import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from pathlib import Path

def create_o3d_pointcloud(depth_map, image_rgb, mask=None, downsample_factor=2):
    """Create point cloud preserving original colors"""
    h, w = depth_map.shape

    # Normalize depth to 0-1 range (don't invert)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

    # Create grid
    step = downsample_factor
    y, x = np.mgrid[0:h:step, 0:w:step]
    z = depth_map[::step, ::step] * 5  # Scale depth for better visualization

    # Convert to 3D coordinates
    x = (x - w/2) / (w/2)  # X: -1 to 1
    y = -(y - h/2) / (h/2)  # Y: -1 to 1 (flipped)

    # Apply mask if provided
    if mask is not None:
        # Ensure mask is boolean and properly downsampled
        mask = mask.astype(bool)
        mask = mask[::step, ::step]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        colors = image_rgb[::step, ::step, :][mask] / 255.0
    else:
        colors = image_rgb[::step, ::step, :].reshape(-1, 3) / 255.0

    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def save_combined_3d_models(pcd_list, mesh_list, output_dir, filename_prefix):
    """Improved robust 3D model export"""
    # Combine point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        combined_pcd += pcd

    # 1. Point cloud pre-processing
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )
    combined_pcd = combined_pcd.voxel_down_sample(0.01)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    # Save PLY (unchanged)
    ply_path = output_dir / f"{filename_prefix}_combined.ply"
    o3d.io.write_point_cloud(str(ply_path), combined_pcd)

    # 2. Improved Poisson reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            combined_pcd,
            depth=11,
            width=0,
            scale=1.1,
            linear_fit=False,
            n_threads=-1
        )
    
    # 3. Enhanced mesh cleaning
    density_threshold = np.quantile(densities, 0.03)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Additional mesh cleaning
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # 4. Optional mesh smoothing
    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh.compute_vertex_normals()

    # Combine with additional meshes
    for additional_mesh in mesh_list:
        mesh += additional_mesh

    # 5. Save OBJ with proper options
    obj_path = output_dir / f"{filename_prefix}_combined.obj"
    o3d.io.write_triangle_mesh(
        str(obj_path),
        mesh,
        write_vertex_normals=True,
        write_vertex_colors=True
    )

    # 6. Save GLB with proper options
    glb_path = output_dir / f"{filename_prefix}_combined.glb"
    o3d.io.write_triangle_mesh(
        str(glb_path),
        mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_ascii=False
    )

    print(f"Saved improved 3D models to {ply_path}, {obj_path}, {glb_path}")

def visualize_with_open3d(depth_map, image_rgb, shadow_mask, object_mask, light_directions=None, output_dir=None):
    """Visualization with exact 3D centroid alignment using only the first detected shadow/object pair"""
    # Create point clouds
    background_pcd = create_o3d_pointcloud(depth_map, image_rgb, downsample_factor=4)
    object_pcd = create_o3d_pointcloud(depth_map, image_rgb, object_mask, downsample_factor=1)
    shadow_pcd = create_o3d_pointcloud(depth_map, image_rgb, shadow_mask, downsample_factor=1)

    # Get exact 3D centroids from point clouds (only for the first shadow/object)
    shadow_points = np.asarray(shadow_pcd.points)
    object_points = np.asarray(object_pcd.points)
    
    if len(shadow_points) == 0 or len(object_points) == 0:
        print("Warning: Empty point cloud detected")
        return

    # Remove outliers from both point clouds
    def remove_outliers(points, radius=0.05, min_neighbors=5):
        if len(points) == 0:
            return points
        tree = KDTree(points)
        mask = np.array([len(tree.query_ball_point(p, radius)) >= min_neighbors for p in points])
        return points[mask]
    
    shadow_points = remove_outliers(shadow_points)
    object_points = remove_outliers(object_points)
  
    if len(shadow_points) == 0 or len(object_points) == 0:
        print("Warning: Empty point cloud after outlier removal")
        return

    # Calculate 3D centroids from cleaned points (only first shadow/object)
    shadow_center = np.mean(shadow_points, axis=0)
    object_center = np.mean(object_points, axis=0)
    
    vis_elements = [background_pcd, shadow_pcd, object_pcd]
    export_meshes = []

    # Visualization parameters
    line_radius = 0.015  # Increased thickness
    arrow_radius = 0.03
    arrow_length = 0.1
    red_color = [1, 0, 0]
    extension_factor = 3 # How far beyond object to extend

    # Calculate exact light direction vector (only for first pair)
    light_vector = object_center - shadow_center
    light_dir = light_vector / np.linalg.norm(light_vector)
    
    # Calculate extended endpoint (passing through object center)
    original_length = np.linalg.norm(light_vector)
    extended_end = shadow_center + light_dir * (original_length * extension_factor)

    # Create the line (cylinder)
    line_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=line_radius,
        height=original_length * extension_factor
    )
    line_cylinder.paint_uniform_color(red_color)
    
    # Position the line (starts at shadow center, goes through object center)
    rot_mat = rotation_matrix_from_vectors([0, 0, 1], light_dir)
    line_cylinder.rotate(rot_mat)
    line_cylinder.translate(shadow_center + light_dir * (original_length * extension_factor / 2))

    # Create arrow head at extended endpoint
    arrow = o3d.geometry.TriangleMesh.create_cone(
        radius=arrow_radius,
        height=arrow_length
    )
    arrow.paint_uniform_color(red_color)
    arrow.rotate(rot_mat)
    arrow.translate(extended_end - light_dir * (arrow_length / 2))

    # Mark centroids with spheres
    shadow_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=line_radius*2)
    shadow_sphere.translate(shadow_center)
    shadow_sphere.paint_uniform_color([1, 0, 0])  # Red for shadow

    object_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=line_radius*2)
    object_sphere.translate(object_center)
    object_sphere.paint_uniform_color([0, 1, 0])  # Green for object

    # Add all elements
    vis_elements.extend([line_cylinder, arrow, shadow_sphere, object_sphere])
    export_meshes.extend([line_cylinder, arrow, shadow_sphere, object_sphere])

    # Save models
    if output_dir:
        save_combined_3d_models(
            [background_pcd, shadow_pcd, object_pcd],
            export_meshes,
            output_dir,
            "scene"
        )

    # Visualization setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768)
    for elem in vis_elements:
        vis.add_geometry(elem)

    # Camera positioned to view the line clearly
    ctr = vis.get_view_control()
    view_dir = -light_dir * 0.7 + np.array([0, 0.3, 0])  # Slight angle
    ctr.set_front(view_dir)
    ctr.set_up([0, 0, 1])
    ctr.set_lookat(object_center)  # Focus on object center
    ctr.set_zoom(0.5)

    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = [0.9, 0.9, 0.9]

    vis.run()
    vis.destroy_window()

def rotation_matrix_from_vectors(vec1, vec2):
    """Find rotation matrix that aligns vec1 to vec2"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
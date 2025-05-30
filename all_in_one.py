import cv2
import torch
import numpy as np
import detectron2
from pathlib import Path
from skimage.feature import canny
from matplotlib import pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.morphology import binary_erosion, binary_dilation
import open3d as o3d
from detectron2 import model_zoo
from detectron2.model_zoo import configs
from detectron2.config import get_cfg
from adet.config import get_cfg as get_adet_cfg
import os
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import KDTree

# ====================== Path Configuration ======================

# Model paths (in home directory)
SSIS_CONFIG = Path.home() / "SSIS/configs/SSIS/MS_R_101_BiFPN_with_offset_class.yaml"
SSIS_WEIGHTS = Path.home() / "SSIS/tools/output/SSIS_MS_R_101_bifpn_with_offset_class/model_ssis_final.pth"
DEPTH_MODEL_NAME = "LiheYoung/depth-anything-large-hf"

# Input/output paths (relative to project directory)
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ====================== Model Loading ======================

def load_models():
    cfg = get_adet_cfg()
    cfg.merge_from_file(str(SSIS_CONFIG))
    cfg.MODEL.WEIGHTS = str(SSIS_WEIGHTS)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize predictor
    ssis_predictor = DefaultPredictor(cfg)

    # Load Depth Anything model
    depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
    depth_model = depth_model.to(cfg.MODEL.DEVICE)

    return ssis_predictor, depth_processor, depth_model

# ====================== Open3D Visualization Functions ======================

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
    """Improved robust 3D model export that handles both TriangleMesh and LineSet"""
    # Combine point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        combined_pcd += pcd

    # 1. Point cloud pre-processing
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=50
        )
    )
    combined_pcd = combined_pcd.voxel_down_sample(0.005)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(
        nb_neighbors=30, std_ratio=1
    )

    # Save PLY (unchanged)
    ply_path = output_dir / f"{filename_prefix}_combined.ply"
    o3d.io.write_point_cloud(str(ply_path), combined_pcd)

    # 2. Improved Poisson reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            combined_pcd,
            depth=10,                # Increased depth for more detail
            width=0,                 # 0 for auto-determination
            scale=1.2,               # Slightly larger scale
            linear_fit=True,         # Better for flat surfaces
            n_threads=-1             
        )
    
    # 3. Enhanced mesh cleaning
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
   

    # Create a combined mesh for export (separate from visualization)
    export_mesh = mesh
    
    # Handle additional meshes (both TriangleMesh and LineSet)
    for additional_geom in mesh_list:
        if isinstance(additional_geom, o3d.geometry.TriangleMesh):
            export_mesh += additional_geom
        elif isinstance(additional_geom, o3d.geometry.LineSet):
            # Convert LineSet to TriangleMesh for export
            line_mesh = o3d.geometry.TriangleMesh()
            vertices = np.asarray(additional_geom.points)
            lines = np.asarray(additional_geom.lines)
            
            # Create small cylinders for each line segment
            for line in lines:
                start = vertices[line[0]]
                end = vertices[line[1]]
                direction = end - start
                length = np.linalg.norm(direction)
                
                if length > 0:
                    direction /= length
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=0.005, height=length
                    )
                    rot_mat = rotation_matrix_from_vectors([0, 0, 1], direction)
                    cylinder.rotate(rot_mat)
                    cylinder.translate((start + end) / 2)
                    export_mesh += cylinder

    # 5. Save OBJ with proper options
    obj_path = output_dir / f"{filename_prefix}_combined.obj"
    o3d.io.write_triangle_mesh(
        str(obj_path),
        export_mesh,
        
        write_vertex_normals=True,
        write_vertex_colors=True
    )

    # 6. Save GLB with proper options
    glb_path = output_dir / f"{filename_prefix}_combined.glb"
    o3d.io.write_triangle_mesh(
        str(glb_path),
        export_mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_ascii=False
    )

    print(f"Saved improved 3D models to {ply_path}, {obj_path}, {glb_path}")



def visualize_with_open3d(depth_map, image_rgb, shadow_masks, object_masks, light_directions=None, output_dir=None):
    """Visualize each shadow-object pair with its own light direction using simple lines"""
    # Create background point cloud (downsampled for performance)
    background_pcd = create_o3d_pointcloud(depth_map, image_rgb, downsample_factor=4)
    
    # Visualization parameters
    line_radius = 0.005  # Thinner lines since we're not using cylinders
    arrow_radius = 0.05
    arrow_length = 0.2
    extension_factor = 3  # Extend light direction beyond object
    
    # Color palette for different pairs - changed to red
    colors = [(1, 0, 0)] * 10  # All red colors
    
    # Store all visualization elements
    vis_elements = [background_pcd]
    export_meshes = []

    # Process each shadow-object pair
    for i, (shadow_mask, object_mask) in enumerate(zip(shadow_masks, object_masks)):
        # Get point clouds for this specific pair
        shadow_pcd = create_o3d_pointcloud(depth_map, image_rgb, shadow_mask, downsample_factor=1)
        object_pcd = create_o3d_pointcloud(depth_map, image_rgb, object_mask, downsample_factor=1)
        
        # Skip if either point cloud is empty
        if len(shadow_pcd.points) == 0 or len(object_pcd.points) == 0:
            print(f"Warning: Empty point cloud detected for pair {i}")
            continue

        # Remove outliers
        shadow_points = np.asarray(shadow_pcd.points)
        object_points = np.asarray(object_pcd.points)
        
        def remove_outliers(points, radius=0.05, min_neighbors=50):
            if len(points) == 0:
                return points
            tree = KDTree(points)
            mask = np.array([len(tree.query_ball_point(p, radius)) >= min_neighbors for p in points])
            return points[mask]
        
        shadow_points = remove_outliers(shadow_points)
        object_points = remove_outliers(object_points)
        
        if len(shadow_points) == 0 or len(object_points) == 0:
            print(f"Warning: Empty point cloud after outlier removal for pair {i}")
            continue

        # Calculate centroids
        shadow_center = np.mean(shadow_points, axis=0)
        object_center = np.mean(object_points, axis=0)
        
        # Compute light direction (normalized)
        light_vector = object_center - shadow_center
        light_dir = light_vector / np.linalg.norm(light_vector)
        print("estimated light direction: ",i,light_dir)
        # Extend the light direction
        original_length = np.linalg.norm(light_vector)
        extended_end = shadow_center + light_dir * (original_length * extension_factor)

        # Create simple line instead of cylinder
        line_points = [shadow_center, extended_end]
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(line_points)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([colors[i % len(colors)]])

        # Create arrow head
        arrow = o3d.geometry.TriangleMesh.create_cone(
            radius=arrow_radius,
            height=arrow_length
        )
        arrow.paint_uniform_color(colors[i % len(colors)])
        
        # Rotate and position arrow
        rot_mat = rotation_matrix_from_vectors([0, 0, 1], light_dir)
        arrow.rotate(rot_mat)
        arrow.translate(extended_end - light_dir * (arrow_length / 2))

        # Mark centroids with spheres
        shadow_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=line_radius*3)
        shadow_sphere.translate(shadow_center)
        shadow_sphere.paint_uniform_color(colors[i % len(colors)])

        object_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=line_radius*3)
        object_sphere.translate(object_center)
        object_sphere.paint_uniform_color(colors[i % len(colors)])

        # Add all elements for this pair
        vis_elements.extend([shadow_pcd, object_pcd, line, arrow, shadow_sphere, object_sphere])
        export_meshes.extend([shadow_pcd, object_pcd, line, arrow, shadow_sphere, object_sphere])

    # Save models (optional)
        if output_dir:
            # Only export TriangleMesh objects (convert lines to cylinders for export)
            export_meshes = []
            for elem in [shadow_pcd, object_pcd, line, arrow, shadow_sphere, object_sphere]:
                if isinstance(elem, o3d.geometry.TriangleMesh):
                    export_meshes.append(elem)
            
            save_combined_3d_models(
                [background_pcd],
                export_meshes,
                output_dir,
                "scene"
            )

    # Visualization setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768)
    for elem in vis_elements:
        vis.add_geometry(elem)

    # Camera positioning (focus on first pair if available)
    ctr = vis.get_view_control()
    if len(shadow_masks) > 0:
        first_shadow_center = np.mean(np.asarray(
            create_o3d_pointcloud(depth_map, image_rgb, shadow_masks[0], downsample_factor=1).points), 
            axis=0
        )
        first_object_center = np.mean(np.asarray(
            create_o3d_pointcloud(depth_map, image_rgb, object_masks[0], downsample_factor=1).points), 
            axis=0
        )
        light_dir = first_object_center - first_shadow_center
        light_dir = light_dir / np.linalg.norm(light_dir)
        view_dir = -light_dir * 0.7 + np.array([0, 0.3, 0])  # Slight angle
        ctr.set_front(view_dir)
        ctr.set_up([0, 0, 1])
        ctr.set_lookat(first_object_center)  # Focus on first object center
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

def get_centroid_3d(mask, depth_map):
    """Get 3D centroid of a masked region"""
    h, w = depth_map.shape
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return np.array([0, 0, 0])

    # Get 2D centroid
    cx = int(np.mean(x))
    cy = int(np.mean(y))

    # Convert to 3D
    z = depth_map[cy, cx]
    return np.array([
        (cx - w/2) / w,
        -(cy - h/2) / h,  # Flip Y-axis
        z * 2
    ])    

# ======================  Pipeline Function ======================

def estimate_light_direction(image_path, ssis_predictor, depth_processor, depth_model, output_dir=Path("output")):
    """Process a single image with enhanced Open3D visualization for all shadows"""
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # Get depth map
    inputs = depth_processor(images=image_rgb, return_tensors="pt").to(depth_model.device)
    with torch.no_grad():
        depth_outputs = depth_model(**inputs)
    depth_map = depth_outputs.predicted_depth.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Detect instances
    ssis_outputs = ssis_predictor(image_rgb)
    instances = ssis_outputs["instances"] if hasattr(ssis_outputs, "instances") else ssis_outputs[0]["instances"]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()

    # Separate shadows and objects
    shadow_indices = np.where(pred_classes == 1)[0]
    object_indices = np.where(pred_classes == 0)[0]

    if len(shadow_indices) == 0:
        raise ValueError("No shadows detected in image")
    if len(object_indices) == 0:
        raise ValueError("No objects detected in image")

    # Prepare visualization image
    result_img = image.copy()
    light_directions = []
    shadow_masks = []
    object_masks = []

    # Process all shadow-object pairs
    for shadow_idx in shadow_indices:
        shadow_mask = pred_masks[shadow_idx]
        shadow_center = get_centroid(shadow_mask)
        
        if shadow_center:
            object_idx, object_mask = find_nearest_object_by_centroid(
                shadow_center, 
                pred_masks[object_indices]
            )
            
            if object_idx is not None:
                object_center = get_centroid(object_mask)
                
                if object_center:
                    light_dir = calculate_light_direction(
                        image_rgb, 
                        shadow_mask, 
                        object_mask, 
                        depth_processor, 
                        depth_model
                    )
                    
                    if light_dir is not None:
                        light_directions.append(light_dir)
                        shadow_masks.append(shadow_mask)
                        object_masks.append(object_mask)
                        
                        # Draw this pair on the 2D image
                        draw_light_direction(
                            result_img, 
                            shadow_center, 
                            object_center, 
                            idx=len(light_directions)-1  # Current index
                        )

    # Save 2D results
    cv2.imwrite(str(output_dir / "light_directions.jpg"), result_img)
    cv2.imwrite(str(output_dir / "original.jpg"), image)

    # Create enhanced Open3D visualization and save 3D models
    if shadow_masks and object_masks:
        visualize_with_open3d(
            depth_map, 
            image_rgb, 
            shadow_masks, 
            object_masks, 
            light_directions,
            output_dir
        )

    # Also keep matplotlib plot for light directions
   

    return light_directions


def find_nearest_object_by_centroid(shadow_center, object_masks):
    """Find the object whose centroid is closest to the shadow centroid"""
    min_dist = float('inf')
    best_idx = None
    best_mask = None

    for i, obj_mask in enumerate(object_masks):
        obj_center = get_centroid(obj_mask)
        if not obj_center:
            continue
        dist = np.linalg.norm(np.array(shadow_center) - np.array(obj_center))
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            best_mask = obj_mask

    return (best_idx, best_mask) if best_idx is not None else (None, None)

def calculate_light_direction(image_rgb, shadow_mask, object_mask, depth_processor, depth_model):
    """Calculate 3D light direction between shadow and object"""
    H, W = image_rgb.shape[:2]

    # Get depth map
    inputs = depth_processor(images=image_rgb, return_tensors="pt").to(depth_model.device)
    with torch.no_grad():
        depth_outputs = depth_model(**inputs)
    depth_map = depth_outputs.predicted_depth.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Get centroids
    shadow_center = get_centroid(shadow_mask)
    object_center = get_centroid(object_mask)

    if not shadow_center or not object_center:
        return None

    # Convert to 3D coordinates (without camera intrinsics)
    def point_to_3d(point, depth_map):
        x, y = point
        z = depth_map[y, x]
        return np.array([
            (x - W/2) * z,
            (y - H/2) * z,
            z
        ])

    shadow_3d = point_to_3d(shadow_center, depth_map)
    object_3d = point_to_3d(object_center, depth_map)

    # Calculate and normalize direction vector
    light_dir = object_3d - shadow_3d
    return light_dir / np.linalg.norm(light_dir)

def get_centroid(mask):
    """Calculate centroid of a binary mask"""
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    return (int(np.mean(x)), int(np.mean(y)))

def draw_light_direction(image, shadow_center, object_center, idx):
    """Draw light direction arrow between points with unique colors"""
    # Color palette for multiple light directions - changed to red
    colors = [(1, 0, 0)] * 10  # All red colors
    
    # Draw centroids
    color = [int(c * 255) for c in colors[idx % len(colors)]]
    cv2.circle(image, shadow_center, 5, color, -1)
    
    object_color = [int(c * 255) for c in colors[(idx + 5) % len(colors)]]  # Different but consistent color
    cv2.circle(image, object_center, 5, object_color, -1)

    # Draw arrow extending beyond the object
    arrow_length = 1.5  # Extend 50% beyond the object
    end_point = (
        int(shadow_center[0] + arrow_length * (object_center[0] - shadow_center[0])),
        int(shadow_center[1] + arrow_length * (object_center[1] - shadow_center[1]))
    )

    cv2.arrowedLine(image, shadow_center, end_point, color, 3, tipLength=0.2)


# ====================== Main Execution ======================

if __name__ == "__main__":
    # Load models
    ssis_predictor, depth_processor, depth_model = load_models()

    # Process input image
    input_image_path = Path("crosswalk.jpg")

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found at {input_image_path}")

    light_directions = estimate_light_direction(
        input_image_path,
        ssis_predictor,
        depth_processor,
        depth_model, 
        output_dir=Path("light-direction-estimation/outputs")
    )
   
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.spatial import KDTree

from .visualization import visualize_with_open3d
from .utils import get_centroid, get_centroid_3d

def estimate_light_direction(image_path, ssis_predictor, depth_processor, depth_model, output_dir=Path("output")):
    """Process a single image with enhanced Open3D visualization"""
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
    combined_shadow_mask = np.zeros((H, W), dtype=bool)
    combined_object_mask = np.zeros((H, W), dtype=bool)

    # Only process the first shadow-object pair
    first_shadow_idx = shadow_indices[0]
    shadow_mask = pred_masks[first_shadow_idx]
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
                    draw_light_direction(
                        result_img, 
                        shadow_center, 
                        object_center, 
                        idx=0  # Only one pair now
                    )
                    combined_shadow_mask = shadow_mask
                    combined_object_mask = object_mask

    # Save 2D results
    cv2.imwrite(str(output_dir / "light_directions.jpg"), result_img)
    cv2.imwrite(str(output_dir / "original.jpg"), image)

    # Create enhanced Open3D visualization and save 3D models
    visualize_with_open3d(
        depth_map, 
        image_rgb, 
        combined_shadow_mask, 
        combined_object_mask, 
        light_directions,
        output_dir
    )

    # Also keep matplotlib plot for light directions
    if light_directions:
        plot_light_directions(light_directions, output_dir)

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

def draw_light_direction(image, shadow_center, object_center, idx):
    """Draw light direction arrow between points"""
    # Draw centroids
    cv2.circle(image, shadow_center, 5, (0, 0, 255), -1)  # Red for shadow
    cv2.circle(image, object_center, 5, (0, 255, 0), -1)   # Green for object

    # Draw arrow extending beyond the object
    arrow_length = 1.5  # Extend 50% beyond the object
    end_point = (
        int(shadow_center[0] + arrow_length * (object_center[0] - shadow_center[0])),
        int(shadow_center[1] + arrow_length * (object_center[1] - shadow_center[1]))
    )

    cv2.arrowedLine(image, shadow_center, end_point, (0, 0, 255), 3, tipLength=0.2)

def plot_light_directions(light_directions, output_dir):
    """Create 3D plot of light directions"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, dir in enumerate(light_directions):
        ax.quiver(0, 0, 0, dir[0], dir[1], dir[2], 
                 length=0.5, color=plt.cm.tab10(i % 10), 
                 arrow_length_ratio=0.1, label=f"Light {i+1}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated Light Directions')
    ax.legend()
    plt.savefig(str(output_dir / "light_directions_3d.png"))
    plt.close()

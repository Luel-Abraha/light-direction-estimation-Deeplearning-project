import numpy as np
import cv2

def get_centroid(mask):
    """Calculate centroid of a binary mask"""
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    return (int(np.mean(x)), int(np.mean(y)))

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
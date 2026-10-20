"""Depth-image helpers used by contact validation and RGB-D candidate filtering."""

import numpy as np


def depth_patch_stats(depth_img, x, y, window=5):
    h, w = depth_img.shape
    x1 = max(0, x - window)
    x2 = min(w, x + window + 1)
    y1 = max(0, y - window)
    y2 = min(h, y + window + 1)

    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]
    valid_ratio = float(len(valid) / max(patch.size, 1))

    if len(valid) == 0:
        return {
            "depth_mm": None,
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "total_pixels": int(patch.size),
        }

    return {
        "depth_mm": float(np.median(valid)),
        "valid_ratio": valid_ratio,
        "valid_pixels": int(len(valid)),
        "total_pixels": int(patch.size),
    }


def get_depth_mm(depth_img, x, y, window=5):
    return depth_patch_stats(depth_img, x, y, window)["depth_mm"]


def pixel_to_3d(x, y, z_mm, fx, fy, cx, cy):
    z = z_mm / 1000.0
    x_m = (x - cx) * z / fx
    y_m = (y - cy) * z / fy
    return np.array([x_m, y_m, z])


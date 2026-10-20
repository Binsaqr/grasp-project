import cv2
import numpy as np


USE_DEPTH_SMOOTHING = True
DEPTH_MEDIAN_KERNEL = 5
DEPTH_BILATERAL_ENABLED = True
DEPTH_PATCH_SIZE = 11
DEPTH_VALID_MIN_MM = 50
DEPTH_VALID_MAX_MM = 2000


def valid_depth_mask(depth):
    if depth is None:
        return np.zeros((0, 0), dtype=bool)
    depth = np.asarray(depth)
    return (
        np.isfinite(depth)
        & (depth >= DEPTH_VALID_MIN_MM)
        & (depth <= DEPTH_VALID_MAX_MM)
    )


def _odd_kernel(value, minimum=3):
    value = int(value)
    if value < minimum:
        value = minimum
    if value % 2 == 0:
        value += 1
    return value


def _filled_depth_for_filtering(depth):
    depth = np.asarray(depth)
    valid = valid_depth_mask(depth)
    if not np.any(valid):
        return depth.astype(np.float32), valid

    filled = depth.astype(np.float32).copy()
    global_median = float(np.median(filled[valid]))
    filled[~valid] = global_median
    local_median = cv2.medianBlur(filled, _odd_kernel(DEPTH_MEDIAN_KERNEL))
    filled[~valid] = local_median[~valid]
    return filled, valid


def smooth_depth_median(depth, kernel_size=5):
    """Median-filter valid depth while keeping invalid pixels invalid."""
    if depth is None:
        return None
    filled, valid = _filled_depth_for_filtering(depth)
    if not np.any(valid):
        return np.zeros_like(depth)

    filtered = cv2.medianBlur(filled, _odd_kernel(kernel_size))
    filtered[~valid] = 0
    return filtered.astype(depth.dtype, copy=False)


def smooth_depth_bilateral(depth, diameter=7, sigma_color=30, sigma_space=7):
    """Bilateral-filter depth surfaces while preserving invalid zeros."""
    if depth is None:
        return None
    filled, valid = _filled_depth_for_filtering(depth)
    if not np.any(valid):
        return np.zeros_like(depth)

    filtered = cv2.bilateralFilter(
        filled.astype(np.float32),
        int(diameter),
        float(sigma_color),
        float(sigma_space),
    )
    filtered[~valid] = 0
    return np.clip(filtered, 0, np.iinfo(depth.dtype).max if np.issubdtype(depth.dtype, np.integer) else np.inf).astype(depth.dtype, copy=False)


def smooth_depth_pipeline(depth):
    """Default depth helper used for visualization, mask cleanup, and tracking support."""
    if depth is None or not USE_DEPTH_SMOOTHING:
        return depth
    smoothed = smooth_depth_median(depth, DEPTH_MEDIAN_KERNEL)
    if DEPTH_BILATERAL_ENABLED:
        smoothed = smooth_depth_bilateral(smoothed, diameter=7, sigma_color=30, sigma_space=7)
    return smoothed


def depth_stats(depth, mask):
    valid = valid_depth_mask(depth) & (mask > 0)
    values = np.asarray(depth)[valid]
    total = int(np.count_nonzero(mask))
    if values.size == 0:
        return {
            "median_mm": None,
            "mean_mm": None,
            "std_mm": None,
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "mask_pixels": total,
        }
    return {
        "median_mm": float(np.median(values)),
        "mean_mm": float(np.mean(values)),
        "std_mm": float(np.std(values)),
        "valid_ratio": float(values.size / max(total, 1)),
        "valid_pixels": int(values.size),
        "mask_pixels": total,
    }


def depth_patch_stats(depth, point_xy, patch_size=None, window=None):
    if point_xy is None:
        return {
            "depth_mm": None,
            "median_mm": None,
            "mean_mm": None,
            "std_mm": None,
            "min_mm": None,
            "max_mm": None,
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "patch_pixels": 0,
        }
    if patch_size is None:
        if window is not None:
            patch_size = int(window) * 2 + 1
        else:
            patch_size = DEPTH_PATCH_SIZE
    patch_size = _odd_kernel(patch_size, minimum=1)
    radius = patch_size // 2
    x, y = point_xy
    h, w = depth.shape[:2]
    x1 = max(0, int(round(x)) - radius)
    x2 = min(w, int(round(x)) + radius + 1)
    y1 = max(0, int(round(y)) - radius)
    y2 = min(h, int(round(y)) + radius + 1)
    patch = np.asarray(depth)[y1:y2, x1:x2]
    valid = patch[valid_depth_mask(patch)]
    valid_ratio = float(valid.size / max(patch.size, 1))
    if valid.size == 0:
        return {
            "depth_mm": None,
            "median_mm": None,
            "mean_mm": None,
            "std_mm": None,
            "min_mm": None,
            "max_mm": None,
            "valid_ratio": valid_ratio,
            "valid_pixels": 0,
            "patch_pixels": int(patch.size),
            "patch_size": int(patch_size),
        }
    median = float(np.median(valid))
    return {
        "depth_mm": median,
        "median_mm": median,
        "mean_mm": float(np.mean(valid)),
        "valid_ratio": valid_ratio,
        "std_mm": float(np.std(valid)),
        "min_mm": float(np.min(valid)),
        "max_mm": float(np.max(valid)),
        "valid_pixels": int(valid.size),
        "patch_pixels": int(patch.size),
        "patch_size": int(patch_size),
    }


def pixel_to_3d(point_xy, z_mm, intrinsics):
    if intrinsics is None or z_mm is None:
        return None
    x, y = point_xy
    z = float(z_mm) / 1000.0
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return [
        float((x - cx) * z / fx),
        float((y - cy) * z / fy),
        float(z),
    ]

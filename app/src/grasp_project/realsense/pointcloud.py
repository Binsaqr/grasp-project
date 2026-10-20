"""Optional RealSense RGB-D point cloud support.

This module is dependency-light by default.  Open3D is used when available, but
all core conversions and summaries work with NumPy so the live demo can keep
running on machines where Open3D is not installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from grasp_project.common.config import CONFIG_DIR, OUTPUT_DIR, ensure_output_dir, get_env_bool, get_env_float, get_env_int, load_named_config
from grasp_project.common.file_io import json_safe, write_json_atomic
from grasp_project.realsense.multiview_reconstruction import (
    multiview_input_gate,
    multiview_status_from_file,
    multiview_target_context,
    submit_multiview_candidate_async,
)

POINTCLOUD_ENABLED = get_env_bool("POINTCLOUD_ENABLED", True)
POINTCLOUD_GENERATE_FPS = get_env_float("POINTCLOUD_GENERATE_FPS", 1.0)
POINTCLOUD_SAVE_PLY_ON_LOCK = get_env_bool("POINTCLOUD_SAVE_PLY_ON_LOCK", True)
POINTCLOUD_DOWNSAMPLE = get_env_bool("POINTCLOUD_DOWNSAMPLE", True)
POINTCLOUD_VOXEL_SIZE_M = get_env_float("POINTCLOUD_VOXEL_SIZE_M", 0.005)
POINTCLOUD_USE_OPEN3D = get_env_bool("POINTCLOUD_USE_OPEN3D", False)
POINTCLOUD_LIVE_STRIDE = max(1, get_env_int("POINTCLOUD_LIVE_STRIDE", 2))
POINTCLOUD_PREVIEW_ENABLED = get_env_bool("POINTCLOUD_PREVIEW_ENABLED", True)
POINTCLOUD_PREVIEW_FPS = get_env_float("POINTCLOUD_PREVIEW_FPS", 1.0)
POINTCLOUD_CONTEXT_PREVIEW_FPS = get_env_float("POINTCLOUD_CONTEXT_PREVIEW_FPS", min(0.5, POINTCLOUD_PREVIEW_FPS))
POINTCLOUD_MAX_TEST_FPS = get_env_float("POINTCLOUD_MAX_TEST_FPS", 5.0)
POINTCLOUD_PREVIEW_MAX_POINTS = get_env_int("POINTCLOUD_PREVIEW_MAX_POINTS", 6_000)
POINTCLOUD_SCENE_MAX_POINTS = get_env_int("POINTCLOUD_SCENE_MAX_POINTS", POINTCLOUD_PREVIEW_MAX_POINTS)
POINTCLOUD_BOTTLE_MAX_POINTS = get_env_int("POINTCLOUD_BOTTLE_MAX_POINTS", 3_000)
POINTCLOUD_PREVIEW_WIDTH = get_env_int("POINTCLOUD_PREVIEW_WIDTH", 900)
POINTCLOUD_PREVIEW_HEIGHT = get_env_int("POINTCLOUD_PREVIEW_HEIGHT", 600)
POINTCLOUD_JSON_ENABLED = get_env_bool("POINTCLOUD_JSON_ENABLED", True)
POINTCLOUD_JSON_MAX_POINTS = get_env_int("POINTCLOUD_JSON_MAX_POINTS", min(2_000, POINTCLOUD_PREVIEW_MAX_POINTS))
POINTCLOUD_SAVE_PLY_EVERY_FRAME = get_env_bool("POINTCLOUD_SAVE_PLY_EVERY_FRAME", False)
POINTCLOUD_PREVIEW_AUTOSCALE = get_env_bool("POINTCLOUD_PREVIEW_AUTOSCALE", True)
POINTCLOUD_PREVIEW_CENTER_OBJECT = get_env_bool("POINTCLOUD_PREVIEW_CENTER_OBJECT", True)
POINTCLOUD_PREVIEW_DRAW_AXES = get_env_bool("POINTCLOUD_PREVIEW_DRAW_AXES", True)
POINTCLOUD_PREVIEW_DRAW_BBOX = get_env_bool("POINTCLOUD_PREVIEW_DRAW_BBOX", True)
POINTCLOUD_PREVIEW_REMOVE_OUTLIERS = get_env_bool("POINTCLOUD_PREVIEW_REMOVE_OUTLIERS", True)
POINTCLOUD_PREVIEW_DEPTH_MIN_M = get_env_float("POINTCLOUD_PREVIEW_DEPTH_MIN_M", 0.10)
POINTCLOUD_PREVIEW_DEPTH_MAX_M = get_env_float("POINTCLOUD_PREVIEW_DEPTH_MAX_M", 1.20)
POINTCLOUD_PREVIEW_POINT_SIZE = get_env_float("POINTCLOUD_PREVIEW_POINT_SIZE", 2.5)
POINTCLOUD_PREVIEW_BRIGHTNESS_BOOST = get_env_float("POINTCLOUD_PREVIEW_BRIGHTNESS_BOOST", 1.5)
POINTCLOUD_PREVIEW_COLOR_MODE = os.environ.get("POINTCLOUD_PREVIEW_COLOR_MODE", "auto").strip().lower() or "auto"
POINTCLOUD_PREVIEW_FPS_TEST_VALUES = os.environ.get("POINTCLOUD_PREVIEW_FPS_TEST_VALUES", "1,2,5")
POINTCLOUD_BOTTLE_DETAIL_MODE = get_env_bool("POINTCLOUD_BOTTLE_DETAIL_MODE", True)
POINTCLOUD_BOTTLE_PREVIEW_STRIDE = max(1, get_env_int("POINTCLOUD_BOTTLE_PREVIEW_STRIDE", 1))
POINTCLOUD_MIN_DEPTH_M = get_env_float("POINTCLOUD_MIN_DEPTH_M", 0.05)
POINTCLOUD_MAX_DEPTH_M = get_env_float("POINTCLOUD_MAX_DEPTH_M", 2.0)
POINTCLOUD_BOTTLE_DEPTH_BAND_MM = get_env_float("POINTCLOUD_BOTTLE_DEPTH_BAND_MM", 90.0)
POINTCLOUD_BOTTLE_MIN_POINTS = get_env_int("POINTCLOUD_BOTTLE_MIN_POINTS", 120)
POINTCLOUD_BOTTLE_MAX_SCENE_FRACTION = get_env_float("POINTCLOUD_BOTTLE_MAX_SCENE_FRACTION", 0.65)
POINTCLOUD_CONTACT_PATCH_RADIUS_PX = get_env_int("POINTCLOUD_CONTACT_PATCH_RADIUS_PX", 4)
POINTCLOUD_ROLLING_WINDOW_SEC = get_env_float("POINTCLOUD_ROLLING_WINDOW_SEC", 30.0)
POINTCLOUD_ROLLING_MAX_POINTS = get_env_int("POINTCLOUD_ROLLING_MAX_POINTS", 120_000)
POINTCLOUD_ROLLING_VOXEL_SIZE_M = get_env_float("POINTCLOUD_ROLLING_VOXEL_SIZE_M", POINTCLOUD_VOXEL_SIZE_M)
POINTCLOUD_ROLLING_RESET_ON_TARGET_CHANGE = get_env_bool("POINTCLOUD_ROLLING_RESET_ON_TARGET_CHANGE", True)
POINTCLOUD_ROLLING_RESET_ON_BOTTLE_MOVED = get_env_bool("POINTCLOUD_ROLLING_RESET_ON_BOTTLE_MOVED", True)
POINTCLOUD_ROLLING_MIN_VALID_DEPTH_RATIO = get_env_float("POINTCLOUD_ROLLING_MIN_VALID_DEPTH_RATIO", 0.50)
BOTTLE_BBOX_SEED_FRACTION = get_env_float("BOTTLE_BBOX_SEED_FRACTION", 0.35)
BOTTLE_DEPTH_MAD_MULTIPLIER = get_env_float("BOTTLE_DEPTH_MAD_MULTIPLIER", 3.0)
BOTTLE_DEPTH_BAND_MIN_MM = get_env_float("BOTTLE_DEPTH_BAND_MIN_MM", 10.0)
BOTTLE_DEPTH_BAND_MAX_MM = get_env_float("BOTTLE_DEPTH_BAND_MAX_MM", 60.0)
BOTTLE_MAX_BORDER_TOUCH_RATIO = get_env_float("BOTTLE_MAX_BORDER_TOUCH_RATIO", 0.35)
BOTTLE_MIN_COMPONENT_RATIO = get_env_float("BOTTLE_MIN_COMPONENT_RATIO", 0.15)

POINTCLOUD_SESSION_ID = f"pc-{os.getpid()}-{int(time.time() * 1000)}"

POINTCLOUD_STATUS_PATH = OUTPUT_DIR / "pointcloud_status.json"
SCENE_PLY_PATH = OUTPUT_DIR / "latest_realsense_pointcloud.ply"
BOTTLE_PLY_PATH = OUTPUT_DIR / "latest_bottle_pointcloud.ply"
POINTCLOUD_PREVIEW_PATH = OUTPUT_DIR / "latest_pointcloud_preview.jpg"
BOTTLE_POINTCLOUD_PREVIEW_PATH = OUTPUT_DIR / "latest_bottle_pointcloud_preview.jpg"
POINTCLOUD_CONTEXT_PREVIEW_PATH = OUTPUT_DIR / "latest_pointcloud_context.jpg"
POINTCLOUD_LIVE_JSON_PATH = OUTPUT_DIR / "latest_pointcloud_live.json"


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception:
        return None


def dependency_status() -> dict[str, Any]:
    """Return point-cloud dependency availability without requiring hardware."""
    o3d = _try_import_open3d()
    try:
        import cv2  # noqa: F401

        opencv_available = True
    except Exception:
        opencv_available = False
    try:
        from sensor_msgs.msg import PointCloud2  # noqa: F401

        ros_pointcloud_available = True
    except Exception:
        ros_pointcloud_available = False
    return {
        "open3d_available": o3d is not None,
        "numpy_available": True,
        "opencv_available": opencv_available,
        "ros_pointcloud_available": ros_pointcloud_available,
        "open3d_enabled": bool(POINTCLOUD_USE_OPEN3D and o3d is not None),
        "selected_pointcloud_backend": "open3d" if POINTCLOUD_USE_OPEN3D and o3d is not None else "numpy_ascii_ply",
    }


def _depth_meters(depth: np.ndarray, intrinsics: dict[str, Any] | None = None) -> np.ndarray:
    arr = np.asarray(depth)
    intrinsics = intrinsics or {}
    if arr.dtype == np.uint16 or np.nanmax(arr) > 50.0:
        scale = float(intrinsics.get("depth_scale_m") or intrinsics.get("depth_scale") or 0.001)
        return arr.astype(np.float32) * scale
    return arr.astype(np.float32)


def _valid_depth_m_mask(depth_m: np.ndarray) -> np.ndarray:
    """Reject missing/sentinel depths using the configured working range."""
    return (
        np.isfinite(depth_m)
        & (depth_m >= float(POINTCLOUD_MIN_DEPTH_M))
        & (depth_m <= float(POINTCLOUD_MAX_DEPTH_M))
    )


def _intrinsics_complete(intrinsics: dict[str, Any] | None) -> bool:
    intrinsics = intrinsics or {}
    return all(intrinsics.get(k) is not None for k in ("fx", "fy", "cx", "cy"))


def depth_to_xyz(depth: np.ndarray, intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert an aligned depth image into Nx3 XYZ meters and valid mask."""
    if depth is None or not _intrinsics_complete(intrinsics):
        return np.empty((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
    depth_m = _depth_meters(np.asarray(depth), intrinsics)
    valid = _valid_depth_m_mask(depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), valid.reshape(-1)
    h, w = depth_m.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = depth_m
    x = (xx - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (yy - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    return xyz[flat_valid].astype(np.float32, copy=False), flat_valid


def masked_depth_to_xyz(
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deproject every valid masked aligned-depth pixel.

    Returns XYZ in metres, its matching ``(x, y)`` image coordinates, and
    explicit coverage metadata.  This is the shared primitive used by the
    bottle-local 3D grasp lock; it deliberately does not invent depth for
    invalid pixels.
    """
    empty_points = np.empty((0, 3), dtype=np.float32)
    empty_pixels = np.empty((0, 2), dtype=np.float32)
    if depth is None or mask is None or not _intrinsics_complete(intrinsics):
        return empty_points, empty_pixels, {
            "mask_pixel_count": 0,
            "valid_point_count": 0,
            "depth_valid_ratio": 0.0,
            "reason": "missing_depth_mask_or_intrinsics",
        }
    depth_arr = np.asarray(depth)
    mask_arr = np.asarray(mask)
    if depth_arr.ndim != 2 or mask_arr.shape[:2] != depth_arr.shape[:2]:
        return empty_points, empty_pixels, {
            "mask_pixel_count": 0,
            "valid_point_count": 0,
            "depth_valid_ratio": 0.0,
            "reason": "mask_depth_shape_mismatch",
        }
    selected = mask_arr.astype(bool)
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        return empty_points, empty_pixels, {
            "mask_pixel_count": 0,
            "valid_point_count": 0,
            "depth_valid_ratio": 0.0,
            "reason": "empty_mask",
        }
    depth_m = _depth_meters(depth_arr, intrinsics)
    valid = selected & _valid_depth_m_mask(depth_m)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return empty_points, empty_pixels, {
            "mask_pixel_count": selected_count,
            "valid_point_count": 0,
            "depth_valid_ratio": 0.0,
            "reason": "no_valid_masked_depth",
        }
    yy, xx = np.nonzero(valid)
    z = depth_m[yy, xx]
    x = (xx.astype(np.float32) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (yy.astype(np.float32) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    xyz = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    pixels = np.column_stack((xx, yy)).astype(np.float32, copy=False)
    return xyz, pixels, {
        "mask_pixel_count": selected_count,
        "valid_point_count": valid_count,
        "depth_valid_ratio": float(valid_count / selected_count),
        "reason": "ok",
    }


def colorize_pointcloud(points: np.ndarray, rgb: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Return Nx3 float colors in 0..1 matching ``points``."""
    if rgb is None or np.asarray(rgb).ndim != 3:
        return np.zeros((len(points), 3), dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    if valid_mask is not None and len(valid_mask) == len(colors):
        colors = colors[np.asarray(valid_mask, dtype=bool)]
    if len(colors) != len(points):
        return np.zeros((len(points), 3), dtype=np.float32)
    return colors.astype(np.float32, copy=False)


def create_pointcloud_from_rgbd(rgb: np.ndarray, depth: np.ndarray, intrinsics: dict[str, Any]) -> dict[str, Any]:
    """Create a colored point cloud from an aligned RGB-D frame."""
    points, valid_mask = depth_to_xyz(depth, intrinsics)
    colors = colorize_pointcloud(points, rgb, valid_mask)
    total_pixels = int(np.asarray(depth).size) if depth is not None else 0
    valid_ratio = float(len(points) / total_pixels) if total_pixels else 0.0
    return {
        "points": points,
        "colors": colors,
        "valid_mask": valid_mask,
        "summary": pointcloud_summary(points, valid_ratio=valid_ratio),
    }


def _depth_range_mm(depth: np.ndarray | None, intrinsics: dict[str, Any] | None = None) -> dict[str, Any]:
    if depth is None:
        return {"pointcloud_depth_min_mm": None, "pointcloud_depth_max_mm": None, "pointcloud_depth_valid_ratio": 0.0}
    depth_m = _depth_meters(np.asarray(depth), intrinsics or {})
    valid = _valid_depth_m_mask(depth_m)
    if not np.any(valid):
        return {"pointcloud_depth_min_mm": None, "pointcloud_depth_max_mm": None, "pointcloud_depth_valid_ratio": 0.0}
    values = depth_m[valid]
    return {
        "pointcloud_depth_min_mm": round(float(np.percentile(values, 1)) * 1000.0, 1),
        "pointcloud_depth_max_mm": round(float(np.percentile(values, 99)) * 1000.0, 1),
        "pointcloud_depth_valid_ratio": round(float(np.count_nonzero(valid)) / float(depth_m.size), 4),
    }


def _filter_visual_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    min_depth_m: float = POINTCLOUD_MIN_DEPTH_M,
    max_depth_m: float = POINTCLOUD_MAX_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove invalid visualization outliers without affecting tracking/math."""
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3)) if np.asarray(points).size else np.empty((0, 3), dtype=np.float32)
    colors = np.zeros((len(points), 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return points, colors, 0
    finite = np.all(np.isfinite(points), axis=1)
    z = points[:, 2]
    keep = finite & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    if not np.any(keep):
        keep = finite
    filtered = points[keep]
    filtered_colors = colors[keep] if len(colors) == len(points) else np.zeros((len(filtered), 3), dtype=np.float32)
    return filtered, filtered_colors, int(np.count_nonzero(keep))


def _preview_depth_filter(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply display-only depth/outlier limits for readable previews."""
    if not POINTCLOUD_PREVIEW_REMOVE_OUTLIERS:
        return points, colors, "disabled"
    filtered, filtered_colors, kept = _filter_visual_pointcloud(
        points,
        colors,
        min_depth_m=POINTCLOUD_PREVIEW_DEPTH_MIN_M,
        max_depth_m=POINTCLOUD_PREVIEW_DEPTH_MAX_M,
    )
    if kept <= 0 and len(points):
        return points, colors, "fallback_all_points"
    return filtered, filtered_colors, "preview_depth_range"


def _crop_mask_from_bbox(image_shape: tuple[int, int], bbox_or_mask: Any) -> np.ndarray | None:
    if bbox_or_mask is None:
        return None
    if isinstance(bbox_or_mask, np.ndarray):
        mask = bbox_or_mask.astype(bool)
        if mask.shape[:2] == image_shape:
            return mask.reshape(-1)
        return None
    if isinstance(bbox_or_mask, dict):
        bbox_or_mask = bbox_or_mask.get("bbox_xyxy") or bbox_or_mask.get("bottle_extent_bbox_xyxy") or bbox_or_mask.get("bbox")
    if not isinstance(bbox_or_mask, (list, tuple)) or len(bbox_or_mask) < 4:
        return None
    h, w = image_shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_or_mask[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask.reshape(-1)


def _bbox_xyxy_from_any(bbox_or_mask: Any) -> list[float] | None:
    if isinstance(bbox_or_mask, dict):
        bbox_or_mask = bbox_or_mask.get("bbox_xyxy") or bbox_or_mask.get("bottle_extent_bbox_xyxy") or bbox_or_mask.get("bbox")
    if not isinstance(bbox_or_mask, (list, tuple)) or len(bbox_or_mask) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_or_mask[:4]]
    except Exception:
        return None
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _clip_bbox_xyxy(bbox: Any, image_shape: tuple[int, int]) -> list[int] | None:
    parsed = _bbox_xyxy_from_any(bbox)
    if parsed is None:
        return None
    h, w = image_shape
    x1, y1, x2, y2 = [int(round(v)) for v in parsed]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _frame_ids_close(a: Any, b: Any, *, max_delta: int = 1) -> bool | None:
    if a is None or b is None:
        return None
    try:
        return abs(int(a) - int(b)) <= int(max_delta)
    except Exception:
        return str(a) == str(b)


def _mask_from_candidate(candidate: Any, image_shape: tuple[int, int]) -> np.ndarray | None:
    if candidate is None:
        return None
    if isinstance(candidate, np.ndarray):
        arr = np.asarray(candidate)
        if arr.shape[:2] == image_shape:
            return arr.astype(bool)
        if arr.size >= 6 and arr.shape[-1] == 2:
            contour = arr.reshape((-1, 1, 2)).astype(np.int32)
            mask = np.zeros(image_shape, dtype=np.uint8)
            try:
                import cv2

                cv2.fillPoly(mask, [contour], 1)
                return mask.astype(bool)
            except Exception:
                return None
        return None
    if isinstance(candidate, (list, tuple)):
        try:
            arr = np.asarray(candidate, dtype=np.float32)
            if arr.size >= 6 and arr.shape[-1] == 2:
                contour = arr.reshape((-1, 1, 2)).astype(np.int32)
                mask = np.zeros(image_shape, dtype=np.uint8)
                import cv2

                cv2.fillPoly(mask, [contour], 1)
                return mask.astype(bool)
        except Exception:
            pass
    flat = _crop_mask_from_bbox(image_shape, candidate)
    if flat is None:
        return None
    return flat.reshape(image_shape)


def _mask_metadata(source: str, candidate: Any, mask: np.ndarray | None, report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    geometry = report.get("bottle_geometry") or {}
    lock = report.get("target_lock") or {}
    locked = (lock.get("locked_target") or {}).get("locked_bottle_geometry") or {}
    owner = locked if source.startswith("locked_") else geometry
    geometry_status = owner.get("measurement_status")
    mask_status = owner.get("mask_measurement_status")
    explicitly_noncurrent = bool(
        geometry_status == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
        or str(mask_status or "").startswith("NOT_AVAILABLE")
    )
    explicitly_current = bool(
        geometry_status == "CURRENT_MEASURED_RGBD"
        and mask_status == "CURRENT_MEASURED_RGBD"
    )
    frame_id = owner.get("mask_frame_id")
    if frame_id is None:
        frame_id = owner.get("frame_id")
    if frame_id is None:
        frame_id = owner.get("source_mask_frame_id")
    if frame_id is None:
        frame_id = owner.get("source_geometry_frame_id")
    exact_frame_match = (
        _frame_ids_close(frame_id, report.get("frame_id"), max_delta=0)
        if frame_id is not None
        else None
    )
    return {
        "bottle_mask_available": mask is not None,
        "bottle_mask_source": source if mask is not None and source != "bbox_fallback" else None,
        "bottle_mask_frame_id": frame_id,
        "bottle_geometry_measurement_status": geometry_status,
        "bottle_mask_measurement_status": mask_status,
        "bottle_geometry_explicitly_noncurrent": explicitly_noncurrent,
        "bottle_geometry_explicitly_current": explicitly_current,
        "bottle_mask_provenance_current": bool(
            exact_frame_match is True and explicitly_current
        ),
        "bottle_mask_shape_valid": mask is not None,
        "bottle_mask_pixel_count": int(np.count_nonzero(mask)) if mask is not None else 0,
        "bottle_mask_rejection_reason": None if mask is not None else "no_mask_candidate",
        "bottle_mask_matches_rgb_frame": exact_frame_match,
    }


def _find_bottle_roi_candidate(report: dict[str, Any] | None, image_shape: tuple[int, int]) -> tuple[np.ndarray | None, str, Any, str, dict[str, Any]]:
    """Return the best available image-space bottle ROI mask without doing detection."""
    report = report or {}
    geometry = report.get("bottle_geometry") or {}
    lock = report.get("target_lock") or {}
    locked = lock.get("locked_target") or {}
    locked_geometry = locked.get("locked_bottle_geometry") or {}
    candidates = (
        ("cleaned_mask", geometry.get("cleaned_mask")),
        ("body_mask", geometry.get("body_mask")),
        ("mask", geometry.get("mask")),
        ("body_contour", geometry.get("body_contour")),
        ("contour", geometry.get("contour")),
        ("oriented_bbox", geometry.get("oriented_bbox")),
        ("locked_cleaned_mask", locked_geometry.get("cleaned_mask")),
        ("locked_body_mask", locked_geometry.get("body_mask")),
        ("locked_mask", locked_geometry.get("mask")),
        ("locked_body_contour", locked_geometry.get("body_contour")),
        ("locked_contour", locked_geometry.get("contour")),
        ("locked_oriented_bbox", locked_geometry.get("oriented_bbox")),
        ("bbox_fallback", _bbox_from_report(report)),
    )
    for source, candidate in candidates:
        mask = _mask_from_candidate(candidate, image_shape)
        if mask is not None and int(np.count_nonzero(mask)) > 0:
            return mask, source, candidate, "ok", _mask_metadata(source, candidate, mask, report)
    return None, "unavailable", None, "no_bottle_mask_or_bbox", _mask_metadata("unavailable", None, None, report)


def _source_is_selective_mask(source: str) -> bool:
    text = str(source or "")
    return text.endswith("mask") or "contour" in text


def _finite_depth_stats_mm(depth_m: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = depth_m[np.asarray(mask, dtype=bool) & _valid_depth_m_mask(depth_m)]
    if values.size == 0:
        return {"crop_depth_std_mm": None, "crop_depth_mad_mm": None, "crop_depth_range_mm": None}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "crop_depth_std_mm": round(float(np.std(values)) * 1000.0, 1),
        "crop_depth_mad_mm": round(mad * 1000.0, 1),
        "crop_depth_range_mm": round(float(np.percentile(values, 95) - np.percentile(values, 5)) * 1000.0, 1),
    }


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _component_stats(mask: np.ndarray, reference_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    raw_pixels = int(np.count_nonzero(mask))
    if raw_pixels <= 0:
        return mask, {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "border_touch_ratio": 0.0,
            "crop_compactness": 0.0,
            "crop_verticality": 0.0,
        }
    try:
        import cv2

        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        component_count = max(0, int(labels_count - 1))
        if component_count <= 0:
            component = mask
            largest_ratio = 1.0
        else:
            ref = reference_mask if reference_mask is not None and np.any(reference_mask) else mask
            ys, xs = np.nonzero(ref)
            ref_center = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32) if xs.size else np.array([0.0, 0.0], dtype=np.float32)
            best_label = 1
            best_score = float("inf")
            largest_area = 0.0
            for label in range(1, labels_count):
                area = float(stats[label, cv2.CC_STAT_AREA])
                largest_area = max(largest_area, area)
                dist = float(np.linalg.norm(np.asarray(centroids[label], dtype=np.float32) - ref_center))
                score = dist - 0.01 * area
                if score < best_score:
                    best_score = score
                    best_label = label
            component = labels == best_label
            largest_ratio = float(largest_area / max(raw_pixels, 1))
    except Exception:
        component = mask
        component_count = 1
        largest_ratio = 1.0
    bbox = _bbox_from_mask(component)
    border_touch_ratio = 0.0
    compactness = 0.0
    verticality = 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        h, w = component.shape[:2]
        border = np.zeros_like(component, dtype=bool)
        border[y1:y2, x1] = True
        border[y1:y2, max(x1, x2 - 1)] = True
        border[y1, x1:x2] = True
        border[max(y1, y2 - 1), x1:x2] = True
        border_touch_ratio = float(np.count_nonzero(component & border) / max(np.count_nonzero(component), 1))
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        compactness = float(np.count_nonzero(component) / max(bw * bh, 1))
        verticality = float(bh / max(bw, 1))
        if x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h:
            border_touch_ratio = max(border_touch_ratio, 0.5)
    return component.astype(bool), {
        "component_count": int(component_count),
        "largest_component_ratio": round(float(largest_ratio), 4),
        "border_touch_ratio": round(float(border_touch_ratio), 4),
        "crop_compactness": round(float(compactness), 4),
        "crop_verticality": round(float(verticality), 4),
    }


def _morph_mask(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        arr = cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return arr.astype(bool)
    except Exception:
        return np.asarray(mask, dtype=bool)


def _largest_component_near_centroid(mask: np.ndarray, reference_mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Keep the connected component closest to the bottle ROI centroid."""
    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return mask, 0
    try:
        import cv2

        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if labels_count <= 1:
            return mask, 1
        ref = reference_mask if reference_mask is not None and np.any(reference_mask) else mask
        ys, xs = np.nonzero(ref)
        ref_center = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32) if xs.size else np.array([0.0, 0.0], dtype=np.float32)
        best_label = 1
        best_score = float("inf")
        for label in range(1, labels_count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            center = np.asarray(centroids[label], dtype=np.float32)
            dist = float(np.linalg.norm(center - ref_center))
            score = dist - 0.01 * area
            if score < best_score:
                best_score = score
                best_label = label
        return labels == best_label, int(labels_count - 1)
    except Exception:
        return mask, 1


def _clean_bottle_roi_mask(
    roi_mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, Any] | None,
    *,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply depth and component filtering to keep the bottle cloud compact."""
    depth_m = _depth_meters(np.asarray(depth), intrinsics or {})
    valid_depth = _valid_depth_m_mask(depth_m)
    raw_mask = np.asarray(roi_mask).astype(bool)
    raw_pixels = int(np.count_nonzero(raw_mask))
    valid_in_roi = valid_depth & raw_mask
    valid_pixels = int(np.count_nonzero(valid_in_roi))
    status: dict[str, Any] = {
        "bottle_mask_pixel_count": raw_pixels,
        "bottle_depth_valid_pixel_count": valid_pixels,
        "bottle_depth_filter_source": source,
        "bottle_connected_component_count": 0,
        "bottle_depth_median_mm": None,
        "bottle_depth_band_mm": None,
        "bottle_crop_status": "UNAVAILABLE",
        "bottle_crop_confidence": 0.0,
        "bottle_crop_quality": "unavailable",
        "bottle_crop_degraded": True,
        "bottle_crop_safe_for_dimensions": False,
        "bottle_crop_safe_for_contacts": False,
        "bottle_crop_safe_for_quality_support": False,
        "crop_background_leak_score": 0.0,
        "crop_hand_or_gripper_risk": "unknown",
        "crop_table_plane_risk": "unknown",
        "crop_extent_profile_score": None,
        **_finite_depth_stats_mm(depth_m, raw_mask),
    }
    if raw_pixels <= 0 or valid_pixels <= 0:
        return np.zeros_like(raw_mask, dtype=bool), {**status, "reason": "no_valid_depth_in_roi", "bottle_crop_reason": "no_valid_depth_in_roi"}
    depth_values = depth_m[valid_in_roi]
    median_m = float(np.median(depth_values))
    seed_band_m = max(float(POINTCLOUD_BOTTLE_DEPTH_BAND_MM) / 1000.0, 0.03)
    status["bottle_depth_median_mm"] = round(median_m * 1000.0, 1)
    if _source_is_selective_mask(source) or source.startswith("locked_"):
        # Masks are already spatially selective; keep a generous depth window to avoid
        # punching holes in the bottle surface.
        band_m = max(seed_band_m, 0.14)
        status["bottle_depth_band_mm"] = round(band_m * 1000.0, 1)
        depth_consistent = valid_in_roi & (np.abs(depth_m - median_m) <= band_m)
        depth_consistent = _morph_mask(depth_consistent)
        component_mask, component_status = _component_stats(depth_consistent, raw_mask)
        status.update(component_status)
        status["bottle_connected_component_count"] = int(component_status.get("component_count") or 0)
        if not np.any(component_mask):
            return depth_consistent, {**status, "reason": "empty_after_depth_component_filter", "bottle_crop_reason": "empty_after_depth_component_filter"}
        locked = source.startswith("locked_")
        contour = "contour" in str(source)
        status.update({
            "bottle_crop_status": "VALID_LOCKED_MASK_CROP" if locked else ("VALID_CONTOUR_CROP" if contour else "VALID_MASK_CROP"),
            "bottle_crop_confidence": 0.78 if locked else 0.95,
            "bottle_crop_quality": "locked_mask" if locked else ("contour_mask" if contour else "cleaned_mask"),
            "bottle_crop_degraded": bool(locked),
            "bottle_crop_safe_for_dimensions": True,
            "bottle_crop_safe_for_contacts": not locked,
            "bottle_crop_safe_for_quality_support": not locked,
            "crop_background_leak_score": round(max(0.0, float(component_status.get("border_touch_ratio") or 0.0) - 0.2), 4),
            "crop_hand_or_gripper_risk": "low" if (component_status.get("component_count") or 0) <= 2 else "medium",
            "crop_table_plane_risk": "low",
        })
        return component_mask.astype(bool), {**status, "reason": "valid", "bottle_crop_reason": "valid"}

    bbox = _bbox_from_mask(raw_mask)
    if bbox is None:
        return np.zeros_like(raw_mask, dtype=bool), {**status, "reason": "invalid_bbox_mask", "bottle_crop_reason": "invalid_bbox_mask"}
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    seed_fraction = float(np.clip(BOTTLE_BBOX_SEED_FRACTION, 0.10, 0.90))
    sw, sh = max(3, int(round(bw * seed_fraction))), max(3, int(round(bh * seed_fraction)))
    sx1 = int(round((x1 + x2 - sw) / 2.0))
    sy1 = int(round((y1 + y2 - sh) / 2.0))
    sx2, sy2 = min(depth_m.shape[1], sx1 + sw), min(depth_m.shape[0], sy1 + sh)
    sx1, sy1 = max(0, sx1), max(0, sy1)
    seed = np.zeros_like(raw_mask, dtype=bool)
    seed[sy1:sy2, sx1:sx2] = True
    seed &= raw_mask
    seed_values = depth_m[seed & valid_depth]
    if seed_values.size < 4:
        seed_values = depth_values
    seed_median_m = float(np.median(seed_values))
    seed_mad_m = float(np.median(np.abs(seed_values - seed_median_m))) if seed_values.size else 0.0
    band_mm = float(np.clip(seed_mad_m * 1000.0 * float(BOTTLE_DEPTH_MAD_MULTIPLIER), BOTTLE_DEPTH_BAND_MIN_MM, BOTTLE_DEPTH_BAND_MAX_MM))
    band_m = band_mm / 1000.0
    status["bottle_depth_median_mm"] = round(seed_median_m * 1000.0, 1)
    status["bottle_depth_band_mm"] = round(band_mm, 1)
    depth_consistent = valid_in_roi & (np.abs(depth_m - median_m) <= band_m)
    depth_consistent = valid_in_roi & (np.abs(depth_m - seed_median_m) <= band_m)
    depth_consistent = _morph_mask(depth_consistent)
    component_mask, component_status = _component_stats(depth_consistent, seed)
    component_pixels = int(np.count_nonzero(component_mask))
    component_ratio = float(component_pixels / max(raw_pixels, 1))
    border_touch = float(component_status.get("border_touch_ratio") or 0.0)
    component_count = int(component_status.get("component_count") or 0)
    status.update(component_status)
    status["bottle_connected_component_count"] = component_count
    status["bottle_component_ratio"] = round(component_ratio, 4)
    status["crop_background_leak_score"] = round(min(1.0, border_touch + max(0.0, 1.0 - float(component_status.get("largest_component_ratio") or 0.0))), 4)
    status["crop_table_plane_risk"] = "high" if border_touch > BOTTLE_MAX_BORDER_TOUCH_RATIO else "medium" if border_touch > 0.18 else "low"
    status["crop_hand_or_gripper_risk"] = "high" if component_count > 3 else "medium" if component_count > 1 else "low"
    reason = "valid_depth_component"
    crop_status = "DEGRADED_BBOX_CROP"
    confidence = 0.62
    if component_pixels < POINTCLOUD_BOTTLE_MIN_POINTS:
        reason = "too_few_bbox_component_points"
        crop_status = "INVALID"
        confidence = 0.0
    elif component_ratio < float(BOTTLE_MIN_COMPONENT_RATIO):
        reason = "bbox_component_too_small"
        crop_status = "LOW_CONFIDENCE_BBOX_CROP"
        confidence = 0.28
    elif border_touch > float(BOTTLE_MAX_BORDER_TOUCH_RATIO):
        reason = "bbox_component_touches_border"
        crop_status = "LOW_CONFIDENCE_BBOX_CROP"
        confidence = 0.35
    status.update({
        "bottle_crop_status": crop_status,
        "bottle_crop_confidence": round(float(confidence), 3),
        "bottle_crop_quality": "bbox_depth_component" if crop_status == "DEGRADED_BBOX_CROP" else "bbox_low_confidence",
        "bottle_crop_degraded": True,
        "bottle_crop_safe_for_dimensions": crop_status == "DEGRADED_BBOX_CROP",
        "bottle_crop_safe_for_contacts": False,
        "bottle_crop_safe_for_quality_support": False,
    })
    return component_mask.astype(bool), {**status, "reason": reason, "bottle_crop_reason": reason}


def _crop_pointcloud_with_mask(
    points: np.ndarray,
    colors: np.ndarray,
    valid_mask: np.ndarray | None,
    crop_mask_2d: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_mask is None or crop_mask_2d is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    flat_crop = np.asarray(crop_mask_2d, dtype=bool).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool)
    if len(flat_crop) != len(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    keep = flat_crop[valid]
    return points[keep], colors[keep]


def crop_pointcloud_to_bottle_roi(
    points: np.ndarray,
    colors: np.ndarray,
    bbox_or_mask: Any,
    *,
    valid_mask: np.ndarray | None = None,
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop point cloud points/colors to a bottle bbox or full-frame mask."""
    if valid_mask is None or image_shape is None:
        return points, colors
    flat_crop = _crop_mask_from_bbox(image_shape, bbox_or_mask)
    if flat_crop is None or len(flat_crop) != len(valid_mask):
        return points, colors
    keep = flat_crop[np.asarray(valid_mask, dtype=bool)]
    return points[keep], colors[keep]


def _known_bottle_profile() -> dict[str, Any]:
    profile = load_named_config("known_bottle_profile.json", default={}) or {}
    legacy = profile.get("legacy_profile")
    if legacy:
        try:
            legacy_path = (CONFIG_DIR / str(legacy)).resolve()
            if legacy_path.exists():
                with open(legacy_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        profile = {**profile, **loaded}
        except Exception:
            pass
    return profile if isinstance(profile, dict) else {}


def _range_from_profile(profile: dict[str, Any], names: tuple[str, ...], fallback: tuple[float, float]) -> tuple[float, float]:
    for name in names:
        value = profile.get(name)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                pass
        if isinstance(value, dict):
            lo = value.get("min") or value.get("min_mm")
            hi = value.get("max") or value.get("max_mm")
            try:
                if lo is not None and hi is not None:
                    return float(lo), float(hi)
            except Exception:
                pass
    return fallback


def _profile_agreement_score(oriented_extents_mm: np.ndarray, point_count: int) -> tuple[float, str]:
    profile = _known_bottle_profile()
    dims = sorted([float(v) for v in np.asarray(oriented_extents_mm).reshape(-1) if np.isfinite(v)], reverse=True)
    if len(dims) < 3:
        return 0.0, "missing_oriented_extents"
    height_mm, width_mm, depth_mm = dims[0], dims[1], dims[2]
    height_range = _range_from_profile(profile, ("height_range_mm", "expected_height_range_mm", "expected_height_mm", "bottle_height_mm"), (70.0, 330.0))
    width_range = _range_from_profile(profile, ("diameter_range_mm", "width_range_mm", "expected_diameter_range_mm", "expected_diameter_mm", "contact_width_range_mm"), (15.0, 100.0))
    depth_range = _range_from_profile(profile, ("depth_range_mm", "thickness_range_mm", "expected_depth_range_mm"), (5.0, 90.0))

    checks = [
        height_range[0] <= height_mm <= height_range[1],
        width_range[0] <= width_mm <= width_range[1],
        depth_range[0] <= depth_mm <= depth_range[1],
        point_count >= POINTCLOUD_BOTTLE_MIN_POINTS,
    ]
    score = float(sum(1 for item in checks if item) / len(checks))
    if score >= 0.75:
        return round(score, 3), "matches_known_bottle_profile"
    return round(score, 3), f"profile_mismatch_h{height_mm:.1f}_w{width_mm:.1f}_d{depth_mm:.1f}_mm"


def estimate_bottle_3d_properties(points: np.ndarray) -> dict[str, Any]:
    """Estimate simple 3D centroid and extents in meters."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return {
            "bottle_cloud_valid": False,
            "bottle_3d_centroid_m": None,
            "bottle_3d_extents_m": None,
            "bottle_point_count": 0,
            "bottle_dimensions_valid": False,
            "bottle_principal_axis_m": None,
            "bottle_point_density": 0.0,
            "bottle_compactness": 0.0,
            "centroid_xyz_mm": None,
            "robust_centroid_xyz_mm": None,
            "axis_aligned_extents_mm": None,
            "oriented_extents_mm": None,
            "principal_axis": None,
            "geometry_profile_score": 0.0,
            "geometry_valid": False,
            "geometry_validation_reason": "no_points",
        }
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) == 0:
        return {
            "bottle_cloud_valid": False,
            "bottle_3d_centroid_m": None,
            "bottle_3d_extents_m": None,
            "bottle_point_count": 0,
            "bottle_dimensions_valid": False,
            "bottle_principal_axis_m": None,
            "bottle_point_density": 0.0,
            "bottle_compactness": 0.0,
            "centroid_xyz_mm": None,
            "robust_centroid_xyz_mm": None,
            "axis_aligned_extents_mm": None,
            "oriented_extents_mm": None,
            "principal_axis": None,
            "geometry_profile_score": 0.0,
            "geometry_valid": False,
            "geometry_validation_reason": "no_finite_points",
        }
    lo = np.percentile(finite, 2, axis=0)
    hi = np.percentile(finite, 98, axis=0)
    centroid = np.mean(finite, axis=0)
    robust_centroid = np.median(finite, axis=0)
    extents = np.maximum(0.0, hi - lo)
    trim_keep = np.all((finite >= lo) & (finite <= hi), axis=1)
    trimmed = finite[trim_keep] if np.count_nonzero(trim_keep) >= max(3, min(len(finite), POINTCLOUD_BOTTLE_MIN_POINTS // 2)) else finite
    centered = trimmed - np.mean(trimmed, axis=0)
    principal_axis = None
    oriented_extents = extents.copy()
    if len(finite) >= 3:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                principal_axis = [round(float(v / norm), 5) for v in axis]
            oriented = centered @ vh.T
            oriented_lo = np.percentile(oriented, 2, axis=0)
            oriented_hi = np.percentile(oriented, 98, axis=0)
            oriented_extents = np.maximum(0.0, oriented_hi - oriented_lo)
        except Exception:
            principal_axis = None
    volume = float(max(extents[0], 1e-5) * max(extents[1], 1e-5) * max(extents[2], 1e-5))
    density = float(len(finite) / volume)
    sorted_extents = sorted([float(v) for v in extents], reverse=True)
    compactness = float(sorted_extents[-1] / max(sorted_extents[0], 1e-6)) if sorted_extents else 0.0
    dims_mm = oriented_extents * 1000.0
    max_dim_mm = float(np.max(dims_mm)) if dims_mm.size else 0.0
    min_dim_mm = float(np.min(dims_mm)) if dims_mm.size else 0.0
    profile_score, profile_reason = _profile_agreement_score(dims_mm, len(finite))
    dimensions_valid = bool(profile_score >= 0.5 and 25.0 <= max_dim_mm <= 420.0 and 2.0 <= min_dim_mm <= 180.0 and len(finite) >= POINTCLOUD_BOTTLE_MIN_POINTS)
    return {
        "bottle_cloud_valid": True,
        "bottle_3d_centroid_m": [round(float(v), 5) for v in centroid],
        "bottle_3d_extents_m": {
            "width_x": round(float(extents[0]), 5),
            "height_y": round(float(extents[1]), 5),
            "depth_z": round(float(extents[2]), 5),
        },
        "bottle_point_count": int(len(finite)),
        "bottle_principal_axis_m": principal_axis,
        "bottle_point_density": round(density, 1),
        "bottle_compactness": round(compactness, 4),
        "bottle_dimensions_valid": dimensions_valid,
        "bottle_dimensions_warning": None if dimensions_valid else "outside_expected_bottle_profile_or_too_few_points",
        "centroid_xyz_mm": [round(float(v) * 1000.0, 1) for v in centroid],
        "robust_centroid_xyz_mm": [round(float(v) * 1000.0, 1) for v in robust_centroid],
        "axis_aligned_extents_mm": [round(float(v) * 1000.0, 1) for v in extents],
        "oriented_extents_mm": [round(float(v) * 1000.0, 1) for v in oriented_extents],
        "principal_axis": principal_axis,
        "geometry_profile_score": profile_score,
        "geometry_valid": dimensions_valid,
        "geometry_validation_reason": profile_reason if dimensions_valid else profile_reason,
    }


def _pixel_depth_mm(depth: np.ndarray, intrinsics: dict[str, Any] | None, point_px: Any, radius_px: int = POINTCLOUD_CONTACT_PATCH_RADIUS_PX) -> float | None:
    if depth is None or point_px is None:
        return None
    try:
        x, y = [float(v) for v in point_px[:2]]
    except Exception:
        return None
    depth_m = _depth_meters(np.asarray(depth), intrinsics or {})
    h, w = depth_m.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    r = max(1, int(radius_px))
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return round(float(np.median(valid)) * 1000.0, 1)


def _pixel_patch_xyz_m(depth: np.ndarray, intrinsics: dict[str, Any] | None, point_px: Any, radius_px: int = POINTCLOUD_CONTACT_PATCH_RADIUS_PX) -> np.ndarray:
    if depth is None or point_px is None or not _intrinsics_complete(intrinsics):
        return np.empty((0, 3), dtype=np.float32)
    try:
        x, y = [float(v) for v in point_px[:2]]
    except Exception:
        return np.empty((0, 3), dtype=np.float32)
    depth_m = _depth_meters(np.asarray(depth), intrinsics or {})
    h, w = depth_m.shape[:2]
    cx_i, cy_i = int(round(x)), int(round(y))
    r = max(1, int(radius_px))
    x1, x2 = max(0, cx_i - r), min(w, cx_i + r + 1)
    y1, y2 = max(0, cy_i - r), min(h, cy_i + r + 1)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.float32)
    patch = depth_m[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    yy, xx = np.indices(patch.shape, dtype=np.float32)
    xx = xx + float(x1)
    yy = yy + float(y1)
    z = patch
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    xyz = np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z), axis=-1)
    return xyz[valid].astype(np.float32, copy=False)


def _pixel_to_xyz_m(point_px: Any, depth_mm: float | None, intrinsics: dict[str, Any] | None) -> list[float] | None:
    if point_px is None or depth_mm is None or not _intrinsics_complete(intrinsics):
        return None
    try:
        x, y = [float(v) for v in point_px[:2]]
        z = float(depth_mm) / 1000.0
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    except Exception:
        return None
    if z <= 0 or fx == 0 or fy == 0:
        return None
    return [round((x - cx) * z / fx, 5), round((y - cy) * z / fy, 5), round(z, 5)]


def _nearest_cloud_distance_mm(point_xyz_m: Any, bottle_points: np.ndarray) -> float | None:
    if point_xyz_m is None or bottle_points is None or len(bottle_points) == 0:
        return None
    try:
        point = np.asarray(point_xyz_m, dtype=np.float32).reshape(1, 3)
        cloud = np.asarray(bottle_points, dtype=np.float32).reshape((-1, 3))
        cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
        if len(cloud) == 0:
            return None
        distances = np.linalg.norm(cloud - point, axis=1)
        return round(float(np.min(distances)) * 1000.0, 1)
    except Exception:
        return None


def _first_point(*values: Any) -> list[float] | None:
    for value in values:
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return [float(arr[0]), float(arr[1])]
    return None


def _patch_polygon_from_report(report: dict[str, Any] | None, side: str) -> Any:
    report = report or {}
    target = report.get("target_clamp") or {}
    lock = report.get("target_lock") or {}
    locked = lock.get("locked_target") or {}
    lock_target = lock.get("target_clamp") if isinstance(lock.get("target_clamp"), dict) else {}
    model = report.get("bottle_contact_model") or lock.get("bottle_contact_model") or locked.get("bottle_contact_model") or {}
    names = (
        f"{side}_clamp_patch",
        f"{side}_contact_patch",
    )
    for parent in (target, lock_target, model):
        for name in names:
            patch = parent.get(name) if isinstance(parent, dict) else None
            if isinstance(patch, dict):
                poly = patch.get("polygon") or patch.get("patch_polygon_px")
                if poly:
                    return poly
    return None


def _contact_points_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    target = report.get("target_clamp") or {}
    lock = report.get("target_lock") or {}
    tracking = (lock.get("current_tracked_target") or {}) if isinstance(lock, dict) else {}
    left_patch = target.get("left_clamp_patch") if isinstance(target.get("left_clamp_patch"), dict) else {}
    right_patch = target.get("right_clamp_patch") if isinstance(target.get("right_clamp_patch"), dict) else {}
    left = _first_point(
        target.get("left_contact_px"),
        left_patch.get("center_px"),
        report.get("left_contact_px"),
        tracking.get("current_left_edge_point"),
    )
    right = _first_point(
        target.get("right_contact_px"),
        right_patch.get("center_px"),
        report.get("right_contact_px"),
        tracking.get("current_right_edge_point"),
    )
    center = _first_point(target.get("clamp_center_px"), report.get("object_center_px"), report.get("bottle_center_px"))
    if center is None and left is not None and right is not None:
        center = [round((left[0] + right[0]) / 2.0, 2), round((left[1] + right[1]) / 2.0, 2)]
    return {
        "left_contact_px": left,
        "right_contact_px": right,
        "object_center_px": center,
        "left_contact_patch_polygon": _patch_polygon_from_report(report, "left"),
        "right_contact_patch_polygon": _patch_polygon_from_report(report, "right"),
    }


def _contact_3d_summary(
    report: dict[str, Any] | None,
    depth: np.ndarray,
    intrinsics: dict[str, Any] | None,
    bottle_points: np.ndarray,
) -> dict[str, Any]:
    contacts = _contact_points_from_report(report)
    left_px = contacts.get("left_contact_px")
    right_px = contacts.get("right_contact_px")
    center_px = contacts.get("object_center_px")
    left_depth = _pixel_depth_mm(depth, intrinsics, left_px)
    right_depth = _pixel_depth_mm(depth, intrinsics, right_px)
    center_depth = _pixel_depth_mm(depth, intrinsics, center_px)
    left_patch_xyz = _pixel_patch_xyz_m(depth, intrinsics, left_px)
    right_patch_xyz = _pixel_patch_xyz_m(depth, intrinsics, right_px)
    left_xyz = [round(float(v), 5) for v in np.median(left_patch_xyz, axis=0)] if len(left_patch_xyz) else _pixel_to_xyz_m(left_px, left_depth, intrinsics)
    right_xyz = [round(float(v), 5) for v in np.median(right_patch_xyz, axis=0)] if len(right_patch_xyz) else _pixel_to_xyz_m(right_px, right_depth, intrinsics)
    center_xyz = _pixel_to_xyz_m(center_px, center_depth, intrinsics)
    width_mm = None
    contact_quality = "invalid"
    left_cloud_distance = _nearest_cloud_distance_mm(left_xyz, bottle_points)
    right_cloud_distance = _nearest_cloud_distance_mm(right_xyz, bottle_points)
    height_difference_mm = None
    opposite_sides = None
    if left_xyz is not None and right_xyz is not None:
        left_arr = np.asarray(left_xyz, dtype=np.float32)
        right_arr = np.asarray(right_xyz, dtype=np.float32)
        width_mm = round(float(np.linalg.norm(right_arr - left_arr)) * 1000.0, 1)
        height_difference_mm = round(abs(float(left_arr[1] - right_arr[1])) * 1000.0, 1)
        if center_xyz is not None:
            center_arr_for_side = np.asarray(center_xyz, dtype=np.float32)
            side_axis = right_arr - left_arr
            if float(np.linalg.norm(side_axis)) > 1e-9:
                opposite_sides = bool(np.dot(left_arr - center_arr_for_side, side_axis) < 0 and np.dot(right_arr - center_arr_for_side, side_axis) > 0)
        on_cloud = (
            left_cloud_distance is not None
            and right_cloud_distance is not None
            and left_cloud_distance <= 25.0
            and right_cloud_distance <= 25.0
            and len(bottle_points) >= POINTCLOUD_BOTTLE_MIN_POINTS
        )
        if on_cloud and (opposite_sides is not False) and (height_difference_mm is None or height_difference_mm <= 35.0):
            contact_quality = "valid"
        elif len(bottle_points) < POINTCLOUD_BOTTLE_MIN_POINTS:
            contact_quality = "valid_contacts_bottle_cloud_weak"
        elif not on_cloud:
            contact_quality = "contacts_not_on_filtered_bottle_cloud"
        elif opposite_sides is False:
            contact_quality = "contacts_not_on_opposite_sides"
        else:
            contact_quality = "contact_height_mismatch"
    elif left_px is None or right_px is None:
        contact_quality = "missing_contact_pixels"
    elif left_depth is None or right_depth is None:
        contact_quality = "invalid_contact_depth"
    normals = {"left": None, "right": None}
    if center_xyz is not None:
        center_arr = np.asarray(center_xyz, dtype=np.float32)
        for side, xyz in (("left", left_xyz), ("right", right_xyz)):
            if xyz is None:
                continue
            vec = center_arr - np.asarray(xyz, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                normals[side] = [round(float(v), 5) for v in (vec / norm)]
    return {
        **contacts,
        "left_contact_depth_mm": left_depth,
        "right_contact_depth_mm": right_depth,
        "object_center_depth_mm": center_depth,
        "left_contact_xyz_m": left_xyz,
        "right_contact_xyz_m": right_xyz,
        "object_center_xyz_m": center_xyz,
        "left_contact_xyz_mm": _meters_list_to_mm(left_xyz),
        "right_contact_xyz_mm": _meters_list_to_mm(right_xyz),
        "object_center_xyz_mm": _meters_list_to_mm(center_xyz),
        "left_contact_cloud_distance_mm": left_cloud_distance,
        "right_contact_cloud_distance_mm": right_cloud_distance,
        "contact_height_difference_mm": height_difference_mm,
        "left_normal": normals["left"],
        "right_normal": normals["right"],
        "local_normals_source": "center_to_contact_approximate",
        "contact_width_3d_mm": width_mm,
        "contact_depth_valid": bool(left_depth is not None and right_depth is not None),
        "contacts_on_opposite_sides": opposite_sides,
        "contacts_on_filtered_bottle_cloud": bool(
            left_cloud_distance is not None
            and right_cloud_distance is not None
            and left_cloud_distance <= 25.0
            and right_cloud_distance <= 25.0
        ),
        "contact_points_on_bottle_cloud": bool(left_xyz is not None and right_xyz is not None and len(bottle_points) >= POINTCLOUD_BOTTLE_MIN_POINTS),
        "contact_3d_valid": contact_quality == "valid",
        "contact_3d_reason": contact_quality,
        "contact_3d_quality": contact_quality,
    }


def _meters_list_to_mm(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    try:
        return [round(float(v) * 1000.0, 1) for v in values[:3]]
    except Exception:
        return None


def _meter_extents_to_mm(extents: Any) -> dict[str, float] | None:
    if not isinstance(extents, dict):
        return None
    out: dict[str, float] = {}
    for key, value in extents.items():
        try:
            out[key] = round(float(value) * 1000.0, 1)
        except Exception:
            continue
    return out or None


def _bbox_area_px(bbox_or_mask: Any) -> float | None:
    if isinstance(bbox_or_mask, np.ndarray):
        return float(np.count_nonzero(bbox_or_mask))
    if isinstance(bbox_or_mask, dict):
        bbox_or_mask = bbox_or_mask.get("bbox_xyxy") or bbox_or_mask.get("bottle_extent_bbox_xyxy") or bbox_or_mask.get("bbox")
    if not isinstance(bbox_or_mask, (list, tuple)) or len(bbox_or_mask) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_or_mask[:4]]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    except Exception:
        return None


def _sample_points_for_preview(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3)) if np.asarray(points).size else np.empty((0, 3), dtype=np.float32)
    colors = np.zeros((len(points), 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32).reshape((-1, 3))
    if len(colors) != len(points):
        colors = np.zeros((len(points), 3), dtype=np.float32)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    colors = colors[finite]
    if len(points) > max(1, int(max_points)):
        idx = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[idx]
        colors = colors[idx]
    return points, np.clip(colors, 0.0, 1.0)


def _point_bounds(points: np.ndarray, bottle_points: np.ndarray | None = None) -> dict[str, Any]:
    combined = np.asarray(points, dtype=np.float32).reshape((-1, 3)) if np.asarray(points).size else np.empty((0, 3), dtype=np.float32)
    bottle_arr = (
        np.asarray(bottle_points, dtype=np.float32).reshape((-1, 3))
        if bottle_points is not None and np.asarray(bottle_points).size
        else np.empty((0, 3), dtype=np.float32)
    )
    if len(bottle_arr):
        combined = np.concatenate([combined, bottle_arr], axis=0) if len(combined) else bottle_arr
    if not len(combined):
        return {"min": None, "max": None, "center": None, "span": None}
    finite = combined[np.all(np.isfinite(combined), axis=1)]
    if not len(finite):
        return {"min": None, "max": None, "center": None, "span": None}
    lo = np.percentile(finite, 2, axis=0)
    hi = np.percentile(finite, 98, axis=0)
    center = np.mean(finite, axis=0)
    span = np.maximum(0.0, hi - lo)
    return {
        "min": [round(float(v), 5) for v in lo],
        "max": [round(float(v), 5) for v in hi],
        "center": [round(float(v), 5) for v in center],
        "span": [round(float(v), 5) for v in span],
    }


def _select_preview_color_mode(points: np.ndarray, colors: np.ndarray | None, requested: str = POINTCLOUD_PREVIEW_COLOR_MODE) -> str:
    requested = (requested or "auto").strip().lower()
    if requested in {"rgb", "depth", "height", "white"}:
        return requested
    colors_arr = np.asarray(colors, dtype=np.float32).reshape((-1, 3)) if colors is not None and np.asarray(colors).size else np.empty((0, 3))
    if len(colors_arr):
        brightness = float(np.mean(np.clip(colors_arr, 0.0, 1.0)))
        if brightness >= 0.26:
            return "rgb"
    points_arr = np.asarray(points, dtype=np.float32).reshape((-1, 3)) if np.asarray(points).size else np.empty((0, 3))
    return "depth" if len(points_arr) else "white"


def _normalized(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.percentile(finite, 2) if lo is None else lo)
    high = float(np.percentile(finite, 98) if hi is None else hi)
    if high <= low:
        high = low + 1e-6
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _preview_bgr_colors(points: np.ndarray, colors: np.ndarray | None, mode: str) -> np.ndarray:
    import cv2

    points_arr = np.asarray(points, dtype=np.float32).reshape((-1, 3)) if np.asarray(points).size else np.empty((0, 3), dtype=np.float32)
    count = len(points_arr)
    if count == 0:
        return np.empty((0, 3), dtype=np.uint8)
    mode = _select_preview_color_mode(points_arr, colors, mode)
    if mode == "rgb":
        rgb = np.zeros((count, 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32).reshape((-1, 3))
        if len(rgb) != count:
            rgb = np.zeros((count, 3), dtype=np.float32)
        rgb = np.clip(rgb * float(POINTCLOUD_PREVIEW_BRIGHTNESS_BOOST), 0.0, 1.0)
        rgb = np.maximum(rgb, 0.10)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)[:, ::-1]
    if mode == "white":
        return np.full((count, 3), (232, 238, 245), dtype=np.uint8)
    axis_values = points_arr[:, 2] if mode == "depth" else -points_arr[:, 1]
    norm = (_normalized(axis_values) * 255.0).astype(np.uint8).reshape((-1, 1))
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    return cv2.applyColorMap(norm, cmap).reshape((-1, 3))


def _blank_preview(width: int, height: int, title: str, message: str) -> np.ndarray:
    import cv2

    img = np.full((int(height), int(width), 3), (22, 28, 38), dtype=np.uint8)
    cv2.putText(img, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 241, 248), 2, cv2.LINE_AA)
    cv2.putText(img, message, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 174, 192), 1, cv2.LINE_AA)
    return img


def _draw_projection_panel(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    bottle_points: np.ndarray,
    axes: tuple[int, int],
    title: str,
    *,
    color_mode: str = POINTCLOUD_PREVIEW_COLOR_MODE,
    contacts_xyz: dict[str, Any] | None = None,
) -> None:
    import cv2

    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), (44, 55, 72), 1, cv2.LINE_AA)
    cv2.putText(img, title, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 241, 248), 1, cv2.LINE_AA)
    points_for_bounds = scene_points
    if len(bottle_points):
        points_for_bounds = np.concatenate([scene_points, bottle_points], axis=0) if len(scene_points) else bottle_points
    if len(points_for_bounds) == 0:
        cv2.putText(img, "no points", (x0 + 10, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (154, 165, 182), 1, cv2.LINE_AA)
        return
    pad = 16
    px0, py0, px1, py1 = x0 + pad, y0 + 40, x1 - pad, y1 - pad
    vals = points_for_bounds[:, list(axes)]
    lo = np.percentile(vals, 3, axis=0)
    hi = np.percentile(vals, 97, axis=0)
    raw_span = np.maximum(hi - lo, 1e-5)
    max_span = float(np.max(raw_span))
    mid = (lo + hi) / 2.0
    span = np.array([max_span, max_span], dtype=np.float32)
    lo = mid - span / 2.0
    hi = mid + span / 2.0
    for frac in (0.25, 0.5, 0.75):
        gx = int(round(px0 + frac * max(1, px1 - px0)))
        gy = int(round(py0 + frac * max(1, py1 - py0)))
        cv2.line(img, (gx, py0), (gx, py1), (24, 33, 48), 1, cv2.LINE_AA)
        cv2.line(img, (px0, gy), (px1, gy), (24, 33, 48), 1, cv2.LINE_AA)

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected = points[:, list(axes)]
        nx = np.clip((projected[:, 0] - lo[0]) / span[0], 0.0, 1.0)
        ny = np.clip((projected[:, 1] - lo[1]) / span[1], 0.0, 1.0)
        xs = np.round(px0 + nx * max(1, px1 - px0)).astype(np.int32)
        ys = np.round(py1 - ny * max(1, py1 - py0)).astype(np.int32)
        return xs, ys

    if len(scene_points):
        xs, ys = project(scene_points)
        keep = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        bgr = _preview_bgr_colors(scene_points, scene_colors, color_mode)
        img[ys[keep], xs[keep]] = bgr[keep]
        radius = max(1, int(round(float(POINTCLOUD_PREVIEW_POINT_SIZE) / 2.0)))
        if radius > 1 and np.count_nonzero(keep):
            kept_idx = np.flatnonzero(keep)
            step = max(1, len(kept_idx) // 2500)
            for idx in kept_idx[::step]:
                cv2.circle(img, (int(xs[idx]), int(ys[idx])), radius, tuple(int(v) for v in bgr[idx]), -1, cv2.LINE_AA)
    if len(bottle_points):
        xs, ys = project(bottle_points)
        keep = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        kept_x = xs[keep]
        kept_y = ys[keep]
        if POINTCLOUD_PREVIEW_DRAW_BBOX and len(kept_x):
            cv2.rectangle(
                img,
                (int(np.min(kept_x)), int(np.min(kept_y))),
                (int(np.max(kept_x)), int(np.max(kept_y))),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        for x, y in zip(kept_x[:: max(1, len(kept_x) // 1500)], kept_y[:: max(1, len(kept_y) // 1500)]):
            cv2.circle(img, (int(x), int(y)), max(2, int(round(POINTCLOUD_PREVIEW_POINT_SIZE))), (255, 0, 255), -1, cv2.LINE_AA)
        if len(kept_x):
            cv2.drawMarker(
                img,
                (int(np.median(kept_x)), int(np.median(kept_y))),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=13,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
    contacts_xyz = contacts_xyz or {}
    contact_points = []
    for label, key, color in (
        ("L", "left_contact_xyz_m", (255, 80, 255)),
        ("R", "right_contact_xyz_m", (80, 220, 255)),
    ):
        xyz = contacts_xyz.get(key)
        if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
            try:
                arr = np.asarray(xyz[:3], dtype=np.float32).reshape(1, 3)
                xs, ys = project(arr)
                px, py = int(xs[0]), int(ys[0])
                if x0 <= px < x1 and y0 <= py < y1:
                    cv2.circle(img, (px, py), 5, color, -1, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 7, (10, 14, 23), 1, cv2.LINE_AA)
                    cv2.putText(img, label, (px + 7, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                    contact_points.append((px, py))
            except Exception:
                pass
    if len(contact_points) == 2:
        cv2.line(img, contact_points[0], contact_points[1], (255, 255, 255), 1, cv2.LINE_AA)
    if POINTCLOUD_PREVIEW_DRAW_AXES:
        cv2.line(img, (px0, py1), (px0 + 28, py1), (90, 180, 255), 1, cv2.LINE_AA)
        cv2.line(img, (px0, py1), (px0, py1 - 28), (90, 255, 170), 1, cv2.LINE_AA)
        cv2.putText(img, "m", (px0 + 32, py1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (154, 165, 182), 1, cv2.LINE_AA)


def save_pointcloud_preview(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    bottle_points: np.ndarray | None = None,
    bottle_colors: np.ndarray | None = None,
    title: str = "RealSense Point Cloud",
    width: int = POINTCLOUD_PREVIEW_WIDTH,
    height: int = POINTCLOUD_PREVIEW_HEIGHT,
    max_points: int = POINTCLOUD_PREVIEW_MAX_POINTS,
    color_mode: str = POINTCLOUD_PREVIEW_COLOR_MODE,
    density_mode: str | None = None,
    contacts_xyz: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a lightweight three-view JPEG point-cloud preview.

    The preview is a visualization artifact only; it never feeds tracking or
    quality math.  It intentionally avoids Open3D so a missing 3D viewer stack
    does not break the live demo.
    """
    path = Path(path)
    try:
        import cv2
    except Exception as exc:
        return {"saved": False, "error": f"opencv_unavailable_for_pointcloud_preview: {exc}"}
    raw_scene_points = np.asarray(points, dtype=np.float32).reshape((-1, 3)) if np.asarray(points).size else np.empty((0, 3), dtype=np.float32)
    raw_scene_colors = np.zeros((len(raw_scene_points), 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32).reshape((-1, 3))
    raw_scene_points, raw_scene_colors, outlier_status = _preview_depth_filter(raw_scene_points, raw_scene_colors)
    scene_points, scene_colors = _sample_points_for_preview(raw_scene_points, raw_scene_colors, max_points)
    bottle_points_arr, bottle_colors_arr = _sample_points_for_preview(
        np.empty((0, 3), dtype=np.float32) if bottle_points is None else bottle_points,
        bottle_colors,
        POINTCLOUD_BOTTLE_MAX_POINTS if density_mode == "bottle_detail" else max_points,
    )
    selected_mode = _select_preview_color_mode(scene_points if len(scene_points) else bottle_points_arr, scene_colors, color_mode)
    view_bounds = _point_bounds(scene_points, bottle_points_arr)
    if len(scene_points) == 0 and len(bottle_points_arr) == 0:
        img = _blank_preview(width, height, title, "Point cloud unavailable: no valid RGB-D points.")
        visible = False
        reason = "no_valid_points"
    else:
        visible = True
        reason = "ok"
        img = np.full((int(height), int(width), 3), (10, 14, 23), dtype=np.uint8)
        cv2.putText(img, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 246, 252), 2, cv2.LINE_AA)
        count_text = f"scene points {len(scene_points)}"
        if len(bottle_points_arr):
            count_text += f" | bottle crop {len(bottle_points_arr)}"
        count_text += f" | color {selected_mode}"
        cv2.putText(img, count_text, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (164, 176, 194), 1, cv2.LINE_AA)
        panel_top = 78
        gap = 10
        panel_w = max(80, (int(width) - 2 * gap - 24) // 3)
        panels = [
            (12, panel_top, 12 + panel_w, int(height) - 12, (0, 2), "Top X/Z"),
            (12 + panel_w + gap, panel_top, 12 + 2 * panel_w + gap, int(height) - 12, (0, 1), "Front X/Y"),
            (12 + 2 * (panel_w + gap), panel_top, int(width) - 12, int(height) - 12, (2, 1), "Side Z/Y"),
        ]
        for x0, y0, x1, y1, axes, label in panels:
            _draw_projection_panel(
                img,
                (x0, y0, x1, y1),
                scene_points,
                scene_colors,
                bottle_points_arr,
                axes,
                label,
                color_mode=selected_mode,
                contacts_xyz=contacts_xyz,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.jpg"
    )
    ok = bool(cv2.imwrite(str(tmp), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88]))
    if ok:
        tmp.replace(path)
    elif tmp.exists():
        tmp.unlink(missing_ok=True)
    return {
        "saved": ok,
        "path": str(path),
        "width": int(width),
        "height": int(height),
        "rendered_scene_points": int(len(scene_points)),
        "rendered_bottle_points": int(len(bottle_points_arr)),
        "visible": bool(visible),
        "reason": reason,
        "color_mode": selected_mode,
        "view_bounds": view_bounds,
        "outlier_filter": outlier_status,
        "density_mode": density_mode or "standard",
        "error": None if ok else "cv2_imwrite_failed",
    }


def _resize_to_tile(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    target_w, target_h = size
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), (12, 17, 28), dtype=np.uint8)
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.full((target_h, target_w, 3), (12, 17, 28), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    tile[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return tile


def save_pointcloud_context_preview(
    path: str | Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, Any] | None,
    bbox: Any,
    scene_preview_path: str | Path,
    bottle_preview_path: str | Path,
    *,
    width: int = POINTCLOUD_PREVIEW_WIDTH,
    height: int = POINTCLOUD_PREVIEW_HEIGHT,
) -> dict[str, Any]:
    """Save a dashboard context image: RGB, depth, full cloud, and bottle crop."""
    path = Path(path)
    try:
        import cv2
    except Exception as exc:
        return {"saved": False, "error": f"opencv_unavailable_for_pointcloud_context: {exc}"}
    t0 = time.perf_counter()
    out_w, out_h = int(width), int(height)
    tile_w = max(160, (out_w - 36) // 2)
    tile_h = max(120, (out_h - 84) // 2)
    canvas = np.full((out_h, out_w, 3), (8, 13, 22), dtype=np.uint8)
    cv2.putText(canvas, "RGB-D + Point Cloud Context", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 246, 252), 2, cv2.LINE_AA)

    rgb_img = np.asarray(rgb)
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) if rgb_img.ndim == 3 else np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    if isinstance(bbox, dict):
        bbox = bbox.get("bbox_xyxy") or bbox.get("bottle_extent_bbox_xyxy") or bbox.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (80, 255, 120), 2, cv2.LINE_AA)
        except Exception:
            pass
    depth_m = _depth_meters(np.asarray(depth), intrinsics or {})
    depth_valid = _valid_depth_m_mask(depth_m)
    if np.any(depth_valid):
        d_norm = _normalized(depth_m, POINTCLOUD_PREVIEW_DEPTH_MIN_M, POINTCLOUD_PREVIEW_DEPTH_MAX_M)
        depth_vis = cv2.applyColorMap(np.clip(d_norm * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET)
        depth_vis[~depth_valid] = (8, 13, 22)
    else:
        depth_vis = np.full(rgb_bgr.shape, (8, 13, 22), dtype=np.uint8)
    scene_img = cv2.imread(str(scene_preview_path))
    bottle_img = cv2.imread(str(bottle_preview_path))
    if scene_img is None:
        scene_img = _blank_preview(tile_w, tile_h, "Full Scene Cloud", "waiting for point-cloud preview")
    if bottle_img is None:
        bottle_img = _blank_preview(tile_w, tile_h, "Bottle Cloud Crop", "waiting for bottle crop")

    tiles = [
        ("RGB + bottle bbox", _resize_to_tile(rgb_bgr, (tile_w, tile_h))),
        ("Aligned depth", _resize_to_tile(depth_vis, (tile_w, tile_h))),
        ("Full scene cloud", _resize_to_tile(scene_img, (tile_w, tile_h))),
        ("Bottle crop cloud", _resize_to_tile(bottle_img, (tile_w, tile_h))),
    ]
    positions = [(18, 54), (18 + tile_w + 12, 54), (18, 54 + tile_h + 30), (18 + tile_w + 12, 54 + tile_h + 30)]
    for (label, tile), (x, y) in zip(tiles, positions):
        canvas[y : y + tile_h, x : x + tile_w] = tile
        cv2.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), (58, 70, 90), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x + 1, y + tile_h - 27), (x + tile_w - 1, y + tile_h - 1), (8, 13, 22), -1)
        cv2.putText(canvas, label, (x + 8, y + tile_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 241, 248), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.jpg"
    )
    ok = bool(cv2.imwrite(str(tmp), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88]))
    if ok:
        tmp.replace(path)
    elif tmp.exists():
        tmp.unlink(missing_ok=True)
    return {
        "saved": ok,
        "path": str(path),
        "width": out_w,
        "height": out_h,
        "processing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "error": None if ok else "cv2_imwrite_failed",
    }


def _round_flat(values: np.ndarray, decimals: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    return np.round(arr.reshape(-1), int(decimals)).astype(float).tolist()


def pointcloud_live_json_payload(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    bottle_points: np.ndarray | None = None,
    bottle_colors: np.ndarray | None = None,
    max_points: int = POINTCLOUD_JSON_MAX_POINTS,
    timestamp: float | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact browser-facing point-cloud payload.

    Positions are meters in camera coordinates and colors are 0..1 floats.
    The browser intentionally receives sampled points, not the full cloud.
    """
    scene_points, scene_colors = _sample_points_for_preview(points, colors, max_points)
    bottle_points_arr, bottle_colors_arr = _sample_points_for_preview(
        np.empty((0, 3), dtype=np.float32) if bottle_points is None else bottle_points,
        bottle_colors,
        max_points,
    )
    combined_for_bounds = scene_points
    if len(bottle_points_arr):
        combined_for_bounds = np.concatenate([scene_points, bottle_points_arr], axis=0) if len(scene_points) else bottle_points_arr
    if len(combined_for_bounds):
        lo = np.percentile(combined_for_bounds, 2, axis=0)
        hi = np.percentile(combined_for_bounds, 98, axis=0)
        center = np.mean(combined_for_bounds, axis=0)
    else:
        lo = hi = center = np.zeros(3, dtype=np.float32)
    status = status or {}
    return {
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "status": "updated" if len(scene_points) or len(bottle_points_arr) else "empty",
        "coordinate_frame": "realsense_camera_m",
        "coordinate_frame_type": "current_camera_frame",
        "coordinate_frame_stable": False,
        "world_registered": False,
        "multiview_registered": False,
        "pointcloud_session_id": status.get("pointcloud_session_id"),
        "pointcloud_generation_id": status.get("pointcloud_generation_id"),
        "pointcloud_frame_id": status.get("pointcloud_frame_id"),
        "pointcloud_timestamp": status.get("pointcloud_timestamp") or timestamp,
        "pointcloud_freshness": status.get("pointcloud_freshness"),
        "viewer_watermark": status.get("viewer_watermark"),
        "units": "meters",
        "positions_format": "flat_xyz",
        "colors_format": "flat_rgb_float_0_1",
        "scene": {
            "point_count": int(len(scene_points)),
            "positions": _round_flat(scene_points, 5),
            "colors": _round_flat(np.clip(scene_colors, 0.0, 1.0), 3),
        },
        "bottle": {
            "point_count": int(len(bottle_points_arr)),
            "positions": _round_flat(bottle_points_arr, 5),
            "colors": _round_flat(np.clip(bottle_colors_arr, 0.0, 1.0), 3),
            "valid": bool(status.get("bottle_pointcloud_valid") or status.get("bottle_cloud_valid")),
            "raw_point_count": status.get("bottle_raw_point_count"),
            "filtered_point_count": status.get("bottle_filtered_point_count"),
            "display_point_count": int(len(bottle_points_arr)),
            "roi_source": status.get("bottle_roi_source") or status.get("bottle_pointcloud_roi_source"),
        },
        "contacts": {
            "left_xyz_m": status.get("left_contact_xyz_m"),
            "right_xyz_m": status.get("right_contact_xyz_m"),
            "center_xyz_m": status.get("object_center_xyz_m"),
            "left_px": status.get("left_contact_px"),
            "right_px": status.get("right_contact_px"),
            "width_mm": status.get("contact_width_3d_mm"),
            "valid": bool(
                status.get("contact_3d_valid")
                and status.get("cloud_valid_for_contact_validation")
            ),
            "quality": status.get("contact_3d_quality"),
        },
        "bounds": {
            "min": _round_flat(lo, 5),
            "max": _round_flat(hi, 5),
            "center": _round_flat(center, 5),
        },
        "source": {
            "backend": status.get("realsense_pointcloud_backend") or status.get("pointcloud_backend"),
            "fps": status.get("realsense_pointcloud_fps") or status.get("pointcloud_fps"),
            "original_scene_point_count": status.get("scene_raw_point_count") or status.get("realsense_pointcloud_point_count"),
            "filtered_scene_point_count": status.get("scene_filtered_point_count"),
            "display_scene_point_count": int(len(scene_points)),
            "original_bottle_point_count": status.get("bottle_raw_point_count") or status.get("bottle_pointcloud_point_count") or status.get("bottle_point_count"),
            "filtered_bottle_point_count": status.get("bottle_filtered_point_count") or status.get("bottle_pointcloud_point_count"),
            "display_bottle_point_count": int(len(bottle_points_arr)),
            "live_stride": status.get("pointcloud_live_stride"),
            "json_max_points": int(max_points),
            "color_mode": status.get("pointcloud_preview_color_mode") or POINTCLOUD_PREVIEW_COLOR_MODE,
            "scene_display_max_points": status.get("pointcloud_scene_display_max_points") or POINTCLOUD_SCENE_MAX_POINTS,
            "bottle_display_max_points": status.get("pointcloud_bottle_display_max_points") or POINTCLOUD_BOTTLE_MAX_POINTS,
            "bottle_density_mode": status.get("bottle_pointcloud_density_mode"),
        },
    }


def save_pointcloud_live_json(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    bottle_points: np.ndarray | None = None,
    bottle_colors: np.ndarray | None = None,
    max_points: int = POINTCLOUD_JSON_MAX_POINTS,
    timestamp: float | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the sampled point cloud JSON consumed by the dashboard."""
    path = Path(path)
    try:
        t0 = time.perf_counter()
        payload = pointcloud_live_json_payload(
            points,
            colors,
            bottle_points=bottle_points,
            bottle_colors=bottle_colors,
            max_points=max_points,
            timestamp=timestamp,
            status=status,
        )
        write_json_atomic(path, payload, indent=None)
        return {
            "saved": True,
            "path": str(path),
            "scene_points": payload["scene"]["point_count"],
            "bottle_points": payload["bottle"]["point_count"],
            "json_processing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "saved": False,
            "path": str(path),
            "scene_points": 0,
            "bottle_points": 0,
            "json_processing_ms": None,
            "error": str(exc),
        }


def downsample_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float = POINTCLOUD_VOXEL_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel downsample without requiring Open3D."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.zeros((len(points), 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32)
    if len(points) == 0 or voxel_size <= 0:
        return points, colors
    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_idx = np.unique(voxels, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx], colors[unique_idx]


def save_pointcloud_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    use_open3d: bool | None = None,
) -> bool:
    """Save an ASCII PLY point cloud; uses Open3D if available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.zeros((len(points), 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32)
    if len(points) == 0:
        return False
    use_open3d = POINTCLOUD_USE_OPEN3D if use_open3d is None else bool(use_open3d)
    o3d = _try_import_open3d() if use_open3d else None
    if o3d is not None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
        return bool(o3d.io.write_point_cloud(str(path), cloud, write_ascii=True))
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    return True


def pointcloud_summary(points: np.ndarray, valid_ratio: float | None = None) -> dict[str, Any]:
    """Summarize point cloud size, spread, and quality."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return {
            "point_count": 0,
            "valid_depth_ratio": round(float(valid_ratio or 0.0), 4),
            "pointcloud_quality_score": 0.0,
            "centroid_m": None,
            "extents_m": None,
        }
    finite = points[np.all(np.isfinite(points), axis=1)]
    centroid = np.mean(finite, axis=0) if len(finite) else np.zeros(3, dtype=np.float32)
    lo = np.percentile(finite, 2, axis=0) if len(finite) else np.zeros(3)
    hi = np.percentile(finite, 98, axis=0) if len(finite) else np.zeros(3)
    valid_depth_ratio = float(valid_ratio if valid_ratio is not None else min(1.0, len(finite) / max(len(points), 1)))
    quality = float(np.clip(valid_depth_ratio * 100.0, 0.0, 100.0))
    return {
        "point_count": int(len(finite)),
        "valid_depth_ratio": round(valid_depth_ratio, 4),
        "pointcloud_quality_score": round(quality, 1),
        "centroid_m": [round(float(v), 5) for v in centroid],
        "extents_m": [round(float(v), 5) for v in np.maximum(0.0, hi - lo)],
    }


@dataclass
class RollingPointCloudObservation:
    """One bottle observation retained without changing its camera frame."""

    timestamp: float
    frame_id: int | str | None
    target_id: str
    target_id_scope: str
    points: np.ndarray
    colors: np.ndarray
    valid_depth_ratio: float
    registration_status: str = "NOT_REGISTERED_DIAGNOSTIC"


@dataclass
class RollingObjectCloudAccumulator:
    """Bounded rolling diagnostics for a single persistently locked bottle.

    Observations remain separate because a hand-mounted RealSense changes
    camera frame as the hand moves.  This class deliberately does not concatenate
    those frames or expose geometry/contact/benchmark validity.  Registered
    multi-view fusion remains the responsibility of ``multiview_reconstruction``.
    """

    window_sec: float = POINTCLOUD_ROLLING_WINDOW_SEC
    max_points: int = POINTCLOUD_ROLLING_MAX_POINTS
    voxel_size_m: float = POINTCLOUD_ROLLING_VOXEL_SIZE_M
    min_valid_depth_ratio: float = POINTCLOUD_ROLLING_MIN_VALID_DEPTH_RATIO
    reset_on_target_change: bool = POINTCLOUD_ROLLING_RESET_ON_TARGET_CHANGE
    reset_on_bottle_moved: bool = POINTCLOUD_ROLLING_RESET_ON_BOTTLE_MOVED
    observations: deque[RollingPointCloudObservation] = field(default_factory=deque, init=False, repr=False)
    target_id: str | None = field(default=None, init=False)
    target_id_scope: str | None = field(default=None, init=False)
    last_reset_reason: str = field(default="initialization", init=False)
    last_reset_timestamp: float | None = field(default=None, init=False)
    last_rejection_reason: str | None = field(default=None, init=False)
    latest_accepted_timestamp: float | None = field(default=None, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def _prune_unlocked(self, now: float) -> None:
        cutoff = float(now) - max(0.0, float(self.window_sec))
        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()

    def _point_count_unlocked(self) -> int:
        return int(sum(len(observation.points) for observation in self.observations))

    def _reset_unlocked(self, reason: str, timestamp: float) -> None:
        self.observations.clear()
        self.target_id = None
        self.target_id_scope = None
        self.latest_accepted_timestamp = None
        self.last_rejection_reason = reason
        self.last_reset_reason = str(reason)
        self.last_reset_timestamp = float(timestamp)

    def reset(self, reason: str = "explicit_reset", timestamp: float | None = None) -> dict[str, Any]:
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            self._reset_unlocked(reason, now)
            return self._status_unlocked(now)

    def add_observation(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        *,
        timestamp: float | None = None,
        frame_id: int | str | None = None,
        target_id: str | None,
        target_id_scope: str | None,
        target_valid: bool,
        valid_depth_ratio: float | None,
        crop_valid: bool,
        snapshot_consistent: bool,
        bottle_moved: bool = False,
        registration_status: str = "NOT_REGISTERED_DIAGNOSTIC",
        explicit_reset: bool = False,
    ) -> dict[str, Any]:
        """Retain a validated observation as diagnostics, never fused geometry."""
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            self._prune_unlocked(now)
            if explicit_reset:
                self._reset_unlocked("explicit_reset", now)
                return self._status_unlocked(now, accepted=False)
            if not target_valid:
                if self.target_id is not None or self.observations:
                    self._reset_unlocked("target_invalid_or_lost", now)
                else:
                    self.last_rejection_reason = "valid_locked_target_required"
                return self._status_unlocked(now, accepted=False)
            target_text = str(target_id or "").strip()
            scope_text = str(target_id_scope or "").strip()
            if not target_text or not scope_text or "FRAME_LOCAL" in scope_text.upper():
                self.last_rejection_reason = "persistent_target_identity_required"
                return self._status_unlocked(now, accepted=False)
            if self.target_id is not None and target_text != self.target_id and self.reset_on_target_change:
                self._reset_unlocked("target_changed", now)
            if bottle_moved and self.reset_on_bottle_moved:
                self._reset_unlocked("bottle_moved", now)
                return self._status_unlocked(now, accepted=False)
            registration_text = str(registration_status or "NOT_REGISTERED_DIAGNOSTIC").upper()
            if registration_text in {"INVALID", "FAILED", "REGISTRATION_FAILED", "INVALID_REGISTRATION"}:
                self._reset_unlocked("invalid_registration", now)
                return self._status_unlocked(now, accepted=False)
            try:
                depth_ratio = float(valid_depth_ratio)
            except (TypeError, ValueError):
                depth_ratio = 0.0
            if depth_ratio < float(self.min_valid_depth_ratio):
                self.last_rejection_reason = "valid_depth_ratio_below_threshold"
                return self._status_unlocked(now, accepted=False)
            if not crop_valid:
                self.last_rejection_reason = "validated_bottle_crop_required"
                return self._status_unlocked(now, accepted=False)
            if not snapshot_consistent:
                self.last_rejection_reason = "snapshot_inconsistent"
                return self._status_unlocked(now, accepted=False)

            pts = np.asarray(points, dtype=np.float32)
            pts = pts.reshape((-1, 3)) if pts.size else np.empty((0, 3), dtype=np.float32)
            cols = np.zeros((len(pts), 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32)
            cols = cols.reshape((-1, 3)) if cols.size and len(cols) == len(pts) else np.zeros((len(pts), 3), dtype=np.float32)
            finite = np.all(np.isfinite(pts), axis=1)
            pts = pts[finite]
            cols = cols[finite]
            if len(pts) == 0:
                self.last_rejection_reason = "empty_or_nonfinite_bottle_cloud"
                return self._status_unlocked(now, accepted=False)
            pts, cols = downsample_pointcloud(pts, cols, self.voxel_size_m)
            if len(pts) > int(self.max_points):
                indices = np.linspace(0, len(pts) - 1, int(self.max_points), dtype=int)
                pts, cols = pts[indices], cols[indices]

            self.target_id = target_text
            self.target_id_scope = scope_text
            self.observations.append(
                RollingPointCloudObservation(
                    timestamp=now,
                    frame_id=frame_id,
                    target_id=target_text,
                    target_id_scope=scope_text,
                    points=pts.copy(),
                    colors=cols.copy(),
                    valid_depth_ratio=depth_ratio,
                    registration_status=registration_text,
                )
            )
            while len(self.observations) > 1 and self._point_count_unlocked() > int(self.max_points):
                self.observations.popleft()
            self.latest_accepted_timestamp = now
            self.last_rejection_reason = None
            return self._status_unlocked(now, accepted=True)

    def _status_unlocked(self, now: float, accepted: bool | None = None) -> dict[str, Any]:
        self._prune_unlocked(now)
        count = len(self.observations)
        point_count = self._point_count_unlocked()
        oldest = self.observations[0].timestamp if count else None
        latest = self.observations[-1].timestamp if count else None
        age = None if latest is None else max(0.0, float(now) - float(latest))
        span = 0.0 if oldest is None or latest is None else max(0.0, float(latest) - float(oldest))
        latest_frame_id = self.observations[-1].frame_id if count else None
        return {
            "rolling_window_active": bool(count),
            "rolling_cloud_age_sec": None if age is None else round(age, 3),
            "rolling_cloud_window_sec": float(self.window_sec),
            "rolling_cloud_point_count": point_count,
            "rolling_observation_count": count,
            "rolling_reset_reason": self.last_reset_reason,
            "rolling_valid_for_geometry": False,
            "rolling_valid_for_contact": False,
            "rolling_valid_for_benchmark": False,
            "rolling_pointcloud_active": bool(count),
            "rolling_pointcloud_status": "NOT_REGISTERED_DIAGNOSTIC" if count else "WAITING_FOR_VALID_LOCKED_TARGET",
            "rolling_pointcloud_mode": "NOT_REGISTERED_DIAGNOSTIC",
            "rolling_pointcloud_observation_accepted": accepted,
            "rolling_pointcloud_age_sec": None if age is None else round(age, 3),
            "rolling_pointcloud_window_sec": float(self.window_sec),
            "rolling_pointcloud_max_points": int(self.max_points),
            "rolling_pointcloud_voxel_size_m": float(self.voxel_size_m),
            "rolling_pointcloud_min_valid_depth_ratio": float(self.min_valid_depth_ratio),
            "rolling_pointcloud_observation_span_sec": round(span, 3),
            "rolling_pointcloud_point_count": point_count,
            "rolling_pointcloud_merged_point_count": 0,
            "rolling_pointcloud_observation_count": count,
            "rolling_pointcloud_latest_frame_id": latest_frame_id,
            "rolling_pointcloud_target_id": self.target_id,
            "rolling_pointcloud_target_id_scope": self.target_id_scope,
            "rolling_pointcloud_last_reset_reason": self.last_reset_reason,
            "rolling_pointcloud_last_reset_timestamp": self.last_reset_timestamp,
            "rolling_pointcloud_last_rejection_reason": self.last_rejection_reason,
            "rolling_pointcloud_registration_status": "NOT_REGISTERED_DIAGNOSTIC",
            "rolling_pointcloud_coordinate_frame": "separate_per_observation_realsense_camera_frames",
            "rolling_pointcloud_valid_for_geometry": False,
            "rolling_pointcloud_valid_for_contact": False,
            "rolling_pointcloud_valid_for_benchmark": False,
            "rolling_pointcloud_physically_validated": False,
            "rolling_pointcloud_limitations": [
                "Hand-camera observations are retained separately and are not spatially fused.",
                "Use validated multi-view registration for accumulated geometry.",
                "Rolling observations never replace current-frame contact validation.",
            ],
            "rolling_pointcloud_metric_provenance": {
                "age_sec": "measured_from_observation_timestamps",
                "point_count": "measured_retained_diagnostic_points",
                "observation_count": "measured",
                "geometry": "not_applicable_unregistered",
                "contact": "not_applicable_unregistered",
                "benchmark": "not_applicable_unregistered",
            },
        }

    def status(self, now: float | None = None) -> dict[str, Any]:
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            return self._status_unlocked(timestamp)


@dataclass
class PointCloudRuntimeState:
    last_generate_time: float = 0.0
    last_preview_time: float = 0.0
    last_context_time: float = 0.0
    last_lock_frame_id: int | None = None
    last_status: dict[str, Any] | None = None
    generated_times: list[float] | None = None
    preview_times: list[float] | None = None
    context_times: list[float] | None = None
    force_save_ply: bool = False
    force_preview_update: bool = False
    worker_busy: bool = False
    skipped_updates: int = 0
    last_duration_ms: float | None = None
    last_bottleneck: str | None = None
    last_worker_started_time: float | None = None
    generation_counter: int = 0
    rolling_object_cloud: RollingObjectCloudAccumulator | None = None


def _rate_from_times(times: list[float] | None) -> float:
    if not times or len(times) < 2:
        return 0.0
    span = max(times[-1] - times[0], 1e-6)
    return float((len(times) - 1) / span)


def pointcloud_update_due(state: PointCloudRuntimeState | None, now: float | None = None) -> bool:
    """Return whether a point-cloud update is due without doing heavy work."""
    if not POINTCLOUD_ENABLED:
        return False
    state = state or PointCloudRuntimeState()
    now = time.time() if now is None else float(now)
    if state.force_save_ply or state.force_preview_update:
        return True
    last_status = state.last_status or {}
    last_update = last_status.get("pointcloud_timestamp") or last_status.get("realsense_pointcloud_last_update_time")
    try:
        if last_update is not None and now - float(last_update) > 5.0:
            return True
    except Exception:
        pass
    min_period = 1.0 / max(float(POINTCLOUD_GENERATE_FPS), 1e-6)
    return state.last_status is None or (now - float(state.last_generate_time or 0.0)) >= min_period


def pointcloud_status_from_state(state: PointCloudRuntimeState | None, now: float | None = None) -> dict[str, Any]:
    """Return a cheap status payload while async point-cloud work is idle/busy."""
    now = time.time() if now is None else float(now)
    state = state or PointCloudRuntimeState()
    status = dict(state.last_status or {})
    status.setdefault("timestamp", now)
    status.setdefault("pointcloud_status", "waiting")
    status.setdefault("pointcloud_enabled", bool(POINTCLOUD_ENABLED))
    status.setdefault("pointcloud_preview_enabled", bool(POINTCLOUD_PREVIEW_ENABLED))
    status.setdefault("pointcloud_generate_fps_target", POINTCLOUD_GENERATE_FPS)
    status.setdefault("pointcloud_preview_fps_target", POINTCLOUD_PREVIEW_FPS)
    status.setdefault("pointcloud_context_preview_fps_target", POINTCLOUD_CONTEXT_PREVIEW_FPS)
    status.setdefault("pointcloud_live_stride", POINTCLOUD_LIVE_STRIDE)
    status["pointcloud_worker_busy"] = bool(state.worker_busy)
    status["pointcloud_skipped_updates"] = int(state.skipped_updates or 0)
    status["pointcloud_last_duration_ms"] = state.last_duration_ms
    status["pointcloud_bottleneck"] = state.last_bottleneck
    active = bool(status.get("realsense_pointcloud_active"))
    last_update = status.get("pointcloud_timestamp") or status.get("realsense_pointcloud_last_update_time")
    try:
        age = None if not active or last_update is None else max(0.0, now - float(last_update))
    except Exception:
        age = None
    if age is None:
        freshness = "UNAVAILABLE"
    elif age <= 2.0:
        freshness = "CURRENT"
    elif age <= 5.0:
        freshness = "AGING"
    else:
        freshness = "STALE"
    status["pointcloud_age_sec"] = None if age is None else round(float(age), 2)
    status["pointcloud_freshness"] = freshness
    status["pointcloud_current"] = bool(active and freshness == "CURRENT")
    if freshness != "CURRENT":
        status["cloud_valid_for_dimension_estimation"] = False
        status["cloud_valid_for_contact_validation"] = False
        status["cloud_valid_for_multiview_input"] = False
    if freshness in {"AGING", "STALE"}:
        if freshness == "AGING":
            status["viewer_watermark"] = "Point cloud aging - refresh soon"
    if freshness == "STALE":
        status["pointcloud_status"] = "stale"
        status["backend_running"] = bool(state.worker_busy)
        status["pointcloud_current"] = False
        status["cloud_valid_for_dimension_estimation"] = False
        status["cloud_valid_for_contact_validation"] = False
        status["viewer_watermark"] = "Last known point cloud - not current"
        if not status.get("pointcloud_error"):
            status["pointcloud_error"] = "Point cloud stale - refresh required"
    elif freshness == "UNAVAILABLE":
        status["viewer_watermark"] = "Point cloud unavailable - no current cloud"
    rolling = state.rolling_object_cloud
    status.update(
        rolling.status(now)
        if rolling is not None
        else RollingObjectCloudAccumulator().status(now)
    )
    return _pointcloud_alias_status(
        status,
        active=bool(status.get("realsense_pointcloud_active")),
        point_count=status.get("realsense_pointcloud_point_count"),
        fps=status.get("realsense_pointcloud_fps"),
        error=status.get("pointcloud_error"),
    )


def _bbox_from_report(report: dict[str, Any] | None) -> Any:
    report = report or {}
    geometry = report.get("bottle_geometry") or {}
    tracking = ((report.get("target_lock") or {}).get("current_tracked_target") or {})
    bbox = (
        geometry.get("bottle_extent_bbox_xyxy")
        or geometry.get("bbox_xyxy")
        or report.get("bottle_extent_bbox_xyxy")
        or tracking.get("accepted_current_bottle_bbox")
        or tracking.get("last_good_bottle_bbox")
    )
    if bbox is not None:
        return bbox
    xywh = geometry.get("bbox_xywh") or report.get("bbox_xywh")
    if not isinstance(xywh, (list, tuple)) or len(xywh) < 4:
        return None
    try:
        x, y, width, height = [float(value) for value in xywh[:4]]
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return [x, y, x + width, y + height]


def _decimate_rgbd_for_live(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a lower-cost RGB-D copy for optional live point-cloud status."""
    stride = max(1, int(stride))
    if stride <= 1:
        return rgb, depth, dict(intrinsics or {})
    rgb_small = np.asarray(rgb)[::stride, ::stride].copy()
    depth_small = np.asarray(depth)[::stride, ::stride].copy()
    scaled = dict(intrinsics or {})
    for key in ("fx", "fy", "cx", "cy"):
        if scaled.get(key) is not None:
            scaled[key] = float(scaled[key]) / float(stride)
    scaled["width"] = int(rgb_small.shape[1])
    scaled["height"] = int(rgb_small.shape[0])
    scaled["pointcloud_live_stride"] = stride
    return rgb_small, depth_small, scaled


def _scale_bbox_for_stride(bbox_or_mask: Any, stride: int) -> Any:
    """Scale an image-space bbox/mask into the decimated point-cloud frame."""
    stride = max(1, int(stride))
    if stride <= 1 or bbox_or_mask is None:
        return bbox_or_mask
    if isinstance(bbox_or_mask, np.ndarray):
        return bbox_or_mask[::stride, ::stride]
    if isinstance(bbox_or_mask, dict):
        scaled = dict(bbox_or_mask)
        for key in ("bbox_xyxy", "bottle_extent_bbox_xyxy", "bbox"):
            if isinstance(scaled.get(key), (list, tuple)) and len(scaled[key]) >= 4:
                scaled[key] = [float(v) / float(stride) for v in scaled[key][:4]]
                break
        return scaled
    if isinstance(bbox_or_mask, (list, tuple)) and len(bbox_or_mask) >= 4:
        return [float(v) / float(stride) for v in bbox_or_mask[:4]]
    return bbox_or_mask


def _pointcloud_alias_status(
    status: dict[str, Any],
    *,
    active: bool,
    point_count: int | None = None,
    fps: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    status["pointcloud_active"] = bool(active)
    status["pointcloud_backend"] = status.get("realsense_pointcloud_backend")
    status["pointcloud_fps"] = fps if fps is not None else status.get("realsense_pointcloud_fps")
    status["pointcloud_point_count"] = point_count if point_count is not None else status.get("realsense_pointcloud_point_count")
    status["pointcloud_last_update_time"] = status.get("realsense_pointcloud_last_update_time") or status.get("pointcloud_timestamp")
    status["pointcloud_error"] = error
    status["pointcloud_preview_enabled"] = bool(POINTCLOUD_PREVIEW_ENABLED)
    status["pointcloud_preview_path"] = str(POINTCLOUD_PREVIEW_PATH)
    status["bottle_pointcloud_preview_path"] = str(BOTTLE_POINTCLOUD_PREVIEW_PATH)
    status["pointcloud_context_preview_path"] = str(POINTCLOUD_CONTEXT_PREVIEW_PATH)
    status["pointcloud_json_enabled"] = bool(POINTCLOUD_JSON_ENABLED)
    status["pointcloud_live_json_path"] = str(POINTCLOUD_LIVE_JSON_PATH)
    return status


def _snapshot_consistency(
    report: dict[str, Any] | None,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, Any] | None,
    mask_meta: dict[str, Any],
) -> dict[str, Any]:
    report = report or {}
    rgb_frame = report.get("frame_id")
    depth_frame = report.get("depth_frame_id") or rgb_frame
    mask_frame = mask_meta.get("bottle_mask_frame_id")
    ids = [v for v in (rgb_frame, depth_frame, mask_frame) if v is not None]
    max_delta = 0
    numeric_ids = []
    for value in ids:
        try:
            numeric_ids.append(int(value))
        except Exception:
            pass
    if len(numeric_ids) >= 2:
        max_delta = int(max(numeric_ids) - min(numeric_ids))
    rejection = None
    consistent = True
    if rgb is None or depth is None:
        consistent = False
        rejection = "missing_rgb_or_depth"
    elif np.asarray(rgb).shape[:2] != np.asarray(depth).shape[:2]:
        consistent = False
        rejection = "rgb_depth_shape_mismatch"
    elif intrinsics and (
        (intrinsics.get("width") is not None and int(intrinsics.get("width")) != int(np.asarray(depth).shape[1]))
        or (intrinsics.get("height") is not None and int(intrinsics.get("height")) != int(np.asarray(depth).shape[0]))
    ):
        consistent = False
        rejection = "intrinsics_dimension_mismatch"
    elif max_delta > 1:
        consistent = False
        rejection = "frame_id_mismatch"
    elif mask_meta.get("bottle_mask_shape_valid") is False and mask_meta.get("bottle_mask_available"):
        consistent = False
        rejection = "mask_resolution_mismatch"
    return {
        "pointcloud_snapshot_consistent": bool(consistent),
        "pointcloud_snapshot_max_frame_delta": int(max_delta),
        "pointcloud_snapshot_rejection_reason": rejection,
        "pointcloud_source_rgb_frame_id": rgb_frame,
        "pointcloud_source_depth_frame_id": depth_frame,
    }


def update_pointcloud_from_frame(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    intrinsics: dict[str, Any] | None,
    report: dict[str, Any] | None = None,
    *,
    state: PointCloudRuntimeState | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Throttled point-cloud generation/status writer for the live backend."""
    ensure_output_dir()
    now = time.time() if now is None else float(now)
    state = state or PointCloudRuntimeState()
    if state.rolling_object_cloud is None:
        state.rolling_object_cloud = RollingObjectCloudAccumulator()
    state.generated_times = state.generated_times or []
    state.preview_times = state.preview_times or []
    state.context_times = state.context_times or []
    state.generation_counter = int(state.generation_counter or 0) + 1
    generation_id = f"{POINTCLOUD_SESSION_ID}-{state.generation_counter}"
    deps = dependency_status()
    base = {
        "timestamp": now,
        "pointcloud_session_id": POINTCLOUD_SESSION_ID,
        "pointcloud_generation_id": generation_id,
        "backend_writer_session_id": (report or {}).get("writer_session_id"),
        "backend_running": True,
        "coordinate_frame": "realsense_camera_m",
        "coordinate_frame_type": "current_camera_frame",
        "coordinate_frame_stable": False,
        "world_registered": False,
        "multiview_registered": False,
        "pointcloud_enabled": bool(POINTCLOUD_ENABLED),
        "pointcloud_preview_enabled": bool(POINTCLOUD_PREVIEW_ENABLED),
        "pointcloud_json_enabled": bool(POINTCLOUD_JSON_ENABLED),
        "pointcloud_rgb_available": rgb is not None,
        "pointcloud_depth_available": depth is not None,
        "pointcloud_intrinsics_available": _intrinsics_complete(intrinsics),
        "realsense_rgb_active": rgb is not None,
        "realsense_depth_active": depth is not None,
        "realsense_intrinsics_available": _intrinsics_complete(intrinsics),
        "realsense_pointcloud_backend": deps["selected_pointcloud_backend"],
        "open3d_available": deps["open3d_available"],
        "ros_pointcloud_available": deps["ros_pointcloud_available"],
        "pointcloud_generate_fps_target": POINTCLOUD_GENERATE_FPS,
        "pointcloud_preview_fps_target": POINTCLOUD_PREVIEW_FPS,
        "pointcloud_context_preview_fps_target": POINTCLOUD_CONTEXT_PREVIEW_FPS,
        "pointcloud_max_test_fps": POINTCLOUD_MAX_TEST_FPS,
        "pointcloud_save_ply_every_frame": POINTCLOUD_SAVE_PLY_EVERY_FRAME,
        "pointcloud_preview_autoscale": POINTCLOUD_PREVIEW_AUTOSCALE,
        "pointcloud_preview_center_object": POINTCLOUD_PREVIEW_CENTER_OBJECT,
        "pointcloud_preview_draw_axes": POINTCLOUD_PREVIEW_DRAW_AXES,
        "pointcloud_preview_draw_bbox": POINTCLOUD_PREVIEW_DRAW_BBOX,
        "pointcloud_preview_remove_outliers": POINTCLOUD_PREVIEW_REMOVE_OUTLIERS,
        "pointcloud_preview_depth_min_m": POINTCLOUD_PREVIEW_DEPTH_MIN_M,
        "pointcloud_preview_depth_max_m": POINTCLOUD_PREVIEW_DEPTH_MAX_M,
        "pointcloud_preview_point_size": POINTCLOUD_PREVIEW_POINT_SIZE,
        "pointcloud_preview_brightness_boost": POINTCLOUD_PREVIEW_BRIGHTNESS_BOOST,
        "pointcloud_preview_color_mode_requested": POINTCLOUD_PREVIEW_COLOR_MODE,
        "pointcloud_preview_fps_test_values": POINTCLOUD_PREVIEW_FPS_TEST_VALUES,
        "pointcloud_scene_max_points": POINTCLOUD_SCENE_MAX_POINTS,
        "pointcloud_bottle_max_points": POINTCLOUD_BOTTLE_MAX_POINTS,
        "pointcloud_bottle_detail_mode": POINTCLOUD_BOTTLE_DETAIL_MODE,
        "pointcloud_bottle_preview_stride": POINTCLOUD_BOTTLE_PREVIEW_STRIDE,
        **state.rolling_object_cloud.status(now),
        **_depth_range_mm(depth, intrinsics),
    }
    if not POINTCLOUD_ENABLED:
        status = _pointcloud_alias_status(
            {**base, "realsense_pointcloud_active": False, "pointcloud_status": "disabled"},
            active=False,
            error="point_cloud_disabled",
        )
        write_json_atomic(POINTCLOUD_STATUS_PATH, status)
        state.last_status = status
        return status
    if rgb is None or depth is None or not _intrinsics_complete(intrinsics):
        missing = []
        if rgb is None:
            missing.append("rgb")
        if depth is None:
            missing.append("aligned_depth")
        if not _intrinsics_complete(intrinsics):
            missing.append("camera_intrinsics")
        reason = "Point cloud unavailable: waiting for " + ", ".join(missing)
        status = _pointcloud_alias_status(
            {**base, "realsense_pointcloud_active": False, "pointcloud_status": "waiting_for_rgb_depth_intrinsics"},
            active=False,
            error=reason,
        )
        write_json_atomic(POINTCLOUD_STATUS_PATH, status)
        state.last_status = status
        return status
    min_period = 1.0 / max(float(POINTCLOUD_GENERATE_FPS), 1e-6)
    force_update = bool(state.force_save_ply or state.force_preview_update)
    if state.last_status and not force_update and now - state.last_generate_time < min_period:
        return state.last_status
    t0 = time.perf_counter()
    live_stride = POINTCLOUD_LIVE_STRIDE
    rgb_for_cloud, depth_for_cloud, intrinsics_for_cloud = _decimate_rgbd_for_live(
        rgb,
        depth,
        intrinsics or {},
        live_stride,
    )
    created = create_pointcloud_from_rgbd(rgb_for_cloud, depth_for_cloud, intrinsics_for_cloud)
    points = created["points"]
    colors = created["colors"]
    scene_raw_point_count = int(len(points))
    points, colors, filtered_count = _filter_visual_pointcloud(points, colors)
    scene_filtered_before_downsample = int(len(points))
    if POINTCLOUD_DOWNSAMPLE:
        points, colors = downsample_pointcloud(points, colors, POINTCLOUD_VOXEL_SIZE_M)
    scene_filtered_point_count = int(len(points))
    image_shape = tuple(np.asarray(depth_for_cloud).shape[:2])
    full_image_shape = tuple(np.asarray(depth).shape[:2])
    bbox = _bbox_from_report(report)
    crop_bbox = _scale_bbox_for_stride(bbox, live_stride)
    bottle_points = np.empty((0, 3), dtype=np.float32)
    bottle_colors = np.empty((0, 3), dtype=np.float32)
    bottle_density_mode = "standard"
    crop_t0 = time.perf_counter()
    bottle_roi_mask, bottle_roi_source, bottle_roi_candidate, bottle_roi_reason, mask_meta = _find_bottle_roi_candidate(report, full_image_shape)
    snapshot_meta = _snapshot_consistency(report, rgb, depth, intrinsics or {}, mask_meta)
    cleaned_bottle_mask = None
    bottle_crop_meta: dict[str, Any] = {
        **mask_meta,
        **snapshot_meta,
        "bottle_roi_source": bottle_roi_source,
        "bottle_crop_reason": bottle_roi_reason,
        "bottle_mask_pixel_count": int(mask_meta.get("bottle_mask_pixel_count") or 0),
        "bottle_depth_valid_pixel_count": 0,
        "bottle_crop_fraction_of_scene": 0.0,
        "bottle_raw_point_count": 0,
        "bottle_filtered_point_count": 0,
        "bottle_display_point_count": 0,
        "bottle_connected_component_count": 0,
        "bottle_crop_status": "UNAVAILABLE",
        "bottle_crop_confidence": 0.0,
        "bottle_crop_quality": "unavailable",
        "bottle_crop_degraded": True,
        "bottle_crop_safe_for_dimensions": False,
        "bottle_crop_safe_for_contacts": False,
        "bottle_crop_safe_for_quality_support": False,
    }
    if bottle_roi_mask is not None:
        cleaned_bottle_mask, clean_meta = _clean_bottle_roi_mask(
            bottle_roi_mask,
            depth,
            intrinsics or {},
            source=bottle_roi_source,
        )
        bottle_crop_meta.update(clean_meta)
        bottle_crop_meta["bottle_roi_source"] = bottle_roi_source
        if mask_meta.get("bottle_mask_provenance_current") is not True:
            # A one-frame-old or explicitly held contour can still bound a
            # useful visualization/dimension estimate against current depth,
            # but it is not exact current-pixel evidence for contacts or live
            # grasp-quality support.
            bottle_crop_meta.update({
                "bottle_crop_provenance_status": "NOT_CURRENT_EXACT_FRAME_CONTACT_QUALITY_BLOCKED",
                "bottle_crop_degraded": True,
                "bottle_crop_safe_for_contacts": False,
                "bottle_crop_safe_for_quality_support": False,
            })
        else:
            bottle_crop_meta["bottle_crop_provenance_status"] = "CURRENT_EXACT_RGBD_FRAME"
        bottle_crop_meta["bottle_crop_fraction_of_scene"] = round(
            float(np.count_nonzero(cleaned_bottle_mask)) / float(max(cleaned_bottle_mask.size, 1)),
            5,
        )
        bottle_created = create_pointcloud_from_rgbd(rgb, depth, intrinsics or {})
        bottle_density_mode = "full_resolution_mask_crop" if bottle_roi_source.endswith("mask") else "full_resolution_bbox_depth_crop"
        bottle_raw_points, bottle_raw_colors = _crop_pointcloud_with_mask(
            bottle_created["points"],
            bottle_created["colors"],
            bottle_created["valid_mask"],
            cleaned_bottle_mask,
        )
        bottle_crop_meta["bottle_raw_point_count"] = int(len(bottle_raw_points))
        bottle_points, bottle_colors, _bottle_kept = _filter_visual_pointcloud(
            bottle_raw_points,
            bottle_raw_colors,
            min_depth_m=POINTCLOUD_PREVIEW_DEPTH_MIN_M,
            max_depth_m=POINTCLOUD_PREVIEW_DEPTH_MAX_M,
        )
        if POINTCLOUD_DOWNSAMPLE and not POINTCLOUD_BOTTLE_DETAIL_MODE:
            bottle_points, bottle_colors = downsample_pointcloud(bottle_points, bottle_colors, POINTCLOUD_VOXEL_SIZE_M)
        bottle_crop_meta["bottle_filtered_point_count"] = int(len(bottle_points))
    crop_processing_ms = round((time.perf_counter() - crop_t0) * 1000.0, 2)
    bottle_crop_meta["scene_raw_point_count"] = int(scene_raw_point_count)
    bottle_crop_meta["scene_filtered_point_count"] = int(scene_filtered_point_count)
    if (
        bottle_crop_meta.get("bottle_raw_point_count")
        and scene_raw_point_count
        and abs(int(bottle_crop_meta["bottle_raw_point_count"]) - int(scene_raw_point_count)) <= max(10, int(scene_raw_point_count * 0.01))
    ):
        bottle_crop_meta["bottle_crop_reason"] = "bottle_raw_count_matches_scene_raw_count"
    if float(bottle_crop_meta.get("bottle_crop_fraction_of_scene") or 0.0) > float(POINTCLOUD_BOTTLE_MAX_SCENE_FRACTION):
        bottle_crop_meta["bottle_crop_reason"] = "bottle_crop_too_large_fraction_of_scene"
    geometry_t0 = time.perf_counter()
    bottle_props = estimate_bottle_3d_properties(bottle_points)
    geometry_processing_ms = round((time.perf_counter() - geometry_t0) * 1000.0, 2)
    if int(bottle_crop_meta.get("bottle_filtered_point_count") or 0) < int(POINTCLOUD_BOTTLE_MIN_POINTS):
        bottle_props["bottle_cloud_valid"] = False
        bottle_crop_meta["bottle_crop_reason"] = bottle_crop_meta.get("bottle_crop_reason") or "too_few_filtered_bottle_points"
    if bottle_crop_meta.get("bottle_crop_reason") in {"bottle_raw_count_matches_scene_raw_count", "bottle_crop_too_large_fraction_of_scene"}:
        bottle_props["bottle_cloud_valid"] = False
        bottle_crop_meta["bottle_crop_status"] = "INVALID"
        bottle_crop_meta["bottle_crop_confidence"] = 0.0
    if not bottle_crop_meta.get("pointcloud_snapshot_consistent", True):
        bottle_props["bottle_cloud_valid"] = False
        bottle_crop_meta["bottle_crop_status"] = "INVALID"
        bottle_crop_meta["bottle_crop_reason"] = bottle_crop_meta.get("pointcloud_snapshot_rejection_reason") or "pointcloud_snapshot_inconsistent"
        bottle_crop_meta["bottle_crop_confidence"] = 0.0
        # Snapshot provenance is a hard gate. A clean crop computed from an
        # older mask must not leave any downstream dimension/contact/quality
        # capability enabled for the current RGB-D frame.
        bottle_crop_meta["bottle_crop_safe_for_dimensions"] = False
        bottle_crop_meta["bottle_crop_safe_for_contacts"] = False
        bottle_crop_meta["bottle_crop_safe_for_quality_support"] = False
    crop_status = bottle_crop_meta.get("bottle_crop_status") or ("VALID_MASK_CROP" if bottle_props.get("bottle_cloud_valid") else "INVALID")
    crop_confidence = float(bottle_crop_meta.get("bottle_crop_confidence") or (0.95 if crop_status == "VALID_MASK_CROP" else 0.0))
    crop_safe_dimensions = bool(bottle_crop_meta.get("bottle_crop_safe_for_dimensions") and bottle_props.get("geometry_valid"))
    crop_safe_contacts = bool(bottle_crop_meta.get("bottle_crop_safe_for_contacts"))
    crop_safe_quality = bool(bottle_crop_meta.get("bottle_crop_safe_for_quality_support"))
    bottle_crop_meta.update({
        "bottle_crop_status": crop_status,
        "bottle_crop_confidence": round(crop_confidence, 3),
        "bottle_crop_safe_for_dimensions": crop_safe_dimensions,
        "bottle_crop_safe_for_contacts": crop_safe_contacts,
        "bottle_crop_safe_for_quality_support": crop_safe_quality,
    })
    contact_t0 = time.perf_counter()
    contact_3d = _contact_3d_summary(report, depth, intrinsics or {}, bottle_points)
    contact_validation_ms = round((time.perf_counter() - contact_t0) * 1000.0, 2)
    bottle_area_px = float(np.count_nonzero(bottle_roi_mask)) if bottle_roi_mask is not None else _bbox_area_px(crop_bbox)
    bottle_depth_ratio = None
    if bottle_area_px and bottle_area_px > 0:
        bottle_depth_ratio = round(float(len(bottle_points)) / float(bottle_area_px), 4)
    lock_frame_id = ((report or {}).get("target_lock") or {}).get("lock_frame_id")
    ply_saved = {"scene": False, "bottle": False}
    save_on_lock = bool(
        POINTCLOUD_SAVE_PLY_EVERY_FRAME
        or
        state.force_save_ply
        or (POINTCLOUD_SAVE_PLY_ON_LOCK and lock_frame_id is not None and lock_frame_id != state.last_lock_frame_id)
    )
    if save_on_lock:
        scene_subset = points
        scene_colors = colors
        if len(scene_subset) > 120_000:
            idx = np.linspace(0, len(scene_subset) - 1, 120_000).astype(int)
            scene_subset = scene_subset[idx]
            scene_colors = scene_colors[idx]
        ply_saved["scene"] = save_pointcloud_ply(SCENE_PLY_PATH, scene_subset, scene_colors, use_open3d=POINTCLOUD_USE_OPEN3D)
        ply_saved["bottle"] = (
            save_pointcloud_ply(BOTTLE_PLY_PATH, bottle_points, bottle_colors, use_open3d=POINTCLOUD_USE_OPEN3D)
            if len(bottle_points)
            else False
        )
        if lock_frame_id is not None:
            state.last_lock_frame_id = int(lock_frame_id)
    previous_pointcloud_status = state.last_status or {}
    preview_status: dict[str, Any] = {
        "pointcloud_preview_status": "disabled" if not POINTCLOUD_PREVIEW_ENABLED else "waiting",
        "pointcloud_preview_path": str(POINTCLOUD_PREVIEW_PATH),
        "bottle_pointcloud_preview_path": str(BOTTLE_POINTCLOUD_PREVIEW_PATH),
        "pointcloud_context_preview_path": str(POINTCLOUD_CONTEXT_PREVIEW_PATH),
        "pointcloud_preview_updated": False,
        "pointcloud_preview_fps": round(_rate_from_times(state.preview_times), 2),
        "pointcloud_preview_processing_ms": None,
        "pointcloud_preview_error": None,
        "pointcloud_context_status": (
            "disabled" if not POINTCLOUD_PREVIEW_ENABLED else previous_pointcloud_status.get("pointcloud_context_status", "waiting")
        ),
        "pointcloud_context_updated": False,
        "pointcloud_context_fps": previous_pointcloud_status.get("pointcloud_context_fps", round(_rate_from_times(state.context_times), 2)),
        "pointcloud_context_fps_target": POINTCLOUD_CONTEXT_PREVIEW_FPS,
        "pointcloud_context_processing_ms": previous_pointcloud_status.get("pointcloud_context_processing_ms"),
        "pointcloud_context_error": previous_pointcloud_status.get("pointcloud_context_error"),
    }
    preview_min_period = 1.0 / max(float(POINTCLOUD_PREVIEW_FPS), 1e-6)
    context_min_period = 1.0 / max(float(POINTCLOUD_CONTEXT_PREVIEW_FPS), 1e-6)
    preview_due = bool(POINTCLOUD_PREVIEW_ENABLED and (state.force_preview_update or now - state.last_preview_time >= preview_min_period))
    context_due = bool(POINTCLOUD_PREVIEW_ENABLED and (state.force_preview_update or now - state.last_context_time >= context_min_period))
    if preview_due:
        preview_t0 = time.perf_counter()
        scene_preview = save_pointcloud_preview(
            POINTCLOUD_PREVIEW_PATH,
            points,
            colors,
            bottle_points=bottle_points,
            bottle_colors=bottle_colors,
            title="RealSense Point Cloud",
            width=POINTCLOUD_PREVIEW_WIDTH,
            height=POINTCLOUD_PREVIEW_HEIGHT,
            max_points=POINTCLOUD_SCENE_MAX_POINTS,
            color_mode=POINTCLOUD_PREVIEW_COLOR_MODE,
            density_mode="scene_context",
            contacts_xyz=contact_3d,
        )
        bottle_preview = save_pointcloud_preview(
            BOTTLE_POINTCLOUD_PREVIEW_PATH,
            bottle_points,
            bottle_colors,
            bottle_points=bottle_points,
            bottle_colors=bottle_colors,
            title="Bottle Point Cloud Crop",
            width=POINTCLOUD_PREVIEW_WIDTH,
            height=POINTCLOUD_PREVIEW_HEIGHT,
            max_points=POINTCLOUD_BOTTLE_MAX_POINTS,
            color_mode=POINTCLOUD_PREVIEW_COLOR_MODE,
            density_mode="bottle_detail" if POINTCLOUD_BOTTLE_DETAIL_MODE else "standard",
            contacts_xyz=contact_3d,
        )
        state.last_preview_time = now
        state.preview_times.append(now)
        state.preview_times = state.preview_times[-30:]
        preview_errors = [p.get("error") for p in (scene_preview, bottle_preview) if p.get("error")]
        preview_status.update({
            "pointcloud_preview_status": "updated" if scene_preview.get("saved") else "failed",
            "pointcloud_preview_updated": bool(scene_preview.get("saved") or bottle_preview.get("saved")),
            "pointcloud_preview_fps": round(_rate_from_times(state.preview_times), 2),
            "pointcloud_preview_processing_ms": round((time.perf_counter() - preview_t0) * 1000.0, 2),
            "pointcloud_preview_error": "; ".join(preview_errors) if preview_errors else None,
            "pointcloud_preview_rendered_points": scene_preview.get("rendered_scene_points"),
            "bottle_pointcloud_preview_rendered_points": bottle_preview.get("rendered_bottle_points"),
            "pointcloud_preview_visible": scene_preview.get("visible"),
            "pointcloud_preview_reason": scene_preview.get("reason"),
            "pointcloud_preview_color_mode": scene_preview.get("color_mode"),
            "pointcloud_color_mode": scene_preview.get("color_mode"),
            "pointcloud_view_bounds": scene_preview.get("view_bounds"),
            "pointcloud_preview_outlier_filter": scene_preview.get("outlier_filter"),
            "bottle_pointcloud_preview_visible": bottle_preview.get("visible"),
            "bottle_pointcloud_preview_reason": bottle_preview.get("reason"),
            "bottle_pointcloud_density_mode": bottle_density_mode,
        })
        preview_status["pointcloud_render_ms"] = preview_status["pointcloud_preview_processing_ms"]
    if context_due:
        context_preview = save_pointcloud_context_preview(
            POINTCLOUD_CONTEXT_PREVIEW_PATH,
            rgb,
            depth,
            intrinsics or {},
            bbox,
            POINTCLOUD_PREVIEW_PATH,
            BOTTLE_POINTCLOUD_PREVIEW_PATH,
            width=POINTCLOUD_PREVIEW_WIDTH,
            height=POINTCLOUD_PREVIEW_HEIGHT,
        )
        state.last_context_time = now
        state.context_times.append(now)
        state.context_times = state.context_times[-30:]
        preview_status.update({
            "pointcloud_context_status": "updated" if context_preview.get("saved") else "failed",
            "pointcloud_context_updated": bool(context_preview.get("saved")),
            "pointcloud_context_fps": round(_rate_from_times(state.context_times), 2),
            "pointcloud_context_processing_ms": context_preview.get("processing_ms"),
            "pointcloud_context_error": context_preview.get("error"),
        })
    state.generated_times.append(now)
    state.generated_times = state.generated_times[-30:]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    summary = created["summary"]
    pointcloud_fps = round(_rate_from_times(state.generated_times), 2)
    pointcloud_frame_id = (report or {}).get("frame_id")
    pointcloud_status_label = "CURRENT"
    scene_input_pixel_count = int(np.asarray(depth_for_cloud).size)
    scene_valid_depth_count = int(scene_raw_point_count)
    bottle_roi_pixel_count = int(np.count_nonzero(bottle_roi_mask)) if bottle_roi_mask is not None else 0
    bottle_valid_depth_count = int(bottle_crop_meta.get("bottle_depth_valid_pixel_count") or 0)
    pointcloud_current = True
    scene_cloud_valid = bool(scene_filtered_point_count > 0)
    bottle_crop_valid = str(bottle_crop_meta.get("bottle_crop_status") or "").startswith("VALID") or str(bottle_crop_meta.get("bottle_crop_status")) == "DEGRADED_BBOX_CROP"
    bottle_geometry_valid = bool(bottle_props.get("geometry_valid") and bottle_crop_meta.get("bottle_crop_safe_for_dimensions"))
    contact_3d_valid = bool(contact_3d.get("contact_3d_valid") and bottle_crop_meta.get("bottle_crop_safe_for_contacts"))
    cloud_valid_for_visualization = bool(scene_cloud_valid)
    cloud_valid_for_dimension_estimation = bool(pointcloud_current and bottle_geometry_valid)
    cloud_valid_for_contact_validation = bool(pointcloud_current and contact_3d_valid)
    cloud_valid_for_multiview_input = bool(pointcloud_current and bottle_crop_meta.get("pointcloud_snapshot_consistent") and bottle_crop_valid and bottle_props.get("geometry_valid"))
    status = {
        **base,
        "realsense_pointcloud_active": True,
        "pointcloud_status": "updated",
        "pointcloud_freshness": pointcloud_status_label,
        "pointcloud_frame_id": pointcloud_frame_id,
        "pointcloud_timestamp": now,
        "pointcloud_age_sec": 0.0,
        "pointcloud_source_rgb_frame_id": pointcloud_frame_id,
        "pointcloud_source_depth_frame_id": (report or {}).get("depth_frame_id") or pointcloud_frame_id,
        "realsense_pointcloud_fps": pointcloud_fps,
        "realsense_pointcloud_last_update_time": now,
        "realsense_pointcloud_point_count": int(len(points)),
        "realsense_pointcloud_quality": summary.get("pointcloud_quality_score"),
        "pointcloud_live_stride": live_stride,
        "scene_input_pixel_count": scene_input_pixel_count,
        "scene_valid_depth_count": scene_valid_depth_count,
        "scene_raw_point_count": int(scene_raw_point_count),
        "scene_filtered_point_count": int(scene_filtered_point_count),
        "scene_display_point_count": int(min(len(points), POINTCLOUD_PREVIEW_MAX_POINTS)),
        "scene_saved_ply_point_count": int(len(points)) if save_on_lock else 0,
        "scene_sampling_stride": int(live_stride),
        "scene_sampling_domain": "decimated_rgbd" if live_stride > 1 else "full_rgbd",
        "pointcloud_raw_point_count": int(scene_raw_point_count),
        "pointcloud_filtered_point_count": int(scene_filtered_point_count),
        "pointcloud_preview_point_count": int(min(len(points), POINTCLOUD_PREVIEW_MAX_POINTS)),
        "pointcloud_scene_display_max_points": int(POINTCLOUD_SCENE_MAX_POINTS),
        "pointcloud_bottle_display_max_points": int(POINTCLOUD_BOTTLE_MAX_POINTS),
        "full_point_count_before_downsample": int(scene_filtered_before_downsample),
        "pointcloud_downsampled": bool(POINTCLOUD_DOWNSAMPLE),
        "pointcloud_voxel_size_m": POINTCLOUD_VOXEL_SIZE_M,
        "valid_depth_ratio": summary.get("valid_depth_ratio"),
        "scene_centroid_m": summary.get("centroid_m"),
        "scene_extents_m": summary.get("extents_m"),
        "open3d_enabled": deps.get("open3d_enabled"),
        **bottle_props,
        **bottle_crop_meta,
        **contact_3d,
        "contact_3d_valid": contact_3d_valid,
        "contact_3d_reason": contact_3d.get("contact_3d_reason") if contact_3d_valid else (contact_3d.get("contact_3d_reason") or "crop_not_safe_for_contact_validation"),
        "bottle_roi_pixel_count": bottle_roi_pixel_count,
        "bottle_valid_depth_count": bottle_valid_depth_count,
        "bottle_saved_ply_point_count": int(len(bottle_points)) if save_on_lock and len(bottle_points) else 0,
        "bottle_sampling_stride": 1,
        "bottle_sampling_domain": "full_resolution_mask_or_bbox_crop",
        "counts_directly_comparable": bool(live_stride == 1),
        "bottle_roi_available": bottle_roi_mask is not None,
        "bottle_pointcloud_valid": bool(bottle_props.get("bottle_cloud_valid")),
        "bottle_crop_status": bottle_crop_meta.get("bottle_crop_status"),
        "bottle_pointcloud_reason": "valid" if bottle_props.get("bottle_cloud_valid") else (bottle_crop_meta.get("bottle_crop_reason") or "invalid_bottle_crop"),
        "bottle_crop_reason": bottle_crop_meta.get("bottle_crop_reason") or ("valid" if bottle_props.get("bottle_cloud_valid") else "invalid_bottle_crop"),
        "bottle_pointcloud_point_count": bottle_props.get("bottle_point_count"),
        "bottle_display_point_count": int(min(len(bottle_points), POINTCLOUD_BOTTLE_MAX_POINTS)),
        "bottle_pointcloud_valid_depth_ratio": bottle_depth_ratio,
        "bottle_pointcloud_centroid_mm": _meters_list_to_mm(bottle_props.get("bottle_3d_centroid_m")),
        "bottle_pointcloud_extents_mm": _meter_extents_to_mm(bottle_props.get("bottle_3d_extents_m")),
        "bottle_pointcloud_roi_source": bottle_roi_source,
        "bottle_roi_source": bottle_roi_source,
        "bottle_pointcloud_density_mode": bottle_density_mode,
        "pointcloud_context_preview_path": str(POINTCLOUD_CONTEXT_PREVIEW_PATH),
        "bottle_bbox_source": "current_realsense_report_bbox" if bbox is not None else "none",
        "bottle_bbox_xyxy": bbox,
        "pointcloud_current": pointcloud_current,
        "scene_cloud_valid": scene_cloud_valid,
        "bottle_crop_valid": bottle_crop_valid,
        "bottle_geometry_valid": bottle_geometry_valid,
        "cloud_valid_for_visualization": cloud_valid_for_visualization,
        "cloud_valid_for_dimension_estimation": cloud_valid_for_dimension_estimation,
        "cloud_valid_for_contact_validation": cloud_valid_for_contact_validation,
        "cloud_valid_for_multiview_input": cloud_valid_for_multiview_input,
        "ply_save_policy": "on_lock",
        "ply_saved_this_update": ply_saved,
        "latest_realsense_pointcloud_ply": str(SCENE_PLY_PATH),
        "latest_bottle_pointcloud_ply": str(BOTTLE_PLY_PATH),
        "crop_processing_ms": crop_processing_ms,
        "geometry_processing_ms": geometry_processing_ms,
        "contact_validation_ms": contact_validation_ms,
        "total_processing_ms": round(elapsed_ms, 2),
        "pointcloud_processing_ms": round(elapsed_ms, 2),
        **preview_status,
        "pointcloud_worker_busy": bool(state.worker_busy),
        "pointcloud_skipped_updates": int(state.skipped_updates or 0),
    }
    target_context = multiview_target_context(report)
    report_data = report or {}
    rolling_bottle_moved = bool(
        report_data.get("bottle_moved") is True
        or report_data.get("rolling_pointcloud_bottle_moved") is True
        or (
            report_data.get("reconstruction_requires_reset") is True
            and report_data.get("bottle_static") is False
        )
    )
    rolling_status = state.rolling_object_cloud.add_observation(
        bottle_points,
        bottle_colors,
        timestamp=now,
        frame_id=pointcloud_frame_id,
        target_id=target_context.get("target_id"),
        target_id_scope=target_context.get("target_id_scope"),
        target_valid=bool(target_context.get("valid")),
        valid_depth_ratio=bottle_depth_ratio,
        crop_valid=bool(bottle_crop_valid and bottle_props.get("bottle_cloud_valid")),
        snapshot_consistent=bool(bottle_crop_meta.get("pointcloud_snapshot_consistent")),
        bottle_moved=rolling_bottle_moved,
        registration_status=str(report_data.get("rolling_pointcloud_registration_status") or "NOT_REGISTERED_DIAGNOSTIC"),
        explicit_reset=bool(report_data.get("reset_rolling_pointcloud") is True),
    )
    status.update(rolling_status)
    json_status = {
        "pointcloud_json_status": "disabled",
        "pointcloud_live_json_path": str(POINTCLOUD_LIVE_JSON_PATH),
        "pointcloud_json_point_count": None,
        "pointcloud_json_bottle_point_count": None,
        "pointcloud_json_processing_ms": None,
        "pointcloud_json_error": None,
    }
    if POINTCLOUD_JSON_ENABLED:
        json_result = save_pointcloud_live_json(
            POINTCLOUD_LIVE_JSON_PATH,
            points,
            colors,
            bottle_points=bottle_points,
            bottle_colors=bottle_colors,
            max_points=POINTCLOUD_JSON_MAX_POINTS,
            timestamp=now,
            status=status,
        )
        json_status.update({
            "pointcloud_json_status": "updated" if json_result.get("saved") else "failed",
            "pointcloud_json_point_count": json_result.get("scene_points"),
            "pointcloud_json_bottle_point_count": json_result.get("bottle_points"),
            "scene_display_point_count": json_result.get("scene_points"),
            "bottle_display_point_count": json_result.get("bottle_points"),
            "pointcloud_json_processing_ms": json_result.get("json_processing_ms"),
            "pointcloud_json_error": json_result.get("error"),
            "pointcloud_json_max_points": POINTCLOUD_JSON_MAX_POINTS,
        })
    status.update(json_status)
    multiview_gate = multiview_input_gate(
        {**(report or {}), **status},
        scene_points=points,
        bottle_points=bottle_points,
        timestamp=now,
    )
    status.update(multiview_gate)
    if multiview_gate.get("multiview_input_valid"):
        multiview_submit = submit_multiview_candidate_async(
            scene_points=points,
            scene_colors=colors,
            bottle_points=bottle_points,
            bottle_colors=bottle_colors,
            report={**(report or {}), **status},
            timestamp=now,
            frame_id=pointcloud_frame_id,
        )
    else:
        multiview_submit = {"submitted": False, "reason": multiview_gate.get("multiview_input_reject_reason") or "single_frame_cloud_not_valid_for_multiview_input"}
    status.update({
        "multiview_reconstruction": multiview_status_from_file(),
        "multiview_candidate_submitted": bool(multiview_submit.get("submitted")),
        "multiview_candidate_submit_reason": multiview_submit.get("reason"),
        "multiview_candidate_input_gate": multiview_gate,
    })
    _pointcloud_alias_status(status, active=True, point_count=int(len(points)), fps=pointcloud_fps)
    write_json_atomic(POINTCLOUD_STATUS_PATH, status)
    state.last_generate_time = now
    state.last_status = status
    state.last_duration_ms = round(elapsed_ms, 2)
    state.last_bottleneck = "preview_render" if (preview_status.get("pointcloud_preview_processing_ms") or 0) > elapsed_ms * 0.5 else "pointcloud_generation"
    state.force_save_ply = False
    state.force_preview_update = False
    return status


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


def _test_dummy() -> dict[str, Any]:
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 80
    rgb[..., 1] = 140
    depth = np.full((48, 64), 500, dtype=np.uint16)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48}
    pc = create_pointcloud_from_rgbd(rgb, depth, intrinsics)
    points, colors = downsample_pointcloud(pc["points"], pc["colors"], 0.01)
    path = OUTPUT_DIR / "pointcloud_dummy_test.ply"
    saved = save_pointcloud_ply(path, points, colors)
    return {**dependency_status(), "dummy_point_count": len(points), "dummy_ply_saved": saved, "dummy_ply_path": str(path)}


def _test_from_live(duration: float) -> dict[str, Any]:
    try:
        import cv2  # noqa: F401
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except Exception as exc:
        return {"status": "failed", "reason": f"ROS2 live test dependencies unavailable: {exc}", **dependency_status()}

    latest: dict[str, Any] = {"rgb": None, "depth": None, "intrinsics": None, "color_count": 0, "depth_count": 0}

    def color_cb(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "rgb8":
            latest["rgb"] = arr.reshape((msg.height, msg.width, 3)).copy()
        elif msg.encoding == "bgr8":
            import cv2

            latest["rgb"] = cv2.cvtColor(arr.reshape((msg.height, msg.width, 3)), cv2.COLOR_BGR2RGB)
        latest["color_count"] += 1

    def depth_cb(msg):
        if msg.encoding in {"16UC1", "mono16"}:
            latest["depth"] = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).copy()
        elif msg.encoding == "32FC1":
            latest["depth"] = (np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)) * 1000.0).astype(np.uint16)
        latest["depth_count"] += 1

    def info_cb(msg):
        latest["intrinsics"] = {"fx": float(msg.k[0]), "fy": float(msg.k[4]), "cx": float(msg.k[2]), "cy": float(msg.k[5]), "width": int(msg.width), "height": int(msg.height)}

    rclpy.init(args=None)
    node = rclpy.create_node("grasp_pointcloud_live_test")
    node.create_subscription(Image, "/camera/camera/color/image_raw", color_cb, qos_profile_sensor_data)
    node.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", depth_cb, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", info_cb, qos_profile_sensor_data)
    end = time.time() + max(0.1, float(duration))
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    status = update_pointcloud_from_frame(latest["rgb"], latest["depth"], latest["intrinsics"], report={"target_lock": {}}, state=PointCloudRuntimeState(), now=time.time())
    status.update({
        "status": "complete" if status.get("realsense_pointcloud_active") else "failed",
        "rgb_frame_received": latest["rgb"] is not None,
        "depth_frame_received": latest["depth"] is not None,
        "camera_info_received": latest["intrinsics"] is not None,
        "color_frames": latest["color_count"],
        "depth_frames": latest["depth_count"],
    })
    write_json_atomic(OUTPUT_DIR / "pointcloud_live_test_status.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSense RGB-D point cloud diagnostics.")
    parser.add_argument("--diagnose", action="store_true", help="Report optional point-cloud backend availability.")
    parser.add_argument("--test-dummy", action="store_true", help="Run a camera-free dummy RGB-D point-cloud test.")
    parser.add_argument("--test-from-live", action="store_true", help="Subscribe to live RealSense RGB-D topics and generate one point cloud.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration for live test collection.")
    args = parser.parse_args(argv)
    if args.test_from_live:
        _print_json(_test_from_live(args.duration))
        return 0
    if args.test_dummy:
        _print_json(_test_dummy())
        return 0
    _print_json(dependency_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

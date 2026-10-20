"""Persistent multi-view bottle reconstruction for RealSense point clouds.

This module is intentionally separate from the live grasp tracker.  It consumes
already-produced RGB-D point clouds, estimates a camera transform when possible,
and fuses accepted bottle-only keyframes into a reconstruction-world model.
All functions are camera-free and safe to import in tests.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grasp_project.common.config import OUTPUT_DIR, get_env_bool, get_env_float, get_env_int
from grasp_project.common.file_io import append_jsonl, read_json, write_json_atomic, write_text_atomic


MULTIVIEW_OUTPUT_DIR = OUTPUT_DIR / "multiview"
MULTIVIEW_AUDIT_PATH = OUTPUT_DIR / "multiview_pointcloud_audit.txt"
MULTIVIEW_STATUS_PATH = MULTIVIEW_OUTPUT_DIR / "reconstruction_status.json"
MULTIVIEW_PREVIEW_PATH = MULTIVIEW_OUTPUT_DIR / "reconstruction_preview.json"
MULTIVIEW_LATEST_STATUS_PATH = MULTIVIEW_OUTPUT_DIR / "latest_status.json"
MULTIVIEW_LATEST_PREVIEW_PATH = MULTIVIEW_OUTPUT_DIR / "latest_preview.json"
MULTIVIEW_SESSION_PATH = MULTIVIEW_OUTPUT_DIR / "current_session.json"
MULTIVIEW_TRAJECTORY_PATH = MULTIVIEW_OUTPUT_DIR / "camera_trajectory.json"
MULTIVIEW_ACCEPTED_PATH = MULTIVIEW_OUTPUT_DIR / "accepted_keyframes.jsonl"
MULTIVIEW_REJECTED_PATH = MULTIVIEW_OUTPUT_DIR / "rejected_keyframes.jsonl"
MULTIVIEW_REPORT_PATH = MULTIVIEW_OUTPUT_DIR / "reconstruction_report.json"
MULTIVIEW_COVERAGE_PATH = MULTIVIEW_OUTPUT_DIR / "coverage.json"
MULTIVIEW_CONTACT_SUPPORT_PATH = MULTIVIEW_OUTPUT_DIR / "contact_support.json"
MULTIVIEW_SCENE_PLY_PATH = MULTIVIEW_OUTPUT_DIR / "accumulated_scene.ply"
MULTIVIEW_BOTTLE_PLY_PATH = MULTIVIEW_OUTPUT_DIR / "accumulated_bottle.ply"
MULTIVIEW_MESH_PLY_PATH = MULTIVIEW_OUTPUT_DIR / "reconstructed_bottle_mesh.ply"
MULTIVIEW_SESSIONS_DIR = MULTIVIEW_OUTPUT_DIR / "sessions"

MULTIVIEW_ENABLED = get_env_bool("MULTIVIEW_ENABLED", True)
MULTIVIEW_MIN_REGISTRATION_FITNESS = get_env_float("MULTIVIEW_MIN_REGISTRATION_FITNESS", 0.35)
MULTIVIEW_MAX_INLIER_RMSE_M = get_env_float("MULTIVIEW_MAX_INLIER_RMSE_M", 0.015)
MULTIVIEW_MAX_TRANSLATION_STEP_M = get_env_float("MULTIVIEW_MAX_TRANSLATION_STEP_M", 0.10)
MULTIVIEW_MAX_ROTATION_STEP_DEG = get_env_float("MULTIVIEW_MAX_ROTATION_STEP_DEG", 20.0)
MULTIVIEW_MIN_OVERLAP_RATIO = get_env_float("MULTIVIEW_MIN_OVERLAP_RATIO", 0.25)
MULTIVIEW_MIN_KEYFRAME_TRANSLATION_MM = get_env_float("MULTIVIEW_MIN_KEYFRAME_TRANSLATION_MM", 10.0)
MULTIVIEW_MIN_KEYFRAME_ROTATION_DEG = get_env_float("MULTIVIEW_MIN_KEYFRAME_ROTATION_DEG", 3.0)
MULTIVIEW_MAX_KEYFRAME_INTERVAL_SEC = get_env_float("MULTIVIEW_MAX_KEYFRAME_INTERVAL_SEC", 1.0)
MULTIVIEW_MIN_BOTTLE_DEPTH_VALID_RATIO = get_env_float("MULTIVIEW_MIN_BOTTLE_DEPTH_VALID_RATIO", 0.50)
MULTIVIEW_FUSION_MODE = os.environ.get("MULTIVIEW_FUSION_MODE", "voxel_cloud").strip() or "voxel_cloud"
MULTIVIEW_VOXEL_SIZE_M = get_env_float("MULTIVIEW_VOXEL_SIZE_M", 0.004)
MULTIVIEW_TSDF_ENABLED = get_env_bool("MULTIVIEW_TSDF_ENABLED", False)
MULTIVIEW_TSDF_SDF_TRUNC_M = get_env_float("MULTIVIEW_TSDF_SDF_TRUNC_M", 0.02)
MULTIVIEW_MAX_ACCUMULATED_POINTS = get_env_int("MULTIVIEW_MAX_ACCUMULATED_POINTS", 250_000)
MULTIVIEW_PREVIEW_MAX_POINTS = get_env_int("MULTIVIEW_PREVIEW_MAX_POINTS", 40_000)
MULTIVIEW_BOTTLE_MOVED_THRESHOLD_MM = get_env_float("MULTIVIEW_BOTTLE_MOVED_THRESHOLD_MM", 35.0)
MULTIVIEW_CONTACT_SURFACE_TOLERANCE_M = get_env_float("MULTIVIEW_CONTACT_SURFACE_TOLERANCE_M", 0.020)
MULTIVIEW_ALLOW_NUMPY_CENTROID_REGISTRATION = get_env_bool("MULTIVIEW_ALLOW_NUMPY_CENTROID_REGISTRATION", False)
MULTIVIEW_MIN_BOTTLE_POINTS = get_env_int("MULTIVIEW_MIN_BOTTLE_POINTS", 80)
MULTIVIEW_MIN_SCENE_POINTS = get_env_int("MULTIVIEW_MIN_SCENE_POINTS", 200)
MULTIVIEW_SCENE_VOXEL_SIZE_M = get_env_float("MULTIVIEW_SCENE_VOXEL_SIZE_M", max(0.008, MULTIVIEW_VOXEL_SIZE_M * 2.0))
MULTIVIEW_BOTTLE_VOXEL_SIZE_M = get_env_float("MULTIVIEW_BOTTLE_VOXEL_SIZE_M", MULTIVIEW_VOXEL_SIZE_M)
MULTIVIEW_MAX_SCENE_VOXELS = get_env_int("MULTIVIEW_MAX_SCENE_VOXELS", 80_000)
MULTIVIEW_MAX_QUEUE_SIZE = get_env_int("MULTIVIEW_MAX_QUEUE_SIZE", 1)
MULTIVIEW_MAX_POINTCLOUD_AGE_SEC = get_env_float("MULTIVIEW_MAX_POINTCLOUD_AGE_SEC", 2.0)
MULTIVIEW_ALLOW_BBOX_ROI = get_env_bool("MULTIVIEW_ALLOW_BBOX_ROI", False)
MULTIVIEW_ENABLE_CONTACT_CANDIDATES = get_env_bool("MULTIVIEW_ENABLE_CONTACT_CANDIDATES", False)
MULTIVIEW_DAVIS_GATE_ENABLED = get_env_bool("MULTIVIEW_DAVIS_GATE_ENABLED", False)

SCAN_IDLE = "IDLE"
SCAN_WAITING = "WAITING_FOR_VALID_TARGET"
SCAN_INITIALIZING = "INITIALIZING_REFERENCE"
SCAN_SCANNING = "SCANNING"
SCAN_PAUSED = "PAUSED"
SCAN_REGISTRATION_WARNING = "REGISTRATION_WARNING"
SCAN_BOTTLE_MOVED = "BOTTLE_MOVED"
SCAN_COMPLETE = "COMPLETE"
SCAN_ERROR = "ERROR"


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception:
        return None


def _now() -> float:
    return time.time()


def _as_points(points: np.ndarray | list | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return arr.reshape((-1, 3))


def _as_colors(colors: np.ndarray | list | None, count: int) -> np.ndarray:
    if colors is None:
        return np.full((count, 3), 0.82, dtype=np.float32)
    arr = np.asarray(colors, dtype=np.float32)
    if arr.size == 0:
        return np.full((count, 3), 0.82, dtype=np.float32)
    arr = arr.reshape((-1, 3))
    if len(arr) != count:
        return np.full((count, 3), 0.82, dtype=np.float32)
    if np.nanmax(arr) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def finite_points(points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    pts = _as_points(points)
    if len(pts) == 0:
        return pts, None if colors is None else _as_colors(colors, 0)
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    if colors is None:
        return pts, None
    cols = _as_colors(colors, len(mask))[mask]
    return pts, cols


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Transform Nx3 points by a 4x4 homogeneous transform."""
    pts = _as_points(points)
    if len(pts) == 0:
        return pts.copy()
    transform = np.asarray(transform_4x4, dtype=np.float64).reshape((4, 4))
    hom = np.ones((len(pts), 4), dtype=np.float64)
    hom[:, :3] = pts.astype(np.float64)
    out = hom @ transform.T
    return out[:, :3].astype(np.float32)


def transform_translation_m(transform_4x4: np.ndarray) -> float:
    t = np.asarray(transform_4x4, dtype=np.float64).reshape((4, 4))[:3, 3]
    return float(np.linalg.norm(t))


def transform_rotation_deg(transform_4x4: np.ndarray) -> float:
    r = np.asarray(transform_4x4, dtype=np.float64).reshape((4, 4))[:3, :3]
    trace = float(np.trace(r))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(math.degrees(math.acos(cos_theta)))


def make_translation_transform(delta_xyz_m: np.ndarray | list[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = np.asarray(delta_xyz_m, dtype=np.float64).reshape(3)
    return t


def _sample_points(points: np.ndarray, colors: np.ndarray | None = None, max_points: int = 50_000) -> tuple[np.ndarray, np.ndarray | None]:
    pts = _as_points(points)
    cols = None if colors is None else _as_colors(colors, len(pts))
    if len(pts) <= int(max_points):
        return pts, cols
    idx = np.linspace(0, len(pts) - 1, int(max_points)).astype(int)
    return pts[idx], None if cols is None else cols[idx]


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    pts = _as_points(points)
    if len(pts) == 0:
        zero = [0.0, 0.0, 0.0]
        return {"min": zero, "max": zero, "center": zero}
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    center = np.mean(pts, axis=0)
    return {
        "min": np.round(lo, 5).astype(float).tolist(),
        "max": np.round(hi, 5).astype(float).tolist(),
        "center": np.round(center, 5).astype(float).tolist(),
    }


def _flat(values: np.ndarray, decimals: int = 5) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    return np.round(arr.reshape(-1), decimals).astype(float).tolist()


def _coverage_from_points(points: np.ndarray) -> dict[str, Any]:
    pts = _as_points(points)
    if len(pts) < 8:
        return {
            "covered_view_angle_deg": 0.0,
            "estimated_surface_coverage_percent": 0.0,
            "observed_front": False,
            "observed_left": False,
            "observed_right": False,
            "observed_rear": False,
            "observed_top": False,
            "observed_bottom": False,
            "missing_view_directions": ["front", "left", "right", "rear", "top", "bottom"],
            "recommended_camera_motion": "collect more bottle views",
        }
    center = np.mean(pts, axis=0)
    rel = pts - center
    angles = (np.degrees(np.arctan2(rel[:, 0], rel[:, 2])) + 360.0) % 360.0
    bins = np.zeros(12, dtype=bool)
    bins[np.floor(angles / 30.0).astype(int).clip(0, 11)] = True
    covered_angle = float(np.count_nonzero(bins) * 30.0)
    y = rel[:, 1]
    y_span = float(np.ptp(y)) if len(y) else 0.0
    top = bool(y_span > 0 and np.percentile(y, 90) > 0.25 * y_span)
    bottom = bool(y_span > 0 and np.percentile(y, 10) < -0.25 * y_span)
    quadrants = {
        "front": np.any((angles >= 315) | (angles < 45)),
        "right": np.any((angles >= 45) & (angles < 135)),
        "rear": np.any((angles >= 135) & (angles < 225)),
        "left": np.any((angles >= 225) & (angles < 315)),
        "top": top,
        "bottom": bottom,
    }
    missing = [name for name, ok in quadrants.items() if not ok]
    coverage = min(100.0, (covered_angle / 360.0) * 85.0 + (7.5 if top else 0.0) + (7.5 if bottom else 0.0))
    recommended = "continue clockwise around bottle" if "rear" in missing or "left" in missing or "right" in missing else "tilt slightly to capture top/bottom"
    return {
        "covered_view_angle_deg": round(covered_angle, 1),
        "estimated_surface_coverage_percent": round(float(coverage), 1),
        "observed_front": bool(quadrants["front"]),
        "observed_left": bool(quadrants["left"]),
        "observed_right": bool(quadrants["right"]),
        "observed_rear": bool(quadrants["rear"]),
        "observed_top": bool(top),
        "observed_bottom": bool(bottom),
        "missing_view_directions": missing,
        "recommended_camera_motion": recommended,
    }


def _zero_coverage_status(recommended: str = "collect more bottle views") -> dict[str, Any]:
    return {
        "covered_view_angle_deg": 0.0,
        "estimated_surface_coverage_percent": 0.0,
        "observed_front": False,
        "observed_left": False,
        "observed_right": False,
        "observed_rear": False,
        "observed_top": False,
        "observed_bottom": False,
        "missing_view_directions": ["front", "left", "right", "rear", "top", "bottom"],
        "recommended_camera_motion": recommended,
    }


def _circular_span_deg(angles_deg: list[float]) -> float:
    if len(angles_deg) <= 1:
        return 0.0
    angles = np.sort(np.asarray(angles_deg, dtype=np.float64) % 360.0)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    return float(max(0.0, 360.0 - np.max(gaps)))


def _yaw_deg_from_transform(transform_4x4: np.ndarray) -> float:
    r = np.asarray(transform_4x4, dtype=np.float64).reshape((4, 4))[:3, :3]
    return float(math.degrees(math.atan2(r[0, 2], r[2, 2])))


def _coverage_from_trajectory(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate view coverage from accepted camera poses, not bottle surface points.

    A single front view of a rounded bottle can contain points around much of the
    object's image-space silhouette.  Counting those points as view coverage
    overstates reconstruction completeness, so coverage is tied to accepted
    camera trajectory bins instead.
    """
    if not trajectory:
        return _zero_coverage_status()
    view_angles: list[float] = []
    y_offsets: list[float] = []
    radii: list[float] = []
    for item in trajectory:
        try:
            transform = np.asarray(item.get("transform_to_reference"), dtype=np.float64).reshape((4, 4))
        except Exception:
            continue
        t = transform[:3, 3]
        radius = float(math.hypot(float(t[0]), float(t[2])))
        radii.append(radius)
        y_offsets.append(float(t[1]))
        if radius >= 0.02:
            angle = math.degrees(math.atan2(float(t[0]), float(t[2])))
        else:
            angle = _yaw_deg_from_transform(transform)
        view_angles.append((angle + 360.0) % 360.0)
    if not view_angles:
        return _zero_coverage_status()

    bins = np.zeros(12, dtype=bool)
    bins[np.floor(np.asarray(view_angles) / 30.0).astype(int).clip(0, 11)] = True
    covered_angle = float(np.count_nonzero(bins) * 30.0)
    angle_span = _circular_span_deg(view_angles)
    radial_span = float(max(radii) - min(radii)) if radii else 0.0
    if angle_span < 15.0 and radial_span < 0.015:
        covered_angle = min(covered_angle, 30.0)

    observed_front = bool(np.any((np.asarray(view_angles) >= 315.0) | (np.asarray(view_angles) < 45.0)))
    observed_right = bool(np.any((np.asarray(view_angles) >= 45.0) & (np.asarray(view_angles) < 135.0)))
    observed_rear = bool(np.any((np.asarray(view_angles) >= 135.0) & (np.asarray(view_angles) < 225.0)))
    observed_left = bool(np.any((np.asarray(view_angles) >= 225.0) & (np.asarray(view_angles) < 315.0)))
    y_span = float(max(y_offsets) - min(y_offsets)) if y_offsets else 0.0
    observed_top = y_span > 0.035 and max(y_offsets) > 0.02
    observed_bottom = y_span > 0.035 and min(y_offsets) < -0.02
    quadrants = {
        "front": observed_front,
        "left": observed_left,
        "right": observed_right,
        "rear": observed_rear,
        "top": observed_top,
        "bottom": observed_bottom,
    }
    missing = [name for name, ok in quadrants.items() if not ok]
    coverage = min(100.0, (covered_angle / 360.0) * 85.0 + (7.5 if observed_top else 0.0) + (7.5 if observed_bottom else 0.0))
    recommended = "continue clockwise around bottle" if any(name in missing for name in ("left", "right", "rear")) else "tilt slightly to capture top/bottom"
    return {
        "covered_view_angle_deg": round(covered_angle, 1),
        "estimated_surface_coverage_percent": round(float(coverage), 1),
        "observed_front": observed_front,
        "observed_left": observed_left,
        "observed_right": observed_right,
        "observed_rear": observed_rear,
        "observed_top": observed_top,
        "observed_bottom": observed_bottom,
        "missing_view_directions": missing,
        "recommended_camera_motion": recommended,
    }


def _session_dir(session_id: str | None) -> Path:
    sid = session_id or "no_session"
    return MULTIVIEW_SESSIONS_DIR / sid


def _bool_from_report(report: dict[str, Any], key: str, default: bool = False) -> bool:
    value = report.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ok", "valid", "current"}
    return bool(value)


def _float_from_report(report: dict[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        value = report.get(key)
        return default if value is None else float(value)
    except Exception:
        return default


def _point3_from_report(report: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        value = report.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(arr)):
                return arr
        except Exception:
            continue
    return None


def _looks_like_bbox_roi(source: Any) -> bool:
    text = str(source or "").strip().lower()
    return "bbox" in text or "box" in text or text in {"extent", "fallback"}


def multiview_target_context(report: dict[str, Any] | None) -> dict[str, Any]:
    """Return only explicit, persistent target-lock evidence.

    A non-empty ``target_lock`` mapping is not evidence of a valid lock.  The
    live reporter intentionally emits a small structural mapping even while no
    target is locked, so scan admission requires both an explicit LOCKED or
    TRACKING state and ``locked_target.valid is True`` (or its flattened
    equivalent).
    """
    report = report or {}
    target_lock = report.get("target_lock") if isinstance(report.get("target_lock"), dict) else {}
    locked_target = report.get("locked_target") if isinstance(report.get("locked_target"), dict) else {}
    if not locked_target and isinstance(target_lock.get("locked_target"), dict):
        locked_target = target_lock["locked_target"]
    target_status = str(
        report.get("target_status")
        or target_lock.get("target_status")
        or ""
    ).upper()
    explicit_valid = (
        report.get("locked_target_valid") is True
        or locked_target.get("valid") is True
    )
    target_id = (
        report.get("pointcloud_target_id")
        or report.get("bottle_target_id")
        or locked_target.get("bottle_target_id")
        or locked_target.get("target_id")
        or (locked_target.get("locked_bottle_geometry") or {}).get("target_id")
    )
    target_scope = (
        report.get("pointcloud_target_id_scope")
        or report.get("target_id_scope")
        or (locked_target.get("locked_bottle_geometry") or {}).get("target_id_scope")
        or ("LOCK_PERSISTENT" if target_id is not None and explicit_valid else None)
    )
    status_valid = target_status in {"TRACKING", "LOCKED"}
    scope_text = str(target_scope or "").upper()
    identity_valid = (
        target_id is not None
        and str(target_id).strip() != ""
        and "FRAME_LOCAL" not in scope_text
    )
    return {
        "target_status": target_status or None,
        "locked_target_valid": bool(explicit_valid),
        "target_id": None if target_id is None else str(target_id),
        "target_id_scope": None if target_scope is None else str(target_scope),
        "identity_valid": bool(identity_valid),
        "valid": bool(status_valid and explicit_valid and identity_valid),
    }


def multiview_input_gate(
    report: dict[str, Any] | None,
    *,
    scene_points: np.ndarray | None = None,
    bottle_points: np.ndarray | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Validate that a single-frame cloud is safe to use as a scan keyframe.

    The gate intentionally prefers cleaned/locked masks over bbox crops.  Bbox
    crops are treated as invalid unless an explicit config/test override allows
    them and their depth/geometry metadata is strong enough.
    """
    report = report or {}
    now = _now() if timestamp is None else float(timestamp)
    scene_count = int(len(_as_points(scene_points))) if scene_points is not None else int(report.get("scene_filtered_point_count") or 0)
    bottle_count = int(len(_as_points(bottle_points))) if bottle_points is not None else int(report.get("bottle_pointcloud_point_count") or report.get("bottle_filtered_point_count") or 0)
    roi_source = report.get("bottle_roi_source") or report.get("bottle_pointcloud_roi_source")
    freshness = str(report.get("pointcloud_freshness") or "").upper()
    pointcloud_ts = _float_from_report(report, "pointcloud_timestamp", None)
    age_sec = _float_from_report(report, "pointcloud_age_sec", None)
    if age_sec is None and pointcloud_ts is not None:
        age_sec = max(0.0, now - float(pointcloud_ts))
    target_context = multiview_target_context(report)
    target_status = str(target_context.get("target_status") or "")
    intrinsics_available = _bool_from_report(report, "pointcloud_intrinsics_available", bool(report.get("intrinsics") or report.get("camera_intrinsics")))
    rgb_frame_id = report.get("pointcloud_source_rgb_frame_id") or report.get("rgb_frame_id") or report.get("frame_id")
    depth_frame_id = report.get("pointcloud_source_depth_frame_id") or report.get("depth_frame_id") or report.get("frame_id")
    snapshot_consistent = _bool_from_report(report, "pointcloud_snapshot_consistent", False)
    scene_valid = _bool_from_report(report, "scene_cloud_valid", scene_count >= MULTIVIEW_MIN_SCENE_POINTS)
    bottle_crop_valid = _bool_from_report(report, "bottle_crop_valid", False)
    bottle_geometry_valid = _bool_from_report(report, "bottle_geometry_valid", False)
    cloud_valid = _bool_from_report(report, "cloud_valid_for_multiview_input", False)
    depth_ratio = _float_from_report(report, "bottle_pointcloud_valid_depth_ratio", _float_from_report(report, "valid_depth_ratio", 0.0)) or 0.0
    allow_bbox = bool(MULTIVIEW_ALLOW_BBOX_ROI or report.get("allow_bbox_multiview_input"))

    reasons: list[str] = []
    if freshness != "CURRENT":
        reasons.append("pointcloud_not_current")
    if pointcloud_ts is None or age_sec is None or age_sec > MULTIVIEW_MAX_POINTCLOUD_AGE_SEC:
        reasons.append("pointcloud_stale_or_missing_timestamp")
    if not snapshot_consistent:
        reasons.append("pointcloud_snapshot_inconsistent")
    if not scene_valid or scene_count < MULTIVIEW_MIN_SCENE_POINTS:
        reasons.append("scene_cloud_invalid_or_too_small")
    if not bottle_crop_valid:
        reasons.append("bottle_crop_invalid")
    if not bottle_geometry_valid:
        reasons.append("bottle_geometry_invalid")
    if not cloud_valid:
        reasons.append("single_frame_cloud_not_valid_for_multiview_input")
    if rgb_frame_id is None or depth_frame_id is None:
        reasons.append("missing_source_frame_ids")
    if not intrinsics_available:
        reasons.append("camera_intrinsics_missing")
    if target_status not in {"TRACKING", "LOCKED"}:
        reasons.append("locked_or_stably_tracked_target_required")
    if not target_context.get("locked_target_valid"):
        reasons.append("explicit_valid_locked_target_required")
    if not target_context.get("identity_valid"):
        reasons.append("persistent_target_identity_required")
    if bottle_count < MULTIVIEW_MIN_BOTTLE_POINTS:
        reasons.append("too_few_bottle_points")
    if _looks_like_bbox_roi(roi_source):
        if not allow_bbox:
            reasons.append("bbox_roi_not_allowed_for_multiview")
        elif depth_ratio < max(0.75, MULTIVIEW_MIN_BOTTLE_DEPTH_VALID_RATIO):
            reasons.append("bbox_roi_depth_consistency_too_low")

    valid = not reasons
    return {
        "multiview_input_valid": valid,
        "multiview_input_reject_reason": None if valid else reasons[0],
        "multiview_input_reject_reasons": reasons,
        "multiview_input_scene_point_count": scene_count,
        "multiview_input_bottle_point_count": bottle_count,
        "multiview_input_roi_source": roi_source,
        "multiview_input_pointcloud_age_sec": None if age_sec is None else round(float(age_sec), 3),
        "multiview_input_snapshot_consistent": bool(snapshot_consistent),
        "multiview_input_intrinsics_available": bool(intrinsics_available),
        "multiview_input_rgb_frame_id": rgb_frame_id,
        "multiview_input_depth_frame_id": depth_frame_id,
        "multiview_input_target_status": target_status or None,
        "multiview_input_locked_target_valid": bool(target_context.get("locked_target_valid")),
        "multiview_input_target_id": target_context.get("target_id"),
        "multiview_input_target_id_scope": target_context.get("target_id_scope"),
        "multiview_input_depth_valid_ratio": round(float(depth_ratio), 4),
        "multiview_input_bbox_allowed": bool(allow_bbox),
    }


def write_ascii_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> bool:
    """Write a small ASCII PLY point cloud."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts, cols = finite_points(points, colors)
    cols = _as_colors(cols, len(pts)) if cols is not None else np.full((len(pts), 3), 0.85, dtype=np.float32)
    try:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with open(tmp, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
            for p, c in zip(pts, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        tmp.replace(path)
        return True
    except Exception:
        return False


@dataclass
class PointCloudFusionResult:
    """Summary returned after a fusion update."""

    accepted: bool
    keyframe_id: int | None = None
    rejection_reason: str | None = None
    raw_accumulated_point_count: int = 0
    filtered_accumulated_point_count: int = 0
    fused_voxel_count: int = 0
    average_observations_per_voxel: float = 0.0
    model_coverage_score: float = 0.0
    model_quality_score: float = 0.0


@dataclass
class ReconstructionStatus:
    """Serializable reconstruction status for dashboard/status JSON."""

    reconstruction_session_id: str | None = None
    reconstruction_target_id: str | None = None
    reconstruction_target_id_scope: str | None = None
    scan_status: str = SCAN_IDLE
    session_start_timestamp: float | None = None
    session_state_updated_timestamp: float | None = None
    reference_coordinate_frame: str = "reconstruction_world"
    pose_source: str = "registration_unavailable"
    pose_confidence: float = 0.0
    pose_matrix_4x4: list[list[float]] | None = None
    pose_timestamp: float | None = None
    pose_valid: bool = False
    pose_failure_reason: str | None = None
    transform_valid: bool = False
    accepted_keyframe_count: int = 0
    rejected_keyframe_count: int = 0
    candidate_frame_count: int = 0
    reconstruction_queue_size: int = 0
    reconstruction_skipped_candidates: int = 0
    raw_accumulated_point_count: int = 0
    filtered_accumulated_point_count: int = 0
    accumulated_scene_point_count: int = 0
    accumulated_bottle_point_count: int = 0
    fused_voxel_count: int = 0
    average_observations_per_voxel: float = 0.0
    model_coverage_score: float = 0.0
    model_quality_score: float = 0.0
    covered_view_angle_deg: float = 0.0
    estimated_surface_coverage_percent: float = 0.0
    missing_view_directions: list[str] = field(default_factory=list)
    recommended_camera_motion: str = "start scan"
    registration_fitness: float | None = None
    registration_inlier_rmse_m: float | None = None
    overlap_ratio: float | None = None
    last_registration_fitness: float | None = None
    last_registration_inlier_rmse_m: float | None = None
    registration_ms: float | None = None
    fusion_ms: float | None = None
    preview_ms: float | None = None
    reconstruction_bottleneck: str | None = None
    estimated_translation_mm: float | None = None
    estimated_rotation_deg: float | None = None
    bottle_static: bool = True
    bottle_world_centroid_delta_mm: float | None = None
    bottle_pose_consistency_score: float | None = None
    reconstruction_requires_reset: bool = False
    worker_busy: bool = False
    last_rejection_reason: str | None = None
    current_view_angle_deg: float | None = None
    reconstruction_age_sec: float | None = None
    reference_frame_id: int | str | None = None
    reference_timestamp: float | None = None
    latest_frame_id: int | str | None = None
    latest_keyframe_timestamp: float | None = None
    latest_candidate_timestamp: float | None = None
    fusion_mode: str = MULTIVIEW_FUSION_MODE
    voxel_size_m: float = MULTIVIEW_VOXEL_SIZE_M
    tsdf_enabled: bool = MULTIVIEW_TSDF_ENABLED
    davis_motion_gate_enabled: bool = False
    davis_motion_too_high: bool = False
    keyframe_skipped_due_to_motion: bool = False
    multiview_input_valid: bool = False
    multiview_input_reject_reason: str | None = None
    multiview_input_reject_reasons: list[str] = field(default_factory=list)
    multiview_input_scene_point_count: int = 0
    multiview_input_bottle_point_count: int = 0
    multiview_input_roi_source: str | None = None
    multiview_input_pointcloud_age_sec: float | None = None
    multiview_input_snapshot_consistent: bool = False
    multiview_input_intrinsics_available: bool = False
    multiview_input_rgb_frame_id: int | str | None = None
    multiview_input_depth_frame_id: int | str | None = None
    multiview_input_target_status: str | None = None
    multiview_input_locked_target_valid: bool = False
    multiview_input_target_id: str | None = None
    multiview_input_target_id_scope: str | None = None
    multiview_input_depth_valid_ratio: float | None = None
    multiview_input_bbox_allowed: bool = False
    contact_support_status: str = "unavailable"
    contact_support_left_distance_mm: float | None = None
    contact_support_right_distance_mm: float | None = None
    contact_support_left_points: int = 0
    contact_support_right_points: int = 0
    contact_width_3d_mm: float | None = None
    contact_candidates_enabled: bool = MULTIVIEW_ENABLE_CONTACT_CANDIDATES
    drift_warning: str | None = None
    ghosting_score: float | None = None
    bottle_profile_valid: bool | None = None
    coverage_confidence: float = 0.0
    output_session_dir: str | None = None
    live_overlay_fps_during_scan: float | None = None


@dataclass
class PoseEstimate:
    transform_to_reference: np.ndarray
    source: str
    confidence: float
    valid: bool
    odometry_success: bool = False
    icp_success: bool = False
    fitness: float | None = None
    inlier_rmse_m: float | None = None
    overlap_ratio: float | None = None
    rejection_reason: str | None = None


class CameraPoseEstimator:
    """Estimate current-camera to reconstruction-reference transform."""

    def __init__(
        self,
        *,
        min_registration_fitness: float = MULTIVIEW_MIN_REGISTRATION_FITNESS,
        max_inlier_rmse_m: float = MULTIVIEW_MAX_INLIER_RMSE_M,
        max_translation_step_m: float = MULTIVIEW_MAX_TRANSLATION_STEP_M,
        max_rotation_step_deg: float = MULTIVIEW_MAX_ROTATION_STEP_DEG,
        min_overlap_ratio: float = MULTIVIEW_MIN_OVERLAP_RATIO,
        allow_numpy_centroid_registration: bool = MULTIVIEW_ALLOW_NUMPY_CENTROID_REGISTRATION,
        use_open3d: bool = True,
    ):
        self.min_registration_fitness = float(min_registration_fitness)
        self.max_inlier_rmse_m = float(max_inlier_rmse_m)
        self.max_translation_step_m = float(max_translation_step_m)
        self.max_rotation_step_deg = float(max_rotation_step_deg)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.allow_numpy_centroid_registration = bool(allow_numpy_centroid_registration)
        self.o3d = _try_import_open3d() if use_open3d else None

    def estimate(
        self,
        *,
        reference_scene_points: np.ndarray | None,
        current_scene_points: np.ndarray,
        robot_pose: np.ndarray | None = None,
        tf_camera_pose: np.ndarray | None = None,
        previous_pose_to_reference: np.ndarray | None = None,
    ) -> PoseEstimate:
        if robot_pose is not None:
            return self._gate(PoseEstimate(np.asarray(robot_pose, dtype=np.float64), "robot_pose_with_hand_eye_calibration", 1.0, True), previous_pose_to_reference)
        if tf_camera_pose is not None:
            return self._gate(PoseEstimate(np.asarray(tf_camera_pose, dtype=np.float64), "ros_tf_camera_pose", 0.9, True), previous_pose_to_reference)
        ref = _as_points(reference_scene_points)
        cur = _as_points(current_scene_points)
        if len(ref) == 0:
            return PoseEstimate(np.eye(4), "reference_initial", 1.0, True, fitness=1.0, inlier_rmse_m=0.0, overlap_ratio=1.0)
        if len(cur) < 50 or len(ref) < 50:
            return PoseEstimate(np.eye(4), "registration_unavailable", 0.0, False, rejection_reason="too_few_scene_points_for_registration")
        if self.o3d is not None:
            estimate = self._estimate_open3d_icp(ref, cur)
            return self._gate(estimate, previous_pose_to_reference)
        if self.allow_numpy_centroid_registration:
            # Test fallback only.  It is intentionally opt-in because centroid
            # alignment is not a trustworthy live camera pose source.
            delta = np.mean(ref, axis=0) - np.mean(cur, axis=0)
            t = make_translation_transform(delta)
            fitness = 1.0
            return self._gate(PoseEstimate(t, "rgbd_odometry_unavailable_numpy_centroid", fitness, True, fitness=fitness, inlier_rmse_m=0.0, overlap_ratio=fitness), previous_pose_to_reference)
        return PoseEstimate(np.eye(4), "registration_unavailable", 0.0, False, rejection_reason="open3d_unavailable_and_no_pose_provider")

    def _estimate_open3d_icp(self, reference_points: np.ndarray, current_points: np.ndarray) -> PoseEstimate:
        o3d = self.o3d
        assert o3d is not None
        try:
            ref_sample, _ = _sample_points(reference_points, max_points=20_000)
            cur_sample, _ = _sample_points(current_points, max_points=20_000)
            ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_sample.astype(np.float64)))
            cur = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cur_sample.astype(np.float64)))
            voxel = max(0.006, float(MULTIVIEW_VOXEL_SIZE_M) * 1.5)
            ref = ref.voxel_down_sample(voxel)
            cur = cur.voxel_down_sample(voxel)
            if len(ref.points) < 50 or len(cur.points) < 50:
                return PoseEstimate(np.eye(4), "rgbd_odometry_plus_icp", 0.0, False, rejection_reason="too_few_downsampled_scene_points")
            radius = voxel * 3.0
            ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=24))
            cur.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=24))
            result = o3d.pipelines.registration.registration_icp(
                cur,
                ref,
                max_correspondence_distance=max(0.025, voxel * 5.0),
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            confidence = float(result.fitness)
            return PoseEstimate(
                np.asarray(result.transformation, dtype=np.float64),
                "rgbd_odometry_plus_icp",
                confidence,
                True,
                odometry_success=False,
                icp_success=True,
                fitness=float(result.fitness),
                inlier_rmse_m=float(result.inlier_rmse),
                overlap_ratio=float(result.fitness),
            )
        except Exception as exc:
            return PoseEstimate(np.eye(4), "rgbd_odometry_plus_icp", 0.0, False, icp_success=False, rejection_reason=f"icp_error:{exc}")

    def _gate(self, estimate: PoseEstimate, previous_pose_to_reference: np.ndarray | None = None) -> PoseEstimate:
        if not estimate.valid:
            return estimate
        # The translation/rotation thresholds bound the INCREMENTAL step between
        # consecutive accepted keyframes, not the cumulative pose relative to the
        # first keyframe. Measuring them against the cumulative transform-to-
        # reference made an orbit impossible: once the camera moved past the
        # threshold from the start, every keyframe was rejected. Compute the true
        # step relative to the previous accepted keyframe pose.
        cumulative = np.asarray(estimate.transform_to_reference, dtype=np.float64)
        if previous_pose_to_reference is not None:
            try:
                step = np.linalg.inv(np.asarray(previous_pose_to_reference, dtype=np.float64)) @ cumulative
            except np.linalg.LinAlgError:
                step = cumulative
        else:
            step = cumulative
        translation = transform_translation_m(step)
        rotation = transform_rotation_deg(step)
        fitness = 1.0 if estimate.fitness is None else float(estimate.fitness)
        rmse = 0.0 if estimate.inlier_rmse_m is None else float(estimate.inlier_rmse_m)
        overlap = fitness if estimate.overlap_ratio is None else float(estimate.overlap_ratio)
        reason = None
        if fitness < self.min_registration_fitness:
            reason = "registration_fitness_below_threshold"
        elif rmse > self.max_inlier_rmse_m:
            reason = "registration_rmse_above_threshold"
        elif overlap < self.min_overlap_ratio:
            reason = "registration_overlap_below_threshold"
        elif translation > self.max_translation_step_m:
            reason = "translation_step_above_threshold"
        elif rotation > self.max_rotation_step_deg:
            reason = "rotation_step_above_threshold"
        if reason:
            estimate.valid = False
            estimate.rejection_reason = reason
            estimate.confidence = min(float(estimate.confidence), 0.0)
        return estimate


class MultiViewBottleReconstructor:
    """Fuse accepted bottle keyframes into a persistent reconstruction."""

    def __init__(
        self,
        *,
        output_dir: str | Path = MULTIVIEW_OUTPUT_DIR,
        pose_estimator: CameraPoseEstimator | None = None,
        voxel_size_m: float = MULTIVIEW_VOXEL_SIZE_M,
        max_accumulated_points: int = MULTIVIEW_MAX_ACCUMULATED_POINTS,
    ):
        self.output_dir = Path(output_dir)
        self.pose_estimator = pose_estimator or CameraPoseEstimator()
        self.voxel_size_m = float(voxel_size_m)
        self.max_accumulated_points = int(max_accumulated_points)
        self.lock = threading.RLock()
        self.reset(clear_files=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_status()

    def reset(self, *, clear_files: bool = True) -> ReconstructionStatus:
        with getattr(self, "lock", threading.RLock()):
            self.reconstruction_session_id: str | None = None
            self.reconstruction_target_id: str | None = None
            self.reconstruction_target_id_scope: str | None = None
            self.reference_frame_id: int | str | None = None
            self.reference_timestamp: float | None = None
            self.reference_scene_points = np.empty((0, 3), dtype=np.float32)
            self.current_camera_pose_4x4 = np.eye(4, dtype=np.float64)
            self.accumulated_camera_trajectory: list[dict[str, Any]] = []
            self.accepted_keyframe_count = 0
            self.rejected_keyframe_count = 0
            self.voxels: dict[tuple[int, int, int], dict[str, np.ndarray | int]] = {}
            self.scene_voxels: dict[tuple[int, int, int], dict[str, np.ndarray | int]] = {}
            self.first_bottle_centroid_world: np.ndarray | None = None
            self.last_keyframe_pose = np.eye(4, dtype=np.float64)
            self.last_keyframe_timestamp: float | None = None
            self.session_start_timestamp: float | None = None
            self.candidate_frame_count = 0
            self.reconstruction_skipped_candidates = 0
            self.last_contact_support: dict[str, Any] = {}
            self.paused_reason: str | None = None
            self.status = ReconstructionStatus(scan_status=SCAN_IDLE)
            if clear_files:
                self._clear_generated_artifacts()
                self._write_session_files()
                self.write_preview()
                self.write_status()
            return self.status

    def start_scan(
        self,
        *,
        require_locked: bool = True,
        locked_valid: bool = True,
        target_id: str | None = None,
        target_id_scope: str | None = None,
        source: str = "dashboard",
    ) -> ReconstructionStatus:
        with self.lock:
            if require_locked and not locked_valid:
                self.status.scan_status = SCAN_WAITING
                self.status.last_rejection_reason = "locked_bottle_required"
                self.status.session_state_updated_timestamp = time.time()
                self.write_status()
                return self.status
            self.reset(clear_files=False)
            self.reconstruction_session_id = f"multiview-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            self.reconstruction_target_id = None if target_id is None else str(target_id)
            self.reconstruction_target_id_scope = None if target_id_scope is None else str(target_id_scope)
            self.session_start_timestamp = time.time()
            self.status = ReconstructionStatus(
                reconstruction_session_id=self.reconstruction_session_id,
                reconstruction_target_id=self.reconstruction_target_id,
                reconstruction_target_id_scope=self.reconstruction_target_id_scope,
                scan_status=SCAN_INITIALIZING,
                session_start_timestamp=self.session_start_timestamp,
                session_state_updated_timestamp=self.session_start_timestamp,
                fusion_mode=MULTIVIEW_FUSION_MODE,
                voxel_size_m=self.voxel_size_m,
                tsdf_enabled=MULTIVIEW_TSDF_ENABLED,
                output_session_dir=str(_session_dir(self.reconstruction_session_id)),
            )
            self.paused_reason = None
            self._clear_generated_artifacts()
            self._write_session_files(extra={"started_by": source})
            self.write_preview()
            self.write_status()
            return self.status

    def pause_scan(self, reason: str = "paused_by_user") -> ReconstructionStatus:
        with self.lock:
            if self.status.scan_status != SCAN_IDLE:
                self.status.scan_status = SCAN_PAUSED
                self.paused_reason = reason
                self.status.last_rejection_reason = reason
                self.status.session_state_updated_timestamp = time.time()
            self.write_status()
            return self.status

    def resume_scan(self) -> ReconstructionStatus:
        with self.lock:
            if self.reconstruction_session_id:
                self.status.scan_status = SCAN_INITIALIZING if self.accepted_keyframe_count == 0 else SCAN_SCANNING
                self.paused_reason = None
                self.status.session_state_updated_timestamp = time.time()
            self.write_status()
            return self.status

    def finish_scan(self, reason: str = "finished_by_user") -> ReconstructionStatus:
        with self.lock:
            if self.reconstruction_session_id:
                self.status.scan_status = SCAN_COMPLETE
                self.status.last_rejection_reason = reason
                self.status.session_state_updated_timestamp = time.time()
                self.save_accumulated_ply()
            self._write_session_files(extra={"finished_reason": reason})
            self.write_preview()
            self.write_status()
            return self.status

    def process_keyframe(
        self,
        *,
        scene_points: np.ndarray,
        scene_colors: np.ndarray | None = None,
        bottle_points: np.ndarray | None = None,
        bottle_colors: np.ndarray | None = None,
        report: dict[str, Any] | None = None,
        timestamp: float | None = None,
        frame_id: int | str | None = None,
        robot_pose: np.ndarray | None = None,
        tf_camera_pose: np.ndarray | None = None,
    ) -> PointCloudFusionResult:
        """Try to accept and fuse one candidate keyframe."""
        with self.lock:
            timestamp = _now() if timestamp is None else float(timestamp)
            report = report or {}
            self.candidate_frame_count += 1
            self.status.candidate_frame_count = self.candidate_frame_count
            self.status.latest_candidate_timestamp = timestamp
            active_states = {SCAN_INITIALIZING, SCAN_SCANNING, SCAN_REGISTRATION_WARNING}
            if self.status.scan_status not in active_states:
                return self._reject("scan_not_active", frame_id, timestamp, write=False)
            scene_pts, scene_cols = finite_points(scene_points, scene_colors)
            bottle_pts, bottle_cols = finite_points(_as_points(bottle_points), bottle_colors)
            bottle_cols = _as_colors(bottle_cols, len(bottle_pts)) if bottle_cols is not None else _as_colors(None, len(bottle_pts))
            gate = multiview_input_gate(report, scene_points=scene_pts, bottle_points=bottle_pts, timestamp=timestamp)
            self._apply_input_gate_status(gate)
            if not gate["multiview_input_valid"]:
                return self._reject(str(gate["multiview_input_reject_reason"] or "multiview_input_invalid"), frame_id, timestamp, extra=gate)
            candidate_target_id = str(gate.get("multiview_input_target_id") or "")
            candidate_target_scope = gate.get("multiview_input_target_id_scope")
            if self.reconstruction_target_id is None:
                self.reconstruction_target_id = candidate_target_id
                self.reconstruction_target_id_scope = None if candidate_target_scope is None else str(candidate_target_scope)
                self.status.reconstruction_target_id = self.reconstruction_target_id
                self.status.reconstruction_target_id_scope = self.reconstruction_target_id_scope
            elif candidate_target_id != self.reconstruction_target_id:
                self.status.scan_status = SCAN_BOTTLE_MOVED
                self.status.reconstruction_requires_reset = True
                self.paused_reason = "target_identity_changed_requires_reset"
                return self._reject(
                    "target_identity_changed_requires_reset",
                    frame_id,
                    timestamp,
                    extra={
                        **gate,
                        "reconstruction_target_id": self.reconstruction_target_id,
                        "candidate_target_id": candidate_target_id,
                    },
                )
            if len(bottle_pts) == 0:
                return self._reject("empty_bottle_cloud", frame_id, timestamp)
            depth_valid_ratio = self._depth_valid_ratio(report, bottle_pts)
            if depth_valid_ratio < MULTIVIEW_MIN_BOTTLE_DEPTH_VALID_RATIO:
                return self._reject("bottle_depth_valid_ratio_below_threshold", frame_id, timestamp)

            registration_t0 = time.perf_counter()
            pose = self.pose_estimator.estimate(
                reference_scene_points=None if self.accepted_keyframe_count == 0 else self.reference_scene_points,
                current_scene_points=scene_pts,
                robot_pose=robot_pose,
                tf_camera_pose=tf_camera_pose,
                previous_pose_to_reference=None if self.accepted_keyframe_count == 0 else self.last_keyframe_pose,
            )
            self.status.registration_ms = round((time.perf_counter() - registration_t0) * 1000.0, 2)
            self._update_pose_status(pose, frame_id, timestamp)
            if not pose.valid:
                return self._reject(pose.rejection_reason or "registration_failed", frame_id, timestamp, pose=pose)

            bottle_world = transform_points(bottle_pts, pose.transform_to_reference)
            scene_world = transform_points(scene_pts, pose.transform_to_reference)
            moved_reason = self._bottle_moved_reason(bottle_world)
            if moved_reason:
                self.status.scan_status = SCAN_BOTTLE_MOVED
                self.status.reconstruction_requires_reset = True
                self.paused_reason = moved_reason
                return self._reject(moved_reason, frame_id, timestamp, pose=pose)

            if self.accepted_keyframe_count > 0 and self._is_duplicate_keyframe(pose.transform_to_reference, timestamp):
                return self._reject("duplicate_view", frame_id, timestamp, pose=pose)

            if self.accepted_keyframe_count == 0:
                self.reference_scene_points, _ = _sample_points(scene_pts, max_points=80_000)
                self.reference_frame_id = frame_id
                self.reference_timestamp = timestamp
                self.status.reference_frame_id = frame_id
                self.status.reference_timestamp = timestamp
                self.first_bottle_centroid_world = np.mean(bottle_world, axis=0)

            fusion_t0 = time.perf_counter()
            self._fuse_voxel_cloud(bottle_world, bottle_cols, self.voxels, voxel_size_m=MULTIVIEW_BOTTLE_VOXEL_SIZE_M)
            if len(scene_world):
                scene_world_sample, scene_cols_sample = _sample_points(scene_world, scene_cols, max_points=40_000)
                self._fuse_voxel_cloud(scene_world_sample, scene_cols_sample, self.scene_voxels, max_voxels=MULTIVIEW_MAX_SCENE_VOXELS, voxel_size_m=MULTIVIEW_SCENE_VOXEL_SIZE_M)
            self.status.fusion_ms = round((time.perf_counter() - fusion_t0) * 1000.0, 2)
            self.accepted_keyframe_count += 1
            keyframe_id = self.accepted_keyframe_count
            self.last_keyframe_pose = np.asarray(pose.transform_to_reference, dtype=np.float64)
            self.last_keyframe_timestamp = timestamp
            self.current_camera_pose_4x4 = np.asarray(pose.transform_to_reference, dtype=np.float64)
            self.accumulated_camera_trajectory.append({
                "keyframe_id": keyframe_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "pose_source": pose.source,
                "pose_confidence": round(float(pose.confidence), 4),
                "transform_to_reference": np.round(pose.transform_to_reference, 6).astype(float).tolist(),
            })
            self._trim_voxels()
            result = self._fusion_result(True, keyframe_id=keyframe_id)
            self._refresh_status_after_fusion(pose, frame_id, timestamp, result)
            contact_support = self._contact_support_from_report(report, pose.transform_to_reference)
            self.last_contact_support = contact_support
            self._apply_contact_support_status(contact_support)
            record = {
                "session_id": self.reconstruction_session_id,
                "keyframe_id": keyframe_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "candidate_frame_count": self.candidate_frame_count,
                "pose_source": pose.source,
                "pose_confidence": pose.confidence,
                "pose_matrix_4x4": np.round(pose.transform_to_reference, 6).astype(float).tolist(),
                "registration_fitness": pose.fitness,
                "registration_inlier_rmse_m": pose.inlier_rmse_m,
                "overlap_ratio": pose.overlap_ratio,
                "input_gate": gate,
                "bottle_point_count": int(len(bottle_pts)),
                "fused_voxel_count": result.fused_voxel_count,
                "contact_support": contact_support,
            }
            self._append_accepted(record)
            self._write_session_files()
            self.write_preview()
            self.write_status()
            return result

    def _depth_valid_ratio(self, report: dict[str, Any], bottle_points: np.ndarray) -> float:
        for key in ("bottle_pointcloud_valid_depth_ratio", "valid_depth_ratio", "bottle_depth_valid_ratio"):
            value = report.get(key)
            try:
                if value is not None:
                    v = float(value)
                    return v if v <= 1.0 else v / 100.0
            except Exception:
                pass
        return 1.0 if len(bottle_points) else 0.0

    def _is_duplicate_keyframe(self, pose: np.ndarray, timestamp: float) -> bool:
        translation_mm = transform_translation_m(np.linalg.inv(self.last_keyframe_pose) @ pose) * 1000.0
        rotation_deg = transform_rotation_deg(np.linalg.inv(self.last_keyframe_pose) @ pose)
        return translation_mm < MULTIVIEW_MIN_KEYFRAME_TRANSLATION_MM and rotation_deg < MULTIVIEW_MIN_KEYFRAME_ROTATION_DEG

    def _bottle_moved_reason(self, bottle_world: np.ndarray) -> str | None:
        if self.first_bottle_centroid_world is None or len(bottle_world) == 0:
            return None
        centroid = np.mean(bottle_world, axis=0)
        delta_mm = float(np.linalg.norm(centroid - self.first_bottle_centroid_world) * 1000.0)
        self.status.bottle_world_centroid_delta_mm = round(delta_mm, 2)
        self.status.bottle_static = delta_mm <= MULTIVIEW_BOTTLE_MOVED_THRESHOLD_MM
        self.status.bottle_pose_consistency_score = round(max(0.0, 1.0 - delta_mm / max(MULTIVIEW_BOTTLE_MOVED_THRESHOLD_MM, 1e-6)), 4)
        if delta_mm > MULTIVIEW_BOTTLE_MOVED_THRESHOLD_MM:
            return "bottle_moved_requires_reset"
        return None

    def _apply_input_gate_status(self, gate: dict[str, Any]) -> None:
        for key, value in gate.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
        self.status.last_rejection_reason = gate.get("multiview_input_reject_reason") if not gate.get("multiview_input_valid") else self.status.last_rejection_reason

    def _contact_support_from_report(self, report: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
        left = _point3_from_report(report, "left_contact_xyz_m", "left_contact_3d_m", "left_contact_3d")
        right = _point3_from_report(report, "right_contact_xyz_m", "right_contact_3d_m", "right_contact_3d")
        bottle_pts, _ = self.fused_points(include_scene=False)
        if left is None or right is None:
            return {
                "status": "unavailable",
                "reason": "missing_locked_contact_xyz",
                "contacts_in_reconstruction_frame": False,
                "contact_candidates_enabled": bool(MULTIVIEW_ENABLE_CONTACT_CANDIDATES),
            }
        contacts_world = transform_points(np.vstack([left, right]).astype(np.float32), transform)
        width_mm = float(np.linalg.norm(contacts_world[1] - contacts_world[0]) * 1000.0)

        def nearest_summary(point: np.ndarray) -> tuple[float | None, int]:
            if len(bottle_pts) == 0:
                return None, 0
            d = np.linalg.norm(bottle_pts - point.reshape(1, 3), axis=1)
            return float(np.min(d) * 1000.0), int(np.count_nonzero(d <= MULTIVIEW_CONTACT_SURFACE_TOLERANCE_M))

        left_dist, left_support = nearest_summary(contacts_world[0])
        right_dist, right_support = nearest_summary(contacts_world[1])
        supported = (
            left_dist is not None
            and right_dist is not None
            and left_dist <= MULTIVIEW_CONTACT_SURFACE_TOLERANCE_M * 1000.0
            and right_dist <= MULTIVIEW_CONTACT_SURFACE_TOLERANCE_M * 1000.0
        )
        return {
            "status": "supported" if supported else "weak_or_outside_cloud",
            "contacts_in_reconstruction_frame": True,
            "left_contact_xyz_world_m": np.round(contacts_world[0], 5).astype(float).tolist(),
            "right_contact_xyz_world_m": np.round(contacts_world[1], 5).astype(float).tolist(),
            "left_contact_surface_distance_mm": None if left_dist is None else round(left_dist, 2),
            "right_contact_surface_distance_mm": None if right_dist is None else round(right_dist, 2),
            "left_contact_support_point_count": left_support,
            "right_contact_support_point_count": right_support,
            "contact_width_3d_mm": round(width_mm, 2),
            "contact_candidates_enabled": bool(MULTIVIEW_ENABLE_CONTACT_CANDIDATES),
        }

    def _apply_contact_support_status(self, support: dict[str, Any]) -> None:
        self.status.contact_support_status = str(support.get("status") or "unavailable")
        self.status.contact_support_left_distance_mm = support.get("left_contact_surface_distance_mm")
        self.status.contact_support_right_distance_mm = support.get("right_contact_surface_distance_mm")
        self.status.contact_support_left_points = int(support.get("left_contact_support_point_count") or 0)
        self.status.contact_support_right_points = int(support.get("right_contact_support_point_count") or 0)
        self.status.contact_width_3d_mm = support.get("contact_width_3d_mm")

    def _session_path(self, name: str) -> Path:
        return _session_dir(self.reconstruction_session_id) / name

    def _append_accepted(self, record: dict[str, Any]) -> None:
        append_jsonl(MULTIVIEW_ACCEPTED_PATH, record)
        if self.reconstruction_session_id:
            append_jsonl(self._session_path("accepted_keyframes.jsonl"), record)
            self._append_trajectory_csv(record)

    def _append_rejected(self, record: dict[str, Any]) -> None:
        append_jsonl(MULTIVIEW_REJECTED_PATH, record)
        if self.reconstruction_session_id:
            append_jsonl(self._session_path("rejected_keyframes.jsonl"), record)

    def _append_trajectory_csv(self, record: dict[str, Any]) -> None:
        if not self.reconstruction_session_id:
            return
        path = self._session_path("camera_motion.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        transform = record.get("pose_matrix_4x4") or np.eye(4).tolist()
        try:
            t = np.asarray(transform, dtype=float).reshape(4, 4)[:3, 3] * 1000.0
        except Exception:
            t = np.zeros(3)
        with open(path, "a") as f:
            if new_file:
                f.write("timestamp,frame_id,keyframe_id,pose_source,pose_confidence,tx_mm,ty_mm,tz_mm,registration_fitness,registration_rmse_m\n")
            f.write(
                f"{record.get('timestamp')},{record.get('frame_id')},{record.get('keyframe_id')},"
                f"{record.get('pose_source')},{record.get('pose_confidence')},"
                f"{t[0]:.3f},{t[1]:.3f},{t[2]:.3f},"
                f"{record.get('registration_fitness')},{record.get('registration_inlier_rmse_m')}\n"
            )

    def _fuse_voxel_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray | None,
        voxels: dict[tuple[int, int, int], dict[str, np.ndarray | int]],
        *,
        max_voxels: int | None = None,
        voxel_size_m: float | None = None,
    ) -> None:
        pts, cols = finite_points(points, colors)
        cols = _as_colors(cols, len(pts)) if cols is not None else _as_colors(None, len(pts))
        if len(pts) == 0:
            return
        voxel = self.voxel_size_m if voxel_size_m is None else float(voxel_size_m)
        keys = np.floor(pts / max(voxel, 1e-6)).astype(np.int64)
        for key_arr, p, c in zip(keys, pts, cols):
            key = tuple(int(v) for v in key_arr)
            item = voxels.get(key)
            if item is None:
                voxels[key] = {"sum_xyz": p.astype(np.float64), "sum_rgb": c.astype(np.float64), "count": 1}
            else:
                item["sum_xyz"] = item["sum_xyz"] + p.astype(np.float64)  # type: ignore[operator]
                item["sum_rgb"] = item["sum_rgb"] + c.astype(np.float64)  # type: ignore[operator]
                item["count"] = int(item["count"]) + 1
        if max_voxels is not None and len(voxels) > max_voxels:
            self._trim_voxel_dict(voxels, max_voxels)

    def _trim_voxels(self) -> None:
        self._trim_voxel_dict(self.voxels, self.max_accumulated_points)

    def _trim_voxel_dict(self, voxels: dict[tuple[int, int, int], dict[str, np.ndarray | int]], max_count: int) -> None:
        if len(voxels) <= max_count:
            return
        ranked = sorted(voxels.items(), key=lambda kv: int(kv[1].get("count", 0)), reverse=True)[:max_count]
        voxels.clear()
        voxels.update(dict(ranked))

    def _clear_generated_artifacts(self) -> None:
        for path in (
            MULTIVIEW_ACCEPTED_PATH,
            MULTIVIEW_REJECTED_PATH,
            MULTIVIEW_TRAJECTORY_PATH,
            MULTIVIEW_SCENE_PLY_PATH,
            MULTIVIEW_BOTTLE_PLY_PATH,
            MULTIVIEW_MESH_PLY_PATH,
        ):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    def fused_points(self, *, include_scene: bool = False) -> tuple[np.ndarray, np.ndarray]:
        voxels = self.scene_voxels if include_scene else self.voxels
        if not voxels:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        pts = []
        cols = []
        for item in voxels.values():
            count = max(1, int(item["count"]))
            pts.append(np.asarray(item["sum_xyz"], dtype=np.float64) / count)
            cols.append(np.asarray(item["sum_rgb"], dtype=np.float64) / count)
        return np.asarray(pts, dtype=np.float32), np.clip(np.asarray(cols, dtype=np.float32), 0.0, 1.0)

    def save_accumulated_ply(self) -> dict[str, Any]:
        with self.lock:
            scene_pts, scene_cols = self.fused_points(include_scene=True)
            bottle_pts, bottle_cols = self.fused_points(include_scene=False)
            scene_ok = write_ascii_ply(MULTIVIEW_SCENE_PLY_PATH, scene_pts, scene_cols)
            bottle_ok = write_ascii_ply(MULTIVIEW_BOTTLE_PLY_PATH, bottle_pts, bottle_cols)
            session_scene = session_bottle = None
            if self.reconstruction_session_id:
                session_scene = self._session_path("scene_map.ply")
                session_bottle = self._session_path("bottle_model.ply")
                scene_ok = bool(write_ascii_ply(session_scene, scene_pts, scene_cols) and scene_ok)
                bottle_ok = bool(write_ascii_ply(session_bottle, bottle_pts, bottle_cols) and bottle_ok)
            self.write_status()
            return {
                "scene_saved": scene_ok,
                "bottle_saved": bottle_ok,
                "accumulated_scene_path": str(MULTIVIEW_SCENE_PLY_PATH),
                "accumulated_bottle_path": str(MULTIVIEW_BOTTLE_PLY_PATH),
                "session_scene_path": None if session_scene is None else str(session_scene),
                "session_bottle_path": None if session_bottle is None else str(session_bottle),
                "scene_points": int(len(scene_pts)),
                "bottle_points": int(len(bottle_pts)),
            }

    def save_mesh(self) -> dict[str, Any]:
        with self.lock:
            bottle_pts, bottle_cols = self.fused_points(include_scene=False)
            # TSDF/mesh extraction is intentionally disabled by default.  Save a
            # PLY point representation at the mesh path so the command has a
            # useful artifact while reporting that mesh extraction is pending.
            saved = write_ascii_ply(MULTIVIEW_MESH_PLY_PATH, bottle_pts, bottle_cols)
            session_mesh = None
            if self.reconstruction_session_id:
                session_mesh = self._session_path("bottle_mesh_or_placeholder.ply")
                saved = bool(write_ascii_ply(session_mesh, bottle_pts, bottle_cols) and saved)
            self.write_status()
            return {
                "mesh_saved": saved,
                "mesh_path": str(MULTIVIEW_MESH_PLY_PATH),
                "session_mesh_path": None if session_mesh is None else str(session_mesh),
                "mesh_status": "point_cloud_placeholder_tsdf_disabled" if not MULTIVIEW_TSDF_ENABLED else "tsdf_mesh_not_available",
                "points": int(len(bottle_pts)),
            }

    def rebuild_preview(self) -> dict[str, Any]:
        with self.lock:
            self.write_preview()
            self.write_status()
            return {"ok": True, "preview_path": str(MULTIVIEW_PREVIEW_PATH)}

    def _reject(
        self,
        reason: str,
        frame_id: int | str | None,
        timestamp: float,
        *,
        pose: PoseEstimate | None = None,
        extra: dict[str, Any] | None = None,
        write: bool = True,
    ) -> PointCloudFusionResult:
        if write:
            self.rejected_keyframe_count += 1
            self.status.rejected_keyframe_count = self.rejected_keyframe_count
            self.status.last_rejection_reason = reason
            if pose is not None:
                self._update_pose_status(pose, frame_id, timestamp)
            record = {
                "session_id": self.reconstruction_session_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "candidate_frame_count": self.candidate_frame_count,
                "reason": reason,
                "pose_source": None if pose is None else pose.source,
                "registration_fitness": None if pose is None else pose.fitness,
                "registration_inlier_rmse_m": None if pose is None else pose.inlier_rmse_m,
                "overlap_ratio": None if pose is None else pose.overlap_ratio,
                **(extra or {}),
            }
            self._append_rejected(record)
            self.write_status()
        return self._fusion_result(False, rejection_reason=reason)

    def _fusion_result(self, accepted: bool, keyframe_id: int | None = None, rejection_reason: str | None = None) -> PointCloudFusionResult:
        counts = [int(item["count"]) for item in self.voxels.values()]
        avg_obs = float(np.mean(counts)) if counts else 0.0
        coverage = _coverage_from_trajectory(self.accumulated_camera_trajectory)
        fused_count = len(self.voxels)
        return PointCloudFusionResult(
            accepted=accepted,
            keyframe_id=keyframe_id,
            rejection_reason=rejection_reason,
            raw_accumulated_point_count=int(sum(counts)),
            filtered_accumulated_point_count=fused_count,
            fused_voxel_count=fused_count,
            average_observations_per_voxel=round(avg_obs, 3),
            model_coverage_score=float(coverage["estimated_surface_coverage_percent"]),
            model_quality_score=round(min(1.0, fused_count / 5000.0) * 0.45 + min(1.0, avg_obs / 3.0) * 0.25 + float(coverage["estimated_surface_coverage_percent"]) / 100.0 * 0.30, 4),
        )

    def _update_pose_status(self, pose: PoseEstimate, frame_id: int | str | None, timestamp: float) -> None:
        self.status.pose_source = pose.source
        self.status.pose_confidence = round(float(pose.confidence), 4)
        self.status.transform_valid = bool(pose.valid)
        self.status.pose_valid = bool(pose.valid)
        self.status.pose_timestamp = timestamp
        self.status.pose_failure_reason = pose.rejection_reason
        self.status.pose_matrix_4x4 = np.round(pose.transform_to_reference, 6).astype(float).tolist()
        self.status.registration_fitness = None if pose.fitness is None else round(float(pose.fitness), 4)
        self.status.registration_inlier_rmse_m = None if pose.inlier_rmse_m is None else round(float(pose.inlier_rmse_m), 5)
        self.status.last_registration_fitness = self.status.registration_fitness
        self.status.last_registration_inlier_rmse_m = self.status.registration_inlier_rmse_m
        self.status.overlap_ratio = None if pose.overlap_ratio is None else round(float(pose.overlap_ratio), 4)
        self.status.estimated_translation_mm = round(transform_translation_m(pose.transform_to_reference) * 1000.0, 2)
        self.status.estimated_rotation_deg = round(transform_rotation_deg(pose.transform_to_reference), 2)
        self.status.current_view_angle_deg = self.status.estimated_rotation_deg
        self.status.latest_frame_id = frame_id

    def _refresh_status_after_fusion(
        self,
        pose: PoseEstimate,
        frame_id: int | str | None,
        timestamp: float,
        result: PointCloudFusionResult,
    ) -> None:
        coverage = _coverage_from_trajectory(self.accumulated_camera_trajectory)
        self.status.reconstruction_session_id = self.reconstruction_session_id
        self.status.reconstruction_target_id = self.reconstruction_target_id
        self.status.reconstruction_target_id_scope = self.reconstruction_target_id_scope
        self.status.scan_status = SCAN_SCANNING
        self.status.accepted_keyframe_count = self.accepted_keyframe_count
        self.status.rejected_keyframe_count = self.rejected_keyframe_count
        self.status.raw_accumulated_point_count = result.raw_accumulated_point_count
        self.status.filtered_accumulated_point_count = result.filtered_accumulated_point_count
        self.status.accumulated_bottle_point_count = result.filtered_accumulated_point_count
        self.status.accumulated_scene_point_count = len(self.scene_voxels)
        self.status.fused_voxel_count = result.fused_voxel_count
        self.status.average_observations_per_voxel = result.average_observations_per_voxel
        self.status.model_coverage_score = result.model_coverage_score
        self.status.model_quality_score = result.model_quality_score
        self.status.covered_view_angle_deg = float(coverage["covered_view_angle_deg"])
        self.status.estimated_surface_coverage_percent = float(coverage["estimated_surface_coverage_percent"])
        self.status.coverage_confidence = round(min(1.0, float(coverage["estimated_surface_coverage_percent"]) / 100.0), 4)
        self.status.missing_view_directions = list(coverage["missing_view_directions"])
        self.status.recommended_camera_motion = str(coverage["recommended_camera_motion"])
        self.status.latest_frame_id = frame_id
        self.status.latest_keyframe_timestamp = timestamp
        self.status.reference_frame_id = self.reference_frame_id
        self.status.reference_timestamp = self.reference_timestamp
        self.status.reconstruction_age_sec = None if self.reference_timestamp is None else round(max(0.0, time.time() - float(self.reference_timestamp)), 2)
        self._update_pose_status(pose, frame_id, timestamp)

    def preview_payload(self) -> dict[str, Any]:
        bottle_pts, bottle_cols = self.fused_points(include_scene=False)
        scene_pts, scene_cols = self.fused_points(include_scene=True)
        bottle_show, bottle_cols_show = _sample_points(bottle_pts, bottle_cols, MULTIVIEW_PREVIEW_MAX_POINTS)
        scene_show, scene_cols_show = _sample_points(scene_pts, scene_cols, min(10_000, MULTIVIEW_PREVIEW_MAX_POINTS // 4))
        bounds = _bounds(np.concatenate([bottle_show, scene_show], axis=0) if len(scene_show) and len(bottle_show) else (bottle_show if len(bottle_show) else scene_show))
        return {
            "timestamp": time.time(),
            "status": self.status.scan_status,
            "coordinate_frame": "reconstruction_world",
            "units": "meters",
            "positions_format": "flat_xyz",
            "colors_format": "flat_rgb_float_0_1",
            "scene": {
                "point_count": int(len(scene_show)),
                "positions": _flat(scene_show),
                "colors": _flat(scene_cols_show if scene_cols_show is not None else np.empty((0, 3), dtype=np.float32), 3),
            },
            "bottle": {
                "point_count": int(len(bottle_show)),
                "positions": _flat(bottle_show),
                "colors": _flat(bottle_cols_show if bottle_cols_show is not None else np.empty((0, 3), dtype=np.float32), 3),
                "raw_point_count": int(self.status.raw_accumulated_point_count),
                "filtered_point_count": int(self.status.filtered_accumulated_point_count),
                "display_point_count": int(len(bottle_show)),
                "valid": bool(len(bottle_show)),
                "roi_source": "accumulated_bottle_model",
            },
            "contacts": self.last_contact_support,
            "trajectory": self.accumulated_camera_trajectory[-120:],
            "bounds": bounds,
            "source": {
                "backend": "multiview_reconstruction",
                "original_scene_point_count": len(self.scene_voxels),
                "original_bottle_point_count": int(self.status.raw_accumulated_point_count),
                "filtered_bottle_point_count": int(self.status.filtered_accumulated_point_count),
                "display_bottle_point_count": int(len(bottle_show)),
                "json_max_points": int(MULTIVIEW_PREVIEW_MAX_POINTS),
                "color_mode": "rgb",
                "reconstruction_session_id": self.reconstruction_session_id,
            },
            "reconstruction": asdict(self.status),
        }

    def write_preview(self) -> None:
        t0 = time.perf_counter()
        payload = self.preview_payload()
        write_json_atomic(MULTIVIEW_PREVIEW_PATH, payload, indent=None)
        write_json_atomic(MULTIVIEW_LATEST_PREVIEW_PATH, payload, indent=None)
        self.status.preview_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def write_status(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status.worker_busy = bool(getattr(self, "_worker_busy", False))
        self.status.accepted_keyframe_count = self.accepted_keyframe_count
        self.status.rejected_keyframe_count = self.rejected_keyframe_count
        self.status.candidate_frame_count = self.candidate_frame_count
        self.status.reconstruction_skipped_candidates = self.reconstruction_skipped_candidates
        self.status.reconstruction_queue_size = int(bool(getattr(self, "_worker_busy", False)))
        self.status.output_session_dir = str(_session_dir(self.reconstruction_session_id)) if self.reconstruction_session_id else None
        if self.status.reference_timestamp is not None:
            self.status.reconstruction_age_sec = round(max(0.0, time.time() - float(self.status.reference_timestamp)), 2)
        status_payload = asdict(self.status)
        write_json_atomic(MULTIVIEW_STATUS_PATH, status_payload, indent=None)
        write_json_atomic(MULTIVIEW_LATEST_STATUS_PATH, status_payload, indent=None)
        coverage = _coverage_from_trajectory(self.accumulated_camera_trajectory)
        write_json_atomic(MULTIVIEW_COVERAGE_PATH, {"session_id": self.reconstruction_session_id, **coverage}, indent=2)
        write_json_atomic(MULTIVIEW_CONTACT_SUPPORT_PATH, {"session_id": self.reconstruction_session_id, **self.last_contact_support}, indent=2)
        report = {
            **asdict(self.status),
            "output_paths": multiview_output_paths(),
            "thresholds": multiview_thresholds(),
            "limitations": [
                "Robot hand-eye and ROS TF pose providers are interface-ready but not wired in this demo path.",
                "RGB-D odometry is not used directly; optional Open3D ICP refines scene alignment when available.",
                "TSDF mesh extraction is disabled by default; mesh export saves a point-cloud placeholder unless TSDF is enabled later.",
            ],
        }
        write_json_atomic(MULTIVIEW_REPORT_PATH, report, indent=2)
        if self.reconstruction_session_id:
            write_json_atomic(self._session_path("reconstruction_report.json"), report, indent=2)
            write_json_atomic(self._session_path("latest_status.json"), status_payload, indent=None)

    def _write_session_files(self, extra: dict[str, Any] | None = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        session = {
            "reconstruction_session_id": self.reconstruction_session_id,
            "reconstruction_target_id": self.reconstruction_target_id,
            "reconstruction_target_id_scope": self.reconstruction_target_id_scope,
            "reference_frame_id": self.reference_frame_id,
            "reference_timestamp": self.reference_timestamp,
            "scan_status": self.status.scan_status,
            "created_at": time.time(),
            "thresholds": multiview_thresholds(),
            **(extra or {}),
        }
        write_json_atomic(MULTIVIEW_SESSION_PATH, session, indent=2)
        trajectory_payload = {
            "reconstruction_session_id": self.reconstruction_session_id,
            "trajectory": self.accumulated_camera_trajectory,
        }
        write_json_atomic(MULTIVIEW_TRAJECTORY_PATH, trajectory_payload, indent=2)
        if self.reconstruction_session_id:
            session_dir = _session_dir(self.reconstruction_session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(session_dir / "session_manifest.json", session, indent=2)
            write_json_atomic(session_dir / "camera_trajectory.json", trajectory_payload, indent=2)


def multiview_thresholds() -> dict[str, Any]:
    return {
        "MULTIVIEW_MIN_REGISTRATION_FITNESS": MULTIVIEW_MIN_REGISTRATION_FITNESS,
        "MULTIVIEW_MAX_INLIER_RMSE_M": MULTIVIEW_MAX_INLIER_RMSE_M,
        "MULTIVIEW_MAX_TRANSLATION_STEP_M": MULTIVIEW_MAX_TRANSLATION_STEP_M,
        "MULTIVIEW_MAX_ROTATION_STEP_DEG": MULTIVIEW_MAX_ROTATION_STEP_DEG,
        "MULTIVIEW_MIN_OVERLAP_RATIO": MULTIVIEW_MIN_OVERLAP_RATIO,
        "MULTIVIEW_MIN_KEYFRAME_TRANSLATION_MM": MULTIVIEW_MIN_KEYFRAME_TRANSLATION_MM,
        "MULTIVIEW_MIN_KEYFRAME_ROTATION_DEG": MULTIVIEW_MIN_KEYFRAME_ROTATION_DEG,
        "MULTIVIEW_MAX_KEYFRAME_INTERVAL_SEC": MULTIVIEW_MAX_KEYFRAME_INTERVAL_SEC,
        "MULTIVIEW_MIN_BOTTLE_DEPTH_VALID_RATIO": MULTIVIEW_MIN_BOTTLE_DEPTH_VALID_RATIO,
        "MULTIVIEW_FUSION_MODE": MULTIVIEW_FUSION_MODE,
        "MULTIVIEW_VOXEL_SIZE_M": MULTIVIEW_VOXEL_SIZE_M,
        "MULTIVIEW_TSDF_ENABLED": MULTIVIEW_TSDF_ENABLED,
        "MULTIVIEW_TSDF_SDF_TRUNC_M": MULTIVIEW_TSDF_SDF_TRUNC_M,
        "MULTIVIEW_MAX_ACCUMULATED_POINTS": MULTIVIEW_MAX_ACCUMULATED_POINTS,
        "MULTIVIEW_MIN_BOTTLE_POINTS": MULTIVIEW_MIN_BOTTLE_POINTS,
        "MULTIVIEW_MIN_SCENE_POINTS": MULTIVIEW_MIN_SCENE_POINTS,
        "MULTIVIEW_SCENE_VOXEL_SIZE_M": MULTIVIEW_SCENE_VOXEL_SIZE_M,
        "MULTIVIEW_BOTTLE_VOXEL_SIZE_M": MULTIVIEW_BOTTLE_VOXEL_SIZE_M,
        "MULTIVIEW_MAX_SCENE_VOXELS": MULTIVIEW_MAX_SCENE_VOXELS,
        "MULTIVIEW_MAX_QUEUE_SIZE": MULTIVIEW_MAX_QUEUE_SIZE,
        "MULTIVIEW_MAX_POINTCLOUD_AGE_SEC": MULTIVIEW_MAX_POINTCLOUD_AGE_SEC,
        "MULTIVIEW_ALLOW_BBOX_ROI": MULTIVIEW_ALLOW_BBOX_ROI,
        "MULTIVIEW_ENABLE_CONTACT_CANDIDATES": MULTIVIEW_ENABLE_CONTACT_CANDIDATES,
        "MULTIVIEW_DAVIS_GATE_ENABLED": MULTIVIEW_DAVIS_GATE_ENABLED,
    }


def multiview_output_paths() -> dict[str, str]:
    return {
        "current_session": str(MULTIVIEW_SESSION_PATH),
        "camera_trajectory": str(MULTIVIEW_TRAJECTORY_PATH),
        "accepted_keyframes": str(MULTIVIEW_ACCEPTED_PATH),
        "rejected_keyframes": str(MULTIVIEW_REJECTED_PATH),
        "accumulated_scene_ply": str(MULTIVIEW_SCENE_PLY_PATH),
        "accumulated_bottle_ply": str(MULTIVIEW_BOTTLE_PLY_PATH),
        "reconstructed_bottle_mesh_ply": str(MULTIVIEW_MESH_PLY_PATH),
        "reconstruction_status": str(MULTIVIEW_STATUS_PATH),
        "reconstruction_preview": str(MULTIVIEW_PREVIEW_PATH),
        "latest_status": str(MULTIVIEW_LATEST_STATUS_PATH),
        "latest_preview": str(MULTIVIEW_LATEST_PREVIEW_PATH),
        "reconstruction_report": str(MULTIVIEW_REPORT_PATH),
        "coverage": str(MULTIVIEW_COVERAGE_PATH),
        "contact_support": str(MULTIVIEW_CONTACT_SUPPORT_PATH),
        "sessions_dir": str(MULTIVIEW_SESSIONS_DIR),
    }


def write_pointcloud_lifecycle_audit() -> None:
    text = """Current RealSense point-cloud lifecycle audit

Current point-cloud source:
- src/grasp_project/realsense/pointcloud.py:update_pointcloud_from_frame()
- Input is the latest aligned RealSense RGB-D frame plus camera intrinsics and the current RealSense report.

Coordinate frame:
- Single-frame point clouds are in realsense_camera_m / current camera coordinates.
- Browser JSON reports positions in meters in the current RealSense camera frame.

Previous-cloud retention:
- The existing single-frame path does not retain previous views.
- output/latest_pointcloud_live.json is overwritten by save_pointcloud_live_json().
- output/latest_realsense_pointcloud.ply and output/latest_bottle_pointcloud.ply are overwritten on refresh/lock.
- PointCloudRuntimeState.last_status stores only the latest status snapshot.

Bottle crop source:
- The bottle cloud is generated in pointcloud.py from cleaned bottle mask when available, then locked/current mask, then bbox/depth fallback.
- It is still a latest-frame crop, not an accumulated model.

Web viewer:
- src/grasp_project/dashboard/web_dashboard.py:/api/pointcloud-live reads output/latest_pointcloud_live.json.
- The Three.js viewer replaces scene/bottle layers when the JSON timestamp changes.

Camera pose / registration:
- No persistent camera trajectory or point-cloud registration existed in the single-frame path.
- The new multiview subsystem estimates a transform into reconstruction_world and rejects unreliable frames.

Latest-frame overwrite points:
- latest_pointcloud_live.json
- latest_pointcloud_preview.jpg
- latest_bottle_pointcloud_preview.jpg
- latest_pointcloud_context.jpg
- latest_realsense_pointcloud.ply
- latest_bottle_pointcloud.ply
"""
    write_text_atomic(MULTIVIEW_AUDIT_PATH, text)


_GLOBAL_RECONSTRUCTOR: MultiViewBottleReconstructor | None = None
_GLOBAL_LOCK = threading.RLock()


def get_global_reconstructor() -> MultiViewBottleReconstructor:
    global _GLOBAL_RECONSTRUCTOR
    with _GLOBAL_LOCK:
        if _GLOBAL_RECONSTRUCTOR is None:
            write_pointcloud_lifecycle_audit()
            _GLOBAL_RECONSTRUCTOR = MultiViewBottleReconstructor()
        return _GLOBAL_RECONSTRUCTOR


def multiview_status_from_file() -> dict[str, Any]:
    status = read_json(MULTIVIEW_STATUS_PATH, {})
    if not status:
        status = asdict(get_global_reconstructor().status)
    status["output_paths"] = multiview_output_paths()
    status["thresholds"] = multiview_thresholds()
    return status


def handle_multiview_command(command: str, payload: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, Any]]:
    recon = get_global_reconstructor()
    payload = payload or {}
    if command == "start_multiview_scan":
        realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
        locked_report = read_json(OUTPUT_DIR / "locked_target.json", {})
        target_context = multiview_target_context(
            {
                **locked_report,
                "target_status": realtime.get("target_status") or locked_report.get("target_status"),
            }
        )
        status = recon.start_scan(
            locked_valid=bool(target_context.get("valid")),
            target_id=target_context.get("target_id"),
            target_id_scope=target_context.get("target_id_scope"),
            source=str(payload.get("source") or "backend_command"),
        )
        ok = status.scan_status in {SCAN_INITIALIZING, SCAN_SCANNING}
        message = "Multi-view scan started." if ok else "Multi-view scan requires a valid locked/tracked bottle."
        return ok, message, asdict(status)
    if command == "pause_multiview_scan":
        return True, "Multi-view scan paused.", asdict(recon.pause_scan())
    if command == "resume_multiview_scan":
        return True, "Multi-view scan resumed.", asdict(recon.resume_scan())
    if command == "finish_multiview_scan":
        return True, "Multi-view scan finished.", asdict(recon.finish_scan())
    if command == "reset_multiview_scan":
        return True, "Multi-view reconstruction reset.", asdict(recon.reset())
    if command in {"save_multiview_ply", "save_accumulated_bottle_ply", "save_scene_map_ply"}:
        return True, "Accumulated multi-view PLY saved.", recon.save_accumulated_ply()
    if command == "save_multiview_report":
        recon.write_status()
        return True, "Multi-view report saved.", {"report_path": str(MULTIVIEW_REPORT_PATH), "session_report_path": None if not recon.reconstruction_session_id else str(recon._session_path("reconstruction_report.json"))}
    if command == "save_multiview_mesh":
        return True, "Bottle mesh export requested.", recon.save_mesh()
    if command == "rebuild_multiview_preview":
        return True, "Multi-view preview rebuilt.", recon.rebuild_preview()
    return False, f"Unknown multi-view command: {command}", {}


def submit_multiview_candidate_async(
    *,
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None,
    bottle_points: np.ndarray,
    bottle_colors: np.ndarray | None,
    report: dict[str, Any] | None,
    timestamp: float | None,
    frame_id: int | str | None,
) -> dict[str, Any]:
    """Submit a keyframe candidate without blocking the live RGB-D loop."""
    if not MULTIVIEW_ENABLED:
        return {"submitted": False, "reason": "multiview_disabled"}
    recon = get_global_reconstructor()
    if recon.status.scan_status not in {SCAN_INITIALIZING, SCAN_SCANNING, SCAN_REGISTRATION_WARNING}:
        return {"submitted": False, "reason": "scan_not_active", "status": asdict(recon.status)}
    gate = multiview_input_gate(report, scene_points=scene_points, bottle_points=bottle_points, timestamp=timestamp)
    recon._apply_input_gate_status(gate)
    if not gate["multiview_input_valid"]:
        recon.candidate_frame_count += 1
        recon.status.candidate_frame_count = recon.candidate_frame_count
        recon.rejected_keyframe_count += 1
        recon.status.rejected_keyframe_count = recon.rejected_keyframe_count
        recon.status.last_rejection_reason = gate["multiview_input_reject_reason"]
        recon._append_rejected({
            "session_id": recon.reconstruction_session_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "candidate_frame_count": recon.candidate_frame_count,
            "reason": gate["multiview_input_reject_reason"],
            **gate,
        })
        recon.write_status()
        return {"submitted": False, "reason": gate["multiview_input_reject_reason"], "status": asdict(recon.status), "input_gate": gate}
    if getattr(recon, "_worker_busy", False):
        recon.status.worker_busy = True
        recon.reconstruction_skipped_candidates += 1
        recon.status.reconstruction_skipped_candidates = recon.reconstruction_skipped_candidates
        recon.status.reconstruction_queue_size = min(MULTIVIEW_MAX_QUEUE_SIZE, 1)
        recon.write_status()
        return {"submitted": False, "reason": "worker_busy", "status": asdict(recon.status)}

    def _worker():
        recon._worker_busy = True
        try:
            recon.process_keyframe(
                scene_points=scene_points.copy(),
                scene_colors=None if scene_colors is None else scene_colors.copy(),
                bottle_points=bottle_points.copy(),
                bottle_colors=None if bottle_colors is None else bottle_colors.copy(),
                report=dict(report or {}),
                timestamp=timestamp,
                frame_id=frame_id,
            )
        finally:
            recon._worker_busy = False
            recon.write_status()

    thread = threading.Thread(target=_worker, name="multiview-reconstruction-worker", daemon=True)
    thread.start()
    return {"submitted": True, "reason": "worker_started", "status": asdict(recon.status)}


__all__ = [
    "CameraPoseEstimator",
    "MultiViewBottleReconstructor",
    "PointCloudFusionResult",
    "ReconstructionStatus",
    "handle_multiview_command",
    "multiview_output_paths",
    "multiview_input_gate",
    "multiview_status_from_file",
    "submit_multiview_candidate_async",
    "transform_points",
    "write_pointcloud_lifecycle_audit",
]

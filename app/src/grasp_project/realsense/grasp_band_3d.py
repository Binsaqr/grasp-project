"""Bottle-local 3D geometry for an immutable RGB-D grasp-band lock.

This module contains geometry only: it does not select a grasp candidate and it
does not provide an image-space fallback.  All XYZ values exposed here use
millimetres in the RealSense optical frame (+x right, +y down, +z forward).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from grasp_project.realsense.pointcloud import masked_depth_to_xyz


MIN_BOTTLE_FRAME_POINTS = 64
MIN_BOTTLE_FRAME_CONFIDENCE = 0.45
DEPTH_SURFACE_MIN_TOLERANCE_MM = 12.0
DEPTH_SURFACE_MAX_TOLERANCE_MM = 60.0
CONTACT_LOCAL_RADIUS_PX = 3.5
CONTACT_RECOVERY_RADIUS_PX = 14.0


def polygon_mask(image_shape: tuple[int, ...], polygon: Any) -> np.ndarray:
    """Rasterize a polygon without altering the source image/mask."""
    height, width = int(image_shape[0]), int(image_shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return result
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return result
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    cv2.fillPoly(result, [np.round(points).astype(np.int32)], 255)
    return result


def deproject_mask_mm(
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    mask: np.ndarray,
    *,
    expected_depth_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return same-surface masked points in millimetres and source pixels.

    ``masked_depth_to_xyz`` performs the aligned pinhole deprojection.  This
    second, bottle-specific stage removes background/edge leakage before PCA.
    The robust median and MAD are deliberately computed only inside the mask;
    no depth is invented for missing pixels.
    """
    points_m, pixels, metadata = masked_depth_to_xyz(depth, intrinsics, mask)
    points_mm = points_m.astype(np.float64) * 1000.0
    pixels = pixels.astype(np.float64)
    raw_count = int(len(points_mm))
    metadata = dict(metadata or {})
    metadata["raw_valid_point_count"] = raw_count
    if raw_count == 0:
        metadata.update({
            "surface_depth_median_mm": None,
            "surface_depth_mad_mm": None,
            "surface_depth_tolerance_mm": None,
            "rejected_depth_outlier_count": 0,
            "depth_filter_status": "NO_VALID_DEPTH",
        })
        return points_mm, pixels, metadata

    z_mm = points_mm[:, 2]
    center_mm = float(np.median(z_mm)) if expected_depth_mm is None else float(expected_depth_mm)
    if not np.isfinite(center_mm) or center_mm <= 0.0:
        center_mm = float(np.median(z_mm))
    absolute_deviation = np.abs(z_mm - center_mm)
    mad_mm = float(np.median(absolute_deviation))
    robust_sigma_mm = 1.4826 * mad_mm
    tolerance_mm = float(np.clip(
        max(DEPTH_SURFACE_MIN_TOLERANCE_MM, 4.0 * robust_sigma_mm),
        DEPTH_SURFACE_MIN_TOLERANCE_MM,
        DEPTH_SURFACE_MAX_TOLERANCE_MM,
    ))
    same_surface = absolute_deviation <= tolerance_mm
    # If an external expected depth was stale, fail back to the current masked
    # surface only when it has clear majority support.  This is still current
    # measured depth, never a temporal/fabricated XYZ value.
    if int(np.count_nonzero(same_surface)) < max(3, int(round(0.05 * raw_count))):
        center_mm = float(np.median(z_mm))
        absolute_deviation = np.abs(z_mm - center_mm)
        mad_mm = float(np.median(absolute_deviation))
        robust_sigma_mm = 1.4826 * mad_mm
        tolerance_mm = float(np.clip(
            max(DEPTH_SURFACE_MIN_TOLERANCE_MM, 4.0 * robust_sigma_mm),
            DEPTH_SURFACE_MIN_TOLERANCE_MM,
            DEPTH_SURFACE_MAX_TOLERANCE_MM,
        ))
        same_surface = absolute_deviation <= tolerance_mm
        expected_depth_status = "EXPECTED_DEPTH_REJECTED_USED_CURRENT_MASK_MEDIAN"
    else:
        expected_depth_status = "EXPECTED_DEPTH_USED" if expected_depth_mm is not None else "CURRENT_MASK_MEDIAN"

    points_mm = points_mm[same_surface]
    pixels = pixels[same_surface]
    mask_count = int(metadata.get("mask_pixel_count", 0) or 0)
    accepted_count = int(len(points_mm))
    metadata.update({
        "valid_point_count": accepted_count,
        "depth_valid_ratio": float(accepted_count / mask_count) if mask_count else 0.0,
        "surface_depth_median_mm": center_mm,
        "surface_depth_mad_mm": mad_mm,
        "surface_depth_tolerance_mm": tolerance_mm,
        "rejected_depth_outlier_count": raw_count - accepted_count,
        "depth_filter_status": "ROBUST_SAME_SURFACE",
        "expected_depth_status": expected_depth_status,
        "reason": "ok" if accepted_count else "no_same_surface_depth",
    })
    return points_mm, pixels, metadata


def robust_masked_contact_xyz_mm(
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    mask: np.ndarray,
    point_px: Any,
    *,
    expected_depth_mm: float | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Recover one contact from exact, local-median, or nearest in-mask depth."""
    try:
        target = np.asarray(point_px, dtype=float).reshape(-1)[:2]
    except (TypeError, ValueError):
        target = np.asarray([], dtype=float)
    if target.size != 2 or not np.all(np.isfinite(target)):
        return None, {"valid": False, "source": "unavailable", "reason": "invalid_contact_pixel"}
    points, pixels, metadata = deproject_mask_mm(
        depth,
        intrinsics,
        mask,
        expected_depth_mm=expected_depth_mm,
    )
    if len(points) == 0:
        return None, {
            "valid": False,
            "source": "unavailable",
            "reason": metadata.get("reason") or "no_valid_in_mask_depth",
            "depth_metadata": metadata,
        }
    distances = np.linalg.norm(pixels - target.reshape(1, 2), axis=1)
    exact_indices = np.flatnonzero(distances <= 0.25)
    if exact_indices.size:
        index = int(exact_indices[0])
        return points[index], {
            "valid": True,
            "source": "exact_pixel",
            "source_pixel": pixels[index].tolist(),
            "distance_px": float(distances[index]),
            "sample_count": 1,
            "depth_quality": 1.0,
            "depth_metadata": metadata,
        }

    local_indices = np.flatnonzero(distances <= CONTACT_LOCAL_RADIUS_PX)
    if local_indices.size >= 3:
        z_mm = float(np.median(points[local_indices, 2]))
        x_mm = (float(target[0]) - float(intrinsics["cx"])) * z_mm / float(intrinsics["fx"])
        y_mm = (float(target[1]) - float(intrinsics["cy"])) * z_mm / float(intrinsics["fy"])
        quality = float(np.clip(local_indices.size / 12.0, 0.35, 1.0))
        return np.asarray([x_mm, y_mm, z_mm], dtype=float), {
            "valid": True,
            "source": "local_in_mask_median",
            "source_pixel": target.tolist(),
            "distance_px": float(np.min(distances[local_indices])),
            "sample_count": int(local_indices.size),
            "depth_quality": quality,
            "depth_metadata": metadata,
        }

    index = int(np.argmin(distances))
    if float(distances[index]) <= CONTACT_RECOVERY_RADIUS_PX:
        quality = float(np.clip(1.0 - distances[index] / CONTACT_RECOVERY_RADIUS_PX, 0.15, 0.8))
        return points[index], {
            "valid": True,
            "source": "nearest_valid_in_mask_pixel",
            "source_pixel": pixels[index].tolist(),
            "distance_px": float(distances[index]),
            "sample_count": 1,
            "depth_quality": quality,
            "depth_metadata": metadata,
        }
    return None, {
        "valid": False,
        "source": "unavailable",
        "reason": "nearest_same_surface_depth_too_far",
        "distance_px": float(distances[index]),
        "depth_metadata": metadata,
    }


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 3 or not np.isfinite(norm) or norm <= 1e-9:
        return None
    return vector / norm


def _invalid_frame(reason: str, point_count: int = 0, depth_valid_ratio: float = 0.0) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "origin_xyz_mm": None,
        "axis_unit": None,
        "side_unit": None,
        "depth_unit": None,
        "bottle_length_mm": None,
        "bottle_width_mm": None,
        "bottle_radius_mm": None,
        "point_count": int(point_count),
        "depth_valid_ratio": float(depth_valid_ratio),
        "confidence": 0.0,
        "pca_eigenvalues_mm2": None,
    }


def fit_bottle_local_frame(
    points_xyz_mm: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    axis_px: Any = None,
    depth_valid_ratio: float = 1.0,
    min_points: int = MIN_BOTTLE_FRAME_POINTS,
) -> dict[str, Any]:
    """Fit an oriented bottle frame from an actual masked RGB-D point cloud.

    PCA estimates the major bottle axis.  The observed image-axis sign removes
    the arbitrary PCA sign; the side direction is the image-plane radial
    direction orthogonalized against that 3D axis.  No 2D location or scale is
    stored as the coordinate frame itself.
    """
    try:
        points = np.asarray(points_xyz_mm, dtype=float).reshape(-1, 3)
    except (TypeError, ValueError):
        return _invalid_frame("invalid_points")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < int(min_points):
        return _invalid_frame("too_few_3d_points", len(points), depth_valid_ratio)
    lo = np.percentile(points, 2.0, axis=0)
    hi = np.percentile(points, 98.0, axis=0)
    trimmed = points[np.all((points >= lo) & (points <= hi), axis=1)]
    if len(trimmed) < int(min_points):
        trimmed = points
    origin = np.median(trimmed, axis=0)
    centered = trimmed - origin
    try:
        _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return _invalid_frame("pca_failed", len(points), depth_valid_ratio)
    if vh.shape != (3, 3) or singular.size < 3:
        return _invalid_frame("pca_failed", len(points), depth_valid_ratio)
    axis = _unit(vh[0])
    if axis is None:
        return _invalid_frame("degenerate_axis", len(points), depth_valid_ratio)

    try:
        image_axis = np.asarray(axis_px, dtype=float).reshape(-1)[:2]
    except (TypeError, ValueError):
        image_axis = np.asarray([], dtype=float)
    if image_axis.size != 2 or not np.all(np.isfinite(image_axis)) or np.linalg.norm(image_axis) <= 1e-9:
        return _invalid_frame("missing_axis_orientation", len(points), depth_valid_ratio)
    image_axis = image_axis / np.linalg.norm(image_axis)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    axis_hint = _unit(np.asarray([image_axis[0] / fx, image_axis[1] / fy, 0.0]))
    if axis_hint is None:
        return _invalid_frame("invalid_axis_orientation", len(points), depth_valid_ratio)
    if float(np.dot(axis, axis_hint)) < 0.0:
        axis = -axis

    image_side = np.asarray([image_axis[1], -image_axis[0]], dtype=float)
    side_hint = _unit(np.asarray([image_side[0] / fx, image_side[1] / fy, 0.0]))
    if side_hint is None:
        return _invalid_frame("invalid_side_orientation", len(points), depth_valid_ratio)
    side = _unit(side_hint - axis * float(np.dot(side_hint, axis)))
    if side is None:
        # The second PCA vector is a measured fallback for an edge-on view. It
        # still comes from the point cloud, never from a bbox scale.
        side = _unit(vh[1] - axis * float(np.dot(vh[1], axis)))
    if side is None:
        return _invalid_frame("degenerate_side_axis", len(points), depth_valid_ratio)
    if float(np.dot(side, side_hint)) < 0.0:
        side = -side
    depth_unit = _unit(np.cross(side, axis))
    if depth_unit is None:
        return _invalid_frame("degenerate_depth_axis", len(points), depth_valid_ratio)
    if depth_unit[2] < 0.0:
        depth_unit = -depth_unit

    local = camera_to_local(points, {
        "origin_xyz_mm": origin,
        "side_unit": side,
        "axis_unit": axis,
        "depth_unit": depth_unit,
    })
    extents = np.percentile(local, 98.0, axis=0) - np.percentile(local, 2.0, axis=0)
    length_mm = float(max(extents[1], 0.0))
    width_mm = float(max(extents[0], 0.0))
    eigenvalues = (singular ** 2) / max(len(trimmed) - 1, 1)
    axis_separation = float(np.clip(1.0 - eigenvalues[1] / max(eigenvalues[0], 1e-9), 0.0, 1.0))
    point_support = float(np.clip(len(points) / max(4.0 * min_points, 1.0), 0.0, 1.0))
    geometry_support = float(np.clip(length_mm / max(1.35 * max(width_mm, 1e-6), 1e-6), 0.0, 1.0))
    confidence = float(np.clip(
        0.40 * float(np.clip(depth_valid_ratio, 0.0, 1.0))
        + 0.25 * point_support
        + 0.20 * axis_separation
        + 0.15 * geometry_support,
        0.0,
        1.0,
    ))
    valid = bool(
        confidence >= MIN_BOTTLE_FRAME_CONFIDENCE
        and length_mm > 1.0
        and width_mm > 1.0
        and np.all(np.isfinite(extents))
    )
    return {
        "valid": valid,
        "reason": "ok" if valid else "low_frame_confidence",
        "origin_xyz_mm": origin.tolist(),
        "axis_unit": axis.tolist(),
        "side_unit": side.tolist(),
        "depth_unit": depth_unit.tolist(),
        "bottle_length_mm": length_mm,
        "bottle_width_mm": width_mm,
        "bottle_radius_mm": width_mm * 0.5,
        "point_count": int(len(points)),
        "depth_valid_ratio": float(depth_valid_ratio),
        "confidence": confidence,
        "pca_eigenvalues_mm2": eigenvalues.tolist(),
    }


def camera_to_local(points_xyz_mm: Any, frame: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points_xyz_mm, dtype=float).reshape(-1, 3)
    origin = np.asarray(frame["origin_xyz_mm"], dtype=float).reshape(3)
    basis = np.column_stack((
        np.asarray(frame["side_unit"], dtype=float),
        np.asarray(frame["axis_unit"], dtype=float),
        np.asarray(frame["depth_unit"], dtype=float),
    ))
    return (points - origin) @ basis


def local_to_camera(points_local_xyz_mm: Any, frame: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points_local_xyz_mm, dtype=float).reshape(-1, 3)
    origin = np.asarray(frame["origin_xyz_mm"], dtype=float).reshape(3)
    basis = np.column_stack((
        np.asarray(frame["side_unit"], dtype=float),
        np.asarray(frame["axis_unit"], dtype=float),
        np.asarray(frame["depth_unit"], dtype=float),
    ))
    return origin + points @ basis.T


def local_boundary(points_local_xyz_mm: Any) -> np.ndarray:
    """Return measured 3D points on the side/axis convex-hull boundary."""
    points = np.asarray(points_local_xyz_mm, dtype=float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return np.empty((0, 3), dtype=float)
    hull_indices = cv2.convexHull(points[:, :2].astype(np.float32), returnPoints=False)
    if hull_indices is None:
        return np.empty((0, 3), dtype=float)
    return points[np.asarray(hull_indices).reshape(-1)]


def project_camera_xyz_mm(points_xyz_mm: Any, intrinsics: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points_xyz_mm, dtype=float).reshape(-1, 3)
    if len(points) == 0 or not np.all(np.isfinite(points)) or np.any(points[:, 2] <= 1e-6):
        return np.empty((0, 2), dtype=float)
    x = float(intrinsics["fx"]) * points[:, 0] / points[:, 2] + float(intrinsics["cx"])
    y = float(intrinsics["fy"]) * points[:, 1] / points[:, 2] + float(intrinsics["cy"])
    return np.column_stack((x, y))

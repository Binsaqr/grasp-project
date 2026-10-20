"""Bottle orientation and pickup-availability estimation from existing evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from bottle_pose import pca_axis_from_mask
from depth_utils import valid_depth_mask


ORIENTATIONS = {"STANDING", "LYING", "TILTED", "UNKNOWN"}


@dataclass(frozen=True)
class ObjectState:
    orientation: str
    tilt_angle_deg: Optional[float]
    standing_confidence: float
    occluded: bool
    available_for_pickup: bool
    reasons: list[str]
    evidence: dict[str, Any]
    pose_class: str = "UNKNOWN"
    pose_confidence: float = 0.0
    major_axis_px: Optional[list[float]] = None
    minor_axis_px: Optional[list[float]] = None
    pose_reason: str = "pose_evidence_unavailable"
    pose_valid_for_grasp: bool = False
    recommended_strategy: str = "NEED_BETTER_VIEW"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_object_state(
    mask,
    depth=None,
    *,
    pose: Optional[Mapping[str, Any]] = None,
    pointcloud: Optional[Mapping[str, Any]] = None,
    standing_max_tilt_deg: float = 25.0,
    lying_min_tilt_deg: float = 65.0,
    occluder_depth_margin_mm: float = 40.0,
    image_roll_deg: float = 0.0,
    image_roll_compensation_status: Optional[str] = None,
    pointcloud_veto_enabled: bool = True,
) -> ObjectState:
    """Estimate state without opening a camera or asserting world calibration.

    ``image_roll_deg`` maps the hand-mount working-up direction into raw image
    pixels.  It corrects the 2-D orientation/clearance reference only; it does
    not alter RGB-D samples or claim a gravity calibration.
    """
    reasons: list[str] = []
    pose = pose or {}
    pointcloud = pointcloud or {}
    depth_arr = None if depth is None else np.asarray(depth, dtype=float)
    mask_arr = None if mask is None else (np.asarray(mask) > 0)
    mask_source = "provided_mask"
    if (mask_arr is None or mask_arr.ndim != 2 or np.count_nonzero(mask_arr) < 5) and depth_arr is not None:
        reconstructed = _mask_from_pose_contour(pose, depth_arr.shape)
        if reconstructed is not None:
            mask_arr = reconstructed
            mask_source = "reconstructed_from_current_pose_contour"
    valid_mask = bool(mask_arr is not None and mask_arr.ndim == 2 and np.count_nonzero(mask_arr) >= 5)

    raw_axis = _axis_from_pose_or_mask(pose, mask_arr if valid_mask else None)
    roll_deg = _finite_roll_deg(image_roll_deg)
    axis = _mount_compensated_axis(raw_axis, roll_deg)
    tilt = None if axis is None else float(np.degrees(np.arccos(np.clip(abs(float(axis[1])), 0.0, 1.0))))
    orientation, standing_confidence = _orientation_from_tilt(tilt, standing_max_tilt_deg, lying_min_tilt_deg)
    pose_class, pose_confidence, pose_reason, recommended_strategy = _pose_classification(
        orientation,
        tilt,
        pose,
        mask_arr if valid_mask else None,
    )
    if pose_class in {"TOP_DOWN_CAP", "TOP_DOWN_BODY"}:
        orientation = "UNKNOWN"
        standing_confidence = 0.0
    pc_axis = _pointcloud_axis(pointcloud)
    pc_tilt = None if pc_axis is None else float(np.degrees(np.arccos(np.clip(abs(float(pc_axis[1])), 0.0, 1.0))))
    extent_evidence = _pointcloud_extent_evidence(pointcloud)
    pc_frame_status = "MEASURED_CAMERA_FRAME_NOT_GRAVITY_CALIBRATED" if pc_axis is not None else "NOT_AVAILABLE"
    if roll_deg == 0.0 and tilt is not None and pc_tilt is not None:
        agreement = max(0.0, 1.0 - abs(tilt - pc_tilt) / 45.0)
        standing_confidence *= 0.7 + 0.3 * agreement
    else:
        agreement = None
    extent_hint = extent_evidence.get("camera_frame_orientation_hint")
    if orientation == "STANDING" and extent_hint == "STANDING":
        standing_confidence = min(1.0, standing_confidence + 0.1)
    elif orientation == "STANDING" and extent_hint == "LYING":
        standing_confidence *= 0.7

    occluded = False
    mask_completeness = 0.0
    depth_valid_ratio = 0.0
    closer_on_mask_ratio = 0.0
    clear_space_closer_ratio = 0.0
    clear_space_valid_ratio = 0.0
    median_depth_mm = None
    pointcloud_blocked = False
    pointcloud_applicable = _pointcloud_evidence_applicable(pointcloud, pose)
    if not valid_mask:
        reasons.append("bottle_mask_unavailable")
    else:
        ys, xs = np.where(mask_arr)
        bbox_area = max((int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1), 1)
        mask_completeness = float(np.count_nonzero(mask_arr) / bbox_area)
        if mask_completeness < 0.35:
            occluded = True
            reasons.append("mask_completeness_low")

    if valid_mask and depth_arr is not None and depth_arr.shape == mask_arr.shape:
        # Use the same bounded RealSense validity definition as the rest of
        # the RGB-D pipeline.  In particular, saturated uint16 values such as
        # 65535 are not measured clear space.
        valid_depth = valid_depth_mask(depth_arr)
        object_depth = depth_arr[mask_arr & valid_depth]
        depth_valid_ratio = float(object_depth.size / max(np.count_nonzero(mask_arr), 1))
        if object_depth.size:
            median_depth_mm = float(np.median(object_depth))
            closer_on_mask_ratio = float(np.mean(object_depth < median_depth_mm - occluder_depth_margin_mm))
            if closer_on_mask_ratio > 0.08:
                occluded = True
                reasons.append("closer_depth_discontinuity_on_bottle")
            clear_space_closer_ratio, clear_space_valid_ratio = _clear_space_depth_evidence(
                mask_arr,
                depth_arr,
                valid_depth,
                median_depth_mm,
                occluder_depth_margin_mm,
                outward_direction=_mount_up_direction_in_raw_image(roll_deg),
            )
            if clear_space_closer_ratio > 0.08:
                occluded = True
                reasons.append("clear_space_above_top_band_blocked")
            if clear_space_valid_ratio < 0.25:
                reasons.append("insufficient_clear_space_depth")
        if depth_valid_ratio < 0.50:
            reasons.append("insufficient_valid_bottle_depth")
    else:
        reasons.append("aligned_depth_unavailable")

    if pointcloud_applicable and pointcloud.get("bottle_crop_safe_for_contacts") is False:
        pointcloud_blocked = bool(pointcloud_veto_enabled)
        reasons.append(
            "pointcloud_crop_not_safe_for_contacts"
            if pointcloud_blocked
            else "pointcloud_crop_advisory_not_used_as_hand_demo_veto"
        )
    if pointcloud_applicable and pointcloud.get("bottle_cloud_valid") is False:
        reasons.append("bottle_pointcloud_invalid")
    available = bool(
        valid_mask
        and depth_arr is not None
        and depth_arr.shape == mask_arr.shape
        and depth_valid_ratio >= 0.50
        and clear_space_valid_ratio >= 0.25
        and (orientation != "UNKNOWN" or pose_class in {"TOP_DOWN_CAP", "TOP_DOWN_BODY"})
        and not occluded
        and not pointcloud_blocked
    )
    if available:
        reasons.append("graspable_region_clear_in_current_rgbd_estimate")
    elif not occluded and orientation == "UNKNOWN":
        reasons.append("orientation_unavailable")

    return ObjectState(
        orientation=orientation,
        tilt_angle_deg=None if tilt is None else round(tilt, 2),
        standing_confidence=round(float(np.clip(standing_confidence, 0.0, 1.0)), 3),
        occluded=occluded,
        available_for_pickup=available,
        reasons=list(dict.fromkeys(reasons)),
        evidence={
            "orientation": {
                "status": "ESTIMATED_FROM_RGB_MASK_PCA" if axis is not None else "NOT_AVAILABLE",
                "mask_source": mask_source if valid_mask else "NOT_AVAILABLE",
                "axis_vector_px": None if axis is None else [round(float(v), 5) for v in axis],
                "raw_image_axis_vector_px": None if raw_axis is None else [round(float(v), 5) for v in raw_axis],
                "camera_frame_tilt_deg": None if tilt is None else round(tilt, 2),
                "orientation_reference_frame": (
                    "HAND_MOUNT_COMPENSATED_IMAGE_FRAME"
                    if roll_deg != 0.0
                    else "RAW_CAMERA_IMAGE_FRAME"
                ),
                "image_roll_compensation_deg": roll_deg,
                "image_roll_compensation_status": (
                    image_roll_compensation_status
                    or ("NOT_CONFIGURED" if roll_deg == 0.0 else "CONFIGURED_NOT_PHYSICALLY_CALIBRATED")
                ),
                "gravity_frame_calibrated": False,
            },
            "pointcloud_orientation": {
                "status": pc_frame_status,
                "principal_axis": None if pc_axis is None else [round(float(v), 5) for v in pc_axis],
                "camera_frame_tilt_deg": None if pc_tilt is None else round(pc_tilt, 2),
                "mask_pointcloud_tilt_agreement": None if agreement is None else round(agreement, 3),
                "extents": pointcloud.get("bottle_3d_extents_m") or pointcloud.get("bottle_pointcloud_extents_mm"),
                **extent_evidence,
            },
            "availability": {
                "status": "ESTIMATED_FROM_ALIGNED_MASK_AND_DEPTH" if valid_mask and depth_arr is not None and depth_arr.shape == mask_arr.shape else "NOT_AVAILABLE",
                "mask_completeness": round(mask_completeness, 4),
                "depth_valid_ratio": round(depth_valid_ratio, 4),
                "median_depth_mm": None if median_depth_mm is None else round(median_depth_mm, 2),
                "closer_on_mask_ratio": round(closer_on_mask_ratio, 4),
                "clear_space_closer_ratio": round(clear_space_closer_ratio, 4),
                "clear_space_valid_ratio": round(clear_space_valid_ratio, 4),
                "pointcloud_blocked": pointcloud_blocked,
                "pointcloud_evidence_applicable": pointcloud_applicable,
                "pointcloud_evidence_role": (
                    "VETO_WHEN_CURRENT"
                    if pointcloud_veto_enabled
                    else "ADVISORY_GEOMETRY_ONLY"
                ),
            },
            "physical_validation": "NOT_TESTED",
        },
        pose_class=pose_class,
        pose_confidence=round(float(np.clip(pose_confidence, 0.0, 1.0)), 3),
        major_axis_px=None if axis is None else [round(float(axis[0]), 5), round(float(axis[1]), 5)],
        minor_axis_px=None if axis is None else [round(float(-axis[1]), 5), round(float(axis[0]), 5)],
        pose_reason=pose_reason,
        pose_valid_for_grasp=bool(available and pose_class not in {"PARTIAL_VIEW", "UNKNOWN"}),
        recommended_strategy=recommended_strategy if available else "CANNOT_GRASP",
    )


def object_state_from_status(realtime: Mapping[str, Any], pointcloud: Optional[Mapping[str, Any]] = None) -> ObjectState:
    """Build an honest diagnostic when dashboard JSON omits pixel arrays."""
    realtime = realtime or {}
    pointcloud = pointcloud or realtime.get("pointcloud_status") or {}
    geometry = realtime.get("bottle_geometry") or {}
    bbox = geometry.get("bbox_xyxy") or realtime.get("raw_bottle_bbox")
    axis = _pointcloud_axis(pointcloud)
    if axis is not None:
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(axis[1])), 0.0, 1.0))))
        orientation, confidence = _orientation_from_tilt(tilt, 25.0, 65.0)
    else:
        tilt, orientation, confidence = None, "UNKNOWN", 0.0
    crop_safe = bool(pointcloud.get("bottle_crop_safe_for_contacts"))
    current = bool(pointcloud.get("pointcloud_current") or pointcloud.get("pointcloud_freshness") == "CURRENT")
    reasons = []
    if not bbox:
        reasons.append("bottle_bbox_unavailable")
    if not crop_safe:
        reasons.append("pointcloud_crop_not_safe_for_contacts")
    if not current:
        reasons.append("pointcloud_not_current")
    reasons.append("pixel_mask_and_depth_not_serialized_for_occlusion_test")
    return ObjectState(
        orientation=orientation,
        tilt_angle_deg=None if tilt is None else round(tilt, 2),
        standing_confidence=round(confidence, 3),
        occluded=False,
        available_for_pickup=False,
        reasons=reasons,
        evidence={
            "orientation": {"status": "ESTIMATED_FROM_POINTCLOUD_CAMERA_FRAME" if axis is not None else "NOT_AVAILABLE", "gravity_frame_calibrated": False},
            "availability": {"status": "NOT_TESTED_MISSING_PIXEL_ARRAYS", "bbox_available": bool(bbox), "pointcloud_current": current, "crop_safe_for_contacts": crop_safe},
            "physical_validation": "NOT_TESTED",
        },
        pose_class=_pose_classification(orientation, tilt, geometry, None)[0],
        pose_confidence=round(confidence, 3),
        major_axis_px=None,
        minor_axis_px=None,
        pose_reason="estimated_from_current_pointcloud_camera_frame_without_pixel_mask",
        pose_valid_for_grasp=False,
        recommended_strategy="NEED_BETTER_VIEW",
    )


def _mask_from_pose_contour(pose, image_shape):
    """Recover the current selective mask from an already-computed contour."""
    if len(image_shape) < 2:
        return None
    contour = pose.get("body_contour") if pose else None
    if contour is None and pose:
        contour = pose.get("contour")
    if contour is None:
        return None
    points = np.asarray(contour, dtype=np.float32)
    if points.size < 6:
        return None
    points = np.rint(points.reshape(-1, 2)).astype(np.int32)
    height, width = int(image_shape[0]), int(image_shape[1])
    points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    recovered = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(recovered, [points], 1)
    return recovered > 0 if np.count_nonzero(recovered) >= 5 else None


def _pose_classification(orientation, tilt, pose, mask):
    metrics = (pose or {}).get("candidate_evidence_scores") or {}
    border_touch = bool(metrics.get("border_touch"))
    if not border_touch and mask is not None and np.count_nonzero(mask):
        ys, xs = np.where(mask)
        height, width = mask.shape
        border_touch = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= width - 2 or ys.max() >= height - 2)
    aspect = metrics.get("oriented_aspect_ratio")
    if aspect is None and mask is not None and np.count_nonzero(mask):
        ys, xs = np.where(mask)
        aspect = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / max(
            min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1),
            1,
        )
    try:
        aspect_value = float(aspect)
    except (TypeError, ValueError):
        aspect_value = None
    top_down = bool((pose or {}).get("top_down_view_candidate")) or (
        aspect_value is not None
        and aspect_value <= 1.35
        and float(metrics.get("solidity_score") or 0.0) >= 0.70
    )
    if border_touch:
        return "PARTIAL_VIEW", 0.35, "candidate_touches_image_boundary", "NEED_BETTER_VIEW"
    if top_down:
        confidence = min(0.9, 0.55 + 0.25 * float(metrics.get("solidity_score") or 0.0))
        cap_like = bool((pose or {}).get("cap_view_candidate"))
        return (
            "TOP_DOWN_CAP" if cap_like else "TOP_DOWN_BODY",
            confidence,
            "near_circular_depth_supported_camera_view",
            "TOP_DOWN_PICK",
        )
    if orientation == "STANDING":
        if tilt is not None and tilt <= 10.0:
            return "UPRIGHT_FRONT", 1.0 - tilt / 25.0, "major_axis_near_camera_vertical", "GRASP_UPRIGHT_BODY"
        return "UPRIGHT_TILTED", 0.75, "major_axis_tilted_from_camera_vertical", "GRASP_UPRIGHT_BODY"
    if orientation == "LYING":
        if tilt is not None and tilt >= 80.0:
            return "SIDEWAYS_HORIZONTAL", min(1.0, (tilt - 65.0) / 25.0), "major_axis_near_camera_horizontal", "STAND_UP_FROM_SIDE"
        return "SIDEWAYS_DIAGONAL", 0.7, "major_axis_diagonal_near_camera_horizontal", "STAND_UP_FROM_SIDE"
    if orientation == "TILTED":
        return "SIDEWAYS_DIAGONAL", 0.65, "major_axis_diagonal_in_camera_frame", "REORIENT_THEN_GRASP"
    return "UNKNOWN", 0.0, "insufficient_pose_evidence", "NEED_BETTER_VIEW"


def _axis_from_pose_or_mask(pose, mask):
    candidate = pose.get("axis_vector") if pose else None
    if candidate is not None:
        axis = np.asarray(candidate, dtype=float).reshape(-1)[:2]
    elif mask is not None:
        axis, _angle, _length = pca_axis_from_mask(mask.astype(np.uint8))
    else:
        return None
    norm = float(np.linalg.norm(axis))
    return None if norm < 1e-9 else axis / norm


def _finite_roll_deg(value):
    try:
        roll = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(roll) if np.isfinite(roll) else 0.0


def _mount_compensated_axis(axis, image_roll_deg):
    if axis is None:
        return None
    radians = np.deg2rad(_finite_roll_deg(image_roll_deg))
    # Raw image = R(roll) * hand-mount image, so undo it for orientation.
    rotation = np.asarray(
        [[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]],
        dtype=float,
    )
    corrected = rotation @ np.asarray(axis, dtype=float).reshape(2)
    norm = float(np.linalg.norm(corrected))
    return None if norm < 1e-9 else corrected / norm


def _mount_up_direction_in_raw_image(image_roll_deg):
    radians = np.deg2rad(_finite_roll_deg(image_roll_deg))
    # Map mount-frame image-up [0, -1] into measured raw-image pixels.
    return np.asarray([np.sin(radians), -np.cos(radians)], dtype=float)


def _pointcloud_axis(pointcloud):
    candidate = pointcloud.get("bottle_principal_axis_m")
    if candidate is None:
        candidate = pointcloud.get("principal_axis")
    if candidate is None:
        return None
    axis = np.asarray(candidate, dtype=float).reshape(-1)
    if axis.size < 2 or np.linalg.norm(axis) < 1e-9:
        return None
    return axis / np.linalg.norm(axis)


def _pointcloud_evidence_applicable(pointcloud, pose):
    if not pointcloud:
        return False
    current = bool(pointcloud.get("pointcloud_current") or pointcloud.get("pointcloud_freshness") == "CURRENT")
    if not current or pointcloud.get("bottle_roi_available") is not True:
        return False
    if pointcloud.get("pointcloud_snapshot_consistent") is False:
        return False
    pose_target = (pose or {}).get("target_id")
    cloud_target = pointcloud.get("selected_bottle_target_id") or pointcloud.get("pointcloud_target_id")
    if pose_target and cloud_target and str(pose_target) != str(cloud_target):
        return False
    return True


def _pointcloud_extent_evidence(pointcloud):
    extents = pointcloud.get("bottle_3d_extents_m") or pointcloud.get("bottle_pointcloud_extents_mm")
    if not isinstance(extents, Mapping):
        return {"extent_status": "NOT_AVAILABLE", "elongation_aspect_ratio": None, "camera_frame_vertical_aspect_ratio": None, "camera_frame_orientation_hint": "UNKNOWN"}
    width = extents.get("width_x")
    height = extents.get("height_y")
    depth = extents.get("depth_z")
    try:
        dimensions = [abs(float(width)), abs(float(height)), abs(float(depth))]
    except (TypeError, ValueError):
        return {"extent_status": "NOT_AVAILABLE", "elongation_aspect_ratio": None, "camera_frame_vertical_aspect_ratio": None, "camera_frame_orientation_hint": "UNKNOWN"}
    if min(dimensions) <= 1e-9:
        return {"extent_status": "NOT_AVAILABLE", "elongation_aspect_ratio": None, "camera_frame_vertical_aspect_ratio": None, "camera_frame_orientation_hint": "UNKNOWN"}
    elongation = max(dimensions) / min(dimensions)
    vertical_aspect = dimensions[1] / max(dimensions[0], dimensions[2])
    hint = "STANDING" if vertical_aspect >= 1.4 else ("LYING" if vertical_aspect <= 0.75 else "TILTED")
    return {
        "extent_status": "MEASURED_CAMERA_FRAME_NOT_GRAVITY_CALIBRATED",
        "elongation_aspect_ratio": round(elongation, 3),
        "camera_frame_vertical_aspect_ratio": round(vertical_aspect, 3),
        "camera_frame_orientation_hint": hint,
    }


def _orientation_from_tilt(tilt, standing_max, lying_min):
    if tilt is None:
        return "UNKNOWN", 0.0
    if tilt <= standing_max:
        return "STANDING", float(np.clip(1.0 - tilt / max(standing_max, 1e-6), 0.0, 1.0))
    if tilt >= lying_min:
        return "LYING", 0.0
    midpoint = (standing_max + lying_min) / 2.0
    confidence = 1.0 - abs(tilt - midpoint) / max((lying_min - standing_max) / 2.0, 1e-6)
    return "TILTED", float(np.clip(confidence, 0.0, 1.0))


def _clear_space_depth_evidence(mask, depth, valid_depth, median_depth, margin, outward_direction=None):
    if outward_direction is not None:
        direction = np.asarray(outward_direction, dtype=float).reshape(-1)[:2]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-9 and np.isfinite(direction).all():
            direction = direction / direction_norm
            ys, xs = np.where(mask)
            points = np.column_stack([xs, ys]).astype(float)
            longitudinal = points @ direction
            transverse_direction = np.asarray([-direction[1], direction[0]], dtype=float)
            transverse = points @ transverse_direction
            longitudinal_span = float(np.ptp(longitudinal))
            transverse_span = float(np.ptp(transverse))
            band_length = max(
                4,
                min(
                    int(round(longitudinal_span * 0.20)),
                    int(round(transverse_span * 0.50)),
                ),
            )
            grid_y, grid_x = np.indices(mask.shape)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(float)
            grid_longitudinal = (grid_points @ direction).reshape(mask.shape)
            grid_transverse = (grid_points @ transverse_direction).reshape(mask.shape)
            outward_edge = float(np.max(longitudinal))
            band = (
                (grid_longitudinal > outward_edge)
                & (grid_longitudinal <= outward_edge + float(band_length))
                & (grid_transverse >= float(np.min(transverse)))
                & (grid_transverse <= float(np.max(transverse)))
            )
            if not np.any(band):
                return 0.0, 0.0
            valid = band & valid_depth
            valid_ratio = float(np.count_nonzero(valid) / max(np.count_nonzero(band), 1))
            if not np.any(valid):
                return 0.0, valid_ratio
            closer_ratio = float(np.mean(depth[valid] < median_depth - margin))
            return closer_ratio, valid_ratio
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # Keep this an actual boundary neighbourhood.  Scaling only with the
    # major-axis height makes the search extend far into unrelated background
    # when a tall bottle is close to the camera.  The transverse extent caps
    # that growth while retaining the original 20% rule for compact/lying
    # objects.
    band_height = max(
        4,
        min(
            int(round((y1 - y0) * 0.20)),
            int(round((x1 - x0) * 0.50)),
        ),
    )
    top = max(0, y0 - band_height)
    if top >= y0:
        return 0.0, 0.0
    clear_valid = valid_depth[top:y0, x0:x1]
    if not np.any(clear_valid):
        return 0.0, 0.0
    clear_depth = depth[top:y0, x0:x1]
    valid_count = np.count_nonzero(clear_valid)
    return (
        float(np.count_nonzero(clear_valid & (clear_depth < median_depth - margin)) / valid_count),
        float(valid_count / max(clear_valid.size, 1)),
    )

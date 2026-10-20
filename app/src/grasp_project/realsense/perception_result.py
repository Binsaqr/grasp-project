"""Canonical RealSense perception result and fail-safe validators.

This module is intentionally camera-free. It does not detect objects itself; it
normalizes the active RealSense backend status into one frame-level contract so
downstream quality, readiness, and dashboard code can tell whether the evidence
is current, bilateral, and honest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FINGER_DETECTED = "DETECTED"
FINGER_NOT_VISIBLE = "NOT_VISIBLE"
FINGER_LOW_CONFIDENCE = "LOW_CONFIDENCE"
FINGER_OCCLUDED = "OCCLUDED"
FINGER_AMBIGUOUS = "AMBIGUOUS"
FINGER_INVALID = "INVALID"

CONTACT_NO_EVIDENCE = "NO_CONTACT_EVIDENCE"
CONTACT_POSSIBLE = "POSSIBLE_CONTACT"
CONTACT_VISUAL_CONFIRMED = "VISUAL_CONTACT_CONFIRMED"
CONTACT_DEPTH_CONFIRMED = "DEPTH_CONTACT_CONFIRMED"
CONTACT_OCCLUDED = "CONTACT_OCCLUDED"
CONTACT_INVALID = "INVALID"

CONTACT_SOURCE_DETECTED_GRIPPER = "detected_gripper_contact"
CONTACT_SOURCE_VISUAL = "visual_contact"
CONTACT_SOURCE_PLANNED = "planned_contact"
CONTACT_SOURCE_UNAVAILABLE = "unavailable"


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _point(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("point") or value.get("center_px") or value.get("px")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        xyxy = value.get("bbox_xyxy") or value.get("bottle_extent_bbox_xyxy")
        xywh = value.get("bbox_xywh")
        if xyxy is not None:
            value = xyxy
        elif isinstance(xywh, (list, tuple)) and len(xywh) >= 4:
            try:
                x, y, width, height = [float(item) for item in xywh[:4]]
                value = [x, y, x + width, y + height]
            except (TypeError, ValueError):
                return None
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _status_from_finger(side_status: str | None, confidence: float | None, point: list[float] | None) -> str:
    status = str(side_status or "").upper()
    if status in {FINGER_OCCLUDED, "OCCLUDED"}:
        return FINGER_OCCLUDED
    if status in {"AMBIGUOUS", "HAND_OR_FOREIGN_OCCLUSION"}:
        return FINGER_AMBIGUOUS
    if point is None:
        return FINGER_NOT_VISIBLE
    if confidence is not None and confidence < 35.0:
        return FINGER_LOW_CONFIDENCE
    return FINGER_DETECTED


def _snapshot_consistency(report: dict) -> tuple[bool, int | None, list[str]]:
    frame_ids = [
        report.get("frame_id"),
        report.get("rgb_frame_id"),
        report.get("depth_frame_id"),
        report.get("mask_frame_id"),
        report.get("contact_frame_id"),
    ]
    numeric = []
    for value in frame_ids:
        try:
            if value is not None:
                numeric.append(int(value))
        except (TypeError, ValueError):
            pass
    if len(numeric) < 2:
        return True, 0, []
    max_delta = max(numeric) - min(numeric)
    reasons = [] if max_delta <= 1 else [f"stale_frame_bundle_delta_{max_delta}"]
    return max_delta <= 1, max_delta, reasons


@dataclass
class RealSensePerceptionResult:
    """One-frame RealSense perception contract for bottle, fingers, and contacts."""

    perception_frame_id: int | None = None
    rgb_frame_id: int | None = None
    depth_frame_id: int | None = None
    mask_frame_id: int | None = None
    contact_frame_id: int | None = None
    timestamp: str | float | None = None
    snapshot_consistent: bool = True
    max_frame_delta: int | None = None

    bottle_detection_status: str = "NOT_DETECTED"
    bottle_detection_confidence: float | None = None
    bottle_bbox_xyxy: list[float] | None = None
    bottle_center_px: list[float] | None = None
    bottle_axis: list[float] | None = None
    bottle_angle_deg: float | None = None
    bottle_depth_mm: float | None = None
    bottle_depth_valid_ratio: float | None = None
    full_bottle_visible: bool | None = None
    bottle_pose_confidence: float | None = None
    bottle_geometry_confidence: float | None = None
    bottle_rejection_reasons: list[str] = field(default_factory=list)

    target_status: str | None = None
    tracking_source: str | None = None
    tracking_confidence: float | None = None
    stable_detection_frames: int | None = None
    low_confidence_frames: int | None = None
    lost_frames: int | None = None
    target_locked: bool = False
    lock_frame_id: int | None = None
    body_fraction_locked: float | None = None
    relock_required: bool = False

    left_finger_status: str = FINGER_NOT_VISIBLE
    right_finger_status: str = FINGER_NOT_VISIBLE
    left_finger_confidence: float | None = None
    right_finger_confidence: float | None = None
    left_finger_mask_or_polygon: Any = None
    right_finger_mask_or_polygon: Any = None
    left_inner_edge: Any = None
    right_inner_edge: Any = None
    left_tip_or_contact_region: Any = None
    right_tip_or_contact_region: Any = None
    finger_pair_geometry_valid: bool = False
    finger_detection_reasons: list[str] = field(default_factory=list)

    contact_source: str = CONTACT_SOURCE_UNAVAILABLE
    contact_source_confidence: float | None = None
    left_contact_status: str = CONTACT_NO_EVIDENCE
    right_contact_status: str = CONTACT_NO_EVIDENCE
    left_contact_patch: Any = None
    right_contact_patch: Any = None
    contact_pair_valid: bool = False
    contact_depth_consistent: bool | None = None
    contact_geometry_stable: bool | None = None
    contact_validation_reasons: list[str] = field(default_factory=list)
    detected_contact_count: int = 0
    expected_contact_count: int = 2
    contact_source_validation_reason: str | None = None
    contact_source_fallback_reason: str | None = None

    perception_ready: bool = False
    perception_blockers: list[str] = field(default_factory=list)
    recommended_action: str = "SEARCH_TARGET"

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_contact_source(raw_source: Any, contact_pair_valid: bool = False) -> str:
    """Map legacy source labels to the canonical source priority vocabulary."""
    source = str(raw_source or "").strip().lower()
    if source in {"detected_gripper", "detected_gripper_contact"}:
        return CONTACT_SOURCE_DETECTED_GRIPPER if contact_pair_valid else CONTACT_SOURCE_UNAVAILABLE
    if source in {"detected_finger", "detected_finger_contacts"}:
        return CONTACT_SOURCE_VISUAL if contact_pair_valid else CONTACT_SOURCE_UNAVAILABLE
    if source in {"visual_contact", "visual_contact_model"}:
        return CONTACT_SOURCE_VISUAL
    if source in {"planned_contact", "locked_contact_model_or_current_target_clamp"}:
        return CONTACT_SOURCE_PLANNED
    return CONTACT_SOURCE_UNAVAILABLE


def validate_contact_source(
    *,
    raw_source: Any,
    left_point: Any = None,
    right_point: Any = None,
    left_depth_mm: Any = None,
    right_depth_mm: Any = None,
    left_confidence: Any = None,
    right_confidence: Any = None,
    stable_frames: Any = None,
    expected_count: int = 2,
    min_confidence: float = 35.0,
    min_stable_frames: int = 3,
) -> dict:
    """Validate whether a source can honestly claim bilateral detected contact."""
    reasons = []
    lp = _point(left_point)
    rp = _point(right_point)
    detected_count = int(lp is not None) + int(rp is not None)
    left_conf = _number(left_confidence, 0.0) or 0.0
    right_conf = _number(right_confidence, 0.0) or 0.0
    stable = int(_number(stable_frames, 0) or 0)
    left_depth = _number(left_depth_mm)
    right_depth = _number(right_depth_mm)
    if detected_count < expected_count:
        reasons.append(f"detected_contact_count_{detected_count}_below_expected_{expected_count}")
    if lp is None:
        reasons.append("missing_left_contact_point")
    if rp is None:
        reasons.append("missing_right_contact_point")
    if left_depth is None:
        reasons.append("missing_left_contact_depth")
    if right_depth is None:
        reasons.append("missing_right_contact_depth")
    if left_conf < min_confidence:
        reasons.append("left_contact_confidence_below_threshold")
    if right_conf < min_confidence:
        reasons.append("right_contact_confidence_below_threshold")
    if stable < min_stable_frames:
        reasons.append("contact_stability_below_threshold")
    raw = str(raw_source or "").strip().lower()
    wants_detected = raw in {"detected_gripper", "detected_finger", "detected_finger_contacts", "detected_gripper_contact"}
    valid = bool(not reasons and wants_detected)
    return {
        "raw_contact_source": raw_source,
        "contact_source": canonical_contact_source(raw_source, valid),
        "detected_contact_count": detected_count,
        "expected_contact_count": expected_count,
        "contact_pair_valid": valid,
        "reasons": reasons,
        "fallback_reason": None if valid else ";".join(reasons) or "detected_contact_source_unavailable",
        "contact_source_validation_reason": "bilateral_detected_contact_valid" if valid else "detected_contact_not_validated",
    }


def build_realsense_perception_result(report: dict | None) -> RealSensePerceptionResult:
    """Build a canonical perception result from the active backend report."""
    report = report or {}
    lock = report.get("target_lock") or {}
    locked = lock.get("locked_target") or {}
    tracking = lock.get("current_tracked_target") or {}
    geometry = report.get("bottle_geometry") or {}
    finger = report.get("finger_detection") or {}
    left_contact = _point(_first_value(report.get("left_finger_contact_px"), finger.get("left_finger_contact_px")))
    right_contact = _point(_first_value(report.get("right_finger_contact_px"), finger.get("right_finger_contact_px")))
    left_conf = _number(_first_value(report.get("left_finger_confidence"), finger.get("left_finger_confidence")))
    right_conf = _number(_first_value(report.get("right_finger_confidence"), finger.get("right_finger_confidence")))
    finger_pair_valid = bool(left_contact is not None and right_contact is not None and (left_conf or 0.0) >= 35.0 and (right_conf or 0.0) >= 35.0)
    snapshot_ok, max_delta, snapshot_reasons = _snapshot_consistency(report)
    source_check = validate_contact_source(
        raw_source=report.get("contact_source") or report.get("quality_input_source"),
        left_point=_first_value(report.get("left_contact_px"), left_contact),
        right_point=_first_value(report.get("right_contact_px"), right_contact),
        left_depth_mm=_first_value(report.get("left_contact_depth_mm"), report.get("left_finger_depth_mm"), finger.get("left_finger_depth_mm")),
        right_depth_mm=_first_value(report.get("right_contact_depth_mm"), report.get("right_finger_depth_mm"), finger.get("right_finger_depth_mm")),
        left_confidence=_first_value(report.get("contact_confidence_left"), left_conf),
        right_confidence=_first_value(report.get("contact_confidence_right"), right_conf),
        stable_frames=_first_value(report.get("stable_frames"), report.get("finger_stable_frames"), finger.get("finger_stable_frames")),
    )
    blockers = list(snapshot_reasons)
    target_status = report.get("target_status") or lock.get("target_status")
    detection_confidence = _number(report.get("detection_confidence"))
    tracking_confidence = _number(_first_value(report.get("tracking_confidence"), tracking.get("tracking_confidence")))
    bbox = _bbox(_first_value(report.get("bottle_extent_bbox_xyxy"), report.get("raw_bottle_bbox"), geometry))
    if bbox is None:
        blockers.append("missing_bottle_bbox")
    if target_status not in {"VERIFIED", "LOCKED", "TRACKING"} and not report.get("target_locked"):
        blockers.append(f"target_status_{target_status or 'missing'}")
    if not source_check["contact_pair_valid"] and str(report.get("contact_source") or "").startswith("detected"):
        blockers.extend(source_check["reasons"])
    if not snapshot_ok:
        blockers.append("snapshot_inconsistent")
    blockers = list(dict.fromkeys(str(item) for item in blockers if item))
    ready = bool(not blockers and source_check["contact_source"] in {CONTACT_SOURCE_DETECTED_GRIPPER, CONTACT_SOURCE_VISUAL, CONTACT_SOURCE_PLANNED})
    if ready:
        action = "READY_TO_CLOSE" if report.get("ready_to_close") else "HOLD_STEADY"
    elif bbox is None:
        action = "SEARCH_TARGET"
    elif not source_check["contact_pair_valid"]:
        action = "WAIT_FOR_CONTACTS"
    else:
        action = "HOLD_STEADY"
    physically_validated_contact = bool(report.get("physical_contact_validated", False))
    contact_status = (
        CONTACT_DEPTH_CONFIRMED
        if source_check["contact_pair_valid"] and physically_validated_contact
        else CONTACT_POSSIBLE
        if source_check["contact_pair_valid"]
        else CONTACT_NO_EVIDENCE
    )
    return RealSensePerceptionResult(
        perception_frame_id=report.get("frame_id"),
        rgb_frame_id=report.get("rgb_frame_id"),
        depth_frame_id=report.get("depth_frame_id"),
        mask_frame_id=report.get("mask_frame_id"),
        contact_frame_id=report.get("contact_frame_id"),
        timestamp=report.get("timestamp"),
        snapshot_consistent=snapshot_ok,
        max_frame_delta=max_delta,
        bottle_detection_status=target_status or ("CANDIDATE" if bbox else "NOT_DETECTED"),
        bottle_detection_confidence=detection_confidence,
        bottle_bbox_xyxy=bbox,
        bottle_center_px=_point(_first_value(report.get("bottle_center_px"), geometry.get("center_px"))),
        bottle_axis=_first_value(report.get("bottle_y_axis"), report.get("bottle_y_axis_px"), geometry.get("axis_vector")),
        bottle_angle_deg=_number(_first_value(report.get("raw_bottle_angle_deg"), geometry.get("axis_angle_degrees"))),
        bottle_depth_mm=_number(_first_value(report.get("bottle_depth_median_mm"), report.get("depth_mm"), geometry.get("depth_median_mm"))),
        bottle_depth_valid_ratio=_number(_first_value(report.get("bottle_depth_valid_ratio"), geometry.get("depth_confidence"))),
        full_bottle_visible=report.get("full_bottle_visible"),
        bottle_pose_confidence=_number(_first_value(report.get("bottle_pose_confidence"), detection_confidence)),
        bottle_geometry_confidence=_number(_first_value(report.get("bottle_geometry_confidence"), tracking_confidence)),
        bottle_rejection_reasons=_as_list(report.get("bottle_rejection_reasons") or report.get("failure_reasons")),
        target_status=target_status,
        tracking_source=report.get("tracking_source") or tracking.get("tracking_source"),
        tracking_confidence=tracking_confidence,
        stable_detection_frames=report.get("stable_detection_frames"),
        low_confidence_frames=report.get("low_confidence_frames"),
        lost_frames=report.get("lost_frames"),
        target_locked=bool(report.get("target_locked") or lock.get("lock_frame_id")),
        lock_frame_id=lock.get("lock_frame_id"),
        body_fraction_locked=_number(_first_value(lock.get("locked_body_fraction"), locked.get("locked_body_fraction"))),
        relock_required=bool(report.get("relock_required")),
        left_finger_status=_status_from_finger(report.get("left_finger_status"), left_conf, left_contact),
        right_finger_status=_status_from_finger(report.get("right_finger_status"), right_conf, right_contact),
        left_finger_confidence=left_conf,
        right_finger_confidence=right_conf,
        left_finger_mask_or_polygon=report.get("left_finger_contour") or finger.get("left_finger_contour"),
        right_finger_mask_or_polygon=report.get("right_finger_contour") or finger.get("right_finger_contour"),
        left_inner_edge=report.get("left_inner_edge") or finger.get("left_inner_edge"),
        right_inner_edge=report.get("right_inner_edge") or finger.get("right_inner_edge"),
        left_tip_or_contact_region=left_contact,
        right_tip_or_contact_region=right_contact,
        finger_pair_geometry_valid=finger_pair_valid,
        finger_detection_reasons=_as_list(report.get("finger_quality_gate_reasons") or finger.get("finger_detection_reasons")),
        contact_source=source_check["contact_source"],
        contact_source_confidence=min(left_conf or 0.0, right_conf or 0.0) if source_check["contact_pair_valid"] else 0.0,
        left_contact_status=contact_status,
        right_contact_status=contact_status,
        left_contact_patch=report.get("left_contact_patch") or report.get("left_clamp_patch"),
        right_contact_patch=report.get("right_contact_patch") or report.get("right_clamp_patch"),
        contact_pair_valid=source_check["contact_pair_valid"],
        contact_depth_consistent=None if source_check["contact_pair_valid"] else False,
        contact_geometry_stable=source_check["contact_pair_valid"],
        contact_validation_reasons=source_check["reasons"],
        detected_contact_count=source_check["detected_contact_count"],
        expected_contact_count=source_check["expected_contact_count"],
        contact_source_validation_reason=source_check["contact_source_validation_reason"],
        contact_source_fallback_reason=source_check["fallback_reason"],
        perception_ready=ready,
        perception_blockers=blockers,
        recommended_action=action,
    )


__all__ = [
    "CONTACT_SOURCE_DETECTED_GRIPPER",
    "CONTACT_SOURCE_PLANNED",
    "CONTACT_SOURCE_UNAVAILABLE",
    "CONTACT_SOURCE_VISUAL",
    "RealSensePerceptionResult",
    "build_realsense_perception_result",
    "canonical_contact_source",
    "validate_contact_source",
]

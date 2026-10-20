"""Strict bottle-only Benchmark Lab evaluators.

Evaluators in this module only read through ``MaskedFeatureView``. They never
read the full production decision, ready-to-close state, or epsilon output
unless the corresponding feature is enabled and applicable.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .feature_registry import FEATURE_BY_ID, canonical_feature_ids
from .masked_feature_view import (
    DEPENDENCY_MISSING,
    DISABLED,
    NOT_APPLICABLE,
    SENSOR_UNAVAILABLE,
    STALE_DATA,
    UNSUPPORTED,
    VALID,
    FeaturePacket,
    MaskedFeatureView,
)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def _is_valid(result: dict[str, Any]) -> bool:
    return result.get("status") == VALID


def _value(result: dict[str, Any]) -> Any:
    return result.get("returned_value")


def _status_priority(statuses: list[str]) -> str:
    if DEPENDENCY_MISSING in statuses:
        return DEPENDENCY_MISSING
    if SENSOR_UNAVAILABLE in statuses:
        return SENSOR_UNAVAILABLE
    if STALE_DATA in statuses:
        return STALE_DATA
    if UNSUPPORTED in statuses:
        return UNSUPPORTED
    if VALID in statuses:
        return VALID
    if NOT_APPLICABLE in statuses:
        return NOT_APPLICABLE
    return DISABLED


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def evaluate_bottle_detection(masked_view: MaskedFeatureView) -> dict[str, Any]:
    rgb = masked_view.get_rgb_detection()
    detection = rgb["detection"]
    if _is_valid(detection):
        payload = _value(detection) or {}
        return {
            "category": "Bottle Detection",
            "status": VALID,
            "bottle_detected": bool(payload.get("detected")),
            "detection_status": payload.get("status") or "rgb_detection",
            "detection_confidence": _clamp_score(_num(payload.get("confidence"))),
            "bbox_xyxy": payload.get("bbox_xyxy"),
            "center_px": _value(rgb["center"]) if _is_valid(rgb["center"]) else None,
            "angle_deg": _num(_value(rgb["angle"])) if _is_valid(rgb["angle"]) else None,
            "mask_available": bool((_value(rgb["mask"]) or {}).get("mask_available")) if _is_valid(rgb["mask"]) else False,
            "reason": "rgb_detection_enabled",
        }
    if masked_view.is_enabled("depth_bottle_distance") or masked_view.is_enabled("depth_object_geometry"):
        return {
            "category": "Bottle Detection",
            "status": NOT_APPLICABLE,
            "bottle_detected": False,
            "detection_status": "depth_only_identity_not_applicable",
            "detection_confidence": None,
            "bbox_xyxy": None,
            "center_px": None,
            "angle_deg": None,
            "mask_available": False,
            "reason": "Depth-only geometry cannot identify a stationary bottle.",
        }
    if masked_view.is_enabled("pointcloud_bottle_geometry"):
        pc = masked_view.get_pointcloud_geometry()
        status = _status_priority([item["status"] for item in pc.values()])
        return {
            "category": "Bottle Detection",
            "status": DEPENDENCY_MISSING if status == DEPENDENCY_MISSING else NOT_APPLICABLE,
            "bottle_detected": False,
            "detection_status": "pointcloud_geometry_not_independent_identity",
            "detection_confidence": None,
            "bbox_xyxy": None,
            "center_px": None,
            "angle_deg": None,
            "mask_available": False,
            "reason": "Point cloud is geometry support; object identity depends on depth/intrinsics/ROI.",
        }
    if masked_view.is_enabled("event_rate") or masked_view.is_enabled("event_motion_activity"):
        return {
            "category": "Bottle Detection",
            "status": NOT_APPLICABLE,
            "bottle_detected": False,
            "detection_status": "stationary_bottle_not_identifiable_from_events_only",
            "detection_confidence": 0.0,
            "bbox_xyxy": None,
            "center_px": None,
            "angle_deg": None,
            "mask_available": False,
            "reason": "DAVIS events measure motion/stability, not stationary object identity.",
        }
    if masked_view.is_enabled("davis_aps_frame"):
        return {
            "category": "Bottle Detection",
            "status": NOT_APPLICABLE,
            "bottle_detected": False,
            "detection_status": "aps_context_not_realsense_identity",
            "detection_confidence": None,
            "bbox_xyxy": None,
            "center_px": None,
            "angle_deg": None,
            "mask_available": False,
            "reason": "DAVIS APS is visual context unless a DAVIS-specific detector is enabled.",
        }
    return {
        "category": "Bottle Detection",
        "status": DISABLED,
        "bottle_detected": False,
        "detection_status": "no_detection_features_enabled",
        "detection_confidence": None,
        "bbox_xyxy": None,
        "center_px": None,
        "angle_deg": None,
        "mask_available": False,
        "reason": "No bottle identity feature was enabled.",
    }


def evaluate_geometry(masked_view: MaskedFeatureView) -> dict[str, Any]:
    depth = masked_view.get_depth_geometry()
    rgbd = masked_view.get_rgbd_tracking()
    pc = masked_view.get_pointcloud_geometry()
    distance = _value(depth["distance"]) if _is_valid(depth["distance"]) else None
    valid_ratio = _value(depth["valid_ratio"]) if _is_valid(depth["valid_ratio"]) else None
    object_geometry = _value(depth["object_geometry"]) if _is_valid(depth["object_geometry"]) else None
    pointcloud_geometry = _value(pc["bottle"]) if _is_valid(pc["bottle"]) else None
    pointcloud_extents = _value(pc["extents"]) if _is_valid(pc["extents"]) else None
    grip_width = _value(pc["grip_width"]) if _is_valid(pc["grip_width"]) else None
    statuses = [item["status"] for item in [*depth.values(), *rgbd.values(), *pc.values()]]
    valid_sources = [
        bool(distance is not None),
        bool(valid_ratio is not None),
        bool(pointcloud_geometry and pointcloud_geometry.get("valid")),
        bool(pointcloud_extents),
    ]
    confidence = _mean([
        100.0 if distance is not None else None,
        min(100.0, max(0.0, float(valid_ratio) * 100.0)) if valid_ratio is not None and float(valid_ratio) <= 1.0 else _clamp_score(_num(valid_ratio)),
        90.0 if pointcloud_geometry and pointcloud_geometry.get("valid") else None,
    ])
    status = VALID if any(valid_sources) else _status_priority(statuses)
    if status == VALID and (DEPENDENCY_MISSING in statuses or STALE_DATA in statuses or SENSOR_UNAVAILABLE in statuses):
        status = "VALID_WITH_LIMITATIONS"
    return {
        "category": "Bottle Geometry",
        "status": status,
        "distance_available": distance is not None,
        "distance_mm": distance,
        "depth_valid_ratio": valid_ratio,
        "object_dimensions": pointcloud_extents or (object_geometry if isinstance(object_geometry, dict) else None),
        "object_dimensions_available": bool(pointcloud_extents or object_geometry),
        "grip_width_estimate_available": grip_width is not None,
        "grip_width_mm": grip_width,
        "pointcloud_valid": bool(pointcloud_geometry and pointcloud_geometry.get("valid")),
        "pointcloud_roi_source": (pointcloud_geometry or {}).get("roi_source") if isinstance(pointcloud_geometry, dict) else None,
        "geometry_confidence": confidence,
        "reason": "geometry_evaluated_from_enabled_features" if any(valid_sources) else "no_valid_geometry_feature",
    }


def evaluate_motion(masked_view: MaskedFeatureView) -> dict[str, Any]:
    aps = masked_view.get_davis_aps_motion()
    events = masked_view.get_davis_event_motion()
    aps_frame = _value(aps["frame"]) if _is_valid(aps["frame"]) else None
    event_rate = _value(events["rate"]) if _is_valid(events["rate"]) else None
    activity = _value(events["activity"]) if _is_valid(events["activity"]) else {}
    stability = _value(events["stability"]) if _is_valid(events["stability"]) else None
    rotation = _value(events["rotation"]) if _is_valid(events["rotation"]) else None
    vibration = _value(events["vibration"]) if _is_valid(events["vibration"]) else None
    impact = _value(events["impact"]) if _is_valid(events["impact"]) else None
    incipient = _value(events["incipient_slip"]) if _is_valid(events["incipient_slip"]) else None
    gross = _value(events["gross_slip"]) if _is_valid(events["gross_slip"]) else None
    statuses = [item["status"] for item in [*aps.values(), *events.values()]]
    has_aps = aps_frame is not None
    has_events = event_rate is not None or stability is not None
    status = VALID if has_aps or has_events else _status_priority(statuses)
    activity_payload = activity if isinstance(activity, dict) else {}
    displacement = _num(activity_payload.get("displacement_px"))
    rotation_delta = _num(activity_payload.get("rotation_delta_deg"))
    disturbance = activity_payload.get("disturbance_type")
    if displacement is not None:
        reason = f"DAVIS Events measured {displacement:.1f} px motion"
        if rotation_delta is not None:
            reason += f" and {rotation_delta:.1f} deg rotation"
        if disturbance and disturbance not in {"OBJECT_MOTION", "none"}:
            reason += f"; disturbance classified as {disturbance}, not object slip"
        reason += "."
    else:
        reason = "motion_features_evaluated" if has_aps or has_events else "no_motion_feature_enabled_or_available"
    aps_visibility_status = (aps_frame or {}).get("aps_visibility_status") if isinstance(aps_frame, dict) else None
    aps_visibility_score = (aps_frame or {}).get("aps_visibility_score") if isinstance(aps_frame, dict) else None
    aps_visual_detection_valid = None
    aps_limitation = None
    if has_aps and aps_visibility_status is not None:
        aps_visual_detection_valid = str(aps_visibility_status).upper() not in {"LOW_LIGHT", "UNUSABLE", "UNKNOWN"}
        if not aps_visual_detection_valid:
            aps_limitation = (
                f"DAVIS APS visibility is {aps_visibility_status} (score {aps_visibility_score}); "
                "APS is context only under current lighting, not a reliable visual detector."
            )
    return {
        "category": "Motion Sensing",
        "status": status,
        "aps_available": bool(has_aps),
        "aps_mode": (aps_frame or {}).get("aps_state") if isinstance(aps_frame, dict) else None,
        "aps_fps": (aps_frame or {}).get("aps_fps") if isinstance(aps_frame, dict) else None,
        "davis_aps_visibility_score": aps_visibility_score,
        "davis_aps_visibility_status": aps_visibility_status,
        "davis_aps_visual_detection_valid": aps_visual_detection_valid,
        "davis_aps_limitation": aps_limitation,
        "events_available": bool(has_events),
        "event_stream_active": bool(has_events),
        "event_rate": event_rate,
        "event_motion_activity": activity_payload or None,
        "motion_detected": bool(event_rate and event_rate > 0),
        "motion_class": activity_payload.get("motion_class"),
        "davis_dx_px": _num(activity_payload.get("dx_px")),
        "davis_dy_px": _num(activity_payload.get("dy_px")),
        "davis_displacement_px": displacement,
        "davis_cumulative_slip_px": _num(activity_payload.get("cumulative_slip_px")),
        "davis_downward_slip_px": _num(activity_payload.get("downward_slip_px")),
        "davis_rotation_delta_deg": rotation_delta,
        "davis_cumulative_rotation_deg": _num(activity_payload.get("cumulative_rotation_deg")),
        "davis_angular_velocity_deg_s": _num(activity_payload.get("angular_velocity_deg_s")),
        "davis_vibration_rms_px": _num(activity_payload.get("vibration_amplitude_px_rms")),
        "davis_motion_estimate_confidence": _num(activity_payload.get("motion_estimation_confidence")),
        "davis_motion_estimate_valid": activity_payload.get("motion_estimate_valid"),
        "davis_motion_validity": "VALID" if activity_payload.get("motion_estimate_valid") is True else ("INVALID" if activity_payload.get("motion_estimate_valid") is False else "UNKNOWN"),
        "davis_disturbance_type": disturbance,
        "stability_score": stability,
        "rotation_score": rotation,
        "vibration_score": vibration,
        "impact_score": impact,
        "incipient_slip_score": incipient,
        "gross_slip_score": gross,
        "reason": reason,
    }


def evaluate_quality(masked_view: MaskedFeatureView, detection: dict[str, Any], geometry: dict[str, Any], motion: dict[str, Any]) -> dict[str, Any]:
    quality = masked_view.get_quality_metric()
    visual_metric = quality["visual_object_quality"]
    geometry_metric = quality["geometry_quality"]
    visual_score = _clamp_score(_num(_value(visual_metric))) if _is_valid(visual_metric) else None
    geometry_score = _clamp_score(_num(_value(geometry_metric))) if _is_valid(geometry_metric) else None
    stability_score = None
    if visual_score is None and detection.get("status") == VALID and detection.get("bottle_detected"):
        visual_score = _mean([
            _num(detection.get("detection_confidence")),
            90.0 if detection.get("mask_available") else 65.0,
        ])
    if geometry_score is None and geometry.get("status") in {VALID, "VALID_WITH_LIMITATIONS"}:
        geometry_score = _mean([
            _num(geometry.get("geometry_confidence")),
            90.0 if geometry.get("object_dimensions_available") else None,
        ])
    if motion.get("status") == VALID:
        stability_score = _num(motion.get("stability_score"))
        if stability_score is None and motion.get("events_available"):
            stability_score = 80.0
    epsilon = quality["analytic_epsilon"]
    live_quality = quality["live_operational_quality"]
    statuses = [item["status"] for item in quality.values()]
    status = VALID if any(v is not None for v in [visual_score, geometry_score, stability_score]) else _status_priority(statuses)
    overall = _mean([visual_score, geometry_score, stability_score])
    return {
        "category": "Bottle-Only Quality",
        "status": status,
        "visual_quality": visual_score,
        "geometry_quality": geometry_score,
        "stability_quality": stability_score,
        "overall_bottle_only_quality": overall,
        "analytic_epsilon_status": epsilon["status"],
        "analytic_epsilon": None if epsilon["status"] != VALID else epsilon["returned_value"],
        "live_operational_quality": None if live_quality["status"] != VALID else live_quality["returned_value"],
        "quality_result_validity": "NOT_APPLICABLE_WITHOUT_CONTACT_GEOMETRY" if epsilon["status"] == NOT_APPLICABLE else epsilon["status"],
        "not_applicable": {k: v for k, v in masked_view.not_applicable_features.items()},
        "reason": "bottle_only_quality_from_enabled_features" if overall is not None else "quality_not_applicable_or_unavailable",
    }


def evaluate_decision(masked_view: MaskedFeatureView, detection: dict[str, Any], geometry: dict[str, Any], motion: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    missing = bool(masked_view.missing_dependencies)
    unavailable = bool(masked_view.unavailable_features)
    stale = bool(masked_view.stale_features)
    used = bool(masked_view.used_features)
    if missing and not used:
        validity = DEPENDENCY_MISSING
        reason = "enabled feature dependencies are disabled"
    elif not used and unavailable:
        validity = SENSOR_UNAVAILABLE
        reason = "enabled sensor source unavailable"
    elif not used:
        validity = NOT_APPLICABLE
        reason = "no enabled feature can produce a bottle-only metric"
    elif stale:
        validity = "VALID_WITH_LIMITATIONS"
        reason = "one or more enabled features are stale"
    elif missing or unavailable:
        validity = "VALID_WITH_LIMITATIONS"
        reason = "some enabled features were blocked or unavailable"
    else:
        validity = VALID
        reason = "strict masked evaluation completed"

    score = _mean([
        _num(detection.get("detection_confidence")) if detection.get("status") == VALID else None,
        _num(geometry.get("geometry_confidence")) if geometry.get("status") in {VALID, "VALID_WITH_LIMITATIONS"} else None,
        _num(motion.get("stability_score")) if motion.get("status") == VALID else None,
        _num(quality.get("overall_bottle_only_quality")) if quality.get("status") == VALID else None,
    ])
    return {
        "result_validity": validity,
        "result_validity_reason": reason,
        "decision": "BENCHMARK_VALID" if validity in {VALID, "VALID_WITH_LIMITATIONS"} else validity,
        "confidence": score if score is not None else 0.0,
        "production_control": False,
        "controls_robot": False,
    }


def evaluate_configuration_card(configuration: dict[str, Any], feature_packet: FeaturePacket) -> dict[str, Any]:
    started = time.perf_counter()
    feature_ids = canonical_feature_ids(configuration.get("feature_ids") or configuration.get("enabled_features") or [])
    view = MaskedFeatureView(
        feature_packet,
        feature_ids,
        bottle_only=bool(configuration.get("benchmark_object_mode") == "bottle_only"),
    )
    detection = evaluate_bottle_detection(view)
    geometry = evaluate_geometry(view)
    motion = evaluate_motion(view)
    quality = evaluate_quality(view, detection, geometry, motion)
    decision = evaluate_decision(view, detection, geometry, motion, quality)
    proof = view.proof_payload()
    categories = {FEATURE_BY_ID[fid].category for fid in feature_ids if fid in FEATURE_BY_ID}
    sync_info = feature_packet.values.get("synchronization_confidence") if isinstance(feature_packet.values.get("synchronization_confidence"), dict) else {}

    card = {
        "configuration_id": configuration.get("configuration_id") or configuration.get("preset_id") or "custom",
        "display_name": configuration.get("display_name") or configuration.get("configuration_id") or "Custom",
        "preset_id": configuration.get("preset_id"),
        "benchmark_only": True,
        "controls_robot": False,
        "production_isolated": True,
        "same_bundle_id": feature_packet.bundle_id,
        "sync_mode": sync_info.get("sync_mode"),
        "sync_quality": sync_info.get("sync_quality"),
        "sync_offset_ms": sync_info.get("max_sensor_time_offset_ms"),
        "stale_modalities": sync_info.get("stale_modalities") or [],
        "unavailable_modalities": sync_info.get("unavailable_modalities") or [],
        "bundle_valid_for_benchmark": sync_info.get("bundle_valid_for_benchmark"),
        "bundle_valid_for_live_quality": sync_info.get("bundle_valid_for_live_quality"),
        "feature_selection_valid": not bool(view.missing_dependencies),
        "feature_selection_reason": "Feature selection is valid." if not view.missing_dependencies else "Missing dependencies are shown per feature.",
        "enabled_features": feature_ids,
        "enabled_feature_categories": sorted(categories),
        **detection,
        "distance_available": geometry.get("distance_available"),
        "distance_mm": geometry.get("distance_mm"),
        "depth_valid_ratio": geometry.get("depth_valid_ratio"),
        "object_dimensions": geometry.get("object_dimensions"),
        "object_dimensions_available": geometry.get("object_dimensions_available"),
        "grip_width_estimate_available": geometry.get("grip_width_estimate_available"),
        "grip_width_mm": geometry.get("grip_width_mm"),
        "pointcloud_valid": geometry.get("pointcloud_valid"),
        "pointcloud_roi_source": geometry.get("pointcloud_roi_source"),
        "geometry_confidence": geometry.get("geometry_confidence"),
        "aps_available": motion.get("aps_available"),
        "aps_mode": motion.get("aps_mode"),
        "aps_fps": motion.get("aps_fps"),
        "events_available": motion.get("events_available"),
        "event_stream_active": motion.get("event_stream_active"),
        "event_rate": motion.get("event_rate"),
        "motion_class": motion.get("motion_class"),
        "davis_dx_px": motion.get("davis_dx_px"),
        "davis_dy_px": motion.get("davis_dy_px"),
        "davis_displacement_px": motion.get("davis_displacement_px"),
        "davis_cumulative_slip_px": motion.get("davis_cumulative_slip_px"),
        "davis_downward_slip_px": motion.get("davis_downward_slip_px"),
        "davis_rotation_delta_deg": motion.get("davis_rotation_delta_deg"),
        "davis_cumulative_rotation_deg": motion.get("davis_cumulative_rotation_deg"),
        "davis_angular_velocity_deg_s": motion.get("davis_angular_velocity_deg_s"),
        "davis_vibration_rms_px": motion.get("davis_vibration_rms_px"),
        "davis_motion_estimate_confidence": motion.get("davis_motion_estimate_confidence"),
        "davis_motion_estimate_valid": motion.get("davis_motion_estimate_valid"),
        "davis_motion_validity": motion.get("davis_motion_validity"),
        "davis_disturbance_type": motion.get("davis_disturbance_type"),
        # carry the full event-motion activity block (event_health_valid,
        # object_motion_valid, disturbance_detected, davis_motion_validity,
        # exclusion reason) so the runner sample separates event health from
        # object motion instead of recording UNKNOWN
        "event_motion_activity": motion.get("event_motion_activity"),
        "stability_score": motion.get("stability_score"),
        "rotation_score": motion.get("rotation_score"),
        "vibration_score": motion.get("vibration_score"),
        "impact_score": motion.get("impact_score"),
        "incipient_slip_score": motion.get("incipient_slip_score"),
        "gross_slip_score": motion.get("gross_slip_score"),
        "visual_quality": quality.get("visual_quality"),
        "geometry_quality": quality.get("geometry_quality"),
        "stability_quality": quality.get("stability_quality"),
        "overall_bottle_only_quality": quality.get("overall_bottle_only_quality"),
        "epsilon_available": quality.get("analytic_epsilon") is not None,
        "epsilon_value": quality.get("analytic_epsilon"),
        "quality_score": quality.get("overall_bottle_only_quality"),
        "analytic_quality_score": quality.get("analytic_epsilon"),
        "live_quality_score": quality.get("live_operational_quality"),
        "quality_label": quality.get("quality_result_validity"),
        "quality_result_validity": quality.get("quality_result_validity"),
        "contact_validity": None,
        "contact_confidence": None,
        "contact_source": "not_applicable_bottle_only",
        "contact_count": None,
        "ready_to_close": None,
        "ready_to_lift": None,
        "latency_ms": None,
        "latency_source": "masked_evaluator_compute",
        "hard_blockers": [],
        "contributions": [
            {"feature": "Bottle Detection", "effect": detection.get("status"), "message": detection.get("reason")},
            {"feature": "Geometry", "effect": geometry.get("status"), "message": geometry.get("reason")},
            {"feature": "Motion", "effect": motion.get("status"), "message": motion.get("reason")},
            {"feature": "Bottle-Only Quality", "effect": quality.get("status"), "message": quality.get("reason")},
        ],
        "profiling": {},
        **decision,
        **proof,
    }
    card["missing_features"] = list(card["blocked_features"].keys())
    card["main_limitation"] = decision["result_validity_reason"]
    if "analytic_epsilon" in feature_ids and not card["epsilon_available"]:
        card["main_limitation"] = "epsilon unavailable: contact geometry missing, disabled, or not applicable"
    feature_mask_ms = max(0.001, round((time.perf_counter() - started) * 1000.0, 3))
    card["profiling"]["feature_mask_ms"] = feature_mask_ms
    card["latency_ms"] = feature_mask_ms
    return card


def poison_disabled_packet(packet: FeaturePacket, enabled_features: list[str]) -> FeaturePacket:
    """Return a packet with impossible values in disabled features."""
    values = copy.deepcopy(packet.values)
    poison_values = {
        "rgb_bottle_detection": {"detected": True, "status": "POISON_RGB", "confidence": 999.0, "bbox_xyxy": [-1, -1, -1, -1]},
        "depth_bottle_distance": -99999.0,
        "depth_valid_ratio": -123.0,
        "pointcloud_bottle_geometry": {"valid": True, "roi_source": "POISON", "filtered_point_count": 99999999},
        "event_rate": 999999.0,
        "event_gross_slip": 100.0,
        "event_rotation": 100.0,
        "analytic_epsilon": 999.0,
        "live_operational_quality": 999.0,
    }
    enabled = set(enabled_features)
    for feature_id, value in poison_values.items():
        if feature_id not in enabled:
            values[feature_id] = value
    return FeaturePacket(
        values=values,
        source_fields=dict(packet.source_fields),
        availability=dict(packet.availability),
        freshness=dict(packet.freshness),
        production_reference=dict(packet.production_reference),
        timestamp=packet.timestamp,
        bundle_id=packet.bundle_id,
    )


def run_poison_value_tests(output_path: str | Path | None = None) -> dict[str, Any]:
    """Run camera-free poison checks proving disabled features cannot affect results."""
    base = FeaturePacket(
        values={
            "rgb_bottle_detection": {"detected": True, "status": "TRACKING", "confidence": 90.0, "bbox_xyxy": [1, 2, 3, 4]},
            "rgb_bottle_segmentation": {"mask_available": True, "mask_confidence": 85.0},
            "rgb_bottle_center": [10, 20],
            "rgb_bottle_angle": 3.0,
            "depth_bottle_distance": 320.0,
            "depth_valid_ratio": 0.8,
            "depth_alignment_quality": {"intrinsics_available": True},
            "depth_object_geometry": {"geometry_available": True, "distance_mm": 320.0, "depth_valid_ratio": 0.8},
            "rgbd_bottle_tracking": {"tracking_status": "TRACKING", "tracking_confidence": 90.0},
            "pointcloud_scene_geometry": {"scene_raw_point_count": 1000},
            "pointcloud_bottle_geometry": {"valid": True, "roi_source": "body_contour", "filtered_point_count": 100},
            "pointcloud_bottle_extents": {"width_x": 50, "height_y": 200, "depth_z": 30},
            "pointcloud_grip_width_3d": 50.0,
            "davis_aps_frame": {"aps_state": "APS_COLOR_VERIFIED", "aps_fps": 10.0},
            "event_rate": 0.0,
            "event_motion_activity": {"event_rate": 0.0, "motion_class": "STABLE"},
            "event_stability": 90.0,
            "event_rotation": 0.0,
            "event_vibration": 0.0,
            "event_incipient_slip": 0.0,
            "event_gross_slip": 0.0,
            "event_impact": 0.0,
            "visual_object_quality": None,
            "geometry_quality": None,
            "analytic_epsilon": 0.2,
            "wrench_quality": {"force_closure": True, "quality_label": "GOOD"},
            "live_operational_quality": 80.0,
            "synchronization_confidence": {"sync_quality": "approximate", "max_sensor_time_offset_ms": 6.0},
        },
        source_fields={},
        availability={k: True for k in ["rgb", "depth", "rgbd", "pointcloud", "davis_aps", "davis_events", "quality", "sync"]},
        freshness={"pointcloud": "CURRENT"},
    )
    cases = [
        ("rgb_only_poisoned_depth", ["rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle"]),
        # distinct DAVIS-isolation case: enable RGB+depth (so depth is NOT
        # poisoned) and prove poisoned DAVIS still cannot leak -- previously this
        # case was byte-identical to rgb_only_poisoned_depth
        ("rgb_depth_poisoned_davis", ["rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle", "depth_bottle_distance", "depth_valid_ratio", "depth_alignment_quality", "depth_object_geometry"]),
        ("depth_only_poisoned_rgb", ["depth_bottle_distance", "depth_valid_ratio", "depth_alignment_quality", "depth_object_geometry"]),
        ("events_only_poisoned_realsense", ["event_rate", "event_motion_activity", "event_stability", "event_rotation", "event_vibration", "event_incipient_slip", "event_gross_slip", "event_impact"]),
        ("pointcloud_disabled_poisoned_pointcloud", ["rgb_bottle_detection"]),
        ("epsilon_disabled_poisoned_epsilon", ["rgb_bottle_detection", "depth_bottle_distance"]),
    ]
    rows = []
    passed = True
    for name, enabled in cases:
        config = {"configuration_id": name, "display_name": name, "feature_ids": enabled, "benchmark_object_mode": "bottle_only"}
        clean = evaluate_configuration_card(config, base)
        poisoned = evaluate_configuration_card(config, poison_disabled_packet(base, enabled))
        stable = {
            key: clean.get(key)
            for key in ["bottle_detected", "detection_status", "detection_confidence", "distance_mm", "event_rate", "epsilon_value", "result_validity"]
        } == {
            key: poisoned.get(key)
            for key in ["bottle_detected", "detection_status", "detection_confidence", "distance_mm", "event_rate", "epsilon_value", "result_validity"]
        }
        no_leak = not poisoned.get("feature_leakage_detected")
        ok = bool(stable and no_leak)
        passed = passed and ok
        rows.append({
            "case": name,
            "passed": ok,
            "stable_result": stable,
            "feature_leakage_detected": poisoned.get("feature_leakage_detected"),
            "used_features": poisoned.get("used_features"),
            "blocked_features": poisoned.get("blocked_features"),
        })
    payload = {
        "passed": passed,
        "feature_leakage_detected": not passed or any(row.get("feature_leakage_detected") for row in rows),
        "case_count": len(rows),
        "cases": rows,
        "rows": rows,
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

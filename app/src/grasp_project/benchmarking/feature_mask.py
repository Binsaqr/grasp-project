"""Feature-mask evaluator for display-only benchmark comparisons.

The evaluator reads existing status dictionaries and hides any value whose
feature dependencies are disabled. It does not reopen cameras, run detectors,
or modify production decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from .feature_registry import (
    FEATURE_BY_ID,
    FEATURE_CATEGORIES,
    canonical_feature_ids,
    default_comparison_configurations,
    feature_registry_payload,
)
from .evaluators import evaluate_configuration_card
from .masked_feature_view import build_feature_packet


def _category_enabled(feature_ids: set[str], category: str) -> bool:
    return any((FEATURE_BY_ID.get(fid) and FEATURE_BY_ID[fid].category == category) for fid in feature_ids)


def _has_any(feature_ids: set[str], *ids: str) -> bool:
    return any(fid in feature_ids for fid in ids)


def validate_feature_selection(feature_ids: Iterable[str], *, auto_enable: bool = False) -> dict:
    selected = canonical_feature_ids(feature_ids)
    selected_set = set(selected)
    unknown = [fid for fid in feature_ids if fid not in FEATURE_BY_ID]
    auto_enabled: list[str] = []

    changed = True
    while auto_enable and changed:
        changed = False
        for fid in list(selected_set):
            feature = FEATURE_BY_ID[fid]
            for dep in feature.dependencies:
                if dep not in selected_set and dep in FEATURE_BY_ID:
                    selected_set.add(dep)
                    auto_enabled.append(dep)
                    changed = True

    missing: dict[str, list[str]] = {}
    for fid in selected_set:
        feature = FEATURE_BY_ID[fid]
        deps = [dep for dep in feature.dependencies if dep not in selected_set]
        if deps:
            missing[fid] = deps

    ordered = canonical_feature_ids(selected_set)
    valid = not unknown and not missing
    if unknown:
        reason = f"Unknown features: {', '.join(unknown)}"
    elif missing:
        reason = "Missing dependencies are shown per feature."
    else:
        reason = "Feature selection is valid."
    return {
        "selected_features": ordered,
        "unknown_features": unknown,
        "missing_dependencies": missing,
        "auto_enabled_dependencies": canonical_feature_ids(auto_enabled),
        "feature_selection_valid": valid,
        "feature_selection_reason": reason,
    }


def mask_bundle(bundle: dict[str, Any], feature_ids: Iterable[str]) -> dict[str, Any]:
    """Return a source-pruned bundle for tests and report exports."""
    selected = set(canonical_feature_ids(feature_ids))
    masked = {"production": dict(bundle.get("production") or {})}
    if _category_enabled(selected, "RealSense RGB"):
        masked["rgb"] = dict(bundle.get("rgb") or {})
    if _category_enabled(selected, "RealSense Depth"):
        masked["depth"] = dict(bundle.get("depth") or {})
    if _category_enabled(selected, "RGB-D Combined"):
        masked["rgbd"] = dict(bundle.get("rgbd") or {})
    if _category_enabled(selected, "Point Cloud"):
        masked["pointcloud"] = dict(bundle.get("pointcloud") or {})
    if _category_enabled(selected, "DAVIS APS"):
        masked["davis_aps"] = dict(bundle.get("davis_aps") or {})
    if _category_enabled(selected, "DAVIS Events"):
        masked["davis_events"] = dict(bundle.get("davis_events") or {})
    if _category_enabled(selected, "Quality"):
        masked["quality"] = dict(bundle.get("quality") or {})
    return masked


def _num(*values):
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first(*values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _confidence_from_bool(ok: bool, fallback: float | None = None) -> float:
    if fallback is not None:
        return max(0.0, min(100.0, float(fallback)))
    return 85.0 if ok else 0.0


@dataclass(frozen=True)
class BenchmarkSources:
    realsense: dict[str, Any]
    davis: dict[str, Any]
    pointcloud: dict[str, Any]
    sync: dict[str, Any]
    quality: dict[str, Any]
    live_quality: dict[str, Any]
    embodiment: dict[str, Any]


def _perception(rs: dict[str, Any]) -> dict[str, Any]:
    return dict(rs.get("realsense_perception") or {})


def _production_summary(sources: BenchmarkSources) -> dict:
    rs = sources.realsense
    qm = sources.quality
    lq = sources.live_quality
    return {
        "production_context": "validated_full_pipeline",
        "production_isolated": True,
        "decision": rs.get("decision") or rs.get("ui_primary_status") or qm.get("quality_label") or "N/A",
        "recommended_action": rs.get("recommended_action") or rs.get("ui_primary_action") or lq.get("next_action") or "N/A",
        "ready_to_close": bool(rs.get("ready_to_close")),
        "ready_to_lift": rs.get("ready_to_lift"),
        "quality_label": qm.get("quality_label_display") or qm.get("quality_label"),
        "live_quality_label": lq.get("live_quality_label"),
        "writer_session_id": rs.get("writer_session_id"),
        "frame_id": rs.get("frame_id"),
    }


def evaluate_configuration(configuration: dict[str, Any], sources: BenchmarkSources) -> dict:
    started = time.perf_counter()
    feature_ids = canonical_feature_ids(configuration.get("feature_ids") or [])
    feature_set = set(feature_ids)
    validation = validate_feature_selection(feature_ids)
    rs = sources.realsense
    perception = _perception(rs)
    qm = sources.quality
    pc = sources.pointcloud
    dm = sources.davis
    sync = sources.sync
    lq = sources.live_quality

    has_rgb = _category_enabled(feature_set, "RealSense RGB")
    has_depth = _category_enabled(feature_set, "RealSense Depth")
    has_rgbd = _category_enabled(feature_set, "RGB-D Combined")
    has_pc = _category_enabled(feature_set, "Point Cloud")
    has_aps = _category_enabled(feature_set, "DAVIS APS")
    has_events = _category_enabled(feature_set, "DAVIS Events")
    has_quality = _category_enabled(feature_set, "Quality")

    limitations: list[str] = []
    missing_features: list[str] = []
    contributions: list[dict] = []

    if has_rgb:
        detected = str(perception.get("bottle_detection_status") or rs.get("target_status") or "").upper() not in {"", "LOST", "NO_TARGET", "NO_BOTTLE"}
        detection_status = perception.get("bottle_detection_status") or rs.get("target_status") or "waiting"
        detection_conf = _num(perception.get("detection_confidence"), rs.get("tracking_confidence"))
        contributions.append({"feature": "RGB", "effect": "detection", "message": "RGB can contribute bottle contour, center, angle, and gripper visibility."})
    elif has_depth:
        detected = False
        detection_status = "depth_limited_without_rgb_identity"
        detection_conf = None
        limitations.append("Depth-only cannot identify a stationary bottle appearance without a valid prior ROI or segmentation.")
    elif has_pc:
        detected = bool(pc.get("bottle_pointcloud_valid") or pc.get("bottle_cloud_valid"))
        detection_status = "pointcloud_geometry_only" if detected else "pointcloud_dependency_missing_or_unavailable"
        detection_conf = 55.0 if detected else 0.0
        limitations.append("Point-cloud-only depends on depth/intrinsics and an ROI or mask source.")
    elif has_events:
        event_rate = _num(dm.get("event_rate"), dm.get("davis_event_rate")) or 0.0
        detected = False
        detection_status = "motion_evidence_only" if event_rate > 0 else "stationary_bottle_not_identifiable_from_events_only"
        detection_conf = 35.0 if event_rate > 0 else 0.0
        limitations.append("DAVIS events alone cannot claim stationary bottle detection.")
    elif has_aps:
        detected = False
        detection_status = "aps_context_only"
        detection_conf = None
        limitations.append("DAVIS APS is shown as context and is not a RealSense bottle detector.")
    else:
        detected = False
        detection_status = "no_detection_features_enabled"
        detection_conf = None

    if has_depth:
        contributions.append({"feature": "Depth", "effect": "geometry", "message": "Depth may contribute distance, grip width, and contact-depth consistency."})
    else:
        missing_features.append("depth")

    if has_pc:
        pc_current = str(pc.get("freshness") or pc.get("pointcloud_freshness") or "").upper() in {"CURRENT", "UPDATED", ""}
        pc_contact = pc.get("contact_3d_quality") or ("valid" if pc.get("contact_points_on_bottle_cloud") else "unavailable")
        contributions.append({"feature": "Point Cloud", "effect": "3D validation", "message": f"Point cloud contact validation: {pc_contact}."})
        if not pc_current:
            limitations.append("Point cloud is stale or unavailable and should not be treated as current geometry.")
    else:
        missing_features.append("point cloud")

    if has_events:
        motion_estimate = dm.get("motion_estimate") or {}
        displacement = _num(dm.get("displacement_px"), motion_estimate.get("displacement_px"))
        rotation_deg = _num(dm.get("rotation_delta_deg"), motion_estimate.get("rotation_delta_deg"))
        disturbance = dm.get("disturbance_type") or motion_estimate.get("disturbance_type")
        if displacement is not None:
            message = f"DAVIS Events measured {displacement:.1f} px motion"
            if rotation_deg is not None:
                message += f" and {rotation_deg:.1f} deg rotation"
            if disturbance and disturbance not in {"OBJECT_MOTION", "none"}:
                message += f"; disturbance classified as {disturbance}, not object slip"
            message += "."
        else:
            message = f"DAVIS Events unavailable for precise motion: {dm.get('motion_estimate_invalid_reason') or 'no quantitative estimate'}."
        contributions.append({"feature": "DAVIS Events", "effect": "post-grasp motion", "message": message})
    else:
        missing_features.append("DAVIS events")

    contact_valid = None
    contact_conf = None
    if has_rgbd or _has_any(feature_set, "visual_contact_quality", "analytic_epsilon", "wrench_quality"):
        contact_valid = bool(
            perception.get("contact_pair_valid")
            or rs.get("visual_contact_ready")
            or qm.get("contact_quality_gate_pass")
        )
        contact_conf = _num(
            perception.get("contact_source_confidence"),
            qm.get("contact_confidence"),
            rs.get("visual_contact_pair_quality_score"),
        )
    else:
        limitations.append("Contact validity disabled for this configuration.")

    epsilon_enabled = "analytic_epsilon" in feature_set or "wrench_quality" in feature_set
    epsilon_available = False
    epsilon_value = None
    if epsilon_enabled:
        if contact_valid:
            epsilon_value = _first(qm.get("epsilon_score"), qm.get("epsilon"), qm.get("raw_epsilon"))
            epsilon_available = epsilon_value is not None
        if not epsilon_available:
            limitations.append("epsilon unavailable: contact geometry missing")

    live_enabled = "live_operational_quality" in feature_set
    live_quality_score = None
    if live_enabled:
        live_quality_score = _first(lq.get("live_quality_score"), lq.get("score"))
        if not has_events:
            limitations.append("Live quality excludes DAVIS slip/rotation/vibration penalties because DAVIS Events are disabled.")
    else:
        limitations.append("Live quality disabled.")

    if has_events:
        stability = _num(dm.get("stability"), dm.get("stability_score"), dm.get("post_grasp_stability_score"))
        event_penalty = max(0.0, 100.0 - stability) if stability is not None else 0.0
    else:
        event_penalty = 0.0

    confidence_parts = [
        detection_conf,
        contact_conf if contact_conf is not None else None,
        100.0 - event_penalty if has_events else None,
    ]
    confidence_values = [v for v in confidence_parts if v is not None]
    confidence = round(sum(confidence_values) / len(confidence_values), 1) if confidence_values else 0.0

    if not validation["feature_selection_valid"]:
        decision = "UNAVAILABLE"
        main_limitation = validation["feature_selection_reason"]
    elif epsilon_enabled and epsilon_available:
        decision = qm.get("quality_label_display") or qm.get("quality_label") or "QUALITY_AVAILABLE"
        main_limitation = limitations[0] if limitations else "All selected quality dependencies are available."
    elif has_events and not (has_rgb or has_depth or has_rgbd):
        decision = dm.get("decision") or "MOTION_ONLY_CONTEXT"
        main_limitation = limitations[0] if limitations else "Motion-only comparison cannot produce ready-to-close."
    else:
        decision = rs.get("recommended_action") or rs.get("ui_primary_action") or "HOLD_AND_RECHECK"
        main_limitation = limitations[0] if limitations else "Display-only benchmark card."

    tracking_conf = _num(rs.get("tracking_confidence"), perception.get("tracking_confidence"))
    if not (has_rgbd or (has_rgb and has_depth)):
        tracking_conf = None

    latency = _num(
        (rs.get("processed_overlay_timing") or {}).get("total_processed_overlay_ms"),
        (rs.get("performance_profile") or {}).get("timings_ms", {}).get("total"),
    )
    davis_motion_valid = _first(dm.get("motion_estimate_valid"), (dm.get("motion_estimate") or {}).get("estimate_valid")) if has_events else None
    davis_motion_validity = None
    if has_events:
        davis_motion_validity = "VALID" if davis_motion_valid is True else ("INVALID" if davis_motion_valid is False else "UNKNOWN")
    result = {
        "configuration_id": configuration.get("configuration_id") or configuration.get("preset_id") or "custom",
        "display_name": configuration.get("display_name") or "Custom",
        "preset_id": configuration.get("preset_id"),
        "benchmark_only": True,
        "controls_robot": False,
        "production_isolated": True,
        "same_bundle_id": _first(sync.get("last_bundle_id"), sync.get("bundle_id"), rs.get("frame_id")),
        "enabled_features": feature_ids,
        **validation,
        "bottle_detected": bool(detected),
        "detection_status": detection_status,
        "detection_confidence": detection_conf,
        "bbox_xyxy": perception.get("bottle_bbox_xyxy") if has_rgb else None,
        "center_px": perception.get("bottle_center_px") if has_rgb else None,
        "angle_deg": perception.get("bottle_angle_deg") if "rgb_bottle_angle" in feature_set else None,
        "mask_available": bool(qm.get("image_bottle_hull_px") or qm.get("current_live_bottle_hull_px")) if "rgb_bottle_segmentation" in feature_set else False,
        "mask_confidence": _confidence_from_bool(bool(qm.get("image_bottle_hull_px"))) if "rgb_bottle_segmentation" in feature_set else None,
        "tracking_status": rs.get("target_status") if tracking_conf is not None else "not_enabled",
        "tracking_confidence": tracking_conf,
        "contact_validity": contact_valid,
        "contact_confidence": contact_conf,
        "contact_source": _first(qm.get("contact_source"), perception.get("contact_source")) if contact_valid is not None else "disabled",
        "contact_count": _first(perception.get("detected_contact_count"), qm.get("detected_contact_count"), qm.get("contact_count")),
        "grip_width_mm": _first(rs.get("grip_width_mm"), qm.get("grip_width_mm")) if has_depth else None,
        "contact_depth_validity": perception.get("contact_depth_consistent") if has_depth else None,
        "pointcloud_contact_validity": pc.get("contact_3d_quality") if has_pc else "disabled",
        "davis_dx_px": _first(dm.get("dx_px"), (dm.get("motion_estimate") or {}).get("dx_px")) if has_events else None,
        "davis_dy_px": _first(dm.get("dy_px"), (dm.get("motion_estimate") or {}).get("dy_px")) if has_events else None,
        "davis_displacement_px": _first(dm.get("displacement_px"), (dm.get("motion_estimate") or {}).get("displacement_px")) if has_events else None,
        "davis_cumulative_slip_px": _first(dm.get("cumulative_slip_px"), (dm.get("motion_estimate") or {}).get("cumulative_slip_px")) if has_events else None,
        "davis_downward_slip_px": _first(dm.get("downward_slip_px"), (dm.get("motion_estimate") or {}).get("downward_slip_px")) if has_events else None,
        "davis_rotation_delta_deg": _first(dm.get("rotation_delta_deg"), (dm.get("motion_estimate") or {}).get("rotation_delta_deg")) if has_events else None,
        "davis_cumulative_rotation_deg": _first(dm.get("cumulative_rotation_deg"), (dm.get("motion_estimate") or {}).get("cumulative_rotation_deg")) if has_events else None,
        "davis_angular_velocity_deg_s": _first(dm.get("angular_velocity_deg_s"), (dm.get("motion_estimate") or {}).get("angular_velocity_deg_s")) if has_events else None,
        "davis_vibration_rms_px": _first(dm.get("vibration_amplitude_px_rms"), (dm.get("motion_estimate") or {}).get("vibration_amplitude_px_rms")) if has_events else None,
        "davis_motion_estimate_confidence": _first(dm.get("motion_confidence"), dm.get("motion_estimation_confidence"), (dm.get("motion_estimate") or {}).get("motion_estimation_confidence")) if has_events else None,
        "davis_motion_estimate_valid": davis_motion_valid,
        "davis_motion_validity": davis_motion_validity,
        "davis_disturbance_type": _first(dm.get("disturbance_type"), (dm.get("motion_estimate") or {}).get("disturbance_type")) if has_events else None,
        "epsilon_available": epsilon_available,
        "epsilon_value": epsilon_value if epsilon_available else None,
        "quality_score": _first(qm.get("normalized_quality_score"), qm.get("epsilon_score")) if has_quality else None,
        "analytic_quality_score": _first(qm.get("normalized_quality_score"), qm.get("epsilon_score")) if epsilon_enabled else None,
        "live_quality_score": live_quality_score,
        "quality_label": qm.get("quality_label_display") or qm.get("quality_label") if has_quality else "disabled",
        "quality_confidence": confidence,
        "decision": decision,
        "confidence": confidence,
        "latency_ms": latency,
        "main_limitation": main_limitation,
        "missing_features": missing_features,
        "hard_blockers": list(perception.get("perception_blockers") or rs.get("ready_blockers") or []),
        "recommended_action": decision,
        "contributions": contributions,
        "profiling": {
            "feature_mask_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
    }
    return result


def evaluate_configuration(configuration: dict[str, Any], sources: BenchmarkSources) -> dict:
    """Evaluate one configuration through a strict masked feature view."""
    started = time.perf_counter()
    packet = build_feature_packet(
        realsense=sources.realsense,
        davis=sources.davis,
        pointcloud=sources.pointcloud,
        sync=sources.sync,
        quality=sources.quality,
        live_quality=sources.live_quality,
        embodiment=sources.embodiment,
    )
    card = evaluate_configuration_card(configuration, packet)
    card.setdefault("profiling", {})
    feature_mask_ms = max(0.001, round((time.perf_counter() - started) * 1000.0, 3))
    card["profiling"]["feature_mask_ms"] = feature_mask_ms
    card["latency_ms"] = feature_mask_ms
    card["latency_source"] = "masked_evaluator_compute"
    return card


def _metric_rows(cards: list[dict], production: dict) -> list[dict]:
    rows = []
    for card in cards:
        rows.append({
            "configuration": card["display_name"],
            "bottle_detected": card["bottle_detected"],
            "detection_confidence": card["detection_confidence"],
            "contact_validity": card["contact_validity"],
            "epsilon": card["epsilon_value"] if card["epsilon_available"] else "disabled/unavailable",
            "live_quality": card["live_quality_score"] if card["live_quality_score"] is not None else "disabled/unavailable",
            "decision": card["decision"],
            "latency_ms": card["latency_ms"],
            "production_decision": production.get("decision"),
        })
    return rows


def build_benchmark_lab_status(
    *,
    realsense: dict[str, Any] | None = None,
    davis: dict[str, Any] | None = None,
    pointcloud: dict[str, Any] | None = None,
    sync: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    live_quality: dict[str, Any] | None = None,
    embodiment: dict[str, Any] | None = None,
    configurations: list[dict] | None = None,
    trial_state: dict[str, Any] | None = None,
) -> dict:
    started = time.perf_counter()
    sources = BenchmarkSources(
        realsense=dict(realsense or {}),
        davis=dict(davis or {}),
        pointcloud=dict(pointcloud or {}),
        sync=dict(sync or {}),
        quality=dict(quality or {}),
        live_quality=dict(live_quality or {}),
        embodiment=dict(embodiment or {}),
    )
    configs = list(configurations or default_comparison_configurations())[:4]
    production = _production_summary(sources)
    cards = [evaluate_configuration(config, sources) for config in configs]
    eval_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return {
        "benchmark_mode": "display_report_only",
        "banner": "Benchmark results are experimental and do not control the robot.",
        "production_context": production,
        "production_result_unchanged": True,
        "sync": {
            "sync_mode": sources.sync.get("sync_mode"),
            "sync_quality": sources.sync.get("sync_quality"),
            "max_sensor_time_offset_ms": sources.sync.get("max_sensor_time_offset_ms"),
            "stale_modalities": sources.sync.get("stale_modalities") or [],
            "unavailable_modalities": sources.sync.get("unavailable_modalities") or [],
            "bundle_valid_for_benchmark": sources.sync.get("bundle_valid_for_benchmark"),
            "bundle_valid_for_live_quality": sources.sync.get("bundle_valid_for_live_quality"),
        },
        "max_comparison_cards": 4,
        "registry": feature_registry_payload(),
        "configurations": configs,
        "comparison_cards": cards,
        "metrics_rows": _metric_rows(cards, production),
        "trial": dict(trial_state or {}),
        "profiling": {
            "benchmark_eval_ms": eval_ms,
            "comparison_card_update_ms": eval_ms,
            "trial_record_write_ms": 0.0,
            "dashboard_render_ms": None,
        },
    }


def ablation_configurations() -> list[dict]:
    full = set(FEATURE_BY_ID)
    groups = {
        "full_minus_rgb": "RealSense RGB",
        "full_minus_depth": "RealSense Depth",
        "full_minus_pointcloud": "Point Cloud",
        "full_minus_davis_aps": "DAVIS APS",
        "full_minus_davis_events": "DAVIS Events",
        "full_minus_epsilon": None,
    }
    configs = [{"configuration_id": "full_multimodal", "display_name": "Full Multimodal", "feature_ids": list(full)}]
    for config_id, category in groups.items():
        if category is None:
            feature_ids = [fid for fid in full if fid != "analytic_epsilon"]
            display = "Full minus Epsilon"
        else:
            feature_ids = [fid for fid in full if FEATURE_BY_ID[fid].category != category]
            display = f"Full minus {category}"
        configs.append({"configuration_id": config_id, "display_name": display, "feature_ids": canonical_feature_ids(feature_ids)})
    return configs

"""One-click Benchmark Lab runner.

The runner samples existing dashboard benchmark outputs. It does not open
cameras, disable production safety, alter RealSense tracking, change DAVIS
scoring, or recompute epsilon/wrench math.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .feature_mask import validate_feature_selection
from .feature_registry import FEATURE_BY_ID, canonical_feature_ids, default_run_configurations
from .masked_feature_view import DEPENDENCY_MISSING, NOT_APPLICABLE, SENSOR_UNAVAILABLE, STALE_DATA
from .trial_recorder import evaluate_ground_truth_labels

DEFAULT_CONFIG_DURATION_SEC = 20.0
DEFAULT_SAMPLE_RATE_HZ = 5.0

STATES = {
    "IDLE",
    "PRECHECK",
    "READY",
    "RUNNING",
    "CONFIG_RUNNING",
    "PAUSED",
    "STOPPING",
    "COMPLETE",
    "FAILED",
    "CANCELLED",
}

RUN_MODES = {
    "sequential_live": "Sequential Live",
}

DEFAULT_WEIGHTS = {
    "detection_weight": 0.20,
    "quality_weight": 0.20,
    "decision_weight": 0.20,
    "contact_weight": 0.20,
    "latency_weight": 0.10,
    "reliability_weight": 0.10,
}

SCORING_MODES = {
    "safety_first": "Safety First",
    "detection_first": "Detection First",
    "quality_first": "Quality First",
    "speed_first": "Speed First",
    "motion_safety_first": "Motion Safety First",
}

VALID_WINNER_STATUSES = {"VALID", "VALID_WITH_LIMITATIONS"}


def _now_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{time.time_ns() % 1000000:06d}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def load_benchmark_settings(path: str | Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[3]
    cfg = _read_yaml(Path(path) if path else root / "config" / "benchmarking.yaml")
    mode = str(cfg.get("ranking_mode") or "safety_first")
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in (cfg.get(mode) or {}).items() if k in weights})
    weights.update({k: float(v) for k, v in cfg.items() if k in weights})
    return {
        "duration_sec": float(cfg.get("duration_sec", DEFAULT_CONFIG_DURATION_SEC)),
        "sample_rate_hz": float(cfg.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ)),
        "ranking_mode": mode,
        "ranking_mode_label": SCORING_MODES.get(mode, mode.replace("_", " ").title()),
        "weights": weights,
        "scoring_formula": (
            "weighted_score = detection*{detection_weight:.2f} + contact*{contact_weight:.2f} + "
            "quality*{quality_weight:.2f} + safety*{decision_weight:.2f} + "
            "latency*{latency_weight:.2f} + sensor_reliability*{reliability_weight:.2f}"
        ).format(**weights),
    }


def normalize_run_configurations(configurations: Iterable[dict] | None, duration_sec: float | int | None = None) -> list[dict]:
    if not configurations:
        return default_run_configurations(duration_sec=duration_sec)
    normalized = []
    for idx, config in enumerate(configurations):
        feature_ids = canonical_feature_ids(config.get("feature_ids") or config.get("enabled_features") or [])
        config_duration = float(config.get("duration_sec") or duration_sec or DEFAULT_CONFIG_DURATION_SEC)
        disabled = [fid for fid in FEATURE_BY_ID if fid not in set(feature_ids)]
        deps = sorted({dep for fid in feature_ids for dep in FEATURE_BY_ID[fid].dependencies})
        expected = sorted({field for fid in feature_ids for field in FEATURE_BY_ID[fid].output_fields})
        row = {
            "configuration_id": config.get("configuration_id") or config.get("preset_id") or f"config_{idx + 1}",
            "display_name": config.get("display_name") or config.get("configuration_id") or f"Config {idx + 1}",
            "preset_id": config.get("preset_id"),
            "feature_ids": feature_ids,
            "enabled_features": feature_ids,
            "disabled_features": disabled,
            "dependencies": deps,
            "duration_sec": config_duration,
            "expected_outputs": expected,
            "limitations": config.get("limitations") or [FEATURE_BY_ID[fid].limitation for fid in feature_ids if FEATURE_BY_ID[fid].limitation],
        }
        for key in (
            "enabled_groups",
            "requested_features",
            "dependency_features",
            "derived_features",
            "missing_dependencies",
            "feature_selection_valid",
            "feature_selection_reason",
            "not_applicable_features",
            "unsupported_requested_features",
            "expected_limitations",
            "pointcloud_required",
            "benchmark_object_mode",
            "result_type",
            "gripper_tests_skipped",
            "contact_tests_skipped",
        ):
            if key in config:
                row[key] = config[key]
        normalized.append(row)
    return normalized


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _std(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0 if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(0.95 * len(ordered))) - 1))
    return round(ordered[idx], 3)


def _stat_summary(values: list[float], invalid_count: int = 0) -> dict:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "mean": _mean(clean),
        "median": round(statistics.median(clean), 3) if clean else None,
        "std": _std(clean),
        "min": round(min(clean), 3) if clean else None,
        "max": round(max(clean), 3) if clean else None,
        "p95": _p95(clean),
        "sample_count": len(values) + int(invalid_count),
        "valid_sample_count": len(clean),
        "invalid_or_stale_sample_count": int(invalid_count),
    }


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "ready", "valid", "yes", "1", "tracking", "detected"}


def _numeric(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _distance(a, b) -> float | None:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) >= 2 and len(b) >= 2):
        return None
    try:
        return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
    except Exception:
        return None


def _feature_categories(feature_ids: Iterable[str]) -> set[str]:
    return {FEATURE_BY_ID[fid].category for fid in feature_ids if fid in FEATURE_BY_ID}


def _config_sensor_unavailable_reason(config: dict, precheck: dict) -> str | None:
    optional = precheck.get("optional") or {}
    categories = _feature_categories(config.get("feature_ids") or [])
    checks = {
        "Point Cloud": "pointcloud_active",
        "DAVIS APS": "davis_aps_active",
        "DAVIS Events": "davis_events_active",
    }
    for category, check_id in checks.items():
        if category in categories:
            status = optional.get(check_id) or {}
            if status and not status.get("ok"):
                return status.get("reason") or f"{check_id}_unavailable"
    return None


def _result_validity(status: str, sample_count: int, unavailable_reason: str | None, stale_count: int, invalid_count: int) -> tuple[str, str]:
    reason = unavailable_reason or "ok"
    if status == "unavailable":
        if unavailable_reason and "dependency" in unavailable_reason.lower():
            return "DEPENDENCY_MISSING", unavailable_reason
        return "SENSOR_UNAVAILABLE", unavailable_reason or "Selected sensor or feature was unavailable."
    if status in {"skipped", "cancelled", "failed"}:
        return "INVALID_TRIAL", status
    if sample_count < 2:
        return "INSUFFICIENT_DATA", "Fewer than two valid benchmark samples were recorded."
    if stale_count and stale_count >= max(1, sample_count // 2):
        return "STALE_DATA", "Half or more of samples reported stale data."
    if invalid_count:
        return "VALID_WITH_LIMITATIONS", f"{invalid_count} samples were unavailable or invalid."
    if stale_count:
        return "VALID_WITH_LIMITATIONS", f"{stale_count} stale samples were excluded from interpretation."
    return "VALID", reason


def _motion_distribution(samples: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for sample in samples:
        label = str(sample.get("davis_motion_class") or "N/A")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _baseline_deltas(row: dict, baseline: dict | None) -> dict:
    if not baseline:
        return {}
    fields = {
        "detection_success_delta": "detection_success_rate",
        "quality_score_delta": "average_quality_score",
        "quality_stability_delta": "quality_stability_score",
        "latency_delta": "average_latency_ms",
        "contact_valid_delta": "contact_valid_rate",
        "ready_to_close_delta": "ready_to_close_rate",
        "confidence_delta": "confidence_mean",
    }
    deltas = {}
    for out_key, field in fields.items():
        a = _numeric(row.get(field))
        b = _numeric(baseline.get(field))
        deltas[out_key] = round(a - b, 3) if a is not None and b is not None else None
    return deltas


def _configuration_explanation(result: dict, config: dict) -> str:
    categories = _feature_categories(config.get("feature_ids") or [])
    name = result.get("display_name") or config.get("display_name") or "This configuration"
    if result.get("result_validity") not in VALID_WINNER_STATUSES:
        return f"{name} could not produce a valid comparison result: {result.get('result_validity_reason') or result.get('main_failure_reason') or 'insufficient data'}."
    parts = []
    if categories == {"RealSense RGB"} or ("RealSense RGB" in categories and "RealSense Depth" not in categories):
        parts.append("RGB can compare bottle outline, center, and gripper visibility, but cannot measure metric distance, grip width, or 3D contact validity by itself.")
    if "RealSense Depth" in categories and "RealSense RGB" not in categories:
        parts.append("Depth can compare distance and rough geometry, but object identity is limited without RGB evidence.")
    if "RealSense RGB" in categories and "RealSense Depth" in categories:
        parts.append("RGB-D is the primary pre-close comparison because RGB supplies object evidence and depth supplies distance/contact geometry.")
    if "Point Cloud" in categories:
        parts.append("Point cloud contributes 3D contact and geometry validation when current; stale or bbox-only crops are treated as limitations.")
    if "DAVIS Events" in categories:
        parts.append("DAVIS events contribute post-grasp motion safety such as slip, rotation, vibration, and stability; they do not identify a stationary bottle.")
    if "DAVIS APS" in categories:
        parts.append("DAVIS APS contributes visual context only when APS mode is available and honestly labeled.")
    if "Quality" in categories:
        parts.append("Quality features read existing epsilon/live-quality outputs; benchmarking does not recompute epsilon or change production decisions.")
    if result.get("main_failure_reason"):
        parts.append(f"Main limitation: {result['main_failure_reason']}.")
    return " ".join(parts) or f"{name} was compared using the selected dashboard status fields."


def _sample_from_card(card: dict, benchmark_status: dict, config: dict) -> dict:
    source = benchmark_status.get("source_status") or {}
    realsense = source.get("realsense") or {}
    pointcloud = source.get("pointcloud") or {}
    sync = source.get("sync") or {}
    feature_ids = canonical_feature_ids(card.get("enabled_features") or config.get("feature_ids") or [])
    categories = _feature_categories(feature_ids)
    has_rgb = "RealSense RGB" in categories
    has_depth = "RealSense Depth" in categories
    has_pointcloud = "Point Cloud" in categories
    has_aps = "DAVIS APS" in categories
    has_events = "DAVIS Events" in categories
    has_quality = "Quality" in categories
    event_activity = card.get("event_motion_activity") if isinstance(card.get("event_motion_activity"), dict) else {}
    closeup_valid_value = event_activity.get("closeup_object_motion_valid")
    if closeup_valid_value is None:
        closeup_valid_value = event_activity.get("object_motion_valid")
    closeup_object_motion_valid = closeup_valid_value is True
    sample = {
        "timestamp": time.time(),
        "synchronized_bundle_id": card.get("same_bundle_id"),
        "configuration_id": card.get("configuration_id"),
        "configuration_name": card.get("display_name"),
        "enabled_features": feature_ids,
        "bottle_detected": bool(card.get("bottle_detected")),
        "detection_status": card.get("detection_status"),
        "detection_confidence": card.get("detection_confidence"),
        "tracking_confidence": card.get("tracking_confidence"),
        "center_px": card.get("center_px"),
        "angle_deg": card.get("angle_deg"),
        "mask_available": card.get("mask_available"),
        "distance_mm": card.get("distance_mm"),
        "grip_width_mm": card.get("grip_width_mm"),
        "contact_source": card.get("contact_source"),
        "contact_valid": card.get("contact_validity"),
        "contact_confidence": card.get("contact_confidence"),
        "pointcloud_valid": card.get("pointcloud_valid") if has_pointcloud else None,
        "pointcloud_contact_valid": card.get("pointcloud_contact_validity") if has_pointcloud else None,
        "davis_motion_class": card.get("motion_class") if has_events else None,
        "davis_event_rate": card.get("event_rate") if has_events else None,
        "davis_aps_visibility_status": card.get("davis_aps_visibility_status"),
        "davis_aps_visibility_score": card.get("davis_aps_visibility_score"),
        "davis_aps_visual_detection_valid": card.get("davis_aps_visual_detection_valid"),
        "davis_aps_limitation": card.get("davis_aps_limitation"),
        "event_health_valid": event_activity.get("event_health_valid"),
        "object_motion_valid": event_activity.get("object_motion_valid"),
        "disturbance_detected": event_activity.get("disturbance_detected"),
        "davis_motion_validity": event_activity.get("davis_motion_validity"),
        "davis_motion_exclusion_reason": event_activity.get("davis_motion_exclusion_reason"),
        "closeup_object_motion_valid": closeup_valid_value,
        "closeup_roi_source": event_activity.get("closeup_roi_source"),
        "closeup_motion_class": event_activity.get("closeup_motion_class"),
        "closeup_displacement_px": event_activity.get("closeup_displacement_px") if closeup_object_motion_valid else None,
        "closeup_rotation_deg": event_activity.get("closeup_rotation_deg") if closeup_object_motion_valid else None,
        "closeup_vibration_rms_px": event_activity.get("closeup_vibration_rms_px") if closeup_object_motion_valid else None,
        "closeup_stability_score": event_activity.get("closeup_stability_score") if closeup_object_motion_valid else None,
        "closeup_disturbance_type": event_activity.get("closeup_disturbance_type"),
        "closeup_used_for_live_quality": bool(event_activity.get("closeup_used_for_live_quality") and closeup_object_motion_valid),
        "impact_score": card.get("impact_score") if has_events else None,
        "incipient_slip_score": card.get("incipient_slip_score") if has_events else None,
        "gross_slip_score": card.get("gross_slip_score") if has_events else None,
        "rotation_score": card.get("rotation_score") if has_events else None,
        "vibration_score": card.get("vibration_score") if has_events else None,
        "stability_score": card.get("stability_score") if has_events else None,
        "davis_dx_px": card.get("davis_dx_px") if has_events else None,
        "davis_dy_px": card.get("davis_dy_px") if has_events else None,
        "davis_displacement_px": card.get("davis_displacement_px") if has_events and closeup_object_motion_valid else None,
        "davis_cumulative_slip_px": card.get("davis_cumulative_slip_px") if has_events and closeup_object_motion_valid else None,
        "davis_downward_slip_px": card.get("davis_downward_slip_px") if has_events and closeup_object_motion_valid else None,
        "davis_rotation_delta_deg": card.get("davis_rotation_delta_deg") if has_events and closeup_object_motion_valid else None,
        "davis_cumulative_rotation_deg": card.get("davis_cumulative_rotation_deg") if has_events and closeup_object_motion_valid else None,
        "davis_angular_velocity_deg_s": card.get("davis_angular_velocity_deg_s") if has_events and closeup_object_motion_valid else None,
        "davis_vibration_rms_px": card.get("davis_vibration_rms_px") if has_events and closeup_object_motion_valid else None,
        "davis_motion_estimate_confidence": card.get("davis_motion_estimate_confidence") if has_events else None,
        "davis_motion_estimate_valid": card.get("davis_motion_estimate_valid") if has_events else None,
        # distinct from davis_motion_validity above (the gate-based object-motion
        # vs disturbance label): this is the motion ESTIMATE's own VALID/INVALID
        "davis_motion_estimate_validity": card.get("davis_motion_validity") if has_events else None,
        "davis_disturbance_type": card.get("davis_disturbance_type") if has_events else None,
        "epsilon_available": card.get("epsilon_available") if has_quality else False,
        "epsilon_value": card.get("epsilon_value") if has_quality else None,
        "analytic_quality_score": (card.get("analytic_quality_score") or card.get("quality_score")) if has_quality else None,
        "live_quality_score": card.get("live_quality_score") if has_quality else None,
        "quality_label": card.get("quality_label") if has_quality else "disabled",
        "ready_to_close": card.get("ready_to_close"),
        "ready_to_lift": card.get("ready_to_lift"),
        "decision": card.get("decision"),
        "hard_blockers": card.get("hard_blockers") or [],
        "confidence": card.get("confidence"),
        "latency_ms": card.get("latency_ms"),
        "latency_source": card.get("latency_source"),
        "feature_mask_ms": (card.get("profiling") or {}).get("feature_mask_ms"),
        "sensor_availability": source.get("sensor_availability") or {},
        "rgb_active": bool(realsense.get("realsense_rgb_active") or realsense.get("target_status")) if has_rgb else None,
        "depth_active": bool(realsense.get("realsense_depth_active") or realsense.get("bottle_distance_mm")) if has_depth else None,
        "rgb_fps": (realsense.get("camera_rgb_fps") or realsense.get("live_fps") or realsense.get("overlay_fps")) if has_rgb else None,
        "depth_valid_ratio": card.get("depth_valid_ratio") if has_depth else None,
        "frame_age_ms": realsense.get("frame_age_ms"),
        "pointcloud_freshness": (sync.get("pointcloud_freshness") or pointcloud.get("freshness") or pointcloud.get("pointcloud_freshness")) if has_pointcloud else None,
        "pointcloud_roi_source": card.get("pointcloud_roi_source") if has_pointcloud else None,
        "pointcloud_crop_confidence": pointcloud.get("bottle_crop_confidence") if has_pointcloud else None,
        "pointcloud_contact_3d_quality": card.get("pointcloud_contact_validity") if has_pointcloud else None,
        "aps_state": card.get("aps_mode") if has_aps else None,
        "aps_fps": card.get("aps_fps") if has_aps else None,
        "event_state": "active" if has_events and card.get("events_available") else None,
        "event_baseline_status": None,
        "sync_mode": card.get("sync_mode") or sync.get("sync_mode"),
        "sync_offset_ms": card.get("sync_offset_ms") if card.get("sync_offset_ms") is not None else (sync.get("max_sensor_time_offset_ms") or sync.get("max_offset_ms")),
        "sync_quality": card.get("sync_quality") or sync.get("sync_quality"),
        "realsense_age_ms": sync.get("realsense_age_ms"),
        "davis_event_age_ms": sync.get("davis_event_age_ms") if has_events else None,
        "davis_aps_age_ms": sync.get("davis_aps_age_ms") if has_aps else None,
        "pointcloud_age_sec": sync.get("pointcloud_age_sec") if has_pointcloud else None,
        "stale_modalities": card.get("stale_modalities") or sync.get("stale_modalities") or [],
        "unavailable_modalities": card.get("unavailable_modalities") or sync.get("unavailable_modalities") or [],
        "bundle_valid_for_benchmark": card.get("bundle_valid_for_benchmark") if card.get("bundle_valid_for_benchmark") is not None else sync.get("bundle_valid_for_benchmark"),
        "bundle_valid_for_live_quality": card.get("bundle_valid_for_live_quality") if card.get("bundle_valid_for_live_quality") is not None else sync.get("bundle_valid_for_live_quality"),
        "main_limitation": card.get("main_limitation"),
        "result_validity": card.get("result_validity"),
        "result_validity_reason": card.get("result_validity_reason"),
        "feature_selection_valid": card.get("feature_selection_valid"),
        "used_features": card.get("used_features") or [],
        "attempted_features": card.get("attempted_features") or [],
        "disabled_features": card.get("disabled_features") or [],
        "blocked_features": card.get("blocked_features") or {},
        "missing_dependencies": card.get("missing_dependencies") or {},
        "not_applicable_features": card.get("not_applicable_features") or {},
        "unavailable_features": card.get("unavailable_features") or {},
        "stale_features": card.get("stale_features") or {},
        "unsupported_features": card.get("unsupported_features") or {},
        "production_fields_used": card.get("production_fields_used") or [],
        "feature_leakage_detected": bool(card.get("feature_leakage_detected")),
        "feature_leakage_reason": card.get("feature_leakage_reason"),
        "evaluator_trace": card.get("evaluator_trace") or [],
        "used_feature_groups": sorted(categories),
    }
    return sample


def summarize_samples(config: dict, samples: list[dict], status: str = "complete", unavailable_reason: str | None = None) -> dict:
    elapsed = 0.0
    if samples:
        elapsed = max(0.001, float(samples[-1]["timestamp"]) - float(samples[0]["timestamp"]))
    n = len(samples)
    detections = [s for s in samples if _truthy(s.get("bottle_detected"))]
    detection_conf = [_numeric(s.get("detection_confidence")) for s in samples]
    detection_conf = [v for v in detection_conf if v is not None]
    centers = [s.get("center_px") for s in samples if s.get("center_px") is not None]
    center_steps = [_distance(a, b) for a, b in zip(centers, centers[1:])]
    center_steps = [v for v in center_steps if v is not None]
    angles = [_numeric(s.get("angle_deg")) for s in samples]
    angles = [v for v in angles if v is not None]
    quality = [_numeric(s.get("analytic_quality_score") if s.get("analytic_quality_score") is not None else s.get("live_quality_score")) for s in samples]
    quality = [v for v in quality if v is not None]
    eps = [_numeric(s.get("epsilon_value")) for s in samples if s.get("epsilon_available")]
    eps = [v for v in eps if v is not None]
    contact = [s for s in samples if _truthy(s.get("contact_valid"))]
    ready = [s for s in samples if _truthy(s.get("ready_to_close"))]
    event_rates = [_numeric(s.get("davis_event_rate")) for s in samples]
    event_rates = [v for v in event_rates if v is not None]
    davis_displacements = [_numeric(s.get("davis_displacement_px")) for s in samples]
    davis_displacements = [v for v in davis_displacements if v is not None]
    davis_cumulative_slips = [_numeric(s.get("davis_cumulative_slip_px")) for s in samples]
    davis_cumulative_slips = [v for v in davis_cumulative_slips if v is not None]
    davis_rotations = [_numeric(s.get("davis_rotation_delta_deg")) for s in samples]
    davis_rotations = [v for v in davis_rotations if v is not None]
    davis_cumulative_rotations = [_numeric(s.get("davis_cumulative_rotation_deg")) for s in samples]
    davis_cumulative_rotations = [v for v in davis_cumulative_rotations if v is not None]
    davis_vibrations = [_numeric(s.get("davis_vibration_rms_px")) for s in samples]
    davis_vibrations = [v for v in davis_vibrations if v is not None]
    davis_confidence = [_numeric(s.get("davis_motion_estimate_confidence")) for s in samples]
    davis_confidence = [v for v in davis_confidence if v is not None]
    davis_motion_valid_values = [s.get("davis_motion_estimate_valid") for s in samples if s.get("davis_motion_estimate_valid") is not None]
    davis_motion_valid_count = sum(value is True for value in davis_motion_valid_values)
    davis_motion_invalid_count = sum(value is False for value in davis_motion_valid_values)
    davis_motion_unknown_count = max(0, len(samples) - len(davis_motion_valid_values))
    davis_motion_validity = "UNKNOWN"
    if davis_motion_valid_values:
        davis_motion_validity = "VALID" if davis_motion_valid_count >= davis_motion_invalid_count else "INVALID"
        if davis_motion_unknown_count:
            davis_motion_validity = f"{davis_motion_validity}_WITH_UNKNOWN_SAMPLES"
    stability = [_numeric(s.get("stability_score")) for s in samples]
    stability = [v for v in stability if v is not None]
    latencies = [_numeric(s.get("latency_ms")) for s in samples if s.get("latency_source") == "masked_evaluator_compute"]
    latencies = [v for v in latencies if v is not None]
    unavailable_samples = [s for s in samples if str(s.get("decision") or "").upper() == "UNAVAILABLE"]
    stale_samples = [s for s in samples if "stale" in str(s.get("main_limitation") or "").lower()]
    hard_blocker_count = sum(len(s.get("hard_blockers") or []) for s in samples)
    confidence = [_numeric(s.get("confidence")) for s in samples]
    confidence = [v for v in confidence if v is not None]
    invalid_count = len(unavailable_samples)
    warnings = {
        "incipient_slip_warning_count": sum((_numeric(s.get("incipient_slip_score")) or 0) >= 50 for s in samples),
        "gross_slip_warning_count": sum((_numeric(s.get("gross_slip_score")) or 0) >= 50 for s in samples),
        "rotation_warning_count": sum((_numeric(s.get("rotation_score")) or 0) >= 50 for s in samples),
        "vibration_warning_count": sum((_numeric(s.get("vibration_score")) or 0) >= 50 for s in samples),
    }
    warning_total = sum(int(v) for v in warnings.values())
    failure_reasons = [s.get("main_limitation") for s in samples if s.get("main_limitation")]
    pointcloud_stale = [
        s for s in samples
        if str(s.get("pointcloud_freshness") or "").upper() in {"STALE", "ORPHANED", "UNAVAILABLE"}
        or "stale" in str(s.get("pointcloud_contact_3d_quality") or "").lower()
    ]
    # A large RealSense<->DAVIS software-timestamp offset only affects a
    # configuration that fuses a RealSense-family modality (rgb/depth/pointcloud,
    # one clock) with a DAVIS-family modality (aps/events, a different clock).
    # For single-family configs the cross-sensor offset is irrelevant and must
    # not mark their results stale. Even for fusion configs it is a known
    # limitation of the uncalibrated side-by-side mount, not stale sensor data,
    # so it downgrades to VALID_WITH_LIMITATIONS rather than excluding as
    # STALE_DATA.
    rs_family = {"rgb", "depth", "pointcloud"}
    davis_family = {"davis_aps", "davis_events"}
    config_groups = set(config.get("enabled_groups") or [])
    cross_sensor_fusion = bool(config_groups & rs_family) and bool(config_groups & davis_family)
    sync_stale_samples = [
        s for s in samples
        if s.get("stale_modalities") or str(s.get("sync_quality") or "").lower() in {"stale", "stale_modalities", "offset_too_large", "missing_timestamp"}
    ] if cross_sensor_fusion else []
    dropped_samples = [
        s for s in samples
        if str(s.get("sync_quality") or "").lower() in {"stale", "missing", "error", "offset_too_large", "missing_timestamp"}
    ] if cross_sensor_fusion else []
    stale_count = len(stale_samples) + len(pointcloud_stale)
    result_validity, result_validity_reason = _result_validity(status, n, unavailable_reason, stale_count, invalid_count)
    if cross_sensor_fusion and sync_stale_samples and result_validity in {"VALID", "STALE_DATA"}:
        result_validity, result_validity_reason = (
            "VALID_WITH_LIMITATIONS",
            f"{len(sync_stale_samples)}/{n} samples had a large RealSense<->DAVIS software-timestamp offset "
            "(uncalibrated side-by-side mount); cross-sensor temporal fusion is approximate.",
        )
    sample_validities = [str(s.get("result_validity") or "") for s in samples if s.get("result_validity")]
    if sample_validities and status == "complete":
        if all(v == DEPENDENCY_MISSING for v in sample_validities):
            result_validity, result_validity_reason = DEPENDENCY_MISSING, "All samples reported missing feature dependencies."
        elif all(v == SENSOR_UNAVAILABLE for v in sample_validities):
            result_validity, result_validity_reason = SENSOR_UNAVAILABLE, "All samples reported selected sensor unavailable."
        elif all(v == NOT_APPLICABLE for v in sample_validities):
            result_validity, result_validity_reason = NOT_APPLICABLE, "All samples reported feature set not applicable."
        elif any(v == "VALID_WITH_LIMITATIONS" for v in sample_validities) and result_validity == "VALID":
            result_validity, result_validity_reason = "VALID_WITH_LIMITATIONS", "Some enabled features produced valid outputs with limitations."
        elif any(v in {DEPENDENCY_MISSING, SENSOR_UNAVAILABLE, STALE_DATA, NOT_APPLICABLE} for v in sample_validities) and result_validity == "VALID":
            result_validity, result_validity_reason = "VALID_WITH_LIMITATIONS", "Some enabled features were blocked, unavailable, stale, or not applicable."
    if result_validity == "VALID" and "davis_aps" in config_groups:
        aps_invalid = [s for s in samples if s.get("davis_aps_visual_detection_valid") is False]
        if aps_invalid and len(aps_invalid) >= max(1, n // 2):
            aps_status_seen = next((s.get("davis_aps_visibility_status") for s in aps_invalid if s.get("davis_aps_visibility_status")), "LOW_LIGHT")
            result_validity, result_validity_reason = (
                "VALID_WITH_LIMITATIONS",
                f"DAVIS APS visibility was {aps_status_seen} for {len(aps_invalid)}/{n} samples; "
                "APS is context only under current lighting, not a reliable visual detector.",
            )
    quality_stability = round(max(0.0, 100.0 - (_std(quality) or 0.0)), 3) if quality else None
    latency_stats = _stat_summary(latencies, invalid_count=invalid_count + len(stale_samples))
    result = {
        "configuration_id": config.get("configuration_id"),
        "display_name": config.get("display_name"),
        "enabled_feature_count": len(config.get("feature_ids") or []),
        "enabled_feature_labels": [FEATURE_BY_ID[fid].display_name for fid in config.get("feature_ids") or [] if fid in FEATURE_BY_ID],
        "enabled_features": canonical_feature_ids(config.get("feature_ids") or []),
        "disabled_features": sorted({fid for sample in samples for fid in (sample.get("disabled_features") or [])} or {fid for fid in FEATURE_BY_ID if fid not in set(config.get("feature_ids") or [])}),
        "used_features": sorted({fid for sample in samples for fid in (sample.get("used_features") or [])}),
        "attempted_features": sorted({fid for sample in samples for fid in (sample.get("attempted_features") or [])}),
        "blocked_features": {k: v for sample in samples for k, v in (sample.get("blocked_features") or {}).items()},
        "missing_dependencies": {k: v for sample in samples for k, v in (sample.get("missing_dependencies") or {}).items()},
        "not_applicable_features": {k: v for sample in samples for k, v in (sample.get("not_applicable_features") or {}).items()},
        "unavailable_features": {k: v for sample in samples for k, v in (sample.get("unavailable_features") or {}).items()},
        "stale_features": {k: v for sample in samples for k, v in (sample.get("stale_features") or {}).items()},
        "unsupported_features": {k: v for sample in samples for k, v in (sample.get("unsupported_features") or {}).items()},
        "production_fields_used": sorted({field for sample in samples for field in (sample.get("production_fields_used") or [])}),
        "feature_leakage_detected": any(bool(sample.get("feature_leakage_detected")) for sample in samples),
        "feature_leakage_reason": next((sample.get("feature_leakage_reason") for sample in samples if sample.get("feature_leakage_reason")), ""),
        "evaluator_trace": (samples[-1].get("evaluator_trace") if samples else []),
        "status": status,
        "unavailable_reason": unavailable_reason,
        "result_validity": result_validity,
        "result_validity_reason": result_validity_reason,
        "has_ground_truth": False,
        "result_type": "estimated_performance",
        "ground_truth_labels_used": 0,
        "accuracy_metrics_valid": False,
        "accuracy_metrics_reason": "No ground-truth labels were attached to this benchmark run.",
        "benchmark_only": True,
        "production_control": False,
        "controls_robot": False,
        "sample_count": n,
        "valid_sample_count": max(0, n - invalid_count - len(stale_samples)),
        "invalid_sample_count": invalid_count,
        "stale_sample_count": stale_count,
        "sync_stale_sample_count": len(sync_stale_samples),
        "sync_stale_modalities": sorted({modal for sample in samples for modal in (sample.get("stale_modalities") or [])}),
        "bundle_valid_for_benchmark_rate": round(100.0 * sum(sample.get("bundle_valid_for_benchmark") is True for sample in samples) / n, 2) if n else 0.0,
        "dropped_sample_count": len(dropped_samples),
        "duration_sec": float(config.get("duration_sec") or DEFAULT_CONFIG_DURATION_SEC),
        "actual_elapsed_sec": round(elapsed, 3) if n else 0.0,
        "detection_success_rate": round(100.0 * len(detections) / n, 2) if n else 0.0,
        "detection_confidence_mean": _mean(detection_conf),
        "detection_confidence_std": _std(detection_conf),
        "average_detection_confidence": _mean(detection_conf),
        "target_loss_count": sum(str(s.get("detection_status") or "").upper() in {"LOST", "NO_TARGET", "NO_BOTTLE"} for s in samples),
        "target_loss_rate": round(100.0 * sum(str(s.get("detection_status") or "").upper() in {"LOST", "NO_TARGET", "NO_BOTTLE"} for s in samples) / n, 2) if n else 0.0,
        "false_positive_count": None,
        "false_negative_count": None,
        "tracking_jitter_px": _std(center_steps),
        "center_jitter_px": _std(center_steps),
        "angle_jitter_deg": _std(angles),
        "mask_stability_iou": None,
        "detection_latency_ms_mean": None,
        "detection_latency_ms_p95": None,
        "detection_latency_reason": "detector-only latency is not measured independently",
        "analytic_quality_available": bool(quality),
        "epsilon_available": bool(eps),
        "average_quality_score": _mean(quality),
        "mean_quality_score": _mean(quality),
        "quality_score_std": _std(quality),
        "live_quality_mean": _mean([_numeric(s.get("live_quality_score")) for s in samples if _numeric(s.get("live_quality_score")) is not None]),
        "live_quality_std": _std([_numeric(s.get("live_quality_score")) for s in samples if _numeric(s.get("live_quality_score")) is not None]),
        "minimum_quality_score": round(min(quality), 3) if quality else None,
        "maximum_quality_score": round(max(quality), 3) if quality else None,
        "quality_min": round(min(quality), 3) if quality else None,
        "quality_max": round(max(quality), 3) if quality else None,
        "quality_stability": quality_stability,
        "quality_stability_score": quality_stability,
        "epsilon_mean": _mean(eps),
        "epsilon_std": _std(eps),
        "contact_valid_rate": round(100.0 * len(contact) / n, 2) if n else 0.0,
        "contact_source_used": max(set([s.get("contact_source") for s in samples if s.get("contact_source")] or ["N/A"]), key=([s.get("contact_source") for s in samples if s.get("contact_source")] or ["N/A"]).count),
        "contact_confidence_mean": _mean([_numeric(s.get("contact_confidence")) for s in samples if _numeric(s.get("contact_confidence")) is not None]),
        "grip_width_mean_mm": _mean([_numeric(s.get("grip_width_mm")) for s in samples if _numeric(s.get("grip_width_mm")) is not None]),
        "grip_width_std_mm": _std([_numeric(s.get("grip_width_mm")) for s in samples if _numeric(s.get("grip_width_mm")) is not None]),
        "depth_valid_rate": _mean([_numeric(s.get("depth_valid_ratio")) * 100.0 for s in samples if _numeric(s.get("depth_valid_ratio")) is not None]),
        "pointcloud_contact_valid_rate": round(100.0 * sum(str(s.get("pointcloud_contact_3d_quality") or s.get("pointcloud_contact_valid") or "").lower() in {"valid", "true"} for s in samples) / n, 2) if n else 0.0,
        "recompute_required_count": sum("recompute" in str(s.get("main_limitation") or "").lower() for s in samples),
        "stale_quality_count": sum("stale" in str(s.get("quality_label") or s.get("main_limitation") or "").lower() for s in samples),
        "ready_to_close_rate": round(100.0 * len(ready) / n, 2) if n else 0.0,
        "unsafe_blocker_count": hard_blocker_count,
        "event_stream_available": bool(event_rates),
        "average_event_rate": _mean(event_rates),
        "event_rate_mean": _mean(event_rates),
        "event_rate_peak": round(max(event_rates), 3) if event_rates else None,
        "mean_displacement_px": _mean(davis_displacements),
        "max_displacement_px": round(max(davis_displacements), 3) if davis_displacements else None,
        "cumulative_slip_px": round(max(davis_cumulative_slips), 3) if davis_cumulative_slips else None,
        "mean_rotation_deg": _mean([abs(v) for v in davis_rotations]),
        "max_rotation_deg": round(max([abs(v) for v in davis_rotations]), 3) if davis_rotations else None,
        "cumulative_rotation_deg": round(max([abs(v) for v in davis_cumulative_rotations]), 3) if davis_cumulative_rotations else None,
        "vibration_rms_px": _mean(davis_vibrations),
        "davis_motion_estimate_confidence_mean": _mean(davis_confidence),
        "motion_validity": davis_motion_validity,
        "motion_valid_sample_count": davis_motion_valid_count,
        "motion_invalid_sample_count": davis_motion_invalid_count,
        "motion_unknown_sample_count": davis_motion_unknown_count,
        "motion_validity_rate": round(100.0 * davis_motion_valid_count / len(davis_motion_valid_values), 2) if davis_motion_valid_values else None,
        "davis_disturbance_distribution": max(set([str(s.get("davis_disturbance_type") or "N/A") for s in samples] or ["N/A"]), key=([str(s.get("davis_disturbance_type") or "N/A") for s in samples] or ["N/A"]).count),
        "first_motion_warning_latency_ms": None,
        "impact_count": sum((_numeric(s.get("impact_score")) or 0) >= 50 for s in samples),
        **warnings,
        "motion_warning_count": warning_total,
        "stability_mean": _mean(stability),
        "stability_min": round(min(stability), 3) if stability else None,
        "motion_class_distribution": _motion_distribution(samples),
        "first_warning_latency_ms": None,
        "false_motion_warning_count": None,
        "missed_motion_event_count": None,
        "average_latency_ms": _mean(latencies),
        "mean_latency_ms": _mean(latencies),
        "p95_latency_ms": _p95(latencies),
        "max_latency_ms": round(max(latencies), 3) if latencies else None,
        "latency_measurement": "masked_evaluator_compute" if latencies else "NOT_MEASURED",
        "effective_fps": round(n / elapsed, 3) if n and elapsed > 0 else 0.0,
        "processing_time_by_feature": {"selected_feature_mask": _mean(latencies)} if latencies else {},
        "slowest_feature": None,
        "slowest_feature_reason": "per-feature timing not measured",
        "sensor_update_rate": round(n / elapsed, 3) if n and elapsed > 0 else 0.0,
        "dashboard_update_rate": round(n / elapsed, 3) if n and elapsed > 0 else 0.0,
        "missing_sensor_count": sum(1 for s in samples if any(v is False for v in (s.get("sensor_availability") or {}).values())),
        "stale_data_count": stale_count,
        "unavailable_fraction": round(len(unavailable_samples) / n, 3) if n else (1.0 if status == "unavailable" else 0.0),
        "confidence_mean": _mean(confidence),
        "hard_blocker_count": hard_blocker_count,
        "main_failure_reason": unavailable_reason or (max(set(failure_reasons), key=failure_reasons.count) if failure_reasons else None),
        "estimated_performance": True,
        "ground_truth_available": False,
        "statistical_summaries": {
            "detection_confidence": _stat_summary(detection_conf, invalid_count=invalid_count),
            "quality_score": _stat_summary(quality, invalid_count=invalid_count + len(stale_samples)),
            "epsilon": _stat_summary(eps, invalid_count=n - len(eps) if n else 0),
            "latency_ms": latency_stats,
            "event_rate": _stat_summary(event_rates, invalid_count=n - len(event_rates) if n else 0),
        },
        "sensor_health": {
            "realsense_rgb": {
                "status": "active" if any(s.get("rgb_active") for s in samples) else "unavailable",
                "fps": _mean([_numeric(s.get("rgb_fps")) for s in samples if _numeric(s.get("rgb_fps")) is not None]),
                "frame_age_ms_mean": _mean([_numeric(s.get("frame_age_ms")) for s in samples if _numeric(s.get("frame_age_ms")) is not None]),
            },
            "depth": {
                "status": "active" if any(s.get("depth_active") for s in samples) else "unavailable",
                "valid_depth_ratio": _mean([_numeric(s.get("depth_valid_ratio")) for s in samples if _numeric(s.get("depth_valid_ratio")) is not None]),
            },
            "point_cloud": {
                "status": max(set([str(s.get("pointcloud_freshness") or "unavailable") for s in samples] or ["unavailable"]), key=([str(s.get("pointcloud_freshness") or "unavailable") for s in samples] or ["unavailable"]).count),
                "roi_source": max(set([str(s.get("pointcloud_roi_source") or "N/A") for s in samples] or ["N/A"]), key=([str(s.get("pointcloud_roi_source") or "N/A") for s in samples] or ["N/A"]).count),
                "contact_3d_validity": max(set([str(s.get("pointcloud_contact_3d_quality") or "N/A") for s in samples] or ["N/A"]), key=([str(s.get("pointcloud_contact_3d_quality") or "N/A") for s in samples] or ["N/A"]).count),
            },
            "davis_aps": {
                "status": max(set([str(s.get("aps_state") or "unavailable") for s in samples] or ["unavailable"]), key=([str(s.get("aps_state") or "unavailable") for s in samples] or ["unavailable"]).count),
                "fps": _mean([_numeric(s.get("aps_fps")) for s in samples if _numeric(s.get("aps_fps")) is not None]),
            },
            "davis_events": {
                "status": max(set([str(s.get("event_state") or "unavailable") for s in samples] or ["unavailable"]), key=([str(s.get("event_state") or "unavailable") for s in samples] or ["unavailable"]).count),
                "event_rate_mean": _mean(event_rates),
                "baseline_status": max(set([str(s.get("event_baseline_status") or "N/A") for s in samples] or ["N/A"]), key=([str(s.get("event_baseline_status") or "N/A") for s in samples] or ["N/A"]).count),
            },
            "synchronization": {
                "sync_quality": max(set([str(s.get("sync_quality") or "N/A") for s in samples] or ["N/A"]), key=([str(s.get("sync_quality") or "N/A") for s in samples] or ["N/A"]).count),
                "sync_mode": max(set([str(s.get("sync_mode") or "N/A") for s in samples] or ["N/A"]), key=([str(s.get("sync_mode") or "N/A") for s in samples] or ["N/A"]).count),
                "max_offset_ms": _p95([_numeric(s.get("sync_offset_ms")) for s in samples if _numeric(s.get("sync_offset_ms")) is not None]),
            },
        },
        "plain_language_explanation": None,
    }
    result["plain_language_explanation"] = _configuration_explanation(result, config)
    return result


def _latency_scores(results: list[dict]) -> dict[str, float]:
    valid = [r for r in results if r.get("average_latency_ms") is not None and r.get("latency_measurement") == "masked_evaluator_compute" and r.get("status") == "complete"]
    if not valid:
        return {}
    min_lat = min(float(r["average_latency_ms"]) for r in valid)
    max_lat = max(float(r["average_latency_ms"]) for r in valid)
    span = max(1.0, max_lat - min_lat)
    return {
        r["configuration_id"]: max(0.0, min(100.0, 100.0 - ((float(r["average_latency_ms"]) - min_lat) / span) * 100.0))
        for r in valid
    }


def rank_results(results: list[dict], settings: dict | None = None) -> list[dict]:
    weights = dict(DEFAULT_WEIGHTS)
    weights.update((settings or {}).get("weights") or {})
    latency_scores = _latency_scores(results)
    ranked = []
    for result in results:
        detection = float(result.get("detection_success_rate") or 0.0)
        quality = float(result.get("quality_stability") or result.get("average_quality_score") or 0.0)
        safety = max(0.0, float(result.get("ready_to_close_rate") or 0.0) - min(50.0, float(result.get("unsafe_blocker_count") or 0.0) * 5.0))
        contact = float(result.get("contact_valid_rate") or 0.0)
        latency = latency_scores.get(result.get("configuration_id"), 0.0)
        reliability = max(0.0, 100.0 - 100.0 * float(result.get("unavailable_fraction") or 0.0) - 2.0 * float(result.get("stale_data_count") or 0))
        winner_valid = result.get("result_validity") in VALID_WINNER_STATUSES and result.get("status") in {"complete", "skipped", "repeated"}
        score = (
            detection * weights["detection_weight"]
            + quality * weights["quality_weight"]
            + safety * weights["decision_weight"]
            + contact * weights["contact_weight"]
            + latency * weights["latency_weight"]
            + reliability * weights["reliability_weight"]
        ) if winner_valid else 0.0
        ranked.append({
            **result,
            "combined_score": round(score, 3),
            "weighted_score": round(score, 3),
            "winner_eligible": bool(winner_valid),
            "score_components": {
                "detection": round(detection, 3),
                "quality": round(quality, 3),
                "safety": round(safety, 3),
                "contact": round(contact, 3),
                "latency": round(latency, 3),
                "sensor_reliability": round(reliability, 3),
            },
        })
    ranked.sort(key=lambda r: (r.get("winner_eligible", False), r.get("combined_score", 0.0)), reverse=True)
    rank = 1
    for row in ranked:
        if row.get("winner_eligible"):
            row["rank"] = rank
            rank += 1
        else:
            row["rank"] = None
    return ranked


def _best_by(results: list[dict], key: str, reverse=True) -> dict | None:
    valid = [r for r in results if r.get(key) is not None and r.get("winner_eligible", True)]
    if not valid:
        return None
    return sorted(valid, key=lambda r: r.get(key) or 0.0, reverse=reverse)[0]


def feature_contribution_summary(ranked: list[dict], configurations: list[dict]) -> dict:
    by_config = {r["configuration_id"]: r for r in ranked}
    sensors = {
        "RGB": "RealSense RGB",
        "Depth": "RealSense Depth",
        "Point Cloud": "Point Cloud",
        "DAVIS APS": "DAVIS APS",
        "DAVIS Events": "DAVIS Events",
        "Epsilon": "Quality",
        "Sync": "Quality",
    }
    summary = {}
    for label, category in sensors.items():
        with_scores = []
        without_scores = []
        for config in configurations:
            result = by_config.get(config.get("configuration_id")) or {}
            if not result.get("winner_eligible"):
                continue
            score = result.get("combined_score")
            if score is None:
                continue
            cats = _feature_categories(config.get("feature_ids") or [])
            if category in cats:
                with_scores.append(float(score))
            else:
                without_scores.append(float(score))
        delta = (_mean(with_scores) or 0.0) - (_mean(without_scores) or 0.0) if with_scores and without_scores else None
        categories = []
        if delta is None:
            categories.append("insufficient comparison data")
        elif delta > 5:
            categories.append("higher estimated score")
        elif delta < -5:
            categories.append("lower estimated score")
        else:
            categories.append("little measured value in this run")
        summary[label] = {
            "mean_score_with_feature": _mean(with_scores),
            "mean_score_without_feature": _mean(without_scores),
            "estimated_delta": round(delta, 3) if delta is not None else None,
            "categories": categories,
            "finding": "observed score association in this recorded trial" if delta is not None else "insufficient comparison data",
        }
        if label == "DAVIS Events":
            event_rows = [r for r in ranked if category in _feature_categories((next((c for c in configurations if c.get("configuration_id") == r.get("configuration_id")), {}) or {}).get("feature_ids") or [])]
            displacements = [_numeric(r.get("max_displacement_px")) for r in event_rows if _numeric(r.get("max_displacement_px")) is not None]
            rotations = [_numeric(r.get("max_rotation_deg")) for r in event_rows if _numeric(r.get("max_rotation_deg")) is not None]
            disturbances = [str(r.get("davis_disturbance_distribution") or "") for r in event_rows if r.get("davis_disturbance_distribution")]
            if displacements or rotations:
                summary[label]["finding"] = (
                    f"DAVIS Events measured up to {max(displacements):.1f} px slip"
                    if displacements else "DAVIS Events did not produce a valid displacement estimate"
                )
                if rotations:
                    summary[label]["finding"] += f" and {max(rotations):.1f} deg rotation"
                if disturbances and any(d in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"} for d in disturbances):
                    summary[label]["finding"] += "; a global disturbance was separated from object slip"
                summary[label]["finding"] += "."
    return summary


def _trial_quality(ranked: list[dict], ground_truth_exists: bool) -> dict:
    warnings = []
    if not ranked:
        return {"trial_quality": "INVALID", "warnings": ["No benchmark results were recorded."]}
    if not ground_truth_exists:
        warnings.append("Ground truth missing; values are estimated performance, not accuracy.")
    if any(r.get("sample_count", 0) < 2 for r in ranked):
        warnings.append("One or more configurations had too few samples.")
    if any(r.get("target_loss_count", 0) > 0 for r in ranked):
        warnings.append("RealSense target loss occurred during the benchmark.")
    if any(r.get("stale_data_count", 0) > 0 for r in ranked):
        warnings.append("Stale sensor data was present.")
    if any(r.get("result_validity") in {"SENSOR_UNAVAILABLE", "DEPENDENCY_MISSING"} for r in ranked):
        warnings.append("At least one selected sensor or dependency was unavailable.")
    if any((r.get("sensor_health") or {}).get("synchronization", {}).get("sync_quality") in {"error", "stale"} for r in ranked):
        warnings.append("Synchronization quality warnings were present.")
    if any((r.get("average_latency_ms") or 0) > 100 for r in ranked):
        warnings.append("High benchmark latency was observed.")
    if any(r.get("status") in {"skipped", "cancelled"} for r in ranked):
        warnings.append("Benchmark was stopped early or a configuration was skipped.")
    if any(r.get("result_validity") == "INVALID_TRIAL" for r in ranked):
        return {"trial_quality": "INVALID", "warnings": warnings or ["Invalid trial."]}
    return {"trial_quality": "WARNING" if warnings else "GOOD", "warnings": warnings}


def _recommendations(ranked: list[dict], contribution: dict, baseline_id: str) -> list[str]:
    recommendations = []
    best = next((r for r in ranked if r.get("winner_eligible")), None)
    if best:
        recommendations.append(
            f"Use {best['display_name']} as the best benchmarked configuration for this run: weighted score {best.get('weighted_score')} using recorded detection, contact, quality, safety, latency, and sensor reliability metrics."
        )
    baseline = next((r for r in ranked if r.get("configuration_id") == baseline_id), None)
    full = next((r for r in ranked if r.get("configuration_id") == "full_multimodal"), None)
    if full and baseline:
        deltas = full.get("baseline_deltas") or {}
        if deltas.get("contact_valid_delta") is not None:
            latency_delta = deltas.get("latency_delta")
            latency_text = f"{latency_delta:+.1f} ms" if latency_delta is not None else "N/A"
            recommendations.append(
                f"Full Multimodal changed contact validity by {deltas['contact_valid_delta']:+.1f}% and latency by {latency_text} versus RGB-D baseline."
            )
    pc = contribution.get("Point Cloud") or {}
    if pc.get("estimated_delta") is not None:
        recommendations.append(
            f"Point Cloud configurations had a {pc['estimated_delta']:+.1f} score association; this is confounded by the other enabled features and is not an isolated modality contribution."
        )
    events = contribution.get("DAVIS Events") or {}
    if events.get("estimated_delta") is not None:
        recommendations.append(
            f"DAVIS Events configurations had a {events['estimated_delta']:+.1f} score association; this is not an isolated modality contribution or stationary bottle-detection result."
        )
    stale = [r for r in ranked if r.get("stale_data_count")]
    if stale:
        recommendations.append(
            f"Repeat the benchmark with fresh sensor data: {stale[0]['display_name']} reported {stale[0].get('stale_data_count')} stale samples."
        )
    unavailable = [r for r in ranked if r.get("result_validity") in {"SENSOR_UNAVAILABLE", "DEPENDENCY_MISSING"}]
    if unavailable:
        recommendations.append(
            f"Do not interpret {unavailable[0]['display_name']} as a failure of the algorithm alone; result validity was {unavailable[0].get('result_validity')} because {unavailable[0].get('result_validity_reason')}."
        )
    return recommendations or ["Repeat the benchmark with more samples before drawing a strong conclusion."]


def build_summary(
    results: list[dict],
    configurations: list[dict],
    settings: dict | None = None,
    ground_truth_exists: bool = False,
    ground_truth: dict | list | None = None,
) -> dict:
    ground_truth_summary = evaluate_ground_truth_labels(
        ground_truth,
        declared=ground_truth_exists,
    )
    accuracy_available = ground_truth_summary["accuracy_metrics_valid"]
    ranked = rank_results(results, settings=settings)
    baseline_id = "rgb_depth"
    baseline = next((r for r in ranked if r.get("configuration_id") == baseline_id), None)
    for row in ranked:
        row["baseline_configuration_id"] = baseline_id
        row["baseline_deltas"] = _baseline_deltas(row, baseline)
    best_detection = _best_by(ranked, "detection_success_rate")
    best_quality = _best_by(ranked, "average_quality_score")
    stable_quality = _best_by(ranked, "quality_stability")
    fastest = _best_by(ranked, "average_latency_ms", reverse=False)
    safety_rows = [r for r in ranked if str(r.get("contact_source_used") or "").lower() != "not_applicable_bottle_only"]
    motion_rows = [r for r in ranked if r.get("event_stream_available") and (r.get("motion_valid_sample_count") or 0) > 0]
    safest = _best_by(safety_rows, "ready_to_close_rate")
    best_motion = _best_by(motion_rows, "motion_warning_count")
    best_overall = next((r for r in ranked if r.get("winner_eligible")), None)
    contribution = feature_contribution_summary(ranked, configurations)
    useful = sorted(
        [(k, v.get("estimated_delta")) for k, v in contribution.items() if v.get("estimated_delta") is not None],
        key=lambda item: item[1],
        reverse=True,
    )
    unreliable = sorted(
        [r for r in ranked if r.get("status") != "complete" or (r.get("unavailable_fraction") or 0) > 0],
        key=lambda r: (r.get("unavailable_fraction") or 0, -r.get("combined_score", 0)),
        reverse=True,
    )
    findings = []
    if best_detection:
        findings.append(f"{best_detection['display_name']} had the highest estimated bottle detection rate.")
    if fastest:
        findings.append(f"{fastest['display_name']} had the lowest average latency in this run.")
    if best_overall:
        findings.append(f"{best_overall['display_name']} ranked best overall using the configured weights.")
    for name, delta in useful[:2]:
        if delta and delta > 0:
            findings.append(f"Configurations containing {name} were associated with a higher estimated combined score in this recorded run; other enabled features may explain the difference.")
    if not findings:
        findings.append("No clear winner was supported by the recorded data.")
    if not accuracy_available:
        findings.insert(0, "These values compare system confidence, consistency, latency, and safety behavior. They are not ground-truth accuracy.")
    else:
        for metric_name, metric in ground_truth_summary["accuracy_metrics"].items():
            if metric["valid"]:
                findings.append(
                    f"{metric_name.replace('_', ' ').title()} accuracy was "
                    f"{metric['accuracy_percent']:.1f}% over "
                    f"{metric['applicable_label_count']} current applicable labels."
                )
    trial_quality = _trial_quality(ranked, accuracy_available)
    recommendations = _recommendations(ranked, contribution, baseline_id)
    limitations = sorted({
        str(r.get("result_validity_reason") or r.get("main_failure_reason") or "")
        for r in ranked
        if r.get("result_validity") != "VALID" or r.get("main_failure_reason")
    } - {""})
    if not accuracy_available:
        limitations.append(ground_truth_summary["accuracy_metrics_reason"])
        limitations = sorted(set(limitations))
    if accuracy_available:
        metric_wording = "Estimated Performance + Ground-Truth Accuracy (Applicable Labels)"
        result_type = "estimated_performance_with_ground_truth_accuracy"
    else:
        metric_wording = "Estimated Performance"
        result_type = "estimated_performance"
    return {
        "metric_wording": metric_wording,
        "result_type": result_type,
        **ground_truth_summary,
        "detection_accuracy_valid": ground_truth_summary["accuracy_metrics"][
            "detection"
        ]["valid"],
        "detection_accuracy_percent": ground_truth_summary["accuracy_metrics"][
            "detection"
        ]["accuracy_percent"],
        "pose_classification_accuracy_valid": ground_truth_summary[
            "accuracy_metrics"
        ]["pose_classification"]["valid"],
        "pose_classification_accuracy_percent": ground_truth_summary[
            "accuracy_metrics"
        ]["pose_classification"]["accuracy_percent"],
        "clearance_accuracy_valid": ground_truth_summary["accuracy_metrics"][
            "clearance"
        ]["valid"],
        "clearance_accuracy_percent": ground_truth_summary["accuracy_metrics"][
            "clearance"
        ]["accuracy_percent"],
        "decision_accuracy_valid": ground_truth_summary["accuracy_metrics"][
            "decision"
        ]["valid"],
        "decision_accuracy_percent": ground_truth_summary["accuracy_metrics"][
            "decision"
        ]["accuracy_percent"],
        **trial_quality,
        "baseline_configuration_id": baseline_id,
        "baseline_configuration": baseline.get("display_name") if baseline else "RGB + Depth",
        "ranking_mode": (settings or {}).get("ranking_mode", "safety_first"),
        "ranking_mode_label": (settings or {}).get("ranking_mode_label", "Safety First"),
        "weights": (settings or {}).get("weights", DEFAULT_WEIGHTS),
        "scoring_formula": (settings or {}).get("scoring_formula"),
        "winner_cards": {
            "best_bottle_detection": best_detection,
            "best_grasp_quality": best_quality,
            "best_safety_decision": safest,
            "best_motion_detection": best_motion,
            "most_stable_quality": stable_quality,
            "fastest": fastest,
            "best_overall": best_overall,
            "best_sensor_contribution": None,
            "most_useful_sensor": None,
            "least_useful_feature": None,
            "strongest_positive_score_association": useful[0][0] if useful else None,
            "weakest_score_association": useful[-1][0] if useful else None,
            "most_unreliable_feature": unreliable[0]["display_name"] if unreliable else None,
        },
        "ranking_table": ranked,
        "feature_contribution_summary": contribution,
        "plain_language_findings": findings,
        "recommendations": recommendations,
        "trial_notes": trial_quality.get("warnings", []),
        "limitations": limitations,
        "single_trial_notice": "Single-trial result. Repeat trials needed for statistical confidence.",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()} or {"configuration_id"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_charts(run_dir: Path, ranked: list[dict]) -> None:
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    chart_specs = {
        "detection_success.png": "detection_success_rate",
        "quality_score.png": "average_quality_score",
        "quality_stability.png": "quality_stability_score",
        "ready_to_close.png": "ready_to_close_rate",
        "contact_validity.png": "contact_valid_rate",
        "latency.png": "average_latency_ms",
        "feature_contribution.png": "weighted_score",
        "motion_warnings.png": "motion_warning_count",
        "quality_timeline.png": "average_quality_score",
        "decision_timeline.png": "ready_to_close_rate",
        "sensor_availability.png": "unavailable_fraction",
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.get("display_name") for r in ranked]
        for filename, key in chart_specs.items():
            vals = [r.get(key) or 0 for r in ranked]
            fig, ax = plt.subplots(figsize=(8, 3.6))
            ax.bar(range(len(vals)), vals)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax.set_title(key.replace("_", " ").title())
            ax.set_ylabel("ms" if "latency" in key else "score / percent")
            fig.tight_layout()
            fig.savefig(charts / filename)
            plt.close(fig)
    except Exception:
        for filename in chart_specs:
            (charts / filename).write_bytes(b"")


def write_run_outputs(
    run_dir: Path,
    manifest: dict,
    configurations: list[dict],
    results: list[dict],
    summary: dict,
    timeline: list[dict],
    ground_truth: dict | list | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "benchmark_run_manifest.json", manifest)
    _write_json(run_dir / "benchmark_configurations.json", {"configurations": configurations})
    _write_json(run_dir / "benchmark_results_by_config.json", {"results": results})
    _write_json(run_dir / "benchmark_summary.json", summary)
    _write_json(run_dir / "benchmark_timeline.json", {"timeline": timeline})
    if ground_truth is not None:
        labels = (
            list(ground_truth.get("labels") or [])
            if isinstance(ground_truth, dict)
            else list(ground_truth)
        )
        _write_json(
            run_dir / "benchmark_ground_truth.json",
            {
                "labels": labels,
                **evaluate_ground_truth_labels(
                    labels,
                    declared=True,
                ),
            },
        )
    _write_json(run_dir / "benchmark_validity.json", {
        "trial_quality": summary.get("trial_quality"),
        "warnings": summary.get("warnings") or summary.get("trial_notes") or [],
        "ground_truth_state": summary.get("ground_truth_state"),
        "ground_truth_label_count": summary.get("ground_truth_label_count"),
        "structured_ground_truth_label_count": summary.get("structured_ground_truth_label_count"),
        "current_ground_truth_label_count": summary.get("current_ground_truth_label_count"),
        "ground_truth_labels_used": summary.get("ground_truth_labels_used"),
        "accuracy_metrics_valid": summary.get("accuracy_metrics_valid"),
        "accuracy_metrics_reason": summary.get("accuracy_metrics_reason"),
        "accuracy_metrics": summary.get("accuracy_metrics") or {},
        "results": [
            {
                "configuration_id": row.get("configuration_id"),
                "display_name": row.get("display_name"),
                "result_validity": row.get("result_validity"),
                "result_validity_reason": row.get("result_validity_reason"),
                "winner_eligible": row.get("winner_eligible"),
            }
            for row in summary.get("ranking_table") or []
        ],
    })
    _write_csv(run_dir / "benchmark_results.csv", summary.get("ranking_table") or [])
    contrib_rows = [
        {"feature": feature, **payload}
        for feature, payload in (summary.get("feature_contribution_summary") or {}).items()
    ]
    _write_csv(run_dir / "benchmark_feature_contributions.csv", contrib_rows)
    _write_charts(run_dir, summary.get("ranking_table") or [])
    expanded_outputs = {}
    if any(config.get("benchmark_object_mode") == "bottle_only" and config.get("enabled_groups") for config in configurations):
        try:
            from .bottle_only import feature_leakage_matrix, write_expanded_report_outputs

            samples = []
            samples_path = run_dir / "benchmark_samples.jsonl"
            if samples_path.exists():
                samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
            expanded_outputs = write_expanded_report_outputs(run_dir, configurations, results, samples)
            samples_by_config: dict[str, list[dict]] = {}
            for sample in samples:
                samples_by_config.setdefault(sample.get("configuration_id"), []).append(sample)
            _write_json(run_dir / "feature_leakage_matrix.json", feature_leakage_matrix(configurations, samples_by_config))
        except Exception as exc:
            expanded_outputs = {"expanded_report_error": str(exc)}
    md = [
        "# Benchmark Report",
        "",
        f"Run: `{manifest.get('benchmark_run_id')}`",
        f"Mode: `{manifest.get('run_mode')}`",
        f"Result type: {summary.get('metric_wording')}",
        f"Trial quality: {summary.get('trial_quality')}",
        f"Ground truth: {summary.get('ground_truth_state')} "
        f"({summary.get('ground_truth_labels_used') or 0} current applicable labels used)",
        f"Scoring: {summary.get('ranking_mode_label')} - {summary.get('scoring_formula')}",
        "",
        "## Findings",
        *[f"- {finding}" for finding in summary.get("plain_language_findings") or []],
        "",
        "## Recommendations",
        *[f"- {finding}" for finding in summary.get("recommendations") or []],
        "",
        "## Limitations",
        *[f"- {finding}" for finding in summary.get("limitations") or summary.get("trial_notes") or []],
        "",
        "## Ranking",
        "| Rank | Configuration | Score | Validity | Detection % | Contact % | Quality | Stability | Ready % | Latency ms | Limitation |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("ranking_table") or []:
        md.append(
            f"| {row.get('rank') or '-'} | {row.get('display_name')} | {row.get('combined_score')} | "
            f"{row.get('result_validity')} | {row.get('detection_success_rate')} | {row.get('contact_valid_rate')} | "
            f"{row.get('average_quality_score')} | {row.get('quality_stability_score')} | "
            f"{row.get('ready_to_close_rate')} | {row.get('average_latency_ms')} | {row.get('main_failure_reason') or ''} |"
        )
    (run_dir / "benchmark_findings.md").write_text("\n".join(md) + "\n")
    rows = "".join(
        "<tr>"
        f"<td>{row.get('rank') or '-'}</td><td>{row.get('display_name')}</td><td>{row.get('combined_score')}</td>"
        f"<td>{row.get('result_validity')}</td><td>{row.get('detection_success_rate')}</td>"
        f"<td>{row.get('contact_valid_rate')}</td><td>{row.get('average_quality_score')}</td>"
        f"<td>{row.get('quality_stability_score')}</td><td>{row.get('ready_to_close_rate')}</td>"
        f"<td>{row.get('average_latency_ms')}</td><td>{row.get('main_failure_reason') or ''}</td>"
        "</tr>"
        for row in summary.get("ranking_table") or []
    )
    findings = "".join(f"<li>{finding}</li>" for finding in summary.get("plain_language_findings") or [])
    recommendations = "".join(f"<li>{finding}</li>" for finding in summary.get("recommendations") or [])
    limitations = "".join(f"<li>{finding}</li>" for finding in (summary.get("limitations") or summary.get("trial_notes") or []))
    winners = summary.get("winner_cards") or {}
    (run_dir / "benchmark_report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Benchmark Report</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:32px auto;line-height:1.4}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.card{border:1px solid #ccc;padding:10px;border-radius:6px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccc;padding:6px;text-align:left}</style></head><body>"
        f"<h1>Benchmark Results Summary</h1><p>Run <code>{manifest.get('benchmark_run_id')}</code>. "
        f"{summary.get('metric_wording')}. Trial quality: <b>{summary.get('trial_quality')}</b>.</p>"
        f"<p>Ground truth: <b>{summary.get('ground_truth_state')}</b>; "
        f"{summary.get('ground_truth_labels_used') or 0} current applicable labels used. "
        f"{summary.get('accuracy_metrics_reason')}</p>"
        f"<p>{summary.get('single_trial_notice')}</p><p>Scoring: {summary.get('scoring_formula')}</p>"
        "<h2>Overall Winner Cards</h2><div class='cards'>"
        f"<div class='card'><b>Best Overall</b><br>{(winners.get('best_overall') or {}).get('display_name') or 'N/A'}</div>"
        f"<div class='card'><b>Best Detection</b><br>{(winners.get('best_bottle_detection') or {}).get('display_name') or 'N/A'}</div>"
        f"<div class='card'><b>Best Quality</b><br>{(winners.get('best_grasp_quality') or {}).get('display_name') or 'N/A'}</div>"
        f"<div class='card'><b>Fastest</b><br>{(winners.get('fastest') or {}).get('display_name') or 'N/A'}</div>"
        f"<div class='card'><b>Strongest Positive Score Association</b><br>{winners.get('strongest_positive_score_association') or 'N/A'}</div>"
        "</div>"
        f"<h2>Findings</h2><ul>{findings}</ul><h2>Recommendations</h2><ul>{recommendations}</ul>"
        f"<h2>Limitations</h2><ul>{limitations}</ul>"
        "<h2>Ranking</h2><table><thead><tr><th>Rank</th><th>Configuration</th><th>Score</th><th>Validity</th>"
        "<th>Detection %</th><th>Contact %</th><th>Quality</th><th>Stability</th><th>Ready %</th><th>Latency ms</th><th>Main Limitation</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )
    return {
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "benchmark_run_manifest.json"),
        "configurations": str(run_dir / "benchmark_configurations.json"),
        "samples": str(run_dir / "benchmark_samples.jsonl"),
        "results_by_config": str(run_dir / "benchmark_results_by_config.json"),
        "summary": str(run_dir / "benchmark_summary.json"),
        "results_csv": str(run_dir / "benchmark_results.csv"),
        "feature_contributions_csv": str(run_dir / "benchmark_feature_contributions.csv"),
        "validity": str(run_dir / "benchmark_validity.json"),
        "timeline": str(run_dir / "benchmark_timeline.json"),
        "ground_truth": (
            str(run_dir / "benchmark_ground_truth.json")
            if ground_truth is not None
            else None
        ),
        "report_html": str(run_dir / "benchmark_report.html"),
        "findings_md": str(run_dir / "benchmark_findings.md"),
        "report_md": str(run_dir / "benchmark_findings.md"),
        "charts_dir": str(run_dir / "charts"),
        "feature_leakage_matrix": str(run_dir / "feature_leakage_matrix.json"),
        **expanded_outputs,
    }


class BenchmarkRunner:
    def __init__(
        self,
        output_root: str | Path,
        sample_callback: Callable[[list[dict]], dict],
        precheck_callback: Callable[[], dict] | None = None,
        settings_path: str | Path | None = None,
    ):
        self.output_root = Path(output_root)
        self.runs_root = self.output_root / "runs"
        self.sample_callback = sample_callback
        self.precheck_callback = precheck_callback
        self.settings_path = settings_path
        self._lock = threading.RLock()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._skip_current = False
        self._repeat_current = False
        self._thread: threading.Thread | None = None
        self._state = self._idle_state()
        self._results: list[dict] = []
        self._timeline: list[dict] = []
        self._manifest: dict = {}
        self._last_outputs: dict = {}
        self._ground_truth: dict | list | None = None

    def _idle_state(self) -> dict:
        return {
            "state": "IDLE",
            "benchmark_run_id": None,
            "experiment_id": None,
            "current_config_id": None,
            "current_config_name": None,
            "current_config_index": 0,
            "total_configs": 0,
            "config_elapsed_sec": 0.0,
            "config_remaining_sec": 0.0,
            "total_elapsed_sec": 0.0,
            "progress_percent": 0.0,
            "current_status_message": "Benchmark runner idle.",
            "current_error": None,
            "run_started_at": None,
            "run_finished_at": None,
            "run_mode": "sequential_live",
            "benchmark_sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
            "benchmark_eval_ms": 0.0,
            "benchmark_log_ms": 0.0,
            "benchmark_ui_update_ms": 0.0,
            "overlay_fps_during_benchmark": None,
            "timeline": [],
            "results_ready": False,
            "outputs": {},
            "summary": {},
        }

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def results(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "manifest": self._manifest,
                "results": self._results,
                "timeline": self._timeline,
                "outputs": self._last_outputs,
                "summary": self._state.get("summary") or {},
            }

    def start(
        self,
        configurations: list[dict] | None = None,
        duration_sec: float | int | None = None,
        mode: str = "sequential_live",
        sample_rate_hz: float | None = None,
        ground_truth: dict | list | None = None,
    ) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {**self.snapshot(), "start_accepted": False, "message": "Benchmark is already running."}
            if mode == "replay_fair":
                return {**self.snapshot(), "start_accepted": False, "message": "Fair replay is unavailable in the live runner. Use a verified recording with the offline component replay and evaluator."}
            settings = load_benchmark_settings(self.settings_path)
            duration = float(duration_sec or settings["duration_sec"] or DEFAULT_CONFIG_DURATION_SEC)
            rate = float(sample_rate_hz or settings["sample_rate_hz"] or DEFAULT_SAMPLE_RATE_HZ)
            mode = mode if mode in RUN_MODES else "sequential_live"
            configs = normalize_run_configurations(configurations, duration_sec=duration)
            run_id = _now_id("benchmark-run")
            experiment_id = _now_id("experiment")
            run_dir = self.runs_root / run_id
            self._pause_event.set()
            self._stop_event.clear()
            self._skip_current = False
            self._repeat_current = False
            self._results = []
            self._timeline = []
            self._ground_truth = (
                json.loads(json.dumps(ground_truth))
                if ground_truth is not None
                else None
            )
            ground_truth_summary = evaluate_ground_truth_labels(
                self._ground_truth,
                declared=self._ground_truth is not None,
            )
            self._manifest = {
                "benchmark_run_id": run_id,
                "experiment_id": experiment_id,
                "run_mode": mode,
                "mode_label": RUN_MODES[mode],
                "duration_sec_per_config": duration,
                "sample_rate_hz": rate,
                "configuration_count": len(configs),
                "production_control": False,
                "benchmark_only": True,
                "full_raw_video_recording": False,
                "created_at": time.time(),
                "run_dir": str(run_dir),
                "ground_truth_state_at_start": ground_truth_summary[
                    "ground_truth_state"
                ],
                "ground_truth_label_count_at_start": ground_truth_summary[
                    "ground_truth_label_count"
                ],
            }
            self._last_outputs = {}
            self._state = {
                **self._idle_state(),
                "state": "PRECHECK",
                "benchmark_run_id": run_id,
                "experiment_id": experiment_id,
                "total_configs": len(configs),
                "run_started_at": time.time(),
                "run_mode": mode,
                "benchmark_sample_rate_hz": rate,
                "current_status_message": "Running benchmark precheck.",
                "outputs": {"run_dir": str(run_dir)},
            }
            self._thread = threading.Thread(
                target=self._run_thread,
                args=(configs, run_dir, settings, mode, rate),
                name="benchmark_runner",
                daemon=True,
            )
            self._thread.start()
            return {**self.snapshot(), "start_accepted": True, "message": "Benchmark run started."}

    def stop(self) -> dict:
        self._stop_event.set()
        with self._lock:
            if self._state.get("state") not in {"IDLE", "COMPLETE", "FAILED", "CANCELLED"}:
                self._state["state"] = "STOPPING"
                self._state["current_status_message"] = "Stopping benchmark run."
        return self.snapshot()

    def pause(self) -> dict:
        self._pause_event.clear()
        with self._lock:
            if self._state.get("state") in {"RUNNING", "CONFIG_RUNNING"}:
                self._state["state"] = "PAUSED"
                self._state["current_status_message"] = "Benchmark paused."
        return self.snapshot()

    def resume(self) -> dict:
        self._pause_event.set()
        with self._lock:
            if self._state.get("state") == "PAUSED":
                self._state["state"] = "CONFIG_RUNNING"
                self._state["current_status_message"] = "Benchmark resumed."
        return self.snapshot()

    def skip_current(self) -> dict:
        with self._lock:
            self._skip_current = True
            self._state["current_status_message"] = "Skipping current configuration."
        return self.snapshot()

    def repeat_current(self) -> dict:
        with self._lock:
            self._repeat_current = True
            self._skip_current = True
            self._state["current_status_message"] = "Repeating current configuration after this cycle."
        return self.snapshot()

    def _set_state(self, **updates) -> None:
        with self._lock:
            self._state.update(updates)

    def _run_thread(self, configs: list[dict], run_dir: Path, settings: dict, mode: str, sample_rate_hz: float) -> None:
        started = time.time()
        samples_path = run_dir / "benchmark_samples.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        samples_path.touch()
        _write_json(run_dir / "benchmark_run_manifest.json", self._manifest)
        _write_json(run_dir / "benchmark_configurations.json", {"configurations": configs})
        try:
            precheck = self.precheck_callback() if self.precheck_callback else {"ok": True, "required": {}, "optional": {}}
            self._manifest["precheck"] = precheck
            _write_json(run_dir / "benchmark_run_manifest.json", self._manifest)
            if not precheck.get("ok", True):
                precheck_reason = precheck.get("reason") or "precheck_failed"
                self._set_state(
                    state="FAILED",
                    current_error=f"benchmark_precheck_failed: {precheck_reason}",
                    current_status_message=f"Cannot run benchmark: {precheck_reason}",
                    run_finished_at=time.time(),
                )
                return
            self._set_state(state="READY", current_status_message="Benchmark precheck passed.")
            time.sleep(0.05)
            i = 0
            while i < len(configs):
                if self._stop_event.is_set():
                    self._finish_cancelled(run_dir, configs, settings)
                    return
                config = dict(configs[i])
                unavailable_reason = None
                unavailable_reason = unavailable_reason or _config_sensor_unavailable_reason(config, precheck)
                if unavailable_reason:
                    result = summarize_samples(config, [], status="unavailable", unavailable_reason=unavailable_reason)
                    self._results.append(result)
                    self._timeline.append({"timestamp": time.time(), "configuration_id": config.get("configuration_id"), "status": "unavailable", "reason": unavailable_reason})
                    self._update_progress(i + 1, len(configs), config, started, config.get("duration_sec") or DEFAULT_CONFIG_DURATION_SEC, 0.0, message=f"{config.get('display_name')} unavailable: {unavailable_reason}")
                    i += 1
                    continue
                config_samples = []
                duration = float(config.get("duration_sec") or DEFAULT_CONFIG_DURATION_SEC)
                config_started = time.time()
                self._timeline.append({"timestamp": config_started, "configuration_id": config.get("configuration_id"), "status": "running"})
                while True:
                    if self._stop_event.is_set():
                        self._finish_cancelled(run_dir, configs, settings)
                        return
                    if self._skip_current:
                        with self._lock:
                            repeat = bool(self._repeat_current)
                            self._skip_current = False
                            self._repeat_current = False
                        status = "repeated" if repeat else "skipped"
                        result = summarize_samples(config, config_samples, status=status)
                        self._results.append(result)
                        self._timeline.append({"timestamp": time.time(), "configuration_id": config.get("configuration_id"), "status": status})
                        if repeat:
                            break
                        i += 1
                        break
                    if not self._pause_event.is_set():
                        self._set_state(state="PAUSED", current_status_message="Benchmark paused.")
                        time.sleep(0.1)
                        continue
                    elapsed = time.time() - config_started
                    if elapsed >= duration:
                        result = summarize_samples(config, config_samples, status="complete")
                        self._results.append(result)
                        self._timeline.append({"timestamp": time.time(), "configuration_id": config.get("configuration_id"), "status": "complete"})
                        i += 1
                        break
                    sample_started = time.perf_counter()
                    status = self.sample_callback([config])
                    card = (status.get("comparison_cards") or [{}])[0]
                    sample = _sample_from_card(card, status, config)
                    sample["benchmark_run_id"] = self._manifest["benchmark_run_id"]
                    sample["config_elapsed_sec"] = round(elapsed, 3)
                    config_samples.append(sample)
                    log_started = time.perf_counter()
                    _append_jsonl(samples_path, sample)
                    log_ms = (time.perf_counter() - log_started) * 1000.0
                    eval_ms = (time.perf_counter() - sample_started) * 1000.0
                    self._update_progress(i, len(configs), config, started, duration, elapsed, eval_ms=eval_ms, log_ms=log_ms, message=f"Running {config.get('display_name')}")
                    interval = max(0.02, 1.0 / max(0.1, sample_rate_hz))
                    time.sleep(max(0.0, interval - (time.perf_counter() - sample_started)))
                if self._results and self._results[-1].get("status") == "repeated":
                    self._results.pop()
                    continue
            self._finish_complete(run_dir, configs, settings, started)
        except Exception as exc:
            self._set_state(
                state="FAILED",
                current_error=str(exc),
                current_status_message=f"Benchmark failed: {exc}",
                run_finished_at=time.time(),
            )
            self._manifest["error"] = str(exc)
            _write_json(run_dir / "benchmark_run_manifest.json", self._manifest)

    def _update_progress(self, idx: int, total: int, config: dict, started: float, duration: float, elapsed: float, eval_ms: float = 0.0, log_ms: float = 0.0, message: str = "") -> None:
        complete_fraction = min(1.0, max(0.0, elapsed / max(0.001, duration)))
        progress = ((idx + complete_fraction) / max(1, total)) * 100.0
        self._set_state(
            state="CONFIG_RUNNING",
            current_config_id=config.get("configuration_id"),
            current_config_name=config.get("display_name"),
            current_config_index=idx + 1,
            total_configs=total,
            config_elapsed_sec=round(elapsed, 2),
            config_remaining_sec=round(max(0.0, duration - elapsed), 2),
            total_elapsed_sec=round(time.time() - started, 2),
            progress_percent=round(min(100.0, progress), 2),
            current_status_message=message,
            benchmark_eval_ms=round(eval_ms, 3),
            benchmark_log_ms=round(log_ms, 3),
            timeline=list(self._timeline[-20:]),
        )

    def _finish_complete(self, run_dir: Path, configs: list[dict], settings: dict, started: float) -> None:
        summary = build_summary(
            self._results,
            configs,
            settings=settings,
            ground_truth_exists=self._ground_truth is not None,
            ground_truth=self._ground_truth,
        )
        self._manifest.update({"finished_at": time.time(), "state": "COMPLETE"})
        outputs = write_run_outputs(
            run_dir,
            self._manifest,
            configs,
            self._results,
            summary,
            self._timeline,
            ground_truth=self._ground_truth,
        )
        with self._lock:
            self._last_outputs = outputs
            self._state.update({
                "state": "COMPLETE",
                "progress_percent": 100.0,
                "current_status_message": "Benchmark complete.",
                "run_finished_at": self._manifest["finished_at"],
                "total_elapsed_sec": round(time.time() - started, 2),
                "results_ready": True,
                "outputs": outputs,
                "summary": summary,
                "timeline": list(self._timeline[-30:]),
            })

    def _finish_cancelled(self, run_dir: Path, configs: list[dict], settings: dict) -> None:
        summary = build_summary(
            self._results,
            configs,
            settings=settings,
            ground_truth_exists=self._ground_truth is not None,
            ground_truth=self._ground_truth,
        )
        self._manifest.update({"finished_at": time.time(), "state": "CANCELLED"})
        outputs = write_run_outputs(
            run_dir,
            self._manifest,
            configs,
            self._results,
            summary,
            self._timeline,
            ground_truth=self._ground_truth,
        )
        with self._lock:
            self._last_outputs = outputs
            self._state.update({
                "state": "CANCELLED",
                "current_status_message": "Benchmark cancelled.",
                "run_finished_at": self._manifest["finished_at"],
                "results_ready": bool(self._results),
                "outputs": outputs,
                "summary": summary,
                "timeline": list(self._timeline[-30:]),
            })

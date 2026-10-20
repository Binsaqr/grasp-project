"""Expanded bottle-only benchmark helpers.

This module is metadata/reporting glue for Benchmark Lab. It does not open
cameras, change production tracking, alter DAVIS scoring, or recompute epsilon.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

from .feature_mask import validate_feature_selection
from .feature_registry import FEATURE_BY_ID, canonical_feature_ids


GROUP_ORDER = ("rgb", "depth", "pointcloud", "davis_aps", "davis_events")

GROUPS: dict[str, dict] = {
    "rgb": {
        "display_name": "RGB",
        "category": "RealSense RGB",
        "features": ("rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle"),
        "expected_limitations": ("No metric distance without depth.",),
    },
    "depth": {
        "display_name": "Depth",
        "category": "RealSense Depth",
        "features": ("depth_bottle_distance", "depth_valid_ratio", "depth_alignment_quality", "depth_object_geometry"),
        "expected_limitations": ("Depth alone has limited object identity.",),
    },
    "pointcloud": {
        "display_name": "Point Cloud",
        "category": "Point Cloud",
        "features": ("pointcloud_scene_geometry", "pointcloud_bottle_geometry", "pointcloud_bottle_extents"),
        "expected_limitations": ("Requires depth, intrinsics, and a valid target ROI/mask.",),
    },
    "davis_aps": {
        "display_name": "DAVIS APS",
        "category": "DAVIS APS",
        "features": ("davis_aps_frame", "davis_aps_target_motion", "davis_aps_angle_change", "davis_aps_lighting_change"),
        "expected_limitations": ("APS is context only and is not a RealSense bottle detector.",),
    },
    "davis_events": {
        "display_name": "DAVIS Events",
        "category": "DAVIS Events",
        "features": (
            "event_rate",
            "event_heatmap",
            "event_motion_activity",
            "event_stability",
            "event_rotation",
            "event_vibration",
            "event_incipient_slip",
            "event_gross_slip",
            "event_impact",
        ),
        "expected_limitations": ("Events alone cannot identify a stationary bottle.",),
    },
}

NOT_APPLICABLE_FEATURES: dict[str, str] = {
    "rgb_gripper_detection": "No gripper/finger test is performed in bottle-only mode.",
    "rgb_contact_visibility": "Gripper contact visibility is out of scope without a contact test.",
    "depth_contact_consistency": "Left/right physical contact depth is out of scope without contact.",
    "depth_grip_width": "Grip width is contact/planned-geometry dependent and is not counted as physical contact.",
    "rgbd_contact_patches": "Physical contact patch confirmation is skipped.",
    "rgbd_ready_to_close": "Ready-to-close after gripper closure is skipped.",
    "rgbd_visual_contact_model": "Validated gripper contact model is skipped.",
    "pointcloud_3d_contact_validation": "3D contact validation is not applicable without a contact test.",
    "visual_contact_quality": "Contact quality gate is skipped without fingers/contact.",
    "analytic_epsilon": "Contact-based epsilon is unavailable without valid contact geometry.",
    "wrench_quality": "Wrench quality depends on contact-based epsilon/contact geometry.",
    "live_operational_quality": "Current live quality baseline depends on contact/epsilon in this pipeline.",
    "multiview_reconstruction_support": "Multi-view scanning is not part of the stationary bottle-only run.",
}

UNSUPPORTED_REQUESTED_FEATURES: dict[str, str] = {}

CONTACT_HEAVY_FEATURES = set(NOT_APPLICABLE_FEATURES)


def _features_for_groups(groups: Iterable[str]) -> tuple[list[str], list[str], list[str], dict]:
    requested: list[str] = []
    for group in groups:
        requested.extend(GROUPS[group]["features"])
    if "rgb" in groups and "depth" in groups:
        requested.append("rgbd_bottle_tracking")
        requested.append("rgbd_object_mask_depth_support")
        requested.append("rgbd_grasp_region_estimate")
    validation = validate_feature_selection(requested, auto_enable=False)
    selected = validation["selected_features"]
    dependency_features = sorted({dep for deps in validation.get("missing_dependencies", {}).values() for dep in deps})
    derived_features = [fid for fid in selected if fid == "rgbd_bottle_tracking" and fid not in requested]
    return canonical_feature_ids(requested), canonical_feature_ids(dependency_features), canonical_feature_ids(derived_features), validation


def generate_group_combinations(duration_sec: float | int | None = None) -> list[dict]:
    """Generate all non-empty RGB/depth/pointcloud/DAVIS group combinations."""
    configs: list[dict] = []
    for size in range(1, len(GROUP_ORDER) + 1):
        for groups_tuple in itertools.combinations(GROUP_ORDER, size):
            groups = list(groups_tuple)
            requested, dependency_features, derived_features, validation = _features_for_groups(groups)
            feature_ids = canonical_feature_ids(requested)
            missing_before_auto = validation.get("missing_dependencies") or {}
            group_label = " + ".join(GROUPS[group]["display_name"] for group in groups)
            config_id = "expanded_" + "_".join(groups)
            limitations: list[str] = []
            for group in groups:
                limitations.extend(GROUPS[group]["expected_limitations"])
            if "pointcloud" in groups:
                limitations.append("Point-cloud configs require CURRENT/AGING point-cloud freshness for fair interpretation.")
            if groups == ["davis_events"]:
                limitations.append("DAVIS Events Only is motion-health context, not a stationary bottle detector.")
            if groups == ["davis_aps"]:
                limitations.append("DAVIS APS Only is visual context, not metric RGB-D geometry.")
            config = {
                "configuration_id": config_id,
                "display_name": group_label,
                "preset_id": "expanded_bottle_only",
                "enabled_groups": groups,
                "feature_ids": feature_ids,
                "enabled_features": feature_ids,
                "requested_features": requested,
                "dependency_features": dependency_features,
                "derived_features": derived_features,
                "missing_dependencies": missing_before_auto,
                "feature_selection_valid": validation["feature_selection_valid"],
                "feature_selection_reason": validation["feature_selection_reason"],
                "not_applicable_features": dict(NOT_APPLICABLE_FEATURES),
                "unsupported_requested_features": dict(UNSUPPORTED_REQUESTED_FEATURES),
                "expected_limitations": list(dict.fromkeys(limitations)),
                "pointcloud_required": "pointcloud" in groups,
                "benchmark_object_mode": "bottle_only",
            }
            if duration_sec is not None:
                config["duration_sec"] = float(duration_sec)
            configs.append(config)
    return configs


def coverage_summary(configs: list[dict], results: list[dict] | None = None) -> dict:
    results_by_id = {row.get("configuration_id"): row for row in (results or [])}
    statuses = {}
    for config in configs:
        row = results_by_id.get(config["configuration_id"]) or {}
        validity = row.get("result_validity") or ("UNTESTED" if not row else "UNKNOWN")
        statuses[config["configuration_id"]] = validity
    return {
        "generated_combinations": len(configs),
        "tested_combinations": sum(1 for cid in statuses if cid in results_by_id),
        "valid": sum(1 for value in statuses.values() if value == "VALID"),
        "valid_with_limitations": sum(1 for value in statuses.values() if value == "VALID_WITH_LIMITATIONS"),
        "unavailable": sum(1 for value in statuses.values() if value == "SENSOR_UNAVAILABLE"),
        "stale": sum(1 for value in statuses.values() if value == "STALE_DATA"),
        "dependency_missing": sum(1 for value in statuses.values() if value == "DEPENDENCY_MISSING"),
        "not_applicable": sum(1 for value in statuses.values() if value == "NOT_APPLICABLE"),
        "unsupported": sum(1 for value in statuses.values() if value == "UNSUPPORTED"),
        "failed_or_invalid": sum(1 for value in statuses.values() if value in {"INVALID", "INVALID_TRIAL"}),
        "untested": sum(1 for value in statuses.values() if value == "UNTESTED"),
        "config_statuses": statuses,
    }


def feature_leakage_matrix(configs: list[dict], samples_by_config: dict[str, list[dict]] | None = None) -> dict:
    """Check that configs only use requested/derived/dependency feature ids.

    The matrix is intentionally based on `enabled_features`/`used_feature_groups`
    metadata instead of raw source-status diagnostics. Raw diagnostics may still
    be present in files for troubleshooting, but they must not be listed as used
    benchmark features when disabled.
    """
    rows = []
    leakage = False
    for config in configs:
        allowed = set(config.get("requested_features") or [])
        allowed.update(config.get("dependency_features") or [])
        allowed.update(config.get("derived_features") or [])
        enabled = set(config.get("feature_ids") or [])
        leaked_features = sorted(enabled - allowed)
        disabled_groups = [group for group in GROUP_ORDER if group not in set(config.get("enabled_groups") or [])]
        sample_leaks: list[str] = []
        for sample in (samples_by_config or {}).get(config["configuration_id"], []):
            used_groups = set(sample.get("used_feature_groups") or [])
            expected_groups = {GROUPS[group]["category"] for group in config.get("enabled_groups") or []}
            if "pointcloud" in config.get("enabled_groups", []):
                expected_groups.update({"RealSense RGB", "RealSense Depth", "RGB-D Combined"})
            if "rgb" in config.get("enabled_groups", []) and "depth" in config.get("enabled_groups", []):
                expected_groups.add("RGB-D Combined")
            sample_leaks.extend(sorted(used_groups - expected_groups))
        row_leak = bool(leaked_features or sample_leaks)
        leakage = leakage or row_leak
        rows.append({
            "configuration_id": config["configuration_id"],
            "enabled_groups": config.get("enabled_groups") or [],
            "disabled_groups": disabled_groups,
            "enabled_features": sorted(enabled),
            "allowed_features": sorted(allowed),
            "leaked_features": leaked_features,
            "sample_group_leaks": sorted(set(sample_leaks)),
            "leakage_detected": row_leak,
        })
    return {
        "leakage_detected": leakage,
        "row_count": len(rows),
        "rows": rows,
    }


def _num(value):
    try:
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def _mean(values):
    clean = [_num(value) for value in values if _num(value) is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def _std(values):
    clean = [_num(value) for value in values if _num(value) is not None]
    if len(clean) > 1:
        return round(statistics.pstdev(clean), 3)
    return 0.0 if clean else None


def _p95(values):
    clean = sorted(_num(value) for value in values if _num(value) is not None)
    if not clean:
        return None
    idx = min(len(clean) - 1, max(0, int(math.ceil(0.95 * len(clean))) - 1))
    return round(clean[idx], 3)


def _eligible(row: dict, *, require_pointcloud_current: bool = False) -> bool:
    if row.get("result_validity") not in {"VALID", "VALID_WITH_LIMITATIONS"}:
        return False
    if require_pointcloud_current and row.get("result_validity") == "VALID_WITH_LIMITATIONS":
        reason = str(row.get("result_validity_reason") or "").lower()
        if "stale" in reason:
            return False
    return True


def summarize_expanded_results(configs: list[dict], results: list[dict], samples: list[dict]) -> dict:
    by_id = {config["configuration_id"]: config for config in configs}
    samples_by_config: dict[str, list[dict]] = {}
    for sample in samples:
        samples_by_config.setdefault(sample.get("configuration_id"), []).append(sample)
    ranking = []
    for result in results:
        cid = result.get("configuration_id")
        config = by_id.get(cid, {})
        config_samples = samples_by_config.get(cid, [])
        ranking.append({
            "configuration_id": cid,
            "display_name": result.get("display_name"),
            "enabled_groups": config.get("enabled_groups") or [],
            "used_features": result.get("used_features") or sorted({fid for sample in config_samples for fid in (sample.get("used_features") or [])}),
            "attempted_features": result.get("attempted_features") or sorted({fid for sample in config_samples for fid in (sample.get("attempted_features") or [])}),
            "missing_dependencies": result.get("missing_dependencies") or {},
            "not_applicable_features": result.get("not_applicable_features") or {},
            "blocked_features": result.get("blocked_features") or {},
            "production_fields_used": result.get("production_fields_used") or [],
            "feature_leakage_detected": bool(result.get("feature_leakage_detected") or any(sample.get("feature_leakage_detected") for sample in config_samples)),
            "result_validity": result.get("result_validity"),
            "result_validity_reason": result.get("result_validity_reason"),
            "sample_count": result.get("sample_count"),
            "detection_success_rate": result.get("detection_success_rate"),
            "detection_confidence_mean": result.get("detection_confidence_mean"),
            "detection_confidence_std": result.get("detection_confidence_std"),
            "target_loss_count": result.get("target_loss_count"),
            "target_loss_rate": result.get("target_loss_rate"),
            "center_jitter_px": result.get("center_jitter_px"),
            "angle_jitter_deg": result.get("angle_jitter_deg"),
            "mask_stability_iou": result.get("mask_stability_iou"),
            "detection_latency_ms_mean": result.get("detection_latency_ms_mean") or result.get("average_latency_ms"),
            "detection_latency_ms_p95": result.get("detection_latency_ms_p95") or result.get("p95_latency_ms"),
            "distance_mean_mm": _mean(sample.get("distance_mm") for sample in config_samples),
            "distance_std_mm": _std(sample.get("distance_mm") for sample in config_samples),
            "depth_valid_ratio_mean": _mean(sample.get("depth_valid_ratio") for sample in config_samples),
            "pointcloud_valid_rate": round(100.0 * sum(bool(sample.get("pointcloud_valid")) for sample in config_samples) / len(config_samples), 3) if config_samples else None,
            "pointcloud_freshness_rate": round(100.0 * sum(str(sample.get("pointcloud_freshness") or "").upper() in {"CURRENT", "AGING"} for sample in config_samples) / len(config_samples), 3) if config_samples else None,
            "aps_fps": _mean(sample.get("aps_fps") for sample in config_samples),
            "aps_mode": next((sample.get("aps_state") for sample in config_samples if sample.get("aps_state")), None),
            "event_rate_mean": _mean(sample.get("davis_event_rate") for sample in config_samples),
            "event_rate_peak": max((_num(sample.get("davis_event_rate")) for sample in config_samples if _num(sample.get("davis_event_rate")) is not None), default=None),
            "motion_class_distribution": {
                label: sum(str(sample.get("davis_motion_class")) == label for sample in config_samples)
                for label in sorted({str(sample.get("davis_motion_class")) for sample in config_samples if sample.get("davis_motion_class")})
            },
            "stability_score_mean": _mean(sample.get("stability_score") for sample in config_samples),
            "epsilon_available": any(bool(sample.get("epsilon_available")) for sample in config_samples),
            "quality_result_validity": result.get("quality_result_validity") or "NOT_APPLICABLE_WITHOUT_CONTACT_GEOMETRY",
            "mean_latency_ms": result.get("mean_latency_ms") or result.get("average_latency_ms"),
            "p95_latency_ms": result.get("p95_latency_ms"),
            "effective_fps": result.get("effective_fps"),
            "stale_sample_count": result.get("stale_sample_count"),
            "unavailable_sample_count": result.get("invalid_sample_count"),
            "main_limitation": result.get("main_failure_reason") or result.get("plain_language_explanation"),
        })

    valid = [row for row in ranking if _eligible(row)]
    rgb_only = [row for row in valid if row["enabled_groups"] == ["rgb"]]
    depth_only = [row for row in valid if row["enabled_groups"] == ["depth"]]
    # a stale point cloud must not win best_pointcloud: apply the strict
    # point-cloud-current guard (previously defined but never used, so a
    # minority-stale VALID_WITH_LIMITATIONS config could still win)
    pc_rows = [row for row in ranking if "pointcloud" in row["enabled_groups"] and _eligible(row, require_pointcloud_current=True)]
    aps_rows = [row for row in valid if row["enabled_groups"] == ["davis_aps"]]
    event_rows = [row for row in valid if row["enabled_groups"] == ["davis_events"]]
    pairs = [row for row in valid if len(row["enabled_groups"]) == 2]
    triples = [row for row in valid if len(row["enabled_groups"]) == 3]
    full = [row for row in valid if row["enabled_groups"] == list(GROUP_ORDER)]

    def detection_key(row):
        return ((row.get("detection_success_rate") or 0), (row.get("detection_confidence_mean") or 0), -(row.get("p95_latency_ms") or 99999))

    def geometry_key(row):
        return ((row.get("pointcloud_valid_rate") or 0), (row.get("detection_success_rate") or 0), -(row.get("p95_latency_ms") or 99999))

    def fastest_key(row):
        return row.get("mean_latency_ms") if row.get("mean_latency_ms") is not None else 99999

    def overall_key(row):
        stale_penalty = 15.0 if row.get("result_validity") == "VALID_WITH_LIMITATIONS" and "stale" in str(row.get("result_validity_reason") or "").lower() else 0.0
        return (
            (row.get("detection_success_rate") or 0) * 0.45
            + (row.get("detection_confidence_mean") or 0) * 0.25
            + max(0.0, 100.0 - (row.get("p95_latency_ms") or 100.0)) * 0.15
            + (row.get("pointcloud_valid_rate") or 0) * 0.15
            - stale_penalty
        )

    winners = {
        "best_rgb_only": max(rgb_only, key=detection_key, default=None),
        "best_depth_only": max(depth_only, key=detection_key, default=None),
        "best_pointcloud": max(pc_rows, key=geometry_key, default=None),
        "best_davis_aps": max(aps_rows, key=lambda row: row.get("aps_fps") or 0, default=None),
        "best_davis_events": max(event_rows, key=lambda row: row.get("event_rate_mean") or 0, default=None),
        "best_pair": max(pairs, key=overall_key, default=None),
        "best_triple": max(triples, key=overall_key, default=None),
        "best_full_multimodal": max(full, key=overall_key, default=None),
        "fastest": min(valid, key=fastest_key, default=None),
        "recommended_bottle_only_setup": max(valid, key=overall_key, default=None),
    }
    for row in ranking:
        score = round(overall_key(row), 3)
        row["weighted_score"] = score
        row["combined_score"] = score
        row["winner_eligible"] = _eligible(row)
    return {
        "coverage": coverage_summary(configs, results),
        "leakage": feature_leakage_matrix(configs, samples_by_config),
        "ranking_table": ranking,
        "winners": winners,
        "plain_language_answers": build_plain_language_answers(winners, ranking),
    }


def build_plain_language_answers(winners: dict, ranking: list[dict]) -> list[str]:
    answers = []
    rgb = winners.get("best_rgb_only")
    depth = winners.get("best_depth_only")
    pc = winners.get("best_pointcloud")
    events = winners.get("best_davis_events")
    if rgb:
        answers.append(f"RGB alone detected the bottle at {rgb.get('detection_success_rate')}% using {len(rgb.get('used_features') or [])} enabled RGB features; mean confidence {rgb.get('detection_confidence_mean')}.")
    if depth:
        answers.append(f"Depth Only result was {depth.get('result_validity')}: bottle identity is limited without RGB; depth validity metric was {depth.get('depth_valid_ratio_mean')}.")
    rgb_depth = next((row for row in ranking if row.get("enabled_groups") == ["rgb", "depth"]), None)
    if rgb_depth:
        answers.append(f"RGB + Depth detection was {rgb_depth.get('detection_success_rate')}% with mean latency {rgb_depth.get('mean_latency_ms')} ms.")
    if pc:
        freshness = pc.get("pointcloud_freshness_rate")
        answers.append(f"Point cloud geometry was interpreted only when dependencies were available; best point-cloud row had freshness {freshness}% and validity {pc.get('result_validity')}.")
    aps = winners.get("best_davis_aps")
    if aps:
        answers.append(f"DAVIS APS was available as visual context with mean FPS {aps.get('aps_fps')}.")
    if events:
        answers.append(f"DAVIS Events Only result was {events.get('result_validity')}; bottle detection remained NOT_APPLICABLE while motion/stability fields were evaluated.")
    if winners.get("best_pair"):
        answers.append(f"Strongest pair: {winners['best_pair'].get('configuration_id')}.")
    if winners.get("best_full_multimodal"):
        answers.append(f"Full multimodal result: {winners['best_full_multimodal'].get('result_validity')}.")
    if winners.get("recommended_bottle_only_setup"):
        answers.append(f"Recommended bottle-only setup from this run: {winners['recommended_bottle_only_setup'].get('configuration_id')}.")
    return answers


def write_group_combination_matrix_chart(path: str | Path, configs: list[dict], results: list[dict] | None = None) -> str:
    """Write a compact PNG matrix of group coverage/result status."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_by_id = {row.get("configuration_id"): row for row in (results or [])}
    cell_w, cell_h = 112, 28
    left_w, top_h = 300, 56
    width = left_w + cell_w * len(GROUP_ORDER)
    height = top_h + cell_h * len(configs) + 40
    try:
        import cv2
        import numpy as np

        img = np.full((height, width, 3), 255, dtype=np.uint8)
        colors = {
            "VALID": (92, 168, 85),
            "VALID_WITH_LIMITATIONS": (60, 159, 212),
            "STALE_DATA": (63, 148, 245),
            "SENSOR_UNAVAILABLE": (82, 82, 210),
            "DEPENDENCY_MISSING": (60, 165, 230),
            "UNTESTED": (180, 180, 180),
        }
        cv2.putText(img, "Expanded Bottle-Only Group Combination Matrix", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        for j, group in enumerate(GROUP_ORDER):
            cv2.putText(img, GROUPS[group]["display_name"][:14], (left_w + j * cell_w + 5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        for i, config in enumerate(configs):
            y = top_h + i * cell_h
            row = results_by_id.get(config["configuration_id"]) or {}
            validity = row.get("result_validity") or "UNTESTED"
            cv2.putText(img, config["configuration_id"][:38], (10, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
            for j, group in enumerate(GROUP_ORDER):
                x = left_w + j * cell_w
                enabled = group in set(config.get("enabled_groups") or [])
                color = colors.get(validity, (160, 160, 160)) if enabled else (235, 235, 235)
                cv2.rectangle(img, (x + 3, y + 4), (x + cell_w - 4, y + cell_h - 4), color, -1)
                cv2.rectangle(img, (x + 3, y + 4), (x + cell_w - 4, y + cell_h - 4), (200, 200, 200), 1)
                if enabled:
                    cv2.putText(img, "ON", (x + 38, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, validity[:18], (left_w + cell_w * len(GROUP_ORDER) - 135, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), img)
    except Exception:
        path.write_bytes(b"")
    return str(path)


def _write_table_pair(path_md: Path, path_csv: Path, rows: list[dict], columns: list[str]) -> None:
    path_md.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path_md.write_text("\n".join(lines) + "\n")


def write_report_ready_tables(run_dir: str | Path, ranking: list[dict]) -> dict:
    """Write report-ready Markdown and CSV comparison tables."""
    run_dir = Path(run_dir)
    table_dir = run_dir / "tables"

    def report_score(row: dict) -> float:
        value = row.get("overall_score") or row.get("weighted_score") or row.get("combined_score")
        return float(value) if _num(value) is not None else -1.0

    def report_eligible(row: dict) -> bool:
        if "winner_eligible" in row:
            return bool(row.get("winner_eligible"))
        return _eligible(row)

    ranked_source = sorted(ranking, key=lambda row: (report_eligible(row), report_score(row)), reverse=True)
    ranked = [dict(row, rank=i + 1) for i, row in enumerate(ranked_source)]
    detection_rows = [
        {
            "Rank": row["rank"],
            "Configuration": row.get("display_name"),
            "Enabled Groups": ", ".join(row.get("enabled_groups") or []),
            "Used Features": ", ".join((row.get("used_features") or [])[:8]),
            "Result Validity": row.get("result_validity"),
            "Bottle Detection Rate": row.get("detection_success_rate"),
            "Detection Confidence Mean": row.get("detection_confidence_mean"),
            "Target Loss Rate": row.get("target_loss_rate"),
            "Center Jitter px": row.get("center_jitter_px"),
            "Angle Jitter deg": row.get("angle_jitter_deg"),
            "Detection Latency ms": row.get("detection_latency_ms_mean"),
            "Main Limitation": row.get("main_limitation") or row.get("result_validity_reason"),
        }
        for row in ranked
    ]
    geometry_rows = [
        {
            "Rank": row["rank"],
            "Configuration": row.get("display_name"),
            "Distance Available": row.get("distance_mean_mm") is not None,
            "Depth Validity %": row.get("depth_valid_ratio_mean"),
            "Point Cloud Validity %": row.get("pointcloud_valid_rate"),
            "Geometry Quality": row.get("geometry_quality_mean") or row.get("pointcloud_valid_rate"),
            "Object Dimensions Available": bool(row.get("pointcloud_valid_rate")),
            "Grip Width Estimate Available": "pointcloud_grip_width_3d" in (row.get("used_features") or []),
            "Quality Stability": row.get("quality_result_validity"),
            "Not Applicable Features": ", ".join((row.get("not_applicable_features") or {}).keys()),
            "Main Limitation": row.get("main_limitation") or row.get("result_validity_reason"),
        }
        for row in ranked
    ]
    motion_rows = [
        {
            "Rank": row["rank"],
            "Configuration": row.get("display_name"),
            "APS Available": row.get("aps_mode") is not None,
            "APS Mode": row.get("aps_mode"),
            "APS FPS": row.get("aps_fps"),
            "Events Available": row.get("event_rate_mean") is not None or "davis_events" in (row.get("enabled_groups") or []),
            "Event Rate Mean": row.get("event_rate_mean"),
            "Event Rate Peak": row.get("event_rate_peak"),
            "Motion Class Distribution": row.get("motion_class_distribution"),
            "Stability Score": row.get("stability_score_mean") or row.get("stability_mean"),
            "Rotation/Vibration Warnings": row.get("motion_warnings") or "see benchmark_results.csv",
            "Main Limitation": row.get("main_limitation") or row.get("result_validity_reason"),
        }
        for row in ranked
    ]
    overall_rows = [
        {
            "Rank": row["rank"],
            "Configuration": row.get("display_name"),
            "Detection Score": row.get("detection_success_rate"),
            "Geometry Score": row.get("pointcloud_valid_rate") or row.get("depth_valid_ratio_mean"),
            "Motion Score": row.get("stability_score_mean"),
            "Latency Score": row.get("mean_latency_ms"),
            "Reliability Score": row.get("result_validity"),
            "Weighted Overall Score": row.get("overall_score") or row.get("weighted_score") or row.get("combined_score"),
            "Recommendation": row.get("main_limitation") or row.get("result_validity_reason"),
        }
        for row in ranked
    ]
    specs = {
        "detection_comparison": (detection_rows, list(detection_rows[0].keys()) if detection_rows else ["Rank"]),
        "geometry_comparison": (geometry_rows, list(geometry_rows[0].keys()) if geometry_rows else ["Rank"]),
        "davis_motion_comparison": (motion_rows, list(motion_rows[0].keys()) if motion_rows else ["Rank"]),
        "overall_recommendation": (overall_rows, list(overall_rows[0].keys()) if overall_rows else ["Rank"]),
    }
    outputs = {}
    for name, (rows, columns) in specs.items():
        md = table_dir / f"{name}.md"
        csv_path = table_dir / f"{name}.csv"
        _write_table_pair(md, csv_path, rows, columns)
        outputs[f"{name}_md"] = str(md)
        outputs[f"{name}_csv"] = str(csv_path)
    return outputs


def write_expanded_report_outputs(run_dir: str | Path, configs: list[dict], results: list[dict], samples: list[dict]) -> dict:
    """Write expanded bottle-only coverage/leakage/report artifacts."""
    run_dir = Path(run_dir)
    summary = summarize_expanded_results(configs, results, samples)
    chart = write_group_combination_matrix_chart(run_dir / "charts" / "group_combination_matrix.png", configs, results)
    table_outputs = write_report_ready_tables(run_dir, summary["ranking_table"])
    payload = {
        "benchmark_object_mode": "bottle_only",
        "result_type": "estimated_performance",
        "has_ground_truth": False,
        "gripper_tests_skipped": True,
        "contact_tests_skipped": True,
        "not_applicable_features": NOT_APPLICABLE_FEATURES,
        "unsupported_requested_features": UNSUPPORTED_REQUESTED_FEATURES,
        **summary,
        "chart_group_combination_matrix": chart,
        "report_ready_tables": table_outputs,
    }
    (run_dir / "expanded_bottle_only_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with (run_dir / "expanded_bottle_only_ranking.csv").open("w", newline="") as handle:
        fieldnames = [
            "configuration_id",
            "enabled_groups",
            "result_validity",
            "detection_success_rate",
            "detection_confidence_mean",
            "pointcloud_valid_rate",
            "pointcloud_freshness_rate",
            "event_rate_mean",
            "mean_latency_ms",
            "p95_latency_ms",
            "quality_result_validity",
            "used_features",
            "missing_dependencies",
            "not_applicable_features",
            "feature_leakage_detected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["ranking_table"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    report_md = run_dir / "expanded_bottle_only_report.md"
    lines = [
        "# Expanded Bottle-Only Benchmark Report",
        "",
        f"- Generated combinations: {summary['coverage']['generated_combinations']}",
        f"- Tested combinations: {summary['coverage']['tested_combinations']}",
        f"- Valid: {summary['coverage']['valid']}",
        f"- Valid with limitations: {summary['coverage']['valid_with_limitations']}",
        f"- Stale: {summary['coverage']['stale']}",
        f"- Unavailable: {summary['coverage']['unavailable']}",
        f"- Feature leakage detected: {summary['leakage']['leakage_detected']}",
        "",
        "## Answers",
        "",
    ]
    lines.extend(f"- {answer}" for answer in summary["plain_language_answers"])
    lines.extend(["", "## Winners", ""])
    for key, value in summary["winners"].items():
        if isinstance(value, dict):
            lines.append(f"- {key}: `{value.get('configuration_id')}` ({value.get('result_validity')})")
        else:
            lines.append(f"- {key}: N/A")
    lines.extend(["", "## Not Applicable", ""])
    lines.extend(f"- `{feature}`: {reason}" for feature, reason in NOT_APPLICABLE_FEATURES.items())
    lines.extend(["", "## Unsupported Requested Features", ""])
    lines.extend(f"- `{feature}`: {reason}" for feature, reason in UNSUPPORTED_REQUESTED_FEATURES.items())
    report_md.write_text("\n".join(lines) + "\n")
    report_html = run_dir / "expanded_bottle_only_report.html"
    report_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Expanded Bottle-Only Benchmark</title></head>"
        "<body><h1>Expanded Bottle-Only Benchmark Coverage Summary</h1>"
        f"<p>Generated combinations: {summary['coverage']['generated_combinations']}</p>"
        f"<p>Tested combinations: {summary['coverage']['tested_combinations']}</p>"
        f"<p>Feature leakage detected: {summary['leakage']['leakage_detected']}</p>"
        "<h2>Group Combination Matrix</h2>"
        "<img src='charts/group_combination_matrix.png' style='max-width:100%;height:auto'>"
        "</body></html>"
    )
    return {
        "expanded_summary": str(run_dir / "expanded_bottle_only_summary.json"),
        "expanded_ranking_csv": str(run_dir / "expanded_bottle_only_ranking.csv"),
        "expanded_report_md": str(report_md),
        "expanded_report_html": str(report_html),
        "group_combination_matrix": chart,
        **table_outputs,
    }

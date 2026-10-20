"""Camera-free tests for expanded bottle-only benchmark coverage."""

from __future__ import annotations

import json
from pathlib import Path

from grasp_project.benchmarking.benchmark_runner import BenchmarkRunner
from grasp_project.benchmarking.bottle_only import (
    NOT_APPLICABLE_FEATURES,
    UNSUPPORTED_REQUESTED_FEATURES,
    coverage_summary,
    feature_leakage_matrix,
    generate_group_combinations,
    write_expanded_report_outputs,
)
from grasp_project.benchmarking.feature_mask import build_benchmark_lab_status


def _sources():
    realsense = {
        "target_status": "TRACKING",
        "tracking_confidence": 91.0,
        "decision": "ADJUST",
        "recommended_action": "MOVE_TO_PRE_CLOSE_POSE",
        "ready_to_close": False,
        "processed_overlay_timing": {"total_processed_overlay_ms": 18.0},
        "realsense_perception": {
            "bottle_detection_status": "TRACKING",
            "bottle_bbox_xyxy": [10, 20, 90, 180],
            "bottle_center_px": [50, 100],
            "bottle_angle_deg": 3.0,
        },
        "realsense_rgb_active": True,
        "realsense_depth_active": True,
        "depth_valid_ratio": 0.8,
        "frame_age_ms": 8.0,
    }
    davis = {
        "event_state": "STREAM_ACTIVE_EVENTS",
        "event_rate": 1000.0,
        "motion_class": "LIGHTING CHANGE",
        "stability": 82.0,
        "rotation": 3.0,
        "vibration": 4.0,
        "aps_state": "APS_COLOR_VERIFIED",
        "aps_fps": 10.0,
    }
    pointcloud = {
        "active": True,
        "freshness": "CURRENT",
        "bottle_pointcloud_valid": True,
        "scene_raw_point_count": 5000,
        "bottle_filtered_point_count": 700,
        "bottle_pointcloud_roi_source": "body_contour",
    }
    sync = {"last_bundle_id": "bundle-1", "sync_quality": "approximate", "max_sensor_time_offset_ms": 6.0}
    quality = {"quality_label": "INVALID", "contact_quality_gate_pass": False}
    live_quality = {"live_quality_score": None, "live_quality_label": "INVALID"}
    return realsense, davis, pointcloud, sync, quality, live_quality


def _status(configurations):
    realsense, davis, pointcloud, sync, quality, live_quality = _sources()
    status = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
        configurations=configurations,
    )
    status["source_status"] = {
        "realsense": realsense,
        "davis": davis,
        "pointcloud": pointcloud,
        "sync": sync,
        "quality": quality,
        "live_quality": live_quality,
        "sensor_availability": {
            "pointcloud_active": True,
            "davis_aps_active": True,
            "davis_events_active": True,
        },
    }
    return status


def _precheck():
    return {
        "ok": True,
        "reason": "ok",
        "required": {"realsense_active": {"ok": True, "reason": "ok"}},
        "optional": {
            "pointcloud_active": {"ok": True, "reason": "ok"},
            "davis_aps_active": {"ok": True, "reason": "ok"},
            "davis_events_active": {"ok": True, "reason": "ok"},
        },
    }


def test_generate_31_group_combinations_with_stable_ids():
    configs = generate_group_combinations(duration_sec=3)

    assert len(configs) == 31
    assert len({config["configuration_id"] for config in configs}) == 31
    assert configs[0]["configuration_id"] == "expanded_rgb"
    assert configs[-1]["configuration_id"] == "expanded_rgb_depth_pointcloud_davis_aps_davis_events"
    assert all(config["display_name"] for config in configs)
    assert all(config["benchmark_object_mode"] == "bottle_only" for config in configs)


def test_dependencies_and_pointcloud_support_are_reported():
    pc = next(config for config in generate_group_combinations() if config["enabled_groups"] == ["pointcloud"])

    assert pc["pointcloud_required"] is True
    assert "pointcloud_bottle_geometry" in pc["requested_features"]
    assert "rgbd_bottle_tracking" in pc["dependency_features"]
    assert "depth_valid_ratio" in pc["dependency_features"]
    assert pc["missing_dependencies"]
    assert pc["feature_selection_valid"] is False


def test_contact_and_epsilon_features_not_applicable_for_bottle_only():
    configs = generate_group_combinations()

    for feature in ["analytic_epsilon", "wrench_quality", "rgb_gripper_detection", "rgbd_contact_patches"]:
        assert feature in NOT_APPLICABLE_FEATURES
    assert all("analytic_epsilon" not in config["feature_ids"] for config in configs)
    assert all("rgb_gripper_detection" not in config["feature_ids"] for config in configs)


def test_requested_features_are_registered_or_explicitly_not_applicable():
    assert UNSUPPORTED_REQUESTED_FEATURES == {}
    configs = generate_group_combinations()
    requested = {feature for config in configs for feature in config["requested_features"]}

    for feature in requested:
        assert feature not in NOT_APPLICABLE_FEATURES


def test_events_only_cannot_claim_stationary_bottle_identity():
    config = next(config for config in generate_group_combinations() if config["enabled_groups"] == ["davis_events"])
    card = _status([config])["comparison_cards"][0]

    assert card["bottle_detected"] is False
    assert "event" in card["detection_status"]


def test_feature_leakage_matrix_passes_for_generated_configs():
    configs = generate_group_combinations()
    matrix = feature_leakage_matrix(configs)

    assert matrix["row_count"] == 31
    assert matrix["leakage_detected"] is False


def test_masked_runner_samples_do_not_leak_davis_into_rgb_only(tmp_path: Path):
    rgb = next(config for config in generate_group_combinations(duration_sec=0.05) if config["enabled_groups"] == ["rgb"])
    events = next(config for config in generate_group_combinations(duration_sec=0.05) if config["enabled_groups"] == ["davis_events"])
    runner = BenchmarkRunner(tmp_path, sample_callback=_status, precheck_callback=_precheck)

    runner.start([rgb, events], duration_sec=0.05, sample_rate_hz=20)
    deadline = __import__("time").time() + 5
    while __import__("time").time() < deadline:
        if runner.snapshot()["state"] in {"COMPLETE", "FAILED", "CANCELLED"}:
            break
        __import__("time").sleep(0.02)
    assert runner.snapshot()["state"] == "COMPLETE"

    samples = [json.loads(line) for line in Path(runner.results()["outputs"]["samples"]).read_text().splitlines()]
    rgb_sample = next(sample for sample in samples if sample["configuration_id"] == "expanded_rgb")
    event_sample = next(sample for sample in samples if sample["configuration_id"] == "expanded_davis_events")
    assert rgb_sample["davis_event_rate"] is None
    assert rgb_sample["aps_state"] is None
    assert not set(rgb_sample["used_features"]) & {"event_rate", "event_stability", "davis_aps_frame"}
    assert event_sample["davis_event_rate"] == 1000.0


def test_coverage_and_expanded_report_outputs(tmp_path: Path):
    configs = generate_group_combinations()
    results = [
        {
            "configuration_id": config["configuration_id"],
            "display_name": config["display_name"],
            "result_validity": "VALID",
            "sample_count": 2,
            "detection_success_rate": 100.0 if "rgb" in config["enabled_groups"] else 0.0,
            "detection_confidence_mean": 90.0 if "rgb" in config["enabled_groups"] else None,
            "mean_latency_ms": 20.0,
            "p95_latency_ms": 25.0,
        }
        for config in configs
    ]
    samples = [
        {
            "configuration_id": config["configuration_id"],
            "bottle_detected": "rgb" in config["enabled_groups"],
            "detection_confidence": 90.0 if "rgb" in config["enabled_groups"] else None,
            "pointcloud_valid": "pointcloud" in config["enabled_groups"],
            "pointcloud_freshness": "CURRENT" if "pointcloud" in config["enabled_groups"] else None,
            "used_feature_groups": [],
            "latency_ms": 20.0,
        }
        for config in configs
    ]

    coverage = coverage_summary(configs, results)
    outputs = write_expanded_report_outputs(tmp_path, configs, results, samples)

    assert coverage["generated_combinations"] == 31
    assert coverage["tested_combinations"] == 31
    assert coverage["valid"] == 31
    assert Path(outputs["group_combination_matrix"]).exists()
    assert Path(outputs["expanded_summary"]).exists()
    assert Path(outputs["detection_comparison_md"]).exists()
    assert Path(outputs["geometry_comparison_md"]).exists()
    assert Path(outputs["davis_motion_comparison_md"]).exists()
    assert Path(outputs["overall_recommendation_md"]).exists()
    assert Path(outputs["detection_comparison_csv"]).exists()
    assert Path(outputs["geometry_comparison_csv"]).exists()
    assert Path(outputs["davis_motion_comparison_csv"]).exists()
    assert Path(outputs["overall_recommendation_csv"]).exists()
    assert "Expanded Bottle-Only Benchmark Coverage Summary" in Path(outputs["expanded_report_html"]).read_text()


def test_stale_pointcloud_cannot_win_best_pointcloud():
    from grasp_project.benchmarking.bottle_only import _eligible

    stale_pc = {"result_validity": "VALID_WITH_LIMITATIONS",
                "result_validity_reason": "3/100 stale point-cloud samples excluded"}
    current_pc = {"result_validity": "VALID_WITH_LIMITATIONS",
                  "result_validity_reason": "some enabled features were blocked or unavailable"}
    # strict guard (used for best_pointcloud) rejects the stale one
    assert _eligible(stale_pc, require_pointcloud_current=True) is False
    assert _eligible(current_pc, require_pointcloud_current=True) is True
    # the default guard still accepts both (used for non-pointcloud winners)
    assert _eligible(stale_pc) is True

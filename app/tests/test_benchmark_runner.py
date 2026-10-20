"""Camera-free tests for the one-click Benchmark Lab runner."""

from __future__ import annotations

import time
from pathlib import Path

from grasp_project.benchmarking.benchmark_runner import (
    BenchmarkRunner,
    build_summary,
    feature_contribution_summary,
    summarize_samples,
)
from grasp_project.benchmarking.feature_mask import build_benchmark_lab_status, validate_feature_selection
from grasp_project.benchmarking.feature_registry import default_run_configurations, preset_feature_ids


def test_live_runner_cannot_mislabel_sequential_sampling_as_fair_replay(tmp_path):
    calls = []
    runner = BenchmarkRunner(tmp_path, sample_callback=lambda configs: calls.append(configs))
    result = runner.start(mode="replay_fair")
    assert result["start_accepted"] is False
    assert result["state"] == "IDLE"
    assert result["run_mode"] == "sequential_live"
    assert "verified recording" in result["message"]
    assert not calls
    assert not (tmp_path / "runs").exists()


def _sources():
    realsense = {
        "frame_id": 7,
        "decision": "ADJUST",
        "recommended_action": "HOLD_AND_RECHECK",
        "ready_to_close": False,
        "writer_session_id": "writer-a",
        "target_status": "TRACKING",
        "tracking_confidence": 87.0,
        "processed_overlay_timing": {"total_processed_overlay_ms": 16.0},
        "realsense_perception": {
            "bottle_detection_status": "TRACKING",
            "bottle_bbox_xyxy": [10, 20, 80, 150],
            "bottle_center_px": [45, 86],
            "bottle_angle_deg": 8.0,
            "contact_pair_valid": True,
            "contact_source_confidence": 81.0,
            "detected_contact_count": 2,
            "contact_depth_consistent": True,
        },
    }
    quality = {
        "quality_label": "GOOD",
        "quality_label_display": "GOOD",
        "epsilon_score": 0.4,
        "normalized_quality_score": 84.0,
        "contact_quality_gate_pass": True,
        "contact_source": "detected_finger_contacts",
        "image_bottle_hull_px": [[0, 0], [1, 0], [1, 1]],
    }
    davis = {
        "event_state": "STREAM_ACTIVE_EVENTS",
        "event_rate": 900.0,
        "decision": "READY_TO_LIFT",
        "dx_px": 3.0,
        "dy_px": 1.0,
        "displacement_px": 3.2,
        "cumulative_slip_px": 5.5,
        "downward_slip_px": 1.0,
        "rotation_delta_deg": 2.0,
        "cumulative_rotation_deg": 3.0,
        "angular_velocity_deg_s": 20.0,
        "vibration_amplitude_px_rms": 0.8,
        "motion_estimate_valid": True,
        "motion_estimation_confidence": 0.75,
        "disturbance_type": "OBJECT_MOTION",
        "stability": 88.0,
        "rotation": 3.0,
        "vibration": 4.0,
    }
    pointcloud = {
        "active": True,
        "freshness": "CURRENT",
        "bottle_pointcloud_valid": True,
        "contact_3d_quality": "valid",
        "contact_points_on_bottle_cloud": True,
        "scene_raw_point_count": 4000,
        "bottle_filtered_point_count": 550,
    }
    sync = {"last_bundle_id": "bundle-7", "sync_quality": "approximate", "max_sensor_time_offset_ms": 8.0}
    live_quality = {"live_quality_score": 78.0, "live_quality_label": "OK"}
    return realsense, davis, pointcloud, sync, quality, live_quality


def _sample_status(configurations):
    realsense, davis, pointcloud, sync, quality, live_quality = _sources()
    realsense.update({
        "realsense_rgb_active": True,
        "realsense_depth_active": True,
        "bottle_distance_mm": 300.0,
        "depth_valid_ratio": 0.82,
    })
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
        "davis": davis,
        "pointcloud": pointcloud,
        "sync": sync,
        "sensor_availability": {
            "pointcloud_active": True,
            "davis_aps_active": True,
            "davis_events_active": True,
        },
    }
    return status


def _precheck(ok=True, pointcloud=True, davis_aps=True, davis_events=True):
    return {
        "ok": ok,
        "reason": "ok" if ok else "realsense_inactive",
        "required": {"realsense_active": {"ok": ok, "reason": "ok" if ok else "realsense_inactive"}},
        "optional": {
            "pointcloud_active": {"ok": pointcloud, "reason": "ok" if pointcloud else "pointcloud_unavailable_or_stale"},
            "davis_aps_active": {"ok": davis_aps, "reason": "ok" if davis_aps else "davis_aps_unavailable"},
            "davis_events_active": {"ok": davis_events, "reason": "ok" if davis_events else "davis_events_unavailable"},
        },
    }


def _wait_done(runner: BenchmarkRunner, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = runner.snapshot()["state"]
        if state in {"COMPLETE", "FAILED", "CANCELLED"}:
            return state
        time.sleep(0.02)
    raise AssertionError(f"runner did not finish: {runner.snapshot()}")


def test_default_run_configurations_cover_requested_sequence_and_validate():
    configs = default_run_configurations(duration_sec=5)
    assert [c["configuration_id"] for c in configs] == [
        "rgb_only",
        "depth_only",
        "rgb_depth",
        "pointcloud_only",
        "rgbd_pointcloud",
        "davis_aps_only",
        "davis_events_only",
        "aps_events",
        "rgbd_davis_events",
        "rgbd_pointcloud_davis_events",
        "full_multimodal",
    ]
    assert all(validate_feature_selection(c["feature_ids"])["feature_selection_valid"] for c in configs)
    assert configs[7]["display_name"] == "DAVIS APS + Events"


def test_feature_contributions_exclude_ineligible_rows_and_do_not_guess_latency():
    configurations = [
        {"configuration_id": "rgb", "feature_ids": ["rgb_bottle_detection"]},
        {"configuration_id": "davis", "feature_ids": ["davis_aps_frame"]},
        {"configuration_id": "rgb_davis", "feature_ids": ["rgb_bottle_detection", "davis_aps_frame"]},
    ]
    ranked = [
        {"configuration_id": "rgb", "combined_score": 80.0, "winner_eligible": True},
        {"configuration_id": "davis", "combined_score": 0.0, "winner_eligible": False},
        {"configuration_id": "rgb_davis", "combined_score": 60.0, "winner_eligible": True},
    ]

    summary = feature_contribution_summary(ranked, configurations)

    assert summary["DAVIS APS"]["mean_score_with_feature"] == 60.0
    assert summary["DAVIS APS"]["mean_score_without_feature"] == 80.0
    assert summary["DAVIS APS"]["estimated_delta"] == -20.0
    assert summary["DAVIS APS"]["categories"] == ["lower estimated score"]
    positive = feature_contribution_summary(
        [
            {"configuration_id": "rgb", "combined_score": 40.0, "winner_eligible": True},
            {"configuration_id": "rgb_pc", "combined_score": 60.0, "winner_eligible": True},
        ],
        [
            {"configuration_id": "rgb", "feature_ids": ["rgb_bottle_detection"]},
            {"configuration_id": "rgb_pc", "feature_ids": ["rgb_bottle_detection", "pointcloud_scene_geometry"]},
        ],
    )
    assert positive["Point Cloud"]["estimated_delta"] == 20.0
    assert positive["Point Cloud"]["categories"] == ["higher estimated score"]
    assert positive["Point Cloud"]["finding"] == "observed score association in this recorded trial"

    built = build_summary(
        [
            {"configuration_id": "rgb", "display_name": "RGB", "status": "complete", "result_validity": "VALID", "detection_success_rate": 40.0, "motion_warning_count": 0, "event_stream_available": False, "motion_valid_sample_count": 0, "ready_to_close_rate": 0.0, "contact_source_used": "not_applicable_bottle_only"},
            {"configuration_id": "rgb_pc", "display_name": "RGB + Point Cloud", "status": "complete", "result_validity": "VALID", "detection_success_rate": 60.0, "motion_warning_count": 0, "event_stream_available": False, "motion_valid_sample_count": 0, "ready_to_close_rate": 0.0, "contact_source_used": "not_applicable_bottle_only"},
        ],
        [
            {"configuration_id": "rgb", "feature_ids": ["rgb_bottle_detection"]},
            {"configuration_id": "rgb_pc", "feature_ids": ["rgb_bottle_detection", "pointcloud_scene_geometry"]},
        ],
    )
    assert built["winner_cards"]["best_sensor_contribution"] is None
    assert built["winner_cards"]["strongest_positive_score_association"] == "Point Cloud"
    assert built["winner_cards"]["best_motion_detection"] is None
    assert built["winner_cards"]["best_safety_decision"] is None
    assert any("associated with a higher estimated combined score" in finding for finding in built["plain_language_findings"])


def test_runner_advances_configurations_and_exports_results(tmp_path: Path):
    runner = BenchmarkRunner(tmp_path, sample_callback=_sample_status, precheck_callback=lambda: _precheck())
    configs = [
        {"configuration_id": "rgb_only", "display_name": "RGB Only", "feature_ids": preset_feature_ids("rgb_only")},
        {"configuration_id": "rgb_depth", "display_name": "RGB + Depth", "feature_ids": preset_feature_ids("rgb_depth")},
    ]
    state = runner.start(configurations=configs, duration_sec=0.08, sample_rate_hz=20)
    assert state["start_accepted"] is True
    assert _wait_done(runner) == "COMPLETE"
    results = runner.results()
    assert len(results["results"]) == 2
    outputs = results["outputs"]
    assert Path(outputs["manifest"]).exists()
    assert Path(outputs["samples"]).exists()
    assert Path(outputs["summary"]).exists()
    assert Path(outputs["report_html"]).exists()
    assert results["summary"]["metric_wording"] == "Estimated Performance"
    assert results["summary"]["result_type"] == "estimated_performance"
    assert results["summary"]["accuracy_metrics_valid"] is False
    assert "not ground-truth accuracy" in results["summary"]["plain_language_findings"][0]
    assert Path(outputs["validity"]).exists()
    assert Path(outputs["findings_md"]).exists()
    assert (Path(outputs["charts_dir"]) / "detection_success.png").exists()
    assert (Path(outputs["charts_dir"]) / "sensor_availability.png").exists()
    assert (Path(outputs["charts_dir"]) / "feature_contribution.png").exists()
    assert results["summary"]["ranking_table"][0]["controls_robot"] is False
    assert results["summary"]["ranking_table"][0]["result_validity"] in {"VALID", "VALID_WITH_LIMITATIONS"}
    assert "baseline_deltas" in results["summary"]["ranking_table"][0]


def test_runner_uses_supplied_structured_ground_truth_and_exports_its_state(tmp_path: Path):
    runner = BenchmarkRunner(
        tmp_path,
        sample_callback=_sample_status,
        precheck_callback=lambda: _precheck(),
    )
    config = {
        "configuration_id": "rgb_only",
        "display_name": "RGB Only",
        "feature_ids": preset_feature_ids("rgb_only"),
    }
    ground_truth = {
        "labels": [
            {
                "ground_truth": {"bottle_present": True},
                "prediction": {"bottle_present": True},
                "evidence": {"current": True},
            }
        ]
    }

    runner.start(
        configurations=[config],
        duration_sec=0.08,
        sample_rate_hz=20,
        ground_truth=ground_truth,
    )

    assert _wait_done(runner) == "COMPLETE"
    results = runner.results()
    assert results["summary"]["detection_accuracy_valid"] is True
    assert results["summary"]["pose_classification_accuracy_valid"] is False
    assert results["summary"]["clearance_accuracy_valid"] is False
    assert results["summary"]["ground_truth_labels_used"] == 1
    assert Path(results["outputs"]["ground_truth"]).exists()


def test_runner_marks_optional_sensor_configs_unavailable(tmp_path: Path):
    runner = BenchmarkRunner(
        tmp_path,
        sample_callback=_sample_status,
        precheck_callback=lambda: _precheck(pointcloud=False),
    )
    config = default_run_configurations(duration_sec=0.05)[3]
    runner.start(configurations=[config], duration_sec=0.05, sample_rate_hz=20)
    assert _wait_done(runner) == "COMPLETE"
    result = runner.results()["results"][0]
    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == "pointcloud_unavailable_or_stale"
    assert result["result_validity"] == "SENSOR_UNAVAILABLE"
    assert Path(runner.results()["outputs"]["samples"]).exists()


def test_benchmark_records_sync_quality_and_marks_stale_pointcloud():
    realsense, davis, pointcloud, sync, quality, live_quality = _sources()
    realsense.update({
        "realsense_rgb_active": True,
        "realsense_depth_active": True,
        "bottle_distance_mm": 300.0,
        "depth_valid_ratio": 0.82,
    })
    pointcloud["freshness"] = "STALE"
    sync.update({
        "sync_mode": "software_timestamp",
        "sync_quality": "stale_modalities",
        "max_sensor_time_offset_ms": 18.0,
        "pointcloud_freshness": "STALE",
        "pointcloud_age_sec": 9.0,
        "stale_modalities": ["pointcloud"],
        "bundle_valid_for_benchmark": True,
        "bundle_valid_for_live_quality": False,
    })
    config = {"configuration_id": "rgbd_pointcloud", "display_name": "RGB-D + Point Cloud", "feature_ids": preset_feature_ids("rgbd_pointcloud")}

    status = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
        configurations=[config],
    )
    card = status["comparison_cards"][0]
    row = summarize_samples(config, [{
        "timestamp": 1.0,
        "decision": card["decision"],
        "result_validity": card["result_validity"],
        "main_limitation": card["main_limitation"],
        "pointcloud_freshness": "STALE",
        "sync_quality": "stale_modalities",
        "sync_mode": "software_timestamp",
        "sync_offset_ms": 18.0,
        "stale_modalities": ["pointcloud"],
        "bundle_valid_for_benchmark": True,
    }])

    assert status["sync"]["sync_mode"] == "software_timestamp"
    assert card["sync_quality"] == "stale_modalities"
    assert card["stale_features"]
    assert row["sync_stale_modalities"] == ["pointcloud"]
    assert row["stale_data_count"] >= 1


def test_runner_precheck_failure_blocks_run(tmp_path: Path):
    runner = BenchmarkRunner(tmp_path, sample_callback=_sample_status, precheck_callback=lambda: _precheck(ok=False))
    runner.start(configurations=[default_run_configurations(duration_sec=0.05)[0]], duration_sec=0.05)
    assert _wait_done(runner) == "FAILED"
    state = runner.snapshot()
    assert "precheck" in state["current_error"]


def test_runner_pause_resume_stop_and_unique_ids(tmp_path: Path):
    runner = BenchmarkRunner(tmp_path, sample_callback=_sample_status, precheck_callback=lambda: _precheck())
    config = default_run_configurations(duration_sec=0.4)[0]
    first = runner.start(configurations=[config], duration_sec=0.4, sample_rate_hz=10)
    time.sleep(0.05)
    paused = runner.pause()
    assert paused["state"] in {"PAUSED", "CONFIG_RUNNING", "READY", "PRECHECK"}
    runner.resume()
    runner.stop()
    assert _wait_done(runner) in {"CANCELLED", "COMPLETE"}
    second = runner.start(configurations=[config], duration_sec=0.03, sample_rate_hz=20)
    assert second["benchmark_run_id"] != first["benchmark_run_id"]
    assert _wait_done(runner) == "COMPLETE"


def test_runner_skip_current_does_not_render_charts_until_complete(tmp_path: Path):
    runner = BenchmarkRunner(tmp_path, sample_callback=_sample_status, precheck_callback=lambda: _precheck())
    configs = default_run_configurations(duration_sec=0.3)[:2]
    runner.start(configurations=configs, duration_sec=0.3, sample_rate_hz=10)
    time.sleep(0.06)
    run_dir = Path(runner.snapshot()["outputs"]["run_dir"])
    assert not (run_dir / "benchmark_charts").exists()
    runner.skip_current()
    assert _wait_done(runner) == "COMPLETE"
    assert (run_dir / "charts").exists()


def test_summary_requires_current_applicable_structured_labels_for_accuracy():
    config = {"configuration_id": "rgb_depth", "display_name": "RGB + Depth", "feature_ids": preset_feature_ids("rgb_depth")}
    sample = {
        "timestamp": 1.0,
        "bottle_detected": True,
        "detection_confidence": 90,
        "contact_valid": True,
        "ready_to_close": True,
        "analytic_quality_score": 80,
        "latency_ms": 20,
        "decision": "READY",
    }
    result = summarize_samples(config, [sample, {**sample, "timestamp": 1.1}], status="complete")

    estimated = build_summary([result], [config], ground_truth_exists=False)
    declared_without_labels = build_summary(
        [result],
        [config],
        ground_truth_exists=True,
    )
    accurate = build_summary(
        [result],
        [config],
        ground_truth={
            "labels": [
                {
                    "ground_truth": {
                        "bottle_present": True,
                        "pose_class": "STANDING",
                        "feasible_grasp_region": True,
                        "expected_decision": "READY",
                    },
                    "prediction": {
                        "bottle_present": True,
                        "pose_class": "STANDING",
                        "feasible_grasp_region": True,
                        "decision": "READY",
                    },
                    "evidence": {"current": True},
                }
            ]
        },
    )

    assert estimated["result_type"] == "estimated_performance"
    assert estimated["accuracy_metrics_valid"] is False
    assert declared_without_labels["result_type"] == "estimated_performance"
    assert declared_without_labels["ground_truth_state"] == "DECLARED_WITHOUT_LABELS"
    assert declared_without_labels["ground_truth_labels_used"] == 0
    assert declared_without_labels["accuracy_metrics_valid"] is False
    assert accurate["result_type"] == "estimated_performance_with_ground_truth_accuracy"
    assert accurate["accuracy_metrics_valid"] is True
    assert accurate["ground_truth_labels_used"] == 1
    assert accurate["detection_accuracy_valid"] is True
    assert accurate["pose_classification_accuracy_valid"] is True
    assert accurate["clearance_accuracy_valid"] is True
    assert accurate["decision_accuracy_valid"] is True
    assert accurate["accuracy_metrics"]["detection"]["accuracy_percent"] == 100.0


def test_non_current_and_nonapplicable_ground_truth_evidence_fails_closed():
    config = {"configuration_id": "rgb_depth", "display_name": "RGB + Depth", "feature_ids": preset_feature_ids("rgb_depth")}
    result = summarize_samples(
        config,
        [
            {"timestamp": 1.0, "bottle_detected": True},
            {"timestamp": 1.1, "bottle_detected": True},
        ],
    )
    non_current = build_summary(
        [result],
        [config],
        ground_truth={
            "labels": [
                {
                    "ground_truth": {
                        "bottle_present": True,
                        "pose_class": "STANDING",
                        "feasible_grasp_region": True,
                    },
                    "prediction": {
                        "bottle_present": True,
                        "pose_class": "STANDING",
                        "feasible_grasp_region": True,
                    },
                    "evidence": {"current": False},
                }
            ]
        },
    )
    nonapplicable_pose_clearance = build_summary(
        [result],
        [config],
        ground_truth={
            "labels": [
                {
                    "ground_truth": {
                        "bottle_present": False,
                        "pose_class": "NOT_APPLICABLE",
                        "feasible_grasp_region": False,
                    },
                    "prediction": {
                        "bottle_present": False,
                        "pose_class": "NOT_APPLICABLE",
                        "feasible_grasp_region": False,
                    },
                    "evidence": {"current": True},
                }
            ]
        },
    )

    assert non_current["ground_truth_state"] == "NON_CURRENT"
    assert non_current["accuracy_metrics_valid"] is False
    assert non_current["detection_accuracy_valid"] is False
    assert non_current["pose_classification_accuracy_valid"] is False
    assert non_current["clearance_accuracy_valid"] is False
    assert nonapplicable_pose_clearance["detection_accuracy_valid"] is True
    assert nonapplicable_pose_clearance["pose_classification_accuracy_valid"] is False
    assert nonapplicable_pose_clearance["clearance_accuracy_valid"] is False
    assert nonapplicable_pose_clearance["accuracy_metrics"]["pose_classification"]["status"] == "NOT_APPLICABLE"
    assert nonapplicable_pose_clearance["accuracy_metrics"]["clearance"]["status"] == "NOT_APPLICABLE"


def test_summary_uses_only_measured_mask_evaluator_latency():
    config = {"configuration_id": "rgb", "display_name": "RGB", "feature_ids": preset_feature_ids("rgb_only")}
    shared_overlay = {"timestamp": 1.0, "latency_ms": 99.0, "latency_source": "production_overlay_shared"}
    measured = {"timestamp": 1.1, "latency_ms": 2.5, "feature_mask_ms": 2.5, "latency_source": "masked_evaluator_compute"}

    result = summarize_samples(config, [shared_overlay, measured], status="complete")

    assert result["average_latency_ms"] == 2.5
    assert result["latency_measurement"] == "masked_evaluator_compute"
    assert result["processing_time_by_feature"] == {"selected_feature_mask": 2.5}
    assert result["slowest_feature"] is None
    assert result["slowest_feature_reason"] == "per-feature timing not measured"


def test_summary_reports_quantitative_davis_motion_validity():
    config = {
        "configuration_id": "davis_events_only",
        "display_name": "DAVIS Events Only",
        "feature_ids": preset_feature_ids("davis_events_only"),
    }
    sample = {
        "timestamp": 1.0,
        "davis_event_rate": 900.0,
        "davis_dx_px": 3.0,
        "davis_dy_px": 1.0,
        "davis_displacement_px": 3.2,
        "davis_cumulative_slip_px": 5.5,
        "davis_rotation_delta_deg": 2.0,
        "davis_cumulative_rotation_deg": 3.0,
        "davis_vibration_rms_px": 0.8,
        "davis_motion_estimate_valid": True,
        "davis_motion_validity": "VALID",
        "davis_motion_estimate_confidence": 0.75,
        "davis_disturbance_type": "OBJECT_MOTION",
        "decision": "READY_TO_LIFT",
        "result_validity": "VALID",
    }

    result = summarize_samples(config, [sample], status="complete")

    assert result["mean_displacement_px"] == 3.2
    assert result["cumulative_slip_px"] == 5.5
    assert result["mean_rotation_deg"] == 2.0
    assert result["vibration_rms_px"] == 0.8
    assert result["motion_validity"] == "VALID"
    assert result["motion_valid_sample_count"] == 1
    assert result["motion_validity_rate"] == 100.0


def test_invalid_configuration_is_not_ranked_first():
    valid_config = {"configuration_id": "rgb_depth", "display_name": "RGB + Depth", "feature_ids": preset_feature_ids("rgb_depth")}
    invalid_config = {"configuration_id": "davis_events_only", "display_name": "DAVIS Events", "feature_ids": preset_feature_ids("davis_events_only")}
    valid_sample = {
        "timestamp": 1.0,
        "bottle_detected": True,
        "detection_confidence": 80,
        "contact_valid": True,
        "ready_to_close": True,
        "analytic_quality_score": 75,
        "latency_ms": 25,
    }
    valid = summarize_samples(valid_config, [valid_sample, {**valid_sample, "timestamp": 1.2}], status="complete")
    invalid = summarize_samples(invalid_config, [], status="unavailable", unavailable_reason="davis_events_unavailable")

    summary = build_summary([invalid, valid], [invalid_config, valid_config])

    assert summary["ranking_table"][0]["configuration_id"] == "rgb_depth"
    assert summary["ranking_table"][-1]["result_validity"] == "SENSOR_UNAVAILABLE"
    assert summary["winner_cards"]["best_overall"]["configuration_id"] == "rgb_depth"


def test_stale_pointcloud_trial_marked_limited_or_invalid():
    config = {"configuration_id": "rgbd_pointcloud", "display_name": "RGB-D + Point Cloud", "feature_ids": preset_feature_ids("rgbd_pointcloud")}
    sample = {
        "timestamp": 1.0,
        "bottle_detected": True,
        "detection_confidence": 80,
        "contact_valid": True,
        "ready_to_close": True,
        "analytic_quality_score": 78,
        "latency_ms": 60,
        "pointcloud_freshness": "STALE",
    }
    result = summarize_samples(config, [sample, {**sample, "timestamp": 1.2}], status="complete")

    assert result["result_validity"] in {"STALE_DATA", "VALID_WITH_LIMITATIONS"}
    assert result["stale_data_count"] >= 2

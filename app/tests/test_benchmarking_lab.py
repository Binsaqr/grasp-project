"""Camera-free tests for Benchmark Lab feature masking and trial recording."""

import json
from pathlib import Path

from grasp_project.benchmarking.feature_mask import (
    ablation_configurations,
    build_benchmark_lab_status,
    mask_bundle,
    validate_feature_selection,
)
from grasp_project.benchmarking.feature_registry import (
    FEATURE_BY_ID,
    PRESET_DEFINITIONS,
    default_comparison_configurations,
    preset_feature_ids,
)
from grasp_project.benchmarking.trial_recorder import (
    BenchmarkTrialRecorder,
    evaluate_ground_truth_labels,
)


def _status():
    realsense = {
        "frame_id": 42,
        "decision": "DO_NOT_CLOSE",
        "recommended_action": "ADJUST_POSITION",
        "ready_to_close": False,
        "writer_session_id": "session-a",
        "target_status": "TRACKING",
        "tracking_confidence": 91.0,
        "processed_overlay_timing": {"total_processed_overlay_ms": 18.5},
        "realsense_perception": {
            "bottle_detection_status": "TRACKING",
            "bottle_bbox_xyxy": [10, 20, 80, 150],
            "bottle_center_px": [45, 85],
            "bottle_angle_deg": 12.0,
            "contact_pair_valid": True,
            "contact_source_confidence": 88.0,
            "detected_contact_count": 2,
            "contact_depth_consistent": True,
        },
    }
    quality = {
        "quality_label": "GOOD",
        "quality_label_display": "GOOD",
        "epsilon_score": 0.32,
        "normalized_quality_score": 82.0,
        "contact_quality_gate_pass": True,
        "contact_source": "detected_finger_contacts",
        "image_bottle_hull_px": [[0, 0], [1, 0], [1, 1]],
    }
    davis = {
        "event_rate": 1200.0,
        "decision": "READY_TO_LIFT",
        "stability": 90.0,
        "rotation": 4.0,
        "vibration": 8.0,
    }
    pointcloud = {
        "freshness": "CURRENT",
        "bottle_pointcloud_valid": True,
        "contact_3d_quality": "valid",
        "contact_points_on_bottle_cloud": True,
        "scene_raw_point_count": 5000,
        "bottle_filtered_point_count": 600,
    }
    sync = {"last_bundle_id": "bundle-42", "sync_quality": "approximate", "max_sensor_time_offset_ms": 12.0}
    live_quality = {"live_quality_score": 76.0, "live_quality_label": "OK"}
    return realsense, davis, pointcloud, sync, quality, live_quality


def test_feature_registry_loads_and_presets_generate_configs():
    assert "rgb_bottle_detection" in FEATURE_BY_ID
    assert "full_multimodal" in PRESET_DEFINITIONS
    assert preset_feature_ids("rgb_only")
    defaults = default_comparison_configurations()
    assert len(defaults) == 4
    assert defaults[-1]["preset_id"] == "full_multimodal"


def test_missing_dependency_is_detected():
    result = validate_feature_selection(["analytic_epsilon"])
    assert result["feature_selection_valid"] is False
    assert "analytic_epsilon" in result["missing_dependencies"]


def test_feature_mask_blocks_disabled_sources():
    bundle = {
        "production": {"decision": "DO_NOT_CLOSE"},
        "rgb": {"bbox": [1, 2, 3, 4]},
        "depth": {"distance": 500},
        "pointcloud": {"points": 10},
        "davis_events": {"event_rate": 100},
    }
    rgb_only = mask_bundle(bundle, preset_feature_ids("rgb_only"))
    assert "rgb" in rgb_only
    assert "depth" not in rgb_only
    assert "pointcloud" not in rgb_only
    assert "davis_events" not in rgb_only

    events_only = mask_bundle(bundle, preset_feature_ids("davis_events_only"))
    assert "davis_events" in events_only
    assert "rgb" not in events_only


def test_full_multimodal_uses_all_feature_groups_and_same_bundle():
    realsense, davis, pointcloud, sync, quality, live_quality = _status()
    status = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
        configurations=[{"display_name": "Full", "feature_ids": preset_feature_ids("full_multimodal")}],
    )
    card = status["comparison_cards"][0]
    assert card["same_bundle_id"] == "bundle-42"
    assert card["production_isolated"] is True
    assert card["controls_robot"] is False
    assert card["epsilon_available"] is True
    assert card["live_quality_score"] == 76.0


def test_events_only_cannot_claim_stationary_bottle_detection():
    realsense, davis, pointcloud, sync, quality, live_quality = _status()
    davis["event_rate"] = 0.0
    status = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
        configurations=[{"display_name": "Events", "feature_ids": preset_feature_ids("davis_events_only")}],
    )
    card = status["comparison_cards"][0]
    assert card["bottle_detected"] is False
    assert card["detection_status"] == "stationary_bottle_not_identifiable_from_events_only"


def test_pointcloud_only_reports_depth_dependency():
    status = build_benchmark_lab_status(
        configurations=[{"display_name": "Point Cloud", "feature_ids": preset_feature_ids("pointcloud_only")}],
    )
    card = status["comparison_cards"][0]
    assert card["feature_selection_valid"] is False
    assert card["missing_dependencies"]


def test_epsilon_unavailable_when_contact_geometry_disabled_or_missing():
    realsense, davis, pointcloud, sync, quality, live_quality = _status()
    realsense["realsense_perception"]["contact_pair_valid"] = False
    quality["contact_quality_gate_pass"] = False
    status = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
        configurations=[{"display_name": "Epsilon", "feature_ids": ["analytic_epsilon", "visual_contact_quality", "rgbd_visual_contact_model", "rgbd_contact_patches", "rgbd_bottle_tracking", "depth_contact_consistency", "depth_bottle_distance", "rgb_bottle_detection", "rgb_contact_visibility", "rgb_gripper_detection"]}],
    )
    card = status["comparison_cards"][0]
    assert card["epsilon_available"] is False
    assert "epsilon unavailable" in card["main_limitation"] or any("epsilon unavailable" in x for x in card["missing_features"] + [card["main_limitation"]])


def test_ablation_removes_exact_feature_group():
    configs = {config["configuration_id"]: config for config in ablation_configurations()}
    minus_events = configs["full_minus_davis_events"]
    assert "event_rate" not in minus_events["feature_ids"]
    assert "rgb_bottle_detection" in minus_events["feature_ids"]
    minus_epsilon = configs["full_minus_epsilon"]
    assert "analytic_epsilon" not in minus_epsilon["feature_ids"]
    assert "wrench_quality" in minus_epsilon["feature_ids"]


def test_trial_recorder_keeps_ground_truth_separate_and_exports(tmp_path: Path):
    recorder = BenchmarkTrialRecorder(tmp_path)
    realsense, davis, pointcloud, sync, quality, live_quality = _status()
    benchmark = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
    )
    state = recorder.start_trial(configurations=benchmark["configurations"])
    assert state["active"] is True

    state = recorder.record_label("good_grasp", benchmark_status=benchmark)
    trial_dir = Path(state["trial_dir"])
    assert (trial_dir / "ground_truth.json").exists()
    assert (trial_dir / "benchmark_results.jsonl").exists()
    assert state["ground_truth_state"] == "UNSTRUCTURED_LABELS_ONLY"
    assert state["accuracy_metrics_valid"] is False

    exported = recorder.export_report(benchmark)
    assert Path(exported["report_json"]).exists()
    assert Path(exported["configuration_results_csv"]).exists()
    assert Path(exported["modality_comparison_csv"]).exists()
    assert Path(exported["feature_contributions_csv"]).exists()
    assert Path(exported["benchmark_report_html"]).exists()

    state = recorder.undo_last_label()
    assert state["label_count"] == 0


def test_trial_recorder_structures_current_ground_truth_and_counts_used_labels(tmp_path: Path):
    recorder = BenchmarkTrialRecorder(tmp_path)
    realsense, davis, pointcloud, sync, quality, live_quality = _status()
    sync["bundle_valid_for_benchmark"] = True
    benchmark = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
    )
    recorder.start_trial(configurations=benchmark["configurations"])

    state = recorder.record_label(
        {
            "label": "standing_clear_ready",
            "ground_truth": {
                "bottle_present": True,
                "pose_class": "STANDING",
                "clutter_class": "CLEAR",
                "feasible_grasp_region": True,
                "expected_decision": "DO_NOT_CLOSE",
            },
            "prediction": {
                "bottle_present": True,
                "pose_class": "STANDING",
                "feasible_grasp_region": True,
                "decision": "DO_NOT_CLOSE",
            },
            "evidence": {"current": True},
        },
        benchmark_status=benchmark,
    )

    ground_truth = json.loads((Path(state["trial_dir"]) / "ground_truth.json").read_text())
    assert ground_truth["schema_version"] == "benchmark_ground_truth_v2"
    assert ground_truth["ground_truth_state"] == "AVAILABLE"
    assert ground_truth["ground_truth_label_count"] == 1
    assert ground_truth["ground_truth_labels_used"] == 1
    assert ground_truth["accuracy_metrics"]["detection"]["valid"] is True
    assert ground_truth["accuracy_metrics"]["pose_classification"]["valid"] is True
    assert ground_truth["accuracy_metrics"]["clearance"]["valid"] is True
    assert ground_truth["accuracy_metrics"]["decision"]["valid"] is True
    assert state["accuracy_metrics_valid"] is True


def test_structured_metrics_cover_detection_pose_identity_and_band_drift():
    def entry(ground_truth, prediction):
        return {
            "ground_truth": ground_truth,
            "prediction": prediction,
            "evidence": {"current": True},
        }

    labels = [
        entry(
            {
                "bottle_present": True,
                "pose_angle_deg": 10.0,
                "target_identity": "bottle_a",
                "target_persistent": True,
                "selected_band": "preferred",
                "selected_body_fraction": 0.50,
                "band_center_px": [100, 100],
            },
            {
                "bottle_present": True,
                "pose_angle_deg": 12.0,
                "target_identity": "bottle_a",
                "target_persistent": True,
                "selected_band": "preferred",
                "selected_body_fraction": 0.52,
                "band_center_px": [102, 100],
            },
        ),
        entry({"bottle_present": True}, {"bottle_present": False}),
        entry({"bottle_present": False}, {"bottle_present": True}),
        entry({"bottle_present": False}, {"bottle_present": False}),
        entry(
            {
                "bottle_present": True,
                "pose_angle_deg": 20.0,
                "target_identity": "bottle_a",
                "target_persistent": True,
                "selected_band": "preferred",
                "selected_body_fraction": 0.50,
                "band_center_px": [110, 100],
            },
            {
                "bottle_present": True,
                "pose_angle_deg": 24.0,
                "target_identity": "bottle_b",
                "target_persistent": True,
                "selected_band": "preferred",
                "selected_body_fraction": 0.56,
                "band_center_px": [116, 100],
            },
        ),
    ]

    result = evaluate_ground_truth_labels(labels, declared=True)
    detection = result["structured_metrics"]["detection"]
    assert detection["confusion"] == {
        "true_positive": 2,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert detection["precision_percent"] == 66.67
    assert detection["recall_percent"] == 66.67

    pose = result["structured_metrics"]["pose_angle_error"]
    assert pose["mean_absolute_error_deg"] == 3.0

    identity = result["structured_metrics"]["identity"]
    assert identity["identity_accuracy_percent"] == 50.0
    assert identity["persistence_accuracy_percent"] == 100.0
    assert identity["identity_switch_proxy_opportunities"] == 1
    assert identity["identity_switch_proxy_count"] == 1
    assert identity["identity_switch_proxy_rate_percent"] == 100.0

    band = result["structured_metrics"]["band"]
    assert band["selected_band_accuracy_percent"] == 100.0
    assert band["mean_absolute_body_fraction_error"] == 0.04
    assert band["mean_band_center_error_px"] == 4.0
    assert band["mean_body_fraction_drift_error"] == 0.04
    assert band["mean_band_center_drift_error_px"] == 4.0
    assert result["structured_metrics_valid"] is True


def test_structured_metrics_require_explicit_current_evidence():
    result = evaluate_ground_truth_labels(
        [
            {
                "ground_truth": {
                    "bottle_present": True,
                    "pose_angle_deg": 10.0,
                    "target_identity": "bottle_a",
                    "target_persistent": True,
                    "selected_band": "preferred",
                    "selected_body_fraction": 0.5,
                    "band_center_px": [100, 100],
                },
                "prediction": {
                    "bottle_present": True,
                    "pose_angle_deg": 10.0,
                    "target_identity": "bottle_a",
                    "target_persistent": True,
                    "selected_band": "preferred",
                    "selected_body_fraction": 0.5,
                    "band_center_px": [100, 100],
                },
            }
        ],
        declared=True,
    )

    assert result["ground_truth_state"] == "NON_CURRENT"
    assert result["accuracy_metrics_valid"] is False
    assert result["structured_metrics_valid"] is False
    assert all(
        metric["valid"] is False
        for metric in result["structured_metrics"].values()
    )


def test_nested_prediction_gates_angle_wrap_and_absent_bottle_applicability():
    wrapped = evaluate_ground_truth_labels(
        [
            {
                "ground_truth": {
                    "bottle_present": True,
                    "pose_angle_deg": 359.0,
                    "target_identity": "bottle_a",
                    "selected_body_fraction": 0.5,
                },
                "prediction": {
                    "bottle_present": True,
                    "pose_angle_deg": {
                        "value": {"value": 1.0, "current": True},
                        "current": True,
                    },
                    "target_identity": {
                        "value": {"value": "bottle_a", "current": False},
                        "current": True,
                    },
                    "selected_body_fraction": {
                        "value": {"value": 0.5, "current": False},
                        "current": True,
                    },
                },
                "evidence": {"current": True},
            }
        ],
        declared=True,
    )

    assert wrapped["structured_metrics"]["pose_angle_error"]["mean_absolute_error_deg"] == 2.0
    assert wrapped["structured_metrics"]["identity"]["status"] == "NON_CURRENT"
    assert wrapped["structured_metrics"]["identity"]["identity_pair_count"] == 0
    assert wrapped["structured_metrics"]["band"]["status"] == "NON_CURRENT"
    assert wrapped["structured_metrics"]["band"]["body_fraction_pair_count"] == 0

    absent = evaluate_ground_truth_labels(
        [
            {
                "ground_truth": {
                    "bottle_present": False,
                    "pose_angle_deg": 10.0,
                    "target_identity": "bottle_a",
                    "selected_body_fraction": 0.5,
                },
                "prediction": {
                    "bottle_present": False,
                    "pose_angle_deg": 10.0,
                    "target_identity": "bottle_a",
                    "selected_body_fraction": 0.5,
                },
                "evidence": {"current": True},
            }
        ],
        declared=True,
    )

    assert absent["structured_metrics"]["detection"]["confusion"]["true_negative"] == 1
    assert absent["structured_metrics"]["pose_angle_error"]["status"] == "NOT_APPLICABLE"
    assert absent["structured_metrics"]["identity"]["status"] == "NOT_APPLICABLE"
    assert absent["structured_metrics"]["band"]["status"] == "NOT_APPLICABLE"


def test_trial_recorder_extracts_extended_prediction_snapshot(tmp_path: Path):
    recorder = BenchmarkTrialRecorder(tmp_path)
    realsense, davis, pointcloud, sync, quality, live_quality = _status()
    benchmark = build_benchmark_lab_status(
        realsense=realsense,
        davis=davis,
        pointcloud=pointcloud,
        sync=sync,
        quality=quality,
        live_quality=live_quality,
    )
    benchmark["source_status"] = {
        "realsense": {
            "object_state": {"tilt_angle_deg": 14.5},
            "locked_bottle_target_id": "bottle_07",
            "identity_measurement_current": True,
            "target_identity_preserved": True,
            "grasp_memory": {"selected_candidate_id": "band_07"},
            "locked_body_fraction": 0.4,
            "target_clamp": {"clamp_center_px": [120, 80]},
        }
    }
    recorder.start_trial(configurations=benchmark["configurations"])

    state = recorder.record_label(
        {
            "ground_truth": {
                "bottle_present": True,
                "pose_angle_deg": 14.0,
                "target_identity": "bottle_07",
                "target_persistent": True,
                "selected_band": "band_07",
                "selected_body_fraction": 0.42,
                "band_center_px": [121, 80],
            },
            "evidence": {"current": True},
        },
        benchmark_status=benchmark,
    )

    document = json.loads((Path(state["trial_dir"]) / "ground_truth.json").read_text())
    prediction = document["labels"][-1]["prediction"]
    assert prediction["pose_angle_deg"] == 14.5
    assert prediction["target_identity"] == {
        "value": "BOTTLE_07",
        "current": True,
    }
    assert prediction["target_persistent"] == {"value": True, "current": True}
    assert prediction["selected_band"] == "BAND_07"
    assert prediction["selected_body_fraction"] == 0.4
    assert prediction["band_center_px"] == [120.0, 80.0]
    assert document["structured_metrics_valid"] is True


def test_benchmark_label_route_accepts_structured_and_legacy_payloads(monkeypatch):
    from grasp_project.dashboard import web_dashboard

    calls = []

    class Recorder:
        def record_label(self, label, **kwargs):
            calls.append((label, kwargs))
            return {"active": True}

    monkeypatch.setattr(web_dashboard, "benchmark_recorder", Recorder())
    monkeypatch.setattr(
        web_dashboard,
        "lightweight_current_benchmark_status",
        lambda: {"production_context": {"frame_id": 42}},
    )
    client = web_dashboard.app.test_client()

    response = client.post(
        "/api/benchmark/label",
        json={
            "label": "scene_1",
            "ground_truth": {"bottle_present": True, "pose_angle_deg": 5.0},
            "prediction": {"bottle_present": True, "pose_angle_deg": 6.0},
            "evidence": {"current": True},
        },
    )
    assert response.status_code == 200
    assert calls[-1][0] == {
        "label": "scene_1",
        "ground_truth": {"bottle_present": True, "pose_angle_deg": 5.0},
        "prediction": {"bottle_present": True, "pose_angle_deg": 6.0},
        "evidence": {"current": True},
    }

    response = client.post("/api/benchmark/label", json={"label": "good_grasp"})
    assert response.status_code == 200
    assert calls[-1][0] == "good_grasp"

    assert client.post("/api/benchmark/label", json=[]).status_code == 400
    assert client.post(
        "/api/benchmark/label",
        json={"label": "bad", "ground_truth": []},
    ).status_code == 400


def test_dashboard_exposes_benchmark_report_download_labels():
    dashboard = Path("src/grasp_project/dashboard/web_dashboard.py").read_text()

    assert "Download Comparison Tables" in dashboard
    assert "Download HTML Report" in dashboard
    assert "Download CSV Results" in dashboard
    assert "Download Charts" in dashboard

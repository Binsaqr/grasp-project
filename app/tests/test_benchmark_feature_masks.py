"""Camera-free proof tests for strict Benchmark Lab feature masks."""

from __future__ import annotations

from pathlib import Path

from grasp_project.benchmarking.evaluators import evaluate_configuration_card, run_poison_value_tests
from grasp_project.benchmarking.feature_mask_proof import proof_configurations, run_feature_mask_proof
from grasp_project.benchmarking.masked_feature_view import DEPENDENCY_MISSING, NOT_APPLICABLE, MaskedFeatureView, build_feature_packet


def _packet():
    return build_feature_packet(
        realsense={
            "target_status": "TRACKING",
            "tracking_confidence": 92.0,
            "distance_mm": 430.0,
            "depth_valid_ratio": 0.87,
            "realsense_rgb_active": True,
            "realsense_depth_active": True,
            "realsense_perception": {
                "bottle_detection_status": "TRACKING",
                "bottle_bbox_xyxy": [10, 20, 90, 170],
                "bottle_center_px": [50, 95],
                "bottle_angle_deg": 4.0,
                "mask_available": True,
            },
        },
        davis={
            "event_state": "STREAM_ACTIVE_EVENTS",
            "event_rate": 1200.0,
            "stability": 88.0,
            "rotation": 3.0,
            "vibration": 4.0,
            "aps_state": "APS_COLOR_VERIFIED",
            "aps_fps": 20.0,
        },
        pointcloud={
            "active": True,
            "freshness": "CURRENT",
            "bottle_pointcloud_valid": True,
            "bottle_filtered_point_count": 900,
            "bottle_pointcloud_roi_source": "cleaned_bottle_mask",
            "bottle_pointcloud_extents_mm": {"width_x": 64.0, "height_y": 210.0, "depth_z": 62.0},
        },
        quality={"epsilon_score": 0.4, "force_closure": True, "quality_label": "GOOD"},
        live_quality={"live_quality_score": 82.0},
        sync={"last_bundle_id": "bundle-test", "sync_quality": "good", "max_sensor_time_offset_ms": 3.0},
    )


def test_rgb_only_mask_does_not_use_depth_davis_or_pointcloud():
    card = evaluate_configuration_card(
        {
            "configuration_id": "rgb_only",
            "feature_ids": ["rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle"],
            "benchmark_object_mode": "bottle_only",
        },
        _packet(),
    )

    assert card["bottle_detected"] is True
    assert card["distance_mm"] is None
    assert card["event_rate"] is None
    assert card["pointcloud_valid"] is False
    assert set(card["used_features"]) <= {"rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle"}
    assert "depth_bottle_distance" in card["blocked_features"]
    assert "event_rate" in card["blocked_features"]


def test_pointcloud_only_reports_dependency_missing_without_auto_enabling_rgbd():
    card = evaluate_configuration_card(
        {
            "configuration_id": "pointcloud_only",
            "feature_ids": ["pointcloud_scene_geometry", "pointcloud_bottle_geometry", "pointcloud_bottle_extents"],
            "benchmark_object_mode": "bottle_only",
        },
        _packet(),
    )

    assert card["feature_selection_valid"] is False
    assert card["result_validity"] in {DEPENDENCY_MISSING, "VALID_WITH_LIMITATIONS"}
    assert "pointcloud_bottle_geometry" in card["missing_dependencies"]
    assert "rgbd_bottle_tracking" in card["missing_dependencies"]["pointcloud_bottle_geometry"]


def test_contact_epsilon_is_not_applicable_in_bottle_only_mode():
    card = evaluate_configuration_card(
        {
            "configuration_id": "epsilon_contact",
            "feature_ids": [
                "rgb_bottle_detection",
                "analytic_epsilon",
                "wrench_quality",
                "visual_contact_quality",
            ],
            "benchmark_object_mode": "bottle_only",
        },
        _packet(),
    )

    assert card["epsilon_available"] is False
    assert card["quality_result_validity"] == "NOT_APPLICABLE_WITHOUT_CONTACT_GEOMETRY"
    assert card["not_applicable_features"]["analytic_epsilon"]
    assert card["not_applicable_features"]["wrench_quality"]


def test_poison_value_tests_prove_disabled_feature_isolation(tmp_path: Path):
    output = tmp_path / "poison.json"
    result = run_poison_value_tests(output)

    assert output.exists()
    assert result["feature_leakage_detected"] is False
    assert all(case["passed"] for case in result["cases"])


def test_mapped_visual_and_geometry_quality_features_are_used():
    card = evaluate_configuration_card(
        {
            "configuration_id": "visual_geometry_quality",
            "feature_ids": [
                "rgb_bottle_detection",
                "rgb_bottle_segmentation",
                "rgb_bottle_center",
                "rgb_bottle_angle",
                "depth_valid_ratio",
                "depth_object_geometry",
                "visual_object_quality",
                "geometry_quality",
            ],
            "benchmark_object_mode": "bottle_only",
        },
        _packet(),
    )

    assert "visual_object_quality" in card["used_features"]
    assert "geometry_quality" in card["used_features"]
    assert card["visual_quality"] is not None
    assert card["geometry_quality"] is not None
    assert not card["unsupported_features"]


def test_required_proof_fields_present_on_configuration_card():
    card = evaluate_configuration_card(
        {
            "configuration_id": "rgb_only",
            "feature_ids": ["rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle"],
            "benchmark_object_mode": "bottle_only",
        },
        _packet(),
    )

    for key in [
        "enabled_features",
        "disabled_features",
        "used_features",
        "attempted_features",
        "blocked_features",
        "missing_dependencies",
        "not_applicable_features",
        "unavailable_features",
        "stale_features",
        "unsupported_features",
        "production_fields_used",
        "feature_leakage_detected",
        "evaluator_trace",
    ]:
        assert key in card
    assert card["production_fields_used"] == []
    assert card["feature_leakage_detected"] is False
    assert card["latency_source"] == "masked_evaluator_compute"
    assert card["latency_ms"] == card["profiling"]["feature_mask_ms"]
    assert card["latency_ms"] > 0.0


def test_production_reference_access_is_recorded_by_proof_payload():
    packet = _packet()
    view = MaskedFeatureView(packet, ["rgb_bottle_detection"])

    assert packet.production_reference.get("quality_label") == "GOOD"
    assert view.proof_payload()["production_fields_used"] == ["quality_label"]


def test_proof_configurations_match_required_sequence():
    configs = proof_configurations(duration_sec=5.0)

    assert [config["configuration_id"] for config in configs] == [
        "rgb_only",
        "depth_only",
        "davis_aps_only",
        "davis_events_only",
        "pointcloud_only",
        "rgb_depth",
        "full_multimodal",
    ]
    assert all(config["duration_sec"] == 5.0 for config in configs)
    assert all(config["benchmark_object_mode"] == "bottle_only" for config in configs)


def test_feature_mask_proof_runner_writes_requested_outputs(tmp_path: Path):
    output = tmp_path / "output"
    (output / "davis").mkdir(parents=True)
    (output / "live_quality").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (output / "latest_realtime_status.json").write_text(
        """{
          "target_status": "TRACKING",
          "tracking_confidence": 88.0,
          "distance_mm": 330.0,
          "depth_valid_ratio": 0.82,
          "realsense_rgb_active": true,
          "realsense_depth_active": true,
          "realsense_perception": {
            "bottle_detection_status": "TRACKING",
            "bottle_bbox_xyxy": [10, 20, 90, 170],
            "bottle_center_px": [50, 95],
            "bottle_angle_deg": 4.0,
            "mask_available": true
          }
        }"""
    )
    (output / "pointcloud_status.json").write_text(
        """{
          "pointcloud_freshness": "CURRENT",
          "bottle_pointcloud_valid": true,
          "scene_raw_point_count": 1000,
          "bottle_filtered_point_count": 200,
          "bottle_pointcloud_extents_mm": {"width_x": 50, "height_y": 200, "depth_z": 30}
        }"""
    )
    (output / "davis" / "latest_status.json").write_text(
        """{"aps_state": "APS_COLOR_VERIFIED", "event_state": "STREAM_ACTIVE_NO_MOTION", "motion_class": "STABLE"}"""
    )
    (output / "davis" / "latest_motion_report.json").write_text(
        """{"aps_fps": 10.0, "event_stream_status": "active", "event_rate": 0.0, "stability": 90.0}"""
    )
    (output / "realtime_wrench_matrix_status.json").write_text("""{"quality_label": "GOOD", "epsilon_score": 100.0}""")
    (output / "sync_status.json").write_text("""{"sync_quality": "approximate", "max_sensor_time_offset_ms": 4.0}""")
    (output / "live_quality" / "latest_status.json").write_text("""{"live_quality_score": 80.0}""")

    summary = run_feature_mask_proof(root=tmp_path, duration_sec=0.03, sample_rate_hz=20.0)

    assert summary["state"] == "COMPLETE"
    assert (output / "benchmarks" / "feature_mask_proof_summary.json").exists()
    assert (tmp_path / "docs" / "benchmark_feature_mask_proof_report.md").exists()
    assert summary["poison_tests"]["passed"] is True
    assert all(row["proof_fields_present"] for row in summary["configurations"])
    assert not any(row["feature_leakage_detected"] for row in summary["configurations"])


def test_depth_valid_ratio_reads_realsense_depth_stats_not_pointcloud():
    # RealSense depth-pipeline ratio (depth_stats) populates the depth-only
    # feature; pointcloud/quality pipeline ratios must NOT leak in.
    packet = build_feature_packet(
        realsense={"target_status": "LOCKED", "depth_stats": {"raw_depth_valid_ratio": 0.77}},
        davis={}, pointcloud={"pointcloud_depth_valid_ratio": 0.99}, sync={},
        quality={"depth_valid_ratio": 0.5}, live_quality={}, embodiment={},
    )
    assert packet.values.get("depth_valid_ratio") == 0.77

    leak_guard = build_feature_packet(
        realsense={"target_status": "LOCKED"},
        davis={}, pointcloud={"pointcloud_depth_valid_ratio": 0.99}, sync={},
        quality={"depth_valid_ratio": 0.5}, live_quality={}, embodiment={},
    )
    assert leak_guard.values.get("depth_valid_ratio") is None


def test_depth_valid_ratio_reads_perception_bottle_ratio():
    packet = build_feature_packet(
        realsense={"target_status": "LOCKED", "realsense_perception": {"bottle_depth_valid_ratio": 0.66}},
        davis={}, pointcloud={}, sync={}, quality={}, live_quality={}, embodiment={},
    )
    assert packet.values.get("depth_valid_ratio") == 0.66

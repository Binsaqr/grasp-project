"""Camera-free tests for the optional sensor sync / point cloud / embodiment layer."""

import json
import time

import numpy as np

from grasp_project.embodiment.contact_model import quality_metric_applicability
from grasp_project.embodiment.gripper_model import DEFAULT_MODELS, embodiment_status
from grasp_project.realsense.pointcloud import (
    PointCloudRuntimeState,
    create_pointcloud_from_rgbd,
    downsample_pointcloud,
    estimate_bottle_3d_properties,
    pointcloud_summary,
    pointcloud_live_json_payload,
    save_pointcloud_live_json,
    save_pointcloud_ply,
    save_pointcloud_preview,
    update_pointcloud_from_frame,
)
from grasp_project.sync.calibrate_2d import estimate_homography, main as calibrate_2d_main
from grasp_project.sync.calibration_status import calibration_status
from grasp_project.sync.sensor_sync import (
    SYNC_MODE_ACTION_ANCHORED_WINDOW,
    SYNC_MODE_DISPLAY_LATEST,
    SensorSynchronizer,
    event_features_for_window,
)


def test_pointcloud_from_dummy_rgbd(tmp_path):
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    depth = np.full((4, 5), 500, dtype=np.uint16)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0}

    cloud = create_pointcloud_from_rgbd(rgb, depth, intrinsics)

    assert cloud["points"].shape[1] == 3
    assert cloud["colors"].shape == cloud["points"].shape
    assert cloud["summary"]["point_count"] == 20
    points, colors = downsample_pointcloud(cloud["points"], cloud["colors"], voxel_size=0.01)
    assert len(points) <= 20
    props = estimate_bottle_3d_properties(points)
    assert props["bottle_cloud_valid"] is True
    path = tmp_path / "dummy.ply"
    assert save_pointcloud_ply(path, points, colors) is True
    assert path.read_text().startswith("ply")
    preview_path = tmp_path / "dummy_preview.jpg"
    preview = save_pointcloud_preview(preview_path, points, colors, bottle_points=points[:5], bottle_colors=colors[:5], width=360, height=240)
    assert preview["saved"] is True
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0
    payload = pointcloud_live_json_payload(points, colors, bottle_points=points[:5], bottle_colors=colors[:5], max_points=10, timestamp=123.0)
    assert payload["scene"]["point_count"] <= 10
    assert payload["coordinate_frame"] == "realsense_camera_m"
    assert payload["coordinate_frame_type"] == "current_camera_frame"
    assert payload["world_registered"] is False
    assert len(payload["scene"]["positions"]) == payload["scene"]["point_count"] * 3
    assert len(payload["scene"]["colors"]) == payload["scene"]["point_count"] * 3
    json_path = tmp_path / "dummy_pointcloud.json"
    json_result = save_pointcloud_live_json(json_path, points, colors, bottle_points=points[:5], bottle_colors=colors[:5], max_points=10)
    assert json_result["saved"] is True
    assert json_path.exists()
    assert pointcloud_summary(np.empty((0, 3)))["point_count"] == 0


def test_pointcloud_rejects_realsense_invalid_depth_sentinel():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.full((4, 5), 500, dtype=np.uint16)
    depth[0, 0] = np.iinfo(np.uint16).max
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0}

    cloud = create_pointcloud_from_rgbd(rgb, depth, intrinsics)

    assert cloud["summary"]["point_count"] == 19
    assert float(np.max(cloud["points"][:, 2])) == 0.5


def _rgbd_fixture(size=64):
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 2] = 160
    depth = np.full((size, size), 900, dtype=np.uint16)
    depth[14:54, 25:39] = 500
    intrinsics = {"fx": 120.0, "fy": 120.0, "cx": size / 2.0, "cy": size / 2.0, "width": size, "height": size}
    return rgb, depth, intrinsics


def test_pointcloud_valid_cleaned_mask_crop_is_high_confidence():
    rgb, depth, intrinsics = _rgbd_fixture()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[14:54, 25:39] = True
    report = {
        "frame_id": 11,
        "depth_frame_id": 11,
        "writer_session_id": "test-writer",
        "bottle_geometry": {
            "cleaned_mask": mask,
            "frame_id": 11,
            "geometry_frame_id": 11,
            "mask_frame_id": 11,
            "measurement_status": "CURRENT_MEASURED_RGBD",
            "mask_measurement_status": "CURRENT_MEASURED_RGBD",
            "bottle_extent_bbox_xyxy": [25, 14, 39, 54],
        },
        "target_clamp": {"left_contact_px": [25, 34], "right_contact_px": [38, 34], "clamp_center_px": [31.5, 34]},
    }
    status = update_pointcloud_from_frame(rgb, depth, intrinsics, report, state=PointCloudRuntimeState(), now=100.0)
    assert status["bottle_roi_source"] == "cleaned_mask"
    assert status["bottle_crop_status"] == "VALID_MASK_CROP"
    assert status["bottle_crop_confidence"] >= 0.9
    assert status["pointcloud_snapshot_consistent"] is True
    assert status["cloud_valid_for_visualization"] is True


def test_pointcloud_contour_crop_is_preferred_over_bbox_fallback():
    rgb, depth, intrinsics = _rgbd_fixture()
    depth = depth.copy()
    depth[14:54, 25:39] += np.linspace(0, 36, 40, dtype=np.uint16).reshape(40, 1)
    contour = np.array([[25, 14], [39, 14], [39, 54], [25, 54]], dtype=np.float32)
    report = {
        "frame_id": 13,
        "depth_frame_id": 13,
        "writer_session_id": "test-writer",
        "bottle_geometry": {
            "body_contour": contour,
            "frame_id": 13,
            "geometry_frame_id": 13,
            "mask_frame_id": 13,
            "measurement_status": "CURRENT_MEASURED_RGBD",
            "mask_measurement_status": "CURRENT_MEASURED_RGBD",
            "bottle_extent_bbox_xyxy": [16, 8, 48, 58],
        },
        "target_clamp": {"left_contact_px": [25, 34], "right_contact_px": [38, 34], "clamp_center_px": [31.5, 34]},
    }
    status = update_pointcloud_from_frame(rgb, depth, intrinsics, report, state=PointCloudRuntimeState(), now=105.0)

    assert status["bottle_roi_source"] == "body_contour"
    assert status["bottle_crop_status"] == "VALID_CONTOUR_CROP"
    assert status["bottle_crop_valid"] is True
    assert status["bottle_mask_provenance_current"] is True
    assert status["bottle_crop_provenance_status"] == "CURRENT_EXACT_RGBD_FRAME"
    assert status["bottle_crop_safe_for_contacts"] is True
    assert 0.0 <= status["bottle_pointcloud_valid_depth_ratio"] <= 1.0


def test_pointcloud_held_shifted_contour_is_not_contact_or_quality_evidence():
    rgb, depth, intrinsics = _rgbd_fixture()
    contour = np.array([[26, 14], [40, 14], [40, 54], [26, 54]], dtype=np.float32)
    report = {
        "frame_id": 14,
        "depth_frame_id": 14,
        "writer_session_id": "test-writer",
        "bottle_geometry": {
            "body_contour": contour,
            "frame_id": None,
            "geometry_frame_id": None,
            "mask_frame_id": None,
            "source_geometry_frame_id": 13,
            "source_mask_frame_id": 13,
            "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
            "mask_measurement_status": "NOT_AVAILABLE_AFTER_GEOMETRIC_SHIFT_USE_SHIFTED_CONTOUR",
            "bottle_extent_bbox_xyxy": [26, 14, 40, 54],
        },
        "target_clamp": {
            "left_contact_px": [26, 34],
            "right_contact_px": [39, 34],
            "clamp_center_px": [32.5, 34],
        },
    }

    status = update_pointcloud_from_frame(
        rgb,
        depth,
        intrinsics,
        report,
        state=PointCloudRuntimeState(),
        now=106.0,
    )

    assert status["bottle_roi_source"] == "body_contour"
    assert status["bottle_mask_frame_id"] == 13
    assert status["bottle_mask_matches_rgb_frame"] is False
    assert status["bottle_mask_provenance_current"] is False
    # A one-frame tolerance may keep visualization/dimension processing
    # usable, but exact-pixel contact and quality evidence must fail closed.
    assert status["pointcloud_snapshot_consistent"] is True
    assert status["bottle_crop_provenance_status"] == (
        "NOT_CURRENT_EXACT_FRAME_CONTACT_QUALITY_BLOCKED"
    )
    assert status["bottle_crop_degraded"] is True
    assert status["bottle_crop_safe_for_contacts"] is False
    assert status["bottle_crop_safe_for_quality_support"] is False
    assert status["contact_3d_valid"] is False


def test_pointcloud_missing_measurement_status_is_not_inferred_current():
    rgb, depth, intrinsics = _rgbd_fixture()
    contour = np.array([[25, 14], [39, 14], [39, 54], [25, 54]], dtype=np.float32)
    report = {
        "frame_id": 15,
        "depth_frame_id": 15,
        "bottle_geometry": {
            "body_contour": contour,
            "frame_id": 15,
            "geometry_frame_id": 15,
            "mask_frame_id": 15,
        },
    }

    status = update_pointcloud_from_frame(
        rgb,
        depth,
        intrinsics,
        report,
        state=PointCloudRuntimeState(),
        now=107.0,
    )

    assert status["bottle_mask_matches_rgb_frame"] is True
    assert status["bottle_geometry_explicitly_current"] is False
    assert status["bottle_mask_provenance_current"] is False
    assert status["bottle_crop_safe_for_contacts"] is False
    assert status["bottle_crop_safe_for_quality_support"] is False


def test_pointcloud_bbox_fallback_is_degraded_and_filters_background():
    rgb, depth, intrinsics = _rgbd_fixture()
    report = {
        "frame_id": 12,
        "depth_frame_id": 12,
        "bottle_extent_bbox_xyxy": [16, 8, 48, 58],
        "bottle_geometry": {"bottle_extent_bbox_xyxy": [16, 8, 48, 58]},
    }
    status = update_pointcloud_from_frame(rgb, depth, intrinsics, report, state=PointCloudRuntimeState(), now=110.0)
    assert status["bottle_roi_source"] == "bbox_fallback"
    assert status["bottle_crop_status"] in {"DEGRADED_BBOX_CROP", "LOW_CONFIDENCE_BBOX_CROP", "INVALID"}
    assert status["bottle_crop_confidence"] < 0.8
    assert status["bottle_raw_point_count"] < status["bottle_roi_pixel_count"]
    assert status["bottle_crop_safe_for_quality_support"] is False


def test_pointcloud_mask_depth_frame_mismatch_downgrades_crop():
    rgb, depth, intrinsics = _rgbd_fixture()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[14:54, 25:39] = True
    report = {
        "frame_id": 20,
        "depth_frame_id": 20,
        "bottle_geometry": {
            "cleaned_mask": mask,
            "frame_id": 7,
            "geometry_frame_id": 7,
            "mask_frame_id": 7,
            "measurement_status": "CURRENT_MEASURED_RGBD",
            "mask_measurement_status": "CURRENT_MEASURED_RGBD",
            "bottle_extent_bbox_xyxy": [25, 14, 39, 54],
        },
    }
    status = update_pointcloud_from_frame(rgb, depth, intrinsics, report, state=PointCloudRuntimeState(), now=120.0)
    assert status["pointcloud_snapshot_consistent"] is False
    assert status["pointcloud_snapshot_rejection_reason"] == "frame_id_mismatch"
    assert status["bottle_crop_status"] == "INVALID"
    assert status["bottle_crop_safe_for_dimensions"] is False
    assert status["bottle_crop_safe_for_contacts"] is False
    assert status["bottle_crop_safe_for_quality_support"] is False
    assert status["cloud_valid_for_dimension_estimation"] is False
    assert status["cloud_valid_for_contact_validation"] is False
    assert status["cloud_valid_for_multiview_input"] is False


def test_robust_geometry_ignores_single_far_outlier():
    rng = np.random.default_rng(3)
    cluster = rng.normal(loc=[0.0, 0.0, 0.55], scale=[0.01, 0.04, 0.008], size=(240, 3)).astype(np.float32)
    points = np.vstack([cluster, np.array([[2.0, 2.0, 2.0]], dtype=np.float32)])
    props = estimate_bottle_3d_properties(points)
    assert props["geometry_valid"] is True
    assert max(props["oriented_extents_mm"]) < 180.0
    assert props["axis_aligned_extents_mm"][0] < 100.0


def test_sensor_bundle_creation_and_event_features():
    now = 1000.0
    events = np.array(
        [
            [now - 0.02, 10, 20, 1],
            [now - 0.08, 12, 24, 0],
            [now + 0.03, 30, 40, 1],
        ],
        dtype=float,
    )
    features = event_features_for_window(events, 0.1)
    assert features["event_count"] == 3
    assert features["event_rate"] == 30.0
    syncer = SensorSynchronizer(max_offset_ms=50, event_window_ms=100)
    bundle = syncer.make_bundle(
        realsense_status={"frame_id": 7, "timestamp": now},
        pointcloud_status={"realsense_pointcloud_active": True, "timestamp": now + 0.01},
        events=events,
        now=now,
    )
    assert bundle.realsense_rgb_frame_id == 7
    assert bundle.davis_event_count == 2
    assert round(bundle.davis_event_window_start, 3) == now - 0.1
    assert bundle.davis_event_window_end == now
    assert bundle.event_features["event_window_source"] == "raw_events"
    assert bundle.event_features["event_window_materialized"] is True
    assert bundle.sync_quality == "good"
    assert syncer.stats(bundle).last_bundle_id == bundle.bundle_id


def test_current_realsense_and_davis_sync_is_good():
    now = 2000.0
    syncer = SensorSynchronizer(max_offset_ms=50, event_window_ms=100)
    bundle = syncer.make_bundle(
        realsense_status={"frame_id": 9, "timestamp": now - 0.01},
        pointcloud_status={"realsense_pointcloud_active": True, "timestamp": now - 0.02},
        davis_report={
            "event_state": "STREAM_ACTIVE_EVENTS",
            "event_count": 4,
            "event_rate": 40,
            "motion_estimate": {"host_timestamp": now - 0.015, "motion_estimation_confidence": 0.8},
            "aps_state": "APS_COLOR_VERIFIED",
            "aps_timestamp": now - 0.02,
        },
        now=now,
    )

    assert bundle.sync_quality == "good"
    assert bundle.bundle_valid_for_benchmark is True
    assert bundle.bundle_valid_for_live_quality is True
    assert bundle.sensor_freshness["davis_events"]["freshness"] == "CURRENT"


def test_sync_prefers_realsense_host_acquisition_times_over_report_write_time():
    now = 2025.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={
            "frame_id": 10,
            "timestamp": now - 0.7,
            "rgb_timestamp": now - 0.012,
            "depth_timestamp": now - 0.018,
        },
        davis_report={
            "event_timestamp_status": "CURRENT",
            "event_host_timestamp": now - 0.015,
        },
        now=now,
    )

    assert bundle.realsense_rgb_timestamp == now - 0.012
    assert bundle.realsense_depth_timestamp == now - 0.018
    assert bundle.freshness_gates["davis_realsense_offset_ms"] < 5.0
    assert bundle.davis_realsense_valid_for_live_quality is True


def test_empty_event_report_does_not_use_analysis_creation_time_as_freshness():
    now = 2050.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={"frame_id": 9, "timestamp": now - 0.01},
        davis_report={
            "event_state": "STREAM_ACTIVE_NO_MOTION",
            "event_count": 0,
            "event_rate": 0.0,
            "raw_window_event_count": 0,
            "event_timestamp_status": "EMPTY",
            "event_host_timestamp": None,
            "motion_estimate": {
                "host_timestamp": now - 0.005,
                "motion_estimation_confidence": 0.0,
            },
        },
        now=now,
    )

    assert bundle.sensor_freshness["davis_events"]["freshness"] == "UNAVAILABLE"
    assert bundle.freshness_gates["davis_events_valid_for_live_quality"] is False


def test_file_sync_can_use_current_in_memory_davis_report(monkeypatch):
    from grasp_project.sync import sensor_sync

    now = time.time()
    writes = {}

    def fake_read(path, default=None):
        name = str(path)
        if name.endswith("latest_realtime_status.json"):
            return {"frame_id": 19, "timestamp": now}
        if name.endswith("pointcloud_status.json"):
            return {}
        raise AssertionError(f"unexpected stale DAVIS file read: {path}")

    monkeypatch.setattr(sensor_sync, "ensure_output_dir", lambda: None)
    monkeypatch.setattr(sensor_sync, "read_json", fake_read)
    monkeypatch.setattr(sensor_sync, "write_json_atomic", lambda path, value: writes.setdefault(str(path), value))
    bundle, status = sensor_sync.run_file_sync_once(
        davis_report={
            "event_state": "STREAM_ACTIVE_EVENTS",
            "event_count": 17,
            "event_rate": 170.0,
            "motion_estimate": {
                "host_timestamp": now,
                "motion_estimation_confidence": 0.9,
            },
        }
    )

    assert bundle.davis_event_count == 0
    assert bundle.sensor_freshness["davis_events"]["freshness"] == "CURRENT"
    assert bundle.davis_event_window_start is None
    assert bundle.davis_event_window_end is None
    assert bundle.event_features["cached_report_event_count"] == 17
    assert bundle.event_features["event_window_source"] == "cached_report_diagnostic_only"
    assert status["davis_event_window_count"] == 0
    assert status["cached_report_event_count"] == 17
    assert status["davis_event_window_materialized"] is False
    assert writes


def test_cached_report_diagnostics_preserve_explicit_zero_values():
    now = 2075.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={"frame_id": 9, "timestamp": now},
        davis_report={
            "event_count": 0,
            "post_grasp_event_count": 9,
            "event_rate": 0.0,
            "davis_event_rate": 90.0,
            "motion_estimation_confidence": 0.0,
            "motion_estimate": {
                "host_timestamp": now,
                "motion_estimation_confidence": 0.8,
            },
        },
        now=now,
    )

    assert bundle.davis_event_count == 0
    assert bundle.event_features["cached_report_event_count"] == 0
    assert bundle.event_features["cached_report_event_rate"] == 0.0
    assert bundle.sensor_freshness["davis_events"]["confidence"] == 0.0


def test_stale_optional_pointcloud_does_not_veto_fresh_davis_pair():
    now = 2100.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={"frame_id": 10, "timestamp": now - 0.01},
        pointcloud_status={"realsense_pointcloud_active": True, "timestamp": now - 5.0},
        davis_report={
            "event_state": "STREAM_ACTIVE_EVENTS",
            "event_count": 12,
            "motion_estimate": {"host_timestamp": now - 0.02, "motion_estimation_confidence": 0.8},
        },
        now=now,
    )

    assert bundle.max_sensor_time_offset_ms >= 900.0
    assert bundle.sensor_freshness["pointcloud"]["freshness"] == "STALE"
    assert bundle.sync_quality == "offset_too_large"
    assert bundle.bundle_valid_for_live_quality is False
    assert bundle.davis_realsense_sync_quality == "good"
    assert bundle.davis_realsense_valid_for_live_quality is True
    assert bundle.freshness_gates["davis_realsense_sync_quality"] == "good"
    assert bundle.freshness_gates["davis_events_valid_for_live_quality"] is True
    assert bundle.freshness_gates["pointcloud_valid_for_live_quality"] is False
    assert bundle.freshness_gates["davis_realsense_offset_ms"] <= 20.0


def test_large_davis_realsense_offset_still_rejects_event_quality():
    now = 2200.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={"frame_id": 11, "timestamp": now},
        davis_report={
            "event_state": "STREAM_ACTIVE_EVENTS",
            "event_count": 12,
            "motion_estimate": {"host_timestamp": now - 0.3, "motion_estimation_confidence": 0.9},
        },
        now=now,
    )

    assert bundle.freshness_gates["davis_events_valid_for_live_quality"] is False
    assert bundle.freshness_gates["davis_realsense_offset_ms"] >= 300.0


def test_analyzer_selected_fallback_window_keeps_async_davis_current_without_cross_camera_claim():
    now = 2250.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={
            "frame_id": 12,
            "rgb_timestamp": now - 0.18,
            "depth_timestamp": now - 0.19,
        },
        davis_report={
            "event_stream_status": "active",
            "event_health_valid": True,
            "event_timestamp_status": "CURRENT",
            "event_timestamp_lag_ms": 700.0,
            "event_host_timestamp": now - 0.70,
            "event_window_ms": 800,
            "event_window_mode": "fallback_800ms",
            "event_count": 0,
            "event_rate": 0.0,
        },
        now=now,
    )

    davis_freshness = bundle.sensor_freshness["davis_events"]
    assert davis_freshness["freshness"] == "CURRENT"
    assert davis_freshness["accepted_window_ms"] == 800.0
    assert davis_freshness["freshness_source"] == "wall_clock_event_age_with_analyzer_bound"
    assert bundle.freshness_gates["davis_events_valid_for_live_quality"] is True
    assert bundle.davis_realsense_valid_for_live_quality is False


def test_cached_current_davis_declaration_cannot_revive_frozen_event_timestamp():
    now = 2260.0
    bundle = SensorSynchronizer(max_offset_ms=50, event_window_ms=100).make_bundle(
        realsense_status={"frame_id": 13, "rgb_timestamp": now - 0.02},
        davis_report={
            "event_timestamp_status": "CURRENT",
            "event_timestamp_lag_ms": 50.0,
            "event_host_timestamp": now - 5.0,
            "event_window_ms": 800,
            "event_window_mode": "fallback_800ms",
            "event_count": 20,
            "event_rate": 25.0,
        },
        now=now,
    )

    freshness = bundle.sensor_freshness["davis_events"]
    assert freshness["freshness"] == "STALE"
    assert freshness["age_ms"] >= 5000.0
    assert freshness["valid"] is False
    assert bundle.freshness_gates["davis_events_valid_for_live_quality"] is False
    assert bundle.davis_realsense_sync_quality == "stale_davis_events"


def test_action_anchored_event_window_uses_action_time_range():
    now = 3000.0
    action = now - 0.5
    events = np.array(
        [
            [action + 0.05, 10, 20, 1],
            [action + 0.45, 12, 24, 0],
            [action + 1.1, 30, 40, 1],
            [now + 2.0, 100, 100, 1],
        ],
        dtype=float,
    )
    bundle = SensorSynchronizer(max_offset_ms=200).make_bundle(
        realsense_status={"frame_id": 10, "timestamp": action + 0.1},
        davis_report={"close_command_time": action},
        events=events,
        now=now,
        sync_mode=SYNC_MODE_ACTION_ANCHORED_WINDOW,
        action_timestamp=action,
    )

    assert bundle.sync_mode == SYNC_MODE_ACTION_ANCHORED_WINDOW
    assert bundle.davis_event_window_start == action
    assert bundle.davis_event_window_end == now
    assert bundle.davis_event_count == 2
    assert bundle.event_features["action_event_windows"]["impact"] == [round(action, 6), round(action + 0.3, 6)]
    assert bundle.event_features["action_event_window_planned_end"] == round(action + 1.2, 6)
    assert bundle.event_features["action_event_window_complete"] is False


def test_display_latest_mode_does_not_claim_strict_synchronization():
    now = 4000.0
    bundle = SensorSynchronizer().make_bundle(
        realsense_status={"frame_id": 11, "timestamp": now},
        pointcloud_status={"realsense_pointcloud_active": True, "timestamp": now},
        now=now,
        sync_mode=SYNC_MODE_DISPLAY_LATEST,
    )

    assert bundle.sync_quality == "display_latest"
    assert bundle.bundle_valid_for_benchmark is False
    assert bundle.bundle_valid_for_live_quality is False
    assert "display_latest_no_strict_fusion_claim" in bundle.sync_warnings


def test_software_timestamp_bundle_rejects_excessive_offset():
    now = 5000.0
    bundle = SensorSynchronizer(max_offset_ms=50).make_bundle(
        realsense_status={"frame_id": 12, "timestamp": now},
        pointcloud_status={"realsense_pointcloud_active": True, "timestamp": now - 1.0},
        now=now,
    )

    assert bundle.sync_quality == "offset_too_large"
    assert bundle.bundle_valid_for_benchmark is False
    assert "offset_too_large" in bundle.sync_warnings


def test_missing_timestamp_creates_sync_warning():
    now = 6000.0
    bundle = SensorSynchronizer().make_bundle(
        realsense_status={"frame_id": 13, "target_status": "TRACKING"},
        now=now,
    )

    assert "missing_timestamp:realsense" in bundle.sync_warnings
    assert bundle.sensor_freshness["realsense_rgb"]["freshness"] == "UNAVAILABLE"


def test_embodiment_metric_applicability_defaults():
    status = embodiment_status()
    assert status["active_embodiment"] in DEFAULT_MODELS
    two_point = quality_metric_applicability("two_point_parallel_gripper")
    assert two_point["primary_quality_metric"] == "epsilon_force_closure"
    vacuum = quality_metric_applicability("vacuum_grip")
    assert vacuum["quality_metric_applicability"] == "epsilon_not_primary"
    assert "vacuum_grip" in vacuum["quality_not_applicable_reason"]


def test_homography_estimation_from_four_points():
    pts = [[0, 0], [1, 0], [1, 1], [0, 1]]
    out = [[10, 20], [20, 20], [20, 30], [10, 30]]
    result = estimate_homography(pts, out)
    assert result["reprojection_error_px"] < 1e-4
    assert len(result["homography"]) == 3


def test_homography_status_requires_a_recorded_passing_error(tmp_path):
    path = tmp_path / "homography.json"
    extrinsics = tmp_path / "missing_extrinsics.json"
    path.write_text('{"homography": [[1,0,0],[0,1,0],[0,0,1]]}')

    missing_error = calibration_status(path, extrinsics)

    assert missing_error["homography_available"] is True
    assert missing_error["usable_for_overlay"] is False
    assert missing_error["alignment_status"] == "approximate_side_by_side_only"
    assert missing_error["homography_validation_status"] == "missing_or_invalid_reprojection_error"

    path.write_text(
        '{"homography": [[1,0,0],[0,1,0],[0,0,1]], "reprojection_error_px": 0.5}'
    )
    missing_sizes = calibration_status(path, extrinsics)
    assert missing_sizes["usable_for_overlay"] is False
    assert missing_sizes["homography_validation_status"] == "missing_frame_size_metadata"

    path.write_text(
        '{"homography": [[1,0,0],[0,1,0],[0,0,1]], "reprojection_error_px": 0.5, '
        '"davis_frame_size": [346, 260], "realsense_frame_size": [640, 480]}'
    )
    passing = calibration_status(path, extrinsics)
    assert passing["usable_for_overlay"] is True
    assert passing["alignment_status"] == "calibrated_2d"


def test_calibration_cli_requires_frame_size_metadata(tmp_path):
    input_path = tmp_path / "points.json"
    output_path = tmp_path / "homography.json"
    input_path.write_text(
        '{"davis_points": [[0,0],[1,0],[1,1],[0,1]], '
        '"realsense_points": [[10,20],[20,20],[20,30],[10,30]]}'
    )

    result = calibrate_2d_main(
        ["--input", str(input_path), "--output", str(output_path)]
    )

    assert result == 2
    assert output_path.exists() is False


def test_calibration_cli_records_direction_and_frame_sizes(tmp_path):
    input_path = tmp_path / "points.json"
    output_path = tmp_path / "homography.json"
    input_path.write_text(
        '{"davis_points": [[0,0],[1,0],[1,1],[0,1]], '
        '"realsense_points": [[10,20],[20,20],[20,30],[10,30]], '
        '"davis_frame_size": [346,260], "realsense_frame_size": [640,480]}'
    )

    result = calibrate_2d_main(
        ["--input", str(input_path), "--output", str(output_path)]
    )
    payload = json.loads(output_path.read_text())

    assert result == 0
    assert payload["transform_direction"] == "davis_to_realsense"
    assert payload["davis_frame_size"] == [346, 260]
    assert payload["realsense_frame_size"] == [640, 480]

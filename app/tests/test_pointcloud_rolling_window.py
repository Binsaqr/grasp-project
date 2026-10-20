"""Camera-free safety tests for current and rolling point-cloud lifecycle."""

import time

import numpy as np

from grasp_project.dashboard.web_dashboard import summarize_pointcloud
from grasp_project.realsense.pointcloud import (
    PointCloudRuntimeState,
    RollingObjectCloudAccumulator,
    _bbox_from_report,
    pointcloud_live_json_payload,
    pointcloud_status_from_state,
    update_pointcloud_from_frame,
)


def _points(count=8, offset=0.0):
    values = np.linspace(-0.04, 0.04, count, dtype=np.float32)
    points = np.stack([values + offset, values * 0.5, np.full(count, 0.5, dtype=np.float32)], axis=1)
    colors = np.full((count, 3), [0.2, 0.5, 0.9], dtype=np.float32)
    return points, colors


def _add(accumulator, timestamp, *, target_id="bottle-A", **overrides):
    points, colors = _points()
    kwargs = {
        "timestamp": timestamp,
        "frame_id": int(timestamp),
        "target_id": target_id,
        "target_id_scope": "LOCK_PERSISTENT",
        "target_valid": True,
        "valid_depth_ratio": 0.9,
        "crop_valid": True,
        "snapshot_consistent": True,
    }
    kwargs.update(overrides)
    return accumulator.add_observation(points, colors, **kwargs)


def test_pointcloud_bbox_converts_xywh_to_xyxy():
    assert _bbox_from_report({"bottle_geometry": {"bbox_xywh": [10, 20, 30, 40]}}) == [
        10,
        20,
        40,
        60,
    ]


def test_rolling_window_prunes_observations_older_than_30_seconds():
    accumulator = RollingObjectCloudAccumulator(window_sec=30.0, max_points=100, voxel_size_m=0.0)

    _add(accumulator, 0.0)
    _add(accumulator, 10.0)
    status = _add(accumulator, 31.0)

    assert status["rolling_pointcloud_observation_count"] == 2
    assert status["rolling_pointcloud_observation_span_sec"] == 21.0
    assert status["rolling_pointcloud_age_sec"] == 0.0


def test_rolling_window_enforces_point_cap_without_raw_frame_fusion():
    accumulator = RollingObjectCloudAccumulator(window_sec=30.0, max_points=5, voxel_size_m=0.0)

    status = _add(accumulator, 1.0)

    assert status["rolling_pointcloud_point_count"] == 5
    assert status["rolling_pointcloud_merged_point_count"] == 0
    assert status["rolling_pointcloud_mode"] == "NOT_REGISTERED_DIAGNOSTIC"
    assert status["rolling_pointcloud_valid_for_geometry"] is False
    assert status["rolling_pointcloud_valid_for_contact"] is False
    assert status["rolling_pointcloud_valid_for_benchmark"] is False
    assert status["rolling_window_active"] is True
    assert status["rolling_cloud_window_sec"] == 30.0
    assert status["rolling_valid_for_geometry"] is False


def test_rolling_window_resets_before_accepting_a_different_target():
    accumulator = RollingObjectCloudAccumulator(window_sec=30.0, max_points=100, voxel_size_m=0.0)
    _add(accumulator, 1.0, target_id="bottle-A")

    status = _add(accumulator, 2.0, target_id="bottle-B")

    assert status["rolling_pointcloud_last_reset_reason"] == "target_changed"
    assert status["rolling_pointcloud_target_id"] == "bottle-B"
    assert status["rolling_pointcloud_observation_count"] == 1


def test_rolling_window_resets_and_rejects_moved_bottle():
    accumulator = RollingObjectCloudAccumulator(window_sec=30.0, max_points=100, voxel_size_m=0.0)
    _add(accumulator, 1.0)

    status = _add(accumulator, 2.0, bottle_moved=True)

    assert status["rolling_pointcloud_active"] is False
    assert status["rolling_pointcloud_observation_count"] == 0
    assert status["rolling_pointcloud_last_reset_reason"] == "bottle_moved"


def test_rolling_window_resets_and_rejects_invalid_registration():
    accumulator = RollingObjectCloudAccumulator(window_sec=30.0, max_points=100, voxel_size_m=0.0)
    _add(accumulator, 1.0)

    status = _add(accumulator, 2.0, registration_status="REGISTRATION_FAILED")

    assert status["rolling_pointcloud_observation_count"] == 0
    assert status["rolling_pointcloud_last_reset_reason"] == "invalid_registration"
    assert status["rolling_pointcloud_registration_status"] == "NOT_REGISTERED_DIAGNOSTIC"


def test_rolling_window_rejects_low_depth_and_unvalidated_roi():
    accumulator = RollingObjectCloudAccumulator(window_sec=30.0, max_points=100, voxel_size_m=0.0)

    low_depth = _add(accumulator, 1.0, valid_depth_ratio=0.1)
    bad_roi = _add(accumulator, 2.0, crop_valid=False)

    assert low_depth["rolling_pointcloud_observation_accepted"] is False
    assert bad_roi["rolling_pointcloud_observation_count"] == 0
    assert bad_roi["rolling_pointcloud_last_rejection_reason"] == "validated_bottle_crop_required"


def test_missing_current_cloud_is_unavailable_not_current():
    status = pointcloud_status_from_state(PointCloudRuntimeState(), now=100.0)

    assert status["pointcloud_freshness"] == "UNAVAILABLE"
    assert status["pointcloud_current"] is False
    assert status["cloud_valid_for_dimension_estimation"] is False
    assert status["cloud_valid_for_contact_validation"] is False


def test_dashboard_does_not_treat_status_write_timestamp_as_cloud_timestamp():
    summary = summarize_pointcloud(
        {"timestamp": time.time(), "realsense_pointcloud_active": False},
        backend_running=True,
    )

    assert summary["freshness"] == "UNAVAILABLE"
    assert summary["pointcloud_current"] is False


def test_live_json_requires_full_3d_contact_and_cloud_validity():
    points, colors = _points()
    status = {
        "contact_depth_valid": True,
        "contact_3d_valid": True,
        "cloud_valid_for_contact_validation": False,
    }

    rejected = pointcloud_live_json_payload(points, colors, status=status)
    status["cloud_valid_for_contact_validation"] = True
    accepted = pointcloud_live_json_payload(points, colors, status=status)

    assert rejected["contacts"]["valid"] is False
    assert accepted["contacts"]["valid"] is True


def test_live_update_exposes_unregistered_rolling_diagnostics():
    size = 64
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    depth = np.full((size, size), 900, dtype=np.uint16)
    depth[14:54, 25:39] = 500
    mask = np.zeros((size, size), dtype=bool)
    mask[14:54, 25:39] = True
    intrinsics = {"fx": 120.0, "fy": 120.0, "cx": 32.0, "cy": 32.0, "width": size, "height": size}
    report = {
        "frame_id": 11,
        "depth_frame_id": 11,
        "target_status": "TRACKING",
        "locked_target_valid": True,
        "pointcloud_target_id": "bottle-A",
        "pointcloud_target_id_scope": "LOCK_PERSISTENT",
        "locked_target": {"valid": True, "bottle_target_id": "bottle-A"},
        "bottle_geometry": {
            "cleaned_mask": mask,
            "mask_frame_id": 11,
            "bottle_extent_bbox_xyxy": [25, 14, 39, 54],
        },
    }

    status = update_pointcloud_from_frame(
        rgb,
        depth,
        intrinsics,
        report,
        state=PointCloudRuntimeState(),
        now=100.0,
    )

    assert status["rolling_pointcloud_active"] is True
    assert status["rolling_pointcloud_observation_count"] == 1
    assert status["rolling_pointcloud_registration_status"] == "NOT_REGISTERED_DIAGNOSTIC"
    assert status["rolling_pointcloud_valid_for_geometry"] is False
    assert status["rolling_pointcloud_valid_for_contact"] is False
    assert status["rolling_pointcloud_valid_for_benchmark"] is False


def test_minimal_live_report_preserves_explicit_persistent_target_evidence():
    from grasp_project.realsense.realtime_runner import minimal_pointcloud_report

    compact = minimal_pointcloud_report(
        {
            "frame_id": 7,
            "depth_frame_id": 7,
            "target_lock": {
                "target_status": "TRACKING",
                "locked_target": {
                    "valid": True,
                    "bottle_target_id": "bottle-A",
                    "lock_frame_id": 4,
                    "locked_bottle_geometry": {"target_id": "bottle-A", "target_id_scope": "LOCK_PERSISTENT"},
                },
            },
        }
    )

    assert compact["target_status"] == "TRACKING"
    assert compact["locked_target_valid"] is True
    assert compact["pointcloud_target_id"] == "bottle-A"
    assert compact["pointcloud_target_id_scope"] == "LOCK_PERSISTENT"
    assert compact["target_lock"]["locked_target"]["valid"] is True

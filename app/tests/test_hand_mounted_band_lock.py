from pathlib import Path
from types import SimpleNamespace

import pytest

from grasp_project.common.config import load_named_config
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.realtime_runner import (
    HAND_MOUNTED_CAMERA_MODE,
    MAX_BAND_CENTER_DRIFT_PX,
    EvaluationWorker,
    TargetLockManager,
    attach_grasp_band_projection_residual,
    attach_hand_mounted_camera_status,
    build_lock_transition_event,
    compact_live_status,
    hand_mounted_stage_from_report,
    locked_grasp_band_snapshot,
    locked_grasp_band_status,
)


def locked_target():
    return {
        "valid": True,
        "bottle_target_id": "bottle-1",
        "lock_frame_id": 12,
        "lock_timestamp": "2026-08-06T10:00:00Z",
        "attempt_id": "attempt-12",
        "selected_candidate_id": "body_fraction_0.500",
        "locked_body_fraction": 0.5,
        "lock_confidence": 88.0,
        "locked_grip_width_mm": 62.0,
        "locked_bottle_geometry": {
            "bbox_xyxy": [70, 20, 130, 220],
            "center_px": [100, 120],
            "axis_vector": [0, 1],
            "axis_length_px": 200,
            "depth_median_mm": 400,
            "image_bottle_hull_px": [[70, 20], [130, 20], [130, 220], [70, 220]],
        },
        "fixed_clamp_zone": {
            "clamp_center_px": [100, 120],
            "clamp_band_width_px": 60,
            "clamp_band_height_px": 40,
            "left_clamp_patch": {"center_px": [70, 120], "polygon": [[65, 100], [75, 100], [75, 140], [65, 140]]},
            "right_clamp_patch": {"center_px": [130, 120], "polygon": [[125, 100], [135, 100], [135, 140], [125, 140]]},
        },
        "bottle_contact_model": {
            "body_fraction": 0.5,
            "planned_depth_mm": 400,
            "grip_width_px": 60,
            "grip_width_mm": 62,
            "epsilon_result": {"epsilon": 0.04, "force_closure": True},
        },
        "epsilon_value_at_lock": 0.04,
        "force_closure_at_lock": True,
    }


def projected_tracking(**updates):
    tracking = {
        "tracking_confidence": 90.0,
        "tracking_ok": True,
        "identity_tracking_ok": True,
        "locked_patch_visible": True,
        "locked_body_fraction": 0.5,
        "current_grasp_center_px": [250, 120],
        "current_depth_mm": 250,
        "grasp_band_projection_error_px": 2.0,
        "contact_model_projection_status": "PROJECTED",
        "target_clamp": {
            "current_frame_projection": True,
            "clamp_center_px": [250, 120],
            "clamp_band_width_px": 120,
            "patch_axis_direction_px": [0, 1],
            "clamp_band_polygon": [[190, 90], [310, 90], [310, 150], [190, 150]],
        },
        "clamp_alignment": {
            "contact_model_center_to_clamp_center_error_px": 2.0,
            "band_inside_frame_ratio": 1.0,
            "band_mask_overlap": 0.9,
        },
    }
    tracking.update(updates)
    return tracking


def test_hand_mounted_config_has_required_uncalibrated_camera_frame_contract():
    config = load_named_config("hand_mounted_camera.yaml")

    assert HAND_MOUNTED_CAMERA_MODE is True
    assert config["enabled"] is True
    assert config["mount_location"] == "forearm_behind_wrist"
    assert config["camera_orientation"] == "forward_over_hand"
    assert config["camera_frame"] == "realsense_color_optical_frame"
    assert config["end_effector_frame"] == "hand_or_gripper_frame"
    assert config["use_camera_frame_guidance"] is True
    assert config["require_robot_base_calibration"] is False


def test_locked_band_snapshot_is_bottle_relative_and_schema_complete():
    snapshot = locked_grasp_band_snapshot(locked_target())

    assert snapshot["locked_band_id"] == "attempt-12:body_fraction_0.500"
    assert snapshot["locked_body_fraction_center"] == 0.5
    assert snapshot["locked_body_fraction_start"] == pytest.approx(0.4)
    assert snapshot["locked_body_fraction_end"] == pytest.approx(0.6)
    assert snapshot["coordinate_model"] == "LEGACY_2D_REFERENCE_NOT_A_3D_LOCK"
    assert snapshot["locked_contact_centers_px"] == {"left": [70.0, 120.0], "right": [130.0, 120.0]}
    assert snapshot["locked_quality_label"] == "OK"
    assert snapshot["lock_valid"] is False
    assert snapshot["lock_invalid_reason"] == "legacy_lock_missing_bottle_local_3d_geometry"
    assert snapshot["true_3d_lock"] is False


def test_expected_camera_translation_is_displacement_not_projection_drift():
    status = locked_grasp_band_status(
        locked_target(),
        projected_tracking(),
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["grasp_band_image_displacement_px"] == pytest.approx(150.0)
    assert status["grasp_band_drift_px"] == pytest.approx(2.0)
    assert status["grasp_band_scale_change"] == pytest.approx(2.0)
    assert status["band_drift_state"] == "OK"
    assert status["grasp_band_relock_required"] is False


def test_expected_image_displacement_uses_locked_band_center_not_bottle_center():
    target = locked_target()
    target["fixed_clamp_zone"]["clamp_center_px"] = [100, 160]
    target["fixed_clamp_zone"]["left_clamp_patch"]["center_px"] = [70, 160]
    target["fixed_clamp_zone"]["right_clamp_patch"]["center_px"] = [130, 160]
    status = locked_grasp_band_status(
        target,
        projected_tracking(current_grasp_center_px=[250, 160]),
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["grasp_band_image_displacement_px"] == pytest.approx(150.0)
    assert status["grasp_band_drift_px"] == pytest.approx(2.0)


def test_runtime_projection_residual_compares_independent_current_observations():
    tracking = {}
    residual = attach_grasp_band_projection_residual(
        tracking,
        {"clamp_center_px": [241.0, 120.0]},
        {"clamp_center_px": [250.0, 120.0]},
        observation_current=True,
        geometry_measurement_current=True,
    )

    assert residual == pytest.approx(9.0)
    assert tracking["grasp_band_projection_error_px"] == pytest.approx(9.0)
    assert tracking["grasp_band_projection_residual_valid"] is True
    status = locked_grasp_band_status(
        locked_target(),
        {**projected_tracking(), **tracking},
        target_status="TRACKING",
        lock_state="LOCKED",
    )
    assert status["grasp_band_drift_px"] == pytest.approx(9.0)


def test_projection_residual_is_unknown_when_either_observation_is_not_current():
    tracking = {}
    residual = attach_grasp_band_projection_residual(
        tracking,
        {"clamp_center_px": [241.0, 120.0]},
        {"clamp_center_px": [250.0, 120.0]},
        observation_current=False,
        geometry_measurement_current=True,
    )

    assert residual is None
    assert tracking["grasp_band_projection_error_px"] is None
    assert tracking["grasp_band_projection_residual_valid"] is False
    assert tracking["grasp_band_projection_residual_reason"] == "local_band_observation_not_current"


def test_expected_approach_depth_change_is_progress_not_depth_drift():
    status = locked_grasp_band_status(
        locked_target(),
        projected_tracking(current_depth_mm=40),
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["grasp_band_depth_change_mm"] == 360
    assert status["grasp_band_depth_residual_mm"] is None
    assert status["band_drift_state"] == "OK"
    assert status["grasp_band_relock_required"] is False


def test_large_projection_residual_is_stale_but_preserves_the_owned_lock():
    tracking = projected_tracking(grasp_band_projection_error_px=MAX_BAND_CENTER_DRIFT_PX + 1.0)
    status = locked_grasp_band_status(
        locked_target(),
        tracking,
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["band_drift_state"] == "STALE"
    assert status["grasp_band_stale"] is True
    assert status["grasp_band_current"] is False
    assert status["grasp_band_relock_required"] is False
    assert "band_center_drift_exceeded" in status["band_drift_reason"]


def test_projected_band_keeps_locked_fraction_when_live_candidate_would_change():
    tracking = projected_tracking()
    tracking["new_candidate_body_fraction"] = 0.62
    status = locked_grasp_band_status(
        locked_target(),
        tracking,
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["current_projected_grasp_band"]["body_fraction"] == 0.5
    assert status["grasp_band_body_fraction_drift"] == 0.0
    assert status["grasp_band_relock_required"] is False


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ({}, "SEARCH_TARGET"),
        ({"full_bottle_visible": True}, "VERIFY_FULL_BOTTLE"),
        ({"selected_candidate_id": "candidate-1"}, "SELECT_GRASP_BAND"),
        ({"target_lock": {"grasp_band_locked": True}, "target_locked": True}, "APPROACH_TRACKING"),
        ({"target_lock": {"grasp_band_locked": True}, "distance_to_bottle_mm": 120, "distance_measurement_valid": False}, "APPROACH_TRACKING"),
        ({"target_lock": {"grasp_band_locked": True}, "distance_to_bottle_mm": 120, "distance_measurement_valid": True}, "CLOSE_PROXIMITY_TRACKING"),
        ({"target_lock": {"grasp_band_locked": True}, "target_locked": True, "in_close_range": True}, "CLOSE_PROXIMITY_TRACKING"),
        ({"ready_to_close": True, "target_lock": {"grasp_band_locked": True}}, "READY_TO_GRASP"),
        ({"post_grasp_active": True}, "POST_GRASP_MONITORING"),
        ({"target_lock": {"grasp_band_relock_required": True}}, "RELOCK_REQUIRED"),
    ],
)
def test_hand_mounted_stage_contract(report, expected):
    assert hand_mounted_stage_from_report(report) == expected


def test_hand_status_says_camera_frame_and_never_claims_robot_base():
    report = {
        "target_locked": True,
        "distance_to_bottle_mm": 210,
        "error_x_px": 8,
        "error_y_px": -3,
        "target_lock": {
            "grasp_band_locked": True,
            "grasp_band_projected": True,
            "locked_patch_visible": True,
            "locked_patch_visible_reason": "current_mask_support",
            "band_drift_state": "OK",
        },
    }

    attach_hand_mounted_camera_status(report)

    assert report["hand_mounted_mode_label"] == (
        "Hand-mounted camera mode: hand-reach guidance (camera frame, approximate)"
    )
    assert report["guidance_frame"] == "camera_frame"
    assert report["robot_base_calibrated"] is False
    assert report["robot_base_coordinates_available"] is False
    assert report["partial_view_allowed"] is True
    assert report["full_bottle_required"] is False
    assert report["approach_tracking_active"] is True
    assert report["target_error_x_px"] == 8
    assert report["locked_band_visible"] is True
    assert report["locked_band_current_rgbd_supported"] is True


def test_held_last_valid_3d_band_remains_operator_visible_but_not_currently_supported():
    report = {
        "target_locked": True,
        "target_lock": {
            "grasp_band_locked": True,
            "grasp_band_projected": True,
            "grasp_band_held": True,
            "projection_mode": "HELD_LAST_VALID_3D",
            "locked_patch_visible": False,
            "locked_patch_visible_reason": "current_depth_support_missing",
        },
    }

    attach_hand_mounted_camera_status(report)

    assert report["locked_band_visible"] is True
    assert report["locked_band_current_rgbd_supported"] is False
    assert report["locked_band_visibility_reason"] == "held_last_valid_3d_display_only"


def test_current_bottle_relative_visual_guide_is_visible_without_safety_claims():
    report = {
        "target_locked": True,
        "target_lock": {
            "grasp_band_locked": True,
            "grasp_band_projected": False,
            "projection_mode": "INVALID_3D",
            "projection_valid": False,
            "locked_patch_visible": False,
            "current_tracked_target": {
                "operator_locked_fraction_visual_guide_current": True,
                "display_target_clamp": {
                    "clamp_center_px": [160.0, 120.0],
                    "clamp_band_polygon": [[140, 110], [180, 110], [180, 130], [140, 130]],
                },
            },
        },
    }

    attach_hand_mounted_camera_status(report)

    assert report["locked_band_visible"] is True
    assert report["operator_band_state"] == "CURRENT_BOTTLE_RELATIVE_GUIDE"
    assert report["operator_locked_fraction_visual_guide_current"] is True
    assert report["operator_band_is_safety_measurement"] is False
    assert report["locked_band_current_rgbd_supported"] is False
    assert report["projection_valid"] is False
    assert report["locked_band_visibility_reason"] == (
        "current_identity_validated_bottle_edges_at_immutable_body_fraction"
    )


def test_compact_status_preserves_hand_band_safety_fields():
    report = {
        "state": "APPROACH_TRACKING",
        "target_lock": {
            "target_status": "TRACKING",
            "lock_state": "LOCKED",
            "locked_grasp_band": locked_grasp_band_snapshot(locked_target()),
            "grasp_band_locked": True,
            "grasp_band_projected": True,
            "grasp_band_current": False,
            "grasp_band_held": True,
            "grasp_band_projection_unavailable": False,
            "grasp_band_relock_required": False,
            "band_drift_state": "LOCKED_HELD",
        },
        "hand_mounted_camera_mode": True,
        "hand_mounted_stage": "APPROACH_TRACKING",
        "grasp_band_locked": True,
        "grasp_band_projected": True,
        "grasp_band_current": False,
        "grasp_band_held": True,
        "grasp_band_projection_unavailable": False,
        "grasp_band_relock_required": False,
        "band_drift_state": "LOCKED_HELD",
    }

    compact = compact_live_status(report)

    assert compact["hand_mounted_stage"] == "APPROACH_TRACKING"
    assert compact["grasp_band_locked"] is True
    assert compact["grasp_band_held"] is True
    assert compact["grasp_band_projection_unavailable"] is False
    assert compact["target_lock"]["locked_grasp_band"]["locked_body_fraction_center"] == 0.5
    assert compact["target_lock"]["band_drift_state"] == "LOCKED_HELD"


def test_completed_quality_metadata_is_attached_to_the_same_locked_band(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    target = locked_target()
    target["locked_grasp_band"] = locked_grasp_band_snapshot(target)
    worker = SimpleNamespace(
        lock_manager=SimpleNamespace(locked_target=target),
        latest_report={
            "writer_session_id": "writer-1",
            "target_lock": {"locked_target": dict(target)},
        },
    )

    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-1",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-12",
            "quality_compute_id": "qm-12",
            "quality_label": "GOOD",
            "raw_epsilon": 0.055,
            "normalized_quality_score": 80.0,
            "epsilon_force_closure": True,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    )

    band = target["locked_grasp_band"]
    assert band["locked_quality_compute_id"] == "qm-12"
    assert band["locked_quality_label"] == "GOOD"
    assert band["locked_epsilon"] == 0.055
    assert band["analytic_quality_status"] == "CURRENT"


def test_first_quality_result_is_rejected_without_exact_baseline_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    target = locked_target()
    target["locked_grasp_band"] = locked_grasp_band_snapshot(target)
    worker = SimpleNamespace(
        lock_manager=SimpleNamespace(locked_target=target),
        latest_report={
            "writer_session_id": "writer-1",
            "target_lock": {"locked_target": dict(target)},
        },
    )

    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-1",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-12",
            "quality_compute_id": "qm-invalid-contract-first",
            "quality_label": "WEAK",
            "raw_epsilon": 0.0,
            "quality_baseline_role": "MANUAL_COMPARISON",
            "quality_input_source": "detected_finger",
            "quality_contact_source_forced": False,
            "baseline_frozen": False,
        },
    )
    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-wrong",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-wrong",
            "quality_compute_id": "qm-invalid-lineage-first",
            "quality_label": "GOOD",
            "raw_epsilon": 0.126,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    )

    assert target.get("locked_quality_compute_id") is None
    assert target["locked_grasp_band"].get("locked_quality_compute_id") is None


def test_weak_zero_quality_result_cannot_become_immutable_demo_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    target = locked_target()
    target["locked_grasp_band"] = locked_grasp_band_snapshot(target)
    worker = SimpleNamespace(
        lock_manager=SimpleNamespace(locked_target=target),
        latest_report={
            "writer_session_id": "writer-1",
            "target_lock": {"locked_target": dict(target)},
        },
    )

    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-1",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-12",
            "quality_compute_id": "qm-zero",
            "quality_label": "WEAK",
            "raw_epsilon": 0.0,
            "normalized_quality_score": 0.0,
            "epsilon_force_closure": False,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    )

    assert target.get("locked_quality_compute_id") is None
    assert target["locked_grasp_band"].get("locked_quality_compute_id") is None


def test_completed_quality_baseline_cannot_be_overwritten_by_late_same_lock_result(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    target = locked_target()
    target["locked_grasp_band"] = locked_grasp_band_snapshot(target)
    worker = SimpleNamespace(
        lock_manager=SimpleNamespace(locked_target=target),
        latest_report={
            "writer_session_id": "writer-1",
            "target_lock": {"locked_target": dict(target)},
        },
    )
    baseline = {
        "quality_matrix_status": "updated",
        "writer_session_id": "writer-1",
        "lock_frame_id": 12,
        "lock_attempt_id": "attempt-12",
        "quality_compute_id": "qm-initial",
        "quality_label": "GOOD",
        "raw_epsilon": 0.126,
        "normalized_quality_score": 100.0,
        "epsilon_force_closure": True,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
    }

    EvaluationWorker.attach_locked_quality_snapshot(worker, baseline)
    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            **baseline,
            "quality_compute_id": "qm-late-finger",
            "raw_epsilon": 0.0,
            "quality_label": "WEAK",
            "quality_input_source": "detected_finger",
            "quality_contact_source_forced": False,
        },
    )
    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            **baseline,
            "writer_session_id": "writer-other",
            "lock_attempt_id": "attempt-other",
            "quality_compute_id": "qm-wrong-lineage",
        },
    )

    band = target["locked_grasp_band"]
    assert band["locked_quality_compute_id"] == "qm-initial"
    assert band["locked_epsilon"] == pytest.approx(0.126)
    assert band["locked_quality_input_source"] == "planned_contact"
    assert band["quality_baseline_role"] == "IMMUTABLE_LOCK_BASELINE"
    assert band["baseline_frozen"] is True


def test_new_active_quality_lineage_accepts_its_own_first_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    target = locked_target()
    target.update({"lock_frame_id": 22, "attempt_id": "attempt-22"})
    target["locked_grasp_band"] = locked_grasp_band_snapshot(target)
    worker = SimpleNamespace(
        lock_manager=SimpleNamespace(locked_target=target),
        latest_report={
            "writer_session_id": "writer-2",
            "target_lock": {"locked_target": dict(target)},
        },
    )

    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-2",
            "lock_frame_id": 22,
            "lock_attempt_id": "attempt-22",
            "quality_compute_id": "qm-new-lineage",
            "quality_label": "GOOD",
            "raw_epsilon": 0.101,
            "normalized_quality_score": 100.0,
            "epsilon_force_closure": True,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    )

    assert target["locked_grasp_band"]["locked_quality_compute_id"] == "qm-new-lineage"
    assert target["locked_grasp_band"]["locked_epsilon"] == pytest.approx(0.101)


def test_stale_quality_marks_status_without_mutating_locked_analytic_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    target = locked_target()
    target["locked_quality_compute_id"] = "qm-lock"
    target["locked_quality_label"] = "GOOD"
    target["epsilon_value_at_lock"] = 0.055
    target["locked_grasp_band"] = locked_grasp_band_snapshot(target)
    worker = SimpleNamespace(
        lock_manager=SimpleNamespace(locked_target=target),
        latest_report={
            "writer_session_id": "writer-1",
            "target_lock": {"locked_target": dict(target)},
        },
    )

    EvaluationWorker.attach_locked_quality_snapshot(
        worker,
        {
            "quality_matrix_status": "stale_needs_recompute",
            "writer_session_id": "writer-1",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-12",
            "quality_compute_id": "qm-should-not-replace",
            "quality_label": "WEAK",
            "raw_epsilon": 0.001,
            "quality_stale_reason": "pose_changed",
        },
    )

    band = target["locked_grasp_band"]
    assert band["locked_quality_compute_id"] == "qm-lock"
    assert band["locked_quality_label"] == "GOOD"
    assert band["locked_epsilon"] == 0.055
    assert band["analytic_quality_status"] == "STALE"
    assert band["analytic_quality_stale_reason"] == "pose_changed"


def test_start_approach_is_only_an_operator_monitoring_marker():
    manager = SimpleNamespace(
        status=lambda: "LOCKED",
        approach_mode_requested=False,
        approach_mode_requested_at=None,
    )
    worker = SimpleNamespace(lock_manager=manager)

    EvaluationWorker.start_approach_mode(worker)

    assert manager.approach_mode_requested is True
    assert manager.approach_mode_requested_at is not None


def test_relock_transition_records_old_and_new_identity_and_is_never_silent():
    old = locked_target()
    old["locked_grasp_band"] = locked_grasp_band_snapshot(old)
    new = locked_target()
    new.update({"attempt_id": "attempt-22", "lock_frame_id": 22})
    new["locked_grasp_band"] = locked_grasp_band_snapshot(new)

    event = build_lock_transition_event(
        old,
        new,
        2,
        "automatic_relock_after_loss",
        "target_lost_for_max_frames",
    )

    assert event["event_type"] == "RELOCK"
    assert event["previous_lock"]["attempt_id"] == "attempt-12"
    assert event["current_lock"]["attempt_id"] == "attempt-22"
    assert event["identity_changed"] is True
    assert event["ui_notification_required"] is True
    assert event["silent_relock"] is False


def test_manual_reset_preserves_previous_identity_for_next_explicit_lock_event():
    manager = TargetLockManager()
    manager.locked_target = locked_target()

    manager.clear("manual_reset_from_dashboard")

    assert manager.pending_relock_previous["attempt_id"] == "attempt-12"
    assert manager.pending_relock_reason == "manual_reset_from_dashboard"
    assert manager.status() == "SEARCHING"


def test_sticky_single_bottle_lock_survives_long_dropout_and_recovers_same_attempt(monkeypatch):
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", True)
    manager = TargetLockManager()
    manager.locked_target = locked_target()
    manager.lock_generation = 1
    original_target = manager.locked_target

    for _ in range(manager.max_bad_tracking_frames * 20):
        tracking = {
            "tracking_ok": False,
            "identity_tracking_ok": False,
            "contact_patch_tracking_ok": False,
            "locked_patch_visible": False,
            "identity_measurement_current": False,
            "projection_mode": "INVALID_3D",
            "locked_projection_mode": "INVALID_3D",
            "projection_valid": False,
            "projection_invalid_reason": "current_bottle_3d_frame_not_measured",
        }
        manager.apply_tracking_counters(tracking)

        assert manager.status() == "LOCKED"
        assert manager.locked_target is original_target
        assert manager.locked_target["attempt_id"] == "attempt-12"
        assert manager.lock_generation == 1
        assert manager.has_lost_lock is False

    assert manager.sticky_lock_waiting_reacquisition is True
    assert manager.sticky_lock_dropout_frames == (
        manager.max_bad_tracking_frames * 20 - manager.max_bad_tracking_frames + 1
    )
    report = realtime_runner.target_lock_report(manager, tracking)
    assert report["grasp_band_locked"] is True
    assert report["grasp_band_relock_required"] is False
    assert report["sticky_lock_preserved_attempt_id"] == "attempt-12"

    recovery = projected_tracking(
        identity_measurement_current=True,
        contact_patch_tracking_ok=True,
        projection_mode="TRUE_3D",
        locked_projection_mode="TRUE_3D",
        projection_valid=True,
        contact_model_projection_status="PROJECTED_3D",
    )
    manager.apply_tracking_counters(recovery)

    assert manager.sticky_lock_waiting_reacquisition is True
    manager.apply_tracking_counters(dict(recovery))

    assert manager.status() == "LOCKED"
    assert manager.locked_target is original_target
    assert manager.locked_target["attempt_id"] == "attempt-12"
    assert manager.lock_generation == 1
    assert manager.sticky_lock_waiting_reacquisition is False
    assert manager.sticky_lock_dropout_frames == 0


def test_sticky_demo_backend_restart_starts_clean_instead_of_restoring_lost_relock(monkeypatch):
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", True)
    manager = TargetLockManager()
    report = {
        "grasp_memory": {
            "status": "PRESERVED_AFTER_TRACK_LOSS",
            "locked_body_fraction": 0.52,
            "axis_vector_at_lock": [0.0, 1.0],
            "grip_width_mm_at_lock": 56.0,
            "last_loss_timestamp": "1970-01-01T00:16:40+00:00",
        },
        "locked_target": {"valid": False, "bottle_target_id": "bottle_01"},
        "target_loss_reason": "temporary_tracking_loss",
    }

    assert manager.restore_recent_grasp_memory(report, now_epoch=1050.0) is False
    assert manager.status() == "SEARCHING"
    assert manager.has_lost_lock is False
    assert manager.grasp_memory is None


def test_config_file_is_part_of_repository_config_tree():
    assert (Path(__file__).resolve().parents[1] / "config" / "hand_mounted_camera.yaml").is_file()

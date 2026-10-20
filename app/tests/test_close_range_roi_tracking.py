import copy
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from grasp_project.realsense.realtime_runner import (
    EvaluationWorker,
    RealtimeBackendRunner,
    TargetLockManager,
    blocked_single_bottle_demo_selection,
    compact_live_status,
    compute_approach_guidance,
    detect_prelock_rgbd_poses,
    detect_roi_bottle_candidate,
    dynamic_post_lock_roi,
    fast_roi_bottle_pose,
    local_locked_patch_tracking,
    locked_tracking_reacquisition_due,
    match_locked_bottle_pose,
    prelock_full_resolution_confirmation_due,
    post_lock_extent_visibility,
    prelock_body_fraction,
    refresh_current_depth_corridor_evidence,
    reanchor_target_clamp_to_current_edges,
    reanchor_target_clamp_only_from_current_geometry,
    roi_opencv_depth_bottle_tracking,
    scale_pose_to_full,
    select_lost_target_reacquisition_candidate,
    segmentation_confidence_summary,
    separate_identity_from_contact_tracking,
    stabilize_magenta_contact_geometry,
    ready_blockers_from_values,
    target_lock_report,
    tracking_good_from_values,
    validate_roi_bottle_candidate,
)
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.object_state import ObjectState
from pregrasp_planner import plan_pregrasp


def test_post_lock_roi_anticipates_depth_growth_for_hand_mounted_approach():
    seed_bbox = [264.0, 227.0, 322.0, 312.0]
    manager = SimpleNamespace(
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_good_bottle_depth_mm=425.0,
    )

    roi, telemetry = dynamic_post_lock_roi(seed_bbox, (480, 640, 3), manager)

    # The search neighborhood anticipates growth down to the configured
    # 80 mm locked-tracking floor even when image-center velocity is zero.
    center_x = (seed_bbox[0] + seed_bbox[2]) / 2.0
    center_y = (seed_bbox[1] + seed_bbox[3]) / 2.0
    expected_half_width = (seed_bbox[2] - seed_bbox[0]) * telemetry["roi_depth_growth_scale"] / 2.0
    expected_half_height = (seed_bbox[3] - seed_bbox[1]) * telemetry["roi_depth_growth_scale"] / 2.0
    assert roi[0] <= max(0.0, center_x - expected_half_width)
    assert roi[2] >= min(640.0, center_x + expected_half_width)
    assert roi[1] <= max(0.0, center_y - expected_half_height)
    assert roi[3] >= min(480.0, center_y + expected_half_height)


def test_post_lock_roi_does_not_shrink_or_bypass_frame_clipping():
    manager = SimpleNamespace(
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_good_bottle_depth_mm=180.0,
    )

    roi, telemetry = dynamic_post_lock_roi([2.0, 3.0, 80.0, 120.0], (240, 320, 3), manager)

    assert telemetry["roi_depth_growth_scale"] == 2.25
    assert roi[0] == 0 and roi[1] == 0
    assert roi[2] <= 320 and roi[3] <= 240


def test_locked_tracker_failure_requests_identity_checked_broad_reacquisition():
    manager = SimpleNamespace(
        status=lambda: "LOCKED",
        roi_bottle_fail_frames=2,
    )

    assert locked_tracking_reacquisition_due(manager, {}) is True

    manager.roi_bottle_fail_frames = 0
    healthy = {
        "tracking_source": "roi_opencv_depth",
        "identity_measurement_current": True,
        "projection_mode": "TRUE_3D",
    }
    assert locked_tracking_reacquisition_due(manager, healthy) is False

    degraded = {
        "tracking_source": "shifted_locked_geometry",
        "identity_measurement_current": False,
        "projection_mode": "INVALID_3D",
    }
    assert locked_tracking_reacquisition_due(manager, degraded) is False
    manager.roi_bottle_fail_frames = 2
    assert locked_tracking_reacquisition_due(
        manager,
        degraded,
        fast_frames_since_full=1,
    ) is False
    assert locked_tracking_reacquisition_due(
        manager,
        degraded,
        fast_frames_since_full=2,
    ) is True


def test_locked_recovery_treats_missing_projection_mode_as_degraded():
    manager = SimpleNamespace(
        status=lambda: "LOCKED",
        roi_bottle_fail_frames=2,
    )
    mode_less_full_recovery = {
        "tracking_source": "full_frame_identity_checked_reacquisition",
        "identity_measurement_current": True,
    }

    assert locked_tracking_reacquisition_due(
        manager,
        mode_less_full_recovery,
        fast_frames_since_full=2,
    ) is True


def test_failed_full_projection_keeps_explicit_invalid_3d_contract(monkeypatch):
    monkeypatch.setattr(
        realtime_runner,
        "project_locked_bottle_contact_model",
        lambda *_args, **_kwargs: (
            None,
            None,
            {
                "projection_valid": False,
                "projection_status": "FAILED",
                "projection_source": "BOTTLE_LOCAL_3D_REPROJECTION",
                "projection_invalid_reason": "locked_3d_band_not_supported_by_current_mask",
                "reason": "locked_3d_band_not_supported_by_current_mask",
                "projected_band_mask_overlap": 0.31,
            },
        ),
    )
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:201, 120:201] = 255

    plan, tracking = realtime_runner.evaluate_fixed_clamp_against_bottle(
        {
            "cleaned_mask": mask,
            "center_px": [160.0, 120.0],
            "depth_confidence": 0.9,
        },
        np.full((240, 320), 420, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        {"fixed_clamp_zone": {}},
        {"full_bottle_visible": True},
        (240, 320, 3),
        temporal_state={},
    )

    assert plan is None
    assert tracking["locked_projection_mode"] == "INVALID_3D"
    assert tracking["projection_mode"] == "INVALID_3D"
    assert tracking["projection_valid"] is False
    assert tracking["projection_invalid_reason"] == (
        "locked_3d_band_not_supported_by_current_mask"
    )
    assert tracking["projected_band_mask_overlap"] == 0.31


def test_full_recovery_projection_failure_cannot_move_anchor_or_publish_current_guide(
    monkeypatch,
):
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:201, 120:201] = 255
    clamp = {
        "clamp_center_px": [160.0, 150.0],
        "clamp_band_polygon": [
            [125.0, 143.0],
            [195.0, 143.0],
            [195.0, 157.0],
            [125.0, 157.0],
        ],
        "left_contact_px": [125.0, 150.0],
        "right_contact_px": [195.0, 150.0],
        "grip_width_px": 70.0,
        "left_clamp_patch": {
            "center_px": [125.0, 150.0],
            "polygon": [[120.0, 143.0], [130.0, 143.0], [130.0, 157.0], [120.0, 157.0]],
        },
        "right_clamp_patch": {
            "center_px": [195.0, 150.0],
            "polygon": [[190.0, 143.0], [200.0, 143.0], [200.0, 157.0], [190.0, 157.0]],
        },
        "grasp_line": {
            "left_endpoint_px": [125.0, 150.0],
            "right_endpoint_px": [195.0, 150.0],
            "center_px": [160.0, 150.0],
            "direction_px": [1.0, 0.0],
        },
    }
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "attempt_id": "attempt-1",
        "lock_frame_id": 10,
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.68,
        "fixed_clamp_zone": copy.deepcopy(clamp),
        "locked_bottle_geometry": {
            "bbox_xyxy": [120.0, 40.0, 200.0, 200.0],
            "center_px": [160.0, 120.0],
            "axis_vector": [0.0, 1.0],
        },
        "locked_grasp_band": {
            "lock_valid": True,
            "true_3d_lock": True,
            "locked_band_center_px": [160.0, 150.0],
            "locked_body_fraction_center": 0.68,
        },
    }
    manager.current_tracked_target = {
        "last_verified_rgbd_target_clamp": copy.deepcopy(clamp),
        "last_verified_rgbd_target_clamp_frame_id": 10,
    }
    original_anchor = copy.deepcopy(manager.locked_target["locked_bottle_geometry"])
    manager.last_good_bottle_geometry = copy.deepcopy(original_anchor)
    manager.locked_target["full_bottle_anchor_geometry"] = copy.deepcopy(original_anchor)
    monkeypatch.setattr(
        realtime_runner,
        "evaluate_fixed_clamp_against_bottle",
        lambda *_args, **_kwargs: (
            None,
            {
                "tracking_confidence": 0.0,
                "tracking_ok": False,
                "projection_valid": False,
                "projection_mode": "INVALID_3D",
                "locked_projection_mode": "INVALID_3D",
                "projection_source": "BOTTLE_LOCAL_3D_REPROJECTION",
                "projection_invalid_reason": "locked_3d_band_not_supported_by_current_mask",
                "contact_model_projection_status": "FAILED",
                "reason": "locked_3d_band_not_supported_by_current_mask",
            },
        ),
    )
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "bbox_xyxy": [120.0, 40.0, 200.0, 200.0],
        "bbox_xywh": [120.0, 40.0, 80.0, 160.0],
        "center_px": [160.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "axis_angle_degrees": 90.0,
        "depth_median_mm": 420.0,
        "partial_extent": False,
    }

    plan, tracking = manager.track(
        pose,
        np.full((240, 320), 420, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        {"full_bottle_visible": True},
        (240, 320, 3),
        frame_id=11,
    )

    assert plan is None
    assert tracking["identity_measurement_current"] is False
    assert tracking["tracking_source"] == "full_frame_identity_checked_reacquisition"
    assert tracking["locked_projection_mode"] == "INVALID_3D"
    assert tracking["projection_valid"] is False
    assert tracking["operator_locked_fraction_visual_guide_current"] is False
    assert tracking["operator_band_is_safety_measurement"] is False
    assert tracking.get("display_measurement_current") is not True
    assert manager.last_good_bottle_geometry == original_anchor
    assert manager.locked_target["full_bottle_anchor_geometry"] == original_anchor

    report = {
        "state": "APPROACH_TRACKING",
        "decision": "ADJUST POSITION",
        "ready_to_close": False,
        "target_lock": target_lock_report(manager, tracking),
        "bottle_geometry": pose,
    }
    realtime_runner.draw_fast_overlay(
        np.zeros((240, 320, 3), dtype=np.uint8),
        pose,
        None,
        report,
    )

    assert report["grasp_band_operator_guide_visible_during_depth_dropout"] is False
    assert report["grasp_band_display_continuity"] != (
        "CURRENT_BOTTLE_RELATIVE_GUIDE_DISPLAY_ONLY"
    )


def test_failed_full_projection_never_creates_display_pixels_after_expired_hold(
    monkeypatch,
):
    """Broad association cannot redraw a hand-shortened bottle silhouette."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:201, 120:201] = 255
    clamp = {
        "clamp_center_px": [160.0, 150.0],
        "clamp_band_polygon": [
            [125.0, 143.0],
            [195.0, 143.0],
            [195.0, 157.0],
            [125.0, 157.0],
        ],
        "left_contact_px": [125.0, 150.0],
        "right_contact_px": [195.0, 150.0],
        "left_clamp_patch": {
            "center_px": [125.0, 150.0],
            "polygon": [[120.0, 143.0], [130.0, 143.0], [130.0, 157.0], [120.0, 157.0]],
        },
        "right_clamp_patch": {
            "center_px": [195.0, 150.0],
            "polygon": [[190.0, 143.0], [200.0, 143.0], [200.0, 157.0], [190.0, 157.0]],
        },
    }
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "attempt_id": "attempt-1",
        "lock_frame_id": 10,
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.68,
        "fixed_clamp_zone": copy.deepcopy(clamp),
        "locked_bottle_geometry": {
            "bbox_xyxy": [120.0, 40.0, 200.0, 200.0],
            "center_px": [160.0, 120.0],
            "axis_vector": [0.0, 1.0],
        },
        "locked_grasp_band": {
            "lock_valid": True,
            "true_3d_lock": True,
            "locked_band_center_px": [160.0, 150.0],
            "locked_body_fraction_center": 0.68,
        },
    }
    manager.current_tracked_target = {
        "last_verified_rgbd_target_clamp": copy.deepcopy(clamp),
        "last_verified_rgbd_target_clamp_frame_id": 10,
    }
    manager.temporal_state["last_valid_true_3d_clamp"] = copy.deepcopy(clamp)
    manager.temporal_state["last_valid_true_3d_monotonic"] = (
        realtime_runner.time.monotonic()
        - realtime_runner.TRUE_3D_TEMPORAL_HOLD_SEC
        - 0.5
    )
    monkeypatch.setattr(
        realtime_runner,
        "evaluate_fixed_clamp_against_bottle",
        lambda *_args, **_kwargs: (
            None,
            {
                "tracking_confidence": 0.0,
                "tracking_ok": False,
                "projection_valid": False,
                "projection_mode": "INVALID_3D",
                "locked_projection_mode": "INVALID_3D",
                "projection_invalid_reason": "locked_3d_band_not_supported_by_current_mask",
            },
        ),
    )
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "bbox_xyxy": [120.0, 40.0, 200.0, 200.0],
        "bbox_xywh": [120.0, 40.0, 80.0, 160.0],
        "center_px": [160.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "axis_angle_degrees": 90.0,
        "depth_median_mm": 420.0,
        "partial_extent": False,
    }

    _plan, current = manager.track(
        pose,
        np.full((240, 320), 420, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        {"full_bottle_visible": True},
        (240, 320, 3),
        frame_id=11,
    )
    assert current["locked_projection_mode"] == "INVALID_3D"
    assert current["operator_locked_fraction_visual_guide_current"] is False
    assert current.get("display_target_clamp", {}) == {}

    dropout = realtime_runner.select_true_3d_dropout_display(
        manager.locked_target,
        {
            "identity_measurement_current": False,
            "tracking_ok": False,
            "tracking_confidence": 0.0,
        },
        manager.temporal_state,
        projection_invalid_reason="current_bottle_identity_not_measured",
    )
    assert dropout["locked_projection_mode"] == "INVALID_3D"
    assert dropout["projection_valid"] is False
    assert realtime_runner.target_clamp_has_spatial_extent(
        dropout["target_clamp"]
    ) is False

    report = {
        "state": "APPROACH_TRACKING",
        "decision": "ADJUST POSITION",
        "ready_to_close": False,
        "target_lock": target_lock_report(manager, dropout),
        "bottle_geometry": {},
    }
    realtime_runner.draw_fast_overlay(
        np.zeros((240, 320, 3), dtype=np.uint8),
        {},
        None,
        report,
    )
    assert report["grasp_band_operator_guide_visible_during_depth_dropout"] is False
    assert report["grasp_band_display_continuity"] == "UNAVAILABLE_INVALID_3D"


def test_metadata_only_invalid_clamp_cannot_hide_protected_band_extent():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[80:190, 110:211] = 255
    protected = {
        "clamp_center_px": [160.0, 145.0],
        "clamp_band_polygon": [[125.0, 138.0], [195.0, 138.0], [195.0, 152.0], [125.0, 152.0]],
        "left_clamp_patch": {
            "center_px": [125.0, 145.0],
            "polygon": [[118.0, 136.0], [132.0, 136.0], [132.0, 154.0], [118.0, 154.0]],
        },
        "right_clamp_patch": {
            "center_px": [195.0, 145.0],
            "polygon": [[188.0, 136.0], [202.0, 136.0], [202.0, 154.0], [188.0, 154.0]],
        },
    }
    manager = SimpleNamespace(
        current_tracked_target={
            "target_clamp": {
                "projection_mode": "INVALID_3D",
                "projection_valid": False,
                "projection_invalid_reason": "current_bottle_3d_frame_not_measured",
            },
            "last_verified_rgbd_target_clamp": copy.deepcopy(protected),
        },
        locked_target={
            "valid": True,
            "fixed_clamp_zone": copy.deepcopy(protected),
        },
    )

    selected = realtime_runner.protected_locked_band_clamp(
        manager,
        manager.locked_target,
    )
    evidence = realtime_runner.expected_bottom_hand_occlusion_evidence(
        {
            "skin_occlusion_on_seed": True,
            "skin_exclusion_bbox_xyxy": [90.0, 165.0, 230.0, 239.0],
            "skin_overlap_seed_bbox_xyxy": [120.0, 165.0, 205.0, 230.0],
            "foreground_occlusion_on_seed": False,
            "cleaned_mask": mask,
        },
        selected,
        (240, 320, 3),
        lock_context={
            "full_bottle_lock_active": True,
            "lock_was_lost": False,
            "current_depth_mm": 200.0,
        },
    )

    assert selected == protected
    assert evidence["expected_bottom_hand_occlusion_reason"] != (
        "locked_band_extent_unavailable"
    )
    assert evidence["expected_hand_locked_band_mask_supported"] is True


def test_unverified_hand_occlusion_hold_uses_elapsed_time_not_camera_frame_ids(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    manager = TargetLockManager()
    manager.locked_target = {"valid": True}
    held = {
        "tracking_ok": True,
        "contact_patch_tracking_ok": True,
        "identity_tracking_ok": False,
        "identity_measurement_current": False,
        "roi_candidate_foreground_occlusion_on_seed": True,
        "identity_measurement_age_frames": 500,
    }

    manager.apply_tracking_counters(held)
    assert held["identity_occlusion_hold_active"] is True
    assert held["identity_occlusion_hold_elapsed_sec"] == 0.0
    assert held["identity_occlusion_hold_timebase"] == "monotonic_seconds"

    clock["now"] += realtime_runner.POST_LOCK_HAND_OCCLUSION_HOLD_MAX_SEC + 0.01
    expired = dict(held)
    expired["identity_occlusion_hold_active"] = False
    manager.apply_tracking_counters(expired)
    assert expired["identity_occlusion_hold_active"] is False
    assert expired["identity_occlusion_hold_expired"] is True


def test_identity_recovery_requires_two_current_measurements_after_occlusion(monkeypatch):
    clock = {"now": 20.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    manager = TargetLockManager()
    manager.locked_target = {"valid": True}
    occluded = {
        "tracking_ok": True,
        "contact_patch_tracking_ok": True,
        "identity_tracking_ok": False,
        "identity_measurement_current": False,
        "roi_candidate_hand_occlusion": True,
    }
    manager.apply_tracking_counters(occluded)

    first = {
        "tracking_ok": True,
        "contact_patch_tracking_ok": True,
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
    }
    clock["now"] += 0.05
    manager.apply_tracking_counters(first)
    assert first["identity_occlusion_hold_active"] is True
    assert first["identity_recovery_confirmation_pending"] is True
    assert first["identity_claim_status"] == "CURRENT_RGBD_IDENTITY_RECOVERY_CONFIRMATION_PENDING"

    second = dict(first)
    clock["now"] += 0.05
    manager.apply_tracking_counters(second)
    assert second["identity_occlusion_hold_active"] is False
    assert second["identity_recovery_confirmation_pending"] is False
    assert second["identity_claim_status"] == "CURRENT_RGBD_IDENTITY"


def test_locked_full_evaluation_aux_reports_are_throttled_like_fast_reports(monkeypatch):
    clock = {"now": 50.0}
    monkeypatch.setattr(realtime_runner, "FAST_FINAL_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(realtime_runner, "LAST_FAST_AUX_OUTPUT_TIME", {})
    monkeypatch.setattr(realtime_runner, "LAST_FAST_AUX_ANY_OUTPUT_TIME", 0.0)
    report = {
        "fast_tracking_active": False,
        "fast_tracking_used": False,
        "target_lock": {"lock_state": "LOCKED"},
    }

    assert realtime_runner.should_write_fast_aux_output("locked_target", report) is True
    clock["now"] += 0.2
    assert realtime_runner.should_write_fast_aux_output("locked_target", report) is False


def test_prelock_aux_and_diagnostics_are_throttled_in_fast_demo(monkeypatch):
    clock = {"now": 80.0}
    monkeypatch.setattr(realtime_runner, "FAST_FINAL_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(realtime_runner, "LAST_FAST_AUX_OUTPUT_TIME", {})
    monkeypatch.setattr(realtime_runner, "LAST_FAST_AUX_ANY_OUTPUT_TIME", 0.0)
    report = {
        "fast_tracking_active": False,
        "fast_tracking_used": False,
        "target_lock": {"lock_state": "SEARCHING"},
    }

    assert realtime_runner.should_write_fast_aux_output("pregrasp_report", report) is True
    clock["now"] += 0.2
    assert realtime_runner.should_write_fast_aux_output("pregrasp_report", report) is False

    clock["now"] += realtime_runner.FAST_AUX_OUTPUT_MIN_GAP_SEC
    assert realtime_runner.should_write_fast_diagnostics_output("diagnostics", report) is True
    clock["now"] += 0.2
    assert realtime_runner.should_write_fast_diagnostics_output("diagnostics", report) is False


def test_rejected_prelock_feedback_requires_consistent_observations(monkeypatch):
    monkeypatch.setattr(
        realtime_runner,
        "PRELOCK_OPERATOR_FEEDBACK_STABLE_FRAMES",
        3,
    )
    manager = TargetLockManager()
    pose = {
        "center_px": [200.0, 120.0],
        "bbox_xyxy": [180.0, 60.0, 220.0, 180.0],
        "depth_median_mm": 450.0,
    }

    assert manager.update_prelock_feedback(pose) is False
    assert manager.update_prelock_feedback(dict(pose, center_px=[202.0, 121.0])) is False
    assert manager.update_prelock_feedback(dict(pose, center_px=[203.0, 122.0])) is True
    assert manager.prelock_feedback_frames == 3

    # A large spatial/depth jump starts a new candidate sequence instead of
    # carrying the old rejected contour into operator-facing ADJUST state.
    jumped = dict(pose, center_px=[330.0, 220.0], depth_median_mm=750.0)
    assert manager.update_prelock_feedback(jumped) is False
    assert manager.prelock_feedback_frames == 1

    assert manager.update_prelock_feedback(None) is False
    assert manager.prelock_feedback_frames == 0


def _oriented_rectangle_geometry(angle_deg, *, center=(160.0, 120.0), width=40.0, height=140.0, depth_mm=400.0):
    angle = np.radians(float(angle_deg))
    axis = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    width_axis = np.array([-axis[1], axis[0]], dtype=float)
    center = np.asarray(center, dtype=float)
    box = np.asarray(
        [
            center - 0.5 * height * axis - 0.5 * width * width_axis,
            center - 0.5 * height * axis + 0.5 * width * width_axis,
            center + 0.5 * height * axis + 0.5 * width * width_axis,
            center + 0.5 * height * axis - 0.5 * width * width_axis,
        ],
        dtype=float,
    )
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    return {
        "detected": True,
        "contour": np.round(box).astype(np.int32).reshape(-1, 1, 2),
        "oriented_bbox": box.tolist(),
        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "bbox_xywh": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "center_px": center.tolist(),
        "axis_vector": axis.tolist(),
        "axis_angle_degrees": float(angle_deg),
        "depth_median_mm": float(depth_mm),
        "depth_confidence": 1.0,
        "detection_confidence": 1.0,
        "known_blue_identity_supported": True,
        "known_blue_support_fraction": 1.0,
    }


def test_dynamic_post_lock_roi_is_bounded_without_full_diagonal_rotation_sweep():
    seed_bbox = [140.0, 50.0, 180.0, 190.0]
    manager = SimpleNamespace(last_good_raw_bottle_velocity_px=[0.0, 0.0], last_good_bottle_depth_mm=80.0)

    roi, telemetry = dynamic_post_lock_roi(seed_bbox, (240, 320, 3), manager)

    radius = 0.5 * np.hypot(40.0, 140.0)
    pad_x, pad_y = telemetry["roi_padding_xy_used"]
    assert roi == [
        int(round(seed_bbox[0] - pad_x)),
        int(round(seed_bbox[1] - pad_y)),
        int(round(seed_bbox[2] + pad_x)),
        int(round(seed_bbox[3] + pad_y)),
    ]
    assert roi[0] > 160.0 - radius
    assert roi[2] < 160.0 + radius
    assert telemetry["roi_rotation_sweep_padding_xy"] == [0.0, 0.0]
    assert telemetry["roi_rotation_sweep_padding_applied"] is False


def _paint_oriented_bottle_scene(shape, geometry, depth_mm=400):
    height, width = shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 900, dtype=np.uint16)
    polygon = np.round(np.asarray(geometry["oriented_bbox"], dtype=float)).astype(np.int32)
    cv2.fillPoly(rgb, [polygon], (30, 80, 220))
    cv2.fillPoly(depth, [polygon], int(depth_mm))
    return rgb, depth


def test_internal_roi_clip_from_35_degree_rotation_gets_one_bounded_retry():
    seed = _oriented_rectangle_geometry(90.0, center=(160.0, 120.0), width=40.0, height=140.0)
    rotated = _oriented_rectangle_geometry(35.0, center=(160.0, 120.0), width=40.0, height=140.0)
    rgb, depth = _paint_oriented_bottle_scene((240, 320, 3), rotated)
    manager = SimpleNamespace(
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_good_bottle_depth_mm=80.0,
    )

    pose, roi, error, _timings, telemetry = detect_roi_bottle_candidate(
        rgb,
        depth,
        seed,
        rgb.shape,
        lock_manager=manager,
        depth_hint_mm=400.0,
    )

    assert error is None
    assert pose is not None
    assert telemetry["roi_rotation_retry_attempted"] is True
    assert telemetry["roi_rotation_retry_recovered"] is True
    assert telemetry["roi_rotation_retry_initial_internal_boundary_touches"]["left"] is True
    assert telemetry["roi_rotation_retry_initial_internal_boundary_touches"]["right"] is True
    assert not any(telemetry["roi_rotation_retry_final_internal_boundary_touches"].values())
    assert roi == telemetry["roi_rotation_retry_roi_xyxy"]
    assert roi[0] < telemetry["roi_rotation_retry_initial_roi_xyxy"][0]
    assert roi[2] > telemetry["roi_rotation_retry_initial_roi_xyxy"][2]


def test_close_range_rotation_retry_covers_frame_after_rotation_and_translation():
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(400.0, 240.0),
        width=60.0,
        height=350.0,
        depth_mm=180.0,
    )
    moved = _oriented_rectangle_geometry(
        0.0,
        center=(250.0, 390.0),
        width=60.0,
        height=350.0,
        depth_mm=180.0,
    )
    rgb, depth = _paint_oriented_bottle_scene((480, 640, 3), moved, depth_mm=180)
    manager = SimpleNamespace(
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_good_bottle_depth_mm=180.0,
    )

    pose, roi, error, _timings, telemetry = detect_roi_bottle_candidate(
        rgb,
        depth,
        seed,
        rgb.shape,
        lock_manager=manager,
        depth_hint_mm=180.0,
    )

    assert error is None
    assert pose is not None
    assert roi == [0, 0, 640, 480]
    assert telemetry["roi_rotation_retry_attempted"] is True
    assert telemetry["roi_rotation_retry_recovered"] is True
    assert telemetry["roi_rotation_retry_scope"] == "full_frame_close_range_scaled_retry"
    assert np.allclose(pose["center_px"], moved["center_px"], atol=4.0)


def test_previous_local_failure_promotes_clipped_retry_to_full_frame():
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(200.0, 200.0),
        width=40.0,
        height=140.0,
        depth_mm=400.0,
    )
    rotated = _oriented_rectangle_geometry(
        0.0,
        center=(300.0, 200.0),
        width=40.0,
        height=140.0,
        depth_mm=400.0,
    )
    rgb, depth = _paint_oriented_bottle_scene((400, 400, 3), rotated)
    manager = SimpleNamespace(
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_good_bottle_depth_mm=400.0,
        roi_bottle_fail_frames=1,
    )

    pose, roi, error, _timings, telemetry = detect_roi_bottle_candidate(
        rgb,
        depth,
        seed,
        rgb.shape,
        lock_manager=manager,
        depth_hint_mm=400.0,
    )

    assert error is None
    assert pose is not None
    assert roi == [0, 0, 400, 400]
    assert telemetry["roi_rotation_retry_scope"] == (
        "full_frame_after_previous_local_failure"
    )


def test_component_still_clipped_after_rotation_retry_is_never_current():
    seed = _oriented_rectangle_geometry(90.0, center=(200.0, 200.0), width=40.0, height=140.0)
    oversized = _oriented_rectangle_geometry(90.0, center=(200.0, 200.0), width=40.0, height=220.0)
    rgb, depth = _paint_oriented_bottle_scene((400, 400, 3), oversized)
    manager = SimpleNamespace(
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_good_bottle_depth_mm=80.0,
    )

    pose, _roi, error, _timings, telemetry = detect_roi_bottle_candidate(
        rgb,
        depth,
        seed,
        rgb.shape,
        lock_manager=manager,
        depth_hint_mm=400.0,
    )

    assert pose is None
    assert error == "fast_roi_component_clipped_by_internal_search_boundary"
    assert telemetry["roi_rotation_retry_attempted"] is True
    assert telemetry["roi_rotation_retry_recovered"] is False
    assert any(telemetry["roi_rotation_retry_final_internal_boundary_touches"].values())


def test_roi_identity_shape_evidence_is_rotation_invariant():
    seed = _oriented_rectangle_geometry(90.0)
    rotated = _oriented_rectangle_geometry(45.0)

    accepted, telemetry = validate_roi_bottle_candidate(
        rotated,
        seed,
        image_shape=(240, 320, 3),
        current_patch_depth_mm=400.0,
    )

    assert accepted is True
    assert telemetry["roi_candidate_reject_reason"] == []
    assert telemetry["roi_candidate_area_ratio"] == pytest.approx(1.0, rel=0.03)
    assert telemetry["roi_candidate_aspect_ratio"] == pytest.approx(3.5, rel=0.03)
    assert telemetry["roi_candidate_aspect_source"] == "oriented_bbox_rotation_invariant"
    assert telemetry["roi_candidate_axis_aligned_area_ratio"] > 2.5


def test_large_approach_depth_change_is_current_when_contact_patch_agrees():
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(320.0, 240.0),
        width=40.0,
        height=140.0,
        depth_mm=600.0,
    )
    approached = _oriented_rectangle_geometry(
        90.0,
        center=(320.0, 240.0),
        width=80.0,
        height=280.0,
        depth_mm=300.0,
    )

    accepted, telemetry = validate_roi_bottle_candidate(
        approached,
        seed,
        image_shape=(480, 640, 3),
        current_patch_depth_mm=300.0,
    )

    assert accepted is True
    assert telemetry["roi_candidate_depth_delta_mm"] == 300.0
    assert telemetry["roi_candidate_patch_depth_delta_mm"] == 0.0
    assert telemetry["roi_candidate_large_depth_change_confirmed_by_current_patch"] is True
    assert "depth_delta_too_large" not in telemetry["roi_candidate_reject_reason"]


def test_large_depth_change_without_current_patch_confirmation_is_rejected():
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(320.0, 240.0),
        width=40.0,
        height=140.0,
        depth_mm=600.0,
    )
    approached = _oriented_rectangle_geometry(
        90.0,
        center=(320.0, 240.0),
        width=80.0,
        height=280.0,
        depth_mm=300.0,
    )

    accepted, telemetry = validate_roi_bottle_candidate(
        approached,
        seed,
        image_shape=(480, 640, 3),
        current_patch_depth_mm=None,
    )

    assert accepted is False
    assert telemetry["roi_candidate_large_depth_change_confirmed_by_current_patch"] is False
    assert "depth_delta_too_large" in telemetry["roi_candidate_reject_reason"]


def test_roi_identity_still_rejects_true_intrinsic_shape_change():
    seed = _oriented_rectangle_geometry(90.0, width=35.0, height=140.0)
    changed = _oriented_rectangle_geometry(45.0, width=75.0, height=140.0)

    accepted, telemetry = validate_roi_bottle_candidate(
        changed,
        seed,
        image_shape=(240, 320, 3),
        current_patch_depth_mm=400.0,
    )

    assert accepted is False
    assert "aspect_ratio_changed" in telemetry["roi_candidate_reject_reason"]


def test_small_grazing_skin_overlap_does_not_invalidate_current_depth_shape():
    seed = _oriented_rectangle_geometry(90.0)
    candidate = _oriented_rectangle_geometry(90.0, center=(160.0, 128.0))
    candidate.update({
        "skin_exclusion_applied": True,
        "skin_occlusion_on_seed": True,
        "skin_overlap_seed_pixels": 12,
        "skin_overlap_seed_ratio": 0.02,
    })

    accepted, telemetry = validate_roi_bottle_candidate(
        candidate,
        seed,
        image_shape=(240, 320, 3),
        current_patch_depth_mm=400.0,
    )

    assert accepted is True
    assert telemetry["roi_candidate_skin_overlap_observed"] is True
    assert telemetry["roi_candidate_skin_overlap_material"] is False
    assert "hand_skin_occludes_locked_bottle_boundary" not in telemetry["roi_candidate_reject_reason"]


def test_material_skin_overlap_still_invalidates_changed_bottle_boundary():
    seed = _oriented_rectangle_geometry(90.0)
    candidate = _oriented_rectangle_geometry(90.0, center=(160.0, 128.0))
    candidate.update({
        "skin_exclusion_applied": True,
        "skin_occlusion_on_seed": True,
        "skin_overlap_seed_pixels": 48,
        "skin_overlap_seed_ratio": 0.08,
    })

    accepted, telemetry = validate_roi_bottle_candidate(
        candidate,
        seed,
        image_shape=(240, 320, 3),
        current_patch_depth_mm=400.0,
    )

    assert accepted is False
    assert telemetry["roi_candidate_skin_overlap_material"] is True
    assert "hand_skin_occludes_locked_bottle_boundary" in telemetry["roi_candidate_reject_reason"]


def test_post_lock_partial_identity_rejects_lateral_frame_clipping():
    seed = _oriented_rectangle_geometry(90.0)
    candidate = copy.deepcopy(seed)
    candidate["component_image_boundary_touches"] = {
        "left": True,
        "top": False,
        "right": False,
        "bottom": False,
    }

    accepted, telemetry = validate_roi_bottle_candidate(
        candidate,
        seed,
        image_shape=(240, 320, 3),
        current_patch_depth_mm=400.0,
    )

    assert accepted is False
    assert telemetry["partial_clip_axis_relation"] == "LATERAL"
    assert telemetry["partial_observation_class"] == "PARTIAL_UNSAFE_REJECTED"
    assert "lateral_bottle_extent_clipped" in telemetry["roi_candidate_reject_reason"]


def test_fast_roi_prefilter_keeps_slender_tilted_bottle_with_large_aabb():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    mask = np.zeros((240, 320), dtype=np.uint8)
    box = cv2.boxPoints(((160.0, 120.0), (14.0, 154.0), 45.0)).astype(np.int32)
    cv2.fillPoly(mask, [box], 1)
    rgb[mask > 0] = [30, 80, 220]
    depth[mask > 0] = 400
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    seed = {
        "bbox_xyxy": [x, y, x + width, y + height],
        "bbox_xywh": [x, y, width, height],
        "contour": contour,
        "oriented_bbox": cv2.boxPoints(cv2.minAreaRect(contour)).tolist(),
        "center_px": [160.0, 120.0],
        "axis_vector": [2 ** -0.5, 2 ** -0.5],
        "depth_median_mm": 400.0,
        "depth_confidence": 1.0,
        "detection_confidence": 1.0,
    }

    pose, error, _timings = fast_roi_bottle_pose(
        rgb,
        depth,
        seed,
        scale=0.35,
        depth_hint_mm=400.0,
    )

    assert np.count_nonzero(mask) / float(width * height) < 0.18
    assert error is None
    assert pose is not None
    assert pose["candidate_score_breakdown"]["area_ratio_vs_depth_projection"] == pytest.approx(1.0, rel=0.08)


def test_fast_roi_search_seed_moves_centroid_prior_without_double_projecting_expected_area():
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(80.0, 120.0),
        width=20.0,
        height=40.0,
        depth_mm=400.0,
    )
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    for x1, x2 in ((60, 100), (220, 260)):
        rgb[80:160, x1:x2] = [30, 80, 220]
        depth[80:160, x1:x2] = 200

    original_prior, original_error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        seed,
        scale=1.0,
        depth_hint_mm=200.0,
    )
    projected_prior, projected_error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        seed,
        scale=1.0,
        depth_hint_mm=200.0,
        search_seed_bbox=[220.0, 80.0, 260.0, 160.0],
    )

    assert original_error is None and projected_error is None
    assert original_prior["center_px"][0] == pytest.approx(79.5, abs=1.0)
    assert projected_prior["center_px"][0] == pytest.approx(239.5, abs=1.0)
    for pose in (original_prior, projected_prior):
        breakdown = pose["candidate_score_breakdown"]
        assert breakdown["area_ratio_vs_seed"] == pytest.approx(4.0, rel=0.05)
        assert breakdown["area_ratio_vs_depth_projection"] == pytest.approx(1.0, rel=0.05)


def test_demo_fast_roi_does_not_expand_known_blue_identity_into_dark_fixture_pixels():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    rgb[45:205, 130:160] = (35, 90, 210)
    rgb[45:205, 160:190] = (3, 8, 20)
    depth[45:205, 130:190] = 420
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(159.5, 124.5),
        width=60.0,
        height=160.0,
        depth_mm=420.0,
    )

    pose, error, _timings = fast_roi_bottle_pose(
        rgb,
        depth,
        seed,
        scale=0.5,
        depth_hint_mm=420.0,
    )

    assert error is None
    assert pose is not None
    assert 28 <= pose["bbox_xywh"][2] <= 32
    assert abs(float(pose["center_px"][0]) - 144.5) <= 2.0
    assert pose["bottle_color_fraction"] >= 0.95


def test_demo_fast_roi_rejects_dark_depth_fixture_even_with_blue_background():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    # Large blue background crosses the remembered center but is far behind
    # the locked 420 mm bottle surface.
    rgb[10:230, 95:225] = (35, 90, 210)
    # The bottle is deliberately dark/achromatic so RGB-only fallback would
    # select the blue background instead of the current depth-backed body.
    rgb[45:205, 135:185] = (15, 18, 22)
    depth[45:205, 135:185] = 420
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(159.5, 124.5),
        width=50.0,
        height=160.0,
        depth_mm=420.0,
    )

    pose, error, _timings = fast_roi_bottle_pose(
        rgb,
        depth,
        seed,
        scale=0.5,
        depth_hint_mm=420.0,
    )

    assert pose is None
    assert error == "fast_roi_known_blue_rgbd_support_too_small"


def test_fast_roi_reuses_sufficient_locked_blue_support_before_generic_fallback(monkeypatch):
    seed = _oriented_rectangle_geometry(
        90.0,
        center=(160.0, 120.0),
        width=60.0,
        height=160.0,
        depth_mm=420.0,
    )
    rgb, depth = _paint_oriented_bottle_scene((240, 320, 3), seed, depth_mm=420)
    monkeypatch.setattr(
        realtime_runner,
        "candidate_masks_from_rgb",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generic fallback should not run")),
    )

    pose, error, _timings = fast_roi_bottle_pose(
        rgb,
        depth,
        seed,
        scale=0.35,
        depth_hint_mm=420.0,
    )

    assert error is None
    assert pose is not None
    assert pose["candidate_score_breakdown"]["known_blue_fast_path"] is True


def test_post_lock_requires_a_current_rgbd_contour_every_frame():
    assert realtime_runner.POST_LOCK_ROI_FAST_CONTOUR_INTERVAL == 1
    manager = _close_tracking_manager()
    manager.last_full_roi_contour_frame_id = 1
    rgb, depth = _partial_close_scene(200)

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=200.0,
    )

    assert telemetry["roi_tracking_mode"] == "fast_scaled_roi_contour"
    assert telemetry["roi_fast_scale_mode"] == "stable"
    assert telemetry["roi_fast_scale_requested"] == pytest.approx(0.5)
    assert telemetry["identity_measurement_current"] is True
    assert geometry["measurement_status"] == "CURRENT_MEASURED_RGBD"


def test_post_lock_roi_candidate_does_not_clear_recovery_until_3d_projection_validates_it():
    manager = _close_tracking_manager()
    manager.roi_bottle_fail_frames = 1
    rgb, depth = _partial_close_scene(200)

    recovered_geometry, recovered = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=200.0,
    )

    assert recovered["tracking_source"] == "roi_opencv_depth"
    assert recovered["roi_fast_scale_mode"] == "reacquire"
    assert recovered["roi_fast_scale_requested"] == pytest.approx(0.75)
    assert "previous_tracking_failure" in recovered["roi_fast_scale_reacquire_reasons"]
    assert manager.roi_bottle_fail_frames == 1

    stable_geometry, stable = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=3,
        current_patch_depth_mm=200.0,
    )

    assert recovered_geometry is not None and stable_geometry is not None
    assert stable["tracking_source"] == "roi_opencv_depth"
    assert stable["roi_fast_scale_mode"] == "reacquire"
    assert stable["roi_fast_scale_requested"] == pytest.approx(0.75)


def test_normal_approach_motion_and_persistent_failure_keep_cheap_roi_scale():
    rgb, depth = _partial_close_scene(200)
    moving = _close_tracking_manager()
    moving.last_good_raw_bottle_velocity_px = [3.0, 0.0]

    _geometry, ordinary = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        moving,
        frame_id=2,
        current_patch_depth_mm=200.0,
        estimated_translation_px=[3.0, 0.0],
    )

    assert ordinary["roi_fast_scale_mode"] == "stable"
    assert ordinary["roi_fast_scale_requested"] == pytest.approx(0.5)
    assert ordinary["roi_fast_scale_reacquire_reasons"] == []

    persistent = _close_tracking_manager()
    persistent.roi_bottle_fail_frames = 2
    _geometry, waiting_for_broad = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        persistent,
        frame_id=2,
        current_patch_depth_mm=200.0,
    )

    assert waiting_for_broad["roi_previous_failure_count"] == 2
    assert waiting_for_broad["roi_fast_scale_mode"] == "stable"
    assert waiting_for_broad["roi_fast_scale_requested"] == pytest.approx(0.5)


def test_repeated_full_recovery_uses_bounded_backoff_until_true_3d_returns():
    assert realtime_runner.locked_full_recovery_interval_sec(0) == pytest.approx(
        realtime_runner.POST_LOCK_FULL_REACQUIRE_MIN_INTERVAL_SEC
    )
    assert realtime_runner.locked_full_recovery_interval_sec(1) == pytest.approx(
        realtime_runner.POST_LOCK_FULL_REACQUIRE_BACKOFF_SEC
    )
    assert realtime_runner.locked_full_recovery_interval_sec(50) == pytest.approx(
        realtime_runner.POST_LOCK_FULL_REACQUIRE_BACKOFF_SEC
    )
    assert realtime_runner.report_has_current_true_3d_lock_projection(
        {
            "target_lock": {
                "current_tracked_target": {
                    "identity_measurement_current": True,
                    "locked_projection_mode": "TRUE_3D",
                    "projection_valid": True,
                }
            }
        }
    ) is True
    assert realtime_runner.report_has_current_true_3d_lock_projection(
        {
            "target_lock": {
                "current_tracked_target": {
                    "identity_measurement_current": True,
                    "locked_projection_mode": "INVALID_3D",
                    "projection_valid": False,
                    "operator_locked_fraction_visual_guide_current": True,
                }
            }
        }
    ) is False

def _reanchored_band(depth_mm, half_width):
    contour = [[320 - half_width, 100], [320 + half_width, 100], [320 + half_width, 380], [320 - half_width, 380]]
    geometry = {"contour": contour, "center_px": [320, 240], "axis_vector": [0, 1], "depth_median_mm": depth_mm}
    depth = np.full((480, 640), depth_mm, dtype=np.uint16)
    locked = {
        "locked_body_fraction": 0.5,
        "bottle_contact_model": {"patch_height_mm": 50.0, "patch_width_mm": 15.0},
    }
    clamp = {"grasp_line": {"direction_px": [1, 0]}, "clamp_patch_height_px": 60, "clamp_patch_width_px": 18}
    return reanchor_target_clamp_to_current_edges(
        clamp,
        geometry,
        locked,
        depth,
        depth.shape,
        {"fx": 600.0, "fy": 600.0},
    )[0]


def test_grasp_band_tracks_contour_width_and_metric_patch_scale_during_approach():
    far = _reanchored_band(500, 30)
    close = _reanchored_band(250, 60)

    assert close["grip_width_px"] > far["grip_width_px"] * 1.8
    assert close["clamp_patch_height_px"] == far["clamp_patch_height_px"] * 2.0
    assert close["clamp_patch_width_px"] == far["clamp_patch_width_px"] * 2.0
    assert close["contact_patch_scale_source"] == "current_rgbd_depth_and_intrinsics"


def test_current_contact_center_and_metric_patch_scale_reach_decision_geometry_immediately():
    manager = _tilt_lock_manager()
    intrinsics = {"fx": 500.0, "fy": 500.0}
    far_geometry = _oriented_rectangle_geometry(
        90.0,
        center=(160.0, 120.0),
        width=40.0,
        height=140.0,
        depth_mm=400.0,
    )
    close_geometry = _oriented_rectangle_geometry(
        90.0,
        center=(164.0, 120.0),
        width=80.0,
        height=140.0,
        depth_mm=200.0,
    )
    far_depth = np.full((240, 320), 400, dtype=np.uint16)
    close_depth = np.full((240, 320), 200, dtype=np.uint16)

    far_raw, far_trace = reanchor_target_clamp_to_current_edges(
        {"clamp_patch_height_px": 40.0, "clamp_patch_width_px": 12.0},
        far_geometry,
        manager.locked_target,
        far_depth,
        far_depth.shape,
        intrinsics,
    )
    far_current, _ = stabilize_magenta_contact_geometry(
        manager,
        far_raw,
        far_trace,
        far_geometry,
        frame_id=1,
    )
    close_raw, close_trace = reanchor_target_clamp_to_current_edges(
        far_current,
        close_geometry,
        manager.locked_target,
        close_depth,
        close_depth.shape,
        intrinsics,
    )
    close_current, stabilized_trace = stabilize_magenta_contact_geometry(
        manager,
        close_raw,
        close_trace,
        close_geometry,
        frame_id=2,
    )

    np.testing.assert_allclose(close_current["clamp_center_px"], close_raw["clamp_center_px"], atol=1e-6)
    np.testing.assert_allclose(close_current["left_contact_px"], close_raw["left_contact_px"], atol=1e-6)
    np.testing.assert_allclose(close_current["right_contact_px"], close_raw["right_contact_px"], atol=1e-6)
    assert close_current["clamp_center_px"][0] == pytest.approx(164.0)
    assert close_current["clamp_patch_height_px"] == pytest.approx(close_raw["clamp_patch_height_px"])
    assert close_current["clamp_patch_width_px"] == pytest.approx(close_raw["clamp_patch_width_px"])
    assert close_current["clamp_patch_height_px"] == pytest.approx(2.0 * far_current["clamp_patch_height_px"])
    assert close_current["clamp_patch_width_px"] == pytest.approx(2.0 * far_current["clamp_patch_width_px"])
    assert stabilized_trace["edge_smoothing_alpha"] == 1.0
    assert stabilized_trace["width_smoothing_alpha"] == 1.0


def _stabilized_tilt_frame(manager, previous_clamp, angle_deg, frame_id):
    geometry = _oriented_rectangle_geometry(angle_deg)
    depth = np.full((240, 320), 400, dtype=np.uint16)
    raw_clamp, edge_trace = reanchor_target_clamp_to_current_edges(
        previous_clamp,
        geometry,
        manager.locked_target,
        depth,
        depth.shape,
        {"fx": 500.0, "fy": 500.0},
    )
    return stabilize_magenta_contact_geometry(
        manager,
        raw_clamp,
        edge_trace,
        geometry,
        frame_id,
    )


def _tilt_lock_manager():
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "lock_frame_id": 21,
        "locked_body_fraction": 0.5,
        "bottle_contact_model": {"patch_height_mm": 50.0, "patch_width_mm": 15.0},
    }
    return manager


def test_sustained_current_in_place_tilt_rotates_grasp_band_after_confirmation():
    manager = _tilt_lock_manager()
    clamp = {"clamp_patch_height_px": 40.0, "clamp_patch_width_px": 12.0}
    clamp, _trace = _stabilized_tilt_frame(manager, clamp, 90.0, 0)
    traces = []
    for frame_id, angle in zip(range(2, 24, 2), [70.0] * 11):
        clamp, trace = _stabilized_tilt_frame(manager, clamp, angle, frame_id)
        traces.append(trace)

    assert traces[0]["angle_update_rejected_reason"] == "orientation_change_awaiting_temporal_confirmation"
    assert traces[1]["angle_update_rejected_reason"] == "orientation_change_awaiting_temporal_confirmation"
    pending_tracking = {
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "tracking_ok": True,
        "contact_patch_tracking_ok": True,
        "bottle_geometry_tracking_ok": True,
        "locked_patch_visible": True,
        "tracking_confidence": 1.0,
        "patch_depth_confidence": 1.0,
        "patch_width_valid": True,
        "patch_on_bottle_body": True,
        "angle_update_rejected_reason": traces[0]["angle_update_rejected_reason"],
    }
    assert tracking_good_from_values(pending_tracking, target_status="LOCKED") is False
    assert "orientation_measurement_pending" in ready_blockers_from_values(
        {"grip_width_in_range": True, "contacts_on_body": True},
        pending_tracking,
        {"force_closure": True},
        100.0,
        [],
        target_status="LOCKED",
    )
    assert traces[2]["orientation_update_confirmed"] is True
    assert abs(realtime_runner.line_angle_delta_deg(clamp["magenta_box_angle_deg"], 70.0)) < 1.5
    bottle_axis = np.asarray(clamp["bottle_y_axis"], dtype=float)
    grasp_axis = np.asarray(clamp["grasp_line"]["direction_px"], dtype=float)
    polygon = np.asarray(clamp["clamp_band_polygon"], dtype=float)
    band_height_axis = polygon[3] - polygon[0]
    assert abs(float(np.dot(bottle_axis, grasp_axis))) < 1e-6
    assert abs(float(np.cross(bottle_axis, band_height_axis / np.linalg.norm(band_height_axis)))) < 1e-6


def test_single_frame_or_alternating_angle_spikes_do_not_rotate_grasp_band():
    manager = _tilt_lock_manager()
    clamp = {"clamp_patch_height_px": 40.0, "clamp_patch_width_px": 12.0}
    clamp, _trace = _stabilized_tilt_frame(manager, clamp, 90.0, 0)

    for frame_id, angle in zip(range(2, 18, 2), [70.0, 90.0, 70.0, 110.0, 70.0, 110.0, 90.0, 90.0]):
        clamp, trace = _stabilized_tilt_frame(manager, clamp, angle, frame_id)
        assert trace["orientation_update_confirmed"] is False

    assert abs(realtime_runner.line_angle_delta_deg(clamp["magenta_box_angle_deg"], 90.0)) < 1e-6


def _partial_close_scene(
    depth_mm,
    *,
    object_depth_value=None,
    center_x=160,
    reference_depth_mm=200.0,
):
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    half_width = int(round(25.0 * float(reference_depth_mm) / float(depth_mm)))
    x1 = max(0, center_x - half_width)
    x2 = min(320, center_x + half_width)
    # RGB ordering: this is the known navy/blue demo bottle, not a generic
    # chromatic contour. Demo identity must not be recoverable from the black
    # hand mount or a depth-only silhouette.
    rgb[:, x1:x2] = [35, 90, 210]
    depth[:, x1:x2] = int(depth_mm if object_depth_value is None else object_depth_value)
    return rgb, depth


def _close_tracking_manager(seed_depth_mm=200.0):
    seed = {
        "detected": True,
        "bbox_xyxy": [135.0, 40.0, 185.0, 200.0],
        "bbox_xywh": [135.0, 40.0, 50.0, 160.0],
        "center_px": [160.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": float(seed_depth_mm),
        "depth_confidence": 1.0,
    }
    return SimpleNamespace(
        last_good_bottle_geometry=dict(seed),
        last_good_bottle_bbox=list(seed["bbox_xyxy"]),
        last_good_bottle_depth_mm=float(seed_depth_mm),
        last_good_bottle_frame_id=1,
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_full_roi_contour_frame_id=None,
        roi_bottle_fail_frames=0,
        locked_target={"valid": True, "bottle_target_id": "bottle_01"},
        grasp_memory={"status": "LOCKED_BOTTLE_RELATIVE_GRASP", "locked_body_fraction": 0.5},
    )


def _close_range_target_clamp(center_y=120.0):
    left = [120.0, float(center_y)]
    right = [200.0, float(center_y)]
    half_patch_width = 6.0
    half_patch_height = 18.0

    def patch(center):
        x, y = center
        return {
            "center_px": [x, y],
            "polygon": [
                [x - half_patch_width, y - half_patch_height],
                [x + half_patch_width, y - half_patch_height],
                [x + half_patch_width, y + half_patch_height],
                [x - half_patch_width, y + half_patch_height],
            ],
        }

    return {
        "clamp_center_px": [160.0, float(center_y)],
        "left_contact_px": left,
        "right_contact_px": right,
        "left_clamp_patch": patch(left),
        "right_clamp_patch": patch(right),
        "clamp_band_polygon": [
            [120.0, center_y - half_patch_height],
            [200.0, center_y - half_patch_height],
            [200.0, center_y + half_patch_height],
            [120.0, center_y + half_patch_height],
        ],
        "grasp_line": {
            "left_endpoint_px": left,
            "right_endpoint_px": right,
            "center_px": [160.0, float(center_y)],
            "direction_px": [1.0, 0.0],
        },
        "clamp_patch_height_px": 36.0,
        "clamp_patch_width_px": 12.0,
        "grip_width_px": 80.0,
        "bottle_y_axis_px": [0.0, 1.0],
        "bottle_x_axis_px": [1.0, 0.0],
    }


def test_partial_visible_extent_does_not_remap_band_height_onto_cropped_bbox():
    locked_target = {
        "attempt_id": "attempt-no-climb",
        "locked_body_fraction": 0.74,
        "locked_bottle_geometry": {
            "bbox_xyxy": [135.0, 40.0, 185.0, 190.0],
            "center_px": [160.0, 115.0],
            "axis_vector": [0.0, 1.0],
        },
        "fixed_clamp_zone": _close_range_target_clamp(center_y=151.0),
        "locked_grasp_band": {"locked_band_center_px": [160.0, 151.0]},
    }
    depth = np.full((240, 320), 900, dtype=np.uint16)

    def geometry(y1, y2, *, partial):
        mask = np.zeros((240, 320), dtype=np.uint8)
        mask[int(y1):int(y2), 135:186] = 1
        depth[int(y1):int(y2), 135:186] = 200
        return {
            "detected": True,
            "bbox_xyxy": [135.0, float(y1), 185.0, float(y2)],
            "bbox_xywh": [135.0, float(y1), 50.0, float(y2 - y1)],
            "center_px": [160.0, (float(y1) + float(y2)) / 2.0],
            "axis_vector": [0.0, 1.0],
            "depth_median_mm": 200.0,
            "cleaned_mask": mask,
            "body_mask": mask,
            "partial_extent": partial,
        }

    clamp = copy.deepcopy(locked_target["fixed_clamp_zone"])
    full_guide, _ = realtime_runner.build_operator_locked_fraction_guide(
        clamp,
        geometry(40, 190, partial=False),
        locked_target,
        depth,
        (240, 320, 3),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
    )
    assert full_guide["clamp_center_px"][1] == pytest.approx(151.0, abs=1.0)

    partial_centers = []
    for y1, y2 in ((55, 190), (70, 190), (40, 175), (40, 165)):
        guide, _ = realtime_runner.build_operator_locked_fraction_guide(
            full_guide,
            geometry(y1, y2, partial=True),
            locked_target,
            depth,
            (240, 320, 3),
            {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        )
        partial_centers.append(float(guide["clamp_center_px"][1]))
        assert guide["operator_partial_extent_center_preserved"] is True
        assert locked_target["attempt_id"] == "attempt-no-climb"

    assert max(abs(center_y - 151.0) for center_y in partial_centers) <= 3.0


def test_partial_locked_view_rechecks_current_finger_corridors():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:200, 100:221] = 1
    geometry = {
        "body_mask": mask,
        "cleaned_mask": mask,
        "bbox_xyxy": [100.0, 40.0, 221.0, 200.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 200.0,
        "partial_extent": True,
    }
    clamp = {
        "left_contact_px": [100.0, 120.0],
        "right_contact_px": [220.0, 120.0],
        "clamp_band_polygon": [
            [100.0, 102.0],
            [220.0, 102.0],
            [220.0, 138.0],
            [100.0, 138.0],
        ],
    }
    clear_depth = np.full((240, 320), 900, dtype=np.uint16)
    clear_depth[mask > 0] = 200
    clear_tracking = {
        "partial_observation_class": "PARTIAL_AXIAL_FRAME_CLIP_CURRENT",
        "identity_measurement_current": True,
        "current_depth_mm": 200.0,
    }

    refresh_current_depth_corridor_evidence(
        clear_tracking,
        clear_depth,
        clamp,
        geometry,
        (240, 320, 3),
        {"fx": 500.0},
    )

    assert clear_tracking["current_depth_corridor_evidence_current"] is True
    assert clear_tracking["current_left_depth_corridor_blocked"] is False
    assert clear_tracking["current_right_depth_corridor_blocked"] is False

    blocked_depth = clear_depth.copy()
    blocked_depth[102:138, 60:96] = 180
    blocked_tracking = dict(clear_tracking)
    refresh_current_depth_corridor_evidence(
        blocked_tracking,
        blocked_depth,
        clamp,
        geometry,
        (240, 320, 3),
        {"fx": 500.0},
    )

    assert blocked_tracking["current_left_depth_corridor_blocked"] is True
    blockers = ready_blockers_from_values(
        {"contact_model_ready": True, "visual_contact_ready": True},
        {
            **blocked_tracking,
            "identity_tracking_ok": True,
            "locked_patch_visible": True,
            "contact_patch_tracking_ok": True,
            "tracking_confidence": 100.0,
            "patch_depth_confidence": 100.0,
            "bottle_geometry_tracking_ok": True,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
        },
        {"force_closure": True},
        100.0,
        [],
        target_status="LOCKED",
    )
    assert "current_grasp_corridor_blocked" in blockers


def _bottom_clipped_geometry(depth_mm):
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[20:240, 80:241] = 1
    contour = np.asarray(
        [[[80, 20]], [[240, 20]], [[240, 239]], [[80, 239]]],
        dtype=np.int32,
    )
    return {
        "detected": True,
        "target_id": "current_locked_bottle_observation",
        "contour": contour,
        "cleaned_mask": mask,
        "body_mask": mask,
        "bbox_xyxy": [80.0, 20.0, 241.0, 240.0],
        "bbox_xywh": [80.0, 20.0, 161.0, 220.0],
        "bottle_extent_bbox_xyxy": [80.0, 20.0, 241.0, 240.0],
        "center_px": [160.0, 129.5],
        "axis_vector": [0.0, 1.0],
        "axis_angle_degrees": 90.0,
        "depth_median_mm": float(depth_mm),
        "depth_confidence": 1.0,
        "detection_confidence": 1.0,
        "partial_extent": True,
        "measurement_status": "CURRENT_MEASURED_RGBD",
        "geometry_source": "roi_opencv_depth",
    }


def test_bbox_ending_exactly_on_bottom_edge_is_partial_extent():
    projected, telemetry = realtime_runner.project_bbox_for_depth(
        [100.0, 40.0, 200.0, 240.0],
        200.0,
        200.0,
        (240, 320, 3),
    )

    assert projected == [100.0, 40.0, 200.0, 240.0]
    assert telemetry["roi_projected_visible_fraction"] == pytest.approx(1.0)
    assert telemetry["roi_projected_extent_partial"] is True


def test_interior_partial_extent_never_reports_full_bottle_visible():
    geometry = {
        "detected": True,
        "bbox_xyxy": [100.0, 30.0, 220.0, 200.0],
        "bbox_xywh": [100.0, 30.0, 120.0, 170.0],
        "bottle_extent_bbox_xyxy": [100.0, 30.0, 220.0, 200.0],
        "partial_extent": True,
        "projected_visible_fraction": 0.72,
    }

    visibility = post_lock_extent_visibility(geometry, (240, 320, 3))

    assert min(visibility["border_margins"].values()) >= visibility["required_margin_px"]
    assert visibility["bbox_size_ratio"]["width"] < 0.82
    assert visibility["bbox_size_ratio"]["height"] < 0.88
    assert visibility["full_bottle_visible"] is False


def test_depth_projected_partial_bottle_stays_current_at_80_to_100_mm():
    for frame_id, depth_mm in enumerate((100, 90, 80), start=2):
        manager = _close_tracking_manager()
        rgb, depth = _partial_close_scene(depth_mm)
        original_memory = dict(manager.grasp_memory)

        geometry, telemetry = roi_opencv_depth_bottle_tracking(
            rgb,
            depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(depth_mm),
        )

        assert telemetry["tracking_source"] == "roi_opencv_depth"
        assert telemetry["roi_candidate_raw_area_ratio"] > 2.5
        assert telemetry["roi_projected_extent_partial"] is True
        assert geometry["measurement_status"] == "CURRENT_MEASURED_RGBD"
        assert geometry["partial_extent"] is True
        assert manager.locked_target["bottle_target_id"] == "bottle_01"
        assert manager.grasp_memory == original_memory
        assert post_lock_extent_visibility(geometry, rgb.shape)["full_bottle_visible"] is False


def test_one_lock_tracks_sequential_depth_growth_into_hand_length_range():
    manager = _close_tracking_manager(seed_depth_mm=300.0)
    original_memory = dict(manager.grasp_memory)

    for frame_id, depth_mm in enumerate((280, 260, 240, 220, 200, 180), start=2):
        rgb, depth = _partial_close_scene(depth_mm, reference_depth_mm=300.0)
        geometry, telemetry = roi_opencv_depth_bottle_tracking(
            rgb,
            depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(depth_mm),
        )

        assert telemetry["tracking_source"] == "roi_opencv_depth"
        assert telemetry["roi_candidate_reject_reason"] == []
        assert geometry["measurement_status"] == "CURRENT_MEASURED_RGBD"
        assert geometry["depth_median_mm"] == depth_mm
        assert manager.locked_target["bottle_target_id"] == "bottle_01"
        assert manager.grasp_memory == original_memory


def test_sequential_close_search_projection_uses_immutable_full_seed_and_accepted_depth():
    manager = _close_tracking_manager(seed_depth_mm=300.0)
    immutable_full_seed = copy.deepcopy(manager.last_good_bottle_geometry)

    for frame_id, depth_mm in enumerate((280, 240, 200, 180), start=2):
        previous_full_seed = copy.deepcopy(manager.last_good_bottle_geometry)
        rgb, depth = _partial_close_scene(depth_mm, reference_depth_mm=300.0)
        geometry, telemetry = roi_opencv_depth_bottle_tracking(
            rgb,
            depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(depth_mm),
        )
        expected_bbox, expected_projection = realtime_runner.project_bbox_for_depth(
            previous_full_seed["bbox_xyxy"],
            previous_full_seed["depth_median_mm"],
            geometry["depth_median_mm"],
            rgb.shape,
        )

        assert telemetry["identity_measurement_current"] is True
        assert geometry["depth_median_mm"] == depth_mm
        assert telemetry["roi_search_depth_projection_scale"] == pytest.approx(
            previous_full_seed["depth_median_mm"] / geometry["depth_median_mm"]
        )
        assert telemetry["roi_search_depth_projection_bbox_unclipped"] == pytest.approx(
            expected_projection["roi_projected_bbox_unclipped"]
        )
        assert telemetry["roi_search_seed_bbox_xyxy"] == pytest.approx(expected_bbox)
        assert manager.last_good_bottle_geometry == immutable_full_seed


def test_active_lock_and_original_fraction_survive_sequential_close_approach():
    seed_manager = _close_tracking_manager(seed_depth_mm=300.0)
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.37,
    }
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.37,
    }
    manager.last_good_bottle_geometry = dict(seed_manager.last_good_bottle_geometry)
    manager.last_good_bottle_bbox = list(seed_manager.last_good_bottle_bbox)
    manager.last_good_bottle_depth_mm = seed_manager.last_good_bottle_depth_mm
    manager.last_good_bottle_frame_id = 1
    manager.last_full_roi_contour_frame_id = 1

    for frame_id, depth_mm in enumerate((280, 260, 240, 220, 200, 180), start=2):
        rgb, depth = _partial_close_scene(depth_mm, reference_depth_mm=300.0)
        geometry, telemetry = roi_opencv_depth_bottle_tracking(
            rgb,
            depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(depth_mm),
        )
        tracking = {
            "tracking_ok": True,
            "locked_patch_visible": True,
            "patch_depth_confidence": 100.0,
            "tracking_confidence": 90.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
            **telemetry,
        }
        separate_identity_from_contact_tracking(
            tracking,
            bottle_geometry_tracking_ok=telemetry["tracking_source"] == "roi_opencv_depth",
        )
        manager.apply_tracking_counters(tracking)

        assert geometry["depth_median_mm"] == depth_mm
        assert manager.status() == "LOCKED"
        assert manager.locked_target["bottle_target_id"] == "bottle_01"
        assert manager.locked_target["locked_body_fraction"] == 0.37
        assert manager.grasp_memory["locked_body_fraction"] == 0.37
        assert tracking["identity_tracking_ok"] is True


def test_bottom_clipped_hand_length_sequence_does_not_ratchet_or_switch_target(monkeypatch):
    anchor = _close_range_target_clamp()
    locked_geometry = _bottom_clipped_geometry(300.0)
    locked_geometry.update({
        "partial_extent": False,
        "bbox_xyxy": [135.0, 40.0, 185.0, 200.0],
        "bbox_xywh": [135.0, 40.0, 50.0, 160.0],
        "bottle_extent_bbox_xyxy": [135.0, 40.0, 185.0, 200.0],
        "center_px": [160.0, 120.0],
        "depth_median_mm": 300.0,
    })
    contact_model = {
        "left_contact_patch": copy.deepcopy(anchor["left_clamp_patch"]),
        "right_contact_patch": copy.deepcopy(anchor["right_clamp_patch"]),
        "grasp_line": copy.deepcopy(anchor["grasp_line"]),
        "contact_band_polygon": copy.deepcopy(anchor["clamp_band_polygon"]),
        "patch_height_px": 36.0,
        "patch_width_px": 12.0,
        "patch_height_mm": 50.0,
        "patch_width_mm": 15.0,
        "grip_width_px": 80.0,
        "grip_width_mm": 64.0,
        "planned_depth_mm": 300.0,
        "axis_direction_px": [0.0, 1.0],
    }
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-original",
        "lock_frame_id": 1,
        "locked_body_fraction": 0.52,
        "locked_grip_width_mm": 64.0,
        "locked_bottle_geometry": copy.deepcopy(locked_geometry),
        "fixed_clamp_zone": copy.deepcopy(anchor),
        "bottle_contact_model": contact_model,
    }
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
    }
    manager.last_good_bottle_geometry = copy.deepcopy(locked_geometry)
    manager.last_good_bottle_bbox = list(locked_geometry["bbox_xyxy"])
    manager.last_good_bottle_depth_mm = 300.0
    manager.last_good_bottle_frame_id = 1
    manager.current_tracked_target = {
        "target_clamp": copy.deepcopy(anchor),
        "last_verified_rgbd_target_clamp": copy.deepcopy(anchor),
        "last_verified_rgbd_target_clamp_frame_id": 1,
        "identity_measurement_current": False,
        "identity_measurement_age_frames": 1,
    }
    active_depth = {"value": 300.0}

    def biased_local_tracking(_depth, _locked, previous, _shape, **_kwargs):
        verified = copy.deepcopy(previous["last_verified_rgbd_target_clamp"])
        biased = realtime_runner.shifted_target_clamp(
            verified,
            [0.0, -realtime_runner.LOCAL_DEPTH_MOTION_RECENT_IDENTITY_MAX_OFFSET_PX],
        )
        return None, {
            "tracking_ok": True,
            "locked_patch_visible": True,
            "patch_depth_confidence": 100.0,
            "tracking_confidence": 100.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
            "local_patch_tracking_used": True,
            "local_motion_application_status": "BOUNDED_RECENT_IDENTITY_BRIDGE_ESTIMATE",
            "identity_measurement_current": False,
            "identity_measurement_age_frames": 1,
            "target_clamp": biased,
            "last_verified_rgbd_target_clamp": verified,
            "last_verified_rgbd_target_clamp_frame_id": previous.get(
                "last_verified_rgbd_target_clamp_frame_id"
            ),
            "current_depth_mm": active_depth["value"],
            "current_tracked_bbox_xyxy": [80.0, 20.0, 241.0, 240.0],
            "current_tracked_bbox_xywh": [80.0, 20.0, 161.0, 220.0],
            "delta_from_locked_center_px": [0.0, -12.0],
            "movement_deadband_state": manager.locked_target.setdefault(
                "movement_deadband_state", {}
            ),
            "target_identity_preserved": True,
        }

    def current_partial_geometry(*_args, **_kwargs):
        geometry = _bottom_clipped_geometry(active_depth["value"])
        return geometry, {
            "tracking_source": "roi_opencv_depth",
            "identity_measurement_current": True,
            "identity_measurement_age_frames": 0,
            "roi_projected_extent_partial": True,
            "roi_projected_visible_fraction": 0.82,
            "roi_candidate_reject_reason": [],
            "roi_bottle_tracking_timings_ms": {},
            "accepted_current_bottle_bbox": geometry["bbox_xyxy"],
            "last_good_bottle_bbox": geometry["bbox_xyxy"],
        }

    monkeypatch.setattr(realtime_runner, "local_locked_patch_tracking", biased_local_tracking)
    monkeypatch.setattr(realtime_runner, "roi_opencv_depth_bottle_tracking", current_partial_geometry)
    monkeypatch.setattr(realtime_runner, "write_status_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "draw_fast_overlay", lambda rgb, *_args, **_kwargs: rgb)
    monkeypatch.setattr(realtime_runner, "detect_finger_contacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        realtime_runner,
        "apply_display_geometry_smoothing",
        lambda _manager, geometry, guide, *_args, **_kwargs: {
            "display_target_clamp": copy.deepcopy(guide),
            "display_bottle_bbox": geometry.get("bbox_xyxy"),
            "display_magenta_center": guide.get("clamp_center_px"),
        },
    )
    monkeypatch.setattr(realtime_runner, "update_performance_profile", lambda *_args, **_kwargs: None)

    latest_report = {
        "decision": "ADJUST POSITION",
        "safety_score": 90.0,
        "failure_reasons": [],
        "epsilon_result": {"force_closure": True, "epsilon": 0.04},
        "planned_contacts": {
            "contact_model_ready": True,
            "visual_contact_ready": True,
            "contacts_on_body": True,
            "grip_width_in_range": True,
            "target_clamp": copy.deepcopy(anchor),
        },
        "multi_bottle_target_selection": {
            "selected_target_id": "background_bottle_02",
            "candidate_count": 2,
            "eligible_count": 1,
            "candidates": [
                {"target_id": "background_bottle_02", "detection_confidence": 0.99},
                {"target_id": "bottle_01", "detection_confidence": 0.61},
            ],
        },
    }
    centers_y = []
    verified_centers_y = []
    for frame_id, depth_mm in enumerate((280, 260, 240, 220, 210, 180), start=2):
        active_depth["value"] = float(depth_mm)
        depth = np.full((240, 320), 900, dtype=np.uint16)
        depth[20:240, 80:241] = depth_mm
        report = realtime_runner.fast_track_frame(
            np.zeros((240, 320, 3), dtype=np.uint8),
            depth,
            {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
            frame_id,
            SimpleNamespace(),
            manager,
            latest_report=latest_report,
        )
        latest_report = report
        tracking = manager.current_tracked_target
        centers_y.append(float(tracking["target_clamp"]["clamp_center_px"][1]))
        verified_centers_y.append(
            float(tracking["last_verified_rgbd_target_clamp"]["clamp_center_px"][1])
        )
        assert manager.status() == "LOCKED"
        assert manager.locked_target["bottle_target_id"] == "bottle_01"
        assert manager.locked_target["locked_body_fraction"] == 0.52
        assert manager.grasp_memory["bottle_target_id"] == "bottle_01"
        assert report["selected_bottle_target_id"] == "bottle_01"
        assert report["bottle_geometry"]["partial_extent"] is True
        assert report["bottle_geometry"]["depth_median_mm"] == depth_mm
        assert tracking["grasp_band_projection_residual_valid"] is False
        assert tracking["strict_current_bottle_geometry"] is False
        assert tracking["operator_locked_fraction_visual_guide_current"] is False
        assert tracking["display_target_clamp"] == {}
        assert manager.last_good_bottle_geometry["bbox_xyxy"] == locked_geometry["bbox_xyxy"]
        assert np.allclose(
            manager.last_good_bottle_geometry["center_px"],
            locked_geometry["center_px"],
        )

    max_offset = realtime_runner.LOCAL_DEPTH_MOTION_RECENT_IDENTITY_MAX_OFFSET_PX
    assert min(centers_y) >= anchor["clamp_center_px"][1] - max_offset
    assert min(verified_centers_y) >= anchor["clamp_center_px"][1] - max_offset
    assert manager.locked_target["attempt_id"] == "attempt-original"


def test_fast_overlay_receives_only_current_measured_rgbd_mask(monkeypatch):
    seed = _close_tracking_manager(seed_depth_mm=300.0)
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        **seed.locked_target,
        "locked_body_fraction": 0.5,
        "locked_bottle_geometry": copy.deepcopy(seed.last_good_bottle_geometry),
        "fixed_clamp_zone": _close_range_target_clamp(),
    }
    manager.grasp_memory = copy.deepcopy(seed.grasp_memory)
    manager.last_good_bottle_geometry = copy.deepcopy(seed.last_good_bottle_geometry)
    manager.last_good_bottle_bbox = list(seed.last_good_bottle_bbox)
    manager.last_good_bottle_depth_mm = seed.last_good_bottle_depth_mm
    manager.last_good_bottle_frame_id = seed.last_good_bottle_frame_id
    manager.current_tracked_target = {
        "target_clamp": _close_range_target_clamp(),
        "identity_measurement_current": True,
        "current_depth_mm": 300.0,
    }
    rgb, depth = _partial_close_scene(280.0, reference_depth_mm=300.0)
    seen = []

    monkeypatch.setattr(realtime_runner, "write_status_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        realtime_runner,
        "draw_fast_overlay",
        lambda frame, pose, *_args, **_kwargs: seen.append(pose) or frame,
    )
    monkeypatch.setattr(realtime_runner, "detect_finger_contacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(realtime_runner, "update_performance_profile", lambda *_args, **_kwargs: None)

    report = realtime_runner.fast_track_frame(
        rgb,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        2,
        SimpleNamespace(),
        manager,
        latest_report={},
    )

    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0]["measurement_status"] == "CURRENT_MEASURED_RGBD"
    assert seen[0]["mask_measurement_status"] == "CURRENT_MEASURED_RGBD"
    assert seen[0]["cleaned_mask"] is not None
    assert report["overlay_bottle_geometry_mode"] == "CURRENT_MEASURED_RGBD_MASK"

    monkeypatch.setattr(
        realtime_runner,
        "roi_opencv_depth_bottle_tracking",
        lambda *_args, **_kwargs: (
            None,
            {
                "tracking_source": "partial_rgbd_hand_occlusion_motion_estimate",
                "identity_measurement_current": False,
                "roi_candidate_reject_reason": ["hand_occlusion_identity_not_current"],
                "roi_bottle_tracking_timings_ms": {},
            },
        ),
    )
    held_report = realtime_runner.fast_track_frame(
        rgb,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        3,
        SimpleNamespace(),
        manager,
        latest_report=report,
    )

    assert len(seen) == 2
    assert seen[1] is None
    assert held_report["overlay_bottle_geometry_mode"] != "CURRENT_MEASURED_RGBD_MASK"


def test_fast_association_without_valid_3d_cannot_publish_current_bottle_guide(
    monkeypatch,
):
    seed = _close_tracking_manager(seed_depth_mm=300.0)
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        **seed.locked_target,
        "attempt_id": "attempt-visual-guide",
        "lock_frame_id": 1,
        "locked_body_fraction": 0.5,
        "locked_bottle_geometry": copy.deepcopy(seed.last_good_bottle_geometry),
        "fixed_clamp_zone": _close_range_target_clamp(),
    }
    manager.grasp_memory = copy.deepcopy(seed.grasp_memory)
    manager.last_good_bottle_geometry = copy.deepcopy(seed.last_good_bottle_geometry)
    manager.last_good_bottle_bbox = list(seed.last_good_bottle_bbox)
    manager.last_good_bottle_depth_mm = seed.last_good_bottle_depth_mm
    manager.last_good_bottle_frame_id = seed.last_good_bottle_frame_id
    manager.current_tracked_target = {
        "target_clamp": _close_range_target_clamp(),
        "identity_measurement_current": True,
        "current_depth_mm": 300.0,
    }
    rgb, depth = _partial_close_scene(280.0, reference_depth_mm=300.0)

    def analytic_projection_failure(*_args, **_kwargs):
        return (
            {},
            {},
            {
                "locked_projection_mode": "INVALID_3D",
                "projection_mode": "INVALID_3D",
                "projection_valid": False,
                "projection_invalid_reason": "locked_3d_band_not_supported_by_current_mask",
            },
            {},
        )

    monkeypatch.setattr(
        realtime_runner,
        "select_locked_band_projection",
        analytic_projection_failure,
    )
    monkeypatch.setattr(realtime_runner, "write_status_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "draw_fast_overlay", lambda rgb, *_args, **_kwargs: rgb)
    monkeypatch.setattr(realtime_runner, "detect_finger_contacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(realtime_runner, "update_performance_profile", lambda *_args, **_kwargs: None)

    current = realtime_runner.fast_track_frame(
        rgb,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        2,
        SimpleNamespace(),
        manager,
        latest_report={},
    )
    tracking = manager.current_tracked_target
    assert tracking["locked_projection_mode"] == "INVALID_3D"
    assert tracking["projection_valid"] is False
    assert current["ready_to_close"] is False
    assert tracking["operator_locked_fraction_visual_guide_current"] is False
    assert tracking["operator_locked_fraction_visual_guide_role"] is None
    assert tracking["display_target_clamp"] == {}
    assert manager.locked_target["attempt_id"] == "attempt-visual-guide"

    monkeypatch.setattr(
        realtime_runner,
        "roi_opencv_depth_bottle_tracking",
        lambda *_args, **_kwargs: (
            None,
            {
                "tracking_source": "shifted_locked_geometry",
                "identity_measurement_current": False,
                "motion_estimate_current": False,
                "roi_candidate_reject_reason": ["identity_not_current"],
                "roi_bottle_tracking_timings_ms": {},
            },
        ),
    )
    missing = realtime_runner.fast_track_frame(
        rgb,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        3,
        SimpleNamespace(),
        manager,
        latest_report=current,
    )
    missing_tracking = manager.current_tracked_target

    assert missing_tracking["operator_locked_fraction_visual_guide_current"] is False
    assert missing_tracking["display_target_clamp"] == {}
    assert missing_tracking["display_bottle_bbox"] is None
    assert missing["ready_to_close"] is False
    assert manager.locked_target["attempt_id"] == "attempt-visual-guide"


def test_partial_motion_estimate_keeps_explicit_recent_identity_age(monkeypatch):
    manager = _close_tracking_manager()
    manager.last_good_bottle_frame_id = 10
    candidate = _oriented_rectangle_geometry(
        90.0,
        center=(160.0, 100.0),
        width=70.0,
        height=180.0,
        depth_mm=180.0,
    )
    candidate["partial_extent"] = True
    monkeypatch.setattr(
        realtime_runner,
        "detect_roi_bottle_candidate",
        lambda *_args, **_kwargs: (candidate, [0, 0, 320, 240], None, {}, {}),
    )
    monkeypatch.setattr(
        realtime_runner,
        "validate_roi_bottle_candidate",
        lambda *_args, **_kwargs: (
            False,
            {
                "roi_candidate_reject_reason": ["center_shift_too_large"],
                "roi_candidate_hand_occlusion": False,
            },
        ),
    )
    monkeypatch.setattr(
        realtime_runner,
        "partial_hand_occlusion_rigid_motion_estimate",
        lambda *_args, **_kwargs: (
            copy.deepcopy(candidate),
            {
                "motion_estimate_step_translation_px": [0.0, -8.0],
                "motion_estimate_step_magnitude_px": 8.0,
            },
        ),
    )

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 180, dtype=np.uint16),
        manager,
        frame_id=12,
        current_patch_depth_mm=180.0,
    )

    assert geometry is not None
    assert telemetry["motion_estimate_current"] is True
    assert telemetry["identity_measurement_current"] is False
    assert telemetry["identity_measurement_age_frames"] == 2
    assert manager.last_good_bottle_frame_id == 10


def test_current_bottle_identity_keeps_lock_when_contact_depth_temporarily_drops():
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {"valid": True, "bottle_target_id": "bottle_01"}
    manager.ready_candidate_frames = 4
    manager.stable_ready_frames = 4

    for _ in range(manager.max_bad_tracking_frames + 2):
        tracking = {
            "tracking_ok": False,
            "locked_patch_visible": False,
            "patch_depth_confidence": 0.0,
            "tracking_confidence": 12.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
        }
        separate_identity_from_contact_tracking(tracking, bottle_geometry_tracking_ok=True)
        manager.apply_tracking_counters(tracking)

    assert manager.status() == "LOCKED"
    assert manager.bad_tracking_frames == 0
    assert manager.ready_candidate_frames == 0
    assert manager.stable_ready_frames == 0
    assert tracking["identity_tracking_ok"] is True
    assert tracking["contact_patch_tracking_ok"] is False
    assert tracking["tracking_ok"] is False
    assert target_lock_report(manager, tracking)["target_status"] == "TRACKING"
    assert tracking_good_from_values(tracking, target_status=manager.status()) is False
    blockers = ready_blockers_from_values(
        {},
        tracking,
        {"force_closure": True},
        100.0,
        [],
        target_status=manager.status(),
    )
    assert "locked_patch_not_visible" in blockers
    assert "contact_patch_tracking_low" in blockers
    assert "depth_confidence_low" in blockers
    approach = compute_approach_guidance(
        None,
        np.zeros((240, 320), dtype=np.uint16),
        {},
        {},
        tracking,
        (240, 320, 3),
    )
    assert approach["approach_action"] == "HOLD_STEADY_READY_VERIFY"


def test_contact_tracking_flag_alone_remains_a_fail_closed_ready_gate():
    tracking = {
        "tracking_ok": False,
        "locked_patch_visible": True,
        "patch_depth_confidence": 100.0,
        "tracking_confidence": 100.0,
        "patch_width_valid": True,
        "patch_on_bottle_body": True,
    }
    separate_identity_from_contact_tracking(tracking, bottle_geometry_tracking_ok=True)

    assert tracking["identity_tracking_ok"] is True
    assert tracking["contact_patch_tracking_ok"] is False
    assert tracking_good_from_values(tracking, target_status="LOCKED") is False
    blockers = ready_blockers_from_values(
        {},
        tracking,
        {"force_closure": True},
        100.0,
        [],
        target_status="LOCKED",
    )
    assert blockers == ["contact_patch_tracking_low"]


def test_recent_identity_bridge_is_labeled_estimated_and_blocks_ready():
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {"valid": True, "bottle_target_id": "bottle_01"}
    tracking = {
        "tracking_ok": True,
        "locked_patch_visible": True,
        "patch_depth_confidence": 100.0,
        "tracking_confidence": 100.0,
        "patch_width_valid": True,
        "patch_on_bottle_body": True,
        "identity_measurement_current": False,
        "identity_measurement_age_frames": 1,
    }

    separate_identity_from_contact_tracking(tracking, bottle_geometry_tracking_ok=True)
    manager.apply_tracking_counters(tracking)

    assert tracking["identity_tracking_ok"] is True
    assert tracking["identity_claim_status"] == (
        "RECENT_RGBD_IDENTITY_BRIDGE_ESTIMATED_NOT_CURRENT"
    )
    assert tracking_good_from_values(tracking, target_status="LOCKED") is False
    blockers = ready_blockers_from_values(
        {}, tracking, {"force_closure": True}, 100.0, [], target_status="LOCKED"
    )
    assert "identity_measurement_not_current" in blockers


def test_ready_stability_requires_consecutive_good_frames():
    manager = TargetLockManager(enabled=True)

    assert manager.update_ready_stability(True) is False
    assert manager.ready_candidate_frames == 1
    assert manager.update_ready_stability(False) is False
    assert manager.ready_candidate_frames == 0
    assert manager.stable_ready_frames == 0
    assert manager.update_ready_stability(True) is False
    assert manager.ready_candidate_frames == 1
    assert manager.update_ready_stability(True) is False
    assert manager.update_ready_stability(True) is True


def test_contact_depth_without_explicit_hand_evidence_expires_lock_and_preserves_memory(monkeypatch):
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", False)
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.37,
        "attempt_id": "attempt-original",
    }
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.37,
        "center_px_at_lock": [160.0, 120.0],
        "depth_mm_at_lock": 100.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }

    def good_contact_patch(**identity):
        return {
            "tracking_ok": True,
            "locked_patch_visible": True,
            "patch_depth_confidence": 100.0,
            "tracking_confidence": 90.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
            **identity,
        }

    for offset in range(manager.max_bad_tracking_frames):
        tracking = good_contact_patch(
            identity_measurement_current=False,
            identity_measurement_age_frames=(
                realtime_runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + 1 + offset
            ),
        )
        separate_identity_from_contact_tracking(tracking, bottle_geometry_tracking_ok=False)
        assert tracking["contact_patch_tracking_ok"] is True
        assert tracking["identity_tracking_ok"] is False
        assert tracking.get("roi_candidate_hand_occlusion") is not True
        manager.apply_tracking_counters(tracking)

    assert manager.status() == "LOST"
    assert manager.locked_target is None
    assert manager.has_lost_lock is True
    assert manager.last_lost_target["bottle_target_id"] == "bottle_01"
    assert manager.last_lost_target["attempt_id"] == "attempt-original"
    assert manager.last_lost_target["locked_body_fraction"] == 0.37
    assert manager.grasp_memory["status"] == "PRESERVED_AFTER_TRACK_LOSS"
    assert manager.grasp_memory["locked_body_fraction"] == 0.37
    assert tracking["identity_occlusion_hold_active"] is False
    assert tracking["identity_occlusion_evidence_current"] is False
    assert tracking["target_identity_preserved"] is False
    assert tracking["identity_claim_status"] == "NO_CURRENT_TARGET_IDENTITY"
    assert target_lock_report(manager, tracking)["target_status"] == "LOST"
    assert tracking_good_from_values(tracking, target_status=manager.status()) is False
    blockers = ready_blockers_from_values(
        {},
        tracking,
        {"force_closure": True},
        100.0,
        [],
        target_status=manager.status(),
    )
    assert "target_lost" in blockers
    assert "bottle_identity_uncertain" in blockers
    assert "original_target_occluded_hold" not in blockers


def test_recent_restart_restores_only_grasp_metadata_not_a_valid_lock(monkeypatch):
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", False)
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
    manager = TargetLockManager(enabled=True)

    restored = manager.restore_recent_grasp_memory(report, now_epoch=1050.0)

    assert restored is True
    assert manager.status() == "LOST"
    assert manager.locked_target is None
    assert manager.grasp_memory["locked_body_fraction"] == 0.52
    assert manager.grasp_memory["status"] == "RESTORED_RECENT_UNVERIFIED_AFTER_BACKEND_RESTART"
    assert manager.close_up_lock_debug["current_rgbd_identity_required"] is True


def test_stale_restart_grasp_memory_is_not_restored():
    report = {
        "grasp_memory": {
            "locked_body_fraction": 0.52,
            "axis_vector_at_lock": [0.0, 1.0],
            "grip_width_mm_at_lock": 56.0,
            "last_loss_timestamp": "1970-01-01T00:00:00+00:00",
        }
    }
    manager = TargetLockManager(enabled=True)

    assert manager.restore_recent_grasp_memory(report, now_epoch=1000.0) is False
    assert manager.status() == "SEARCHING"
    assert manager.grasp_memory is None


def test_lost_lock_memory_cannot_override_current_occlusion_or_selection_gates(monkeypatch):
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[:, 120:200] = 1
    pose = {
        "detected": True,
        "target_id": "close_candidate",
        "bbox_xywh": [120, 0, 80, 240],
        "center_px": [159.5, 119.5],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 90.0,
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
        "bottle_detection_status": "CANDIDATE",
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_depth_fused",
        "bottle_color_fraction": 0.75,
        "cleaned_mask": mask,
        "body_mask": mask,
        "candidate_evidence_scores": {"border_touch": True},
        "depth_geometry_evidence": {"depth_foreground_separated": True},
        "approximate_bbox_dimensions_mm": {"minor_extent_mm": 56.0},
    }
    blocked_state = ObjectState(
        orientation="STANDING",
        tilt_angle_deg=0.0,
        standing_confidence=1.0,
        occluded=True,
        available_for_pickup=False,
        reasons=["clear_space_above_top_band_blocked"],
        evidence={"availability": {"mask_completeness": 1.0, "depth_valid_ratio": 1.0}},
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_with_yolo_trigger",
        lambda *_args, **_kwargs: (
            [pose],
            None,
            {"yolo_enabled": False, "yolo_available": False, "yolo_detected": False},
        ),
    )
    monkeypatch.setattr(realtime_runner, "estimate_object_state", lambda *_args, **_kwargs: blocked_state)
    monkeypatch.setattr(realtime_runner, "write_latest_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)
    manager = TargetLockManager(enabled=True)
    manager.has_lost_lock = True
    manager.grasp_memory = {
        "status": "PRESERVED_AFTER_TRACK_LOSS",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [160.0, 120.0],
        "axis_vector_at_lock": [0.0, 1.0],
        "depth_mm_at_lock": 465.0,
        "grip_width_mm_at_lock": 56.0,
    }

    report = realtime_runner.evaluate_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 90, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        42,
        SimpleNamespace(),
        lock_manager=manager,
    )

    assert report["state"] == "TARGET_LOST"
    assert report["decision"] == "DO_NOT_CLOSE"
    assert report["selected_bottle_target_id"] is None
    assert "original_target_identity_not_reacquired" in report["failure_reasons"]
    assert manager.status() == "LOST"
    assert manager.locked_target is None
    assert manager.grasp_memory["locked_body_fraction"] == 0.52


def test_generated_target_id_keeps_selection_object_state_as_report_authority(monkeypatch):
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[50:190, 130:190] = 1
    pose = {
        "detected": True,
        "bbox_xywh": [130, 50, 60, 140],
        "center_px": [159.5, 119.5],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 500.0,
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
        "bottle_detection_status": "CANDIDATE",
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_depth_fused",
        "bottle_color_fraction": 0.75,
        "cleaned_mask": mask,
        "candidate_evidence_scores": {"border_touch": False},
        "depth_geometry_evidence": {"depth_foreground_separated": True},
    }
    blocked_state = ObjectState(
        orientation="STANDING",
        tilt_angle_deg=0.0,
        standing_confidence=1.0,
        occluded=False,
        available_for_pickup=False,
        reasons=["insufficient_clear_space_depth"],
        evidence={"availability": {"mask_completeness": 1.0, "depth_valid_ratio": 1.0}},
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_with_yolo_trigger",
        lambda *_args, **_kwargs: (
            [pose],
            None,
            {"yolo_enabled": False, "yolo_available": False, "yolo_detected": False},
        ),
    )
    monkeypatch.setattr(realtime_runner, "estimate_object_state", lambda *_args, **_kwargs: blocked_state)
    monkeypatch.setattr(realtime_runner, "write_latest_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)

    report = realtime_runner.evaluate_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 500, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        43,
        SimpleNamespace(),
        lock_manager=TargetLockManager(enabled=False),
    )

    assert pose["target_id"] == "bottle_01"
    assert report["object_state"]["available_for_pickup"] is False
    assert report["object_state"]["reasons"] == ["insufficient_clear_space_depth"]
    assert report["object_state_source"] == "CURRENT_FRAME_TARGET_SELECTION_RGBD_GATE"


def test_prelock_nonblue_hand_like_candidate_never_advances_lock_stability(monkeypatch):
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:200, 130:190] = 1
    pose = {
        "detected": True,
        "target_id": "hand_fixture",
        "bbox_xywh": [130, 40, 60, 160],
        "center_px": [159.5, 119.5],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 325.0,
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
        "bottle_detection_status": "CANDIDATE",
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_depth_fused",
        "bottle_color_fraction": 0.0,
        "cleaned_mask": mask,
        "body_mask": mask,
    }
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_with_yolo_trigger",
        lambda *_args, **_kwargs: ([pose], None, {"yolo_enabled": False}),
    )
    monkeypatch.setattr(realtime_runner, "write_latest_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)
    manager = TargetLockManager(enabled=True)

    for frame_id in range(1, 5):
        report = realtime_runner.evaluate_frame(
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.where(mask > 0, 325, 900).astype(np.uint16),
            {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
            frame_id,
            SimpleNamespace(),
            lock_manager=manager,
        )
        assert report["state"] == "FIND_FULL_BOTTLE"
        assert report["decision"] == "SEARCH"
        assert manager.stable_full_bottle_frames == 0
        assert manager.locked_target is None

    eligible, reasons = realtime_runner.demo_bottle_identity_eligibility(pose)
    assert eligible is False
    assert "demo_identity_known_blue_support_too_low" in reasons


def test_output_object_state_diagnostic_cannot_overwrite_decision_state(monkeypatch):
    decision_state = {
        "orientation": "STANDING",
        "occluded": False,
        "available_for_pickup": False,
        "reasons": ["insufficient_clear_space_depth"],
    }
    diagnostic_state = ObjectState(
        orientation="STANDING",
        tilt_angle_deg=0.0,
        standing_confidence=1.0,
        occluded=False,
        available_for_pickup=True,
        reasons=["graspable_region_clear_in_current_rgbd_estimate"],
        evidence={"availability": {"clear_space_valid_ratio": 1.0}},
    )
    report = {"object_state": copy.deepcopy(decision_state)}
    monkeypatch.setattr(
        realtime_runner,
        "estimate_object_state",
        lambda *_args, **_kwargs: diagnostic_state,
    )

    result = realtime_runner.attach_output_object_state(
        report,
        pose=None,
        quality_depth=np.full((20, 20), 500, dtype=np.uint16),
        pointcloud_status={},
    )

    assert result == decision_state
    assert report["object_state"] == decision_state
    assert report["object_state_output_diagnostic"]["available_for_pickup"] is True
    assert report["object_state_decision_output_agreement"] is False
    assert report["object_state_output_diagnostic_role"].startswith("NOT_USED_FOR_TARGET_SELECTION")


def test_held_geometry_object_state_is_not_reported_as_current_or_available():
    held_contour = np.array(
        [[[70, 45]], [[109, 45]], [[109, 144]], [[70, 144]]],
        dtype=np.int32,
    )
    pose = {
        "frame_id": None,
        "geometry_frame_id": None,
        "mask_frame_id": None,
        "source_geometry_frame_id": 10,
        "source_mask_frame_id": 10,
        "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
        "mask_measurement_status": "NOT_AVAILABLE_AFTER_GEOMETRIC_SHIFT_USE_SHIFTED_CONTOUR",
        "contour": held_contour,
        "axis_vector": [0.0, 1.0],
        "target_id": "bottle_01",
    }
    depth = np.zeros((160, 180), dtype=np.uint16)
    depth[45:145, 70:110] = 500
    depth[25:45, 70:110] = 800
    prior_selection_state = {
        "orientation": "STANDING",
        "occluded": False,
        "available_for_pickup": True,
        "reasons": ["graspable_region_clear_in_current_rgbd_estimate"],
    }
    report = {
        "frame_id": 11,
        "fast_tracking_used": True,
        "identity_measurement_current": False,
        "object_state": copy.deepcopy(prior_selection_state),
    }

    state = realtime_runner.attach_output_object_state(report, pose, depth, {})

    assert state["orientation"] == "UNKNOWN"
    assert state["available_for_pickup"] is False
    assert state["pose_valid_for_grasp"] is False
    assert state["reasons"] == ["geometry_measurement_not_current"]
    assert state["evidence"]["availability"]["status"] == (
        "NOT_TESTED_GEOMETRY_MEASUREMENT_NOT_CURRENT"
    )
    assert report["object_state_source"] == (
        "NOT_TESTED_GEOMETRY_MEASUREMENT_NOT_CURRENT"
    )
    assert report["object_state_selection_evidence"] == prior_selection_state
    assert report["object_state_selection_evidence_role"].startswith("HISTORICAL")


def test_partial_pose_cannot_be_force_locked_before_or_after_target_loss(monkeypatch):
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[5:235, 120:200] = 1
    pose = {
        "detected": True,
        "target_id": "selected_close_candidate",
        "bbox_xywh": [120, 5, 80, 230],
        "center_px": [159.5, 119.5],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 90.0,
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
        "bottle_detection_status": "CANDIDATE",
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_depth_fused",
        "bottle_color_fraction": 0.75,
        "cleaned_mask": mask,
        "body_mask": mask,
    }
    available_state = ObjectState(
        orientation="STANDING",
        tilt_angle_deg=0.0,
        standing_confidence=1.0,
        occluded=False,
        available_for_pickup=True,
        reasons=["graspable_region_clear_in_current_rgbd_estimate"],
        evidence={"availability": {"mask_completeness": 1.0, "depth_valid_ratio": 1.0}},
    )
    selection = {
        "selected_target_id": pose["target_id"],
        "selection_status": "TARGET_SELECTED",
        "candidate_count": 1,
        "eligible_count": 1,
        "candidates": [
            {
                "target_id": pose["target_id"],
                "eligible": True,
                "selected_grasp_body_fraction": 0.5,
                "reasons": ["side_approach_supported_by_task_policy"],
            }
        ],
        "grasp_feasibility_score": 90.0,
        "grasp_feasibility_label": "OPTIMAL_GRASP",
        "grasp_feasibility_reason": "selected",
        "suggested_operator_action": "continue",
        "suggested_robot_action": "PLAN_ONLY_APPROACH_SELECTED_TARGET",
        "limitations": ["NOT_PHYSICALLY_VALIDATED"],
        "physically_validated": False,
    }
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_with_yolo_trigger",
        lambda *_args, **_kwargs: (
            [pose],
            None,
            {"yolo_enabled": False, "yolo_available": False, "yolo_detected": False},
        ),
    )
    monkeypatch.setattr(realtime_runner, "estimate_object_state", lambda *_args, **_kwargs: available_state)
    monkeypatch.setattr(
        realtime_runner,
        "select_bottle_target",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: selection),
    )
    monkeypatch.setattr(
        realtime_runner,
        "check_full_bottle_visibility",
        lambda *_args, **_kwargs: {
            "full_bottle_visible": False,
            "visibility_score": 70.0,
            "missing_region_reason": ["Top of bottle is near/cut by frame."],
            "recommended_view_action": "MOVE_BACK",
        },
    )
    plan_calls = []
    monkeypatch.setattr(
        realtime_runner,
        "plan_pregrasp",
        lambda *_args, **_kwargs: (plan_calls.append(True), None),
    )
    monkeypatch.setattr(realtime_runner, "write_latest_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)
    cases = (
        (True, ["lost_lock_requires_full_visible_reacquisition"], "LOST", "FIND_FULL_BOTTLE"),
        (False, ["initial_full_visible_acquisition_required"], "SEARCHING", "FIND_FULL_BOTTLE"),
    )
    for has_lost_lock, expected_close_up_blockers, expected_status, expected_state in cases:
        manager = TargetLockManager(enabled=True)
        manager.has_lost_lock = has_lost_lock
        if has_lost_lock:
            manager.grasp_memory = {
                "status": "PRESERVED_AFTER_TRACK_LOSS",
                "bottle_target_id": pose["target_id"],
                "locked_body_fraction": 0.52,
                "center_px_at_lock": [160.0, 120.0],
                "axis_vector_at_lock": [0.0, 1.0],
                "depth_mm_at_lock": 90.0,
                "grip_width_mm_at_lock": 56.0,
            }
            manager.last_lost_target = {
                "valid": False,
                "bottle_target_id": pose["target_id"],
                "attempt_id": "attempt-original",
                "locked_bottle_geometry": {
                    "bbox_xywh": list(pose["bbox_xywh"]),
                    "center_px": list(pose["center_px"]),
                    "axis_vector": list(pose["axis_vector"]),
                    "depth_median_mm": pose["depth_median_mm"],
                },
            }

        observation_count = manager.relock_required_frames + 1 if has_lost_lock else 1
        for frame_id in range(43, 43 + observation_count):
            report = realtime_runner.evaluate_frame(
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.where(mask > 0, 90, 600).astype(np.uint16),
                {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
                frame_id,
                SimpleNamespace(),
                lock_manager=manager,
                force_lock=True,
            )

            assert report["state"] == expected_state
            assert report["ready_to_close"] is False
            assert report["target_lock"]["target_status"] == expected_status
            assert report["target_lock_progress"]["close_up_lock_allowed"] is False
            assert report["target_lock_progress"]["close_up_lock_blockers"] == expected_close_up_blockers
            assert report["full_bottle_visible"] is False
            assert "Full bottle is not visible." in report["failure_reasons"]
            assert manager.stable_full_bottle_frames == 0
            assert manager.ready_to_lock() is False
            assert manager.status() == expected_status
            assert manager.locked_target is None
    assert plan_calls == []


def test_rejected_bottle_identity_still_expires_lock_after_existing_limit(monkeypatch):
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", False)
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {"valid": True, "bottle_target_id": "bottle_01"}

    for _ in range(manager.max_bad_tracking_frames):
        tracking = {
            "tracking_ok": False,
            "locked_patch_visible": False,
            "patch_depth_confidence": 0.0,
            "tracking_confidence": 0.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
            "bad_tracking_increment_reason": "roi_identity_rejected",
        }
        separate_identity_from_contact_tracking(tracking, bottle_geometry_tracking_ok=False)
        manager.apply_tracking_counters(tracking)

    assert manager.status() == "LOST"
    assert tracking["tracking_ok"] is False
    assert "roi_identity_rejected" in tracking["reason"]


def test_partial_extent_uses_tracked_grasp_band_not_visible_contour_fraction():
    manager = _close_tracking_manager()
    rgb, depth = _partial_close_scene(90)
    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=90.0,
    )
    assert telemetry["tracking_source"] == "roi_opencv_depth"
    base_clamp = {
        "clamp_center_px": [160.0, 150.0],
        "grasp_line": {"direction_px": [1.0, 0.0]},
        "clamp_patch_height_px": 60.0,
        "clamp_patch_width_px": 18.0,
    }
    locked = {
        "locked_body_fraction": 0.25,
        "bottle_contact_model": {"patch_height_mm": 50.0, "patch_width_mm": 15.0},
    }

    band, trace = reanchor_target_clamp_to_current_edges(
        base_clamp,
        geometry,
        locked,
        depth,
        rgb.shape,
        {"fx": 500.0, "fy": 500.0},
    )

    assert trace["band_source"] == "current_side_edges_at_tracked_locked_band"
    assert abs(band["clamp_center_px"][1] - 150.0) < 2.0
    assert band["contact_patch_scale_source"] == "current_rgbd_depth_and_intrinsics"


def test_invalid_close_depth_cannot_replace_last_good_locked_geometry():
    manager = _close_tracking_manager()
    original = dict(manager.last_good_bottle_geometry)
    rgb, depth = _partial_close_scene(90, object_depth_value=0)

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=90.0,
    )

    assert telemetry["tracking_source"] != "roi_opencv_depth"
    assert telemetry["accepted_current_bottle_bbox"] == original["bbox_xyxy"]
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert geometry["detected"] is False
    assert manager.last_good_bottle_geometry == original
    assert manager.last_good_bottle_depth_mm == 200.0


def test_identity_age_remains_explicit_after_last_good_hold_expires():
    manager = _close_tracking_manager()
    manager.roi_bottle_fail_frames = realtime_runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.zeros((240, 320), dtype=np.uint16)

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=9,
        current_patch_depth_mm=90.0,
    )

    assert telemetry["tracking_source"] == "lost"
    assert telemetry["identity_measurement_current"] is False
    assert telemetry["identity_measurement_age_frames"] == 8
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert geometry["detected"] is False


def test_rejected_geometry_cannot_rescale_the_locally_tracked_grasp_band():
    target_clamp = {
        "clamp_center_px": [160.0, 120.0],
        "left_contact_px": [130.0, 120.0],
        "right_contact_px": [190.0, 120.0],
        "grasp_line": {
            "direction_px": [1.0, 0.0],
            "left_endpoint_px": [130.0, 120.0],
            "right_endpoint_px": [190.0, 120.0],
        },
        "clamp_patch_height_px": 36.0,
        "clamp_patch_width_px": 12.0,
        "grip_width_px": 60.0,
        "bottle_y_axis_px": [0.0, 1.0],
        "bottle_x_axis_px": [1.0, 0.0],
    }
    held_geometry = {
        "contour": [[80, 0], [240, 0], [240, 239], [80, 239]],
        "depth_median_mm": 90.0,
    }
    depth = np.full((240, 320), 90, dtype=np.uint16)

    preserved, trace = reanchor_target_clamp_only_from_current_geometry(
        target_clamp,
        held_geometry,
        {
            "locked_body_fraction": 0.37,
            "bottle_contact_model": {"patch_height_mm": 50.0, "patch_width_mm": 15.0},
        },
        depth,
        depth.shape,
        {"fx": 500.0, "fy": 500.0},
        geometry_measurement_current=False,
    )

    assert preserved["clamp_center_px"] == target_clamp["clamp_center_px"]
    assert preserved["clamp_patch_height_px"] == 36.0
    assert preserved["clamp_patch_width_px"] == 12.0
    assert preserved["grip_width_px"] == 60.0
    assert trace["locked_body_fraction"] == 0.37
    assert trace["reanchor_applied"] is False
    assert trace["contact_patch_scale_source"] == "preserved_last_verified_rgbd_scale"


def test_stale_depth_neighborhood_freezes_last_verified_clamp_without_rescaling():
    last_verified = {
        "clamp_center_px": [140.0, 120.0],
        "left_contact_px": [110.0, 120.0],
        "right_contact_px": [170.0, 120.0],
        "grasp_line": {
            "direction_px": [1.0, 0.0],
            "left_endpoint_px": [110.0, 120.0],
            "right_endpoint_px": [170.0, 120.0],
            "center_px": [140.0, 120.0],
        },
        "left_clamp_patch": {
            "center_px": [110.0, 120.0],
            "polygon": [[105.0, 105.0], [115.0, 105.0], [115.0, 135.0], [105.0, 135.0]],
        },
        "right_clamp_patch": {
            "center_px": [170.0, 120.0],
            "polygon": [[165.0, 105.0], [175.0, 105.0], [175.0, 135.0], [165.0, 135.0]],
        },
        "clamp_band_polygon": [[105.0, 100.0], [175.0, 100.0], [175.0, 140.0], [105.0, 140.0]],
        "clamp_patch_height_px": 30.0,
        "clamp_patch_width_px": 10.0,
        "grip_width_px": 60.0,
        "bottle_y_axis_px": [0.0, 1.0],
        "bottle_x_axis_px": [1.0, 0.0],
    }
    previous_tracking = {
        "target_clamp": last_verified,
        "last_verified_rgbd_target_clamp": last_verified,
        "last_verified_rgbd_target_clamp_frame_id": 1,
        "current_depth_mm": 100.0,
    }
    locked_target = {
        "locked_body_fraction": 0.37,
        "depth_mm_at_lock": 100.0,
        "movement_deadband_state": {},
    }
    depth = np.full((240, 320), 900, dtype=np.uint16)
    # A same-depth bottle neighborhood moved +12 px right and +8 px down.
    depth[110:147, 115:190] = 100

    centers = []
    per_frame_deltas = []
    tracking = previous_tracking
    for frame_id in range(2, 5):
        _pose, tracking = local_locked_patch_tracking(
            depth,
            locked_target,
            tracking,
            depth.shape,
            frame_id=frame_id,
        )
        tracking["identity_measurement_current"] = False
        tracking["identity_measurement_age_frames"] = (
            realtime_runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + frame_id
        )
        separate_identity_from_contact_tracking(
            tracking,
            bottle_geometry_tracking_ok=False,
        )
        assert tracking["identity_tracking_ok"] is False
        centers.append(np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float))
        per_frame_deltas.append(
            np.asarray(tracking["tracking_delta_this_frame_px"], dtype=float)
        )

    translated = tracking["target_clamp"]
    movement = centers[-1] - np.asarray(last_verified["clamp_center_px"], dtype=float)
    assert tracking["clamp_base_source"] == "last_verified_rgbd_target_clamp"
    assert tracking["local_motion_application_status"] == "FROZEN_STALE_IDENTITY"
    assert tracking["local_motion_filter_mode"] == "freeze_at_last_verified_rgbd_clamp"
    assert all(np.linalg.norm(center - last_verified["clamp_center_px"]) == 0.0 for center in centers)
    assert all(np.linalg.norm(delta) == 0.0 for delta in per_frame_deltas)
    assert np.linalg.norm(movement) == 0.0
    assert translated["grip_width_px"] == last_verified["grip_width_px"]
    assert translated["clamp_patch_height_px"] == last_verified["clamp_patch_height_px"]
    assert translated["clamp_patch_width_px"] == last_verified["clamp_patch_width_px"]
    for side in ("left_clamp_patch", "right_clamp_patch"):
        before = np.asarray(last_verified[side]["polygon"], dtype=float)
        after = np.asarray(translated[side]["polygon"], dtype=float)
        assert np.allclose(after - before, movement, atol=1e-6)

    held_geometry = {
        "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
        "bbox_xyxy": [95.0, 40.0, 209.0, 220.0],
        "depth_median_mm": 100.0,
    }
    preserved, trace = reanchor_target_clamp_only_from_current_geometry(
        translated,
        held_geometry,
        locked_target,
        depth,
        depth.shape,
        {"fx": 500.0, "fy": 500.0},
        geometry_measurement_current=False,
        contact_patch_tracking_current=True,
    )
    assert preserved["clamp_center_px"] == translated["clamp_center_px"]
    assert preserved["grip_width_px"] == translated["grip_width_px"]
    assert preserved["clamp_patch_height_px"] == translated["clamp_patch_height_px"]
    assert preserved["clamp_patch_width_px"] == translated["clamp_patch_width_px"]
    for side in ("left_clamp_patch", "right_clamp_patch"):
        assert preserved[side]["polygon"] == translated[side]["polygon"]
    assert trace["reanchor_applied"] is False
    assert trace["contact_patch_scale_source"] == "preserved_last_verified_rgbd_scale"


def test_local_translation_search_seed_is_bounded_and_cannot_jump_to_distant_object():
    seed = {
        "detected": True,
        "bbox_xyxy": [100.0, 60.0, 140.0, 180.0],
        "bbox_xywh": [100.0, 60.0, 40.0, 120.0],
        "center_px": [120.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 100.0,
        "depth_confidence": 1.0,
    }

    def manager():
        return SimpleNamespace(
            last_good_bottle_geometry=dict(seed),
            last_good_bottle_bbox=list(seed["bbox_xyxy"]),
            last_good_bottle_depth_mm=100.0,
            last_good_bottle_frame_id=1,
            last_good_raw_bottle_velocity_px=[0.0, 0.0],
            last_full_roi_contour_frame_id=None,
            roi_bottle_fail_frames=0,
            locked_target={"valid": True, "bottle_target_id": "bottle_01"},
            grasp_memory={"status": "LOCKED_BOTTLE_RELATIVE_GRASP", "locked_body_fraction": 0.5},
        )

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    rgb[40:200, 170:210] = [45, 205, 65]
    depth[40:200, 170:210] = 100

    accepted_manager = manager()
    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        accepted_manager,
        frame_id=2,
        current_patch_depth_mm=100.0,
        estimated_translation_px=[70.0, 0.0],
    )

    assert telemetry["tracking_source"] == "shifted_locked_geometry"
    assert telemetry["roi_fast_scale_mode"] == "reacquire"
    assert telemetry["roi_fast_scale_requested"] == pytest.approx(0.75)
    assert "local_translation" in telemetry["roi_fast_scale_reacquire_reasons"]
    assert telemetry["identity_measurement_current"] is False
    assert telemetry["roi_estimated_translation_magnitude_px"] <= (
        realtime_runner.LOCAL_DEPTH_MOTION_RECENT_IDENTITY_MAX_OFFSET_PX + 1e-6
    )
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    roi = telemetry["post_lock_roi_xyxy"]
    roi_center_x = (float(roi[0]) + float(roi[2])) / 2.0
    seed_center_x = (seed["bbox_xyxy"][0] + seed["bbox_xyxy"][2]) / 2.0
    assert 0.0 < roi_center_x - seed_center_x <= (
        realtime_runner.LOCAL_DEPTH_MOTION_RECENT_IDENTITY_MAX_OFFSET_PX + 1e-6
    )
    assert accepted_manager.last_good_bottle_geometry == seed

    # Translation changes the search seed, not the identity decision: a moved
    # lookalike at the wrong depth must still be held/rejected.
    wrong_depth = depth.copy()
    wrong_depth[40:200, 170:210] = 300
    rejected_manager = manager()
    held, rejected = roi_opencv_depth_bottle_tracking(
        rgb,
        wrong_depth,
        rejected_manager,
        frame_id=2,
        current_patch_depth_mm=100.0,
        estimated_translation_px=[70.0, 0.0],
    )
    assert rejected["identity_measurement_current"] is False
    # The wrong-depth RGB lookalike may also fill an internal fast-search edge;
    # either safety gate must reject it before it becomes a current identity.
    assert any(
        reason in rejected["roi_candidate_reject_reason"]
        for reason in (
            "candidate_patch_depth_mismatch",
            "fast_roi_component_clipped_by_internal_search_boundary",
            "fast_roi_mask_too_small",
            "fast_roi_known_blue_rgbd_support_too_small",
        )
    ), rejected["roi_candidate_reject_reason"]
    assert held["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert rejected_manager.last_good_bottle_geometry == seed


def test_segmentation_confidence_accepts_json_serialized_contour():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:52, 20:44] = 1
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        # cv2.findContours(...).tolist() preserves the Nx1x2 nesting.
        "contour": [[[20, 12]], [[43, 12]], [[43, 51]], [[20, 51]]],
        "segmentation_source": "held_roi_geometry",
    }

    summary = segmentation_confidence_summary(pose)

    assert summary["contour_quality_score"] > 0.0
    assert np.isfinite(summary["contour_quality_score"])
    assert summary["segmentation_source"] == "held_roi_geometry"


def test_segmentation_confidence_malformed_or_empty_contour_fails_safe():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:52, 20:44] = 1

    for contour in ([], ["not-a-point"], [[20, 12]], [20, 12, 43]):
        summary = segmentation_confidence_summary(
            {
                "cleaned_mask": mask,
                "body_mask": mask,
                "contour": contour,
            }
        )
        assert summary["contour_quality_score"] == 0.0
        assert summary["cleaned_mask_confidence"] == 100.0


def test_segmentation_summary_reports_cap_estimator_evidence_not_body_fraction():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:52, 20:44] = 1
    summary = segmentation_confidence_summary(
        {
            "cleaned_mask": mask,
            "body_mask": mask,
            "cap_like_end_region": {
                "status": "ESTIMATED_FROM_ALIGNED_RGBD_CAP_LIKE_END_GEOMETRY",
                "detected": True,
                "confidence": 0.82,
                "physically_validated": False,
            },
        }
    )

    assert summary["body_mask_confidence"] == 100.0
    assert summary["cap_body_separation_confidence"] == pytest.approx(82.0)
    assert summary["cap_like_end_region_status"] == "ESTIMATED_FROM_ALIGNED_RGBD_CAP_LIKE_END_GEOMETRY"
    assert summary["cap_like_end_region_physically_validated"] is False


def test_compact_lock_reports_authoritative_body_fraction_as_grasp_fraction():
    compact = compact_live_status(
        {
            "state": "APPROACH_TRACKING",
            "target_lock": {
                "target_status": "TRACKING",
                "lock_frame_id": 10,
                "locked_body_fraction": 0.37,
                "attempt_id": "attempt-10",
                "locked_target": {
                    "bottle_target_id": "bottle_01",
                    "locked_body_fraction": 0.37,
                },
                "current_tracked_target": {
                    "identity_tracking_ok": False,
                    "identity_occlusion_hold_active": True,
                    "identity_occlusion_hold_frames": 8,
                    "target_identity_preserved": True,
                },
            },
        }
    )

    assert compact["target_lock"]["locked_body_fraction"] == 0.37
    assert compact["target_lock"]["locked_grasp_fraction"] == 0.37
    assert compact["locked_bottle_target_id"] == "bottle_01"
    assert compact["lock_attempt_id"] == "attempt-10"
    assert compact["identity_occlusion_hold_active"] is True
    assert compact["identity_occlusion_hold_frames"] == 8
    assert compact["target_identity_preserved"] is True


def test_lock_trace_accepts_projected_fraction_equivalent_to_logical_band(monkeypatch):
    monkeypatch.setattr(
        realtime_runner,
        "LAST_TARGET_LOCK_TRACE_TIME",
        time.monotonic(),
    )
    report = {
        "frame_id": 21,
        "planned_contacts": {"body_fraction": 0.5996897925987771},
        "target_lock": {
            "target_status": "TRACKING",
            "locked_body_fraction": 0.6,
            "clamp_source": "locked_target",
            "grasp_band_state": "LOCKED_HELD",
            "band_drift_state": "LOCKED_HELD",
            "grasp_band_current": False,
            "grasp_band_held": True,
            "grasp_band_stale": False,
            "grasp_band_relock_required": False,
            "locked_projection_mode": "HELD_LAST_VALID_3D",
            "projection_valid": False,
            "locked_target": {
                "valid": True,
                "bottle_target_id": "bottle_02",
                "locked_body_fraction": 0.6,
                "locked_body_fraction_at_lock": 0.6,
            },
        },
    }

    trace = realtime_runner.trace_locked_target_frame(report)

    assert trace["locked_body_fraction"] == 0.6
    assert trace["current_body_fraction_used"] == pytest.approx(0.5996897925987771)
    assert trace["grasp_band_state"] == "LOCKED_HELD"
    assert trace["grasp_band_held"] is True
    assert trace["grasp_band_stale"] is False
    assert trace["grasp_band_relock_required"] is False
    assert trace["locked_projection_mode"] == "HELD_LAST_VALID_3D"
    assert trace["warnings"] == []
    assert report.get("warnings", []) == []


def test_lock_trace_ignores_missing_transient_selection_and_is_rate_limited(
    monkeypatch,
    tmp_path,
):
    trace_path = tmp_path / "target_lock_trace.jsonl"
    monkeypatch.setattr(realtime_runner, "TARGET_LOCK_TRACE_PATH", trace_path)
    monkeypatch.setattr(realtime_runner, "FAST_FINAL_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "TARGET_LOCK_TRACE_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(realtime_runner, "LAST_TARGET_LOCK_TRACE_TIME", 0.0)
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: 100.0)
    report = {
        "frame_id": 21,
        "grasp_target_3d_diagnostics": {"dense": list(range(10000))},
        "planned_contacts": {},
        "target_lock": {
            "target_status": "TRACKING",
            "locked_body_fraction": 0.74,
            "clamp_source": "held_last_valid_bottle_local_3d_projection",
            "locked_target": {
                "valid": True,
                "locked_body_fraction": 0.74,
                "locked_body_fraction_at_lock": 0.74,
            },
        },
    }

    for _ in range(100):
        realtime_runner.trace_locked_target_frame(report)

    rows = trace_path.read_text().splitlines()
    assert len(rows) == 1
    assert "BUG: locked target overwritten" not in report.get("warnings", [])
    assert len(rows[0]) < 10000


def test_prelock_progress_trace_records_transitions_but_not_every_frame(monkeypatch):
    emitted = []
    progress = {
        "lock_failure_reason": "no_valid_candidate",
        "lock_blocker": "no_valid_candidate",
        "lock_type": "INITIAL_LOCK",
        "close_up_lock_allowed": False,
        "stable_frames": 0,
        "required_stable_frames": 2,
        "stable_close_up_frames": 0,
        "required_close_up_stable_frames": 3,
        "close_up_lock_blockers": [],
        "detection_confidence": None,
        "bottle_center": None,
        "depth_median": None,
        "mask_area": None,
        "visibility_score": None,
        "full_bottle_visible": False,
    }
    monkeypatch.setattr(realtime_runner, "LAST_LOCK_PROGRESS_TRACE_TIME", 0.0)
    monkeypatch.setattr(realtime_runner, "LAST_LOCK_PROGRESS_TRACE_SIGNATURE", None)
    monkeypatch.setattr(realtime_runner, "TARGET_LOCK_TRACE_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(realtime_runner, "build_lock_progress", lambda *args, **kwargs: dict(progress))
    monkeypatch.setattr(realtime_runner, "target_lock_report", lambda _manager: {"target_status": "SEARCHING"})
    monkeypatch.setattr(realtime_runner, "append_target_lock_trace", emitted.append)
    report = {"frame_id": 1, "state": "FIND_FULL_BOTTLE"}

    for _ in range(100):
        realtime_runner.attach_lock_progress(report, object())
    assert len(emitted) == 1

    progress["stable_frames"] = 1
    progress["lock_failure_reason"] = "stability_pending"
    realtime_runner.attach_lock_progress(report, object())
    assert len(emitted) == 2
    assert emitted[-1]["stable_frames"] == 1


def test_lock_progress_preserves_final_3d_lock_rejection_after_stability_reset(monkeypatch):
    manager = TargetLockManager(enabled=True)
    pose = {
        "detection_confidence": 0.97,
        "depth_confidence": 0.99,
        "cleaned_mask": np.ones((10, 10), dtype=np.uint8),
    }
    monkeypatch.setattr(realtime_runner, "append_target_lock_trace", lambda *_args: None)
    report = {
        "frame_id": 8,
        "state": "FAIL_SAFE",
        "lock_attempt_failure_reason": "full_bottle_metric_length_below_minimum",
    }
    realtime_runner.attach_lock_progress(
        report,
        manager,
        pose=pose,
        visibility={"full_bottle_visible": True},
        plan={"contact_model_ready": True},
    )
    progress = report["target_lock_progress"]

    assert progress["lock_failure_reason"] == "full_bottle_metric_length_below_minimum"
    assert "full_bottle_metric_length_below_minimum" in progress["lock_failure_reasons"]


def test_lock_trace_invalid_projection_allows_missing_clamp_and_rotates(monkeypatch, tmp_path):
    trace_path = tmp_path / "target_lock_trace.jsonl"
    trace_path.write_text("x" * 200)
    monkeypatch.setattr(realtime_runner, "TARGET_LOCK_TRACE_PATH", trace_path)
    monkeypatch.setattr(realtime_runner, "TARGET_LOCK_TRACE_MAX_BYTES", 100)
    monkeypatch.setattr(realtime_runner, "FAST_FINAL_DEMO_MODE", False)
    report = {
        "frame_id": 22,
        "target_lock": {
            "target_status": "LOW_CONFIDENCE",
            "locked_body_fraction": 0.74,
            "locked_projection_mode": "INVALID_3D",
            "projection_valid": False,
            "locked_target": {
                "valid": True,
                "locked_body_fraction": 0.74,
                "locked_body_fraction_at_lock": 0.74,
            },
        },
    }

    trace = realtime_runner.trace_locked_target_frame(report)

    assert trace["warnings"] == []
    assert (tmp_path / "target_lock_trace.jsonl.previous").read_text() == "x" * 200
    rows = trace_path.read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["frame_id"] == 22


def test_timed_json_write_retries_transient_windows_bind_replace(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "latest_realtime_status.json"
    original_replace = realtime_runner.Path.replace
    attempts = {"count": 0}

    def transient_replace(path, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("destination temporarily open")
        return original_replace(path, target)

    monkeypatch.setattr(realtime_runner.Path, "replace", transient_replace)

    result = realtime_runner.write_json_timed(output_path, {"frame_id": 9})

    assert result["success"] is True
    assert result["replace_attempts"] == 3
    assert json.loads(output_path.read_text()) == {"frame_id": 9}


def test_timed_json_write_preserves_last_good_file_on_permanent_replace_failure(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "latest_realtime_status.json"
    output_path.write_text('{"frame_id": 8}')

    def permanent_failure(_path, _target):
        raise PermissionError("destination remains open")

    monkeypatch.setattr(realtime_runner.Path, "replace", permanent_failure)

    result = realtime_runner.write_json_timed(output_path, {"frame_id": 9})

    assert result["success"] is False
    assert result["replace_attempts"] == 4
    assert "PermissionError" in result["error"]
    assert json.loads(output_path.read_text()) == {"frame_id": 8}


def test_wrong_depth_lookalike_cannot_overwrite_locked_bottle():
    manager = _close_tracking_manager()
    original = dict(manager.last_good_bottle_geometry)
    rgb, depth = _partial_close_scene(90, object_depth_value=250)

    _geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=90.0,
    )

    assert telemetry["tracking_source"] != "roi_opencv_depth"
    assert telemetry["roi_candidate_reject_reason"]
    assert manager.last_good_bottle_geometry == original
    assert manager.grasp_memory["locked_body_fraction"] == 0.5


def test_uniform_close_depth_plane_cannot_replace_locked_bottle():
    manager = _close_tracking_manager()
    original = dict(manager.last_good_bottle_geometry)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 90, dtype=np.uint16)

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=90.0,
    )

    assert telemetry["tracking_source"] != "roi_opencv_depth"
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert geometry["detected"] is False
    assert manager.last_good_bottle_geometry == original


def test_locked_redetection_matches_locked_geometry_not_highest_detection_score():
    manager = SimpleNamespace(
        last_good_bottle_geometry={"center_px": [80.0, 120.0], "depth_median_mm": 500.0},
        locked_target={"bottle_target_id": "bottle_01"},
    )
    locked_bottle = {"target_id": "frame_bottle_01", "center_px": [84.0, 118.0], "depth_median_mm": 490.0, "detection_confidence": 0.62}
    distracting_bottle = {"target_id": "frame_bottle_02", "center_px": [280.0, 120.0], "depth_median_mm": 500.0, "detection_confidence": 0.99}

    selected = match_locked_bottle_pose([distracting_bottle, locked_bottle], manager, (240, 360, 3))

    assert selected is locked_bottle


def test_locked_redetection_rejects_distant_replacement_bottle():
    manager = SimpleNamespace(
        last_good_bottle_geometry={
            "center_px": [80.0, 120.0],
            "bbox_xywh": [58.0, 55.0, 44.0, 130.0],
            "depth_median_mm": 500.0,
            "axis_vector": [0.0, 1.0],
        },
        locked_target={"bottle_target_id": "bottle_01"},
    )
    replacement = {
        "target_id": "bottle_01",
        "center_px": [285.0, 120.0],
        "bbox_xywh": [263.0, 55.0, 44.0, 130.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
    }

    assert match_locked_bottle_pose([replacement], manager, (240, 360, 3)) is None


def test_locked_redetection_rejects_adjacent_zero_overlap_lookalike():
    manager = SimpleNamespace(
        last_good_bottle_geometry={
            "center_px": [80.0, 120.0],
            "bbox_xywh": [58.0, 55.0, 44.0, 130.0],
            "depth_median_mm": 500.0,
            "axis_vector": [0.0, 1.0],
        },
        locked_target={"bottle_target_id": "bottle_01"},
    )
    adjacent = {
        "target_id": "frame_bottle_02",
        "center_px": [125.0, 120.0],
        "bbox_xywh": [103.0, 55.0, 44.0, 130.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
    }

    assert match_locked_bottle_pose([adjacent], manager, (240, 360, 3)) is None


def test_identity_checked_broad_observation_refreshes_next_local_search_anchor():
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
    }
    manager.last_good_bottle_geometry = _oriented_rectangle_geometry(
        90.0,
        center=(100.0, 120.0),
        depth_mm=500.0,
    )
    manager.roi_bottle_fail_frames = 4
    recovered = _oriented_rectangle_geometry(
        80.0,
        center=(155.0, 118.0),
        depth_mm=350.0,
    )

    manager.record_current_bottle_observation(
        recovered,
        (240, 360, 3),
        frame_id=42,
        source="full_frame_identity_checked_reacquisition",
    )

    assert manager.roi_bottle_fail_frames == 0
    assert manager.last_good_bottle_frame_id == 42
    assert manager.last_good_bottle_depth_mm == 350.0
    assert np.allclose(manager.last_good_bottle_geometry["center_px"], [155.0, 118.0])
    assert manager.locked_target["full_bottle_anchor_frame_id"] == 42
    assert manager.locked_target["bottle_target_id"] == "bottle_01"


def test_loss_reacquisition_uses_latest_verified_close_scale_not_original_far_lock():
    far_lock_geometry = {
        "center_px": [80.0, 120.0],
        "bbox_xywh": [60.0, 70.0, 40.0, 100.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
    }
    near_verified_geometry = {
        "detected": True,
        "center_px": [185.0, 120.0],
        "bbox_xywh": [105.0, 10.0, 160.0, 220.0],
        "depth_median_mm": 100.0,
        "axis_vector": [0.0, 1.0],
        "measurement_status": "CURRENT_MEASURED_RGBD",
        "geometry_source": "roi_opencv_depth",
    }
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_original",
        "attempt_id": "attempt-original",
        "locked_bottle_geometry": far_lock_geometry,
    }
    manager.last_good_bottle_geometry = copy.deepcopy(near_verified_geometry)
    manager.last_good_bottle_bbox = [105.0, 10.0, 265.0, 230.0]
    manager.last_good_bottle_depth_mm = 100.0
    manager.last_good_bottle_frame_id = 88

    manager.clear("Locked target lost after hand occlusion.")

    assert manager.last_lost_target["locked_bottle_geometry"] == far_lock_geometry
    assert manager.last_lost_target["last_verified_bottle_geometry"]["center_px"] == [185.0, 120.0]
    assert manager.last_lost_target["last_verified_bottle_bbox"] == [105.0, 10.0, 265.0, 230.0]
    assert manager.last_lost_target["last_verified_bottle_depth_mm"] == 100.0
    assert manager.last_lost_target["last_verified_bottle_frame_id"] == 88
    assert manager.last_lost_target["last_verified_reference_status"] == "LAST_VERIFIED_RGBD_REFERENCE_NOT_CURRENT"

    same_near_bottle = {
        "target_id": "frame_candidate_01",
        "center_px": [188.0, 120.0],
        "bbox_xywh": [108.0, 12.0, 160.0, 216.0],
        "depth_median_mm": 102.0,
        "axis_vector": [0.0, 1.0],
    }
    adjacent_neighbor = {
        "target_id": "frame_candidate_02",
        "center_px": [330.0, 120.0],
        "bbox_xywh": [270.0, 10.0, 120.0, 220.0],
        "depth_median_mm": 102.0,
        "axis_vector": [0.0, 1.0],
    }

    selected = match_locked_bottle_pose(
        [adjacent_neighbor, same_near_bottle],
        manager,
        (240, 400, 3),
    )

    assert selected is same_near_bottle
    assert selected["identity_match_status"] == "ESTIMATED_GEOMETRIC_MATCH_TO_LAST_VERIFIED_RGBD_REFERENCE"
    assert selected["identity_reference_measurement_status"] == "LAST_VERIFIED_NOT_CURRENT"
    assert selected["identity_reference_frame_id"] == 88
    assert selected["identity_match_physically_validated"] is False


def test_lost_lock_does_not_auto_retarget_adjacent_bottle(monkeypatch):
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", False)
    original_geometry = {
        "center_px": [80.0, 120.0],
        "bbox_xywh": [58.0, 55.0, 44.0, 130.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
    }
    manager = TargetLockManager(enabled=True)
    manager.has_lost_lock = True
    manager.last_lost_target = {
        "valid": False,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-original",
        "locked_body_fraction": 0.52,
        "locked_bottle_geometry": original_geometry,
    }
    manager.grasp_memory = {
        "status": "PRESERVED_AFTER_TRACK_LOSS",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [80.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }
    adjacent = {
        "target_id": "bottle_02",
        "center_px": [125.0, 120.0],
        "bbox_xywh": [103.0, 55.0, 44.0, 130.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
    }
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_with_yolo_trigger",
        lambda *_args, **_kwargs: (
            [adjacent],
            None,
            {"yolo_enabled": False, "yolo_available": False, "yolo_detected": False},
        ),
    )
    monkeypatch.setattr(
        realtime_runner,
        "select_bottle_target",
        lambda *_args, **_kwargs: pytest.fail("neighbor must not reach target ranking"),
    )
    monkeypatch.setattr(realtime_runner, "write_latest_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)

    report = realtime_runner.evaluate_frame(
        np.zeros((240, 360, 3), dtype=np.uint8),
        np.full((240, 360), 500, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 180.0, "cy": 120.0},
        77,
        SimpleNamespace(),
        lock_manager=manager,
    )

    assert report["state"] == "TARGET_LOST"
    assert report["selected_bottle_target_id"] is None
    assert report["detected_bottle_count"] == 1
    assert report["eligible_bottle_count"] == 0
    selection = report["multi_bottle_target_selection"]
    assert selection["selection_status"] == "CURRENT_BOTTLES_OBSERVED_REACQUISITION_BLOCKED"
    assert selection["candidate_count"] == 1
    assert selection["candidates"][0]["target_id"] == "bottle_02"
    assert selection["candidates"][0]["eligible"] is False
    assert selection["candidates"][0]["evidence_status"] == "CURRENT_RGBD_OBSERVATION_IDENTITY_UNVERIFIED"
    assert manager.status() == "LOST"
    assert manager.locked_target is None
    assert manager.grasp_memory["locked_body_fraction"] == 0.52
    assert manager.last_lost_target["attempt_id"] == "attempt-original"

    manager.clear("manual_reset_from_dashboard")
    assert manager.status() == "SEARCHING"
    assert manager.grasp_memory is None
    assert manager.last_lost_target is None


def test_scaled_pose_scales_axis_length_and_direction():
    pose = {
        "center_px": [20.0, 30.0],
        "axis_vector": [0.6, 0.8],
        "axis_length_px": 100.0,
        "bbox_xywh": [10, 10, 20, 80],
    }

    scaled = scale_pose_to_full(pose, (240, 400, 3), (120, 100, 3))

    expected_scale = np.linalg.norm([0.6 * 4.0, 0.8 * 2.0])
    assert abs(scaled["axis_length_px"] - 100.0 * expected_scale) < 1e-6
    assert abs(np.linalg.norm(scaled["axis_vector"]) - 1.0) < 1e-6


def test_single_bottle_demo_scaled_search_requires_full_resolution_lock_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "FAST_FINAL_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "PRELOCK_FAST_SCALED_RGBD", True)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_scaled",
        lambda *_args, **_kwargs: (calls.append("scaled") or [], None),
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: (calls.append("full") or [], None),
    )
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 500, dtype=np.uint16)

    _poses, _error, search_resolution = detect_prelock_rgbd_poses(rgb, depth)
    _poses, _error, confirmation_resolution = detect_prelock_rgbd_poses(
        rgb,
        depth,
        force_full_resolution=True,
    )

    assert calls == ["scaled", "full"]
    assert search_resolution == "scaled"
    assert confirmation_resolution == "full_resolution_lock_confirmation"


def test_full_resolution_confirmation_is_due_only_on_would_be_lock_frame():
    manager = TargetLockManager(enabled=True)
    manager.stable_full_bottle_frames = 0
    assert prelock_full_resolution_confirmation_due(manager) is (
        realtime_runner.TARGET_LOCK_STABLE_FRAMES == 1
    )
    manager.stable_full_bottle_frames = realtime_runner.TARGET_LOCK_STABLE_FRAMES - 1
    assert prelock_full_resolution_confirmation_due(manager) is True
    manager.has_lost_lock = True
    manager.stable_full_bottle_frames = manager.relock_required_frames - 2
    assert prelock_full_resolution_confirmation_due(manager) is False
    manager.stable_full_bottle_frames += 1
    assert prelock_full_resolution_confirmation_due(manager) is True


def test_empty_prelock_scene_never_forces_full_resolution_confirmation(monkeypatch):
    manager = TargetLockManager(enabled=True)
    calls = []
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", False)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_scaled",
        lambda *_args, **_kwargs: (calls.append("scaled") or [], None),
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: pytest.fail(
            "an empty search scene cannot force full-resolution confirmation"
        ),
    )
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.full((240, 360), 500, dtype=np.uint16)

    for _ in range(5):
        poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
            rgb,
            depth,
            lock_manager=manager,
        )
        assert poses == []
        assert error is None
        assert report["rgbd_search_resolution"] == "scaled"

    assert calls == ["scaled"] * 5
    assert manager.stable_full_bottle_frames == 0


def test_initial_lock_rejects_one_frame_candidate_and_requires_consistent_confirmation():
    manager = TargetLockManager(enabled=True)
    candidate = _translated_current_bottle()
    candidate["target_id"] = "bottle_01"

    manager.update_stability(True, pose=candidate)

    assert realtime_runner.TARGET_LOCK_STABLE_FRAMES >= 2
    assert manager.ready_to_lock() is False

    manager.update_stability(False, pose={})
    assert manager.stable_full_bottle_frames == 0
    assert manager.ready_to_lock() is False

    manager.update_stability(True, pose=candidate)
    manager.update_stability(True, pose=candidate)
    assert manager.ready_to_lock() is True


def test_would_be_single_bottle_lock_uses_full_resolution_candidate_count(monkeypatch):
    manager = TargetLockManager(enabled=True)
    manager.stable_full_bottle_frames = realtime_runner.TARGET_LOCK_STABLE_FRAMES - 1
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", False)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_scaled",
        lambda *_args, **_kwargs: pytest.fail("scaled search cannot authorize the lock frame"),
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: (
            [_translated_current_bottle(), _translated_current_bottle()],
            None,
        ),
    )

    poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
        np.zeros((240, 360, 3), dtype=np.uint8),
        np.full((240, 360), 500, dtype=np.uint16),
        lock_manager=manager,
    )

    assert error is None
    assert len(poses) == 2
    assert report["rgbd_search_resolution"] == "full_resolution_lock_confirmation"


def test_manual_force_lock_always_requests_full_resolution_rgbd(monkeypatch):
    manager = TargetLockManager(enabled=True)
    manager.stable_full_bottle_frames = 0
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", False)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_scaled",
        lambda *_args, **_kwargs: pytest.fail(
            "manual force lock cannot use the scaled search mask"
        ),
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: ([_translated_current_bottle()], None),
    )

    poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
        np.zeros((240, 360, 3), dtype=np.uint8),
        np.full((240, 360), 500, dtype=np.uint16),
        lock_manager=manager,
        force_full_resolution=True,
    )

    assert error is None
    assert len(poses) == 1
    assert report["rgbd_search_resolution"] == "full_resolution_lock_confirmation"


def test_single_bottle_demo_blocks_multiple_full_resolution_lock_candidates():
    selection = blocked_single_bottle_demo_selection(
        [_translated_current_bottle(), _translated_current_bottle()]
    )

    assert selection["selected_target_id"] is None
    assert selection["candidate_count"] == 2
    assert selection["eligible_count"] == 0
    assert selection["selection_status"] == (
        "SINGLE_BOTTLE_DEMO_BLOCKED_MULTIPLE_OR_AMBIGUOUS"
    )
    assert all(candidate["selected"] is False for candidate in selection["candidates"])


def test_confident_single_rgbd_candidate_skips_generic_model_in_demo(monkeypatch):
    manager = TargetLockManager(enabled=True)
    candidate = _translated_current_bottle()
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", True)
    monkeypatch.setattr(
        realtime_runner,
        "detect_prelock_rgbd_poses",
        lambda *_args, **_kwargs: ([candidate], None, "scaled"),
    )
    monkeypatch.setattr(
        realtime_runner,
        "get_yolo_detector",
        lambda: pytest.fail("generic model is fallback-only for confident RGB-D"),
    )

    poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
        np.zeros((240, 360, 3), dtype=np.uint8),
        np.full((240, 360), 500, dtype=np.uint16),
        lock_manager=manager,
    )

    assert error is None
    assert poses == [candidate]
    assert report["yolo_runtime_state"] == "PRELOCK_SKIPPED_CONFIDENT_SINGLE_RGBD"
    assert report["yolo_rgbd_fusion_mode"] == "RGBD_PRIMARY_YOLO_FALLBACK_ONLY"


def _lost_single_bottle_manager():
    manager = TargetLockManager(enabled=True)
    manager.has_lost_lock = True
    manager.lock_generation = 1
    manager.last_lost_target = {
        "valid": False,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-old",
        "locked_body_fraction": 0.52,
        "last_verified_bottle_geometry": {
            "center_px": [70.0, 120.0],
            "bbox_xywh": [45.0, 50.0, 50.0, 140.0],
            "depth_median_mm": 500.0,
            "axis_vector": [0.0, 1.0],
        },
    }
    manager.grasp_memory = {
        "status": "PRESERVED_AFTER_TRACK_LOSS",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [70.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }
    return manager


def _translated_current_bottle(depth_mm=500.0):
    return {
        "target_id": "frame_candidate",
        "measurement_status": "CURRENT_MEASURED_RGBD",
        "center_px": [270.0, 120.0],
        "bbox_xywh": [245.0, 50.0, 50.0, 140.0],
        "depth_median_mm": depth_mm,
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
        "bottle_detection_status": "CANDIDATE",
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_depth_fused",
        "bottle_color_fraction": 0.75,
        "axis_vector": [0.0, 1.0],
    }


def test_single_bottle_auto_reacquisition_accepts_one_compatible_current_candidate():
    manager = _lost_single_bottle_manager()
    candidate = _translated_current_bottle()

    selected, status = select_lost_target_reacquisition_candidate(
        [candidate],
        manager,
        (240, 360, 3),
        single_bottle_demo_mode=True,
    )

    assert selected is candidate
    assert status == "SINGLE_BOTTLE_AUTO_REACQUIRE_STABILITY_PENDING"
    assert selected["reacquired_original_target_identity"] is False
    assert selected["single_bottle_demo_reacquisition_candidate"] is True
    required, remembered, source = prelock_body_fraction(
        manager,
        selected,
        (240, 360, 3),
    )
    assert required == remembered == 0.52
    assert source == "SINGLE_BOTTLE_DEMO_NEW_LOCK_PRESERVES_REMEMBERED_BODY_FRACTION"


def test_single_bottle_auto_reacquisition_blocks_multiple_even_if_one_strictly_matches():
    manager = _lost_single_bottle_manager()
    strict_match = _translated_current_bottle()
    strict_match.update({
        "center_px": [70.0, 120.0],
        "bbox_xywh": [45.0, 50.0, 50.0, 140.0],
    })

    selected, status = select_lost_target_reacquisition_candidate(
        [strict_match, _translated_current_bottle()],
        manager,
        (240, 360, 3),
        single_bottle_demo_mode=True,
    )

    assert selected is None
    assert status == "SINGLE_BOTTLE_AUTO_REACQUIRE_REQUIRES_EXACTLY_ONE_CANDIDATE"


def test_single_bottle_auto_reacquisition_blocks_incompatible_sole_candidate():
    manager = _lost_single_bottle_manager()

    selected, status = select_lost_target_reacquisition_candidate(
        [_translated_current_bottle(depth_mm=1200.0)],
        manager,
        (240, 360, 3),
        single_bottle_demo_mode=True,
    )

    assert selected is None
    assert status == "SINGLE_BOTTLE_AUTO_REACQUIRE_INCOMPATIBLE_CURRENT_RGBD"


def test_active_lock_never_accepts_remote_sole_candidate_by_compatibility():
    manager = TargetLockManager(enabled=True)
    original_anchor = {
        "center_px": [228.0, 179.0],
        "bbox_xywh": [207.0, 110.0, 42.0, 139.0],
        "depth_median_mm": 325.0,
        "axis_vector": [0.0, 1.0],
    }
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-live",
        "locked_bottle_geometry": copy.deepcopy(original_anchor),
        "full_bottle_anchor_geometry": copy.deepcopy(original_anchor),
        "last_verified_bottle_geometry": copy.deepcopy(original_anchor),
    }
    manager.lock_state = "LOCKED"
    remote = {
        "target_id": "frame_candidate",
        "measurement_status": "CURRENT_MEASURED_RGBD",
        "center_px": [438.0, 123.0],
        "bbox_xywh": [416.0, 54.0, 44.0, 138.0],
        "depth_median_mm": 330.0,
        "depth_confidence": 0.95,
        "detection_confidence": 0.95,
        "bottle_detection_status": "CANDIDATE",
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_depth_fused",
        "bottle_color_fraction": 0.75,
        "axis_vector": [0.0, 1.0],
    }
    assert realtime_runner.demo_bottle_identity_eligibility(remote)[0] is True
    assert realtime_runner.single_bottle_reacquisition_compatibility(
        remote,
        manager,
        (270, 480, 3),
    )[0] is True

    selected = realtime_runner.select_active_locked_bottle_recovery_candidate(
        [remote],
        manager,
        (270, 480, 3),
    )

    assert selected is None
    assert manager.locked_target["full_bottle_anchor_geometry"] == original_anchor
    assert manager.locked_target["last_verified_bottle_geometry"] == original_anchor
    assert remote.get("reacquired_original_target_identity") is not True
    assert remote.get("sticky_lock_reacquisition_compatible") is not True


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("bottle_detection_status", "LOW_CONFIDENCE", "demo_identity_detection_status_not_candidate"),
        ("bottle_profile_valid", False, "demo_identity_bottle_profile_invalid"),
        ("segmentation_rejection_reasons", ["possible_background_or_hand_contamination"], "demo_identity_segmentation_rejected"),
        ("bottle_color_fraction", 0.0, "demo_identity_known_blue_support_too_low"),
    ],
)
def test_active_recovery_rejects_overlapping_hand_like_pose_before_positional_match(
    field, value, expected_reason
):
    manager = _lost_single_bottle_manager()
    manager.has_lost_lock = False
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-live",
        "locked_bottle_geometry": copy.deepcopy(
            manager.last_lost_target["last_verified_bottle_geometry"]
        ),
    }
    manager.lock_state = "LOCKED"
    candidate = _translated_current_bottle()
    candidate.update({
        "center_px": [70.0, 120.0],
        "bbox_xywh": [45.0, 50.0, 50.0, 140.0],
        field: value,
    })

    selected = realtime_runner.select_active_locked_bottle_recovery_candidate(
        [candidate], manager, (240, 360, 3)
    )

    assert selected is None
    assert expected_reason in manager.last_active_recovery_candidate_rejections[0]["reasons"]
    assert manager.locked_target["locked_bottle_geometry"]["center_px"] == [70.0, 120.0]


def test_lost_lock_reacquisition_rejects_nonblue_candidate_even_when_geometry_matches():
    manager = _lost_single_bottle_manager()
    candidate = _translated_current_bottle()
    candidate.update({
        "center_px": [70.0, 120.0],
        "bbox_xywh": [45.0, 50.0, 50.0, 140.0],
        "bottle_color_fraction": 0.0,
        "segmentation_source": "depth_only",
    })

    selected, status = select_lost_target_reacquisition_candidate(
        [candidate], manager, (240, 360, 3), single_bottle_demo_mode=True
    )

    assert selected is None
    assert status == "SINGLE_BOTTLE_AUTO_REACQUIRE_REQUIRES_KNOWN_BLUE_CURRENT_RGBD"
    reasons = manager.last_lost_recovery_candidate_rejections[0]["reasons"]
    assert "demo_identity_requires_known_blue_rgb_component" in reasons
    assert "demo_identity_known_blue_support_too_low" in reasons


def test_strict_post_loss_match_uses_stored_fraction_beyond_memory_pixel_radius():
    manager = _lost_single_bottle_manager()
    candidate = _translated_current_bottle()
    candidate.update({
        "center_px": [115.0, 120.0],
        "bbox_xywh": [90.0, 50.0, 50.0, 140.0],
    })

    selected, status = select_lost_target_reacquisition_candidate(
        [candidate],
        manager,
        (240, 360, 3),
        single_bottle_demo_mode=True,
    )
    required, remembered, source = prelock_body_fraction(
        manager,
        selected,
        (240, 360, 3),
    )

    assert status == "ORIGINAL_TARGET_UNAMBIGUOUS_MATCH"
    assert selected["reacquired_original_target_identity"] is True
    assert required == remembered == 0.52
    assert source == "VERIFIED_ORIGINAL_TARGET_REACQUISITION_PRESERVES_LOCKED_BAND"


def test_required_bottle_fraction_is_preserved_in_new_close_view_plan():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        required_body_fraction=0.68,
    )

    assert error is None
    assert plan["grasp_memory_applied"]
    assert plan["grasp_body_fraction_constraint"]["status"] == "APPLIED_BOTTLE_RELATIVE"
    assert abs(plan["body_fraction"] - 0.68) < 0.02


def test_clearance_allowed_set_selects_highest_ranked_valid_preferred_band():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        allowed_body_fractions=[0.45, 0.52, 0.60, 0.32],
    )

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.52
    assert plan["selected_grasp_preference_tier"] == "PREFERRED"
    assert plan["grasp_band_fallback_used"] is False
    assert plan["candidate_ranking_table"][0]["selected"] is True
    assert plan["candidate_ranking_table"][0]["requested_body_fraction"] == 0.52


def test_clearance_blocked_rank_one_uses_best_remaining_preferred_band():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        allowed_body_fractions=[0.45, 0.60, 0.32],
    )

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.45
    blocked_rank = next(
        row for row in plan["candidate_ranking_table"]
        if row["requested_body_fraction"] == 0.52
    )
    assert blocked_rank["clearance_allowed"] is False
    assert blocked_rank["selection_eligible"] is False


def test_clearance_uses_profile_fallback_only_when_no_preferred_band_is_allowed():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        allowed_body_fractions=[0.32],
    )

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.32
    assert plan["selected_grasp_preference_tier"] == "FALLBACK"
    assert plan["grasp_band_fallback_used"] is True


def test_no_planner_candidate_in_clearance_allowed_set_fails_closed():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        allowed_body_fractions=[0.50],
    )

    assert plan is None
    assert "clearance gates" in error


def test_remembered_fraction_outside_current_allowed_set_fails_closed():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        required_body_fraction=0.52,
        allowed_body_fractions=[0.45],
    )

    assert plan is None
    assert "not allowed" in error


def test_remembered_fraction_matches_equivalent_current_clearance_band():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }

    plan, error = plan_pregrasp(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        required_body_fraction=0.5197,
        allowed_body_fractions=[0.52],
    )

    assert error is None
    assert plan is not None
    assert plan["selected_requested_body_fraction"] == pytest.approx(0.5197)
    assert abs(plan["body_fraction"] - 0.5197) <= 0.015


def test_grasp_memory_survives_tracking_loss_but_manual_reset_clears_it():
    manager = TargetLockManager(enabled=True)
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [160.0, 120.0],
    }
    manager.display_smoothing_state = {
        "grasp_reference_id": "original-optimal-band",
        "display_target_clamp": {"clamp_center_px": [160.0, 120.0]},
    }

    manager.clear("temporary_roi_tracking_loss")

    assert manager.grasp_memory["status"] == "PRESERVED_AFTER_TRACK_LOSS"
    assert manager.display_smoothing_state["grasp_reference_id"] == "original-optimal-band"
    assert manager.remembered_body_fraction_for_pose(
        {"target_id": "frame_local_changed", "center_px": [168.0, 124.0]},
        (240, 320, 3),
    ) == 0.52
    manager.clear("manual_reset_from_dashboard")
    assert manager.grasp_memory is None
    assert manager.display_smoothing_state == {}


def test_grasp_memory_does_not_transfer_to_distant_frame_local_bottle_id():
    manager = TargetLockManager(enabled=True)
    manager.grasp_memory = {
        "status": "PRESERVED_AFTER_TRACK_LOSS",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [80.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }

    inherited = manager.remembered_body_fraction_for_pose(
        {
            "target_id": "bottle_01",
            "center_px": [245.0, 120.0],
            "depth_median_mm": 500.0,
            "axis_vector": [0.0, 1.0],
        },
        (240, 320, 3),
    )

    assert inherited is None


def test_lost_target_reacquisition_preserves_original_band_and_fails_closed_if_blocked():
    manager = TargetLockManager(enabled=True)
    manager.has_lost_lock = True
    manager.grasp_memory = {
        "status": "PRESERVED_AFTER_TRACK_LOSS",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [160.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }
    pose = {
        "center_px": [162.0, 121.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
        "selection_grasp_body_fraction": 0.32,
        "inter_bottle_grasp_clearance": {
            "grasp_clearance_status": "ESTIMATED_CLEAR",
            "clear_grasp_body_fractions": [0.32],
        },
    }

    required, remembered, source = prelock_body_fraction(manager, pose, (240, 320, 3))

    assert required == 0.52
    assert remembered == 0.52
    assert source == "VERIFIED_ORIGINAL_TARGET_REACQUISITION_PRESERVES_LOCKED_BAND"
    assert manager.grasp_memory["status"] == "PRESERVED_AFTER_TRACK_LOSS"


def test_current_optimal_clearance_band_replaces_stale_prelock_fallback_memory():
    manager = TargetLockManager(enabled=True)
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "locked_body_fraction": 0.32,
        "center_px_at_lock": [160.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }
    pose = {
        "center_px": [160.0, 120.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
        "selection_grasp_body_fraction": 0.52,
        "inter_bottle_grasp_clearance": {
            "grasp_clearance_status": "ESTIMATED_CLEAR",
            "preferred_grasp_body_fraction": 0.52,
            "selected_grasp_body_fraction": 0.52,
            "grasp_band_fallback_used": False,
            "clear_grasp_body_fractions": [0.45, 0.52, 0.60],
        },
    }

    required, remembered, source = prelock_body_fraction(manager, pose, (240, 320, 3))

    assert required is None
    assert remembered == 0.32
    assert source == "CURRENT_CLEARANCE_ALLOWED_RANKED_SELECTION"


def test_current_clear_fallback_band_remains_when_preferred_band_is_blocked():
    manager = TargetLockManager(enabled=True)
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [160.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }
    pose = {
        "center_px": [160.0, 120.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
        "selection_grasp_body_fraction": 0.32,
        "inter_bottle_grasp_clearance": {
            "grasp_clearance_status": "ESTIMATED_CLEAR",
            "preferred_grasp_body_fraction": 0.52,
            "selected_grasp_body_fraction": 0.32,
            "grasp_band_fallback_used": True,
            "clear_grasp_body_fractions": [0.32],
        },
    }

    required, remembered, source = prelock_body_fraction(manager, pose, (240, 320, 3))

    assert required is None
    assert remembered == 0.52
    assert source == "CURRENT_CLEARANCE_ALLOWED_RANKED_SELECTION"


def test_grasp_memory_lookup_does_not_claim_reacquisition_before_lock():
    manager = TargetLockManager(enabled=True)
    manager.grasp_memory = {
        "status": "PRESERVED_AFTER_TRACK_LOSS",
        "locked_body_fraction": 0.52,
        "center_px_at_lock": [160.0, 120.0],
        "depth_mm_at_lock": 500.0,
        "axis_vector_at_lock": [0.0, 1.0],
    }

    inherited = manager.remembered_body_fraction_for_pose(
        {
            "center_px": [162.0, 121.0],
            "depth_median_mm": 500.0,
            "axis_vector": [0.0, 1.0],
        },
        (240, 320, 3),
    )

    assert inherited == 0.52
    assert manager.grasp_memory["status"] == "PRESERVED_AFTER_TRACK_LOSS"
    assert "last_match_timestamp" not in manager.grasp_memory


def test_manual_gripper_open_ends_post_grasp_but_retains_target_memory():
    worker = EvaluationWorker(lock_target=True)

    class MonitorStub:
        stopped = False

        def stop(self):
            self.stopped = True

    monitor = MonitorStub()
    worker.event_monitor = monitor
    worker.post_grasp_active = True
    worker.post_grasp_baseline = {"frame_id": 10}
    worker.post_grasp_stable_frames = 3
    worker.latest_report = {"state": "READY_TO_LIFT", "gripper_closed_confirmed": True}
    worker.last_full_eval_time = 123.0
    worker.locked_fast_frames_since_full = 5
    worker.lock_manager.locked_target = {"valid": True, "bottle_target_id": "bottle_01"}
    worker.lock_manager.current_tracked_target = {"tracking_ok": True}
    worker.lock_manager.grasp_memory = {"status": "LOCKED_BOTTLE_RELATIVE_GRASP"}
    worker.lock_manager.ready_candidate_frames = 4
    worker.lock_manager.stable_ready_frames = 4
    worker.lock_manager.temporal_state = {
        "finger_stable_frames": 6,
        "last_valid_true_3d_clamp": {"clamp_center_px": [160.0, 120.0]},
    }
    worker.lock_manager.good_frame_history.append({"frame_id": 9})

    worker.mark_gripper_open()

    assert worker.post_grasp_active is False
    assert worker.post_grasp_baseline is None
    assert worker.post_grasp_stable_frames == 0
    assert worker.latest_report["state"] == "APPROACH_TRACKING"
    assert worker.latest_report["post_grasp_active"] is False
    assert worker.latest_report["manual_gripper_closed"] is False
    assert worker.last_full_eval_time == 0.0
    assert worker.locked_fast_frames_since_full == 0
    assert worker.lock_manager.locked_target["bottle_target_id"] == "bottle_01"
    assert worker.lock_manager.current_tracked_target["tracking_ok"] is True
    assert worker.lock_manager.grasp_memory["status"] == "LOCKED_BOTTLE_RELATIVE_GRASP"
    assert worker.lock_manager.ready_candidate_frames == 0
    assert worker.lock_manager.stable_ready_frames == 0
    assert worker.lock_manager.temporal_state["last_valid_true_3d_clamp"]["clamp_center_px"] == [160.0, 120.0]
    assert list(worker.lock_manager.good_frame_history) == []
    assert monitor.stopped is True


def test_compact_pre_close_status_reports_explicit_inactive_manual_state():
    compact = compact_live_status({"state": "APPROACH_TRACKING"})

    assert compact["post_grasp_active"] is False
    assert compact["manual_gripper_closed"] is False


def test_duplicate_manual_close_does_not_restart_post_grasp_monitor():
    worker = EvaluationWorker(lock_target=True)

    class MonitorStub:
        starts = 0

        def start(self, _baseline):
            self.starts += 1
            return {}

    monitor = MonitorStub()
    worker.event_monitor = monitor
    worker.post_grasp_active = True
    worker.mark_gripper_closed()

    assert monitor.starts == 0


def test_manual_close_preserves_original_target_and_starts_matching_baseline():
    frozen_band = {
        "locked_quality_compute_id": "quality-1",
        "locked_quality_writer_session_id": "writer-1",
        "locked_quality_attempt_id": "attempt-original",
        "locked_quality_lock_frame_id": 41,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "locked_quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "locked_epsilon": 0.08,
        "locked_quality_score": 88.0,
        "locked_quality_force_closure": True,
        "locked_quality_label": "GOOD",
    }
    worker = EvaluationWorker(lock_target=True)
    worker.lock_manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-original",
        "lock_frame_id": 41,
        "locked_body_fraction": 0.52,
        "locked_grasp_band": copy.deepcopy(frozen_band),
    }
    worker.lock_manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
    }
    worker.latest_report = {
        "frame_id": 55,
        "writer_session_id": "writer-1",
        "state": "READY_TO_CLOSE",
        "decision": "READY_TO_CLOSE",
        "recommended_action": "CLOSE_GRIPPER",
        "ready_to_close": True,
        "bottle_geometry": {
            "center_px": [160.0, 120.0],
            "axis_angle_degrees": 90.0,
            "depth_median_mm": 95.0,
        },
        "target_lock": {
            "attempt_id": "attempt-original",
            "lock_frame_id": 41,
            "locked_body_fraction": 0.52,
            "locked_target": copy.deepcopy(worker.lock_manager.locked_target),
            "locked_grasp_band": copy.deepcopy(frozen_band),
        },
        "quality_matrix": {
            "quality_compute_id": "quality-1",
            "writer_session_id": "writer-1",
            "lock_frame_id": 41,
            "lock_attempt_id": "attempt-original",
            "quality_matrix_status": "updated",
            "quality_label": "GOOD",
            "raw_epsilon": 0.08,
            "normalized_quality_score": 88.0,
        },
    }

    class MonitorStub:
        baseline = None

        def start(self, baseline):
            self.baseline = copy.deepcopy(baseline)
            return {}

    monitor = MonitorStub()
    worker.event_monitor = monitor
    transition_results = []
    worker.post_grasp_transition_result.connect(transition_results.append)

    worker.mark_gripper_closed_for_command("close-ready-1")

    assert worker.post_grasp_active is True
    assert worker.post_grasp_stable_frames == 0
    assert worker.post_grasp_baseline == monitor.baseline
    assert monitor.baseline["bottle_target_id"] == "bottle_01"
    assert monitor.baseline["attempt_id"] == "attempt-original"
    assert monitor.baseline["lock_frame_id"] == 41
    assert monitor.baseline["locked_body_fraction"] == 0.52
    assert monitor.baseline["writer_session_id"] == "writer-1"
    assert monitor.baseline["quality_compute_id"] == "quality-1"
    assert monitor.baseline["quality_label"] == "GOOD"
    assert monitor.baseline["quality_epsilon"] == 0.08
    assert monitor.baseline["quality_score"] == 88.0
    assert monitor.baseline["baseline_frozen"] is True
    assert transition_results[-1]["ok"] is True
    assert transition_results[-1]["command_id"] == "close-ready-1"
    assert transition_results[-1]["post_grasp_active"] is True
    assert transition_results[-1]["quality_compute_id"] == "quality-1"
    assert worker.lock_manager.locked_target["bottle_target_id"] == "bottle_01"
    assert worker.lock_manager.grasp_memory["locked_body_fraction"] == 0.52


def test_weak_zero_baseline_cannot_start_post_grasp_monitor():
    weak_band = {
        "locked_quality_compute_id": "quality-zero",
        "locked_quality_writer_session_id": "writer-zero",
        "locked_quality_attempt_id": "attempt-zero",
        "locked_quality_lock_frame_id": 9,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "locked_quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "locked_epsilon": 0.0,
        "locked_quality_score": 0.0,
        "locked_quality_force_closure": False,
        "locked_quality_label": "WEAK",
    }
    worker = EvaluationWorker(lock_target=True)
    worker.lock_manager.locked_target = {
        "valid": True,
        "attempt_id": "attempt-zero",
        "lock_frame_id": 9,
        "locked_grasp_band": copy.deepcopy(weak_band),
    }
    worker.latest_report = {
        "writer_session_id": "writer-zero",
        "target_lock": {
            "attempt_id": "attempt-zero",
            "lock_frame_id": 9,
            "locked_target": copy.deepcopy(worker.lock_manager.locked_target),
            "locked_grasp_band": copy.deepcopy(weak_band),
        },
    }

    class MonitorStub:
        starts = 0

        def start(self, _baseline):
            self.starts += 1
            return {}

    monitor = MonitorStub()
    worker.event_monitor = monitor
    results = []
    worker.post_grasp_transition_result.connect(results.append)

    worker.mark_gripper_closed_for_command("close-zero")

    assert worker.post_grasp_active is False
    assert worker.post_grasp_baseline is None
    assert monitor.starts == 0
    assert results[-1]["ok"] is False
    assert results[-1]["post_grasp_baseline_reason"] == "quality_epsilon_not_positive"


def test_davis_monitor_start_failure_rolls_back_post_grasp_transition():
    band = {
        "locked_quality_compute_id": "quality-1",
        "locked_quality_writer_session_id": "writer-1",
        "locked_quality_attempt_id": "attempt-1",
        "locked_quality_lock_frame_id": 41,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "locked_quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "locked_epsilon": 0.08,
        "locked_quality_score": 88.0,
        "locked_quality_force_closure": True,
        "locked_quality_label": "GOOD",
    }
    worker = EvaluationWorker(lock_target=True)
    worker.lock_manager.locked_target = {
        "valid": True,
        "attempt_id": "attempt-1",
        "lock_frame_id": 41,
        "locked_grasp_band": copy.deepcopy(band),
    }
    worker.latest_report = {
        "writer_session_id": "writer-1",
        "target_lock": {
            "attempt_id": "attempt-1",
            "lock_frame_id": 41,
            "locked_target": copy.deepcopy(worker.lock_manager.locked_target),
            "locked_grasp_band": copy.deepcopy(band),
        },
    }

    class MonitorStub:
        @staticmethod
        def start(_baseline):
            raise RuntimeError("synthetic DAVIS start failure")

    worker.event_monitor = MonitorStub()
    results = []
    worker.post_grasp_transition_result.connect(results.append)

    worker.mark_gripper_closed_for_command("close-davis-fail")

    assert worker.post_grasp_active is False
    assert worker.post_grasp_baseline is None
    assert results[-1]["ok"] is False
    assert results[-1]["reason"] == "davis_monitor_start_failed"


def test_post_grasp_baseline_uses_immutable_band_when_transient_matrix_is_absent():
    frozen_band = {
        "locked_quality_compute_id": "quality-frozen",
        "locked_quality_writer_session_id": "writer-frozen",
        "locked_quality_attempt_id": "attempt-frozen",
        "locked_quality_lock_frame_id": 72,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "locked_quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "locked_epsilon": 0.126,
        "locked_quality_score": 84.0,
        "locked_quality_force_closure": True,
        "locked_quality_label": "GOOD",
    }
    report = {
        "writer_session_id": "writer-frozen",
        "target_lock": {
            "attempt_id": "attempt-frozen",
            "lock_frame_id": 72,
            "locked_grasp_band": copy.deepcopy(frozen_band),
            "locked_target": {
                "attempt_id": "attempt-frozen",
                "lock_frame_id": 72,
                "bottle_target_id": "bottle-frozen",
                "locked_grasp_band": copy.deepcopy(frozen_band),
            },
        },
    }

    readiness = realtime_runner.immutable_lock_baseline_readiness(report)
    assert readiness["ready"] is True

    baseline = realtime_runner.post_grasp_baseline_from_report(report)

    assert baseline["writer_session_id"] == readiness["writer_session_id"]
    assert baseline["attempt_id"] == readiness["attempt_id"]
    assert baseline["lock_frame_id"] == readiness["lock_frame_id"]
    assert baseline["quality_lock_attempt_id"] == readiness["attempt_id"]
    assert baseline["quality_lock_frame_id"] == readiness["lock_frame_id"]
    assert baseline["quality_compute_id"] == readiness["quality_compute_id"]
    assert baseline["quality_epsilon"] == readiness["quality_epsilon"] == 0.126
    assert baseline["quality_score"] == readiness["quality_score"] == 84.0
    assert baseline["quality_baseline_role"] == "IMMUTABLE_LOCK_BASELINE"
    assert baseline["quality_input_source"] == "planned_contact"
    assert baseline["quality_contact_source_forced"] is True
    assert baseline["baseline_frozen"] is True


def test_post_grasp_baseline_rejects_mismatched_transient_matrix_authority():
    frozen_band = {
        "locked_quality_compute_id": "quality-authoritative",
        "locked_quality_writer_session_id": "writer-authoritative",
        "locked_quality_attempt_id": "attempt-authoritative",
        "locked_quality_lock_frame_id": 91,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "locked_quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "locked_epsilon": 0.126,
        "locked_quality_score": 84.0,
        "locked_quality_force_closure": True,
        "locked_quality_label": "GOOD",
    }
    report = {
        "writer_session_id": "writer-authoritative",
        "target_lock": {
            "attempt_id": "attempt-authoritative",
            "lock_frame_id": 91,
            "locked_grasp_band": copy.deepcopy(frozen_band),
            "locked_target": {
                "attempt_id": "attempt-authoritative",
                "lock_frame_id": 91,
                "locked_grasp_band": copy.deepcopy(frozen_band),
            },
        },
        "quality_matrix": {
            "quality_compute_id": "quality-stray",
            "writer_session_id": "writer-stray",
            "lock_frame_id": 17,
            "lock_attempt_id": "attempt-stray",
            "raw_epsilon": 0.0,
            "normalized_quality_score": 0.0,
            "quality_matrix_status": "updated",
            "quality_label": "WEAK",
        },
    }

    readiness = realtime_runner.immutable_lock_baseline_readiness(report)
    assert readiness["ready"] is True

    baseline = realtime_runner.post_grasp_baseline_from_report(report)

    assert baseline["writer_session_id"] == readiness["writer_session_id"]
    assert baseline["attempt_id"] == readiness["attempt_id"]
    assert baseline["lock_frame_id"] == readiness["lock_frame_id"]
    assert baseline["quality_lock_attempt_id"] == readiness["attempt_id"]
    assert baseline["quality_lock_frame_id"] == readiness["lock_frame_id"]
    assert baseline["quality_compute_id"] == readiness["quality_compute_id"]
    assert baseline["quality_epsilon"] == readiness["quality_epsilon"] == 0.126
    assert baseline["quality_score"] == readiness["quality_score"] == 84.0
    assert baseline["quality_matrix_status"] == "updated"
    assert baseline["quality_label"] == "GOOD"
    assert baseline["quality_force_closure"] is True


def test_compact_live_status_preserves_frozen_post_grasp_baseline():
    baseline = {
        "writer_session_id": "writer-1",
        "attempt_id": "attempt-1",
        "lock_frame_id": 44,
        "quality_compute_id": "quality-1",
        "quality_epsilon": 0.126,
        "quality_score": 84.0,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
    }

    compact = compact_live_status(
        {
            "post_grasp_active": True,
            "post_grasp_baseline": baseline,
            "writer_session_id": "writer-1",
            "target_lock": {
                "attempt_id": "attempt-1",
                "lock_frame_id": 44,
                "locked_grasp_band": {},
            },
        }
    )

    assert compact["post_grasp_baseline"] == baseline


def test_manual_close_waits_for_exact_immutable_baseline_before_starting_monitor():
    worker = EvaluationWorker(lock_target=True)
    worker.lock_manager.locked_target = {
        "valid": True,
        "attempt_id": "attempt-1",
        "lock_frame_id": 41,
        "locked_grasp_band": {},
    }
    worker.latest_report = {
        "writer_session_id": "writer-1",
        "target_lock": {
            "target_status": "LOCKED",
            "attempt_id": "attempt-1",
            "lock_frame_id": 41,
            "locked_target": copy.deepcopy(worker.lock_manager.locked_target),
            "locked_grasp_band": {},
        },
    }

    class MonitorStub:
        starts = 0

        def start(self, _baseline):
            self.starts += 1
            return {}

    worker.event_monitor = MonitorStub()

    worker.mark_gripper_closed()

    assert worker.post_grasp_active is False
    assert worker.post_grasp_baseline is None
    assert worker.event_monitor.starts == 0


def test_inflight_quality_result_is_discarded_after_manual_release():
    cleared = []
    runner = SimpleNamespace(
        quality_matrix_busy=True,
        discard_next_quality_matrix_result=True,
        clear_quality_matrix_outputs=cleared.append,
    )

    RealtimeBackendRunner.on_quality_matrix_result(
        runner,
        {"quality_matrix_status": "complete", "epsilon_score": 99.0},
    )

    assert runner.quality_matrix_busy is False
    assert runner.discard_next_quality_matrix_result is False
    assert cleared == ["discarded_quality_result_from_released_grasp"]


def test_inactive_manual_release_command_is_an_idempotent_noop(monkeypatch, tmp_path):
    command_path = tmp_path / "command.json"
    command_path.write_text(
        json.dumps(
            {
                "command": "mark_gripper_open",
                "command_id": "release-1",
                "requested_at": time.time(),
            }
        )
    )
    monkeypatch.setattr(realtime_runner, "BACKEND_COMMAND_PATH", command_path)
    emitted = []
    cleared = []
    statuses = []
    runner = SimpleNamespace(
        last_backend_command_mtime_ns=None,
        last_backend_command_id=None,
        backend_start_time=0.0,
        eval_worker=SimpleNamespace(
            post_grasp_active=False,
            lock_manager=SimpleNamespace(status=lambda: "LOCKED"),
        ),
        gripper_open_requested=SimpleNamespace(emit=lambda: emitted.append(True)),
        quality_matrix_busy=False,
        clear_quality_matrix_outputs=cleared.append,
        write_backend_command_status=lambda *args, **kwargs: statuses.append((args, kwargs)),
    )

    RealtimeBackendRunner.handle_backend_command_file(runner)

    assert emitted == []
    assert cleared == []
    assert statuses[-1][0][2] is True
    assert "already inactive" in statuses[-1][0][3]


def test_active_manual_release_preserves_same_lock_analytic_quality(monkeypatch, tmp_path):
    command_path = tmp_path / "command.json"
    command_path.write_text(
        json.dumps(
            {
                "command": "mark_gripper_open",
                "command_id": "release-2",
                "requested_at": time.time(),
            }
        )
    )
    monkeypatch.setattr(realtime_runner, "BACKEND_COMMAND_PATH", command_path)
    emitted = []
    cleared = []
    runner = SimpleNamespace(
        last_backend_command_mtime_ns=None,
        last_backend_command_id=None,
        backend_start_time=0.0,
        eval_worker=SimpleNamespace(
            post_grasp_active=True,
            lock_manager=SimpleNamespace(status=lambda: "LOCKED"),
        ),
        gripper_open_requested=SimpleNamespace(emit=lambda: emitted.append(True)),
        quality_matrix_busy=True,
        last_auto_hull_lock_frame_id=1,
        last_actual_quality_lock_frame_id=1,
        last_finger_quality_signature=("old",),
        discard_next_quality_matrix_result=False,
        last_quality_matrix_request_status={},
        clear_quality_matrix_outputs=cleared.append,
        write_backend_command_status=lambda *args, **kwargs: None,
    )

    RealtimeBackendRunner.handle_backend_command_file(runner)

    assert emitted == [True]
    assert cleared == []
    assert runner.discard_next_quality_matrix_result is False
    assert runner.last_auto_hull_lock_frame_id == 1
    assert runner.last_actual_quality_lock_frame_id == 1
    assert runner.last_finger_quality_signature == ("old",)


def test_queued_post_grasp_lock_change_cannot_receive_false_started_ack(monkeypatch, tmp_path):
    command_path = tmp_path / "command.json"
    command_path.write_text(
        json.dumps({
            "command": "mark_gripper_closed",
            "command_id": "close-race",
            "requested_at": time.time(),
        })
    )
    monkeypatch.setattr(realtime_runner, "BACKEND_COMMAND_PATH", command_path)
    monkeypatch.setattr(
        realtime_runner,
        "immutable_lock_baseline_readiness",
        lambda report: (
            {
                "ready": True,
                "reason": None,
                "reasons": [],
                "quality_compute_id": "quality-ready",
            }
            if (report or {}).get("phase") == "ready"
            else {
                "ready": False,
                "reason": "attempt_baseline_lineage_mismatch",
                "reasons": ["attempt_baseline_lineage_mismatch"],
                "quality_compute_id": None,
            }
        ),
    )
    worker = EvaluationWorker(lock_target=True)
    worker.lock_manager.locked_target = {"valid": True, "attempt_id": "attempt-1"}
    worker.latest_report = {"phase": "ready"}
    statuses = []
    runner = SimpleNamespace(
        last_backend_command_mtime_ns=None,
        last_backend_command_id=None,
        backend_start_time=0.0,
        eval_worker=worker,
        write_backend_command_status=lambda *args, **kwargs: statuses.append((args, kwargs)),
    )
    worker.post_grasp_transition_result.connect(
        lambda result: RealtimeBackendRunner.on_post_grasp_transition_result(runner, result)
    )

    class QueuedSignal:
        @staticmethod
        def emit(command_id):
            # Reproduce the exact queued-thread race: handler precheck passed,
            # then the active report/lineage changed before the worker slot.
            worker.latest_report = {"phase": "changed"}
            worker.mark_gripper_closed_for_command(command_id)

    runner.gripper_closed_for_command_requested = QueuedSignal()

    RealtimeBackendRunner.handle_backend_command_file(runner)

    assert worker.post_grasp_active is False
    assert len(statuses) == 1
    args, details = statuses[0]
    assert args[0] == "mark_gripper_closed"
    assert args[1] == "close-race"
    assert args[2] is False
    assert details["command_transition_state"] == "rejected"
    assert details["post_grasp_active"] is False


def test_new_automatic_lock_requests_planned_contact_baseline_without_button(monkeypatch):
    requests = []
    monkeypatch.setattr(realtime_runner.QTimer, "singleShot", lambda *_args: None)
    runner = SimpleNamespace(
        evaluating=True,
        last_auto_hull_lock_frame_id=None,
        last_actual_quality_lock_frame_id=None,
        last_finger_quality_signature=None,
        maybe_evaluate=lambda: None,
        request_quality_matrix=lambda report, manual=False, require_actual_contact=None: (
            requests.append({
                "report": report,
                "manual": manual,
                "require_actual_contact": require_actual_contact,
            })
            or True
        ),
    )
    report = {
        "writer_session_id": "writer-1",
        "target_lock": {
            "lock_frame_id": 41,
            "attempt_id": "attempt-41",
            "target_status": "LOCKED",
        },
        "finger_detection": {
            "finger_detection_status": "BOTH_FINGERS_DETECTED",
            "finger_stable_frames": realtime_runner.FINGER_STABLE_FRAMES_REQUIRED,
            "left_finger_contact_px": [10.0, 20.0],
            "right_finger_contact_px": [30.0, 20.0],
        },
        "contact_source": "planned_contact",
    }

    RealtimeBackendRunner.on_eval_result(runner, report)

    assert runner.evaluating is False
    assert runner.last_auto_hull_lock_frame_id == 41
    assert len(requests) == 1
    assert requests[0]["manual"] is False
    assert requests[0]["require_actual_contact"] is False
    assert requests[0]["report"]["_quality_force_planned_contact"] is True
    assert requests[0]["report"]["_quality_baseline_role"] == "IMMUTABLE_LOCK_BASELINE"


def test_same_lineage_quality_request_is_blocked_after_baseline_freezes():
    report = {
        "writer_session_id": "writer-1",
        "target_lock": {
            "target_status": "TRACKING",
            "lock_frame_id": 41,
            "attempt_id": "attempt-41",
        },
    }
    runner = SimpleNamespace(
        last_auto_quality_lineage=("writer-1", "attempt-41", 41),
        auto_quality_baseline_in_flight=False,
        last_quality_matrix_request_status={},
    )

    started = RealtimeBackendRunner.request_quality_matrix(
        runner,
        report,
        manual=True,
    )

    assert started is False
    assert runner.last_quality_matrix_request_status["quality_matrix_failure_reasons"] == [
        "immutable_lock_baseline_already_computed"
    ]


def test_normal_approach_motion_is_live_observation_not_analytic_staleness():
    report = {
        "writer_session_id": "writer-a",
        "frame_id": 20,
        "target_lock": {
            "lock_frame_id": 10,
            "attempt_id": "attempt-a",
            "locked_target": {
                "lock_frame_id": 10,
                "attempt_id": "attempt-a",
                "locked_bottle_geometry": {
                    "bbox_xyxy": [10, 10, 50, 110],
                    "center_px": [30, 60],
                    "depth_median_mm": 500,
                },
            },
        },
        "bottle_geometry": {
            "bbox_xyxy": [80, 20, 180, 220],
            "center_px": [130, 120],
            "depth_median_mm": 180,
        },
    }

    status = realtime_runner.hull_staleness_status(report)

    assert status["matrix_stale"] is False
    assert status["matrix_stale_due_to_motion"] is False
    assert status["matrix_stale_reasons"] == []
    assert status["live_observation_changed"] is True
    assert "center_shift_from_analytic_baseline" in status["live_observation_reasons"]
    assert "depth_changed_from_analytic_baseline" in status["live_observation_reasons"]


def test_quality_lineage_change_still_invalidates_analytic_baseline():
    report = {
        "writer_session_id": "writer-new",
        "frame_id": 20,
        "target_lock": {
            "lock_frame_id": 10,
            "attempt_id": "attempt-new",
            "locked_target": {"lock_frame_id": 10, "attempt_id": "attempt-new"},
        },
        "quality_matrix": {
            "writer_session_id": "writer-old",
            "lock_frame_id": 9,
            "lock_attempt_id": "attempt-old",
        },
    }

    status = realtime_runner.hull_staleness_status(report)

    assert status["matrix_stale"] is True
    assert status["matrix_stale_due_to_invalidation"] is True
    assert "quality_writer_session_changed" in status["matrix_stale_reasons"]
    assert "quality_lock_frame_changed" in status["matrix_stale_reasons"]
    assert "quality_lock_attempt_changed" in status["matrix_stale_reasons"]


def test_auto_baseline_retries_without_marking_failed_schedule_and_keeps_lock_snapshot(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    calls = []
    outcomes = iter([False, True])
    runner = SimpleNamespace(
        last_auto_hull_lock_frame_id=None,
        request_quality_matrix=lambda report, **kwargs: (
            calls.append((report, kwargs)) or next(outcomes)
        ),
    )
    lock_report = {
        "frame_id": 41,
        "_quality_snapshot_frame_id": 41,
        "_quality_rgb_snapshot": np.full((2, 2, 3), 41, dtype=np.uint8),
        "_quality_depth_snapshot": np.full((2, 2), 410, dtype=np.uint16),
        "_quality_intrinsics_snapshot": {"fx": 100.0},
        "target_lock": {"lock_frame_id": 41, "target_status": "LOCKED"},
    }
    later_report = {
        "frame_id": 42,
        "target_lock": {"lock_frame_id": 41, "target_status": "TRACKING"},
    }

    assert realtime_runner.schedule_auto_lock_baseline(runner, lock_report) is False
    assert runner.last_auto_hull_lock_frame_id is None
    assert runner.pending_auto_quality_report is lock_report
    clock["now"] += realtime_runner.AUTO_QUALITY_BASELINE_RETRY_SEC + 0.01
    assert realtime_runner.schedule_auto_lock_baseline(runner, later_report) is True
    assert runner.last_auto_hull_lock_frame_id == 41
    assert calls[1][0] is not lock_report
    assert calls[1][0]["_quality_force_planned_contact"] is True
    assert calls[1][0]["_quality_rgb_snapshot"] is lock_report["_quality_rgb_snapshot"]
    assert calls[1][1]["require_actual_contact"] is False


def test_auto_baseline_survives_more_than_four_seconds_of_transient_deferrals(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    calls = []
    transient_reasons = [
        "quality_worker_busy",
        "quality_matrix_auto_throttled",
        "locked_patch_not_visible",
    ]
    runner = SimpleNamespace(last_auto_hull_lock_frame_id=None)

    def request(report, **kwargs):
        calls.append((report, kwargs))
        if len(calls) <= 9:
            reason = transient_reasons[(len(calls) - 1) % len(transient_reasons)]
            runner.last_quality_matrix_request_status = (
                realtime_runner.quality_matrix_request_status(
                    worker_busy=reason == "quality_worker_busy",
                    target_valid=False,
                    started=False,
                    reason=reason,
                    reasons=[reason],
                )
            )
            return False
        return True

    runner.request_quality_matrix = request
    lock_report = {
        "frame_id": 41,
        "_quality_snapshot_frame_id": 41,
        "_quality_rgb_snapshot": np.full((2, 2, 3), 41, dtype=np.uint8),
        "_quality_depth_snapshot": np.full((2, 2), 410, dtype=np.uint16),
        "_quality_intrinsics_snapshot": {"fx": 100.0},
        "target_lock": {"lock_frame_id": 41, "target_status": "LOCKED"},
    }
    later_report = {
        "frame_id": 50,
        "target_lock": {"lock_frame_id": 41, "target_status": "TRACKING"},
    }

    for attempt in range(9):
        report = lock_report if attempt == 0 else later_report
        assert realtime_runner.schedule_auto_lock_baseline(runner, report) is False
        assert runner.last_auto_hull_lock_frame_id is None
        clock["now"] += realtime_runner.AUTO_QUALITY_BASELINE_RETRY_SEC + 0.01

    assert clock["now"] - 10.0 > 4.0
    assert len(calls) > realtime_runner.AUTO_QUALITY_BASELINE_MAX_RETRIES
    assert runner.auto_quality_retry_count == 0
    assert realtime_runner.schedule_auto_lock_baseline(runner, later_report) is True
    assert runner.last_auto_hull_lock_frame_id == 41
    assert calls[-1][0]["_quality_force_planned_contact"] is True
    assert calls[-1][0]["_quality_rgb_snapshot"] is lock_report["_quality_rgb_snapshot"]


def test_fast_mode_retains_rgbd_snapshot_only_for_current_lock_frame(monkeypatch):
    monkeypatch.setattr(realtime_runner, "FAST_FINAL_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "FAST_REPORT_SNAPSHOTS", False)
    lock_report = {
        "frame_id": 41,
        "state": "TARGET_LOCKED",
        "target_lock": {
            "lock_frame_id": 41,
            "grasp_band_locked": True,
        },
    }

    assert realtime_runner.quality_report_snapshot_required(lock_report) is True
    assert realtime_runner.quality_report_snapshot_required({
        **lock_report,
        "frame_id": 42,
    }) is False


def test_failed_auto_baseline_result_reopens_bounded_retry(monkeypatch):
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: 20.0)
    runner = SimpleNamespace(
        auto_quality_baseline_in_flight=True,
        last_auto_hull_lock_frame_id=41,
        auto_quality_retry_count=0,
        pending_auto_quality_report={"frame_id": 41},
        pending_auto_quality_lock_frame_id=41,
    )

    realtime_runner.update_auto_lock_baseline_result(
        runner,
        {"quality_matrix_status": "failed"},
    )

    assert runner.last_auto_hull_lock_frame_id is None
    assert runner.pending_auto_quality_report == {"frame_id": 41}
    assert runner.auto_quality_retry_count == 1
    assert runner.auto_quality_next_retry_time == pytest.approx(
        20.0 + realtime_runner.AUTO_QUALITY_BASELINE_RETRY_SEC
    )


def test_auto_baseline_result_requires_exact_lineage_and_role(monkeypatch):
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: 20.0)
    expected = ("writer-1", "attempt-41", 41)
    runner = SimpleNamespace(
        auto_quality_baseline_in_flight=True,
        last_auto_hull_lock_frame_id=41,
        last_auto_quality_lineage=expected,
        auto_quality_retry_count=0,
        pending_auto_quality_report={"frame_id": 41},
        pending_auto_quality_lock_frame_id=41,
        pending_auto_quality_lineage=expected,
    )

    realtime_runner.update_auto_lock_baseline_result(
        runner,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-other",
            "lock_attempt_id": "attempt-41",
            "lock_frame_id": 41,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
            "quality_compute_id": "quality-41",
            "quality_label": "GOOD",
            "raw_epsilon": 0.126,
            "normalized_quality_score": 100.0,
            "epsilon_force_closure": True,
        },
    )

    assert runner.pending_auto_quality_report == {"frame_id": 41}
    assert runner.pending_auto_quality_lineage == expected
    assert runner.last_auto_quality_lineage is None
    assert runner.auto_quality_retry_count == 1

    runner.auto_quality_baseline_in_flight = True
    runner.last_auto_hull_lock_frame_id = 41
    runner.last_auto_quality_lineage = expected
    realtime_runner.update_auto_lock_baseline_result(
        runner,
        {
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-1",
            "lock_attempt_id": "attempt-41",
            "lock_frame_id": 41,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
            "quality_compute_id": "quality-41",
            "quality_label": "GOOD",
            "raw_epsilon": 0.126,
            "normalized_quality_score": 100.0,
            "epsilon_force_closure": True,
        },
    )

    assert runner.pending_auto_quality_report is None
    assert runner.pending_auto_quality_lineage is None
    assert runner.auto_quality_retry_count == 0


def _post_grasp_identity_fixture(identity_ok, occlusion_hold=False):
    target_clamp = {
        "clamp_center_px": [160.0, 120.0],
        "left_clamp_patch": {
            "polygon": [[130.0, 105.0], [142.0, 105.0], [142.0, 135.0], [130.0, 135.0]],
        },
        "right_clamp_patch": {
            "polygon": [[178.0, 105.0], [190.0, 105.0], [190.0, 135.0], [178.0, 135.0]],
        },
        "clamp_band_polygon": [[130.0, 105.0], [190.0, 105.0], [190.0, 135.0], [130.0, 135.0]],
    }
    manager = TargetLockManager(enabled=True)
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-original",
        "lock_frame_id": 11,
        "locked_body_fraction": 0.52,
    }
    tracking = {
        "identity_tracking_ok": bool(identity_ok),
        "identity_measurement_current": bool(identity_ok),
        "identity_occlusion_hold_active": bool(occlusion_hold),
        "contact_patch_tracking_ok": True,
        "tracking_ok": True,
        "target_clamp": target_clamp,
    }
    manager.current_tracked_target = tracking
    baseline = {
        "center_px": [160.0, 120.0],
        "axis_angle_degrees": 90.0,
        "depth_mm": 500.0,
        "target_clamp": target_clamp,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-original",
        "lock_frame_id": 11,
    }
    tracked_report = {
        "bottle_geometry": {
            "center_px": [160.0, 120.0],
            "axis_angle_degrees": 90.0,
            "depth_median_mm": 500.0,
        },
        "target_lock": {
            "current_tracked_target": tracking,
        },
    }
    return manager, baseline, tracked_report


def test_post_grasp_occlusion_keeps_original_lock_but_never_becomes_ready(monkeypatch):
    manager, baseline, tracked_report = _post_grasp_identity_fixture(False, occlusion_hold=True)
    monkeypatch.setattr(realtime_runner, "fast_track_frame", lambda *_args, **_kwargs: tracked_report)

    report, stable_frames, _pose, _depth_context = realtime_runner.evaluate_post_grasp_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 500, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        12,
        SimpleNamespace(),
        manager,
        baseline,
        stable_frames=4,
        latest_report={},
    )

    assert report["state"] == "POST_GRASP_EVALUATION"
    assert report["recommended_action"] == "HOLD_STEADY_REACQUIRE_ORIGINAL_TARGET"
    assert report["ready_to_lift"] is False
    assert report["post_grasp_identity_tracking_ok"] is False
    assert report["post_grasp_identity_occlusion_hold"] is True
    assert "post_grasp_identity_uncertain" in report["ready_blockers"]
    assert stable_frames == 0
    assert manager.locked_target["bottle_target_id"] == "bottle_01"
    assert manager.locked_target["locked_body_fraction"] == 0.52


def test_post_grasp_ready_counter_requires_same_current_locked_identity(monkeypatch):
    manager, baseline, tracked_report = _post_grasp_identity_fixture(True)
    monkeypatch.setattr(realtime_runner, "fast_track_frame", lambda *_args, **_kwargs: tracked_report)

    report, stable_frames, _pose, _depth_context = realtime_runner.evaluate_post_grasp_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 500, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        12,
        SimpleNamespace(),
        manager,
        baseline,
        stable_frames=4,
        latest_report={},
    )

    assert report["state"] == "READY_TO_LIFT"
    assert report["ready_to_lift"] is True
    assert report["post_grasp_identity_tracking_ok"] is True
    assert stable_frames == 5


def test_post_grasp_recent_identity_bridge_never_counts_toward_ready(monkeypatch):
    manager, baseline, tracked_report = _post_grasp_identity_fixture(True)
    tracking = tracked_report["target_lock"]["current_tracked_target"]
    tracking["identity_measurement_current"] = False
    tracking["identity_measurement_age_frames"] = 2
    tracking["identity_tracking_source"] = "current_locked_contact_patch_recent_identity_bridge"
    monkeypatch.setattr(realtime_runner, "fast_track_frame", lambda *_args, **_kwargs: tracked_report)

    report, stable_frames, _pose, _depth_context = realtime_runner.evaluate_post_grasp_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 500, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        12,
        SimpleNamespace(),
        manager,
        baseline,
        stable_frames=4,
        latest_report={},
    )

    assert report["state"] == "POST_GRASP_EVALUATION"
    assert report["recommended_action"] == "HOLD_STEADY_REACQUIRE_ORIGINAL_TARGET"
    assert report["ready_to_lift"] is False
    assert report["post_grasp_identity_tracking_ok"] is False
    assert report["post_grasp_identity_measurement_current"] is False
    assert report["post_grasp_identity_occlusion_hold"] is True
    assert stable_frames == 0


def test_post_grasp_identity_metadata_change_forces_do_not_lift(monkeypatch):
    manager, baseline, tracked_report = _post_grasp_identity_fixture(True, occlusion_hold=True)
    manager.locked_target["attempt_id"] = "attempt-different"
    monkeypatch.setattr(realtime_runner, "fast_track_frame", lambda *_args, **_kwargs: tracked_report)

    report, stable_frames, _pose, _depth_context = realtime_runner.evaluate_post_grasp_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 500, dtype=np.uint16),
        {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
        12,
        SimpleNamespace(),
        manager,
        baseline,
        stable_frames=4,
        latest_report={},
    )

    assert report["state"] == "DO_NOT_LIFT"
    assert report["ready_to_lift"] is False
    assert report["post_grasp_identity_metadata_matches_baseline"] is False
    assert report["post_grasp_identity_occlusion_hold"] is False
    assert "post_grasp_identity_mismatch" in report["ready_blockers"]
    assert stable_frames == 0


def test_temporal_lock_stability_resets_when_visible_candidate_switches():
    manager = TargetLockManager(enabled=True)
    first = {
        "target_id": "bottle_01",
        "center_px": [80.0, 120.0],
        "bbox_xywh": [58.0, 55.0, 44.0, 130.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
    }
    other = {
        "target_id": "bottle_02",
        "center_px": [124.0, 120.0],
        "bbox_xywh": [102.0, 55.0, 44.0, 130.0],
        "depth_median_mm": 500.0,
        "axis_vector": [0.0, 1.0],
    }

    manager.update_stability(True, pose=first)
    manager.update_stability(True, pose=first)
    assert manager.stable_full_bottle_frames == 2
    manager.update_stability(True, pose=other)
    assert manager.stable_full_bottle_frames == 0

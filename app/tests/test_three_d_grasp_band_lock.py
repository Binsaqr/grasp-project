import copy

import numpy as np
import pytest

from grasp_project.realsense import realtime_runner
from grasp_project.realsense.grasp_band_3d import deproject_mask_mm
from grasp_project.realsense.realtime_runner import (
    LockedGraspBand3D,
    QualityMatrixWorker,
    TargetLockManager,
    create_locked_grasp_band_3d,
    current_lock_frame_snapshot_provenance,
    grasp_target_3d_runtime_diagnostics,
    grasp_band_overlay_label,
    locked_grasp_band_status,
    project_locked_grasp_band_to_current_frame,
    ready_blockers_from_values,
    select_locked_band_projection,
    select_true_3d_dropout_display,
    side_clamp_quality_inputs_from_snapshot,
    target_lock_report,
    write_invalid_locked_grasp_band_artifacts,
)


INTRINSICS = {"fx": 200.0, "fy": 200.0, "cx": 60.0, "cy": 60.0, "source": "synthetic"}


def synthetic_lock_inputs(depth_mm=400):
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[10:110, 40:80] = 255
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = depth_mm
    left = [42.0, 60.0]
    right = [77.0, 60.0]
    left_polygon = [[40, 50], [49, 50], [49, 70], [40, 70]]
    right_polygon = [[70, 50], [79, 50], [79, 70], [70, 70]]
    band_polygon = [[40, 50], [79, 50], [79, 70], [40, 70]]
    pose = {
        "detected": True,
        "target_id": "bottle-synthetic",
        "center_px": [59.5, 59.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 100.0,
        "axis_angle_degrees": 90.0,
        "bbox_xyxy": [40, 10, 80, 110],
        "bbox_xywh": [40, 10, 40, 100],
        "oriented_bbox": [[40, 10], [80, 10], [80, 110], [40, 110]],
        "cleaned_mask": mask,
        "body_mask": mask,
        "depth_median_mm": float(depth_mm),
        "detection_confidence": 0.95,
    }
    plan = {
        "grasp_center_px": [59.5, 60.0],
        "grasp_direction": [1.0, 0.0],
        "body_fraction": 0.5,
        "left_contact_px": left,
        "right_contact_px": right,
        "left_contact_3d_m": [-0.036, 0.0, depth_mm / 1000.0],
        "right_contact_3d_m": [0.034, 0.0, depth_mm / 1000.0],
        "planned_depth_mm": float(depth_mm),
        "grip_width_px": 35.0,
        "grip_width_mm": 70.0,
        "target_contact_quality": 90.0,
        "raw_grip_width_score": 90.0,
        "grip_width_score": 90.0,
        "contact_model_ready": True,
        "selected_candidate_id": "body_fraction_0.500",
        "selected_contact_pair": {"candidate_id": "body_fraction_0.500", "body_fraction": 0.5},
    }
    locked_target = {
        "valid": True,
        "attempt_id": "attempt-synthetic",
        "selected_candidate_id": "body_fraction_0.500",
        "bottle_target_id": pose["target_id"],
        "lock_frame_id": 8,
        "lock_timestamp": "2026-08-12T10:00:00Z",
        "locked_body_fraction": 0.5,
        "locked_grip_width_mm": 70.0,
        "lock_confidence": 92.0,
        "locked_quality_compute_id": "quality-8",
        "locked_quality_label": "GOOD",
        "epsilon_value_at_lock": 0.05,
        "force_closure_at_lock": True,
        "fixed_clamp_zone": {
            "clamp_center_px": [59.5, 60.0],
            "clamp_band_width_px": 35.0,
            "clamp_band_height_px": 20.0,
            "clamp_patch_height_px": 20.0,
            "left_clamp_patch": {"center_px": left, "polygon": left_polygon},
            "right_clamp_patch": {"center_px": right, "polygon": right_polygon},
            "clamp_band_polygon": band_polygon,
        },
        "bottle_contact_model": {
            "body_fraction": 0.5,
            "grasp_line": {"left_endpoint_px": left, "right_endpoint_px": right, "center_px": [59.5, 60.0]},
            "left_contact_patch": {"center_px": left, "polygon": left_polygon},
            "right_contact_patch": {"center_px": right, "polygon": right_polygon},
            "contact_band_polygon": band_polygon,
            "grip_width_px": 35.0,
            "grip_width_mm": 70.0,
            "patch_height_px": 20.0,
            "epsilon_result": {"epsilon": 0.05, "force_closure": True},
        },
        "locked_bottle_geometry": {
            "image_bottle_hull_px": pose["oriented_bbox"],
        },
    }
    return pose, plan, depth, locked_target


def test_locked_band_stores_body_fraction_quality_and_3d_geometry():
    pose, plan, depth, target = synthetic_lock_inputs()

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    payload = band.to_dict()

    assert isinstance(band, LockedGraspBand3D)
    assert payload["lock_valid"] is True
    assert payload["locked_body_fraction_center"] == 0.5
    assert payload["locked_body_fraction_height"] == pytest.approx(0.2)
    assert payload["locked_left_contact_xyz_mm"][2] == pytest.approx(400.0)
    assert payload["locked_right_contact_xyz_mm"][2] == pytest.approx(400.0)
    assert payload["locked_band_centroid_xyz_mm"][2] == pytest.approx(400.0)
    assert len(payload["lock_bottle_points_xyz"]) == 4000
    assert len(payload["lock_band_points_xyz"]) > 100
    assert len(payload["locked_band_points_xyz_mm"]) == len(payload["lock_band_points_xyz"])
    assert len(payload["locked_band_boundary_local_xyz"]) >= 4
    assert payload["locked_band_local_xyz"] == payload["locked_band_boundary_local_xyz"]
    assert len(payload["locked_band_center_local_xyz"]) == 3
    assert len(payload["locked_left_contact_local_xyz"]) == 3
    assert len(payload["locked_right_contact_local_xyz"]) == 3
    assert payload["bottle_local_origin_xyz"][2] == pytest.approx(400.0)
    assert payload["locked_bottle_origin_xyz"] == payload["bottle_local_origin_xyz"]
    assert abs(payload["bottle_local_axis_unit"][1]) > 0.9
    assert abs(payload["bottle_local_side_unit"][0]) > 0.9
    assert payload["bottle_frame_confidence"] >= realtime_runner.MIN_BOTTLE_FRAME_CONFIDENCE
    assert payload["representation_type"] == "BOTTLE_LOCAL_3D_POINT_CLOUD_LOCK"
    assert payload["true_3d_lock"] is True
    assert payload["locked_projection_mode"] == "TRUE_3D"
    assert payload["locked_body_fraction"] == 0.5
    assert payload["projection_invalid_reason"] is None
    assert payload["two_d_fallback_used"] is False
    assert payload["locked_band_original_px"] == target["fixed_clamp_zone"]["clamp_band_polygon"]
    assert payload["locked_quality_compute_id"] == "quality-8"
    assert payload["physically_validated"] is False


def test_full_bottle_lock_rejects_old_surface_crop_that_was_actually_hand_fixture():
    pose, plan, depth, target = synthetic_lock_inputs(depth_mm=180)
    target["full_bottle_lock"] = True

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    assert 80.0 <= band.bottle_length_mm <= 100.0
    assert band.bottle_length_mm < realtime_runner.FULL_BOTTLE_LOCK_MIN_LENGTH_MM
    assert band.lock_valid is False
    assert band.lock_invalid_reason == "full_bottle_metric_length_below_minimum"


def test_full_bottle_lock_rejects_live_black_fixture_metric_even_if_centered(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs(depth_mm=325)
    target["full_bottle_lock"] = True
    original = realtime_runner.fit_bottle_local_frame

    def live_fixture_frame(points, *args, **kwargs):
        frame = original(points, *args, **kwargs)
        frame["bottle_length_mm"] = 109.92
        frame["bottle_width_mm"] = 44.54
        return frame

    monkeypatch.setattr(realtime_runner, "fit_bottle_local_frame", live_fixture_frame)

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    assert band.bottle_length_mm == pytest.approx(109.92)
    assert band.bottle_width_mm == pytest.approx(44.54)
    assert band.lock_valid is False
    assert band.lock_invalid_reason == "full_bottle_metric_length_below_minimum"


def test_strict_known_blue_real_bottle_uses_profile_tolerance_not_generic_fixture_floor(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs(depth_mm=280)
    target["full_bottle_lock"] = True
    pose.update(
        {
            "measurement_status": "CURRENT_MEASURED_RGBD",
            "bottle_detection_status": "CANDIDATE",
            "depth_confidence": 0.997,
            "bottle_profile_valid": True,
            "bottle_profile_agreement_score": 0.89,
            "segmentation_rejection_reasons": [],
            "segmentation_source": "rgb_known_blue",
            "bottle_color_fraction": 0.95,
        }
    )
    original = realtime_runner.fit_bottle_local_frame

    def physical_bottle_frame(points, *args, **kwargs):
        frame = original(points, *args, **kwargs)
        # Exact local-PCA dimensions captured in the failed 30-second test.
        frame["bottle_length_mm"] = 120.916
        frame["bottle_width_mm"] = 42.268
        return frame

    monkeypatch.setattr(realtime_runner, "fit_bottle_local_frame", physical_bottle_frame)

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    assert realtime_runner.demo_bottle_identity_eligibility(pose)[0] is True
    assert band.bottle_length_mm < realtime_runner.FULL_BOTTLE_LOCK_MIN_LENGTH_MM
    assert band.bottle_length_mm >= realtime_runner.KNOWN_BLUE_FULL_BOTTLE_LOCK_MIN_LENGTH_MM
    assert band.bottle_length_mm / band.bottle_width_mm > realtime_runner.FULL_BOTTLE_LOCK_MIN_LOCAL_ASPECT
    assert band.lock_valid is True
    assert band.lock_invalid_reason is None


def test_profile_tolerance_cannot_be_used_by_nonblue_fixture(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs(depth_mm=280)
    target["full_bottle_lock"] = True
    pose.update(
        {
            "measurement_status": "CURRENT_MEASURED_RGBD",
            "bottle_detection_status": "CANDIDATE",
            "depth_confidence": 0.997,
            "bottle_profile_valid": True,
            "bottle_profile_agreement_score": 0.89,
            "segmentation_rejection_reasons": [],
            "segmentation_source": "rgb_generic",
            "bottle_color_fraction": 0.20,
        }
    )
    original = realtime_runner.fit_bottle_local_frame

    def fixture_frame(points, *args, **kwargs):
        frame = original(points, *args, **kwargs)
        frame["bottle_length_mm"] = 120.916
        frame["bottle_width_mm"] = 42.268
        return frame

    monkeypatch.setattr(realtime_runner, "fit_bottle_local_frame", fixture_frame)

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    assert realtime_runner.demo_bottle_identity_eligibility(pose)[0] is False
    assert band.lock_valid is False
    assert band.lock_invalid_reason == "full_bottle_metric_length_below_minimum"


@pytest.mark.parametrize(
    ("depth", "intrinsics", "reason"),
    [(None, INTRINSICS, "missing_depth"), (np.ones((120, 120), dtype=np.uint16), None, "missing_intrinsics")],
)
def test_locked_band_rejects_missing_depth_or_intrinsics(depth, intrinsics, reason):
    pose, plan, _, target = synthetic_lock_inputs()

    band = create_locked_grasp_band_3d(target, pose, plan, depth, intrinsics)

    assert band.lock_valid is False
    assert band.lock_invalid_reason == reason


def test_full_bottle_lock_rejects_small_metric_fixture_fragment():
    pose, plan, depth, target = synthetic_lock_inputs()
    short_mask = np.zeros_like(depth)
    short_mask[40:80, 40:80] = 255
    depth[:] = 0
    depth[short_mask > 0] = 250
    pose.update({
        "center_px": [59.5, 59.5],
        "axis_length_px": 40.0,
        "bbox_xyxy": [40, 40, 80, 80],
        "bbox_xywh": [40, 40, 40, 40],
        "oriented_bbox": [[40, 40], [80, 40], [80, 80], [40, 80]],
        "cleaned_mask": short_mask,
        "body_mask": short_mask,
        "depth_median_mm": 250.0,
    })
    target["full_bottle_lock"] = True

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    assert band.bottle_length_mm < realtime_runner.FULL_BOTTLE_LOCK_MIN_LENGTH_MM
    assert band.lock_valid is False
    assert band.lock_invalid_reason == "full_bottle_metric_length_below_minimum"


def test_full_bottle_lock_rejects_squat_local_3d_fragment(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs(depth_mm=180)
    target["full_bottle_lock"] = True
    original = realtime_runner.fit_bottle_local_frame

    def squat_frame(points, *args, **kwargs):
        frame = original(points, *args, **kwargs)
        frame["bottle_length_mm"] = 130.0
        frame["bottle_width_mm"] = 90.0
        return frame

    monkeypatch.setattr(realtime_runner, "fit_bottle_local_frame", squat_frame)

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    assert band.bottle_length_mm >= realtime_runner.FULL_BOTTLE_LOCK_MIN_LENGTH_MM
    assert band.bottle_length_mm / band.bottle_width_mm < (
        realtime_runner.FULL_BOTTLE_LOCK_MIN_LOCAL_ASPECT
    )
    assert band.lock_valid is False
    assert band.lock_invalid_reason == "full_bottle_metric_aspect_below_minimum"


def test_target_manager_does_not_create_pixel_only_lock(monkeypatch):
    pose, plan, _, _ = synthetic_lock_inputs()
    manager = TargetLockManager()
    monkeypatch.setattr(realtime_runner, "write_json", lambda *_args, **_kwargs: None)

    result = manager.lock_from_plan(
        plan,
        pose,
        {"epsilon": 0.05, "force_closure": True},
        {"final_score": 90.0, "decision": "READY TO CLOSE"},
        8,
        depth=None,
        intrinsics=INTRINSICS,
        image_shape=(120, 120, 3),
    )

    assert result is None
    assert manager.locked_target is None
    assert manager.last_lock_failure_reason == "missing_depth"


def test_target_manager_preserves_originating_lock_command(monkeypatch):
    pose, plan, depth, _ = synthetic_lock_inputs()
    manager = TargetLockManager()
    monkeypatch.setattr(realtime_runner, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "log_grasp_attempt", lambda *_args, **_kwargs: None)

    result = manager.lock_from_plan(
        plan,
        pose,
        {"epsilon": 0.05, "force_closure": True},
        {"final_score": 90.0, "decision": "READY TO CLOSE"},
        8,
        depth=depth,
        intrinsics=INTRINSICS,
        image_shape=(120, 120, 3),
        lock_command_id="fresh-lock-command-1",
    )

    assert result["lock_command_id"] == "fresh-lock-command-1"
    assert manager.grasp_memory["lock_command_id"] == "fresh-lock-command-1"
    assert target_lock_report(manager)["lock_command_id"] == "fresh-lock-command-1"


def test_full_xyz_exists_only_in_canonical_artifact_not_live_target(monkeypatch, tmp_path):
    pose, plan, depth, _ = synthetic_lock_inputs()
    manager = TargetLockManager()
    lock_path = tmp_path / "locked_grasp_band.json"
    debug_path = tmp_path / "locked_grasp_band_debug.json"
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", lock_path)
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", debug_path)
    monkeypatch.setattr(realtime_runner, "append_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime_runner, "log_grasp_attempt", lambda *_args, **_kwargs: None)

    result = manager.lock_from_plan(
        plan,
        pose,
        {"epsilon": 0.05, "force_closure": True},
        {"final_score": 90.0, "decision": "READY TO CLOSE"},
        8,
        depth=depth,
        intrinsics=INTRINSICS,
        image_shape=(120, 120, 3),
    )

    canonical = realtime_runner.json.loads(lock_path.read_text())
    assert len(canonical["lock_bottle_points_xyz"]) == 4000
    assert len(canonical["lock_band_points_xyz"]) > 100
    assert "lock_bottle_points_xyz" not in result["locked_grasp_band_3d"]
    assert "lock_band_points_xyz" not in result["locked_grasp_band_3d"]
    live = target_lock_report(manager)["locked_target"]
    assert "lock_bottle_points_xyz" not in live["locked_grasp_band_3d"]
    assert live["locked_grasp_band_3d"]["lock_bottle_point_count"] == 4000


def test_pose_dropout_freezes_last_valid_3d_and_nonsticky_mode_then_hides(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    target["locked_grasp_band"] = realtime_runner.compact_locked_grasp_band_snapshot(
        band.to_dict()
    )
    last_valid = copy.deepcopy(target["fixed_clamp_zone"])
    last_valid["clamp_center_px"] = [30.0, 40.0]
    last_valid["left_contact_px"] = [20.0, 40.0]
    last_valid["right_contact_px"] = [40.0, 40.0]
    temporal = {
        "last_valid_true_3d_clamp": last_valid,
        "last_valid_true_3d_monotonic": 100.0,
    }
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: 100.1)
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", False)

    estimated = select_true_3d_dropout_display(
        target,
        {
            "target_clamp": copy.deepcopy(target["fixed_clamp_zone"]),
            "local_motion_anchor_offset_px": [5.0, 0.0],
        },
        temporal,
    )

    assert estimated["projection_mode"] == "HELD_LAST_VALID_3D"
    assert estimated["target_clamp"]["clamp_center_px"] == [30.0, 40.0]
    assert estimated["target_clamp"]["clamp_center_px"] != target["fixed_clamp_zone"]["clamp_center_px"]
    assert estimated["projection_valid"] is False
    assert estimated["two_d_fallback_used"] is False
    assert estimated["projected_left_contact_xyz_mm"] is None

    monkeypatch.setattr(
        realtime_runner.time,
        "monotonic",
        lambda: 100.0 + realtime_runner.TRUE_3D_TEMPORAL_HOLD_SEC + 0.01,
    )
    unavailable = select_true_3d_dropout_display(target, {}, temporal)
    assert unavailable["projection_mode"] == "INVALID_3D"
    assert unavailable["target_clamp"].get("clamp_center_px") is None


def test_sticky_demo_preserves_identity_but_expires_stale_screen_projection(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    target["locked_grasp_band"] = realtime_runner.compact_locked_grasp_band_snapshot(
        band.to_dict()
    )
    last_valid = copy.deepcopy(target["fixed_clamp_zone"])
    last_valid["clamp_center_px"] = [30.0, 40.0]
    temporal = {
        "last_valid_true_3d_clamp": last_valid,
        "last_valid_true_3d_monotonic": 100.0,
    }
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", True)
    monkeypatch.setattr(
        realtime_runner.time,
        "monotonic",
        lambda: 100.0 + realtime_runner.TRUE_3D_TEMPORAL_HOLD_SEC + 30.0,
    )

    held = select_true_3d_dropout_display(
        target,
        {"local_motion_anchor_offset_px": [12.0, -12.0]},
        temporal,
    )

    assert held["projection_mode"] == "INVALID_3D"
    assert held["target_clamp"].get("clamp_center_px") is None
    assert held["projection_valid"] is False
    assert held["two_d_fallback_used"] is False


def test_translation_reprojects_same_body_fraction_in_current_frame():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    translated_pose = copy.deepcopy(pose)
    translated_mask = np.zeros_like(pose["cleaned_mask"])
    translated_mask[10:110, 50:90] = 255
    translated_pose.update({
        "center_px": [69.5, 59.5],
        "bbox_xyxy": [50, 10, 90, 110],
        "bbox_xywh": [50, 10, 40, 100],
        "cleaned_mask": translated_mask,
        "body_mask": translated_mask,
    })
    translated_depth = np.zeros_like(depth)
    translated_depth[translated_mask > 0] = 400

    projected = project_locked_grasp_band_to_current_frame(
        band, translated_pose, translated_depth, INTRINSICS
    )

    assert projected["projected_band_valid"] is True
    assert projected["projection_source"] == "BOTTLE_LOCAL_3D_REPROJECTION"
    assert projected["two_d_fallback_used"] is False
    assert projected["projected_body_fraction"] == 0.5
    assert projected["projected_left_contact_px"][0] > band.locked_left_contact_px[0] + 5
    assert projected["band_reselected_this_frame"] is False
    assert projected["band_reselection_blocked_reason"] == "locked_bottle_local_3d_geometry_is_immutable"


def test_vertical_camera_motion_reprojects_target_with_bottle_not_screen():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    moved_pose = copy.deepcopy(pose)
    moved_mask = np.zeros_like(depth, dtype=np.uint8)
    moved_mask[18:118, 40:80] = 255
    moved_pose.update({
        "center_px": [59.5, 67.5],
        "bbox_xyxy": [40, 18, 80, 118],
        "bbox_xywh": [40, 18, 40, 100],
        "cleaned_mask": moved_mask,
        "body_mask": moved_mask,
    })
    moved_depth = np.zeros_like(depth)
    moved_depth[moved_mask > 0] = 400

    projected = project_locked_grasp_band_to_current_frame(
        band, moved_pose, moved_depth, INTRINSICS
    )

    assert projected["projection_valid"] is True
    assert projected["projected_band_center_px"][1] > band.locked_band_center_projection_at_lock_px[1] + 5
    assert projected["projected_body_fraction"] == band.locked_body_fraction
    assert projected["band_reselected_this_frame"] is False


def test_scale_and_approach_depth_reproject_without_changing_fraction():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    current_pose = copy.deepcopy(pose)
    current_mask = np.zeros_like(pose["cleaned_mask"])
    current_mask[5:115, 30:90] = 255
    current_pose.update({
        "center_px": [59.5, 59.5],
        "axis_length_px": 110.0,
        "bbox_xyxy": [30, 5, 90, 115],
        "bbox_xywh": [30, 5, 60, 110],
        "cleaned_mask": current_mask,
        "body_mask": current_mask,
        "depth_median_mm": 200.0,
    })
    current_depth = np.zeros_like(depth)
    current_depth[current_mask > 0] = 200

    projected = project_locked_grasp_band_to_current_frame(
        band, current_pose, current_depth, INTRINSICS
    )

    assert projected["projected_band_valid"] is True
    assert projected["projected_body_fraction"] == 0.5
    assert projected["projected_depth_mm"] == pytest.approx(200.0)
    assert projected["projected_contact_width_mm"] > 50.0


def test_depth_change_changes_pixel_scale_but_preserves_locked_local_geometry():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    lock_projection = project_locked_grasp_band_to_current_frame(band, pose, depth, INTRINSICS)
    current_pose = copy.deepcopy(pose)
    current_mask = np.zeros_like(depth, dtype=np.uint8)
    current_mask[5:115, 30:90] = 255
    current_pose.update({
        "cleaned_mask": current_mask,
        "body_mask": current_mask,
        "bbox_xyxy": [30, 5, 90, 115],
        "bbox_xywh": [30, 5, 60, 110],
        "axis_length_px": 110.0,
        "depth_median_mm": 200.0,
    })
    current_depth = np.zeros_like(depth)
    current_depth[current_mask > 0] = 200
    close_projection = project_locked_grasp_band_to_current_frame(
        band, current_pose, current_depth, INTRINSICS
    )

    lock_width_px = abs(lock_projection["projected_right_contact_px"][0] - lock_projection["projected_left_contact_px"][0])
    close_width_px = abs(close_projection["projected_right_contact_px"][0] - close_projection["projected_left_contact_px"][0])
    assert close_projection["projection_valid"] is True
    assert close_width_px > lock_width_px * 1.4
    assert close_projection["projected_contact_width_mm"] == pytest.approx(
        lock_projection["projected_contact_width_mm"], rel=0.10
    )


def test_raw_pixel_fields_cannot_create_a_true_3d_projection():
    pose, _, depth, _ = synthetic_lock_inputs()
    fake = {
        "lock_valid": True,
        "true_3d_lock": True,
        "locked_left_contact_px": [42, 60],
        "locked_right_contact_px": [77, 60],
        "locked_band_polygon_px": [[40, 50], [79, 50], [79, 70], [40, 70]],
    }

    projected = project_locked_grasp_band_to_current_frame(fake, pose, depth, INTRINSICS)

    assert projected["projection_valid"] is False
    assert projected["projection_invalid_reason"] == "missing_locked_local_3d_geometry"
    assert projected["two_d_fallback_used"] is False


def test_low_current_3d_frame_confidence_invalidates_projection():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    sparse_pose = copy.deepcopy(pose)
    sparse_mask = np.zeros_like(depth, dtype=np.uint8)
    sparse_mask[55:59, 55:59] = 255
    sparse_pose["cleaned_mask"] = sparse_mask
    sparse_pose["body_mask"] = sparse_mask
    sparse_depth = np.zeros_like(depth)
    sparse_depth[sparse_mask > 0] = 400

    projected = project_locked_grasp_band_to_current_frame(band, sparse_pose, sparse_depth, INTRINSICS)

    assert projected["projection_valid"] is False
    assert projected["current_frame_confidence"] < realtime_runner.MIN_BOTTLE_FRAME_CONFIDENCE
    assert projected["projection_invalid_reason"] == "too_few_3d_points"


def test_current_projection_does_not_call_2d_candidate_reselection(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("2D candidate projection/reselection must not run")

    monkeypatch.setattr(realtime_runner, "build_locked_projection_plan", fail_if_called)
    projected = project_locked_grasp_band_to_current_frame(band, pose, depth, INTRINSICS)

    assert projected["projection_valid"] is True
    assert projected["band_reselected_this_frame"] is False


def test_overlay_labels_true_3d_current_stale_and_legacy_bands_honestly():
    assert grasp_band_overlay_label(true_3d_lock=True, projection_valid=True, locked=True) == "Locked Bottle Grasp Band"
    assert grasp_band_overlay_label(true_3d_lock=True, projection_valid=False, locked=True) == "Locked Bottle Grasp Band (stale)"
    assert grasp_band_overlay_label(true_3d_lock=False, projection_valid=False, locked=True) == "2D_FALLBACK Bottle Band"
    assert grasp_band_overlay_label(true_3d_lock=False, projection_valid=False, locked=False) == "Planned Bottle Grasp Band"
    assert grasp_band_overlay_label(
        true_3d_lock=True,
        projection_valid=False,
        locked=True,
        projection_mode="2D_FALLBACK",
    ) == "2D_FALLBACK Bottle Band"
    assert grasp_band_overlay_label(
        true_3d_lock=True,
        projection_valid=False,
        locked=True,
        projection_mode="HELD_LAST_VALID_3D",
    ) == "Locked Bottle Grasp Band"
    assert grasp_band_overlay_label(
        true_3d_lock=True,
        projection_valid=False,
        locked=True,
        projection_mode="ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
    ) == "Locked Bottle Grasp Band"
    assert grasp_band_overlay_label(
        true_3d_lock=True,
        projection_valid=False,
        locked=True,
        projection_mode="INVALID_3D",
    ) == "Locked Bottle Grasp Band (unavailable)"


def test_missing_current_3d_never_substitutes_contour_band_and_blocks_ready():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    target["locked_grasp_band"] = band.to_dict()
    fallback = copy.deepcopy(target["fixed_clamp_zone"])

    clamp, model, projection, _trace = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        fallback,
        {"band_source": "current_2d_contour", "edge_source": "current_2d_contour"},
        geometry_measurement_current=False,
    )

    assert model is None
    assert projection["projection_valid"] is False
    assert projection["locked_projection_mode"] == "INVALID_3D"
    assert projection["two_d_fallback_used"] is False
    assert projection["projection_confidence"] == 0.0
    assert clamp["projection_mode"] == "INVALID_3D"
    assert clamp.get("left_contact_px") is None
    blockers = ready_blockers_from_values(
        plan,
        {
            "locked_projection_mode": "INVALID_3D",
            "identity_measurement_current": True,
            "locked_patch_visible": True,
            "tracking_confidence": 90.0,
            "patch_depth_confidence": 90.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
        },
        {"force_closure": True},
        90.0,
        [],
    )
    assert "true_3d_projection_unavailable" in blockers


def test_short_failure_holds_only_last_valid_3d_then_expires(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    target["locked_grasp_band"] = band.to_dict()
    temporal = {}
    fallback = copy.deepcopy(target["fixed_clamp_zone"])
    clock = {"now": 100.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", False)

    valid_clamp, _, valid_projection, _ = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        fallback,
        {},
        geometry_measurement_current=True,
        temporal_state=temporal,
    )
    assert valid_projection["locked_projection_mode"] == "TRUE_3D"
    held_clamp, _, held_projection, _ = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        fallback,
        {},
        geometry_measurement_current=False,
        temporal_state=temporal,
    )
    assert held_projection["locked_projection_mode"] == "HELD_LAST_VALID_3D"
    assert held_projection["projection_valid"] is False
    assert held_projection["two_d_fallback_used"] is False
    assert held_clamp["grasp_line"]["left_endpoint_px"] == valid_clamp["grasp_line"]["left_endpoint_px"]

    clock["now"] += realtime_runner.TRUE_3D_TEMPORAL_HOLD_SEC + 0.01
    expired_clamp, _, expired_projection, _ = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        fallback,
        {},
        geometry_measurement_current=False,
        temporal_state=temporal,
    )
    assert expired_projection["locked_projection_mode"] == "INVALID_3D"
    assert expired_clamp.get("left_contact_px") is None


def test_full_then_fast_dropout_keeps_same_frozen_center_mode_and_blocks_ready(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    target["locked_grasp_band"] = band.to_dict()
    temporal = {}
    clock = {"now": 100.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])

    _valid_clamp, _, valid_projection, _ = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        target["fixed_clamp_zone"],
        {},
        geometry_measurement_current=True,
        temporal_state=temporal,
    )
    assert valid_projection["projection_mode"] == "TRUE_3D"

    offset = [5.0, 2.0]
    full_dropout = select_true_3d_dropout_display(
        target,
        {"local_motion_anchor_offset_px": offset},
        temporal,
    )
    clock["now"] += 0.01
    fast_clamp, fast_model, fast_projection, fast_trace = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        target["fixed_clamp_zone"],
        {},
        geometry_measurement_current=False,
        temporal_state=temporal,
        local_patch_offset_px=offset,
    )

    assert full_dropout["projection_mode"] == "HELD_LAST_VALID_3D"
    assert fast_projection["projection_mode"] == full_dropout["projection_mode"]
    assert fast_clamp["clamp_center_px"] == full_dropout["target_clamp"]["clamp_center_px"]
    assert fast_trace["magenta_band_polygon"] == full_dropout["target_clamp"]["clamp_band_polygon"]
    assert fast_model is None
    assert fast_projection["projection_valid"] is False
    assert fast_projection["two_d_fallback_used"] is False
    blockers = ready_blockers_from_values(
        plan,
        {
            "locked_projection_mode": fast_projection["projection_mode"],
            "identity_measurement_current": True,
            "locked_patch_visible": True,
            "tracking_confidence": 90.0,
            "patch_depth_confidence": 90.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
        },
        {"force_closure": True},
        90.0,
        [],
    )
    assert "true_3d_projection_unavailable" in blockers


def test_current_bounded_partial_motion_never_moves_operator_guide(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    target["locked_grasp_band"] = create_locked_grasp_band_3d(
        target,
        pose,
        plan,
        depth,
        INTRINSICS,
    ).to_dict()
    temporal = {}
    clock = {"now": 100.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])

    valid_clamp, _, valid_projection, valid_trace = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        target["fixed_clamp_zone"],
        {},
        geometry_measurement_current=True,
        temporal_state=temporal,
    )
    assert valid_projection["projection_mode"] == "TRUE_3D"
    transformed = realtime_runner.shifted_target_clamp(valid_clamp, [8.0, -4.0])

    estimated_clamp, model, estimated_projection, estimated_trace = (
        select_locked_band_projection(
            target,
            pose,
            depth,
            INTRINSICS,
            transformed,
            valid_trace,
            geometry_measurement_current=False,
            temporal_state=temporal,
            motion_estimate_current=True,
        )
    )

    assert model is None
    assert estimated_projection["projection_mode"] == "HELD_LAST_VALID_3D"
    assert estimated_projection["projection_valid"] is False
    assert estimated_projection["two_d_fallback_used"] is False
    assert estimated_clamp["clamp_center_px"] == valid_clamp["clamp_center_px"]
    assert estimated_clamp["clamp_center_px"] != transformed["clamp_center_px"]
    assert estimated_trace["magenta_band_polygon"] == valid_clamp["clamp_band_polygon"]

    blockers = ready_blockers_from_values(
        plan,
        {
            "locked_projection_mode": estimated_projection["projection_mode"],
            "identity_measurement_current": False,
            "locked_patch_visible": True,
            "tracking_confidence": 90.0,
            "patch_depth_confidence": 90.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
        },
        {"force_closure": True},
        90.0,
        [],
    )
    assert "true_3d_projection_unavailable" in blockers
    assert "identity_measurement_not_current" in blockers


def test_full_manager_projection_seeds_next_fast_dropout_display_and_blocks_ready(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    target["locked_grasp_band"] = create_locked_grasp_band_3d(
        target,
        pose,
        plan,
        depth,
        INTRINSICS,
    ).to_dict()
    manager = TargetLockManager(enabled=True)
    manager.locked_target = target
    clock = {"now": 100.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])

    _full_plan, full_tracking = manager.track(
        pose,
        depth,
        INTRINSICS,
        {"full_bottle_visible": True},
        pose["cleaned_mask"].shape,
        frame_id=9,
    )

    assert full_tracking["projection_mode"] == "TRUE_3D"
    assert full_tracking["projection_source"] == "BOTTLE_LOCAL_3D_REPROJECTION"
    assert full_tracking["projection_valid"] is True
    assert manager.temporal_state["last_valid_true_3d_clamp"]["clamp_center_px"] == (
        full_tracking["target_clamp"]["clamp_center_px"]
    )

    clock["now"] += min(0.1, realtime_runner.TRUE_3D_TEMPORAL_HOLD_SEC * 0.5)
    held = select_true_3d_dropout_display(target, {}, manager.temporal_state)

    assert held["projection_mode"] == "HELD_LAST_VALID_3D"
    assert held["target_clamp"]["clamp_center_px"] == full_tracking["target_clamp"]["clamp_center_px"]
    assert held["projection_valid"] is False
    assert held["two_d_fallback_used"] is False
    assert held["projected_left_contact_xyz_mm"] is None
    blockers = ready_blockers_from_values(
        plan,
        {
            **held,
            "identity_measurement_current": False,
            "locked_patch_visible": True,
            "tracking_confidence": 90.0,
            "patch_depth_confidence": 90.0,
            "patch_width_valid": True,
            "patch_on_bottle_body": True,
        },
        {"force_closure": True},
        90.0,
        [],
    )
    assert "true_3d_projection_unavailable" in blockers
    assert "identity_measurement_not_current" in blockers


def test_compact_live_output_preserves_dropout_mode_without_dense_lock_points(monkeypatch):
    monkeypatch.setattr(realtime_runner, "FAST_LIVE_STATUS_COMPACT", True)
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    target["locked_grasp_band"] = realtime_runner.compact_locked_grasp_band_snapshot(
        band.to_dict()
    )
    manager = TargetLockManager(enabled=True)
    manager.locked_target = target
    manager.current_tracked_target = {
        "tracking_ok": True,
        "identity_tracking_ok": True,
        "target_clamp": target["fixed_clamp_zone"],
        "current_patch_center_px": [64.5, 61.5],
        "locked_projection_mode": "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
        "projection_mode": "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
        "projection_valid": False,
        "projection_invalid_reason": "current_bottle_3d_frame_not_measured",
        "two_d_fallback_used": True,
        "local_patch_estimate_used": True,
        "temporal_fallback_used": True,
        "contact_model_projection_status": "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
        "tracking_representation": "display_only_local_patch_estimate_from_last_valid_3d",
    }
    lock_report = target_lock_report(manager)
    compact = realtime_runner.compact_live_status({
        "frame_id": 9,
        "state": "APPROACH_TRACKING",
        "target_lock": lock_report,
        "target_clamp": manager.current_tracked_target["target_clamp"],
        "locked_projection_mode": manager.current_tracked_target["locked_projection_mode"],
        "projection_mode": manager.current_tracked_target["projection_mode"],
        "projection_valid": False,
        "two_d_fallback_used": True,
    })

    live_tracking = compact["target_lock"]["current_tracked_target"]
    assert live_tracking["projection_mode"] == "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D"
    assert live_tracking["projection_valid"] is False
    assert compact["target_clamp"]["clamp_center_px"] == target["fixed_clamp_zone"]["clamp_center_px"]
    assert "lock_bottle_points_xyz" not in str(compact)
    assert "lock_band_points_xyz" not in str(compact)


def test_partial_view_uses_current_depth_backed_band_anchor_not_visible_fragment_center():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    partial_pose = copy.deepcopy(pose)
    partial_mask = np.zeros_like(depth, dtype=np.uint8)
    partial_mask[0:82, 40:80] = 255
    partial_pose.update({
        "center_px": [59.5, 40.5],
        "cleaned_mask": partial_mask,
        "body_mask": partial_mask,
        "partial_extent": True,
        "_locked_band_center_observation_px": [59.5, 40.0],
        "_locked_band_center_observation_current": True,
        "_locked_band_center_observation_depth_mm": 400.0,
    })
    partial_depth = np.zeros_like(depth)
    partial_depth[partial_mask > 0] = 400

    projected = project_locked_grasp_band_to_current_frame(
        band, partial_pose, partial_depth, INTRINSICS
    )

    assert projected["projection_valid"] is True
    assert projected["partial_view_anchor"]["used"] is True
    assert projected["partial_view_anchor"]["source"] == "current_depth_backed_locked_band_observation"
    assert projected["projected_band_center_px"][1] == pytest.approx(40.0, abs=1.0)
    assert projected["projected_body_fraction"] == 0.5


def test_background_depth_near_contact_is_rejected_and_local_surface_recovers():
    pose, plan, depth, target = synthetic_lock_inputs()
    depth[:, 40:44] = 900

    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    payload = band.to_dict()

    assert payload["lock_valid"] is True
    assert payload["locked_left_contact_xyz_mm"][2] == pytest.approx(400.0)
    assert payload["locked_left_contact_depth_source"] in {
        "local_in_mask_median",
        "nearest_valid_in_mask_pixel",
    }
    assert payload["lock_bottle_depth_metadata"]["rejected_depth_outlier_count"] > 0


def test_mask_deprojection_reports_robust_surface_filter_metadata():
    mask = np.ones((20, 20), dtype=np.uint8) * 255
    depth = np.full((20, 20), 400, dtype=np.uint16)
    depth[:, :3] = 1000

    points, _, metadata = deproject_mask_mm(depth, INTRINSICS, mask, expected_depth_mm=400.0)

    assert len(points) == 340
    assert np.allclose(points[:, 2], 400.0)
    assert metadata["raw_valid_point_count"] == 400
    assert metadata["rejected_depth_outlier_count"] == 60
    assert metadata["depth_filter_status"] == "ROBUST_SAME_SURFACE"


def test_quality_prefers_current_projected_3d_for_planned_contacts():
    pose, plan, depth, target = synthetic_lock_inputs()
    report = {
        "planned_contacts": plan,
        "bottle_geometry": pose,
        "target_clamp": target["fixed_clamp_zone"],
        "target_lock": {
            "projection_valid": True,
            "locked_projection_mode": "TRUE_3D",
            "locked_target": target,
            "current_projected_grasp_band": {
                "center_px": [59.5, 60.0],
                "left_contact_px": [42.0, 60.0],
                "right_contact_px": [77.0, 60.0],
                "left_contact_xyz_mm": [-31.0, 0.0, 321.0],
                "right_contact_xyz_mm": [31.0, 0.0, 322.0],
                "band_centroid_xyz_mm": [0.0, 0.0, 321.5],
            },
        },
    }

    result = side_clamp_quality_inputs_from_snapshot(report, depth, INTRINSICS)

    assert result["current_projected_3d_authoritative"] is True
    assert result["contact_3d_source"] == "CURRENT_BOTTLE_LOCAL_3D_REPROJECTION"
    assert result["left_contact_3d_m"] == pytest.approx([-0.031, 0.0, 0.321])
    assert result["right_contact_3d_m"] == pytest.approx([0.031, 0.0, 0.322])


def test_immutable_lock_baseline_forces_planned_contacts_when_fingers_are_visible():
    pose, plan, depth, target = synthetic_lock_inputs()
    plan["epsilon_contacts"] = [
        {"point": plan["left_contact_px"], "inward_normal": [1.0, 0.0]},
        {"point": plan["right_contact_px"], "inward_normal": [-1.0, 0.0]},
    ]
    target["locked_grasp_band"] = create_locked_grasp_band_3d(
        target,
        pose,
        plan,
        depth,
        INTRINSICS,
    ).to_dict()
    report = {
        "writer_session_id": "writer-synthetic",
        "_quality_force_planned_contact": True,
        "_quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "planned_contacts": plan,
        "bottle_geometry": pose,
        "target_clamp": target["fixed_clamp_zone"],
        "finger_detection": {
            "finger_detection_status": "BOTH_FINGERS_DETECTED",
            "finger_stable_frames": realtime_runner.FINGER_STABLE_FRAMES_REQUIRED,
            "left_finger_confidence": 100.0,
            "right_finger_confidence": 100.0,
            "left_finger_contact_px": [5.0, 5.0],
            "right_finger_contact_px": [110.0, 110.0],
            "left_finger_depth_mm": 900.0,
            "right_finger_depth_mm": 900.0,
            "left_finger_depth_verified": True,
            "right_finger_depth_verified": True,
            "left_finger_depth_delta_to_bottle_mm": 0.0,
            "right_finger_depth_delta_to_bottle_mm": 0.0,
            "finger_bottle_depth_tolerance_mm": 60.0,
        },
        "target_lock": {
            "attempt_id": target["attempt_id"],
            "lock_frame_id": target["lock_frame_id"],
            "projection_valid": True,
            "locked_projection_mode": "TRUE_3D",
            "locked_target": target,
            "current_projected_grasp_band": {
                "center_px": [10.0, 10.0],
                "left_contact_px": [1.0, 10.0],
                "right_contact_px": [19.0, 10.0],
                "left_contact_xyz_mm": [-999.0, 0.0, 999.0],
                "right_contact_xyz_mm": [999.0, 0.0, 999.0],
                "band_centroid_xyz_mm": [0.0, 0.0, 999.0],
            },
        },
    }

    result = side_clamp_quality_inputs_from_snapshot(report, depth, INTRINSICS)

    assert result["quality_input_source"] == "planned_contact"
    assert result["quality_contact_source_forced"] is True
    assert result["quality_baseline_role"] == "IMMUTABLE_LOCK_BASELINE"
    assert result["contact_quality_gate_status"] == "PLANNED_BASELINE"
    assert result["normal_source"] == "LOCKED_BAND_LOCAL_SIDE_NORMALS_PLANNED"
    assert result["planned_antipodal_normal_contract"] is True
    assert result["immutable_lock_contract_valid"] is True
    assert result["current_projected_3d_authoritative"] is False
    frozen_band = target["locked_grasp_band"]
    assert result["left_contact_px"] == pytest.approx(
        frozen_band["locked_left_contact_projection_at_lock_px"]
    )
    assert result["right_contact_px"] == pytest.approx(
        frozen_band["locked_right_contact_projection_at_lock_px"]
    )
    assert result["left_contact_3d_m"] == pytest.approx(
        np.asarray(frozen_band["locked_left_contact_xyz_mm"]) / 1000.0
    )
    assert result["right_contact_3d_m"] == pytest.approx(
        np.asarray(frozen_band["locked_right_contact_xyz_mm"]) / 1000.0
    )
    assert result["left_inward_normal_px"] == pytest.approx([1.0, 0.0])
    assert result["right_inward_normal_px"] == pytest.approx([-1.0, 0.0])
    assert np.dot(
        result["left_inward_normal_3d"], result["right_inward_normal_3d"]
    ) == pytest.approx(-1.0)
    assert result["contact_geometry_checks"]["validation_status"] == "valid"
    assert result["contact_geometry_checks"]["contact_opposition_score"] == 100.0
    epsilon = realtime_runner.compute_epsilon_quality(
        result["epsilon_contacts"],
        result["object_center_px"],
        result["grip_width_px"] / 2.0,
        mu=realtime_runner.FRICTION_COEFFICIENT,
        samples_per_contact=realtime_runner.CONTACT_FORCE_SAMPLES,
    )
    assert epsilon["force_closure"] is True
    assert epsilon["epsilon"] > 0.0
    assert realtime_runner.quality_label_from_epsilon(epsilon, result) == "GOOD"


def test_immutable_lock_baseline_rejects_wrong_band_lineage_instead_of_using_current_pixels():
    pose, plan, depth, target = synthetic_lock_inputs()
    target["locked_grasp_band"] = create_locked_grasp_band_3d(
        target, pose, plan, depth, INTRINSICS
    ).to_dict()
    target["locked_grasp_band"]["locked_frame_id"] = 999
    report = {
        "writer_session_id": "writer-synthetic",
        "_quality_force_planned_contact": True,
        "_quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "planned_contacts": plan,
        "bottle_geometry": pose,
        "target_lock": {
            "attempt_id": target["attempt_id"],
            "lock_frame_id": target["lock_frame_id"],
            "locked_target": target,
            "projection_valid": True,
            "locked_projection_mode": "TRUE_3D",
            "current_projected_grasp_band": {
                "left_contact_px": [1.0, 1.0],
                "right_contact_px": [2.0, 2.0],
                "left_contact_xyz_mm": [1.0, 1.0, 1.0],
                "right_contact_xyz_mm": [2.0, 2.0, 2.0],
                "band_centroid_xyz_mm": [1.5, 1.5, 1.5],
            },
        },
    }

    result = side_clamp_quality_inputs_from_snapshot(report, depth, INTRINSICS)

    assert result["status"] == "invalid"
    assert result["immutable_lock_contract_valid"] is False
    assert "immutable_baseline_frame_lineage_mismatch" in result["failure_reasons"]
    assert result["current_projected_3d_authoritative"] is False


def test_immutable_lock_baseline_rejects_invalid_planned_contact_geometry():
    pose, plan, depth, target = synthetic_lock_inputs()
    target["locked_grasp_band"] = create_locked_grasp_band_3d(
        target, pose, plan, depth, INTRINSICS
    ).to_dict()
    # Keep the immutable lineage and 3D contract intact but move one frozen
    # projected contact outside the locked bottle hull.  A large epsilon must
    # never be frozen with a contradictory WEAK/warning geometry label.
    target["locked_grasp_band"]["locked_left_contact_projection_at_lock_px"] = [5.0, 60.0]
    report = {
        "writer_session_id": "writer-synthetic",
        "_quality_force_planned_contact": True,
        "_quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "planned_contacts": plan,
        "bottle_geometry": pose,
        "target_lock": {
            "attempt_id": target["attempt_id"],
            "lock_frame_id": target["lock_frame_id"],
            "locked_target": target,
        },
    }

    result = side_clamp_quality_inputs_from_snapshot(report, depth, INTRINSICS)

    assert result["status"] == "invalid"
    assert result["immutable_lock_contract_valid"] is True
    assert result["contact_geometry_checks"]["validation_status"] == "warning"
    assert "immutable_planned_contact_geometry_invalid" in result["failure_reasons"]


def test_quality_worker_freezes_exact_immutable_band_despite_stray_live_contacts():
    pose, plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS).to_dict()
    target["locked_grasp_band"] = copy.deepcopy(band)
    target["locked_grasp_band_3d"] = copy.deepcopy(band)
    report = {
        "frame_id": target["lock_frame_id"],
        "writer_session_id": "writer-synthetic",
        "writer_pid": 123,
        "_quality_force_planned_contact": True,
        "_quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "planned_contacts": plan,
        "selected_grasp": plan,
        "bottle_geometry": pose,
        "bottle_depth_median_mm": 400.0,
        "finger_detection": {
            "finger_detection_status": "BOTH_FINGERS_DETECTED",
            "left_finger_contact_px": [5.0, 5.0],
            "right_finger_contact_px": [110.0, 110.0],
            "left_finger_depth_mm": 900.0,
            "right_finger_depth_mm": 900.0,
        },
        "target_lock": {
            "target_status": "LOCKED",
            "attempt_id": target["attempt_id"],
            "lock_frame_id": target["lock_frame_id"],
            "locked_patch_visible": True,
            "locked_target": target,
            "current_projected_grasp_band": {
                "left_contact_px": [1.0, 1.0],
                "right_contact_px": [2.0, 2.0],
                "left_contact_xyz_mm": [-999.0, 0.0, 999.0],
                "right_contact_xyz_mm": [999.0, 0.0, 999.0],
            },
        },
    }
    emitted = []
    worker = QualityMatrixWorker()
    worker.result_ready.connect(emitted.append)

    worker.compute({
        "frame_id": target["lock_frame_id"],
        "report": report,
        "intrinsics": INTRINSICS,
        "depth_snapshot": depth,
        "rgb_snapshot": np.zeros((120, 120, 3), dtype=np.uint8),
        "rgb_shape": [120, 120, 3],
        "snapshot_timestamp": realtime_runner.time.time(),
    })

    assert len(emitted) == 1
    result = emitted[0]
    assert result["quality_matrix_status"] == "updated"
    assert result["baseline_frozen"] is True
    assert result["quality_baseline_role"] == "IMMUTABLE_LOCK_BASELINE"
    assert result["quality_input_source"] == "planned_contact"
    assert result["quality_contact_source_forced"] is True
    assert result["force_closure"] is True
    assert result["raw_epsilon"] > 0.0
    assert result["quality_label"] == "GOOD"
    inputs_px = result["contact_points_px"]
    inputs_3d = result["contact_points_3d"]
    assert inputs_px["left"] == pytest.approx(
        band["locked_left_contact_projection_at_lock_px"]
    )
    assert inputs_px["right"] == pytest.approx(
        band["locked_right_contact_projection_at_lock_px"]
    )
    assert inputs_3d["left"] == pytest.approx(
        np.asarray(band["locked_left_contact_xyz_mm"]) / 1000.0
    )
    assert inputs_3d["right"] == pytest.approx(
        np.asarray(band["locked_right_contact_xyz_mm"]) / 1000.0
    )


def test_locked_projection_ready_uses_current_antipodal_band_not_legacy_workspace_or_visual_estimator(monkeypatch):
    pose, plan, depth, target = synthetic_lock_inputs()
    target["locked_grasp_band"] = create_locked_grasp_band_3d(
        target, pose, plan, depth, INTRINSICS
    ).to_dict()
    monkeypatch.setattr(
        realtime_runner,
        "evaluate_workspace_reachability",
        lambda *_args, **_kwargs: {"both_reachable": False, "mode": "approximate"},
    )
    monkeypatch.setattr(
        realtime_runner,
        "compute_visual_contact_model",
        lambda *_args, **_kwargs: {
            "visual_contact_ready": False,
            "visual_contact_reasons": ["diagnostic_patch_not_detected"],
            "visual_contact_warnings": [],
            "visual_contact_quality_score": 0.0,
            "left_visual_clamp_patch": {"on_bottle_body": False},
            "right_visual_clamp_patch": {"on_bottle_body": False},
            "visual_contact_pair": {},
            "epsilon_contacts": [],
        },
    )

    projected, _selected, projection = realtime_runner.project_locked_bottle_contact_model(
        pose,
        depth,
        INTRINSICS,
        target,
        {"full_bottle_visible": True},
        (120, 120, 3),
    )

    assert projection["projection_valid"] is True
    assert projected["contact_model_ready"] is True
    assert projected["workspace_both_reachable"] is False
    assert projected["workspace_hard_gate_active"] is False
    epsilon = realtime_runner.compute_epsilon_quality(
        projected["epsilon_contacts"],
        projected["epsilon_object_centroid_px"],
        projected["grip_width_px"] / 2.0,
        mu=realtime_runner.FRICTION_COEFFICIENT,
        samples_per_contact=8,
    )
    realtime_runner.attach_visual_contact_model(
        projected,
        pose,
        depth,
        INTRINSICS,
        (120, 120, 3),
        epsilon_result=epsilon,
        locked_target=target,
    )
    assert epsilon["force_closure"] is True
    assert projected["visual_contact_ready"] is False
    assert projected["visual_contact_required_for_ready"] is False
    assert projected["contacts_on_body"] is True
    tracking = {
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "locked_patch_visible": True,
        "contact_patch_tracking_ok": True,
        "tracking_confidence": 100.0,
        "patch_depth_confidence": 100.0,
        "bottle_geometry_tracking_ok": True,
        "locked_projection_mode": "TRUE_3D",
        "patch_width_valid": True,
        "patch_on_bottle_body": True,
    }
    blockers = ready_blockers_from_values(
        projected, tracking, epsilon, 100.0, [], target_status="LOCKED"
    )
    assert "contact_model_not_ready" not in blockers
    assert "visual_contact_not_ready" not in blockers
    assert blockers == []


def test_runtime_diagnostics_report_alignment_intrinsics_motion_and_depth_provenance():
    diagnostics = grasp_target_3d_runtime_diagnostics(
        12,
        {
            "bottle_target_id": "bottle-synthetic",
            "locked_grasp_band": {
                "locked_intrinsics_source": "synthetic_camera_info",
                "locked_bottle_depth_mm": 400.0,
            },
        },
        {
            "tracking_source": "roi_opencv_depth",
            "current_bottle_frame_3d": {"origin_xyz_mm": [0.0, 0.0, 350.0]},
            "projected_left_contact_px": [40.0, 60.0],
            "projected_left_contact_xyz_mm": [-30.0, 0.0, 350.0],
            "projected_right_contact_px": [80.0, 60.0],
            "projected_right_contact_xyz_mm": [30.0, 0.0, 350.0],
            "projected_left_contact_depth_source": "local_in_mask_median",
            "projected_right_contact_depth_source": "exact_pixel",
            "locked_projection_mode": "TRUE_3D",
            "projection_valid": True,
            "projected_depth_mm": 350.0,
            "current_to_lock_length_ratio": 1.0,
            "roi_depth_projection_scale": 1.14,
            "roi_candidate_depth_delta_mm": 12.0,
            "tracking_delta_this_frame_px": [2.0, -3.0],
        },
        {"center_px": [60.0, 60.0]},
        INTRINSICS,
        {"rgb_host_timestamp": 1.0, "depth_host_timestamp": 1.001},
    )

    assert diagnostics["rgb_depth_aligned"] is True
    assert diagnostics["intrinsics_used"]["fx"] == 200.0
    assert diagnostics["left_contact_depth_source"] == "local_in_mask_median"
    assert diagnostics["depth_change_from_lock_mm"] == pytest.approx(-50.0)
    assert diagnostics["perspective_scale_from_previous_rgbd_reference"] == 1.14
    assert diagnostics["image_translation_this_frame_px"] == [2.0, -3.0]


def test_current_fast_path_selection_uses_true_3d_clamp_not_2d_fallback():
    pose, _plan, depth, target = synthetic_lock_inputs()
    band = create_locked_grasp_band_3d(target, pose, _plan, depth, INTRINSICS)
    target["locked_grasp_band"] = band.to_dict()
    fallback = copy.deepcopy(target["fixed_clamp_zone"])

    clamp, model, projection, trace = select_locked_band_projection(
        target,
        pose,
        depth,
        INTRINSICS,
        fallback,
        {"band_source": "current_2d_contour", "edge_source": "current_2d_contour"},
        geometry_measurement_current=True,
    )

    assert projection["projection_valid"] is True
    assert projection["locked_projection_mode"] == "TRUE_3D"
    assert projection["two_d_fallback_used"] is False
    assert clamp["clamp_source"] == "locked_bottle_local_3d_reprojection"
    assert clamp["clamp_band_polygon"] == projection["projected_band_polygon_px"]
    assert model["model_coordinate_frame"] == "bottle_local_3d_reprojected_to_camera"
    assert trace["band_source"] == "locked_bottle_local_3d_reprojection"


def test_target_lock_report_treats_projected_3d_contact_model_as_active():
    manager = TargetLockManager()
    manager.locked_target = {
        "valid": True,
        "lock_frame_id": 8,
        "locked_body_fraction": 0.5,
    }
    tracking = {
        "tracking_ok": True,
        "identity_tracking_ok": True,
        "contact_model_projection_status": "PROJECTED_3D",
        "bottle_contact_model": {"projection_status": "PROJECTED_3D"},
        "locked_projection_mode": "TRUE_3D",
        "projection_valid": True,
        "current_patch_center_px": [60.0, 60.0],
    }

    status = target_lock_report(manager, tracking)

    assert status["bottle_contact_model_ready"] is True
    assert status["bottle_contact_model"]["projection_status"] == "PROJECTED_3D"


def test_target_loss_has_explicit_grasp_band_state():
    _, _, _, target = synthetic_lock_inputs()
    status = locked_grasp_band_status(target, {}, target_status="LOST", lock_state="LOST")

    assert status["grasp_band_state"] == "TARGET_LOST"
    assert status["band_relock_required"] is True
    assert status["band_reselected_this_frame"] is False


def test_temporary_3d_dropout_holds_same_lock_without_stale_or_relock():
    _, _, _, target = synthetic_lock_inputs()
    tracking = {
        "current_patch_center_px": [61.0, 62.0],
        "target_clamp": copy.deepcopy(target["fixed_clamp_zone"]),
        "contact_model_projection_status": "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
        "projection_mode": "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
        "locked_projection_mode": "ESTIMATED_LOCAL_PATCH_FROM_LAST_VALID_3D",
        "projection_valid": False,
        "projection_invalid_reason": "current_bottle_3d_frame_not_measured",
        "local_patch_tracking_used": True,
        "two_d_fallback_used": True,
        "tracking_confidence": 35.0,
        "locked_patch_visible": True,
    }

    status = locked_grasp_band_status(
        target,
        tracking,
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["grasp_band_state"] == "LOCKED_HELD"
    assert status["grasp_band_held"] is True
    assert status["grasp_band_current"] is False
    assert status["grasp_band_stale"] is False
    assert status["grasp_band_relock_required"] is False
    assert status["current_projected_grasp_band"]["body_fraction"] == pytest.approx(
        target["locked_body_fraction"]
    )


def test_missing_projection_does_not_relock_while_lock_manager_still_owns_target():
    _, _, _, target = synthetic_lock_inputs()
    status = locked_grasp_band_status(
        target,
        {
            "projection_mode": "INVALID_3D",
            "locked_projection_mode": "INVALID_3D",
            "projection_valid": False,
            "projection_invalid_reason": "current_bottle_3d_frame_not_measured",
        },
        target_status="LOW_CONFIDENCE",
        lock_state="LOCKED",
    )

    assert status["grasp_band_state"] == "LOCKED_PROJECTION_UNAVAILABLE"
    assert status["grasp_band_projection_unavailable"] is True
    assert status["grasp_band_stale"] is False
    assert status["grasp_band_relock_required"] is False


def test_projection_age_beyond_configured_limit_is_stale():
    _, _, _, target = synthetic_lock_inputs()
    tracking = {
        "current_patch_center_px": [59.5, 60.0],
        "contact_model_projection_status": "PROJECTED",
        "tracking_confidence": 90.0,
        "grasp_band_projection_age_sec": realtime_runner.MAX_BAND_STALE_SEC + 0.1,
    }

    status = locked_grasp_band_status(
        target,
        tracking,
        target_status="TRACKING",
        lock_state="LOCKED",
    )

    assert status["grasp_band_state"] == "STALE"
    assert status["grasp_band_current"] is False
    assert status["grasp_band_projection_age_sec"] > realtime_runner.MAX_BAND_STALE_SEC
    assert "projection_stale" in status["band_drift_reason"]


def test_invalid_lock_artifact_omits_previous_geometry(monkeypatch, tmp_path):
    lock_path = tmp_path / "locked_grasp_band.json"
    debug_path = tmp_path / "locked_grasp_band_debug.json"
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", lock_path)
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", debug_path)

    payload = write_invalid_locked_grasp_band_artifacts(
        "manual_reset_from_dashboard",
        {
            "attempt_id": "attempt-test",
            "lock_frame_id": 8,
            "locked_band_points_xyz_mm": [[1.0, 2.0, 3.0]],
        },
    )

    assert payload["lock_valid"] is False
    assert payload["lock_invalid_reason"] == "manual_reset"
    assert payload["locked_grasp_band_3d"] is None
    assert payload["previous_lock_identity"] == {
        "attempt_id": "attempt-test",
        "lock_frame_id": 8,
    }
    assert "locked_band_points_xyz_mm" not in lock_path.read_text()
    assert debug_path.is_file()


def test_lock_snapshot_provenance_is_only_created_on_current_lock_frame():
    report = {
        "writer_session_id": "writer-test",
        "frame_id": 8,
        "state": "TARGET_LOCKED",
        "target_lock": {
            "grasp_band_locked": True,
            "lock_frame_id": 8,
            "attempt_id": "attempt-test",
            "bottle_target_id": "bottle-test",
        },
    }

    provenance = current_lock_frame_snapshot_provenance(report)

    assert provenance["capture_status"] == "CURRENT_LOCK_FRAME"
    assert provenance["writer_session_id"] == "writer-test"
    assert provenance["lock_frame_id"] == 8
    assert provenance["attempt_id"] == "attempt-test"
    assert provenance["bottle_target_id"] == "bottle-test"
    assert current_lock_frame_snapshot_provenance({**report, "frame_id": 9}) is None
    assert current_lock_frame_snapshot_provenance({"frame_id": 8}) is None

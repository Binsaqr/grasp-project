"""Camera-free tests for masked contact-band depth and distance guidance."""

import numpy as np
import pytest

from grasp_project.common.config import load_named_config
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.realtime_runner import (
    compact_live_status,
    compute_approach_guidance,
    compute_view_quality,
    contact_patch_depths,
    current_geometry_guidance_pose,
    draw_fast_overlay,
    publish_quality_bundle,
    save_grasp_quality_bottle_weak_direction_overlay,
)


def _target_clamp():
    return {
        "left_clamp_patch": {
            "center_px": [80.0, 100.0],
            "polygon": [[74, 90], [86, 90], [86, 110], [74, 110]],
        },
        "right_clamp_patch": {
            "center_px": [120.0, 100.0],
            "polygon": [[114, 90], [126, 90], [126, 110], [114, 110]],
        },
        "left_contact_px": [80.0, 100.0],
        "right_contact_px": [120.0, 100.0],
        "clamp_center_px": [100.0, 100.0],
        "clamp_band_polygon": [[74, 88], [126, 88], [126, 112], [74, 112]],
        "grip_width_px": 40.0,
    }


def _bottle_scene():
    depth = np.full((200, 200), 1200, dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[35:166, 80:121] = 1
    depth[mask > 0] = 600
    return depth, mask


def _alignment_arrow_report(*, approach_requested=True, move_x_mm=12.0):
    return {
        "overlay_diagnostics_enabled": True,
        "target_lock": {
            "target_status": "TRACKING",
            "approach_mode_requested": approach_requested,
        },
        "approach_mode_requested": approach_requested,
        "distance_measurement_valid": True,
        "target_x_px": 130.0,
        "target_y_px": 100.0,
        "ideal_x_px": 100.0,
        "ideal_y_px": 100.0,
        "recommended_move_x_mm": move_x_mm,
        "recommended_move_y_mm": 0.0,
        "approach_action": "CENTER_TARGET",
    }


def test_live_alignment_arrow_obeys_disabled_flag_and_guidance_deadband(monkeypatch):
    calls = []
    monkeypatch.setattr(realtime_runner.cv2, "arrowedLine", lambda *args, **kwargs: calls.append((args, kwargs)))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    monkeypatch.setattr(realtime_runner, "SHOW_MAIN_MOVEMENT_ARROW", False)
    monkeypatch.setattr(realtime_runner, "HAND_MOUNTED_CAMERA_MODE", True)
    disabled = _alignment_arrow_report()
    draw_fast_overlay(frame, None, None, disabled)
    assert calls == []
    assert disabled["yellow_arrow_visible"] is False

    monkeypatch.setattr(realtime_runner, "SHOW_MAIN_MOVEMENT_ARROW", True)
    before_approach = _alignment_arrow_report(approach_requested=False)
    draw_fast_overlay(frame, None, None, before_approach)
    deadbanded = _alignment_arrow_report(move_x_mm=0.0)
    draw_fast_overlay(frame, None, None, deadbanded)
    assert calls == []

    actionable = _alignment_arrow_report()
    draw_fast_overlay(frame, None, None, actionable)
    assert len(calls) == 1
    assert calls[0][0][1] == (130, 100)
    assert calls[0][0][2] == (100, 100)
    assert actionable["camera_alignment_arrow_visible"] is True
    assert actionable["camera_alignment_arrow_semantics"] == "TARGET_IMAGE_ALIGNMENT_GUIDANCE_NOT_OBJECT_MOTION"


def test_presentation_hides_unvalidated_virtual_hand_guidance(monkeypatch):
    arrows = []
    monkeypatch.setattr(
        realtime_runner.cv2,
        "arrowedLine",
        lambda *args, **kwargs: arrows.append((args, kwargs)),
    )
    monkeypatch.setattr(realtime_runner, "SHOW_MAIN_MOVEMENT_ARROW", True)
    monkeypatch.setattr(realtime_runner, "HAND_MOUNTED_CAMERA_MODE", True)
    config = dict(realtime_runner.HAND_MOUNTED_CAMERA_CONFIG)
    config["hand_offset_physically_validated"] = False
    monkeypatch.setattr(realtime_runner, "HAND_MOUNTED_CAMERA_CONFIG", config)

    report = _alignment_arrow_report()
    report["overlay_diagnostics_enabled"] = False
    draw_fast_overlay(np.zeros((200, 200, 3), dtype=np.uint8), None, None, report)

    assert arrows == []
    assert report["hand_reach_marker_visible"] is False
    assert report["hand_alignment_arrow_visible"] is False


def test_live_overlay_draws_stabilized_guide_but_keeps_raw_clamp_authoritative(monkeypatch):
    circles = []
    monkeypatch.setattr(
        realtime_runner.cv2,
        "circle",
        lambda _image, center, *_args, **_kwargs: circles.append(center),
    )
    raw_clamp = _target_clamp()
    display_clamp = realtime_runner.shifted_target_clamp(raw_clamp, [-40.0, -40.0])
    report = {
        "target_clamp": raw_clamp,
        "overlay_diagnostics_enabled": True,
        "target_lock": {
            "target_status": "TRACKING",
            "current_tracked_target": {"display_target_clamp": display_clamp},
        },
    }

    draw_fast_overlay(np.zeros((200, 200, 3), dtype=np.uint8), None, None, report)

    assert (40, 60) in circles
    assert (80, 60) in circles
    assert (80, 100) not in circles
    assert (120, 100) not in circles
    assert report["target_clamp"] is raw_clamp


def test_true_3d_overlay_draws_stable_bottle_relative_guide_not_raw_projection(monkeypatch):
    polygons = []

    def capture_polyline(_image, points, *_args, **_kwargs):
        polygons.append(np.asarray(points[0]).reshape(-1, 2).tolist())

    monkeypatch.setattr(realtime_runner.cv2, "polylines", capture_polyline)
    raw_clamp = _target_clamp()
    display_clamp = realtime_runner.shifted_target_clamp(raw_clamp, [-35.0, 25.0])
    report = {
        "target_clamp": raw_clamp,
        "target_lock": {
            "target_status": "TRACKING",
            "locked_target": {
                "locked_grasp_band": {"true_3d_lock": True},
            },
            "current_tracked_target": {
                "projection_source": "BOTTLE_LOCAL_3D_REPROJECTION",
                "projection_valid": True,
                "two_d_fallback_used": False,
                "locked_projection_mode": "TRUE_3D",
                "display_target_clamp": display_clamp,
            },
        },
    }

    realtime_runner.draw_fast_overlay(
        np.zeros((240, 240, 3), dtype=np.uint8), None, None, report
    )

    raw = np.round(np.asarray(raw_clamp["clamp_band_polygon"])).astype(int).tolist()
    assert polygons
    assert raw not in polygons
    ribbon = np.asarray(polygons[-1], dtype=float)
    assert np.ptp(ribbon[:, 1]) <= 14.0
    assert np.mean(ribbon, axis=0) == pytest.approx(display_clamp["clamp_center_px"], abs=1.0)
    assert report["grasp_band_presentation_shape"] == "SLIM_BOTTLE_AXIS_RIBBON"
    assert report["grasp_band_display_continuity"] == "CURRENT_3D"


def test_invalid_analytic_3d_draws_only_explicit_current_bottle_visual_guide(monkeypatch):
    polygons = []
    monkeypatch.setattr(
        realtime_runner.cv2,
        "polylines",
        lambda _image, points, *_args, **_kwargs: polygons.append(
            np.asarray(points[0]).reshape(-1, 2).tolist()
        ),
    )
    display_clamp = realtime_runner.shifted_target_clamp(_target_clamp(), [20.0, -15.0])
    report = {
        "target_clamp": {},
        "target_lock": {
            "target_status": "LOW_CONFIDENCE",
            "grasp_band_relock_required": False,
            "locked_target": {"locked_grasp_band": {"true_3d_lock": True}},
            "current_tracked_target": {
                "locked_projection_mode": "INVALID_3D",
                "projection_valid": False,
                "display_target_clamp": display_clamp,
                "operator_locked_fraction_visual_guide_current": True,
            },
        },
    }

    realtime_runner.draw_fast_overlay(
        np.zeros((240, 240, 3), dtype=np.uint8), None, None, report
    )

    assert polygons
    ribbon = np.asarray(polygons[-1], dtype=float)
    assert np.ptp(ribbon[:, 1]) <= 14.0
    assert np.mean(ribbon, axis=0) == pytest.approx(display_clamp["clamp_center_px"], abs=1.0)
    assert report["grasp_band_operator_guide_visible_during_depth_dropout"] is True
    assert (
        report["grasp_band_display_continuity"]
        == "CURRENT_BOTTLE_RELATIVE_GUIDE_DISPLAY_ONLY"
    )


def test_operator_band_and_bottle_extent_share_one_locked_fraction_transform():
    manager = realtime_runner.TargetLockManager()
    manager.locked_target = {
        "lock_frame_id": 1,
        "grasp_reference_id": "locked-fraction-test",
        "locked_body_fraction": 0.74,
        "locked_bottle_geometry": {"bbox_xywh": [90, 40, 45, 150]},
        "locked_grasp_band": {"locked_band_center_px": [112.5, 151.0]},
    }
    base = _target_clamp()
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 120.0, "cy": 120.0}

    def geometry(y1, y2, x1, x2):
        mask = np.zeros((260, 260), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return {
            "cleaned_mask": mask,
            "body_mask": mask,
            "center_px": [(x1 + x2 - 1) / 2.0, (y1 + y2 - 1) / 2.0],
            "axis_vector": [0.0, 1.0],
            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            "bbox_xyxy": [x1, y1, x2, y2],
            "bottle_extent_bbox_xyxy": [x1, y1, x2, y2],
            "depth_median_mm": 350.0,
        }

    fractions = []
    for frame_id, current in enumerate(
        (geometry(40, 190, 90, 135), geometry(8, 248, 55, 185)), start=1
    ):
        guide, _trace = realtime_runner.build_operator_locked_fraction_guide(
            base,
            current,
            manager.locked_target,
            np.full((260, 260), 350, dtype=np.uint16),
            (260, 260, 3),
            intrinsics,
        )
        result = realtime_runner.apply_display_geometry_smoothing(
            manager,
            current,
            guide,
            {"tracking_source": "roi_opencv_depth", "identity_measurement_current": True},
            (260, 260, 3),
            frame_id,
        )
        center = np.asarray(result["display_magenta_center"], dtype=float)
        bbox = np.asarray(result["display_bottle_bbox"], dtype=float)
        fractions.append((center[1] - bbox[1]) / (bbox[3] - bbox[1]))
        assert result["display_band_bottle_fraction_coupled"] is True
        assert result["display_smoothing_mode"] == "locked_body_fraction_coupled_to_shown_bottle"

    assert all(abs(value - 0.74) < 0.02 for value in fractions)
    assert abs(fractions[1] - fractions[0]) < 0.01


def test_operator_quality_closeup_excludes_analytic_weak_direction_arrow(monkeypatch, tmp_path):
    captured = {}

    def capture_engineering(image, contact_image_rgb=None):
        captured["engineering"] = image.copy()
        return {}

    def capture_operator(image, hull_result):
        captured["operator"] = image.copy()
        return {}

    def mark_analytic_arrow(image, *args, **kwargs):
        image[-1, -1] = [1, 2, 3]

    monkeypatch.setattr(realtime_runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(realtime_runner, "write_quality_overlay_variants", capture_engineering)
    monkeypatch.setattr(realtime_runner, "write_quality_context_closeup", capture_operator)
    monkeypatch.setattr(realtime_runner, "save_contact_patch_debug_images", lambda *args, **kwargs: {})
    monkeypatch.setattr(realtime_runner, "write_contact_hull_debug", lambda *args, **kwargs: "output/contact_hull_debug.json")
    monkeypatch.setattr(realtime_runner.cv2, "arrowedLine", mark_analytic_arrow)

    _, metadata = save_grasp_quality_bottle_weak_direction_overlay(
        np.zeros((180, 220, 3), dtype=np.uint8),
        {
            "quality_compute_id": "quality-arrow-semantics",
            "quality_label": "GOOD",
            "epsilon": 0.1,
            "epsilon_score": 80.0,
            "weak_direction": [1.0, 0.0, 0.0],
            "image_bottle_hull_px": [[70, 20], [150, 20], [150, 160], [70, 160]],
            "contact_points_px": {
                "left": [75.0, 90.0],
                "right": [145.0, 90.0],
                "object_center": [110.0, 90.0],
            },
        },
    )

    assert captured["engineering"][-1, -1].tolist() == [1, 2, 3]
    assert captured["operator"][-1, -1].tolist() != [1, 2, 3]
    assert metadata["weak_direction_is_object_motion"] is False
    assert metadata["operator_quality_view_includes_weak_direction_arrow"] is False
    assert metadata["weak_direction_semantics"] == "ANALYTIC_GRASP_RESISTANCE_NOT_OBJECT_MOTION"


def test_quality_bundle_publishes_lock_attempt_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "OUTPUT_DIR", tmp_path)

    manifest = publish_quality_bundle(
        {
            "quality_compute_id": "quality-attempt-provenance",
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-42",
            "bottle_target_id": "bottle-42",
            "locked_band_id": "band-42",
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
            "quality_compute_started_at": 1.0,
        },
        {},
    )

    assert manifest["writer_session_id"] == "writer-1"
    assert manifest["lock_frame_id"] == 42
    assert manifest["lock_attempt_id"] == "attempt-42"
    assert manifest["bottle_target_id"] == "bottle-42"
    assert manifest["locked_band_id"] == "band-42"
    assert manifest["quality_baseline_role"] == "IMMUTABLE_LOCK_BASELINE"
    assert manifest["quality_input_source"] == "planned_contact"
    assert manifest["quality_contact_source_forced"] is True
    assert manifest["baseline_frozen"] is True


def test_quality_bundle_does_not_reuse_artifact_from_previous_compute(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "OUTPUT_DIR", tmp_path)
    old_closeup = tmp_path / "bottle_quality_closeup.jpg"
    old_closeup.write_bytes(b"old-arrow-image")

    manifest = publish_quality_bundle(
        {
            "quality_compute_id": "new-quality",
            "quality_matrix_status": "updated",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-42",
            "quality_compute_started_at": old_closeup.stat().st_mtime + 10.0,
            "quality_compute_started_at_ns": old_closeup.stat().st_mtime_ns + 1,
        },
        {},
    )

    artifact = manifest["artifact_paths"]["bottle_quality_closeup"]
    assert manifest["quality_bundle_status"] == "PARTIAL"
    assert artifact["generated_for_current_compute"] is False
    assert artifact["artifact_status"] == "stale_from_previous_compute"
    assert artifact.get("run_path") is None


def test_contact_depth_uses_current_bottle_mask_not_background_half_patch():
    depth, mask = _bottle_scene()

    evidence = contact_patch_depths(
        depth,
        _target_clamp(),
        bottle_mask=mask,
        reference_depth_mm=600.0,
    )

    assert evidence["left_patch_depth_mm"] == 600.0
    assert evidence["right_patch_depth_mm"] == 600.0
    assert evidence["clamp_patch_depth_median_mm"] == 600.0
    assert evidence["clamp_patch_depth_bilateral_consistent"] is True
    assert evidence["left_patch_bottle_overlap_ratio"] < 1.0
    assert evidence["depth_evidence_source"] == "CURRENT_BOTTLE_MASK_AND_ALIGNED_DEPTH"


def test_inconsistent_left_right_surface_depth_fails_closed():
    depth, mask = _bottle_scene()
    depth[mask > 0] = 600
    depth[:, 101:][mask[:, 101:] > 0] = 850

    evidence = contact_patch_depths(
        depth,
        _target_clamp(),
        bottle_mask=mask,
        reference_depth_mm=725.0,
    )

    assert evidence["clamp_patch_depth_bilateral_consistent"] is False
    assert evidence["clamp_patch_depth_median_mm"] is None
    assert evidence["clamp_patch_depth_valid_ratio"] == 0.0


def test_center_depth_and_xyz_units_are_separate_and_correct():
    depth, mask = _bottle_scene()
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 600.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    tracking = {
        "target_clamp": _target_clamp(),
        "locked_patch_visible": True,
        "bottle_geometry_tracking_ok": True,
        "identity_measurement_current": True,
        "current_grip_width_px": 40.0,
    }

    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        tracking,
        depth.shape,
    )

    assert guidance["target_center_depth_mm"] == 600.0
    assert guidance["target_center_depth_source"] == "CURRENT_MASKED_CENTER_PATCH"
    assert guidance["camera_to_target_xyz_m"][2] == 0.6
    assert guidance["camera_to_target_xyz_mm"][2] == 600.0
    assert guidance["distance_definition"] == "CAMERA_OPTICAL_AXIS_Z_TO_CURRENT_BOTTLE_CONTACT_BAND_ESTIMATE"


def test_camera_frame_xyz_guidance_uses_documented_sign_convention():
    depth, mask = _bottle_scene()
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 600.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    tracking = {
        "target_clamp": _target_clamp(),
        "locked_patch_visible": True,
        "bottle_geometry_tracking_ok": True,
        "identity_measurement_current": True,
        "current_grip_width_px": 40.0,
        "finalized_preclose_target": {
            "exists": True,
            "finalized_target_x_px": 80.0,
            "finalized_target_y_px": 120.0,
            "finalized_target_depth_mm": 90.0,
        },
    }

    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        tracking,
        depth.shape,
    )

    # The finalized screen target no longer moves the bottle band.  The fixed
    # physical hand-contact projection is left of the optical center for the
    # current mount, so the camera/gripper is told to move right toward it.
    assert guidance["recommended_move_x_mm"] == 25.0
    assert guidance["recommended_move_y_mm"] < 0.0
    assert guidance["recommended_move_z_mm"] < 0.0
    assert guidance["coordinate_frame"] == "camera_frame"
    assert guidance["movement_convention"]["+X"] == "move camera/gripper unit right"
    assert guidance["movement_convention"]["-Y"] == "move camera/gripper unit up"
    assert "negative means move closer" in guidance["movement_convention"]["dz_sign"]


def test_hand_mounted_close_range_comes_from_config_without_calibration_claim():
    depth, mask = _bottle_scene()
    depth[mask > 0] = 210
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 210.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    tracking = {
        "target_clamp": _target_clamp(),
        "locked_patch_visible": True,
        "bottle_geometry_tracking_ok": True,
        "identity_measurement_current": True,
        "current_grip_width_px": 40.0,
    }

    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        tracking,
        depth.shape,
    )

    config = load_named_config("hand_mounted_camera.yaml")
    assert guidance["desired_close_distance_mm"] == (
        config["camera_to_hand_offset_mm"]["z"] + config["desired_contact_gap_mm"]
    ) == 180.0
    assert guidance["min_close_distance_mm"] == 160.0
    assert guidance["max_close_distance_mm"] == 220.0
    assert guidance["distance_status"] == "READY"
    assert guidance["close_distance_range_source"] == config["close_distance_range_source"]
    assert guidance["close_distance_range_validation_status"] == config[
        "close_distance_range_validation_status"
    ]
    assert guidance["close_distance_range_validation_status"] == "APPROXIMATE_NOT_PHYSICALLY_CALIBRATED"
    assert guidance["close_distance_range_physically_validated"] is False


def test_unverified_or_low_confidence_distance_is_diagnostic_not_actionable():
    depth = np.full((200, 200), 1200, dtype=np.uint16)
    clamp = _target_clamp()
    depth[90:111, 74:87] = 90
    depth[90:111, 114:127] = 250
    tracking = {
        "target_clamp": clamp,
        "current_depth_mm": 90.0,
        "locked_patch_visible": True,
        "bottle_geometry_tracking_ok": False,
        "current_grip_width_px": 40.0,
    }

    guidance = compute_approach_guidance(
        None,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        tracking,
        depth.shape,
    )

    assert guidance["bottle_distance_mm_raw"] == 90.0
    assert guidance["distance_status"] == "UNRELIABLE"
    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_ready"] is False
    assert guidance["in_close_range"] is False
    assert guidance["actionable_distance_mm"] is None
    assert guidance["dx_mm"] is None
    assert guidance["dy_mm"] is None
    assert guidance["dz_mm"] is None
    assert "distance_unreliable" in guidance["ready_blockers"]
    assert guidance["approach_action"] == "HOLD_STEADY_READY_VERIFY"


def test_pose_missing_depth_fallback_cannot_emit_actionable_guidance():
    depth = np.full((200, 200), 90, dtype=np.uint16)
    guidance = compute_approach_guidance(
        None,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "current_depth_mm": 90.0,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["bottle_distance_mm_raw"] == 90.0
    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "current_bottle_mask_not_available"
    assert guidance["distance_status"] == "UNRELIABLE"
    assert guidance["actionable_distance_mm"] is None
    assert guidance["dx_mm"] is None
    assert guidance["dy_mm"] is None
    assert guidance["dz_mm"] is None


def test_held_3d_band_cannot_use_bilateral_background_depth_as_guidance():
    depth = np.full((200, 200), 180, dtype=np.uint16)
    guidance = compute_approach_guidance(
        None,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {"planned_depth_mm": 185.0},
        {
            "target_clamp": _target_clamp(),
            "locked_projection_mode": "HELD_LAST_VALID_3D",
            "projection_mode": "HELD_LAST_VALID_3D",
            "locked_patch_visible": True,
            "current_depth_mm": 185.0,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "current_bottle_mask_not_available"
    assert guidance["distance_locked_local_patch_support_valid"] is False
    assert guidance["bottle_distance_mm_raw"] == 180.0
    assert guidance["distance_status"] == "UNRELIABLE"
    assert guidance["actionable_distance_mm"] is None


def test_missing_current_geometry_flags_fail_closed():
    depth, mask = _bottle_scene()
    depth[mask > 0] = 90
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 90.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "current_bottle_geometry_not_confirmed"
    assert guidance["actionable_distance_mm"] is None


def test_wrong_shape_bottle_mask_cannot_be_labeled_current_depth_evidence():
    depth = np.full((200, 200), 90, dtype=np.uint16)
    wrong_shape_mask = np.ones((1, 1), dtype=np.uint8)
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 90.0,
        "body_mask": wrong_shape_mask,
        "cleaned_mask": wrong_shape_mask,
    }
    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "bottle_geometry_tracking_ok": True,
            "identity_measurement_current": True,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["depth_evidence_source"] == "INVALID_BOTTLE_MASK_UNMASKED_DIAGNOSTIC"
    assert guidance["bottle_mask_usable_for_distance"] is False
    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "current_bottle_mask_unusable"
    assert guidance["actionable_distance_mm"] is None


def test_held_geometry_depth_is_diagnostic_until_current_measurement_returns():
    depth, mask = _bottle_scene()
    depth[mask > 0] = 90
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 90.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "bottle_geometry_tracking_ok": True,
            "identity_measurement_current": False,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "current_bottle_geometry_not_confirmed"
    assert guidance["distance_status"] == "UNRELIABLE"
    assert guidance["actionable_distance_mm"] is None


def test_tiny_mask_slivers_cannot_make_contact_distance_actionable():
    depth = np.full((200, 200), 90, dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[99:102, 79:82] = 1
    mask[99:102, 119:122] = 1
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 90.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "bottle_geometry_tracking_ok": True,
            "identity_measurement_current": True,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["distance_confidence"] == 1.0
    assert guidance["distance_mask_support_valid"] is False
    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "insufficient_masked_contact_support"
    assert guidance["distance_status"] == "UNRELIABLE"
    assert guidance["actionable_distance_mm"] is None


def test_mask_overlap_without_enough_valid_depth_pixels_is_not_actionable():
    depth = np.zeros((200, 200), dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[90:111, 74:88] = 1
    mask[90:111, 113:127] = 1
    depth[90:105, 75:76] = 90
    depth[90:105, 125:126] = 90
    pose = {
        "center_px": [100.0, 100.0],
        "depth_median_mm": 90.0,
        "body_mask": mask,
        "cleaned_mask": mask,
    }
    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "bottle_geometry_tracking_ok": True,
            "identity_measurement_current": True,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert guidance["left_patch_depth_eligible_pixels"] >= 30
    assert guidance["right_patch_depth_eligible_pixels"] >= 30
    assert guidance["left_patch_depth_valid_pixels"] < 30
    assert guidance["right_patch_depth_valid_pixels"] < 30
    assert guidance["distance_mask_support_valid"] is False
    assert guidance["distance_measurement_valid"] is False
    assert guidance["distance_validity_reason"] == "insufficient_masked_contact_support"


def test_current_geometry_contour_masks_background_for_distance_guidance():
    depth = np.full((200, 200), 1200, dtype=np.uint16)
    depth[35:166, 80:121] = 260
    geometry = {
        "detected": True,
        "depth_median_mm": 260.0,
        "contour": [[80, 35], [120, 35], [120, 165], [80, 165]],
    }
    pose = current_geometry_guidance_pose(geometry, True, depth.shape)

    guidance = compute_approach_guidance(
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": _target_clamp(),
            "locked_patch_visible": True,
            "bottle_geometry_tracking_ok": True,
            "identity_measurement_current": True,
            "current_grip_width_px": 40.0,
        },
        depth.shape,
    )

    assert pose["body_mask"] is not None
    assert guidance["depth_evidence_source"] == "CURRENT_BOTTLE_MASK_AND_ALIGNED_DEPTH"
    assert guidance["bottle_distance_mm_raw"] == 260.0
    assert guidance["distance_measurement_valid"] is True
    assert guidance["distance_status"] == "TOO_FAR"


def test_compact_live_status_keeps_close_range_and_depth_projection_diagnostics():
    compact = compact_live_status(
        {
            "state": "APPROACH_TRACKING",
            "distance_status": "READY",
            "distance_ready": True,
            "in_close_range": True,
            "desired_close_distance_mm": 210.0,
            "min_close_distance_mm": 180.0,
            "max_close_distance_mm": 240.0,
            "too_close_distance_mm": 160.0,
            "too_far_distance_mm": 280.0,
            "close_distance_range_source": "hand_mounted_live_observation",
            "close_distance_range_validation_status": "APPROXIMATE_NOT_PHYSICALLY_CALIBRATED",
            "close_distance_range_physically_validated": False,
            "bottle_geometry_tracking_ok": True,
            "identity_tracking_ok": True,
            "identity_tracking_source": "recent_held_rgbd_bottle_geometry",
            "identity_measurement_current": False,
            "identity_measurement_age_frames": 1,
            "contact_patch_tracking_ok": False,
            "contact_patch_tracking_reason": "depth_confidence_low",
            "bottle_geometry_measurement_status": "CURRENT_MEASURED_RGBD",
            "roi_depth_projection_scale": 2.0,
            "roi_projected_bbox_unclipped": [10.0, -20.0, 210.0, 300.0],
            "roi_projected_visible_fraction": 0.75,
            "roi_projected_extent_partial": True,
        }
    )

    assert compact["distance_status"] == "READY"
    assert compact["distance_ready"] is True
    assert compact["in_close_range"] is True
    assert compact["desired_close_distance_mm"] == 210.0
    assert compact["min_close_distance_mm"] == 180.0
    assert compact["max_close_distance_mm"] == 240.0
    assert compact["close_distance_range_validation_status"] == "APPROXIMATE_NOT_PHYSICALLY_CALIBRATED"
    assert compact["close_distance_range_physically_validated"] is False
    assert compact["bottle_geometry_tracking_ok"] is True
    assert compact["identity_tracking_ok"] is True
    assert compact["identity_tracking_source"] == "recent_held_rgbd_bottle_geometry"
    assert compact["identity_measurement_current"] is False
    assert compact["identity_measurement_age_frames"] == 1
    assert compact["contact_patch_tracking_ok"] is False
    assert compact["contact_patch_tracking_reason"] == "depth_confidence_low"
    assert compact["bottle_geometry_measurement_status"] == "CURRENT_MEASURED_RGBD"
    assert compact["roi_depth_projection_scale"] == 2.0
    assert compact["roi_projected_visible_fraction"] == 0.75
    assert compact["roi_projected_extent_partial"] is True


def test_post_lock_view_quality_treats_hand_reach_rgbd_measurement_as_detected():
    _depth, mask = _bottle_scene()
    quality = compute_view_quality(
        {
            "detected": True,
            "depth_median_mm": 260.0,
            "depth_confidence": 1.0,
            "detection_confidence": 1.0,
            "cleaned_mask": mask,
            "body_mask": mask,
        },
        {"full_bottle_visible": False, "visibility_score": 70.0},
        mask.shape,
        phase="post_lock",
        tracking={"locked_patch_visible": True},
    )

    assert quality["view_status"] != "SEARCH"
    assert quality["components"]["bottle_detected"] == 100.0
    assert quality["components"]["distance_range"] == 100.0
    assert quality["preferred_distance_range_mm"][0] == 160.0


def test_compact_status_keeps_configured_close_range_visible_while_searching():
    compact = compact_live_status({"state": "SELECT_FEASIBLE_TARGET"})

    assert compact["desired_close_distance_mm"] == 180.0
    assert compact["min_close_distance_mm"] == 160.0
    assert compact["max_close_distance_mm"] == 220.0
    assert compact["close_distance_range_source"] == "camera_to_hand_offset_plus_contact_gap_estimate"
    assert compact["close_distance_range_validation_status"] == "APPROXIMATE_NOT_PHYSICALLY_CALIBRATED"
    assert compact["close_distance_range_physically_validated"] is False

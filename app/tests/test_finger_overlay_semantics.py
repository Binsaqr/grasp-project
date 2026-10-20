"""Camera-free checks for honest bottle-band and detected-finger overlays."""

import numpy as np
import pytest

from grasp_project.realsense import realtime_runner


@pytest.fixture(autouse=True)
def enable_engineering_visual_finger_tests(monkeypatch):
    monkeypatch.setattr(
        realtime_runner,
        "ENABLE_VISUAL_FINGER_DETECTION_IN_DEMO",
        True,
    )


def test_final_demo_uses_fixed_mount_geometry_and_never_runs_skin_finger_detector(monkeypatch):
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "ENABLE_FINGER_DETECTION", True)
    monkeypatch.setattr(
        realtime_runner,
        "ENABLE_VISUAL_FINGER_DETECTION_IN_DEMO",
        False,
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_finger_in_roi",
        lambda *_args, **_kwargs: pytest.fail("skin detector must not run in final demo"),
    )
    target = {
        "left_contact_px": [80.0, 120.0],
        "right_contact_px": [120.0, 120.0],
        "clamp_band_polygon": [[80, 110], [120, 110], [120, 130], [80, 130]],
    }

    result = realtime_runner.detect_finger_contacts(
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320), 400, dtype=np.uint16),
        target,
        (240, 320, 3),
        {},
        reference_bottle_depth_mm=400.0,
        intrinsics={"fx": 200.0},
    )

    assert result["finger_detection_status"] == "DISABLED_FIXED_CAMERA_TO_HAND_GEOMETRY"
    assert result["left_finger_detected"] is False
    assert result["right_finger_detected"] is False
    assert result["contact_source"] == "planned_contact"
    assert result["quality_input_source"] == "planned_contact"
    assert result["finger_detection_source"] == (
        "fixed_camera_to_hand_geometry_no_visual_finger_authority"
    )


def _clamp():
    return {
        "left_clamp_patch": {
            "center_px": [80.0, 75.0],
            "polygon": [[74, 60], [86, 60], [86, 90], [74, 90]],
        },
        "right_clamp_patch": {
            "center_px": [120.0, 75.0],
            "polygon": [[114, 60], [126, 60], [126, 90], [114, 90]],
        },
        "left_contact_px": [80.0, 75.0],
        "right_contact_px": [120.0, 75.0],
        "clamp_center_px": [100.0, 75.0],
        "clamp_band_polygon": [[74, 58], [126, 58], [126, 92], [74, 92]],
        "grasp_line": {
            "left_endpoint_px": [80.0, 75.0],
            "right_endpoint_px": [120.0, 75.0],
        },
    }


def _report(finger=None, mode="presentation"):
    clamp = _clamp()
    return {
        "state": "APPROACH_TRACKING",
        "decision": "CONTINUE_APPROACH",
        "grasp_overlay_layer_mode": mode,
        "target_clamp": clamp,
        "finger_detection": finger or {},
        "target_lock": {
            "target_status": "TRACKING",
            "band_drift_state": "OK",
            "target_clamp": clamp,
            "current_tracked_target": {"display_target_clamp": clamp},
        },
    }


def _fake_finger(side):
    left = side == "left"
    return {
        "detected": True,
        "visual_detected": True,
        "bbox": [42, 61, 67, 89] if left else [133, 61, 158, 89],
        "contour": (
            [[42, 61], [67, 61], [67, 89], [42, 89]]
            if left
            else [[133, 61], [158, 61], [158, 89], [133, 89]]
        ),
        "contact_px": [80.0, 75.0] if left else [120.0, 75.0],
        "expected_contact_px": [80.0, 75.0] if left else [120.0, 75.0],
        "depth_mm": 600.0,
        "depth_verified_against_bottle": True,
        "depth_delta_to_bottle_mm": 0.0,
        "confidence": 92.0,
        "distance_to_expected_px": 0.0,
    }


def test_planned_band_is_not_drawn_as_finger_boxes():
    frame = np.zeros((150, 200, 3), dtype=np.uint8)
    report = _report()

    overlay = realtime_runner.draw_fast_overlay(frame, None, None, report)

    assert report["grasp_overlay_semantics"]["planned_contact"] == (
        "cyan_dashed_never_labeled_as_finger"
    )
    assert np.all(overlay[75, 42] == 0)
    assert np.all(overlay[75, 158] == 0)


def test_detected_fingers_draw_at_actual_finger_regions_only():
    frame = np.zeros((150, 200, 3), dtype=np.uint8)
    finger = {
        "left_finger_detected": True,
        "right_finger_detected": True,
        "left_finger_bbox": [42, 61, 67, 89],
        "right_finger_bbox": [133, 61, 158, 89],
        "left_inner_edge": [67, 75],
        "right_inner_edge": [133, 75],
        "left_expected_bottle_contact_px": [80, 75],
        "right_expected_bottle_contact_px": [120, 75],
        "left_contact_status": "APPROACHING",
        "right_contact_status": "APPROACHING",
    }

    overlay = realtime_runner.draw_fast_overlay(frame, None, None, _report(finger))

    assert tuple(overlay[75, 42]) == (0, 220, 255)
    assert tuple(overlay[75, 158]) == (220, 80, 190)


def test_bottle_only_layer_hides_real_finger_boxes():
    frame = np.zeros((150, 200, 3), dtype=np.uint8)
    finger = {
        "left_finger_detected": True,
        "right_finger_detected": True,
        "left_finger_bbox": [42, 61, 67, 89],
        "right_finger_bbox": [133, 61, 158, 89],
    }

    overlay = realtime_runner.draw_fast_overlay(
        frame, None, None, _report(finger, mode="bottle_only")
    )

    assert np.all(overlay[75, 42] == 0)
    assert np.all(overlay[75, 158] == 0)


def test_bilateral_contact_requires_persistent_depth_matched_fingers(monkeypatch):
    monkeypatch.setattr(
        realtime_runner,
        "detect_finger_in_roi",
        lambda _rgb, _depth, side, *_args, **_kwargs: _fake_finger(side),
    )
    monkeypatch.setattr(realtime_runner, "FINGER_STABLE_FRAMES_REQUIRED", 3)
    monkeypatch.setattr(realtime_runner, "FINGER_CONTACT_CONFIRM_GAP_PX", 6.0)
    rgb = np.zeros((150, 200, 3), dtype=np.uint8)
    depth = np.full((150, 200), 600, dtype=np.uint16)
    state = {}

    first = realtime_runner.detect_finger_contacts(
        rgb, depth, _clamp(), rgb.shape, state, reference_bottle_depth_mm=600.0
    )
    second = realtime_runner.detect_finger_contacts(
        rgb, depth, _clamp(), rgb.shape, state, reference_bottle_depth_mm=600.0
    )
    confirmed = realtime_runner.detect_finger_contacts(
        rgb, depth, _clamp(), rgb.shape, state, reference_bottle_depth_mm=600.0
    )

    assert first["contact_source"] == "visual_contact"
    assert first["left_contact_status"] in {"APPROACHING", "POSSIBLE_CONTACT"}
    assert first["left_contact_status"] != "DEPTH_CONTACT_CONFIRMED"
    assert second["contact_source"] == "visual_contact"
    assert confirmed["contact_source"] == "detected_gripper_contact"
    assert confirmed["left_contact_status"] == "DEPTH_CONTACT_CONFIRMED"
    assert confirmed["right_contact_status"] == "DEPTH_CONTACT_CONFIRMED"
    assert confirmed["left_finger_to_bottle_gap_px"] == 0.0
    assert confirmed["right_finger_to_bottle_gap_px"] == 0.0
    assert confirmed["physical_contact_validated"] is False


def test_unilateral_finger_keeps_bilateral_quality_source_planned(monkeypatch):
    def fake(_rgb, _depth, side, *_args, **_kwargs):
        return _fake_finger(side) if side == "left" else {"detected": False}

    monkeypatch.setattr(realtime_runner, "detect_finger_in_roi", fake)
    rgb = np.zeros((150, 200, 3), dtype=np.uint8)
    depth = np.full((150, 200), 600, dtype=np.uint16)
    result = realtime_runner.detect_finger_contacts(
        rgb, depth, _clamp(), rgb.shape, {}, reference_bottle_depth_mm=600.0
    )

    assert result["left_finger_detected"] is True
    assert result["right_finger_detected"] is False
    assert result["contact_source"] == "planned_contact"
    assert result["quality_input_source"] == "planned_contact"
    assert result["quality_source_display"] == "planned bottle contact"

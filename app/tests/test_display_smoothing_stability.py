"""Regression tests for green-bbox / magenta-clamp display stabilization.

Covers the 2026-07-08 stability upgrades: median prefilter against
single-frame measurement spikes, exact freeze on static scenes, snap
responsiveness on sustained motion, and cross-frame jitter telemetry.
"""

import copy

import numpy as np
import pytest

runner = pytest.importorskip("grasp_project.realsense.realtime_runner")


IMAGE_SHAPE = (480, 640)


class FakeLockManager:
    def __init__(self):
        self.display_smoothing_state = {}
        self.locked_target = {"lock_frame_id": 77}


def make_inputs(center_x, center_y, angle_deg=88.0, half_w=40.0, half_h=90.0):
    geometry = {
        "bbox_xyxy": [center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h],
        "axis_angle_degrees": angle_deg,
    }
    band = [
        [center_x - half_w, center_y - 10.0],
        [center_x + half_w, center_y - 10.0],
        [center_x + half_w, center_y + 10.0],
        [center_x - half_w, center_y + 10.0],
    ]
    target_clamp = {
        "clamp_center_px": [center_x, center_y],
        "clamp_band_polygon": band,
        "bottle_y_axis": [np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))],
        "bottle_y_axis_angle_deg": angle_deg,
        "magenta_box_angle_deg": angle_deg,
        "left_contact_px": [center_x - half_w, center_y],
        "right_contact_px": [center_x + half_w, center_y],
    }
    tracking = {
        "tracking_source": "roi_opencv_depth",
        "identity_measurement_current": True,
        "raw_bottle_angle_deg": angle_deg,
    }
    return geometry, target_clamp, tracking


def run_frames(manager, samples, start_frame=0):
    results = []
    for i, (cx, cy, ang) in enumerate(samples):
        geometry, clamp, tracking = make_inputs(cx, cy, ang)
        results.append(
            runner.apply_display_geometry_smoothing(
                manager, geometry, clamp, tracking, IMAGE_SHAPE, start_frame + i
            )
        )
    return results


def display_center(result):
    center = result.get("display_magenta_center")
    assert center is not None
    return np.asarray(center, dtype=float)


def test_single_frame_spike_does_not_move_boxes():
    manager = FakeLockManager()
    base = [(320.0, 240.0, 88.0)] * 5
    spike = [(335.0, 249.0, 79.0)]  # one bad detection frame
    settle = [(320.0, 240.0, 88.0)] * 3
    results = run_frames(manager, base + spike + settle)
    anchor = display_center(results[4])
    for r in results[5:]:
        assert float(np.linalg.norm(display_center(r) - anchor)) <= 2.0
        delta = r.get("display_magenta_angle_deg")
        assert abs(((delta - 88.0) + 90.0) % 180.0 - 90.0) <= 2.0
    bbox_before = np.asarray(results[4]["display_bottle_bbox"], dtype=float)
    bbox_spike = np.asarray(results[5]["display_bottle_bbox"], dtype=float)
    assert float(np.max(np.abs(bbox_spike - bbox_before))) <= 3.0


def test_static_scene_stays_pixel_still():
    manager = FakeLockManager()
    rng = np.random.default_rng(7)
    samples = [(320.0 + rng.uniform(-0.8, 0.8), 240.0 + rng.uniform(-0.8, 0.8), 88.0) for _ in range(12)]
    results = run_frames(manager, samples)
    anchor = display_center(results[4])
    for r in results[5:]:
        assert float(np.linalg.norm(display_center(r) - anchor)) == pytest.approx(0.0, abs=1e-9)
        assert r["display_smoothing_mode"] in {"static", "static_angle_reject"}


def test_sustained_step_still_converges_quickly():
    manager = FakeLockManager()
    before = [(320.0, 240.0, 88.0)] * 5
    after = [(360.0, 240.0, 88.0)] * 12
    results = run_frames(manager, before + after)
    final = display_center(results[-1])
    assert float(np.linalg.norm(final - np.asarray([360.0, 240.0]))) <= 3.0
    # responsiveness: within 4 frames of the sustained step the box must be
    # more than halfway there (lag guards snap on real motion)
    early = display_center(results[5 + 3])
    assert early[0] >= 340.0


def test_temporally_confirmed_in_place_rotation_does_not_deadlock_display_angle():
    manager = FakeLockManager()
    run_frames(manager, [(320.0, 240.0, 90.0)] * 5)
    results = []
    for frame_id in range(5, 17):
        geometry, clamp, tracking = make_inputs(320.0, 240.0, 60.0)
        tracking["orientation_update_confirmed"] = True
        clamp["orientation_update_confirmed"] = True
        results.append(
            runner.apply_display_geometry_smoothing(
                manager, geometry, clamp, tracking, IMAGE_SHAPE, frame_id
            )
        )

    assert any(result["display_smoothing_mode"] == "confirmed_in_place_rotation" for result in results)
    assert abs(runner.line_angle_delta_deg(results[-1]["display_magenta_angle_deg"], 60.0)) < 2.0
    assert all(result["display_smoothing_mode"] != "static_angle_reject" for result in results[1:])


def test_jitter_history_accumulates_across_frames():
    manager = FakeLockManager()
    results = run_frames(manager, [(320.0, 240.0, 88.0)] * 6)
    assert results[-1]["display_jitter_history_frames"] >= 5


@pytest.mark.parametrize("held_source", ["shifted_locked_geometry", "lost"])
def test_display_smoothing_state_survives_held_estimated_tracking(held_source):
    manager = FakeLockManager()
    measured = run_frames(manager, [(320.0, 240.0, 88.0)] * 4)
    history_before_hold = measured[-1]["display_jitter_history_frames"]

    held_results = []
    for frame_id, center_x in enumerate((324.0, 328.0), start=4):
        geometry, clamp, tracking = make_inputs(center_x, 240.0, 88.0)
        geometry.update(
            {
                "detected": False,
                "tracking_source": held_source,
                "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
            }
        )
        tracking.update(
            {
                "tracking_source": held_source,
                "tracking_ok": False,
                "bottle_geometry_tracking_ok": False,
                "identity_tracking_ok": False,
                "identity_measurement_current": False,
                "contact_patch_tracking_ok": False,
                "ready_to_close": False,
                "ready_blockers": ["current_bottle_geometry_not_measured"],
                "safety_score": 17.0,
            }
        )
        tracking_before = copy.deepcopy(tracking)

        result = runner.apply_display_geometry_smoothing(
            manager, geometry, clamp, tracking, IMAGE_SHAPE, frame_id
        )

        assert result["display_smoothing_reset_reason"] is None
        assert result["display_smoothing_mode"] != "reset"
        assert result["display_smoothing_active"] is True
        assert result["display_smoothing_only"] is True
        assert tracking == tracking_before
        for validity_key in (
            "tracking_ok",
            "bottle_geometry_tracking_ok",
            "identity_tracking_ok",
            "identity_measurement_current",
            "contact_patch_tracking_ok",
            "ready_to_close",
            "ready_blockers",
            "safety_score",
        ):
            assert validity_key not in result
        held_results.append(result)

    assert held_results[0]["display_jitter_history_frames"] > history_before_hold
    assert held_results[1]["display_jitter_history_frames"] > held_results[0]["display_jitter_history_frames"]
    anchor = display_center(measured[-1])
    assert np.array_equal(display_center(held_results[0]), anchor)
    assert np.array_equal(display_center(held_results[1]), anchor)
    assert all(result["display_locked_reference_frozen"] is True for result in held_results)
    assert all(
        result["display_smoothing_mode"] == "locked_reference_frozen_identity_unverified"
        for result in held_results
    )
    assert manager.display_smoothing_state["frame_id"] == 5


def test_missing_identity_provenance_is_never_treated_as_current_measurement():
    manager = FakeLockManager()
    geometry, clamp, tracking = make_inputs(320.0, 240.0, 88.0)
    tracking.pop("identity_measurement_current")

    result = runner.apply_display_geometry_smoothing(
        manager, geometry, clamp, tracking, IMAGE_SHAPE, 0
    )

    assert result["display_measurement_current"] is False
    assert result["display_held_estimate"] is True


def test_bounded_depth_motion_moves_guide_with_object_not_camera_pixels():
    manager = FakeLockManager()
    measured = run_frames(manager, [(320.0, 240.0, 88.0)] * 4)
    anchor = display_center(measured[-1])
    results = []
    for frame_id, center_x in enumerate((326.0, 332.0, 338.0, 344.0), start=4):
        geometry, clamp, tracking = make_inputs(center_x, 240.0, 88.0)
        tracking.update(
            {
                "tracking_source": "shifted_locked_geometry",
                "identity_measurement_current": False,
                "identity_tracking_ok": False,
                "local_motion_application_status": "BOUNDED_OCCLUSION_DEPTH_ESTIMATE",
            }
        )
        results.append(
            runner.apply_display_geometry_smoothing(
                manager, geometry, clamp, tracking, IMAGE_SHAPE, frame_id
            )
        )

    assert results[-1]["display_magenta_center"][0] > anchor[0] + 10.0
    assert results[-1]["display_magenta_center"][0] <= 344.0
    final_bbox = np.asarray(results[-1]["display_bottle_bbox"], dtype=float)
    assert float((final_bbox[0] + final_bbox[2]) * 0.5) > anchor[0] + 10.0
    assert all(result["display_locked_reference_frozen"] is False for result in results)
    assert all(result["display_bounded_object_motion_estimate"] is True for result in results)
    assert all(
        result["display_smoothing_mode"].startswith("object_relative_depth_estimate_")
        for result in results
    )


def test_display_smoothing_writer_change_still_resets(monkeypatch):
    manager = FakeLockManager()
    run_frames(manager, [(320.0, 240.0, 88.0)] * 3)
    monkeypatch.setattr(runner, "WRITER_SESSION_ID", f"{runner.WRITER_SESSION_ID}-replacement")
    geometry, clamp, tracking = make_inputs(322.0, 240.0, 88.0)

    result = runner.apply_display_geometry_smoothing(
        manager, geometry, clamp, tracking, IMAGE_SHAPE, 3
    )

    assert result["display_smoothing_reset_reason"] == "writer_session_changed"
    assert result["display_smoothing_mode"] == "reset"
    assert result["display_smoothing_active"] is False
    assert result["display_smoothing_only"] is True


def test_display_smoothing_lock_change_still_resets():
    manager = FakeLockManager()
    run_frames(manager, [(320.0, 240.0, 88.0)] * 3)
    manager.locked_target["lock_frame_id"] = 78
    geometry, clamp, tracking = make_inputs(322.0, 240.0, 88.0)

    result = runner.apply_display_geometry_smoothing(
        manager, geometry, clamp, tracking, IMAGE_SHAPE, 3
    )

    assert result["display_smoothing_reset_reason"] == "grasp_reference_changed"
    assert result["display_smoothing_mode"] == "reset"
    assert result["display_smoothing_active"] is False
    assert result["display_smoothing_only"] is True


def test_automatic_relock_with_same_grasp_reference_does_not_reset_contacts():
    manager = FakeLockManager()
    manager.locked_target["grasp_reference_id"] = "original-optimal-band"
    measured = run_frames(manager, [(320.0, 240.0, 88.0)] * 3)
    anchor = display_center(measured[-1])
    manager.locked_target["lock_frame_id"] = 120
    geometry, clamp, tracking = make_inputs(326.0, 240.0, 88.0)

    result = runner.apply_display_geometry_smoothing(
        manager, geometry, clamp, tracking, IMAGE_SHAPE, 3
    )

    assert result["display_smoothing_reset_reason"] is None
    assert result["display_smoothing_mode"] != "reset"
    assert float(np.linalg.norm(display_center(result) - anchor)) <= 6.0

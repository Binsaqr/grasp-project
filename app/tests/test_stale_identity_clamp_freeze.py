"""Camera-free regressions for stale-identity clamp drift.

Depth-only contact support may bridge a short RGB-D identity interruption, but it
must not become a new recursive coordinate frame or move an unverified target
indefinitely.
"""

import copy

import numpy as np

from grasp_project.realsense import realtime_runner as runner


IMAGE_SHAPE = (200, 200)
DEPTH_MM = 100


def _verified_clamp():
    center_y = 100.0
    top = center_y - 10.0
    bottom = center_y + 10.0
    return {
        "left_clamp_patch": {
            "center_px": [80.0, center_y],
            "polygon": [[74.0, top], [86.0, top], [86.0, bottom], [74.0, bottom]],
        },
        "right_clamp_patch": {
            "center_px": [120.0, center_y],
            "polygon": [[114.0, top], [126.0, top], [126.0, bottom], [114.0, bottom]],
        },
        "left_contact_px": [80.0, center_y],
        "right_contact_px": [120.0, center_y],
        "clamp_center_px": [100.0, center_y],
        "clamp_band_polygon": [
            [74.0, top],
            [126.0, top],
            [126.0, bottom],
            [74.0, bottom],
        ],
        "grasp_line": {
            "left_endpoint_px": [80.0, center_y],
            "right_endpoint_px": [120.0, center_y],
            "center_px": [100.0, center_y],
            "direction_px": [1.0, 0.0],
        },
        "grip_width_px": 40.0,
        "clamp_patch_height_px": 20.0,
        "clamp_patch_width_px": 12.0,
        "bottle_y_axis_px": [0.0, 1.0],
        "bottle_x_axis_px": [1.0, 0.0],
    }


def _same_depth_occluder(*, top=35, bottom=116):
    """A coherent non-bottle depth region that overlaps both contact patches."""
    depth = np.full(IMAGE_SHAPE, 900, dtype=np.uint16)
    depth[top:bottom, 60:141] = DEPTH_MM
    return depth


def _locked_target():
    return {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "lock_frame_id": 1,
        "locked_body_fraction": 0.5,
        "depth_mm_at_lock": float(DEPTH_MM),
        "movement_deadband_state": {},
    }


def _previous_tracking(
    verified,
    identity_age_frames,
    *,
    observed_history=None,
    hand_occlusion=True,
):
    tracking = {
        "target_clamp": copy.deepcopy(verified),
        "last_verified_rgbd_target_clamp": copy.deepcopy(verified),
        "last_verified_rgbd_target_clamp_frame_id": 0,
        "current_depth_mm": float(DEPTH_MM),
        "identity_measurement_current": False,
        "identity_measurement_age_frames": int(identity_age_frames),
        "roi_candidate_hand_occlusion": bool(hand_occlusion),
    }
    if observed_history is not None:
        tracking["local_motion_observed_center_history_px"] = copy.deepcopy(observed_history)
    return tracking


def _mark_identity_unverified(tracking, identity_age_frames, *, hand_occlusion=True):
    tracking["identity_measurement_current"] = False
    tracking["identity_measurement_age_frames"] = int(identity_age_frames)
    # Recorded fixtures use the legacy nominal 15 Hz age. Live tracking now
    # publishes monotonic seconds directly, so keep both representations
    # coherent when a test advances the recorded identity age.
    tracking["identity_measurement_age_sec"] = float(identity_age_frames) / 15.0
    tracking["roi_candidate_hand_occlusion"] = bool(hand_occlusion)
    runner.separate_identity_from_contact_tracking(
        tracking,
        bottle_geometry_tracking_ok=False,
    )


def _assert_same_clamp_coordinates(actual, expected):
    for key in ("clamp_center_px", "left_contact_px", "right_contact_px", "clamp_band_polygon"):
        np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(expected[key]))
    for side in ("left_clamp_patch", "right_clamp_patch"):
        np.testing.assert_array_equal(
            np.asarray(actual[side]["center_px"]),
            np.asarray(expected[side]["center_px"]),
        )
        np.testing.assert_array_equal(
            np.asarray(actual[side]["polygon"]),
            np.asarray(expected[side]["polygon"]),
        )
    for key in ("left_endpoint_px", "right_endpoint_px", "center_px"):
        np.testing.assert_array_equal(
            np.asarray(actual["grasp_line"][key]),
            np.asarray(expected["grasp_line"][key]),
        )


def test_stale_identity_coherent_depth_motion_freezes_clamp_then_expires_lock(monkeypatch):
    monkeypatch.setattr(runner, "STICKY_SINGLE_BOTTLE_LOCK", False)
    verified = _verified_clamp()
    locked_target = _locked_target()
    stale_age = runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + 1
    previous = _previous_tracking(
        verified,
        stale_age,
        hand_occlusion=False,
    )
    manager = runner.TargetLockManager(enabled=True)
    manager.locked_target = copy.deepcopy(locked_target)

    for frame_id in range(1, 8):
        _plan, tracking = runner.local_locked_patch_tracking(
            _same_depth_occluder(),
            locked_target,
            previous,
            IMAGE_SHAPE,
            frame_id=frame_id,
        )
        _mark_identity_unverified(
            tracking,
            stale_age + frame_id,
            hand_occlusion=False,
        )
        manager.apply_tracking_counters(tracking)
        previous = tracking

    _assert_same_clamp_coordinates(tracking["target_clamp"], verified)
    _assert_same_clamp_coordinates(tracking["last_verified_rgbd_target_clamp"], verified)
    assert tracking["identity_tracking_ok"] is False
    assert runner.target_lock_report(manager, tracking)["target_status"] == "LOST"
    assert manager.last_lost_target["bottle_target_id"] == "bottle_01"
    assert manager.last_lost_target["locked_body_fraction"] == 0.5
    assert runner.tracking_good_from_values(tracking, target_status=manager.status()) is False
    blockers = runner.ready_blockers_from_values(
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


def test_recent_identity_bridge_offset_is_bounded_from_immutable_verified_clamp():
    verified = _verified_clamp()
    locked_target = _locked_target()
    previous = _previous_tracking(
        verified,
        identity_age_frames=1,
        # The coherent depth neighborhood was already observed while the RGB-D
        # identity bridge began; every bridge frame is therefore actionable.
        observed_history=[[100.0, 86.0], [100.0, 86.0]],
        hand_occlusion=False,
    )
    verified_center = np.asarray(verified["clamp_center_px"], dtype=float)

    offsets = []
    for identity_age in range(1, runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + 1):
        _plan, tracking = runner.local_locked_patch_tracking(
            _same_depth_occluder(top=50, bottom=101),
            locked_target,
            previous,
            IMAGE_SHAPE,
            frame_id=identity_age,
        )
        _mark_identity_unverified(
            tracking,
            identity_age,
            hand_occlusion=False,
        )
        assert tracking["identity_tracking_ok"] is True
        current_center = np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float)
        offsets.append(float(np.linalg.norm(current_center - verified_center)))
        _assert_same_clamp_coordinates(
            tracking["last_verified_rgbd_target_clamp"],
            verified,
        )
        previous = tracking

    assert max(offsets) > 0.5 * runner.LOCAL_DEPTH_MOTION_RECENT_IDENTITY_MAX_OFFSET_PX
    assert max(offsets) <= runner.LOCAL_DEPTH_MOTION_RECENT_IDENTITY_MAX_OFFSET_PX


def test_expired_identity_with_hand_evidence_keeps_bounded_depth_projection():
    verified = _verified_clamp()
    locked_target = _locked_target()
    previous = _previous_tracking(
        verified,
        identity_age_frames=1,
        observed_history=[[100.0, 86.0], [100.0, 86.0]],
        hand_occlusion=True,
    )

    _plan, bridged = runner.local_locked_patch_tracking(
        _same_depth_occluder(top=50, bottom=101),
        locked_target,
        previous,
        IMAGE_SHAPE,
        frame_id=1,
    )
    assert not np.array_equal(
        np.asarray(bridged["target_clamp"]["clamp_center_px"]),
        np.asarray(verified["clamp_center_px"]),
    )
    _mark_identity_unverified(
        bridged,
        runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + 1,
        hand_occlusion=True,
    )

    _plan, estimated = runner.local_locked_patch_tracking(
        _same_depth_occluder(top=20, bottom=81),
        locked_target,
        bridged,
        IMAGE_SHAPE,
        frame_id=2,
    )

    estimated_center = np.asarray(estimated["target_clamp"]["clamp_center_px"], dtype=float)
    verified_center = np.asarray(verified["clamp_center_px"], dtype=float)
    assert float(np.linalg.norm(estimated_center - verified_center)) <= (
        runner.LOCAL_DEPTH_MOTION_OCCLUDED_MAX_OFFSET_PX
    )
    _assert_same_clamp_coordinates(
        estimated["last_verified_rgbd_target_clamp"],
        verified,
    )
    assert estimated["local_motion_application_status"] == "BOUNDED_OCCLUSION_DEPTH_ESTIMATE"
    assert estimated["identity_measurement_current"] is False


def test_stale_hand_occlusion_projection_stays_bounded_from_verified_clamp():
    verified = _verified_clamp()
    locked_target = _locked_target()
    stale_age = runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + 1
    previous = _previous_tracking(
        verified,
        stale_age,
        observed_history=[[100.0, 70.0], [100.0, 70.0]],
        hand_occlusion=True,
    )
    for frame_id in range(1, 20):
        _plan, tracking = runner.local_locked_patch_tracking(
            _same_depth_occluder(top=20, bottom=81),
            locked_target,
            previous,
            IMAGE_SHAPE,
            frame_id=frame_id,
        )
        _mark_identity_unverified(
            tracking,
            stale_age + frame_id,
            hand_occlusion=True,
        )
        current_center = np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float)
        verified_center = np.asarray(verified["clamp_center_px"], dtype=float)
        assert float(np.linalg.norm(current_center - verified_center)) <= (
            runner.LOCAL_DEPTH_MOTION_OCCLUDED_MAX_OFFSET_PX
        )
        _assert_same_clamp_coordinates(
            tracking["last_verified_rgbd_target_clamp"],
            verified,
        )
        previous = tracking

    assert tracking["local_motion_application_status"] == "BOUNDED_OCCLUSION_DEPTH_ESTIMATE"
    assert tracking["identity_measurement_current"] is False

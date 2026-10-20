"""Camera-free regressions for post-lock bottle tracking around a touching hand."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_project.realsense import realtime_runner
from grasp_project.realsense.realtime_runner import (
    MAX_DISPLAY_LAG_PX,
    TargetLockManager,
    apply_display_geometry_smoothing,
    contact_patch_depths,
    dynamic_post_lock_roi,
    fast_roi_bottle_pose,
    local_locked_patch_tracking,
    partial_hand_occlusion_rigid_motion_estimate,
    ready_blockers_from_values,
    reanchor_target_clamp_only_from_current_geometry,
    reanchor_target_clamp_to_current_edges,
    refresh_contact_tracking_from_final_clamp,
    roi_opencv_depth_bottle_tracking,
    separate_identity_from_contact_tracking,
    shifted_bottle_geometry,
    shifted_target_clamp,
    validate_roi_bottle_candidate,
)


FRAME_SHAPE = (240, 320)
BOTTLE_BBOX = [135.0, 40.0, 185.0, 200.0]
BOTTLE_DEPTH_MM = 200
BLUE_RGB = np.array([30, 80, 220], dtype=np.uint8)
SKIN_RGB = np.array([210, 145, 105], dtype=np.uint8)
SATURATED_SKIN_RGB = np.array([255, 35, 5], dtype=np.uint8)
DARK_GLOVE_RGB = np.array([25, 25, 25], dtype=np.uint8)


def _seed_geometry():
    return {
        "detected": True,
        "bbox_xyxy": list(BOTTLE_BBOX),
        "bbox_xywh": [135.0, 40.0, 50.0, 160.0],
        "bottle_extent_bbox_xyxy": list(BOTTLE_BBOX),
        "center_px": [160.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": float(BOTTLE_DEPTH_MM),
        "depth_confidence": 1.0,
    }


def _tracking_manager():
    seed = _seed_geometry()
    return SimpleNamespace(
        last_good_bottle_geometry=copy.deepcopy(seed),
        last_good_bottle_bbox=list(BOTTLE_BBOX),
        last_good_bottle_depth_mm=float(BOTTLE_DEPTH_MM),
        last_good_bottle_frame_id=1,
        last_good_raw_bottle_velocity_px=[0.0, 0.0],
        last_full_roi_contour_frame_id=None,
        roi_bottle_fail_frames=0,
        locked_target={"valid": True, "bottle_target_id": "bottle_01"},
        grasp_memory={"status": "LOCKED_BOTTLE_RELATIVE_GRASP", "locked_body_fraction": 0.5},
    )


def _scene(hand_placement=None):
    rgb = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
    depth = np.full(FRAME_SHAPE, 900, dtype=np.uint16)
    rgb[40:200, 135:185] = BLUE_RGB
    depth[40:200, 135:185] = BOTTLE_DEPTH_MM

    if hand_placement == "side":
        # An elongated finger/hand component touching the bottle's right side.
        rgb[92:122, 185:245] = SKIN_RGB
        depth[92:122, 185:245] = BOTTLE_DEPTH_MM
    elif hand_placement == "cap":
        # A palm across the cap, overlapping the top few bottle rows.
        rgb[10:55, 105:215] = SKIN_RGB
        depth[10:55, 105:215] = BOTTLE_DEPTH_MM
    elif hand_placement == "covered_cap":
        # Leave too little current bottle body for a trustworthy extent update.
        rgb[10:170, 105:215] = SKIN_RGB
        depth[10:170, 105:215] = BOTTLE_DEPTH_MM
    elif hand_placement == "full_cover":
        # Foreground hand removes the complete visible body candidate while
        # extending beyond the remembered bottle boundary.
        rgb[10:225, 105:215] = SKIN_RGB
        depth[10:225, 105:215] = BOTTLE_DEPTH_MM
    elif hand_placement == "bottom_same_depth":
        # The user's hand enters from the camera-side image edge and covers
        # only the bottle below the already locked grasp band.
        rgb[165:240, 105:215] = SKIN_RGB
        depth[165:240, 105:215] = BOTTLE_DEPTH_MM
    elif hand_placement == "bottom_foreground_skin":
        rgb[165:240, 105:215] = SKIN_RGB
        depth[165:240, 105:215] = BOTTLE_DEPTH_MM - 60
    elif hand_placement == "bottom_reaches_band":
        rgb[120:240, 105:215] = SKIN_RGB
        depth[120:240, 105:215] = BOTTLE_DEPTH_MM - 60
    elif hand_placement == "bottom_foreground_non_skin":
        rgb[165:240, 105:215] = np.array([70, 190, 70], dtype=np.uint8)
        depth[165:240, 105:215] = BOTTLE_DEPTH_MM - 60
    elif hand_placement == "bottom_skin_plus_separate_foreground":
        rgb[165:240, 105:215] = SKIN_RGB
        depth[165:240, 105:215] = BOTTLE_DEPTH_MM - 60
        rgb[140:152, 135:165] = np.array([70, 190, 70], dtype=np.uint8)
        depth[140:152, 135:165] = BOTTLE_DEPTH_MM - 60
    elif hand_placement == "bottom_skin_plus_dark_glove":
        # Exposed fingers and a dark glove can be separate RGB components even
        # though both belong to the same hand entering from the image bottom.
        rgb[165:240, 105:145] = SKIN_RGB
        depth[165:240, 105:145] = BOTTLE_DEPTH_MM - 60
        rgb[165:240, 157:215] = DARK_GLOVE_RGB
        depth[165:240, 157:215] = BOTTLE_DEPTH_MM - 60
    return rgb, depth


def _bbox_xyxy(geometry):
    if geometry.get("bbox_xyxy"):
        return [float(value) for value in geometry["bbox_xyxy"]]
    x, y, width, height = [float(value) for value in geometry["bbox_xywh"]]
    return [x, y, x + width, y + height]


def _bbox_size(bbox):
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _locked_clamp():
    return {
        "clamp_center_px": [160.0, 120.0],
        "left_contact_px": [135.0, 120.0],
        "right_contact_px": [185.0, 120.0],
        "grasp_line": {
            "left_endpoint_px": [135.0, 120.0],
            "right_endpoint_px": [185.0, 120.0],
            "center_px": [160.0, 120.0],
            "direction_px": [1.0, 0.0],
        },
        "grip_width_px": 50.0,
        "clamp_patch_height_px": 36.0,
        "clamp_patch_width_px": 12.0,
        "clamp_band_height_px": 36.0,
        "left_clamp_patch": {
            "center_px": [135.0, 120.0],
            "polygon": [[129.0, 102.0], [141.0, 102.0], [141.0, 138.0], [129.0, 138.0]],
        },
        "right_clamp_patch": {
            "center_px": [185.0, 120.0],
            "polygon": [[179.0, 102.0], [191.0, 102.0], [191.0, 138.0], [179.0, 138.0]],
        },
        "clamp_band_polygon": [
            [129.0, 102.0],
            [191.0, 102.0],
            [191.0, 138.0],
            [129.0, 138.0],
        ],
        "bottle_y_axis_px": [0.0, 1.0],
        "bottle_x_axis_px": [1.0, 0.0],
    }


def _expected_bottom_hand_manager():
    manager = _tracking_manager()
    protected_clamp = _locked_clamp()
    manager.locked_target.update({
        "attempt_id": "attempt-bottom-hand",
        "lock_type": "FULL_BOTTLE_LOCK",
        "full_bottle_lock": True,
        "locked_body_fraction": 0.5,
        "fixed_clamp_zone": copy.deepcopy(protected_clamp),
    })
    manager.current_tracked_target = {
        "target_clamp": copy.deepcopy(protected_clamp),
    }
    manager.grasp_memory.update({
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.5,
    })
    return manager


def _contact_clamp_at_y(center_y):
    top = float(center_y - 10.0)
    bottom = float(center_y + 10.0)
    return {
        "left_clamp_patch": {
            "center_px": [80.0, float(center_y)],
            "polygon": [[74.0, top], [86.0, top], [86.0, bottom], [74.0, bottom]],
        },
        "right_clamp_patch": {
            "center_px": [120.0, float(center_y)],
            "polygon": [[114.0, top], [126.0, top], [126.0, bottom], [114.0, bottom]],
        },
        "left_contact_px": [80.0, float(center_y)],
        "right_contact_px": [120.0, float(center_y)],
        "clamp_center_px": [100.0, float(center_y)],
        "clamp_band_polygon": [[74.0, top], [126.0, top], [126.0, bottom], [74.0, bottom]],
        "grasp_line": {
            "left_endpoint_px": [80.0, float(center_y)],
            "right_endpoint_px": [120.0, float(center_y)],
            "center_px": [100.0, float(center_y)],
            "direction_px": [1.0, 0.0],
        },
        "grip_width_px": 40.0,
        "clamp_patch_height_px": 20.0,
        "clamp_patch_width_px": 12.0,
        "bottle_y_axis_px": [0.0, 1.0],
        "bottle_x_axis_px": [1.0, 0.0],
    }


def _motion_depth(offset_x=0, occluder_side=None):
    depth = np.full((200, 200), 900, dtype=np.uint16)
    x1 = 70 + int(offset_x)
    depth[80:121, x1 : x1 + 61] = 100
    if occluder_side == "left":
        depth[85:116, 20:70] = 100
    elif occluder_side == "right":
        depth[85:116, 131:181] = 100
    return depth


def _run_held_local_motion_sequence(depth_frames):
    verified_clamp = _contact_clamp_at_y(100.0)
    previous = {
        "target_clamp": copy.deepcopy(verified_clamp),
        "last_verified_rgbd_target_clamp": copy.deepcopy(verified_clamp),
        "last_verified_rgbd_target_clamp_frame_id": 0,
        "current_depth_mm": 100.0,
        "roi_candidate_hand_occlusion": True,
    }
    locked_target = {
        "lock_frame_id": 1,
        "locked_body_fraction": 0.5,
        "depth_mm_at_lock": 100.0,
        "movement_deadband_state": {},
    }
    display_manager = SimpleNamespace(
        display_smoothing_state={},
        locked_target=locked_target,
    )
    base_geometry = {
        "bbox_xyxy": [70.0, 25.0, 130.0, 175.0],
        "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
        "detected": False,
        "axis_angle_degrees": 90.0,
    }
    base_display = apply_display_geometry_smoothing(
        display_manager,
        base_geometry,
        verified_clamp,
        {"tracking_source": "shifted_locked_geometry"},
        (200, 200),
        0,
    )
    trackings = []
    displays = [base_display]
    for frame_id, depth in enumerate(depth_frames, start=1):
        _plan, tracking = local_locked_patch_tracking(
            depth,
            locked_target,
            previous,
            depth.shape,
            frame_id=frame_id,
        )
        tracking["roi_candidate_hand_occlusion"] = True
        tracking["tracking_source"] = "shifted_locked_geometry"
        delta = np.asarray(tracking["tracking_delta_this_frame_px"], dtype=float)
        tracking["center_velocity_vector_px_per_frame"] = delta.tolist()
        tracking["center_velocity_px_per_frame"] = float(np.linalg.norm(delta))
        center = np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float)
        geometry = {
            "bbox_xyxy": [center[0] - 30.0, 25.0, center[0] + 30.0, 175.0],
            "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
            "detected": False,
            "axis_angle_degrees": 90.0,
        }
        displays.append(
            apply_display_geometry_smoothing(
                display_manager,
                geometry,
                tracking["target_clamp"],
                tracking,
                depth.shape,
                frame_id,
            )
        )
        trackings.append(tracking)
        previous = tracking
    return verified_clamp, trackings, displays


@pytest.mark.parametrize("hand_placement", ["side", "cap"])
def test_fast_roi_excludes_touching_skin_without_stretching_bottle(hand_placement):
    clean_rgb, clean_depth = _scene()
    obscured_rgb, obscured_depth = _scene(hand_placement)

    clean, clean_error, _ = fast_roi_bottle_pose(
        clean_rgb,
        clean_depth,
        _seed_geometry(),
        scale=1.0,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )
    obscured, obscured_error, _ = fast_roi_bottle_pose(
        obscured_rgb,
        obscured_depth,
        _seed_geometry(),
        scale=1.0,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )

    assert clean_error is None and clean is not None
    assert obscured_error is None and obscured is not None
    clean_bbox = _bbox_xyxy(clean)
    obscured_bbox = _bbox_xyxy(obscured)

    # Occlusion may trim the hidden cap/side, but skin must never enlarge the
    # current bottle extent beyond the clean RGB-D component.
    assert obscured_bbox[0] >= clean_bbox[0] - 3.0
    assert obscured_bbox[1] >= clean_bbox[1] - 3.0
    assert obscured_bbox[2] <= clean_bbox[2] + 3.0
    assert obscured_bbox[3] <= clean_bbox[3] + 3.0
    assert obscured["depth_median_mm"] == pytest.approx(BOTTLE_DEPTH_MM)
    assert obscured["candidate_score_breakdown"]["skin_exclusion_applied"] is True
    assert obscured["candidate_score_breakdown"]["skin_exclusion_pixel_count"] > 0


def test_fast_roi_does_not_treat_different_depth_warm_bottle_as_hand_occlusion():
    rgb, depth = _scene()
    # A warm-coloured neighbouring bottle touches the target in RGB but is
    # clearly behind it in aligned depth.
    rgb[70:190, 185:235] = SKIN_RGB
    depth[70:190, 185:235] = 800

    candidate, error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        _seed_geometry(),
        scale=0.5,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )

    assert error is None and candidate is not None
    breakdown = candidate["candidate_score_breakdown"]
    assert breakdown["raw_skin_color_pixel_count"] > 0
    assert breakdown["skin_depth_gate_applied"] is True
    assert breakdown["skin_exclusion_pixel_count"] == 0
    assert candidate["skin_occlusion_on_seed"] is False
    accepted, telemetry = validate_roi_bottle_candidate(
        candidate,
        _seed_geometry(),
        image_shape=(*FRAME_SHAPE, 3),
        current_patch_depth_mm=BOTTLE_DEPTH_MM,
    )
    assert accepted is True
    assert "hand_skin_occludes_locked_bottle_boundary" not in telemetry["roi_candidate_reject_reason"]


def test_demo_fast_roi_rejects_warm_hand_colored_depth_component():
    rgb = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
    depth = np.full(FRAME_SHAPE, 900, dtype=np.uint16)
    rgb[40:200, 135:185] = SKIN_RGB
    depth[40:200, 135:185] = BOTTLE_DEPTH_MM

    candidate, error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        _seed_geometry(),
        scale=0.5,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )

    assert candidate is None
    assert error == "fast_roi_known_blue_rgbd_support_too_small"


def test_fast_roi_retains_internal_warm_bottle_label():
    rgb, depth = _scene()
    rgb[90:135, 145:175] = SKIN_RGB

    candidate, error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        _seed_geometry(),
        scale=0.5,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )

    assert error is None and candidate is not None
    breakdown = candidate["candidate_score_breakdown"]
    assert breakdown["same_depth_skin_color_pixel_count"] > 0
    assert breakdown["skin_qualified_component_count"] == 0
    assert breakdown["skin_exclusion_pixel_count"] == 0
    assert candidate["skin_occlusion_on_seed"] is False
    width, height = _bbox_size(_bbox_xyxy(candidate))
    assert width == pytest.approx(50.0, abs=4.0)
    assert height == pytest.approx(160.0, abs=4.0)


def test_saturated_finger_above_cap_is_excluded_from_bottle_extent():
    rgb, depth = _scene()
    rgb[10:55, 150:171] = SATURATED_SKIN_RGB
    depth[10:55, 150:171] = BOTTLE_DEPTH_MM

    candidate, error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        _seed_geometry(),
        scale=0.5,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )

    assert error is None and candidate is not None
    assert candidate["skin_exclusion_applied"] is True
    assert candidate["skin_overlap_seed_pixels"] >= 8
    assert _bbox_xyxy(candidate)[1] >= BOTTLE_BBOX[1] - 3.0


def test_fast_roi_skin_overlap_uses_scaled_seed_coordinates():
    rgb, depth = _scene()
    # Same-depth skin-like pixels are intentionally far to the right of the
    # target. At scale=0.5 the old code indexed the half-size mask with this
    # full-size seed box and falsely counted this region as overlapping.
    rgb[80:200, 270:315] = SKIN_RGB
    depth[80:200, 270:315] = BOTTLE_DEPTH_MM

    candidate, error, _ = fast_roi_bottle_pose(
        rgb,
        depth,
        _seed_geometry(),
        scale=0.5,
        depth_hint_mm=BOTTLE_DEPTH_MM,
    )

    assert error is None and candidate is not None
    breakdown = candidate["candidate_score_breakdown"]
    assert breakdown["same_depth_skin_color_pixel_count"] > 0
    assert breakdown["skin_qualified_component_count"] == 0
    assert breakdown["skin_exclusion_pixel_count"] == 0
    assert candidate["skin_overlap_seed_pixels"] == 0
    assert candidate["skin_overlap_seed_ratio"] == pytest.approx(0.0)
    assert candidate["skin_occlusion_on_seed"] is False


@pytest.mark.parametrize("hand_placement", ["side", "cap"])
def test_post_lock_tracker_cleans_or_holds_touching_hand_without_resizing_grasp(hand_placement):
    manager = _tracking_manager()
    original_geometry = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb, depth = _scene(hand_placement)

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    bbox = _bbox_xyxy(geometry)
    seed_width, seed_height = _bbox_size(BOTTLE_BBOX)
    width, height = _bbox_size(bbox)
    assert width <= seed_width + 4.0
    assert height <= seed_height + 4.0

    if telemetry["identity_measurement_current"]:
        assert telemetry["tracking_source"] == "roi_opencv_depth"
        assert geometry["measurement_status"] == "CURRENT_MEASURED_RGBD"
    else:
        assert telemetry["tracking_source"] in {
            "shifted_locked_geometry",
            "partial_rgbd_hand_occlusion_motion_estimate",
        }
        if telemetry["tracking_source"] == "partial_rgbd_hand_occlusion_motion_estimate":
            assert geometry["measurement_status"] == (
                "CURRENT_PARTIAL_RGBD_RIGID_MOTION_ESTIMATE_NOT_IDENTITY_VALIDATED"
            )
            assert telemetry["motion_estimate_current"] is True
            assert telemetry["motion_estimate_physically_validated"] is False
            assert telemetry["motion_estimate_rigid_extent_preserved"] is True
        else:
            assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
            assert bbox == BOTTLE_BBOX
        assert manager.last_good_bottle_geometry == original_geometry

    clamp, _trace = reanchor_target_clamp_only_from_current_geometry(
        _locked_clamp(),
        geometry,
        {
            "locked_body_fraction": 0.5,
            "bottle_contact_model": {"patch_height_mm": 14.0, "patch_width_mm": 8.0},
        },
        depth,
        rgb.shape,
        {"fx": 500.0, "fy": 500.0},
        geometry_measurement_current=telemetry["identity_measurement_current"],
    )
    assert clamp["grip_width_px"] <= 54.0


@pytest.mark.parametrize(
    "hand_placement,foreground_expected",
    [
        ("bottom_same_depth", False),
        ("bottom_foreground_skin", True),
        ("bottom_skin_plus_dark_glove", True),
    ],
)
def test_expected_bottom_hand_below_locked_band_stays_current_partial_beyond_hold_limit(
    hand_placement,
    foreground_expected,
    monkeypatch,
):
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.HAND_MOUNTED_CAMERA_MODE",
        True,
    )
    manager = _expected_bottom_hand_manager()
    original_full_anchor = copy.deepcopy(manager.last_good_bottle_geometry)
    original_locked_target = copy.deepcopy(manager.locked_target)
    original_tracked_target = copy.deepcopy(manager.current_tracked_target)
    original_body_fraction = manager.grasp_memory["locked_body_fraction"]
    rgb, depth = _scene(hand_placement)

    frame_ids = list(range(2, 35))
    assert len(frame_ids) > 30
    for frame_id in frame_ids:
        geometry, telemetry = roi_opencv_depth_bottle_tracking(
            rgb,
            depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
        )

        assert telemetry["identity_measurement_current"] is True
        assert telemetry["tracking_source"] == "roi_opencv_depth"
        assert telemetry["expected_bottom_hand_occlusion"] is True
        assert telemetry["expected_hand_bottom_connected"] is True
        assert telemetry["expected_hand_overlap_below_locked_band"] is True
        assert telemetry["expected_hand_locked_band_mask_supported"] is True
        assert telemetry["roi_candidate_foreground_occlusion_on_seed"] is foreground_expected
        assert telemetry["expected_hand_full_lock_continuity"] is True
        assert telemetry["expected_hand_close_proximity"] is True
        assert telemetry["expected_hand_self_occlusion_physically_validated"] is False
        if hand_placement == "bottom_skin_plus_dark_glove":
            assert telemetry["roi_candidate_foreground_occlusion_explained_by_skin"] is False
            assert telemetry["roi_candidate_foreground_unexplained_component_count"] == 1
            assert telemetry["expected_hand_foreground_bottom_entry_supported"] is True
            assert telemetry["expected_hand_foreground_unsupported_component_count"] == 0
        assert telemetry["roi_candidate_reject_reason"] == []
        assert geometry["partial_extent"] is True
        assert geometry["expected_bottom_hand_occlusion"] is True
        assert geometry["measurement_status"] == (
            "CURRENT_PARTIAL_RGBD_IDENTITY_EXPECTED_BOTTOM_HAND_OCCLUSION"
        )

        # A current partial identity can keep tracking indefinitely, but it
        # must not become the next full-extent projection/identity anchor.
        assert manager.last_good_bottle_geometry == original_full_anchor
        assert manager.last_good_bottle_bbox == BOTTLE_BBOX
        assert manager.locked_target == original_locked_target
        assert manager.current_tracked_target == original_tracked_target
        assert manager.grasp_memory["locked_body_fraction"] == original_body_fraction


def test_bottom_hand_reaching_locked_band_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.HAND_MOUNTED_CAMERA_MODE",
        True,
    )
    manager = _expected_bottom_hand_manager()
    original_full_anchor = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb, depth = _scene("bottom_reaches_band")

    _geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["expected_bottom_hand_occlusion"] is False
    assert telemetry["expected_hand_overlap_below_locked_band"] is False
    assert "hand_skin_occludes_locked_bottle_boundary" in telemetry["roi_candidate_reject_reason"]
    assert manager.last_good_bottle_geometry == original_full_anchor


def test_non_skin_closer_bottom_foreground_is_rejected_without_anchor_promotion(monkeypatch):
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.HAND_MOUNTED_CAMERA_MODE",
        True,
    )
    manager = _expected_bottom_hand_manager()
    original_full_anchor = copy.deepcopy(manager.last_good_bottle_geometry)
    original_locked_target = copy.deepcopy(manager.locked_target)
    rgb, depth = _scene("bottom_foreground_non_skin")

    _geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["expected_bottom_hand_occlusion"] is False
    assert telemetry["roi_candidate_foreground_occlusion_on_seed"] is True
    assert "unclassified_foreground_occludes_locked_bottle" in telemetry[
        "roi_candidate_reject_reason"
    ]
    assert manager.last_good_bottle_geometry == original_full_anchor
    assert manager.locked_target == original_locked_target


def test_bottom_skin_does_not_explain_a_separate_closer_foreground_object(monkeypatch):
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.HAND_MOUNTED_CAMERA_MODE",
        True,
    )
    manager = _expected_bottom_hand_manager()
    original_full_anchor = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb, depth = _scene("bottom_skin_plus_separate_foreground")

    _geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["expected_bottom_hand_occlusion"] is False
    assert telemetry["roi_candidate_foreground_occlusion_on_seed"] is True
    assert telemetry["roi_candidate_foreground_occlusion_explained_by_skin"] is False
    assert telemetry["roi_candidate_foreground_unexplained_component_count"] == 1
    assert "unclassified_foreground_occludes_locked_bottle" in telemetry[
        "roi_candidate_reject_reason"
    ]
    assert manager.last_good_bottle_geometry == original_full_anchor


def test_bottom_skin_is_not_special_cased_outside_hand_mounted_mode(monkeypatch):
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.HAND_MOUNTED_CAMERA_MODE",
        False,
    )
    manager = _expected_bottom_hand_manager()
    original_full_anchor = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb, depth = _scene("bottom_same_depth")

    _geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["expected_bottom_hand_occlusion"] is False
    assert telemetry["expected_bottom_hand_occlusion_reason"] == "hand_mounted_mode_disabled"
    assert "hand_skin_occludes_locked_bottle_boundary" in telemetry["roi_candidate_reject_reason"]
    assert manager.last_good_bottle_geometry == original_full_anchor


def test_bottom_trim_motion_estimate_preserves_original_vertical_center_and_extent():
    seed = _seed_geometry()
    candidate = {
        "bbox_xyxy": [135.0, 40.0, 185.0, 180.0],
        "bbox_xywh": [135.0, 40.0, 50.0, 140.0],
        "oriented_bbox": [[135.0, 40.0], [185.0, 40.0], [185.0, 180.0], [135.0, 180.0]],
        "contour": [[135.0, 40.0], [185.0, 40.0], [185.0, 180.0], [135.0, 180.0]],
        "center_px": [160.0, 110.0],
        "axis_vector": [0.0, 1.0],
        "skin_exclusion_center_px": [160.0, 205.0],
        "depth_median_mm": float(BOTTLE_DEPTH_MM),
        "depth_confidence": 1.0,
        "detection_confidence": 0.9,
    }

    estimate, trace = partial_hand_occlusion_rigid_motion_estimate(
        seed,
        candidate,
        (*FRAME_SHAPE, 3),
        frame_id=2,
    )

    assert estimate is not None
    assert estimate["center_px"] == pytest.approx([160.0, 120.0], abs=0.5)
    assert _bbox_xyxy(estimate) == pytest.approx(BOTTLE_BBOX, abs=0.5)
    assert trace["motion_estimate_step_translation_px"] == pytest.approx([0.0, 0.0], abs=0.5)
    assert trace["motion_estimate_boundary_anchor_modes"][0] == (
        "boundary_opposite_measured_hand"
    )


def test_partial_hand_motion_keeps_full_extent_center_and_follows_tilt():
    seed = _seed_geometry()
    seed["oriented_bbox"] = [
        [135.0, 40.0],
        [185.0, 40.0],
        [185.0, 200.0],
        [135.0, 200.0],
    ]
    seed["contour"] = copy.deepcopy(seed["oriented_bbox"])
    true_center = np.array([168.0, 96.0], dtype=float)
    angle_deg = 18.0
    theta = np.radians(angle_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    # The current hand-trimmed contour is missing part of the cap end. Its
    # visible center is not the bottle center, so a naive bbox-center update
    # would move the grasp band along the bottle axis.
    visible_local = np.array(
        [[-25.0, -50.0], [25.0, -50.0], [25.0, 80.0], [-25.0, 80.0]],
        dtype=float,
    )
    visible = visible_local @ rotation.T + true_center
    candidate = {
        "bbox_xyxy": [
            float(np.min(visible[:, 0])),
            float(np.min(visible[:, 1])),
            float(np.max(visible[:, 0])),
            float(np.max(visible[:, 1])),
        ],
        "center_px": np.mean(visible, axis=0).tolist(),
        "axis_vector": (-rotation[:, 1]).tolist(),
        "contour": visible.tolist(),
        "skin_exclusion_center_px": (true_center + rotation[:, 1] * -65.0).tolist(),
        "depth_median_mm": float(BOTTLE_DEPTH_MM),
        "depth_confidence": 1.0,
        "detection_confidence": 0.9,
    }

    estimate, trace = partial_hand_occlusion_rigid_motion_estimate(
        seed,
        candidate,
        (*FRAME_SHAPE, 3),
        frame_id=2,
    )

    assert estimate is not None
    assert estimate["detected"] is False
    assert estimate["physically_validated"] is False
    assert estimate["rigid_extent_preserved"] is True
    assert np.allclose(estimate["center_px"], true_center, atol=0.75)
    assert abs(float(estimate["axis_angle_degrees"]) - 108.0) < 0.75
    oriented = np.asarray(estimate["oriented_bbox"], dtype=float)
    edge_lengths = sorted(
        [float(np.linalg.norm(oriented[(index + 1) % 4] - oriented[index])) for index in range(4)]
    )
    assert edge_lengths == pytest.approx([50.0, 50.0, 160.0, 160.0], abs=0.75)
    assert trace["motion_estimate_boundary_anchor_modes"][0] == "boundary_opposite_measured_hand"


def test_moved_bottle_with_touching_hand_uses_motion_estimate_without_identity_claim():
    manager = _tracking_manager()
    original_geometry = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
    depth = np.full(FRAME_SHAPE, 900, dtype=np.uint16)
    rgb[22:182, 135:185] = BLUE_RGB
    depth[22:182, 135:185] = BOTTLE_DEPTH_MM
    rgb[72:106, 185:245] = SKIN_RGB
    depth[72:106, 185:245] = BOTTLE_DEPTH_MM

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["tracking_source"] == "partial_rgbd_hand_occlusion_motion_estimate"
    assert telemetry["identity_measurement_current"] is False
    assert telemetry["motion_estimate_current"] is True
    assert geometry["detected"] is False
    assert geometry["center_px"][1] == pytest.approx(102.0, abs=3.0)
    assert _bbox_size(_bbox_xyxy(geometry)) == pytest.approx(_bbox_size(BOTTLE_BBOX), abs=1.0)
    assert manager.last_good_bottle_geometry == original_geometry
    assert manager.last_partial_hand_motion_geometry is not None


def test_large_lateral_hand_motion_is_rejected_outside_identity_anchor_bound(monkeypatch):
    manager = _tracking_manager()
    original_geometry = copy.deepcopy(manager.last_good_bottle_geometry)
    candidate = {
        "detected": True,
        "bbox_xyxy": [220.0, 40.0, 270.0, 200.0],
        "bbox_xywh": [220.0, 40.0, 50.0, 160.0],
        "oriented_bbox": [[220.0, 40.0], [270.0, 40.0], [270.0, 200.0], [220.0, 200.0]],
        "contour": [[220.0, 40.0], [270.0, 40.0], [270.0, 200.0], [220.0, 200.0]],
        "center_px": [245.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": float(BOTTLE_DEPTH_MM),
        "depth_confidence": 1.0,
        "detection_confidence": 0.9,
        "skin_occlusion_on_seed": True,
        "skin_exclusion_center_px": [275.0, 120.0],
    }
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.detect_roi_bottle_candidate",
        lambda *_args, **_kwargs: (copy.deepcopy(candidate), [100, 10, 310, 225], None, {}, {}),
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.validate_roi_bottle_candidate",
        lambda *_args, **_kwargs: (
            False,
            {
                "roi_candidate_hand_occlusion": True,
                "roi_candidate_reject_reason": [
                    "bbox_iou_too_low",
                    "center_shift_too_large",
                    "aspect_ratio_changed",
                    "hand_skin_occludes_locked_bottle_boundary",
                ],
            },
        ),
    )
    rgb, depth = _scene()

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["tracking_source"] == "shifted_locked_geometry"
    assert telemetry["motion_estimate_current"] is False
    assert telemetry["motion_estimate_reject_reason"] == "anchor_relative_translation_limit_exceeded"
    assert geometry["center_px"] == pytest.approx(original_geometry["center_px"], abs=0.75)
    assert manager.last_good_bottle_geometry == original_geometry
    assert manager.last_partial_hand_motion_velocity_px == [0.0, 0.0]


def test_bounded_partial_rgbd_motion_cannot_exceed_identity_anchor_bound(monkeypatch):
    manager = _tracking_manager()
    original_geometry = copy.deepcopy(manager.last_good_bottle_geometry)
    candidate = {
        "detected": True,
        "bbox_xyxy": [220.0, 60.0, 250.0, 180.0],
        "bbox_xywh": [220.0, 60.0, 30.0, 120.0],
        "oriented_bbox": [[220.0, 60.0], [250.0, 60.0], [250.0, 180.0], [220.0, 180.0]],
        "contour": [[220.0, 60.0], [250.0, 60.0], [250.0, 180.0], [220.0, 180.0]],
        "center_px": [235.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": float(BOTTLE_DEPTH_MM),
        "depth_confidence": 1.0,
        "detection_confidence": 0.9,
    }
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.detect_roi_bottle_candidate",
        lambda *_args, **_kwargs: (copy.deepcopy(candidate), [100, 10, 300, 225], None, {}, {}),
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.validate_roi_bottle_candidate",
        lambda *_args, **_kwargs: (
            False,
            {
                "roi_candidate_hand_occlusion": False,
                "roi_candidate_reject_reason": [
                    "bbox_iou_too_low",
                    "center_shift_too_large",
                    "candidate_too_small",
                ],
            },
        ),
    )
    rgb, depth = _scene()

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["tracking_source"] == "shifted_locked_geometry"
    assert telemetry["identity_measurement_current"] is False
    assert telemetry["motion_estimate_current"] is False
    assert telemetry["motion_estimate_reject_reason"] == "anchor_relative_translation_limit_exceeded"
    assert geometry["detected"] is False
    assert geometry["center_px"] == pytest.approx(original_geometry["center_px"], abs=0.75)
    assert _bbox_size(_bbox_xyxy(geometry)) == pytest.approx(_bbox_size(BOTTLE_BBOX), abs=1.0)
    assert manager.last_good_bottle_geometry == original_geometry


def test_repeated_partial_motion_cannot_walk_past_identity_anchor(
    monkeypatch,
):
    manager = _tracking_manager()
    manager.last_identity_observation_monotonic = realtime_runner.time.monotonic()
    call_index = {"value": 0}

    def candidate_for_call(*_args, **_kwargs):
        call_index["value"] += 1
        dx = float(call_index["value"] * 5)
        candidate = _seed_geometry()
        candidate["center_px"] = [160.0 + dx, 120.0]
        candidate["bbox_xyxy"] = [135.0 + dx, 40.0, 185.0 + dx, 200.0]
        candidate["bbox_xywh"] = [135.0 + dx, 40.0, 50.0, 160.0]
        candidate["oriented_bbox"] = [
            [135.0 + dx, 40.0],
            [185.0 + dx, 40.0],
            [185.0 + dx, 200.0],
            [135.0 + dx, 200.0],
        ]
        candidate["contour"] = copy.deepcopy(candidate["oriented_bbox"])
        return candidate, [80, 10, 300, 225], None, {}, {}

    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.detect_roi_bottle_candidate",
        candidate_for_call,
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.validate_roi_bottle_candidate",
        lambda *_args, **_kwargs: (
            False,
            {
                "roi_candidate_hand_occlusion": False,
                "roi_candidate_reject_reason": ["center_shift_too_large"],
            },
        ),
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.partial_hand_occlusion_rigid_motion_estimate",
        lambda _seed, candidate, *_args, **_kwargs: (
            copy.deepcopy(candidate),
            {"motion_estimate_step_magnitude_px": 5.0},
        ),
    )
    rgb, depth = _scene()
    accepted_centers = []

    for frame_id in range(2, 52):
        _geometry, telemetry = roi_opencv_depth_bottle_tracking(
            rgb,
            depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
        )
        if telemetry.get("motion_estimate_current") is True:
            accepted_centers.append(float(telemetry["motion_estimate_current_center_px"][0]))

    assert accepted_centers
    assert max(abs(center - 160.0) for center in accepted_centers) <= (
        realtime_runner.POST_LOCK_HAND_MOTION_MAX_CUMULATIVE_PX + 1e-6
    )
    assert manager.partial_motion_rejected_until_identity_recovery is True
    assert manager.last_partial_hand_motion_geometry is None
    assert manager.locked_target["valid"] is True


def test_partial_motion_expires_from_last_current_identity_without_dropping_lock(
    monkeypatch,
):
    manager = _tracking_manager()
    manager.last_identity_observation_monotonic = (
        realtime_runner.time.monotonic()
        - realtime_runner.POST_LOCK_HAND_OCCLUSION_HOLD_MAX_SEC
        - 0.1
    )
    candidate = _seed_geometry()
    candidate["center_px"] = [170.0, 120.0]
    candidate["bbox_xyxy"] = [145.0, 40.0, 195.0, 200.0]
    candidate["bbox_xywh"] = [145.0, 40.0, 50.0, 160.0]

    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.detect_roi_bottle_candidate",
        lambda *_args, **_kwargs: (copy.deepcopy(candidate), [80, 10, 260, 225], None, {}, {}),
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.validate_roi_bottle_candidate",
        lambda *_args, **_kwargs: (
            False,
            {
                "roi_candidate_hand_occlusion": False,
                "roi_candidate_reject_reason": ["center_shift_too_large"],
            },
        ),
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.partial_hand_occlusion_rigid_motion_estimate",
        lambda *_args, **_kwargs: (copy.deepcopy(candidate), {}),
    )
    rgb, depth = _scene()

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["motion_estimate_current"] is False
    assert telemetry["motion_estimate_reject_reason"] == (
        "identity_anchor_motion_estimate_expired"
    )
    assert manager.partial_motion_rejected_until_identity_recovery is True
    assert manager.last_partial_hand_motion_geometry is None
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert manager.locked_target["valid"] is True

    current_candidate = _seed_geometry()
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.detect_roi_bottle_candidate",
        lambda *_args, **_kwargs: (
            copy.deepcopy(current_candidate),
            [80, 10, 240, 225],
            None,
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.validate_roi_bottle_candidate",
        lambda *_args, **_kwargs: (True, {"roi_candidate_reject_reason": []}),
    )

    recovered, recovery = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=52,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert recovery["identity_measurement_current"] is True
    assert recovered["measurement_status"] == "CURRENT_MEASURED_RGBD"
    # A broad 2D ROI association is not enough to restore display authority.
    # The latch clears only after fast_track_frame also validates the current
    # locked bottle-local TRUE_3D projection.
    assert manager.partial_motion_rejected_until_identity_recovery is True
    assert manager.locked_target["valid"] is True


def test_current_rgbd_identity_is_reported_as_preserved():
    manager = _explicit_hand_lock_manager()
    tracking = {
        "tracking_ok": True,
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "contact_patch_tracking_ok": True,
    }

    manager.apply_tracking_counters(tracking)

    assert tracking["target_identity_preserved"] is True
    assert tracking["identity_claim_status"] == "CURRENT_RGBD_IDENTITY"


def test_post_lock_tracker_holds_last_good_extent_when_hand_hides_too_much_cap():
    manager = _tracking_manager()
    original_geometry = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb, depth = _scene("covered_cap")

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["tracking_source"] == "shifted_locked_geometry"
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert _bbox_xyxy(geometry) == BOTTLE_BBOX
    assert manager.last_good_bottle_geometry == original_geometry


def test_post_lock_tracker_preserves_occlusion_evidence_when_hand_removes_candidate():
    manager = _tracking_manager()
    original_geometry = copy.deepcopy(manager.last_good_bottle_geometry)
    rgb, depth = _scene("full_cover")

    geometry, telemetry = roi_opencv_depth_bottle_tracking(
        rgb,
        depth,
        manager,
        frame_id=2,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )

    assert telemetry["identity_measurement_current"] is False
    assert telemetry["roi_candidate_hand_occlusion"] is True
    assert telemetry["roi_candidate_hand_occlusion_evidence_status"] == (
        "ESTIMATED_SAME_DEPTH_SKIN_OVERLAP_NO_BOTTLE_CANDIDATE"
    )
    assert telemetry["roi_candidate_hand_occlusion_physically_validated"] is False
    assert geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert _bbox_xyxy(geometry) == BOTTLE_BBOX
    assert manager.last_good_bottle_geometry == original_geometry


def _explicit_hand_lock_manager():
    manager = TargetLockManager(enabled=True)
    seed = _seed_geometry()
    manager.last_good_bottle_geometry = copy.deepcopy(seed)
    manager.last_good_bottle_bbox = list(BOTTLE_BBOX)
    manager.last_good_bottle_depth_mm = float(BOTTLE_DEPTH_MM)
    manager.last_good_bottle_frame_id = 1
    manager.locked_target = {
        "valid": True,
        "bottle_target_id": "bottle_01",
        "attempt_id": "attempt-original",
        "locked_body_fraction": 0.52,
        "locked_bottle_geometry": copy.deepcopy(seed),
    }
    manager.grasp_memory = {
        "status": "LOCKED_BOTTLE_RELATIVE_GRASP",
        "bottle_target_id": "bottle_01",
        "locked_body_fraction": 0.52,
    }
    return manager


def test_explicit_hand_cover_recovers_before_bounded_occlusion_hold_expires(monkeypatch):
    manager = _explicit_hand_lock_manager()
    hold_limit = manager.max_bad_tracking_frames + 3
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.POST_LOCK_HAND_OCCLUSION_HOLD_MAX_FRAMES",
        hold_limit,
    )
    covered_rgb, covered_depth = _scene("full_cover")

    covered_frame_count = manager.max_bad_tracking_frames + 1
    for frame_id in range(2, 2 + covered_frame_count):
        _geometry, telemetry = roi_opencv_depth_bottle_tracking(
            covered_rgb,
            covered_depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
        )
        assert telemetry["roi_candidate_hand_occlusion"] is True
        telemetry.update({"tracking_ok": False, "locked_patch_visible": False})
        separate_identity_from_contact_tracking(
            telemetry,
            bottle_geometry_tracking_ok=False,
        )
        manager.apply_tracking_counters(telemetry)

    assert covered_frame_count > manager.max_bad_tracking_frames
    assert covered_frame_count < hold_limit
    assert manager.status() == "LOCKED"
    assert telemetry["identity_occlusion_hold_active"] is True
    assert telemetry["identity_occlusion_hold_expired"] is False
    assert telemetry["identity_occlusion_evidence_current"] is True
    assert telemetry["identity_claim_status"] == (
        "UNVERIFIED_ORIGINAL_TARGET_HYPOTHESIS_PRESERVED"
    )
    assert manager.locked_target["bottle_target_id"] == "bottle_01"
    assert manager.locked_target["attempt_id"] == "attempt-original"
    assert manager.locked_target["locked_body_fraction"] == 0.52

    clean_rgb, clean_depth = _scene()
    _geometry, recovered = roi_opencv_depth_bottle_tracking(
        clean_rgb,
        clean_depth,
        manager,
        frame_id=2 + covered_frame_count,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )
    recovered["tracking_ok"] = True
    recovered["locked_patch_visible"] = True
    separate_identity_from_contact_tracking(
        recovered,
        bottle_geometry_tracking_ok=True,
    )
    manager.apply_tracking_counters(recovered)

    assert recovered["identity_measurement_current"] is True
    assert recovered["identity_occlusion_hold_active"] is True
    assert recovered["identity_recovery_confirmation_pending"] is True

    _geometry, recovered_again = roi_opencv_depth_bottle_tracking(
        clean_rgb,
        clean_depth,
        manager,
        frame_id=3 + covered_frame_count,
        current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
    )
    recovered_again["tracking_ok"] = True
    recovered_again["locked_patch_visible"] = True
    separate_identity_from_contact_tracking(
        recovered_again,
        bottle_geometry_tracking_ok=True,
    )
    manager.apply_tracking_counters(recovered_again)

    assert recovered_again["identity_occlusion_hold_active"] is False
    assert recovered_again["identity_recovery_confirmation_pending"] is False
    assert manager.bad_tracking_frames == 0
    assert manager.status() == "LOCKED"
    assert manager.locked_target["bottle_target_id"] == "bottle_01"
    assert manager.locked_target["locked_body_fraction"] == 0.52


def test_unclassified_foreground_mistake_holds_original_lock_and_blocks_ready():
    manager = _explicit_hand_lock_manager()

    for _ in range(manager.max_bad_tracking_frames + 1):
        tracking = {
            "tracking_ok": False,
            "locked_patch_visible": False,
            "patch_depth_confidence": 0.0,
            "tracking_confidence": 0.0,
            "identity_measurement_current": False,
            "identity_measurement_age_frames": 99,
            "roi_candidate_foreground_occlusion_on_seed": True,
            "roi_candidate_reject_reason": [
                "unclassified_foreground_occludes_locked_bottle"
            ],
        }
        separate_identity_from_contact_tracking(
            tracking, bottle_geometry_tracking_ok=False
        )
        manager.apply_tracking_counters(tracking)

    assert manager.status() == "LOCKED"
    assert manager.locked_target["bottle_target_id"] == "bottle_01"
    assert manager.locked_target["locked_body_fraction"] == 0.52
    assert tracking["identity_occlusion_hold_active"] is True
    assert tracking["identity_unclassified_foreground_hold_candidate"] is True
    assert tracking["identity_occlusion_hold_evidence_source"] == (
        "current_unclassified_foreground_at_locked_bottle"
    )
    blockers = ready_blockers_from_values(
        {}, tracking, {"force_closure": True}, 100.0, [], target_status=manager.status()
    )
    assert "bottle_identity_uncertain" in blockers
    assert "identity_measurement_not_current" in blockers
    assert "original_target_occluded_hold" in blockers


def test_expired_hand_hold_survives_brief_evidence_dropout_then_loses(monkeypatch):
    monkeypatch.setattr(realtime_runner, "STICKY_SINGLE_BOTTLE_LOCK", False)
    manager = _explicit_hand_lock_manager()
    clock = {"now": 100.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(realtime_runner, "POST_LOCK_HAND_OCCLUSION_HOLD_MAX_SEC", 0.2)
    hold_limit = manager.max_bad_tracking_frames + 2
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.POST_LOCK_HAND_OCCLUSION_HOLD_MAX_FRAMES",
        hold_limit,
    )
    covered_rgb, covered_depth = _scene("full_cover")

    for frame_id in range(2, 2 + hold_limit + 1):
        _geometry, telemetry = roi_opencv_depth_bottle_tracking(
            covered_rgb,
            covered_depth,
            manager,
            frame_id=frame_id,
            current_patch_depth_mm=float(BOTTLE_DEPTH_MM),
        )
        assert telemetry["roi_candidate_hand_occlusion"] is True
        telemetry.update({"tracking_ok": False, "locked_patch_visible": False})
        separate_identity_from_contact_tracking(
            telemetry,
            bottle_geometry_tracking_ok=False,
        )
        manager.apply_tracking_counters(telemetry)
        clock["now"] += 0.05
    assert manager.status() == "LOCKED"
    assert manager.locked_target["bottle_target_id"] == "bottle_01"
    assert manager.locked_target["locked_body_fraction"] == 0.52
    assert telemetry["identity_occlusion_hold_active"] is False
    assert telemetry["identity_occlusion_hold_expired"] is True
    assert telemetry["identity_reacquisition_pending"] is True
    assert telemetry["identity_occlusion_evidence_current"] is True
    assert telemetry["target_identity_preserved"] is True
    assert telemetry["identity_occlusion_hold_safety_status"] == (
        "HOLD_EXPIRED_LOCK_RETAINED_READY_BLOCKED"
    )
    blockers = ready_blockers_from_values(
        {}, telemetry, {"force_closure": True}, 100.0, [], target_status=manager.status()
    )
    assert "original_target_reacquisition_pending" in blockers

    # A recent contact/depth bridge can briefly look valid without providing a
    # new current bottle-identity measurement. It must not erase the pending
    # reacquisition state that protects the same physical grasp target.
    recent_bridge = {
        "tracking_ok": True,
        "locked_patch_visible": True,
        "patch_depth_confidence": 90.0,
        "identity_measurement_current": False,
        "identity_measurement_age_frames": 1,
    }
    separate_identity_from_contact_tracking(
        recent_bridge, bottle_geometry_tracking_ok=False
    )
    manager.apply_tracking_counters(recent_bridge)
    assert recent_bridge["identity_tracking_ok"] is True
    assert manager.status() == "LOCKED"

    def apply_no_evidence():
        no_evidence = {
            "tracking_ok": False,
            "locked_patch_visible": False,
            "patch_depth_confidence": 0.0,
            "identity_measurement_current": False,
            "identity_measurement_age_frames": 99,
        }
        separate_identity_from_contact_tracking(
            no_evidence, bottle_geometry_tracking_ok=False
        )
        manager.apply_tracking_counters(no_evidence)
        return no_evidence

    for expected in range(1, realtime_runner.POST_LOCK_ROI_HOLD_LAST_GOOD_FRAMES + 1):
        no_evidence = apply_no_evidence()
        assert manager.status() == "LOCKED"
        assert no_evidence["identity_reacquisition_pending"] is True
        assert no_evidence["identity_reacquisition_dropout_grace_active"] is True
        assert no_evidence["identity_reacquisition_dropout_frames"] == expected

    no_evidence = apply_no_evidence()

    assert manager.status() == "LOST"
    assert no_evidence["identity_reacquisition_pending"] is False
    assert no_evidence["identity_reacquisition_dropout_grace_active"] is False
    assert manager.locked_target is None
    assert manager.has_lost_lock is True
    assert manager.last_lost_target["bottle_target_id"] == "bottle_01"
    assert manager.last_lost_target["attempt_id"] == "attempt-original"
    assert manager.last_lost_target["locked_body_fraction"] == 0.52
    assert manager.grasp_memory["status"] == "PRESERVED_AFTER_TRACK_LOSS"
    assert manager.grasp_memory["bottle_target_id"] == "bottle_01"
    assert manager.grasp_memory["locked_body_fraction"] == 0.52


def test_hand_occlusion_hold_counts_actual_hold_frames_not_prior_bad_frames(monkeypatch):
    manager = _explicit_hand_lock_manager()
    clock = {"now": 200.0}
    monkeypatch.setattr(realtime_runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(realtime_runner, "POST_LOCK_HAND_OCCLUSION_HOLD_MAX_SEC", 0.2)
    manager.bad_tracking_frames = manager.max_bad_tracking_frames - 1
    manager.identity_occlusion_hold_frames = 0
    monkeypatch.setattr(
        "grasp_project.realsense.realtime_runner.POST_LOCK_HAND_OCCLUSION_HOLD_MAX_FRAMES",
        2,
    )

    def apply_occluded_frame():
        tracking = {
            "tracking_ok": False,
            "locked_patch_visible": False,
            "patch_depth_confidence": 0.0,
            "identity_measurement_current": False,
            "identity_measurement_age_frames": 99,
            "roi_candidate_hand_occlusion": True,
        }
        separate_identity_from_contact_tracking(
            tracking, bottle_geometry_tracking_ok=False
        )
        manager.apply_tracking_counters(tracking)
        return tracking

    first = apply_occluded_frame()
    assert manager.status() == "LOCKED"
    assert first["identity_occlusion_hold_active"] is True
    assert first["identity_occlusion_hold_frames"] == 1

    second = apply_occluded_frame()
    assert manager.status() == "LOCKED"
    assert second["identity_occlusion_hold_active"] is True
    assert second["identity_occlusion_hold_frames"] == 2

    clock["now"] += 0.21
    third = apply_occluded_frame()
    assert manager.status() == "LOCKED"
    assert third["identity_occlusion_hold_active"] is False
    assert third["identity_occlusion_hold_expired"] is True
    assert third["identity_reacquisition_pending"] is True
    assert third["target_identity_preserved"] is True


def test_held_geometry_and_grasp_band_follow_local_translation_without_resizing():
    stale_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    stale_mask[40:200, 135:185] = 1
    geometry = {
        **_seed_geometry(),
        "frame_id": 40,
        "geometry_frame_id": 40,
        "mask_frame_id": 40,
        "cleaned_mask": stale_mask,
        "body_mask": stale_mask,
        "measurement_status": "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT",
        "oriented_bbox": [[135, 40], [185, 40], [185, 200], [135, 200]],
        "contour": [[135, 40], [185, 40], [185, 200], [135, 200]],
        "body_contour": [[135, 40], [185, 40], [185, 200], [135, 200]],
    }
    delta = [14.0, -9.0]

    moved_geometry = shifted_bottle_geometry(geometry, delta, (*FRAME_SHAPE, 3))
    moved_clamp = shifted_target_clamp(_locked_clamp(), delta)

    assert moved_geometry["center_px"] == [174.0, 111.0]
    assert moved_geometry["bbox_xyxy"] == [149.0, 31.0, 199.0, 191.0]
    assert moved_geometry["bottle_extent_bbox_xyxy"] == [149, 31, 199, 191]
    assert _bbox_size(moved_geometry["bbox_xyxy"]) == _bbox_size(BOTTLE_BBOX)
    assert moved_geometry["tracking_delta_px"] == delta
    assert moved_geometry["measurement_status"] == "HELD_ESTIMATED_NOT_CURRENT_MEASUREMENT"
    assert "cleaned_mask" not in moved_geometry
    assert "body_mask" not in moved_geometry
    assert moved_geometry["source_geometry_frame_id"] == 40
    assert moved_geometry["source_mask_frame_id"] == 40
    assert moved_geometry["frame_id"] is None
    assert moved_geometry["geometry_frame_id"] is None
    assert moved_geometry["mask_frame_id"] is None
    assert moved_geometry["mask_measurement_status"] == (
        "NOT_AVAILABLE_AFTER_GEOMETRIC_SHIFT_USE_SHIFTED_CONTOUR"
    )
    assert moved_clamp["clamp_center_px"] == [174.0, 111.0]
    assert moved_clamp["left_contact_px"] == [149.0, 111.0]
    assert moved_clamp["right_contact_px"] == [199.0, 111.0]
    assert moved_clamp["grasp_line"]["left_endpoint_px"] == [149.0, 111.0]
    assert moved_clamp["grasp_line"]["right_endpoint_px"] == [199.0, 111.0]
    assert moved_clamp["grip_width_px"] == 50.0


def test_final_contact_evidence_is_recomputed_from_final_reanchored_clamp():
    depth = np.full((200, 200), 1200, dtype=np.uint16)
    bottle_mask = np.zeros_like(depth, dtype=np.uint8)
    bottle_mask[25:176, 60:141] = 1
    depth[bottle_mask > 0] = 600
    stale_clamp = _contact_clamp_at_y(60.0)
    final_clamp = _contact_clamp_at_y(130.0)
    depth[50:71, 74:87] = 350
    depth[50:71, 114:127] = 900

    stale_evidence = contact_patch_depths(
        depth,
        stale_clamp,
        bottle_mask=bottle_mask,
        reference_depth_mm=600.0,
    )
    assert stale_evidence["clamp_patch_depth_median_mm"] is None

    tracking = {
        "current_depth_mm": 350.0,
        "patch_bbox_current_px": [74.0, 50.0, 126.0, 70.0],
        "locked_patch_visible": False,
    }
    refresh_contact_tracking_from_final_clamp(
        tracking,
        depth,
        final_clamp,
        {"body_mask": bottle_mask, "depth_median_mm": 600.0},
        depth.shape,
        geometry_measurement_current=True,
    )

    assert tracking["final_contact_patch_evidence_current"] is True
    assert tracking["final_contact_patch_evidence_source"] == "final_reanchored_clamp_current_bottle_mask_and_depth"
    assert tracking["patch_bbox_current_px"] == [74.0, 120.0, 126.0, 140.0]
    assert tracking["current_depth_mm"] == 600.0
    assert tracking["locked_patch_visible"] is True
    assert tracking["tracking_ok"] is True


def test_foreground_hand_center_depth_cannot_rescale_accepted_bottle_band():
    depth = np.full(FRAME_SHAPE, 900, dtype=np.uint16)
    bottle_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    bottle_mask[40:200, 135:185] = 1
    depth[bottle_mask > 0] = 600
    depth[112:129, 152:169] = 180  # Closer hand over the bottle center.
    masked_bottle_depth = float(np.median(depth[(bottle_mask > 0) & (depth > 300)]))
    geometry = {
        "contour": [[135, 40], [185, 40], [185, 200], [135, 200]],
        "body_contour": [[135, 40], [185, 40], [185, 200], [135, 200]],
        "body_mask": bottle_mask,
        "center_px": [160.0, 120.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": masked_bottle_depth,
    }

    band, trace = reanchor_target_clamp_to_current_edges(
        _locked_clamp(),
        geometry,
        {
            "locked_body_fraction": 0.5,
            "bottle_contact_model": {"patch_height_mm": 50.0, "patch_width_mm": 15.0},
        },
        depth,
        depth.shape,
        {"fx": 600.0, "fy": 600.0},
    )

    assert depth[120, 160] == 180
    assert masked_bottle_depth == 600.0
    assert trace["contact_patch_depth_mm"] == 600.0
    assert trace["contact_patch_scale_source"] == "current_rgbd_depth_and_intrinsics"
    assert band["clamp_patch_height_px"] == pytest.approx(50.0)
    assert band["clamp_patch_width_px"] == pytest.approx(15.0)
    assert band["clamp_band_height_px"] == pytest.approx(50.0)


def test_alternating_occluder_support_cannot_accumulate_stationary_clamp_jumps():
    depth_frames = [
        _motion_depth(occluder_side=side)
        for side in ("left", "right", "left", "right", "left", "right", "left", "right")
    ]
    verified, trackings, displays = _run_held_local_motion_sequence(depth_frames)
    anchor = np.asarray(verified["clamp_center_px"], dtype=float)
    raw_centers = [anchor] + [
        np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float)
        for tracking in trackings
    ]
    display_centers = [
        np.asarray(result["display_magenta_center"], dtype=float)
        for result in displays
    ]
    raw_steps = [float(np.linalg.norm(current - previous)) for previous, current in zip(raw_centers, raw_centers[1:])]
    display_steps = [
        float(np.linalg.norm(current - previous))
        for previous, current in zip(display_centers, display_centers[1:])
    ]

    assert max(raw_steps) == 0.0
    assert sum(raw_steps) == 0.0
    assert max(display_steps) == 0.0
    assert sum(display_steps) == 0.0
    assert all(result["display_smoothing_mode"] != "reset" for result in displays[1:])
    final = trackings[-1]
    assert final["local_motion_observed_center_history_px"] == []
    assert final["local_motion_observation_jitter_px"] is None
    assert np.linalg.norm(np.asarray(final["local_motion_filtered_center_px"], dtype=float) - anchor) == 0.0
    assert final["local_motion_filter_mode"] == "freeze_at_last_verified_rgbd_clamp"


def test_unverified_hand_neighborhood_cannot_move_authoritative_clamp():
    offsets = [0, 3, 6, 9, 12, 15, 18]
    verified, trackings, displays = _run_held_local_motion_sequence(
        [_motion_depth(offset_x=offset) for offset in offsets]
    )
    anchor = np.asarray(verified["clamp_center_px"], dtype=float)
    raw_centers = [anchor] + [
        np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float)
        for tracking in trackings
    ]
    raw_x = np.asarray([center[0] for center in raw_centers], dtype=float)
    raw_steps = np.linalg.norm(np.diff(np.asarray(raw_centers), axis=0), axis=1)
    display_x = np.asarray(
        [result["display_magenta_center"][0] for result in displays],
        dtype=float,
    )

    assert np.allclose(raw_x, raw_x[0])
    assert float(np.max(raw_steps)) == 0.0
    assert np.allclose(display_x, display_x[0])
    assert all(
        tracking["local_motion_application_status"] == "FROZEN_STALE_IDENTITY"
        for tracking in trackings
    )
    for tracking in trackings:
        clamp = tracking["target_clamp"]
        assert clamp["grip_width_px"] == verified["grip_width_px"]
        assert clamp["clamp_patch_height_px"] == verified["clamp_patch_height_px"]
        assert clamp["clamp_patch_width_px"] == verified["clamp_patch_width_px"]
    final = trackings[-1]
    assert final["local_motion_observed_center_history_px"] == []
    assert final["local_motion_filtered_center_px"] is not None
    assert final["local_motion_filter_mode"] == "freeze_at_last_verified_rgbd_clamp"


def test_foreground_occlusion_uses_bounded_depth_to_follow_same_grasp_location():
    verified = _contact_clamp_at_y(100.0)
    previous = {
        "target_clamp": copy.deepcopy(verified),
        "last_verified_rgbd_target_clamp": copy.deepcopy(verified),
        "identity_measurement_current": False,
        "identity_measurement_age_frames": 99,
        "roi_candidate_foreground_occlusion_on_seed": True,
        "roi_candidate_reject_reason": [
            "unclassified_foreground_occludes_locked_bottle"
        ],
        "current_depth_mm": 100.0,
    }
    locked = {
        "valid": True,
        "lock_frame_id": 1,
        "locked_body_fraction": 0.5,
        "depth_mm_at_lock": 100.0,
        "movement_deadband_state": {},
    }
    centers = []
    for frame_id, offset in enumerate((0, 4, 8, 12, 16, 20), start=1):
        _plan, tracking = local_locked_patch_tracking(
            _motion_depth(offset_x=offset),
            locked,
            previous,
            (200, 200),
            frame_id=frame_id,
        )
        centers.append(np.asarray(tracking["target_clamp"]["clamp_center_px"], dtype=float))
        previous = tracking

    assert centers[-1][0] > centers[0][0] + 5.0
    assert max(
        float(np.linalg.norm(current - prior))
        for prior, current in zip(centers, centers[1:])
    ) <= 8.0
    assert tracking["local_motion_application_status"] == "BOUNDED_OCCLUSION_DEPTH_ESTIMATE"
    assert tracking["identity_measurement_current"] is False
    assert tracking["target_clamp"]["tracking_projection_source"] == (
        "bounded_local_depth_hand_occlusion_estimate"
    )

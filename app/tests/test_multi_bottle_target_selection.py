"""Camera-free tests for multi-bottle detection and feasibility selection."""

import cv2
import numpy as np

from bottle_pose import approximate_bbox_dimensions_mm, detect_bottle_pose, detect_bottle_poses
import grasp_project.realsense.realtime_runner as realtime_runner
from grasp_project.planning.task_grasp_policy import (
    evaluate_grasp_band_clearance,
    finalize_planned_grasp_selection,
    mark_planned_grasp_selection_failed,
    select_bottle_target,
)
from grasp_project.realsense.object_state import ObjectState
from grasp_project.realsense.realtime_runner import compact_live_status, match_yolo_detection_to_pose
from yolo_bottle_detector import YoloBottleDetector, expand_bbox_xyxy


def _draw_bottle(rgb, depth, center, size=(42, 126), angle=0.0, depth_mm=500):
    box = cv2.boxPoints((tuple(map(float, center)), tuple(map(float, size)), float(angle))).astype(np.int32)
    cv2.fillConvexPoly(rgb, box, (35, 90, 210))
    cv2.fillConvexPoly(depth, box, int(depth_mm))


def _draw_gradient_bottle(rgb, depth, center, size=(60, 126), depth_mm=620):
    cx, cy = map(int, center)
    width, height = map(int, size)
    x1, x2 = cx - width // 2, cx + width // 2
    y1, y2 = cy - height // 2, cy + height // 2
    split = cy
    cv2.rectangle(rgb, (x1, y1), (x2, split - 5), (35, 145, 185), thickness=-1)
    cv2.rectangle(rgb, (x1 + 2, split + 5), (x2 - 2, y2), (205, 82, 45), thickness=-1)
    cv2.rectangle(depth, (x1, y1), (x2, y2), int(depth_mm), thickness=-1)


def test_dark_navy_center_bottle_remains_a_rgbd_candidate():
    rgb = np.full((270, 480, 3), 28, dtype=np.uint8)
    depth = np.full((270, 480), 760, dtype=np.uint16)
    cv2.rectangle(rgb, (218, 70), (268, 235), (18, 30, 62), thickness=-1)
    cv2.rectangle(depth, (218, 70), (268, 235), 270, thickness=-1)

    poses, error = detect_bottle_poses(rgb, depth)

    assert error is None
    assert any(
        abs(float(pose["center_px"][0]) - 243.0) < 12.0
        and float(pose["depth_median_mm"]) < 350.0
        for pose in poses
    )


def test_dark_navy_bottle_is_recovered_when_touching_blue_screen_in_rgb():
    rgb = np.full((270, 480, 3), 28, dtype=np.uint8)
    depth = np.full((270, 480), 760, dtype=np.uint16)
    # Bright screen and dark bottle overlap in image space, so the ordinary
    # blue mask is one connected component even though the bottle is nearer.
    cv2.rectangle(rgb, (85, 0), (395, 112), (35, 115, 220), thickness=-1)
    cv2.rectangle(depth, (85, 0), (395, 112), 365, thickness=-1)
    cv2.rectangle(rgb, (218, 78), (272, 236), (18, 30, 62), thickness=-1)
    cv2.rectangle(depth, (218, 78), (272, 236), 270, thickness=-1)

    poses, error = detect_bottle_poses(rgb, depth)

    assert error is None
    assert any(
        abs(float(pose["center_px"][0]) - 245.0) < 15.0
        and float(pose["depth_median_mm"]) < 320.0
        for pose in poses
    )


def test_low_saturation_navy_bottle_uses_central_near_depth_with_color_support():
    rgb = np.full((270, 480, 3), 45, dtype=np.uint8)
    depth = np.full((270, 480), 720, dtype=np.uint16)
    # Mostly low-saturation navy body with sparse stronger-blue markings, like
    # the live D405 frame under changing monitor illumination.
    cv2.rectangle(rgb, (222, 82), (280, 236), (68, 72, 78), thickness=-1)
    cv2.rectangle(rgb, (232, 108), (270, 210), (48, 57, 76), thickness=-1)
    cv2.rectangle(depth, (222, 82), (280, 236), 275, thickness=-1)
    cv2.rectangle(rgb, (85, 0), (395, 105), (35, 115, 220), thickness=-1)
    cv2.rectangle(depth, (85, 0), (395, 105), 350, thickness=-1)

    poses, error = detect_bottle_poses(rgb, depth)

    assert error is None
    assert any(
        abs(float(pose["center_px"][0]) - 251.0) < 15.0
        and float(pose["depth_median_mm"]) < 320.0
        for pose in poses
    )


def test_single_bottle_demo_selects_sole_central_near_candidate_over_screen_duplicates():
    physical = {
        "center_px": [248.0, 165.0],
        "depth_median_mm": 275.0,
        "detection_confidence": 0.90,
        "depth_confidence": 0.95,
    }
    screen = {
        "center_px": [143.0, 96.0],
        "depth_median_mm": 560.0,
        "detection_confidence": 0.82,
        "depth_confidence": 0.92,
    }

    selected = realtime_runner.select_central_near_single_bottle_candidate(
        [screen, physical],
        (270, 480, 3),
    )

    assert selected is physical
    assert selected["single_bottle_central_near_selection"] is True
    assert selected["single_bottle_central_near_depth_separation_mm"] == 285.0


def test_single_bottle_demo_keeps_two_central_near_bottles_ambiguous():
    candidates = [
        {
            "center_px": [220.0, 160.0],
            "depth_median_mm": 280.0,
            "detection_confidence": 0.90,
            "depth_confidence": 0.95,
        },
        {
            "center_px": [275.0, 165.0],
            "depth_median_mm": 470.0,
            "detection_confidence": 0.88,
            "depth_confidence": 0.94,
        },
    ]

    assert realtime_runner.select_central_near_single_bottle_candidate(
        candidates,
        (270, 480, 3),
    ) is None


def _state(orientation="STANDING", *, available=True, occluded=False, completeness=0.8, depth_ratio=1.0):
    return ObjectState(
        orientation=orientation,
        tilt_angle_deg=0.0 if orientation == "STANDING" else 90.0,
        standing_confidence=1.0 if orientation == "STANDING" else 0.0,
        occluded=occluded,
        available_for_pickup=available,
        reasons=["clear"] if available else ["clear_space_above_top_band_blocked"],
        evidence={
            "availability": {
                "mask_completeness": completeness,
                "depth_valid_ratio": depth_ratio,
            }
        },
    )


def _pose(target_id, x, confidence=0.8, border_touch=False):
    return {
        "target_id": target_id,
        "bbox_xywh": [x, 40, 42, 126],
        "center_px": [x + 21.0, 103.0],
        "depth_median_mm": 500.0,
        "detection_confidence": confidence,
        "bottle_profile_agreement_score": 0.8,
        "candidate_evidence_scores": {"border_touch": border_touch},
    }


def test_existing_detector_returns_two_distinct_bottles_without_duplicates():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.zeros((240, 360), dtype=np.uint16)
    _draw_bottle(rgb, depth, (95, 120), depth_mm=480)
    _draw_bottle(rgb, depth, (265, 120), depth_mm=620)

    poses, error = detect_bottle_poses(rgb, depth, intrinsics={"fx": 500.0, "fy": 500.0})
    legacy_pose, legacy_error = detect_bottle_pose(rgb, depth)

    assert error is None
    assert legacy_error is None
    assert len(poses) == 2
    assert {pose["target_id"] for pose in poses} == {"bottle_01", "bottle_02"}
    assert all(pose["detected_bottle_count"] == 2 for pose in poses)
    assert legacy_pose["target_id"] in {"bottle_01", "bottle_02"}


def test_rgb_supported_scene_rejects_unrelated_depth_only_shape():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.zeros((240, 360), dtype=np.uint16)
    _draw_bottle(rgb, depth, (95, 120), depth_mm=480)
    cv2.rectangle(depth, (220, 55), (285, 190), 510, thickness=-1)

    poses, error = detect_bottle_poses(rgb, depth, intrinsics={"fx": 500.0, "fy": 500.0})

    assert error is None
    assert len(poses) == 1
    assert poses[0]["center_px"][0] < 160


def test_border_attached_keyboard_and_monitor_depth_surfaces_are_not_bottles():
    rgb = np.zeros((300, 500, 3), dtype=np.uint8)
    depth = np.full((300, 500), 1000, dtype=np.uint16)
    _draw_bottle(rgb, depth, (180, 145), size=(46, 150), depth_mm=520)
    keyboard = np.asarray(
        [[240, 250], [499, 230], [499, 299], [260, 299]],
        dtype=np.int32,
    )
    monitor = np.asarray(
        [[450, 20], [499, 30], [499, 220], [445, 200]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(depth, keyboard, 420)
    cv2.fillConvexPoly(depth, monitor, 600)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 1
    assert poses[0]["center_px"][0] < 240
    assert poses[0]["candidate_evidence_scores"]["border_touch"] is False


def test_different_color_and_size_bottle_sections_fuse_into_full_pose():
    rgb = np.zeros((240, 380, 3), dtype=np.uint8)
    depth = np.zeros((240, 380), dtype=np.uint16)
    _draw_bottle(rgb, depth, (90, 120), size=(42, 148), depth_mm=480)
    _draw_gradient_bottle(rgb, depth, (275, 120), size=(62, 122), depth_mm=640)

    poses, error = detect_bottle_poses(rgb, depth, intrinsics={"fx": 500.0, "fy": 500.0})

    assert error is None
    assert len(poses) == 2
    right = max(poses, key=lambda pose: pose["center_px"][0])
    assert right["bbox_xywh"][3] >= 110
    assert right["bbox_xywh"][2] >= 50
    assert right["chromatic_color_fraction"] > 0.70
    assert right["color_evidence_mode"] == "known_blue_or_generic_chromatic_depth_fused"
    assert right["approximate_bbox_dimensions_mm"]["broad_bottle_dimensions_valid"]


def test_dark_low_color_bottle_is_found_from_depth_shape_not_blue_threshold():
    rgb = np.full((240, 360, 3), 20, dtype=np.uint8)
    depth = np.full((240, 360), 900, dtype=np.uint16)
    cv2.rectangle(rgb, (158, 48), (202, 192), (8, 8, 8), thickness=-1)
    cv2.rectangle(depth, (158, 48), (202, 192), 560, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 1
    pose = poses[0]
    assert pose["bottle_color_fraction"] == 0.0
    assert pose["segmentation_source"] == "depth"
    assert pose["candidate_evidence_scores"]["depth_support_score"] > 0.7
    assert pose["bottle_detection_status"] == "CANDIDATE"


def test_candidate_score_breakdown_labels_unmeasured_temporal_and_pointcloud_cues():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.full((240, 360), 900, dtype=np.uint16)
    _draw_gradient_bottle(rgb, depth, (180, 120), depth_mm=620)

    poses, error = detect_bottle_poses(rgb, depth)

    assert error is None
    evidence = poses[0]["candidate_evidence_scores"]
    assert evidence["rgb_boundary_score"] is not None
    assert evidence["contour_score"] is not None
    assert evidence["shape_score"] is not None
    assert evidence["profile_score"] is not None
    assert evidence["temporal_score"] is None
    assert evidence["temporal_score_status"] == "NOT_TESTED_SINGLE_FRAME_CANDIDATE"
    assert evidence["pointcloud_score"] is None


def test_colorful_square_is_not_promoted_to_generic_bottle():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.zeros((240, 360), dtype=np.uint16)
    cv2.rectangle(rgb, (130, 70), (220, 160), (210, 80, 40), thickness=-1)
    cv2.rectangle(depth, (130, 70), (220, 160), 550, thickness=-1)

    poses, error = detect_bottle_poses(rgb, depth)

    assert poses == []
    assert error is not None


def test_depth_supported_round_top_view_is_diagnostic_top_down_candidate():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.full((240, 360), 900, dtype=np.uint16)
    cv2.circle(rgb, (180, 120), 33, (35, 90, 210), thickness=-1)
    cv2.circle(depth, (180, 120), 33, 600, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 1
    assert poses[0]["top_down_view_candidate"] is True
    assert poses[0]["approximate_bbox_dimensions_mm"]["broad_top_down_bottle_diameter_valid"] is True
    assert poses[0]["bottle_detection_status"] == "CANDIDATE"


def test_intrinsics_reject_physically_overwide_bottle_like_distractor():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.zeros((240, 360), dtype=np.uint16)
    cv2.rectangle(rgb, (150, 65), (200, 175), (35, 145, 185), thickness=-1)
    cv2.rectangle(depth, (150, 65), (200, 175), 1300, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 390.0, "fy": 390.0},
    )

    assert poses == []
    assert error is not None


def test_aligned_depth_rejects_colored_bottle_shape_on_flat_background_plane():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.full((240, 360), 700, dtype=np.uint16)
    cv2.rectangle(rgb, (145, 50), (195, 190), (35, 145, 185), thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert poses == []
    assert error is not None


def test_aligned_depth_does_not_treat_a_deeper_recess_as_foreground():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.full((240, 360), 600, dtype=np.uint16)
    cv2.rectangle(rgb, (150, 50), (200, 190), (35, 145, 185), thickness=-1)
    cv2.rectangle(depth, (150, 50), (200, 190), 760, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert poses == []
    assert error is not None


def test_pose_reports_local_depth_foreground_identity_evidence():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.full((240, 360), 900, dtype=np.uint16)
    _draw_gradient_bottle(rgb, depth, (180, 120), size=(60, 126), depth_mm=620)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 1
    evidence = poses[0]["depth_geometry_evidence"]
    assert evidence["depth_geometry_status"] == "ESTIMATED_FROM_ALIGNED_DEPTH_NEIGHBORHOOD"
    assert evidence["depth_foreground_separated"] is True
    assert evidence["depth_foreground_separation_mm"] == 280.0
    selection = select_bottle_target([{"pose": poses[0], "object_state": _state()}])
    selected = selection.candidates[0]
    assert selected.depth_geometry_score == evidence["depth_geometry_score"]
    assert selected.depth_foreground_separation_mm == 280.0
    assert selected.metric_dimensions_valid is True


def test_target_selection_rejects_rgb_fragment_on_same_depth_surface():
    pose = _pose("fragment", 150, confidence=0.92)
    pose["depth_geometry_evidence"] = {
        "depth_geometry_status": "ESTIMATED_FROM_ALIGNED_DEPTH_NEIGHBORHOOD",
        "depth_geometry_score": 0.55,
        "depth_foreground_separation_mm": 4.0,
        "depth_foreground_separated": False,
    }

    selection = select_bottle_target(
        [{"target_id": "fragment", "pose": pose, "object_state": _state()}],
        image_shape=(240, 360, 3),
    )

    assert selection.selected_target_id is None
    assert selection.eligible_count == 0
    candidate = selection.candidates[0]
    assert candidate.depth_foreground_separated is False
    assert "candidate_not_separated_from_surrounding_depth_surface" in candidate.reasons


def test_metric_bbox_dimensions_use_separate_horizontal_and_vertical_intrinsics():
    dimensions = approximate_bbox_dimensions_mm(
        (0, 0, 50, 150),
        600,
        {"fx": 500.0, "fy": 750.0},
    )

    assert dimensions["width_extent_mm"] == 60.0
    assert dimensions["height_extent_mm"] == 120.0
    assert dimensions["minor_extent_mm"] == 60.0
    assert dimensions["major_extent_mm"] == 120.0
    assert dimensions["broad_bottle_dimensions_valid"]


def test_diagonal_bottle_uses_oriented_metric_dimensions():
    rgb = np.zeros((280, 380, 3), dtype=np.uint8)
    depth = np.full((280, 380), 900, dtype=np.uint16)
    _draw_bottle(rgb, depth, (190, 140), size=(46, 154), angle=42.0, depth_mm=560)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 520.0, "fy": 520.0},
    )

    assert error is None
    assert len(poses) == 1
    dimensions = poses[0]["approximate_bbox_dimensions_mm"]
    assert dimensions["dimension_source"] == "ORIENTED_MIN_AREA_RECT"
    assert dimensions["broad_bottle_dimensions_valid"] is True


def test_depth_modes_separate_touching_differently_colored_bottles():
    rgb = np.zeros((260, 360, 3), dtype=np.uint8)
    depth = np.zeros((260, 360), dtype=np.uint16)
    _draw_bottle(rgb, depth, (145, 130), size=(58, 170), depth_mm=430)
    _draw_gradient_bottle(rgb, depth, (198, 130), size=(62, 142), depth_mm=650)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 2
    assert sorted(round(pose["depth_median_mm"]) for pose in poses) == [430, 650]


def test_narrow_gap_same_color_same_depth_bottles_remain_distinct_instances():
    rgb = np.zeros((260, 380, 3), dtype=np.uint8)
    depth = np.zeros((260, 380), dtype=np.uint16)
    _draw_bottle(rgb, depth, (150, 130), size=(58, 170), depth_mm=520)
    _draw_bottle(rgb, depth, (211, 130), size=(58, 170), depth_mm=520)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 2
    assert {pose["target_id"] for pose in poses} == {"bottle_01", "bottle_02"}
    left, right = sorted(poses, key=lambda pose: pose["center_px"][0])
    assert left["target_id"] == "bottle_01"
    assert right["target_id"] == "bottle_02"
    assert all(pose["bbox_xywh"][2] < 80 for pose in poses)
    assert abs(poses[0]["depth_median_mm"] - poses[1]["depth_median_mm"]) < 1.0


def test_touching_blue_bottles_split_at_supported_rgb_boundary():
    rgb = np.zeros((260, 380, 3), dtype=np.uint8)
    depth = np.full((260, 380), 900, dtype=np.uint16)
    left_box = cv2.boxPoints(((150.0, 130.0), (58.0, 170.0), 0.0)).astype(np.int32)
    right_box = cv2.boxPoints(((208.0, 130.0), (58.0, 170.0), 0.0)).astype(np.int32)
    cv2.fillConvexPoly(rgb, left_box, (35, 90, 210))
    cv2.fillConvexPoly(rgb, right_box, (20, 65, 175))
    cv2.fillConvexPoly(depth, left_box, 520)
    cv2.fillConvexPoly(depth, right_box, 520)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 2
    assert all(pose["bbox_xywh"][2] < 80 for pose in poses)


def test_uniform_overwide_single_bottle_is_not_split_without_boundary_evidence():
    rgb = np.zeros((260, 380, 3), dtype=np.uint8)
    depth = np.full((260, 380), 900, dtype=np.uint16)
    cv2.rectangle(rgb, (140, 65), (220, 195), (35, 90, 210), thickness=-1)
    cv2.rectangle(depth, (140, 65), (220, 195), 520, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 1


def test_dark_label_gap_does_not_split_one_bottle_into_two_instances():
    rgb = np.zeros((260, 380, 3), dtype=np.uint8)
    depth = np.zeros((260, 380), dtype=np.uint16)
    cv2.rectangle(rgb, (150, 40), (208, 126), (35, 90, 210), thickness=-1)
    cv2.rectangle(rgb, (150, 132), (208, 220), (35, 90, 210), thickness=-1)
    cv2.rectangle(depth, (150, 40), (208, 220), 520, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert error is None
    assert len(poses) == 1
    assert poses[0]["bbox_xywh"][3] >= 175


def test_low_solidity_colorful_distractor_is_rejected():
    rgb = np.zeros((240, 360, 3), dtype=np.uint8)
    depth = np.zeros((240, 360), dtype=np.uint16)
    cv2.rectangle(rgb, (165, 50), (180, 185), (50, 190, 70), thickness=-1)
    cv2.rectangle(rgb, (140, 90), (205, 105), (50, 190, 70), thickness=-1)
    cv2.rectangle(depth, (165, 50), (180, 185), 600, thickness=-1)
    cv2.rectangle(depth, (140, 90), (205, 105), 600, thickness=-1)

    poses, error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics={"fx": 400.0, "fy": 400.0},
    )

    assert poses == []
    assert error is not None


def test_feasible_target_beats_higher_confidence_occluded_target():
    result = select_bottle_target(
        [
            {"pose": _pose("bottle_01", 40, confidence=0.98), "object_state": _state(available=False, occluded=True)},
            {"pose": _pose("bottle_02", 180, confidence=0.72), "object_state": _state()},
        ]
    )

    assert result.selected_target_id == "bottle_02"
    assert result.eligible_count == 1
    assert result.candidates[0].selected
    assert result.candidates[0].approach == "SIDE"
    assert result.grasp_feasibility_label == "OPTIMAL_GRASP"
    assert result.suggested_robot_action == "PLAN_ONLY_APPROACH_SELECTED_TARGET"
    assert not result.physically_validated


def test_touching_standing_bottles_are_blocked_when_fingers_have_no_side_space():
    left = _pose("bottle_01", 80)
    right = _pose("bottle_02", 122)

    result = select_bottle_target(
        [
            {"pose": left, "object_state": _state()},
            {"pose": right, "object_state": _state()},
        ],
        image_shape=(240, 360, 3),
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert result.selected_target_id is None
    assert result.eligible_count == 0
    assert all(candidate.grasp_clearance_status == "BLOCKED_BY_NEIGHBOR_OR_IMAGE_EDGE" for candidate in result.candidates)
    assert all("no_collision_free_side_grasp_band" in candidate.reasons for candidate in result.candidates)
    assert result.grasp_feasibility_label == "CANNOT_GRASP_MULTIPLE_OBJECTS"


def test_non_bottle_depth_obstacle_blocks_side_finger_corridor():
    target = _pose("bottle_01", 100)
    target_mask = np.zeros((240, 360), dtype=np.uint8)
    target_mask[40:166, 100:142] = 1
    target["cleaned_mask"] = target_mask
    depth = np.full((240, 360), 900, dtype=np.uint16)
    depth[target_mask > 0] = 500
    depth[35:171, 82:100] = 480

    result = select_bottle_target(
        [{"pose": target, "object_state": _state()}],
        image_shape=depth.shape,
        intrinsics={"fx": 500.0, "fy": 500.0},
        depth_image=depth,
    )

    assert result.selected_target_id is None
    candidate = result.candidates[0]
    assert candidate.depth_corridor_status == "ESTIMATED_FROM_CURRENT_ALIGNED_DEPTH_CORRIDORS"
    assert candidate.left_depth_obstacle_ratio >= 0.18
    assert "depth_obstacle_left" in candidate.blocking_target_ids
    assert "no_collision_free_side_grasp_band" in candidate.reasons


def test_background_plane_behind_contact_depth_does_not_block_clear_sides():
    target = _pose("bottle_01", 100)
    target_mask = np.zeros((240, 360), dtype=np.uint8)
    target_mask[40:166, 100:142] = 1
    target["cleaned_mask"] = target_mask
    depth = np.full((240, 360), 540, dtype=np.uint16)
    depth[target_mask > 0] = 500

    result = select_bottle_target(
        [{"pose": target, "object_state": _state()}],
        image_shape=depth.shape,
        intrinsics={"fx": 500.0, "fy": 500.0},
        depth_image=depth,
    )

    assert result.selected_target_id == "bottle_01"
    candidate = result.candidates[0]
    assert candidate.grasp_clearance_status == "ESTIMATED_CLEAR"
    assert candidate.left_depth_obstacle_ratio == 0.0
    assert candidate.right_depth_obstacle_ratio == 0.0
    assert candidate.blocking_target_ids == []


def test_lower_neighbor_uses_clear_less_optimal_grasp_band():
    target = _pose("bottle_01", 100)
    target["preferred_grasp_body_fractions"] = [0.45, 0.52, 0.60]
    target["fallback_grasp_body_fractions"] = [0.32]
    lower_neighbor = _pose("bottle_02", 142)
    lower_neighbor["bbox_xywh"] = [142, 115, 42, 60]
    lower_neighbor["center_px"] = [163.0, 145.0]

    clearance = evaluate_grasp_band_clearance(
        0,
        [
            {"pose": target, "object_state": _state()},
            {"pose": lower_neighbor, "object_state": _state(available=False, occluded=True)},
        ],
        image_shape=(240, 360, 3),
        intrinsics={"fx": 500.0, "fy": 500.0},
    )
    result = select_bottle_target(
        [
            {"pose": target, "object_state": _state()},
            {"pose": lower_neighbor, "object_state": _state(available=False, occluded=True)},
        ],
        image_shape=(240, 360, 3),
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert clearance["grasp_clearance_status"] == "ESTIMATED_CLEAR"
    assert clearance["selected_grasp_body_fraction"] == 0.32
    assert clearance["grasp_band_fallback_used"]
    assert clearance["clear_grasp_body_fractions"] == [0.32]
    assert clearance["preferred_grasp_body_fractions"] == [0.45, 0.52, 0.60]
    assert clearance["fallback_grasp_body_fractions"] == [0.32]
    assert clearance["blocking_target_ids"] == []
    assert result.selected_target_id == "bottle_01"
    selected = next(candidate for candidate in result.candidates if candidate.selected)
    assert selected.selected_grasp_body_fraction == 0.32
    assert "less_optimal_clear_grasp_band_selected" in selected.reasons
    assert result.grasp_feasibility_label == "LESS_OPTIMAL_BUT_ACCEPTABLE"


def test_second_clear_preferred_band_is_optimal_not_fallback():
    target = _pose("bottle_01", 100)
    target["preferred_grasp_body_fractions"] = [0.45, 0.52, 0.60]
    target["fallback_grasp_body_fractions"] = [0.32]
    neighbor = _pose("bottle_02", 142)
    neighbor["bbox_xywh"] = [142, 94, 42, 5]

    clearance = evaluate_grasp_band_clearance(
        0,
        [
            {"pose": target, "object_state": _state()},
            {"pose": neighbor, "object_state": _state(available=False, occluded=True)},
        ],
        image_shape=(240, 360, 3),
        intrinsics={"fx": 500.0, "fy": 500.0},
        patch_height_mm=8.0,
    )

    assert clearance["selected_grasp_body_fraction"] == 0.52
    assert clearance["grasp_band_fallback_used"] is False
    assert 0.45 not in clearance["clear_grasp_body_fractions"]
    assert 0.52 in clearance["clear_grasp_body_fractions"]
    rows = {row["body_fraction"]: row for row in clearance["grasp_band_clearance_table"]}
    assert rows[0.45]["clear"] is False
    assert rows[0.52]["clear"] is True


def test_planner_winner_replaces_provisional_first_clear_band_in_report():
    selection = {
        "selected_target_id": "bottle_01",
        "grasp_feasibility_label": "OPTIMAL_GRASP",
        "candidates": [
            {
                "target_id": "bottle_01",
                "eligible": True,
                "approach": "SIDE",
                "feasibility_score": 88.0,
                "selected_grasp_body_fraction": 0.45,
                "clear_grasp_body_fractions": [0.45, 0.52, 0.60],
                "preferred_grasp_body_fractions": [0.45, 0.52, 0.60],
                "grasp_band_clearance_table": [
                    {"body_fraction": 0.45, "clear": True, "left_clearance_px": 20.0},
                    {"body_fraction": 0.52, "clear": True, "left_clearance_px": 24.0},
                ],
                "reasons": ["side_approach_supported_by_task_policy"],
            }
        ],
    }
    plan = {
        "body_fraction": 0.5211,
        "selected_requested_body_fraction": 0.52,
        "selected_candidate_id": "cand_01_fraction_0.520",
        "selected_candidate_score": 88.24,
        "selected_grasp_preference_tier": "PREFERRED",
        "selected_contact_pair": {
            "body_fraction": 0.5211,
            "requested_body_fraction": 0.52,
        },
    }

    finalize_planned_grasp_selection(selection, "bottle_01", plan)

    selected = selection["candidates"][0]
    assert selected["selected_grasp_body_fraction"] == 0.52
    assert selected["measured_contact_body_fraction"] == 0.5211
    assert selected["grasp_band_fallback_used"] is False
    assert selected["left_clearance_px"] == 24.0
    assert selection["grasp_feasibility_label"] == "OPTIMAL_GRASP"


def test_planner_failure_clears_provisional_optimal_band_claim():
    selection = {
        "selected_target_id": "bottle_01",
        "grasp_feasibility_label": "OPTIMAL_GRASP",
        "candidates": [
            {
                "target_id": "bottle_01",
                "selected_grasp_body_fraction": 0.45,
                "grasp_band_fallback_used": False,
                "reasons": ["side_approach_supported_by_task_policy"],
            }
        ],
    }

    mark_planned_grasp_selection_failed(
        selection,
        "bottle_01",
        "No planner-valid candidate also passed the current grasp-band clearance gates.",
    )

    selected = selection["candidates"][0]
    assert selected["selected_grasp_body_fraction"] is None
    assert selected["grasp_plan_status"] == "BLOCKED_BY_PLANNER_VALIDITY"
    assert "no_planner_valid_clear_band" in selected["reasons"]
    assert selection["grasp_feasibility_label"] == "CANNOT_GRASP_PLANNER_INVALID"
    assert selection["suggested_robot_action"] == "HOLD_NO_ACTUATION"


def test_closer_standing_bottle_wins_when_other_evidence_is_equal():
    near = _pose("bottle_near", 45, confidence=0.85)
    far = _pose("bottle_far", 220, confidence=0.85)
    near["depth_median_mm"] = 350.0
    far["depth_median_mm"] = 700.0

    result = select_bottle_target(
        [
            {"pose": far, "object_state": _state()},
            {"pose": near, "object_state": _state()},
        ],
        image_shape=(240, 360, 3),
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert result.selected_target_id == "bottle_near"
    near_result = next(candidate for candidate in result.candidates if candidate.target_id == "bottle_near")
    far_result = next(candidate for candidate in result.candidates if candidate.target_id == "bottle_far")
    assert near_result.distance_priority_score > far_result.distance_priority_score


def test_top_exposed_standing_bottle_ranks_before_lower_stack_candidate():
    top = _pose("bottle_top", 150)
    bottom = _pose("bottle_bottom", 150)
    top["bbox_xywh"] = [150, 10, 42, 90]
    top["center_px"] = [171.0, 55.0]
    bottom["bbox_xywh"] = [150, 120, 42, 110]
    bottom["center_px"] = [171.0, 175.0]

    result = select_bottle_target(
        [
            {"pose": bottom, "object_state": _state()},
            {"pose": top, "object_state": _state()},
        ],
        image_shape=(260, 360, 3),
        intrinsics={"fx": 500.0, "fy": 500.0},
    )

    assert result.selected_target_id == "bottle_top"
    top_result = next(candidate for candidate in result.candidates if candidate.target_id == "bottle_top")
    bottom_result = next(candidate for candidate in result.candidates if candidate.target_id == "bottle_bottom")
    assert top_result.top_exposure_score > bottom_result.top_exposure_score


def test_lying_bottle_selects_top_rotate_upright_approach():
    result = select_bottle_target(
        [{"pose": _pose("bottle_01", 80), "object_state": _state("LYING")}]
    )

    assert result.selected_target_id == "bottle_01"
    assert result.candidates[0].strategy == "TOP_GRASP_ROTATE_UPRIGHT"
    assert result.candidates[0].approach == "TOP"
    assert "top_approach_is_planning_only_without_top_contact_geometry" in result.candidates[0].reasons
    assert result.grasp_feasibility_label == "HOLD_AND_ADJUST"


def test_all_blocked_targets_emit_no_selection():
    result = select_bottle_target(
        [
            {"pose": _pose("bottle_01", 30), "object_state": _state(available=False, occluded=True)},
            {"pose": _pose("bottle_02", 170), "object_state": _state("TILTED")},
        ]
    )

    assert result.selected_target_id is None
    assert result.selection_status == "NO_FEASIBLE_TARGET"
    assert result.eligible_count == 0
    assert all(not candidate.selected for candidate in result.candidates)
    assert result.grasp_feasibility_label == "CANNOT_GRASP_POSE"


def test_unavailable_visible_candidate_reports_availability_not_pose_failure():
    unavailable = _state(available=False, occluded=False)
    unavailable = ObjectState(
        **{
            **unavailable.to_dict(),
            "reasons": ["insufficient_clear_space_depth"],
        }
    )

    result = select_bottle_target(
        [{"pose": _pose("bottle_01", 80), "object_state": unavailable}]
    )

    assert result.selected_target_id is None
    assert result.grasp_feasibility_label == "CANNOT_GRASP_AVAILABILITY"
    assert result.grasp_feasibility_reason == "current_object_state_availability_evidence_failed"
    assert "insufficient_clear_space_depth" in result.candidates[0].reasons


def test_border_cropped_candidate_is_not_eligible_for_pickup():
    result = select_bottle_target(
        [{"pose": _pose("bottle_01", 0, border_touch=True), "object_state": _state()}]
    )

    assert result.selected_target_id is None
    assert result.candidates[0].eligible is False
    assert "candidate_touches_image_border" in result.candidates[0].reasons


def test_optional_yolo_trigger_preserves_all_bottle_boxes():
    class Boxes:
        xyxy = np.asarray([[10, 20, 60, 150], [180, 30, 230, 160]], dtype=float)
        conf = np.asarray([0.91, 0.83], dtype=float)
        cls = np.asarray([39, 39], dtype=float)

    class Output:
        names = {39: "bottle"}
        boxes = Boxes()
        masks = None

    class Model:
        def predict(self, *_args, **_kwargs):
            return [Output()]

    detector = YoloBottleDetector.__new__(YoloBottleDetector)
    detector.model_path = "synthetic-test.pt"
    detector.conf_threshold = 0.5
    detector.model = Model()
    detector.available = True
    detector.unavailable_reason = None

    result = detector.detect(np.zeros((180, 260, 3), dtype=np.uint8))

    assert result["yolo_detected"]
    assert result["detection_count"] == 2
    assert len(result["detections"]) == 2
    assert result["bbox_xyxy"] == [10, 20, 60, 150]


def test_opencv_onnx_yolo_parser_keeps_only_coco_bottles():
    predictions = np.zeros((1, 84, 2), dtype=np.float32)
    predictions[0, :4, 0] = [150.0, 280.0, 100.0, 200.0]
    predictions[0, 4 + 39, 0] = 0.90
    predictions[0, :4, 1] = [400.0, 260.0, 120.0, 220.0]
    predictions[0, 4, 1] = 0.99

    class FakeOnnxNet:
        def setInput(self, _blob):
            return None

        def forward(self):
            return predictions

    detector = YoloBottleDetector.__new__(YoloBottleDetector)
    detector.model_path = "models/yolo_bottle.onnx"
    detector.conf_threshold = 0.25
    detector.model = FakeOnnxNet()
    detector.model_backend = "opencv_dnn_onnx"
    detector.available = True
    detector.unavailable_reason = None

    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result["yolo_detected"]
    assert result["detection_count"] == 1
    assert result["bbox_xyxy"] == [100, 100, 200, 300]
    assert result["class_name"] == "bottle"
    assert result["yolo_model_backend"] == "opencv_dnn_onnx"


def test_opencv_darknet_yolo_parser_keeps_only_coco_bottles():
    predictions = np.zeros((2, 85), dtype=np.float32)
    predictions[0, :5] = [0.25, 0.50, 0.20, 0.50, 1.0]
    predictions[0, 5 + 39] = 0.90
    predictions[1, :5] = [0.70, 0.50, 0.20, 0.50, 1.0]
    predictions[1, 5] = 0.99

    class FakeDarknetNet:
        def setInput(self, _blob):
            return None

        def getUnconnectedOutLayersNames(self):
            return ["yolo"]

        def forward(self, _names):
            return [predictions]

    detector = YoloBottleDetector.__new__(YoloBottleDetector)
    detector.model_path = "models/yolo_bottle.weights"
    detector.conf_threshold = 0.25
    detector.model = FakeDarknetNet()
    detector.model_backend = "opencv_dnn_darknet"
    detector.available = True
    detector.unavailable_reason = None

    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result["yolo_detected"]
    assert result["detection_count"] == 1
    assert result["bbox_xyxy"] == [96, 120, 224, 360]
    assert result["class_name"] == "bottle"
    assert result["yolo_model_backend"] == "opencv_dnn_darknet"


def test_yolo_rgbd_fusion_keeps_depth_valid_candidate_that_yolo_misses(monkeypatch):
    rgbd_poses = [_pose("bottle_01", 40), _pose("bottle_02", 180)]

    class Detector:
        def detect(self, _rgb):
            return {
                "yolo_available": True,
                "yolo_detected": True,
                "yolo_model_path": "models/yolo_bottle.weights",
                "yolo_model_backend": "opencv_dnn_darknet",
                "yolo_inference_ms": 25.0,
                "confidence": 0.91,
                "bbox_xyxy": [38, 38, 84, 168],
                "bbox_xywh": [38, 38, 46, 130],
                "class_name": "bottle",
                "detections": [
                    {
                        "bbox_xyxy": [38, 38, 84, 168],
                        "bbox_xywh": [38, 38, 46, 130],
                        "confidence": 0.91,
                        "class_name": "bottle",
                    }
                ],
                "detection_source": "yolo_opencv_dnn_darknet",
                "yolo_unavailable_reason": None,
            }

    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", True)
    monkeypatch.setattr(realtime_runner, "get_yolo_detector", lambda: Detector())
    monkeypatch.setattr(realtime_runner, "expand_bbox_xyxy", expand_bbox_xyxy)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: ([dict(pose) for pose in rgbd_poses], None),
    )

    poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
        np.zeros((240, 360, 3), dtype=np.uint8),
        np.full((240, 360), 700, dtype=np.uint16),
    )

    assert error is None
    assert len(poses) == 2
    assert poses[0]["detection_source"] == "yolo_confirmed_opencv_depth"
    assert poses[0]["yolo_semantic_match"] is True
    assert poses[1]["detection_source"] == "opencv_depth_supplemental_yolo_missed"
    assert poses[1]["yolo_semantic_match"] is False
    assert report["yolo_matched_rgbd_count"] == 1
    assert report["rgbd_candidate_count"] == 2
    assert report["fused_pose_count"] == 2
    assert report["yolo_rgbd_fusion_mode"] == "ADDITIVE_SEMANTIC_PROPOSALS_RGBD_GATES_AUTHORITATIVE"


def test_yolo_zero_detections_does_not_erase_rgbd_candidates(monkeypatch):
    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", True)
    monkeypatch.setattr(
        realtime_runner,
        "get_yolo_detector",
        lambda: type(
            "Detector",
            (),
            {
                "detect": lambda _self, _rgb: {
                    "yolo_available": True,
                    "yolo_detected": False,
                    "detections": [],
                    "detection_source": "yolo_opencv_dnn_darknet",
                }
            },
        )(),
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: ([_pose("bottle_01", 40)], None),
    )

    poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
        np.zeros((240, 360, 3), dtype=np.uint8),
        np.full((240, 360), 700, dtype=np.uint16),
    )

    assert error is None
    assert len(poses) == 1
    assert poses[0]["detection_source"] == "opencv_depth_supplemental_yolo_missed"
    assert report["yolo_detected"] is False
    assert report["fused_pose_count"] == 1


def test_two_yolo_boxes_split_one_ambiguous_rgbd_candidate(monkeypatch):
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", False)
    detections = [
        {
            "bbox_xyxy": [35, 35, 85, 175],
            "bbox_xywh": [35, 35, 50, 140],
            "confidence": 0.91,
            "class_name": "bottle",
        },
        {
            "bbox_xyxy": [85, 35, 135, 175],
            "bbox_xywh": [85, 35, 50, 140],
            "confidence": 0.88,
            "class_name": "bottle",
        },
    ]
    merged = _pose("merged", 40)
    merged["bbox_xywh"] = [40, 40, 100, 126]
    merged["center_px"] = [90.0, 103.0]

    class Detector:
        def detect(self, _rgb):
            return {
                "yolo_available": True,
                "yolo_detected": True,
                "detections": detections,
                "detection_count": 2,
                "bbox_xyxy": detections[0]["bbox_xyxy"],
                "bbox_xywh": detections[0]["bbox_xywh"],
                "confidence": detections[0]["confidence"],
                "class_name": "bottle",
                "detection_source": "yolo_opencv_dnn_darknet",
                "yolo_model_backend": "opencv_dnn_darknet",
            }

    def roi_pose(rgb, _depth, intrinsics=None):
        height, width = rgb.shape[:2]
        return {
            **_pose("roi", 4),
            "bbox_xywh": [4, 4, max(12, width - 8), max(20, height - 8)],
            "center_px": [width / 2.0, height / 2.0],
        }, None

    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", True)
    monkeypatch.setattr(realtime_runner, "get_yolo_detector", lambda: Detector())
    monkeypatch.setattr(realtime_runner, "expand_bbox_xyxy", expand_bbox_xyxy)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: ([merged], None),
    )
    monkeypatch.setattr(realtime_runner, "detect_bottle_pose", roi_pose)

    poses, error, report = realtime_runner.detect_bottle_poses_with_yolo_trigger(
        np.zeros((220, 220, 3), dtype=np.uint8),
        np.full((220, 220), 600, dtype=np.uint16),
        intrinsics={"fx": 500.0, "fy": 500.0, "cx": 110.0, "cy": 110.0},
    )

    assert error is None
    assert len(poses) == 2
    assert all(pose["detection_source"] == "yolo_split_rgbd_validated" for pose in poses)
    assert report["yolo_ambiguous_merged_rgbd_count"] == 1
    assert report["yolo_split_replacement_count"] == 2


def test_yolo_pose_matching_rejects_spatially_unrelated_candidate():
    poses = [_pose("bottle_01", 40)]

    assert match_yolo_detection_to_pose({"bbox_xyxy": [38, 38, 84, 168]}, poses) == 0
    assert match_yolo_detection_to_pose({"bbox_xyxy": [250, 40, 292, 166]}, poses) is None


def test_compact_live_status_keeps_multi_bottle_diagnostics():
    selection = {
        "selected_target_id": "bottle_02",
        "selection_status": "TARGET_SELECTED",
        "candidate_count": 2,
        "eligible_count": 1,
        "grasp_feasibility_score": 88.0,
        "grasp_feasibility_label": "OPTIMAL_GRASP",
        "grasp_feasibility_reason": "preferred_side_contact_band_has_estimated_clearance",
        "suggested_operator_action": "Follow camera-frame guidance.",
        "suggested_robot_action": "PLAN_ONLY_APPROACH_SELECTED_TARGET",
        "candidates": [{"target_id": "bottle_02", "selected": True, "approach": "SIDE"}],
    }

    compact = compact_live_status(
        {
            "state": "SELECT_BEST_GRASP",
            "detected_bottle_count": 2,
            "eligible_bottle_count": 1,
            "selected_bottle_target_id": "bottle_02",
            "yolo_model_backend": "opencv_dnn_darknet",
            "yolo_inference_ms": 25.0,
            "yolo_runtime_state": "PRELOCK_FUSED",
            "yolo_rgbd_fusion_mode": "ADDITIVE_SEMANTIC_PROPOSALS_RGBD_GATES_AUTHORITATIVE",
            "fused_pose_count": 2,
            "grasp_feasibility_score": 88.0,
            "grasp_feasibility_label": "OPTIMAL_GRASP",
            "grasp_feasibility_reason": "preferred_side_contact_band_has_estimated_clearance",
            "suggested_operator_action": "Follow camera-frame guidance.",
            "suggested_robot_action": "PLAN_ONLY_APPROACH_SELECTED_TARGET",
            "multi_bottle_target_selection": selection,
            "bottle_target_candidates": selection["candidates"],
            "selected_target_task_plan": {"strategy": "SIDE_ANTIPODAL", "approach": "SIDE"},
        }
    )

    assert compact["detected_bottle_count"] == 2
    assert compact["yolo_model_backend"] == "opencv_dnn_darknet"
    assert compact["yolo_runtime_state"] == "PRELOCK_FUSED"
    assert compact["fused_pose_count"] == 2
    assert compact["selected_bottle_target_id"] == "bottle_02"
    assert compact["multi_bottle_target_selection"]["candidate_count"] == 2
    assert compact["grasp_feasibility_label"] == "OPTIMAL_GRASP"
    assert compact["suggested_robot_action"] == "PLAN_ONLY_APPROACH_SELECTED_TARGET"
    assert compact["selected_target_task_plan"]["approach"] == "SIDE"

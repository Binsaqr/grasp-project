"""Camera-free tests for conservative RGB-D cap-like end-region estimation."""

import cv2
import numpy as np
import pytest

from bottle_pose import _pose_from_scored_component, depth_supported_achromatic_endcap
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.realtime_runner import (
    apply_skin_exclusion_to_cap_like_mask,
    fast_roi_bottle_pose,
    offset_pose_to_full,
    scale_pose_to_full,
)


FRAME_SHAPE = (240, 320)
OBJECT_DEPTH_MM = 400
BLUE_RGB = (30, 80, 220)
SILVER_RGB = (175, 175, 175)


def _oriented_rectangle(center, axis, length, width):
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    width_axis = np.array([-axis[1], axis[0]], dtype=float)
    points = np.asarray(
        [
            center - 0.5 * length * axis - 0.5 * width * width_axis,
            center - 0.5 * length * axis + 0.5 * width * width_axis,
            center + 0.5 * length * axis + 0.5 * width * width_axis,
            center + 0.5 * length * axis - 0.5 * width * width_axis,
        ]
    )
    return np.round(points).astype(np.int32)


def _scene(
    angle_deg=90.0,
    cap_depth_mm=OBJECT_DEPTH_MM,
    include_cap=True,
    include_side_object=False,
    cap_gap_px=0.0,
):
    angle = np.deg2rad(float(angle_deg))
    axis = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    center = np.array([160.0, 130.0], dtype=float)
    body_length = 120.0
    body_width = 44.0
    cap_length = 20.0
    cap_width = 36.0

    body_polygon = _oriented_rectangle(center, axis, body_length, body_width)
    negative_end = center - 0.5 * body_length * axis
    cap_center = negative_end - (0.5 * cap_length + float(cap_gap_px)) * axis
    cap_polygon = _oriented_rectangle(cap_center, axis, cap_length, cap_width)

    rgb = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
    depth = np.full(FRAME_SHAPE, 900, dtype=np.uint16)
    body_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    cap_truth = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    cv2.fillPoly(body_mask, [body_polygon], 1)
    rgb[body_mask > 0] = BLUE_RGB
    depth[body_mask > 0] = OBJECT_DEPTH_MM

    if include_cap:
        cv2.fillPoly(cap_truth, [cap_polygon], 1)
        rgb[cap_truth > 0] = SILVER_RGB
        depth[cap_truth > 0] = int(cap_depth_mm)

    if include_side_object:
        width_axis = np.array([-axis[1], axis[0]], dtype=float)
        side_center = center + 0.5 * (body_width + 30.0) * width_axis
        side_polygon = _oriented_rectangle(side_center, axis, 62.0, 30.0)
        side_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        cv2.fillPoly(side_mask, [side_polygon], 1)
        rgb[side_mask > 0] = SILVER_RGB
        depth[side_mask > 0] = OBJECT_DEPTH_MM

    return rgb, depth, body_mask, cap_truth


def _add_same_depth_support_plane(rgb, depth, angle_deg):
    """Add a broad shallow plane that the cap corridor must not crop into a cap."""
    angle = np.deg2rad(float(angle_deg))
    axis = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    center = np.array([160.0, 130.0], dtype=float)
    positive_end = center + 0.5 * 120.0 * axis
    support_polygon = _oriented_rectangle(
        positive_end + 8.0 * axis,
        axis,
        16.0,
        150.0,
    )
    support_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    cv2.fillPoly(support_mask, [support_polygon], 1)
    rgb[support_mask > 0] = SILVER_RGB
    depth[support_mask > 0] = OBJECT_DEPTH_MM
    return support_mask


def _seed_geometry(body_mask, angle_deg):
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    angle = np.deg2rad(float(angle_deg))
    return {
        "bbox_xyxy": [x, y, x + width, y + height],
        "bbox_xywh": [x, y, width, height],
        "contour": contour,
        "oriented_bbox": cv2.boxPoints(cv2.minAreaRect(contour)).tolist(),
        "center_px": [float(np.mean(np.where(body_mask > 0)[1])), float(np.mean(np.where(body_mask > 0)[0]))],
        "axis_vector": [float(np.cos(angle)), float(np.sin(angle))],
        "depth_median_mm": float(OBJECT_DEPTH_MM),
        "depth_confidence": 1.0,
        "detection_confidence": 1.0,
    }


@pytest.mark.parametrize("angle_deg", [90.0, 35.0])
def test_same_depth_cap_like_region_extends_upright_and_tilted_masks(angle_deg):
    rgb, depth, body_mask, cap_truth = _scene(angle_deg=angle_deg)

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    overlap = np.count_nonzero((cap_mask > 0) & (cap_truth > 0))
    assert evidence["status"] == "ESTIMATED_FROM_ALIGNED_RGBD_CAP_LIKE_END_GEOMETRY"
    assert evidence["detected"] is True
    assert evidence["physically_validated"] is False
    assert evidence["semantic_identity"] == "CAP_LIKE_END_REGION_NOT_SEMANTICALLY_CLASSIFIED"
    assert overlap >= 0.80 * np.count_nonzero(cap_mask)
    assert np.count_nonzero(extended) > np.count_nonzero(body_mask)
    assert np.all(depth[cap_mask > 0] == OBJECT_DEPTH_MM)


def test_different_depth_achromatic_end_region_is_not_merged():
    rgb, depth, body_mask, _cap_truth = _scene(cap_depth_mm=560)

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is False
    assert evidence["status"] == "NOT_DETECTED_NO_SUPPORTED_ACHROMATIC_END_REGION"
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


def test_same_depth_achromatic_side_object_is_not_treated_as_endcap():
    rgb, depth, body_mask, _cap_truth = _scene(include_cap=False, include_side_object=True)

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is False
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


@pytest.mark.parametrize("angle_deg", [90.0, 35.0])
def test_same_depth_support_plane_is_not_cropped_into_cap_shape(angle_deg):
    rgb, depth, body_mask, _cap_truth = _scene(angle_deg=angle_deg, include_cap=False)
    _add_same_depth_support_plane(rgb, depth, angle_deg)

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is False
    assert evidence["status"] == "NOT_DETECTED_UNBOUNDED_ACHROMATIC_SURFACE"
    assert evidence["unbounded_surface_reject_count"] >= 1
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


@pytest.mark.parametrize("angle_deg", [90.0, 35.0])
def test_true_cap_wins_when_opposite_same_depth_support_plane_is_present(angle_deg):
    rgb, depth, body_mask, cap_truth = _scene(angle_deg=angle_deg)
    support_mask = _add_same_depth_support_plane(rgb, depth, angle_deg)

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    cap_overlap = np.count_nonzero((cap_mask > 0) & (cap_truth > 0))
    support_overlap = np.count_nonzero((cap_mask > 0) & (support_mask > 0))
    assert evidence["detected"] is True
    assert evidence["axis_end_sign"] == -1
    assert evidence["end_window_support_fraction"] >= evidence["minimum_end_window_support_fraction"]
    assert cap_overlap >= 0.80 * np.count_nonzero(cap_mask)
    assert support_overlap == 0
    assert np.count_nonzero(extended) > np.count_nonzero(body_mask)


def test_sibling_instance_exclusion_prevents_cap_region_identity_bridge():
    rgb, depth, body_mask, _cap_truth = _scene()
    sibling_mask = np.zeros_like(body_mask)
    sibling_mask[20:50, 140:181] = 1

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(
        body_mask,
        rgb,
        depth,
        exclusion_mask=sibling_mask,
    )

    assert evidence["sibling_exclusion_applied"] is True
    assert evidence["detected"] is False
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


def test_missing_cap_depth_does_not_enable_rgb_only_extension():
    rgb, depth, body_mask, _cap_truth = _scene(cap_depth_mm=0)

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is False
    assert evidence["physically_validated"] is False
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


def test_cap_depth_is_compared_with_local_tilted_end_not_global_body_median():
    rgb, depth, body_mask, cap_truth = _scene()
    body_y, body_x = np.where(body_mask > 0)
    y_low, y_high = float(body_y.min()), float(body_y.max())
    depth[body_y, body_x] = np.round(
        460.0 - 100.0 * (body_y - y_low) / max(y_high - y_low, 1.0)
    ).astype(np.uint16)
    depth[cap_truth > 0] = 460

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is True
    assert evidence["end_reference_depth_source"] == "LOCAL_TERMINAL_BODY_DEPTH"
    assert abs(evidence["source_depth_median_mm"] - 410.0) < 5.0
    assert abs(evidence["end_reference_depth_median_mm"] - 460.0) < 15.0
    assert np.count_nonzero(cap_mask) > 0
    assert np.count_nonzero(extended) > np.count_nonzero(body_mask)


def test_tilted_end_still_rejects_cap_at_wrong_local_depth():
    rgb, depth, body_mask, cap_truth = _scene()
    body_y, body_x = np.where(body_mask > 0)
    y_low, y_high = float(body_y.min()), float(body_y.max())
    depth[body_y, body_x] = np.round(
        460.0 - 100.0 * (body_y - y_low) / max(y_high - y_low, 1.0)
    ).astype(np.uint16)
    depth[cap_truth > 0] = 540

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is False
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


def test_same_depth_long_narrow_axial_fragment_is_not_cap_like():
    rgb, depth, body_mask, _cap_truth = _scene(include_cap=False)
    rgb[31:69, 151:168] = SILVER_RGB
    depth[31:69, 151:168] = OBJECT_DEPTH_MM

    extended, cap_mask, evidence = depth_supported_achromatic_endcap(body_mask, rgb, depth)

    assert evidence["detected"] is False
    assert np.count_nonzero(cap_mask) == 0
    np.testing.assert_array_equal(extended, body_mask)


@pytest.mark.parametrize("angle_deg", [90.0, 35.0])
def test_pose_uses_cap_like_extent_but_preserves_pre_extension_grasp_body(angle_deg):
    rgb, depth, body_mask, _cap_truth = _scene(angle_deg=angle_deg)
    best = {
        "component": body_mask,
        "score": 8.0,
        "source": "rgb",
        "top_down_view_candidate": False,
        "depth_exclusion_mask": None,
    }

    pose, error = _pose_from_scored_component(rgb, depth, best)

    assert error is None
    assert pose is not None
    assert pose["cap_like_end_region"]["detected"] is True
    assert np.count_nonzero(pose["cap_like_mask"]) > 0
    assert np.count_nonzero((pose["cleaned_mask"] > 0) & (pose["cap_like_mask"] > 0)) == 0
    assert np.count_nonzero((pose["body_mask"] > 0) & (pose["cap_like_mask"] > 0)) == 0
    expected_y, expected_x = np.where(body_mask > 0)
    assert pose["center_px"] == pytest.approx(
        [float(np.mean(expected_x)), float(np.mean(expected_y))],
        abs=0.5,
    )
    cap_evidence = pose["cap_like_end_region"]
    assert cap_evidence["estimated_whole_extent_role"] == "DIAGNOSTIC_EXTENT_ONLY_NOT_TRACKING_GEOMETRY"
    assert cap_evidence["applied_to_tracking_geometry"] is False
    assert cap_evidence["tracking_geometry_source"] == "pre_extension_rgbd_body"
    assert cap_evidence["physically_validated"] is False
    assert pose["candidate_score_breakdown"]["cap_like_end_region"]["physically_validated"] is False


def test_pose_whole_extent_includes_depth_supported_cap_across_small_seam():
    rgb, depth, body_mask, cap_truth = _scene(cap_gap_px=2.0)
    best = {
        "component": body_mask,
        "score": 8.0,
        "source": "rgb",
        "top_down_view_candidate": False,
        "depth_exclusion_mask": None,
    }

    pose, error = _pose_from_scored_component(rgb, depth, best)

    assert error is None
    assert pose["cap_like_end_region"]["detected"] is True
    cap_y, _cap_x = np.where(cap_truth > 0)
    assert pose["bbox_xywh"][1] > int(cap_y.min())
    assert pose["cap_like_end_region"]["estimated_whole_extent_bbox_xywh"][1] <= int(cap_y.min())
    assert cv2.boundingRect(np.asarray(pose["body_contour"], dtype=np.int32))[1] > int(cap_y.min())


@pytest.mark.parametrize("angle_deg", [90.0, 35.0])
def test_fast_locked_target_path_keeps_cap_extent_diagnostic_and_body_authoritative(angle_deg):
    rgb, depth, body_mask, cap_truth = _scene(angle_deg=angle_deg)

    pose, error, _timings = fast_roi_bottle_pose(
        rgb,
        depth,
        _seed_geometry(body_mask, angle_deg),
        scale=1.0,
        depth_hint_mm=OBJECT_DEPTH_MM,
    )

    assert error is None
    assert pose["cap_like_end_region"]["detected"] is True
    tracking_bbox = cv2.boundingRect(np.asarray(pose["contour"], dtype=np.int32))
    grasp_body_bbox = cv2.boundingRect(np.asarray(pose["body_contour"], dtype=np.int32))
    estimated_bbox = pose["cap_like_end_region"]["estimated_whole_extent_bbox_xywh"]
    cap_y, cap_x = np.where(cap_truth > 0)
    assert tracking_bbox != grasp_body_bbox
    assert tracking_bbox[0] > int(cap_x.min()) or tracking_bbox[1] > int(cap_y.min())
    assert estimated_bbox[0] <= int(cap_x.min())
    assert estimated_bbox[1] <= int(cap_y.min())
    assert estimated_bbox[0] + estimated_bbox[2] > int(cap_x.max())
    assert estimated_bbox[1] + estimated_bbox[3] > int(cap_y.max())
    assert pose["cap_like_end_region"]["applied_to_tracking_geometry"] is False
    assert pose["candidate_score_breakdown"]["cap_like_end_region"]["physically_validated"] is False


def test_scaled_pose_cap_metrics_use_full_image_units_and_mask_is_not_serialized():
    cap_mask = np.zeros((20, 30), dtype=np.uint8)
    cap_mask[2:7, 8:18] = 1
    evidence = {
        "detected": True,
        "confidence": 0.9,
        "added_pixel_count": 50,
        "minimum_area_px": 8,
        "axial_span_px": 5.0,
        "lateral_span_px": 10.0,
        "search_roi_xyxy": [4, 1, 24, 12],
        "estimated_whole_extent_bbox_xywh": [5, 2, 18, 14],
        "coordinate_frame": "prelock_scaled_pixels",
        "physically_validated": False,
    }
    pose = {
        "axis_vector": [0.0, 1.0],
        "cap_like_mask": cap_mask,
        "cap_like_end_region": evidence,
        "candidate_score_breakdown": {"cap_like_end_region": evidence},
    }

    scaled = scale_pose_to_full(pose, (40, 60, 3), (20, 30, 3))
    scaled_evidence = scaled["cap_like_end_region"]

    assert scaled["cap_like_mask"].shape == (40, 60)
    assert scaled_evidence["coordinate_frame"] == "full_image_pixels"
    assert scaled_evidence["added_pixel_count"] == 200
    assert scaled_evidence["minimum_area_px"] == 32
    assert scaled_evidence["axial_span_px"] == pytest.approx(10.0)
    assert scaled_evidence["lateral_span_px"] == pytest.approx(20.0)
    assert scaled_evidence["search_roi_xyxy"] == [8.0, 2.0, 48.0, 24.0]
    assert scaled_evidence["estimated_whole_extent_bbox_xywh"] == [10.0, 4.0, 36.0, 28.0]
    assert scaled["candidate_score_breakdown"]["cap_like_end_region"] == scaled_evidence
    assert "cap_like_mask" not in realtime_runner.json_safe({"cap_like_mask": scaled["cap_like_mask"]})


def test_roi_cap_evidence_is_offset_into_full_image_coordinates():
    cap_mask = np.zeros((20, 30), dtype=np.uint8)
    cap_mask[2:7, 8:18] = 1
    evidence = {
        "detected": True,
        "search_roi_xyxy": [4, 1, 24, 12],
        "estimated_whole_extent_bbox_xywh": [5, 2, 18, 14],
        "coordinate_frame": "roi_crop_pixels",
        "physically_validated": False,
    }
    pose = {
        "cap_like_mask": cap_mask,
        "cap_like_end_region": evidence,
        "candidate_score_breakdown": {"cap_like_end_region": evidence},
        "bbox_xywh": [5, 2, 18, 14],
        "center_px": [14.0, 9.0],
    }

    offset = offset_pose_to_full(pose, (100, 200, 3), (100, 50, 130, 70))
    offset_evidence = offset["cap_like_end_region"]

    assert offset["cap_like_mask"].shape == (100, 200)
    assert offset_evidence["search_roi_xyxy"] == [104.0, 51.0, 124.0, 62.0]
    assert offset_evidence["estimated_whole_extent_bbox_xywh"] == [105.0, 52.0, 18.0, 14.0]
    assert offset_evidence["coordinate_frame"] == "full_image_pixels"
    assert offset_evidence["source_coordinate_frame"] == "roi_crop_pixels"
    assert offset_evidence["coordinate_offset_xy"] == [100.0, 50.0]
    assert offset["candidate_score_breakdown"]["cap_like_end_region"] == offset_evidence
    assert offset["fast_roi_xyxy"] == [100, 50, 130, 70]


def test_partial_skin_overlap_must_revalidate_cap_support_and_confidence():
    body = np.zeros((30, 30), dtype=np.uint8)
    body[12:25, 8:22] = 1
    cap = np.zeros_like(body)
    cap[2:12, 10:20] = 1
    evidence = {
        "status": "ESTIMATED_FROM_ALIGNED_RGBD_CAP_LIKE_END_GEOMETRY",
        "detected": True,
        "confidence": 0.95,
        "minimum_confidence": 0.55,
        "minimum_area_px": 8,
        "added_pixel_count": 100,
        "physically_validated": False,
    }

    mostly_occluded = np.zeros_like(body)
    mostly_occluded[2:8, 10:20] = 1
    whole, retained, rejected = apply_skin_exclusion_to_cap_like_mask(
        body, cap, evidence, mostly_occluded
    )
    assert rejected["detected"] is False
    assert rejected["cap_like_skin_retained_ratio"] == pytest.approx(0.4)
    assert np.count_nonzero(retained) == 0
    np.testing.assert_array_equal(whole, body)

    partly_occluded = np.zeros_like(body)
    partly_occluded[2:5, 10:20] = 1
    whole, retained, accepted = apply_skin_exclusion_to_cap_like_mask(
        body, cap, evidence, partly_occluded
    )
    assert accepted["detected"] is True
    assert accepted["cap_like_skin_retained_ratio"] == pytest.approx(0.7)
    assert accepted["confidence"] == pytest.approx(0.95 * 0.7)
    assert accepted["confidence"] >= accepted["minimum_confidence"]
    assert np.count_nonzero(retained) == 70
    assert np.count_nonzero(whole) == np.count_nonzero(body) + 70

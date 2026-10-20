"""Camera-free tests for hardened known-bottle detection cues."""

import cv2
import numpy as np

from bottle_pose import _pose_from_scored_component
from bottle_pose import (
    candidate_masks_from_rgb,
    depth_geometry_evidence,
    detect_bottle_pose,
    known_profile_agreement,
)


def _blank_frame(width=320, height=240):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint16)
    return rgb, depth


def _draw_blue_bottle(rgb, depth, center=(160, 120), size=(46, 150), angle=0.0, depth_mm=260):
    rect = (tuple(map(float, center)), tuple(map(float, size)), float(angle))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(rgb, [box], (35, 90, 210))
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box], 1)
    depth[mask > 0] = depth_mm
    return box


def test_known_bottle_candidate_passes_with_profile_scores():
    rgb, depth = _blank_frame()
    _draw_blue_bottle(rgb, depth)

    pose, error = detect_bottle_pose(rgb, depth)

    assert error is None
    assert pose["detected"] is True
    assert pose["bottle_profile_valid"] is True
    assert pose["bottle_profile_agreement_score"] >= 0.35
    assert pose["candidate_evidence_scores"]["RGB_contrast_score"] > 0.5
    assert pose["depth_support_ratio"] > 0.9
    assert pose["bottle_detection_status"] in {"CANDIDATE", "LOW_CONFIDENCE"}


def test_background_contour_is_rejected():
    rgb, depth = _blank_frame()
    rgb[:, :] = (120, 120, 120)
    depth[:, :] = 900

    pose, error = detect_bottle_pose(rgb, depth)

    assert pose is None
    assert error is not None


def test_tilted_bottle_uses_oriented_geometry():
    rgb, depth = _blank_frame()
    _draw_blue_bottle(rgb, depth, angle=24.0)

    pose, error = detect_bottle_pose(rgb, depth)

    assert error is None
    assert pose["oriented_bbox"]
    assert abs(float(pose["axis_angle_degrees"])) > 5.0
    assert pose["candidate_evidence_scores"]["oriented_aspect_score"] > 0.35


def test_profile_agreement_flags_bad_shape():
    result = known_profile_agreement(
        {
            "oriented_aspect_ratio": 1.0,
            "aspect_ratio": 1.0,
            "bottle_color_fraction": 0.0,
            "depth_valid_ratio": 0.1,
            "solidity_score": 0.2,
            "extent_score": 0.2,
            "mask_contamination_score": 0.8,
        }
    )

    assert result["bottle_profile_valid"] is False
    assert result["bottle_profile_failures"]


def test_shadowed_bottle_side_is_completed_only_by_connected_depth():
    rgb, depth = _blank_frame()
    rgb[45:205, 130:160] = (35, 90, 210)
    rgb[45:205, 160:190] = (3, 8, 20)
    depth[45:205, 130:190] = 420

    masks = candidate_masks_from_rgb(rgb, depth)
    pose, error = detect_bottle_pose(rgb, depth)

    assert any(cv2.boundingRect(mask.astype(np.uint8))[2] >= 58 for mask in masks)
    assert error is None
    assert pose["bbox_xywh"][2] >= 58
    assert abs(float(pose["center_px"][0]) - 159.5) <= 2.0


def test_shadow_completion_does_not_merge_disconnected_same_depth_bottle():
    rgb, depth = _blank_frame()
    rgb[45:205, 80:110] = (35, 90, 210)
    rgb[45:205, 110:140] = (3, 8, 20)
    depth[45:205, 80:140] = 420
    rgb[45:205, 152:182] = (35, 90, 210)
    depth[45:205, 152:212] = 420

    masks = candidate_masks_from_rgb(rgb, depth)
    boxes = [cv2.boundingRect(mask.astype(np.uint8)) for mask in masks]

    assert any(x <= 82 and 58 <= width <= 62 for x, _y, width, _height in boxes)
    assert all(width < 100 for _x, _y, width, _height in boxes)


def test_raw_rgb_extent_cannot_replace_depth_refined_tracking_geometry():
    rgb, depth = _blank_frame(width=480, height=270)
    raw_mask = np.zeros(depth.shape, dtype=np.uint8)
    raw_mask[80:217, 219:261] = 1
    rgb[raw_mask > 0] = (35, 90, 210)
    # The larger lower lobe sets the median. The upper lobe is valid depth but
    # far enough away that the median band removes it, reproducing the live
    # curved/reflective bottle fragmentation.
    depth[80:130, 219:261] = 470
    depth[130:217, 219:261] = 372
    best = {
        "component": raw_mask,
        "score": 8.0,
        "source": "rgb",
        "top_down_view_candidate": False,
        "depth_exclusion_mask": None,
    }

    pose, error = _pose_from_scored_component(
        rgb,
        depth,
        best,
        intrinsics={"fx": 420.0, "fy": 420.0},
    )

    assert pose is None
    assert "dimension plausibility" in error.lower()


def _mixed_boundary_depth_fixture(*, planar=False, clustered=False):
    """Build the live D405 edge-fill signature without camera hardware."""
    height, width = 240, 360
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[48:202, 145:205] = 1
    depth = np.full((height, width), 289, dtype=np.uint16)
    ys, xs = np.where(mask > 0)
    if planar:
        depth[mask > 0] = 280
    else:
        # Robustly non-planar by eight millimetres, matching the saved bottle
        # evidence (inside median about 280 mm and MAD about 4 mm).
        depth[ys, xs] = np.where(xs < 175, 276, 284).astype(np.uint16)

    radius = int(np.clip(round(0.14 * 60), 4, 14))
    dilated = cv2.dilate(
        mask,
        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
    )
    ring = (dilated > 0) & (mask == 0)
    ring_y, ring_x = np.where(ring)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    angle = np.mod(np.arctan2(ring_y - center_y, ring_x - center_x), 2.0 * np.pi)
    sector = np.floor((angle + np.pi / 8.0) / (np.pi / 4.0)).astype(int) % 8
    far_budget = int(round(0.107 * ring_x.size))
    selected = []
    allowed_sectors = (0,) if clustered else (0, 4)
    per_sector = max(1, far_budget // len(allowed_sectors))
    for sector_index in allowed_sectors:
        indices = np.flatnonzero(sector == sector_index)[:per_sector]
        selected.extend(indices.tolist())
    if len(selected) < far_budget:
        remaining = [index for index in range(ring_x.size) if index not in set(selected)]
        selected.extend(remaining[: far_budget - len(selected)])
    selected = np.asarray(selected[:far_budget], dtype=int)
    depth[ring_y[selected], ring_x[selected]] = 381
    return mask, depth


def test_mixed_boundary_accepts_live_curved_bottle_signature_with_distributed_background():
    mask, depth = _mixed_boundary_depth_fixture()

    evidence = depth_geometry_evidence(mask, depth)

    assert evidence["foreground_separated"] is False
    assert 0.09 <= evidence["far_background_support_fraction"] <= 0.12
    assert evidence["far_background_support_pixel_count"] >= 32
    assert evidence["far_background_supported_sector_count"] >= 2
    assert evidence["far_background_opposite_sector_pair_count"] >= 1
    assert evidence["far_background_distributed"] is True
    assert evidence["surface_depth_interdecile_range_mm"] >= 6.0
    assert evidence["mixed_boundary_foreground_supported"] is True


def test_mixed_boundary_rejects_clustered_far_patch_even_at_live_support_ratio():
    mask, depth = _mixed_boundary_depth_fixture(clustered=True)

    evidence = depth_geometry_evidence(mask, depth)

    assert evidence["far_background_support_fraction"] >= 0.08
    assert evidence["far_background_distributed"] is False
    assert evidence["mixed_boundary_foreground_supported"] is False


def test_mixed_boundary_rejects_planar_blue_card_on_support_plane():
    mask, depth = _mixed_boundary_depth_fixture(planar=True)

    evidence = depth_geometry_evidence(mask, depth)

    assert evidence["far_background_distributed"] is True
    assert evidence["surface_depth_interdecile_range_mm"] == 0.0
    assert evidence["mixed_boundary_foreground_supported"] is False


def test_nearest_hand_depth_cannot_own_candidate_from_sparse_navy_overlap():
    rgb, depth = _blank_frame()
    depth[:, :] = 650
    # A close hand/fixture-like rectangle fills the former central near-depth
    # fallback. Only a thin navy stripe overlaps it, below bottle geometry.
    depth[60:205, 130:190] = 250
    rgb[60:205, 136:144] = (15, 35, 80)

    masks = candidate_masks_from_rgb(rgb, depth)
    boxes = [cv2.boundingRect(candidate.astype(np.uint8)) for candidate in masks]

    assert all(not (width >= 50 and height >= 120) for _x, _y, width, height in boxes)

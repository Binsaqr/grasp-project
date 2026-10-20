import cv2
import numpy as np

from depth_utils import smooth_depth_pipeline
from grasp_project.realsense.object_state import estimate_object_state


def _scene(angle_deg=0.0, occluded=False):
    mask = np.zeros((140, 160), dtype=np.uint8)
    box = cv2.boxPoints(((80, 78), (28, 88), angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 1)
    depth = np.full(mask.shape, 1250.0, dtype=float)
    depth[mask > 0] = 1000.0
    if occluded:
        ys, xs = np.where(mask > 0)
        y0 = int(ys.min())
        depth[max(0, y0 - 8):y0, int(xs.min()):int(xs.max()) + 1] = 850.0
    return mask, depth


def test_standing_bottle_and_clear_pickup_region():
    mask, depth = _scene(0.0)
    state = estimate_object_state(mask, depth)
    assert state.orientation == "STANDING"
    assert state.available_for_pickup
    assert not state.occluded
    assert state.evidence["availability"]["status"] == "ESTIMATED_FROM_ALIGNED_MASK_AND_DEPTH"


def test_lying_bottle():
    mask, depth = _scene(90.0)
    state = estimate_object_state(mask, depth)
    assert state.orientation == "LYING"
    assert state.available_for_pickup


def test_tilted_bottle():
    mask, depth = _scene(45.0)
    state = estimate_object_state(mask, depth)
    assert state.orientation == "TILTED"
    assert 25 < state.tilt_angle_deg < 65


def test_top_down_pose_is_diagnostic_and_available_but_not_claimed_standing():
    mask = np.zeros((140, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 78), 28, 1, thickness=-1)
    depth = np.full(mask.shape, 1250.0, dtype=float)
    depth[mask > 0] = 1000.0

    state = estimate_object_state(
        mask,
        depth,
        pose={
            "top_down_view_candidate": True,
            "candidate_evidence_scores": {
                "oriented_aspect_ratio": 1.0,
                "solidity_score": 0.95,
            },
        },
    )

    assert state.orientation == "UNKNOWN"
    assert state.pose_class == "TOP_DOWN_BODY"
    assert state.available_for_pickup
    assert state.recommended_strategy == "TOP_DOWN_PICK"


def test_occluder_above_top_band_blocks_pickup():
    mask, depth = _scene(0.0, occluded=True)
    state = estimate_object_state(mask, depth)
    assert state.occluded
    assert not state.available_for_pickup
    assert "clear_space_above_top_band_blocked" in state.reasons


def test_depth_discontinuity_on_bottle_blocks_pickup():
    mask, depth = _scene(0.0)
    ys, xs = np.where(mask > 0)
    selected = np.arange(len(xs))[: max(1, len(xs) // 6)]
    depth[ys[selected], xs[selected]] = 850.0
    state = estimate_object_state(mask, depth)
    assert state.occluded
    assert "closer_depth_discontinuity_on_bottle" in state.reasons


def test_missing_depth_above_bottle_is_unknown_not_clear():
    mask, depth = _scene(0.0)
    ys, xs = np.where(mask > 0)
    y0 = int(ys.min())
    depth[max(0, y0 - 20):y0, int(xs.min()):int(xs.max()) + 1] = 0.0
    state = estimate_object_state(mask, depth)

    assert not state.available_for_pickup
    assert not state.occluded
    assert "insufficient_clear_space_depth" in state.reasons


def test_close_tall_bottle_uses_local_top_boundary_neighborhood():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[129:407, 240:324] = 1
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = 322
    # Twelve rows of valid, safely more-distant depth immediately above the
    # cap are enough for the local 42 px neighbourhood.  They would be diluted
    # below the unchanged 25% gate by the old 56 px height-scaled region.
    depth[117:129, 240:324] = 1200

    state = estimate_object_state(mask, depth)

    assert state.available_for_pickup
    assert not state.occluded
    assert state.evidence["availability"]["clear_space_valid_ratio"] >= 0.25


def test_saturated_depth_above_bottle_is_not_counted_as_clear_space():
    mask = np.zeros((160, 180), dtype=np.uint8)
    mask[45:145, 70:110] = 1
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = 500
    depth[25:45, 70:110] = np.iinfo(np.uint16).max

    state = estimate_object_state(mask, depth)

    assert not state.available_for_pickup
    assert not state.occluded
    assert state.evidence["availability"]["clear_space_valid_ratio"] == 0.0
    assert "insufficient_clear_space_depth" in state.reasons


def test_close_tall_bottle_local_top_occluder_still_blocks_pickup():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[129:407, 240:324] = 1
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = 322
    depth[117:129, 240:324] = 250

    state = estimate_object_state(mask, depth)

    assert state.occluded
    assert not state.available_for_pickup
    assert "clear_space_above_top_band_blocked" in state.reasons


def test_raw_and_smoothed_depth_agree_when_far_values_are_saturated():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[129:407, 240:324] = 1
    raw_depth = np.full(mask.shape, np.iinfo(np.uint16).max, dtype=np.uint16)
    raw_depth[mask > 0] = 322
    raw_depth[117:129, 240:324] = 1200

    raw_state = estimate_object_state(mask, raw_depth)
    smooth_state = estimate_object_state(mask, smooth_depth_pipeline(raw_depth))

    assert raw_state.available_for_pickup == smooth_state.available_for_pickup
    assert raw_state.occluded == smooth_state.occluded
    assert (
        raw_state.evidence["availability"]["clear_space_valid_ratio"]
        == smooth_state.evidence["availability"]["clear_space_valid_ratio"]
    )


def test_unsafe_pointcloud_crop_blocks_without_claiming_occlusion():
    mask, depth = _scene(0.0)
    state = estimate_object_state(
        mask,
        depth,
        pointcloud={
            "pointcloud_current": True,
            "bottle_roi_available": True,
            "pointcloud_snapshot_consistent": True,
            "bottle_crop_safe_for_contacts": False,
        },
    )

    assert not state.available_for_pickup
    assert not state.occluded
    assert "pointcloud_crop_not_safe_for_contacts" in state.reasons


def test_hand_demo_can_use_current_pointcloud_as_advisory_only():
    mask, depth = _scene(0.0)
    state = estimate_object_state(
        mask,
        depth,
        pointcloud={
            "pointcloud_current": True,
            "bottle_roi_available": True,
            "pointcloud_snapshot_consistent": True,
            "bottle_crop_safe_for_contacts": False,
        },
        pointcloud_veto_enabled=False,
    )

    assert state.available_for_pickup
    assert state.evidence["availability"]["pointcloud_blocked"] is False
    assert state.evidence["availability"]["pointcloud_evidence_role"] == "ADVISORY_GEOMETRY_ONLY"
    assert "pointcloud_crop_advisory_not_used_as_hand_demo_veto" in state.reasons


def test_stale_or_unscoped_pointcloud_cannot_veto_current_rgbd_state():
    mask, depth = _scene(0.0)
    state = estimate_object_state(
        mask,
        depth,
        pointcloud={
            "pointcloud_freshness": "STALE",
            "bottle_roi_available": False,
            "bottle_crop_safe_for_contacts": False,
        },
    )

    assert state.available_for_pickup
    assert state.evidence["availability"]["pointcloud_evidence_applicable"] is False
    assert "pointcloud_crop_not_safe_for_contacts" not in state.reasons

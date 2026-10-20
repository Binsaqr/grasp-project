"""Camera-free regressions for RGB-D finger evidence near the clamp guides."""

import numpy as np
import pytest

from grasp_project.dashboard import web_dashboard
from grasp_project.realsense import realtime_runner


SKIN_RGB = np.array([210, 145, 105], dtype=np.uint8)


@pytest.fixture(autouse=True)
def _deterministic_finger_config(monkeypatch):
    monkeypatch.setattr(realtime_runner, "ENABLE_FINGER_DETECTION", True)
    monkeypatch.setattr(realtime_runner, "FINGER_DETECTION_MODE", "skin_depth")
    monkeypatch.setattr(realtime_runner, "FINGER_MIN_AREA_PX", 80)
    monkeypatch.setattr(realtime_runner, "FINGER_MAX_AREA_PX", 9000)
    monkeypatch.setattr(realtime_runner, "FINGER_CONFIDENCE_THRESHOLD", 35.0)
    monkeypatch.setattr(realtime_runner, "FINGER_DEPTH_TOLERANCE_MM", 90.0)
    monkeypatch.setattr(realtime_runner, "FINGER_ROI_PAD_PX", 36.0)
    monkeypatch.setattr(realtime_runner, "FINGER_ROI_MAX_PAD_PX", 120.0)
    monkeypatch.setattr(realtime_runner, "FINGER_ROI_PATCH_SCALE", 0.35)
    monkeypatch.setattr(realtime_runner, "FINGER_MAX_CONTACT_DISTANCE_PX", 55.0)
    monkeypatch.setattr(realtime_runner, "FINGER_BOTTLE_DEPTH_MAX_DELTA_MM", 60.0)
    monkeypatch.setattr(realtime_runner, "FINGER_BOTTLE_DEPTH_RELATIVE_DELTA", 0.15)
    monkeypatch.setattr(realtime_runner, "FINGER_STABLE_FRAMES_REQUIRED", 3)


def _target_clamp():
    """Mirror the synthetic clamp already used by depth-guidance tests."""
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


def _scene(shape=(200, 200), depth_mm=600):
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    depth = np.full(shape, depth_mm, dtype=np.uint16)
    return rgb, depth


def _paint_skin(rgb, depth, bbox, *, depth_mm=600):
    x1, y1, x2, y2 = bbox
    rgb[y1:y2, x1:x2] = SKIN_RGB
    depth[y1:y2, x1:x2] = depth_mm


def test_elongated_component_reaching_guide_wins_despite_far_centroid(monkeypatch):
    rgb, depth = _scene()
    _paint_skin(rgb, depth, [48, 95, 83, 106])
    # The component centroid is about 15 px from the guide, but its near edge
    # reaches it. Tightening the normal gate proves scoring is contour-nearest,
    # rather than centroid-only.
    monkeypatch.setattr(realtime_runner, "FINGER_MAX_CONTACT_DISTANCE_PX", 12.0)

    result = realtime_runner.detect_finger_in_roi(
        rgb,
        depth,
        "left",
        _target_clamp(),
        rgb.shape,
        reference_bottle_depth_mm=600.0,
    )

    assert result["detected"] is True
    assert result["bbox"] is not None
    assert result["contact_px"] is not None
    assert result["distance_to_expected_px"] <= 12.0


def test_wrong_depth_skin_is_only_a_visual_candidate():
    rgb, depth = _scene()
    _paint_skin(rgb, depth, [71, 92, 90, 109], depth_mm=1050)

    result = realtime_runner.detect_finger_in_roi(
        rgb,
        depth,
        "left",
        _target_clamp(),
        rgb.shape,
        reference_bottle_depth_mm=600.0,
    )

    assert result["detected"] is False
    assert result["candidate_bbox"] is not None
    assert result["candidate_contour"]
    assert result.get("bbox") is None
    assert result.get("contour") is None
    assert result.get("contact_px") is None
    assert result["expected_depth_mm"] == 600.0


def test_one_rgbd_component_never_becomes_bilateral_contact(monkeypatch):
    monkeypatch.setattr(
        realtime_runner, "ENABLE_VISUAL_FINGER_DETECTION_IN_DEMO", True
    )
    rgb, depth = _scene()
    # Both search ROIs overlap here and both guides are within the raw distance
    # gate. The same physical component still owns only its nearest (left) side.
    _paint_skin(rgb, depth, [76, 91, 96, 110])
    state = {}

    for _ in range(realtime_runner.FINGER_STABLE_FRAMES_REQUIRED + 1):
        result = realtime_runner.detect_finger_contacts(
            rgb,
            depth,
            _target_clamp(),
            rgb.shape,
            state,
            reference_bottle_depth_mm=600.0,
        )

        assert result["raw_finger_detection_status"] == "LEFT_ONLY"
        assert result["finger_detection_status"] == "LEFT_ONLY"
        assert result["left_finger_bbox"] is not None
        assert result.get("right_finger_bbox") is None
        assert result.get("right_finger_contact_px") is None
        assert result["finger_stable_frames"] == 0
        assert result["contact_source"] == "planned_contact"


def test_finger_contact_points_hold_small_jitter_but_follow_real_motion(monkeypatch):
    monkeypatch.setattr(
        realtime_runner, "ENABLE_VISUAL_FINGER_DETECTION_IN_DEMO", True
    )
    state = {}
    samples = {
        "left": [[79.0, 100.0], [80.5, 100.5], [89.0, 100.0]],
        "right": [[121.0, 100.0], [120.0, 99.0], [131.0, 100.0]],
    }
    indices = {"left": 0, "right": 0}

    def fake_detect(_rgb, _depth, side, target_clamp, _image_shape, reference_bottle_depth_mm=None):
        index = indices[side]
        indices[side] += 1
        contact = samples[side][index]
        return {
            "detected": True,
            "visual_detected": True,
            "bbox": [40, 80, 70, 120] if side == "left" else [130, 80, 160, 120],
            "contour": [[40, 80], [70, 80], [70, 120], [40, 120]] if side == "left" else [[130, 80], [160, 80], [160, 120], [130, 120]],
            "contact_px": contact,
            "depth_mm": reference_bottle_depth_mm,
            "depth_verified_against_bottle": True,
            "depth_delta_to_bottle_mm": 0.0,
            "confidence": 90.0,
            "distance_to_expected_px": 1.0,
        }

    monkeypatch.setattr(realtime_runner, "detect_finger_in_roi", fake_detect)
    rgb, depth = _scene()
    results = [
        realtime_runner.detect_finger_contacts(
            rgb,
            depth,
            _target_clamp(),
            rgb.shape,
            state,
            reference_bottle_depth_mm=600.0,
        )
        for _ in range(3)
    ]

    assert results[1]["raw_left_finger_contact_px"] != results[0]["raw_left_finger_contact_px"]
    assert results[1]["left_finger_contact_px"] == results[0]["left_finger_contact_px"]
    assert results[1]["right_finger_contact_px"] == results[0]["right_finger_contact_px"]
    assert results[1]["left_finger_contact_filter"]["filter_mode"] == "stable_deadband"
    assert results[2]["left_finger_contact_px"][0] > results[1]["left_finger_contact_px"][0]
    assert results[2]["left_finger_contact_px"][0] < results[2]["raw_left_finger_contact_px"][0]
    assert results[2]["finger_contact_midpoint_px"] is not None
    assert results[2]["finger_contact_display_smoothed"] is True


def _single_patch_clamp(scale):
    cx = cy = 400.0
    half_width = 6.0 * scale
    half_height = 10.0 * scale
    return {
        "left_contact_px": [cx, cy],
        "left_clamp_patch": {
            "center_px": [cx, cy],
            "polygon": [
                [cx - half_width, cy - half_height],
                [cx + half_width, cy - half_height],
                [cx + half_width, cy + half_height],
                [cx - half_width, cy + half_height],
            ],
        },
    }


def test_finger_roi_padding_scales_with_projected_patch_size():
    image_shape = (800, 800, 3)
    small_clamp = _single_patch_clamp(scale=1.0)
    large_clamp = _single_patch_clamp(scale=8.0)

    small_roi = realtime_runner.finger_search_roi("left", small_clamp, image_shape)
    large_roi = realtime_runner.finger_search_roi("left", large_clamp, image_shape)
    small_patch_width = 12.0
    large_patch_width = 96.0
    small_horizontal_padding = small_roi[2] - small_roi[0] - small_patch_width
    large_horizontal_padding = large_roi[2] - large_roi[0] - large_patch_width

    assert large_horizontal_padding > small_horizontal_padding
    assert 0 <= large_roi[0] < large_roi[2] <= image_shape[1]
    assert 0 <= large_roi[1] < large_roi[3] <= image_shape[0]


def test_estimated_finger_depth_and_skin_occlusion_reach_compact_dashboard(monkeypatch):
    monkeypatch.setattr(realtime_runner, "FAST_LIVE_STATUS_COMPACT", True)
    compact = realtime_runner.compact_live_status(
        {
            "state": "APPROACH_TRACKING",
            "left_finger_depth_mm": 605.0,
            "right_finger_depth_mm": 612.0,
            "left_finger_depth_verified": True,
            "right_finger_depth_verified": True,
            "left_finger_depth_delta_to_bottle_mm": 5.0,
            "right_finger_depth_delta_to_bottle_mm": 12.0,
            "finger_bottle_depth_reference_mm": 600.0,
            "finger_bottle_depth_reference_source": "current_masked_bottle_depth_estimate",
            "finger_bottle_depth_tolerance_mm": 60.0,
            "finger_detection": {
                "contact_estimate_type": "ESTIMATED_VISUAL_NEAR_CONTACT",
                "physical_contact_validated": False,
                "validation_status": "NOT_PHYSICALLY_VALIDATED",
            },
            "target_lock": {
                "current_tracked_target": {
                    "roi_candidate_skin_exclusion_applied": True,
                    "roi_candidate_skin_overlap_seed_pixels": 42,
                    "roi_candidate_hand_occlusion": True,
                }
            },
        }
    )

    assert compact["finger_contact_estimate_type"] == "ESTIMATED_VISUAL_NEAR_CONTACT"
    assert compact["finger_physical_contact_validated"] is False
    assert compact["finger_validation_status"] == "NOT_PHYSICALLY_VALIDATED"
    assert compact["roi_candidate_skin_exclusion_applied"] is True
    assert compact["roi_candidate_skin_overlap_seed_pixels"] == 42
    assert compact["roi_candidate_hand_occlusion"] is True
    assert compact["roi_candidate_hand_occlusion_state"] == "POSSIBLE_HAND_SKIN_OCCLUSION_ESTIMATED"
    assert compact["roi_candidate_hand_occlusion_validation_status"] == "NOT_PHYSICALLY_VALIDATED"

    summary = web_dashboard.summarize_realsense(compact)

    assert summary["finger_bottle_depth_reference_mm"] == 600.0
    assert summary["finger_contact_estimate_type"] == "ESTIMATED_VISUAL_NEAR_CONTACT"
    assert summary["finger_physical_contact_validated"] is False
    assert summary["finger_validation_status"] == "NOT_PHYSICALLY_VALIDATED"
    assert summary["roi_candidate_hand_occlusion"] is True
    assert summary["roi_candidate_hand_occlusion_state"] == "POSSIBLE_HAND_SKIN_OCCLUSION_ESTIMATED"
    assert summary["roi_candidate_hand_occlusion_validation_status"] == "NOT_PHYSICALLY_VALIDATED"
    assert "function fmtFingerBottleDepth(rs)" in web_dashboard.INDEX_HTML
    assert "function fmtSkinOcclusion(rs)" in web_dashboard.INDEX_HTML
    assert "['Finger RGB-D', fmtFingerBottleDepth(rs)]" in web_dashboard.INDEX_HTML
    assert "['Skin Occlusion', fmtSkinOcclusion(rs)]" in web_dashboard.INDEX_HTML

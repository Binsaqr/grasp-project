"""Camera-free fail-safe tests for canonical RealSense perception status."""

from grasp_project.realsense.perception_result import (
    CONTACT_POSSIBLE,
    CONTACT_SOURCE_DETECTED_GRIPPER,
    CONTACT_SOURCE_PLANNED,
    CONTACT_SOURCE_UNAVAILABLE,
    CONTACT_SOURCE_VISUAL,
    build_realsense_perception_result,
    validate_contact_source,
)


def test_no_bottle_frame_never_becomes_verified():
    result = build_realsense_perception_result({"frame_id": 1, "target_status": "NO_BOTTLE"})

    assert result.bottle_detection_status == "NO_BOTTLE"
    assert result.perception_ready is False
    assert "missing_bottle_bbox" in result.perception_blockers


def test_stale_frame_bundle_is_rejected():
    result = build_realsense_perception_result(
        {
            "frame_id": 10,
            "rgb_frame_id": 10,
            "depth_frame_id": 7,
            "target_status": "TRACKING",
            "bottle_extent_bbox_xyxy": [10, 20, 80, 180],
        }
    )

    assert result.snapshot_consistent is False
    assert result.max_frame_delta == 3
    assert "snapshot_inconsistent" in result.perception_blockers


def test_one_finger_cannot_create_bilateral_detected_contact():
    check = validate_contact_source(
        raw_source="detected_finger",
        left_point=[25, 50],
        right_point=None,
        left_depth_mm=250,
        right_depth_mm=None,
        left_confidence=80,
        right_confidence=0,
        stable_frames=8,
    )

    assert check["contact_source"] == CONTACT_SOURCE_UNAVAILABLE
    assert check["contact_pair_valid"] is False
    assert check["detected_contact_count"] == 1
    assert "missing_right_contact_point" in check["reasons"]


def test_stable_bilateral_detected_contact_passes():
    check = validate_contact_source(
        raw_source="detected_gripper",
        left_point=[25, 50],
        right_point=[85, 51],
        left_depth_mm=250,
        right_depth_mm=253,
        left_confidence=75,
        right_confidence=78,
        stable_frames=5,
    )

    assert check["contact_source"] == CONTACT_SOURCE_DETECTED_GRIPPER
    assert check["contact_pair_valid"] is True
    assert check["fallback_reason"] is None


def test_visual_finger_pair_is_estimated_not_physical_contact():
    result = build_realsense_perception_result(
        {
            "frame_id": 5,
            "target_status": "TRACKING",
            "bottle_extent_bbox_xyxy": [10, 20, 80, 180],
            "contact_source": "detected_finger",
            "left_contact_px": [10, 80],
            "right_contact_px": [80, 80],
            "left_contact_depth_mm": 240,
            "right_contact_depth_mm": 242,
            "contact_confidence_left": 75,
            "contact_confidence_right": 78,
            "stable_frames": 5,
            "physical_contact_validated": False,
        }
    )

    assert result.contact_pair_valid is True
    assert result.contact_source == CONTACT_SOURCE_VISUAL
    assert result.left_contact_status == CONTACT_POSSIBLE
    assert result.right_contact_status == CONTACT_POSSIBLE


def test_planned_contact_is_honest_not_detected():
    result = build_realsense_perception_result(
        {
            "frame_id": 3,
            "target_status": "TRACKING",
            "bottle_extent_bbox_xyxy": [10, 20, 80, 180],
            "contact_source": "planned_contact",
            "left_contact_px": [10, 80],
            "right_contact_px": [80, 80],
        }
    )

    assert result.contact_source == CONTACT_SOURCE_PLANNED
    assert result.contact_pair_valid is False
    assert result.contact_source_fallback_reason is not None


def test_ready_to_close_blocked_by_weak_detected_contact_evidence():
    result = build_realsense_perception_result(
        {
            "frame_id": 4,
            "target_status": "TRACKING",
            "ready_to_close": True,
            "bottle_extent_bbox_xyxy": [10, 20, 80, 180],
            "contact_source": "detected_finger",
            "left_contact_px": [10, 80],
            "right_contact_px": [80, 80],
            "left_contact_depth_mm": 240,
            "right_contact_depth_mm": 241,
            "contact_confidence_left": 20,
            "contact_confidence_right": 70,
            "stable_frames": 5,
        }
    )

    assert result.perception_ready is False
    assert result.contact_source == CONTACT_SOURCE_UNAVAILABLE
    assert "left_contact_confidence_below_threshold" in result.perception_blockers

import numpy as np

from bottle_visibility import check_full_bottle_visibility


def _pose(bbox):
    x, y, width, height = bbox
    mask = np.zeros((270, 480), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 1
    return {
        "detected": True,
        "bbox_xywh": list(bbox),
        "cleaned_mask": mask,
        "center_px": [x + width / 2.0, y + height / 2.0],
        "depth_confidence": 0.95,
        "area_ratio": float(np.count_nonzero(mask) / mask.size),
    }


def test_physical_false_lock_fragment_is_not_a_full_bottle():
    result = check_full_bottle_visibility(_pose((12, 132, 45, 88)), (270, 480, 3))

    assert result["full_bottle_visible"] is False
    assert result["required_margin_px"] == 13.5
    assert result["bbox_size_ratio"]["long_extent"] < result["min_long_extent_ratio"]
    assert any("left" in reason.lower() for reason in result["missing_region_reason"])
    assert any("too short" in reason.lower() for reason in result["missing_region_reason"])


def test_live_black_hand_fixture_bbox_is_rejected_at_bottom_margin():
    # Exact deployed false-lock artifact: [x=206,y=120,w=42,h=139] in 480x270.
    result = check_full_bottle_visibility(_pose((206, 120, 42, 139)), (270, 480, 3))

    assert result["full_bottle_visible"] is False
    assert any("bottom" in reason.lower() for reason in result["missing_region_reason"])


def test_centered_full_length_bottle_passes_strict_extent_and_margin_gate():
    result = check_full_bottle_visibility(_pose((210, 45, 50, 170)), (270, 480, 3))

    assert result["full_bottle_visible"] is True
    assert result["bbox_size_ratio"]["long_extent"] >= result["min_long_extent_ratio"]
    assert result["missing_region_reason"] == []

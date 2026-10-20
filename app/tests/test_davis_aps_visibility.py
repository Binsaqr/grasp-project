"""Camera-free tests for DAVIS APS brightness/visibility diagnostics.

Raw APS statistics are the honest measurement; display enhancement is
presentation only and must never alter them.
"""

import numpy as np
import pytest

from grasp_project.davis.aps_stream import (
    classify_aps_visibility,
    compute_aps_stats,
    enhance_aps_display,
)

H, W = 260, 346


def frame(values):
    return np.asarray(values, dtype=np.uint8)


def dark_frame():
    rng = np.random.default_rng(1)
    return frame(rng.integers(0, 25, size=(H, W)))


def good_frame():
    rng = np.random.default_rng(2)
    return frame(np.clip(rng.normal(120, 55, size=(H, W)), 0, 255))


def flat_frame():
    rng = np.random.default_rng(3)
    return frame(np.clip(rng.normal(128, 4, size=(H, W)), 0, 255))


def saturated_frame():
    rng = np.random.default_rng(4)
    return frame(np.clip(rng.normal(247, 4, size=(H, W)), 220, 255))


def test_dark_frame_reports_low_light():
    stats = compute_aps_stats(dark_frame())
    vis = classify_aps_visibility(stats)
    assert vis["aps_visibility_status"] in {"LOW_LIGHT", "UNUSABLE"}
    assert vis["aps_underexposed"] is True
    assert "underexposed" in (vis["aps_display_warning"] or "").lower()
    assert vis["aps_visibility_score"] is not None and vis["aps_visibility_score"] < 30.0


def test_high_dynamic_range_frame_is_good():
    vis = classify_aps_visibility(compute_aps_stats(good_frame()))
    assert vis["aps_visibility_status"] == "GOOD"
    assert vis["aps_underexposed"] is False and vis["aps_overexposed"] is False
    assert vis["aps_visibility_score"] > 50.0


def test_flat_frame_reports_low_contrast():
    vis = classify_aps_visibility(compute_aps_stats(flat_frame()))
    assert vis["aps_visibility_status"] == "LOW_CONTRAST"
    assert vis["aps_low_contrast"] is True


def test_saturated_frame_reports_overexposed():
    vis = classify_aps_visibility(compute_aps_stats(saturated_frame()))
    assert vis["aps_visibility_status"] == "OVEREXPOSED"
    assert vis["aps_overexposed"] is True


def test_display_enhancement_brightens_preview_but_raw_stats_unchanged():
    cv2 = pytest.importorskip("cv2")
    raw = dark_frame()
    stats_before = compute_aps_stats(raw)
    rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    enhanced, meta = enhance_aps_display(rgb, mode="auto_normalize", gamma=0.7)
    assert meta["aps_display_enhanced"] is True
    assert meta["aps_display_label"] == "Display enhanced for visibility."
    assert float(np.mean(enhanced)) > float(np.mean(rgb)) + 20.0
    stats_after = compute_aps_stats(raw)
    assert stats_after == stats_before
    raw_out, raw_meta = enhance_aps_display(rgb, mode="raw")
    assert raw_meta["aps_display_enhanced"] is False
    assert np.array_equal(raw_out, rgb)


def test_stats_shape_and_channel_fields():
    stats = compute_aps_stats(good_frame())
    for key in ("aps_min", "aps_p1", "aps_p50", "aps_p95", "aps_p99", "aps_max",
                "aps_mean", "aps_std", "aps_dtype", "aps_shape", "aps_channel_count",
                "aps_channel_means", "aps_channel_stds"):
        assert key in stats
    assert stats["aps_dtype"] == "uint8"
    assert stats["aps_shape"] == [H, W]
    assert stats["aps_channel_count"] == 1


def test_low_aps_visibility_downgrades_benchmark_card():
    from grasp_project.benchmarking.evaluators import evaluate_motion
    from grasp_project.benchmarking.masked_feature_view import MaskedFeatureView, build_feature_packet

    packet = build_feature_packet(
        realsense={},
        davis={
            "aps_state": "APS_GRAYSCALE",
            "aps_fps": 20.0,
            "aps_visibility_status": "LOW_LIGHT",
            "aps_visibility_score": 12.5,
            "aps_underexposed": True,
            "aps_display_warning": "DAVIS APS is underexposed.",
        },
        pointcloud={}, sync={}, quality={}, live_quality={}, embodiment={},
    )
    view = MaskedFeatureView(packet, enabled_features=["davis_aps_frame"])
    card = evaluate_motion(view)
    assert card["davis_aps_visibility_status"] == "LOW_LIGHT"
    assert card["davis_aps_visual_detection_valid"] is False
    assert "context only" in (card["davis_aps_limitation"] or "")

    good_packet = build_feature_packet(
        realsense={},
        davis={"aps_state": "APS_GRAYSCALE", "aps_fps": 20.0, "aps_visibility_status": "GOOD", "aps_visibility_score": 78.0},
        pointcloud={}, sync={}, quality={}, live_quality={}, embodiment={},
    )
    good_card = evaluate_motion(MaskedFeatureView(good_packet, enabled_features=["davis_aps_frame"]))
    assert good_card["davis_aps_visual_detection_valid"] is True
    assert good_card["davis_aps_limitation"] is None

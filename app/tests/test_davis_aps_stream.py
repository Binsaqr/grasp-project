"""Camera-free tests for DAVIS APS frame classification and features."""

import time

import numpy as np
import pytest

from grasp_project.davis.aps_stream import (
    APS_DISPLAY_MODES,
    aps_backend_metadata,
    aps_display_image,
    aps_features,
    classify_aps_array,
    demosaic_bayer,
    latest_aps_array,
    normalize_aps_frame,
    normalize_backend_aps_frame,
)


def test_grayscale_aps_is_not_called_rgb():
    frame = np.arange(64, dtype=np.uint8).reshape(8, 8)
    status = classify_aps_array(frame)
    assert status["aps_mode"] == "DAVIS APS Grayscale"
    assert status["aps_color_verified"] is False
    assert status["channel_count"] == 1


def test_raw_bayer_aps_uses_conservative_label():
    frame = np.arange(64, dtype=np.uint8).reshape(8, 8)
    status = classify_aps_array(frame, metadata={"color_sensor": True}, config={"bayer_pattern": "rggb"})
    assert status["aps_mode"] == "DAVIS APS Raw Bayer"
    assert status["aps_bayer_pattern"] == "rggb"
    rgb = demosaic_bayer(frame, "rggb")
    assert rgb.shape == (8, 8, 3)


@pytest.mark.parametrize("pattern", ["rggb", "bggr", "grbg", "gbrg"])
def test_bayer_patterns_preserve_rgb_channel_values(pattern):
    values = {"r": 240, "g": 120, "b": 20}
    frame = np.zeros((12, 12), dtype=np.uint8)
    for y, x, channel in (
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ):
        frame[y::2, x::2] = values[channel]

    rgb = demosaic_bayer(frame, pattern)

    assert np.median(rgb[2:-2, 2:-2], axis=(0, 1)) == pytest.approx([240, 120, 20], abs=2)


def test_independent_three_channel_aps_is_color_but_order_unverified():
    red = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    green = np.flipud(red)
    blue = np.full((8, 8), 120, dtype=np.uint8)
    frame = np.dstack([red, green, blue])
    status = classify_aps_array(frame, config={"channel_order": "auto"})
    assert status["aps_mode"] == "DAVIS APS Color"
    assert status["aps_color_verified"] is False
    assert status["aps_channel_order"] == "unverified_color_order"
    assert status["aps_channel_order_source"] == "not_available"


def test_pyaer_rgb_contract_verifies_encoding_not_color_accuracy():
    frame = np.full((8, 8, 3), 32, dtype=np.uint8)
    metadata = aps_backend_metadata("pyaer", frame)

    status = classify_aps_array(frame, metadata=metadata)

    assert status["aps_color_verified"] is True
    assert status["aps_channel_order"] == "rgb"
    assert status["aps_channel_order_source"] == "pyaer_rgb_array_contract"


def test_backend_normalization_keeps_device_and_host_time_separate():
    before = time.time()
    record = normalize_backend_aps_frame(
        np.full((8, 8, 3), 32, dtype=np.uint8),
        backend="pyaer",
        frame_id=9,
        device_timestamp_us=1_025_000,
    )

    assert record.device_timestamp == 1_025_000
    assert record.device_timestamp_unit == "microseconds"
    assert before <= record.host_timestamp <= time.time()


def test_configured_bgr_order_is_converted_for_rgb_display():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 200
    status = classify_aps_array(frame, config={"channel_order": "bgr"})

    display = aps_display_image(frame, channel_order=status["aps_channel_order"])

    assert status["aps_color_verified"] is True
    assert status["aps_channel_order_source"] == "configured"
    assert display[0, 0].tolist() == [200, 20, 10]


def test_identical_three_channel_aps_is_unknown_not_color():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    frame = np.dstack([gray, gray, gray])
    status = classify_aps_array(frame)
    assert status["aps_mode"] == "DAVIS APS Unknown"
    assert status["aps_color_verified"] is False


def test_latest_aps_array_distinguishes_mono_stack_from_color_frame():
    mono_stack = np.stack([
        np.full((260, 346), 17, dtype=np.uint8),
        np.full((260, 346), 29, dtype=np.uint8),
    ])
    color = np.full((260, 346, 3), 41, dtype=np.uint8)

    latest_mono = latest_aps_array(mono_stack)

    assert latest_mono.shape == (260, 346)
    assert np.all(latest_mono == 29)
    assert latest_aps_array(color) is color


def test_removed_bayer_debug_mode_is_not_advertised():
    assert "bayer_debug" not in APS_DISPLAY_MODES


def test_normalize_none_keeps_channel_source_explicitly_unavailable():
    record = normalize_aps_frame(None)

    assert record.aps_frame_valid is False
    assert record.aps_channel_order_source is None


def test_aps_features_include_motion_between_frames():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = a.copy()
    b[:, 5:] = 200
    features = aps_features(b, previous_frame=a)
    assert features["aps_feature_valid"] is True
    assert features["brightness"] > 0
    assert features["frame_to_frame_motion"] > 0
    record = normalize_aps_frame(b, frame_id=3)
    assert record.frame_id == 3
    assert record.aps_frame_valid is True

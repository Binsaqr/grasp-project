"""Camera-free regression tests for the live DAVIS APS adapter."""

import sys
import time
import types

import cv2
import numpy as np
import pytest

from grasp_project.davis import event_stream, motion_gui
from grasp_project.davis.event_stream import EventPostGraspMonitor


class _FakeConfigDevice:
    def __init__(self):
        self.values = {(10, 11): 6000, (10, 12): 1}
        self.writes = []

    def get_config(self, module, key):
        return self.values[(module, key)]

    def set_config(self, module, key, value):
        self.writes.append((module, key, value))
        self.values[(module, key)] = value
        return True


def test_pyaer_exposure_adapter_uses_exported_libcaer_and_applies_manual_config(monkeypatch):
    libcaer = types.SimpleNamespace(
        DAVIS_CONFIG_APS=10,
        DAVIS_CONFIG_APS_EXPOSURE=11,
        DAVIS_CONFIG_APS_AUTOEXPOSURE=12,
    )
    pyaer = types.ModuleType("pyaer")
    pyaer.libcaer = libcaer
    monkeypatch.setitem(sys.modules, "pyaer", pyaer)
    monkeypatch.delitem(sys.modules, "pyaer.libcaer", raising=False)
    device = _FakeConfigDevice()
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.monitor = types.SimpleNamespace(device=device, backend_name="pyaer")
    analyzer.aps_config = {"exposure_auto": False, "exposure_us": 12_000}

    analyzer._refresh_aps_exposure_info()

    assert device.writes == [(10, 12, 0), (10, 11, 12_000)]
    assert analyzer.aps_exposure_info["aps_exposure_control_available"] is True
    assert analyzer.aps_exposure_info["aps_exposure_configuration_applied"] is True
    assert analyzer.aps_exposure_info["aps_exposure_auto"] is False
    assert analyzer.aps_exposure_info["aps_exposure_us"] == 12_000
    assert analyzer.aps_exposure_info["aps_exposure_source"] == "pyaer_libcaer_config_readback"
    assert analyzer.aps_exposure_info["aps_gain_status"] == "NOT_APPLICABLE_BACKEND_API_UNAVAILABLE"


def test_pyaer_exposure_adapter_enables_configured_auto_exposure(monkeypatch):
    libcaer = types.SimpleNamespace(
        DAVIS_CONFIG_APS=10,
        DAVIS_CONFIG_APS_EXPOSURE=11,
        DAVIS_CONFIG_APS_AUTOEXPOSURE=12,
    )
    pyaer = types.ModuleType("pyaer")
    pyaer.libcaer = libcaer
    monkeypatch.setitem(sys.modules, "pyaer", pyaer)
    device = _FakeConfigDevice()
    device.values[(10, 12)] = 0
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.monitor = types.SimpleNamespace(device=device, backend_name="pyaer")
    analyzer.aps_config = {"exposure_auto": True, "exposure_us": None}

    analyzer._refresh_aps_exposure_info()

    assert device.writes == [(10, 12, 1)]
    assert analyzer.aps_exposure_info["aps_exposure_auto"] is True
    assert analyzer.aps_exposure_info["aps_exposure_configuration_applied"] is True


def test_motion_analyzer_loads_live_bayer_and_channel_configuration(monkeypatch):
    class _FakeMonitor:
        def __init__(self, **_kwargs):
            self.owner_info = {}

        def start(self):
            return {}

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", _FakeMonitor)
    monkeypatch.setattr(
        motion_gui,
        "load_named_config",
        lambda *_args, **_kwargs: {
            "aps": {
                "bayer_pattern": "rggb",
                "channel_order": "bgr",
                "display_enhancement": "raw",
            }
        },
    )

    analyzer = motion_gui.DavisMotionAnalyzer()

    assert analyzer.aps_config["bayer_pattern"] == "rggb"
    assert analyzer.aps_config["channel_order"] == "bgr"
    assert analyzer.aps_config["display_mode"] == "raw"
    assert analyzer.set_aps_display_mode("bayer_debug") is False


def test_motion_analyzer_reuses_existing_aps_detector_to_publish_target_context(monkeypatch):
    import davis_grasp_planner

    monkeypatch.setattr(
        davis_grasp_planner,
        "detect_bottle_from_aps",
        lambda *_args, **_kwargs: (
            {
                "davis_bottle_detected": True,
                "davis_detection_status": "DETECTED",
                "davis_bbox_xyxy": [60, 20, 150, 240],
                "davis_detection_confidence": 88.0,
                "aps_detection_source": "aps_contour_shape_score",
                "davis_detection_backend": "opencv_aps",
                "detection_reason": "test APS bottle",
            },
            {},
        ),
    )
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.latest_aps_display_rgb = np.zeros((260, 346, 3), dtype=np.uint8)
    analyzer.latest_aps_record = types.SimpleNamespace(
        aps_mode="MONO8",
        host_timestamp=123.0,
    )
    analyzer.aps_frame_id = 7
    analyzer.previous_aps_target_bbox = None
    analyzer.latest_aps_target_detection = None
    analyzer.latest_aps_target_context = None
    analyzer.latest_aps_target_failure_reason = "not_evaluated"
    analyzer.aps_config = {"target_confirmation_frames": 3}

    analyzer._update_aps_target_detection()
    assert analyzer.latest_aps_target_context is None
    assert analyzer.latest_aps_target_failure_reason.endswith("1_of_3")
    analyzer._update_aps_target_detection()
    assert analyzer.latest_aps_target_context is None
    analyzer._update_aps_target_detection()

    assert analyzer.latest_aps_target_context["roi_source"] == "aps_target"
    assert analyzer.latest_aps_target_context["roi_priority_selected"] == "davis_aps_target_roi"
    assert analyzer.latest_aps_target_context["source_frame_id"] == 7
    assert analyzer.latest_aps_target_context["target_confidence"] == 0.88
    assert analyzer.previous_aps_target_bbox == [60, 20, 150, 240]
    assert analyzer.latest_aps_target_failure_reason is None


def test_opencv_aps_detector_accepts_horizontal_bottle_and_exports_long_axis():
    from davis_grasp_planner import detect_bottle_from_aps

    gray = np.full((260, 346), 25, dtype=np.uint8)
    cv2.ellipse(gray, (170, 130), (105, 28), 0, 0, 360, 125, -1)
    cv2.line(gray, (80, 130), (260, 130), 80, 3)

    detection, _debug = detect_bottle_from_aps(gray)

    assert detection["davis_bottle_detected"] is True
    assert detection["davis_contour_available"] is True
    assert abs(detection["davis_axis_vector_px"][0]) > 0.9
    assert detection["selected_candidate"]["oriented_aspect_ratio_long_over_short"] > 3.0
    assert "candidate_not_vertical" not in detection["rejection_reasons"]


def test_raw_aps_saturated_textureless_glare_cannot_become_detected_target():
    from davis_grasp_planner import detect_bottle_from_aps

    gray = np.full((260, 346), 25, dtype=np.uint8)
    cv2.rectangle(gray, (150, 35), (205, 230), 255, -1)

    detection, _debug = detect_bottle_from_aps(gray)

    assert detection["davis_bottle_detected"] is False
    assert detection["davis_detection_status"] != "DETECTED"
    assert any(
        candidate.get("glare_like_low_texture")
        for candidate in detection["davis_detection_candidates"]
    )


def test_detector_glare_evidence_uses_separate_raw_aps_not_display_input():
    from davis_grasp_planner import detect_bottle_from_aps

    display_gray = np.full((260, 346), 25, dtype=np.uint8)
    cv2.ellipse(display_gray, (170, 130), (105, 28), 0, 0, 360, 125, -1)
    cv2.line(display_gray, (80, 130), (260, 130), 80, 3)
    raw_gray = np.full_like(display_gray, 25)
    cv2.ellipse(raw_gray, (170, 130), (105, 28), 0, 0, 360, 255, -1)

    detection, _debug = detect_bottle_from_aps(display_gray, raw_gray=raw_gray)

    assert detection["aps_raw_evidence_source"] == "raw_aps_frame"
    assert detection["davis_bottle_detected"] is False
    assert any(
        candidate.get("glare_like_low_texture")
        for candidate in detection["davis_detection_candidates"]
    )


def test_motion_analyzer_passes_backend_raw_aps_frame_to_detector(monkeypatch):
    import davis_grasp_planner

    captured = {}

    def fake_detector(*_args, **kwargs):
        captured.update(kwargs)
        return ({"davis_bottle_detected": False, "detection_reason": "test"}, {})

    monkeypatch.setattr(davis_grasp_planner, "detect_bottle_from_aps", fake_detector)
    raw = np.arange(260 * 346, dtype=np.uint16).reshape(260, 346)
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.latest_aps_display_rgb = np.full((260, 346, 3), 127, dtype=np.uint8)
    analyzer.latest_aps_record = types.SimpleNamespace(
        aps_mode="MONO16",
        host_timestamp=123.0,
        image=raw,
    )
    analyzer.aps_frame_id = 1
    analyzer.previous_aps_target_bbox = None
    analyzer.latest_aps_target_detection = None
    analyzer.latest_aps_target_context = None
    analyzer.latest_aps_target_failure_reason = "not_evaluated"
    analyzer.aps_config = {}

    analyzer._update_aps_target_detection()

    assert captured["raw_gray"] is raw


@pytest.mark.parametrize("center_x", [85, 260])
def test_plausible_off_center_bottle_is_not_hard_capped_below_detected(center_x):
    from davis_grasp_planner import detect_bottle_from_aps

    gray = np.full((260, 346), 25, dtype=np.uint8)
    cv2.ellipse(gray, (center_x, 130), (65, 24), 0, 0, 360, 125, -1)
    cv2.line(gray, (center_x - 45, 130), (center_x + 45, 130), 80, 3)

    detection, _debug = detect_bottle_from_aps(gray)

    assert detection["davis_bottle_detected"] is True
    assert detection["davis_contour_available"] is True
    assert detection["davis_detection_confidence"] >= 70.0


@pytest.mark.parametrize("adc_max,expected_bits", [(1023, 10), (4095, 12)])
def test_uint16_raw_aps_glare_uses_inferred_adc_code_range(adc_max, expected_bits):
    from davis_grasp_planner import detect_bottle_from_aps

    display_gray = np.full((260, 346), 25, dtype=np.uint8)
    cv2.ellipse(display_gray, (170, 130), (105, 28), 0, 0, 360, 125, -1)
    cv2.line(display_gray, (80, 130), (260, 130), 80, 3)
    raw_gray = np.full((260, 346), adc_max // 10, dtype=np.uint16)
    cv2.ellipse(raw_gray, (170, 130), (105, 28), 0, 0, 360, adc_max, -1)

    detection, _debug = detect_bottle_from_aps(display_gray, raw_gray=raw_gray)

    assert detection["aps_raw_normalization_source"] == "inferred_common_adc_range"
    assert detection["aps_raw_normalization_bit_depth"] == expected_bits
    assert detection["aps_raw_normalization_code_max"] == adc_max
    assert detection["davis_bottle_detected"] is False
    assert any(
        candidate.get("glare_like_low_texture")
        for candidate in detection["davis_detection_candidates"]
    )


def test_confirmed_aps_target_requires_hysteresis_before_nonoverlap_switch(monkeypatch):
    import davis_grasp_planner

    def detection(bbox):
        return {
            "davis_bottle_detected": True,
            "davis_detection_status": "DETECTED",
            "davis_bbox_xyxy": bbox,
            "davis_detection_confidence": 88.0,
            "aps_detection_source": "aps_contour_shape_score",
            "davis_detection_backend": "opencv_aps",
            "detection_reason": "synthetic target",
        }

    results = iter([detection([40, 30, 130, 220])] * 3 + [detection([210, 35, 300, 225])] * 3)
    monkeypatch.setattr(
        davis_grasp_planner,
        "detect_bottle_from_aps",
        lambda *_args, **_kwargs: (next(results), {}),
    )
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.latest_aps_display_rgb = np.zeros((260, 346, 3), dtype=np.uint8)
    analyzer.latest_aps_record = types.SimpleNamespace(aps_mode="MONO8", host_timestamp=123.0)
    analyzer.aps_frame_id = 1
    analyzer.previous_aps_target_bbox = None
    analyzer.latest_aps_target_detection = None
    analyzer.latest_aps_target_context = None
    analyzer.latest_aps_target_failure_reason = "not_evaluated"
    analyzer.aps_config = {
        "target_confirmation_frames": 3,
        "target_switch_confirmation_frames": 3,
        "target_same_iou_min": 0.2,
        "target_pending_iou_min": 0.5,
    }

    for _ in range(3):
        analyzer._update_aps_target_detection()
    first_roi = list(analyzer.latest_aps_target_context["roi_xywh"])
    analyzer._update_aps_target_detection()
    assert analyzer.latest_aps_target_context["roi_xywh"] == first_roi
    assert "switch_pending_confirmation_1_of_3" in analyzer.latest_aps_target_failure_reason
    analyzer._update_aps_target_detection()
    assert analyzer.latest_aps_target_context["roi_xywh"] == first_roi
    analyzer._update_aps_target_detection()
    assert analyzer.latest_aps_target_context["roi_xywh"] != first_roi


def test_confirmed_target_expires_after_misses_and_same_bbox_must_reconfirm(monkeypatch):
    import davis_grasp_planner

    detected = {
        "davis_bottle_detected": True,
        "davis_detection_status": "DETECTED",
        "davis_bbox_xyxy": [40, 30, 130, 220],
        "davis_detection_confidence": 88.0,
        "detection_reason": "synthetic target",
    }
    missed = {
        "davis_bottle_detected": False,
        "davis_detection_status": "NOT_DETECTED",
        "davis_bbox_xyxy": None,
        "davis_detection_confidence": 0.0,
        "detection_reason": "synthetic miss",
    }
    results = iter([detected] * 3 + [missed] * 2 + [detected] * 3)
    monkeypatch.setattr(
        davis_grasp_planner,
        "detect_bottle_from_aps",
        lambda *_args, **_kwargs: (next(results), {}),
    )
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.latest_aps_display_rgb = np.zeros((260, 346, 3), dtype=np.uint8)
    analyzer.latest_aps_record = types.SimpleNamespace(
        aps_mode="MONO8",
        host_timestamp=123.0,
        image=np.zeros((260, 346), dtype=np.uint8),
    )
    analyzer.aps_frame_id = 1
    analyzer.previous_aps_target_bbox = None
    analyzer.latest_aps_target_detection = None
    analyzer.latest_aps_target_context = None
    analyzer.latest_aps_target_failure_reason = "not_evaluated"
    analyzer.aps_config = {
        "target_confirmation_frames": 3,
        "target_switch_confirmation_frames": 3,
        "target_max_missed_detections": 2,
        "target_confirmed_max_age_sec": 0.75,
    }

    for _ in range(3):
        analyzer._update_aps_target_detection()
    assert analyzer.confirmed_aps_target_context is not None
    analyzer._update_aps_target_detection()
    assert analyzer.confirmed_aps_target_context is not None
    analyzer._update_aps_target_detection()
    assert analyzer.confirmed_aps_target_context is None
    assert analyzer.latest_aps_target_context is None
    assert analyzer.previous_aps_target_bbox is None
    analyzer._update_aps_target_detection()
    assert analyzer.confirmed_aps_target_context is None
    assert analyzer.pending_aps_target_count == 1
    analyzer._update_aps_target_detection()
    assert analyzer.confirmed_aps_target_context is None
    analyzer._update_aps_target_detection()
    assert analyzer.confirmed_aps_target_context is not None


def test_stale_confirmed_target_expires_before_same_bbox_candidate(monkeypatch):
    import davis_grasp_planner

    detected = {
        "davis_bottle_detected": True,
        "davis_detection_status": "DETECTED",
        "davis_bbox_xyxy": [40, 30, 130, 220],
        "davis_detection_confidence": 88.0,
        "detection_reason": "synthetic target",
    }
    monkeypatch.setattr(
        davis_grasp_planner,
        "detect_bottle_from_aps",
        lambda *_args, **_kwargs: (detected, {}),
    )
    stale_context = motion_gui.aps_target_context_from_detection(
        detected,
        frame_id=1,
        captured_monotonic=time.monotonic() - 2.0,
    )
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.latest_aps_display_rgb = np.zeros((260, 346, 3), dtype=np.uint8)
    analyzer.latest_aps_record = types.SimpleNamespace(
        aps_mode="MONO8",
        host_timestamp=123.0,
        image=np.zeros((260, 346), dtype=np.uint8),
    )
    analyzer.aps_frame_id = 2
    analyzer.previous_aps_target_bbox = list(detected["davis_bbox_xyxy"])
    analyzer.confirmed_aps_target_bbox = list(detected["davis_bbox_xyxy"])
    analyzer.confirmed_aps_target_context = stale_context
    analyzer.latest_aps_target_context = stale_context
    analyzer.latest_aps_target_detection = detected
    analyzer.latest_aps_target_failure_reason = None
    analyzer.aps_config = {
        "target_confirmation_frames": 3,
        "target_confirmed_max_age_sec": 0.75,
    }

    analyzer._update_aps_target_detection()

    assert analyzer.confirmed_aps_target_context is None
    assert analyzer.latest_aps_target_context is None
    assert analyzer.pending_aps_target_count == 1
    assert analyzer.latest_aps_target_failure_reason.endswith("1_of_3")


def test_confirmation_age_tracks_detector_cadence_without_masking_stale_gap(monkeypatch):
    import davis_grasp_planner

    detected = {
        "davis_bottle_detected": True,
        "davis_detection_status": "DETECTED",
        "davis_bbox_xyxy": [40, 30, 130, 220],
        "davis_detection_confidence": 92.0,
        "detection_reason": "continuous synthetic target",
    }
    monkeypatch.setattr(
        davis_grasp_planner,
        "detect_bottle_from_aps",
        lambda *_args, **_kwargs: (detected, {}),
    )
    clock = [100.0]
    monkeypatch.setattr(motion_gui.time, "monotonic", lambda: clock[0])
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.latest_aps_display_rgb = np.zeros((260, 346, 3), dtype=np.uint8)
    analyzer.latest_aps_record = types.SimpleNamespace(
        aps_mode="MONO8",
        host_timestamp=123.0,
        image=np.zeros((260, 346), dtype=np.uint8),
    )
    analyzer.aps_frame_id = 1
    analyzer.previous_aps_target_bbox = None
    analyzer.latest_aps_target_detection = None
    analyzer.latest_aps_target_context = None
    analyzer.latest_aps_target_failure_reason = "not_evaluated"
    analyzer.aps_config = {
        "target_confirmation_frames": 3,
        "target_confirmed_max_age_sec": 0.75,
        "target_cadence_jitter_multiplier": 1.5,
        "target_confirmed_age_cap_sec": 3.0,
    }
    detector_cadence = 1.0 / 6.7

    for _ in range(8):
        analyzer.aps_times = [
            clock[0] - index / 6.7
            for index in range(19, -1, -1)
        ]
        analyzer._update_aps_target_detection()
        clock[0] += detector_cadence

    assert analyzer.confirmed_aps_target_context is not None
    assert analyzer.aps_target_confirmation_expired_reason is None
    assert analyzer._aps_target_detector_cadence_sec() == pytest.approx(detector_cadence)
    assert analyzer._aps_target_max_age_sec() == pytest.approx(0.75)
    assert (
        analyzer.latest_aps_target_context["aps_target_context_max_age_sec"]
        > detector_cadence
    )

    clock[0] += analyzer._aps_target_max_age_sec() + 0.01
    analyzer._update_aps_target_detection()

    assert analyzer.confirmed_aps_target_context is None
    assert analyzer.pending_aps_target_count == 1
    assert analyzer.latest_aps_target_failure_reason.endswith("1_of_3")


def test_confirmed_contour_context_exposes_estimated_nonphysical_midbody_band():
    from davis_grasp_planner import detect_bottle_from_aps

    gray = np.full((260, 346), 25, dtype=np.uint8)
    cv2.ellipse(gray, (170, 130), (105, 28), 0, 0, 360, 125, -1)
    cv2.line(gray, (80, 130), (260, 130), 80, 3)
    detection, _debug = detect_bottle_from_aps(gray)

    context = motion_gui.aps_target_context_from_detection(
        detection,
        frame_id=9,
        captured_monotonic=10.0,
    )

    assert context["boundary_source"] == "davis_aps_target_contour"
    assert len(context["boundary_points"]) >= 3
    assert len(context["contact_band_points"]) == 4
    assert context["contact_band_source"] == "aps_native_estimated_midbody_band"
    assert context["contact_band_physically_validated"] is False
    assert any("not a measured gripper contact" in item for item in context["mapping_limitations"])


def test_post_grasp_window_latches_only_current_confirmed_aps_context():
    analyzer = object.__new__(motion_gui.DavisMotionAnalyzer)
    analyzer.stage = "PRE_CLOSE"
    analyzer.roi_xywh = [40, 30, 100, 160]
    analyzer.close_action_origin = None
    analyzer.close_action_id = None
    analyzer.close_action_started_at = None
    analyzer._reset_recent_motion = lambda **_kwargs: None
    current_context = {
        "roi_source": "aps_target",
        "roi_xywh": [40, 30, 100, 160],
        "captured_monotonic": time.monotonic(),
    }
    analyzer.confirmed_aps_target_context = current_context
    analyzer.latest_aps_target_context = current_context

    assert analyzer.set_stage("CLOSE_IMPACT") is True

    assert analyzer.latest_aps_target_context["aps_post_grasp_latched"] is True
    assert analyzer.latest_aps_target_context["aps_post_grasp_latch_duration_sec"] == 120.0
    assert analyzer.post_grasp_aps_target_latch_until - analyzer.close_action_origin == pytest.approx(120.0)


def test_pyaer_frame_timestamp_tracks_the_latest_frame():
    frames = np.stack([
        np.full((6, 8), 17, dtype=np.uint8),
        np.full((6, 8), 29, dtype=np.uint8),
    ])
    data = (
        None,
        0,
        None,
        0,
        np.asarray([1_000_000, 1_025_000], dtype=np.uint64),
        frames,
        None,
        0,
    )
    monitor = object.__new__(EventPostGraspMonitor)
    monitor.active = True
    monitor.device = types.SimpleNamespace(get_event=lambda _mode: data)
    monitor.unavailable_reason = None
    monitor.event_time_zero_us = None
    monitor.event_time_wall_zero = None
    monitor.start_time = time.monotonic()
    monitor.latest_aps_device_timestamp_us = None

    events, returned_frames = monitor.poll_event_and_frame_arrays()

    assert events.shape == (0, 4)
    assert np.array_equal(returned_frames, frames)
    assert monitor.latest_aps_device_timestamp_us == 1_025_000


def _event_monitor_for_data(data):
    monitor = object.__new__(EventPostGraspMonitor)
    monitor.active = True
    monitor.device = types.SimpleNamespace(get_event=lambda _mode: data)
    monitor.unavailable_reason = None
    monitor.event_time_zero_us = None
    monitor.event_time_wall_zero = None
    monitor.event_device_last_us = None
    monitor.event_normalized_last = None
    monitor.event_timestamp_reanchor_count = 0
    monitor.event_timestamp_basis = "device_microseconds_host_anchored"
    monitor.start_time = 10.0
    monitor.latest_aps_device_timestamp_us = None
    return monitor


def test_pyaer_event_timestamps_anchor_to_first_poll_not_preopen_start(monkeypatch):
    events = np.asarray(
        [
            [1_000_000, 10, 20, 1],
            [1_050_000, 11, 21, 0],
        ],
        dtype=np.float64,
    )
    data = (events, 0, None, 0, None, None, None, 0)
    monitor = _event_monitor_for_data(data)
    monkeypatch.setattr(event_stream.time, "monotonic", lambda: 100.0)

    normalized, _frames = monitor.poll_event_and_frame_arrays()

    assert np.allclose(normalized[:, 0], [99.95, 100.0])
    assert normalized[-1, 0] != monitor.start_time + 0.05
    assert monitor.event_normalized_last == 100.0
    normalized_with_future = np.vstack(
        [normalized, np.asarray([[100.01, 12, 22, 1]], dtype=np.float64)]
    )
    assert motion_gui._events_since(
        normalized_with_future,
        now=100.0,
        seconds=0.1,
        roi_xywh=[0, 0, 346, 260],
    ).shape[0] == 2


def test_pyaer_event_timestamp_clock_reset_reanchors_to_host(monkeypatch):
    packets = iter(
        [
            (
                np.asarray(
                    [[1_000_000, 10, 20, 1], [1_050_000, 11, 21, 0]],
                    dtype=np.float64,
                ),
                0,
                None,
                0,
                None,
                None,
                None,
                0,
            ),
            (
                np.asarray(
                    [[1_000, 12, 22, 1], [51_000, 13, 23, 0]],
                    dtype=np.float64,
                ),
                0,
                None,
                0,
                None,
                None,
                None,
                0,
            ),
        ]
    )
    poll_times = iter([100.0, 101.0])
    monitor = _event_monitor_for_data(None)
    monitor.device = types.SimpleNamespace(get_event=lambda _mode: next(packets))
    monkeypatch.setattr(event_stream.time, "monotonic", lambda: next(poll_times))

    first, _frames = monitor.poll_event_and_frame_arrays()
    second, _frames = monitor.poll_event_and_frame_arrays()

    assert first[-1, 0] == 100.0
    assert second[-1, 0] == 101.0
    assert np.all(np.diff(second[:, 0]) >= 0.0)
    assert monitor.event_timestamp_reanchor_count == 1

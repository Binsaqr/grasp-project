"""Camera-free tests for DAVIS event ring and raw event features."""

import numpy as np

from grasp_project.davis.camera import DavisEventRingBuffer
from grasp_project.davis.motion_features import calibrate_event_baseline, event_features_for_window
from grasp_project.davis.motion_gui import cap_event_packet_preserving_timestamps


def _events(count=20, start=1.0, step=0.001):
    t = start + np.arange(count, dtype=float) * step
    x = np.linspace(10, 120, count)
    y = np.linspace(20, 180, count)
    p = np.arange(count) % 2
    return np.column_stack([t, x, y, p])


def test_event_ring_is_bounded_by_count():
    ring = DavisEventRingBuffer(max_duration_sec=10.0, max_event_count=10)
    batch = ring.add(_events(25))
    assert batch.event_count == 25
    assert ring.event_count == 10
    assert ring.dropped_event_count == 15
    assert ring.monotonic_timestamps is True


def test_event_ring_recent_and_between_queries():
    ring = DavisEventRingBuffer(max_duration_sec=10.0, max_event_count=100)
    ring.add(_events(30, start=5.0, step=0.01))
    recent = ring.get_recent_events(50.0)
    assert 1 <= recent.shape[0] <= 6
    between = ring.get_events_between(5.05, 5.10)
    assert between.shape[0] > 0
    assert ring.get_event_count(50.0) == recent.shape[0]


def test_event_features_report_values_and_baseline_normalization():
    events = _events(40)
    baseline = calibrate_event_baseline(events[:10], duration_sec=0.1)
    features = event_features_for_window(events, window_ms=100.0, baseline=baseline)
    assert features["valid"] is True
    assert features["event_count"]["value"] == 40.0
    assert features["event_rate"]["value"] > 0
    assert features["event_rate"]["baseline_normalized"] is not None
    assert features["motion_bbox"] is not None
    assert features["disturbance_type"] in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "OBJECT_MOTION", "UNKNOWN_MOTION"}


def test_empty_event_features_are_honest():
    features = event_features_for_window(np.empty((0, 4)), window_ms=100)
    assert features["valid"] is False
    assert features["invalid_reason"] == "no_events"
    assert features["event_motion_class"] == "STREAM_ACTIVE_NO_MOTION"


def test_live_tick_cap_preserves_normalized_event_timestamps():
    events = _events(12, start=7.0, step=0.003)
    events[:, 0] += np.asarray([0.0, 0.001, 0.0, 0.002] * 3)

    capped, dropped = cap_event_packet_preserving_timestamps(events, max_events=5)

    assert dropped == 7
    np.testing.assert_array_equal(capped[:, 0], events[-5:, 0])
    assert not np.shares_memory(capped, events)

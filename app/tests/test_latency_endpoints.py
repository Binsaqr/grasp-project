"""Deterministic software endpoint tests; no physical latency validation."""

import json

import pytest

from grasp_project.common.latency import (
    ClockStamp,
    SoftwareLatencyTracker,
    endpoint_interval,
)


def interval(start, end, start_clock="host_monotonic", end_clock="host_monotonic"):
    return endpoint_interval(
        ClockStamp(start, start_clock), ClockStamp(end, end_clock),
        start_name="observed", end_name="available",
    )


def test_same_clock_interval_preserves_explicit_endpoints_and_units():
    result = interval(30.0, 30.125)
    assert result == {
        "start_endpoint": "observed", "end_endpoint": "available",
        "clock_domain": "host_monotonic", "status": "MEASURED",
        "start_timestamp_sec": 30.0, "end_timestamp_sec": 30.125,
        "duration_ms": 125.0, "reason": None,
    }


@pytest.mark.parametrize("start,end,reason", [
    (None, 1.0, "endpoint_not_observed"),
    (1.0, None, "endpoint_not_observed"),
    (float("nan"), 1.0, "nonfinite_timestamp"),
    (1.0, float("inf"), "nonfinite_timestamp"),
    (float("-inf"), 1.0, "nonfinite_timestamp"),
    ("not a timestamp", 1.0, "invalid_timestamp"),
    (True, 1.0, "invalid_timestamp"),
    (100.0, 99.0, "negative_interval_clock_reset_or_rollover"),
    (2**32 - 1, 0, "negative_interval_clock_reset_or_rollover"),
    (0, 1e308, "nonfinite_interval"),
])
def test_missing_corrupt_or_reset_endpoint_is_not_a_measured_delay(start, end, reason):
    result = interval(start, end)
    assert result["duration_ms"] is None
    assert result["reason"] == reason
    assert result["status"] in {"UNAVAILABLE", "INVALID"}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("start_clock,end_clock,reason", [
    ("device_microseconds", "host_monotonic", "cross_clock_interval_not_allowed"),
    ("host_epoch", "host_monotonic", "cross_clock_interval_not_allowed"),
    ("", "host_monotonic", "clock_domain_required"),
])
def test_unmapped_clock_domains_cannot_be_subtracted(start_clock, end_clock, reason):
    result = interval(1.0, 100.0, start_clock, end_clock)
    assert result["status"] == "INVALID"
    assert result["duration_ms"] is None
    assert result["reason"] == reason


def test_equal_timestamps_are_a_valid_zero_interval():
    assert interval(1.0, 1.0)["duration_ms"] == 0.0


def clock_from(*values):
    values = iter(values)
    return lambda: next(values)


def test_injected_clock_records_actual_processing_completion_and_first_label_once():
    tracker = SoftwareLatencyTracker(clock_from(10.0, 11.0, 11.05, 11.06, 12.0, 12.1, 12.2))
    tracker.begin_update()
    tracker.observe_event_receipt(10.99)
    tracker.inference_complete(accepted_slip=True)
    report = tracker.report_payload_ready()
    intervals = report["intervals"]
    assert intervals["analysis_epoch_to_first_accepted_slip_label"]["duration_ms"] == 1050
    assert intervals["update_to_inference_complete"]["duration_ms"] == 50
    assert intervals["last_packet_receipt_to_inference_complete"]["duration_ms"] == 60
    assert intervals["last_packet_receipt_to_report_payload_ready"]["duration_ms"] == 70
    assert intervals["first_accepted_slip_label_to_first_report_payload_ready"]["duration_ms"] == 10
    assert report["physical_onset_to_alert_ms"] is None
    assert report["browser_display_latency_ms"] is None
    assert report["report_ready_scope"] == "in_memory_payload_before_file_serialization_or_transport"
    tracker.begin_update()
    tracker.inference_complete(accepted_slip=True)
    next_report = tracker.report_payload_ready()
    assert next_report["intervals"]["analysis_epoch_to_first_accepted_slip_label"] == intervals["analysis_epoch_to_first_accepted_slip_label"]
    assert next_report["intervals"]["first_accepted_slip_label_to_first_report_payload_ready"] == intervals["first_accepted_slip_label_to_first_report_payload_ready"]
    assert next_report["intervals"]["last_packet_receipt_to_inference_complete"]["reason"] == "no_event_observation_in_update"


def test_no_observation_is_unavailable_even_when_analysis_runs():
    tracker = SoftwareLatencyTracker(clock_from(1.0, 2.0, 2.1, 2.2))
    tracker.begin_update()
    tracker.inference_complete()
    report = tracker.report_payload_ready()
    assert report["intervals"]["update_to_inference_complete"]["status"] == "MEASURED"
    assert report["intervals"]["last_packet_receipt_to_inference_complete"]["status"] == "UNAVAILABLE"
    assert report["intervals"]["analysis_epoch_to_first_accepted_slip_label"]["reason"] == "accepted_slip_not_observed"
    assert report["observed_packet_count_current_update"] == 0


def test_packet_without_arrival_metadata_does_not_use_poll_time_or_device_time():
    tracker = SoftwareLatencyTracker(clock_from(1.0, 2.0, 2.1, 2.2))
    tracker.begin_update()
    tracker.observe_event_receipt(None)
    tracker.inference_complete()
    report = tracker.report_payload_ready()
    assert report["observed_packet_count_current_update"] == 1
    assert report["intervals"]["last_packet_receipt_to_inference_complete"]["reason"] == "application_receipt_not_recorded"


def test_unavailable_sensor_does_not_claim_inference_ran():
    tracker = SoftwareLatencyTracker(clock_from(1.0, 2.0, 2.1))
    tracker.begin_update()
    report = tracker.report_payload_ready()
    assert report["intervals"]["update_to_inference_complete"]["reason"] == "inference_not_run"
    assert report["intervals"]["update_to_report_payload_ready"]["duration_ms"] == 100


def test_last_packet_in_current_update_has_an_explicit_receipt_endpoint():
    tracker = SoftwareLatencyTracker(clock_from(1.0, 2.0, 2.3, 2.4))
    tracker.begin_update()
    tracker.observe_event_receipt(2.01)
    tracker.observe_event_receipt(2.2)
    tracker.inference_complete()
    report = tracker.report_payload_ready()
    assert report["observed_packet_count_current_update"] == 2
    assert report["intervals"]["last_packet_receipt_to_inference_complete"]["duration_ms"] == 100
    assert report["intervals"]["last_packet_receipt_to_inference_complete"]["start_timestamp_sec"] == 2.2


def test_reset_clears_first_label_and_can_preserve_analysis_origin():
    tracker = SoftwareLatencyTracker(clock_from(1.0, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 4.0))
    tracker.begin_update()
    tracker.inference_complete(accepted_slip=True)
    tracker.report_payload_ready()
    tracker.reset_epoch(reset_origin=False)
    tracker.begin_update()
    tracker.inference_complete()
    report = tracker.report_payload_ready()
    assert report["analysis_epoch_index"] == 2
    assert report["intervals"]["analysis_epoch_to_first_accepted_slip_label"]["duration_ms"] is None
    assert tracker.epoch_start.timestamp_sec == 1.0
    tracker.reset_epoch()
    assert tracker.epoch_start.timestamp_sec == 4.0


def test_device_counter_or_clock_reset_is_reported_as_invalid():
    tracker = SoftwareLatencyTracker(clock_from(100.0, 101.0, 1.0, 2.0))
    tracker.begin_update()
    tracker.observe_event_receipt(100.9)
    tracker.inference_complete(accepted_slip=True)
    report = tracker.report_payload_ready()
    for name in ("analysis_epoch_to_first_accepted_slip_label", "update_to_inference_complete", "last_packet_receipt_to_inference_complete"):
        assert report["intervals"][name]["status"] == "INVALID"
        assert report["intervals"][name]["duration_ms"] is None


def analyzer_with_stub(monkeypatch, *, active=True, with_packet=False, arrival=1.99):
    import time

    import numpy as np

    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **kwargs):
            self.roi_xywh = kwargs.get("roi_xywh")
            self.configured_roi = self.roi_xywh
            self.active = active
            self.backend_name = "synthetic_test"
            self.owner_info = {}
            self.latest_event_host_arrival_monotonic = arrival
            self.sent_packet = False

        def start(self):
            return {}

        def poll_event_and_frame_arrays(self):
            if with_packet and not self.sent_packet:
                self.sent_packet = True
                return np.asarray([[time.monotonic(), 150, 100, 1]], dtype=float), []
            return np.empty((0, 4), dtype=float), []

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    return motion_gui.DavisMotionAnalyzer(
        latency_clock=clock_from(1.0, 2.0, 2.05, 2.06),
    )


def test_actual_analyzer_inference_and_compatibility_serialization_have_endpoint_metadata(monkeypatch):
    from grasp_project.davis import motion_gui

    analyzer = analyzer_with_stub(monkeypatch, with_packet=True)
    report = analyzer.update()
    assert report["software_timing"]["intervals"]["last_packet_receipt_to_inference_complete"]["duration_ms"] == 60
    assert report["software_timing"]["intervals"]["update_to_report_payload_ready"]["duration_ms"] == 60
    assert report["slip_detection_latency_basis"] == "analysis_epoch_start_to_first_accepted_slip_window_reference"
    assert report["slip_detection_latency_physical_onset_measured"] is False
    assert report["slip_detection_latency_ms"] is None
    assert report["davis_motion_usable_for_grasp"] is False
    writes = {}
    monkeypatch.setattr(motion_gui, "_write_json", lambda path, value: writes.update({path.name: value}))
    monkeypatch.setattr(motion_gui, "_append_jsonl", lambda *_args: None)
    analyzer.save_outputs(include_images=False)
    assert writes["latest_motion_report.json"]["software_timing"] == report["software_timing"]
    assert writes["davis_motion_evaluation.json"]["software_timing"] == report["software_timing"]
    assert writes["davis_motion_evaluation.json"]["slip_detection_latency_basis"] == report["slip_detection_latency_basis"]


@pytest.mark.parametrize("with_packet,arrival,reason", [
    (False, 1.99, "no_event_observation_in_update"),
    (True, None, "application_receipt_not_recorded"),
])
def test_actual_analyzer_never_invents_event_receipt_time(monkeypatch, with_packet, arrival, reason):
    analyzer = analyzer_with_stub(monkeypatch, with_packet=with_packet, arrival=arrival)
    report = analyzer.update()
    measured = report["software_timing"]["intervals"]["last_packet_receipt_to_inference_complete"]
    assert measured["status"] == "UNAVAILABLE"
    assert measured["duration_ms"] is None
    assert measured["reason"] == reason
    assert report["software_timing"]["physical_onset_to_alert_ms"] is None
    assert report["slip_detection_latency_ms"] is None


def test_actual_analyzer_unavailable_path_has_no_stale_inference_endpoint(monkeypatch):
    analyzer = analyzer_with_stub(monkeypatch, active=False)
    report = analyzer.update()
    assert report["motion_class"] == "SENSOR_UNAVAILABLE"
    assert report["decision"] == "UNAVAILABLE"
    assert report["software_timing"]["intervals"]["update_to_inference_complete"]["reason"] == "inference_not_run"


def test_actual_analyzer_stale_evidence_reset_preserves_current_update_endpoint(monkeypatch):
    analyzer = analyzer_with_stub(monkeypatch)
    analyzer.latency_tracker = SoftwareLatencyTracker(clock_from(1.0, 2.0, 2.01, 2.05, 2.06))
    analyzer.object_evidence_context = {
        "captured_monotonic": 0.0, "aps_target_context_max_age_sec": 0.05,
    }
    report = analyzer.update()
    assert report["object_evidence_invalid_reason"] == "aps_object_evidence_stale"
    assert report["software_timing"]["analysis_epoch_index"] == 2
    intervals = report["software_timing"]["intervals"]
    assert intervals["update_to_inference_complete"]["duration_ms"] == 50
    assert intervals["update_to_report_payload_ready"]["duration_ms"] == 60
    assert intervals["analysis_epoch_to_first_accepted_slip_label"]["status"] == "UNAVAILABLE"


def test_recorded_window_parameters_override_current_environment_defaults():
    import numpy as np

    from grasp_project.davis.motion_gui import select_event_analysis_window

    events = np.asarray([[9.3, 100, 100, 1], [9.8, 100, 100, 1], [9.99, 100, 100, 1]])
    selected, window, mode, end = select_event_analysis_window(
        events, 10.0, [0, 0, 346, 260], primary_window_sec=0.05,
        fallback_window_sec=0.3, min_primary_events=2,
    )
    assert selected[:, 0].tolist() == [9.8, 9.99]
    assert (window, mode, end) == (0.3, "fallback_300ms", 10.0)


@pytest.mark.parametrize("kwargs", [
    {"primary_window_sec": 0}, {"primary_window_sec": float("nan")},
    {"primary_window_sec": True}, {"fallback_window_sec": -1},
    {"fallback_window_sec": float("inf")}, {"fallback_window_sec": "invalid"},
    {"min_primary_events": 0}, {"min_primary_events": 2.5},
    {"min_primary_events": False}, {"min_primary_events": float("nan")},
])
def test_recorded_window_configuration_rejects_invalid_values(kwargs):
    from grasp_project.davis.motion_gui import select_event_analysis_window

    with pytest.raises(ValueError):
        select_event_analysis_window([], 10.0, None, **kwargs)


def test_default_window_parameters_preserve_existing_selection():
    import numpy as np

    from grasp_project.davis import motion_gui

    events = np.asarray([[9.3, 100, 100, 1], [9.8, 100, 100, 1], [9.99, 100, 100, 1]])
    defaults = motion_gui.select_event_analysis_window(events, 10.0, None)
    explicit = motion_gui.select_event_analysis_window(
        events, 10.0, None,
        primary_window_sec=motion_gui.PRIMARY_EVENT_WINDOW_SEC,
        fallback_window_sec=motion_gui.FALLBACK_EVENT_WINDOW_SEC,
        min_primary_events=motion_gui.MIN_PRIMARY_WINDOW_EVENTS,
    )
    assert np.array_equal(defaults[0], explicit[0])
    assert defaults[1:] == explicit[1:]

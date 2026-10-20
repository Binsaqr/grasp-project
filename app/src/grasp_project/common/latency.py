"""Software endpoint intervals; never infer physical onset or clock alignment."""

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ClockStamp:
    """An endpoint in seconds in an explicitly named clock domain."""

    timestamp_sec: float | None
    clock_domain: str


def endpoint_interval(start, end, *, start_name, end_name, missing_reason="endpoint_not_observed"):
    """Return an interval only for finite, ordered endpoints in one clock.

    A decreasing device counter or a reset clock is invalid, not a zero delay.
    Clock mappings and counter rollover correction belong in acquisition code.
    """
    result = {
        "start_endpoint": start_name,
        "end_endpoint": end_name,
        "clock_domain": None,
        "start_timestamp_sec": None,
        "end_timestamp_sec": None,
        "status": "UNAVAILABLE",
        "duration_ms": None,
        "reason": missing_reason,
    }
    if start is None or end is None:
        return result
    if not start.clock_domain or not end.clock_domain:
        return {**result, "status": "INVALID", "reason": "clock_domain_required"}
    if start.clock_domain != end.clock_domain:
        return {**result, "status": "INVALID", "reason": "cross_clock_interval_not_allowed"}
    result["clock_domain"] = start.clock_domain
    if start.timestamp_sec is None or end.timestamp_sec is None:
        return result
    values = (start.timestamp_sec, end.timestamp_sec)
    try:
        if any(isinstance(value, bool) for value in values):
            raise ValueError("boolean timestamp")
        start_sec, end_sec = map(float, values)
    except (TypeError, ValueError, OverflowError):
        return {**result, "status": "INVALID", "reason": "invalid_timestamp"}
    if not all(math.isfinite(value) for value in (start_sec, end_sec)):
        return {**result, "status": "INVALID", "reason": "nonfinite_timestamp"}
    result.update(start_timestamp_sec=start_sec, end_timestamp_sec=end_sec)
    if end_sec < start_sec:
        return {**result, "status": "INVALID", "reason": "negative_interval_clock_reset_or_rollover"}
    duration_ms = (end_sec - start_sec) * 1000.0
    if not math.isfinite(duration_ms):
        return {**result, "status": "INVALID", "reason": "nonfinite_interval"}
    return {**result, "status": "MEASURED", "duration_ms": round(duration_ms, 6), "reason": None}


class SoftwareLatencyTracker:
    """Record host processing boundaries without modifying inference policy.

    ``report_payload_ready`` means the Python decision payload is assembled.
    It excludes file serialization, transport, browser receipt and rendering.
    Application receipt is the nonempty backend poll's return, stamped by
    acquisition. It does not measure hardware, bus or kernel arrival time.
    """

    def __init__(self, clock=None, *, clock_domain="host_monotonic", epoch_start=None):
        self.clock = clock or time.monotonic
        self.clock_domain = clock_domain
        self.epoch_index = 0
        self._update_in_progress = False
        self.reset_epoch(started_at=epoch_start)

    def _stamp(self):
        return ClockStamp(self.clock(), self.clock_domain)

    def reset_epoch(self, *, started_at=None, reset_origin=True):
        """Clear first-label endpoints when analyzer motion evidence resets."""
        if reset_origin or not hasattr(self, "epoch_start"):
            self.epoch_start = self._stamp() if started_at is None else ClockStamp(started_at, self.clock_domain)
        self.epoch_index += 1
        self.first_accepted_slip_label = None
        self.first_slip_report_ready = None
        if not self._update_in_progress:
            self.update_entry = None
            self.last_receipt = None
            self.inference_completion = None
            self.observed_packet_count = 0
        self.accepted_in_update = False

    def begin_update(self):
        self._update_in_progress = True
        self.update_entry = self._stamp()
        self.last_receipt = None
        self.inference_completion = None
        self.observed_packet_count = 0
        self.accepted_in_update = False

    def observe_event_receipt(self, timestamp_sec, *, clock_domain="host_monotonic"):
        """Use acquisition's stamp of the last nonempty backend poll return."""
        self.observed_packet_count += 1
        self.last_receipt = ClockStamp(timestamp_sec, clock_domain)

    def inference_complete(self, *, accepted_slip=False):
        self.inference_completion = self._stamp()
        if accepted_slip and self.first_accepted_slip_label is None:
            self.first_accepted_slip_label = self.inference_completion
            self.accepted_in_update = True

    def report_payload_ready(self):
        ready = self._stamp()
        self._update_in_progress = False
        if self.accepted_in_update and self.first_slip_report_ready is None:
            self.first_slip_report_ready = ready
        receipt_reason = "application_receipt_not_recorded" if self.observed_packet_count else "no_event_observation_in_update"
        intervals = {}
        for name, start, end, start_name, end_name, missing in (
            ("analysis_epoch_to_first_accepted_slip_label", self.epoch_start, self.first_accepted_slip_label,
             "analysis_epoch_start", "first_accepted_slip_label", "accepted_slip_not_observed"),
            ("update_to_inference_complete", self.update_entry, self.inference_completion,
             "update_entry", "inference_complete", "inference_not_run"),
            ("last_packet_receipt_to_inference_complete", self.last_receipt, self.inference_completion,
             "last_nonempty_backend_poll_return", "inference_complete", receipt_reason),
            ("last_packet_receipt_to_report_payload_ready", self.last_receipt, ready,
             "last_nonempty_backend_poll_return", "report_payload_ready", receipt_reason),
            ("update_to_report_payload_ready", self.update_entry, ready,
             "update_entry", "report_payload_ready", "update_not_observed"),
            ("first_accepted_slip_label_to_first_report_payload_ready", self.first_accepted_slip_label,
             self.first_slip_report_ready, "first_accepted_slip_label", "first_slip_report_payload_ready",
             "accepted_slip_not_observed"),
        ):
            intervals[name] = endpoint_interval(start, end, start_name=start_name, end_name=end_name, missing_reason=missing)
        return {
            "schema_version": 1,
            "measurement_kind": "SOFTWARE_ENDPOINT_INTERVALS",
            "timestamp_unit": "seconds",
            "duration_unit": "milliseconds",
            "clock_domain": self.clock_domain,
            "analysis_epoch_index": self.epoch_index,
            "observed_packet_count_current_update": self.observed_packet_count,
            "application_receipt_endpoint_basis": "device_get_event_return",
            "hardware_arrival_measured": False,
            "intervals": intervals,
            "physical_onset_to_alert_ms": None,
            "physical_onset_to_alert_status": "UNAVAILABLE_INDEPENDENT_ONSET_AND_CLOCK_MAPPING_REQUIRED",
            "browser_display_latency_ms": None,
            "browser_display_latency_status": "UNAVAILABLE_RECEIPT_AND_RENDER_ENDPOINTS_NOT_INSTRUMENTED",
            "report_ready_scope": "in_memory_payload_before_file_serialization_or_transport",
        }

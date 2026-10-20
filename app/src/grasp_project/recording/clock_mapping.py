"""Offline affine clock diagnostics from explicitly declared correspondences.

No pairing, clock reset repair, live-policy application or physical calibration
is inferred. Integer-centered rational arithmetic retains large native clocks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction

SCHEMA_VERSION = "grasp_clock_correspondences_v1"
REPORT_SCHEMA_VERSION = "grasp_clock_mapping_diagnostics_v1"
UNIT_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
KINDS = ("simultaneous_event", "host_receipt_association")
MAX_PAIRS = 10_000
RECORDING_STREAMS = ("davis_poll", "rgb", "depth", "camera_info")


class ClockMappingError(ValueError):
    """Correspondences do not satisfy the explicit clock contract."""


def _require(condition, message):
    if not condition:
        raise ClockMappingError(message)


def _object(value, name):
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _list(value, name):
    _require(isinstance(value, list), f"{name} must be a list")
    return value


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def _integer(value, name, *, minimum=None):
    _require(type(value) is int and -(2**127) <= value < 2**127, f"{name} must be a signed 128-bit JSON integer")
    _require(minimum is None or value >= minimum, f"{name} must be >= {minimum}")
    return value


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _clock(value, name):
    value = _object(value, name)
    _text(value.get("domain"), f"{name}.domain")
    _require(isinstance(value.get("unit"), str) and value["unit"] in UNIT_NS, f"{name}.unit must be ns/us/ms/s")
    return value


def _fit_epoch(epoch, source_factor, target_factor, residual_alert_ns):
    pairs = epoch["pairs"]
    result = {
        "epoch_id": epoch["epoch_id"], "source_epoch_id": epoch["source_epoch_id"],
        "target_epoch_id": epoch["target_epoch_id"],
        "correspondence_kind": epoch["correspondence_kind"],
        "interpretation": (
            "DECLARED_SIMULTANEOUS_EVENTS_UNVALIDATED"
            if epoch["correspondence_kind"] == "simultaneous_event"
            else "HOST_RECEIPT_ASSOCIATION_ONLY"
        ),
        "provenance": epoch["provenance"], "pairs": pairs,
        "pair_count": len(pairs), "excluded_pair_count": 0,
        "status": "UNAVAILABLE_INSUFFICIENT_PAIRS", "model": None,
    }
    if len(pairs) < 3:
        return result, None
    source_origin, target_origin = pairs[0]["source_timestamp"], pairs[0]["target_timestamp"]
    # Subtract and scale as integers before any floating-point conversion.
    xs = [(row["source_timestamp"] - source_origin) * source_factor for row in pairs]
    ys = [(row["target_timestamp"] - target_origin) * target_factor for row in pairs]
    count, sx, sy = len(pairs), sum(xs), sum(ys)
    denominator = count * sum(x * x for x in xs) - sx * sx
    rate = Fraction(count * sum(x * y for x, y in zip(xs, ys)) - sx * sy, denominator)
    offset = Fraction(sy, count) - rate * Fraction(sx, count)
    residuals = [Fraction(y) - offset - rate * x for x, y in zip(xs, ys)]
    sse = sum((residual * residual for residual in residuals), Fraction())
    variance = sse / (count - 2)
    sxx = Fraction(denominator, count)
    sigma_ns = math.sqrt(float(variance))
    diagnostics = []
    for row, residual in zip(pairs, residuals):
        source_uncertainty = row.get("source_uncertainty_ns")
        target_uncertainty = row.get("target_uncertainty_ns")
        declared = None
        if source_uncertainty is not None and target_uncertainty is not None:
            declared = Fraction(target_uncertainty) + abs(rate) * source_uncertainty
        diagnostics.append({
            "pair_id": row["pair_id"], "residual_ns": float(residual),
            "exceeds_residual_alert": None if residual_alert_ns is None else abs(residual) > residual_alert_ns,
            "declared_combined_uncertainty_ns": None if declared is None else float(declared),
            "residual_exceeds_declared_uncertainty": None if declared is None else abs(residual) > declared,
        })
    result.update({
        "status": "ESTIMATED_UNVALIDATED",
        "fit_method": "ordinary_least_squares_all_declared_pairs",
        "model": {
            "source_origin_native": source_origin, "target_origin_native": target_origin,
            "source_unit_ns": source_factor, "target_unit_ns": target_factor,
            "rate_ratio": float(rate), "rate_ratio_exact": _fraction(rate),
            "drift_ppm": float((rate - 1) * 1_000_000),
            "target_offset_from_target_origin_ns_exact": _fraction(offset),
            "target_minus_source_at_source_origin_ns_exact": _fraction(
                target_origin * target_factor + offset - source_origin * source_factor
            ),
            "formula": "target_native = target_origin_native + (offset_ns + rate_ratio * (source_native - source_origin_native) * source_unit_ns) / target_unit_ns",
            "source_range_native": [pairs[0]["source_timestamp"], pairs[-1]["source_timestamp"]],
            "target_range_native": [pairs[0]["target_timestamp"], pairs[-1]["target_timestamp"]],
        },
        "diagnostics": {
            "residual_unit": "ns", "residual_definition": "observed_target_minus_fitted_target",
            "source_span_ns": xs[-1] - xs[0], "target_span_ns": ys[-1] - ys[0],
            "residual_rmse_ns": math.sqrt(float(sse / count)),
            "residual_max_abs_ns": float(max(map(abs, residuals))),
            "residual_alert_ns": residual_alert_ns,
            "residual_alert_count": None if residual_alert_ns is None else sum(row["exceeds_residual_alert"] for row in diagnostics),
            "pairs": diagnostics,
            "residual_standard_error_ns": sigma_ns,
            "rate_ratio_standard_error": math.sqrt(float(variance / sxx)),
            "offset_from_target_origin_standard_error_ns": sigma_ns * math.sqrt(float(Fraction(1, count) + Fraction(sx, count) ** 2 / sxx)),
            "uncertainty_assumptions": "OLS independent homoscedastic target errors and exact source values; not a physical error bound",
            "includes_pairing_or_acquisition_delay_uncertainty": False,
            "zero_residual_proves_zero_physical_uncertainty": False,
        },
    })
    return result, (source_origin, target_origin, rate, offset, count, Fraction(sx, count), sxx, variance)


def _query(row, epoch, fit, source_clock, source_factor, target_factor):
    result = {
        **row, "interpretation": epoch["interpretation"],
        "status": "UNAVAILABLE_NO_FIT", "target_estimate_native": None,
        "target_estimate_native_exact": None,
    }
    if fit is None:
        return result
    source_origin, target_origin, rate, offset, count, mean_x, sxx, variance = fit
    value = row["source_timestamp"]
    lo, hi = epoch["model"]["source_range_native"]
    distance = max(lo - value, value - hi, 0) * source_factor
    result["extrapolation_distance_ns"] = distance
    if distance > row.get("max_extrapolation_ns", 0):
        result["status"] = "UNAVAILABLE_EXTRAPOLATION_LIMIT"
        return result
    x = (value - source_origin) * source_factor
    target = Fraction(target_origin) + (offset + rate * x) / target_factor
    estimate = round(target)
    leverage = Fraction(1, count) + (x - mean_x) ** 2 / sxx
    result.update({
        "status": "EXTRAPOLATED_UNVALIDATED" if distance else "INTERPOLATED_UNVALIDATED",
        "target_estimate_native": estimate, "target_estimate_native_exact": _fraction(target),
        "target_rounding_error_ns": float((estimate - target) * target_factor),
        "conditional_mean_fit_standard_error_ns": math.sqrt(float(variance * leverage)),
        "conditional_new_pair_prediction_standard_error_ns": math.sqrt(float(variance * (1 + leverage))),
        "source_clock_domain": source_clock["domain"],
    })
    return result


def estimate_clock_mapping(document):
    """Fit explicit epochs and diagnose optional bounded offline queries.

    Correspondence order is evidence: nonincreasing native clocks are rejected,
    never sorted or repaired. Declare a new epoch for each reset or rollover.
    """
    document = _object(document, "correspondences")
    try:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ClockMappingError(f"correspondences must be finite JSON: {exc}") from exc
    _require(document.get("schema_version") == SCHEMA_VERSION, f"expected schema_version {SCHEMA_VERSION}")
    source = _clock(document.get("source_clock"), "source_clock")
    target = _clock(document.get("target_clock"), "target_clock")
    source_factor, target_factor = UNIT_NS[source["unit"]], UNIT_NS[target["unit"]]
    threshold = document.get("residual_alert_ns")
    if threshold is not None:
        _integer(threshold, "residual_alert_ns", minimum=0)
    epochs = _list(document.get("epochs"), "epochs")
    queries = _list(document.get("queries", []), "queries")
    _require(0 < len(epochs) <= 1_000, "epochs must contain 1..1000 explicit epochs")
    _require(len(queries) <= MAX_PAIRS, "too many queries")
    seen_epochs, seen_pairs, fits, results = set(), set(), {}, {}
    for raw in epochs:
        epoch = _object(raw, "epoch")
        epoch_id = _text(epoch.get("epoch_id"), "epoch_id")
        _require(epoch_id not in seen_epochs, f"duplicate epoch_id: {epoch_id}")
        seen_epochs.add(epoch_id)
        for name in ("source_epoch_id", "target_epoch_id"):
            _text(epoch.get(name), name)
        _require(epoch.get("correspondence_kind") in KINDS, "correspondence_kind must explicitly distinguish simultaneous_event or host_receipt_association")
        provenance = _object(epoch.get("provenance"), "provenance")
        for name in ("method", "reference"):
            _text(provenance.get(name), f"provenance.{name}")
        pairs = _list(epoch.get("pairs"), "pairs")
        previous = None
        for raw_pair in pairs:
            pair = _object(raw_pair, "pair")
            pair_id = _text(pair.get("pair_id"), "pair_id")
            _require(pair_id not in seen_pairs, f"duplicate pair_id: {pair_id}")
            seen_pairs.add(pair_id)
            _require(len(seen_pairs) <= MAX_PAIRS, "too many correspondence pairs")
            values = tuple(_integer(pair.get(name), name) for name in ("source_timestamp", "target_timestamp"))
            _require(previous is None or all(now > before for now, before in zip(values, previous)), f"nonincreasing clock in epoch {epoch_id}; split resets/rollovers into explicit epochs")
            previous = values
            for name in ("source_uncertainty_ns", "target_uncertainty_ns"):
                if name in pair:
                    _integer(pair[name], name, minimum=0)
        results[epoch_id], fits[epoch_id] = _fit_epoch(epoch, source_factor, target_factor, threshold)
    query_results, seen_queries = [], set()
    for raw in queries:
        row = _object(raw, "query")
        query_id = _text(row.get("query_id"), "query_id")
        _require(query_id not in seen_queries, f"duplicate query_id: {query_id}")
        seen_queries.add(query_id)
        epoch_id = _text(row.get("epoch_id"), "query.epoch_id")
        _require(epoch_id in results, f"unknown query epoch: {epoch_id}")
        _require(row.get("source_clock_domain") == source["domain"] and row.get("source_unit") == source["unit"], "query source clock domain/unit mismatch")
        _require(row.get("source_epoch_id") == results[epoch_id]["source_epoch_id"], "query source epoch mismatch")
        _integer(row.get("source_timestamp"), "query.source_timestamp")
        _integer(row.get("max_extrapolation_ns", 0), "max_extrapolation_ns", minimum=0)
        query_results.append(_query(row, results[epoch_id], fits[epoch_id], source, source_factor, target_factor))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "source_clock": source, "target_clock": target,
        "epochs": list(results.values()), "queries": query_results,
        "physical_alignment_established": False, "applied_to_live_policy": False,
        "limitations": [
            "Correspondence simultaneity and provenance are declarations, not authenticated measurements.",
            "Host receipt associations include acquisition/transport/queue delay and do not identify simultaneous event times.",
            "All declared pairs are fitted; outliers are retained and can bias offset/rate estimates.",
            "OLS uncertainty is conditional on the affine/noise model and does not bound physical clock error.",
            "Resets/rollovers require explicit epochs; extrapolation is forbidden unless explicitly bounded per query.",
            "No hardware synchronization, physical latency or slip lead time is established.",
        ],
    }


def _recorded_native_time(record, stream):
    """Extract one receipt association, retaining the native timestamp basis."""
    clock, metadata = record["native_clock"], record["metadata"]
    if stream == "davis_poll":
        events = record["arrays"].get("events")
        if events is None or events.ndim != 2 or events.shape[1] < 4:
            return None, None, "invalid_native_event_array"
        if not len(events):
            return None, None, "empty_poll_no_native_event_timestamp"
        if clock["domain"] != "davis_device" or clock["unit"] != "us":
            return None, None, "invalid_davis_native_clock_domain_or_unit"
        if events.dtype.kind not in "ui":
            return None, None, "native_event_timestamps_not_recorded_as_integers"
        epoch = metadata.get("normalization_epoch")
        if type(epoch) is not int or epoch < 0:
            return None, None, "normalization_epoch_not_recorded"
        native_epoch = clock.get("epoch_id")
        if native_epoch is not None and (not isinstance(native_epoch, str) or not native_epoch.strip()):
            return None, None, "native_clock_epoch_id_invalid"
        timestamps = events[:, 0]
        if (timestamps[1:] < timestamps[:-1]).any():
            return None, (epoch, native_epoch), "native_reset_within_packet"
        return int(timestamps[-1]), (epoch, native_epoch), None
    if clock["domain"].strip().lower() in {
        "unknown", "unspecified", "ros_header_unspecified", "not_available",
    }:
        return None, None, "ros_native_clock_domain_unspecified"
    if clock["unit"] != "ns":
        return None, None, "invalid_ros_native_clock_unit"
    epoch = clock.get("epoch_id")
    if not isinstance(epoch, str) or not epoch.strip():
        return None, None, "ros_native_clock_epoch_not_recorded"
    sec, nanosec = metadata.get("header_stamp_sec"), metadata.get("header_stamp_nanosec")
    if type(sec) is not int or type(nanosec) is not int or not 0 <= nanosec < 1_000_000_000:
        return None, (None, epoch), "ros_native_header_stamp_not_exact_integers"
    timestamp = sec * 1_000_000_000 + nanosec
    if type(clock.get("timestamp")) is not int or clock["timestamp"] != timestamp:
        return None, (None, epoch), "ros_native_timestamp_disagrees_with_header"
    return timestamp, (None, epoch), None


def extract_recording_receipt_correspondences(recording_dir, stream):
    """Verify an archive and extract associations only within recorded epochs.

    Unsupported observations remain explicit unavailable rows. A native reset
    without a newly recorded epoch blocks the remaining segment; no reset epoch
    or missing physical clock identity is inferred from observed counter values.
    """
    from .session import read_recording

    _require(stream in RECORDING_STREAMS, "unsupported recording stream")
    recording = read_recording(recording_dir)
    manifest = recording.manifest
    rows, epochs, reasons = [], [], Counter()
    source_clock = None
    current_key, previous_native, previous_receipt, blocked_key = None, None, None, None
    seen_keys = set()
    drops = [record for record in recording.records if record["record_type"] == "drop"
             and record["stream"] in ({"davis_poll", "davis_events"} if stream == "davis_poll" else {stream})]
    for record in recording.records:
        if record["record_type"] != "observation" or record["kind"] != stream:
            continue
        clock, metadata = record["native_clock"], record["metadata"]
        native, key, reason = _recorded_native_time(record, stream)
        row = {
            "observation_id": record["observation_id"], "record_sequence": record["sequence"],
            "native_clock": clock, "host_receipt_monotonic_ns": record["host_arrival_monotonic_ns"],
            "normalization_epoch": metadata.get("normalization_epoch"),
            "native_epoch_id": clock.get("epoch_id"),
            "header_stamp_sec": metadata.get("header_stamp_sec"),
            "header_stamp_nanosec": metadata.get("header_stamp_nanosec"),
            "source_sequence_gap": record["source_sequence_gap"],
            "status": "UNAVAILABLE", "reason": reason,
        }
        if stream == "davis_poll" and key is not None and current_key is not None and key[0] < current_key[0]:
            reason = "recorded_normalization_epoch_decreased"
        if key is not None and key != current_key:
            if key in seen_keys:
                reason = "recorded_clock_epoch_reentered"
            else:
                seen_keys.add(key)
                current_key, previous_native, previous_receipt, blocked_key = key, None, None, None
        if reason is None and key == blocked_key:
            reason = "clock_continuity_unknown_until_new_recorded_epoch"
        descriptor = {"domain": clock["domain"], "unit": clock["unit"]}
        if reason is None and source_clock is not None and descriptor != source_clock:
            reason = "native_clock_domain_or_unit_changed"
        if reason is None and previous_native is not None and native < previous_native:
            reason = "native_reset_without_recorded_epoch_change"
        if reason is None and native == previous_native:
            reason = "duplicate_native_timestamp_no_distinct_correspondence"
        if reason is None and record["host_arrival_monotonic_ns"] == previous_receipt:
            reason = "duplicate_host_receipt_no_distinct_correspondence"
        if reason is not None:
            if reason not in {"empty_poll_no_native_event_timestamp", "duplicate_native_timestamp_no_distinct_correspondence", "duplicate_host_receipt_no_distinct_correspondence"}:
                blocked_key = current_key
            row["reason"] = reason
            reasons[reason] += 1
            rows.append(row)
            continue
        source_clock = descriptor
        _integer(native, "recorded native timestamp")
        if not epochs or epochs[-1]["recorded_epoch_key"] != list(key):
            epochs.append({
                "epoch_id": f"{manifest['recording_id']}:{stream}:segment-{len(epochs)}",
                "source_epoch_id": f"native:{key[1]}" if key[1] is not None else f"normalization:{key[0]}",
                "target_epoch_id": f"archive:{manifest['recording_id']}:host_monotonic_continuity",
                "recorded_epoch_key": list(key), "correspondence_kind": "host_receipt_association",
                "provenance": {"method": "verified_raw_archive_application_receipt_association",
                               "reference": f"manifest-sha256:{manifest['manifest_sha256']}",
                               "recorded_epoch_key": list(key)},
                "pairs": [],
            })
        pair = {"pair_id": record["observation_id"], "source_timestamp": native,
                "target_timestamp": record["host_arrival_monotonic_ns"],
                "native_clock": clock, "normalization_epoch": metadata.get("normalization_epoch"),
                "acquisition_endpoint": metadata.get("acquisition_endpoint"),
                "receipt_endpoint_status": "RECORDED" if metadata.get("acquisition_endpoint") else "NOT_RECORDED"}
        if stream != "davis_poll":
            pair["native_header_stamp"] = {"sec": metadata["header_stamp_sec"], "nanosec": metadata["header_stamp_nanosec"]}
        epochs[-1]["pairs"].append(pair)
        previous_native, previous_receipt = native, record["host_arrival_monotonic_ns"]
        row.update(status="USED_HOST_RECEIPT_ASSOCIATION", reason=None, source_timestamp=native)
        rows.append(row)
    correspondences = None if source_clock is None else {
        "schema_version": SCHEMA_VERSION, "source_clock": source_clock,
        "target_clock": {"domain": "host_monotonic", "unit": "ns"}, "epochs": epochs, "queries": [],
    }
    return {
        "correspondences": correspondences,
        "archive": {name: manifest[name] for name in (
            "recording_id", "trial_id", "manifest_sha256", "config_sha256", "revision_sha256", "metadata_sha256",
        )},
        "extraction": {
            "stream": stream, "interpretation": "HOST_RECEIPT_ASSOCIATION_ONLY",
            "selected_observation_count": len(rows),
            "used_pair_count": sum(len(epoch["pairs"]) for epoch in epochs),
            "unavailable_observation_count": sum(reasons.values()), "unavailable_reasons": dict(sorted(reasons.items())),
            "drop_records": drops,
            "known_dropped_observations": sum(record["count"] or 0 for record in drops),
            "unknown_drop_events": sum(record["count"] is None for record in drops),
            "source_sequence_gaps": sum(row["source_sequence_gap"] for row in rows),
            "observations": rows,
            "source_epoch_scope": "recorded normalization/native epoch only; normalization epoch is not a verified device boot",
            "target_epoch_scope": "verified archive-local monotonic continuity; not a cross-archive host boot identity",
        },
    }


def estimate_recording_clock_mapping(recording_dir, stream, *, residual_alert_ns=None):
    """Estimate only archive host receipt associations, retaining unavailable input."""
    if residual_alert_ns is not None:
        _integer(residual_alert_ns, "residual_alert_ns", minimum=0)
    extracted = extract_recording_receipt_correspondences(recording_dir, stream)
    document = extracted.pop("correspondences")
    if document is None:
        report = {"schema_version": REPORT_SCHEMA_VERSION, "input_sha256": None,
                  "source_clock": None, "target_clock": {"domain": "host_monotonic", "unit": "ns"},
                  "epochs": [], "queries": [], "physical_alignment_established": False,
                  "applied_to_live_policy": False, "status": "UNAVAILABLE_NO_USABLE_ASSOCIATIONS"}
        report["availability_reason"] = "selected_stream_not_recorded" if extracted["extraction"]["selected_observation_count"] == 0 else "no_usable_recorded_native_clock_epoch"
    else:
        document["residual_alert_ns"] = residual_alert_ns
        report = estimate_clock_mapping(document)
        report["status"] = "ESTIMATED_UNVALIDATED" if any(epoch["model"] is not None for epoch in report["epochs"]) else "UNAVAILABLE_INSUFFICIENT_PAIRS"
    report.update(extracted)
    report["archive_input_requirements"] = [
        "DAVIS requires recorded integer native event microseconds and a normalization epoch; the last event in each nonempty packet is associated with poll return.",
        "ROS requires an explicitly documented native clock domain/epoch_id and matching exact header sec/nanosec; ros_header_unspecified is unavailable.",
        "Native resets without a new recorded epoch block that segment; no clock/reset identity or simultaneous event is invented.",
        "Receipt associations include acquisition, transport and queue delays and establish no physical alignment or hardware latency.",
    ]
    return report

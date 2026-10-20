"""Synthetic clock contract tests, not a hardware synchronization study."""

import copy
import json
import runpy
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from grasp_project.recording.clock_mapping import (
    SCHEMA_VERSION,
    ClockMappingError,
    estimate_clock_mapping,
    estimate_recording_clock_mapping,
    extract_recording_receipt_correspondences,
)


def correspondence_document(*, kind="simultaneous_event", count=5):
    return {
        "schema_version": SCHEMA_VERSION,
        "source_clock": {"domain": "davis_device", "unit": "us"},
        "target_clock": {"domain": "host_monotonic", "unit": "ns"},
        "residual_alert_ns": 50,
        "epochs": [{
            "epoch_id": "epoch-0", "source_epoch_id": "device-boot-0",
            "target_epoch_id": "host-boot-0", "correspondence_kind": kind,
            "provenance": {"method": "synthetic_fixture", "reference": "test-only://paired-events"},
            "pairs": [
                {"pair_id": f"pair-{index}", "source_timestamp": 4_000_000_000 + index * 20_000,
                 "target_timestamp": 1_700_000_000_000_000_017 + index * 20_001_000}
                for index in range(count)
            ],
        }],
        "queries": [],
    }


def query(timestamp, **updates):
    return {
        "query_id": "query-0", "epoch_id": "epoch-0",
        "source_epoch_id": "device-boot-0", "source_clock_domain": "davis_device",
        "source_unit": "us", "source_timestamp": timestamp, **updates,
    }


def exact(value):
    return Fraction(value["numerator"], value["denominator"])


def test_exact_centered_offset_and_drift_preserve_native_origins():
    document = correspondence_document()
    original = copy.deepcopy(document)
    document["queries"] = [query(4_000_010_000)]
    report = estimate_clock_mapping(document)
    epoch = report["epochs"][0]
    model, diagnostics = epoch["model"], epoch["diagnostics"]
    assert model["source_origin_native"] == 4_000_000_000
    assert model["target_origin_native"] == 1_700_000_000_000_000_017
    assert exact(model["rate_ratio_exact"]) == Fraction(20_001, 20_000)
    assert model["drift_ppm"] == pytest.approx(50.0)
    assert exact(model["target_offset_from_target_origin_ns_exact"]) == 0
    assert exact(model["target_minus_source_at_source_origin_ns_exact"]) == 1_699_996_000_000_000_017
    assert diagnostics["residual_rmse_ns"] == diagnostics["residual_max_abs_ns"] == 0
    assert diagnostics["rate_ratio_standard_error"] == 0
    assert diagnostics["zero_residual_proves_zero_physical_uncertainty"] is False
    mapped = report["queries"][0]
    assert mapped["status"] == "INTERPOLATED_UNVALIDATED"
    assert mapped["target_estimate_native"] == 1_700_000_000_010_000_517
    assert report["physical_alignment_established"] is report["applied_to_live_policy"] is False
    assert epoch["pairs"] == original["epochs"][0]["pairs"]
    assert document["epochs"] == original["epochs"]
    json.dumps(report, allow_nan=False)


def test_large_nanosecond_timestamps_retain_single_tick_changes():
    document = correspondence_document()
    document["source_clock"]["unit"] = "ns"
    source_origin, target_origin = 2**64 + 17, 2**65 + 29
    document["epochs"][0]["pairs"] = [
        {"pair_id": str(index), "source_timestamp": source_origin + index,
         "target_timestamp": target_origin + index}
        for index in range(5)
    ]
    document["queries"] = [query(source_origin + 3, source_unit="ns")]
    report = estimate_clock_mapping(document)
    assert exact(report["epochs"][0]["model"]["rate_ratio_exact"]) == 1
    mapped = report["queries"][0]
    assert type(mapped["target_estimate_native"]) is int
    assert mapped["target_estimate_native"] == target_origin + 3
    assert exact(mapped["target_estimate_native_exact"]) == target_origin + 3
    assert report["epochs"][0]["diagnostics"]["residual_max_abs_ns"] == 0


def test_different_native_units_produce_a_dimensionless_rate_ratio():
    document = correspondence_document()
    document["source_clock"]["unit"], document["target_clock"]["unit"] = "ns", "us"
    document["epochs"][0]["pairs"] = [
        {"pair_id": str(index), "source_timestamp": 10**18 + index * 20_000_000,
         "target_timestamp": 4_000_000_000 + index * 20_001}
        for index in range(5)
    ]
    report = estimate_clock_mapping(document)
    assert exact(report["epochs"][0]["model"]["rate_ratio_exact"]) == Fraction(20_001, 20_000)
    assert report["epochs"][0]["model"]["drift_ppm"] == pytest.approx(50)


def test_outliers_remain_in_fit_and_raise_explicit_residual_diagnostics():
    document = correspondence_document()
    document["epochs"][0]["pairs"][2]["target_timestamp"] += 100
    report = estimate_clock_mapping(document)
    epoch = report["epochs"][0]
    diagnostics = epoch["diagnostics"]
    assert epoch["excluded_pair_count"] == 0
    assert diagnostics["residual_alert_count"] == 1
    assert diagnostics["residual_max_abs_ns"] == 80
    assert [pair["residual_ns"] for pair in diagnostics["pairs"]] == [-20, -20, 80, -20, -20]
    assert diagnostics["residual_standard_error_ns"] > 0
    assert diagnostics["rate_ratio_standard_error"] > 0
    assert len(epoch["pairs"]) == 5


def test_declared_uncertainty_is_disclosed_without_becoming_a_fit_weight():
    document = correspondence_document()
    document["epochs"][0]["pairs"][2].update(
        target_timestamp=1_700_000_000_040_002_117,
        source_uncertainty_ns=10, target_uncertainty_ns=20,
    )
    report = estimate_clock_mapping(document)
    diagnostics = report["epochs"][0]["diagnostics"]
    pair = diagnostics["pairs"][2]
    assert pair["declared_combined_uncertainty_ns"] == pytest.approx(30.0005)
    assert pair["residual_exceeds_declared_uncertainty"] is True
    assert diagnostics["includes_pairing_or_acquisition_delay_uncertainty"] is False


def test_host_receipt_associations_cannot_be_relabelled_as_simultaneity():
    report = estimate_clock_mapping(correspondence_document(kind="host_receipt_association"))
    assert report["epochs"][0]["interpretation"] == "HOST_RECEIPT_ASSOCIATION_ONLY"
    assert report["physical_alignment_established"] is False
    assert any("queue delay" in limitation for limitation in report["limitations"])


@pytest.mark.parametrize("field", ["source_timestamp", "target_timestamp"])
def test_reset_or_rollover_within_an_epoch_is_rejected_without_sorting(field):
    document = correspondence_document()
    document["epochs"][0]["pairs"][3][field] = 0
    with pytest.raises(ClockMappingError, match="resets/rollovers"):
        estimate_clock_mapping(document)


def test_explicit_reset_epochs_are_fitted_independently_and_cannot_be_crossed():
    document = correspondence_document()
    second = copy.deepcopy(document["epochs"][0])
    second.update(epoch_id="epoch-1", source_epoch_id="device-boot-1")
    for index, pair in enumerate(second["pairs"]):
        pair.update(pair_id=f"after-reset-{index}", source_timestamp=index * 20_000)
        pair["target_timestamp"] += 1_000_000_000
    document["epochs"].append(second)
    document["queries"] = [query(10_000, epoch_id="epoch-1", source_epoch_id="device-boot-1")]
    report = estimate_clock_mapping(document)
    assert len(report["epochs"]) == 2
    assert report["queries"][0]["target_estimate_native"] == 1_700_000_001_010_000_517
    document["queries"][0]["source_epoch_id"] = "device-boot-0"
    with pytest.raises(ClockMappingError, match="epoch mismatch"):
        estimate_clock_mapping(document)


def test_extrapolation_is_forbidden_unless_explicitly_bounded():
    document = correspondence_document()
    document["queries"] = [query(4_000_090_000)]
    forbidden = estimate_clock_mapping(document)["queries"][0]
    assert forbidden["status"] == "UNAVAILABLE_EXTRAPOLATION_LIMIT"
    assert forbidden["extrapolation_distance_ns"] == 10_000_000
    assert forbidden["target_estimate_native"] is None
    document["queries"][0]["max_extrapolation_ns"] = 10_000_000
    allowed = estimate_clock_mapping(document)["queries"][0]
    assert allowed["status"] == "EXTRAPOLATED_UNVALIDATED"
    assert allowed["target_estimate_native"] == 1_700_000_000_090_004_517
    document["queries"][0]["max_extrapolation_ns"] -= 1
    assert estimate_clock_mapping(document)["queries"][0]["target_estimate_native"] is None


def test_uncertainty_increases_away_from_observed_center_under_ols_assumptions():
    document = correspondence_document()
    document["epochs"][0]["pairs"][2]["target_timestamp"] += 100
    document["queries"] = [
        query(4_000_040_000),
        query(4_000_100_000, query_id="far", max_extrapolation_ns=20_000_000),
    ]
    near, far = estimate_clock_mapping(document)["queries"]
    assert far["conditional_mean_fit_standard_error_ns"] > near["conditional_mean_fit_standard_error_ns"] > 0
    assert far["conditional_new_pair_prediction_standard_error_ns"] > far["conditional_mean_fit_standard_error_ns"]


def test_two_pairs_report_no_fit_and_no_false_zero_uncertainty():
    document = correspondence_document(count=2)
    document["queries"] = [query(4_000_000_001)]
    report = estimate_clock_mapping(document)
    assert report["epochs"][0]["status"] == "UNAVAILABLE_INSUFFICIENT_PAIRS"
    assert report["epochs"][0]["model"] is None
    assert report["queries"][0]["status"] == "UNAVAILABLE_NO_FIT"
    assert report["queries"][0]["target_estimate_native"] is None


@pytest.mark.parametrize("update,message", [
    ({"source_clock_domain": "host_epoch"}, "domain/unit mismatch"),
    ({"source_unit": "ns"}, "domain/unit mismatch"),
    ({"source_epoch_id": "different-boot"}, "epoch mismatch"),
    ({"epoch_id": "unknown"}, "unknown query epoch"),
    ({"max_extrapolation_ns": -1}, "must be >="),
])
def test_queries_require_the_exact_declared_clock_and_epoch(update, message):
    document = correspondence_document()
    document["queries"] = [query(4_000_000_001, **update)]
    with pytest.raises(ClockMappingError, match=message):
        estimate_clock_mapping(document)


@pytest.mark.parametrize("value", [True, 1.5, "123", None, 2**128])
def test_native_timestamps_are_exact_bounded_json_integers(value):
    document = correspondence_document()
    document["epochs"][0]["pairs"][0]["source_timestamp"] = value
    with pytest.raises(ClockMappingError, match="JSON integer"):
        estimate_clock_mapping(document)


@pytest.mark.parametrize("mutation", ["duplicate_pair", "duplicate_epoch", "duplicate_query", "zero_span"])
def test_ambiguous_identifiers_or_equal_clock_values_are_rejected(mutation):
    document = correspondence_document()
    if mutation == "duplicate_pair":
        document["epochs"][0]["pairs"][-1]["pair_id"] = "pair-0"
    elif mutation == "duplicate_epoch":
        document["epochs"].append(copy.deepcopy(document["epochs"][0]))
    elif mutation == "duplicate_query":
        document["queries"] = [query(4_000_000_001), query(4_000_000_002)]
    else:
        document["epochs"][0]["pairs"][1]["source_timestamp"] = 4_000_000_000
    with pytest.raises(ClockMappingError, match="duplicate|nonincreasing"):
        estimate_clock_mapping(document)


def test_unknown_units_unqualified_pairs_and_nonfinite_metadata_fail_closed():
    document = correspondence_document()
    document["source_clock"]["unit"] = "ticks"
    with pytest.raises(ClockMappingError, match="unit"):
        estimate_clock_mapping(document)
    document = correspondence_document()
    del document["epochs"][0]["correspondence_kind"]
    with pytest.raises(ClockMappingError, match="correspondence_kind"):
        estimate_clock_mapping(document)
    document = correspondence_document()
    document["metadata"] = {"corrupt": float("nan")}
    with pytest.raises(ClockMappingError, match="finite JSON"):
        estimate_clock_mapping(document)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "estimate_clock_mapping.py"


def test_cli_reuses_strict_json_reader_and_preserves_integer_native_times(tmp_path):
    source, output = tmp_path / "pairs.json", tmp_path / "mapping.json"
    source.write_text(json.dumps(correspondence_document()), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "--output", str(output)], check=False, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["epochs"][0]["pairs"][0]["target_timestamp"] == 1_700_000_000_000_000_017
    assert report["physical_alignment_established"] is False
    original = source.read_bytes()
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "--output", str(source)], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert source.read_bytes() == original


def test_cli_does_not_overwrite_an_input_alias(tmp_path, monkeypatch):
    source, output = tmp_path / "pairs.json", tmp_path / "alias.json"
    source.write_text(json.dumps(correspondence_document()), encoding="utf-8")
    output.write_bytes(source.read_bytes())
    # Simulate OS file identity without needing hardlink permissions on Windows.
    monkeypatch.setattr(Path, "samefile", lambda self, other: self == output and other == source)
    original = source.read_bytes()
    main = runpy.run_path(str(SCRIPT))["main"]
    with pytest.raises(SystemExit) as error:
        main([str(source), "--output", str(output)])
    assert error.value.code == 2
    assert source.read_bytes() == output.read_bytes() == original


@pytest.mark.parametrize("content", ['{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}', '{"a":', '[]'])
def test_cli_rejects_corrupt_json_without_producing_a_report(tmp_path, content):
    source, output = tmp_path / "pairs.json", tmp_path / "mapping.json"
    source.write_text(content, encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "--output", str(output)], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert "clock mapping contract error" in run.stderr
    assert not output.exists()


def davis_archive(tmp_path, observations):
    import numpy as np

    from grasp_project.recording import RecordingWriter

    root = tmp_path / "raw"
    writer = RecordingWriter(root, recording_id="synthetic-clock-archive", config={},
                             revision={"commit": "synthetic-fixture"}, metadata={"data_origin": "synthetic"})
    for index, values in enumerate(observations):
        timestamp = values.get("timestamp")
        native = np.empty((0, 4), dtype=np.uint64) if timestamp is None else np.asarray(
            [[timestamp - 1, 100, 100, 0], [timestamp, 101, 101, 1]],
            dtype=values.get("dtype", np.uint64),
        )
        if values.get("within_packet_reset"):
            native = native[::-1]
        metadata = {"acquisition_endpoint": "device_get_event_return", "poll_state": "packet_returned"}
        if values.get("epoch", 1) is not None:
            metadata["normalization_epoch"] = values.get("epoch", 1)
        clock = {"domain": values.get("domain", "davis_device"), "unit": values.get("unit", "us")}
        if "native_epoch" in values:
            clock["epoch_id"] = values["native_epoch"]
        writer.record("davis_poll", {"events": native}, observation_id=f"poll-{index}",
                      host_arrival_monotonic_ns=10**18 + index * 1_000,
                      host_arrival_epoch_ns=1_700_000_000_000_000_017 + index * 1_000,
                      native_clock=clock, metadata=metadata, source_sequence=index)
    writer.close()
    return root


def test_verified_archive_mode_preserves_large_integer_native_times_and_manifest_provenance(tmp_path):
    origin = 2**63 + 7
    root = davis_archive(tmp_path, [{"timestamp": origin + index} for index in range(5)])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    extracted = extract_recording_receipt_correspondences(root, "davis_poll")
    assert extracted["correspondences"]["epochs"][0]["correspondence_kind"] == "host_receipt_association"
    report = estimate_recording_clock_mapping(root, "davis_poll")
    assert report["status"] == "ESTIMATED_UNVALIDATED"
    assert report["extraction"]["used_pair_count"] == 5
    assert report["epochs"][0]["interpretation"] == "HOST_RECEIPT_ASSOCIATION_ONLY"
    assert report["epochs"][0]["pairs"][0]["source_timestamp"] == origin
    assert report["epochs"][0]["pairs"][0]["target_timestamp"] == 10**18
    assert report["epochs"][0]["model"]["rate_ratio"] == 1.0
    assert report["epochs"][0]["provenance"]["recorded_epoch_key"] == [1, None]
    assert report["archive"]["manifest_sha256"] == json.loads((root / "manifest.json").read_text())["manifest_sha256"]
    assert report["physical_alignment_established"] is report["applied_to_live_policy"] is False
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_empty_polls_are_counted_as_unavailable_not_fake_zero_correspondences(tmp_path):
    root = davis_archive(tmp_path, [{"timestamp": None}, {"timestamp": 100}, {"timestamp": None}, {"timestamp": 101}, {"timestamp": 102}])
    report = estimate_recording_clock_mapping(root, "davis_poll")
    assert report["extraction"]["selected_observation_count"] == 5
    assert report["extraction"]["used_pair_count"] == 3
    assert report["extraction"]["unavailable_reasons"] == {"empty_poll_no_native_event_timestamp": 2}
    assert [row["source_timestamp"] for row in report["epochs"][0]["pairs"]] == [100, 101, 102]
    assert report["extraction"]["observations"][0]["status"] == "UNAVAILABLE"


def test_archive_epochs_stay_separate_and_native_reset_is_not_automatically_unwrapped(tmp_path):
    root = davis_archive(tmp_path, [
        {"timestamp": 100}, {"timestamp": 101}, {"timestamp": 102},
        {"timestamp": 10}, {"timestamp": 11},
        {"timestamp": 10, "epoch": 2}, {"timestamp": 11, "epoch": 2}, {"timestamp": 12, "epoch": 2},
    ])
    report = estimate_recording_clock_mapping(root, "davis_poll")
    assert [epoch["pair_count"] for epoch in report["epochs"]] == [3, 3]
    assert report["extraction"]["unavailable_reasons"] == {
        "native_reset_without_recorded_epoch_change": 1,
        "clock_continuity_unknown_until_new_recorded_epoch": 1,
    }
    assert report["epochs"][0]["model"]["source_range_native"] == [100, 102]
    assert report["epochs"][1]["model"]["source_range_native"] == [10, 12]


@pytest.mark.parametrize("bad,reason", [
    ({"epoch": None}, "normalization_epoch_not_recorded"),
    ({"domain": "host_epoch"}, "invalid_davis_native_clock_domain_or_unit"),
    ({"unit": "ns"}, "invalid_davis_native_clock_domain_or_unit"),
    ({"dtype": "float64"}, "native_event_timestamps_not_recorded_as_integers"),
    ({"native_epoch": {}}, "native_clock_epoch_id_invalid"),
    ({"within_packet_reset": True}, "native_reset_within_packet"),
])
def test_unsupported_or_unknown_archive_native_clocks_are_explicitly_unavailable(tmp_path, bad, reason):
    root = davis_archive(tmp_path, [{"timestamp": 100, **bad}])
    report = estimate_recording_clock_mapping(root, "davis_poll")
    assert report["status"] == "UNAVAILABLE_NO_USABLE_ASSOCIATIONS"
    assert report["epochs"] == []
    assert report["extraction"]["unavailable_reasons"] == {reason: 1}


def test_unknown_epoch_observation_cannot_be_bridged_back_into_existing_fit(tmp_path):
    root = davis_archive(tmp_path, [
        {"timestamp": 100}, {"timestamp": 101, "epoch": None}, {"timestamp": 102},
        {"timestamp": 103}, {"timestamp": 104, "epoch": 2},
    ])
    report = estimate_recording_clock_mapping(root, "davis_poll")
    assert report["status"] == "UNAVAILABLE_INSUFFICIENT_PAIRS"
    assert [epoch["pair_count"] for epoch in report["epochs"]] == [1, 1]
    assert report["extraction"]["unavailable_reasons"]["clock_continuity_unknown_until_new_recorded_epoch"] == 2


@pytest.mark.parametrize("stream", ["rgb", "depth", "camera_info"])
def test_current_ros_archives_do_not_invent_native_domain_or_epoch(tmp_path, stream):
    from grasp_project.recording import RecordingWriter

    root = tmp_path / "ros"
    writer = RecordingWriter(root, recording_id="synthetic-ros", config={}, revision={}, metadata={})
    for index in range(3):
        writer.record(stream, {}, observation_id=f"ros-{index}", host_arrival_monotonic_ns=100 + index,
                      host_arrival_epoch_ns=1_700_000_000_000_000_000 + index,
                      native_clock={"domain": "ros_header_unspecified", "unit": "ns", "timestamp": 1_700_000_003_123_456_789 + index},
                      metadata={"header_stamp_sec": 1_700_000_003, "header_stamp_nanosec": 123_456_789 + index})
    writer.close()
    report = estimate_recording_clock_mapping(root, stream)
    assert report["status"] == "UNAVAILABLE_NO_USABLE_ASSOCIATIONS"
    assert report["extraction"]["unavailable_reasons"] == {"ros_native_clock_domain_unspecified": 3}
    assert report["extraction"]["observations"][0]["header_stamp_sec"] == 1_700_000_003
    assert report["extraction"]["observations"][0]["header_stamp_nanosec"] == 123_456_789
    assert report["epochs"] == []


def test_explicit_documented_ros_native_epoch_uses_exact_header_integer_arithmetic(tmp_path):
    from grasp_project.recording import RecordingWriter

    root = tmp_path / "ros"
    writer = RecordingWriter(root, recording_id="synthetic-documented-ros", config={}, revision={}, metadata={})
    for index in range(3):
        writer.record("rgb", {}, observation_id=f"ros-{index}", host_arrival_monotonic_ns=10**18 + index,
                      host_arrival_epoch_ns=1_700_000_000_000_000_000 + index,
                      native_clock={"domain": "synthetic_documented_ros_clock", "unit": "ns", "epoch_id": "native-epoch-0", "timestamp": 1_700_000_003_123_456_789 + index},
                      metadata={"header_stamp_sec": 1_700_000_003, "header_stamp_nanosec": 123_456_789 + index})
    writer.close()
    report = estimate_recording_clock_mapping(root, "rgb")
    assert report["status"] == "ESTIMATED_UNVALIDATED"
    assert report["epochs"][0]["pairs"][0]["source_timestamp"] == 1_700_000_003_123_456_789
    assert report["epochs"][0]["pairs"][0]["native_header_stamp"] == {"sec": 1_700_000_003, "nanosec": 123_456_789}
    assert report["epochs"][0]["pairs"][0]["acquisition_endpoint"] is None
    assert report["epochs"][0]["pairs"][0]["receipt_endpoint_status"] == "NOT_RECORDED"


def test_archive_mode_cli_and_input_archive_write_guards(tmp_path):
    root = davis_archive(tmp_path, [{"timestamp": 100 + index} for index in range(3)])
    output = tmp_path / "diagnostics.json"
    arguments = [sys.executable, str(SCRIPT), "--recording", str(root), "--stream", "davis_poll", "--output"]
    run = subprocess.run([*arguments, str(output)], check=False, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["extraction"]["used_pair_count"] == 3
    manifest = (root / "manifest.json").read_bytes()
    for unsafe in (root / "manifest.json", root / "nested" / "diagnostics.json"):
        run = subprocess.run([*arguments, str(unsafe)], check=False, capture_output=True, text=True)
        assert run.returncode == 2
        assert "input archive" in run.stderr
    assert (root / "manifest.json").read_bytes() == manifest
    assert not (root / "nested").exists()


def test_archive_mode_verifies_all_raw_integrity_before_producing_a_report(tmp_path):
    root = davis_archive(tmp_path, [{"timestamp": 100 + index} for index in range(3)])
    payload = next((root / "arrays").glob("*.npy"))
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    output = tmp_path / "diagnostics.json"
    run = subprocess.run([sys.executable, str(SCRIPT), "--recording", str(root), "--stream", "davis_poll", "--output", str(output)], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert "checksum mismatch" in run.stderr
    assert not output.exists()


@pytest.mark.parametrize("arguments", [[], ["file.json", "--stream", "rgb"], ["--recording", "raw"], ["file.json", "--recording", "raw", "--stream", "rgb"]])
def test_cli_recording_and_correspondence_sources_are_mutually_exclusive(arguments):
    run = subprocess.run([sys.executable, str(SCRIPT), *arguments], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert "error:" in run.stderr

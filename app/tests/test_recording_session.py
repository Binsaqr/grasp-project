"""Synthetic recording contract checks, never physical sensor validation."""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from grasp_project.recording import RecordingWriter, read_recording


def writer_at(tmp_path, **overrides):
    values = {"recording_id": "trial-1", "config": {"threshold": 12, "required": True},
              "revision": {"commit": "synthetic-fixture", "dirty": False},
              "metadata": {"trial_id": "trial-1", "fixture": "synthetic"}}
    values.update(overrides)
    return RecordingWriter(tmp_path / "raw", **values)


def record(writer, **overrides):
    values = {"kind": "events", "arrays": {"timestamps": np.array([2**53 + 1, 2**53 + 4], dtype=np.uint64),
                                           "xy": np.array([[12, 14], [13, 15]], dtype=np.uint16)},
              "observation_id": "events-1", "host_arrival_monotonic_ns": 100,
              "host_arrival_epoch_ns": 1_800_000_000_000_000_000,
              "native_clock": {"domain": "davis_device", "unit": "us", "epoch_known": False},
              "metadata": {"source": "synthetic"}, "source_sequence": 4}
    values.update(overrides)
    return writer.record(**values)


def completed(tmp_path):
    writer = writer_at(tmp_path)
    record(writer)
    writer.close()
    return tmp_path / "raw"


def resign(root, change):
    path = root / "manifest.json"
    payload = json.loads(path.read_text())
    change(payload)
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_roundtrip_preserves_native_timestamp_precision_payload_and_snapshots(tmp_path):
    config = {"threshold": 12, "nested": {"enabled": True}}
    revision = {"commit": "synthetic-fixture", "dirty": False}
    writer = writer_at(tmp_path, config=config, revision=revision)
    times = np.array([2**53 + 1, 2**53 + 4], dtype=np.uint64)
    descriptor = record(writer, arrays={"timestamps": times})
    config["nested"]["enabled"] = False
    revision["dirty"] = True
    times[:] = 0
    descriptor["native_clock"]["domain"] = "modified_by_caller"
    writer.close()

    recording = read_recording(tmp_path / "raw")
    saved = recording.records[0]
    assert saved["arrays"]["timestamps"].tolist() == [2**53 + 1, 2**53 + 4]
    assert saved["arrays"]["timestamps"].dtype == np.uint64
    assert saved["native_clock"] == {"domain": "davis_device", "unit": "us", "epoch_known": False}
    assert saved["host_arrival_monotonic_ns"] == 100
    assert saved["host_arrival_epoch_ns"] == 1_800_000_000_000_000_000
    assert saved["recording_id"] == saved["trial_id"] == "trial-1"
    assert recording.manifest["config"] == {"threshold": 12, "nested": {"enabled": True}}
    assert recording.manifest["revision"]["dirty"] is False
    assert recording.manifest["clock_alignment"] == "not_estimated"
    assert recording.manifest["physical_validation"] is False
    assert saved["arrays"]["timestamps"].flags.writeable is False


def test_drop_accounting_preserves_unknown_counts_and_sequence_gaps_separately(tmp_path):
    writer = writer_at(tmp_path)
    record(writer, source_sequence=4)
    writer.record_drop("events", 2, "source_overflow_report", 101)
    writer.record_drop("events", None, "source_drop_count_unavailable", 102)
    record(writer, observation_id="events-2", source_sequence=7, host_arrival_monotonic_ns=103)
    writer.close()
    recording = read_recording(tmp_path / "raw")
    assert recording.manifest["summary"] == {
        "record_count": 4, "observation_count": 2, "drop_record_count": 2,
        "known_dropped_observations": 2, "unknown_drop_events": 1, "source_sequence_gaps": 2,
    }
    assert recording.records[2]["count"] is None
    assert recording.records[3]["source_sequence_gap"] == 2


def test_empty_archive_and_metadata_only_observation_do_not_invent_measurements(tmp_path):
    writer = writer_at(tmp_path)
    record(writer, kind="phase", arrays={}, source_sequence=None,
           native_clock={"domain": "operator_host", "unit": "ns"}, metadata={"phase": "holding"})
    writer.close()
    recording = read_recording(tmp_path / "raw")
    assert recording.records[0]["arrays"] == {}
    assert recording.manifest["summary"]["known_dropped_observations"] == 0
    assert recording.manifest["physical_validation"] is False


def test_complete_empty_archive_has_zero_observations(tmp_path):
    writer = writer_at(tmp_path)
    writer.close()
    assert read_recording(tmp_path / "raw").manifest["summary"]["observation_count"] == 0


def test_incomplete_archives_rejected_by_default_and_forensics_still_checks_arrays(tmp_path):
    writer = writer_at(tmp_path)
    descriptor = record(writer)
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(tmp_path / "raw")
    assert read_recording(tmp_path / "raw", require_complete=False).manifest["status"] == "incomplete"
    (tmp_path / "raw" / descriptor["arrays"]["xy"]["path"]).unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        read_recording(tmp_path / "raw", require_complete=False)


def test_existing_directory_not_overwritten_and_close_is_idempotent(tmp_path):
    writer = writer_at(tmp_path)
    record(writer)
    first = writer.close()
    assert writer.close() == first
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        writer_at(tmp_path)
    assert first.read_bytes() == before
    with pytest.raises(ValueError, match="closed"):
        record(writer, observation_id="events-2")
    with pytest.raises(ValueError, match="closed"):
        writer.record_drop("events", 1, "overflow", 110)


def test_write_failure_cannot_be_finalized(tmp_path):
    writer = writer_at(tmp_path)
    with (patch("grasp_project.recording.session.np.save", side_effect=OSError("disk failed")),
          pytest.raises(OSError, match="disk failed")):
        record(writer)
    with pytest.raises(ValueError, match="previous write failed"):
        writer.close()
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(tmp_path / "raw")


def test_global_order_does_not_assert_cross_clock_or_cross_stream_alignment(tmp_path):
    writer = writer_at(tmp_path)
    record(writer, kind="events", host_arrival_monotonic_ns=100)
    record(writer, kind="depth", observation_id="depth-1", host_arrival_monotonic_ns=80)
    writer.close()
    recording = read_recording(tmp_path / "raw")
    assert [r["kind"] for r in recording.records] == ["events", "depth"]
    assert [r["host_arrival_monotonic_ns"] for r in recording.records] == [100, 80]


@pytest.mark.parametrize("overrides", [
    {"observation_id": "../escape"}, {"kind": "events/escape"},
    {"host_arrival_monotonic_ns": True}, {"host_arrival_monotonic_ns": -1},
    {"host_arrival_epoch_ns": 1.5}, {"source_sequence": True}, {"source_sequence": -1},
    {"native_clock": {"domain": "device"}}, {"native_clock": {"domain": "", "unit": "us"}},
    {"metadata": {"gain": float("nan")}}, {"metadata": {"gain": float("inf")}},
    {"metadata": {1: "integer key"}}, {"arrays": {"../escape": np.zeros(1)}},
    {"arrays": {"object": np.array([{"bad": 1}], dtype=object)}},
    {"arrays": {"text": np.array(["unsupported"])}}, {"arrays": {"not_ndarray": [1, 2]}},
])
def test_invalid_observation_rejected_before_writes(tmp_path, overrides):
    writer = writer_at(tmp_path)
    with pytest.raises(ValueError):
        record(writer, **overrides)
    assert list((tmp_path / "raw" / "arrays").iterdir()) == []
    record(writer)
    writer.close()
    assert len(read_recording(tmp_path / "raw").records) == 1


@pytest.mark.parametrize("overrides", [
    {"observation_id": "events-1"},
    {"observation_id": "events-2", "source_sequence": 4},
    {"observation_id": "events-2", "source_sequence": 3},
    {"observation_id": "events-2", "source_sequence": 5, "host_arrival_monotonic_ns": 99},
])
def test_duplicate_or_decreasing_order_rejected(tmp_path, overrides):
    writer = writer_at(tmp_path)
    record(writer)
    with pytest.raises(ValueError):
        record(writer, **overrides)


@pytest.mark.parametrize("count,reason,arrival", [(0, "overflow", 100), (-1, "overflow", 100),
                                                (True, "overflow", 100), (1, "", 100),
                                                (None, "unknown", -1)])
def test_invalid_drop_rejected(tmp_path, count, reason, arrival):
    writer = writer_at(tmp_path)
    with pytest.raises(ValueError):
        writer.record_drop("events", count, reason, arrival)


@pytest.mark.parametrize("field", ["config", "revision", "metadata"])
def test_snapshot_hash_detects_changed_contents_even_if_manifest_resigned(tmp_path, field):
    root = completed(tmp_path)
    resign(root, lambda manifest: manifest[field].update({"modified": True}))
    with pytest.raises(ValueError, match=f"{field} snapshot checksum"):
        read_recording(root)


def test_manifest_checksum_detects_changes(tmp_path):
    root = completed(tmp_path)
    path = root / "manifest.json"
    data = json.loads(path.read_text())
    data["records"][0]["metadata"]["source"] = "modified"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest checksum"):
        read_recording(root)


@pytest.mark.parametrize("mutation", ["missing", "truncated", "same_size_modified"])
def test_every_array_checked_before_return_without_ground_truth(tmp_path, mutation):
    root = completed(tmp_path)
    path = root / "arrays" / "00000000-0000.npy"
    if mutation == "missing":
        path.unlink()
    elif mutation == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        contents = bytearray(path.read_bytes())
        contents[-1] ^= 1
        path.write_bytes(contents)
    with pytest.raises(ValueError):
        read_recording(root)


@pytest.mark.parametrize("relative", ["../outside.npy", "/absolute.npy", "arrays/../outside.npy",
                                       "arrays\\outside.npy", "C:/outside.npy", "arrays:outside.npy"])
def test_unsafe_array_paths_rejected_even_with_resigned_manifest(tmp_path, relative):
    root = completed(tmp_path)
    resign(root, lambda manifest: manifest["records"][0]["arrays"]["xy"].update({"path": relative}))
    with pytest.raises(ValueError):
        read_recording(root)


@pytest.mark.parametrize("change", [
    lambda m: m.update(schema_version="unknown"),
    lambda m: m.update(status="something_else"),
    lambda m: m.update(clock_alignment="synchronized"),
    lambda m: m.update(physical_validation=True),
    lambda m: m["records"][0].update(sequence=4),
    lambda m: m["records"][0].update(sequence=False),
    lambda m: m["records"][0].update(trial_id="wrong-trial"),
    lambda m: m["records"][0].update(source_sequence_gap=3),
    lambda m: m["records"][0].update(host_arrival_epoch_ns=None),
    lambda m: m["records"][0]["arrays"]["xy"].update(dtype="<u8"),
    lambda m: m["records"][0]["arrays"]["xy"].update(shape=[4]),
    lambda m: m["records"][0]["arrays"]["xy"].update(shape=[True]),
    lambda m: m["summary"].update(observation_count=2),
    lambda m: m["summary"].update(observation_count=True),
])
def test_schema_and_consistency_errors_rejected_after_valid_manifest_hash(tmp_path, change):
    root = completed(tmp_path)
    resign(root, change)
    with pytest.raises(ValueError):
        read_recording(root)


def test_duplicate_observation_identity_detected_on_read(tmp_path):
    writer = writer_at(tmp_path)
    record(writer)
    record(writer, observation_id="events-2", source_sequence=5, host_arrival_monotonic_ns=101)
    writer.close()
    root = tmp_path / "raw"
    resign(root, lambda m: m["records"][1].update(observation_id="events-1"))
    with pytest.raises(ValueError, match="duplicate observation_id"):
        read_recording(root)


def test_duplicate_json_keys_nonfinite_numbers_and_nesting_rejected(tmp_path):
    root = completed(tmp_path)
    path = root / "manifest.json"
    original = path.read_text()
    path.write_text(original.replace('"config": {', '"config": {}, "config": {', 1))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_recording(root)
    path.write_text(original.replace('"created_epoch_ns":', '"nonfinite": NaN, "created_epoch_ns":', 1))
    with pytest.raises(ValueError, match="finite"):
        read_recording(root)
    nested = current = {}
    for _ in range(65):
        current["child"] = {}
        current = current["child"]
    with pytest.raises(ValueError, match="nesting"):
        writer_at(tmp_path / "nested", config=nested)


def test_numeric_payload_nan_and_noncontiguous_array_preserved_losslessly(tmp_path):
    writer = writer_at(tmp_path)
    original = np.array([[np.nan, 1.0], [np.inf, -0.0]], dtype=np.float32)[:, ::-1]
    record(writer, arrays={"depth": original})
    writer.close()
    loaded = read_recording(tmp_path / "raw").records[0]["arrays"]["depth"]
    assert loaded.dtype == original.dtype
    np.testing.assert_array_equal(loaded, original)
    assert loaded.tobytes() == original.tobytes()


def test_long_valid_array_identifier_roundtrips(tmp_path):
    writer = writer_at(tmp_path)
    name = "x" * 120
    record(writer, arrays={name: np.zeros(1)})
    writer.close()
    assert name in read_recording(tmp_path / "raw").records[0]["arrays"]


def test_semantic_array_names_do_not_collide_on_case_insensitive_filesystems(tmp_path):
    writer = writer_at(tmp_path)
    record(writer, arrays={"events": np.array([1]), "EVENTS": np.array([2])})
    writer.close()
    arrays = read_recording(tmp_path / "raw").records[0]["arrays"]
    np.testing.assert_array_equal(arrays["events"], [1])
    np.testing.assert_array_equal(arrays["EVENTS"], [2])


@pytest.mark.parametrize("payload", ["pickle", "trailing"])
def test_resigned_payload_cannot_enable_pickle_or_hide_trailing_bytes(tmp_path, payload):
    root = completed(tmp_path)
    path = root / "arrays" / "00000000-0000.npy"
    if payload == "pickle":
        with path.open("wb") as handle:
            np.save(handle, np.array([{"not": "numeric"}], dtype=object), allow_pickle=True)
    else:
        path.write_bytes(path.read_bytes() + b"unexpected trailing payload")

    def change(manifest):
        descriptor = manifest["records"][0]["arrays"]["xy"]
        descriptor.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    resign(root, change)
    with pytest.raises(ValueError):
        read_recording(root)


@pytest.mark.parametrize("change", [
    lambda m: m["records"][1].update(host_arrival_monotonic_ns=99),
    lambda m: m["records"][1].update(source_sequence=4),
    lambda m: m["records"][1].update(source_sequence=3),
    lambda m: m["records"].reverse(),
])
def test_reader_rechecks_sequence_and_monotonic_order(tmp_path, change):
    writer = writer_at(tmp_path)
    record(writer)
    record(writer, observation_id="events-2", source_sequence=5, host_arrival_monotonic_ns=101)
    writer.close()
    root = tmp_path / "raw"
    resign(root, change)
    with pytest.raises(ValueError):
        read_recording(root)


def test_integer_metadata_native_timestamp_is_retained_without_float_conversion(tmp_path):
    writer = writer_at(tmp_path)
    record(writer, native_clock={"domain": "device", "unit": "ns", "timestamp": 2**63 + 1})
    writer.close()
    value = read_recording(tmp_path / "raw").records[0]["native_clock"]["timestamp"]
    assert type(value) is int
    assert value == 2**63 + 1


def test_unknown_source_sequences_do_not_imply_zero_physical_loss(tmp_path):
    writer = writer_at(tmp_path)
    record(writer, source_sequence=None)
    writer.record_drop("events", None, "source_drop_count_unavailable", 101)
    record(writer, source_sequence=None, observation_id="events-2", host_arrival_monotonic_ns=102)
    writer.close()
    recording = read_recording(tmp_path / "raw")
    assert recording.manifest["summary"]["known_dropped_observations"] == 0
    assert recording.manifest["summary"]["unknown_drop_events"] == 1
    assert recording.records[0]["source_sequence"] is None


def test_drop_and_observation_share_per_stream_arrival_order(tmp_path):
    writer = writer_at(tmp_path)
    record(writer)
    with pytest.raises(ValueError, match="decreased"):
        writer.record_drop("events", None, "unknown", 99)
    writer.record_drop("events", None, "unknown", 105)
    with pytest.raises(ValueError, match="decreased"):
        record(writer, observation_id="events-2", source_sequence=5, host_arrival_monotonic_ns=104)

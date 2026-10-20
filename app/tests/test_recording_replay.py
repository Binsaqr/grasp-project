"""Synthetic recording/inference parity tests, never physical measurements."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_project.davis import event_stream
from grasp_project.davis.event_stream import EventPostGraspMonitor
from grasp_project.davis.motion_estimation import DavisMotionEstimator
from grasp_project.davis.motion_gui import select_event_analysis_window
from grasp_project.quality.live_quality_monitor import (
    BaselineGraspQuality,
    LiveGraspObservation,
    LiveGraspQualityMonitor,
)
from grasp_project.recording.replay import ABLATIONS, replay_recording
from grasp_project.recording.session import RecordingWriter, read_recording

WINDOW = {
    "primary_window_sec": 0.1,
    "fallback_window_sec": 0.2,
    "min_primary_events": 50,
}


def test_replay_cli_writes_a_new_report_without_touching_or_overwriting_inputs(tmp_path):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0)
    writer.close()
    manifest = (writer.root / "manifest.json").read_bytes()
    output = tmp_path / "replay.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_recording.py"
    command = [sys.executable, str(script), str(writer.root), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["physical_validation"] is False
    assert report["rgbd_trace"]["verified_fsm"] is False
    contents = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == contents
    inside = writer.root / "new-report.json"
    result = subprocess.run(command[:-1] + [str(inside)], capture_output=True)
    assert result.returncode != 0
    assert not inside.exists()
    assert (writer.root / "manifest.json").read_bytes() == manifest


def writer_at(path):
    return RecordingWriter(
        path,
        recording_id="synthetic",
        config={"davis_motion": {}, "event_window_selection": WINDOW},
        revision={"commit": "fixture"},
        metadata={"data_origin": "synthetic", "trial_id": "trial-1"},
    )


def packet(index):
    rng = np.random.default_rng(5)
    count = 120
    return np.column_stack(
        [
            np.linspace(index * 100000, (index + 1) * 100000 - 1, count),
            np.linspace(-16, 16, count) + 100 + index * 5,
            rng.normal(100, 1, count),
            np.arange(count) % 2,
        ]
    )


def record_packet(writer, index, events=None, **metadata):
    writer.record(
        "davis_poll",
        {"events": packet(index) if events is None else events},
        observation_id=f"packet-{index}",
        host_arrival_monotonic_ns=1_100_000_000 + index * 100_000_000,
        host_arrival_epoch_ns=10_100_000_000 + index * 100_000_000,
        native_clock={"domain": "davis_device", "unit": "us"},
        metadata={"roi_xywh": [40, 40, 160, 160], "normalization_epoch": 0, **metadata},
        source_sequence=index,
    )


def test_native_hook_preserves_integer_clocks_and_payload_before_conversion(
    tmp_path, monkeypatch
):
    writer = writer_at(tmp_path / "recording")
    raw = np.array([[2**53 + 1, 10, 20, 1], [2**53 + 3, 11, 21, 0]], dtype=np.int64)
    monitor = object.__new__(EventPostGraspMonitor)
    monitor.active = True
    monitor.device = SimpleNamespace(get_event=lambda _: [raw.copy()])
    monitor.recording_sink = writer
    monitor.recording_poll_sequence = 0
    monitor.recording_stream_id = "stream"
    monitor.roi_xywh = [0, 0, 346, 260]
    monkeypatch.setattr(event_stream.time, "monotonic", lambda: 50.0)
    monkeypatch.setattr(event_stream.time, "monotonic_ns", lambda: 50_000_000_123)
    monkeypatch.setattr(event_stream.time, "time_ns", lambda: 500_000_000_456)
    normalized, _ = monitor.poll_event_and_frame_arrays()
    writer.close()
    recording = read_recording(writer.root)
    descriptor = recording.records[1]
    np.testing.assert_array_equal(descriptor["arrays"]["events"], raw)
    assert descriptor["host_arrival_monotonic_ns"] == 50_000_000_123
    assert descriptor["host_arrival_epoch_ns"] == 500_000_000_456
    assert descriptor["metadata"]["acquisition_endpoint"] == "device_get_event_return"
    assert descriptor["metadata"]["normalization_reference_monotonic_sec"] == 50.0
    assert normalized[-1, 0] == 50
    assert descriptor["arrays"]["events"].dtype == np.int64
    assert recording.manifest["summary"]["unknown_drop_events"] == 1


def test_replay_is_deterministic_and_matches_canonical_estimator(tmp_path):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0)
    record_packet(writer, 1)
    writer.close()
    replay = replay_recording(writer.root)
    assert replay == replay_recording(writer.root)
    mapper = object.__new__(EventPostGraspMonitor)
    events = np.vstack(
        [
            mapper.normalize_native_event_array(packet(0), 1.1),
            mapper.normalize_native_event_array(packet(1), 1.2),
        ]
    )
    _, duration, _, end = select_event_analysis_window(
        events, 1.2, [40, 40, 160, 160], **WINDOW
    )
    expected = (
        DavisMotionEstimator()
        .estimate(
            events,
            now=1.2,
            analysis_end=end,
            motion_window_ms=duration * 1000,
            roi_xywh=[40, 40, 160, 160],
            roi_source="recorded_roi",
        )
        .to_dict()
    )
    actual = replay["observations"][-1]["raw_estimate"]
    for key in expected:
        if key not in {"estimate_id", "host_timestamp"}:
            assert actual[key] == expected[key], key
    assert replay["observations"][-1]["prediction"]["valid"] is True
    assert replay["physical_validation"] is False
    assert replay["ground_truth_generated"] is False


@pytest.mark.parametrize(
    "count,reason",
    [
        (1, "local_packet_drop"),
        (None, "device_poll_failed"),
        (None, "unknown_size_packet_loss"),
        (1, "backend_packet_loss_not_observable"),
    ],
)
def test_actual_loss_breaks_continuity_even_when_count_is_unknown(
    tmp_path, count, reason
):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0)
    writer.record_drop("davis_events", count, reason, 1_150_000_000)
    record_packet(writer, 1)
    writer.close()
    row = replay_recording(writer.root)["observations"][-1]
    assert row["prediction"]["valid"] is False
    assert row["prediction"]["label"] is None
    assert "recorded_observation_gap" in row["prediction"]["reasons"]


def test_accounting_notice_is_distinct_from_an_observed_loss(tmp_path):
    writer = writer_at(tmp_path / "recording")
    writer.record_drop(
        "davis_events", None, "backend_packet_loss_not_observable", 1_000_000_000
    )
    record_packet(writer, 0)
    record_packet(writer, 1)
    writer.close()
    replay = replay_recording(writer.root)
    assert replay["observations"][-1]["prediction"]["valid"] is True
    assert replay["drop_accounting"]["unknown_drop_events"] == 1


def test_empty_packet_has_no_measured_stability_label(tmp_path):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0, np.empty((0, 4), dtype=np.int64))
    writer.close()
    row = replay_recording(writer.root)["observations"][0]
    assert (
        row["raw_estimate"]["motion_class"] == "STABLE"
    )  # Existing diagnostic retained honestly.
    assert row["prediction"] == {
        "label": None,
        "score": None,
        "decision": None,
        "valid": False,
        "reasons": ["no_events"],
    }


def test_native_clock_reset_cannot_bridge_motion_history(tmp_path):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0)
    record_packet(writer, 1, packet(0))
    record_packet(writer, 2, packet(0) * [0.5, 1, 1, 1])
    writer.close()
    row = replay_recording(writer.root)["observations"][-1]
    assert row["prediction"]["valid"] is False
    assert "recorded_observation_gap" in row["prediction"]["reasons"]


def test_recorded_restart_epoch_replays_native_mapping_reanchor(tmp_path):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0)
    restarted = packet(1) + [10_000_000, 0, 0, 0]
    record_packet(writer, 1, restarted, normalization_epoch=1)
    writer.close()
    actual = replay_recording(writer.root)["observations"][-1]["raw_estimate"]
    mapper = object.__new__(EventPostGraspMonitor)
    normalized = mapper.normalize_native_event_array(restarted, 1.2)
    expected = (
        DavisMotionEstimator()
        .estimate(
            normalized,
            now=1.2,
            analysis_end=1.2,
            motion_window_ms=100,
            roi_xywh=[40, 40, 160, 160],
            roi_source="recorded_roi",
        )
        .to_dict()
    )
    assert {
        key: value
        for key, value in actual.items()
        if key not in {"estimate_id", "host_timestamp"}
    } == {
        key: value
        for key, value in expected.items()
        if key not in {"estimate_id", "host_timestamp"}
    }


def test_estimator_long_lookback_retains_all_required_recorded_history(tmp_path):
    writer = RecordingWriter(
        tmp_path / "recording",
        recording_id="long",
        config={
            "davis_motion": {"stability_window_ms": 1000},
            "event_window_selection": WINDOW,
        },
        revision={},
        metadata={"data_origin": "synthetic"},
    )
    mapper = object.__new__(EventPostGraspMonitor)
    estimator = DavisMotionEstimator({"stability_window_ms": 1000})
    buffer = []
    for index in range(15):
        events = packet(index)
        record_packet(writer, index, events)
        end = (1_100_000_000 + index * 100_000_000) / 1e9
        buffer.extend(mapper.normalize_native_event_array(events, end))
        expected = estimator.estimate(
            np.asarray(buffer),
            now=end,
            analysis_end=end,
            motion_window_ms=100,
            roi_xywh=[40, 40, 160, 160],
            roi_source="recorded_roi",
        )
    writer.close()
    actual = replay_recording(writer.root)["observations"][-1]["raw_estimate"]
    assert actual["vibration_amplitude_px_rms"] == expected.vibration_amplitude_px_rms


def test_replay_refuses_corrupted_payload_and_missing_frozen_windows(tmp_path):
    writer = writer_at(tmp_path / "recording")
    record_packet(writer, 0)
    writer.close()
    next((writer.root / "arrays").iterdir()).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        replay_recording(writer.root)
    other = RecordingWriter(
        tmp_path / "other", recording_id="other", config={}, revision={}, metadata={}
    )
    record_packet(other, 0)
    other.close()
    with pytest.raises(ValueError, match="adaptive window"):
        replay_recording(other.root)


def quality_observation(**changes):
    values = {
        "timestamp": 10,
        "target_status": "TRACKING",
        "ready_to_close": True,
        "detected_contacts": 2,
        "visual_contact_ready": True,
        "event_state": "STREAM_ACTIVE_NO_MOTION",
        "davis_event_freshness": "CURRENT",
        "davis_events_valid_for_live_quality": True,
        "davis_motion_usable_for_grasp": True,
        "davis_object_motion_valid": True,
        "davis_report_heartbeat_current": True,
        "davis_motion_confidence": 0.9,
        "incipient_slip_score": 80,
        "davis_motion_class": "INCIPIENT_SLIP",
        "post_grasp_active": True,
        "grasp_stage": "POST_GRASP_MONITORING",
    }
    values.update(changes)
    return LiveGraspObservation(**values)


def test_recorded_quality_inputs_replay_actual_stateful_monitor_exactly(tmp_path):
    writer = writer_at(tmp_path / "recording")
    monitor = LiveGraspQualityMonitor(recording_sink=writer)
    baseline = BaselineGraspQuality(
        valid=True, analytic_score=84, analytic_label="GOOD"
    )
    expected = [
        monitor.update(
            baseline,
            quality_observation(timestamp=10 + index, incipient_slip_score=score),
        ).to_dict()
        for index, score in enumerate([80, 90, 0])
    ]
    writer.close()
    replay = replay_recording(writer.root)
    assert [row["raw_quality"] for row in replay["observations"]] == expected
    assert replay == replay_recording(writer.root)


def test_quality_replay_cannot_silently_bridge_lost_stateful_inputs(tmp_path):
    writer = writer_at(tmp_path / "recording")
    monitor = LiveGraspQualityMonitor(recording_sink=writer)
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    monitor.update(base, quality_observation())
    last_arrival = read_recording(writer.root, require_complete=False).records[-1][
        "host_arrival_monotonic_ns"
    ]
    writer.record_drop("live_quality_inputs", None, "input_write_missing", last_arrival)
    monitor.update(base, quality_observation(timestamp=11))
    monitor.update(base, quality_observation(timestamp=12))
    monitor.reset()
    expected_after_reset = monitor.update(
        base, quality_observation(timestamp=13)
    ).to_dict()
    writer.close()
    replayed = replay_recording(writer.root)["observations"]
    row = replayed[-2]
    assert row["prediction"]["valid"] is False
    assert row["prediction"]["label"] is None
    assert row["prediction"]["score"] == row["raw_quality"]["live_quality_score"]
    assert "recorded_observation_gap" in row["prediction"]["reasons"]
    assert replayed[-3]["prediction"]["valid"] is False
    assert replayed[-1]["raw_quality"] == expected_after_reset
    assert replayed[-1]["prediction"]["valid"] is True


def test_explicit_quality_reset_is_replayed_without_score_drift(tmp_path):
    writer = writer_at(tmp_path / "recording")
    monitor = LiveGraspQualityMonitor(recording_sink=writer)
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    monitor.update(base, quality_observation(incipient_slip_score=100))
    monitor.reset()
    expected = monitor.update(
        base, quality_observation(incipient_slip_score=0, timestamp=11)
    ).to_dict()
    writer.close()
    assert replay_recording(writer.root)["observations"][-1]["raw_quality"] == expected


def test_applied_quality_config_changes_the_replay_fingerprint(tmp_path):
    hashes, scores = [], []
    for weight in (12, 24):
        writer = writer_at(tmp_path / str(weight))
        monitor = LiveGraspQualityMonitor(
            {"weights": {"incipient_slip": weight}}, recording_sink=writer
        )
        monitor.update(
            BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD"),
            quality_observation(incipient_slip_score=100),
        )
        writer.close()
        replay = replay_recording(writer.root)
        hashes.append(replay["configuration"]["sha256"])
        scores.append(replay["observations"][0]["prediction"]["score"])
    assert hashes[0] != hashes[1]
    assert scores[0] != scores[1]


def test_mid_recording_quality_config_changes_require_an_explicit_new_arm(tmp_path):
    writer = writer_at(tmp_path / "recording")
    monitor = LiveGraspQualityMonitor(recording_sink=writer)
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    monitor.update(base, quality_observation())
    monitor.config["weights"]["incipient_slip"] = 24
    monitor.update(base, quality_observation(timestamp=11))
    writer.close()
    with pytest.raises(ValueError, match="configuration changed"):
        replay_recording(writer.root)


def test_unknown_warm_quality_state_is_rejected(tmp_path):
    writer = writer_at(tmp_path / "recording")
    monitor = LiveGraspQualityMonitor()
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    monitor.update(base, quality_observation())
    monitor.recording_sink = writer
    monitor.update(base, quality_observation(timestamp=11))
    writer.close()
    with pytest.raises(ValueError, match="unknown prior state"):
        replay_recording(writer.root)


def test_missing_recorded_timestamp_cannot_use_current_time(tmp_path):
    writer = writer_at(tmp_path / "recording")
    observation = asdict(quality_observation())
    observation.pop("timestamp")
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    writer.record(
        "live_quality_inputs",
        {},
        observation_id="incomplete-input",
        host_arrival_monotonic_ns=10,
        host_arrival_epoch_ns=20,
        native_clock={"domain": "application_observation_unspecified", "unit": "s"},
        metadata={
            "baseline": asdict(base),
            "observation": observation,
            "effective_config": LiveGraspQualityMonitor().config,
            "state_epoch": 0,
            "epoch_input_index": 0,
            "initial_state": "constructor_reset",
        },
        source_sequence=0,
    )
    writer.close()
    with pytest.raises(ValueError, match="complete recorded"):
        replay_recording(writer.root)


def test_recording_replay_and_independent_evaluation_contracts_integrate(tmp_path):
    from grasp_project.benchmarking.offline_evaluation import (
        ANNOTATION_SCHEMA,
        SPLIT_SCHEMA,
        evaluate_recorded_trials,
        replay_predictions_document,
    )

    writer = writer_at(tmp_path / "recording")
    monitor = LiveGraspQualityMonitor(recording_sink=writer)
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    monitor.update(
        base, quality_observation(incipient_slip_score=0, davis_motion_class="STABLE")
    )
    monitor.update(
        base,
        quality_observation(
            timestamp=11,
            incipient_slip_score=0,
            event_state="UNAVAILABLE",
            davis_event_freshness="UNAVAILABLE",
            davis_events_valid_for_live_quality=False,
            davis_motion_usable_for_grasp=False,
        ),
    )
    writer.close()
    replay = replay_recording(writer.root)
    declaration = {
        "configuration_id": "fixture-predeclared",
        "revision": "synthetic-fixture",
        "parameters": {"replay_configuration": replay["configuration"]},
        "required_evidence": ["davis_events"],
        "frozen_before_test": True,
        "selection_method": "predeclared",
        "fit_trial_ids": [],
        "selection_trial_ids": [],
    }
    predictions = replay_predictions_document(
        replay,
        task_id="quiet_fixture_decision",
        configuration=declaration,
        kind="live_quality_inputs",
    )
    # The fixture's quiet-state labels are fixed independently of model outputs.
    annotations = {
        "schema_version": ANNOTATION_SCHEMA,
        "task_id": "quiet_fixture_decision",
        "label_vocabulary": ["MONITOR", "WAIT_FOR_DAVIS"],
        "trials": [
            {
                "trial_id": "trial-1",
                "session_id": "fixture-session",
                "object_id": "fixture-object",
                "data_origin": "synthetic",
            }
        ],
        "annotations": [
            {
                "annotation_id": f"fixture-{index}",
                "trial_id": "trial-1",
                "observation_id": row["observation_id"],
                "state": "labelled",
                "label": "MONITOR",
                "provenance": {
                    "source": "synthetic_fixture",
                    "annotator_id": "fixture-author",
                    "independent_of_predictions": True,
                    "prediction_blinded": True,
                    "evidence_reference": {"observation_ids": [row["observation_id"]]},
                },
            }
            for index, row in enumerate(replay["observations"])
        ],
    }
    splits = {
        "schema_version": SPLIT_SCHEMA,
        "group_by": ["trial_id", "session_id", "object_id"],
        "assignments": [{"trial_id": "trial-1", "split": "test"}],
    }
    for split in ("train", "validation"):
        trial = f"fixture-{split}"
        annotations["trials"].append(
            {
                "trial_id": trial,
                "session_id": trial,
                "object_id": trial,
                "data_origin": "synthetic",
            }
        )
        splits["assignments"].append({"trial_id": trial, "split": split})
    report = evaluate_recorded_trials(annotations, predictions, splits)
    result = report["results"]["fixture-predeclared"]
    summary = result["splits"]["test"]
    assert summary["paired_count"] == 1
    assert summary["abstention_count"] == 1
    assert summary["coverage_of_labelled"] == 0.5
    unavailable = next(
        row for row in result["outcomes"] if row["outcome"] == "unusable_evidence"
    )
    assert unavailable["prediction"]["score"] == 84
    assert unavailable["evidence"]["davis_events"]["status"] == "unavailable"
    assert report["physical_validation_established"] is False


@pytest.mark.parametrize("name", ABLATIONS)
def test_explicit_offline_ablations_reuse_identical_recorded_inputs(tmp_path, name):
    writer = writer_at(tmp_path / "recording")
    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    original = quality_observation(post_grasp_active=False, grasp_stage="PRE_CLOSE")
    before = asdict(original)
    LiveGraspQualityMonitor(recording_sink=writer).update(base, original)
    writer.close()
    report = replay_recording(writer.root, ablation=name)
    reference = replay_recording(writer.root)
    assert asdict(original) == before
    assert (
        report["observations"][0]["observation_id"]
        == reference["observations"][0]["observation_id"]
    )
    assert report["configuration"]["ablation"] == name
    if name == "phase_conditioning_removed":
        assert (
            report["observations"][0]["raw_quality"][
                "davis_motion_used_for_live_quality"
            ]
            is True
        )
    if name == "events_disabled":
        assert (
            report["observations"][0]["evidence"]["davis_events"]["status"]
            == "unavailable"
        )
    if name == "sync_diagnostics_removed":
        assert (
            report["observations"][0]["prediction"]
            == reference["observations"][0]["prediction"]
        )
    assert json.loads(json.dumps(report))["physical_validation"] is False

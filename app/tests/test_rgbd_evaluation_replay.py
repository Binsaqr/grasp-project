"""Synthetic software parity; never physical grasp or latency validation."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from grasp_project.recording.rgbd_evaluation_replay import (
    ReplayClock,
    _differences,
    _normal,
    _RecordedMonitor,
    _trace_records,
    evaluation_context,
    replay_rgbd_evaluation,
)
from grasp_project.recording.rgbd_trace import (
    RgbdTraceRecorder,
    report_contract,
    runtime_trace_context,
)
from grasp_project.recording.session import Recording, RecordingWriter, read_recording


def test_clock_rejects_missing_nonfinite_and_negative_inputs():
    clock = ReplayClock()
    for value in (None, float("nan"), float("inf"), -1, True):
        with pytest.raises(ValueError, match="recorded clock"):
            clock.set({"epoch_sec": value, "monotonic_sec": 2, "perf_counter_sec": 3})
    clock.set({"epoch_sec": 100, "monotonic_sec": 2, "perf_counter_sec": 3})
    assert clock.proxy().time() == 100
    assert clock.proxy().monotonic() == 2
    assert clock.proxy().perf_counter() == 3


def test_native_integer_ticks_and_identifiers_compare_exactly_at_large_origins():
    value = 2**63 + 1
    expected = {"native_clock": {"timestamp": value}, "frame_id": value}
    assert not _differences(expected, copy.deepcopy(expected))
    changed = copy.deepcopy(expected)
    changed["native_clock"]["timestamp"] += 1
    assert _differences(expected, changed)[0]["path"] == "native_clock.timestamp"
    assert _differences({"frame_id": value}, {"frame_id": value + 1})
    assert _differences(value, float(value))
    assert not _differences(84, 84.0)
    assert not _differences(0.126, 0.126 + 1e-10)


def test_float_acquisition_clocks_are_exact_at_large_epoch_origins():
    epoch = 1_700_000_000.0
    for field in (
        "native_clock.timestamp",
        "host_clock.seconds",
        "status.acquisition.t",
        "intrinsics.camera_timestamp",
        "clock_inputs[0].epoch_sec",
        "host_arrival_epoch_ns",
        "rgb_host_timestamp",
    ):
        for delta in (100.0, 0.000001):
            differences = _differences(epoch, epoch + delta, field)
            assert differences and differences[0]["reason"] == "clock_value", field
        assert not _differences(epoch, epoch, field)
    assert not _differences(epoch, epoch + 1.0, "bottle_geometry.volume")


def test_output_artifact_reference_is_portable_but_keeps_identity_and_geometry():
    expected = {
        "target_lock": {
            "locked_target": {
                "locked_grasp_band": {
                    "full_lock_point_data_path": "C:/capture-output/locked_grasp_band.json",
                    "center": [0.1, 0.2, 0.3],
                }
            }
        },
        "acquisition_path": "C:/capture-output/raw.npy",
    }
    actual = copy.deepcopy(expected)
    band = actual["target_lock"]["locked_target"]["locked_grasp_band"]
    band["full_lock_point_data_path"] = "/tmp/replay/locked_grasp_band.json"
    normalized = _normal(expected, scope="report")
    assert not _differences(normalized, _normal(actual, scope="report"))
    assert _normal(expected) == expected
    assert normalized["acquisition_path"] == expected["acquisition_path"]
    band["full_lock_point_data_path"] = "/tmp/replay/another-artifact.json"
    assert _differences(normalized, _normal(actual, scope="report"))
    band["full_lock_point_data_path"] = "/tmp/replay/locked_grasp_band.json"
    band["center"][0] += 0.01
    assert _differences(normalized, _normal(actual, scope="report"))


def test_comparison_keeps_score_decision_validity_reasons_and_native_clocks():
    expected = {
        "state_history_event": {"timestamp": "generated-a"},
        "state_machine": {"updated_at": "generated-a"},
        "score": 90,
        "decision": "WAIT",
        "valid": False,
        "reasons": ["missing_motion"],
        "native_timestamp_us": 123,
    }
    actual = {
        **expected,
        "state_history_event": {"timestamp": "generated-b"},
        "state_machine": {"updated_at": "generated-b"},
    }
    assert (
        _differences(_normal(expected, scope="report"), _normal(actual, scope="report"))
        == []
    )
    for key, value in {
        "score": 89,
        "decision": "READY",
        "valid": True,
        "reasons": [],
        "native_timestamp_us": 124,
    }.items():
        assert _differences(
            _normal(expected, scope="report"),
            _normal({**actual, key: value}, scope="report"),
        )


def test_nested_acquisition_and_monitor_clocks_are_never_normalized_away():
    value = {
        "native_clock": {"timestamp": 123, "sec": 4, "nsec": 5},
        "status": {
            "acquisition": {"timestamp": 125, "timestamp_sec": 4, "timestamp_nsec": 7}
        },
        "event_monitor": {"timestamp": 126, "event_id": "native-sequence-1"},
    }
    assert _normal(value, scope="report") == value
    for path in (
        ("native_clock", "timestamp"),
        ("native_clock", "sec"),
        ("native_clock", "nsec"),
        ("status", "acquisition", "timestamp"),
        ("status", "acquisition", "timestamp_sec"),
        ("status", "acquisition", "timestamp_nsec"),
        ("event_monitor", "timestamp"),
    ):
        changed = copy.deepcopy(value)
        child = changed
        for key in path[:-1]:
            child = child[key]
        child[path[-1]] += 1
        assert _differences(
            _normal(value, scope="report"), _normal(changed, scope="report")
        )
    monitor = _RecordedMonitor(
        [{"method": "start", "args": [value], "kwargs": {}, "result": {}}]
    )
    changed = copy.deepcopy(value)
    changed["native_clock"]["timestamp"] += 1
    with pytest.raises(ValueError, match="arguments mismatch"):
        monitor.start(changed)


def test_epsilon_contract_detects_metric_force_closure_and_validity_corruption():
    report = {
        "epsilon_score": 86,
        "decision": "WAIT",
        "epsilon_result": {
            "epsilon": 0.126,
            "force_closure": True,
            "valid": True,
            "validity_reason": "current_contact_geometry",
        },
    }
    for key, replacement in {
        "epsilon": 0.001,
        "force_closure": False,
        "valid": False,
        "validity_reason": "missing_contact_geometry",
    }.items():
        changed = copy.deepcopy(report)
        changed["epsilon_result"][key] = replacement
        differences = _differences(
            _normal(report_contract(report), scope="report"),
            _normal(report_contract(changed), scope="report"),
        )
        assert any(item["path"] == f"epsilon_result.{key}" for item in differences)


def test_monitor_cannot_silently_swallow_missing_instrumentation():
    monitor = _RecordedMonitor()
    with pytest.raises(ValueError, match="missing recorded"):
        monitor.stop()
    assert monitor.errors
    monitor = _RecordedMonitor(
        [{"method": "start", "args": [{"score": 20}], "kwargs": {}, "result": {}}]
    )
    with pytest.raises(ValueError, match="arguments mismatch"):
        monitor.start({"score": 90})


def test_journal_rejects_missing_initial_and_dropped_trace():
    with pytest.raises(ValueError, match="initial"):
        _trace_records([])
    with pytest.raises(ValueError, match="dropped"):
        _trace_records([{"record_type": "drop", "stream": "rgbd_trace"}])


def _fingerprints(directory):
    return (
        {
            path.relative_to(directory).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if directory.exists()
        else {}
    )


def _synthetic_rgbd_scene():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[45:196, 137:184] = (35, 90, 210)
    # The pickup gate requires observed clear surrounding space. Missing
    # background depth is unknown space, not evidence that the path is clear.
    depth = np.full((240, 320), 500, dtype=np.uint16)
    depth[45:196, 137:184] = 260
    intrinsics = {
        "fx": 200.0,
        "fy": 200.0,
        "cx": 160.0,
        "cy": 120.0,
        "source": "synthetic_software_fixture",
    }
    return rgb, depth, intrinsics


def test_synthetic_scene_requires_observed_clear_space_for_production_selection():
    from bottle_pose import detect_bottle_pose
    from bottle_visibility import check_full_bottle_visibility
    from grasp_project.planning.task_grasp_policy import select_bottle_target
    from grasp_project.realsense.object_state import estimate_object_state

    rgb, depth, intrinsics = _synthetic_rgbd_scene()
    for background_valid in (True, False):
        observed_depth = depth.copy()
        if not background_valid:
            observed_depth[observed_depth == 500] = 0
        pose, error = detect_bottle_pose(rgb, observed_depth, intrinsics=intrinsics)
        assert error is None and pose["detected"]
        assert check_full_bottle_visibility(pose, rgb.shape)["full_bottle_visible"]
        state = estimate_object_state(pose["cleaned_mask"], observed_depth, pose=pose)
        selection = select_bottle_target(
            [{"target_id": "synthetic-bottle", "pose": pose, "object_state": state}],
            image_shape=rgb.shape,
            intrinsics=intrinsics,
            depth_image=observed_depth,
        ).to_dict()
        if background_valid:
            assert state.available_for_pickup
            assert selection["selected_target_id"] == "synthetic-bottle"
            assert selection["eligible_count"] == 1
        else:
            assert not state.available_for_pickup
            assert "insufficient_clear_space_depth" in state.reasons
            assert selection["selected_target_id"] is None
            assert selection["eligible_count"] == 0


def _synthetic_journal(tmp_path):
    # Actual core imports are required on Linux; missing supported dependencies
    # fail CI rather than silently skipping the canonical evaluator.
    import bottle_pose
    import pregrasp_planner
    from grasp_project.realsense import realtime_runner as runtime

    clock = ReplayClock()
    clock.set(
        {"epoch_sec": 1_700_000_000, "monotonic_sec": 100, "perf_counter_sec": 100}
    )
    embodiment = runtime.current_embodiment_status()
    with patch.object(pregrasp_planner, "active_object_profile_override", lambda: None):
        grasp_profile = pregrasp_planner.load_known_bottle_profile()
    initial = {
        "runtime_context": runtime_trace_context(vars(runtime)),
        "external_inputs": {
            "bottle_profile": bottle_pose.load_known_bottle_profile(),
            "grasp_profile": grasp_profile,
            "embodiment_status": embodiment,
            "metric_applicability": runtime.quality_metric_applicability(
                embodiment.get("embodiment_type") or embodiment.get("active_embodiment")
            ),
        },
    }
    # This fixture selects the supported default detector path explicitly;
    # scoring and readiness thresholds retain their production values.
    initial["runtime_context"]["USE_YOLO_BOTTLE_TRIGGER"] = False
    pointcloud = {
        "pointcloud_status": "unavailable",
        "pointcloud_freshness": "UNAVAILABLE",
        "realsense_rgb_active": True,
        "realsense_depth_active": True,
        "realsense_intrinsics_available": True,
        "realsense_pointcloud_active": False,
        "pointcloud_current": False,
    }
    monitor = _RecordedMonitor(
        [
            {"method": "start", "args": [], "kwargs": {}, "result": {}},
            {"method": "update", "args": [], "kwargs": {}, "result": {}},
            {"method": "stop", "args": [], "kwargs": {}, "result": None},
            {"method": "stop", "args": [], "kwargs": {}, "result": None},
        ]
    )
    # Live baseline argument is captured by TracedEventMonitor, so this fixture
    # monitor accepts it while the replay monitor checks the archived arguments.
    monitor._call = lambda method, *_args, **_kwargs: {} if method != "stop" else None
    recording_dir = tmp_path / "recording"
    with (
        closing(
            RecordingWriter(
                recording_dir,
                recording_id="synthetic-rgbd",
                config={},
                revision={"fixture": "synthetic"},
                metadata={"trial_id": "synthetic-rgbd"},
            )
        ) as writer,
        evaluation_context(
            runtime,
            initial,
            clock=clock,
            monitor=monitor,
            output_dir=tmp_path / "capture-output",
            external_for_frame={"pointcloud_status": pointcloud},
        ),
        patch.object(
            RgbdTraceRecorder,
            "clock",
            staticmethod(
                lambda: {
                    "epoch_sec": clock.epoch_sec,
                    "monotonic_sec": clock.monotonic_sec,
                    "perf_counter_sec": clock.perf_counter_sec,
                }
            ),
        ),
    ):
        worker = runtime.EvaluationWorker(
            lock_target=True,
            restore_grasp_memory=False,
            publish_lock_artifacts=False,
            recording_sink=writer,
        )
        rgb, depth, intrinsics = _synthetic_rgbd_scene()
        pose, error = bottle_pose.detect_bottle_pose(rgb, depth, intrinsics=intrinsics)
        assert error is None and pose["detected"]
        worker.force_lock_current_target_for_command("synthetic-force-lock")
        worker.evaluate(rgb, depth, intrinsics, 1)
        assert worker.lock_manager.status() == "LOCKED"
        target = worker.lock_manager.locked_target
        baseline = {
            "quality_matrix_status": "updated",
            "writer_session_id": runtime.WRITER_SESSION_ID,
            "lock_attempt_id": target["attempt_id"],
            "attempt_id": target["attempt_id"],
            "lock_frame_id": target["lock_frame_id"],
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
            "quality_compute_id": "synthetic-analytic-completion",
            "quality_label": "GOOD",
            "raw_epsilon": 0.126,
            "normalized_quality_score": 86.0,
            "epsilon_force_closure": True,
        }
        worker.attach_locked_quality_snapshot(baseline)
        transition = worker._mark_gripper_closed_result()
        assert transition["ok"] is True
        clock.set(
            {
                "epoch_sec": 1_700_000_001,
                "monotonic_sec": 101,
                "perf_counter_sec": 101,
            }
        )
        worker.evaluate(rgb, depth, intrinsics, 2)
        worker.mark_gripper_open()
        clock.set(
            {
                "epoch_sec": 1_700_000_002,
                "monotonic_sec": 102,
                "perf_counter_sec": 102,
            }
        )
        worker.evaluate(rgb, depth, intrinsics, 3)
        worker.reset_lock_from_dashboard()
        worker.evaluate(rgb, depth, intrinsics, 4)
    return read_recording(recording_dir)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="canonical worker requires supported Linux ROS Humble/PyQt5/fcntl runtime",
)
def test_actual_worker_replay_detection_commands_baseline_and_isolated_outputs(
    tmp_path,
):
    from grasp_project.realsense import realtime_runner as runtime

    root = Path(runtime.__file__).resolve().parents[3]
    before_config = _fingerprints(root / "config")
    before_output = _fingerprints(root / "output")
    recording = _synthetic_journal(tmp_path)
    archive_before = _fingerprints(tmp_path / "recording")
    first = replay_rgbd_evaluation(recording)
    second = replay_rgbd_evaluation(recording)
    assert first == second
    assert first["verified"] is True, first["mismatches"]
    assert len(first["outputs"]) == 4
    assert {command["method"] for command in first["commands"]} >= {
        "force_lock_current_target_for_command",
        "attach_locked_quality_snapshot",
        "_mark_gripper_closed_result",
        "mark_gripper_open",
        "reset_lock_from_dashboard",
    }
    assert first["physical_validation"] is False and first["ground_truth_used"] is False
    assert _fingerprints(root / "config") == before_config
    assert _fingerprints(root / "output") == before_output
    assert _fingerprints(tmp_path / "recording") == archive_before
    destination = tmp_path / "cli-replay.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/replay_recording.py"),
            str(tmp_path / "recording"),
            "--rgbd-evaluation",
            "--output",
            str(destination),
        ],
        cwd=tmp_path,
        # Keep the sourced ROS runtime paths when adding the source package.
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(root / "src"), os.environ.get("PYTHONPATH")))
            ),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    cli_report = json.loads(destination.read_text(encoding="utf-8"))
    assert cli_report["rgbd_evaluation"]["verified"] is True
    assert _fingerprints(root / "config") == before_config
    assert _fingerprints(root / "output") == before_output
    assert _fingerprints(tmp_path / "recording") == archive_before


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="canonical worker requires supported Linux ROS Humble/PyQt5/fcntl runtime",
)
def test_actual_worker_does_not_echo_corrupted_golden_decision(tmp_path):
    recording = _synthetic_journal(tmp_path)
    records = copy.deepcopy(recording.records)
    expected = next(
        record
        for record in records
        if record.get("metadata", {}).get("event") == "output"
    )
    expected["metadata"]["data"]["report"]["decision"] = "INVENTED_GOLDEN_DECISION"
    report = replay_rgbd_evaluation(Recording(recording.manifest, records))
    assert report["verified"] is False
    assert any(item["path"] == "decision" for item in report["mismatches"])

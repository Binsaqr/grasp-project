"""CLI help and mocked lifecycles; these tests never instantiate a camera backend."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_project.recording import read_recording

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_native_trial.py"
SPEC = importlib.util.spec_from_file_location("record_native_trial", SCRIPT)
capture = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capture)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock.now)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(capture.time, "sleep", sleep)
    config = {"davis_motion": {"window_ms": 40.0},
              "event_window_selection": {"primary_window_sec": 0.1, "fallback_window_sec": 0.8,
                                         "min_primary_events": 8}, "config_files": {}}
    monkeypatch.setattr(capture, "_configuration", lambda: config.copy())
    monkeypatch.setattr(capture, "_revision", lambda: {"commit": "synthetic-fixture", "working_tree_dirty": False})
    state = SimpleNamespace(monitor=None, fail_start=False, poll_error=None, fail_stop=False,
                            mismatch_config=False, ros=None, ros_fail=False, ros_empty=False)

    class Monitor:
        def __init__(self, writer):
            self.writer = writer
            self.active = False
            self.stopped = False
            self.polls = 0
            self.device_open_error = "mock backend unavailable"
            self.unavailable_reason = "mock backend unavailable"
            self.motion_estimator = SimpleNamespace(config={} if state.mismatch_config else config["davis_motion"])

        def start(self):
            self.active = not state.fail_start

        def poll_event_and_frame_arrays(self):
            if state.poll_error is not None:
                raise state.poll_error
            self.writer.record(
                "davis_poll", {"events": np.empty((0, 4), dtype=np.int64)},
                observation_id=f"mock-{self.polls}", host_arrival_monotonic_ns=int(clock.now * 1e9),
                host_arrival_epoch_ns=1_800_000_000_000_000_000,
                native_clock={"domain": "davis_device", "unit": "us"},
                metadata={"fixture": "synthetic"}, source_sequence=self.polls,
            )
            self.polls += 1

        def stop(self):
            self.stopped = True
            self.active = False
            if state.fail_stop:
                raise RuntimeError("mock shutdown failure")

    def monitor_factory(writer, destination):
        assert destination == tmp_path / "raw"
        state.monitor = Monitor(writer)
        return state.monitor

    class Ros:
        def __init__(self, writer, args):
            assert writer is not None
            assert args.with_ros
            if state.ros_fail:
                raise ImportError("mock rclpy unavailable")
            state.ros = self
            self.counts = {"rgb": 0, "depth": 0, "camera_info": 0}
            self.closed = False

        def poll(self):
            if not state.ros_empty:
                self.counts = {key: count + 1 for key, count in self.counts.items()}

        def close(self):
            self.closed = True

    monkeypatch.setattr(capture, "_create_monitor", monitor_factory)
    monkeypatch.setattr(capture, "_RosSubscriptions", Ros)
    args = capture.build_parser().parse_args([
        str(tmp_path / "raw"), "--recording-id", "trial-1", "--session-id", "session-1",
        "--object-id", "object-1", "--duration-sec", "0.03",
    ])
    return args, state


def test_subprocess_help_exits_without_optional_dependencies_or_sensor_imports():
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "canonical DAVIS backend" in " ".join(result.stdout.split())
    assert "--with-ros" in result.stdout


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "-inf"])
def test_parser_rejects_unbounded_or_nonpositive_duration(duration, tmp_path):
    with pytest.raises(SystemExit):
        capture.build_parser().parse_args([
            str(tmp_path), "--recording-id", "trial-1", "--session-id", "session-1",
            "--object-id", "object-1", "--duration-sec", duration,
        ])


def test_mock_success_stops_backend_and_finalizes_only_requested_duration(harness):
    args, state = harness
    manifest = capture.run_capture(args)
    assert state.monitor.stopped
    assert state.monitor.polls >= 2
    recording = read_recording(args.destination)
    assert manifest == args.destination / "manifest.json"
    assert recording.manifest["metadata"]["data_origin"] == "recorded"
    assert recording.manifest["metadata"]["ground_truth_status"] == "not_labelled"
    assert recording.manifest["metadata"]["session_id"] == "session-1"
    assert recording.manifest["metadata"]["object_id"] == "object-1"
    assert recording.manifest["config"]["event_window_selection"]["min_primary_events"] == 8
    assert recording.manifest["config"]["davis_motion"] == {"window_ms": 40.0}
    assert recording.manifest["physical_validation"] is False


@pytest.mark.parametrize("failure", ["start", "poll", "interrupt", "stop", "config"])
def test_mock_failures_leave_incomplete_and_release_backend(harness, failure):
    args, state = harness
    if failure == "start":
        state.fail_start = True
    elif failure == "poll":
        state.poll_error = RuntimeError("mock poll failure")
    elif failure == "interrupt":
        state.poll_error = KeyboardInterrupt()
    elif failure == "stop":
        state.fail_stop = True
    else:
        state.mismatch_config = True
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)):
        capture.run_capture(args)
    assert state.monitor.stopped
    assert json.loads((args.destination / "manifest.json").read_text())["status"] == "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(args.destination)


def test_existing_output_rejected_before_backend_creation(harness):
    args, state = harness
    args.destination.mkdir()
    marker = args.destination / "existing-user-file.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        capture.run_capture(args)
    assert state.monitor is None
    assert marker.read_text() == "preserve"


def test_optional_ros_is_subscriber_only_and_all_requested_streams_required(harness):
    args, state = harness
    args.with_ros = True
    capture.run_capture(args)
    assert state.ros.closed
    assert all(state.ros.counts.values())
    assert state.monitor.stopped


def test_missing_ros_messages_cannot_be_called_complete(harness):
    args, state = harness
    args.with_ros = True
    state.ros_empty = True
    with pytest.raises(RuntimeError, match="did not all arrive"):
        capture.run_capture(args)
    assert state.ros.closed
    assert state.monitor.stopped
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(args.destination)


def test_unavailable_ros_dependency_cannot_open_davis(harness):
    args, state = harness
    args.with_ros = True
    state.ros_fail = True
    with pytest.raises(ImportError):
        capture.run_capture(args)
    assert state.monitor is None
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(args.destination)


@pytest.mark.parametrize("field,value", [("session_id", " "), ("object_id", ""),
                                         ("poll_interval_sec", 10.0), ("poll_interval_sec", 0.0)])
def test_invalid_operator_inputs_fail_before_backend_or_directory_creation(harness, field, value):
    args, state = harness
    setattr(args, field, value)
    with pytest.raises(ValueError):
        capture.run_capture(args)
    assert state.monitor is None
    assert not args.destination.exists()


def test_config_file_snapshot_preserves_raw_utf8_contents_and_hash(tmp_path, monkeypatch):
    import hashlib

    directory = tmp_path / "config"
    directory.mkdir()
    contents = b"threshold: 12\r\n"
    (directory / "davis.yaml").write_bytes(contents)
    (directory / "ignored.txt").write_text("not config")
    monkeypatch.setattr(capture, "PROJECT_ROOT", tmp_path)
    files = capture._config_files()
    assert files == {"config/davis.yaml": {
        "contents_utf8": "threshold: 12\r\n", "sha256": hashlib.sha256(contents).hexdigest(),
    }}


def test_configuration_retains_exact_resolved_window_values_without_starting_backend(monkeypatch):
    from grasp_project.davis import motion_estimation, motion_gui

    files = {"config/davis.yaml": {"contents_utf8": "fixture: true\n", "sha256": "fixture"}}
    monkeypatch.setattr(capture, "_config_files", lambda: files)
    monkeypatch.setattr(motion_estimation, "load_davis_motion_config", lambda: {"fixture_threshold": 3.0})
    monkeypatch.setattr(motion_gui, "PRIMARY_EVENT_WINDOW_SEC", 0.123)
    monkeypatch.setattr(motion_gui, "FALLBACK_EVENT_WINDOW_SEC", 0.456)
    monkeypatch.setattr(motion_gui, "MIN_PRIMARY_WINDOW_EVENTS", 7)
    configuration = capture._configuration()
    assert configuration == {
        "davis_motion": {"fixture_threshold": 3.0}, "config_files": files,
        "event_window_selection": {"primary_window_sec": 0.123, "fallback_window_sec": 0.456,
                                   "min_primary_events": 7},
        "runtime_versions": {"python": sys.version, "numpy": np.__version__, "opencv": __import__("cv2").__version__},
    }


def test_nested_configuration_files_are_retained(tmp_path, monkeypatch):
    path = tmp_path / "config" / "embodiments" / "parallel_jaw.yaml"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"contacts: 2\n")
    monkeypatch.setattr(capture, "PROJECT_ROOT", tmp_path)
    assert capture._config_files()["config/embodiments/parallel_jaw.yaml"]["contents_utf8"] == "contacts: 2\n"


def test_configuration_file_drift_is_rejected_before_capture(monkeypatch):
    snapshots = iter([{"config/davis.yaml": {"contents_utf8": "before"}},
                      {"config/davis.yaml": {"contents_utf8": "after"}}])
    monkeypatch.setattr(capture, "_config_files", lambda: next(snapshots))
    with pytest.raises(ValueError, match="changed while resolving"):
        capture._configuration()

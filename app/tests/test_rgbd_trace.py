"""Camera-free software trace checks; fixtures provide no physical evidence."""

import ast
import copy
import signal
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_project.recording import RecordingWriter, read_recording
from grasp_project.recording.rgbd_trace import (
    RgbdTraceRecorder,
    TracedEventMonitor,
    capture_worker_state,
    recording_shutdown_signals,
    replay_rgbd_trace,
    report_contract,
    restore_worker_state,
    runtime_trace_context,
    trace_worker_command,
    unpack_trace_value,
)
from state_machine import GraspStateMachine


def worker():
    return SimpleNamespace(
        machine=GraspStateMachine(state="FULL_BOTTLE_VIEW_CHECK"),
        lock_manager=SimpleNamespace(
            enabled=True, locked_target=None, has_lost_lock=False,
            lock_artifact_invalidator=lambda *_args: None,
            temporal_state={"mask": np.arange(6, dtype=np.uint16).reshape(2, 3)},
            good_frame_history=deque([{"frame_id": 1}], maxlen=30),
            ready_candidate_frames=0, stable_ready_frames=0,
        ),
        event_monitor=SimpleNamespace(active=False, enabled=True, last_report={"motion_class": "SENSOR_UNAVAILABLE"}),
        latest_report=None, last_full_eval_time=0.0, post_grasp_active=False,
        post_grasp_baseline=None, post_grasp_stable_frames=0, last_evaluated_frame_id=0,
        force_lock_next_eval=False, pending_lock_command_id=None,
        locked_full_recovery_failures=0,
        evaluation_times=deque(maxlen=120), evaluation_start_times=deque(maxlen=120),
    )


def writer_at(tmp_path):
    return RecordingWriter(
        tmp_path / "raw", recording_id="synthetic-rgbd", config={"synthetic": True},
        revision={"commit": "synthetic-fixture"}, metadata={"trial_id": "synthetic-trial", "data_origin": "synthetic"},
    )


def trace_at(tmp_path):
    target = worker()
    sink = writer_at(tmp_path)
    trace = RgbdTraceRecorder(sink, target, constructor={"lock_target": True, "restore_grasp_memory": False}, runtime_context={})
    target.rgbd_trace = trace
    return target, sink, trace


def test_typed_worker_snapshot_roundtrip_preserves_arrays_deques_and_nan():
    original = worker()
    original.lock_manager.temporal_state["nonfinite"] = np.array([np.nan, np.inf], dtype=np.float32)
    arrays = {}
    snapshot = capture_worker_state(original, arrays)
    target = worker()
    target.lock_manager.temporal_state = {}
    restore_worker_state(target, snapshot, arrays)
    np.testing.assert_array_equal(target.lock_manager.temporal_state["mask"], original.lock_manager.temporal_state["mask"])
    assert target.lock_manager.temporal_state["mask"].dtype == np.uint16
    assert np.isnan(target.lock_manager.temporal_state["nonfinite"][0])
    assert target.lock_manager.good_frame_history.maxlen == 30
    assert list(target.lock_manager.good_frame_history) == [{"frame_id": 1}]
    target.lock_manager.temporal_state["mask"][0, 0] = 99
    assert original.lock_manager.temporal_state["mask"][0, 0] == 0


def test_missing_snapshot_array_is_rejected():
    with pytest.raises(ValueError, match="missing trace state array"):
        unpack_trace_value({"__ndarray__": "missing"}, {})


def test_journal_roundtrip_uses_actual_fsm_and_retains_commands_frames(tmp_path):
    target, sink, trace = trace_at(tmp_path)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.full((2, 3), 400, dtype=np.uint16)
    intrinsics = {"fx": 100.0, "_recording_camera_info_observation_id": "calibration-1"}
    trace.begin_frame(target, rgb, depth, intrinsics, 2)
    first = trace.transition(target.machine, {
        "stage": "FULL_BOTTLE_VIEW_CHECK", "stage_decision": "VIEW_OK",
        "recommended_action": "MODEL_CONTACTS", "safety_score": 0.0,
        "reasons": ["synthetic_stage_input"], "warnings": [],
    }, frame_id=2, provenance={"writer_session_id": "synthetic-writer"})
    assert first["to_state"] == "CONTACT_MODELING"
    trace.output(target, {"frame_id": 2, "state": target.machine.state, "state_history_event": first})
    trace.emit("command_request", {"payload": {"command": "mark_gripper_closed", "command_id": "operator-1"}})
    trace.transition(target.machine, {
        "stage": "POST_GRASP_EVALUATION", "stage_decision": "DO_NOT_LIFT",
        "reasons": ["missing_event_evidence"], "warnings": [],
    }, frame_id=2, provenance={})
    sink.close()
    recording = read_recording(tmp_path / "raw")
    replay = replay_rgbd_trace(recording.records)
    assert replay["verified_fsm"] is True
    assert replay["state"] == "DO_NOT_LIFT"
    assert replay["transitions"][-1]["reasons"] == ["missing_event_evidence"]
    assert replay["worker_and_image_tracker_parity"] == "NOT_ASSESSED_BY_FSM_COMPONENT"
    assert replay["physical_validation"] is False
    frame = next(record for record in recording.records if record["metadata"]["event"] == "frame")
    np.testing.assert_array_equal(frame["arrays"]["depth"], depth)
    assert frame["arrays"]["rgb"].dtype == np.uint8
    assert frame["metadata"]["data"]["intrinsics"] == intrinsics
    assert set(frame["metadata"]["data"]["clock"]) == {"epoch_sec", "monotonic_sec", "perf_counter_sec"}


@pytest.mark.parametrize("corrupt", ["output", "order", "initial"])
def test_fsm_replay_rejects_semantic_corruption(tmp_path, corrupt):
    target, sink, trace = trace_at(tmp_path)
    trace.transition(target.machine, {"stage": "FULL_BOTTLE_VIEW_CHECK", "stage_decision": "VIEW_OK"}, frame_id=1, provenance={})
    sink.close()
    records = copy.deepcopy(read_recording(tmp_path / "raw").records)
    if corrupt == "output":
        records[-1]["metadata"]["data"]["event"]["to_state"] = "READY_TO_LIFT"
    elif corrupt == "order":
        records[-1]["metadata"]["trace_sequence"] = 3
    else:
        records = records[1:]
    with pytest.raises(ValueError):
        replay_rgbd_trace(records)


def test_report_contract_retains_scores_decisions_validity_and_provenance():
    report = {"frame_id": 3, "state": "DO_NOT_LIFT", "safety_score": 0.0,
              "ready_to_lift": False, "failure_reasons": ["missing_events"],
              "target_lock": {"locked_target": {"valid": False}}, "attempt_id": "attempt-1",
              "pointcloud_status": {"status": "not_available"}, "performance_profile": {"time": 1.0}}
    report["epsilon_result"] = {"epsilon": 0.126, "force_closure": True, "valid": True, "failure_reasons": []}
    contract = report_contract(report)
    assert "performance_profile" not in contract
    assert contract["failure_reasons"] == ["missing_events"]
    assert contract["target_lock"]["locked_target"]["valid"] is False
    assert contract["pointcloud_status"] == {"status": "not_available"}
    assert contract["epsilon_result"] == report["epsilon_result"]
    report["epsilon_result"]["epsilon"] = 999.0
    assert contract["epsilon_result"]["epsilon"] == 0.126


def runtime_class(name, **overrides):
    """Compile an actual class without importing unavailable ROS/Qt modules."""
    path = Path(__file__).parents[1] / "src/grasp_project/realsense/realtime_runner.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    namespace = {
        "QObject": object, "pyqtSignal": lambda *_args: SimpleNamespace(emit=lambda *_args: None),
        "pyqtSlot": lambda *_args: lambda function: function, "trace_worker_command": trace_worker_command,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


@pytest.mark.parametrize("post_grasp", [False, True])
def test_actual_worker_output_boundary_follows_fsm_and_current_report(post_grasp):
    target = worker()
    target.post_grasp_active = post_grasp
    target.locked_fast_frames_since_full = 0
    target.last_started_fps = 0.0
    target.lock_manager.status = lambda: "SEARCHING"
    target.event_monitor.update = lambda: {"motion_class": "SENSOR_UNAVAILABLE"}
    order = []
    report = {"frame_id": 1, "stage": "FULL_BOTTLE_VIEW_CHECK", "stage_decision": "VIEW_OK"}

    def finalize(_rgb, _pose, _plan, result, machine, **_kwargs):
        result["state_history_event"] = machine.record(result)
        order.append("fsm")

    def full(rgb, _depth, _intrinsics, _frame_id, machine, _manager, **_kwargs):
        finalize(rgb, None, None, report, machine)
        return report

    def output(worker, result):
        assert worker.latest_report is result
        assert result["state_history_event"]["to_state"] == "CONTACT_MODELING"
        assert len(worker.machine.history) == 1
        order.append("output")

    target.rgbd_trace = SimpleNamespace(begin_frame=lambda *_args, **_kwargs: order.append("frame"), output=output)
    target.result_ready = SimpleNamespace(emit=lambda _report: order.append("emit"))
    target._attach_evaluation_audit = lambda *_args: None
    actual = runtime_class(
        "EvaluationWorker", time=SimpleNamespace(perf_counter=lambda: 1.0, monotonic=lambda: 2.0),
        HAND_MOUNTED_CAMERA_CONFIG={"enabled": False},
        CameraWorker=SimpleNamespace(_fps=lambda _times: 0.0), locked_full_recovery_interval_sec=lambda _count: 1.0,
        locked_tracking_reacquisition_due=lambda *_args, **_kwargs: False, FAST_FINAL_DEMO_MODE=True,
        FULL_REDETECT_INTERVAL_FRAMES=12, FULL_EVAL_INTERVAL_SEC=1.5, evaluate_frame=full,
        evaluate_post_grasp_frame=lambda *_args, **_kwargs: (report, 0, None, None),
        attach_event_postgrasp_report=lambda result, _events: result,
        write_status_outputs=finalize, draw_fast_overlay=lambda *_args: None,
    )
    actual.evaluate(target, np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint16), {}, 1)
    assert order == ["frame", "fsm", "output", "emit"]


def test_actual_force_lock_command_is_journalled_without_policy_change(tmp_path):
    target, sink, _trace = trace_at(tmp_path)
    actual = runtime_class("EvaluationWorker")
    actual.force_lock_current_target_for_command(target, "manual-42")
    assert target.force_lock_next_eval is True
    assert target.pending_lock_command_id == "manual-42"
    sink.close()
    command = read_recording(tmp_path / "raw").records[-1]
    data = unpack_trace_value(command["metadata"]["data"], command["arrays"])
    assert data["method"] == "force_lock_current_target_for_command"
    assert data["args"] == ["manual-42"]
    assert data["worker_before"]["worker"]["force_lock_next_eval"] is False
    assert data["worker_after"]["worker"]["force_lock_next_eval"] is True
    assert data["sensor_measured"] is False


def test_external_monitor_calls_are_recorded_without_inventing_missing_evidence(tmp_path):
    _target, sink, trace = trace_at(tmp_path)
    monitor = TracedEventMonitor(SimpleNamespace(update=lambda: {"motion_class": "SENSOR_UNAVAILABLE", "event_count": 0}), trace)
    assert monitor.update()["motion_class"] == "SENSOR_UNAVAILABLE"
    sink.close()
    data = read_recording(tmp_path / "raw").records[-1]["metadata"]["data"]
    assert data["method"] == "update"
    assert data["result"] == {"motion_class": "SENSOR_UNAVAILABLE", "event_count": 0}


def test_external_monitor_failure_prevents_archive_completion(tmp_path):
    _target, sink, trace = trace_at(tmp_path)

    def fail():
        raise OSError("synthetic failure")

    monitor = TracedEventMonitor(SimpleNamespace(stop=fail), trace)
    with pytest.raises(OSError, match="synthetic failure"):
        monitor.stop()
    with pytest.raises(ValueError, match="previous write failed"):
        sink.close()


@pytest.mark.parametrize("stopped,owns", [(True, True), (False, True), (True, False), (False, False)])
def test_actual_backend_closes_only_owned_recording_after_threads_stop(stopped, owns):
    calls = []
    thread = SimpleNamespace(quit=lambda: None, wait=lambda _timeout: stopped)
    sink = SimpleNamespace(close=lambda: calls.append("close"), abort=lambda reason: calls.append(reason))
    target = SimpleNamespace(camera_worker=SimpleNamespace(stop=lambda: None), camera_thread=thread,
                             eval_thread=thread, quality_thread=thread, recording_sink=sink, _owns_recording_sink=owns)
    actual = runtime_class("RealtimeBackendRunner")
    if not stopped:
        with pytest.raises(RuntimeError, match="incomplete"):
            actual.stop(target)
        assert calls == ["rgbd_worker_shutdown_did_not_complete"]
    else:
        actual.stop(target)
        assert calls == (["close"] if owns else [])


def test_effective_runtime_context_retains_primitive_globals_without_arrays():
    result = runtime_trace_context({"WRITER_SESSION_ID": "synthetic-writer", "COUNT": 7,
                                    "CONFIG": {"enabled": True}, "ARRAY": np.arange(3), "function": lambda: None})
    assert result == {"WRITER_SESSION_ID": "synthetic-writer", "COUNT": 7, "CONFIG": {"enabled": True}}


def test_recording_signals_quit_then_restore_handlers(monkeypatch):
    original = {signal.SIGTERM: "term-original", signal.SIGINT: "int-original"}
    installed = dict(original)
    calls = []
    monkeypatch.setattr(signal, "getsignal", lambda number: installed[number])
    monkeypatch.setattr(signal, "signal", lambda number, handler: installed.__setitem__(number, handler))
    app = SimpleNamespace(quit=lambda: calls.append("quit"))
    with recording_shutdown_signals(app, enabled=True):
        installed[signal.SIGTERM](signal.SIGTERM, None)
        installed[signal.SIGINT](signal.SIGINT, None)
        assert calls == ["quit", "quit"]
    assert installed == original
    with recording_shutdown_signals(app, enabled=False):
        assert installed == original


def test_constructor_failure_before_all_workers_exist_starts_no_thread():
    calls = []

    class Thread:
        def __init__(self):
            self.started = SimpleNamespace(connect=lambda _callback: None)

        def start(self):
            calls.append("start")

        def quit(self):
            calls.append("quit")

        def wait(self, _timeout):
            return True

    class Camera:
        def __init__(self, **_kwargs):
            self.frame_ready = SimpleNamespace(connect=lambda _callback: None)

        def moveToThread(self, _thread):
            pass

        def run(self):
            pass

        def stop(self):
            calls.append("camera_stop")

    def fail(**_kwargs):
        raise ValueError("synthetic constructor failure")

    actual = runtime_class(
        "RealtimeBackendRunner", QThread=Thread, CameraWorker=Camera, EvaluationWorker=fail,
        deque=deque, time=SimpleNamespace(time=lambda: 1.0),
        quality_matrix_request_status=dict, write_quality_no_current_lock_status=lambda *_args: None,
        write_invalid_locked_grasp_band_artifacts=lambda *_args: None,
    )
    sink = SimpleNamespace(abort=lambda reason: calls.append(reason))
    with pytest.raises(ValueError, match="synthetic constructor failure"):
        actual(recording_sink=sink)
    assert "start" not in calls
    assert "camera_stop" in calls
    assert "rgbd_runtime_initialization_failed" in calls


@pytest.mark.parametrize("outcome", ["signal", "return", "exec_error", "signal_then_exec_error", "constructor_error"])
def test_backend_runtime_restores_signals_stops_once_and_releases_lock(monkeypatch, outcome):
    original = {signal.SIGTERM: "term-original", signal.SIGINT: "int-original"}
    installed = dict(original)
    calls = []
    monkeypatch.setattr(signal, "getsignal", lambda number: installed[number])
    monkeypatch.setattr(signal, "signal", lambda number, handler: installed.__setitem__(number, handler))
    callbacks = []

    def quit_app():
        for callback in callbacks:
            callback()

    def execute():
        if outcome in {"signal", "signal_then_exec_error"}:
            installed[signal.SIGTERM](signal.SIGTERM, None)
        if outcome in {"exec_error", "signal_then_exec_error"}:
            raise OSError("synthetic event loop failure")
        return 42

    app = SimpleNamespace(aboutToQuit=SimpleNamespace(connect=callbacks.append), exec_=execute, quit=quit_app)

    def construct(**kwargs):
        if outcome == "constructor_error":
            raise ValueError("synthetic constructor failure")
        return SimpleNamespace(recording_sink=kwargs["recording_sink"], stop=lambda **fields: calls.append(fields))

    path = Path(__file__).parents[1] / "src/grasp_project/realsense/realtime_runner.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run_live_realtime_backend")
    namespace = {
        "OUTPUT_DIR": SimpleNamespace(mkdir=lambda **_kwargs: None),
        "acquire_backend_writer_lock": lambda: True, "release_backend_writer_lock": lambda: calls.append("release"),
        "QCoreApplication": SimpleNamespace(instance=lambda: app), "RealtimeBackendRunner": construct,
        "recording_shutdown_signals": recording_shutdown_signals,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    run = namespace["run_live_realtime_backend"]
    if outcome in {"exec_error", "signal_then_exec_error", "constructor_error"}:
        with pytest.raises((OSError, ValueError), match="synthetic"):
            run(recording_sink=object())
    else:
        assert run(recording_sink=object()) == 42
    assert installed == original
    expected = [] if outcome == "constructor_error" else [{"finalize_recording": outcome not in {"exec_error", "signal_then_exec_error"}}]
    assert calls == [*expected, "release"]

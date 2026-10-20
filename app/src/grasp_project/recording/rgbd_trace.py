"""RGB-D software execution journal; imports no ROS, Qt or sensor backend.

FSM replay uses the production state machine. Tracker snapshots and commands
are recorded evidence, not independent labels or physical validation.
"""

import copy
import functools
import importlib.util
import signal
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_WORKER_FIELDS = (
    "last_full_eval_time", "latest_report", "last_force_closure", "last_selected_fraction",
    "post_grasp_active", "post_grasp_baseline", "post_grasp_stable_frames",
    "last_evaluated_frame_id", "evaluation_times", "evaluation_start_times",
    "locked_fast_frames_since_full", "locked_full_recovery_failures",
    "last_output_fps", "last_started_fps", "force_lock_next_eval",
    "pending_lock_command_id", "last_lock_debug_time",
)
_MONITOR_FIELDS = (
    "enabled", "configured_roi", "roi_xywh", "sensitivity", "thresholds",
    "active", "backend_name", "backend_available", "unavailable_reason",
    "device_open_error", "start_time", "last_report", "events",
    "event_time_zero_us", "event_time_wall_zero", "event_device_last_us",
    "event_normalized_last", "event_timestamp_reanchor_count",
    "latest_event_host_arrival_monotonic", "latest_event_host_arrival_epoch",
    "latest_aps_device_timestamp_us",
)
_REPORT_FIELDS = (
    "frame_id", "state", "stage", "stage_decision", "decision", "recommended_action",
    "safety_score", "failure_reasons", "ready_blockers", "ready_to_close", "ready_to_lift",
    "post_grasp_active", "manual_gripper_closed", "gripper_closed_confirmed",
    "target_status", "target_lock", "target_lock_progress", "planned_contacts",
    "locked_grasp_band", "locked_grasp_band_3d", "bottle_geometry",
    "epsilon", "raw_epsilon", "epsilon_score", "epsilon_force_closure", "epsilon_result",
    "state_machine", "state_history_event", "writer_session_id", "attempt_id",
    "lock_frame_id", "quality_compute_id", "post_grasp_baseline", "pointcloud_status",
)


def _fsm_owner():
    """Load the one canonical flat-file FSM even in src-only CLI imports."""
    name = "_grasp_project_recording_canonical_fsm"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[3] / "state_machine.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError("canonical state_machine.py is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


@contextmanager
def recording_shutdown_signals(app, *, enabled):
    """Let an explicitly enabled recording finish normal Qt worker teardown."""
    previous = {}
    try:
        if enabled:
            for number in (signal.SIGTERM, signal.SIGINT):
                previous[number] = signal.getsignal(number)
                signal.signal(number, lambda _number, _frame: app.quit())
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        name = f"state_{len(arrays):05d}"
        arrays[name] = value.copy()
        return {"__ndarray__": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, deque):
        return {"__deque__": [_pack(item, arrays) for item in value], "maxlen": value.maxlen}
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(item, arrays) for item in value]}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, dict):
        return {key: _pack(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack(item, arrays) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported trace state type: {type(value).__name__}")


def unpack_trace_value(value, arrays):
    """Restore typed state data from a verified recording's array references."""
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            name = value["__ndarray__"]
            if name not in arrays or not isinstance(arrays[name], np.ndarray):
                raise ValueError("missing trace state array")
            return arrays[name].copy()
        if set(value) == {"__deque__", "maxlen"}:
            return deque((unpack_trace_value(item, arrays) for item in value["__deque__"]), maxlen=value["maxlen"])
        if set(value) == {"__tuple__"}:
            return tuple(unpack_trace_value(item, arrays) for item in value["__tuple__"])
        if set(value) == {"__path__"}:
            return Path(value["__path__"])
        return {key: unpack_trace_value(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack_trace_value(item, arrays) for item in value]
    return value


def capture_worker_state(worker, arrays):
    """Capture complete TargetLockManager fields and explicit worker state.

    Device handles, Qt signals and callbacks are deliberately excluded. NumPy
    state is stored losslessly as arrays, including masks, dtype and nonfinite
    values. This snapshot is a software continuation input, never ground truth.
    """
    manager = getattr(worker, "lock_manager", None)
    monitor = getattr(worker, "event_monitor", None)
    return {
        "machine": _pack({"state": worker.machine.state, "history": worker.machine.history}, arrays),
        "worker": _pack({name: getattr(worker, name) for name in _WORKER_FIELDS if hasattr(worker, name)}, arrays),
        "lock_manager": _pack({
            name: value for name, value in vars(manager).items()
            if name != "lock_artifact_invalidator"
        } if manager is not None else {}, arrays),
        "event_monitor": _pack({
            name: getattr(monitor, name) for name in _MONITOR_FIELDS if hasattr(monitor, name)
        } if monitor is not None else {}, arrays),
        "excluded_resources": ["device_handles", "qt_signals", "callbacks"],
    }


def restore_worker_state(worker, snapshot, arrays):
    """Restore software fields into a caller-created worker with safe backends.

    This does not construct a worker or start a monitor. The offline driver
    must supply a monitor that cannot open hardware and isolate output paths.
    """
    state = unpack_trace_value(snapshot, arrays)
    if state["machine"]["state"] not in _fsm_owner().STATES:
        raise ValueError("unknown recorded FSM state")
    worker.machine.state = state["machine"]["state"]
    worker.machine.history = state["machine"]["history"]
    for name, value in state["worker"].items():
        if name not in _WORKER_FIELDS:
            raise ValueError("unknown worker state field")
        setattr(worker, name, value)
    for name, value in state["lock_manager"].items():
        if name == "lock_artifact_invalidator" or not hasattr(worker.lock_manager, name):
            raise ValueError("unknown or unsafe lock-manager state field")
        setattr(worker.lock_manager, name, value)
    for name, value in state["event_monitor"].items():
        if name not in _MONITOR_FIELDS:
            raise ValueError("unknown monitor state field")
        setattr(worker.event_monitor, name, value)


def report_contract(report):
    """Select substantive runtime outputs; retain provenance for comparison.

    Clock/UUID diagnostics are retained where embedded in these fields. A
    caller must disclose any normalization used for deterministic comparisons.
    """
    selected = {name: report[name] for name in _REPORT_FIELDS if name in report}
    if "realtime_pipeline_audit" in report:
        audit = report["realtime_pipeline_audit"]
        selected["realtime_pipeline_audit"] = {name: audit[name] for name in (
            "evaluation_branch", "full_eval_reason", "manual_force_lock_requested", "manual_lock_command_id",
            "due_full_eval", "lock_active", "locked_full_recovery_failures_before", "locked_full_recovery_failures_after",
        ) if name in audit}
    return copy.deepcopy(selected)


def runtime_trace_context(namespace):
    """Retain effective primitive globals and mutable runtime references."""
    settings = {}
    for name, value in namespace.items():
        if name.isupper() and (value is None or type(value) in (str, int, float, bool)):
            settings[name] = value
        elif name.isupper() and isinstance(value, (dict, list, tuple)):
            try:
                arrays = {}
                packed = _pack(value, arrays)
                if not arrays:
                    settings[name] = packed
            except ValueError:
                continue
    return settings


class RgbdTraceRecorder:
    """Ordered journal on the same thread-safe writer as raw acquisition."""

    def __init__(self, sink, worker, *, constructor, runtime_context, external_inputs=None):
        self.sink = sink
        self.trace_id = uuid.uuid4().hex
        self.sequence = 0
        self.frame_id = None
        self._lock = threading.RLock()
        arrays = {}
        initial = capture_worker_state(worker, arrays)
        self.emit("initial", {
            "constructor": constructor, "runtime_context": runtime_context, "worker": initial,
            "external_inputs": {} if external_inputs is None else external_inputs,
        }, arrays)

    @staticmethod
    def clock():
        return {"epoch_sec": time.time(), "monotonic_sec": time.monotonic(), "perf_counter_sec": time.perf_counter()}

    def emit(self, event, data, arrays=None):
        with self._lock:
            payload_arrays = {} if arrays is None else dict(arrays)
            data = _pack(data, payload_arrays)
            descriptor = self.sink.record(
                "rgbd_trace", payload_arrays,
                observation_id=f"rgbd-{self.trace_id}-{self.sequence}",
                host_arrival_monotonic_ns=time.monotonic_ns(),
                host_arrival_epoch_ns=time.time_ns(),
                native_clock={"domain": "application_trace_order", "unit": "sequence"},
                metadata={"schema_version": 1, "trace_id": self.trace_id,
                          "trace_sequence": self.sequence, "event": event, "data": data},
                source_sequence=self.sequence,
            )
            self.sequence += 1
            return descriptor

    def begin_frame(self, worker, rgb, depth, intrinsics, frame_id, *, runtime_configuration=None):
        if type(frame_id) is not int or frame_id < 1:
            raise ValueError("frame_id must be a positive integer")
        arrays = {"rgb": rgb, "depth": depth}
        state = capture_worker_state(worker, arrays)
        self.frame_id = frame_id
        return self.emit("frame", {
            "frame_id": frame_id, "intrinsics": intrinsics, "worker_before": state, "clock": self.clock(),
            "runtime_configuration": {} if runtime_configuration is None else runtime_configuration,
        }, arrays)

    def transition(self, machine, stage_result, *, frame_id, provenance):
        before = machine.state
        event = machine.record(stage_result)
        self.emit("stage_transition", {
            "frame_id": frame_id, "before_state": before, "stage_result": stage_result,
            "event": event, "history_length": len(machine.history), "provenance": provenance,
        })
        return event

    def output(self, worker, report):
        arrays = {}
        state = capture_worker_state(worker, arrays)
        return self.emit("output", {
            "frame_id": report.get("frame_id", self.frame_id), "report": report_contract(report), "worker_after": state,
        }, arrays)


def trace_worker_command(function):
    """Journal an actual worker command without changing its return behavior."""
    @functools.wraps(function)
    def wrapped(worker, *args, **kwargs):
        trace = getattr(worker, "rgbd_trace", None)
        if trace is None:
            return function(worker, *args, **kwargs)
        arrays = {}
        before = capture_worker_state(worker, arrays)
        clock = trace.clock()
        try:
            result = function(worker, *args, **kwargs)
        except Exception as exc:
            trace.emit("command", {
                "method": function.__name__, "args": list(args), "kwargs": kwargs,
                "exception": repr(exc), "worker_before": before, "clock": clock,
                "sensor_measured": False,
            }, arrays)
            if hasattr(trace.sink, "abort"):
                trace.sink.abort(f"rgbd_worker_command_{function.__name__}_failed")
            raise
        after = capture_worker_state(worker, arrays)
        trace.emit("command", {
            "method": function.__name__, "args": list(args), "kwargs": kwargs,
            "result": result, "worker_before": before, "worker_after": after, "clock": clock,
            "operator_declaration": function.__name__ in {"_mark_gripper_closed_result", "mark_gripper_open"},
            "sensor_measured": False,
        }, arrays)
        return result
    return wrapped


class TracedEventMonitor:
    """Record exact external monitor replies; replay must supply a safe stub."""

    def __init__(self, monitor, trace):
        self._monitor = monitor
        self._trace = trace

    def __getattr__(self, name):
        return getattr(self._monitor, name)

    def _call(self, method, *args, **kwargs):
        clock = self._trace.clock()
        try:
            result = getattr(self._monitor, method)(*args, **kwargs)
        except Exception as exc:
            self._trace.emit("event_monitor", {
                "method": method, "args": list(args), "kwargs": kwargs,
                "exception": repr(exc), "clock": clock,
            })
            if hasattr(self._trace.sink, "abort"):
                self._trace.sink.abort(f"event_monitor_{method}_failed")
            raise
        self._trace.emit("event_monitor", {
            "method": method, "args": list(args), "kwargs": kwargs,
            "result": result, "clock": clock,
        })
        return result

    def start(self, *args, **kwargs):
        return self._call("start", *args, **kwargs)

    def update(self, *args, **kwargs):
        return self._call("update", *args, **kwargs)

    def stop(self, *args, **kwargs):
        return self._call("stop", *args, **kwargs)


def replay_rgbd_trace(records):
    """Verify only the recorded stage transitions with the actual FSM.

    Image tracking and worker-command replay are separate components. Their
    captured context is never silently promoted into verified FSM inputs.
    """
    machine = None
    trace_id = None
    sequence = 0
    transitions = []
    context_counts = {}
    for record in records:
        if record.get("kind") != "rgbd_trace":
            continue
        metadata = record["metadata"]
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported RGB-D trace schema")
        if trace_id is None:
            trace_id = metadata["trace_id"]
        if metadata["trace_id"] != trace_id or metadata["trace_sequence"] != sequence:
            raise ValueError("RGB-D trace ID or ordering mismatch")
        sequence += 1
        event = metadata["event"]
        data = unpack_trace_value(metadata["data"], record["arrays"])
        context_counts[event] = context_counts.get(event, 0) + 1
        if event == "initial":
            if machine is not None or sequence != 1:
                raise ValueError("RGB-D trace must have exactly one initial state")
            initial = data["worker"]["machine"]
            owner = _fsm_owner()
            if initial["state"] not in owner.STATES:
                raise ValueError("unknown initial FSM state")
            machine = owner.GraspStateMachine(state=initial["state"], history=copy.deepcopy(initial["history"]))
        elif machine is None:
            raise ValueError("RGB-D trace missing initial state")
        elif event == "stage_transition":
            if machine.state != data["before_state"]:
                raise ValueError("RGB-D FSM before-state mismatch")
            actual = machine.record(data["stage_result"])
            comparable = lambda value: {key: item for key, item in value.items() if key != "timestamp"}
            if comparable(actual) != comparable(data["event"]) or len(machine.history) != data["history_length"]:
                raise ValueError("RGB-D FSM transition parity mismatch")
            transitions.append({"frame_id": data["frame_id"], **comparable(actual)})
        elif event not in {"frame", "command", "output", "command_request", "command_result", "event_monitor"}:
            raise ValueError("unknown RGB-D trace event")
    return {
        "schema_version": 1, "trace_id": trace_id, "verified_fsm": machine is not None,
        "state": None if machine is None else machine.state, "transitions": transitions,
        "context_counts": context_counts, "physical_validation": False,
        "verification_scope": "production_fsm_on_recorded_stage_inputs",
        "worker_and_image_tracker_parity": "NOT_ASSESSED_BY_FSM_COMPONENT",
        "normalization": "FSM event timestamp excluded; state/action/score/reasons retained",
    }

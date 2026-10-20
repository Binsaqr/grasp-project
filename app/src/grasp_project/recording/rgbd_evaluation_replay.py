"""Canonical RGB-D worker replay with recorded clocks and isolated file output.

Requires the supported ROS/Qt runtime, but constructs no camera, robot, GUI or
worker thread. Asynchronous DAVIS and point-cloud replies are recorded external
inputs; this driver does not claim to recompute those components.
"""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasp_project.recording.rgbd_trace import (
    report_contract,
    restore_worker_state,
    unpack_trace_value,
)
from grasp_project.recording.session import Recording, _file_digest, read_recording

_REPLAY_LOCK = threading.RLock()
_COMMANDS = frozenset(
    {
        "_mark_gripper_closed_result",
        "mark_gripper_open",
        "force_lock_current_target",
        "force_lock_current_target_for_command",
        "reset_lock_from_dashboard",
        "start_approach_mode",
        "attach_locked_quality_snapshot",
    }
)
# Exact production bookkeeping paths only. Acquisition timestamp fields at any
# other path remain comparison inputs, including nested native-clock objects.
_GENERATED_REPORT_PATHS = frozenset(
    {
        ("state_history_event", "timestamp"),
        ("state_machine", "updated_at"),
        ("target_lock", "last_lock_event", "event_id"),
        ("target_lock", "last_relock_event", "event_id"),
    }
)


class ReplayClock:
    """One explicit clock boundary per actual evaluation or command."""

    def __init__(self):
        self.set({"epoch_sec": 1.0, "monotonic_sec": 1.0, "perf_counter_sec": 1.0})

    def set(self, values):
        for name in ("epoch_sec", "monotonic_sec", "perf_counter_sec"):
            value = values.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"recorded clock needs finite nonnegative {name}")
            setattr(self, name, float(value))

    def proxy(self):
        return SimpleNamespace(
            time=lambda: self.epoch_sec,
            time_ns=lambda: round(self.epoch_sec * 1e9),
            monotonic=lambda: self.monotonic_sec,
            monotonic_ns=lambda: round(self.monotonic_sec * 1e9),
            perf_counter=lambda: self.perf_counter_sec,
            perf_counter_ns=lambda: round(self.perf_counter_sec * 1e9),
            sleep=lambda _seconds: None,
            strftime=time.strftime,
            gmtime=time.gmtime,
            localtime=time.localtime,
        )

    def utc(self):
        return datetime.fromtimestamp(self.epoch_sec, timezone.utc)


class _RecordedMonitor:
    """Replies only from a journal; no method can discover or open a device."""

    def __init__(self, calls=()):
        self.calls = list(calls)
        self.position = 0
        self.errors = []
        self.enabled = False
        self.active = False
        self.last_report = {}

    def _call(self, method, *args, **kwargs):
        if self.position >= len(self.calls):
            self.errors.append(f"missing recorded event-monitor {method} reply")
            raise ValueError(f"missing recorded event-monitor {method} reply")
        entry = self.calls[self.position]
        if entry.get("method") != method or "exception" in entry:
            self.errors.append(f"recorded event-monitor call mismatch: {method}")
            raise ValueError(f"recorded event-monitor call mismatch: {method}")
        if _differences(
            _normal(entry.get("args", [])), _normal(list(args))
        ) or _differences(_normal(entry.get("kwargs", {})), _normal(kwargs)):
            self.errors.append(f"recorded event-monitor arguments mismatch: {method}")
            raise ValueError(f"recorded event-monitor arguments mismatch: {method}")
        self.position += 1
        result = copy.deepcopy(entry.get("result"))
        if method == "stop":
            self.active = False
        else:
            self.active = True
            self.last_report = copy.deepcopy(result or {})
        return result

    def start(self, *args, **kwargs):
        return self._call("start", *args, **kwargs)

    def update(self, *args, **kwargs):
        return self._call("update", *args, **kwargs)

    def stop(self, *args, **kwargs):
        return self._call("stop", *args, **kwargs)


def _normal(value, *, scope="value", path=()):
    # This production field references an isolated generated output file, not
    # an acquisition input. Keep its artifact identity while removing only the
    # capture/replay directory, which necessarily differs between executions.
    if (
        scope == "report"
        and path
        and path[-1] == "full_lock_point_data_path"
        and isinstance(value, (str, Path))
        and PureWindowsPath(value).name not in ("", ".", "..")
    ):
        return {"output_artifact_name": PureWindowsPath(value).name}
    if isinstance(value, np.ndarray):
        return _normal(value.tolist(), scope=scope, path=path)
    if isinstance(value, np.generic):
        return _normal(value.item(), scope=scope, path=path)
    if isinstance(value, dict):
        return {
            key: _normal(item, scope=scope, path=(*path, key))
            for key, item in value.items()
            if not (
                scope == "report"
                and (*path, key) in _GENERATED_REPORT_PATHS
                or scope == "transition"
                and not path
                and key == "timestamp"
            )
        }
    if isinstance(value, (list, tuple)):
        return [
            _normal(item, scope=scope, path=(*path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _clock_value_path(path):
    parts = path.replace("[", ".").replace("]", "").split(".")
    return any(
        "timestamp" in part
        or part
        in {
            "clock",
            "clocks",
            "native_clock",
            "host_clock",
            "clock_inputs",
            "acquisition",
            "epoch_sec",
            "monotonic_sec",
            "perf_counter_sec",
        }
        or part.startswith("host_arrival_")
        for part in parts
    )


def _differences(expected, actual, path="", limit=40):
    result = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                result.append({"path": child, "reason": "field_presence"})
            else:
                result.extend(_differences(expected[key], actual[key], child, limit))
            if len(result) >= limit:
                return result[:limit]
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            result.append({"path": path, "reason": "length"})
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(_differences(left, right, f"{path}[{index}]", limit))
            if len(result) >= limit:
                return result[:limit]
    elif type(expected) is int and type(actual) is int:
        if expected != actual:
            result.append(
                {
                    "path": path,
                    "reason": "integer_value",
                    "expected": expected,
                    "actual": actual,
                }
            )
    elif (
        type(expected) in (int, float)
        and type(actual) in (int, float)
        and (type(expected) is int or type(actual) is int)
    ):
        if expected != actual:
            result.append(
                {
                    "path": path,
                    "reason": "numeric_representation",
                    "expected": expected,
                    "actual": actual,
                }
            )
    elif type(expected) is float and type(actual) is float:
        clock_value = _clock_value_path(path)
        if (
            expected != actual
            if clock_value
            else not math.isclose(expected, actual, rel_tol=1e-7, abs_tol=1e-6)
        ):
            result.append(
                {
                    "path": path,
                    "reason": "clock_value" if clock_value else "numeric_value",
                    "expected": expected,
                    "actual": actual,
                }
            )
    elif type(expected) is not type(actual) or expected != actual:
        result.append(
            {"path": path, "reason": "value", "expected": expected, "actual": actual}
        )
    return result


def _setting_safe(value):
    if isinstance(value, np.generic):
        return _setting_safe(value.item())
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_setting_safe(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _setting_safe(item) for key, item in value.items()
        )
    return False


def _command_state(machine, worker, manager):
    locked = manager.get("locked_target") or {}
    return {
        "machine_state": machine["state"],
        "machine_history_length": len(machine["history"]),
        "post_grasp_active": worker.get("post_grasp_active"),
        "force_lock_next_eval": worker.get("force_lock_next_eval"),
        "pending_lock_command_id": worker.get("pending_lock_command_id"),
        "lock_enabled": manager.get("enabled"),
        "lock_valid": locked.get("valid") is True,
        "attempt_id": locked.get("attempt_id"),
        "locked_quality_compute_id": locked.get("locked_quality_compute_id"),
        "locked_quality_score": locked.get("locked_quality_score"),
        "baseline_frozen": locked.get("baseline_frozen"),
    }


@contextmanager
def evaluation_context(
    runtime, initial, *, clock, monitor, output_dir, external_for_frame
):
    """Contain canonical evaluator side effects without replacing its writer.

    Archived configuration replaces profile/embodiment reads. Recorded async
    point-cloud replies replace background scheduling. The actual status writer,
    report enrichment, geometry, scoring and machine.record remain active.
    This process-global context must not run alongside a live backend.
    """
    import bottle_pose
    import pregrasp_planner
    import state_machine

    external = initial.get("external_inputs", {})
    missing = {"bottle_profile", "grasp_profile", "embodiment_status"} - set(external)
    if missing:
        raise ValueError(
            f"missing recorded RGB-D external configuration: {sorted(missing)}"
        )
    settings = unpack_trace_value(initial.get("runtime_context", {}), {})
    if type(settings.get("USE_YOLO_BOTTLE_TRIGGER")) is not bool:
        raise ValueError(
            "missing recorded detector configuration USE_YOLO_BOTTLE_TRIGGER"
        )
    if settings["USE_YOLO_BOTTLE_TRIGGER"]:
        raise ValueError(
            "YOLO-enabled evaluation replay requires archived model/backend support"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_output = Path(runtime.OUTPUT_DIR).resolve()
    with _REPLAY_LOCK, ExitStack() as stack:
        # Snapshot mutable globals so repeated replay cannot inherit caches or
        # modify the caller's counters. Noncopyable thread resources stay idle.
        for name, value in list(vars(runtime).items()):
            if name.isupper():
                try:
                    saved = copy.deepcopy(value)
                except (TypeError, ValueError):
                    continue
                stack.callback(setattr, runtime, name, value)
                setattr(runtime, name, saved)
        for name, value in settings.items():
            if (
                not name.isupper()
                or not hasattr(runtime, name)
                or not _setting_safe(getattr(runtime, name))
                or not _setting_safe(value)
            ):
                raise ValueError(f"unknown recorded runtime setting: {name}")
            setattr(runtime, name, copy.deepcopy(value))
        # Only local project modules are changed, and every patch is restored.
        project = Path(runtime.__file__).resolve().parents[3]
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            try:
                Path(filename).resolve().relative_to(project)
            except (ValueError, OSError):
                continue
            if (
                module is not sys.modules[__name__]
                and getattr(module, "time", None) is time
            ):
                stack.enter_context(patch.object(module, "time", clock.proxy()))
            for name, value in list(vars(module).items()):
                if isinstance(value, Path):
                    try:
                        relative = value.resolve().relative_to(source_output)
                    except (ValueError, OSError):
                        continue
                    stack.enter_context(
                        patch.object(module, name, output_dir / relative)
                    )
        stack.enter_context(patch.object(runtime, "time", clock.proxy()))
        stack.enter_context(
            patch.object(
                runtime,
                "utc_now",
                lambda: clock.utc().replace(tzinfo=None).isoformat() + "Z",
            )
        )
        stack.enter_context(
            patch.object(state_machine, "utc_now_iso", lambda: clock.utc().isoformat())
        )
        identities = iter(range(1_000_000))
        stack.enter_context(
            patch.object(
                runtime,
                "uuid",
                SimpleNamespace(
                    uuid4=lambda: uuid.uuid5(
                        uuid.NAMESPACE_URL, f"rgbd-replay-{next(identities)}"
                    )
                ),
            )
        )
        stack.enter_context(
            patch.object(runtime, "EventPostGraspMonitor", lambda **_kwargs: monitor)
        )
        stack.enter_context(
            patch.object(
                bottle_pose,
                "load_known_bottle_profile",
                lambda *_a, **_k: copy.deepcopy(external["bottle_profile"]),
            )
        )
        stack.enter_context(
            patch.object(
                pregrasp_planner,
                "load_known_bottle_profile",
                lambda *_a, **_k: copy.deepcopy(external["grasp_profile"]),
            )
        )
        stack.enter_context(
            patch.object(
                pregrasp_planner, "active_object_profile_override", lambda: None
            )
        )
        stack.enter_context(
            patch.object(
                runtime,
                "current_embodiment_status",
                lambda: copy.deepcopy(external["embodiment_status"]),
            )
        )
        metrics = external.get("metric_applicability")
        if metrics is None:
            # Applicability depends on the embodiment model; require its
            # captured result rather than silently consulting current config.
            raise ValueError("missing recorded RGB-D metric_applicability")
        stack.enter_context(
            patch.object(
                runtime,
                "quality_metric_applicability",
                lambda *_a, **_k: copy.deepcopy(metrics),
            )
        )
        stack.enter_context(
            patch.object(
                runtime,
                "live_embodiment_status_and_metrics",
                lambda: (
                    copy.deepcopy(external["embodiment_status"]),
                    copy.deepcopy(metrics),
                    False,
                ),
            )
        )

        def pointcloud_reply(*_args, **_kwargs):
            reply = external_for_frame.get("pointcloud_status")
            if not isinstance(reply, dict):
                raise TypeError("missing recorded async pointcloud_status")
            return copy.deepcopy(reply)

        stack.enter_context(
            patch.object(runtime, "start_pointcloud_update_async", pointcloud_reply)
        )
        yield


def _trace_records(recording):
    records = recording.records if isinstance(recording, Recording) else recording
    entries, trace_id = [], None
    for record in records:
        if record.get("record_type") == "drop" and record.get("stream") == "rgbd_trace":
            raise ValueError("RGB-D execution trace contains a dropped observation")
        if record.get("kind") != "rgbd_trace":
            continue
        metadata = record["metadata"]
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported RGB-D execution trace schema")
        if trace_id is None:
            trace_id = metadata.get("trace_id")
        if metadata.get("trace_id") != trace_id or metadata.get(
            "trace_sequence"
        ) != len(entries):
            raise ValueError("RGB-D execution trace ordering mismatch")
        entries.append(
            (
                record,
                metadata["event"],
                unpack_trace_value(metadata["data"], record["arrays"]),
            )
        )
    if (
        not entries
        or entries[0][1] != "initial"
        or sum(event == "initial" for _, event, _ in entries) != 1
    ):
        raise ValueError("RGB-D evaluation replay requires exactly one initial event")
    return entries, trace_id


def replay_rgbd_evaluations(recording) -> dict:
    """Execute actual worker inputs/commands and compare recorded golden fields.

    Boundary clocks are frozen within each call. Explicit FSM bookkeeping paths
    and lock-transition UUID paths are excluded; all selected score, decision,
    validity, reason, geometry and state fields remain checked. Near-threshold
    timing differences are reported as mismatches, never silently accepted.
    """
    if isinstance(recording, (str, Path)):
        recording = read_recording(recording)
    entries, trace_id = _trace_records(recording)
    try:
        from grasp_project.realsense import realtime_runner as runtime
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "RGB-D worker replay requires the supported Linux ROS Humble/PyQt5 runtime"
        ) from exc

    initial_record, _, initial = entries[0]
    constructor = initial.get("constructor", {})
    if constructor.get("restore_grasp_memory") is not False:
        raise ValueError(
            "replay requires explicitly disabled constructor memory restoration"
        )
    if set(constructor) != {
        "lock_target",
        "restore_grasp_memory",
        "publish_lock_artifacts",
    } or any(type(value) is not bool for value in constructor.values()):
        raise ValueError(
            "missing, unknown or invalid EvaluationWorker constructor option"
        )
    clock = ReplayClock()
    monitor = _RecordedMonitor(
        data for _, event, data in entries if event == "event_monitor"
    )
    output_entries = [data for _, event, data in entries if event == "output"]
    outputs_by_frame = {data["frame_id"]: data for data in output_entries}
    if len(outputs_by_frame) != len(output_entries):
        raise ValueError("duplicate RGB-D golden output frame ID")
    frame_externals = {}
    mismatches, outputs, commands, transitions = [], [], [], []
    pending_frame = None
    with (
        TemporaryDirectory(prefix="grasp-rgbd-replay-") as temporary,
        evaluation_context(
            runtime,
            initial,
            clock=clock,
            monitor=monitor,
            output_dir=temporary,
            external_for_frame=frame_externals,
        ),
    ):
        worker = runtime.EvaluationWorker(**constructor)
        restore_worker_state(
            worker,
            initial_record["metadata"]["data"]["worker"],
            initial_record["arrays"],
        )
        # Recorded attempt IDs are identity inputs for asynchronous analytic
        # completion commands, never outcome labels or scores.
        original_attempt_id = runtime.make_attempt_id

        def attempt_id(frame_id):
            report = outputs_by_frame.get(frame_id, {}).get("report", {})
            target = report.get("target_lock", {}).get("locked_target") or {}
            return (
                target.get("attempt_id")
                or report.get("attempt_id")
                or original_attempt_id(frame_id)
            )

        with patch.object(runtime, "make_attempt_id", attempt_id):
            for record, event, data in entries[1:]:
                if event == "frame":
                    if pending_frame is not None:
                        raise ValueError("RGB-D frame has no matching output")
                    frame_id = data["frame_id"]
                    if frame_id not in outputs_by_frame:
                        raise ValueError("RGB-D frame missing golden output")
                    rgb, depth = (
                        record["arrays"].get("rgb"),
                        record["arrays"].get("depth"),
                    )
                    if (
                        not isinstance(rgb, np.ndarray)
                        or not isinstance(depth, np.ndarray)
                        or rgb.ndim != 3
                        or rgb.shape[2] != 3
                        or rgb.dtype != np.uint8
                        or depth.dtype != np.uint16
                        or depth.shape != rgb.shape[:2]
                    ):
                        raise ValueError(
                            "RGB-D replay needs canonical uint8 RGB and aligned uint16 depth"
                        )
                    clock.set(data["clock"])
                    configuration = data.get("runtime_configuration")
                    if (
                        not isinstance(configuration, dict)
                        or set(configuration) != {"HAND_MOUNTED_CAMERA_CONFIG"}
                        or not isinstance(
                            configuration["HAND_MOUNTED_CAMERA_CONFIG"], dict
                        )
                        or not _setting_safe(configuration)
                    ):
                        raise ValueError(
                            "missing or invalid per-frame hand-camera configuration"
                        )
                    runtime.HAND_MOUNTED_CAMERA_CONFIG = copy.deepcopy(
                        configuration["HAND_MOUNTED_CAMERA_CONFIG"]
                    )
                    frame_externals.clear()
                    frame_externals.update(outputs_by_frame[frame_id]["report"])
                    if not isinstance(frame_externals.get("pointcloud_status"), dict):
                        raise ValueError("missing recorded async pointcloud_status")
                    worker.evaluate(
                        rgb.copy(),
                        depth.copy(),
                        copy.deepcopy(data["intrinsics"]),
                        frame_id,
                    )
                    pending_frame = frame_id
                elif event == "command":
                    method = data.get("method")
                    if method not in _COMMANDS or pending_frame is not None:
                        raise ValueError(
                            "unsupported or interleaved RGB-D worker command"
                        )
                    clock.set(data["clock"])
                    result = getattr(worker, method)(
                        *copy.deepcopy(data["args"]), **copy.deepcopy(data["kwargs"])
                    )
                    differences = _differences(
                        _normal(data.get("result")), _normal(result)
                    )
                    expected_state = data["worker_after"]
                    actual_state = _command_state(
                        {
                            "state": worker.machine.state,
                            "history": worker.machine.history,
                        },
                        vars(worker),
                        vars(worker.lock_manager),
                    )
                    differences.extend(
                        _differences(
                            _command_state(
                                expected_state["machine"],
                                expected_state["worker"],
                                expected_state["lock_manager"],
                            ),
                            actual_state,
                            "worker_state",
                        )
                    )
                    mismatches.extend(
                        {"event": "command", "method": method, **item}
                        for item in differences
                    )
                    commands.append(
                        {
                            "method": method,
                            "result": _normal(result),
                            "state": actual_state,
                            "matched": not differences,
                        }
                    )
                elif event == "output":
                    if pending_frame != data["frame_id"]:
                        raise ValueError(
                            "RGB-D output does not match pending evaluation"
                        )
                    actual = _normal(
                        report_contract(worker.latest_report or {}), scope="report"
                    )
                    differences = _differences(
                        _normal(data["report"], scope="report"), actual
                    )
                    mismatches.extend(
                        {"event": "output", "frame_id": pending_frame, **item}
                        for item in differences
                    )
                    outputs.append(
                        {
                            "frame_id": pending_frame,
                            "report": actual,
                            "matched": not differences,
                        }
                    )
                    pending_frame = None
                elif event == "stage_transition":
                    if pending_frame != data["frame_id"]:
                        raise ValueError(
                            "stage transition outside matching RGB-D evaluation"
                        )
                    actual = (
                        worker.machine.history[-1] if worker.machine.history else None
                    )
                    differences = _differences(
                        _normal(data["event"], scope="transition"),
                        _normal(actual, scope="transition"),
                    )
                    if len(worker.machine.history) != data["history_length"]:
                        differences.append(
                            {"path": "history_length", "reason": "value"}
                        )
                    mismatches.extend(
                        {"event": event, "frame_id": data["frame_id"], **item}
                        for item in differences
                    )
                    transitions.append(
                        {"frame_id": data["frame_id"], "matched": not differences}
                    )
                elif event not in {
                    "event_monitor",
                    "command_request",
                    "command_result",
                }:
                    raise ValueError(f"unknown RGB-D execution event: {event}")
        if (
            pending_frame is not None
            or monitor.errors
            or monitor.position != len(monitor.calls)
        ):
            raise ValueError(
                "incomplete RGB-D evaluation or unconsumed monitor replies"
            )
        final_state = worker.machine.state
    return {
        "schema_version": "grasp_rgbd_evaluation_replay_v1",
        "trace_id": trace_id,
        "verified": bool(outputs) and not mismatches,
        "outputs": outputs,
        "commands": commands,
        "transitions": transitions,
        "mismatches": mismatches,
        "final_state": final_state,
        "scope": "canonical_EvaluationWorker_on_recorded_RGB_depth_intrinsics_and_commands",
        "external_replies": "recorded_DAVIS_monitor_async_pointcloud_and_analytic_completion_commands; not_recomputed",
        "normalization": "per_call_boundary_clocks; explicit_FSM_bookkeeping_and_lock_event_UUID_paths_excluded; full_lock_point_data_path_compared_by_output_artifact_basename; acquisition_clocks_compared_exactly; recorded_attempt_IDs_reused",
        "state_continuation": "initial_snapshot_restored_once; temporal_state_propagated_through_actual_worker_calls",
        "supported_path": "YOLO_disabled; canonical_arrays; archived_profiles_and_embodiment; recorded_async_replies",
        "unsupported_reasons": [],
        "clock_inputs": [
            data["clock"] for _, event, data in entries if event in {"frame", "command"}
        ],
        "implementation_source_sha256": {
            "realsense/realtime_runner.py": _file_digest(Path(runtime.__file__)),
            "recording/rgbd_evaluation_replay.py": _file_digest(Path(__file__)),
        },
        "ground_truth_used": False,
        "physical_validation": False,
    }


def replay_rgbd_evaluation(recording_dir) -> dict:
    """CLI-facing singular name; accepts the same verified recording inputs."""
    return replay_rgbd_evaluations(recording_dir)

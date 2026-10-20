# RGB-D execution journal

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

The canonical backend can record the actual image pairs submitted to
`EvaluationWorker.evaluate()`, state-machine inputs, operator commands, monitor
replies and selected outputs. This closes the gap between raw ROS capture and
the stateful inference inputs. Recording is opt-in and changes no scoring rule,
threshold, lock authority or actuation behavior.

For an operator-run recording, set all four variables before launching the
normal backend and dashboard processes:

```bash
export GRASP_RECORDING_ROOT=/absolute/path/to/new-session-directory
export GRASP_RECORDING_TRIAL_ID=trial-001
export GRASP_RECORDING_SESSION_ID=session-001
export GRASP_RECORDING_OBJECT_ID=bottle-001
```

The backend creates a new `rgbd` archive under that root. The dashboard owns a
separate process archive; each part shares the declared trial/session/object
identity. Existing archives are never overwritten. Constructor callers can
instead pass `recording_sink` to `RealtimeBackendRunner` or
`run_live_realtime_backend`; an explicitly supplied writer remains caller-owned.
The backend closes its owned writer only after every worker thread stops.
While recording is enabled, temporary SIGTERM/SIGINT handlers request normal Qt
shutdown and are restored afterwards. All workers are constructed and connected
before threads start. Failed construction, execution or worker teardown leaves
the archive incomplete and prevents finalization.

`rgbd_trace` records identify their trace ID, schema version, trace sequence and
event type. The journal includes:

| Event | Retained input or result |
| --- | --- |
| `initial` | Constructor arguments, FSM state/history, complete TargetLockManager fields, worker continuation fields, monitor configuration/report state, effective primitive runtime globals, writer identity, detector/grasp profiles and embodiment status. |
| `frame` | Actual processed RGB and depth arrays, frame ID, complete intrinsics/audit metadata, references to raw RGB/depth/CameraInfo observations, software state before evaluation, and host clock references. |
| `stage_transition` | Actual production `GraspStateMachine.record()` input/event, before state, history length, frame and lock provenance. |
| `command_request` / `command_result` | Backend command envelope before dispatch filtering and correlated acknowledgement. Queued acceptance is distinct from worker execution. |
| `command` | Actual worker handler, arguments, returned result, and before/after state. Recorded after the handler returns; its clock references are captured before execution. |
| `event_monitor` | Actual `start`, `update` or `stop` reply and call order. Missing event evidence remains unavailable; these replies are external inputs to RGB-D replay. |
| `output` | Substantive state, score, decision, reasons, validity, lock/geometry/provenance and point-cloud status, plus worker continuation state. |

Frame inputs also retain the effective mutable hand-camera configuration at
evaluation entry. This configuration is an explicit external input; replay does
not restore predicted tracker state from every frame. Output contracts retain
the full nested `epsilon_result`, including value, closure, validity and errors.

NumPy state, including masks, remains separate typed arrays. Deques retain their
maximum lengths, and tuples retain their types. Device handles, Qt signals and
callbacks are excluded. Restoring a snapshot never opens hardware: the caller
must create a worker with safe external adapters and isolated output paths.

Reset commands retain both the state before and after the actual reset handler.
The existing handler clears the target lock and post-grasp state; it does not
silently replace the FSM with an invented reset state. Close/open actions remain
operator declarations and are explicitly marked as not sensor measured.

`replay_rgbd_trace(records)` verifies the stage transitions with the real
production FSM. It compares the event fields and history length, excluding only
the generated UTC event timestamp. It rejects missing initial state, reordered
records, unsupported events and transition mismatches. Its verification scope is
explicitly `production_fsm_on_recorded_stage_inputs`; captured tracker outputs
and command contexts are not automatically claimed as image-tracker parity.
The canonical evaluator replay driver consumes the same frame/command journal
and reports its own supported scope and comparison results separately.

The evaluator comparator checks integer values and acquisition clock values
exactly, including float timestamps at large epoch origins. Geometry floats
retain the stated numerical tolerance. In report fields named
`full_lock_point_data_path`, it compares the generated output artifact basename
rather than the capture/replay temporary directory. A changed artifact name,
missing reference or changed geometry still fails comparison. Other file paths
are preserved. This portable reference checks identity, not artifact bytes.

There is no physical validation, independent annotation or measured stability
in a synthetic fixture. Software parity does not establish sensor accuracy,
grasp success, physical slip timing or sample adequacy.

Portable contract checks:

```bash
python -m pytest tests/test_rgbd_trace.py tests/test_ros_recording_adapter.py tests/test_recording_session.py -o addopts='' -q
```

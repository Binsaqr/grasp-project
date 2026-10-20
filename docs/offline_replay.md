# Component replay and matched-observation controls

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

```bash
python scripts/replay_recording.py path/to/complete_recording > output/replay.json
python scripts/replay_recording.py path/to/complete_recording --ablation events_disabled > output/replay_no_events.json
```

The reader validates the entire raw recording before inference. Replay opens no
camera and uses the canonical native DAVIS timestamp normalization, adaptive
window selector, `DavisMotionEstimator.estimate`, and
`LiveGraspQualityMonitor.update`. The former two accept recorded clocks/config
instead of the current machine's arrival time or environment settings.

The default mode is **component replay**. Raw ROS images and calibration are
preserved and listed as unsupported observations. The optional
`--rgbd-evaluation` mode replays the canonical RGB-D worker from a production
execution journal in the supported ROS/Qt runtime. See the
[trace contract](rgbd_trace.md) and [production capture guide](runtime_recording.md).
It uses actual evaluated pairs and ordered worker commands rather than guessing
which raw callbacks the live worker consumed. It does not reproduce GUI scheduling,
actuator state, physical timing or every asynchronous producer.
The native-capture command records raw inputs. It does not create the separately
recorded `live_quality_inputs` needed for quality replay.

`--output NEW_PATH` writes a new report outside the archive. Component contract
processing can succeed while predictions abstain. With `--rgbd-evaluation`, a
golden-output mismatch is retained in the report and returns exit code 1.

## Preserve the actual inference inputs

`EventPostGraspMonitor(..., recording_sink=writer)` captures arrays before native
microseconds become host-monotonic estimates. Application receipt is the return
of `device.get_event()`, sampled with integer host monotonic/epoch clocks. A
separate recorded floating-point normalization reference preserves the exact
legacy mapping input. It is never renamed as a device/hardware arrival time.
The normalization epoch changes on backend start/restart.

`LiveGraspQualityMonitor(config, recording_sink=writer)` records the complete
baseline/observation dataclasses and effective configuration before each update.
It also retains reset epochs and per-epoch input indices. Recording defaults off.
A monitor attached after earlier unrecorded updates has unknown prior state and
reference replay rejects it unless a genuine recorded reset establishes state.
Attach at construction for reproducible smoothing/hysteresis.

Quality inputs require explicit timestamps and complete dataclass fields; replay
never fills a missing timestamp with the current clock. Operator reset epochs
are replayed. A configuration change within a recording requires a new frozen
comparison arm rather than silently reusing one configuration identity.

The replay fingerprint includes the applied quality configuration, resolved
motion defaults, source configuration digest, replay source-file hashes, Python,
NumPy, OpenCV and platform. Acquisition source revision is retained separately.
The motion estimator's nondeterministic UUID and current wall-clock diagnostic
fields are replaced by stable observation identity and recorded host time.
Numerical outputs and classification are otherwise produced by the real core.
Repeatability is tested within a fixed implementation/numerical stack; no
cross-version bitwise equivalence or replay-speed measurement is claimed.

## Gaps and unavailable evidence

Known losses, unknown-size real polling failures, missing source sequences and
native clock resets invalidate motion continuity. An accounting notice that
native loss counters are unavailable is distinct from a measured loss. A positive
drop count is a loss regardless of its reason text. Motion continuity remains
qualified until a recorded normalization restart establishes a new epoch.
Quality input loss similarly remains invalid until an explicit recorded state
reset. Resetting an offline component does not prove the unrecorded live state
was recovered.

The existing estimator can emit diagnostic `motion_class: STABLE` for an empty
event window while `estimate_valid` is false. Replay retains that raw diagnostic
but publishes no valid prediction label or confidence for it. Missing events are
unavailable evidence. Quality outputs retain high baseline scores, decisions,
validity, use/exclusion flags and reasons together; a score does not override
missing evidence in the evaluator.

The reader loads arrays into memory and the replay retains history for all
configured estimator lookbacks. This implementation targets bounded recordings;
long-session throughput and memory limits require measurement before use.

## Predeclared offline controls

Each command uses the same recorded observation IDs and produces a distinct
fingerprinted configuration. None changes live configuration or thresholds.

| `--ablation` | Exact offline intervention |
|---|---|
| `reference` | Recorded inputs and production component policy |
| `events_disabled` | Quality event evidence marked unavailable; event-only motion outputs abstain |
| `sync_diagnostics_removed` | Remove cross-camera synchronization diagnostic fields, preserving native-event freshness/validity |
| `phase_conditioning_removed` | Apply the existing post-grasp branch to all recorded quality phases, explicitly as an offline control |

The current quality policy treats cross-camera skew as diagnostic, so removing
those diagnostics can leave scores unchanged. This control does not disable a
hardware clock alignment system, since none is established. Phase removal does
not mean an operator actually declared closure. For event-only estimator outputs,
sync/phase controls leave the estimator inputs unchanged; their intervention
applies to quality inputs. These are matched-input component controls, not
measured modality advantages or fair full-pipeline benchmarks.

The old live-runner `replay_fair` option is rejected before any sampling starts;
its dashboard button is disabled. That implementation sampled configurations
sequentially and must not label the result as fair recorded replay.

To evaluate, use `replay_predictions_document` with an explicit predeclared or
train/validation-selected configuration, task vocabulary and required evidence.
Then pass the result, independent annotations and grouped splits to the
[offline evaluator](offline_evaluation.md). The adapter generates no labels and
does not infer that configurations were frozen before test. The integration
regression exercises recording, validation, real inference, interchange and
held-out evaluation together, including abstention despite a high score.
For a complete command-line workflow, use the [aggregate experiment
CLI](recording_experiments.md), which runs selected or all arms directly from
archive parts and supplied study declarations without Python conversion glue.

No retained fixture establishes physical accuracy, sample adequacy, slip lead
time, hardware latency, successful grasping or autonomous actuation.

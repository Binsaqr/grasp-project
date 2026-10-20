# Raw recording contract

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

`grasp_project.recording` stores typed raw observations without opening a sensor.
It complements the existing bounded snapshot capture: snapshots retain published
outputs, while this contract retains arrays that can become offline inference
inputs. All tests use constructed arrays. Passing these tests establishes a file
contract; it supplies no physical validation, clock alignment, grasp success rate,
slip latency, or sample adequacy evidence.

## Minimal API

```python
import time
import numpy as np
from grasp_project.recording import RecordingWriter, read_recording

writer = RecordingWriter(
    "output/raw/example-trial",  # must not already exist
    recording_id="example-trial",
    config={"event_analyzer": {"window_ms": 40}},  # actual config contents
    revision={"commit": "actual-git-sha", "dirty": False},
    metadata={"trial_id": "example-trial", "source": "synthetic_example"},
)
writer.record(
    "events",
    arrays={
        "native_t": np.array([1000, 1002], dtype=np.int64),
        "x": np.array([20, 21], dtype=np.int16),
        "y": np.array([30, 30], dtype=np.int16),
        "polarity": np.array([1, 0], dtype=np.int8),
    },
    observation_id="events-000001",
    host_arrival_monotonic_ns=time.monotonic_ns(),
    host_arrival_epoch_ns=time.time_ns(),
    native_clock={"domain": "davis_device", "unit": "us", "epoch_known": False},
    source_sequence=1,
    metadata={"fixture": True},
)
writer.record_drop(
    "events", None, "source_does_not_expose_drop_count", time.monotonic_ns(),
)
writer.close()

recording = read_recording("output/raw/example-trial")
for item in recording.records:
    if item["record_type"] == "observation":
        native_t = item["arrays"].get("native_t")  # read-only NumPy array
```

The caller supplies revision provenance; the writer never invents a Git revision,
clean working tree, device timestamp, or sensor clock source. Record actual config
contents and an explicit unavailable revision value when provenance is unknown.
The initial config, revision and metadata are copied and individually hashed.
Mutating the caller's dictionaries after construction cannot alter the archive.
Each recording has a stable `recording_id`; `trial_id` comes from
`metadata.trial_id` or defaults to `recording_id`. Observation IDs must be unique.
Identifiers use ASCII letters, digits, `.`, `_`, and `-`, starting with a letter
or digit, with at most 120 characters.

## Clock and ordering semantics

Every observation requires these distinct fields:

| Field | Meaning |
| --- | --- |
| `host_arrival_monotonic_ns` | Caller-measured host callback/receipt endpoint; integer nanoseconds. |
| `host_arrival_epoch_ns` | Caller-measured wall clock at receipt; integer nanoseconds. Wall-clock adjustment is allowed. |
| `native_clock.domain` | Explicit source clock identifier; use an honest unknown/unspecified domain when needed. |
| `native_clock.unit` | Explicit unit of native timestamp values. |
| Other `native_clock` fields | Original finite JSON scalar timestamp and provenance metadata, if supplied. |

Native timestamp arrays are stored separately with their original NumPy dtype,
shape, byte order and values. Integer timestamps above `2**53` are preserved;
avoid converting them to floating point or combining them into a floating-point
event matrix before recording. Raw numeric payloads can contain NaN/Inf, which
are retained for downstream validity checks. JSON metadata must always be finite.
The contract never interprets NaN/Inf as a valid measurement.

`sequence` is contiguous append order across all observation and drop records.
Host monotonic arrival is nondecreasing within each stream (`kind`), including
drop records for that stream. Cross-stream callback order can differ from host
arrival order, and native timestamp order can reset or roll over. Neither case
establishes clock alignment. The manifest states `clock_alignment=not_estimated`.
When known, `source_sequence` must increase within its stream. A restarted source
requires a new recording or a separately identified stream.

## Loss and evidence availability

`record_drop(stream, count, reason, host_arrival_monotonic_ns)` records a known
positive count, or `count=None` when the quantity is unavailable. A nonempty
reason is mandatory. An unknown-count record can describe unavailable source
accounting; it does not assert that a physical drop occurred.

The summary preserves three independent quantities:

- `known_dropped_observations`: sum of explicit positive drop counts.
- `unknown_drop_events`: records whose count is unknown, including unavailable
  accounting notices; this is not an estimated number of dropped observations.
- `source_sequence_gaps`: directly observed gaps between declared source sequence
  numbers; the first source number does not infer earlier missing observations.

Do not add explicit drop counts to source sequence gaps: they can describe the
same missing observations. Zero explicit drops does not establish loss-free
capture when the source exposes no drop counter. An empty recording, absent
event arrays, or metadata-only phase observation provides no measured stability
evidence. Consumers must report that evidence as unavailable.

## Archive and validation

Each recording directory contains an atomic `manifest.json` and lossless numeric
or boolean `.npy` payloads under `arrays/`. Object, structured, and text dtypes
are rejected; named arrays retain heterogeneous types without coercion. Arrays
are copied when submitted and loaded with `allow_pickle=False`.
Payload filenames use short record/array ordinals while semantic array names
remain in the manifest, avoiding long names expanding Windows path lengths.

The manifest includes the versioned schema, recording and trial identity,
configuration/provenance snapshots, every record, payload byte size/SHA-256/dtype/
shape, record summaries, and a canonical manifest digest. These hashes detect
accidental corruption. They are not a signature or an authenticity guarantee:
someone who rewrites both data and checksums can create a different valid archive.

The reader validates the entire archive before returning any record: finite JSON,
duplicate JSON keys, nesting, identities, order, source gaps, per-stream host
monotonic order, config/revision/metadata hashes, summaries, relative paths, regular
non-symlink payload files, size/hash/dtype/shape, and trailing payload bytes.
This validation requires no prediction or label files. Missing or corrupt arrays
are rejected even if no ground truth exists.

The writer refuses an existing output directory, marks each persisted manifest
`incomplete`, and marks it `complete` only on explicit `close()`. Repeated close
is harmless; later writes fail. A write failure prevents finalization. An
interrupted process leaves an incomplete archive rejected by default.
`read_recording(path, require_complete=False)` permits forensic inspection only;
all other integrity checks still apply. It does not repair or resume a session.
Atomic replacement protects against a partial manifest write; this is not a
power-loss durability or storage hardware guarantee.

## Current limits and validation command

The synchronous writer rewrites the manifest after each observation and the reader
loads validated arrays into memory. Capture throughput, long-session scaling,
callback overhead, and physical sensor completeness remain unmeasured. This is a
bounded recording building block, not a sustained-throughput recorder claim.
Hardware capture must retain actual source metadata and explicitly available or
unknown loss accounting. Independent annotation and recorded-data evaluation are
separate steps; this archive never derives ground truth from predictions.

```powershell
$env:PYTHONPATH='src'
..\.venv\Scripts\python.exe -m pytest tests/test_recording_session.py -o addopts='' -q
```

## Explicit user-run native capture

`scripts/record_native_trial.py` connects this contract to the existing canonical
`EventPostGraspMonitor` backend and its opt-in recording sink. Running a capture
opens DAVIS. It does not operate a gripper or robot, start a RealSense driver,
reset a USB device, or generate labels. `--help` parses using the standard library
before importing sensor dependencies, and opens no hardware.

On the supported acquisition computer, the operator can first inspect the help:

```bash
python scripts/record_native_trial.py --help
```

An operator-run bounded DAVIS capture uses a new destination and explicit
acquisition identifiers:

```bash
python scripts/record_native_trial.py output/raw/session01-object01-trial01 \
  --recording-id session01-object01-trial01 \
  --session-id session01 --object-id object01 --duration-sec 20
```

To subscribe to RGB, aligned depth, and calibration, add `--with-ros` **after the
operator has started the existing D405 ROS producer** in the intended ROS domain.
The CLI uses installed `rclpy`/`sensor_msgs` and the existing sensor-free
`ros_capture` adapters; it never starts another sensor backend. Topic options
default to `/camera/camera/color/image_raw`,
`/camera/camera/aligned_depth_to_color/image_raw`, and
`/camera/camera/color/camera_info`. Override them for an existing publisher when
needed. The ROS header clock is declared `ros_header_unspecified`, preserving the
stamp without claiming a RealSense hardware timestamp or cross-camera alignment.
Missing requested ROS streams cause an incomplete capture rather than a complete
multimodal archive.

The CLI records resolved DAVIS motion thresholds, the exact primary/fallback/min-
event window-selection values, UTF-8 config file contents and hashes, Git commit/
branch/dirty status, requested duration/topics, and operator-supplied session and
object IDs. Config files are checked for changes while resolving the snapshot;
the backend's resolved motion config must agree before the camera is started.
Metadata says `data_origin=recorded` and `ground_truth_status=not_labelled`.
Dirty status records that source provenance is incomplete: modified source files
are not bundled with this archive. Record from a reviewed clean commit for a
reproducible acquisition.

The backend must start successfully and remain active for the requested finite
duration. Successful backend/subscriber teardown precedes finalization. Errors,
missing requested ROS streams, and Ctrl+C leave the manifest incomplete; the
backend is stopped in cleanup. There is no resume or overwrite path. A successful
capture can contain empty native event polls: completion describes the recording
lifecycle and does not establish valid event motion, measured stability, or
physical quality. No capture or hardware operation was used to test this CLI.

The camera-free CLI tests invoke `--help` without site packages and use a mocked
backend for complete, failed, interrupted, missing-stream, and teardown cases:

```bash
python -m pytest tests/test_record_native_trial_cli.py tests/test_recording_session.py \
  tests/test_ros_recording_adapter.py -o addopts='' -q
```

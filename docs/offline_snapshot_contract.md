# Offline trial snapshot contract

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

The capture command writes `grasp_trial_snapshot_v3`. It archives
derived status files, images and bounded trace tails that already exist on disk;
it opens no sensors. The companion verifier checks the archive and presents
recorded quality scores, decisions, validity flags and motion exclusion reasons
together. This is a prerequisite for reviewing evidence, not a raw sensor replay
or an evaluation of physical grasp performance.

## Capture and inspect

From the application directory (`app/`), with Python 3.10 or newer:

```bash
python scripts/capture_grasp_trial.py trial_name --stage post_grasp
python scripts/verify_grasp_trial.py output/trial_snapshots/ACTUAL_TRIAL_DIRECTORY
python scripts/verify_grasp_trial.py output/trial_snapshots/ACTUAL_TRIAL_DIRECTORY --require-complete
```

Use the actual directory printed by the capture command. Capturing existing files
is camera-free; starting a live session to produce those files is a separate
operator action. Do not interpret old files in `output/` as a current trial.
Both commands use only the Python standard library. To retain a verification
report, redirect its JSON stdout to a file outside the snapshot directory.

The verifier exits with `0` for a valid archive, `1` for an invalid or corrupt
archive, and `2` when `--require-complete` is set and snapshot evidence is
incomplete. An intact archive containing missing, stale or excluded evidence can
be useful to inspect. Exit `0` is not research readiness, sensor freshness,
stability, successful grasping, or permission to operate hardware.

## What v3 preserves

| Item | Contract |
|---|---|
| Artifacts | Existing bytes, relative POSIX path, size, SHA-256, source filesystem mtime and age at capture |
| Prediction | JSON object derived from the copied realtime status; its own size and SHA-256; no second read from the live status |
| Config | Actual files under `config/`, with relative source/archive paths, size and SHA-256 in `config_hashes` |
| Trace tails | Complete bounded trailing rows, source byte size, captured row/byte counts and limits; rows retain their recorded fields |
| Revision | Git commit and working-tree status; unavailable Git is represented as unknown, not a clean working tree |
| Ground truth | Separate unlabelled `ground_truth.json`; capture never fills labels from predictions |

Files are copied sequentially. A source can change during copying, and separate
artifacts can describe different observations. Copying configs preserves the
bytes present during capture; it does not prove a running process loaded those
bytes. Source metadata is sampled before each copy to avoid assigning a later
mtime to earlier bytes. This does not make the snapshot atomic.

The legacy `capture_readiness` field describes file presence and filesystem
mtime only. `capture_readiness_scope` states that limit explicitly. The verifier
recomputes age from recorded times, rejects future-dated sources, requires
nonempty required status objects, checks JSON/JSONL syntax and finite numbers,
and validates declared sizes and hashes. Extraction time and the verification
machine's current clock do not affect the result. Image contents are hash
checked but not decoded. JSON and trace inspection is limited to 16 MiB per file
and 100 nesting levels.

Hash checks detect accidental corruption against the supplied manifest. The
manifest is not signed; these checks do not establish provenance or authenticity.
Ground-truth annotations are intentionally outside prediction integrity checks
and must be maintained independently with their own annotation provenance.

## Reading the verification report

`archive_integrity` reports consistency with declared hashes and structure.
`file_availability_at_capture` reports missing, empty/invalid, stale and
future-dated evidence. `snapshot_inspection` reports whether that bounded set of
required files is available for inspection. None of these fields validates a
sensor, an algorithm, or a research claim.

Stale optional artifacts remain listed as diagnostics without hiding availability
of the required files. Stale live-quality/DAVIS evidence still qualifies the
motion report as unavailable; an available snapshot can contain excluded motion.

`recorded_live_quality` retains the stored score, decision, hard gates, baseline
validity and risk information. `motion_evidence` separately distinguishes
missing, invalid/empty, stale/unavailable, explicitly excluded, included as
recorded, inconsistent flags, and unknown evidence. Missing event evidence never
becomes a measured stable-motion label. A high stored score can coexist with
unavailable or excluded events; the verifier preserves both facts and does not
change runtime scores or thresholds.

Version 2 snapshots remain inspectable with the explicit limitation
`legacy_v2_prediction_and_config_contents_not_integrity_covered`. Their original
config contents cannot be recovered from hashes. The verifier accepts their
Windows path separators but rejects absolute paths, traversal and symlink
escapes from the snapshot directory.

## Scope and interpretation

A snapshot preserves derived files; it does not recreate a raw sensor sequence,
make the capture atomic or validate the predictions. Use the separate
[raw recording contract](raw_recording_contract.md) and
[replay guide](offline_replay.md) when actual inference inputs are required.

Retain native timestamps and clock domains without relabelling host receipt
estimates as device time. Analyzer startup-to-label timing is distinct from
physical slip onset-to-alert timing. Physical evaluation requires independently
observed outcomes, raw evidence, reference timing and calibration uncertainty.

## Reproduce software checks

```bash
python -m pytest tests/test_capture_grasp_trial.py tests/test_verify_grasp_trial.py tests/test_live_quality_monitor.py tests/test_pregrasp_required_evidence.py tests/test_realsense_perception_result.py
python -m compileall -q scripts/capture_grasp_trial.py scripts/verify_grasp_trial.py tests/test_capture_grasp_trial.py tests/test_verify_grasp_trial.py
```

The full runtime suite requires the supported Linux ROS/Qt environment; report
unavailable checks separately from the focused snapshot tests.

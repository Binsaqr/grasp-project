# Synthetic offline workflow smoke command

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

From the application directory (`app/`), using the existing repository Python environment:

```bash
python scripts/smoke_offline_workflow.py ../synthetic-offline-smoke-new
```

The destination must not exist. Prefer a destination outside the repository so
generated files do not change the source working-tree state. The command also
works from an unrelated current directory when invoked with the script's
absolute path. It uses the current Python interpreter for every subprocess and
installs nothing.

This command creates **constructed fixtures only** and exercises the actual
offline CLI workflow. It never opens a camera, imports a hardware capture
entrypoint, calls `record_native_trial.py`, runs a robot/gripper, or changes live
policy. It is a local software smoke check, not physical research validation or
a substitute for the repository's supported full test suite.

## What runs

The new destination contains three synthetic trials with disjoint trial,
session and object identities assigned to train, validation and test. Each
trial contains:

- Three native DAVIS event packets stored as integer arrays, with explicit
  synthetic receipt clocks, normalization reference and epoch metadata.
- Two complete quality-monitor inputs: a quiet observation with current event
  evidence, and a high-score observation with missing event evidence.
- Fixed literal `MONITOR` fixture labels written before inference. Their
  provenance is `synthetic_fixture`; no prediction is read to create a label.

For each trial it invokes the actual commands:

```bash
python scripts/replay_recording.py RAW_TRIAL --output REPLAY_REPORT
python scripts/estimate_clock_mapping.py --recording RAW_TRIAL --stream davis_poll --output CLOCK_REPORT
```

It then runs `evaluate_recordings.py --all-ablations` on all three archives with
explicit fixture-only frozen declarations, independently specified fixture
labels, and grouped splits. It retains the prediction interchange and invokes
`evaluate_recorded_trials.py` on that document, requiring exact agreement with
the aggregate evaluation.

Finally, `epsilon_sensitivity.py` runs on explicit constructed antipodal contacts
with a literal constant baseline of `0.2` and `recorded_samples: []`. Geometric
invariance controls remain distinct from a recorded grasp distribution. These
are nine actual subprocess commands in total; their exact argument arrays,
working directory, logs and exit codes are retained.

The smoke checks require:

- Expected report schemas and explicit false physical-validation/alignment
  claims.
- Three usable integer timestamp receipt associations per trial, without
  promoting them to physical clock alignment.
- Identical quality observation support across all four current ablation arms.
- Per split, one valid pair and one abstention for reference, removed sync
  diagnostics and removed phase conditioning; two abstentions for disabled
  events under the declared event-evidence requirement.
- Every high-score missing-event observation remains an `unusable_evidence`
  outcome. Its raw score cannot turn absence into measured stability.
- Standalone and aggregate evaluation agree, and the constructed epsilon
  controls pass with no recorded samples or fitted constant baseline.

The checks do not require a synthetic accuracy or success-rate target.

## Retained outcome and failures

Inspect `smoke-summary.json` in the destination. It includes:

- `status`: `PASSED_SYNTHETIC_ONLY` or `FAILED_SYNTHETIC_SMOKE`.
- Source Git commit and working-tree dirty state when available, Python version,
  SHA-256 hashes of listed scripts/implementation files, and their scoped
  `implementation_file_set_sha256` fingerprint. A dirty checkout remains
  explicitly dirty; a commit SHA alone is not claimed to describe those edits.
- Every attempted subprocess command, exit code, report path and stdout/stderr
  log path.
- Named semantic checks with their result and interpretation.
- Sizes and SHA-256 hashes of every retained archive, input, report and log.
  The summary excludes its own hash to avoid a self-reference.

An exported source tree or environment without Git can still run the smoke.
It retains `commit: null`, an explicit Git-unavailable reason, and unknown
working-tree state. Configuration provenance then uses
`implementation-files-sha256:<digest>`, identifying only the listed
implementation file set. This is not a Git SHA or a hash of the entire source
tree. When a real Git checkout and Git executable are available, the actual
checkout commit is required and retained.

Raw archives are under `raw/`, independent fixture inputs under `inputs/`, CLI
reports under `reports/`, and subprocess logs under `logs/`. With identical
source, source state and runtime, new destinations produce the same constructed
fixture/report bytes and summary. No run timestamp is invented as a measurement;
the archive creation timestamp is an explicitly fixed synthetic fixture value.

Exit 0 means all commands and semantic checks passed. Exit 1 means a command,
schema or expected outcome failed. A newly created destination retains a
partial failure summary, completed artifacts and subprocess logs for diagnosis.
Timed-out commands retain available partial stdout/stderr, a timeout flag and
an unknown exit code rather than inventing a terminal exit status.
An existing destination is refused without altering it or writing a summary
there. Storage failures can prevent retaining a complete summary; the command
then reports the write error.

The recorded argument arrays document what ran. To repeat the complete smoke,
use another new destination. Individual CLIs deliberately refuse existing
report paths in parts of this workflow, so direct reruns need fresh output
filenames.

## Portable regression tests

```bash
python -m pytest tests/test_offline_workflow_smoke.py -o addopts='' -q
```

The tests execute the complete command from an unrelated directory and check
retained hashes, repeatability, existing-content preservation, corruption,
schema failures and abstention mismatches. Synthetic passing checks establish
software behavior only. Physical labels, camera alignment, hardware latency,
slip lead time, grasp success and sample adequacy remain unvalidated.

# One offline command: archives, replay, independent evaluation

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

`scripts/evaluate_recordings.py` composes the verified raw archive reader,
actual component replay, the prediction interchange adapter, and the independent
held-out evaluator. It accepts multiple trials and multiple process-local archive
parts for one trial. It does not open sensors, create annotations, fit thresholds,
choose a winning configuration, or alter live policy.

Run this after collecting archives and independently annotating the observations.
The [raw contract](raw_recording_contract.md) describes archive integrity and
clock semantics. The [evaluation contract](offline_evaluation.md) defines label
provenance, unknown/unlabelled states, grouped splits, evidence and denominators.

## Inputs and command

Supply every train, validation and test trial in the annotation document. Each
archive manifest must retain `trial_id` and metadata `session_id`, `object_id`
and `data_origin`, matching the independently supplied trial metadata exactly.
Process-local archives may share those values, but must have distinct
`recording_id` values and disjoint `(trial_id, observation_id)` pairs. The
command refuses conflicting parts rather than silently combining them.

Repeat `--recording` for each part. This example includes a native sensor part
and a dashboard quality-input part for one trial, plus the other two trials:

```bash
python scripts/evaluate_recordings.py \
  --recording output/raw/trial-0-native \
  --recording output/raw/trial-0-quality \
  --recording output/raw/trial-1 \
  --recording output/raw/trial-2 \
  --annotations annotations.json \
  --splits splits.json \
  --declarations experiment.json \
  --all-ablations \
  --output output/evaluation/experiment-report.json
```

Use your actual archive paths, independent annotations and predeclared study.
`--ablation reference --ablation events_disabled` selects a subset;
`--all-ablations` selects every supported arm. With neither option, only
`reference` runs. The selected arms are never chosen from test metrics.

Omit `--output` for JSON on stdout. A file output must be new, outside all input
archives, and different from the input documents. Existing reports are not
overwritten. Contract, integrity or replay errors exit 2 and do not emit a new
report. Exit 0 means the pipeline completed, including when every prediction
abstains; inspect the reported metrics and coverage.

## Explicit study declarations

The declaration file uses `schema_version: "grasp_recording_experiment_v1"`,
the same `task_id` as annotations, a component `kind`, and an `arms` mapping.
Each selected arm requires a complete, explicit frozen configuration. For
example, this shape declares a quality-decision task and all current arms:

```json
{
  "schema_version": "grasp_recording_experiment_v1",
  "task_id": "independent-quality-decision-task",
  "kind": "live_quality_inputs",
  "arms": {
    "reference": {
      "configuration_id": "quality-reference",
      "revision": "REPLACE_WITH_ACTUAL_STUDY_REVISION",
      "parameters": {"protocol": "REPLACE_WITH_ACTUAL_PREDECLARED_PROTOCOL"},
      "required_evidence": ["davis_events"],
      "frozen_before_test": true,
      "selection_method": "predeclared",
      "fit_trial_ids": [],
      "selection_trial_ids": []
    },
    "events_disabled": {
      "configuration_id": "quality-events-disabled",
      "revision": "REPLACE_WITH_ACTUAL_STUDY_REVISION",
      "parameters": {"protocol": "REPLACE_WITH_ACTUAL_PREDECLARED_PROTOCOL"},
      "required_evidence": ["davis_events"],
      "frozen_before_test": true,
      "selection_method": "predeclared",
      "fit_trial_ids": [],
      "selection_trial_ids": []
    },
    "sync_diagnostics_removed": {
      "configuration_id": "quality-sync-diagnostics-removed",
      "revision": "REPLACE_WITH_ACTUAL_STUDY_REVISION",
      "parameters": {"protocol": "REPLACE_WITH_ACTUAL_PREDECLARED_PROTOCOL"},
      "required_evidence": ["davis_events"],
      "frozen_before_test": true,
      "selection_method": "predeclared",
      "fit_trial_ids": [],
      "selection_trial_ids": []
    },
    "phase_conditioning_removed": {
      "configuration_id": "quality-phase-conditioning-removed",
      "revision": "REPLACE_WITH_ACTUAL_STUDY_REVISION",
      "parameters": {"protocol": "REPLACE_WITH_ACTUAL_PREDECLARED_PROTOCOL"},
      "required_evidence": ["davis_events"],
      "frozen_before_test": true,
      "selection_method": "predeclared",
      "fit_trial_ids": [],
      "selection_trial_ids": []
    }
  }
}
```

Replace placeholders with real provenance. Declare `frozen_before_test: true`
only if the configuration and study choices were actually frozen before test
inspection. The command requires the declaration and cannot authenticate it.
For trained/selected configurations use `selection_method: "train_validation"`
and actual tuning trial IDs. Fitting may use train only; selection may use train
or validation. Test and unknown tuning trials are rejected before replay.
No thresholds are fitted by the command.

Use `kind: "davis_poll"` for motion classification, with its own task and motion
label vocabulary. `kind: "live_quality_inputs"` evaluates the actual quality
monitor decisions. The command does not mix these vocabularies or remap classes.
Independently inspect raw evidence to supply labels; never copy replay output
labels into the annotations. Empty event packets remain unavailable evidence,
even if a retained raw diagnostic happens to say `STABLE`.

`parameters.recording_experiment` and `parameters.replay_configuration` are
reserved for generated source provenance. Other study parameters are retained
verbatim and included in the configuration hash.

## Source configuration identity

Every recording retains its own manifest hash, recorded configuration snapshot,
revision, metadata and drop summary. Every arm retains each recording's complete
replay configuration and digest. Different sensor metadata or recording
provenance can produce different source hashes even when the applied inference
settings are identical; those sources are preserved separately.

The aggregate arm hash covers the explicitly declared study configuration and
the complete ordered source snapshots. Prediction rows retain their original
trial/observation IDs and source `recording_id`, and reference the aggregate arm
hash. Temporary per-recording adapter hashes are not falsely presented as one
identical source configuration.

Applied inference fingerprints compare the effective component configuration,
event-window settings for motion, and replay implementation/runtime. Differing
applied settings across contributing recordings are refused by default. If the
predeclared protocol intentionally uses recording-specific models, add
`"allow_recording_specific_configs": true` to each affected arm declaration.
The report then states `configuration_semantics: "recording_dependent_protocol"`
and exposes all fingerprints. That result must not be described as performance
of one uniform frozen model.

Parts with no observations of the requested component remain in source
provenance as `trace_only_for_selected_component`. Their RGB/depth/calibration
or other raw traces are retained without creating extra predictions or treating
the archive part itself as a missing prediction. An annotation explicitly
targeting an unsupported raw observation still produces a disclosed
`missing_prediction` outcome. Unknown annotation IDs absent from every archive
are rejected as an unresolved join.

## Ablations, support and evidence

All selected arms run through the same archive reader and component inference
path. The command verifies that each arm emits exactly the union of selected
raw observation IDs across parts. No missing output can disappear silently from
an arm. It reports the common valid labelled observations and whether valid-pair
coverage differs, without ranking arms or optimizing on test results.

The current arms are the registered component controls:

- `reference`: recorded inputs and production component policy.
- `events_disabled`: DAVIS unavailable; other recorded inputs retained.
- `sync_diagnostics_removed`: removes cross-camera synchronization diagnostics,
  preserving native event freshness. This is not an aligned-clock control.
- `phase_conditioning_removed`: applies the production post-grasp monitoring
  branch to every recorded phase as an offline control.

The declared evidence requirement remains explicit per arm. The example keeps
`davis_events` required in all arms, so `events_disabled` necessarily abstains on
accuracy under that requirement. This is an evidence-availability result; it
must not be converted into a measured stability result or compared as if it had
full coverage. A different, independently justified evaluation target may declare
a different stream requirement, retained visibly in its configuration snapshot.

The report embeds the prediction interchange and full evaluation report. Inspect
`labelled_count`, `paired_count`, `abstention_count`, valid-pair accuracy,
coverage, confusion matrices and per-observation reasons together. Missing,
stale, excluded or unavailable required evidence remains an abstention even
with a high recorded score. Corrupt raw bytes or manifests fail the whole run
before metrics are emitted. Synthetic and recorded annotation strata remain
separate in evaluation summaries.

## Verification and remaining study work

```bash
python -m pytest tests/test_recording_experiment_cli.py tests/test_recording_replay.py tests/test_offline_evaluation.py -o addopts='' -q
```

These tests exercise the CLI as a subprocess with constructed archives, actual
component inference, independent fixture labels, grouped splits and all arms.
Repeated runs on identical archives, declarations and implementation produce
identical report bytes. Changes to runtime or implementation are retained in
the replay fingerprint and intentionally change provenance.

Physical recordings, independently adjudicated labels and an appropriate split
plan are still required for an empirical study. This pipeline does not execute
the full RGB-D tracker, reproduce dashboard scheduling, authenticate blinding,
or establish camera alignment, physical success, sample adequacy, hardware
latency or slip lead time. Every report keeps physical validation false.

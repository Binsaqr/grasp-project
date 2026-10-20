# Independent annotations and grouped offline evaluation

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

`scripts/evaluate_recorded_trials.py` joins predictions to independent labels by
`(trial_id, observation_id)`. It checks configuration provenance and split
leakage, then reports outcomes, denominators, abstentions and confusion matrices
for train, validation and test separately. It never creates labels from model
outputs, fits thresholds, changes live policy, or opens a camera.

The existing `benchmarking/trial_recorder.py` supports manually labelled
dashboard snapshots. This stricter interchange adds independently sourced label
provenance, stable observation IDs, frozen configuration digests and explicit
held-out splits. Existing dashboard labels are not automatically promoted into
independent truth. Inspect their source and label them independently first.

## Run

From the application directory (`app/`), using the repository Python environment:

```bash
python scripts/evaluate_recorded_trials.py \
  --annotations annotations.json \
  --predictions predictions.json \
  --splits splits.json \
  --output output/evaluation/report.json
python -m pytest tests/test_offline_evaluation.py -o addopts='' -q
```

The command exits 0 when the contract is valid, including when there are no
valid labelled pairs. It exits 2 for malformed input, conflicting stable IDs,
leakage or configuration digest mismatches. Consumers must inspect
`metric_status`, `paired_count`, coverage and abstentions; exit 0 is not an
accuracy or readiness threshold. Inputs are strict UTF-8 JSON objects with no
duplicate keys or nonfinite numbers, limited to 64 MiB per document. The output
must not overwrite an input.

## Annotation document

Use `schema_version: "grasp_independent_annotations_v1"`, a `task_id` shared by
the predictions, a nonempty `label_vocabulary`, and these lists:

- `trials`: each entry has unique `trial_id`, `session_id`, `object_id`, and
  `data_origin` (`recorded` or `synthetic`). Preserve the actual recorded IDs.
- `annotations`: one entry per `(trial_id, observation_id)`, with unique
  `annotation_id`, `state` (`labelled`, `unknown`, or `unlabelled`), `label` and
  `provenance`. Labelled entries use the declared vocabulary; other states use
  JSON `null`. Unresolved disagreements should remain `unknown` until an
  independent adjudication, rather than becoming extra labels on the same ID.

Each annotation requires provenance:

```json
{
  "source": "independent_video",
  "annotator_id": "annotator-A",
  "independent_of_predictions": true,
  "prediction_blinded": true,
  "evidence_reference": {
    "media_uri": "recordings/trial-2-independent-video.mp4",
    "sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_LOWERCASE_SHA256",
    "clock_domain": "independent-video-presentation-time",
    "clock_units": "seconds",
    "start": 1.2,
    "end": 1.4
  }
}
```

This is a shape example; supply actual independent media and its digest. The
interval must have finite `0 <= start <= end`, with explicit clock domain and
units. A timestamp in this interval is not automatically aligned with a camera
or host clock. Retain the synchronization mapping and its uncertainty outside
the annotation file when an onset-time analysis needs one.

Alternatively, set `evidence_reference` to
`{"observation_ids": ["the-recorded-observation-id"]}`. The annotation's stable
observation ID must appear in that list. Use exactly one reference form.
Observation references identify the independently inspected raw observation;
they must not be references to prediction overlays or model-generated labels.

Supported `source` declarations are `independent_video`, `direct_observation`,
`external_measurement`, or `synthetic_fixture`. The last requires synthetic
trial origin; the others require recorded origin. Prediction-derived sources,
embedded prediction objects and explicit prediction references are rejected even
if an independence boolean is also true. The tool cannot authenticate the
declarations, prove blinding, or verify referenced media bytes. Keep the actual
raw evidence and independent annotation protocol for review.

## Split manifest

Declare groups and assign every trial exactly once:

```json
{
  "schema_version": "grasp_grouped_split_v1",
  "group_by": ["trial_id", "session_id", "object_id"],
  "assignments": [
    {"trial_id": "trial-0", "split": "train"},
    {"trial_id": "trial-1", "split": "validation"},
    {"trial_id": "trial-2", "split": "test"}
  ]
}
```

All three splits must contain trials. `trial_id` grouping is mandatory.
Each additional declared axis is checked **independently** across splits. This
avoids leakage caused by treating `(session_id, object_id)` as a composite key
while reusing the same session in train and test. A strict session-and-object
holdout may require assigning connected groups together. The evaluator refuses
infeasible assignments instead of silently weakening grouping.

Use `group_by: ["trial_id", "session_id"]` only for an explicitly declared
within-object evaluation; that manifest permits object overlap and does not
support an unseen-object claim. Split assignment is supplied by the researcher,
not optimized against predictions by this tool. Freeze the split before fitting
or choosing configurations, and retain its revision with the experiment.

## Frozen configurations and predictions

The prediction document uses `schema_version: "grasp_offline_predictions_v1"`,
the same `task_id`, `configurations` and `predictions` lists. A configuration
contains:

```json
{
  "configuration_id": "motion-full-v1",
  "revision": "ACTUAL_SOURCE_REVISION",
  "parameters": {"configuration_snapshot": "retain_actual_complete_parameters"},
  "required_evidence": ["events"],
  "frozen_before_test": true,
  "selection_method": "predeclared",
  "fit_trial_ids": [],
  "selection_trial_ids": []
}
```

Retain actual parameters or the complete content-addressed replay configuration
snapshot in `parameters`. For fitted/selected configurations use
`selection_method: "train_validation"` and list the actual tuning trials.
`fit_trial_ids` may contain only train trials; `selection_trial_ids` may contain
train or validation trials. Test or unknown trials are rejected. Predeclared
configurations must have both lists empty. The schema declarations do not prove
that somebody avoided inspecting the test set; maintain that experimental
discipline independently.

Compute the digest using the exported helper, which hashes canonical JSON for
the complete configuration except its optional `sha256` field:

```python
from grasp_project.benchmarking.offline_evaluation import configuration_digest

digest = configuration_digest(configuration)
```

Each prediction has `configuration_id`, matching `config_sha256`, `trial_id`,
`observation_id`, `prediction` and `evidence`:

```json
{
  "label": null,
  "score": 92.0,
  "decision": "RETAIN_ACTUAL_RECORDED_DECISION",
  "valid": false,
  "reasons": ["event_evidence_unavailable"]
}
```

That shape is the inner `prediction` object. Its label is null or a vocabulary
member; score is finite or null; decision is text or null; validity is an
explicit boolean. Invalid predictions require reasons. Scores are retained in
their original scale; they are not interpreted as calibrated probabilities.

`evidence` maps stream names to explicit status and reasons, for example:

```json
{"events": {"status": "missing", "reasons": ["no_recorded_event_observation"]}}
```

Statuses are `valid`, `missing`, `stale`, `excluded`, `corrupt`, or `unavailable`.
Every nonvalid status requires reasons. Omitted required streams are `missing`.
A high raw score, a stable label, or a permissive decision cannot override
missing, stale, excluded, corrupt or unavailable required evidence. These rows
become abstentions for accuracy. A configuration may explicitly require a
different stream set for an ablation; keep that change visible in its snapshot.

## Reading results and comparing ablations

Component replay can be adapted without assigning study provenance implicitly:

```python
from grasp_project.benchmarking.offline_evaluation import replay_predictions_document

# The caller fills the complete frozen configuration fields above, including
# selection_method, required_evidence and actual tuning trial IDs, then retains
# the exact source replay.configuration snapshot:
configuration["parameters"]["replay_configuration"] = replay["configuration"]
predictions = replay_predictions_document(
    replay, task_id="motion_class", configuration=configuration, kind="davis_poll",
)
```

Use evidence names from the replay, such as `davis_events`. `kind` is explicitly
`davis_poll` for motion or `live_quality_inputs` for quality decisions, with an
appropriate separate task/vocabulary. The helper verifies source configuration
digests and identities, copies the actual prediction/evidence fields, and
preserves absent/excluded streams. It does not infer labels, remap classes,
claim a configuration was frozen, or choose required evidence. Whole-trial
tuning restrictions are checked when the document is evaluated against splits.
To combine trials from the same arm, use one identical declared configuration
and concatenate the adapted prediction rows; conflicting IDs remain errors.

For each configuration and split:

- `accuracy_on_valid_pairs = correct_count / paired_count`, or null with no
  valid labelled pairs.
- `coverage_of_labelled = paired_count / labelled_count`, or null without labels.
- `abstention_count = labelled_count - paired_count`, including missing
  predictions, unusable evidence, invalid outputs and explicit abstentions.
- `correct_fraction_of_labelled = correct_count / labelled_count` retains
  abstentions in the denominator; it is a disclosed fraction, not an imputed
  accuracy for missing outputs.
- `outcome_counts`, confusion matrices and stable-ID `outcomes` retain reasons,
  raw scores, decisions, validity and evidence statuses. Unknown/unlabelled
  observations and predictions with no annotation remain visible without
  inventing truth. `by_data_origin` separates synthetic and recorded strata.

Metrics from configurations with different coverage need support-aware review.
The tool does not rank configurations, choose thresholds or pool train and test
metrics. Existing feature-mask ablations in
[`feature_mask.py`](../app/src/grasp_project/benchmarking/feature_mask.py) remain
display/report comparisons; do not relabel them as full inference replay.
For a replay ablation retain each actual frozen configuration, use identical
observation IDs and split assignments across arms, and preserve unavailable
outputs. Changing synchronization, event use or phase conditioning requires
running the corresponding offline inference arm; editing exported predictions
does not constitute an ablation experiment.

All reports set `physical_validation_established: false`. Synthetic fixtures
exercise software contracts. Even metrics from recorded independent labels do
not establish sample adequacy, physical grasp success, hardware latency or slip
lead time without the corresponding study and clock/onset evidence.

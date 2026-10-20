"""Synthetic contract tests; no measured grasp performance or hardware claims."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from grasp_project.benchmarking.offline_evaluation import (
    ANNOTATION_SCHEMA,
    PREDICTION_SCHEMA,
    SPLIT_SCHEMA,
    EvaluationContractError,
    configuration_digest,
    evaluate_recorded_trials,
    read_document,
    replay_predictions_document,
)


def _inputs():
    trials = [
        {"trial_id": f"trial-{index}", "session_id": f"session-{index}",
         "object_id": f"object-{index}", "data_origin": "synthetic"}
        for index in range(3)
    ]
    annotations = {
        "schema_version": ANNOTATION_SCHEMA, "task_id": "motion_class",
        "label_vocabulary": ["stable", "slip"], "trials": trials,
        "annotations": [
            {"annotation_id": f"annotation-{index}", "trial_id": trial["trial_id"],
             "observation_id": "observation-1", "state": "labelled", "label": "slip",
             "provenance": {
                 "source": "synthetic_fixture", "annotator_id": "fixture-author",
                 "independent_of_predictions": True, "prediction_blinded": True,
                 "evidence_reference": {"observation_ids": ["observation-1"]},
             }}
            for index, trial in enumerate(trials)
        ],
    }
    config = {
        "configuration_id": "frozen", "revision": "0123456789abcdef",
        "parameters": {"model": "existing-policy", "threshold": 0.5},
        "required_evidence": ["events"], "frozen_before_test": True,
        "selection_method": "predeclared", "fit_trial_ids": [], "selection_trial_ids": [],
    }
    predictions = {
        "schema_version": PREDICTION_SCHEMA, "task_id": "motion_class",
        "configurations": [config], "predictions": [
            {"configuration_id": "frozen", "config_sha256": configuration_digest(config),
             "trial_id": trial["trial_id"], "observation_id": "observation-1",
             "prediction": {"label": "slip", "score": 99.0, "decision": "WARN",
                            "valid": True, "reasons": []},
             "evidence": {"events": {"status": "valid", "reasons": []}}}
            for trial in trials
        ],
    }
    splits = {
        "schema_version": SPLIT_SCHEMA, "group_by": ["trial_id", "session_id", "object_id"],
        "assignments": [
            {"trial_id": f"trial-{index}", "split": split}
            for index, split in enumerate(("train", "validation", "test"))
        ],
    }
    return annotations, predictions, splits


def _test_report(annotations, predictions, splits):
    report = evaluate_recorded_trials(annotations, predictions, splits)
    return report["results"]["frozen"]["splits"]["test"]


def test_stable_id_join_deterministic_and_explicit_synthetic_limits():
    annotations, predictions, splits = _inputs()
    report = evaluate_recorded_trials(annotations, predictions, splits)
    predictions["predictions"].reverse()
    assert evaluate_recorded_trials(annotations, predictions, splits) == report
    assert report["data_origins"] == {"synthetic": 3}
    assert report["physical_validation_established"] is False
    test = report["results"]["frozen"]["splits"]["test"]
    assert test["paired_count"] == test["labelled_count"] == 1
    assert test["accuracy_on_valid_pairs"] == test["coverage_of_labelled"] == 1.0
    assert test["confusion_matrix"]["slip"]["slip"] == 1


@pytest.mark.parametrize("status", ["missing", "stale", "excluded", "corrupt", "unavailable"])
def test_high_score_cannot_override_missing_stale_excluded_corrupt_event_evidence(status):
    annotations, predictions, splits = _inputs()
    predictions["predictions"][-1]["evidence"]["events"] = {
        "status": status, "reasons": [f"fixture_{status}"]
    }
    report = evaluate_recorded_trials(annotations, predictions, splits)
    test = report["results"]["frozen"]["splits"]["test"]
    assert test["accuracy_on_valid_pairs"] is None
    assert test["correct_fraction_of_labelled"] == test["coverage_of_labelled"] == 0.0
    assert test["abstention_count"] == test["labelled_count"] == 1
    assert test["outcome_counts"] == {"unusable_evidence": 1}
    outcome = report["results"]["frozen"]["outcomes"][-1]
    assert outcome["prediction"] == {"label": "slip", "score": 99.0, "decision": "WARN", "valid": True, "reasons": []}
    assert outcome["unusable_required_evidence"]["events"]["status"] == status
    assert f"events:{status}:fixture_{status}" in outcome["reasons"]


def test_absent_event_stream_is_missing_never_stability():
    annotations, predictions, splits = _inputs()
    predictions["predictions"][-1]["evidence"] = {}
    predictions["predictions"][-1]["prediction"]["label"] = "stable"
    report = evaluate_recorded_trials(annotations, predictions, splits)
    outcome = report["results"]["frozen"]["outcomes"][-1]
    assert outcome["outcome"] == "unusable_evidence"
    assert outcome["unusable_required_evidence"]["events"]["status"] == "missing"
    assert _test_report(annotations, predictions, splits)["paired_count"] == 0


@pytest.mark.parametrize("state", ["unknown", "unlabelled"])
def test_unknown_and_unlabelled_never_become_prediction_derived_truth(state):
    annotations, predictions, splits = _inputs()
    annotations["annotations"][-1].update(state=state, label=None)
    test = _test_report(annotations, predictions, splits)
    assert test["annotation_count"] == 1
    assert test["labelled_count"] == test["paired_count"] == 0
    assert test["accuracy_on_valid_pairs"] is test["coverage_of_labelled"] is None
    assert test["outcome_counts"] == {state: 1}


@pytest.mark.parametrize("mutation, outcome", [
    ("remove", "missing_prediction"), ("invalid", "invalid_prediction"),
    ("null", "abstained"), ("incorrect", "incorrect"),
])
def test_outcomes_and_denominators(mutation, outcome):
    annotations, predictions, splits = _inputs()
    if mutation == "remove":
        predictions["predictions"].pop()
    else:
        prediction = predictions["predictions"][-1]["prediction"]
        if mutation == "invalid":
            prediction.update(valid=False, reasons=["output_not_valid"])
        else:
            prediction["label"] = None if mutation == "null" else "stable"
    test = _test_report(annotations, predictions, splits)
    assert test["outcome_counts"] == {outcome: 1}
    assert test["labelled_count"] == 1
    assert test["abstention_count"] == (0 if mutation == "incorrect" else 1)
    assert test["accuracy_on_valid_pairs"] == (0.0 if mutation == "incorrect" else None)


def test_unannotated_predictions_remain_visible_without_invented_labels():
    annotations, predictions, splits = _inputs()
    extra = copy.deepcopy(predictions["predictions"][-1])
    extra["observation_id"] = "unlabelled-observation"
    predictions["predictions"].append(extra)
    report = evaluate_recorded_trials(annotations, predictions, splits)
    assert report["results"]["frozen"]["unannotated_prediction_count"] == 1
    assert report["results"]["frozen"]["splits"]["test"]["annotation_count"] == 1


@pytest.mark.parametrize("axis", ["session_id", "object_id"])
def test_each_declared_grouping_axis_is_disjoint_not_just_composite_tuple(axis):
    annotations, predictions, splits = _inputs()
    annotations["trials"][-1][axis] = annotations["trials"][0][axis]
    with pytest.raises(EvaluationContractError, match=f"{axis} leakage"):
        evaluate_recorded_trials(annotations, predictions, splits)


def test_object_overlap_requires_explicit_within_object_scope():
    annotations, predictions, splits = _inputs()
    for trial in annotations["trials"]:
        trial["object_id"] = "shared-object"
    splits["group_by"].remove("object_id")
    report = evaluate_recorded_trials(annotations, predictions, splits)
    assert "object_id" not in report["group_by"]


@pytest.mark.parametrize("field", ["fit_trial_ids", "selection_trial_ids"])
def test_test_set_cannot_fit_thresholds_or_select_configuration(field):
    annotations, predictions, splits = _inputs()
    predictions["configurations"][0].update(selection_method="train_validation", **{field: ["trial-2"]})
    with pytest.raises(EvaluationContractError, match="train trials only|cannot use test"):
        evaluate_recorded_trials(annotations, predictions, splits)


def test_valid_training_provenance_requires_matching_frozen_snapshot():
    annotations, predictions, splits = _inputs()
    config = predictions["configurations"][0]
    config.update(selection_method="train_validation", fit_trial_ids=["trial-0"], selection_trial_ids=["trial-1"])
    for row in predictions["predictions"]:
        row["config_sha256"] = configuration_digest(config)
    assert _test_report(annotations, predictions, splits)["paired_count"] == 1


@pytest.mark.parametrize("location", ["configuration", "prediction"])
def test_configuration_snapshot_corruption_is_rejected(location):
    annotations, predictions, splits = _inputs()
    config = predictions["configurations"][0]
    if location == "configuration":
        config["sha256"] = configuration_digest(config)
        config["parameters"]["threshold"] = 0.8
    else:
        predictions["predictions"][-1]["config_sha256"] = "0" * 64
    with pytest.raises(EvaluationContractError, match="digest mismatch"):
        evaluate_recorded_trials(annotations, predictions, splits)


@pytest.mark.parametrize("change, message", [
    ({"source": "model_prediction"}, "prediction-derived"),
    ({"source": "prediction_snapshot", "independent_of_predictions": True}, "prediction-derived"),
    ({"independent_of_predictions": False}, "independence"),
    ({"prediction_blinded": False}, "blinding"),
    ({"prediction_reference": "prediction-3"}, "reference predictions"),
])
def test_prediction_derived_labels_rejected_despite_claimed_independence(change, message):
    annotations, predictions, splits = _inputs()
    annotations["annotations"][-1]["provenance"].update(change)
    with pytest.raises(EvaluationContractError, match=message):
        evaluate_recorded_trials(annotations, predictions, splits)


def test_embedded_prediction_is_not_a_label_source():
    annotations, predictions, splits = _inputs()
    annotations["annotations"][-1]["prediction"] = {"label": "slip"}
    with pytest.raises(EvaluationContractError, match="embedded"):
        evaluate_recorded_trials(annotations, predictions, splits)


def test_media_interval_provenance_has_explicit_clock_domain_units_and_digest():
    annotations, predictions, splits = _inputs()
    annotations["trials"][-1]["data_origin"] = "recorded"
    provenance = annotations["annotations"][-1]["provenance"]
    provenance.update(source="independent_video", evidence_reference={
        "media_uri": "manual://independent-camera/trial-2", "sha256": "a" * 64,
        "clock_domain": "annotation-video-frame", "clock_units": "frames", "start": 30, "end": 42,
    })
    assert _test_report(annotations, predictions, splits)["paired_count"] == 1
    provenance["evidence_reference"]["end"] = 10
    with pytest.raises(EvaluationContractError, match="clock interval"):
        evaluate_recorded_trials(annotations, predictions, splits)


@pytest.mark.parametrize("duplicate", ["annotations", "predictions", "assignments"])
def test_duplicate_stable_identifiers_never_silently_overwrite(duplicate):
    annotations, predictions, splits = _inputs()
    document = annotations if duplicate == "annotations" else predictions if duplicate == "predictions" else splits
    document[duplicate].append(copy.deepcopy(document[duplicate][-1]))
    with pytest.raises(EvaluationContractError, match="duplicate|multiple assignments"):
        evaluate_recorded_trials(annotations, predictions, splits)


@pytest.mark.parametrize("content", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{"a": 1e10000}', '{"a":', '[]'])
def test_strict_reader_rejects_duplicate_nonfinite_truncated_or_nonobject_json(tmp_path, content):
    path = tmp_path / "corrupt.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EvaluationContractError):
        read_document(path)


def test_zero_predictions_exposes_no_accuracy_and_full_abstention():
    annotations, predictions, splits = _inputs()
    predictions["predictions"] = []
    test = _test_report(annotations, predictions, splits)
    assert test["metric_status"] == "UNAVAILABLE_NO_VALID_PAIRS"
    assert test["accuracy_on_valid_pairs"] is None
    assert test["paired_count"] == 0
    assert test["abstention_count"] == test["labelled_count"] == 1


@pytest.mark.parametrize("location", ["origin", "state", "selection_method", "evidence_status"])
def test_malformed_object_values_raise_contract_errors_not_python_type_errors(location):
    annotations, predictions, splits = _inputs()
    if location == "origin":
        annotations["trials"][-1]["data_origin"] = {}
    elif location == "state":
        annotations["annotations"][-1]["state"] = []
    elif location == "selection_method":
        predictions["configurations"][0]["selection_method"] = {}
    else:
        predictions["predictions"][-1]["evidence"]["events"]["status"] = []
    with pytest.raises(EvaluationContractError):
        evaluate_recorded_trials(annotations, predictions, splits)


def test_in_memory_nonfinite_input_also_fails_closed():
    annotations, predictions, splits = _inputs()
    predictions["predictions"][-1]["prediction"]["score"] = float("nan")
    with pytest.raises(EvaluationContractError, match="finite JSON"):
        evaluate_recorded_trials(annotations, predictions, splits)


def test_recorded_and_synthetic_strata_are_separate():
    annotations, predictions, splits = _inputs()
    annotations["trials"][-1]["data_origin"] = "recorded"
    annotations["annotations"][-1]["provenance"]["source"] = "direct_observation"
    report = evaluate_recorded_trials(annotations, predictions, splits)
    results = report["results"]["frozen"]["splits"]
    assert results["test"]["by_data_origin"]["recorded"]["paired_count"] == 1
    assert "synthetic" not in results["test"]["by_data_origin"]
    assert results["train"]["by_data_origin"]["synthetic"]["paired_count"] == 1


def _replay_inputs():
    annotations, predictions, splits = _inputs()
    source = {"ablation": "reference", "recorded_config_sha256": "c" * 64,
              "replay_schema": "grasp_component_replay_v1"}
    source["sha256"] = configuration_digest(source)
    row = copy.deepcopy(predictions["predictions"][-1])
    row.update(configuration_id="reference", config_sha256=source["sha256"], kind="davis_poll")
    replay = {"schema_version": "grasp_component_replay_v1", "trial_id": "trial-2",
              "configuration": source, "observations": [row]}
    config = predictions["configurations"][0]
    config["parameters"]["replay_configuration"] = copy.deepcopy(source)
    return annotations, replay, splits, config


@pytest.mark.parametrize("status", ["valid", "missing", "stale", "excluded", "corrupt", "unavailable"])
def test_replay_adapter_joins_explicit_declarations_without_inventing_truth(status):
    annotations, replay, splits, config = _replay_inputs()
    replay["observations"][0]["evidence"]["events"] = {
        "status": status, "reasons": [] if status == "valid" else ["retained_fixture_reason"]
    }
    document = replay_predictions_document(replay, task_id="motion_class", configuration=config, kind="davis_poll")
    assert document["predictions"][0]["prediction"] == replay["observations"][0]["prediction"]
    assert document["predictions"][0]["evidence"] == replay["observations"][0]["evidence"]
    assert "annotations" not in document
    report = evaluate_recorded_trials(annotations, document, splits)
    test = report["results"]["frozen"]["splits"]["test"]
    assert test["paired_count"] == (1 if status == "valid" else 0)
    assert test["abstention_count"] == (0 if status == "valid" else 1)


def test_replay_adapter_preserves_absence_and_selects_one_component():
    annotations, replay, splits, config = _replay_inputs()
    replay["observations"][0]["evidence"] = {}
    quality = copy.deepcopy(replay["observations"][0])
    quality.update(kind="live_quality_inputs", observation_id="quality-1")
    quality["prediction"]["label"] = "READY_TO_LIFT"
    replay["observations"].append(quality)
    document = replay_predictions_document(replay, task_id="motion_class", configuration=config, kind="davis_poll")
    assert len(document["predictions"]) == 1
    assert document["predictions"][0]["evidence"] == {}
    assert _test_report(annotations, document, splits)["outcome_counts"] == {"unusable_evidence": 1}
    config["parameters"]["threshold"] = 999
    replay["observations"][0]["prediction"]["score"] = 0
    assert document["configurations"][0]["parameters"]["threshold"] == 0.5
    assert document["predictions"][0]["prediction"]["score"] == 99


@pytest.mark.parametrize("field", ["frozen_before_test", "selection_method", "fit_trial_ids", "selection_trial_ids", "required_evidence"])
def test_replay_adapter_never_invents_frozen_or_tuning_declarations(field):
    _, replay, _, config = _replay_inputs()
    del config[field]
    with pytest.raises(EvaluationContractError):
        replay_predictions_document(replay, task_id="motion_class", configuration=config, kind="davis_poll")


@pytest.mark.parametrize("mutation", ["source_hash", "source_snapshot", "row_hash", "row_config", "row_trial", "duplicate"])
def test_replay_adapter_rejects_corrupt_or_inconsistent_source(mutation):
    _, replay, _, config = _replay_inputs()
    if mutation == "source_hash":
        replay["configuration"]["sha256"] = "0" * 64
    elif mutation == "source_snapshot":
        config["parameters"]["replay_configuration"]["ablation"] = "events_disabled"
    elif mutation == "row_hash":
        replay["observations"][0]["config_sha256"] = "0" * 64
    elif mutation == "row_config":
        replay["observations"][0]["configuration_id"] = "other"
    elif mutation == "row_trial":
        replay["observations"][0]["trial_id"] = "other"
    else:
        replay["observations"].append(copy.deepcopy(replay["observations"][0]))
    with pytest.raises(EvaluationContractError):
        replay_predictions_document(replay, task_id="motion_class", configuration=config, kind="davis_poll")


def test_cli_writes_report_and_fails_closed_on_corrupt_input(tmp_path):
    documents = _inputs()
    arguments = []
    for name, document in zip(("annotations", "predictions", "splits"), documents):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        arguments.extend([f"--{name}", str(path)])
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_recorded_trials.py"
    output = tmp_path / "report.json"
    run = subprocess.run([sys.executable, str(script), *arguments, "--output", str(output)], check=False, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["physical_validation_established"] is False
    output.unlink()
    (tmp_path / "predictions.json").write_text("{", encoding="utf-8")
    run = subprocess.run([sys.executable, str(script), *arguments, "--output", str(output)], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert "contract error" in run.stderr
    assert not output.exists()

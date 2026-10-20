"""Strict, camera-free evaluation of predictions against independent annotations.

This module never creates labels, fits thresholds, or runs a live policy. Schema
declarations preserve an audit trail; they cannot authenticate an annotator's
independence or establish physical validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA = "grasp_independent_annotations_v1"
PREDICTION_SCHEMA = "grasp_offline_predictions_v1"
SPLIT_SCHEMA = "grasp_grouped_split_v1"
REPORT_SCHEMA = "grasp_offline_evaluation_v1"
SPLITS = ("train", "validation", "test")
EVIDENCE_STATES = ("valid", "missing", "stale", "excluded", "corrupt", "unavailable")
INDEPENDENT_SOURCES = (
    "independent_video", "direct_observation", "external_measurement", "synthetic_fixture"
)


class EvaluationContractError(ValueError):
    """An input cannot be evaluated without an ambiguous or unsafe assumption."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def _text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def _list(value: Any, name: str) -> list:
    _require(isinstance(value, list), f"{name} must be a list")
    return value


def _strings(value: Any, name: str, *, nonempty: bool = False) -> list[str]:
    result = [_text(item, name) for item in _list(value, name)]
    _require(len(result) == len(set(result)), f"{name} contains duplicates")
    _require(bool(result) or not nonempty, f"{name} must not be empty")
    return result


def _object(value: Any, name: str) -> dict:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _schema(document: Any, expected: str) -> dict:
    document = _object(document, expected)
    _require(document.get("schema_version") == expected, f"expected schema_version {expected}")
    return document


def _finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_document(path: str | Path) -> dict:
    """Read strict JSON, rejecting duplicate keys and nonfinite/corrupted input."""
    try:
        path = Path(path)
        _require(path.stat().st_size <= 64 * 1024 * 1024, "JSON input exceeds 64 MiB")
        document = _object(json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
            parse_constant=lambda value: _require(False, f"nonfinite JSON number: {value}"),
        ), str(path))
        _finite_json(document)
        return document
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise EvaluationContractError(f"cannot read JSON document {path}: {exc}") from exc


def _finite_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise EvaluationContractError(f"input is not finite JSON: {exc}") from exc


def configuration_digest(configuration: dict) -> str:
    """Hash the complete frozen configuration, excluding its optional digest."""
    configuration = _object(configuration, "configuration")
    try:
        encoded = json.dumps(
            {key: value for key, value in configuration.items() if key != "sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise EvaluationContractError(f"configuration is not finite JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _annotations(document: dict) -> tuple[dict, dict, list[str]]:
    document = _schema(document, ANNOTATION_SCHEMA)
    _text(document.get("task_id"), "task_id")
    vocabulary = _strings(document.get("label_vocabulary"), "label_vocabulary", nonempty=True)
    _require(not set(vocabulary) & {"unknown", "unlabelled"}, "unknown/unlabelled are states, not labels")
    trials = {}
    for raw in _list(document.get("trials"), "trials"):
        trial = _object(raw, "trial")
        trial_id = _text(trial.get("trial_id"), "trial_id")
        _require(trial_id not in trials, f"duplicate trial_id: {trial_id}")
        for name in ("session_id", "object_id"):
            _text(trial.get(name), name)
        _require(trial.get("data_origin") in ("recorded", "synthetic"), "data_origin must be recorded or synthetic")
        trials[trial_id] = trial
    labels = {}
    annotation_ids = set()
    for raw in _list(document.get("annotations"), "annotations"):
        entry = _object(raw, "annotation")
        annotation_id = _text(entry.get("annotation_id"), "annotation_id")
        _require(annotation_id not in annotation_ids, f"duplicate annotation_id: {annotation_id}")
        annotation_ids.add(annotation_id)
        key = (_text(entry.get("trial_id"), "trial_id"), _text(entry.get("observation_id"), "observation_id"))
        _require(key[0] in trials, f"unknown annotation trial_id: {key[0]}")
        _require(key not in labels, f"conflicting or duplicate annotation for {key}")
        state = entry.get("state")
        _require(state in ("labelled", "unknown", "unlabelled"), "invalid annotation state")
        if state == "labelled":
            _require(entry.get("label") in vocabulary, "labelled annotation must use label_vocabulary")
        else:
            _require(entry.get("label") is None, "unknown/unlabelled annotation cannot contain a label")
        provenance = _object(entry.get("provenance"), "annotation provenance")
        _require(provenance.get("source") in INDEPENDENT_SOURCES, "prediction-derived or unknown annotation source")
        _text(provenance.get("annotator_id"), "annotator_id")
        _require(provenance.get("independent_of_predictions") is True, "annotation must declare independence")
        _require(provenance.get("prediction_blinded") is True, "annotation must declare prediction blinding")
        _require(not {"prediction", "prediction_snapshot", "predicted_label"} & entry.keys(), "predictions cannot be embedded in annotations")
        _require(not {"prediction_id", "prediction_reference", "model_output"} & provenance.keys(), "annotation provenance cannot reference predictions")
        if provenance["source"] == "synthetic_fixture":
            _require(trials[key[0]]["data_origin"] == "synthetic", "synthetic labels require synthetic trials")
        else:
            _require(trials[key[0]]["data_origin"] == "recorded", "recorded labels require recorded trials")
        reference = _object(provenance.get("evidence_reference"), "evidence_reference")
        _require(("observation_ids" in reference) != ("media_uri" in reference), "reference must select observation IDs or media interval")
        if "observation_ids" in reference:
            ids = _strings(reference["observation_ids"], "observation_ids", nonempty=True)
            _require(key[1] in ids, "annotation observation_id missing from evidence reference")
        else:
            _text(reference.get("media_uri"), "media_uri")
            digest = reference.get("sha256")
            _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid media sha256")
            _text(reference.get("clock_domain"), "clock_domain")
            start, end = reference.get("start"), reference.get("end")
            _require(_finite_number(start) and _finite_number(end) and 0 <= start <= end, "invalid annotation clock interval")
            _text(reference.get("clock_units"), "clock_units")
        labels[key] = entry
    return trials, labels, vocabulary


def _split_assignments(document: dict, trials: dict) -> tuple[dict, list[str]]:
    document = _schema(document, SPLIT_SCHEMA)
    group_by = _strings(document.get("group_by"), "group_by", nonempty=True)
    _require(set(group_by) <= {"trial_id", "session_id", "object_id"}, "unsupported grouping axis")
    _require("trial_id" in group_by, "trial_id grouping is required")
    assignments = {}
    for raw in _list(document.get("assignments"), "assignments"):
        row = _object(raw, "split assignment")
        trial_id = _text(row.get("trial_id"), "trial_id")
        _require(trial_id in trials, f"split references unknown trial: {trial_id}")
        _require(trial_id not in assignments, f"trial appears in multiple assignments: {trial_id}")
        _require(row.get("split") in SPLITS, "invalid split")
        assignments[trial_id] = row["split"]
    _require(set(assignments) == set(trials), "every trial must have exactly one split assignment")
    _require(set(assignments.values()) == set(SPLITS), "train, validation and test must each contain trials")
    for axis in group_by:
        seen = {}
        for trial_id, trial in trials.items():
            group = trial[axis]
            split = assignments[trial_id]
            _require(group not in seen or seen[group] == split, f"{axis} leakage across splits: {group}")
            seen[group] = split
    return assignments, group_by


def _frozen_configuration(config: dict, assignments: dict | None = None) -> dict:
    config = _object(config, "configuration")
    _text(config.get("configuration_id"), "configuration_id")
    _text(config.get("revision"), "revision")
    _object(config.get("parameters"), "parameters")
    _strings(config.get("required_evidence"), "required_evidence", nonempty=True)
    _require(config.get("frozen_before_test") is True, "configuration must be frozen before test")
    method = config.get("selection_method")
    _require(method in ("predeclared", "train_validation"), "invalid configuration selection_method")
    fit = _strings(config.get("fit_trial_ids"), "fit_trial_ids")
    selected = _strings(config.get("selection_trial_ids"), "selection_trial_ids")
    if assignments is not None:
        _require(all(assignments.get(item) == "train" for item in fit), "fitting may use train trials only")
        _require(all(assignments.get(item) in {"train", "validation"} for item in selected), "selection cannot use test or unknown trials")
    _require(method != "predeclared" or not (fit or selected), "predeclared configuration cannot list fitted/selected trials")
    _require(method != "train_validation" or bool(fit or selected), "train_validation requires tuning provenance")
    digest = configuration_digest(config)
    _require("sha256" not in config or config["sha256"] == digest, "configuration snapshot digest mismatch")
    return {**config, "sha256": digest}


def replay_predictions_document(replay: dict, *, task_id: str, configuration: dict, kind: str) -> dict:
    """Adapt one component replay without inferring labels or study provenance.

    The caller explicitly declares the frozen evaluation configuration, including
    ``parameters.replay_configuration`` equal to the complete source snapshot.
    Tuning provenance is checked against actual splits only during evaluation.
    ``kind`` selects motion or quality outputs, never mixes their vocabularies.
    """
    _finite_json(replay)
    _finite_json(configuration)
    replay = _schema(replay, "grasp_component_replay_v1")
    _text(task_id, "task_id")
    _require(kind in ("davis_poll", "live_quality_inputs"), "unsupported replay component kind")
    source = _object(replay.get("configuration"), "replay configuration")
    source_id = _text(source.get("ablation"), "replay ablation")
    source_digest = configuration_digest(source)
    _require(source.get("sha256") == source_digest, "source replay configuration digest mismatch")
    config = _frozen_configuration(configuration)
    _require(config["parameters"].get("replay_configuration") == source, "configuration must retain complete replay_configuration snapshot")
    trial_id = _text(replay.get("trial_id"), "replay trial_id")
    predictions, seen = [], set()
    for raw in _list(replay.get("observations"), "replay observations"):
        row = _object(raw, "replay observation")
        if row.get("kind") != kind:
            continue
        _require(row.get("trial_id") == trial_id, "replay observation trial differs from source")
        _require(row.get("configuration_id") == source_id, "replay observation configuration_id differs from source")
        _require(row.get("config_sha256") == source_digest, "replay observation configuration digest mismatch")
        observation_id = _text(row.get("observation_id"), "replay observation_id")
        _require(observation_id not in seen, "duplicate replay observation_id")
        seen.add(observation_id)
        predictions.append({
            "configuration_id": config["configuration_id"], "config_sha256": config["sha256"],
            "trial_id": trial_id, "observation_id": observation_id,
            "prediction": _object(row.get("prediction"), "replay prediction"),
            "evidence": _object(row.get("evidence"), "replay evidence"),
        })
    # Detach snapshots so later caller mutations cannot change the interchange.
    return json.loads(json.dumps({
        "schema_version": PREDICTION_SCHEMA, "task_id": task_id,
        "configurations": [config], "predictions": predictions,
    }, allow_nan=False))


def _predictions(document: dict, trials: dict, assignments: dict, vocabulary: list[str], task_id: str) -> tuple[dict, dict]:
    document = _schema(document, PREDICTION_SCHEMA)
    _require(document.get("task_id") == task_id, "prediction and annotation tasks differ")
    configurations = {}
    for raw in _list(document.get("configurations"), "configurations"):
        config = _frozen_configuration(raw, assignments)
        config_id = config["configuration_id"]
        _require(config_id not in configurations, f"duplicate configuration_id: {config_id}")
        configurations[config_id] = config
    _require(bool(configurations), "at least one configuration is required")
    predictions = {}
    for raw in _list(document.get("predictions"), "predictions"):
        row = _object(raw, "prediction row")
        config_id = _text(row.get("configuration_id"), "configuration_id")
        _require(config_id in configurations, f"unknown prediction configuration: {config_id}")
        trial_id = _text(row.get("trial_id"), "trial_id")
        _require(trial_id in trials, f"unknown prediction trial: {trial_id}")
        key = (config_id, trial_id, _text(row.get("observation_id"), "observation_id"))
        _require(key not in predictions, f"duplicate prediction stable ID: {key}")
        _require(row.get("config_sha256") == configurations[config_id]["sha256"], "prediction configuration digest mismatch")
        prediction = _object(row.get("prediction"), "prediction")
        _require(prediction.get("label") is None or prediction["label"] in vocabulary, "prediction label outside label_vocabulary")
        _require(isinstance(prediction.get("valid"), bool), "prediction.valid must be explicit boolean")
        _require(prediction.get("score") is None or _finite_number(prediction["score"]), "prediction score must be finite or null")
        _require(prediction.get("decision") is None or isinstance(prediction["decision"], str), "prediction decision must be text or null")
        _strings(prediction.get("reasons"), "prediction reasons", nonempty=not prediction["valid"])
        evidence = _object(row.get("evidence"), "prediction evidence")
        for name, raw_item in evidence.items():
            _text(name, "evidence name")
            item = _object(raw_item, "evidence item")
            _require(item.get("status") in EVIDENCE_STATES, "invalid evidence status")
            _strings(item.get("reasons"), "evidence reasons", nonempty=item["status"] != "valid")
        predictions[key] = row
    return configurations, predictions


def _summary(outcomes: list[dict], vocabulary: list[str]) -> dict:
    counts = Counter(row["outcome"] for row in outcomes)
    labelled = [row for row in outcomes if row["annotation_state"] == "labelled"]
    pairs = [row for row in outcomes if row["outcome"] in {"correct", "incorrect"}]
    confusion = {truth: {predicted: 0 for predicted in vocabulary} for truth in vocabulary}
    for row in pairs:
        confusion[row["truth"]][row["prediction"]["label"]] += 1
    return {
        "annotation_count": len(outcomes), "labelled_count": len(labelled),
        "paired_count": len(pairs), "correct_count": counts["correct"],
        "abstention_count": len(labelled) - len(pairs),
        "accuracy_on_valid_pairs": counts["correct"] / len(pairs) if pairs else None,
        "coverage_of_labelled": len(pairs) / len(labelled) if labelled else None,
        "correct_fraction_of_labelled": counts["correct"] / len(labelled) if labelled else None,
        "outcome_counts": dict(sorted(counts.items())), "confusion_matrix": confusion,
        "metric_status": "AVAILABLE" if pairs else "UNAVAILABLE_NO_VALID_PAIRS",
    }


def _split_summary(outcomes: list[dict], vocabulary: list[str]) -> dict:
    origins = sorted({row["data_origin"] for row in outcomes})
    return {
        **_summary(outcomes, vocabulary),
        "data_origin_counts": dict(sorted(Counter(row["data_origin"] for row in outcomes).items())),
        "by_data_origin": {
            origin: _summary([row for row in outcomes if row["data_origin"] == origin], vocabulary)
            for origin in origins
        },
    }


def evaluate_recorded_trials(annotations: dict, predictions: dict, splits: dict) -> dict:
    """Join stable IDs, reject declared leakage, and retain every abstention.

    Accuracy denominates only valid labelled pairs. Coverage and abstentions
    disclose the observations omitted from that denominator. No fit or threshold
    selection is performed, and no metric authenticates physical validation.
    """
    for document in (annotations, predictions, splits):
        _finite_json(document)
    trials, labels, vocabulary = _annotations(annotations)
    assignments, group_by = _split_assignments(splits, trials)
    configurations, indexed = _predictions(predictions, trials, assignments, vocabulary, annotations["task_id"])
    results = {}
    for config_id, config in configurations.items():
        outcomes = []
        for (trial_id, observation_id), annotation in sorted(labels.items()):
            row = indexed.get((config_id, trial_id, observation_id))
            state = annotation["state"]
            reasons = []
            unavailable = {}
            if row is not None:
                for name in config["required_evidence"]:
                    item = row["evidence"].get(name, {"status": "missing", "reasons": ["required_evidence_not_recorded"]})
                    if item["status"] != "valid":
                        unavailable[name] = item
                        reasons.extend(f"{name}:{item['status']}:{reason}" for reason in item["reasons"])
                reasons.extend(row["prediction"]["reasons"])
            if state != "labelled":
                outcome = state
            elif row is None:
                outcome = "missing_prediction"
                reasons.append("no_prediction_for_stable_id")
            elif unavailable:
                outcome = "unusable_evidence"
            elif row["prediction"]["valid"] is not True:
                outcome = "invalid_prediction"
            elif row["prediction"]["label"] is None:
                outcome = "abstained"
            else:
                outcome = "correct" if row["prediction"]["label"] == annotation["label"] else "incorrect"
            outcomes.append({
                "annotation_id": annotation["annotation_id"], "trial_id": trial_id,
                "observation_id": observation_id, "split": assignments[trial_id],
                "data_origin": trials[trial_id]["data_origin"],
                "annotation_state": state, "truth": annotation.get("label"),
                "prediction": row["prediction"] if row else None,
                "evidence": row["evidence"] if row else {},
                "unusable_required_evidence": unavailable, "outcome": outcome,
                "reasons": list(dict.fromkeys(reasons)),
            })
        extra = sum(1 for key in indexed if key[0] == config_id and (key[1], key[2]) not in labels)
        results[config_id] = {
            "configuration_sha256": config["sha256"], "unannotated_prediction_count": extra,
            "splits": {split: _split_summary([row for row in outcomes if row["split"] == split], vocabulary) for split in SPLITS},
            "outcomes": outcomes,
        }
    return {
        "schema_version": REPORT_SCHEMA, "task_id": annotations["task_id"],
        "group_by": group_by, "grouping_rule": "each_declared_axis_disjoint_across_splits",
        "trial_counts": dict(sorted(Counter(assignments.values()).items())),
        "data_origins": dict(sorted(Counter(trial["data_origin"] for trial in trials.values()).items())),
        "configurations": configurations, "results": results,
        "physical_validation_established": False,
        "limitations": [
            "Annotation independence and blinding are declarations, not authenticated by this tool.",
            "Media references and hashes are retained; this tool does not verify referenced media bytes.",
            "Accuracy uses valid labelled pairs; inspect coverage and abstentions before comparing configurations.",
            "No sample adequacy, physical success rate, hardware latency or slip lead time is established.",
        ],
    }

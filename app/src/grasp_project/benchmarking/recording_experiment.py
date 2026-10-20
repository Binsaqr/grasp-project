"""Run declared offline arms on verified archive parts and independent labels.

This composes existing recording, replay and evaluation contracts. It performs
no fitting, annotation, camera access, or selection based on test performance.
"""

from __future__ import annotations

import json
from pathlib import Path

from grasp_project.recording.replay import ABLATIONS, replay_recording
from grasp_project.recording.session import read_recording

from .offline_evaluation import (
    PREDICTION_SCHEMA,
    EvaluationContractError,
    _annotations,
    _finite_json,
    _frozen_configuration,
    _object,
    _require,
    _schema,
    _split_assignments,
    _strings,
    configuration_digest,
    evaluate_recorded_trials,
    read_document,
    replay_predictions_document,
)

DECLARATION_SCHEMA = "grasp_recording_experiment_v1"
EXPERIMENT_SCHEMA = "grasp_recording_experiment_report_v1"


def _digest(value) -> str:
    return configuration_digest({"content": value})


def safe_output_path(output, *, input_paths=(), recording_dirs=()) -> Path:
    """Allow only a new file outside every input archive and input document."""
    path = Path(output).resolve()
    _require(
        not path.exists() and not Path(output).is_symlink(),
        "output already exists; choose a new report path",
    )
    _require(
        path not in {Path(item).resolve() for item in input_paths},
        "output must not overwrite an input document",
    )
    for recording_dir in recording_dirs:
        root = Path(recording_dir).resolve()
        _require(
            not path.is_relative_to(root),
            "output must not be inside an input recording archive",
        )
    return path


def _declarations(document, assignments, selected):
    document = _schema(document, DECLARATION_SCHEMA)
    kind = document.get("kind")
    _require(
        kind in ("davis_poll", "live_quality_inputs"),
        "kind must select davis_poll or live_quality_inputs",
    )
    arms = _object(document.get("arms"), "declared arms")
    _require(set(arms) <= set(ABLATIONS), "unknown ablation in declarations")
    declarations, ids = {}, set()
    for arm in selected:
        _require(arm in arms, f"missing explicit frozen declaration for arm: {arm}")
        config = _frozen_configuration(arms[arm], assignments)
        _require(
            config["configuration_id"] not in ids,
            "configuration_id must be unique across selected arms",
        )
        ids.add(config["configuration_id"])
        _require(
            not {"recording_experiment", "replay_configuration"}
            & config["parameters"].keys(),
            "parameters use a reserved replay provenance field",
        )
        _require(
            type(config.get("allow_recording_specific_configs", False)) is bool,
            "allow_recording_specific_configs must be boolean",
        )
        declarations[arm] = config
    return document, kind, declarations


def _archive_parts(paths, trials):
    _require(bool(paths), "at least one recording archive is required")
    parts, recording_ids, raw_ids = [], set(), set()
    for path in paths:
        recording = read_recording(path)
        manifest = recording.manifest
        recording_id, trial_id = manifest["recording_id"], manifest["trial_id"]
        _require(
            recording_id not in recording_ids, f"duplicate recording_id: {recording_id}"
        )
        recording_ids.add(recording_id)
        _require(
            trial_id in trials, f"archive trial is absent from annotations: {trial_id}"
        )
        for field in ("session_id", "object_id", "data_origin"):
            _require(
                manifest["metadata"].get(field) == trials[trial_id][field],
                f"archive {recording_id} {field} differs from independent trial metadata",
            )
        ids = set()
        for record in recording.records:
            if record["record_type"] != "observation":
                continue
            key = (trial_id, record["observation_id"])
            _require(
                key not in raw_ids,
                f"overlapping observation ID across archive parts: {key}",
            )
            ids.add(key)
            raw_ids.add(key)
        parts.append(
            {
                "path": Path(path),
                "manifest": manifest,
                "raw_ids": ids,
                "kind_ids": {
                    kind: {
                        (trial_id, row["observation_id"])
                        for row in recording.records
                        if row.get("kind") == kind
                    }
                    for kind in ("davis_poll", "live_quality_inputs")
                },
            }
        )
    _require(
        {part["manifest"]["trial_id"] for part in parts} == set(trials),
        "archives must cover every declared trial",
    )
    return sorted(
        parts,
        key=lambda part: (
            part["manifest"]["trial_id"],
            part["manifest"]["recording_id"],
        ),
    ), raw_ids


def _applied_configuration(replay, manifest, kind):
    source = replay["configuration"]
    effective = {
        "kind": kind,
        "ablation": source.get("ablation"),
        "replay_implementation": source.get("replay_implementation"),
    }
    if kind == "davis_poll":
        effective["motion"] = source.get("effective_davis_motion_config")
        effective["window"] = manifest["config"].get("event_window_selection")
    else:
        effective["quality"] = source.get("effective_quality_config")
    return effective


def run_recording_experiment(
    recording_dirs, *, annotations, splits, declarations, ablations=None
):
    """Replay selected arms with identical raw observation support and evaluate.

    Parts from separate processes can share one trial when observation IDs are
    disjoint and session/object/origin match. Source archive hashes and replay
    snapshots remain distinct; only the explicitly declared arm is aggregated.
    """
    for document in (annotations, splits, declarations):
        _finite_json(document)
    trials, labels, _ = _annotations(annotations)
    assignments, _ = _split_assignments(splits, trials)
    selected = (
        list(ABLATIONS)
        if ablations is None
        else _strings(list(ablations), "ablations", nonempty=True)
    )
    _require(set(selected) <= set(ABLATIONS), "unknown selected ablation")
    selected = [arm for arm in ABLATIONS if arm in selected]
    document, kind, declared = _declarations(declarations, assignments, selected)
    _require(
        document.get("task_id") == annotations["task_id"],
        "declaration and annotation task_id differ",
    )
    parts, raw_ids = _archive_parts(list(recording_dirs), trials)
    _require(
        set(labels) <= raw_ids,
        "annotation observation ID does not exist in the supplied archives",
    )
    expected_ids = set().union(*(part["kind_ids"][kind] for part in parts))
    prediction_document = {
        "schema_version": PREDICTION_SCHEMA,
        "task_id": annotations["task_id"],
        "configurations": [],
        "predictions": [],
    }
    source_reports, arm_reports = {}, {}
    for part in parts:
        manifest = part["manifest"]
        source_reports[manifest["recording_id"]] = {
            "recording_id": manifest["recording_id"],
            "trial_id": manifest["trial_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "recorded_config_sha256": manifest["config_sha256"],
            "recorded_config": manifest["config"],
            "source_revision": manifest["revision"],
            "metadata": manifest["metadata"],
            "drop_accounting": manifest["summary"],
            "selected_component_observation_count": len(part["kind_ids"][kind]),
            "trace_only_for_selected_component": not bool(part["kind_ids"][kind]),
        }
    for arm in selected:
        sources, replays, fingerprints = [], [], set()
        for part in parts:
            manifest = part["manifest"]
            try:
                replay = replay_recording(part["path"], ablation=arm)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                raise EvaluationContractError(
                    f"cannot replay {manifest['recording_id']} in arm {arm}: {exc}"
                ) from exc
            _require(
                read_document(part["path"] / "manifest.json").get("manifest_sha256")
                == manifest["manifest_sha256"],
                "archive manifest changed during experiment",
            )
            _require(
                replay["recording_id"] == manifest["recording_id"]
                and replay["trial_id"] == manifest["trial_id"],
                "replay identity differs from archive",
            )
            rows = [row for row in replay["observations"] if row.get("kind") == kind]
            observed = {(row["trial_id"], row["observation_id"]) for row in rows}
            _require(
                len(rows) == len(observed) and observed == part["kind_ids"][kind],
                "replay selected observation coverage differs from raw archive",
            )
            applied = _applied_configuration(replay, manifest, kind)
            fingerprint = _digest(applied) if rows else None
            if fingerprint:
                fingerprints.add(fingerprint)
            sources.append(
                {
                    "recording_id": manifest["recording_id"],
                    "trial_id": manifest["trial_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "replay_configuration": replay["configuration"],
                    "applied_inference_configuration": applied if rows else None,
                    "applied_inference_sha256": fingerprint,
                    "observation_ids": sorted(key[1] for key in observed),
                    "trace_only_for_selected_component": not bool(rows),
                }
            )
            replays.append(replay)
        heterogeneous = len(fingerprints) > 1
        _require(
            not heterogeneous
            or declared[arm].get("allow_recording_specific_configs") is True,
            f"arm {arm} has differing applied inference configurations; explicitly declare allow_recording_specific_configs or use uniform recordings",
        )
        config = {key: value for key, value in declared[arm].items() if key != "sha256"}
        config["parameters"] = {
            **config["parameters"],
            "recording_experiment": {
                "ablation": arm,
                "kind": kind,
                "recording_sources": sources,
                "heterogeneous_applied_inference": heterogeneous,
                "configuration_semantics": "recording_dependent_protocol"
                if heterogeneous
                else "uniform_applied_inference",
            },
        }
        config["sha256"] = configuration_digest(config)
        prediction_document["configurations"].append(config)
        arm_ids = set()
        for replay in replays:
            if not any(row.get("kind") == kind for row in replay["observations"]):
                continue
            temporary = {key: value for key, value in config.items() if key != "sha256"}
            temporary["parameters"] = {
                **config["parameters"],
                "replay_configuration": replay["configuration"],
            }
            adapted = replay_predictions_document(
                replay,
                task_id=annotations["task_id"],
                configuration=temporary,
                kind=kind,
            )
            for row in adapted["predictions"]:
                row["config_sha256"] = config["sha256"]
                row["recording_id"] = replay["recording_id"]
                prediction_document["predictions"].append(row)
                arm_ids.add((row["trial_id"], row["observation_id"]))
        _require(
            arm_ids == expected_ids,
            "ablation arms do not share identical selected observation IDs",
        )
        arm_reports[arm] = {
            "configuration_id": config["configuration_id"],
            "selected_observation_count": len(arm_ids),
            "heterogeneous_applied_inference": heterogeneous,
            "applied_inference_sha256": sorted(fingerprints),
        }
    evaluation = evaluate_recorded_trials(annotations, prediction_document, splits)
    pair_sets = [
        {
            (row["trial_id"], row["observation_id"])
            for row in evaluation["results"][item["configuration_id"]]["outcomes"]
            if row["outcome"] in {"correct", "incorrect"}
        }
        for item in arm_reports.values()
    ]
    common_pairs = set.intersection(*pair_sets)
    return {
        "schema_version": EXPERIMENT_SCHEMA,
        "task_id": annotations["task_id"],
        "kind": kind,
        "input_sha256": {
            "annotations": _digest(annotations),
            "splits": _digest(splits),
            "declarations": _digest(declarations),
        },
        "selected_ablations": selected,
        "sources": list(source_reports.values()),
        "arms": arm_reports,
        "coverage": {
            "identical_observation_ids_across_arms": True,
            "selected_raw_observation_count": len(expected_ids),
            "valid_pair_coverage_differs_across_arms": any(
                pairs != pair_sets[0] for pairs in pair_sets[1:]
            ),
            "common_valid_labelled_observations": [
                {"trial_id": trial, "observation_id": observation}
                for trial, observation in sorted(common_pairs)
            ],
        },
        "prediction_document": prediction_document,
        "evaluation": evaluation,
        "physical_validation_established": False,
        "ground_truth_generated": False,
        "limitations": [
            "Component replay does not execute the full RGB-D tracker or reproduce hardware timing.",
            "Frozen configuration and independent annotation declarations cannot be authenticated by this tool.",
            "Equal observation support does not imply equal valid-pair coverage; compare coverage and abstentions.",
            "No physical grasp success, sample adequacy, hardware latency or slip lead time is established.",
        ],
    }


def write_experiment_report(
    report, output, *, input_paths=(), recording_dirs=()
) -> Path:
    path = safe_output_path(
        output, input_paths=input_paths, recording_dirs=recording_dirs
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return path

#!/usr/bin/env python3
"""Exercise offline CLIs using new, explicitly synthetic fixtures only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA = "grasp_synthetic_offline_smoke_v1"
ARMS = (
    "reference",
    "events_disabled",
    "sync_diagnostics_removed",
    "phase_conditioning_removed",
)
TRIALS = ("synthetic-train", "synthetic-validation", "synthetic-test")


class SmokeFailure(RuntimeError):
    """A command or semantic smoke expectation failed."""


def _json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read(path):
    from grasp_project.benchmarking.offline_evaluation import read_document

    return read_document(path)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance():
    unknown = {
        "commit": None,
        "working_tree_dirty": None,
        "git_provenance_status": "UNAVAILABLE",
    }
    executable = shutil.which("git")
    if executable is None:
        return {**unknown, "git_provenance_reason": "git_executable_not_on_path"}
    options = {"capture_output": True, "text": True, "timeout": 30}
    try:
        root = subprocess.run(
            [executable, "-C", str(REPOSITORY), "rev-parse", "--show-toplevel"],
            check=False,
            **options,
        )
        if (
            root.returncode
            or Path(root.stdout.strip()).resolve() != REPOSITORY.resolve()
        ):
            return {
                **unknown,
                "git_provenance_reason": "source_directory_is_not_a_git_worktree_root",
            }
        revision = subprocess.run(
            [executable, "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            check=False,
            **options,
        )
        commit = revision.stdout.strip()
        if (
            revision.returncode
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise SmokeFailure(
                "Git identified a source checkout but its commit could not be established"
            )
        status = subprocess.run(
            [executable, "-C", str(REPOSITORY), "status", "--porcelain"],
            check=False,
            **options,
        )
        if status.returncode:
            return {
                **unknown,
                "commit": commit,
                "git_provenance_status": "PARTIAL",
                "git_provenance_reason": "source_working_tree_state_unavailable",
            }
        return {
            "commit": commit,
            "working_tree_dirty": bool(status.stdout),
            "git_provenance_status": "AVAILABLE",
            "git_provenance_reason": None,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            **unknown,
            "git_provenance_reason": f"git_command_unavailable:{type(exc).__name__}",
        }


def _source():
    files = (
        "scripts/smoke_offline_workflow.py",
        "scripts/replay_recording.py",
        "scripts/estimate_clock_mapping.py",
        "scripts/evaluate_recordings.py",
        "scripts/evaluate_recorded_trials.py",
        "scripts/epsilon_sensitivity.py",
        "src/grasp_project/recording/session.py",
        "src/grasp_project/recording/replay.py",
        "src/grasp_project/recording/clock_mapping.py",
        "src/grasp_project/benchmarking/recording_experiment.py",
        "src/grasp_project/benchmarking/offline_evaluation.py",
        "src/grasp_project/benchmarking/epsilon_sensitivity.py",
        "src/grasp_project/quality/live_quality_monitor.py",
        "src/grasp_project/davis/event_stream.py",
        "src/grasp_project/davis/motion_estimation.py",
        "src/grasp_project/davis/motion_gui.py",
        "src/grasp_project/realsense/epsilon_wrench.py",
    )
    hashes = {name: _sha(REPOSITORY / name) for name in files}
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **_git_provenance(),
        "files_sha256": hashes,
        "implementation_file_set_sha256": fingerprint,
        "implementation_fingerprint_scope": "listed_implementation_files_only_not_full_source_tree",
        "python": sys.version,
    }


def _fixtures(destination, source):
    """Create only constructed data; annotation labels never read predictions."""
    from dataclasses import asdict
    from unittest.mock import patch

    import numpy as np

    from grasp_project.quality.live_quality_monitor import (
        BaselineGraspQuality,
        LiveGraspObservation,
        LiveGraspQualityMonitor,
    )
    from grasp_project.recording.session import RecordingWriter

    window = {
        "primary_window_sec": 0.1,
        "fallback_window_sec": 0.2,
        "min_primary_events": 50,
    }
    effective = LiveGraspQualityMonitor().config
    baseline = asdict(
        BaselineGraspQuality(valid=True, analytic_score=84.0, analytic_label="GOOD")
    )
    revision_identifier = (
        source["commit"]
        or f"implementation-files-sha256:{source['implementation_file_set_sha256']}"
    )
    trials, labels = [], []
    for trial_index, trial_id in enumerate(TRIALS):
        trial = {
            "trial_id": trial_id,
            "session_id": f"synthetic-session-{trial_index}",
            "object_id": f"synthetic-object-{trial_index}",
            "data_origin": "synthetic",
        }
        trials.append(trial)
        # Deterministic fixture creation time is explicitly synthetic, not a measurement.
        with patch(
            "grasp_project.recording.session.time.time_ns",
            return_value=1_800_000_000_000_000_000,
        ):
            writer = RecordingWriter(
                destination / "raw" / trial_id,
                recording_id=trial_id,
                config={"davis_motion": {}, "event_window_selection": window},
                revision={
                    "commit": source["commit"],
                    "working_tree_dirty": source["working_tree_dirty"],
                    "implementation_file_set_sha256": source[
                        "implementation_file_set_sha256"
                    ],
                    "git_provenance_status": source["git_provenance_status"],
                    "synthetic_fixture": True,
                },
                metadata={
                    **trial,
                    "fixture": "constructed_offline_smoke_only",
                    "physical_validation": False,
                },
            )
        for packet_index in range(3):
            indexes = np.arange(120, dtype=np.int64)
            native = np.column_stack(
                (
                    packet_index * 100_000 + indexes * 800,
                    80 + indexes % 40 + packet_index * 2,
                    80 + indexes // 40,
                    indexes % 2,
                )
            )
            arrival_ns = 1_100_000_000 + packet_index * 100_000_000
            writer.record(
                "davis_poll",
                {"events": native},
                observation_id=f"native-packet-{packet_index}",
                host_arrival_monotonic_ns=arrival_ns,
                host_arrival_epoch_ns=1_800_000_000_000_000_000 + arrival_ns,
                native_clock={
                    "domain": "davis_device",
                    "unit": "us",
                    "epoch_id": "synthetic-device-epoch",
                },
                metadata={
                    "normalization_epoch": 0,
                    "normalization_reference_monotonic_sec": arrival_ns / 1_000_000_000,
                    "roi_xywh": [40, 40, 160, 160],
                    "acquisition_endpoint": "synthetic_receipt_fixture",
                },
                source_sequence=packet_index,
            )
        for index, observation_id in enumerate(
            ("quality-quiet", "quality-missing-events")
        ):
            current = index == 0
            observation = LiveGraspObservation(
                timestamp=10.0 + index,
                target_status="TRACKING",
                ready_to_close=True,
                detected_contacts=2,
                visual_contact_ready=True,
                post_grasp_active=True,
                grasp_stage="POST_GRASP_MONITORING",
                event_state="STREAM_ACTIVE_NO_MOTION" if current else "UNAVAILABLE",
                davis_event_freshness="CURRENT" if current else "UNAVAILABLE",
                davis_events_valid_for_live_quality=current,
                davis_motion_usable_for_grasp=current,
                davis_object_motion_valid=current,
                davis_report_heartbeat_current=True,
                davis_motion_confidence=0.9 if current else None,
                davis_motion_class="STABLE" if current else None,
            )
            writer.record(
                "live_quality_inputs",
                {},
                observation_id=observation_id,
                host_arrival_monotonic_ns=10_000_000_000 + index * 1_000_000_000,
                host_arrival_epoch_ns=1_800_000_000_000_000_000 + index * 1_000_000_000,
                native_clock={
                    "domain": "synthetic-application-observation",
                    "unit": "s",
                },
                metadata={
                    "baseline": baseline,
                    "observation": asdict(observation),
                    "effective_config": effective,
                    "state_epoch": 0,
                    "epoch_input_index": index,
                    "initial_state": "constructor_reset"
                    if index == 0
                    else "continuing",
                },
                source_sequence=index,
            )
            labels.append(
                {
                    "annotation_id": f"{trial_id}-{observation_id}",
                    "trial_id": trial_id,
                    "observation_id": observation_id,
                    "state": "labelled",
                    "label": "MONITOR",
                    "provenance": {
                        "source": "synthetic_fixture",
                        "annotator_id": "fixed-fixture-specification",
                        "independent_of_predictions": True,
                        "prediction_blinded": True,
                        "evidence_reference": {"observation_ids": [observation_id]},
                    },
                }
            )
        writer.close()
    _json(
        destination / "inputs/annotations.json",
        {
            "schema_version": "grasp_independent_annotations_v1",
            "task_id": "synthetic-quiet-decision",
            "label_vocabulary": [
                "MONITOR",
                "WAIT_FOR_DAVIS",
                "READY_TO_CLOSE",
                "HOLD_AND_RECHECK",
                "ADJUST",
                "WAIT_FOR_ANALYTIC_QUALITY",
                "DO_NOT_CLOSE",
                "HOLD_AND_RELOCK",
                "REGRIP",
            ],
            "trials": trials,
            "annotations": labels,
        },
    )
    _json(
        destination / "inputs/splits.json",
        {
            "schema_version": "grasp_grouped_split_v1",
            "group_by": ["trial_id", "session_id", "object_id"],
            "assignments": [
                {"trial_id": trial, "split": split}
                for trial, split in zip(TRIALS, ("train", "validation", "test"))
            ],
        },
    )
    _json(
        destination / "inputs/declarations.json",
        {
            "schema_version": "grasp_recording_experiment_v1",
            "task_id": "synthetic-quiet-decision",
            "kind": "live_quality_inputs",
            "arms": {
                arm: {
                    "configuration_id": f"synthetic-{arm}",
                    "revision": revision_identifier,
                    "parameters": {
                        "protocol": "fixed synthetic smoke fixture; no empirical study"
                    },
                    "required_evidence": ["davis_events"],
                    "frozen_before_test": True,
                    "selection_method": "predeclared",
                    "fit_trial_ids": [],
                    "selection_trial_ids": [],
                }
                for arm in ARMS
            },
        },
    )
    _json(
        destination / "inputs/epsilon-samples.json",
        {
            "schema": "grasp_epsilon_samples/v1",
            "constructed_samples": [
                {
                    "sample_id": "synthetic-antipodal-control",
                    "source": {
                        "kind": "constructed",
                        "description": "constructed planar antipodal control; no physical measurement",
                    },
                    "contacts": [
                        {"point": [-1.0, 0.0], "inward_normal": [1.0, 0.0]},
                        {"point": [1.0, 0.0], "inward_normal": [-1.0, 0.0]},
                    ],
                    "object_centroid": [0.0, 0.0],
                    "object_radius": 1.0,
                }
            ],
            "recorded_samples": [],
            "parameters": {"constant_baseline": 0.2},
        },
    )


def _retain(destination, summary):
    summary["artifacts"] = [
        {
            "path": path.relative_to(destination).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
        and path.name not in {"smoke-summary.json", ".smoke-summary.tmp"}
    ]
    temporary = destination / ".smoke-summary.tmp"
    _json(temporary, summary)
    temporary.replace(destination / "smoke-summary.json")


def _command(destination, summary, name, script, arguments, output):
    entry = {
        "name": name,
        "argv": [sys.executable, str(REPOSITORY / "scripts" / script), *arguments],
        "cwd": ".",
        "exit_code": None,
        "output": output,
        "stdout": f"logs/{name}.stdout.txt",
        "stderr": f"logs/{name}.stderr.txt",
    }
    summary["commands"].append(entry)
    _retain(destination, summary)
    try:
        result = subprocess.run(
            entry["argv"],
            cwd=destination,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        entry["error"] = str(exc)
        entry["timed_out"] = isinstance(exc, subprocess.TimeoutExpired)
        _command_logs(
            destination,
            entry,
            getattr(exc, "stdout", None),
            getattr(exc, "stderr", None) or str(exc),
        )
        _retain(destination, summary)
        raise SmokeFailure(f"command {name} could not complete: {exc}") from exc
    entry["exit_code"] = result.returncode
    _command_logs(destination, entry, result.stdout, result.stderr)
    _retain(destination, summary)
    if result.returncode:
        raise SmokeFailure(
            f"command {name} exited {result.returncode}; see {entry['stderr']}"
        )
    return _read(destination / output)


def _command_logs(destination, entry, stdout, stderr):
    for key, value in (("stdout", stdout), ("stderr", stderr)):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        path = destination / entry[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value or "", encoding="utf-8")


def _check(summary, name, condition, details):
    summary["checks"].append(
        {"name": name, "passed": bool(condition), "details": details}
    )
    if not condition:
        raise SmokeFailure(f"smoke expectation failed: {name}")


def _verify_replay(summary, report, trial):
    _check(
        summary,
        f"{trial}:replay-schema",
        report.get("schema_version") == "grasp_component_replay_v1"
        and report.get("physical_validation") is False
        and report.get("ground_truth_generated") is False,
        "Component replay retains no physical-validation or generated-ground-truth claim.",
    )
    quality = {
        row["observation_id"]: row
        for row in report["observations"]
        if row["kind"] == "live_quality_inputs"
    }
    _check(
        summary,
        f"{trial}:quality-support",
        set(quality) == {"quality-quiet", "quality-missing-events"},
        "Both independently specified fixture observations are retained.",
    )
    quiet, missing = quality["quality-quiet"], quality["quality-missing-events"]
    _check(
        summary,
        f"{trial}:quiet-evidence",
        quiet["prediction"]["valid"] is True
        and quiet["evidence"]["davis_events"]["status"] == "valid",
        "Quiet constructed evidence is explicitly current and usable.",
    )
    _check(
        summary,
        f"{trial}:high-score-missing-events",
        missing["prediction"]["score"] >= 75
        and missing["evidence"]["davis_events"]["status"] == "unavailable"
        and missing["prediction"]["decision"] == "WAIT_FOR_DAVIS",
        "A retained high raw score with missing events does not become measured stability.",
    )
    _check(
        summary,
        f"{trial}:motion-component",
        sum(row["kind"] == "davis_poll" for row in report["observations"]) == 3,
        "All integer native event packets traverse the motion component.",
    )


def _verify_clock(summary, report, trial):
    _check(
        summary,
        f"{trial}:clock-contract",
        report.get("schema_version") == "grasp_clock_mapping_diagnostics_v1"
        and report.get("status") == "ESTIMATED_UNVALIDATED"
        and report.get("physical_alignment_established") is False
        and report.get("applied_to_live_policy") is False,
        "Clock diagnostics remain unvalidated host receipt associations.",
    )
    extraction = report["extraction"]
    _check(
        summary,
        f"{trial}:integer-clock-pairs",
        extraction["used_pair_count"] == 3
        and extraction["unavailable_observation_count"] == 0
        and extraction["interpretation"] == "HOST_RECEIPT_ASSOCIATION_ONLY",
        "Three exact integer native packet endpoints are associated with synthetic host receipts.",
    )
    _check(
        summary,
        f"{trial}:clock-model",
        len(report["epochs"]) == 1
        and report["epochs"][0]["model"] is not None
        and report["epochs"][0]["diagnostics"][
            "zero_residual_proves_zero_physical_uncertainty"
        ]
        is False,
        "A constructed affine fit does not establish physical alignment or timing uncertainty.",
    )


def _verify_evaluation(summary, experiment):
    _check(
        summary,
        "experiment-schema",
        experiment.get("schema_version") == "grasp_recording_experiment_report_v1"
        and experiment.get("physical_validation_established") is False
        and experiment.get("ground_truth_generated") is False,
        "The complete offline pipeline preserves synthetic-only interpretation.",
    )
    _check(
        summary,
        "all-ablation-arms",
        experiment["selected_ablations"] == list(ARMS)
        and experiment["coverage"]["identical_observation_ids_across_arms"] is True
        and experiment["coverage"]["selected_raw_observation_count"] == 6,
        "All four registered arms use the same six quality observation IDs.",
    )
    for arm in ARMS:
        result = experiment["evaluation"]["results"][f"synthetic-{arm}"]
        expected_pairs = 0 if arm == "events_disabled" else 1
        for split in ("train", "validation", "test"):
            metrics = result["splits"][split]
            _check(
                summary,
                f"{arm}:{split}:abstentions",
                metrics["labelled_count"] == 2
                and metrics["paired_count"] == expected_pairs
                and metrics["abstention_count"] == 2 - expected_pairs,
                "Fixed independent fixture labels remain separate from unavailable event evidence.",
            )
        missing = [
            row
            for row in result["outcomes"]
            if row["observation_id"] == "quality-missing-events"
        ]
        _check(
            summary,
            f"{arm}:missing-event-outcomes",
            len(missing) == 3
            and all(
                row["outcome"] == "unusable_evidence"
                and row["prediction"]["score"] >= 75
                for row in missing
            ),
            "Every high-score missing-event observation is an evidence abstention.",
        )
    _check(
        summary,
        "synthetic-strata-only",
        experiment["evaluation"]["data_origins"] == {"synthetic": 3},
        "No physical recordings or empirical labels are present.",
    )


def _verify_epsilon(summary, epsilon):
    _check(
        summary,
        "epsilon-schema-and-origin",
        epsilon.get("schema") == "grasp_epsilon_sensitivity/v1"
        and epsilon.get("physical_validation") is False
        and epsilon["recorded"]["distribution"]["sample_count"] == 0
        and epsilon["constructed"]["distribution"]["sample_count"] == 1,
        "Constructed geometry controls remain separate from absent recorded distributions.",
    )
    baseline = epsilon["constructed"]["distribution"]["constant_baseline"]
    _check(
        summary,
        "epsilon-fixed-constant-control",
        baseline["value"] == 0.2
        and baseline["selection"] == "fixed_user_parameter_not_fitted",
        "The constant baseline is literal and never fitted to fixture predictions.",
    )
    controls = epsilon["constructed"]["samples"][0]["constructed_controls"]
    invariants = [row for row in controls if row["expected_geometric_invariance"]]
    _check(
        summary,
        "epsilon-constructed-invariances",
        len(invariants) == 3
        and {row["name"] for row in invariants}
        == {"translation", "uniform_scale", "rotation"}
        and all(row["invariance_passed"] is True for row in invariants),
        "All three named joint geometry invariances pass without an empirical grasp claim.",
    )


def run_smoke(destination):
    requested = Path(destination)
    if requested.exists() or requested.is_symlink():
        raise SmokeFailure(
            "destination must be new; existing content is never overwritten"
        )
    destination = requested.resolve()
    if destination.exists():
        raise SmokeFailure(
            "destination must be new; existing content is never overwritten"
        )
    destination.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": SCHEMA,
        "status": "RUNNING",
        "synthetic_only": True,
        "physical_validation_established": False,
        "physical_alignment_established": False,
        "commands": [],
        "checks": [],
        "artifacts": [],
        "limitations": [
            "Constructed fixtures establish software behavior only; no physical trial or independent empirical label is supplied.",
            "No hardware latency, slip lead time, success rate or sample adequacy is established.",
            "Synthetic receipt associations do not align physical camera clocks.",
        ],
    }
    try:
        sys.path.insert(0, str(REPOSITORY / "src"))
        summary["source"] = _source()
        _retain(destination, summary)
        _fixtures(destination, summary["source"])
        for trial in TRIALS:
            raw, replay_output, clock_output = (
                f"raw/{trial}",
                f"reports/{trial}-replay.json",
                f"reports/{trial}-clock.json",
            )
            replay = _command(
                destination,
                summary,
                f"{trial}-replay",
                "replay_recording.py",
                [raw, "--output", replay_output],
                replay_output,
            )
            _verify_replay(summary, replay, trial)
            clock = _command(
                destination,
                summary,
                f"{trial}-clock",
                "estimate_clock_mapping.py",
                [
                    "--recording",
                    raw,
                    "--stream",
                    "davis_poll",
                    "--output",
                    clock_output,
                ],
                clock_output,
            )
            _verify_clock(summary, clock, trial)
        aggregate = [
            argument for trial in TRIALS for argument in ("--recording", f"raw/{trial}")
        ]
        aggregate += [
            "--annotations",
            "inputs/annotations.json",
            "--splits",
            "inputs/splits.json",
            "--declarations",
            "inputs/declarations.json",
            "--all-ablations",
            "--output",
            "reports/experiment.json",
        ]
        experiment = _command(
            destination,
            summary,
            "all-arm-experiment",
            "evaluate_recordings.py",
            aggregate,
            "reports/experiment.json",
        )
        _verify_evaluation(summary, experiment)
        _json(
            destination / "inputs/replayed-predictions.json",
            experiment["prediction_document"],
        )
        standalone = _command(
            destination,
            summary,
            "standalone-evaluation",
            "evaluate_recorded_trials.py",
            [
                "--annotations",
                "inputs/annotations.json",
                "--predictions",
                "inputs/replayed-predictions.json",
                "--splits",
                "inputs/splits.json",
                "--output",
                "reports/standalone-evaluation.json",
            ],
            "reports/standalone-evaluation.json",
        )
        _check(
            summary,
            "standalone-evaluation-agrees",
            standalone == experiment["evaluation"],
            "Standalone and aggregate evaluators produce identical reports from the retained prediction interchange.",
        )
        epsilon = _command(
            destination,
            summary,
            "constructed-epsilon",
            "epsilon_sensitivity.py",
            ["inputs/epsilon-samples.json", "--output", "reports/epsilon.json"],
            "reports/epsilon.json",
        )
        _verify_epsilon(summary, epsilon)
        summary["status"] = "PASSED_SYNTHETIC_ONLY"
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ImportError,
        subprocess.SubprocessError,
    ) as exc:
        summary["status"] = "FAILED_SYNTHETIC_SMOKE"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        _retain(destination, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        help="NEW directory for synthetic archives, reports, command logs and summary",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_smoke(args.destination)
    except (OSError, SmokeFailure) as exc:
        print(f"offline synthetic smoke: {exc}", file=sys.stderr)
        return 1
    print(f"{summary['status']}: {args.destination.resolve() / 'smoke-summary.json'}")
    if summary["status"] != "PASSED_SYNTHETIC_ONLY":
        print(summary.get("error", {}).get("message", "smoke failed"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

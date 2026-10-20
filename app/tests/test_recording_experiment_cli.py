"""Subprocess archive -> actual replay -> evaluation on synthetic fixtures only."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from grasp_project.benchmarking.offline_evaluation import (
    ANNOTATION_SCHEMA,
    SPLIT_SCHEMA,
)
from grasp_project.benchmarking.recording_experiment import (
    DECLARATION_SCHEMA,
    run_recording_experiment,
)
from grasp_project.quality.live_quality_monitor import (
    BaselineGraspQuality,
    LiveGraspObservation,
    LiveGraspQualityMonitor,
)
from grasp_project.recording.replay import ABLATIONS
from grasp_project.recording.session import RecordingWriter, read_recording

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_recordings.py"
WINDOW = {
    "primary_window_sec": 0.1,
    "fallback_window_sec": 0.2,
    "min_primary_events": 50,
}
DECISIONS = [
    "MONITOR",
    "WAIT_FOR_DAVIS",
    "READY_TO_CLOSE",
    "HOLD_AND_RECHECK",
    "ADJUST",
    "WAIT_FOR_ANALYTIC_QUALITY",
    "DO_NOT_CLOSE",
    "HOLD_AND_RELOCK",
    "REGRIP",
]
MOTIONS = [
    "STABLE",
    "UNKNOWN_MOTION",
    "BACKGROUND_MOTION",
    "CAMERA_SHAKE",
    "LIGHTING_CHANGE",
    "IMPACT_ONLY",
    "GROSS_SLIP",
    "SPIN_OR_ROTATION",
    "SHAKE_OR_VIBRATION",
    "INCIPIENT_SLIP",
]


def _writer(path, trial, *, recording_id=None):
    return RecordingWriter(
        path,
        recording_id=recording_id or trial["trial_id"],
        config={
            "davis_motion": {},
            "event_window_selection": WINDOW,
            "device_serial": trial["object_id"],
        },
        revision={"commit": "synthetic-fixture", "dirty": False},
        metadata=trial,
    )


def _observation(**changes):
    values = {
        "timestamp": 10.0,
        "target_status": "TRACKING",
        "ready_to_close": True,
        "detected_contacts": 2,
        "visual_contact_ready": True,
        "event_state": "STREAM_ACTIVE_NO_MOTION",
        "davis_event_freshness": "CURRENT",
        "davis_events_valid_for_live_quality": True,
        "davis_motion_usable_for_grasp": True,
        "davis_object_motion_valid": True,
        "davis_report_heartbeat_current": True,
        "davis_motion_confidence": 0.9,
        "davis_motion_class": "STABLE",
        "post_grasp_active": True,
        "grasp_stage": "POST_GRASP_MONITORING",
    }
    values.update(changes)
    return LiveGraspObservation(**values)


def _write_quality(writer, *, config=None):
    config = LiveGraspQualityMonitor(config).config
    baseline = asdict(
        BaselineGraspQuality(valid=True, analytic_score=84.0, analytic_label="GOOD")
    )
    changes = [
        {},
        {"davis_event_freshness": "STALE"},
        {
            "davis_motion_usable_for_grasp": False,
            "davis_motion_excluded_reason": "outside_object_roi",
        },
        {
            "event_state": "UNAVAILABLE",
            "davis_event_freshness": "UNAVAILABLE",
            "davis_events_valid_for_live_quality": False,
            "davis_motion_usable_for_grasp": False,
        },
    ]
    for index, change in enumerate(changes):
        writer.record(
            "live_quality_inputs",
            {},
            observation_id=f"quality-{index}",
            host_arrival_monotonic_ns=10_000_000_000 + index * 1_000_000_000,
            host_arrival_epoch_ns=1_800_000_000_000_000_000 + index * 1_000_000_000,
            native_clock={"domain": "application_observation_unspecified", "unit": "s"},
            metadata={
                "baseline": baseline,
                "observation": asdict(_observation(timestamp=10.0 + index, **change)),
                "effective_config": config,
                "state_epoch": 0,
                "epoch_input_index": index,
                "initial_state": "constructor_reset" if index == 0 else "continuing",
            },
            source_sequence=index,
        )


def _write_motion(writer):
    for index in range(2):
        rng = np.random.default_rng(5)
        native = np.column_stack(
            [
                np.linspace(index * 100000, (index + 1) * 100000 - 1, 120),
                np.linspace(-16, 16, 120) + 100 + index * 5,
                rng.normal(100, 1, 120),
                np.arange(120) % 2,
            ]
        )
        writer.record(
            "davis_poll",
            {"events": native},
            observation_id=f"packet-{index}",
            host_arrival_monotonic_ns=1_100_000_000 + index * 100_000_000,
            host_arrival_epoch_ns=10_100_000_000 + index * 100_000_000,
            native_clock={"domain": "davis_device", "unit": "us"},
            metadata={"normalization_epoch": 0, "roi_xywh": [40, 40, 160, 160]},
            source_sequence=index,
        )


def _label(trial_id, observation_id):
    return {
        "annotation_id": f"{trial_id}-{observation_id}",
        "trial_id": trial_id,
        "observation_id": observation_id,
        "state": "labelled",
        "label": "MONITOR",
        "provenance": {
            "source": "synthetic_fixture",
            "annotator_id": "fixture-author",
            "independent_of_predictions": True,
            "prediction_blinded": True,
            "evidence_reference": {"observation_ids": [observation_id]},
        },
    }


@pytest.fixture
def experiment(tmp_path):
    trials = [
        {
            "trial_id": f"trial-{index}",
            "session_id": f"session-{index}",
            "object_id": f"object-{index}",
            "data_origin": "synthetic",
        }
        for index in range(3)
    ]
    archives, labels = [], []
    for trial in trials:
        writer = _writer(tmp_path / trial["trial_id"], trial)
        _write_quality(writer)
        _write_motion(writer)
        writer.close()
        archives.append(writer.root)
        # Quiet-scene fixture labels are specified independently of every replay output.
        labels.extend(
            _label(trial["trial_id"], f"quality-{index}") for index in range(4)
        )
    annotations = {
        "schema_version": ANNOTATION_SCHEMA,
        "task_id": "fixture_quality_decision",
        "label_vocabulary": DECISIONS,
        "trials": trials,
        "annotations": labels,
    }
    splits = {
        "schema_version": SPLIT_SCHEMA,
        "group_by": ["trial_id", "session_id", "object_id"],
        "assignments": [
            {"trial_id": trial["trial_id"], "split": split}
            for trial, split in zip(trials, ("train", "validation", "test"))
        ],
    }
    declarations = {
        "schema_version": DECLARATION_SCHEMA,
        "task_id": annotations["task_id"],
        "kind": "live_quality_inputs",
        "arms": {
            arm: {
                "configuration_id": f"fixture-{arm}",
                "revision": "synthetic-fixture",
                "parameters": {"study": "synthetic contract test"},
                "required_evidence": ["davis_events"],
                "frozen_before_test": True,
                "selection_method": "predeclared",
                "fit_trial_ids": [],
                "selection_trial_ids": [],
            }
            for arm in ABLATIONS
        },
    }
    return {
        "archives": archives,
        "annotations": annotations,
        "splits": splits,
        "declarations": declarations,
        "root": tmp_path,
    }


def _arguments(experiment, *, output="report.json", flags=()):
    arguments = []
    for archive in experiment["archives"]:
        arguments += ["--recording", str(archive)]
    for name in ("annotations", "splits", "declarations"):
        path = experiment["root"] / f"{name}.json"
        path.write_text(json.dumps(experiment[name]), encoding="utf-8")
        arguments += [f"--{name}", str(path)]
    if output is not None:
        arguments += ["--output", str(experiment["root"] / output)]
    return [*arguments, *flags]


def _run(experiment, *, output="report.json", flags=()):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *_arguments(experiment, output=output, flags=flags),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_needs_no_sensor_or_installed_site_packages():
    run = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "--all-ablations" in run.stdout


def test_full_cli_all_arms_deterministic_real_replay_and_independent_evaluation(
    experiment,
):
    first = _run(experiment, flags=["--all-ablations"])
    assert first.returncode == 0, first.stderr
    second = _run(experiment, output="repeat.json", flags=["--all-ablations"])
    assert second.returncode == 0, second.stderr
    report_path = experiment["root"] / "report.json"
    assert report_path.read_bytes() == (experiment["root"] / "repeat.json").read_bytes()
    report = json.loads(report_path.read_text())
    assert report["selected_ablations"] == list(ABLATIONS)
    assert (
        report["physical_validation_established"]
        is report["ground_truth_generated"]
        is False
    )
    assert report["coverage"]["identical_observation_ids_across_arms"] is True
    assert report["coverage"]["selected_raw_observation_count"] == 12
    assert report["coverage"]["valid_pair_coverage_differs_across_arms"] is True
    assert report["coverage"]["common_valid_labelled_observations"] == []
    test = report["evaluation"]["results"]["fixture-reference"]["splits"]["test"]
    assert test["labelled_count"] == 4
    assert test["paired_count"] == 1
    assert test["abstention_count"] == 3
    assert test["coverage_of_labelled"] == 0.25
    outcomes = report["evaluation"]["results"]["fixture-reference"]["outcomes"][-4:]
    assert [row["evidence"]["davis_events"]["status"] for row in outcomes] == [
        "valid",
        "stale",
        "excluded",
        "unavailable",
    ]
    assert all(row["prediction"]["score"] > 50 for row in outcomes)
    disabled = report["evaluation"]["results"]["fixture-events_disabled"]["splits"][
        "test"
    ]
    assert disabled["paired_count"] == 0
    assert disabled["accuracy_on_valid_pairs"] is None
    # Per-recording config provenance differs, while applied inference remains uniform.
    assert len({source["recorded_config_sha256"] for source in report["sources"]}) == 3
    assert all(
        not arm["heterogeneous_applied_inference"] for arm in report["arms"].values()
    )
    snapshots = report["evaluation"]["configurations"]["fixture-reference"][
        "parameters"
    ]["recording_experiment"]["recording_sources"]
    assert len(snapshots) == 3
    assert len({source["replay_configuration"]["sha256"] for source in snapshots}) == 3


def test_cli_selected_subset_and_stdout(experiment):
    run = _run(
        experiment,
        output=None,
        flags=["--ablation", "phase_conditioning_removed", "--ablation", "reference"],
    )
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["selected_ablations"] == ["reference", "phase_conditioning_removed"]


def test_required_evidence_absent_from_actual_replay_is_missing_never_stable(
    experiment,
):
    experiment["declarations"]["arms"]["reference"]["required_evidence"].append(
        "independent_rgbd_evidence"
    )
    run = _run(experiment)
    assert run.returncode == 0, run.stderr
    report = json.loads((experiment["root"] / "report.json").read_text())
    result = report["evaluation"]["results"]["fixture-reference"]
    assert result["splits"]["test"]["paired_count"] == 0
    assert result["splits"]["test"]["accuracy_on_valid_pairs"] is None
    assert all(
        row["unusable_required_evidence"]["independent_rgbd_evidence"]["status"]
        == "missing"
        for row in result["outcomes"]
    )


def test_cli_motion_component_uses_actual_estimator_all_arms(experiment):
    experiment["declarations"]["kind"] = "davis_poll"
    experiment["annotations"]["label_vocabulary"] = MOTIONS
    experiment["annotations"]["annotations"] = []
    for trial in experiment["annotations"]["trials"]:
        for index in range(2):
            label = _label(trial["trial_id"], f"packet-{index}")
            label["label"] = (
                "STABLE"  # Constructed fixture control, independent of the estimator.
            )
            experiment["annotations"]["annotations"].append(label)
    run = _run(experiment, flags=["--all-ablations"])
    assert run.returncode == 0, run.stderr
    report = json.loads((experiment["root"] / "report.json").read_text())
    assert report["coverage"]["selected_raw_observation_count"] == 6
    rows = report["prediction_document"]["predictions"]
    assert all(row["prediction"]["decision"] is None for row in rows)
    disabled = [
        row for row in rows if row["configuration_id"] == "fixture-events_disabled"
    ]
    assert all(
        row["prediction"]["valid"] is False and row["prediction"]["label"] is None
        for row in disabled
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_frozen",
        "test_tuning",
        "prediction_labels",
        "missing_arm",
        "bad_task",
        "unknown_arm",
    ],
)
def test_cli_rejects_ambiguous_study_contracts_without_report(experiment, problem):
    config = experiment["declarations"]["arms"]["reference"]
    flags = []
    if problem == "missing_frozen":
        del config["frozen_before_test"]
    elif problem == "test_tuning":
        config.update(selection_method="train_validation", fit_trial_ids=["trial-2"])
    elif problem == "prediction_labels":
        experiment["annotations"]["annotations"][0]["provenance"]["source"] = (
            "model_prediction"
        )
    elif problem == "missing_arm":
        del experiment["declarations"]["arms"]["events_disabled"]
        flags = ["--all-ablations"]
    elif problem == "bad_task":
        experiment["declarations"]["task_id"] = "other-task"
    else:
        flags = ["--ablation", "not-an-arm"]
    run = _run(experiment, flags=flags)
    assert run.returncode == 2
    assert "recording experiment error:" in run.stderr
    assert not (experiment["root"] / "report.json").exists()


@pytest.mark.parametrize(
    "target",
    [
        "annotations.json",
        "trial-0/manifest.json",
        "trial-0/new-output.json",
        "existing.json",
    ],
)
def test_cli_cannot_overwrite_inputs_existing_files_or_archive_contents(
    experiment, target
):
    if target == "existing.json":
        (experiment["root"] / target).write_text("retain me", encoding="utf-8")
    before = {
        str(path): path.read_bytes()
        for archive in experiment["archives"]
        for path in archive.rglob("*")
        if path.is_file()
    }
    run = _run(experiment, output=target)
    assert run.returncode == 2
    assert "output" in run.stderr
    assert before == {
        str(path): path.read_bytes()
        for archive in experiment["archives"]
        for path in archive.rglob("*")
        if path.is_file()
    }
    if target == "existing.json":
        assert (experiment["root"] / target).read_text() == "retain me"


@pytest.mark.parametrize("corruption", ["array", "manifest", "annotation_json"])
def test_cli_corruption_fails_closed_no_report(experiment, corruption):
    arguments = _arguments(experiment)
    if corruption == "array":
        payload = next((experiment["archives"][0] / "arrays").glob("*.npy"))
        data = bytearray(payload.read_bytes())
        data[-1] ^= 1
        payload.write_bytes(data)
    elif corruption == "manifest":
        path = experiment["archives"][0] / "manifest.json"
        path.write_text("{", encoding="utf-8")
    else:
        (experiment["root"] / "annotations.json").write_text(
            '{"task_id": "one", "task_id": "two"}', encoding="utf-8"
        )
    run = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 2
    assert "recording experiment error" in run.stderr
    assert not (experiment["root"] / "report.json").exists()


def test_disjoint_process_parts_same_trial_and_auxiliary_traces_are_retained(
    experiment,
):
    trial = experiment["annotations"]["trials"][0]
    auxiliary = _writer(
        experiment["root"] / "trial-0-rgb-process",
        trial,
        recording_id="trial-0-rgb-process",
    )
    auxiliary.record(
        "rgb",
        {"raw_bytes": np.zeros(6, dtype=np.uint8)},
        observation_id="rgb-only",
        host_arrival_monotonic_ns=100,
        host_arrival_epoch_ns=200,
        native_clock={"domain": "ros_header_unspecified", "unit": "ns"},
    )
    auxiliary.close()
    experiment["archives"].append(auxiliary.root)
    run = _run(experiment, flags=["--all-ablations"])
    assert run.returncode == 0, run.stderr
    report = json.loads((experiment["root"] / "report.json").read_text())
    source = next(
        source
        for source in report["sources"]
        if source["recording_id"] == "trial-0-rgb-process"
    )
    assert source["trace_only_for_selected_component"] is True
    assert source["selected_component_observation_count"] == 0
    assert report["coverage"]["selected_raw_observation_count"] == 12
    assert (
        report["evaluation"]["results"]["fixture-reference"]["splits"]["train"][
            "paired_count"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "conflict",
    [
        "duplicate_recording",
        "overlap",
        "metadata",
        "missing_trial",
        "missing_observation",
    ],
)
def test_archive_identity_or_coverage_conflicts_rejected(experiment, conflict):
    if conflict == "duplicate_recording":
        experiment["archives"].append(experiment["archives"][0])
    elif conflict in {"overlap", "metadata"}:
        trial = copy.deepcopy(experiment["annotations"]["trials"][0])
        if conflict == "metadata":
            trial["session_id"] = "wrong-session"
        auxiliary = _writer(experiment["root"] / "extra", trial, recording_id="extra")
        auxiliary.record(
            "rgb",
            {},
            observation_id="quality-0"
            if conflict == "overlap"
            else "extra-observation",
            host_arrival_monotonic_ns=100,
            host_arrival_epoch_ns=200,
            native_clock={"domain": "synthetic", "unit": "ns"},
        )
        auxiliary.close()
        experiment["archives"].append(auxiliary.root)
    elif conflict == "missing_trial":
        experiment["archives"].pop()
    else:
        experiment["annotations"]["annotations"][0]["observation_id"] = "nonexistent"
        experiment["annotations"]["annotations"][0]["provenance"][
            "evidence_reference"
        ] = {"observation_ids": ["nonexistent"]}
    run = _run(experiment)
    assert run.returncode == 2
    assert not (experiment["root"] / "report.json").exists()


def test_differing_applied_settings_need_explicit_recording_dependent_protocol(
    experiment,
):
    trial = experiment["annotations"]["trials"][0]
    original = next(
        row["metadata"]["effective_config"]
        for row in read_recording(experiment["archives"][0]).records
        if row.get("kind") == "live_quality_inputs"
    )
    changed = copy.deepcopy(original)
    changed["weights"]["incipient_slip"] += 12
    replacement = _writer(
        experiment["root"] / "different-config", trial, recording_id="different-config"
    )
    _write_quality(replacement, config=changed)
    _write_motion(replacement)
    replacement.close()
    experiment["archives"][0] = replacement.root
    run = _run(experiment)
    assert run.returncode == 2
    assert "differing applied inference" in run.stderr
    experiment["declarations"]["arms"]["reference"][
        "allow_recording_specific_configs"
    ] = True
    run = _run(experiment)
    assert run.returncode == 0, run.stderr
    report = json.loads((experiment["root"] / "report.json").read_text())
    assert report["arms"]["reference"]["heterogeneous_applied_inference"] is True
    parameters = report["evaluation"]["configurations"]["fixture-reference"][
        "parameters"
    ]["recording_experiment"]
    assert parameters["configuration_semantics"] == "recording_dependent_protocol"


def test_reordered_recording_inputs_have_identical_report(experiment):
    report = run_recording_experiment(
        experiment["archives"],
        annotations=experiment["annotations"],
        splits=experiment["splits"],
        declarations=experiment["declarations"],
    )
    assert report == run_recording_experiment(
        list(reversed(experiment["archives"])),
        annotations=experiment["annotations"],
        splits=experiment["splits"],
        declarations=experiment["declarations"],
    )

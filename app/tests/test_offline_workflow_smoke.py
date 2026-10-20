"""Actual offline CLI smoke on constructed fixtures; never physical validation."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_offline_workflow.py"
SPEC = importlib.util.spec_from_file_location("smoke_offline_workflow", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


def _run(destination, cwd, *, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(destination)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_help_works_without_site_packages_or_hardware_imports(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "synthetic" in result.stdout


def test_actual_full_command_from_unrelated_cwd_retains_verified_synthetic_outcome(
    tmp_path,
):
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    destination = tmp_path / "synthetic-smoke"
    result = _run(destination, unrelated)
    assert result.returncode == 0, result.stderr
    assert "PASSED_SYNTHETIC_ONLY" in result.stdout
    summary = json.loads((destination / "smoke-summary.json").read_text())
    assert summary["schema_version"] == "grasp_synthetic_offline_smoke_v1"
    assert summary["status"] == "PASSED_SYNTHETIC_ONLY"
    assert summary["synthetic_only"] is True
    assert (
        summary["physical_validation_established"]
        is summary["physical_alignment_established"]
        is False
    )
    assert len(summary["commands"]) == 9
    assert all(command["exit_code"] == 0 for command in summary["commands"])
    assert all(check["passed"] for check in summary["checks"])
    source = summary["source"]
    if source["commit"] is None:
        assert source["git_provenance_status"] == "UNAVAILABLE"
        assert source["working_tree_dirty"] is None
        assert source["git_provenance_reason"]
    else:
        assert len(source["commit"]) == 40
    assert len(source["implementation_file_set_sha256"]) == 64
    assert (
        source["implementation_fingerprint_scope"]
        == "listed_implementation_files_only_not_full_source_tree"
    )
    assert all(
        len(digest) == 64 for digest in summary["source"]["files_sha256"].values()
    )
    assert all(
        command["cwd"] == "." and "record_native_trial.py" not in command["argv"][1]
        for command in summary["commands"]
    )
    for artifact in summary["artifacts"]:
        path = destination / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    annotations = json.loads((destination / "inputs/annotations.json").read_text())
    assert {label["label"] for label in annotations["annotations"]} == {"MONITOR"}
    assert all(
        label["provenance"]["source"] == "synthetic_fixture"
        for label in annotations["annotations"]
    )
    experiment = json.loads((destination / "reports/experiment.json").read_text())
    reference = experiment["evaluation"]["results"]["synthetic-reference"]["splits"][
        "test"
    ]
    assert reference["labelled_count"] == 2
    assert reference["paired_count"] == reference["abstention_count"] == 1
    for trial in smoke.TRIALS:
        manifest = json.loads(
            (destination / "raw" / trial / "manifest.json").read_text()
        )
        packets = [
            row for row in manifest["records"] if row.get("kind") == "davis_poll"
        ]
        assert len(packets) == 3
        assert all(row["arrays"]["events"]["dtype"].endswith("i8") for row in packets)
        assert all(
            type(row["metadata"]["normalization_epoch"]) is int for row in packets
        )
        assert all(
            "normalization_reference_monotonic_sec" in row["metadata"]
            for row in packets
        )
    epsilon = json.loads((destination / "reports/epsilon.json").read_text())
    assert epsilon["recorded"]["distribution"]["sample_count"] == 0
    assert epsilon["constructed"]["distribution"]["constant_baseline"]["value"] == 0.2


def test_identical_fixtures_and_implementation_produce_identical_retained_summary(
    tmp_path,
):
    first, second = tmp_path / "first", tmp_path / "second"
    initial = _run(first, tmp_path)
    repeated = _run(second, tmp_path)
    assert initial.returncode == repeated.returncode == 0, (
        initial.stderr,
        repeated.stderr,
    )
    assert (first / "smoke-summary.json").read_bytes() == (
        second / "smoke-summary.json"
    ).read_bytes()


def test_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / "retain-existing"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("retain user content", encoding="utf-8")
    result = _run(destination, tmp_path)
    assert result.returncode == 1
    assert "destination must be new" in result.stderr
    assert sentinel.read_text() == "retain user content"
    assert not (destination / "smoke-summary.json").exists()


def test_failed_actual_subcommand_retains_partial_summary_logs_and_nonzero_exit(
    tmp_path, monkeypatch
):
    original = smoke._fixtures

    def corrupted_fixtures(destination, source):
        original(destination, source)
        payload = next((destination / "raw" / smoke.TRIALS[0] / "arrays").glob("*.npy"))
        data = bytearray(payload.read_bytes())
        data[-1] ^= 1
        payload.write_bytes(data)

    monkeypatch.setattr(smoke, "_fixtures", corrupted_fixtures)
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "corrupted-fixture"
    assert smoke.main([str(destination)]) == 1
    summary = json.loads((destination / "smoke-summary.json").read_text())
    assert summary["status"] == "FAILED_SYNTHETIC_SMOKE"
    assert len(summary["commands"]) == 1
    assert summary["commands"][0]["exit_code"] == 1
    stderr = destination / summary["commands"][0]["stderr"]
    assert "checksum mismatch" in stderr.read_text()
    assert any(
        artifact["path"].endswith(".stderr.txt") for artifact in summary["artifacts"]
    )


def test_successful_commands_with_wrong_abstention_denominators_fail_smoke(
    tmp_path, monkeypatch
):
    original = smoke._fixtures

    def mismatched_requirements(destination, source):
        original(destination, source)
        path = destination / "inputs/declarations.json"
        document = json.loads(path.read_text())
        for config in document["arms"].values():
            config["required_evidence"] = ["deliberately_missing_fixture_requirement"]
        smoke._json(path, document)

    monkeypatch.setattr(smoke, "_fixtures", mismatched_requirements)
    destination = tmp_path / "wrong-denominator"
    assert smoke.main([str(destination)]) == 1
    summary = json.loads((destination / "smoke-summary.json").read_text())
    assert summary["status"] == "FAILED_SYNTHETIC_SMOKE"
    assert all(command["exit_code"] == 0 for command in summary["commands"])
    assert summary["checks"][-1]["passed"] is False
    assert summary["checks"][-1]["name"] == "reference:train:abstentions"


def test_wrong_report_schema_fails_smoke_even_after_successful_command(
    tmp_path, monkeypatch
):
    original = smoke._command

    def wrong_schema(*args, **kwargs):
        report = original(*args, **kwargs)
        report["schema_version"] = "deliberately-invalid-schema"
        return report

    monkeypatch.setattr(smoke, "_command", wrong_schema)
    destination = tmp_path / "wrong-schema"
    assert smoke.main([str(destination)]) == 1
    summary = json.loads((destination / "smoke-summary.json").read_text())
    assert summary["commands"][0]["exit_code"] == 0
    assert summary["checks"][-1]["passed"] is False
    assert summary["error"]["message"].endswith("replay-schema")


def test_unreadable_implementation_provenance_still_retains_failure_summary(
    tmp_path, monkeypatch
):
    def unavailable_source():
        raise smoke.SmokeFailure("synthetic source provenance failure")

    monkeypatch.setattr(smoke, "_source", unavailable_source)
    destination = tmp_path / "source-failure"
    assert smoke.main([str(destination)]) == 1
    summary = json.loads((destination / "smoke-summary.json").read_text())
    assert summary["status"] == "FAILED_SYNTHETIC_SMOKE"
    assert summary["commands"] == []
    assert summary["error"]["message"] == "synthetic source provenance failure"


def test_full_command_without_git_keeps_explicit_unknown_commit_and_file_fingerprint(
    tmp_path,
):
    environment = os.environ.copy()
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    environment["PATH"] = str(empty_path)
    destination = tmp_path / "no-git-smoke"
    result = _run(destination, tmp_path, env=environment)
    assert result.returncode == 0, result.stderr
    summary = json.loads((destination / "smoke-summary.json").read_text())
    source = summary["source"]
    assert source["commit"] is source["working_tree_dirty"] is None
    assert source["git_provenance_reason"] == "git_executable_not_on_path"
    assert source["git_provenance_status"] == "UNAVAILABLE"
    assert len(source["implementation_file_set_sha256"]) == 64
    declarations = json.loads((destination / "inputs/declarations.json").read_text())
    expected_revision = (
        f"implementation-files-sha256:{source['implementation_file_set_sha256']}"
    )
    assert all(
        config["revision"] == expected_revision
        for config in declarations["arms"].values()
    )
    archive = json.loads(
        (destination / "raw" / smoke.TRIALS[0] / "manifest.json").read_text()
    )
    assert archive["revision"]["commit"] is None
    assert (
        archive["revision"]["implementation_file_set_sha256"]
        == source["implementation_file_set_sha256"]
    )


def test_exported_tree_never_borrows_an_unrelated_git_commit(tmp_path, monkeypatch):
    known = smoke._source()
    exported = tmp_path / "exported-tree"
    for name in known["files_sha256"]:
        target = exported / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(smoke.REPOSITORY / name, target)
    monkeypatch.setattr(smoke, "REPOSITORY", exported)
    # Existing git may inspect an exported tree, but it must not invent a commit
    # or borrow a surrounding checkout's identity. The listed file hashes remain exact.
    unknown = smoke._source()
    assert unknown["commit"] is unknown["working_tree_dirty"] is None
    assert unknown["git_provenance_status"] == "UNAVAILABLE"
    assert unknown["git_provenance_reason"] in {
        "source_directory_is_not_a_git_worktree_root",
        "git_executable_not_on_path",
    }
    assert unknown["files_sha256"] == known["files_sha256"]
    assert (
        unknown["implementation_file_set_sha256"]
        == known["implementation_file_set_sha256"]
    )


def test_timeout_retains_partial_streams_and_explicit_unknown_exit_code(
    tmp_path, monkeypatch
):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            "synthetic-command",
            60,
            output=b"partial stdout\xff",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(smoke.subprocess, "run", timeout)
    summary = {"commands": [], "checks": [], "artifacts": []}
    with pytest.raises(smoke.SmokeFailure, match="could not complete"):
        smoke._command(
            tmp_path,
            summary,
            "timeout-fixture",
            "replay_recording.py",
            [],
            "reports/not-created.json",
        )
    entry = summary["commands"][0]
    assert entry["exit_code"] is None
    assert entry["timed_out"] is True
    assert (tmp_path / entry["stdout"]).read_text(
        encoding="utf-8"
    ) == "partial stdout\ufffd"
    assert (tmp_path / entry["stderr"]).read_text(encoding="utf-8") == "partial stderr"
    retained = json.loads((tmp_path / "smoke-summary.json").read_text())
    assert len(retained["artifacts"]) == 2


@pytest.mark.parametrize("mutation", ["empty", "only-noninvariants", "missing-one"])
def test_expected_named_epsilon_invariants_cannot_disappear(tmp_path, mutation):
    smoke._fixtures(tmp_path, smoke._source())
    summary = {"commands": [], "checks": [], "artifacts": []}
    report = smoke._command(
        tmp_path,
        summary,
        "epsilon-control-regression",
        "epsilon_sensitivity.py",
        ["inputs/epsilon-samples.json", "--output", "reports/epsilon.json"],
        "reports/epsilon.json",
    )
    controls = report["constructed"]["samples"][0]["constructed_controls"]
    if mutation == "empty":
        controls.clear()
    elif mutation == "only-noninvariants":
        controls[:] = [
            row for row in controls if not row["expected_geometric_invariance"]
        ]
    else:
        controls[:] = [row for row in controls if row["name"] != "translation"]
    with pytest.raises(smoke.SmokeFailure, match="epsilon-constructed-invariances"):
        smoke._verify_epsilon(summary, report)
    assert summary["checks"][-1]["passed"] is False

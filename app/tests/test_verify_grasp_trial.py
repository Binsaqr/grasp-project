"""Synthetic archive regressions; these are not physical sensor validation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import capture_grasp_trial as capture
from scripts.verify_grasp_trial import verify_trial


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "_git", lambda *args: None)
    project = tmp_path / "project"
    for relative in capture.REQUIRED_CURRENT_ARTIFACTS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b'{"timestamp": 99, "stream": "STREAM_ACTIVE"}'
            if path.suffix == ".json"
            else b"image fixture"
        )
    config = project / "config" / "live_quality.yaml"
    config.parent.mkdir()
    config.write_text("required_contacts: 2\n", encoding="utf-8")
    trace = project / "output" / "target_lock_trace.jsonl"
    trace.write_text(
        '{"timestamp": 98, "frame": 1}\n{"timestamp": 99, "frame": 2}\n',
        encoding="utf-8",
    )
    for path in project.rglob("*"):
        if path.is_file():
            os.utime(path, (100, 100))
    return capture.capture_trial(
        project, tmp_path / "trials", label="synthetic", stage="close", captured_at=102
    )


def load_manifest(snapshot):
    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(snapshot, manifest):
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def replace_artifact(snapshot, source, payload):
    """Change a synthetic recording and its digest to test semantic inspection."""
    manifest = load_manifest(snapshot)
    entry = next(
        (item for item in manifest["artifacts"] if item["source"] == source), None
    )
    if entry is None:
        entry = {
            "source": source,
            "captured_as": "artifacts/" + source.removeprefix("output/"),
            "source_mtime": 100,
        }
        manifest["artifacts"].append(entry)
    path = snapshot / entry["captured_as"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    entry.update(
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    save_manifest(snapshot, manifest)


def test_inspection_uses_archived_time_and_config_without_research_claims(snapshot):
    first = verify_trial(snapshot)
    for path in snapshot.rglob("*"):
        if path.is_file():
            os.utime(path, (1, 1))  # Copy/extraction time must not change readiness.
    assert verify_trial(snapshot) == first
    assert first["archive_integrity"] == "VALID"
    assert first["snapshot_inspection"] == "AVAILABLE"
    assert first["physical_validation"] is False
    assert first["raw_sensor_replay_available"] is False
    assert first["motion_evidence"]["state"] == "UNKNOWN"
    assert first["motion_evidence"]["stability_inferred"] is False


@pytest.mark.parametrize("mutation", ["missing", "truncated", "same_size_corruption"])
def test_artifact_corruption_fails_closed(snapshot, mutation):
    path = snapshot / "artifacts/davis/latest_status.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "truncated":
        path.write_bytes(b"{")
    else:
        data = path.read_bytes()
        path.write_bytes(b"X" + data[1:])
    report = verify_trial(snapshot)
    assert report["archive_integrity"] == "INVALID"
    assert report["snapshot_inspection"] == "INCOMPLETE"
    assert report["motion_evidence"]["state"] == "MISSING"
    assert report["errors"]


@pytest.mark.parametrize(
    "payload", ["{", "[]", '{"bad": NaN}', '{"bad": 1e999}', "\xff"]
)
def test_malformed_required_status_is_not_current_evidence(snapshot, payload):
    replace_artifact(snapshot, "output/davis/latest_status.json", payload)
    report = verify_trial(snapshot)
    assert report["archive_integrity"] == "INVALID"
    assert report["snapshot_inspection"] == "INCOMPLETE"


def test_empty_required_status_does_not_pass_on_fresh_mtime(snapshot):
    replace_artifact(snapshot, "output/davis/latest_status.json", {})
    report = verify_trial(snapshot)
    assert report["archive_integrity"] == "VALID"
    assert report["snapshot_inspection"] == "INCOMPLETE"
    assert (
        "output/davis/latest_status.json"
        in report["file_availability_at_capture"]["empty_or_invalid_required_json"]
    )


@pytest.mark.parametrize("payload", ["{", "{}"])
def test_invalid_motion_input_cannot_inherit_included_flag(snapshot, payload):
    replace_artifact(
        snapshot,
        "output/live_quality/latest_status.json",
        {"live_quality_score": 92, "davis_motion_used_for_live_quality": True},
    )
    replace_artifact(snapshot, "output/davis/latest_status.json", payload)
    report = verify_trial(snapshot)
    assert report["recorded_live_quality"]["live_quality_score"] == 92
    assert report["motion_evidence"]["state"] == "INVALID_OR_EMPTY"


@pytest.mark.parametrize(
    "case", ["stale_output", "stale_heartbeat", "inconsistent_flags"]
)
def test_included_flag_requires_available_recorded_context(snapshot, case):
    live = {"davis_motion_used_for_live_quality": True, "observation": {}}
    if case == "stale_heartbeat":
        live["observation"]["davis_report_heartbeat_current"] = False
    elif case == "inconsistent_flags":
        live["observation"]["davis_events_valid_for_live_quality"] = False
    replace_artifact(snapshot, "output/live_quality/latest_status.json", live)
    if case == "stale_output":
        manifest = load_manifest(snapshot)
        next(
            item
            for item in manifest["artifacts"]
            if item["source"] == "output/live_quality/latest_status.json"
        )["source_mtime"] = 0
        save_manifest(snapshot, manifest)
    state = verify_trial(snapshot)["motion_evidence"]["state"]
    assert state == (
        "INCONSISTENT_RECORDED_FLAGS"
        if case == "inconsistent_flags"
        else "STALE_OR_UNAVAILABLE"
    )


@pytest.mark.parametrize(
    "case,expected_state,expected_decision",
    [
        ("stopped", "STALE_OR_UNAVAILABLE", "WAIT_FOR_DAVIS"),
        ("quiet_stale", "STALE_OR_UNAVAILABLE", "MONITOR"),
        ("excluded", "EXCLUDED_AS_RECORDED", "MONITOR"),
    ],
)
def test_actual_monitor_outputs_keep_score_decision_validity_and_reasons(
    snapshot, case, expected_state, expected_decision
):
    from grasp_project.quality.live_quality_monitor import (
        BaselineGraspQuality,
        LiveGraspObservation,
        LiveGraspQualityMonitor,
    )

    base = BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD")
    obs = LiveGraspObservation(
        timestamp=99,
        post_grasp_active=True,
        grasp_stage="POST_GRASP_MONITORING",
        event_state="NOT_RUNNING" if case == "stopped" else "STREAM_ACTIVE_NO_MOTION",
        davis_report_heartbeat_current=case != "stopped",
        davis_event_freshness="CURRENT" if case == "excluded" else "STALE",
        davis_events_valid_for_live_quality=case == "excluded",
        davis_motion_usable_for_grasp=False,
        davis_object_motion_valid=False,
        davis_motion_excluded_reason="global_disturbance"
        if case == "excluded"
        else None,
    )
    recorded = LiveGraspQualityMonitor().update(base, obs).to_dict()
    replace_artifact(snapshot, "output/live_quality/latest_status.json", recorded)
    report = verify_trial(snapshot)
    quality, motion = report["recorded_live_quality"], report["motion_evidence"]
    assert quality["live_quality_score"] == recorded["live_quality_score"] == 84
    assert quality["decision"] == recorded["decision"] == expected_decision
    assert quality["baseline_valid_as_recorded"] is True
    assert motion["events_valid_as_recorded"] is (case == "excluded")
    assert motion["reason_as_recorded"] == recorded["davis_motion_live_quality_reason"]
    assert motion["state"] == expected_state
    assert motion["stability_inferred"] is False


@pytest.mark.parametrize("mtime", [0, 103])
def test_stale_and_future_mtime_cannot_be_overridden_by_readiness_flag(snapshot, mtime):
    manifest = load_manifest(snapshot)
    entry = next(
        item
        for item in manifest["artifacts"]
        if item["source"] == "output/davis/latest_status.json"
    )
    entry.update(source_mtime=mtime, source_age_sec=0, fresh_at_capture=True)
    manifest["capture_readiness"] = "CURRENT_EVIDENCE_AVAILABLE"
    save_manifest(snapshot, manifest)
    report = verify_trial(snapshot)
    assert report["archive_integrity"] == "VALID"
    assert report["snapshot_inspection"] == "INCOMPLETE"
    assert report["motion_evidence"]["state"] == "STALE_OR_UNAVAILABLE"


def test_stale_optional_file_is_diagnostic_without_hiding_required_availability(
    snapshot,
):
    source = "output/multiview/latest_status.json"
    replace_artifact(snapshot, source, {"status": "historical"})
    manifest = load_manifest(snapshot)
    next(item for item in manifest["artifacts"] if item["source"] == source)[
        "source_mtime"
    ] = 0
    save_manifest(snapshot, manifest)
    report = verify_trial(snapshot)
    assert report["snapshot_inspection"] == "AVAILABLE"
    assert source in report["file_availability_at_capture"]["stale_or_future_dated"]


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "/tmp/escape",
        "C:/escape",
        "C:escape",
        "\\\\host\\share\\file",
        "artifacts/../../escape",
    ],
)
def test_unsafe_manifest_paths_are_rejected(snapshot, relative):
    manifest = load_manifest(snapshot)
    manifest["artifacts"][0]["captured_as"] = relative
    save_manifest(snapshot, manifest)
    assert verify_trial(snapshot)["archive_integrity"] == "INVALID"


def test_legacy_v2_inspection_preserves_limitations_and_windows_paths(snapshot):
    manifest = load_manifest(snapshot)
    manifest["schema_version"] = "grasp_trial_snapshot_v2"
    manifest.pop("prediction")
    for entry in manifest["config_hashes"]:
        entry.pop("captured_as")
        entry.pop("size_bytes")
    for entry in manifest["artifacts"]:
        entry["captured_as"] = entry["captured_as"].replace("/", "\\")
    save_manifest(snapshot, manifest)
    report = verify_trial(snapshot)
    assert report["archive_integrity"] == "VALID"
    assert (
        "legacy_v2_prediction_and_config_contents_not_integrity_covered"
        in report["limitations"]
    )


def test_configs_predictions_and_traces_are_all_integrity_checked(snapshot):
    manifest = load_manifest(snapshot)
    for entry in [
        manifest["config_hashes"][0],
        manifest["prediction"],
        manifest["trace_tails"][0],
    ]:
        path = snapshot / entry["captured_as"]
        original = path.read_bytes()
        path.write_bytes(original + b"x")
        assert verify_trial(snapshot)["archive_integrity"] == "INVALID"
        path.write_bytes(original)
    assert verify_trial(snapshot)["archive_integrity"] == "VALID"


def test_prediction_must_match_captured_status_even_if_hash_updated(snapshot):
    manifest = load_manifest(snapshot)
    path = snapshot / "prediction.json"
    path.write_text('{"frame_id": 999}', encoding="utf-8")
    manifest["prediction"].update(
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    save_manifest(snapshot, manifest)
    assert any(
        "prediction differs" in error for error in verify_trial(snapshot)["errors"]
    )


@pytest.mark.parametrize(
    "state,freshness,used,reason",
    [
        ("INCLUDED_AS_RECORDED", "CURRENT", True, "object_motion_valid"),
        ("STALE_OR_UNAVAILABLE", "STALE", False, "stale_events"),
        ("EXCLUDED_AS_RECORDED", "CURRENT", False, "global_camera_motion"),
        ("UNKNOWN", None, None, None),
    ],
)
def test_high_score_preserves_decision_validity_and_motion_reason_together(
    snapshot, state, freshness, used, reason
):
    live = {
        "timestamp": 99,
        "live_quality_score": 92,
        "live_quality_label": "GOOD",
        "decision": "CONTINUE_MONITORING",
        "baseline": {"valid": True},
        "hard_gates": [],
        "davis_motion_used_for_live_quality": used,
        "davis_motion_live_quality_reason": reason,
        "observation": {
            "davis_event_freshness": freshness,
            "davis_events_valid_for_live_quality": used,
        },
    }
    replace_artifact(snapshot, "output/live_quality/latest_status.json", live)
    report = verify_trial(snapshot)
    quality = report["recorded_live_quality"]
    assert (
        quality["live_quality_score"],
        quality["decision"],
        quality["baseline_valid_as_recorded"],
        quality["hard_gates"],
    ) == (92, "CONTINUE_MONITORING", True, [])
    assert report["motion_evidence"]["state"] == state
    assert report["motion_evidence"]["reason_as_recorded"] == reason
    assert report["motion_evidence"]["events_valid_as_recorded"] is used
    assert report["motion_evidence"]["stability_inferred"] is False


def test_manifest_schema_and_shape_errors_return_reports(snapshot):
    for manifest in [
        None,
        [],
        {"schema_version": "future_v99"},
        {"schema_version": "grasp_trial_snapshot_v3", "artifacts": None},
    ]:
        save_manifest(snapshot, manifest)
        assert verify_trial(snapshot)["archive_integrity"] == "INVALID"


@pytest.mark.parametrize("target", ["manifest", "status", "trace"])
@pytest.mark.parametrize("depth", [101, 1100, 5000])
def test_excessive_json_nesting_is_a_reported_error(snapshot, target, depth):
    payload = '{"nested":' + "[" * depth + "0" + "]" * depth + "}"
    if target == "manifest":
        (snapshot / "manifest.json").write_text(payload, encoding="utf-8")
    elif target == "status":
        replace_artifact(snapshot, "output/davis/latest_status.json", payload)
    else:
        manifest = load_manifest(snapshot)
        entry = manifest["trace_tails"][0]
        path = snapshot / entry["captured_as"]
        path.write_text(payload + "\n", encoding="utf-8")
        entry.update(
            captured_size_bytes=path.stat().st_size,
            captured_line_count=1,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        save_manifest(snapshot, manifest)
    assert verify_trial(snapshot)["archive_integrity"] == "INVALID"


def test_cli_exit_codes_and_pure_standard_library_execution(snapshot):
    script = Path(capture.__file__).with_name("verify_grasp_trial.py")

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-S", str(script), str(snapshot), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert json.loads(result.stdout)["physical_validation"] is False
        return result.returncode

    assert run("--require-complete") == 0
    replace_artifact(snapshot, "output/davis/latest_status.json", {})
    assert run() == 0
    assert run("--require-complete") == 2
    (snapshot / "prediction.json").write_bytes(b"broken")
    assert run() == 1

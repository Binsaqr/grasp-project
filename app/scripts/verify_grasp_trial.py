#!/usr/bin/env python3
"""Verify a bounded trial snapshot offline; never open sensors or rescore a grasp."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

try:
    from .capture_grasp_trial import (
        REQUIRED_CURRENT_ARTIFACTS,
        _load_snapshot_json,
        _sha256,
    )
except ImportError:  # Direct script invocation.
    from capture_grasp_trial import (
        REQUIRED_CURRENT_ARTIFACTS,
        _load_snapshot_json,
        _sha256,
    )


MAX_JSON_BYTES = 16 * 1024 * 1024


def _read_json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds 16 MiB inspection limit")
    return _load_snapshot_json(path.read_text(encoding="utf-8"))


def _number(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _inside(root, relative):
    """Accept old Windows separators, but reject escapes on every host platform."""
    if not isinstance(relative, str) or not relative or PureWindowsPath(relative).drive:
        raise ValueError("invalid captured path")
    parts = PurePosixPath(relative.replace("\\", "/"))
    if parts.is_absolute() or ".." in parts.parts or ":" in relative:
        raise ValueError("captured path escapes snapshot")
    path = (root / parts).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("captured path escapes snapshot")
    return path


def _verified_entry(root, entry, size_key, errors):
    """Return a checked path only; never expose data with a failed digest."""
    try:
        if not isinstance(entry, dict):
            raise TypeError("entry must be an object")
        path = _inside(root, entry.get("captured_as"))
        expected_size = entry.get(size_key)
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError("invalid declared size")
        if path.stat().st_size != expected_size:
            raise ValueError("size mismatch")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError("sha256 mismatch")
        return path
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        label = entry.get("captured_as") if isinstance(entry, dict) else repr(entry)
        errors.append(f"{label}: {exc}")
        return None


def _motion_qualification(live, sources, stale, json_objects):
    """Describe recorded evidence, with no new thresholds or stability inference."""
    required = {
        "output/davis/latest_status.json",
        "output/davis/latest_motion_report.json",
    }
    obs = live.get("observation") if isinstance(live.get("observation"), dict) else {}
    reason = live.get("davis_motion_live_quality_reason") or live.get(
        "davis_adjustment_reason"
    )
    freshness = str(obs.get("davis_event_freshness") or "UNKNOWN").upper()
    if not required.issubset(sources):
        state = "MISSING"
    elif any(not json_objects.get(source) for source in required):
        state = "INVALID_OR_EMPTY"
    elif (
        (required | {"output/live_quality/latest_status.json"}).intersection(stale)
        or freshness in {"STALE", "AGING", "UNAVAILABLE"}
        or obs.get("davis_report_heartbeat_current") is False
    ):
        state = "STALE_OR_UNAVAILABLE"
    elif live.get("davis_motion_used_for_live_quality") is True and (
        obs.get("davis_events_valid_for_live_quality") is False
        or obs.get("davis_motion_usable_for_grasp") is False
    ):
        state = "INCONSISTENT_RECORDED_FLAGS"
    elif live.get("davis_motion_used_for_live_quality") is False:
        state = "EXCLUDED_AS_RECORDED"
    elif live.get("davis_motion_used_for_live_quality") is True:
        state = "INCLUDED_AS_RECORDED"
    else:
        state = "UNKNOWN"
    return {
        "state": state,
        "event_freshness_as_recorded": freshness,
        "used_for_live_quality_as_recorded": live.get(
            "davis_motion_used_for_live_quality"
        ),
        "events_valid_as_recorded": obs.get("davis_events_valid_for_live_quality"),
        "motion_usable_as_recorded": obs.get("davis_motion_usable_for_grasp"),
        "reason_as_recorded": reason,
        "stability_inferred": False,
    }


def verify_trial(trial_dir: Path) -> dict:
    """Inspect v2/v3 archives without live files, current time, or model execution.

    Hashes detect accidental corruption, not authenticity. A valid archive may
    contain incomplete or excluded evidence. It is never physical validation.
    """
    errors, limitations = [], []
    report = {
        "schema_version": "grasp_trial_verification_v1",
        "archive_integrity": "INVALID",
        "errors": errors,
        "limitations": limitations,
        "physical_validation": False,
        "raw_sensor_replay_available": False,
    }
    try:
        root = Path(trial_dir).resolve()
        manifest = _read_json(root / "manifest.json")
        if not isinstance(manifest, dict):
            raise TypeError("manifest must be an object")
        version = manifest.get("schema_version")
        if version not in {"grasp_trial_snapshot_v2", "grasp_trial_snapshot_v3"}:
            raise ValueError("unsupported snapshot schema")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        errors.append(f"manifest.json: {exc}")
        return report

    report["snapshot_schema"] = version
    report["git_as_recorded"] = manifest.get("git")
    created, age_limit = (
        manifest.get("created_at_epoch_sec"),
        manifest.get("max_current_age_sec"),
    )
    if not _number(created) or not _number(age_limit) or age_limit <= 0:
        errors.append(
            "manifest: finite capture time and positive max_current_age_sec required"
        )
    sources, stale, json_objects, seen_paths = {}, [], {}, set()
    for group, size_key in (
        ("artifacts", "size_bytes"),
        ("trace_tails", "captured_size_bytes"),
        ("config_hashes", "size_bytes"),
    ):
        entries = manifest.get(group)
        if not isinstance(entries, list):
            errors.append(f"{group}: expected list")
            continue
        for entry in entries:
            if version == "grasp_trial_snapshot_v2" and group == "config_hashes":
                continue  # Legacy configs were hashes only; they cannot be recovered.
            path = _verified_entry(root, entry, size_key, errors)
            if path is None:
                continue
            if path in seen_paths:
                errors.append(
                    f"{group}: duplicate captured path {entry['captured_as']}"
                )
                continue
            seen_paths.add(path)
            if group == "artifacts":
                source = entry.get("source")
                if not isinstance(source, str) or source in sources:
                    errors.append("artifacts: missing or duplicate source")
                    continue
                sources[source] = entry
                mtime = entry.get("source_mtime")
                if (
                    not _number(mtime)
                    or not _number(created)
                    or not _number(age_limit)
                    or age_limit <= 0
                ):
                    errors.append(f"{source}: invalid freshness metadata")
                    stale.append(source)
                elif mtime > created or created - mtime > age_limit:
                    stale.append(source)
                if source.endswith(".json"):
                    try:
                        payload = _read_json(path)
                        if not isinstance(payload, dict):
                            raise TypeError("status must be a JSON object")
                        json_objects[source] = payload
                    except (OSError, ValueError, TypeError) as exc:
                        errors.append(f"{source}: {exc}")
            elif group == "trace_tails":
                try:
                    if path.stat().st_size > MAX_JSON_BYTES:
                        raise ValueError("trace exceeds 16 MiB inspection limit")
                    lines = path.read_text(encoding="utf-8").splitlines()
                    if len(lines) != entry.get("captured_line_count"):
                        raise ValueError("trace line count mismatch")
                    for line in lines:
                        if not isinstance(
                            _load_snapshot_json(line),
                            dict,
                        ):
                            raise TypeError("trace rows must be JSON objects")
                except (OSError, ValueError, TypeError, RecursionError) as exc:
                    errors.append(f"{entry['captured_as']}: {exc}")

    if version == "grasp_trial_snapshot_v3":
        prediction_entry = manifest.get("prediction")
        if (
            isinstance(prediction_entry, dict)
            and prediction_entry.get("captured_as") != "prediction.json"
        ):
            errors.append("prediction: expected prediction.json path")
        prediction_path = _verified_entry(root, prediction_entry, "size_bytes", errors)
        if prediction_path:
            try:
                prediction = _read_json(prediction_path)
                valid_source = "output/latest_realtime_status.json" in json_objects
                if prediction_entry.get("source_json_valid") is not valid_source:
                    raise ValueError(
                        "prediction source_json_valid disagrees with captured status"
                    )
                if prediction != json_objects.get("output/latest_realtime_status.json"):
                    raise ValueError("prediction differs from captured realtime status")
            except (OSError, ValueError, RecursionError) as exc:
                errors.append(f"prediction.json: {exc}")
    else:
        limitations.append(
            "legacy_v2_prediction_and_config_contents_not_integrity_covered"
        )

    missing = sorted(set(REQUIRED_CURRENT_ARTIFACTS) - sources.keys())
    empty = sorted(
        source
        for source in REQUIRED_CURRENT_ARTIFACTS
        if source.endswith(".json") and not json_objects.get(source)
    )
    report["file_availability_at_capture"] = {
        "missing_required": missing,
        "stale_or_future_dated": sorted(stale),
        "empty_or_invalid_required_json": empty,
        "scope": "filesystem age at capture; not native sensor age or clock alignment",
    }
    live = json_objects.get("output/live_quality/latest_status.json", {})
    report["recorded_live_quality"] = {
        key: live.get(key)
        for key in (
            "timestamp",
            "analytic_quality_score",
            "analytic_quality_status",
            "live_quality_score",
            "live_quality_label",
            "decision",
            "hard_gates",
            "primary_risk",
            "next_action",
            "davis_adjustment_status",
            "davis_adjustment_reason",
        )
    }
    baseline = live.get("baseline") if isinstance(live.get("baseline"), dict) else {}
    report["recorded_live_quality"]["baseline_valid_as_recorded"] = baseline.get(
        "valid"
    )
    report["motion_evidence"] = _motion_qualification(
        live, sources, stale, json_objects
    )
    report["snapshot_inspection"] = (
        "AVAILABLE"
        if not errors
        and not missing
        and not set(REQUIRED_CURRENT_ARTIFACTS).intersection(stale)
        and not empty
        else "INCOMPLETE"
    )
    limitations.extend(
        [
            "bounded_status_snapshot_not_raw_RGBD_or_native_event_recording",
            "sequential_file_copies_not_an_atomic_synchronized_observation",
            "recorded_scores_and_decisions_not_recomputed_or_physically_validated",
            "ground_truth_requires_independent_labels_not_predictions",
            "no_physical_onset_latency_or_slip_lead_time_endpoint",
            "image_bytes_hash_checked_but_image_content_not_decoded",
            "trace_tails_may_omit_observations_no_dropped_packet_counts_available",
        ]
    )
    report["archive_integrity"] = "VALID" if not errors else "INVALID"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="also fail for incomplete snapshot evidence",
    )
    args = parser.parse_args()
    report = verify_trial(args.trial_dir)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["archive_integrity"] != "VALID":
        return 1
    return (
        2
        if args.require_complete and report["snapshot_inspection"] != "AVAILABLE"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

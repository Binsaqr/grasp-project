#!/usr/bin/env python3
"""Capture one bounded grasp-system evidence snapshot without opening sensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

ARTIFACTS = (
    "output/latest_realtime_status.json",
    "output/latest_realtime_diagnostics.json",
    "output/latest_overlay_live.jpg",
    "output/latest_rgb_live.jpg",
    "output/latest_depth_live.jpg",
    "output/locked_target.json",
    "output/candidate_grasp_report.json",
    "output/current_quality_bundle.json",
    "output/fresh_lock_quality_status.json",
    "output/latest_pregrasp_safety_report.json",
    "output/realtime_epsilon_result.json",
    "output/realtime_wrench_matrix_status.json",
    "output/pointcloud_status.json",
    "output/sync_status.json",
    "output/event_postgrasp_report.json",
    "output/davis/latest_status.json",
    "output/davis/latest_motion_estimate.json",
    "output/davis/latest_motion_report.json",
    "output/davis/latest_timeline.json",
    "output/davis/latest_aps.jpg",
    "output/davis/latest_event_heatmap.jpg",
    "output/live_quality/latest_status.json",
    "output/multiview/latest_status.json",
)
TRACE_ARTIFACTS = (
    "output/target_lock_trace.jsonl",
    "output/patch_coordinate_trace.jsonl",
    "output/davis/motion_timeline.jsonl",
    "output/live_quality/timeline.jsonl",
)
REQUIRED_CURRENT_ARTIFACTS = (
    "output/latest_realtime_status.json",
    "output/latest_overlay_live.jpg",
    "output/davis/latest_status.json",
    "output/davis/latest_motion_report.json",
)
VALID_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _reject_json_constant(value)
    return number


def _load_snapshot_json(text: str):
    """Reject non-finite or excessively nested data before inspection/serialization."""
    try:
        data = json.loads(text, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser limit") from exc
    pending = [(data, 0)]
    while pending:
        value, depth = pending.pop()
        if isinstance(value, (dict, list)):
            if depth >= 100:
                raise ValueError("JSON nesting exceeds 100 levels")
            children = value.values() if isinstance(value, dict) else value
            pending.extend((child, depth + 1) for child in children)
    return data


def _copy_entry(
    project_root: Path,
    trial_dir: Path,
    relative: str,
    captured_at: float,
    max_current_age_sec: float,
) -> dict | None:
    source = project_root / relative
    if not source.is_file():
        return None
    destination = trial_dir / "artifacts" / relative.removeprefix("output/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    shutil.copy2(source, destination)
    age = max(0.0, captured_at - stat.st_mtime)
    return {
        "source": relative,
        "captured_as": destination.relative_to(trial_dir).as_posix(),
        "source_mtime": stat.st_mtime,
        "source_age_sec": age,
        "fresh_at_capture": age <= max_current_age_sec,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _read_tail_lines(path: Path, max_lines: int, max_bytes: int) -> tuple[list[bytes], int]:
    """Read complete trailing lines without scanning an unbounded JSONL trace."""
    size = path.stat().st_size
    if size == 0 or max_lines <= 0 or max_bytes <= 0:
        return [], size
    block_size = 64 * 1024
    position = size
    chunks = []
    bytes_read = 0
    newline_count = 0
    with path.open("rb") as handle:
        while position > 0 and bytes_read < max_bytes and newline_count <= max_lines:
            read_size = min(block_size, position, max_bytes - bytes_read)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            bytes_read += len(chunk)
            newline_count += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    if position > 0:
        first_newline = data.find(b"\n")
        data = b"" if first_newline < 0 else data[first_newline + 1 :]
    return [line for line in data.splitlines() if line.strip()][-max_lines:], size


def _copy_trace_tail(
    project_root: Path,
    trial_dir: Path,
    relative: str,
    captured_at: float,
    max_current_age_sec: float,
    max_lines: int,
    max_bytes: int,
) -> dict | None:
    source = project_root / relative
    if not source.is_file():
        return None
    stat = source.stat()
    lines, source_size = _read_tail_lines(source, max_lines, max_bytes)
    destination = trial_dir / "trace_tails" / relative.removeprefix("output/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"\n".join(lines) + (b"\n" if lines else b""))
    age = max(0.0, captured_at - stat.st_mtime)
    return {
        "source": relative,
        "captured_as": destination.relative_to(trial_dir).as_posix(),
        "source_mtime": stat.st_mtime,
        "source_age_sec": age,
        "fresh_at_capture": age <= max_current_age_sec,
        "source_size_bytes_at_capture": source_size,
        "captured_size_bytes": destination.stat().st_size,
        "captured_line_count": len(lines),
        "tail_line_limit": max_lines,
        "tail_byte_read_limit": max_bytes,
        "sha256": _sha256(destination),
    }


def capture_trial(
    project_root: Path,
    output_root: Path,
    *,
    label: str,
    stage: str,
    notes: str = "",
    captured_at: float | None = None,
    max_current_age_sec: float = 5.0,
    trace_lines: int = 120,
    trace_max_bytes: int = 8 * 1024 * 1024,
) -> Path:
    """Copy current evidence and return the new trial directory."""
    if not VALID_LABEL.fullmatch(label):
        raise ValueError("label must be 1-80 path-safe letters, digits, '.', '_' or '-'")
    if not math.isfinite(max_current_age_sec) or max_current_age_sec <= 0:
        raise ValueError("max_current_age_sec must be finite and positive")
    if trace_lines < 0 or trace_max_bytes < 0:
        raise ValueError("trace limits cannot be negative")
    captured_at = time.time() if captured_at is None else float(captured_at)
    if not math.isfinite(captured_at) or captured_at <= 0:
        raise ValueError("captured_at must be finite and positive")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(captured_at))
    trial_dir = output_root / f"{stamp}_{time.time_ns() % 1_000_000:06d}_{label}"
    trial_dir.mkdir(parents=True, exist_ok=False)

    artifacts = []
    missing = []
    for relative in ARTIFACTS:
        entry = _copy_entry(
            project_root,
            trial_dir,
            relative,
            captured_at,
            max_current_age_sec,
        )
        if entry is None:
            missing.append(relative)
        else:
            artifacts.append(entry)

    trace_tails = []
    missing_trace_tails = []
    for relative in TRACE_ARTIFACTS:
        entry = _copy_trace_tail(
            project_root,
            trial_dir,
            relative,
            captured_at,
            max_current_age_sec,
            trace_lines,
            trace_max_bytes,
        )
        if entry is None:
            missing_trace_tails.append(relative)
        else:
            trace_tails.append(entry)

    configs = []
    for path in sorted((project_root / "config").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(project_root)
        destination = trial_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        configs.append({
            "path": relative.as_posix(),
            "captured_as": destination.relative_to(trial_dir).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        })

    status_source = "output/latest_realtime_status.json"
    artifacts_by_source = {entry["source"]: entry for entry in artifacts}
    status_entry = artifacts_by_source.get(status_source)
    prediction = None
    try:
        if status_entry is not None:
            status_path = trial_dir / status_entry["captured_as"]
            archived_status = _load_snapshot_json(status_path.read_text(encoding="utf-8"))
            if isinstance(archived_status, dict):
                prediction = archived_status
    except (OSError, ValueError, RecursionError):
        pass
    prediction_path = trial_dir / "prediction.json"
    prediction_path.write_text(
        json.dumps(prediction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (trial_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "schema_version": "grasp_trial_truth_v1",
                "labelled": False,
                "ground_truth": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    git_status = _git(project_root, "status", "--short")
    missing_required = [
        relative for relative in REQUIRED_CURRENT_ARTIFACTS if relative not in artifacts_by_source
    ]
    stale_required = [
        relative
        for relative in REQUIRED_CURRENT_ARTIFACTS
        if relative in artifacts_by_source
        and artifacts_by_source[relative]["fresh_at_capture"] is not True
    ]
    capture_ready = not missing_required and not stale_required
    manifest = {
        "schema_version": "grasp_trial_snapshot_v3",
        "created_at_epoch_sec": captured_at,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(captured_at)),
        "label": label,
        "stage": stage,
        "notes": notes,
        "capture_mode": "bounded_existing_files_no_sensor_open",
        "snapshot_scope": "derived_artifacts_only; no_raw_streams_or_timestamp_alignment",
        "synchronization_claim": "not_synchronized; source_age_sec is recorded per artifact",
        "max_current_age_sec": max_current_age_sec,
        "capture_readiness": "CURRENT_EVIDENCE_AVAILABLE" if capture_ready else "INCOMPLETE_CURRENT_EVIDENCE",
        "capture_readiness_scope": "file_presence_and_mtime_only; content_and_sensor_freshness_not_validated",
        "required_current_artifacts": list(REQUIRED_CURRENT_ARTIFACTS),
        "missing_required_current_artifacts": missing_required,
        "stale_required_current_artifacts": stale_required,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "trace_tails": trace_tails,
        "missing_trace_tails": missing_trace_tails,
        "config_hashes": configs,
        "prediction": {
            "source": status_source,
            "source_json_valid": prediction is not None,
            "captured_as": "prediction.json",
            "size_bytes": prediction_path.stat().st_size,
            "sha256": _sha256(prediction_path),
        },
        "git": {
            "commit": _git(project_root, "rev-parse", "HEAD"),
            "working_tree_dirty": None if git_status is None else bool(git_status),
            "status_short": None if git_status is None else git_status.splitlines(),
        },
        "ground_truth_status": "not_labelled",
    }
    (trial_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trial_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Path-safe trial label, for example upright_bright")
    parser.add_argument(
        "--stage",
        choices=("distant", "approach", "close", "post_grasp", "combined"),
        default="distant",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-current-age-sec", type=float, default=5.0)
    parser.add_argument("--trace-lines", type=int, default=120)
    parser.add_argument("--trace-max-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root or project_root / "output" / "trial_snapshots"
    trial_dir = capture_trial(
        project_root,
        output_root,
        label=args.label,
        stage=args.stage,
        notes=args.notes,
        max_current_age_sec=args.max_current_age_sec,
        trace_lines=args.trace_lines,
        trace_max_bytes=args.trace_max_bytes,
    )
    print(f"Saved bounded trial snapshot: {trial_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

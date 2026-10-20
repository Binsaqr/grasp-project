"""Validate fresh runtime intrinsics evidence without importing sensor stacks."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STATUS_RELATIVE_PATHS = (
    Path("output/latest_realtime_status.json"),
    Path("output/pointcloud_status.json"),
    Path("output/current_quality_bundle.json"),
    Path("output/realtime_grasp_hulls.json"),
)
PROVENANCE_FIELDS = (
    "writer_session_id",
    "frame_id",
    "camera_frame_id",
    "pointcloud_frame_id",
    "quality_compute_id",
    "matrix_compute_id",
)


@dataclass(frozen=True)
class IntrinsicsEvidence:
    path: Path
    source: str
    age_sec: float


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _timestamp_epoch(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            epoch = float(value)
            return epoch if epoch > 1_000_000_000 and math.isfinite(epoch) else None
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _complete_intrinsics(value: Any) -> bool:
    return isinstance(value, dict) and all(_finite_number(value.get(key)) for key in ("fx", "fy", "cx", "cy")) and float(value["fx"]) > 0.0 and float(value["fy"]) > 0.0


def _intrinsics_sources(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    sources = []
    for key in ("realsense_intrinsics_available", "pointcloud_intrinsics_available", "intrinsics_available", "_quality_intrinsics_available"):
        if data.get(key) is True:
            sources.append(prefix + key)
    for key in ("intrinsics", "camera_intrinsics", "_quality_intrinsics_snapshot"):
        if _complete_intrinsics(data.get(key)):
            sources.append(prefix + key)
    for key in ("pointcloud_status", "quality_matrix", "realsense_summary"):
        nested = data.get(key)
        if isinstance(nested, dict):
            sources.extend(_intrinsics_sources(nested, prefix + key + "."))
    return sources


def fresh_intrinsics_matches(root: Path, *, now: float | None = None, max_age_sec: float = 600.0) -> list[IntrinsicsEvidence]:
    """Return only timestamped, runtime-provenance intrinsics evidence."""
    current_time = time.time() if now is None else float(now)
    matches = []
    for relative_path in STATUS_RELATIVE_PATHS:
        path = root / relative_path
        data = _read_json(path)
        if not data or data.get("synthetic") is True or str(data.get("evidence_source") or "").lower() in {"test", "synthetic"}:
            continue
        epoch = _timestamp_epoch(data.get("timestamp") or data.get("pointcloud_timestamp") or data.get("quality_compute_timestamp"))
        if epoch is None:
            continue
        age_sec = current_time - epoch
        if age_sec < 0.0 or age_sec > max_age_sec or not any(data.get(field) is not None for field in PROVENANCE_FIELDS):
            continue
        sources = _intrinsics_sources(data)
        if sources:
            matches.append(IntrinsicsEvidence(path=path, source=sources[0], age_sec=age_sec))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-age-sec", type=float, default=600.0)
    args = parser.parse_args()
    matches = fresh_intrinsics_matches(args.root, max_age_sec=args.max_age_sec)
    if not matches:
        print("no fresh timestamped runtime intrinsics source found")
        return 1
    print("; ".join(f"{match.path}:{match.source}:age={match.age_sec:.1f}s" for match in matches[:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

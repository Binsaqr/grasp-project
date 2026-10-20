"""Proof runner for strict Benchmark Lab feature masking.

This module reuses the existing BenchmarkRunner and masked evaluators. It reads
current status files only; it does not open sensors, change production
decisions, or run the 31-combination benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .benchmark_runner import BenchmarkRunner
from .evaluators import run_poison_value_tests
from .feature_mask import build_benchmark_lab_status
from .feature_registry import FEATURE_BY_ID


ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def proof_configurations(duration_sec: float = 5.0) -> list[dict[str, Any]]:
    """Return the seven required feature-mask proof configurations."""
    groups = {
        "rgb_only": (
            "RGB Only",
            ["rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_center", "rgb_bottle_angle"],
        ),
        "depth_only": (
            "Depth Only",
            ["depth_bottle_distance", "depth_valid_ratio", "depth_alignment_quality", "depth_object_geometry"],
        ),
        "davis_aps_only": (
            "DAVIS APS Only",
            ["davis_aps_frame", "davis_aps_target_motion", "davis_aps_angle_change", "davis_aps_lighting_change"],
        ),
        "davis_events_only": (
            "DAVIS Events Only",
            [
                "event_rate",
                "event_heatmap",
                "event_motion_activity",
                "event_stability",
                "event_rotation",
                "event_vibration",
                "event_incipient_slip",
                "event_gross_slip",
                "event_impact",
            ],
        ),
        "pointcloud_only": (
            "Point Cloud Only",
            ["pointcloud_scene_geometry", "pointcloud_bottle_geometry", "pointcloud_bottle_extents"],
        ),
        "rgb_depth": (
            "RGB + Depth",
            [
                "rgb_bottle_detection",
                "rgb_bottle_segmentation",
                "rgb_bottle_center",
                "rgb_bottle_angle",
                "depth_bottle_distance",
                "depth_valid_ratio",
                "depth_alignment_quality",
                "depth_object_geometry",
            ],
        ),
        "full_multimodal": ("Full Multimodal", list(FEATURE_BY_ID)),
    }
    return [
        {
            "configuration_id": config_id,
            "display_name": label,
            "feature_ids": feature_ids,
            "enabled_features": feature_ids,
            "benchmark_object_mode": "bottle_only",
            "duration_sec": float(duration_sec),
        }
        for config_id, (label, feature_ids) in groups.items()
    ]


def load_current_sources(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Load current dashboard/backend source dictionaries from output files."""
    output = root / "output"
    davis_status = _read_json(output / "davis" / "latest_status.json")
    davis_report = _read_json(output / "davis" / "latest_motion_report.json")
    davis = {**davis_report, **davis_status}
    quality = _read_json(output / "realtime_wrench_matrix_status.json")
    return {
        "realsense": _read_json(output / "latest_realtime_status.json"),
        "davis": davis,
        "pointcloud": _read_json(output / "pointcloud_status.json"),
        "sync": _read_json(output / "sync_status.json") or _read_json(output / "sensor_sync_report.json"),
        "quality": quality,
        "live_quality": _read_json(output / "live_quality" / "latest_status.json"),
        "embodiment": _read_json(output / "embodiment_status.json"),
    }


def _availability(sources: dict[str, dict[str, Any]]) -> dict[str, bool]:
    rs = sources["realsense"]
    pc = sources["pointcloud"]
    davis = sources["davis"]
    pointcloud_fresh = str(pc.get("pointcloud_freshness") or pc.get("freshness") or "").upper()
    aps_state = str(davis.get("aps_state") or davis.get("aps_mode") or "").upper()
    event_state = str(davis.get("event_state") or davis.get("event_stream_status") or "").upper()
    return {
        "realsense_active": bool(rs.get("target_status") or rs.get("realsense_rgb_active")),
        "rgb_active": bool(rs.get("target_status") or rs.get("realsense_rgb_active")),
        "depth_active": bool(rs.get("target_status") or rs.get("realsense_depth_active") or rs.get("distance_mm")),
        "pointcloud_active": pointcloud_fresh in {"CURRENT", "AGING", "UPDATED"},
        "davis_aps_active": aps_state in {"APS_COLOR_VERIFIED", "APS_RAW_BAYER", "APS_GRAYSCALE", "ACTIVE"},
        "davis_events_active": event_state in {"STREAM_ACTIVE_EVENTS", "STREAM_ACTIVE_NO_MOTION", "ACTIVE"},
    }


def build_status_from_current_files(configurations: list[dict[str, Any]], root: Path = ROOT) -> dict[str, Any]:
    """Build a Benchmark Lab status snapshot from current output files."""
    sources = load_current_sources(root)
    status = build_benchmark_lab_status(
        realsense=sources["realsense"],
        davis=sources["davis"],
        pointcloud=sources["pointcloud"],
        sync=sources["sync"],
        quality=sources["quality"],
        live_quality=sources["live_quality"],
        embodiment=sources["embodiment"],
        configurations=configurations,
    )
    status["source_status"] = {**sources, "sensor_availability": _availability(sources)}
    return status


def proof_precheck(root: Path = ROOT) -> dict[str, Any]:
    """Return a non-invasive precheck for the proof benchmark."""
    sources = load_current_sources(root)
    availability = _availability(sources)
    required = {
        "realsense_status_available": {
            "ok": bool(sources["realsense"]),
            "reason": "ok" if sources["realsense"] else "latest_realtime_status_missing",
        }
    }
    optional = {
        "pointcloud_active": {
            "ok": availability["pointcloud_active"],
            "reason": "ok" if availability["pointcloud_active"] else "pointcloud_unavailable_or_stale",
        },
        "davis_aps_active": {
            "ok": availability["davis_aps_active"],
            "reason": "ok" if availability["davis_aps_active"] else "davis_aps_unavailable",
        },
        "davis_events_active": {
            "ok": availability["davis_events_active"],
            "reason": "ok" if availability["davis_events_active"] else "davis_events_unavailable",
        },
    }
    failed = [name for name, check in required.items() if not check["ok"]]
    return {"ok": not failed, "reason": ",".join(failed) if failed else "ok", "required": required, "optional": optional}


def _wait_done(runner: BenchmarkRunner, timeout_sec: float) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        state = runner.snapshot().get("state")
        if state in {"COMPLETE", "FAILED", "CANCELLED"}:
            return state
        time.sleep(0.05)
    return runner.snapshot().get("state") or "TIMEOUT"


def _required_proof_fields_present(row: dict[str, Any]) -> bool:
    required = {
        "enabled_features",
        "disabled_features",
        "used_features",
        "attempted_features",
        "blocked_features",
        "missing_dependencies",
        "not_applicable_features",
        "unavailable_features",
        "stale_features",
        "unsupported_features",
        "production_fields_used",
        "feature_leakage_detected",
        "evaluator_trace",
    }
    return required.issubset(row)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Benchmark Feature Mask Proof Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Proof run ID: `{summary['proof_run_id']}`",
        f"- Proof passed: **{summary['proof_passed']}**",
        f"- Benchmark trustworthy now: **{summary['benchmark_lab_trustworthy']}**",
        f"- Run directory: `{summary['run_dir']}`",
        "",
        "## Root Cause",
        "",
        summary["root_cause"],
        "",
        "## Previous Benchmark Risk",
        "",
        summary["previous_benchmark_risk"],
        "",
        "## Poison Tests",
        "",
        f"- Passed: `{summary['poison_tests']['passed']}`",
        f"- Case count: `{summary['poison_tests']['case_count']}`",
        f"- Feature leakage detected: `{summary['poison_tests']['feature_leakage_detected']}`",
        "",
        "## Proof Configurations",
        "",
        "| Configuration | Result | Used Features | Missing Dependencies | Not Applicable | Leakage |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["configurations"]:
        lines.append(
            "| {name} | {validity} | {used} | {missing} | {na} | {leak} |".format(
                name=row["display_name"],
                validity=row["result_validity"],
                used=", ".join(row["used_features"]) or "none",
                missing=", ".join(row["missing_dependencies"]) or "none",
                na=", ".join(row["not_applicable_features"]) or "none",
                leak=row["feature_leakage_detected"],
            )
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Production safety behavior, RealSense tracking, DAVIS scoring, and epsilon/wrench math were not modified.",
        "- The proof run samples existing status outputs through `MaskedFeatureView`; disabled features are reported as blocked rather than read.",
        "- The 31-combination benchmark should stay paused until this proof remains passing on the live setup.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_feature_mask_proof(
    *,
    root: Path = ROOT,
    duration_sec: float = 5.0,
    sample_rate_hz: float = 5.0,
) -> dict[str, Any]:
    """Run the seven-configuration strict-mask proof benchmark."""
    output_root = root / "output" / "benchmarks"
    configs = proof_configurations(duration_sec=duration_sec)
    runner = BenchmarkRunner(
        output_root,
        sample_callback=lambda active_configs: build_status_from_current_files(active_configs, root),
        precheck_callback=lambda: proof_precheck(root),
    )
    start = runner.start(configurations=configs, duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    state = _wait_done(runner, timeout_sec=max(30.0, duration_sec * len(configs) + 20.0))
    results = runner.results()
    poison_path = output_root / "feature_leakage_poison_tests.json"
    poison = run_poison_value_tests(poison_path)
    rows = results.get("results") or []
    config_rows = []
    for row in rows:
        config_rows.append({
            "configuration_id": row.get("configuration_id"),
            "display_name": row.get("display_name"),
            "result_validity": row.get("result_validity"),
            "enabled_features": row.get("enabled_features") or [],
            "disabled_features": row.get("disabled_features") or [],
            "used_features": row.get("used_features") or [],
            "attempted_features": row.get("attempted_features") or [],
            "blocked_features": row.get("blocked_features") or {},
            "missing_dependencies": row.get("missing_dependencies") or {},
            "not_applicable_features": row.get("not_applicable_features") or {},
            "unavailable_features": row.get("unavailable_features") or {},
            "stale_features": row.get("stale_features") or {},
            "unsupported_features": row.get("unsupported_features") or {},
            "production_fields_used": row.get("production_fields_used") or [],
            "feature_leakage_detected": bool(row.get("feature_leakage_detected")),
            "evaluator_trace_count": len(row.get("evaluator_trace") or []),
            "proof_fields_present": _required_proof_fields_present(row),
        })
    no_leak = not any(row["feature_leakage_detected"] for row in config_rows)
    fields_present = bool(config_rows) and all(row["proof_fields_present"] for row in config_rows)
    no_production_fields = not any(row["production_fields_used"] for row in config_rows)
    proof_passed = bool(state == "COMPLETE" and poison.get("passed") and no_leak and fields_present and no_production_fields)
    run_dir = (results.get("outputs") or {}).get("run_dir") or (start.get("outputs") or {}).get("run_dir")
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "proof_run_id": (results.get("manifest") or {}).get("benchmark_run_id"),
        "state": state,
        "run_dir": run_dir,
        "proof_passed": proof_passed,
        "benchmark_lab_trustworthy": proof_passed,
        "root_cause": (
            "Benchmark Lab previously had a direct source-dictionary evaluator path that made "
            "feature-card outputs easy to confuse with production-context fields. The active "
            "path now evaluates cards through FeaturePacket and MaskedFeatureView only."
        ),
        "previous_benchmark_risk": (
            "Older expanded runs may be report-useful for timing/status, but should not be used "
            "as strict feature-ablation proof unless their result rows include the proof fields "
            "and feature_leakage_detected=false."
        ),
        "poison_tests": poison,
        "configurations": config_rows,
        "runner_outputs": results.get("outputs") or {},
    }
    summary_path = output_root / "feature_mask_proof_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_report(root / "docs" / "benchmark_feature_mask_proof_report.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strict Benchmark Lab feature-mask proof benchmark.")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per proof configuration.")
    parser.add_argument("--sample-rate", type=float, default=5.0, help="Samples per second.")
    args = parser.parse_args()
    summary = run_feature_mask_proof(duration_sec=args.duration, sample_rate_hz=args.sample_rate)
    print(json.dumps({
        "proof_passed": summary["proof_passed"],
        "state": summary["state"],
        "proof_run_id": summary["proof_run_id"],
        "summary": "output/benchmarks/feature_mask_proof_summary.json",
        "report": "docs/benchmark_feature_mask_proof_report.md",
    }, indent=2))
    return 0 if summary["proof_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

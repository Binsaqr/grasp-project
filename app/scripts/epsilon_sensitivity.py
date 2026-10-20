#!/usr/bin/env python3
"""Run camera-free geometric epsilon controls from an explicit sample manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_project.benchmarking.epsilon_sensitivity import analyze_epsilon_sensitivity


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.manifest.resolve() == args.output.resolve():
            raise ValueError("output must not overwrite the input manifest")
        raw = args.manifest.read_bytes()
        payload = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "grasp_epsilon_samples/v1"
        ):
            raise ValueError("manifest schema must be grasp_epsilon_samples/v1")
        unknown = set(payload) - {
            "schema",
            "constructed_samples",
            "recorded_samples",
            "parameters",
        }
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be an object")
        report = analyze_epsilon_sensitivity(
            payload.get("constructed_samples", []),
            payload.get("recorded_samples", []),
            **parameters,
        )
        report["input_manifest"] = {
            "path": str(args.manifest),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"epsilon sensitivity: {exc}\n")
    invalid = sum(
        report[kind]["distribution"]["invalid_or_unavailable_count"]
        for kind in ("constructed", "recorded")
    )
    failed_invariances = any(
        control["expected_geometric_invariance"]
        and control["invariance_passed"] is not True
        for row in report["constructed"]["samples"]
        for control in row["constructed_controls"]
    )
    total = sum(
        report[kind]["distribution"]["sample_count"]
        for kind in ("constructed", "recorded")
    )
    return 1 if not total or invalid or failed_invariances else 0


if __name__ == "__main__":
    raise SystemExit(main())

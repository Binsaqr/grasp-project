#!/usr/bin/env python3
"""Evaluate frozen offline predictions; no sensors, model fitting or label inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_project.benchmarking.offline_evaluation import (
    EvaluationContractError,
    evaluate_recorded_trials,
    read_document,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="JSON report path; otherwise stdout")
    args = parser.parse_args(argv)
    if args.output and args.output.resolve() in {
        path.resolve() for path in (args.annotations, args.predictions, args.splits)
    }:
        parser.error("output must not overwrite an input document")
    try:
        report = evaluate_recorded_trials(
            read_document(args.annotations), read_document(args.predictions), read_document(args.splits)
        )
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except (EvaluationContractError, OSError) as exc:
        print(f"evaluation contract error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

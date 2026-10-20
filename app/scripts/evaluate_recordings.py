#!/usr/bin/env python3
"""Evaluate verified recording parts through declared offline ablation arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        action="append",
        required=True,
        help="raw recording directory; repeat for trials or process-local parts",
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument(
        "--declarations",
        type=Path,
        required=True,
        help="explicit task/component and frozen configurations for selected arms",
    )
    arms = parser.add_mutually_exclusive_group()
    arms.add_argument(
        "--ablation",
        action="append",
        help="selected replay arm; repeat (default: reference)",
    )
    arms.add_argument(
        "--all-ablations",
        action="store_true",
        help="run every supported replay arm with explicit declarations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new JSON report outside input archives; otherwise stdout",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from grasp_project.benchmarking.offline_evaluation import read_document
    from grasp_project.benchmarking.recording_experiment import (
        run_recording_experiment,
        safe_output_path,
        write_experiment_report,
    )

    inputs = (args.annotations, args.splits, args.declarations)
    try:
        if args.output:
            safe_output_path(
                args.output, input_paths=inputs, recording_dirs=args.recording
            )
        report = run_recording_experiment(
            args.recording,
            annotations=read_document(args.annotations),
            splits=read_document(args.splits),
            declarations=read_document(args.declarations),
            ablations=None if args.all_ablations else args.ablation or ["reference"],
        )
        if args.output:
            write_experiment_report(
                report, args.output, input_paths=inputs, recording_dirs=args.recording
            )
        else:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"recording experiment error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

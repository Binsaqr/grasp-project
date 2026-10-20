#!/usr/bin/env python3
"""Estimate offline clock diagnostics from explicit paired correspondences."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_project.benchmarking.offline_evaluation import read_document
from grasp_project.recording.clock_mapping import (
    RECORDING_STREAMS,
    ClockMappingError,
    estimate_clock_mapping,
    estimate_recording_clock_mapping,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correspondences", nargs="?", type=Path)
    parser.add_argument("--recording", type=Path, help="verified raw archive; receipt associations only")
    parser.add_argument("--stream", choices=RECORDING_STREAMS)
    parser.add_argument("--output", type=Path, help="report path; otherwise stdout")
    args = parser.parse_args(argv)
    if (args.correspondences is None) == (args.recording is None):
        parser.error("provide either a correspondence file or --recording, exclusively")
    if (args.recording is None) != (args.stream is None):
        parser.error("--recording requires --stream; --stream is only valid with --recording")
    if args.correspondences and args.output and (
        args.output.resolve() == args.correspondences.resolve()
        or (args.output.exists() and args.correspondences.exists() and args.output.samefile(args.correspondences))
    ):
        parser.error("output must not overwrite input correspondences")
    try:
        if args.recording:
            if args.output and (
                args.output.resolve().is_relative_to(args.recording.resolve())
                or (args.output.exists() and any(
                    args.output.samefile(path) for path in args.recording.rglob("*") if path.is_file()
                ))
            ):
                parser.error("output must not write inside or overwrite any file in the input archive")
            report = estimate_recording_clock_mapping(args.recording, args.stream)
        else:
            report = estimate_clock_mapping(read_document(args.correspondences))
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except (ClockMappingError, OSError, ValueError) as exc:
        print(f"clock mapping contract error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

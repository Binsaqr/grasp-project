#!/usr/bin/env python3
"""Replay a verified recording through existing motion and quality components."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_project.recording.replay import ABLATIONS, replay_recording


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--ablation", choices=ABLATIONS, default="reference")
    parser.add_argument("--output", type=Path, help="new report path outside the input archive; otherwise stdout")
    parser.add_argument("--rgbd-evaluation", action="store_true", help="also replay canonical RGB-D worker calls in the supported ROS/Qt runtime")
    args = parser.parse_args(argv)
    try:
        if args.output and (args.output.exists() or args.output.is_symlink()
                            or args.output.resolve().is_relative_to(args.recording_dir.resolve())):
            raise ValueError("output must be a new file outside the input archive")
        report = replay_recording(args.recording_dir, ablation=args.ablation)
        if args.rgbd_evaluation:
            from grasp_project.recording.rgbd_evaluation_replay import (
                replay_rgbd_evaluation,
            )

            report["rgbd_evaluation"] = replay_rgbd_evaluation(args.recording_dir)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
        return 1 if args.rgbd_evaluation and not report["rgbd_evaluation"]["verified"] else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Recording replay failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

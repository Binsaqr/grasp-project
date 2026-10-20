"""Small command wrapper for the optional RealSense + DAVIS sync demo layer."""

from __future__ import annotations

import argparse
import json

from grasp_project.sync.sensor_sync import test_live


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run optional sensor sync demo from current status files.")
    parser.add_argument("--mode", choices=["2d", "3d", "compare"], default="2d")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args(argv)
    result = test_live(args.duration)
    result["mode"] = args.mode
    if args.mode == "3d" and result.get("sync_quality") == "waiting_for_pointcloud":
        result["mode_warning"] = "3D comparison needs pointcloud_status.json from the RealSense point cloud layer."
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Main entry point for the grasp-quality pipeline.

This keeps the run path small while the legacy scripts remain in place for
behavior parity during the staged module refactor.
"""

import argparse
import json
import os
import shutil
import runpy
import sys
from pathlib import Path

from camera.orientation import correct_saved_rgbd


ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = ROOT / "sample_data"
OUTPUT_DIR = ROOT / "output"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

DEMO_FILES = {
    "captured_frame.png": OUTPUT_DIR / "captured_frame.png",
    "depth_frame.png": OUTPUT_DIR / "depth_frame.png",
    "depth_frame.npy": OUTPUT_DIR / "depth_frame.npy",
    "camera_intrinsics.json": OUTPUT_DIR / "camera_intrinsics.json",
}


def run_script(relative_path):
    runpy.run_path(str(ROOT / relative_path), run_name="__main__")


def clear_realtime_outputs():
    for filename in ("realtime_status.json", "latest_realtime_status.json", "stage_history.json", "locked_target.json"):
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def clear_multiview_outputs():
    for filename in ("multiview_report.json", "multiview_summary.png"):
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def prepare_demo_inputs(apply_rotation=True):
    missing = [
        str(SAMPLE_DATA_DIR / filename)
        for filename in DEMO_FILES
        if not (SAMPLE_DATA_DIR / filename).exists()
    ]
    if missing:
        print("Demo sample files are missing.")
        print("Expected these files:")
        for path in missing:
            print(f"  - {path}")
        print("\nAdd saved RGB-D sample files to sample_data/ and run again.")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, destination in DEMO_FILES.items():
        shutil.copy2(SAMPLE_DATA_DIR / filename, destination)

    with open(OUTPUT_DIR / "camera_intrinsics.json", "r") as f:
        json.load(f)

    print("Demo mode: loaded saved RGB-D sample files from sample_data/.")
    correct_saved_rgbd(OUTPUT_DIR, apply_rotation=apply_rotation, force=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the RGB-D grasp quality pipeline.")
    parser.add_argument("--demo", action="store_true", help="Use saved RGB-D sample files from sample_data/ instead of the live camera.")
    parser.add_argument("--bottle", action="store_true", help="Run the bottle-focused grasp planner instead of the general pipeline.")
    parser.add_argument("--no-capture", action="store_true", help="Skip camera capture and reuse output/captured_frame.png.")
    parser.add_argument("--no-gui", action="store_true", help="Skip opening the PyQt dashboard.")
    parser.add_argument("--no-rotate", action="store_true", help="Compatibility flag; camera frames are already upright and are never rotated.")
    parser.add_argument("--realtime", action="store_true", help="Run staged bottle grasp safety state-machine mode.")
    parser.add_argument("--lock-target", action="store_true", help="Enable target locking in live realtime bottle mode. This is the live default.")
    parser.add_argument("--no-lock-target", action="store_true", help="Disable target locking in live realtime bottle mode.")
    parser.add_argument("--multiview", action="store_true", help="Run optional saved-frame multi-view bottle pre-grasp evaluation.")
    parser.add_argument("--realtime-iterations", type=int, default=1, help="Number of staged realtime frames to evaluate.")
    parser.add_argument("--realtime-interval", type=float, default=0.0, help="Seconds to wait between realtime frames.")
    args = parser.parse_args(argv)
    apply_rotation = False
    os.environ["APPLY_CAMERA_ROTATION"] = "0"

    if args.bottle and args.multiview:
        from viewpoint_evaluator import evaluate_multiview

        clear_realtime_outputs()
        evaluate_multiview()
        if not args.no_gui:
            run_script("visualization/gui_dashboard.py")
        return

    if args.demo:
        prepare_demo_inputs(apply_rotation=apply_rotation)
    elif not args.no_capture and not args.realtime:
        run_script("scripts/capture_rgb_depth.py")

    if args.bottle:
        if args.realtime:
            clear_multiview_outputs()
            if args.demo or args.no_gui or args.no_capture:
                from real_time_loop import run_realtime_loop

                run_realtime_loop(
                    iterations=max(1, args.realtime_iterations),
                    interval_s=max(0.0, args.realtime_interval),
                    capture_each_frame=not (args.demo or args.no_capture),
                )
                if not args.no_gui:
                    run_script("visualization/gui_dashboard.py")
            else:
                qt_platforms = Path("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
                if qt_platforms.exists():
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_platforms)
                from real_time_live import run_live_realtime_app

                run_live_realtime_app(lock_target=not args.no_lock_target)
            return
        clear_realtime_outputs()
        clear_multiview_outputs()
        run_script("scripts/bottle_grasp_planner.py")
        if not args.no_gui:
            run_script("visualization/gui_dashboard.py")
        return

    run_script("scripts/hybrid_opencv_sam.py")
    run_script("scripts/rgbd_grasp_eval.py")

    if not args.no_gui:
        run_script("visualization/gui_dashboard.py")


if __name__ == "__main__":
    main(sys.argv[1:])

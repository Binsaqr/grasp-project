import argparse
import os
from pathlib import Path

QT_PLATFORM_PATH = Path("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
if QT_PLATFORM_PATH.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLATFORM_PATH)

from davis_grasp_planner import DAVIS_APS_DISPLAY_MODE, DAVIS_APS_DISPLAY_MODES, run_gui


def main(argv=None):
    parser = argparse.ArgumentParser(description="DAVIS346 live APS/event visual verifier.")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"), help="Manual ROI x y w h for event/detection debugging.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means run until closed.")
    parser.add_argument("--sensitivity", choices=["conservative", "balanced", "sensitive"], default="balanced")
    parser.add_argument("--display-mode", choices=sorted(DAVIS_APS_DISPLAY_MODES), default=DAVIS_APS_DISPLAY_MODE, help="APS display interpretation.")
    args = parser.parse_args(argv)
    return run_gui(duration=args.duration, roi_xywh=args.roi, sensitivity=args.sensitivity, display_mode=args.display_mode)


if __name__ == "__main__":
    raise SystemExit(main())

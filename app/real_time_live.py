"""Compatibility wrapper for the migrated RealSense realtime runner.

Purpose: preserve old imports and commands such as
`python3 real_time_live.py --backend-only`.
Implementation: `src/grasp_project/realsense/realtime_runner.py`.
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    runpy.run_module("grasp_project.realsense.realtime_runner", run_name="__main__")
else:
    from grasp_project.realsense.realtime_runner import *  # noqa: F401,F403

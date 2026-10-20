"""Compatibility wrapper for the migrated DAVIS motion GUI.

Purpose: preserve old imports and commands such as
`python3 davis_motion_gui.py --live`.
Implementation: `src/grasp_project/davis/motion_gui.py`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasp_project.davis.motion_gui import *  # noqa: E402,F401,F403
from grasp_project.davis.motion_gui import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

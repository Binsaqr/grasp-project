"""Compatibility wrapper for the migrated Flask dashboard.

Purpose: preserve old commands such as `python3 web_dashboard.py --host ...`.
Implementation: `src/grasp_project/dashboard/web_dashboard.py`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasp_project.dashboard.web_dashboard import *  # noqa: E402,F401,F403
from grasp_project.dashboard.web_dashboard import app, main  # noqa: E402


if __name__ == "__main__":
    main()

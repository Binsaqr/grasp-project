"""Compatibility wrapper for migrated epsilon/wrench-space quality helpers.

Purpose: preserve old imports such as `from epsilon_quality import ...`.
Implementation: `src/grasp_project/realsense/epsilon_wrench.py`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasp_project.realsense.epsilon_wrench import *  # noqa: E402,F401,F403

"""Compatibility wrapper for the migrated DAVIS event monitor.

Purpose: preserve old imports and commands such as
`python3 event_postgrasp_monitor.py --enable ...`.
Implementation: `src/grasp_project/davis/event_stream.py`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasp_project.davis import event_stream as _impl  # noqa: E402
from grasp_project.davis.event_stream import *  # noqa: E402,F401,F403

_clip_roi = _impl._clip_roi
_backend_import_status = _impl._backend_import_status
_detect_davis_usb_devices = _impl._detect_davis_usb_devices
_json_safe = _impl._json_safe
_write_json = _impl._write_json


def run_cli(argv=None):
    """Run the migrated DAVIS event monitor CLI."""
    return _impl.run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(run_cli())

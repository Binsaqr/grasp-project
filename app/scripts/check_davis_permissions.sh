#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "DAVIS346 permission diagnostic"
echo "=============================="
echo "Selection identity: VID/PID and configured serial; USB bus/device node is diagnostic only."
echo

# keep the diagnostics below reachable when the check fails (set -e would abort here)
rc=0
python3 -m grasp_project.hardware.device_discovery --davis-check || rc="$?"

echo
echo "Backend import check:"
backend_rc=0
require_backend=0
if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    require_backend=1
fi
GRASP_REQUIRE_DAVIS_BACKEND="$require_backend" python3 - <<'PY' || backend_rc="$?"
import importlib
import os

required_ok = True
try:
    from pyaer.davis import DAVIS  # noqa: F401
    print("PASS: required release adapter pyaer.davis.DAVIS")
except Exception as exc:
    required_ok = False
    label = "FAIL" if os.environ.get("GRASP_REQUIRE_DAVIS_BACKEND") == "1" else "HOST-ONLY OPTIONAL"
    print(f"{label}: pyaer.davis.DAVIS unavailable: {exc}")

for name in ("dv_processing", "dv"):
    try:
        importlib.import_module(name)
        print(f"OPTIONAL_PRESENT: Python import {name}; no live adapter is implemented")
    except Exception as exc:
        print(f"OPTIONAL_MISSING: Python import {name}: {exc}")

if not required_ok and os.environ.get("GRASP_REQUIRE_DAVIS_BACKEND") == "1":
    raise SystemExit(3)
PY
if [ "$backend_rc" -ne 0 ]; then
    rc="$backend_rc"
fi

echo
echo "Suggested streaming test:"
echo "  python3 event_postgrasp_monitor.py --enable --duration 3.0 --live-print"

if [ "$require_backend" -eq 0 ]; then
    echo "Host note: only USB visibility/permissions are required here; pyaer is required inside the release container."
fi

exit "$rc"

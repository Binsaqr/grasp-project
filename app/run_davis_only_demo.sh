#!/usr/bin/env bash
# LEGACY/DEBUG ONLY. Final demo command is ./run_final_grasp_demo.sh --with-davis
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" output

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

echo "Starting deprecated DAVIS-only path as the main web dashboard DAVIS tab."
echo "This launcher no longer opens the experimental DAVIS-only desktop planner."
echo "Open: http://$HOST:$PORT/?tab=davis"
echo "DAVIS has no metric depth or grip width; production grasp planning remains RealSense-based."

python3 -m grasp_project.hardware.device_discovery --davis-check || {
    echo "WARNING: DAVIS auto-discovery did not select a usable camera."
    echo "Run: bash scripts/setup_davis_udev.sh, then unplug/replug the camera."
}

python3 - <<'PY'
try:
    import pyaer  # noqa: F401
    print("PASS: pyaer import OK")
except Exception as exc:
    print(f"WARNING: pyaer import failed: {exc}")
PY

exec python3 web_dashboard.py --host "$HOST" --port "$PORT" --start-davis

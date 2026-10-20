#!/usr/bin/env bash
# Purpose: open the DAVIS Motion tab through the main web dashboard.
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BASE_URL="http://$HOST:$PORT"

export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export DAVIS_DEBUG_HEATMAP="${DAVIS_DEBUG_HEATMAP:-0}"
mkdir -p "$MPLCONFIGDIR" output/logs

api_request() {
    local method="$1"
    local path="$2"
    python3 - "$method" "$BASE_URL$path" <<'PY'
import sys
import urllib.request

method, url = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, method=method)
try:
    with urllib.request.urlopen(req, timeout=3.0) as response:
        sys.stdout.write(response.read().decode("utf-8", "replace"))
except Exception as exc:
    sys.stderr.write(str(exc) + "\n")
    sys.exit(1)
PY
}

dashboard_live() {
    api_request GET /api/status >/dev/null 2>&1
}

echo "Starting DAVIS motion/event demo in the main web dashboard..."
echo "Mode: canonical dashboard-owned DAVIS backend; no standalone desktop GUI."
echo "Open: $BASE_URL/?tab=davis"
echo "Outputs: output/davis/latest_status.json, output/davis/latest_event_heatmap.jpg, output/davis/latest_motion_report.json"
echo "Debug heatmap is disabled by default. Use DAVIS_DEBUG_HEATMAP=1 only when needed."

python3 -m grasp_project.hardware.device_discovery --davis-check || {
    echo "WARNING: DAVIS auto-discovery did not select a usable camera."
    echo "Run: ./scripts/setup_davis_udev.sh, then unplug/replug the camera."
}

if dashboard_live; then
    echo "Dashboard: already running; starting DAVIS backend through web API."
    api_request POST /api/davis/start >/dev/null || true
    echo "Open: $BASE_URL/?tab=davis"
    exit 0
fi

echo "Dashboard: starting with DAVIS backend enabled."
exec python3 web_dashboard.py --host "$HOST" --port "$PORT" --start-davis

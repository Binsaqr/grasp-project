#!/usr/bin/env bash
# Purpose: launch the final combined web demo from one operator command.
# Main demo: RealSense RGB-D grasp planning plus optional DAVIS post-grasp motion.
# Canonical UI: src/grasp_project/dashboard/web_dashboard.py.
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
START_DAVIS=0
POINTCLOUD_FPS="${POINTCLOUD_GENERATE_FPS:-0.5}"
DASHBOARD_PID=""
DASHBOARD_STARTED_BY_LAUNCHER=0
REALSENSE_STARTED_BY_LAUNCHER=0
DAVIS_STARTED_BY_LAUNCHER=0
CLEANUP_STARTED=0
REALSENSE_STATUS="UNKNOWN"

usage() {
    cat <<EOF
Usage: ./run_final_grasp_demo.sh [--with-davis] [--pointcloud-fps 0.5|1|2|5]

One-command final demo startup:
  cd <path-to>/GRASP_PROJECT
  ./run_final_grasp_demo.sh --with-davis

Then open:
  http://$HOST:$PORT

Starts or reuses:
  - main web dashboard
  - RealSense backend
  - RealSense camera node if needed
  - dashboard-owned DAVIS backend when --with-davis is passed

Environment overrides:
  HOST=127.0.0.1
  PORT=8080
  ROS_DOMAIN_ID=67
  REALSENSE_FPS=30
  REALSENSE_COLOR_PROFILE=480x270x30
  REALSENSE_DEPTH_PROFILE=480x270x30
  POINTCLOUD_GENERATE_FPS=0.5
  POINTCLOUD_PREVIEW_FPS=0.5
  DEMO_LOG_MAX_BYTES=8388608
  USE_YOLO_BOTTLE_TRIGGER=0  # optional fallback only; provide model assets before enabling
  SINGLE_BOTTLE_DEMO_MODE=1
  PRELOCK_FAST_SCALED_RGBD=1
  YOLO_MODEL_PATH=models/yolo_bottle.weights
  YOLO_CONF_THRESHOLD=0.25
EOF
}

while [ "$#" -gt 0 ]; do
    arg="$1"
    case "$arg" in
        --with-davis) START_DAVIS=1 ;;
        --pointcloud-fps)
            if [ "$#" -lt 2 ]; then
                echo "--pointcloud-fps requires a value."
                exit 1
            fi
            POINTCLOUD_FPS="$2"
            shift
            ;;
        --pointcloud-fps=*)
            POINTCLOUD_FPS="${arg#*=}"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
    shift
done

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export REALSENSE_FPS="${REALSENSE_FPS:-30}"
DEFAULT_REALSENSE_PROFILE="480x270x${REALSENSE_FPS}"
export REALSENSE_COLOR_PROFILE="${REALSENSE_COLOR_PROFILE:-$DEFAULT_REALSENSE_PROFILE}"
export REALSENSE_DEPTH_PROFILE="${REALSENSE_DEPTH_PROFILE:-$DEFAULT_REALSENSE_PROFILE}"
export POINTCLOUD_GENERATE_FPS="${POINTCLOUD_GENERATE_FPS:-$POINTCLOUD_FPS}"
export POINTCLOUD_PREVIEW_FPS="${POINTCLOUD_PREVIEW_FPS:-$POINTCLOUD_FPS}"
export DAVIS_DEBUG_HEATMAP="${DAVIS_DEBUG_HEATMAP:-0}"
export USE_YOLO_BOTTLE_TRIGGER="${USE_YOLO_BOTTLE_TRIGGER:-0}"
export SINGLE_BOTTLE_DEMO_MODE="${SINGLE_BOTTLE_DEMO_MODE:-1}"
export PRELOCK_FAST_SCALED_RGBD="${PRELOCK_FAST_SCALED_RGBD:-1}"

LOG_DIR="$PROJECT_DIR/output/logs"
DASHBOARD_LOG="$LOG_DIR/dashboard.log"
REALSENSE_LOG="$LOG_DIR/realsense_backend.log"
DAVIS_LOG="$LOG_DIR/davis_backend.log"
BASE_URL="http://$HOST:$PORT"
DEMO_LOG_MAX_BYTES="${DEMO_LOG_MAX_BYTES:-8388608}"

mkdir -p "$MPLCONFIGDIR" output "$LOG_DIR"

case "$DEMO_LOG_MAX_BYTES" in
    ''|*[!0-9]*)
        echo "DEMO_LOG_MAX_BYTES must be a non-negative integer."
        exit 1
        ;;
esac

prepare_bounded_log() {
    local log_path="$1"
    local prior_path="${log_path}.previous"
    local temporary_path="${prior_path}.tmp"
    touch "$log_path"
    if [ -s "$log_path" ]; then
        tail -c "$DEMO_LOG_MAX_BYTES" "$log_path" >"$temporary_path"
        mv -f "$temporary_path" "$prior_path"
    fi
    : >"$log_path"
    printf '\n=== Final demo session %s (prior tail capped at %s bytes) ===\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$DEMO_LOG_MAX_BYTES" >>"$log_path"
}

prepare_bounded_log "$DASHBOARD_LOG"
prepare_bounded_log "$REALSENSE_LOG"
prepare_bounded_log "$DAVIS_LOG"

api_request() {
    local method="$1"
    local path="$2"
    python3 - "$method" "$BASE_URL$path" <<'PY'
import sys
import urllib.request

method, url = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, method=method)
timeout = 120.0 if method == "POST" and "/api/start/realsense" in url else 10.0
try:
    with urllib.request.urlopen(req, timeout=timeout) as response:
        sys.stdout.write(response.read().decode("utf-8", "replace"))
except Exception as exc:
    sys.stderr.write(str(exc) + "\n")
    sys.exit(1)
PY
}

dashboard_live() {
    api_request GET /api/status >/dev/null 2>&1
}

api_running() {
    local name="$1"
    python3 - "$BASE_URL/api/status" "$name" <<'PY'
import json
import sys
import urllib.request

url, name = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(url, timeout=3.0) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    print("true" if bool((payload.get("running") or {}).get(name)) else "false")
except Exception:
    print("false")
    sys.exit(1)
PY
}

davis_running() {
    python3 - "$BASE_URL/api/davis/status" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=3.0) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    state = str(payload.get("state") or payload.get("event_state") or "").upper()
    owner = str(payload.get("davis_owner_status") or "unowned").lower()
    open_count = int(payload.get("davis_device_open_count") or 0)
    invalid = state in {"BACKEND_UNAVAILABLE", "ERROR", "DISCONNECTED", "NOT_CONNECTED", "DEVICE_NOT_FOUND"}
    healthy = bool(payload.get("running") and not invalid and owner == "owned" and open_count > 0)
    print("true" if healthy else "false")
except Exception:
    print("false")
    sys.exit(1)
PY
}

realsense_backend_process_count() {
    pgrep -fc 'real_time_live.py --backend-only|run_bottle_realtime.sh --headless-web' 2>/dev/null || true
}

realsense_backend_process_list() {
    pgrep -af 'real_time_live.py --backend-only|run_bottle_realtime.sh --headless-web' 2>/dev/null || true
}

wait_for_dashboard() {
    local deadline=$((SECONDS + 25))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if dashboard_live; then
            return 0
        fi
        if [ "$DASHBOARD_STARTED_BY_LAUNCHER" -eq 1 ] && [ -n "$DASHBOARD_PID" ] && ! kill -0 "$DASHBOARD_PID" >/dev/null 2>&1; then
            return 1
        fi
        sleep 0.5
    done
    return 1
}

wait_for_realsense() {
    python3 - <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path("output/latest_realtime_status.json")
deadline = time.time() + 50.0
last = {}
while time.time() < deadline:
    try:
        if path.exists():
            last = json.loads(path.read_text())
            age = time.time() - path.stat().st_mtime
            writer_pid = last.get("writer_pid")
            cmdline_path = Path(f"/proc/{writer_pid}/cmdline") if writer_pid else None
            writer_is_current = False
            if cmdline_path is not None and cmdline_path.exists():
                try:
                    writer_is_current = "real_time_live.py" in cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
                except OSError:
                    writer_is_current = False
            if age <= 5.0 and last.get("writer_session_id") and writer_is_current:
                rgb = bool(last.get("realsense_rgb_active"))
                depth = bool(last.get("realsense_depth_active"))
                target = last.get("target_status") or last.get("state") or "starting"
                fps = last.get("overlay_fps") or last.get("processed_overlay_fps") or last.get("fps")
                if rgb and depth:
                    print(f"RealSense: OK, target={target}, rgb={rgb}, depth={depth}, fps={fps if fps is not None else 'pending'}")
                    sys.exit(0)
                last["error"] = "RGB and aligned-depth streams are not active yet"
    except Exception as exc:
        last = {"error": str(exc)}
    time.sleep(0.5)
reason = last.get("camera_error") or last.get("error") or last.get("target_status") or "no fresh backend status"
print(f"RealSense: not ready - {reason}")
sys.exit(1)
PY
}

summarize_davis() {
    python3 - "$BASE_URL/api/davis/status" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=3.0) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
except Exception as exc:
    print(f"DAVIS: unavailable - dashboard status unavailable: {exc}")
    sys.exit(1)

state = payload.get("state") or payload.get("event_state") or "unknown"
event_state = payload.get("event_state") or "unknown"
aps_state = payload.get("aps_state") or "unknown"
rate = payload.get("event_rate")
owner = payload.get("davis_owner_status") or "unowned"
open_count = int(payload.get("davis_device_open_count") or 0)
error = payload.get("error") or ""
if state == "BACKEND_UNAVAILABLE" or owner != "owned" or open_count <= 0:
    print(f"DAVIS: unavailable - backend ownership/open failed (state={state}, owner={owner}, opens={open_count}).")
    sys.exit(2)
if state == "PERMISSION_DENIED":
    print("DAVIS: permission denied - run ./scripts/setup_davis_udev.sh, then unplug/replug DAVIS.")
    sys.exit(2)
if state in {"NOT_CONNECTED", "DEVICE_NOT_FOUND", "DISCONNECTED"}:
    print("DAVIS: unavailable - DAVIS not detected. Continuing without DAVIS.")
    sys.exit(2)
if state == "ERROR":
    lower = str(error).lower()
    if "permission" in lower or "udev" in lower:
        print("DAVIS: permission denied - run ./scripts/setup_davis_udev.sh, then unplug/replug DAVIS.")
    elif "not found" in lower or "not detected" in lower or "not connected" in lower or owner == "unowned":
        print("DAVIS: unavailable - DAVIS not detected. Continuing without DAVIS.")
    elif error:
        print(f"DAVIS: unavailable - {error}")
    else:
        print(f"DAVIS: unavailable - state={state}, APS={aps_state}, events={event_state}")
    sys.exit(2)
if state in {"STREAM_ACTIVE_EVENTS", "STREAM_ACTIVE_NO_MOTION", "EVENTS_ONLY", "APS_ONLY", "OPENING"} or payload.get("running"):
    print(f"DAVIS: {state}, APS={aps_state}, events={event_state}, rate={rate if rate is not None else 'pending'}, owner={owner}")
    sys.exit(0)
if error:
    print(f"DAVIS: unavailable - {error}")
else:
    print(f"DAVIS: unavailable - state={state}, APS={aps_state}, events={event_state}")
sys.exit(2)
PY
}

cleanup() {
    if [ "$CLEANUP_STARTED" -eq 1 ]; then
        return
    fi
    CLEANUP_STARTED=1
    trap - INT TERM EXIT
    echo
    echo "Stopping demo..."
    if [ "$START_DAVIS" -eq 1 ] && [ "$DAVIS_STARTED_BY_LAUNCHER" -eq 1 ]; then
        api_request POST /api/davis/stop >/dev/null 2>&1 || true
    fi
    if [ "$REALSENSE_STARTED_BY_LAUNCHER" -eq 1 ]; then
        api_request POST /api/stop/realsense >/dev/null 2>&1 || true
    fi
    if [ "$DASHBOARD_STARTED_BY_LAUNCHER" -eq 1 ] && [ -n "$DASHBOARD_PID" ]; then
        kill "$DASHBOARD_PID" >/dev/null 2>&1 || true
        wait "$DASHBOARD_PID" >/dev/null 2>&1 || true
    fi
    echo "Demo stopped cleanly."
}

trap cleanup INT TERM EXIT

echo "Starting Multimodal Grasp Safety Demo..."
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "Dashboard URL: $BASE_URL"
echo "RealSense profile: color=$REALSENSE_COLOR_PROFILE depth=$REALSENSE_DEPTH_PROFILE"
echo "Point cloud FPS: generate=$POINTCLOUD_GENERATE_FPS preview=$POINTCLOUD_PREVIEW_FPS"
echo
echo "Hardware discovery: delegated to each canonical camera backend to avoid duplicate USB probes."
echo

if dashboard_live; then
    echo "Dashboard: already running at $BASE_URL"
else
    echo "Dashboard: starting..."
    python3 web_dashboard.py --host "$HOST" --port "$PORT" >>"$DASHBOARD_LOG" 2>&1 &
    DASHBOARD_PID="$!"
    DASHBOARD_STARTED_BY_LAUNCHER=1
    if ! wait_for_dashboard; then
        echo "Dashboard: failed to start. See $DASHBOARD_LOG"
        exit 1
    fi
    echo "Dashboard: OK $BASE_URL"
fi

if [ "$START_DAVIS" -eq 1 ]; then
    if [ "$(davis_running 2>/dev/null || echo false)" = "true" ]; then
        echo "DAVIS: already running; reusing dashboard-owned backend."
    else
        echo "DAVIS: starting dashboard-owned backend..."
        DAVIS_STARTED_BY_LAUNCHER=1
        if ! api_request POST /api/davis/start >/dev/null; then
            echo "DAVIS: start request failed. See $DAVIS_LOG and $DASHBOARD_LOG"
        fi
    fi
    sleep 1
    if ! summarize_davis; then
        echo "DAVIS: first open was unhealthy; performing one bounded dashboard-owned stop/start retry..."
        api_request POST /api/davis/stop >/dev/null 2>&1 || true
        sleep 0.5
        DAVIS_STARTED_BY_LAUNCHER=1
        if ! api_request POST /api/davis/start >/dev/null; then
            echo "DAVIS: retry start request failed. See $DAVIS_LOG and $DASHBOARD_LOG"
        fi
        sleep 1
        summarize_davis || true
    fi
else
    echo "DAVIS: optional - pass --with-davis to start event-camera monitoring."
fi

if [ "$(api_running realsense 2>/dev/null || echo false)" = "true" ] && [ "$(realsense_backend_process_count)" -gt 0 ]; then
    echo "RealSense: already running; reusing existing backend."
elif [ "$(realsense_backend_process_count)" -gt 0 ]; then
    REALSENSE_STATUS="STALE_PROCESS"
    echo "RealSense: backend process already exists, but dashboard status is stale or unhealthy."
    echo "RealSense: not starting a duplicate. Check $REALSENSE_LOG and run ./scripts/check_demo_system.sh."
    echo "Existing RealSense process:"
    realsense_backend_process_list | sed 's/^/  /'
else
    echo "RealSense: starting backend through dashboard API..."
    REALSENSE_STARTED_BY_LAUNCHER=1
    if ! api_request POST /api/start/realsense >/dev/null; then
        echo "RealSense: start request failed. See $REALSENSE_LOG"
    fi
fi

if [ "$REALSENSE_STATUS" = "STALE_PROCESS" ]; then
    echo "RealSense: not ready - existing backend process has stale dashboard status."
elif wait_for_realsense; then
    REALSENSE_STATUS="OK"
else
    REALSENSE_STATUS="NOT_READY"
    echo "RealSense: not detected or not streaming - demo cannot run until the camera is fixed."
    echo "Check USB, ROS_DOMAIN_ID=67, and $REALSENSE_LOG"
fi

echo
echo "Startup summary"
echo "Dashboard: OK $BASE_URL"
if [ "$REALSENSE_STATUS" = "OK" ]; then
    echo "RealSense: OK, RGB and aligned depth verified active"
elif [ "$REALSENSE_STATUS" = "STALE_PROCESS" ]; then
    echo "RealSense: existing backend process has stale/unhealthy status; no duplicate started."
else
    echo "RealSense: not ready - demo cannot run. Check USB connection and logs."
fi
if [ "$START_DAVIS" -eq 1 ]; then
    summarize_davis || true
else
    echo "DAVIS: not requested"
fi
echo "Point Cloud: current-frame support runs at $POINTCLOUD_GENERATE_FPS FPS when valid RGB-D is available"
echo "Benchmark Lab: available"
echo
echo "Logs:"
echo "$DASHBOARD_LOG"
echo "$REALSENSE_LOG"
echo "$DAVIS_LOG"
echo
echo "Open: $BASE_URL"
echo "Press Ctrl+C here to stop processes started by this launcher."

while true; do
    if [ "$DASHBOARD_STARTED_BY_LAUNCHER" -eq 1 ] && [ -n "$DASHBOARD_PID" ] && ! kill -0 "$DASHBOARD_PID" >/dev/null 2>&1; then
        echo "Dashboard process exited. See $DASHBOARD_LOG"
        exit 1
    fi
    sleep 1
done

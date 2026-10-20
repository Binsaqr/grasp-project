#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$PROJECT_DIR/.cache/ros_log}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$ROS_LOG_DIR" "$MPLCONFIGDIR"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() { echo "[PASS] $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
warn() { echo "[WARN] $*"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { echo "[FAIL] $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "Final demo system check"
echo "Project: $PROJECT_DIR"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

if [ "$ROS_DOMAIN_ID" = "67" ]; then
    pass "ROS_DOMAIN_ID is 67"
else
    fail "ROS_DOMAIN_ID must be 67 for this demo"
fi

mkdir -p output
if [ -w output ]; then
    pass "output/ is writable"
else
    fail "output/ is not writable"
fi

python3 - <<'PY' >/tmp/grasp_demo_import_check.txt 2>&1
import cv2
import flask
import numpy
import rclpy
from sensor_msgs.msg import CameraInfo, Image
PY
if [ "$?" -eq 0 ]; then
    pass "Python imports available: cv2, flask, numpy, rclpy, sensor_msgs"
else
    fail "Python import check failed: $(tr '\n' ' ' </tmp/grasp_demo_import_check.txt)"
    echo "       Native dependencies are incomplete. On Fedora use: podman exec grasp-dashboard ./scripts/check_demo_system.sh"
fi

if command -v ros2 >/dev/null 2>&1; then
    pass "ros2 command available"
else
    fail "ros2 command not found"
    echo "       ROS 2 is provided by the supported container: podman exec grasp-dashboard ./scripts/check_demo_system.sh"
fi

if python3 -m grasp_project.hardware.device_discovery --realsense-check >/tmp/grasp_demo_rs_discovery.txt 2>&1; then
    pass "RealSense detected dynamically: $(head -n 1 /tmp/grasp_demo_rs_discovery.txt)"
else
    warn "RealSense dynamic discovery did not select a camera: $(tail -n 2 /tmp/grasp_demo_rs_discovery.txt | tr '\n' ' ')"
fi

if [ -x scripts/realsense_transport_health.py ]; then
    if timeout 8s scripts/realsense_transport_health.py --timeout 5 --min-fps 5 >/tmp/grasp_demo_rs_health.txt 2>&1; then
        pass "RealSense color/depth topics are readable"
    else
        warn "RealSense health check did not pass: $(tail -n 2 /tmp/grasp_demo_rs_health.txt | tr '\n' ' ')"
    fi
else
    warn "scripts/realsense_transport_health.py is missing"
fi

if timeout 5s ros2 topic echo --once /camera/camera/color/camera_info >/tmp/grasp_demo_camera_info.txt 2>&1; then
    pass "RealSense camera_info topic is available"
else
    python3 -m grasp_project.hardware.status_evidence --root "$PROJECT_DIR" --max-age-sec 600 >/tmp/grasp_demo_intrinsics_status.txt 2>&1
    if [ "$?" -eq 0 ]; then
        pass "RealSense camera_info ROS topic missing, but intrinsics available through live status: $(tr '\n' ' ' </tmp/grasp_demo_intrinsics_status.txt)"
    else
        warn "RealSense camera_info topic is not available and no fresh live status intrinsics were found"
    fi
fi

if [ -x scripts/check_davis_permissions.sh ]; then
    if timeout 8s scripts/check_davis_permissions.sh >/tmp/grasp_demo_davis_check.txt 2>&1; then
        pass "DAVIS detected dynamically and permissions are OK"
    else
        warn "DAVIS dynamic discovery/permission check reported an issue: $(tail -n 3 /tmp/grasp_demo_davis_check.txt | tr '\n' ' ')"
    fi
else
    warn "DAVIS permission script missing"
fi

if curl -fsS --max-time 2 http://127.0.0.1:8080/api/status >/tmp/grasp_demo_dashboard_status.json 2>/tmp/grasp_demo_dashboard_curl.txt; then
    pass "dashboard is responding on http://127.0.0.1:8080"
else
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn | grep -q ':8080 '; then
            port_state="occupied"
        else
            port_state="free"
        fi
    else
        port_state="unknown"
    fi
    if [ "$port_state" = "free" ]; then
        pass "port 8080 is free"
    elif [ "$port_state" = "unknown" ]; then
        warn "dashboard /api/status did not respond and port state could not be checked"
    else
        warn "port 8080 is occupied but dashboard /api/status did not respond"
    fi
fi

writer_count="$(pgrep -fc 'real_time_live.py --backend-only' || true)"
if [ "$writer_count" -le 1 ]; then
    pass "RealSense backend writer count is $writer_count"
else
    fail "duplicate RealSense backend writers detected: $writer_count"
fi

wrapper_count="$(pgrep -fc 'run_bottle_realtime.sh --headless-web' || true)"
if [ "$wrapper_count" -le 1 ]; then
    pass "headless wrapper count is $wrapper_count"
else
    fail "duplicate headless wrappers detected: $wrapper_count"
fi

available_kb="$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')"
if [ "${available_kb:-0}" -gt 1048576 ]; then
    pass "disk space OK: $((available_kb / 1024)) MB available"
else
    warn "low disk space: $((available_kb / 1024)) MB available"
fi

echo "Summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0

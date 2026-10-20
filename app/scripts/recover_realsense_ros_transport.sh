#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

ROS_SETUP="/opt/ros/humble/setup.bash"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REALSENSE_FPS="${REALSENSE_FPS:-15}"
DEFAULT_REALSENSE_PROFILE="480x270x${REALSENSE_FPS}"
COLOR_PROFILE="${REALSENSE_COLOR_PROFILE:-$DEFAULT_REALSENSE_PROFILE}"
DEPTH_PROFILE="${REALSENSE_DEPTH_PROFILE:-$DEFAULT_REALSENSE_PROFILE}"
HARDWARE_RESET="${REALSENSE_HARDWARE_RESET:-0}"
LOG_PATH="$PROJECT_DIR/output/realsense_safe_driver.log"
PID_PATH="$PROJECT_DIR/output/realsense_camera_launch.pid"

mkdir -p output .cache/ros_log

if [ ! -f "$ROS_SETUP" ]; then
    echo "FAIL: ROS setup file not found: $ROS_SETUP"
    exit 2
fi

# shellcheck disable=SC1091
set +u
source "$ROS_SETUP"
set -u

echo "RealSense ROS transport recovery"
echo "================================"
echo "Project: $PROJECT_DIR"
echo "FPS override: $REALSENSE_FPS"
echo "Color profile: $COLOR_PROFILE"
echo "Depth profile: $DEPTH_PROFILE"

echo
echo "1) Stopping old RealSense/grasp camera processes..."
if [ -f "$PID_PATH" ]; then
    stale_driver_pid="$(tr -cd '0-9' < "$PID_PATH")"
    if [ -n "$stale_driver_pid" ]; then
        kill -TERM -- "-$stale_driver_pid" >/dev/null 2>&1 || true
        sleep 1
        kill -KILL -- "-$stale_driver_pid" >/dev/null 2>&1 || true
    fi
    rm -f "$PID_PATH"
fi
pkill -f "main.py --bottle --realtime" >/dev/null 2>&1 || true
pkill -f "run_bottle_realtime.sh" >/dev/null 2>&1 || true
pkill -f "real_time_live.py --backend-only" >/dev/null 2>&1 || true
pkill -f "realsense_transport_health.py" >/dev/null 2>&1 || true
pkill -f "realsense2_camera_node" >/dev/null 2>&1 || true
pkill -f "ros2 launch realsense2_camera rs_launch.py" >/dev/null 2>&1 || true
sleep 1
# A disconnected UVC node can emit gigabytes of ENODEV errors.  Once its
# process group is stopped, discard that failed-session log before retrying.
: > "$LOG_PATH"
# Never let a previous successful session masquerade as the current recovery.
rm -f "$PROJECT_DIR/output/realsense_camera_health.json"

echo
echo "2) Discovering RealSense hardware..."
if ! "$PYTHON_BIN" -m grasp_project.hardware.device_discovery --realsense-check; then
    echo "CAMERA STREAM ERROR: No selectable RealSense camera was discovered."
    echo "Selection uses serial/model identity, never a fixed USB bus/device path."
    echo "If multiple RealSense cameras are connected, set REALSENSE_SERIAL or config/hardware.yaml."
    "$PYTHON_BIN" scripts/realsense_transport_health.py \
        --device-unavailable \
        --output "$PROJECT_DIR/output/realsense_camera_health.json" >/dev/null || true
    exit 2
fi

echo
if [[ "${HARDWARE_RESET,,}" =~ ^(1|true|yes|on)$ ]]; then
    echo "2b) Hardware-resetting RealSense device (explicit REALSENSE_HARDWARE_RESET opt-in)..."
    "$PYTHON_BIN" - <<'PYEOF' || echo "WARN: hardware reset skipped/failed; continuing with driver restart only."
import time
import pyrealsense2 as rs

devices = list(rs.context().query_devices())
if not devices:
    raise SystemExit("no RealSense device visible to librealsense")
for dev in devices:
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"hardware_reset -> serial {serial}")
    dev.hardware_reset()
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    if list(rs.context().query_devices()):
        print("device re-enumerated after reset")
        break
    time.sleep(1)
else:
    raise SystemExit("device did not re-enumerate within 15s after reset")
PYEOF
else
    echo "2b) Skipping RealSense hardware reset during normal startup."
    echo "    Set REALSENSE_HARDWARE_RESET=1 only for an explicit firmware recovery."
fi

echo
echo "3) Restarting ROS daemon..."
ros2 daemon stop >/dev/null 2>&1 || true
sleep 1
ros2 daemon start >/dev/null 2>&1 || true
sleep 1

echo
echo "4) Starting RealSense with safer transport settings..."
rm -f "$LOG_PATH" "$PID_PATH"
setsid ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    pointcloud.enable:=false \
    enable_sync:=false \
    rgb_camera.color_profile:="$COLOR_PROFILE" \
    depth_module.color_profile:="$COLOR_PROFILE" \
    depth_module.depth_profile:="$DEPTH_PROFILE" \
    enable_infra1:=false \
    enable_infra2:=false \
    >"$LOG_PATH" 2>&1 &
driver_pid="$!"
echo "$driver_pid" > "$PID_PATH"
echo "Driver launch PID: $driver_pid"
echo "Driver log: $LOG_PATH"

echo
echo "5) Waiting for topics..."
sleep 6

echo
echo "6) Raw topic list:"
ros2 topic list | grep -E "/camera/camera/(color|aligned_depth|depth|.*info)" || true

echo
echo "7) Camera info one-shot:"
timeout 5 ros2 topic echo /camera/camera/color/camera_info --once || true

echo
echo "8) Standalone image decode/FPS health check:"
health_rc=2
for attempt in 1 2 3; do
    echo "Health attempt $attempt/3..."
    if "$PYTHON_BIN" scripts/realsense_transport_health.py --timeout 8 --min-fps 5; then
        health_rc=0
        break
    fi
    health_rc="$?"
    if [ "$attempt" -lt 3 ]; then
        echo "Health check failed while streams are warming up; retrying..."
        sleep 3
    fi
done

# A failed ROS camera node retries UVC extension-control requests indefinitely.
# Stop that process group before returning failure so a transient stream fault
# cannot become a USB retry storm that prevents clean re-enumeration.
if [ "$health_rc" -ne 0 ]; then
    echo "Health gate failed; stopping the failed RealSense driver before returning."
    kill -TERM -- "-$driver_pid" >/dev/null 2>&1 || true
    sleep 1
    kill -KILL -- "-$driver_pid" >/dev/null 2>&1 || true
    rm -f "$PID_PATH"
fi
exit "$health_rc"

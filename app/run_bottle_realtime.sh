#!/usr/bin/env bash

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ROS_SETUP="/opt/ros/humble/setup.bash"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REALSENSE_FPS="${REALSENSE_FPS:-15}"
DEFAULT_REALSENSE_PROFILE="480x270x${REALSENSE_FPS}"
COLOR_PROFILE="${REALSENSE_COLOR_PROFILE:-$DEFAULT_REALSENSE_PROFILE}"
DEPTH_PROFILE="${REALSENSE_DEPTH_PROFILE:-$DEFAULT_REALSENSE_PROFILE}"
HEADLESS_WEB=0

usage() {
    cat <<EOF
Usage: ./run_bottle_realtime.sh [--headless-web]

Starts the RealSense ROS stream if needed and launches:
  desktop legacy: python3 main.py --bottle --realtime --lock-target
  headless web:   python3 real_time_live.py --backend-only --lock-target

No robot commands are sent. The final web demo uses --headless-web so no desktop GUI opens.
Default RealSense transport profile is conservative:
  color=$COLOR_PROFILE depth=$DEPTH_PROFILE
Override with REALSENSE_FPS=30 or explicit REALSENSE_COLOR_PROFILE/REALSENSE_DEPTH_PROFILE.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --headless-web|--backend-only)
            HEADLESS_WEB=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
    shift
done

if [ ! -f "$ROS_SETUP" ]; then
    echo "ERROR: ROS setup file not found: $ROS_SETUP"
    exit 1
fi

export ROS_LOG_DIR="${ROS_LOG_DIR:-$PROJECT_DIR/.cache/ros_log}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export APPLY_CAMERA_ROTATION=0
export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
mkdir -p "$ROS_LOG_DIR" "$MPLCONFIGDIR" output

source "$ROS_SETUP"

started_camera=0
camera_launch_pid=""

stop_camera_if_needed() {
    if [ "$started_camera" -eq 1 ] && [ -n "$camera_launch_pid" ]; then
        echo "Stopping RealSense driver..."
        kill "$camera_launch_pid" >/dev/null 2>&1 || true
        wait "$camera_launch_pid" >/dev/null 2>&1 || true
    fi
}
trap stop_camera_if_needed EXIT

if pgrep -af "realsense2_camera|rs_launch" >/dev/null 2>&1; then
    echo "RealSense driver already running."
else
    echo "Discovering RealSense camera by stable hardware identity..."
    if ! "$PYTHON_BIN" -m grasp_project.hardware.device_discovery --realsense-check; then
        echo
        echo "CAMERA STREAM ERROR: No selectable RealSense camera was discovered."
        echo "Move/replug the camera, or set REALSENSE_SERIAL when multiple RealSense devices are connected."
        exit 2
    fi
    echo "Starting RealSense driver..."
    ros2 launch realsense2_camera rs_launch.py \
        align_depth.enable:=true \
        pointcloud.enable:=false \
        enable_sync:=false \
        rgb_camera.color_profile:="$COLOR_PROFILE" \
        depth_module.color_profile:="$COLOR_PROFILE" \
        depth_module.depth_profile:="$DEPTH_PROFILE" \
        enable_infra1:=false \
        enable_infra2:=false &
    camera_launch_pid="$!"
    started_camera=1
    sleep 6
fi

if [[ "${REALSENSE_TRANSPORT_PREVALIDATED:-0}" =~ ^(1|true|yes|on)$ ]]; then
    echo "RealSense transport was just validated by the canonical recovery gate; skipping duplicate health subscription."
else
    echo "Checking RealSense ROS image transport before launching grasp GUI..."
    if ! "$PYTHON_BIN" scripts/realsense_transport_health.py --timeout 5 --min-fps 5; then
        echo
        echo "CAMERA STREAM ERROR: RealSense ROS topic unreadable"
        echo "Suggested fix: bash scripts/recover_realsense_ros_transport.sh"
        exit 2
    fi
fi

if [ "$HEADLESS_WEB" -eq 1 ]; then
    echo "Launching realtime bottle pre-grasp backend for web dashboard..."
    "$PYTHON_BIN" real_time_live.py --backend-only --lock-target
else
    echo "Launching legacy realtime bottle pre-grasp desktop GUI..."
    "$PYTHON_BIN" main.py --bottle --realtime --lock-target
fi

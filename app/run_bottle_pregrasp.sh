#!/usr/bin/env bash
# LEGACY/DEBUG ONLY. Final demo command is ./run_final_grasp_demo.sh --with-davis

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ROS_SETUP="/opt/ros/humble/setup.bash"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/sam_env/bin/python}"
GUI=1
KEEP_CAMERA=0
NO_ROTATE=0

usage() {
    cat <<EOF
Usage: ./run_bottle_pregrasp.sh [options]

Options:
  --no-gui       Run headless and only save output files.
  --keep-camera  Leave the RealSense ROS driver running after exit.
  --no-rotate    Compatibility flag; frames are already upright and unrotated.
  -h, --help     Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-gui)
            GUI=0
            ;;
        --keep-camera)
            KEEP_CAMERA=1
            ;;
        --no-rotate)
            NO_ROTATE=1
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

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN"
    exit 1
fi

export ROS_LOG_DIR="${ROS_LOG_DIR:-$PROJECT_DIR/.cache/ros_log}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$ROS_LOG_DIR" "$MPLCONFIGDIR" output

source "$ROS_SETUP"

started_camera=0
camera_launch_pid=""

stop_camera_if_needed() {
    if [ "$started_camera" -eq 1 ] && [ "$KEEP_CAMERA" -eq 0 ] && [ -n "$camera_launch_pid" ]; then
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
        echo "CAMERA STREAM ERROR: No selectable RealSense camera was discovered."
        echo "Move/replug the camera, or set REALSENSE_SERIAL when multiple RealSense devices are connected."
        exit 2
    fi
    echo "Starting RealSense driver..."
    ros2 launch realsense2_camera rs_launch.py \
        align_depth.enable:=true \
        rgb_camera.color_profile:=848x480x30 \
        depth_module.depth_profile:=848x480x30 &
    camera_launch_pid="$!"
    started_camera=1
    sleep 4
fi

echo "Capturing fresh RGB-D frame..."
export APPLY_CAMERA_ROTATION=0
python3 scripts/capture_rgb_depth.py

args=(main.py --no-capture --bottle)
if [ "$GUI" -eq 0 ]; then
    args+=(--no-gui)
fi
if [ "$NO_ROTATE" -eq 1 ]; then
    args+=(--no-rotate)
fi

echo "Running bottle pre-grasp safety evaluation..."
"$PYTHON_BIN" "${args[@]}"

echo
echo "Latest outputs:"
echo "  output/pregrasp_safety_report.json"
echo "  output/bottle_grasp_debug.png"
echo "  output/dashboard_summary.png"
echo "  output/wrench_debug_dashboard.png"

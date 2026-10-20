#!/usr/bin/env bash
# LEGACY/DEBUG ONLY. Final demo command is ./run_final_grasp_demo.sh --with-davis
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" output

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "WARNING: ROS Humble setup not found at /opt/ros/humble/setup.bash"
fi

python3 -m grasp_project.hardware.device_discovery --realsense-check || {
    echo "WARNING: RealSense auto-discovery did not select a camera."
    echo "If multiple RealSense cameras are connected, set REALSENSE_SERIAL or config/hardware.yaml."
}

echo "Starting main working demo: RealSense RGB-D grasp planner."
echo "Shows RGB, depth, bottle mask, clamp patches, movement X/Y/Z, visual contact, epsilon, final decision."
exec ./run_bottle_realtime.sh

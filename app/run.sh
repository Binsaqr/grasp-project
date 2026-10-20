#!/bin/bash
# LEGACY/DEBUG ONLY. Final demo command is ./run_final_grasp_demo.sh --with-davis

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$PROJECT_DIR/.cache/ros_log}"
mkdir -p "$ROS_LOG_DIR"

echo "1/4 Capturing RGB-D image..."
source /opt/ros/humble/setup.bash
python3 scripts/capture_rgb_depth.py

if [ $? -ne 0 ]; then
    echo "ERROR: RGB-D capture failed."
    exit 1
fi

echo "2/4 Running SAM + epsilon grasp evaluation..."
source sam_env/bin/activate
python3 scripts/hybrid_opencv_sam.py

if [ $? -ne 0 ]; then
    echo "ERROR: SAM pipeline failed."
    exit 1
fi

echo "3/4 Running RGB-D validation..."
python3 scripts/rgbd_grasp_eval.py

if [ $? -ne 0 ]; then
    echo "ERROR: RGB-D validation failed."
    exit 1
fi

echo "4/4 Opening GUI..."
python3 visualization/gui_dashboard.py

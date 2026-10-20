#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Launching: RealSense RGB-D Grasp Planner"
echo "Requires: Intel RealSense color + aligned depth ROS2 topics"
echo "Outputs: output/latest_overlay_live.jpg, output/latest_realtime_status.json"
echo "If the camera is missing: run ./scripts/check_demo_system.sh and verify ROS_DOMAIN_ID=67"
echo
exec ./run_realsense_grasp_demo.sh "$@"

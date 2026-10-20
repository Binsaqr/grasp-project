#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "Running RealSense point cloud diagnostic demo"
echo "Requires: RealSense RGB + aligned depth topics"
echo "Outputs: output/pointcloud_status.json, output/latest_realsense_pointcloud.ply, output/latest_bottle_pointcloud.ply"
python3 -m grasp_project.realsense.pointcloud --test-from-live --duration "${1:-5}"

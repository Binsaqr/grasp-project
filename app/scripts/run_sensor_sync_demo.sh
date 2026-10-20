#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-2d}"
echo "Running synchronized sensor demo mode: $MODE"
echo "Requires: existing RealSense/DAVIS status outputs. DAVIS may report stream-active-no-motion if no events occur."
echo "Outputs: output/sync_status.json, output/latest_sensor_bundle.json, output/sensor_sync_report.json"
python3 -m grasp_project.demos.run_sensor_sync_demo --mode "$MODE" --duration "${SYNC_DEMO_DURATION:-10}"

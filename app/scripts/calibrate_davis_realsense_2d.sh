#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/calibrate_davis_realsense_2d.sh correspondences.json"
  echo "JSON must contain davis_points and realsense_points arrays."
  exit 2
fi

python3 -m grasp_project.sync.calibrate_2d --input "$1" --output config/davis_to_realsense_homography.json

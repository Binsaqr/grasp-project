#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Launching: DAVIS Motion Detector in the main web dashboard"
echo "Requires: DAVIS346 USB camera; permissions follow VID/PID udev rule, not a fixed USB port"
echo "Outputs: output/davis_motion_report.json and the DAVIS Motion web tab"
echo "If the camera is missing or denied: run bash scripts/check_davis_permissions.sh"
echo "No standalone desktop DAVIS GUI is launched."
echo
exec ./run_davis_motion_demo.sh "$@"

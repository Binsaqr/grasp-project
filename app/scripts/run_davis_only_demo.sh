#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Launching: Deprecated DAVIS-only path via the main web dashboard"
echo "Requires: DAVIS346 USB camera"
echo "Outputs: output/davis_motion_report.json and the DAVIS Motion web tab"
echo "Note: the old DAVIS-only desktop planner is no longer a normal-use GUI"
echo
exec ./run_davis_only_demo.sh "$@"

#!/usr/bin/env bash
# LEGACY/DEBUG ONLY. Final demo command is ./run_final_grasp_demo.sh --with-davis

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
mkdir -p "$MPLCONFIGDIR" output

echo "Starting combined web dashboard..."
echo "Open: http://127.0.0.1:8080"
echo
echo "Use the web buttons to start:"
echo "  - RealSense RGB-D grasp planner"
echo "  - DAVIS motion safety monitor"
echo "  - DAVIS Motion tab inside this dashboard"

exec python3 web_dashboard.py --host 127.0.0.1 --port 8080 --start-davis

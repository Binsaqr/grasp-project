#!/usr/bin/env bash
# LEGACY/DEBUG ONLY. Final demo command is ./run_final_grasp_demo.sh --with-davis

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" output

echo "Starting RealSense RGB-D bottle grasp demo..."
echo "Panels: RGB, depth, bottle mask, clamp patches, X/Y/Z movement, visual contact, epsilon, READY_TO_CLOSE."
exec ./run_bottle_realtime.sh

#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Launching: Demo Menu"
echo "Use this when you are unsure which demo to run."
echo "Outputs are written under output/."
echo
exec ./run_all_demos_menu.sh "$@"

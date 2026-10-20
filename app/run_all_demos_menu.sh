#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

while true; do
    cat <<'MENU'
===============================
FINAL GRASP PROJECT LAUNCHER
===============================
1) Final Combined Web Demo
2) Final Combined Web Demo + DAVIS Post-Grasp Monitor
3) Check Cameras / Dependencies
4) Exit
MENU
    read -r -p "Select demo: " choice
    case "$choice" in
        1) exec ./run_final_grasp_demo.sh ;;
        2) exec ./run_final_grasp_demo.sh --with-davis ;;
        3) ./check_demo_system.sh ;;
        4) exit 0 ;;
        *) echo "Invalid choice: $choice" ;;
    esac
    echo
done

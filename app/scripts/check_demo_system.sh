#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -r /opt/ros/humble/setup.bash ]; then
    # Make the documented `podman exec ... ./scripts/check_demo_system.sh`
    # command self-contained inside the supported ROS image.
    source /opt/ros/humble/setup.bash
fi

echo "Running: Final demo system check"
echo "Checks: ROS_DOMAIN_ID, Python imports, RealSense/DAVIS availability, output folder, writer count"
echo "If hardware is unplugged, camera warnings are expected; code/import failures should not be."
echo
exec ./check_demo_system.sh "$@"

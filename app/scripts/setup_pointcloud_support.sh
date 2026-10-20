#!/usr/bin/env bash
set -eo pipefail

echo "Point cloud support check"
echo "Python: $(python3 --version 2>&1)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

INSTALL=0
if [ "${1:-}" = "--install" ]; then
  INSTALL=1
fi

python3 - <<'PY'
import importlib.util
mods = ["numpy", "cv2", "open3d", "sensor_msgs"]
for name in mods:
    print(f"{name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY

if ! python3 -c "import open3d" >/dev/null 2>&1; then
  if [ "$INSTALL" -eq 1 ]; then
    echo "Installing Open3D with python3 -m pip install open3d"
    python3 -m pip install open3d || echo "WARN: Open3D install failed; NumPy/ASCII PLY fallback remains available."
  else
    echo "WARN: Open3D missing. Re-run with --install to attempt python3 -m pip install open3d."
  fi
fi

python3 -m grasp_project.realsense.pointcloud --diagnose

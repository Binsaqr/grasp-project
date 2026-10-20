#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LAYOUT="${1:-unknown}"
echo "Running mounting experiment summary for layout: $LAYOUT"
python3 -m grasp_project.sync.sensor_sync --test-live --duration "${MOUNTING_EXPERIMENT_DURATION:-10}"
python3 - <<PY
import json, time
from pathlib import Path
out = Path("output/mounting_experiment_report.json")
sync = json.loads(Path("output/sensor_sync_report.json").read_text()) if Path("output/sensor_sync_report.json").exists() else {}
cal = json.loads(Path("output/sync_status.json").read_text()) if Path("output/sync_status.json").exists() else {}
payload = {
    "timestamp": time.time(),
    "mount_layout": "$LAYOUT",
    "sync": sync,
    "calibration_status": cal,
    "required_measurements": ["horizontal_offset_mm", "vertical_offset_mm", "approximate_baseline_mm", "angle_difference_deg"],
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

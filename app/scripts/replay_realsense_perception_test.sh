#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SEQUENCE_DIR="${1:-output/perception_validation/sequences}"

echo "Running camera-free RealSense perception replay checks"
echo "Sequence root: $SEQUENCE_DIR"
echo "Current replay gate uses synthetic/unit fixtures until labeled sequences are available."

python3 -m pytest tests/test_realsense_perception_result.py

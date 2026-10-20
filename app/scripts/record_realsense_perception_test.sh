#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/capture_grasp_trial.py" "${1:-unlabeled}" --stage distant

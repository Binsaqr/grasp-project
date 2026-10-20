#!/usr/bin/env bash
# Preserve the original launch command with the runtime grouped under app/.
set -euo pipefail
REPOSITORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$REPOSITORY_DIR/app/run_docker_demo.sh" "$@"

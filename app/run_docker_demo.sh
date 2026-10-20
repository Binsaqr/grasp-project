#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8080}"

mkdir -p output sample_data

if ! command -v podman >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
    echo "Podman or Docker is required to run the camera application." >&2
    exit 1
fi

echo "Building the current grasp application..."
if command -v podman >/dev/null 2>&1; then
    podman build -t localhost/grasp-project:release .
    # Remove only the former project container name used by older releases.
    podman rm -f grasp-project-dashboard >/dev/null 2>&1 || true
    echo "Starting the dashboard, RealSense, and DAVIS cameras..."
    echo "Open http://$HOST:$PORT after startup completes."
    echo "Press Ctrl+C to stop."
    exec podman run --rm --replace --name grasp-dashboard --network host --privileged --security-opt label=disable --ipc host \
        -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}" \
        -e HOST="$HOST" \
        -e PORT="$PORT" \
        -v "$PROJECT_DIR/output:/app/output" \
        -v "$PROJECT_DIR/config:/app/config" \
        -v /dev:/dev \
        -v /run/udev:/run/udev:ro \
        localhost/grasp-project:release
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required. Install the Docker Compose plugin, then retry." >&2
    exit 1
fi

echo "Starting the dashboard, RealSense, and DAVIS cameras with Docker Compose..."
echo "Open http://$HOST:$PORT after startup completes."
echo "Press Ctrl+C to stop."
exec docker compose up --build grasp-dashboard

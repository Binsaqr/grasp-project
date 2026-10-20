# Container setup

The existing image runs `./run_final_grasp_demo.sh --with-davis` with Ubuntu
22.04 / ROS 2 Humble, the RealSense ROS driver, the pyaer DAVIS backend and the
Flask dashboard. It retains the application's original launch path and writes
results to the mounted `output/` directory.

Unless stated otherwise, commands below run from `app/` (`cd app` from the
repository root). Image and container names remain `localhost/grasp-project:release`
and `grasp-dashboard`.

## Linux prerequisites and launch

Install Podman or Docker Engine with Compose v2. Connect the D405 and DAVIS346,
provide host USB permissions and ensure port 8080 is available. The first build
requires network access. From the repository root, run:

```bash
./run_docker_demo.sh
```

The wrapper delegates to the original application launcher, preferring Podman
when installed and otherwise using Docker Compose. Open
[http://127.0.0.1:8080](http://127.0.0.1:8080). Keep the default `HOST=127.0.0.1`;
the hardware path uses host networking, privileged device access and mounted
`/dev` and `/run/udev`. This is a local hardware workflow, not a hardened
internet-facing service.

## Direct build and Compose use

From `app/`:

```bash
podman build -t localhost/grasp-project:release .
# Docker alternative:
# docker compose up --build grasp-dashboard
```

The existing Compose file reads settings including `HOST`, `PORT`,
`ROS_DOMAIN_ID`, camera-selection mode, optional device serials, RealSense
profiles and DAVIS processing rates. Its default profiles are 640x480x30.
Set device selectors from local configuration when needed; do not commit unit
serials. For example, to change the dashboard port:

```bash
PORT=8081 docker compose up grasp-dashboard
```

The main Podman launcher forwards a smaller set of variables than Compose.
Inspect `app/run_docker_demo.sh` and `app/docker-compose.yml` when choosing a
custom launch. [Runtime recording](runtime_recording.md) supplies its required
explicit environment flags; the normal wrapper does not forward them.

Local `app/config/` and `app/output/` are mounted at `/app/config` and
`/app/output`. Training datasets, model checkpoints and generated exports are
not included in the image. For a RealSense-only Compose session:

```bash
docker compose run --rm grasp-dashboard \
  bash -lc 'source /opt/ros/humble/setup.bash && exec ./run_final_grasp_demo.sh'
```

## Windows USB/IP procedure

The existing Windows launcher uses WSL 2, Docker Desktop and usbipd-win. Docker
Desktop does not directly pass physical USB devices to Linux containers. In an
Administrator PowerShell, inspect `usbipd list` and share the actual camera bus IDs:

```powershell
usbipd bind --busid <D405_BUSID>
usbipd bind --busid <DAVIS_BUSID>
```

Then run the existing launcher from `app/` in a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_docker_demo_windows.ps1
```

The script discovers the supported device VID/PIDs, attaches them to WSL, loads
`uvcvideo` and starts the existing application. `-Restart` rebuilds and replaces
an existing `grasp-dashboard` container. Windows-native applications cannot open
cameras while they are attached to WSL. This procedure is supplied with the
source; it does not establish hardware commissioning or validated sensor
operation on every Windows installation. macOS live-camera operation is not
established here.

## Health checks and permissions

```bash
podman exec grasp-dashboard ./scripts/check_demo_system.sh
# Docker alternative:
# docker exec grasp-dashboard ./scripts/check_demo_system.sh
```

Inspect dashboard availability, readable RealSense topics, DAVIS state and
producer ownership together. On the host, if DAVIS permissions fail:

```bash
./scripts/setup_davis_udev.sh
```

Unplug and reconnect the camera, then restart the application. Use dynamic
device discovery rather than hardcoded transient USB bus paths. See
[camera setup](camera_setup_guide.md) and [troubleshooting](troubleshooting.md).

## Stop and retain outputs

Stop the foreground launcher with Ctrl+C. For Compose-managed services,
`docker compose down` stops the service. Retain required logs, recordings and
configuration before starting another trial. Application results remain in
`app/output/` on the host; they are not source files or proof of a successful
physical grasp.

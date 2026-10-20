# Operator guide

[Project overview](../README.md) · [Documentation](index.md)

The application combines RealSense D405 RGB-D with DAVIS346 APS/events in a
hand-operated bottle demonstration. It provides perception and quality estimates;
it does not command a robot or gripper. Camera availability and software readiness
do not establish calibrated physical grasp performance or hardware safety.

## Start the application

On Linux, install Podman or Docker Engine with Compose v2, connect the cameras
and set the required USB permissions. From the repository root:

```bash
./run_docker_demo.sh
```

The first build downloads the ROS 2 Humble / Ubuntu 22.04 runtime. Open
[http://127.0.0.1:8080](http://127.0.0.1:8080). Keep the default loopback binding;
the hardware container uses privileged device access and host networking.

Check the running system from another terminal:

```bash
podman exec grasp-dashboard ./scripts/check_demo_system.sh
# Docker alternative:
# docker exec grasp-dashboard ./scripts/check_demo_system.sh
```

A missing camera remains unavailable. Do not treat a dashboard response or
camera-free test pass as evidence that both streams are operating. For device
setup and diagnostics, see [camera setup](camera_setup_guide.md),
[container setup](container.md) and [troubleshooting](troubleshooting.md).

## Documented hand-operated sequence

1. Secure the cameras and cables, position the bottle in view and confirm the active streams.
2. Hold the view steady for automatic target locking and the current analytic baseline; inspect the grasp band and distance guidance.
3. Use **Start Approach Mode** if required and approach manually.
4. Select **Start Post-Grasp Monitor** after the manual grasp. This declares a phase; it does not measure contact or physical closure.
5. Observe the fixed analytic baseline, eligible DAVIS motion contribution and live explanation during settling and holding. Stale, disturbed or unsuitable evidence may be excluded; exclusion does not prove stability.
6. End the manual trial using the existing release/end-test controls and use **Export Demo Summary** to retain the displayed result.

**Fresh Lock + Quality** and the Engineering **Reset Lock** controls support
recovery and recomputation. Automatic locking is the normal workflow. Keep
phase declarations separate from independently measured physical outcomes.

Existing freshness and display-validity limitations can allow old inputs to
produce current-looking guidance. Inspect source timestamps and stream health;
readiness labels are not a verified safety interlock. The
[quality guide](quality_matrix_explained.md) explains analytic versus operational
quality and the [sync guide](sensor_sync.md) explains timing eligibility.

## Engineering workflows

The existing application also includes point clouds, multi-view reconstruction,
object profiles, embodiment settings and Benchmark Lab. These are separate
engineering capabilities. Comparisons without independent ground truth remain
**Estimated Performance**; feature-isolation checks do not establish accuracy.

For a reproducible trial, follow [runtime recording](runtime_recording.md).
Recording requires its explicit Compose command; exporting recording variables
and calling the normal launcher does not enable the recording contract.

## Native Linux launch

With ROS 2 Humble, its RealSense packages, the Python dependencies and supported
native DAVIS driver installed:

```bash
cd app
source /opt/ros/humble/setup.bash
./run_final_grasp_demo.sh --with-davis
```

Other existing launchers remain in `app/` and `app/scripts/`. Run their commands
from `app/` to preserve their original relative paths. Live logs and generated
files are under `app/output/` on the host. Stop the foreground launcher with
Ctrl+C and retain required logs before another session rotates them.

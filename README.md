# RGB-D + Neuromorphic Grasp Quality

**Vision-based grasp assessment before and during manual handling.**

A research application combining RealSense RGB-D geometry with DAVIS346 APS and
event-motion evidence. The dashboard keeps the analytic quality of a locked
grasp separate from a live operational estimate, showing eligible motion
penalties and the reasons other evidence is excluded.

[Run the application](docs/operator-guide.md) · [Research method](docs/research_methodology.md) · [Documentation](docs/index.md) · [Contributing](CONTRIBUTING.md)

![Method overview: RealSense establishes a locked analytic baseline, DAVIS supplies gated motion evidence, and the dashboard presents a separate live estimate.](docs/assets/system-overview.svg)

*Method schematic; not a screenshot or a validation result.*

## What is included

- Live RealSense D405 color/depth and DAVIS346 APS/event workflows.
- Automatic target locking, analytic grasp quality, manual approach and post-grasp monitoring.
- Explainable live quality, motion diagnostics, point clouds, reconstruction, object profiles and benchmarking tools.
- Runtime recording, offline replay, clock diagnostics, evaluation and software regression tests.

The current demonstration is hand-operated and focused on a single bottle. It
does not command a robot or gripper. Readiness labels, analytic epsilon and motion
cues are estimates; they do not establish contact, force, slip accuracy or
hardware safety. Excluded or unavailable motion evidence does not prove stability.

## Run the application

The full sensor workflow uses the existing Ubuntu 22.04 / ROS 2 Humble container
on Linux with Podman or Docker Compose. Connect the cameras and provide the
required host USB permissions; the first build requires network access.

```bash
git clone https://github.com/Binsaqr/grasp-project.git
cd grasp-project
./run_docker_demo.sh
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). In another terminal, check the
running system:

```bash
podman exec grasp-dashboard ./scripts/check_demo_system.sh
# Docker alternative:
# docker exec grasp-dashboard ./scripts/check_demo_system.sh
```

The launcher uses privileged device access and host networking. Keep the default
loopback binding. The [operator guide](docs/operator-guide.md) explains the manual
sequence and exports; [container setup](docs/container.md) covers dependencies,
configuration and the existing Windows USB/IP procedure. Native Linux use is
available from `app/` with ROS 2 and sensor drivers installed:

```bash
cd app
source /opt/ros/humble/setup.bash
./run_final_grasp_demo.sh --with-davis
```

## Find your workflow

| Goal | Guide |
|---|---|
| Operate the live dashboard | [Operator guide](docs/operator-guide.md) |
| Understand analytic and live quality | [Quality matrix](docs/quality_matrix_explained.md) · [Live-quality method](docs/live_quality_methodology.md) |
| Understand evidence timing | [Software synchronization](docs/sensor_sync.md) |
| Record a trial | [Runtime recording](docs/runtime_recording.md) |
| Inspect an archive without cameras | [Offline replay](docs/offline_replay.md) |
| Design and evaluate experiments | [Recording experiments](docs/recording_experiments.md) · [Offline evaluation](docs/offline_evaluation.md) |
| Develop or run tests | [Contributing](CONTRIBUTING.md) |

## Repository layout

| Location | Contents |
|---|---|
| [`app/src/grasp_project/`](app/src/grasp_project/) | Canonical sensor, quality, dashboard, synchronization and support modules |
| [`app/`](app/) | Existing launchers, compatibility modules and optional legacy workflows |
| [`app/scripts/`](app/scripts/) · [`app/config/`](app/config/) | Capture, replay, evaluation and runtime configuration |
| [`app/tests/`](app/tests/) | Software regression tests |
| [`docs/`](docs/index.md) | Operator and technical guides, contracts and method diagrams |

Application-relative paths and launch behavior are preserved under `app/`.
The root launcher delegates to the existing application launcher. Generated
outputs belong in `app/output/`; raw recordings, model weights, personal
configuration and prior repository history are not part of this source package.

## Interpretation and licensing

Software tests and offline fixtures do not establish hardware commissioning,
physical grasp reliability, cross-camera calibration or an early-warning advantage.
Timing gates use software timestamps; stale-input and display behavior must be
checked against retained trial evidence. Benchmark results without ground truth
retain the **Estimated Performance** qualification.

This repository presents existing project source. No project reuse license has
been specified. See [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution,
dependencies and redistribution obligations; public visibility does not grant a
general reuse license.

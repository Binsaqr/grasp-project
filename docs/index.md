# Documentation

Start with the [project overview](../README.md) or the
[operator guide](operator-guide.md). Unless stated otherwise, technical commands
run from the application directory: `cd app` from the repository root.

## Operate and configure

| Guide | Purpose |
|---|---|
| [Operator guide](operator-guide.md) | Start the application, run the manual sequence and export results |
| [Container setup](container.md) | Linux container setup and the existing Windows USB/IP procedure |
| [Camera setup](camera_setup_guide.md) | Native sensor dependencies, discovery and permissions |
| [Troubleshooting](troubleshooting.md) | Diagnose camera, dashboard and quality issues |
| [Benchmarking](benchmark_user_guide.md) | Engineering comparisons and Estimated Performance limits |

## Understand the method

| Guide | Purpose |
|---|---|
| [Research method](research_methodology.md) | Sensor roles, research questions and required evidence |
| [Quality matrix](quality_matrix_explained.md) | Analytic baseline and live operational quality |
| [Live-quality method](live_quality_methodology.md) | Scoring, gates and interpretation |
| [Software synchronization](sensor_sync.md) | Timestamp provenance, freshness and sensor alignment |

## Record, replay and evaluate

| Guide | Purpose |
|---|---|
| [Runtime recording](runtime_recording.md) | Capture a reproducible runtime trial and check its artifacts |
| [Raw recording contract](raw_recording_contract.md) | Archive format, manifests and integrity |
| [Offline snapshot contract](offline_snapshot_contract.md) | Camera-free inspection boundary |
| [Offline replay](offline_replay.md) | Replay commands and result interpretation |
| [RGB-D trace](rgbd_trace.md) | Recorded RGB-D processing inputs and parity checks |
| [Clock mapping](clock_mapping.md) | Clock-domain mapping and uncertainty |
| [Timing endpoints](timing_endpoints.md) | Defined timing measurements and their limits |
| [Recording experiments](recording_experiments.md) | Study design, splits and independent labels |
| [Offline evaluation](offline_evaluation.md) | Evaluation inputs, metrics and unavailable-result states |
| [Epsilon sensitivity](epsilon_sensitivity.md) | Analytic sensitivity analysis |
| [Offline smoke test](offline_smoke.md) | Synthetic software-path checks, separate from hardware evidence |

These guides describe existing software and its contracts. They are not a
record of hardware commissioning or measured grasp performance. Raw capture
data and generated results remain outside the source repository.

For development and attribution, see [Contributing](../CONTRIBUTING.md) and
[Third-party components](../THIRD_PARTY.md).

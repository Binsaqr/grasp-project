# Contributing

Keep changes focused and preserve the existing live-camera workflows. The
application lives in `app/`; its relative imports, compatibility entrypoints and
subprocess paths are intentional. Read the [operator guide](docs/operator-guide.md)
and [third-party notes](THIRD_PARTY.md) before changing launch behavior or adding
external code.

## Development setup

For the full sensor environment, use the [ROS 2 Humble container](docs/container.md).
A pip installation alone does not provide ROS 2 or the native DAVIS driver.
For camera-free Python development, use an isolated Python 3.10–3.12 environment:

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Native
libraries and available wheels remain platform-specific. Dependencies are not
locked; retain the resolved versions when reporting results.

## Implementation approach

Confirm the need, search existing code, and extend the canonical implementation
before introducing a helper, dependency or service. Prefer the standard library,
existing platform capabilities and already installed dependencies. Keep wrappers
thin and avoid duplicate sensor backends, geometry helpers or quality calculations.

Canonical implementations include:

- `app/src/grasp_project/realsense/realtime_runner.py`: RealSense runtime.
- `app/src/grasp_project/davis/event_stream.py`: DAVIS event stream.
- `app/src/grasp_project/dashboard/web_dashboard.py`: dashboard.
- `app/src/grasp_project/realsense/epsilon_wrench.py`: analytic epsilon.
- `app/src/grasp_project/common/`: shared file, geometry and timing helpers.

Do not change epsilon mathematics, sensor scoring or runtime behavior as part of
a presentation-only or packaging change. Preserve
`./run_final_grasp_demo.sh --with-davis` from `app/` and the root container launcher.

## Verification

Run software tests from `app/`:

```bash
python -m pytest
python3 -m py_compile real_time_live.py web_dashboard.py davis_motion_gui.py event_postgrasp_monitor.py epsilon_quality.py
```

The complete suite includes ROS, Qt and JavaScript-dependent checks. The original
CI recipe is retained at `app/.github/workflows/camera-free.yml` as a reference
for reproducing the ROS/Qt/Node environment. It is not active GitHub automation
in this layout. Record missing prerequisites and skips explicitly. For live checks, use `ROS_DOMAIN_ID=67` and
run `./scripts/check_demo_system.sh` in the application environment.

A change report should identify the behavior changed, reused implementation,
commands run, complete test summary and remaining limitations. Keep hardware
validation separate from camera-free checks. Do not infer physical grasp or
safety behavior from passing software tests.

## Source and data hygiene

Use focused branches and pull requests. Keep generated images, profiles, logs
and reports under `app/output/`. Do not commit recordings, raw samples, model
weights, virtual environments, secrets, personal machine paths or internal
conversation and planning records. Use neutral placeholders for device IDs in
examples; retain real capture provenance with the local experiment records.

Before incorporating external material, record its original source, immutable
revision, exact files, license, attribution, modifications and obligations in
[THIRD_PARTY.md](THIRD_PARTY.md). Do not infer permission from public availability.

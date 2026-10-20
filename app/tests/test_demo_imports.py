"""Demo launcher file checks that do not require hardware."""

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_scripts_exist_and_are_executable():
    scripts = [
        "run_final_grasp_demo.sh",
        "run_realsense_grasp_demo.sh",
        "run_davis_motion_demo.sh",
        "run_davis_only_demo.sh",
        "run_all_demos_menu.sh",
        "scripts/run_realsense_grasp_demo.sh",
        "scripts/run_davis_motion_demo.sh",
        "scripts/run_davis_only_demo.sh",
        "scripts/run_all_demos_menu.sh",
        "scripts/setup_pointcloud_support.sh",
        "scripts/run_pointcloud_demo.sh",
        "scripts/run_sensor_sync_demo.sh",
        "scripts/calibrate_davis_realsense_2d.sh",
        "scripts/run_mounting_experiment.sh",
        "scripts/check_demo_system.sh",
    ]
    for rel in scripts:
        path = ROOT / rel
        assert path.exists(), rel
        assert os.access(path, os.X_OK), rel


def test_final_launcher_uses_dashboard_api_and_no_legacy_davis_gui():
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()
    assert "api/start/realsense" in launcher
    assert "api/davis/start" in launcher
    assert 'api_running realsense 2>/dev/null || echo false)" = "true" ] && [ "$(realsense_backend_process_count)" -gt 0' in launcher
    assert 'Path(f"/proc/{writer_pid}/cmdline")' in launcher
    assert "if rgb and depth:" in launcher
    assert 'LOG_DIR="$PROJECT_DIR/output/logs"' in launcher
    assert 'DASHBOARD_LOG="$LOG_DIR/dashboard.log"' in launcher
    assert "davis_motion_gui.py --live" not in launcher
    assert "cv2.imshow" not in launcher


def test_final_launcher_starts_davis_before_bounded_realsense_wait():
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()

    davis_start = launcher.index('echo "DAVIS: starting dashboard-owned backend..."')
    realsense_start = launcher.index('echo "RealSense: starting backend through dashboard API..."')
    realsense_wait = launcher.index("elif wait_for_realsense; then", realsense_start)
    assert davis_start < realsense_start < realsense_wait
    assert launcher.count("api_request POST /api/davis/start") == 2
    assert launcher.count("first open was unhealthy; performing one bounded") == 1
    retry = launcher.index("first open was unhealthy; performing one bounded")
    assert launcher.index("api_request POST /api/davis/stop", retry) < launcher.index(
        "api_request POST /api/davis/start", retry
    )


def test_final_launcher_rejects_running_but_unowned_or_unopened_davis():
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()

    davis_check = launcher.split("davis_running() {", 1)[1].split("\n}\n", 1)[0]
    summary = launcher.split("summarize_davis() {", 1)[1].split("\n}\n", 1)[0]
    assert 'owner == "owned" and open_count > 0' in davis_check
    assert '"BACKEND_UNAVAILABLE"' in davis_check
    assert 'state == "BACKEND_UNAVAILABLE" or owner != "owned" or open_count <= 0' in summary


def test_final_launcher_rotates_each_log_to_one_bounded_prior_tail(tmp_path):
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()
    function_start = launcher.index("prepare_bounded_log() {")
    function_end = launcher.index("\n}\n", function_start) + len("\n}\n")
    function_source = launcher[function_start:function_end]
    log_path = tmp_path / "dashboard.log"
    original = ("old-session-line\n" * 20).encode()
    log_path.write_bytes(original)

    subprocess.run(
        [
            "bash",
            "-c",
            f'{function_source}\nDEMO_LOG_MAX_BYTES=64\nprepare_bounded_log "$1"',
            "bounded-log-test",
            str(log_path),
        ],
        check=True,
    )

    assert log_path.with_suffix(".log.previous").read_bytes() == original[-64:]
    current = log_path.read_text()
    assert "=== Final demo session" in current
    assert "prior tail capped at 64 bytes" in current


def test_camera_recovery_does_not_leave_failed_driver_or_probe_storm():
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()
    recovery = (ROOT / "scripts/recover_realsense_ros_transport.sh").read_text()
    dashboard = (ROOT / "src/grasp_project/dashboard/web_dashboard.py").read_text()

    assert "device_discovery --summary" not in launcher
    assert 'if [ "$health_rc" -ne 0 ]; then' in recovery
    assert 'kill -TERM -- "-$driver_pid"' in recovery
    assert recovery.index("Stopping old RealSense/grasp camera processes") < recovery.index(
        "Discovering RealSense hardware"
    )
    assert 'pkill -f "realsense_transport_health.py"' in recovery
    assert ': > "$LOG_PATH"' in recovery
    assert "stop_realsense_recovery_processes()" in dashboard
    assert "except subprocess.TimeoutExpired:" in dashboard
    assert "RECONNECT_CIRCUIT_BREAKER_FAILURES = 3" in dashboard
    assert "RECONNECT_BASE_DELAY_SEC" in dashboard
    assert "automatic USB probing stopped" in dashboard
    assert "--device-unavailable" in recovery
    assert "re-arming one bounded recovery attempt" in dashboard
    assert 'RECONNECT_CIRCUIT_BREAKER_FAILURES - 1' in dashboard


def test_usb2_hand_camera_profile_keeps_30fps_at_lower_bandwidth():
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()
    recovery = (ROOT / "scripts/recover_realsense_ros_transport.sh").read_text()
    config = (ROOT / "config/realsense.yaml").read_text()
    dashboard = (ROOT / "src/grasp_project/dashboard/web_dashboard.py").read_text()

    assert 'DEFAULT_REALSENSE_PROFILE="480x270x${REALSENSE_FPS}"' in launcher
    assert 'DEFAULT_REALSENSE_PROFILE="480x270x${REALSENSE_FPS}"' in recovery
    assert 'REALSENSE_COLOR_PROFILE", f"480x270x' in dashboard
    assert 'REALSENSE_DEPTH_PROFILE", f"480x270x' in dashboard
    assert '"single_bottle_demo_mode": status.get("single_bottle_demo_mode")' in dashboard
    assert '"automatic_reacquisition_enabled": status.get("automatic_reacquisition_enabled")' in dashboard
    assert 'color_profile: "480x270x30"' in config
    assert 'depth_profile: "480x270x30"' in config


def test_final_single_bottle_release_has_no_required_yolo_download():
    launcher = (ROOT / "run_final_grasp_demo.sh").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert 'USE_YOLO_BOTTLE_TRIGGER="${USE_YOLO_BOTTLE_TRIGGER:-0}"' in launcher
    assert "ENV USE_YOLO_BOTTLE_TRIGGER=0" in dockerfile
    assert "urlretrieve" not in dockerfile


def test_container_launchers_disable_selinux_label_isolation():
    """The privileged camera container must not relabel saved evidence files."""
    launcher = (ROOT / "run_docker_demo.sh").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "--security-opt label=disable" in launcher
    assert "- label=disable" in compose
    assert "output:/app/output:z" not in launcher
    assert "./output:/app/output:z" not in compose


def test_smoke_fps_gate_requires_current_locked_fast_tracking_phase():
    smoke = (ROOT / "scripts/final_demo_smoke_test.sh").read_text()
    phase_gate = smoke.index('if [ "$fast_path" -eq 1 ]; then')
    fps_sample = smoke.index('status_path = Path("output/latest_realtime_status.json")')

    assert 'target_status not in {"LOCKED", "TRACKING"}' in smoke
    assert 'overlay_mode != "fast_tracking_local_locked_patch"' in smoke
    assert 'if not identity_current:' in smoke
    assert phase_gate < fps_sample


def test_config_and_output_placeholders_exist():
    required = [
        ".env.example",
        "pyproject.toml",
        "config/realsense.yaml",
        "config/davis.yaml",
        "config/demo.yaml",
        "config/quality_matrix.yaml",
        "config/camera_mount.yaml",
        "config/embodiment.yaml",
        "config/known_bottle_profile.json",
        "output/.gitkeep",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel

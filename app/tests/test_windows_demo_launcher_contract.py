from pathlib import Path


SOURCE = Path("run_docker_demo_windows.ps1").read_text()
DAVIS_READY = SOURCE.split("function Test-DavisReady", 1)[1].split(
    "function Test-DashboardReady", 1
)[0]
DASHBOARD_READY = SOURCE.split("function Test-DashboardReady", 1)[1].split(
    "if ($SelfTestReadiness)", 1
)[0]


def test_quiet_davis_readiness_uses_stream_and_heartbeat_not_event_rate():
    assert 'Stream.ToLowerInvariant() -eq "active"' in DAVIS_READY
    assert "report_heartbeat_current -eq $true" in DAVIS_READY
    assert "event_rate" not in DAVIS_READY
    assert "event_rate = 0.0" in SOURCE
    assert "Test-DavisReady $QuietHealthy" in SOURCE


def test_launcher_rejects_stale_davis_and_restarts_app_container_on_failure():
    assert "Test-DavisReady $Stopped" in SOURCE
    assert "Test-DavisReady $StaleHeartbeat" in SOURCE
    assert "--restart unless-stopped" in SOURCE


def test_launcher_waits_for_fresh_operator_pipeline_and_allows_slow_camera_startup():
    assert "$MaxAttempts = 240" in SOURCE
    assert "$Status.operator_snapshot_stale -eq $false" in DASHBOARD_READY
    assert "$Status.realtime_status_age_unknown -ne $true" in DASHBOARD_READY
    assert "Test-DashboardReady $StaleOperatorSnapshot" in SOURCE
    assert "Test-DashboardReady $IncompleteStatus" in SOURCE
    assert "Test-DavisReady $IncompleteStatus" in SOURCE
    assert "status payload is incomplete" in SOURCE
    assert "($Attempt % 10) -eq 0" in SOURCE

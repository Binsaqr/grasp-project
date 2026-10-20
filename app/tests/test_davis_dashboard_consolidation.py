"""Camera-free checks for DAVIS dashboard consolidation."""

import json
import os
from pathlib import Path
import time

import numpy as np


def test_davis_owner_guard_blocks_duplicate(monkeypatch, tmp_path):
    from grasp_project.davis import event_stream

    lock_path = tmp_path / "davis_backend.lock"
    monkeypatch.setattr(event_stream, "DAVIS_OWNER_LOCK_PATH", lock_path)
    first_ok, first = event_stream.acquire_davis_owner("test_owner")
    second_ok, second = event_stream.acquire_davis_owner("test_duplicate")
    try:
        assert first_ok is True
        assert first["davis_device_open_count"] == 1
        assert first["duplicate_open_prevented"] is False
        assert second_ok is False
        assert second["duplicate_davis_owner_detected"] is True
        assert second["duplicate_open_prevented"] is True
        assert second["davis_owner_status"] == "duplicate_owner_blocked"
    finally:
        event_stream.release_davis_owner(first.get("davis_writer_session_id"))


def test_davis_owner_guard_retires_reused_pid(monkeypatch, tmp_path):
    from grasp_project.davis import event_stream

    lock_path = tmp_path / "davis_backend.lock"
    monkeypatch.setattr(event_stream, "DAVIS_OWNER_LOCK_PATH", lock_path)
    lock_path.write_text(json.dumps({
        "davis_writer_pid": os.getpid(),
        "davis_writer_process_identity": "old-container-process",
        "davis_owner_start_time": time.time() - 30.0,
        "davis_owner_status": "owned",
        "davis_device_open_count": 1,
        "duplicate_davis_owner_detected": False,
    }))

    acquired, owner = event_stream.acquire_davis_owner("replacement_owner")
    try:
        assert acquired is True
        assert owner["davis_owner_status"] == "owned"
        assert owner["davis_writer_process_identity"] != "old-container-process"
    finally:
        event_stream.release_davis_owner(owner.get("davis_writer_session_id"))


def test_deprecated_event_monitor_gui_cli_does_not_launch_desktop_gui(capsys):
    from grasp_project.davis import event_stream

    code = event_stream.run_cli(["--gui"])
    captured = capsys.readouterr()
    assert code == 2
    assert "web dashboard" in captured.out.lower()


def test_davis_event_buffer_age_helper_returns_float():
    from grasp_project.davis.motion_gui import event_buffer_age_sec

    events = np.array([[1.0, 10.0, 20.0, 1.0], [1.25, 11.0, 21.0, 0.0]])
    assert event_buffer_age_sec(events) == 0.25
    assert event_buffer_age_sec(None) == 0.0


def test_davis_vertical_slip_sign_matches_canonical_convention():
    """motion_gui.classify_motion must agree with event_stream.py's slip_down
    convention (increasing DAVIS image Y = physical downward motion). This
    previously used a -1.0 flip found nowhere else in the codebase."""
    from grasp_project.davis.motion_gui import DAVIS_VERTICAL_SLIP_SIGN, classify_motion

    assert DAVIS_VERTICAL_SLIP_SIGN == 1.0

    def window(event_rate, vertical_event_bias, centroid_shift_y):
        return {
            "event_count": 200,
            "event_rate": event_rate,
            "event_density": 0.01,
            "centroid_x": 100.0,
            "centroid_y": 106.0,
            "centroid_shift_x": 0.0,
            "centroid_shift_y": centroid_shift_y,
            "vertical_event_bias": vertical_event_bias,
            "horizontal_event_bias": 0.0,
            "left_right_asymmetry": 0.0,
            "top_bottom_asymmetry": abs(vertical_event_bias),
            "event_cluster_spread": 0.3,
            "event_persistence": 20.0,
        }

    empty = window(0.0, 0.0, 0.0)
    # Bias toward more events in the lower half of the frame (larger pixel Y):
    # under the canonical convention this is a downward slip, so it must
    # surface as positive downward evidence, not be zeroed out by the sign.
    downward = window(1500.0, 0.30, 5.0)
    scores = classify_motion(empty, downward, empty, downward, downward, {}, "configured_roi")
    assert scores["downward_bias_abs"] > 0.0
    assert scores["downward_centroid_abs"] > 0.0


def test_davis_web_routes_are_available_without_camera():
    from grasp_project.dashboard import web_dashboard

    client = web_dashboard.app.test_client()
    for path in ("/api/davis/status", "/api/davis/features", "/api/davis/timeline"):
        response = client.get(path)
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        if path == "/api/davis/status":
            assert "event_state" in payload
            assert "aps_state" in payload
            assert "object_state" in payload
        if path == "/api/davis/features":
            assert "boundary_event_field" in payload

    response = client.post("/api/davis/calibrate")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is False
    assert "not running" in payload["message"].lower()

    response = client.post("/api/davis/snapshot")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert "manifest" in payload


def test_davis_missing_device_maps_to_device_not_found():
    from grasp_project.dashboard import web_dashboard

    report = {"davis_unavailable_reason": "No DAVIS device detected with VID:PID 152a:841a."}
    assert web_dashboard.davis_web_state(report, running=False) == "DEVICE_NOT_FOUND"
    assert web_dashboard.davis_event_state(report, running=False) == "DEVICE_NOT_FOUND"


def test_davis_stream_health_uses_raw_activity_not_object_roi_rate():
    from grasp_project.dashboard import web_dashboard

    report = {
        "event_stream_status": "active",
        "aps_stream_status": "active",
        "raw_window_event_count": 250,
        "object_roi_window_event_count": 0,
        "current_metrics": {"event_rate": 0.0},
        "event_analysis_status": "RAW_EVENTS_OUTSIDE_OBJECT_ROI",
    }

    assert web_dashboard.davis_web_state(report, running=True) == "STREAM_ACTIVE_EVENTS"
    assert web_dashboard.davis_event_state(report, running=True) == "STREAM_ACTIVE_EVENTS"


def test_compact_davis_summary_preserves_current_aps_state():
    from grasp_project.dashboard import web_dashboard

    summary = web_dashboard.summarize_davis(
        {
            "aps_state": "APS_COLOR_VERIFIED",
            "aps_stream_status": "active",
            "aps_fps": 19.5,
        }
    )

    assert summary["aps_state"] == "APS_COLOR_VERIFIED"
    assert summary["aps_stream_status"] == "active"
    assert summary["aps_fps"] == 19.5


def test_stopped_davis_report_is_explicitly_historical(monkeypatch, tmp_path):
    from grasp_project.dashboard import web_dashboard
    from grasp_project.davis import event_stream

    monkeypatch.setattr(
        web_dashboard,
        "read_davis_report",
        lambda: {
            "event_stream_status": "active",
            "aps_stream_status": "active",
            "motion_class": "GROSS SLIP",
            "event_rate": 50000.0,
        },
    )
    monkeypatch.setattr(web_dashboard, "DAVIS_LATEST_MOTION_REPORT_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(web_dashboard, "davis_thread", None)
    monkeypatch.setattr(
        event_stream,
        "read_davis_owner_lock",
        lambda: {"davis_owner_status": "unowned", "davis_device_open_count": 0},
    )

    payload = web_dashboard.davis_status_payload()

    assert payload["report_status"] == "HISTORICAL_NOT_CURRENT"
    assert payload["event_state"] == "STREAM_NOT_STARTED"
    assert payload["aps_state"] == "APS_UNAVAILABLE"
    assert payload["motion_class"] is None
    assert payload["event_rate"] is None


def test_sync_summary_preserves_live_quality_freshness_gates():
    from grasp_project.dashboard import web_dashboard

    summary = web_dashboard.summarize_sync_status(
        {
            "sync_quality": "good",
            "davis_realsense_sync_quality": "good",
            "davis_realsense_valid_for_live_quality": True,
            "bundle_valid_for_live_quality": True,
            "davis_event_age_ms": 20.0,
            "davis_event_freshness": "CURRENT",
            "davis_events_valid_for_live_quality": True,
            "pointcloud_freshness": "AGING",
            "pointcloud_valid_for_live_quality": False,
            "sensor_freshness": {"davis_events": {"freshness": "CURRENT"}},
            "davis_event_window_count": 0,
            "davis_event_rate": 0.0,
            "davis_event_window_source": "cached_report_diagnostic_only",
            "davis_event_window_materialized": False,
            "cached_report_event_count": 17,
            "cached_report_event_rate": 170.0,
        },
        {"davis_event_count": 17, "davis_event_rate": 170.0},
        {},
        {},
    )

    assert summary["davis_event_freshness"] == "CURRENT"
    assert summary["davis_realsense_sync_quality"] == "good"
    assert summary["davis_realsense_valid_for_live_quality"] is True
    assert summary["davis_events_valid_for_live_quality"] is True
    assert summary["pointcloud_valid_for_live_quality"] is False
    assert summary["sensor_freshness"]["davis_events"]["freshness"] == "CURRENT"
    assert summary["davis_event_window_count"] == 0
    assert summary["davis_event_rate"] == 0.0
    assert summary["davis_event_window_source"] == "cached_report_diagnostic_only"
    assert summary["davis_event_window_materialized"] is False
    assert summary["cached_report_event_count"] == 17
    assert summary["cached_report_event_rate"] == 170.0


def test_davis_launchers_route_to_web_dashboard():
    assert "web_dashboard.py" in Path("run_davis_motion_demo.sh").read_text()
    assert "davis_motion_gui.py --live" not in Path("run_davis_motion_demo.sh").read_text()
    assert "web_dashboard.py" in Path("run_davis_only_demo.sh").read_text()
    assert "davis_grasp_planner.py --live --gui" not in Path("run_davis_only_demo.sh").read_text()


def test_legacy_davis_viewer_wrappers_do_not_open_camera(capsys):
    import davis_event_viewer
    import davis_live_viewer
    import davis_slip_detection_gui

    assert davis_event_viewer.main([]) == 2
    assert davis_live_viewer.main([]) == 2
    assert davis_slip_detection_gui.main([]) == 2
    captured = capsys.readouterr()
    assert "web dashboard" in captured.out.lower()


def test_davis_grasp_planner_live_path_is_gated(capsys, monkeypatch):
    import davis_grasp_planner

    monkeypatch.delenv("DAVIS_ALLOW_STANDALONE_BACKEND", raising=False)
    assert davis_grasp_planner.main(["--live", "--duration", "0.1"]) == 2
    captured = capsys.readouterr()
    assert "competing camera owner" in captured.out.lower()

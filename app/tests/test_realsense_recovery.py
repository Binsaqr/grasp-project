"""Camera-free regressions for bounded RealSense USB recovery."""

from types import SimpleNamespace

from grasp_project.dashboard import web_dashboard


def test_d405_sysfs_transition_rearms_once_without_opening_camera(tmp_path):
    sysfs = tmp_path / "usb"
    device = sysfs / "1-9"
    device.mkdir(parents=True)
    (device / "idVendor").write_text("8086\n")
    (device / "idProduct").write_text("0b5b\n")

    original = dict(web_dashboard.realsense_usb_rearm)
    try:
        web_dashboard.realsense_usb_rearm.update(waiting=True, last_present=False)
        assert web_dashboard.realsense_d405_enumerated(sysfs) is True
        assert web_dashboard.observe_realsense_usb_for_rearm(True) is True
        assert web_dashboard.observe_realsense_usb_for_rearm(True) is False
    finally:
        web_dashboard.realsense_usb_rearm.clear()
        web_dashboard.realsense_usb_rearm.update(original)


def test_no_device_recovery_message_does_not_report_generic_topic_fault(monkeypatch, tmp_path):
    output = tmp_path / "output"
    logs = output / "logs"
    logs.mkdir(parents=True)
    web_dashboard.write_json_atomic(
        output / "realsense_camera_health.json",
        {
            "status": "DEVICE_NOT_CONNECTED",
            "pass": False,
            "last_error": "No selectable RealSense camera is currently enumerated.",
        },
    )
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    monkeypatch.setattr(web_dashboard, "LOG_DIR", logs)
    monkeypatch.setattr(
        web_dashboard.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2),
    )

    ok, message = web_dashboard.run_realsense_transport_recovery()

    assert ok is False
    assert "D405 is not enumerated" in message
    assert "ROS topic unreadable" not in message

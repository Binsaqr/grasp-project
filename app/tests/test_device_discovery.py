"""Camera-free tests for stable hardware discovery and selection."""

import json
import time
from pathlib import Path

from grasp_project.hardware import device_discovery as dd
from grasp_project.hardware import status_evidence as se


def _rs(serial="RS123", model="D435", node=None):
    return dd.RealSenseDeviceInfo(
        serial_number=serial,
        model_name=model,
        usb_speed="3.2",
        device_node=node,
    )


def _davis(node="/dev/bus/usb/001/002", permission_ok=True):
    return {
        "vendor_id": "152a",
        "product_id": "841a",
        "description": "INI DAViS FX3",
        "bus": "001",
        "device_number": Path(node).name,
        "device_node": node,
        "line": f"Bus 001 Device {Path(node).name}: ID 152a:841a INI DAViS FX3",
        "permission_ok": permission_ok,
    }


def _config(rs=None, davis=None):
    return {
        "realsense": {"selection": "auto", "serial": None, "preferred_model": None, "require_usb3": True, **(rs or {})},
        "davis": {"selection": "auto", "serial": None, "vendor_id": "152a", "product_id": "841a", **(davis or {})},
    }


def test_one_realsense_is_selected_automatically(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config())
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([_rs()], None))
    result = dd.select_realsense_device()
    assert result.selected.serial_number == "RS123"
    assert result.status == dd.DEVICE_DISCOVERED


def test_one_davis_is_selected_automatically(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config())
    monkeypatch.setattr(dd, "_backend_available", lambda name: (True, None))
    monkeypatch.setattr(dd, "find_usb_node_by_vid_pid", lambda vendor, product: [_davis()])
    monkeypatch.setattr(dd, "check_usb_access", lambda path: {"permission_ok": True, "exists": True, "readable": True, "writable": True})
    result = dd.select_davis_device()
    assert result.selected.vendor_id == "152a"
    assert result.selected.product_id == "841a"
    assert result.status == dd.DEVICE_DISCOVERED


def test_no_realsense_reports_unavailable(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config())
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([], "no camera"))
    monkeypatch.setattr(dd, "_rs_enumerate_devices", lambda: ([], "no camera"))
    result = dd.select_realsense_device()
    assert result.selected is None
    assert result.status == dd.DEVICE_NOT_CONNECTED


def test_multiple_realsense_without_serial_is_ambiguous(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config())
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([_rs("A"), _rs("B")], None))
    result = dd.select_realsense_device()
    assert result.selected is None
    assert result.status == dd.MULTIPLE_DEVICES


def test_configured_realsense_serial_selects_correct_device(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config(rs={"serial": "B"}))
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([_rs("A"), _rs("B")], None))
    result = dd.select_realsense_device()
    assert result.selected.serial_number == "B"


def test_configured_serial_absent_fails_clearly(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config(rs={"serial": "MISSING"}))
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([_rs("A")], None))
    result = dd.select_realsense_device()
    assert result.selected is None
    assert result.status == dd.DEVICE_NOT_CONNECTED
    assert "MISSING" in result.reason


def test_bus_device_path_change_does_not_affect_serial_selection(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config(rs={"serial": "A"}))
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([_rs("A", node="/dev/bus/usb/001/002")], None))
    first = dd.select_realsense_device()
    monkeypatch.setattr(dd, "_pyrealsense_devices", lambda: ([_rs("A", node="/dev/bus/usb/003/004")], None))
    second = dd.select_realsense_device()
    assert first.selected.serial_number == second.selected.serial_number == "A"
    assert first.selected.device_node != second.selected.device_node


def test_davis_permission_denied_diagnosis(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config())
    monkeypatch.setattr(dd, "_backend_available", lambda name: (True, None))
    monkeypatch.setattr(dd, "find_usb_node_by_vid_pid", lambda vendor, product: [_davis(permission_ok=False)])
    monkeypatch.setattr(dd, "check_usb_access", lambda path: {"permission_ok": False, "exists": True, "readable": True, "writable": False})
    result = dd.select_davis_device()
    assert result.status == dd.PERMISSION_DENIED
    assert "setup_davis_udev" in result.reason


def test_davis_reconnect_updates_transient_node(monkeypatch):
    monkeypatch.setattr(dd, "_hardware_config", lambda: _config())
    monkeypatch.setattr(dd, "_backend_available", lambda name: (True, None))
    node = {"value": "/dev/bus/usb/001/002"}
    monkeypatch.setattr(dd, "find_usb_node_by_vid_pid", lambda vendor, product: [_davis(node=node["value"])])
    monkeypatch.setattr(dd, "check_usb_access", lambda path: {"permission_ok": True, "exists": True, "readable": True, "writable": True})
    first = dd.select_davis_device()
    node["value"] = "/dev/bus/usb/003/004"
    second = dd.select_davis_device()
    assert first.selected.device_node != second.selected.device_node
    assert first.selected.vendor_id == second.selected.vendor_id == "152a"


def test_public_launchers_do_not_contain_fixed_usb_node_paths():
    root = Path(__file__).resolve().parents[1]
    launchers = list(root.glob("run*.sh")) + list((root / "scripts").glob("run*.sh"))
    for path in launchers:
        text = path.read_text()
        assert "/dev/bus/usb/002/" not in text
        assert "usb_port_id" not in text


def test_recent_runtime_intrinsics_are_accepted(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "latest_realtime_status.json").write_text(json.dumps({
        "timestamp": time.time(),
        "writer_session_id": "writer-live",
        "frame_id": 42,
        "intrinsics": {"fx": 390.0, "fy": 391.0, "cx": 320.0, "cy": 240.0},
    }))

    matches = se.fresh_intrinsics_matches(tmp_path, now=time.time())

    assert len(matches) == 1
    assert matches[0].path.name == "latest_realtime_status.json"


def test_stale_missing_timestamp_and_synthetic_intrinsics_are_rejected(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    intrinsics = {"fx": 390.0, "fy": 391.0, "cx": 320.0, "cy": 240.0}
    payloads = {
        "latest_realtime_status.json": {"writer_session_id": "writer", "intrinsics": intrinsics},
        "pointcloud_status.json": {"timestamp": 120.0, "pointcloud_frame_id": 1, "intrinsics": intrinsics},
        "current_quality_bundle.json": {"timestamp": time.time(), "synthetic": True, "quality_compute_id": "test", "intrinsics": intrinsics},
        "realtime_grasp_hulls.json": {"timestamp": time.time() - 601.0, "quality_compute_id": "old", "intrinsics": intrinsics},
    }
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload))

    assert se.fresh_intrinsics_matches(tmp_path, now=time.time(), max_age_sec=600.0) == []

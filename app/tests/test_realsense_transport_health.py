import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_health_module():
    path = ROOT / "scripts" / "realsense_transport_health.py"
    spec = importlib.util.spec_from_file_location("realsense_transport_health", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_health_requires_frames_at_end_of_window():
    health = _load_health_module()

    assert health.MAX_LAST_MESSAGE_AGE_SEC <= 0.5
    source = (ROOT / "scripts" / "realsense_transport_health.py").read_text()
    assert "color stream stopped before health window ended" in source
    assert "depth stream stopped before health window ended" in source
    assert "terminal_frames_current" in source


def test_device_unavailable_overwrites_health_with_zero_frame_failure(tmp_path):
    health = _load_health_module()
    output = tmp_path / "realsense_camera_health.json"
    output.write_text(json.dumps({"pass": True, "color_fps": 30.0}))

    report = health.device_unavailable_report()
    health.write_report(output, report)
    saved = json.loads(output.read_text())

    assert saved["status"] == "DEVICE_NOT_CONNECTED"
    assert saved["pass"] is False
    assert saved["device_present"] is False
    assert saved["color_fps"] == saved["depth_fps"] == 0.0
    assert "No selectable RealSense camera is currently enumerated" in saved["last_error"]

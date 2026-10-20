"""Camera-free tests for shared common helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from grasp_project.common.config import config_path, get_env_bool, get_env_float, get_env_int, load_config_file, output_path, parse_bool
from grasp_project.common.file_io import (
    append_jsonl,
    append_jsonl_bounded,
    json_safe,
    read_json,
    read_text,
    write_json_atomic,
    write_text_atomic,
)
from grasp_project.common.geometry import (
    bbox_area_px,
    bbox_aspect_long_short,
    bbox_center,
    bbox_iou_xyxy,
    bbox_xywh_to_xyxy,
    bbox_xyxy_from_geometry,
    bbox_xyxy_to_xywh,
    clip_bbox_xyxy,
    angle_delta_deg,
    distance_point_to_line,
    expand_bbox_xyxy,
    line_angle_deg,
    merge_bboxes_xyxy,
    normalize_angle_deg,
    point_in_bbox,
    point_distance_px,
    polygon_bbox_xyxy,
)
from grasp_project.common.image_utils import (
    bgr_to_rgb,
    encode_rgb_to_jpeg,
    fit_image_to_canvas,
    rgb_to_bgr,
    to_uint8_display,
    write_image_atomic,
)
from grasp_project.common.logging_utils import compact_exception, format_status_line, utc_iso
from grasp_project.common.timing import RateTracker, fps_from_timestamps, profile_set, profile_snapshot


def test_json_read_write_atomic(tmp_path):
    path = tmp_path / "nested" / "status.json"
    write_json_atomic(path, {"state": "TRACKING", "frame": np.int64(7), "mask": np.ones((2, 2), dtype=np.uint8)})

    assert json.loads(path.read_text())["frame"] == 7
    assert "mask" not in json.loads(path.read_text())
    assert read_json(path)["state"] == "TRACKING"
    assert read_json(tmp_path / "missing.json", default={"ok": False}) == {"ok": False}

    jsonl_path = tmp_path / "events.jsonl"
    append_jsonl(jsonl_path, {"value": np.float32(1.5)})
    assert json.loads(jsonl_path.read_text().strip()) == {"value": 1.5}
    assert json_safe({"arr": np.array([1, 2])}) == {"arr": [1, 2]}

    text_path = tmp_path / "notes" / "status.txt"
    write_text_atomic(text_path, "ready")
    assert read_text(text_path) == "ready"
    assert read_text(tmp_path / "missing.txt", default="missing") == "missing"


def test_bounded_jsonl_rotates_once_and_keeps_both_files_under_cap(tmp_path):
    path = tmp_path / "timeline.jsonl"
    limit = 160

    for index in range(30):
        assert append_jsonl_bounded(
            path,
            {"index": index, "state": "stable"},
            max_bytes=limit,
        )

    backup = tmp_path / "timeline.jsonl.previous"
    assert path.exists()
    assert backup.exists()
    assert path.stat().st_size <= limit
    assert backup.stat().st_size <= limit
    assert json.loads(path.read_text().splitlines()[-1])["index"] == 29


def test_bounded_jsonl_permission_failure_is_nonfatal_and_preserves_active_file(monkeypatch, tmp_path):
    path = tmp_path / "timeline.jsonl"
    path.write_text("x" * 160)

    def blocked_replace(_self, _target):
        raise PermissionError("destination temporarily open")

    monkeypatch.setattr(Path, "replace", blocked_replace)
    assert append_jsonl_bounded(path, {"state": "new"}, max_bytes=160) is False
    assert path.read_text() == "x" * 160


def test_bounded_jsonl_trims_inherited_oversized_timeline_before_append(tmp_path):
    path = tmp_path / "timeline.jsonl"
    path.write_text("".join(json.dumps({"index": index}) + "\n" for index in range(100)))

    assert append_jsonl_bounded(path, {"index": 100}, max_bytes=160)

    backup = tmp_path / "timeline.jsonl.previous"
    assert path.stat().st_size <= 160
    assert backup.stat().st_size <= 160
    assert json.loads(path.read_text().strip()) == {"index": 100}
    assert all(json.loads(line)["index"] < 100 for line in backup.read_text().splitlines())


def test_common_atomic_json_retries_transient_replace_lock(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    original_replace = Path.replace
    calls = {"count": 0}

    def transient_replace(self, target):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("destination temporarily open")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", transient_replace)
    write_json_atomic(path, {"state": "CURRENT"})
    assert read_json(path)["state"] == "CURRENT"
    assert calls["count"] == 3


def test_common_atomic_json_permanent_replace_failure_cleans_temp(monkeypatch, tmp_path):
    path = tmp_path / "status.json"

    def blocked_replace(_self, _target):
        raise PermissionError("destination remains open")

    monkeypatch.setattr(Path, "replace", blocked_replace)
    with pytest.raises(PermissionError):
        write_json_atomic(path, {"state": "CURRENT"})
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_bbox_helpers_accept_common_geometry_shapes():
    assert bbox_xyxy_from_geometry({"bbox_xyxy": [1, 2, 11, 22]}) == [1.0, 2.0, 11.0, 22.0]
    assert bbox_xyxy_from_geometry({"bbox_xywh": [1, 2, 10, 20]}) == [1.0, 2.0, 11.0, 22.0]
    assert bbox_xyxy_from_geometry({"oriented_box_points": [[4, 8], [1, 9], [3, 2], [6, 5]]}) == [
        1.0,
        2.0,
        6.0,
        9.0,
    ]

    assert bbox_center([0, 2, 10, 22]) == [5.0, 12.0]
    assert bbox_xywh_to_xyxy([1, 2, 10, 20]) == [1.0, 2.0, 11.0, 22.0]
    assert bbox_xyxy_to_xywh([1, 2, 11, 22]) == [1.0, 2.0, 10.0, 20.0]
    assert bbox_area_px([0, 0, 10, 20]) == 200.0
    assert bbox_aspect_long_short([0, 0, 10, 20]) == 2.0
    assert bbox_iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3
    assert bbox_iou_xyxy([0, 0, 0, 10], [5, 0, 15, 10]) == 0.0
    assert clip_bbox_xyxy([-5, 2, 12, 22], (20, 10)) == [0.0, 2.0, 10.0, 20.0]
    assert point_in_bbox([5, 6], [0, 0, 10, 10]) is True
    assert point_in_bbox([15, 6], [0, 0, 10, 10]) is False
    assert point_distance_px([0, 0], [3, 4]) == 5.0
    assert distance_point_to_line([1, 1], [0, 0], [2, 0]) == 1.0
    assert normalize_angle_deg(190) == -170.0
    assert angle_delta_deg(179, -179) == -2.0
    assert line_angle_deg([0, 0], [0, 1]) == 90.0
    assert merge_bboxes_xyxy([[0, 0, 2, 2], [1, 1, 5, 6]]) == [0.0, 0.0, 5.0, 6.0]
    assert expand_bbox_xyxy([2, 2, 8, 8], image_shape=(12, 12), min_pad_px=3) == [0.0, 0.0, 11.0, 11.0]
    assert polygon_bbox_xyxy([[1, 2], [5, 8], [3, 4]], image_shape=(20, 20), expand_px=1) == [
        0.0,
        1.0,
        6.0,
        9.0,
    ]


def test_image_helpers_encode_and_write(tmp_path):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255

    jpeg, convert_ms, encode_ms = encode_rgb_to_jpeg(rgb)

    assert isinstance(jpeg, bytes)
    assert jpeg.startswith(b"\xff\xd8")
    assert convert_ms >= 0.0
    assert encode_ms >= 0.0

    canvas = fit_image_to_canvas(rgb, 16, 10)
    assert canvas.shape == (10, 16, 3)
    assert np.array_equal(bgr_to_rgb(rgb_to_bgr(rgb)), rgb)
    scaled = to_uint8_display(np.array([[0.0, 0.5], [1.0, 2.0]], dtype=float))
    assert scaled.dtype == np.uint8
    assert int(scaled.max()) == 255
    image_path = tmp_path / "image.png"
    assert write_image_atomic(image_path, rgb_to_bgr(rgb)) is True
    assert image_path.read_bytes().startswith(b"\x89PNG")


def test_profile_timing_snapshot_records_stage():
    profile_set("unit_test_common_helper", 12.34)
    snapshot = profile_snapshot()

    assert snapshot["unit_test_common_helper"]["ms"] == 12.34
    assert snapshot["unit_test_common_helper"]["samples"] >= 1
    assert fps_from_timestamps([0.0, 0.5, 1.0]) == 2.0
    tracker = RateTracker(maxlen=3, clock=lambda: 0.0)
    tracker.tick(0.0)
    tracker.tick(0.5)
    tracker.tick(1.0)
    assert tracker.fps() == 2.0
    tracker.clear()
    assert tracker.fps() == 0.0


def test_config_helpers_load_json_flat_yaml_and_env(tmp_path, monkeypatch):
    json_path = tmp_path / "demo.json"
    json_path.write_text('{"port": 8080, "enabled": true}')
    yaml_path = tmp_path / "demo.yaml"
    yaml_path.write_text('host: "127.0.0.1"\nport: 8080\nenabled: true\n')

    assert load_config_file(json_path)["port"] == 8080
    assert load_config_file(yaml_path)["host"] == "127.0.0.1"
    assert load_config_file(yaml_path)["enabled"] is True
    assert load_config_file(tmp_path / "missing.yaml", default={"missing": True}) == {"missing": True}
    assert config_path("demo.yaml").name == "demo.yaml"
    assert output_path("status.json").name == "status.json"
    assert parse_bool("yes") is True
    assert parse_bool("off", default=True) is False
    assert parse_bool("maybe", default=True) is True
    monkeypatch.setenv("GRASP_TEST_BOOL", "true")
    monkeypatch.setenv("GRASP_TEST_INT", "42")
    monkeypatch.setenv("GRASP_TEST_FLOAT", "4.25")
    assert get_env_bool("GRASP_TEST_BOOL") is True
    assert get_env_int("GRASP_TEST_INT") == 42
    assert get_env_float("GRASP_TEST_FLOAT") == 4.25


def test_logging_helpers_format_stable_strings():
    assert utc_iso(0) == "1970-01-01T00:00:00Z"
    assert format_status_line("PASS", "ready", fps=30, skipped=None) == "PASS | ready | fps=30"
    try:
        raise ValueError("bad demo input")
    except ValueError as exc:
        assert compact_exception(exc) == "ValueError: bad demo input"

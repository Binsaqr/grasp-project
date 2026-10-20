"""Synthetic ROS messages only; no sensor or physical timing validation."""

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_project.recording import RecordingWriter, read_recording
from grasp_project.recording.ros_capture import record_ros_camera_info, record_ros_image


class MemorySink:
    def __init__(self):
        self.records = []
        self.drops = []

    def record(self, kind, arrays, **fields):
        record = {"kind": kind, "arrays": arrays, **fields}
        self.records.append(record)
        return record

    def record_drop(self, stream, count, reason, **fields):
        self.drops.append({"stream": stream, "count": count, "reason": reason, **fields})


def header():
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=1_700_000_003, nanosec=123_456_789),
        frame_id="realsense_color_optical_frame",
    )


def image(encoding="rgb8", *, data=None, step=None, is_bigendian=0):
    byte_count = {"rgb8": 3, "bgr8": 3, "16UC1": 2, "mono16": 2, "32FC1": 4}[encoding]
    width, height = 3, 2
    step = width * byte_count if step is None else step
    if data is None:
        data = bytearray(range(height * step))
    return SimpleNamespace(
        header=header(),
        width=width,
        height=height,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=step,
        data=data,
    )


def camera_info():
    return SimpleNamespace(
        header=header(),
        width=3,
        height=2,
        distortion_model="plumb_bob",
        k=[2.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0],
        r=np.eye(3).ravel(),
        p=list(range(12)),
        d=[0.1, 0.2, 0.3, 0.4, 0.5],
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(x_offset=0, y_offset=0, width=3, height=2, do_rectify=False),
    )


def record_image(sink, msg, stream="rgb"):
    return record_ros_image(
        sink,
        msg,
        stream=stream,
        host_arrival_monotonic_ns=123_456_789,
        host_arrival_epoch_ns=1_700_000_004_987_654_321,
    )


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "16UC1", "mono16", "32FC1"])
def test_preserves_exact_original_bytes_and_header(encoding):
    sink = MemorySink()
    msg = image(encoding)
    expected = bytes(msg.data)
    record = record_image(sink, msg, "rgb" if encoding in {"rgb8", "bgr8"} else "depth")
    msg.data[0] = 255

    assert record["arrays"]["raw_bytes"].dtype == np.uint8
    assert record["arrays"]["raw_bytes"].tobytes() == expected
    assert record["host_arrival_monotonic_ns"] == 123_456_789
    assert record["host_arrival_epoch_ns"] == 1_700_000_004_987_654_321
    assert record["native_clock"] == {
        "domain": "ros_header_unspecified",
        "unit": "ns",
        "timestamp": 1_700_000_003_123_456_789,
    }
    assert record["metadata"]["header_stamp_sec"] == 1_700_000_003
    assert record["metadata"]["header_stamp_nanosec"] == 123_456_789
    assert record["metadata"]["frame_id"] == msg.header.frame_id
    assert record["metadata"]["encoding"] == encoding
    assert record["source_sequence"] is None
    assert record["metadata"]["source_drop_count"] is None
    assert sink.drops[0]["count"] is None


def test_preserves_big_endian_float32_values_and_row_padding_without_conversion():
    sink = MemorySink()
    # Include NaN and infinity payloads: raw capture must not normalize them.
    values = np.array([0.25, 1.5, np.nan, 0.0, np.inf, -1.0], dtype=">f4")
    padded = values[:3].tobytes() + b"PAD!" + values[3:].tobytes() + b"END!"
    record = record_image(sink, image("32FC1", data=padded, step=16, is_bigendian=1), "depth")
    assert record["arrays"]["raw_bytes"].tobytes() == padded
    assert record["metadata"]["is_bigendian"] == 1
    assert record["metadata"]["step"] == 16


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("width", 0, "width"),
        ("height", 2.0, "integer"),
        ("step", 8, "row width"),
        ("data", b"short", "byte length"),
        ("encoding", "unsupported", "encoding"),
        ("is_bigendian", 3, "is_bigendian"),
    ],
)
def test_malformed_message_records_rejection_and_raises(field, value, reason):
    sink = MemorySink()
    msg = image()
    setattr(msg, field, value)
    with pytest.raises(ValueError, match=reason):
        record_image(sink, msg)
    assert sink.records == []
    assert sink.drops[0]["count"] == 1
    assert "invalid_ros_image" in sink.drops[0]["reason"]


def test_invalid_header_nanoseconds_are_rejected():
    sink = MemorySink()
    msg = image()
    msg.header.stamp.nanosec = 1_000_000_000
    with pytest.raises(ValueError, match="nanosec"):
        record_image(sink, msg)
    assert sink.records == []


def test_camera_info_preserves_complete_calibration_metadata():
    sink = MemorySink()
    msg = camera_info()
    record = record_ros_camera_info(sink, msg, host_arrival_monotonic_ns=100, host_arrival_epoch_ns=200)
    assert record["kind"] == "camera_info"
    assert record["arrays"] == {}
    assert record["metadata"]["k"] == msg.k
    assert record["metadata"]["r"] == msg.r.tolist()
    assert record["metadata"]["p"] == msg.p
    assert record["metadata"]["d"] == msg.d
    assert record["metadata"]["distortion_model"] == "plumb_bob"
    assert record["metadata"]["roi"]["width"] == 3
    msg.k[0] = 999
    assert record["metadata"]["k"][0] == 2.0


@pytest.mark.parametrize("field,value", [("k", [0.0]), ("d", [float("nan")])])
def test_camera_info_rejects_corrupt_calibration(field, value):
    sink = MemorySink()
    msg = camera_info()
    setattr(msg, field, value)
    with pytest.raises(ValueError, match="CameraInfo"):
        record_ros_camera_info(sink, msg, host_arrival_monotonic_ns=100, host_arrival_epoch_ns=200)
    assert sink.records == []
    assert sink.drops[0]["count"] == 1


def test_writer_failure_propagates_instead_of_successful_recording():
    class FailedSink(MemorySink):
        def record(self, *args, **kwargs):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        record_image(FailedSink(), image())


def test_adapters_roundtrip_through_recording_contract(tmp_path):
    writer = RecordingWriter(
        tmp_path / "raw",
        recording_id="synthetic-ros-fixture",
        config={"synthetic": True},
        revision={"commit": "synthetic-fixture"},
        metadata={"source": "synthetic", "trial_id": "synthetic-trial"},
    )
    depth = image("16UC1", data=np.array([1, 256, 500, 1024, 0, 65535], dtype=">u2").tobytes(), is_bigendian=1)
    record_image(writer, depth, "depth")
    record_ros_camera_info(writer, camera_info(), host_arrival_monotonic_ns=100, host_arrival_epoch_ns=200)
    writer.close()
    recording = read_recording(tmp_path / "raw")
    observations = [record for record in recording.records if record["record_type"] == "observation"]
    assert [record["kind"] for record in observations] == ["depth", "camera_info"]
    assert observations[0]["arrays"]["raw_bytes"].tobytes() == depth.data
    assert observations[0]["metadata"]["is_bigendian"] == 1
    assert observations[1]["metadata"]["k"][0] == 2.0
    assert recording.manifest["physical_validation"] is False
    assert all(record["count"] is None for record in recording.records if record["record_type"] == "drop")


def camera_worker_type():
    """Compile the actual callback class without importing Linux/ROS/Qt."""
    path = Path(__file__).parents[1] / "src/grasp_project/realsense/realtime_runner.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    camera_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "CameraWorker")
    signal = SimpleNamespace(emit=lambda *args: None)
    clock = SimpleNamespace(
        perf_counter=lambda: 1.0,
        time=lambda: 2.0,
        monotonic=lambda: 3.0,
        monotonic_ns=lambda: 3_123_456_789,
        time_ns=lambda: 2_987_654_321,
    )
    namespace = {
        "QObject": object,
        "pyqtSignal": lambda *args: signal,
        "pyqtSlot": lambda: lambda func: func,
        "time": clock,
        "deque": deque,
        "np": np,
        "FAST_ZERO_COPY_FRAME_HANDOFF": True,
    }
    # Execute only the actual repository callback class with synthetic dependencies.
    exec(compile(ast.Module(body=[camera_class], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["CameraWorker"]


def test_opt_in_callback_records_depth_before_float32_conversion():
    worker_class = camera_worker_type()
    sink = MemorySink()
    worker = worker_class(recording_sink=sink)
    original = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    msg = image("32FC1", data=original.tobytes())
    worker.depth_callback(msg)
    assert sink.records[0]["arrays"]["raw_bytes"].tobytes() == original.tobytes()
    assert sink.records[0]["host_arrival_monotonic_ns"] == 3_123_456_789
    assert sink.records[0]["host_arrival_epoch_ns"] == 2_987_654_321
    np.testing.assert_array_equal(worker.depth, (original * 1000).astype(np.uint16).reshape(2, 3))


def test_opt_in_rgb_and_camera_info_hooks_preserve_live_values():
    worker_class = camera_worker_type()
    sink = MemorySink()
    worker = worker_class(recording_sink=sink)
    msg = image()
    worker.color_callback(msg)
    worker.info_callback(camera_info())
    assert [record["kind"] for record in sink.records] == ["rgb", "camera_info"]
    np.testing.assert_array_equal(worker.color, np.frombuffer(msg.data, dtype=np.uint8).reshape(2, 3, 3))
    assert worker.intrinsics == {"fx": 2.0, "fy": 3.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 2}


def test_evaluated_pair_links_exact_raw_rgb_depth_and_calibration_ids():
    sink = MemorySink()
    worker = camera_worker_type()(recording_sink=sink)
    emitted = []
    worker.frame_ready = SimpleNamespace(emit=lambda *args: emitted.append(args))
    worker.info_callback(camera_info())
    worker.depth_callback(image("16UC1"))
    worker.color_callback(image())
    assert len(emitted) == 1
    intrinsics = emitted[0][2]
    assert intrinsics["_recording_camera_info_observation_id"] == sink.records[0]["observation_id"]
    assert intrinsics["_recording_depth_observation_id"] == sink.records[1]["observation_id"]
    assert intrinsics["_recording_rgb_observation_id"] == sink.records[2]["observation_id"]


def test_default_off_callbacks_need_no_recording_metadata():
    worker = camera_worker_type()()
    msg = image()
    del msg.header
    del msg.step
    del msg.is_bigendian
    worker.color_callback(msg)
    assert worker.color.shape == (2, 3, 3)


def test_callback_recording_failure_stops_before_processing():
    class FailedSink(MemorySink):
        def record(self, *args, **kwargs):
            raise OSError("disk full")

    worker = camera_worker_type()(recording_sink=FailedSink())
    with pytest.raises(OSError, match="disk full"):
        worker.color_callback(image())
    assert worker.color is None

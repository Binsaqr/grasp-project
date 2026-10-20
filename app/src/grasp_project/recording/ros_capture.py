"""Sensor-free adapters for recording ROS image messages before conversion.

ROS header stamps do not establish a hardware clock domain or synchronization.
Source losses are unknown: ROS 2 Image/CameraInfo headers have no sequence field.
These adapters deliberately propagate validation and writer failures.
"""

import math
import operator
import uuid

import numpy as np

_IMAGE_BYTES_PER_PIXEL = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
    "mono16": 2,
    "16UC1": 2,
    "32FC1": 4,
}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _header_metadata(msg):
    sec = _integer(msg.header.stamp.sec, "header.stamp.sec", minimum=None)
    nanosec = _integer(msg.header.stamp.nanosec, "header.stamp.nanosec")
    if nanosec >= 1_000_000_000:
        raise ValueError("header.stamp.nanosec must be < 1000000000")
    if not isinstance(msg.header.frame_id, str):
        raise TypeError("header.frame_id must be a string")
    return {
        "header_stamp_sec": sec,
        "header_stamp_nanosec": nanosec,
        "frame_id": msg.header.frame_id,
        "source_sequence_available": False,
        "source_drop_count": None,
        "source_drop_reason": "ros_header_has_no_source_sequence",
    }


def _record(sink, stream, arrays, metadata, monotonic_ns, epoch_ns):
    sink.record_drop(
        stream,
        None,
        "ros_header_has_no_source_sequence",
        host_arrival_monotonic_ns=monotonic_ns,
    )
    return sink.record(
        stream,
        arrays,
        observation_id=uuid.uuid4().hex,
        host_arrival_monotonic_ns=monotonic_ns,
        host_arrival_epoch_ns=epoch_ns,
        native_clock={
            "domain": "ros_header_unspecified",
            "unit": "ns",
            "timestamp": metadata["header_stamp_sec"] * 1_000_000_000
            + metadata["header_stamp_nanosec"],
        },
        metadata=metadata,
        source_sequence=None,
    )


def record_ros_image(
    sink,
    msg,
    *,
    stream,
    host_arrival_monotonic_ns,
    host_arrival_epoch_ns,
):
    """Record exact message bytes, including row padding and original endian.

    Supported encodings cover the current RGB/depth callbacks. No pixel
    conversion or reinterpretation is performed. Malformed observations are
    counted as locally rejected and raise instead of producing a valid record.
    """
    if stream not in {"rgb", "depth"}:
        raise ValueError("image stream must be rgb or depth")
    try:
        metadata = _header_metadata(msg)
        width = _integer(msg.width, "width", minimum=1)
        height = _integer(msg.height, "height", minimum=1)
        step = _integer(msg.step, "step", minimum=1)
        encoding = msg.encoding
        if encoding not in _IMAGE_BYTES_PER_PIXEL:
            raise ValueError(f"unsupported image encoding: {encoding!r}")
        endian = msg.is_bigendian
        if not isinstance(endian, (int, bool, np.integer, np.bool_)) or endian not in (0, 1):
            raise ValueError("is_bigendian must be 0 or 1")
        if step < width * _IMAGE_BYTES_PER_PIXEL[encoding]:
            raise ValueError("image step is smaller than encoded row width")
        raw_bytes = np.frombuffer(msg.data, dtype=np.uint8).copy()
        if raw_bytes.size != height * step:
            raise ValueError("image byte length does not equal height * step")
        metadata.update(
            width=width,
            height=height,
            step=step,
            encoding=encoding,
            is_bigendian=int(endian),
        )
    except (AttributeError, BufferError, TypeError, ValueError) as exc:
        sink.record_drop(
            stream,
            1,
            f"invalid_ros_image: {exc}",
            host_arrival_monotonic_ns=host_arrival_monotonic_ns,
        )
        raise ValueError(f"invalid ROS {stream} image: {exc}") from exc
    return _record(
        sink,
        stream,
        {"raw_bytes": raw_bytes},
        metadata,
        host_arrival_monotonic_ns,
        host_arrival_epoch_ns,
    )


def record_ros_camera_info(sink, msg, *, host_arrival_monotonic_ns, host_arrival_epoch_ns):
    """Retain calibration values from CameraInfo without importing ROS."""
    try:
        metadata = _header_metadata(msg)
        metadata.update(
            width=_integer(msg.width, "width", minimum=1),
            height=_integer(msg.height, "height", minimum=1),
            distortion_model=msg.distortion_model,
            binning_x=_integer(msg.binning_x, "binning_x"),
            binning_y=_integer(msg.binning_y, "binning_y"),
        )
        if not isinstance(metadata["distortion_model"], str):
            raise TypeError("distortion_model must be a string")
        for name, length in (("k", 9), ("r", 9), ("p", 12), ("d", None)):
            values = [float(value) for value in getattr(msg, name)]
            if length is not None and len(values) != length:
                raise ValueError(f"{name} must contain {length} values")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} contains nonfinite calibration values")
            metadata[name] = values
        metadata["roi"] = {
            name: _integer(getattr(msg.roi, name), f"roi.{name}")
            for name in ("x_offset", "y_offset", "height", "width")
        }
        if not isinstance(msg.roi.do_rectify, (bool, np.bool_)):
            raise TypeError("roi.do_rectify must be a boolean")
        metadata["roi"]["do_rectify"] = bool(msg.roi.do_rectify)
    except (AttributeError, TypeError, ValueError) as exc:
        sink.record_drop(
            "camera_info",
            1,
            f"invalid_ros_camera_info: {exc}",
            host_arrival_monotonic_ns=host_arrival_monotonic_ns,
        )
        raise ValueError(f"invalid ROS CameraInfo: {exc}") from exc
    return _record(
        sink,
        "camera_info",
        {},
        metadata,
        host_arrival_monotonic_ns,
        host_arrival_epoch_ns,
    )

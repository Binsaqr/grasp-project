#!/usr/bin/env python3
"""User-run bounded native capture; executing capture opens the canonical DAVIS backend.

--help never opens a sensor. Optional ROS subscribers consume an already-running
D405 producer; this command does not start a RealSense driver or robot hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _positive_seconds(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New recording directory; existing paths are rejected")
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--session-id", required=True, help="Operator-supplied acquisition session/group identifier")
    parser.add_argument("--object-id", required=True, help="Operator-supplied physical object identifier")
    parser.add_argument("--duration-sec", required=True, type=_positive_seconds)
    parser.add_argument("--poll-interval-sec", type=_positive_seconds, default=0.01)
    parser.add_argument("--with-ros", action="store_true", help="Also subscribe to existing D405 ROS image/calibration topics")
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    return parser


def _config_files():
    files = {}
    for path in sorted((PROJECT_ROOT / "config").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
            contents = path.read_bytes()
            files[path.relative_to(PROJECT_ROOT).as_posix()] = {
                "contents_utf8": contents.decode("utf-8"),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
    return files


def _configuration():
    before = _config_files()
    import cv2
    import numpy as np

    from grasp_project.davis import motion_gui
    from grasp_project.davis.motion_estimation import load_davis_motion_config

    resolved = {
        "davis_motion": load_davis_motion_config(),
        "event_window_selection": {
            "primary_window_sec": motion_gui.PRIMARY_EVENT_WINDOW_SEC,
            "fallback_window_sec": motion_gui.FALLBACK_EVENT_WINDOW_SEC,
            "min_primary_events": motion_gui.MIN_PRIMARY_WINDOW_EVENTS,
        },
        "config_files": before,
        "runtime_versions": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__},
    }
    if _config_files() != before:
        raise ValueError("configuration files changed while resolving the capture snapshot")
    return resolved


def _revision():
    from scripts.capture_grasp_trial import _git

    status = _git(PROJECT_ROOT, "status", "--short")
    return {
        "commit": _git(PROJECT_ROOT, "rev-parse", "HEAD"),
        "branch": _git(PROJECT_ROOT, "branch", "--show-current"),
        "working_tree_dirty": None if status is None else bool(status),
        "status_short": None if status is None else status.splitlines(),
    }


def _create_monitor(writer, destination):
    from grasp_project.davis.event_stream import EventPostGraspMonitor

    return EventPostGraspMonitor(enabled=True, recording_sink=writer, output_dir=destination)


class _RosSubscriptions:
    def __init__(self, writer, args):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.rclpy = rclpy
        self.writer = writer
        self.node = None
        self.counts = {"rgb": 0, "depth": 0, "camera_info": 0}
        rclpy.init(args=[])
        try:
            self.node = rclpy.create_node("grasp_native_trial_recorder")
            self.subscriptions = [
                self.node.create_subscription(Image, args.rgb_topic, self._image_callback("rgb"), qos_profile_sensor_data),
                self.node.create_subscription(Image, args.depth_topic, self._image_callback("depth"), qos_profile_sensor_data),
                self.node.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_callback, qos_profile_sensor_data),
            ]
        except BaseException:
            self.close()
            raise

    def _image_callback(self, stream):
        def callback(message):
            monotonic_ns, epoch_ns = time.monotonic_ns(), time.time_ns()
            from grasp_project.recording.ros_capture import record_ros_image

            record_ros_image(self.writer, message, stream=stream,
                             host_arrival_monotonic_ns=monotonic_ns, host_arrival_epoch_ns=epoch_ns)
            self.counts[stream] += 1

        return callback

    def _camera_info_callback(self, message):
        monotonic_ns, epoch_ns = time.monotonic_ns(), time.time_ns()
        from grasp_project.recording.ros_capture import record_ros_camera_info

        record_ros_camera_info(self.writer, message,
                               host_arrival_monotonic_ns=monotonic_ns, host_arrival_epoch_ns=epoch_ns)
        self.counts["camera_info"] += 1

    def poll(self):
        if not self.rclpy.ok():
            raise RuntimeError("ROS context stopped before capture duration elapsed")
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        try:
            if self.node is not None:
                self.node.destroy_node()
        finally:
            if self.rclpy.ok():
                self.rclpy.shutdown()


def run_capture(args):
    """Capture only when explicitly invoked; errors/interrupts never finalize."""
    from grasp_project.recording import RecordingWriter

    if not args.session_id.strip() or not args.object_id.strip():
        raise ValueError("session_id and object_id must be nonempty")
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0:
        raise ValueError("duration_sec must be positive and finite")
    if not math.isfinite(args.poll_interval_sec) or not 0 < args.poll_interval_sec <= 1:
        raise ValueError("poll_interval_sec must be positive, finite, and <= 1 second")
    configuration = _configuration()
    configuration["capture"] = {
        "duration_sec": args.duration_sec, "poll_interval_sec": args.poll_interval_sec,
        "with_ros": args.with_ros,
        "topics": {"rgb": args.rgb_topic, "depth": args.depth_topic, "camera_info": args.camera_info_topic},
    }
    writer = RecordingWriter(
        args.destination, recording_id=args.recording_id, config=configuration, revision=_revision(),
        metadata={"trial_id": args.recording_id, "session_id": args.session_id, "object_id": args.object_id,
                  "data_origin": "recorded", "ground_truth_status": "not_labelled",
                  "capture_scope": "native_DAVIS_and_optional_existing_ROS_topics_no_tracker_replay_claim"},
    )
    monitor, ros = None, None
    try:
        if args.with_ros:
            ros = _RosSubscriptions(writer, args)
        monitor = _create_monitor(writer, args.destination)
        if monitor.motion_estimator.config != configuration["davis_motion"]:
            raise ValueError("DAVIS motion configuration changed before backend start")
        monitor.start()
        if not monitor.active:
            raise RuntimeError(monitor.device_open_error or monitor.unavailable_reason or "DAVIS backend failed to start")
        print("DAVIS capture active. Missing events remain unavailable evidence; Ctrl+C leaves an incomplete archive.", flush=True)
        deadline = time.monotonic() + args.duration_sec
        while time.monotonic() < deadline:
            monitor.poll_event_and_frame_arrays()
            if not monitor.active:
                raise RuntimeError(monitor.unavailable_reason or "DAVIS polling stopped")
            if ros is not None:
                ros.poll()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.poll_interval_sec, remaining))
        if ros is not None and any(count == 0 for count in ros.counts.values()):
            raise RuntimeError(f"requested ROS streams did not all arrive: {ros.counts}")
    finally:
        try:
            if monitor is not None:
                monitor.stop()
        finally:
            if ros is not None:
                ros.close()
    return writer.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    print("This user-run command opens DAVIS; optional ROS mode only subscribes to an existing D405 producer.", flush=True)
    try:
        manifest = run_capture(args)
    except KeyboardInterrupt:
        print("Interrupted: archive remains incomplete; no physical performance result was produced.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"Capture failed; any created archive remains incomplete: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "manifest": str(manifest), "physical_validation": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Standalone RealSense ROS image transport health check.

This intentionally checks only ROS transport and message readability. It does
not import or run any grasp planning code.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"
OUTPUT_PATH = Path("output/realsense_camera_health.json")
MAX_LAST_MESSAGE_AGE_SEC = 0.5


def write_report(path, report):
    """Atomically publish the one canonical transport-health artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(output)


def device_unavailable_report():
    """Return an honest zero-frame result when discovery found no D405."""
    return {
        "timestamp": time.time(),
        "status": "DEVICE_NOT_CONNECTED",
        "pass": False,
        "device_present": False,
        "color_topic": COLOR_TOPIC,
        "depth_topic": DEPTH_TOPIC,
        "camera_info_topic": INFO_TOPIC,
        "color_messages": 0,
        "depth_messages": 0,
        "color_fps": 0.0,
        "depth_fps": 0.0,
        "color_decode_ok": False,
        "depth_decode_ok": False,
        "camera_info_ok": False,
        "health_pass_source": "hardware_discovery",
        "last_error": "No selectable RealSense camera is currently enumerated.",
        "suggested_fix": "Reconnect the D405, confirm USB enumeration, then retry Start RealSense.",
    }


def run_cmd(cmd, timeout=5.0):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "cmd": " ".join(cmd),
            "returncode": 999,
            "stdout": "",
            "stderr": str(exc),
        }


def parse_topic_hz(stdout):
    matches = re.findall(r"average rate:\s*([0-9.]+)", stdout or "")
    if not matches:
        return 0.0
    try:
        return float(matches[-1])
    except ValueError:
        return 0.0


def topic_hz_report(topic, timeout_sec=8.0):
    result = run_cmd(["timeout", str(int(timeout_sec)), "ros2", "topic", "hz", topic], timeout=timeout_sec + 2.0)
    result["average_rate"] = parse_topic_hz(result.get("stdout", "") + "\n" + result.get("stderr", ""))
    return result


def expected_image_bytes(msg):
    if not msg:
        return None
    return int(msg.height) * int(msg.step)


def check_with_rclpy(timeout_sec, min_fps):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    result = {
        "color_topic": COLOR_TOPIC,
        "depth_topic": DEPTH_TOPIC,
        "camera_info_topic": INFO_TOPIC,
        "color_messages": 0,
        "depth_messages": 0,
        "color_fps": 0.0,
        "depth_fps": 0.0,
        "color_decode_ok": False,
        "depth_decode_ok": False,
        "camera_info_ok": False,
        "last_error": None,
        "color": {},
        "depth": {},
    }
    color_times = []
    depth_times = []

    def image_summary(msg):
        expected = expected_image_bytes(msg)
        actual = len(msg.data)
        return {
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "step": int(msg.step),
            "expected_bytes": expected,
            "actual_bytes": actual,
            "bytes_ok": bool(expected == actual),
        }

    def color_cb(msg):
        now = time.perf_counter()
        color_times.append(now)
        result["color_messages"] += 1
        result["color"] = image_summary(msg)
        result["color_decode_ok"] = bool(result["color"]["bytes_ok"] and msg.encoding in {"rgb8", "bgr8"})

    def depth_cb(msg):
        now = time.perf_counter()
        depth_times.append(now)
        result["depth_messages"] += 1
        result["depth"] = image_summary(msg)
        result["depth_decode_ok"] = bool(result["depth"]["bytes_ok"] and msg.encoding in {"16UC1", "mono16"})

    def info_cb(msg):
        result["camera_info_ok"] = bool(msg.width > 0 and msg.height > 0 and len(msg.k) == 9)
        result["camera_info"] = {
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": str(msg.distortion_model),
        }

    rclpy.init(args=None)
    node = rclpy.create_node("realsense_transport_health")
    node.create_subscription(Image, COLOR_TOPIC, color_cb, qos_profile_sensor_data)
    node.create_subscription(Image, DEPTH_TOPIC, depth_cb, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, INFO_TOPIC, info_cb, qos_profile_sensor_data)
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
    except Exception as exc:
        result["last_error"] = str(exc)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    elapsed = max(time.perf_counter() - start, 1e-6)
    finished = time.perf_counter()
    color_last_message_age_sec = (
        None if not color_times else max(0.0, finished - color_times[-1])
    )
    depth_last_message_age_sec = (
        None if not depth_times else max(0.0, finished - depth_times[-1])
    )
    result["elapsed_sec"] = round(elapsed, 3)
    result["color_fps"] = round(len(color_times) / elapsed, 2)
    result["depth_fps"] = round(len(depth_times) / elapsed, 2)
    result["color_last_message_age_sec"] = (
        None if color_last_message_age_sec is None else round(color_last_message_age_sec, 3)
    )
    result["depth_last_message_age_sec"] = (
        None if depth_last_message_age_sec is None else round(depth_last_message_age_sec, 3)
    )
    result["max_last_message_age_sec"] = MAX_LAST_MESSAGE_AGE_SEC
    result["pass"] = bool(
        result["color_decode_ok"]
        and result["depth_decode_ok"]
        and result["camera_info_ok"]
        and result["color_fps"] >= min_fps
        and result["depth_fps"] >= min_fps
        and color_last_message_age_sec is not None
        and depth_last_message_age_sec is not None
        and color_last_message_age_sec <= MAX_LAST_MESSAGE_AGE_SEC
        and depth_last_message_age_sec <= MAX_LAST_MESSAGE_AGE_SEC
    )
    if not result["pass"] and not result["last_error"]:
        reasons = []
        if not result["color_decode_ok"]:
            reasons.append("color image unreadable or wrong encoding")
        if not result["depth_decode_ok"]:
            reasons.append("aligned depth image unreadable or wrong encoding")
        if not result["camera_info_ok"]:
            reasons.append("camera_info missing")
        if result["color_fps"] < min_fps:
            reasons.append(f"color FPS below {min_fps}")
        if result["depth_fps"] < min_fps:
            reasons.append(f"depth FPS below {min_fps}")
        if color_last_message_age_sec is None or color_last_message_age_sec > MAX_LAST_MESSAGE_AGE_SEC:
            reasons.append("color stream stopped before health window ended")
        if depth_last_message_age_sec is None or depth_last_message_age_sec > MAX_LAST_MESSAGE_AGE_SEC:
            reasons.append("depth stream stopped before health window ended")
        result["last_error"] = "; ".join(reasons)
    return result


def main():
    parser = argparse.ArgumentParser(description="Check RealSense ROS image transport health.")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--min-fps", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--device-unavailable",
        action="store_true",
        help="Publish DEVICE_NOT_CONNECTED without probing ROS topics.",
    )
    args = parser.parse_args()

    if args.device_unavailable:
        report = device_unavailable_report()
        write_report(args.output, report)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print("RealSense ROS transport health: DEVICE_NOT_CONNECTED")
            print(f"  last error: {report['last_error']}")
            print(f"  suggested fix: {report['suggested_fix']}")
        return 2

    report = {
        "timestamp": time.time(),
        "ros_topics": run_cmd(["ros2", "topic", "list"], timeout=4.0),
        "color_topic_info": run_cmd(["ros2", "topic", "info", COLOR_TOPIC], timeout=4.0),
        "depth_topic_info": run_cmd(["ros2", "topic", "info", DEPTH_TOPIC], timeout=4.0),
        "suggested_fix": "restart camera driver / lower resolution / restart ROS daemon",
    }
    try:
        report.update(check_with_rclpy(args.timeout, args.min_fps))
    except Exception as exc:
        report.update({
            "pass": False,
            "last_error": str(exc),
            "color_fps": 0.0,
            "depth_fps": 0.0,
        })

    if not report.get("pass"):
        color_hz = topic_hz_report(COLOR_TOPIC, timeout_sec=max(5.0, min(args.timeout, 8.0)))
        depth_hz = topic_hz_report(DEPTH_TOPIC, timeout_sec=max(5.0, min(args.timeout, 8.0)))
        report["color_topic_hz_cli"] = color_hz
        report["depth_topic_hz_cli"] = depth_hz
        cli_color_ok = float(color_hz.get("average_rate", 0.0) or 0.0) >= args.min_fps
        cli_depth_ok = float(depth_hz.get("average_rate", 0.0) or 0.0) >= args.min_fps
        terminal_frames_current = bool(
            report.get("color_last_message_age_sec") is not None
            and report.get("depth_last_message_age_sec") is not None
            and float(report["color_last_message_age_sec"]) <= MAX_LAST_MESSAGE_AGE_SEC
            and float(report["depth_last_message_age_sec"]) <= MAX_LAST_MESSAGE_AGE_SEC
        )
        if cli_color_ok and cli_depth_ok and report.get("color_decode_ok") and report.get("depth_decode_ok") and report.get("camera_info_ok") and terminal_frames_current:
            report["pass"] = True
            report["health_pass_source"] = "ros2_topic_hz_cli_fallback"
            report["last_error"] = None
        else:
            report["health_pass_source"] = "rclpy_subscriber"
    else:
        report["health_pass_source"] = "rclpy_subscriber"

    write_report(args.output, report)

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        status = "PASS" if report.get("pass") else "FAIL"
        print(f"RealSense ROS transport health: {status}")
        print(f"  color: {report.get('color_fps', 0.0)} FPS {report.get('color', {})}")
        print(f"  depth: {report.get('depth_fps', 0.0)} FPS {report.get('depth', {})}")
        print(f"  camera_info_ok: {report.get('camera_info_ok')}")
        if not report.get("pass"):
            print("  CAMERA STREAM ERROR: RealSense ROS topic unreadable")
            print(f"  last error: {report.get('last_error')}")
            print(f"  suggested fix: {report.get('suggested_fix')}")
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    sys.exit(main())

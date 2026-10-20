import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
import os
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.orientation import (  # noqa: E402
    ORIENTATION_LOG,
    rotate_image_if_needed,
    rotate_intrinsics_if_needed,
    rotation_enabled,
    write_orientation_metadata,
)


class RGBDepthSaver(Node):
    def __init__(self):
        super().__init__("rgb_depth_saver")

        os.makedirs("output", exist_ok=True)
        self.apply_camera_rotation = rotation_enabled(
            os.environ.get("APPLY_CAMERA_ROTATION", "0")
        )

        self.color_saved = False
        self.depth_saved = False
        self.info_saved = False
        self.done = False

        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_callback,
            10
        )

        # Prefer depth aligned to the color frame so RGB overlays and pixel contacts
        # share one coordinate system. Keep the older raw topic as a fallback.
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10
        )

        self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
            self.depth_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.info_callback,
            10
        )

    def color_callback(self, msg):
        if self.color_saved:
            return

        arr = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.encoding == "rgb8":
            rgb = arr.reshape((msg.height, msg.width, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "bgr8":
            bgr = arr.reshape((msg.height, msg.width, 3))
        else:
            self.get_logger().error(f"Unsupported RGB encoding: {msg.encoding}")
            return

        bgr = rotate_image_if_needed(bgr, self.apply_camera_rotation)
        cv2.imwrite("output/captured_frame.png", bgr)

        self.color_saved = True
        self.get_logger().info("Saved RGB")
        self.check_done()

    def depth_callback(self, msg):
        if self.depth_saved:
            return

        if msg.encoding in ["16UC1", "mono16"]:
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                (msg.height, msg.width)
            )
        elif msg.encoding == "32FC1":
            depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width)
            )
            depth = (depth_m * 1000).astype(np.uint16)
        else:
            self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}")
            return

        depth = rotate_image_if_needed(depth, self.apply_camera_rotation)
        cv2.imwrite("output/depth_frame.png", depth)
        np.save("output/depth_frame.npy", depth)

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("output/depth_visual.png", depth_vis)

        self.depth_saved = True
        self.get_logger().info("Saved depth")
        self.check_done()

    def info_callback(self, msg):
        if self.info_saved:
            return

        intrinsics = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5],
            "width": msg.width,
            "height": msg.height
        }
        intrinsics = rotate_intrinsics_if_needed(intrinsics, self.apply_camera_rotation)

        with open("output/camera_intrinsics.json", "w") as f:
            json.dump(intrinsics, f, indent=4)

        write_orientation_metadata("output", applied=self.apply_camera_rotation)
        self.get_logger().info(ORIENTATION_LOG)

        self.info_saved = True
        self.get_logger().info("Saved camera intrinsics")
        self.check_done()

    def check_done(self):
        if self.color_saved and self.depth_saved and self.info_saved:
            self.get_logger().info("Saved RGB + depth + intrinsics")
            self.done = True


def main():
    rclpy.init()
    node = RGBDepthSaver()

    timeout_seconds = 10
    start_time = time.time()

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

            if time.time() - start_time > timeout_seconds:
                node.get_logger().error("Timeout: failed to capture RGB, depth, and intrinsics")
                break

    except KeyboardInterrupt:
        pass

    success = node.done

    node.destroy_node()
    rclpy.shutdown()

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

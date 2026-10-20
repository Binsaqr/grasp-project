import numpy as np


WORKSPACE_CONFIG = {
    "center_deadband_ratio": 0.015,
    "side_reach_ratio": 0.36,
    "vertical_min_ratio": 0.14,
    "vertical_max_ratio": 0.94,
    "score_tolerance_ratio": 0.08,
}

WORKSPACE_CONFIG_SOURCE = "default"


def calibrated_gripper_zones(image_shape, config=None):
    config = config or WORKSPACE_CONFIG
    h, w = image_shape[:2]
    center_x = w / 2.0
    deadband = w * config["center_deadband_ratio"]
    reach = w * config["side_reach_ratio"]
    y_min = h * config["vertical_min_ratio"]
    y_max = h * config["vertical_max_ratio"]
    return {
        "frame_size": {"width": int(w), "height": int(h)},
        "image_center_x": float(center_x),
        "left": {
            "x_min": float(max(0.0, center_x - reach)),
            "x_max": float(max(0.0, center_x - deadband)),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "right": {
            "x_min": float(min(float(w), center_x + deadband)),
            "x_max": float(min(float(w), center_x + reach)),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "notes": "Calibrated fixed gripper reach zones in raw camera coordinates.",
    }


def point_distance_outside_zone(point_xy, zone):
    x, y = np.asarray(point_xy, dtype=float)
    dx = max(zone["x_min"] - x, 0.0, x - zone["x_max"])
    dy = max(zone["y_min"] - y, 0.0, y - zone["y_max"])
    return float(np.hypot(dx, dy))


def score_point_in_zone(point_xy, zone, image_shape):
    h, w = image_shape[:2]
    tolerance = max(12.0, w * WORKSPACE_CONFIG["score_tolerance_ratio"])
    distance = point_distance_outside_zone(point_xy, zone)
    score = float(np.clip(100.0 * (1.0 - distance / tolerance), 0.0, 100.0))
    return score, bool(distance <= 1.0)


def evaluate_workspace_reachability(left_contact, right_contact, image_shape):
    zones = calibrated_gripper_zones(image_shape)
    left_score, left_reachable = score_point_in_zone(left_contact, zones["left"], image_shape)
    right_score, right_reachable = score_point_in_zone(right_contact, zones["right"], image_shape)
    workspace_score = float(min(left_score, right_score))
    return {
        "zones": zones,
        "left_reachable": left_reachable,
        "right_reachable": right_reachable,
        "left_score": left_score,
        "right_score": right_score,
        "workspace_score": workspace_score,
        "both_reachable": bool(left_reachable and right_reachable),
    }

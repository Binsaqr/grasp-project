import numpy as np


IDEAL_ZONE_CONFIG = {
    "center_x_ratio": 0.50,
    "center_y_ratio": 0.60,
    "center_tolerance_x_ratio": 0.06,
    "center_tolerance_y_ratio": 0.08,
    "grasp_y_ratio": 0.60,
    "grasp_tolerance_y_ratio": 0.08,
    "depth_min_mm": 90.0,
    "depth_max_mm": 360.0,
    "depth_ideal_mm": 220.0,
    "minimum_body_visibility": 0.035,
}


def ideal_pregrasp_zone(image_shape, config=None):
    config = config or IDEAL_ZONE_CONFIG
    h, w = image_shape[:2]
    center = [float(w * config["center_x_ratio"]), float(h * config["center_y_ratio"])]
    center_tol_x = float(w * config["center_tolerance_x_ratio"])
    center_tol_y = float(h * config["center_tolerance_y_ratio"])
    grasp_y = float(h * config["grasp_y_ratio"])
    grasp_tol_y = float(h * config["grasp_tolerance_y_ratio"])
    return {
        "image_size": {"width": int(w), "height": int(h)},
        "gripper_centerline_x": float(w / 2.0),
        "ideal_center_px": center,
        "ideal_center_region": {
            "x_min": center[0] - center_tol_x,
            "x_max": center[0] + center_tol_x,
            "y_min": center[1] - center_tol_y,
            "y_max": center[1] + center_tol_y,
        },
        "ideal_grasp_height_region": {
            "y_min": grasp_y - grasp_tol_y,
            "y_max": grasp_y + grasp_tol_y,
            "center_y": grasp_y,
        },
        "acceptable_depth_range_mm": [config["depth_min_mm"], config["depth_max_mm"]],
        "ideal_depth_mm": config["depth_ideal_mm"],
        "minimum_body_visibility": config["minimum_body_visibility"],
    }


def _inside(value, low, high):
    return low <= value <= high


def _dominant_single_action(dx, dy, depth_error, tolerances):
    candidates = []
    if abs(dx) > tolerances["x"]:
        candidates.append((abs(dx) / max(tolerances["x"], 1.0), "MOVE LEFT" if dx < 0 else "MOVE RIGHT"))
    if abs(dy) > tolerances["y"]:
        candidates.append((abs(dy) / max(tolerances["y"], 1.0), "MOVE UP" if dy < 0 else "MOVE DOWN"))
    if depth_error is not None and abs(depth_error) > tolerances["depth"]:
        candidates.append((abs(depth_error) / max(tolerances["depth"], 1.0), "MOVE CLOSER" if depth_error > 0 else "MOVE BACK"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def recommend_pregrasp_action(pose, plan, safety, image_shape):
    zone = ideal_pregrasp_zone(image_shape)
    if pose is None or not pose.get("detected"):
        return {
            "state": "SEARCHING",
            "recommended_action": "SEARCH",
            "reason": "Bottle is not detected in the current RGB-D frame.",
            "ideal_zone": zone,
            "current_bottle_center_px": None,
            "ideal_center_px": zone["ideal_center_px"],
            "dx_error_px": None,
            "dy_error_px": None,
            "depth_error_mm": None,
            "action_dx_px": None,
            "action_dy_px": None,
            "action_depth_mm": None,
        }

    center = np.asarray(pose["center_px"], dtype=float)
    ideal_center = np.asarray(zone["ideal_center_px"], dtype=float)
    dx = float(center[0] - ideal_center[0])
    dy = float(center[1] - ideal_center[1])
    action_dx = float(ideal_center[0] - center[0])
    action_dy = float(ideal_center[1] - center[1])

    current_depth = pose.get("depth_median_mm")
    depth_error = None
    action_depth = None
    depth_ok = False
    if current_depth is not None:
        depth_error = float(current_depth - zone["ideal_depth_mm"])
        action_depth = float(zone["ideal_depth_mm"] - current_depth)
        depth_ok = _inside(
            float(current_depth),
            zone["acceptable_depth_range_mm"][0],
            zone["acceptable_depth_range_mm"][1],
        )

    grasp_center = np.asarray(plan.get("grasp_center_px", center) if plan else center, dtype=float)
    grasp_y_error = float(grasp_center[1] - zone["ideal_grasp_height_region"]["center_y"])
    body_visibility = float(pose.get("area_ratio", 0.0) or 0.0)
    body_visible = body_visibility >= zone["minimum_body_visibility"]
    center_region = zone["ideal_center_region"]
    grasp_region = zone["ideal_grasp_height_region"]
    centered_x = _inside(center[0], center_region["x_min"], center_region["x_max"])
    centered_y = _inside(center[1], center_region["y_min"], center_region["y_max"])
    grasp_y_ok = _inside(grasp_center[1], grasp_region["y_min"], grasp_region["y_max"])
    width_feasible = bool(plan and plan.get("width_reachable"))
    contacts_in_preferred_zone = bool(plan and plan.get("left_reachable") and plan.get("right_reachable"))
    safe = safety.get("decision") in {"READY TO CLOSE", "SAFE TO GRIP/LIFT"} and not safety.get("failure_reasons")

    corrections = []
    if not centered_x:
        corrections.append("horizontal alignment")
    if not centered_y or not grasp_y_ok:
        corrections.append("vertical grasp height")
    if not depth_ok:
        corrections.append("depth")
    if not width_feasible:
        corrections.append("physical grip width")
    if not contacts_in_preferred_zone:
        corrections.append("positioning guide")
    if not body_visible:
        corrections.append("bottle body visibility")

    if safety.get("failure_reasons"):
        state = "UNSAFE"
        action = "DO NOT CLOSE"
        reason = "; ".join(safety.get("failure_reasons", []))
    elif safe and not corrections:
        state = "READY_TO_CLOSE"
        action = "READY TO CLOSE"
        reason = "Bottle is centered, reachable, depth is acceptable, and planned grasp is safe."
    else:
        state = "POSITIONING"
        if len(corrections) > 1:
            action = "ADJUST POSITION"
        else:
            action = _dominant_single_action(
                dx,
                dy if not grasp_y_ok else 0.0,
                depth_error if not depth_ok else 0.0,
                {
                    "x": max(1.0, center_region["x_max"] - ideal_center[0]),
                    "y": max(1.0, center_region["y_max"] - ideal_center[1]),
                    "depth": 45.0,
                },
            ) or "ADJUST POSITION"
        reason = "Correct " + ", ".join(corrections) + " before closing the grippers."

    return {
        "state": state,
        "recommended_action": action,
        "reason": reason,
        "ideal_zone": zone,
        "current_bottle_center_px": [float(center[0]), float(center[1])],
        "ideal_center_px": [float(ideal_center[0]), float(ideal_center[1])],
        "current_grasp_center_px": [float(grasp_center[0]), float(grasp_center[1])],
        "dx_error_px": dx,
        "dy_error_px": dy,
        "grasp_y_error_px": grasp_y_error,
        "depth_error_mm": depth_error,
        "action_dx_px": action_dx,
        "action_dy_px": action_dy,
        "action_depth_mm": action_depth,
        "horizontal_aligned": bool(centered_x),
        "vertical_aligned": bool(centered_y and grasp_y_ok),
        "depth_in_range": bool(depth_ok),
        "grip_width_feasible": width_feasible,
        "left_right_reachable": contacts_in_preferred_zone,
        "left_right_reachable_meaning": "preferred image positioning guide only; not a hard physical reach gate",
        "bottle_body_visible": bool(body_visible),
        "body_visibility_ratio": body_visibility,
    }

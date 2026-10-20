import cv2
import numpy as np

from depth_utils import pixel_to_3d, valid_depth_mask


CONTACT_MODEL_TYPE = "pure_visual_patch_model"
NO_FORCE_SENSOR_NOTE = (
    "Contact is estimated visually from RGB-D geometry and patch evidence. "
    "No force or pressure is directly measured."
)


DEFAULT_CONFIG = {
    "patch_height_mm": 50.0,
    "patch_width_mm": 15.0,
    "patch_height_px_fallback": 56.0,
    "patch_width_px_fallback": 18.0,
    "patch_height_px_range": (24.0, 130.0),
    "patch_width_px_range": (8.0, 45.0),
    "body_fraction_min": 0.22,
    "body_fraction_max": 0.78,
    "min_mask_overlap": 0.20,
    "min_depth_valid_ratio": 0.20,
    "max_depth_std_mm": 90.0,
    "min_patch_quality": 55.0,
    "min_pair_quality": 60.0,
    "min_grip_width_mm": 30.0,
    "max_grip_width_mm": 130.0,
}


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def metric_length_to_px(length_mm, depth_mm, focal_px, fallback_px, px_range):
    if depth_mm is not None and focal_px is not None and float(focal_px) > 1e-6:
        depth_m = max(float(depth_mm) / 1000.0, 1e-3)
        px = float(focal_px) * (float(length_mm) / 1000.0) / depth_m
    else:
        px = float(fallback_px)
    lo, hi = px_range
    return float(np.clip(px, lo, hi))


def oriented_box_points(center, length_direction, width_direction, length_px, width_px):
    center = np.asarray(center, dtype=float)
    length_direction = unit(length_direction)
    width_direction = unit(width_direction)
    half_l = float(length_px) / 2.0
    half_w = float(width_px) / 2.0
    return np.array(
        [
            center - length_direction * half_l - width_direction * half_w,
            center + length_direction * half_l - width_direction * half_w,
            center + length_direction * half_l + width_direction * half_w,
            center - length_direction * half_l + width_direction * half_w,
        ],
        dtype=float,
    )


def polygon_mask(shape, polygon):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polygon is None:
        return mask
    points = np.round(np.asarray(polygon, dtype=float)).astype(np.int32)
    if points.size:
        cv2.fillPoly(mask, [points], 1)
    return mask


def polygon_bbox_xywh(polygon, image_shape):
    points = np.asarray(polygon, dtype=float)
    h, w = image_shape[:2]
    if points.size == 0:
        return [0, 0, 0, 0]
    x1 = int(np.clip(np.floor(np.min(points[:, 0])), 0, max(w - 1, 0)))
    y1 = int(np.clip(np.floor(np.min(points[:, 1])), 0, max(h - 1, 0)))
    x2 = int(np.clip(np.ceil(np.max(points[:, 0])), 0, w))
    y2 = int(np.clip(np.ceil(np.max(points[:, 1])), 0, h))
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def polygon_inside_frame_ratio(polygon, image_shape):
    points = np.asarray(polygon, dtype=float)
    if points.size == 0:
        return 0.0
    h, w = image_shape[:2]
    inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    return float(np.count_nonzero(inside) / max(points.shape[0], 1))


def mask_overlap_score(mask, patch_mask):
    patch_pixels = int(np.count_nonzero(patch_mask))
    if mask is None or patch_pixels == 0:
        return 0.0
    return float(np.count_nonzero((mask > 0) & (patch_mask > 0)) / patch_pixels)


def depth_stats_for_patch(depth, patch_mask):
    patch_pixels = int(np.count_nonzero(patch_mask))
    if depth is None or patch_pixels == 0:
        return {"depth_median_mm": None, "depth_valid_ratio": 0.0, "depth_std_mm": None}
    valid = (patch_mask > 0) & valid_depth_mask(depth)
    values = depth[valid]
    if values.size == 0:
        return {"depth_median_mm": None, "depth_valid_ratio": 0.0, "depth_std_mm": None}
    return {
        "depth_median_mm": float(np.median(values)),
        "depth_valid_ratio": float(values.size / max(patch_pixels, 1)),
        "depth_std_mm": float(np.std(values)),
    }


def score_from_range(value, good, bad):
    if value is None:
        return 0.0
    value = abs(float(value))
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return float(np.clip(100.0 * (1.0 - (value - good) / max(bad - good, 1e-6)), 0.0, 100.0))


def score_body_fraction(fraction, config):
    lo = float(config["body_fraction_min"])
    hi = float(config["body_fraction_max"])
    if fraction is None:
        return 0.0
    fraction = float(fraction)
    if lo <= fraction <= hi:
        center = (lo + hi) / 2.0
        return score_from_range(fraction - center, 0.12, max(center - lo, hi - center))
    return 0.0


def patch_from_center(side, center_px, pose, depth, intrinsics, grasp_direction, body_fraction, config, image_shape):
    center = np.asarray(center_px, dtype=float)
    axis = unit(np.asarray(pose.get("axis_vector", [0.0, 1.0]), dtype=float))
    bottle_center = np.asarray(pose.get("center_px", center), dtype=float)
    depth_mm = None
    if depth is not None:
        cx, cy = int(round(center[0])), int(round(center[1]))
        if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] and depth[cy, cx] > 0:
            depth_mm = float(depth[cy, cx])
    if depth_mm is None:
        depth_mm = pose.get("depth_median_mm")
    fx = (intrinsics or {}).get("fx")
    fy = (intrinsics or {}).get("fy")
    patch_height_px = metric_length_to_px(
        config["patch_height_mm"], depth_mm, fy, config["patch_height_px_fallback"], config["patch_height_px_range"]
    )
    patch_width_px = metric_length_to_px(
        config["patch_width_mm"], depth_mm, fx, config["patch_width_px_fallback"], config["patch_width_px_range"]
    )
    polygon = oriented_box_points(center, axis, grasp_direction, patch_height_px, patch_width_px)
    body_mask = pose.get("body_mask") if pose.get("body_mask") is not None else pose.get("cleaned_mask")
    cleaned_mask = pose.get("cleaned_mask", body_mask)
    patch_mask = polygon_mask(image_shape, polygon)
    mask_overlap = mask_overlap_score(body_mask, patch_mask)
    cleaned_overlap = mask_overlap_score(cleaned_mask, patch_mask)
    depth_stats = depth_stats_for_patch(depth, patch_mask)
    inside_ratio = polygon_inside_frame_ratio(polygon, image_shape)
    eroded_overlap = 0.0
    if body_mask is not None:
        kernel = np.ones((5, 5), dtype=np.uint8)
        eroded_overlap = mask_overlap_score(cv2.erode((body_mask > 0).astype(np.uint8), kernel, iterations=1), patch_mask)
    inward_normal = unit(bottle_center - center)
    expected = unit(bottle_center - center)
    normal_score = float(np.clip(max(float(np.dot(inward_normal, expected)), 0.0) * 100.0, 0.0, 100.0))
    body_region_score = score_body_fraction(body_fraction, config)
    visibility_score = float(np.clip(inside_ratio * 100.0, 0.0, 100.0))
    mask_score = float(np.clip(min(mask_overlap, cleaned_overlap) * 100.0 / max(config["min_mask_overlap"], 1e-6), 0.0, 100.0))
    valid_ratio = depth_stats["depth_valid_ratio"]
    std_mm = depth_stats["depth_std_mm"]
    depth_score = float(np.clip(0.62 * min(valid_ratio / max(config["min_depth_valid_ratio"], 1e-6), 1.0) * 100.0
                                + 0.38 * score_from_range(std_mm, 25.0, config["max_depth_std_mm"]), 0.0, 100.0))
    edge_margin_score = float(np.clip(eroded_overlap * 100.0 / max(config["min_mask_overlap"], 1e-6), 0.0, 100.0))
    stability_score = 100.0
    quality = float(np.clip(
        0.18 * body_region_score
        + 0.16 * visibility_score
        + 0.22 * mask_score
        + 0.22 * depth_score
        + 0.10 * normal_score
        + 0.07 * edge_margin_score
        + 0.05 * stability_score,
        0.0,
        100.0,
    ))
    visible = bool(inside_ratio >= 0.50 and mask_overlap >= config["min_mask_overlap"] and valid_ratio >= config["min_depth_valid_ratio"])
    on_body = bool(body_region_score > 0.0 and mask_overlap >= config["min_mask_overlap"])
    return {
        "side": side,
        "patch_center_px": [float(center[0]), float(center[1])],
        "center_px": [float(center[0]), float(center[1])],
        "patch_bbox_px": polygon_bbox_xywh(polygon, image_shape),
        "patch_polygon_px": polygon.tolist(),
        "polygon": polygon.tolist(),
        "patch_height_mm": float(config["patch_height_mm"]),
        "patch_height_px": float(patch_height_px),
        "patch_width_mm": float(config["patch_width_mm"]),
        "patch_width_px": float(patch_width_px),
        "body_fraction": None if body_fraction is None else float(body_fraction),
        "visible": visible,
        "on_bottle_body": on_body,
        "depth_median_mm": depth_stats["depth_median_mm"],
        "depth_valid_ratio": depth_stats["depth_valid_ratio"],
        "depth_std_mm": depth_stats["depth_std_mm"],
        "normal_vector": inward_normal.tolist(),
        "inward_normal": inward_normal.tolist(),
        "body_region_score": body_region_score,
        "visibility_score": visibility_score,
        "mask_overlap_score": mask_score,
        "mask_overlap_ratio": mask_overlap,
        "depth_score": depth_score,
        "normal_score": normal_score,
        "edge_margin_score": edge_margin_score,
        "stability_score": stability_score,
        "patch_quality_score": quality,
        "visual_patch_quality_score": quality,
    }


def grip_width_mm_from_patches(left_patch, right_patch, intrinsics):
    left_depth = left_patch.get("depth_median_mm")
    right_depth = right_patch.get("depth_median_mm")
    fx = (intrinsics or {}).get("fx")
    if left_depth is None or right_depth is None or fx is None or float(fx) <= 1e-6:
        return None
    left = np.asarray(left_patch["patch_center_px"], dtype=float)
    right = np.asarray(right_patch["patch_center_px"], dtype=float)
    z = (float(left_depth) + float(right_depth)) / 2.0
    return float(np.linalg.norm(right - left) * z / float(fx))


def compute_visual_contact_model(pose, depth, intrinsics=None, plan=None, locked_target=None, image_shape=None, epsilon_result=None, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    image_shape = image_shape or (depth.shape if depth is not None else pose.get("cleaned_mask").shape)
    reasons = []
    warnings = [NO_FORCE_SENSOR_NOTE]
    if pose is None or pose.get("cleaned_mask") is None:
        return {
            "visual_contact_model_enabled": True,
            "contact_model_type": CONTACT_MODEL_TYPE,
            "no_force_sensor": True,
            "visual_contact_ready": False,
            "visual_contact_reasons": ["Bottle mask unavailable."],
            "visual_contact_warnings": warnings,
        }
    plan = plan or {}
    target_clamp = (locked_target or {}).get("target_clamp") or (locked_target or {}).get("fixed_clamp_zone") or {}
    left_center = plan.get("left_contact_px") or ((target_clamp.get("left_clamp_patch") or {}).get("center_px"))
    right_center = plan.get("right_contact_px") or ((target_clamp.get("right_clamp_patch") or {}).get("center_px"))
    if left_center is None or right_center is None:
        return {
            "visual_contact_model_enabled": True,
            "contact_model_type": CONTACT_MODEL_TYPE,
            "no_force_sensor": True,
            "visual_contact_ready": False,
            "visual_contact_reasons": ["Left/right clamp patch centers unavailable."],
            "visual_contact_warnings": warnings,
        }
    left = np.asarray(left_center, dtype=float)
    right = np.asarray(right_center, dtype=float)
    grasp_direction = unit(np.asarray(plan.get("grasp_direction", right - left), dtype=float))
    if np.linalg.norm(grasp_direction) < 1e-9:
        grasp_direction = unit(right - left)
    body_fraction = plan.get("body_fraction", ((locked_target or {}).get("locked_body_fraction")))
    left_patch = patch_from_center("left", left, pose, depth, intrinsics, grasp_direction, body_fraction, config, image_shape)
    right_patch = patch_from_center("right", right, pose, depth, intrinsics, grasp_direction, body_fraction, config, image_shape)
    pair_center = ((left + right) / 2.0).tolist()
    grip_width_px = float(np.linalg.norm(right - left))
    grip_width_mm = plan.get("grip_width_mm")
    if grip_width_mm is None:
        grip_width_mm = grip_width_mm_from_patches(left_patch, right_patch, intrinsics)
    if grip_width_mm is not None:
        grip_width_valid = bool(config["min_grip_width_mm"] <= float(grip_width_mm) <= config["max_grip_width_mm"])
    else:
        grip_width_valid = bool(plan.get("grip_width_in_range", False))
    depth_delta = None
    if left_patch.get("depth_median_mm") is not None and right_patch.get("depth_median_mm") is not None:
        depth_delta = abs(float(left_patch["depth_median_mm"]) - float(right_patch["depth_median_mm"]))
    depth_consistency_score = score_from_range(depth_delta, 18.0, 95.0)
    bottle_center = np.asarray(pose.get("center_px", pair_center), dtype=float)
    balance_error = float(np.linalg.norm(np.asarray(pair_center, dtype=float) - bottle_center))
    balance_score = score_from_range(balance_error, max(grip_width_px * 0.10, 8.0), max(grip_width_px * 0.40, 25.0))
    fraction_similarity_score = 100.0
    width_score = 100.0 if grip_width_valid else 0.0
    epsilon_ready = None
    if epsilon_result:
        epsilon_ready = epsilon_result.get("force_closure")
    if not left_patch["visible"]:
        reasons.append("left_visual_patch_not_visible")
    if not right_patch["visible"]:
        reasons.append("right_visual_patch_not_visible")
    if not left_patch["on_bottle_body"] or not right_patch["on_bottle_body"]:
        reasons.append("visual_patch_not_on_bottle_body")
    if not grip_width_valid:
        reasons.append("grip_width_invalid")
    if depth_consistency_score < 45.0:
        reasons.append("left_right_depth_inconsistent")
    if epsilon_ready is False:
        reasons.append("epsilon_not_ready")
    pair_quality = float(np.clip(
        0.23 * left_patch["patch_quality_score"]
        + 0.23 * right_patch["patch_quality_score"]
        + 0.14 * fraction_similarity_score
        + 0.14 * width_score
        + 0.10 * depth_consistency_score
        + 0.10 * balance_score
        + 0.06 * (100.0 if epsilon_ready is not False else 0.0),
        0.0,
        100.0,
    ))
    pair_ready = bool(
        left_patch["patch_quality_score"] >= config["min_patch_quality"]
        and right_patch["patch_quality_score"] >= config["min_patch_quality"]
        and pair_quality >= config["min_pair_quality"]
        and grip_width_valid
        and epsilon_ready is not False
    )
    if pair_ready and not reasons:
        reasons.append("visual clamp patches are depth-validated on the bottle body")
    epsilon_contacts = [
        {"point": left_patch["patch_center_px"], "inward_normal": left_patch["inward_normal"]},
        {"point": right_patch["patch_center_px"], "inward_normal": right_patch["inward_normal"]},
    ]
    return {
        "visual_contact_model_enabled": True,
        "contact_model_type": CONTACT_MODEL_TYPE,
        "no_force_sensor": True,
        "contact_model_note": NO_FORCE_SENSOR_NOTE,
        "patch_height_mm": float(config["patch_height_mm"]),
        "left_visual_clamp_patch": left_patch,
        "right_visual_clamp_patch": right_patch,
        "left_clamp_patch": left_patch,
        "right_clamp_patch": right_patch,
        "visual_contact_pair": {
            "left_patch_center_px": left_patch["patch_center_px"],
            "right_patch_center_px": right_patch["patch_center_px"],
            "pair_center_px": pair_center,
            "body_fraction": None if body_fraction is None else float(body_fraction),
            "grip_width_px": grip_width_px,
            "grip_width_mm": grip_width_mm,
            "grip_width_valid": grip_width_valid,
            "left_right_body_fraction_similarity_score": fraction_similarity_score,
            "left_right_depth_consistency_score": depth_consistency_score,
            "balance_around_bottle_center_score": balance_score,
            "rotation_risk_estimate": float(np.clip(100.0 - balance_score, 0.0, 100.0)),
            "slip_risk_estimate": float(np.clip(100.0 - min(left_patch["edge_margin_score"], right_patch["edge_margin_score"]), 0.0, 100.0)),
            "epsilon_force_closure": epsilon_ready,
            "epsilon_contacts": epsilon_contacts,
        },
        "patch_quality_scores": {
            "left": left_patch["patch_quality_score"],
            "right": right_patch["patch_quality_score"],
        },
        "visual_contact_pair_quality_score": pair_quality,
        "pair_quality_score": pair_quality,
        "visual_contact_quality_score": pair_quality,
        "visual_contact_pair_ready": pair_ready,
        "visual_contact_ready": pair_ready,
        "visual_contact_reasons": reasons,
        "visual_contact_warnings": warnings,
        "epsilon_contacts": epsilon_contacts,
    }


def contact_set_from_visual_contact_model(result, embodiment_id="parallel_jaw_two_finger"):
    """Adapt the legacy visual-contact result into a universal ContactSet.

    This is an adapter only. It preserves the existing left/right patch geometry
    and lets embodiment-aware status/benchmark code consume the same evidence
    without forcing non-parallel grippers through this two-contact path.
    """

    try:
        from grasp_project.embodiment.contact_primitives import ContactSet, contact_from_legacy_patch
    except Exception:
        return None
    result = result or {}
    left_patch = (
        result.get("left_clamp_patch")
        or result.get("left_visual_clamp_patch")
        or result.get("left_contact_patch")
        or {}
    )
    right_patch = (
        result.get("right_clamp_patch")
        or result.get("right_visual_clamp_patch")
        or result.get("right_contact_patch")
        or {}
    )
    contacts = [
        contact_from_legacy_patch(
            "left_contact",
            embodiment_id,
            left_patch,
            source="depth_supported_contact",
            associated_finger_or_element="left_inner_jaw",
        ),
        contact_from_legacy_patch(
            "right_contact",
            embodiment_id,
            right_patch,
            source="depth_supported_contact",
            associated_finger_or_element="right_inner_jaw",
        ),
    ]
    contact_set = ContactSet(
        embodiment_id=embodiment_id,
        expected_contact_count=2,
        contacts=contacts,
        contact_hull_2d=_contact_hull_from_result(result),
        contact_set_confidence=float(result.get("visual_contact_pair_quality_score") or result.get("pair_quality_score") or 0.0),
        contact_set_valid=bool(result.get("visual_contact_ready")),
        blockers=list(result.get("visual_contact_reasons") or []),
        limitations=[NO_FORCE_SENSOR_NOTE],
    )
    if not contact_set.contact_set_valid and not contact_set.blockers:
        contact_set.blockers.append("visual_contact_pair_not_ready")
    return contact_set


def _contact_hull_from_result(result):
    pair = result.get("visual_contact_pair") or {}
    hull = result.get("contact_hull_2d") or pair.get("contact_hull_2d")
    if hull:
        return hull
    points = []
    for key in ("left_clamp_patch", "right_clamp_patch", "left_visual_clamp_patch", "right_visual_clamp_patch"):
        patch = result.get(key) or {}
        points.extend(patch.get("polygon") or patch.get("patch_polygon_px") or [])
    if len(points) < 3:
        return []
    try:
        arr = np.asarray(points, dtype=np.float32)
        hull = cv2.convexHull(arr.reshape((-1, 1, 2))).reshape((-1, 2))
        return hull.tolist()
    except Exception:
        return []

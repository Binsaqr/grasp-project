import numpy as np


MIN_DEPTH_VALID_RATIO = 0.60
# Initial lock must be based on a genuinely complete bottle silhouette.  The
# former eight-pixel margin admitted the 45x88 left-edge fragment reproduced in
# the physical demo.  Five percent is still easy to satisfy with a centered
# bottle while rejecting clipped edge fragments before they can own a lock.
MIN_MARGIN_RATIO = 0.05
MIN_LONG_EXTENT_RATIO = 0.34
MAX_BBOX_WIDTH_RATIO = 0.82
MAX_BBOX_HEIGHT_RATIO = 0.88
MAX_AREA_RATIO = 0.32
CENTER_WARNING_RATIO = 0.12


def clamp_score(value):
    return float(np.clip(float(value or 0.0), 0.0, 100.0))


def score_margin(margins, required_margin):
    values = [float(v) for v in margins.values()]
    if not values:
        return 0.0
    return clamp_score(100.0 * min(values) / max(float(required_margin), 1.0))


def recommended_action(reasons, center_error_x, center_error_y, image_shape):
    h, w = image_shape[:2]
    normalized = [str(reason).lower() for reason in reasons]
    if any("not detected" in reason for reason in normalized):
        return "SEARCH"
    if any("too large" in reason or "too close" in reason for reason in normalized):
        return "MOVE_BACK"
    if any("left edge" in reason for reason in normalized):
        return "MOVE_RIGHT"
    if any("right edge" in reason for reason in normalized):
        return "MOVE_LEFT"
    if any("top" in reason for reason in normalized):
        return "MOVE_DOWN"
    if any("bottom" in reason for reason in normalized):
        return "MOVE_UP"
    if abs(center_error_x) > w * CENTER_WARNING_RATIO or abs(center_error_y) > h * CENTER_WARNING_RATIO:
        return "CENTER_BOTTLE"
    return "VIEW_OK"


def check_full_bottle_visibility(pose, image_shape):
    h, w = image_shape[:2]
    required_margin = max(8.0, min(h, w) * MIN_MARGIN_RATIO)
    if pose is None or not pose.get("detected"):
        return {
            "full_bottle_visible": False,
            "visibility_score": 0.0,
            "border_margins": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            "required_margin_px": float(required_margin),
            "missing_region_reason": ["Bottle not detected."],
            "recommended_view_action": "SEARCH",
            "contact_model_ready": False,
            "view_status": "SEARCH_BOTTLE",
        }

    mask = pose.get("cleaned_mask")
    reasons = []
    if mask is None or np.count_nonzero(mask) == 0:
        reasons.append("Cleaned bottle mask is missing.")

    projected_visible_fraction = pose.get("projected_visible_fraction")
    projected_partial = bool(
        projected_visible_fraction is not None
        and float(projected_visible_fraction) < 0.98
    )
    partial_extent = bool(pose.get("partial_extent", False) or projected_partial)
    if partial_extent:
        reasons.append(
            "Current observation is only a partial bottle extent; a full view is required for locking."
        )

    x, y, bw, bh = [float(v) for v in pose.get("bbox_xywh", [0, 0, 0, 0])]
    margins = {
        "left": x,
        "right": float(w) - (x + bw),
        "top": y,
        "bottom": float(h) - (y + bh),
    }

    if margins["left"] < required_margin:
        reasons.append("Left bottle edge is cut off or too close to border.")
    if margins["right"] < required_margin:
        reasons.append("Right bottle edge is cut off or too close to border.")
    if margins["top"] < required_margin:
        reasons.append("Top of bottle is not fully visible.")
    if margins["bottom"] < required_margin:
        reasons.append("Bottom of bottle is not fully visible.")

    bbox_width_ratio = bw / max(float(w), 1.0)
    bbox_height_ratio = bh / max(float(h), 1.0)
    bbox_long_extent_ratio = max(bw, bh) / max(float(min(h, w)), 1.0)
    area_ratio = float(pose.get("area_ratio", 0.0) or 0.0)
    if bbox_long_extent_ratio < MIN_LONG_EXTENT_RATIO:
        reasons.append(
            "Bottle visible extent is too short for a reliable full-bottle lock."
        )
    if bbox_width_ratio > MAX_BBOX_WIDTH_RATIO or bbox_height_ratio > MAX_BBOX_HEIGHT_RATIO or area_ratio > MAX_AREA_RATIO:
        reasons.append("Bottle appears too large or too close for full-bottle contact modeling.")

    depth_ratio = float(pose.get("depth_confidence", 0.0) or 0.0)
    if depth_ratio < MIN_DEPTH_VALID_RATIO:
        reasons.append("Bottle depth valid ratio is too low for full-bottle visibility.")

    center = np.asarray(pose.get("center_px", [w / 2.0, h / 2.0]), dtype=float)
    center_error_x = float(center[0] - w / 2.0)
    center_error_y = float(center[1] - h / 2.0)

    margin_score = score_margin(margins, required_margin)
    size_score = clamp_score(
        min(
            100.0,
            100.0 * (1.0 - max(0.0, bbox_width_ratio - 0.55) / max(MAX_BBOX_WIDTH_RATIO - 0.55, 1e-6)),
            100.0 * (1.0 - max(0.0, bbox_height_ratio - 0.70) / max(MAX_BBOX_HEIGHT_RATIO - 0.70, 1e-6)),
            100.0 * (1.0 - max(0.0, area_ratio - 0.14) / max(MAX_AREA_RATIO - 0.14, 1e-6)),
        )
    )
    depth_score = clamp_score(depth_ratio * 100.0)
    center_score = clamp_score(
        100.0
        * (1.0 - abs(center_error_x) / max(w * 0.50, 1.0))
        * (1.0 - abs(center_error_y) / max(h * 0.50, 1.0))
    )
    visibility_score = clamp_score(0.38 * margin_score + 0.25 * size_score + 0.25 * depth_score + 0.12 * center_score)
    action = recommended_action(reasons, center_error_x, center_error_y, image_shape)
    full_visible = not reasons
    return {
        "full_bottle_visible": bool(full_visible),
        "visibility_score": visibility_score,
        "border_margins": {key: float(value) for key, value in margins.items()},
        "required_margin_px": float(required_margin),
        "bbox_size_ratio": {
            "width": float(bbox_width_ratio),
            "height": float(bbox_height_ratio),
            "long_extent": float(bbox_long_extent_ratio),
            "area": area_ratio,
        },
        "min_long_extent_ratio": float(MIN_LONG_EXTENT_RATIO),
        "center_error_px": {"x": center_error_x, "y": center_error_y},
        "depth_valid_ratio": depth_ratio,
        "missing_region_reason": reasons,
        "recommended_view_action": action,
        "contact_model_ready": bool(full_visible),
        "view_status": "VIEW_OK" if full_visible else "ADJUST_VIEW",
        "partial_extent": partial_extent,
        "projected_visible_fraction": projected_visible_fraction,
        "full_extent_evidence_status": (
            "CURRENT_FULL_EXTENT"
            if full_visible
            else "PARTIAL_EXTENT_NOT_VALID_FOR_LOCK"
            if partial_extent
            else "FULL_EXTENT_REQUIREMENTS_NOT_MET"
        ),
    }

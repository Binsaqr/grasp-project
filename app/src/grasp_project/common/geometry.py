"""Common geometry helpers that are safe to import without hardware."""

import numpy as np


def point_distance_px(a, b):
    """Return Euclidean distance between two 2D points."""
    if a is None or b is None:
        return None
    try:
        ax, ay = [float(v) for v in a[:2]]
        bx, by = [float(v) for v in b[:2]]
    except (TypeError, ValueError):
        return None
    return float(np.hypot(ax - bx, ay - by))


def distance_point_to_line(point, a, b):
    """Return perpendicular distance from a point to a line through ``a`` and ``b``."""
    if point is None or a is None or b is None:
        return None
    try:
        p = np.asarray(point[:2], dtype=float)
        a_arr = np.asarray(a[:2], dtype=float)
        b_arr = np.asarray(b[:2], dtype=float)
    except (TypeError, ValueError):
        return None
    ab = b_arr - a_arr
    denom = float(np.linalg.norm(ab))
    if denom <= 1e-9:
        return point_distance_px(p, a_arr)
    # scalar 2D cross product; np.cross no longer accepts 2D vectors on NumPy >= 2.0
    ap = p - a_arr
    return float(abs(ab[0] * ap[1] - ab[1] * ap[0]) / denom)


def normalize_angle_deg(angle):
    """Normalize an angle to ``[-180, 180)`` degrees."""
    try:
        return float((float(angle) + 180.0) % 360.0 - 180.0)
    except (TypeError, ValueError):
        return None


def angle_delta_deg(current, previous):
    """Return the shortest signed delta from ``previous`` to ``current`` degrees."""
    current_norm = normalize_angle_deg(current)
    previous_norm = normalize_angle_deg(previous)
    if current_norm is None or previous_norm is None:
        return None
    return normalize_angle_deg(current_norm - previous_norm)


def line_angle_deg(a, b):
    """Return the image-plane angle of the line from point ``a`` to point ``b``."""
    if a is None or b is None:
        return None
    try:
        ax, ay = [float(v) for v in a[:2]]
        bx, by = [float(v) for v in b[:2]]
    except (TypeError, ValueError):
        return None
    return float(np.degrees(np.arctan2(by - ay, bx - ax)))


def bbox_xyxy_from_geometry(geometry):
    """Extract an ``[x1, y1, x2, y2]`` bbox from common geometry dictionaries."""
    geometry = geometry or {}
    bbox = (
        geometry.get("bottle_extent_bbox_xyxy")
        or geometry.get("bbox_xyxy")
        or geometry.get("accepted_current_bottle_bbox")
        or geometry.get("raw_bottle_bbox")
    )
    if bbox and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
        except (TypeError, ValueError):
            pass
    bbox = geometry.get("bbox_xywh")
    if bbox and len(bbox) == 4:
        try:
            x, y, w, h = [float(v) for v in bbox]
            if w > 0 and h > 0:
                return [x, y, x + w, y + h]
        except (TypeError, ValueError):
            pass
    points = geometry.get("oriented_box") or geometry.get("oriented_box_points") or geometry.get("contour_hull_px")
    if points:
        try:
            arr = np.asarray(points, dtype=float).reshape(-1, 2)
            if arr.size >= 4:
                return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]
        except (TypeError, ValueError):
            pass
    return None


def bbox_xywh_to_xyxy(bbox):
    """Convert ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``."""
    if not bbox:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def bbox_xyxy_to_xywh(bbox):
    """Convert ``[x1, y1, x2, y2]`` to ``[x, y, w, h]``."""
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_center(bbox):
    """Return the center point of an ``[x1, y1, x2, y2]`` bbox."""
    if not bbox or len(bbox) != 4:
        return None
    return [(float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5]


def bbox_area_px(bbox):
    """Return axis-aligned bbox area in pixels."""
    if not bbox:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def point_in_bbox(point, bbox):
    """Return whether a 2D point lies inside an ``xyxy`` bbox."""
    if point is None or bbox is None:
        return False
    try:
        x, y = [float(v) for v in point[:2]]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return False
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_aspect_long_short(bbox):
    """Return long-side/short-side aspect ratio for a bbox."""
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    short = min(width, height)
    long = max(width, height)
    if short <= 1e-6:
        return None
    return float(long / short)


def bbox_iou_xyxy(a, b):
    """Return IoU for two ``[x1, y1, x2, y2]`` boxes, or ``None`` if invalid."""
    if not a or not b:
        return None
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return None
    return inter / union


def merge_bboxes_xyxy(bboxes, image_shape=None):
    """Return the union bbox for non-empty ``xyxy`` boxes."""
    valid = []
    for bbox in bboxes or []:
        if not bbox:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            valid.append([x1, y1, x2, y2])
    if not valid:
        return None
    merged = [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]
    if image_shape is not None:
        return clip_bbox_xyxy(merged, image_shape)
    return merged


def expand_bbox_xyxy(bbox, image_shape=None, expand_ratio=0.0, min_pad_px=0.0):
    """Expand an ``xyxy`` bbox by ratio/min padding, optionally clipping to image."""
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = max((x2 - x1) * float(expand_ratio), float(min_pad_px))
    pad_y = max((y2 - y1) * float(expand_ratio), float(min_pad_px))
    expanded = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    if image_shape is not None:
        return clip_bbox_xyxy(expanded, image_shape)
    return expanded


def clip_bbox_xyxy(bbox, image_shape):
    """Clip an ``xyxy`` bbox to image bounds."""
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    h, w = image_shape[:2]
    clipped = [
        float(np.clip(x1, 0, max(w - 1, 0))),
        float(np.clip(y1, 0, max(h - 1, 0))),
        float(np.clip(x2, 0, max(w, 0))),
        float(np.clip(y2, 0, max(h, 0))),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def polygon_bbox_xyxy(points, image_shape=None, expand_px=0.0):
    """Return the bbox of polygon points, optionally expanded and clipped."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return None
    bbox = [
        float(np.min(arr[:, 0])) - float(expand_px),
        float(np.min(arr[:, 1])) - float(expand_px),
        float(np.max(arr[:, 0])) + float(expand_px),
        float(np.max(arr[:, 1])) + float(expand_px),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if image_shape is not None:
        return clip_bbox_xyxy(bbox, image_shape)
    return bbox


__all__ = [
    "bbox_area_px",
    "bbox_aspect_long_short",
    "bbox_center",
    "bbox_iou_xyxy",
    "bbox_xywh_to_xyxy",
    "bbox_xyxy_from_geometry",
    "bbox_xyxy_to_xywh",
    "clip_bbox_xyxy",
    "angle_delta_deg",
    "distance_point_to_line",
    "expand_bbox_xyxy",
    "line_angle_deg",
    "merge_bboxes_xyxy",
    "normalize_angle_deg",
    "point_in_bbox",
    "point_distance_px",
    "polygon_bbox_xyxy",
]

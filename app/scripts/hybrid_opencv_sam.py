import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from epsilon_quality import (
    compute_epsilon_quality,
    save_all_wrench_space_projections,
    save_epsilon_result_json,
    save_wrench_space_projection,
)


IMAGE_PATH = "output/captured_frame.png"
CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml"

FRICTION_COEFFICIENT = 0.5
CONTACT_DEPTH_THRESHOLD_MM = 15.0
CONTACT_PIXEL_THRESHOLD_PX = 20.0
MIN_VALID_DEPTH_RATIO = 0.25
MIN_OBJECT_GRIPPER_SAMPLE_DISTANCE_PX = 5.0
MANUAL_GRIPPER_BOXES_PATH = "gripper_bboxes.json"


def find_prompt_in_roi(image_rgb, roi, min_area=200, exclude_mask=None):
    x1, y1, x2, y2 = roi
    crop = image_rgb[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if exclude_mask is not None:
        exclude_crop = exclude_mask[y1:y2, x1:x2].astype(np.uint8)
        if exclude_crop.shape == edges.shape:
            edges[exclude_crop > 0] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        valid.append(c)

    if not valid:
        return None

    best = max(valid, key=cv2.contourArea)
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1
    return [cx, cy]


def sam_predict(predictor, point):
    mask, _, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=False
    )
    return mask[0].astype(np.uint8)


def mask_boundary(mask):
    kernel = np.ones((3, 3), np.uint8)
    edge = mask - cv2.erode(mask, kernel)
    return np.column_stack(np.where(edge > 0))  # [y, x]

def load_depth_optional(path="output/depth_frame.png"):
    if not os.path.exists(path):
        return None
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return depth


def get_depth_mm(depth_img, point_yx, window=5):
    if depth_img is None or point_yx is None:
        return None

    y, x = int(point_yx[0]), int(point_yx[1])
    h, w = depth_img.shape[:2]

    x1 = max(0, x - window)
    x2 = min(w, x + window + 1)
    y1 = max(0, y - window)
    y2 = min(h, y + window + 1)

    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if len(valid) == 0:
        return None

    return float(np.median(valid))


def depth_patch_stats_yx(depth_img, point_yx, window=5):
    if depth_img is None or point_yx is None:
        return {
            "depth_mm": None,
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "total_pixels": 0,
        }

    y, x = int(round(point_yx[0])), int(round(point_yx[1]))
    h, w = depth_img.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return {
            "depth_mm": None,
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "total_pixels": 0,
        }

    x1 = max(0, x - window)
    x2 = min(w, x + window + 1)
    y1 = max(0, y - window)
    y2 = min(h, y + window + 1)
    patch = depth_img[y1:y2, x1:x2]
    valid = patch[(patch > 0) & (patch < 10000)]
    valid_ratio = float(len(valid) / max(patch.size, 1))
    if len(valid) == 0:
        depth_mm = None
    else:
        depth_mm = float(np.median(valid))
    return {
        "depth_mm": depth_mm,
        "valid_ratio": valid_ratio,
        "valid_pixels": int(len(valid)),
        "total_pixels": int(patch.size),
    }


def point_inside_mask_yx(mask, point_yx):
    if mask is None or point_yx is None:
        return False
    y, x = int(round(point_yx[0])), int(round(point_yx[1]))
    h, w = mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return bool(mask[y, x] > 0)


def shift_gripper_sample_yx(obj_p, face_p, object_mask=None, direction_xy=None, min_shift=6, max_steps=30):
    sample = np.asarray(face_p, dtype=float).copy()
    if object_mask is None:
        if direction_xy is None:
            return sample, False
        direction_xy = np.asarray(direction_xy, dtype=float)
        norm = np.linalg.norm(direction_xy)
        if norm < 1e-6:
            return sample, False
        direction_yx = np.array([direction_xy[1], direction_xy[0]], dtype=float) / norm
        return sample + direction_yx * min_shift, True

    directions = []
    if direction_xy is not None:
        direction_xy = np.asarray(direction_xy, dtype=float)
        norm = np.linalg.norm(direction_xy)
        if norm >= 1e-6:
            directions.append(np.array([direction_xy[1], direction_xy[0]], dtype=float) / norm)

    away = np.asarray(face_p, dtype=float) - np.asarray(obj_p, dtype=float)
    away_norm = np.linalg.norm(away)
    if away_norm >= 1e-6:
        directions.append(away / away_norm)

    h, w = object_mask.shape[:2]
    if not point_inside_mask_yx(object_mask, sample):
        if directions:
            candidate = sample + directions[0] * min_shift
            y, x = int(round(candidate[0])), int(round(candidate[1]))
            if 0 <= x < w and 0 <= y < h and not point_inside_mask_yx(object_mask, candidate):
                return candidate, True
        return sample, False

    for direction_yx in directions:
        for step in range(min_shift, max_steps + 1):
            candidate = np.asarray(face_p, dtype=float) + direction_yx * step
            y, x = int(round(candidate[0])), int(round(candidate[1]))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if not point_inside_mask_yx(object_mask, candidate):
                return candidate, True

    return sample, False


def evaluate_contact_candidate(
    obj_p,
    face_p,
    depth_img=None,
    object_mask=None,
    gripper_direction_xy=None,
    object_depth_stats=None,
):
    obj_p = np.asarray(obj_p, dtype=float)
    raw_face_p = np.asarray(face_p, dtype=float)
    face_sample_p, shifted = shift_gripper_sample_yx(
        obj_p,
        raw_face_p,
        object_mask,
        direction_xy=gripper_direction_xy,
    )
    pixel_dist = float(np.linalg.norm(face_sample_p - obj_p))
    near_same = pixel_dist < MIN_OBJECT_GRIPPER_SAMPLE_DISTANCE_PX
    gripper_inside_object = point_inside_mask_yx(object_mask, face_sample_p)

    obj_stats = depth_patch_stats_yx(depth_img, obj_p, window=5)
    face_stats = depth_patch_stats_yx(depth_img, face_sample_p, window=5)
    obj_depth = obj_stats["depth_mm"]
    face_depth = face_stats["depth_mm"]
    depth_diff = None
    depth_confidence = 0.0
    depth_available = depth_img is not None

    reasons = []
    if near_same:
        reasons.append("object and gripper samples within 5 px")
    if gripper_inside_object:
        reasons.append("gripper sample inside object mask")

    if depth_available and object_depth_stats and object_depth_stats.get("available"):
        if obj_depth is not None and not depth_in_object_band(obj_depth, object_depth_stats):
            reasons.append("object sample outside object depth band")
        if face_depth is not None and not depth_in_object_band(face_depth, object_depth_stats):
            if face_depth > object_depth_stats["band_high"]:
                reasons.append("gripper sample behind object depth band")
            else:
                reasons.append("gripper sample in front of object depth band")

    if depth_available:
        object_depth_ok = obj_stats["valid_ratio"] >= MIN_VALID_DEPTH_RATIO
        gripper_depth_ok = face_stats["valid_ratio"] >= MIN_VALID_DEPTH_RATIO
        if obj_depth is None or face_depth is None:
            reasons.append("missing valid depth")
        else:
            depth_diff = abs(obj_depth - face_depth)
            depth_confidence = float(np.clip(1.0 - depth_diff / CONTACT_DEPTH_THRESHOLD_MM, 0.0, 1.0))
            if depth_diff > CONTACT_DEPTH_THRESHOLD_MM:
                reasons.append(f"depth mismatch {depth_diff:.1f} mm")
        if not object_depth_ok:
            reasons.append("low object depth ratio")
        if not gripper_depth_ok:
            reasons.append("low gripper depth ratio")
    else:
        object_depth_ok = True
        gripper_depth_ok = True
        depth_confidence = 0.5
        reasons.append("geometry-only: depth unreliable")

    pixel_ok = pixel_dist <= CONTACT_PIXEL_THRESHOLD_PX
    if not pixel_ok:
        reasons.append("contact gap too large")

    depth_valid = (
        not depth_available or (
            obj_depth is not None
            and face_depth is not None
            and obj_stats["valid_ratio"] >= MIN_VALID_DEPTH_RATIO
            and face_stats["valid_ratio"] >= MIN_VALID_DEPTH_RATIO
            and depth_diff is not None
            and depth_diff <= CONTACT_DEPTH_THRESHOLD_MM
            and depth_in_object_band(obj_depth, object_depth_stats)
            and depth_in_object_band(face_depth, object_depth_stats)
        )
    )
    valid = pixel_ok and depth_valid and not near_same and not gripper_inside_object
    geometry_score = float(np.clip(1.0 - pixel_dist / CONTACT_PIXEL_THRESHOLD_PX, 0.0, 1.0))
    if depth_available:
        ratio_score = min(obj_stats["valid_ratio"], face_stats["valid_ratio"])
        band_score = min(depth_band_score(obj_depth, object_depth_stats), depth_band_score(face_depth, object_depth_stats))
        depth_score = depth_confidence * ratio_score * band_score
    else:
        ratio_score = 1.0
        depth_score = 0.5
    mask_score = 0.0 if gripper_inside_object else 1.0
    sample_score = 0.0 if near_same else 1.0
    combined_score = (
        0.40 * geometry_score
        + 0.35 * depth_score
        + 0.15 * mask_score
        + 0.10 * sample_score
    )
    if not valid:
        combined_score -= 1.0

    return {
        "object_point": obj_p,
        "raw_face_point": raw_face_p,
        "face_point": face_sample_p,
        "pixel_dist": pixel_dist,
        "object_depth_mm": obj_depth,
        "face_depth_mm": face_depth,
        "depth_difference_mm": depth_diff,
        "object_depth_valid_ratio": obj_stats["valid_ratio"],
        "gripper_depth_valid_ratio": face_stats["valid_ratio"],
        "object_depth_band_low": object_depth_stats.get("band_low") if object_depth_stats else None,
        "object_depth_band_high": object_depth_stats.get("band_high") if object_depth_stats else None,
        "gripper_depth_difference_from_object_median_mm": (
            abs(face_depth - object_depth_stats["median"])
            if face_depth is not None and object_depth_stats and object_depth_stats.get("available")
            else None
        ),
        "depth_confidence": depth_confidence,
        "gripper_sample_inside_object": gripper_inside_object,
        "near_same_sample": near_same,
        "sample_shifted": shifted,
        "valid": valid,
        "depth_available": depth_available,
        "score": combined_score,
        "reasons": reasons,
        "selection_source": "RGB-D validated" if depth_available and valid else ("geometry-only: depth unreliable" if not depth_available or not (object_depth_stats or {}).get("available", False) else "RGB-D invalid"),
        "depth_affected_selection": bool(depth_available and object_depth_stats and object_depth_stats.get("available")),
    }


def mask_bbox(mask):
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) == 0:
        return None
    y1, x1 = np.min(pts, axis=0)
    y2, x2 = np.max(pts, axis=0)
    return (int(x1), int(y1), int(x2) + 1, int(y2) + 1)


def clip_roi(roi, width, height):
    x1, y1, x2, y2 = roi
    return (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def object_guided_gripper_rois(image_shape, object_box):
    h, w = image_shape[:2]
    if object_box is None:
        return (
            (int(w * 0.12), int(h * 0.25), int(w * 0.50), int(h * 0.75)),
            (int(w * 0.50), int(h * 0.25), int(w * 0.88), int(h * 0.75)),
        )

    ox1, oy1, ox2, oy2 = object_box
    object_width = max(ox2 - ox1, 1)
    object_height = max(oy2 - oy1, 1)
    x_overlap = max(12, int(0.03 * w))
    side_span = max(int(object_width * 1.35), int(0.16 * w))

    # Search next to the object, at the object's middle/lower-middle height.
    # This avoids the lower white/base support and upper rail/background hardware.
    y1 = oy1 + int(0.18 * object_height)
    y2 = oy1 + int(0.88 * object_height)

    left_roi = clip_roi((ox1 - side_span, y1, ox1 + x_overlap, y2), w, h)
    right_roi = clip_roi((ox2 - x_overlap, y1, ox2 + side_span, y2), w, h)
    return left_roi, right_roi


def load_manual_gripper_boxes(path=MANUAL_GRIPPER_BOXES_PATH):
    if not os.path.exists(path):
        return {}
    try:
        import json
        with open(path, "r") as f:
            data = json.load(f)
        boxes = {}
        for side in ["left", "right"]:
            if side in data and len(data[side]) == 4:
                boxes[side] = tuple(int(v) for v in data[side])
        return boxes
    except Exception as exc:
        print(f"Warning: could not load manual gripper boxes from {path}: {exc}")
        return {}


def rect_mask(shape, box):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = clip_roi(box, shape[1], shape[0])
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def box_area(box):
    if box is None:
        return 0
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_intersection_area(a, b):
    if a is None or b is None:
        return 0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def box_overlap_fraction(inner, outer):
    return box_intersection_area(inner, outer) / max(box_area(inner), 1)


def local_depth_stats(depth_img, points_yx, window=5):
    if depth_img is None or points_yx is None or len(points_yx) == 0:
        return None

    h, w = depth_img.shape[:2]
    values = []
    for point in points_yx:
        y, x = int(point[0]), int(point[1])
        x1 = max(0, x - window)
        x2 = min(w, x + window + 1)
        y1 = max(0, y - window)
        y2 = min(h, y + window + 1)
        patch = depth_img[y1:y2, x1:x2]
        valid = patch[(patch > 0) & (patch < 10000)]
        if len(valid) > 0:
            values.extend(valid.astype(float).tolist())

    if len(values) == 0:
        return None

    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
    }


def compute_object_depth_stats(object_mask, depth_img):
    if object_mask is None or depth_img is None:
        return {
            "available": False,
            "median": None,
            "mean": None,
            "std": None,
            "valid_ratio": 0.0,
            "band_low": None,
            "band_high": None,
            "band_width": None,
            "warning": "geometry-only: depth unreliable",
        }

    values = depth_img[(object_mask > 0) & (depth_img > 0) & (depth_img < 10000)]
    object_pixels = int(np.count_nonzero(object_mask > 0))
    valid_ratio = float(len(values) / max(object_pixels, 1))
    if len(values) == 0 or valid_ratio < MIN_VALID_DEPTH_RATIO:
        return {
            "available": False,
            "median": None,
            "mean": None,
            "std": None,
            "valid_ratio": valid_ratio,
            "band_low": None,
            "band_high": None,
            "band_width": None,
            "warning": "geometry-only: depth unreliable",
        }

    values = values.astype(float)
    median = float(np.median(values))
    mean = float(np.mean(values))
    std = float(np.std(values))
    band_width = max(30.0, 2.0 * std)
    return {
        "available": True,
        "median": median,
        "mean": mean,
        "std": std,
        "valid_ratio": valid_ratio,
        "band_low": median - band_width,
        "band_high": median + band_width,
        "band_width": band_width,
        "warning": None,
    }


def depth_in_object_band(depth_mm, object_depth_stats):
    if not object_depth_stats or not object_depth_stats.get("available") or depth_mm is None:
        return True
    return object_depth_stats["band_low"] <= depth_mm <= object_depth_stats["band_high"]


def depth_band_score(depth_mm, object_depth_stats):
    if not object_depth_stats or not object_depth_stats.get("available") or depth_mm is None:
        return 0.5
    diff = abs(float(depth_mm) - float(object_depth_stats["median"]))
    return float(np.clip(1.0 - diff / max(float(object_depth_stats["band_width"]), 1.0), 0.0, 1.0))


def candidate_mask_depth_stats(candidate_mask, depth_img):
    if candidate_mask is None or depth_img is None:
        return {"median": None, "valid_ratio": 0.0}
    total = int(np.count_nonzero(candidate_mask > 0))
    if total == 0:
        return {"median": None, "valid_ratio": 0.0}
    values = depth_img[(candidate_mask > 0) & (depth_img > 0) & (depth_img < 10000)]
    if len(values) == 0:
        return {"median": None, "valid_ratio": 0.0}
    return {
        "median": float(np.median(values.astype(float))),
        "valid_ratio": float(len(values) / max(total, 1)),
    }


def bbox_face_points(box, side, object_box=None, samples=80):
    if box is None:
        return np.empty((0, 2), dtype=int)

    x1, y1, x2, y2 = box
    if object_box is not None:
        _, oy1, _, oy2 = object_box
        y1 = max(y1, oy1)
        y2 = min(y2, oy2)
        if y2 <= y1:
            y1, y2 = box[1], box[3]

    x = x2 - 1 if side == "left" else x1
    ys = np.linspace(y1, max(y1, y2 - 1), max(2, samples)).astype(int)
    xs = np.full_like(ys, int(x))
    return np.column_stack([ys, xs])


def candidate_face_to_object_distance(box, side, object_points, object_box=None):
    face_points = bbox_face_points(box, side, object_box)
    if len(face_points) == 0 or len(object_points) == 0:
        return None, None, None

    best = None
    for face_p in face_points:
        dists = np.linalg.norm(object_points - face_p, axis=1)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        if best is None or dist < best[0]:
            best = (dist, face_p, object_points[idx])

    return best


def gripper_body_bbox_status(side, bbox, image_shape, object_box=None):
    if bbox is None:
        return False, "Missing gripper body bbox."

    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width < int(0.06 * w) or height < int(0.08 * h):
        return False, "Detected gripper body is too small."

    if object_box is not None:
        ox1, oy1, ox2, oy2 = object_box
        object_height = max(oy2 - oy1, 1)
        vertical_overlap = max(0, min(y2, oy2) - max(y1, oy1))
        overlap_fraction = vertical_overlap / max(min(height, object_height), 1)
        below_fraction = max(0, y2 - oy2) / max(height, 1)

        if overlap_fraction < 0.30:
            return False, "Rejected gripper fallback: detected support/base instead of metal jaw."
        if below_fraction > 0.45 or y1 > oy2 - int(0.12 * object_height):
            return False, "Rejected gripper fallback: detected support/base instead of metal jaw."

        reach = int(0.06 * w)
        if side == "left" and x2 < ox1 - reach:
            return False, "Detected left gripper body is too far from the object."
        if side == "right" and x1 > ox2 + reach:
            return False, "Detected right gripper body is too far from the object."

    return True, None


def probable_gripper_body_bbox(side, bbox, image_shape, object_box=None):
    ok, _ = gripper_body_bbox_status(side, bbox, image_shape, object_box)
    return ok


def reliable_block_candidate(candidate):
    if not candidate:
        return False
    return (
        candidate.get("accepted", False)
        and candidate.get("confidence", 0.0) >= 0.58
        and candidate.get("solidness", 0.0) >= 0.18
        and candidate.get("vertical_overlap", 0.0) >= 0.35
        and (candidate.get("face_distance") is None or candidate.get("face_distance") <= 35.0)
        and (candidate.get("candidate_depth_valid_ratio", 1.0) >= MIN_VALID_DEPTH_RATIO)
        and not any("behind object depth band" in reason or "slot/gap" in reason for reason in candidate.get("reasons", []))
    )


def mode_reason_for_block(block_candidate, thin_candidate=None):
    thin_conf = thin_candidate.get("confidence", 0.0) if thin_candidate else 0.0
    return (
        "block preferred: solid filled area, object overlap, close inner face, and depth near object "
        f"(block {block_candidate.get('confidence', 0.0):.2f}, thin {thin_conf:.2f})"
    )


def score_gripper_candidate(
    candidate,
    side,
    image_shape,
    object_box,
    object_mask,
    object_points,
    depth_img=None,
    object_depth_stats=None,
):
    box = candidate["bbox"]
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    area = max(width * height, 1)

    if object_box is None:
        return None

    ox1, oy1, ox2, oy2 = object_box
    object_width = max(ox2 - ox1, 1)
    object_height = max(oy2 - oy1, 1)
    object_cx = (ox1 + ox2) / 2.0
    candidate_cx = (x1 + x2) / 2.0

    reasons = []
    side_ok = candidate_cx < object_cx if side == "left" else candidate_cx > object_cx
    if not side_ok:
        reasons.append("wrong side of object")

    vertical_overlap = max(0, min(y2, oy2) - max(y1, oy1))
    vertical_overlap_fraction = vertical_overlap / max(min(height, object_height), 1)
    below_fraction = max(0, y2 - oy2) / height
    object_overlap = box_intersection_area(box, object_box) / area

    dist_result = candidate_face_to_object_distance(box, side, object_points, object_box)
    face_distance = dist_result[0] if dist_result is not None else float("inf")
    max_reasonable_distance = max(18.0, object_width * 0.55)

    size_ratio = area / max(box_area(object_box), 1)
    solidness = float(candidate.get("solidness", 0.0))

    face_points = bbox_face_points(box, side, object_box, samples=20)
    obj_near_face = dist_result[2] if dist_result is not None else None
    depth_confidence = 0.5
    depth_diff = None
    candidate_depth = None
    candidate_depth_valid_ratio = 0.0
    depth_difference_from_object_median = None
    object_band_score = 0.5
    depth_valid = True
    if depth_img is not None and obj_near_face is not None:
        face_depth = local_depth_stats(depth_img, face_points, window=4)
        obj_depth = local_depth_stats(depth_img, np.asarray([obj_near_face]), window=5)
        candidate_stats = candidate_mask_depth_stats(candidate.get("mask"), depth_img)
        candidate_depth = candidate_stats["median"]
        candidate_depth_valid_ratio = candidate_stats["valid_ratio"]
        if face_depth is None or obj_depth is None:
            depth_confidence = 0.0
            depth_valid = False
            reasons.append("invalid depth")
        else:
            depth_diff = abs(face_depth["median"] - obj_depth["median"])
            depth_confidence = float(np.clip(1.0 - depth_diff / 80.0, 0.0, 1.0))
            if depth_diff > 90.0:
                depth_valid = False
                reasons.append("depth mismatch")
        if object_depth_stats and object_depth_stats.get("available"):
            if candidate_depth is None:
                depth_valid = False
                reasons.append("candidate has no valid depth")
            else:
                depth_difference_from_object_median = abs(candidate_depth - object_depth_stats["median"])
                object_band_score = depth_band_score(candidate_depth, object_depth_stats)
                if candidate_depth_valid_ratio < MIN_VALID_DEPTH_RATIO:
                    depth_valid = False
                    reasons.append("low candidate depth ratio")
                if not depth_in_object_band(candidate_depth, object_depth_stats):
                    depth_valid = False
                    if candidate_depth > object_depth_stats["band_high"]:
                        reasons.append("candidate behind object depth band")
                    else:
                        reasons.append("candidate in front of object depth band")

    if vertical_overlap_fraction < 0.25:
        reasons.append("poor vertical overlap")
    if below_fraction > 0.45:
        reasons.append("mostly below object")
    if object_overlap > 0.35:
        reasons.append("overlaps object")
    if face_distance > max_reasonable_distance:
        reasons.append("inner face too far")
    if size_ratio < 0.08 or size_ratio > 4.5:
        reasons.append("unreasonable size")
    if candidate_depth_valid_ratio < MIN_VALID_DEPTH_RATIO and object_depth_stats and object_depth_stats.get("available"):
        reasons.append("low valid depth ratio before scoring")

    proximity_score = float(np.clip(1.0 - face_distance / max_reasonable_distance, 0.0, 1.0))
    vertical_score = float(np.clip(vertical_overlap_fraction, 0.0, 1.0))
    below_score = float(np.clip(1.0 - below_fraction, 0.0, 1.0))
    object_overlap_score = float(np.clip(1.0 - object_overlap / 0.35, 0.0, 1.0))
    size_score = float(np.clip(1.0 - abs(np.log(max(size_ratio, 1e-6)) - np.log(0.65)) / 2.4, 0.0, 1.0))
    solid_score = float(np.clip(solidness, 0.0, 1.0))
    side_score = 1.0 if side_ok else 0.0

    score = (
        0.28 * proximity_score +
        0.22 * vertical_score +
        0.12 * object_overlap_score +
        0.12 * size_score +
        0.10 * solid_score +
        0.10 * (depth_confidence * object_band_score) +
        0.04 * below_score +
        0.02 * side_score
    )

    accepted = (
        len(reasons) == 0
        and score >= 0.38
        and depth_valid
    )

    candidate.update({
        "score": float(score),
        "confidence": float(score),
        "accepted": accepted,
        "reasons": reasons,
        "face_distance": float(face_distance) if np.isfinite(face_distance) else None,
        "vertical_overlap": float(vertical_overlap_fraction),
        "below_fraction": float(below_fraction),
        "object_overlap": float(object_overlap),
        "depth_confidence": float(depth_confidence),
        "candidate_depth_difference_mm": float(depth_diff) if depth_diff is not None else None,
        "candidate_depth_median_mm": float(candidate_depth) if candidate_depth is not None else None,
        "candidate_depth_valid_ratio": float(candidate_depth_valid_ratio),
        "candidate_depth_difference_from_object_median_mm": (
            float(depth_difference_from_object_median)
            if depth_difference_from_object_median is not None
            else None
        ),
        "object_depth_band_low": object_depth_stats.get("band_low") if object_depth_stats else None,
        "object_depth_band_high": object_depth_stats.get("band_high") if object_depth_stats else None,
    })
    return candidate


def candidate_from_contour(contour, roi_offset, image_shape, source):
    x, y, w, h = cv2.boundingRect(contour)
    if w < 8 or h < 8:
        return None

    full_box = (x + roi_offset[0], y + roi_offset[1], x + roi_offset[0] + w, y + roi_offset[1] + h)
    contour_area = max(float(cv2.contourArea(contour)), 1.0)
    rect_area = max(float(w * h), 1.0)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    shifted = contour + np.array([[[roi_offset[0], roi_offset[1]]]], dtype=contour.dtype)
    cv2.drawContours(mask, [shifted], -1, 1, thickness=cv2.FILLED)
    return {
        "bbox": full_box,
        "mask": mask,
        "solidness": float(np.clip(contour_area / rect_area, 0.0, 1.0)),
        "source": source,
    }


def generate_gripper_candidates(image_rgb, roi, object_mask, depth_img=None, object_depth_stats=None):
    x1, y1, x2, y2 = roi
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 115)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    exclude_crop = object_mask[y1:y2, x1:x2].astype(np.uint8)
    edges[exclude_crop > 0] = 0
    if depth_img is not None and object_depth_stats and object_depth_stats.get("available"):
        depth_crop = depth_img[y1:y2, x1:x2]
        band_pad = 10.0
        valid_depth = (depth_crop > 0) & (depth_crop < 10000)
        outside_band = valid_depth & (
            (depth_crop.astype(float) < object_depth_stats["band_low"] - band_pad)
            | (depth_crop.astype(float) > object_depth_stats["band_high"] + band_pad)
        )
        edges[outside_band] = 0

    candidates = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        candidate = candidate_from_contour(contour, (x1, y1), image_rgb.shape, "edge")
        if candidate is not None:
            candidates.append(candidate)

    merged = {}
    for candidate in candidates:
        bx = candidate["bbox"]
        key = tuple(int(round(v / 8.0) * 8) for v in bx)
        existing = merged.get(key)
        if existing is None or candidate["solidness"] > existing["solidness"]:
            merged[key] = candidate

    return list(merged.values())


def refine_candidate_by_depth(
    candidate,
    side,
    image_shape,
    object_box,
    object_mask,
    object_points,
    depth_img,
    roi,
    object_depth_stats=None,
):
    if depth_img is None or candidate is None:
        return candidate

    dist_result = candidate_face_to_object_distance(candidate["bbox"], side, object_points, object_box)
    if dist_result is None:
        return candidate

    _, seed_face, _ = dist_result
    seed_depth = get_depth_mm(depth_img, seed_face, window=5)
    if seed_depth is None:
        return candidate

    x1, y1, x2, y2 = roi
    depth_crop = depth_img[y1:y2, x1:x2]
    object_crop = object_mask[y1:y2, x1:x2].astype(np.uint8)
    if depth_crop.size == 0:
        return candidate

    if object_depth_stats and object_depth_stats.get("available"):
        low = object_depth_stats["band_low"]
        high = object_depth_stats["band_high"]
    else:
        tolerance = max(25.0, CONTACT_DEPTH_THRESHOLD_MM * 2.0)
        low = seed_depth - tolerance
        high = seed_depth + tolerance
    depth_mask = (
        (depth_crop > 0) &
        (depth_crop < 10000) &
        (depth_crop.astype(float) >= low) &
        (depth_crop.astype(float) <= high) &
        (object_crop == 0)
    ).astype(np.uint8)

    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(depth_mask, 8)
    if num_labels <= 1:
        return candidate

    sy = int(seed_face[0]) - y1
    sx = int(seed_face[1]) - x1
    if not (0 <= sy < labels.shape[0] and 0 <= sx < labels.shape[1]):
        return candidate

    label = labels[sy, sx]
    if label == 0:
        search = labels[max(0, sy - 6):sy + 7, max(0, sx - 6):sx + 7]
        nonzero = search[search > 0]
        if len(nonzero) == 0:
            return candidate
        label = int(np.bincount(nonzero.ravel()).argmax())

    component = (labels == label).astype(np.uint8)
    if np.count_nonzero(component) < 120:
        return candidate

    refined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    refined_mask[y1:y2, x1:x2] = component
    refined_box = mask_bbox(refined_mask)
    if refined_box is None:
        return candidate

    refined = {
        "bbox": refined_box,
        "mask": refined_mask,
        "solidness": min(1.0, np.count_nonzero(refined_mask) / max(box_area(refined_box), 1)),
        "source": f"{candidate.get('source', 'candidate')}_depth_refined",
    }
    scored = score_gripper_candidate(
        refined,
        side,
        image_shape,
        object_box,
        object_mask,
        object_points,
        depth_img,
        object_depth_stats,
    )
    if scored is None:
        return candidate

    original_width = max(candidate["bbox"][2] - candidate["bbox"][0], 1)
    refined_width = max(refined_box[2] - refined_box[0], 1)
    if scored.get("accepted") and refined_width <= original_width * 1.05:
        return scored

    return candidate


def trim_candidate_to_inner_band(
    candidate,
    side,
    image_shape,
    object_box,
    object_mask,
    object_points,
    depth_img=None,
    object_depth_stats=None,
):
    if candidate is None or object_box is None:
        return candidate

    h, w = image_shape[:2]
    x1, y1, x2, y2 = candidate["bbox"]
    ox1, _, ox2, _ = object_box
    object_width = max(ox2 - ox1, 1)
    max_jaw_width = min(max(int(object_width * 0.75), int(0.08 * w)), int(0.20 * w))

    if side == "left":
        trimmed_box = (max(x1, x2 - max_jaw_width), y1, x2, y2)
    else:
        trimmed_box = (x1, y1, min(x2, x1 + max_jaw_width), y2)

    if box_area(trimmed_box) >= box_area(candidate["bbox"]) * 0.95:
        return candidate

    rect = rect_mask(image_shape, trimmed_box)
    source_mask = candidate.get("mask")
    if source_mask is not None and np.count_nonzero(source_mask) > 0:
        trimmed_mask = np.logical_and(source_mask > 0, rect > 0).astype(np.uint8)
        if np.count_nonzero(trimmed_mask) < 80:
            trimmed_mask = rect
    else:
        trimmed_mask = rect

    trimmed_bbox = mask_bbox(trimmed_mask)
    if trimmed_bbox is None:
        return candidate

    trimmed = {
        "bbox": trimmed_bbox,
        "mask": trimmed_mask,
        "solidness": min(1.0, np.count_nonzero(trimmed_mask) / max(box_area(trimmed_bbox), 1)),
        "source": f"{candidate.get('source', 'candidate')}_inner_band",
    }
    scored = score_gripper_candidate(
        trimmed,
        side,
        image_shape,
        object_box,
        object_mask,
        object_points,
        depth_img,
        object_depth_stats,
    )
    if scored is not None and scored.get("accepted"):
        return scored

    return candidate


def line_points_xy(p1, p2, count=40):
    xs = np.linspace(p1[0], p2[0], count)
    ys = np.linspace(p1[1], p2[1], count)
    return np.column_stack([xs, ys])


def score_thin_finger_candidate(
    candidate,
    side,
    image_shape,
    object_box,
    object_mask,
    object_points,
    edge_image,
    depth_img=None,
    object_depth_stats=None,
):
    if object_box is None or len(object_points) == 0:
        return None

    h, w = image_shape[:2]
    ox1, oy1, ox2, oy2 = object_box
    object_width = max(ox2 - ox1, 1)
    object_cx = (ox1 + ox2) / 2.0

    p1 = np.asarray(candidate["p1_xy"], dtype=float)
    p2 = np.asarray(candidate["p2_xy"], dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 18.0:
        return None

    endpoints_yx = [
        np.array([p1[1], p1[0]], dtype=float),
        np.array([p2[1], p2[0]], dtype=float),
    ]
    endpoint_scores = []
    for endpoint_yx in endpoints_yx:
        dists = np.linalg.norm(object_points - endpoint_yx, axis=1)
        idx = int(np.argmin(dists))
        endpoint_scores.append((float(dists[idx]), endpoint_yx, object_points[idx]))

    endpoint_scores.sort(key=lambda item: item[0])
    face_distance, tip_yx, object_contact_yx = endpoint_scores[0]
    tail_yx = endpoints_yx[1] if np.allclose(tip_yx, endpoints_yx[0]) else endpoints_yx[0]
    tip_xy = np.array([tip_yx[1], tip_yx[0]], dtype=float)
    tail_xy = np.array([tail_yx[1], tail_yx[0]], dtype=float)

    away_xy = tail_xy - tip_xy
    away_norm = np.linalg.norm(away_xy)
    if away_norm < 1e-6:
        return None
    away_xy = away_xy / away_norm

    side_ok = tip_xy[0] <= object_cx if side == "left" else tip_xy[0] >= object_cx
    approach_ok = tail_xy[0] < tip_xy[0] if side == "left" else tail_xy[0] > tip_xy[0]
    if approach_ok:
        side_ok = True

    object_overlap = 0.0
    samples = line_points_xy(tip_xy, tail_xy, count=40)
    inside_count = 0
    edge_values = []
    for sample_x, sample_y in samples:
        sx = int(round(sample_x))
        sy = int(round(sample_y))
        if 0 <= sx < w and 0 <= sy < h:
            if object_mask[sy, sx] > 0:
                inside_count += 1
            edge_values.append(edge_image[sy, sx] if edge_image is not None else 255)
    object_overlap = inside_count / max(len(samples), 1)
    edge_strength = float(np.mean(edge_values) / 255.0) if edge_values else 0.0

    max_reasonable_distance = max(18.0, object_width * 0.45)
    proximity_score = float(np.clip(1.0 - face_distance / max_reasonable_distance, 0.0, 1.0))
    length_score = float(np.clip((length - 18.0) / max(object_width * 1.2, 40.0), 0.0, 1.0))
    side_score = 1.0 if side_ok else 0.0
    overlap_score = float(np.clip(1.0 - object_overlap / 0.20, 0.0, 1.0))

    depth_confidence = 0.5
    depth_diff = None
    gripper_depth_median = None
    gripper_valid_ratio = 0.0
    depth_difference_from_object_median = None
    object_band_score = 0.5
    if depth_img is not None:
        sample_xy = tip_xy + away_xy * 8.0
        gripper_depth = local_depth_stats(
            depth_img,
            np.asarray([[sample_xy[1], sample_xy[0]]]),
            window=5,
        )
        object_depth = local_depth_stats(
            depth_img,
            np.asarray([object_contact_yx]),
            window=5,
        )
        if gripper_depth is None or object_depth is None:
            depth_confidence = 0.0
        else:
            gripper_depth_median = gripper_depth["median"]
            gripper_valid_ratio = float(min(1.0, gripper_depth["count"] / 121.0))
            depth_diff = abs(gripper_depth["median"] - object_depth["median"])
            depth_confidence = float(np.clip(1.0 - depth_diff / 90.0, 0.0, 1.0))
        if object_depth_stats and object_depth_stats.get("available"):
            if gripper_depth_median is None:
                depth_confidence = 0.0
            else:
                depth_difference_from_object_median = abs(gripper_depth_median - object_depth_stats["median"])
                object_band_score = depth_band_score(gripper_depth_median, object_depth_stats)

    score = (
        0.34 * proximity_score +
        0.18 * length_score +
        0.16 * side_score +
        0.12 * overlap_score +
        0.10 * edge_strength +
        0.10 * (depth_confidence * object_band_score)
    )

    reasons = []
    if face_distance > max_reasonable_distance:
        reasons.append("tip too far from object")
    if object_overlap > 0.25:
        reasons.append("line overlaps object")
    if not side_ok:
        reasons.append("wrong side/approach")
    if edge_strength < 0.05:
        reasons.append("weak edge support")
    if length < max(24.0, object_width * 0.22) and face_distance > 8.0:
        reasons.append("short slot/gap-like edge")
    if depth_img is not None and object_depth_stats and object_depth_stats.get("available"):
        if gripper_depth_median is None:
            reasons.append("invalid thin-finger depth")
        elif gripper_valid_ratio < MIN_VALID_DEPTH_RATIO:
            reasons.append("low thin-finger depth ratio")
        elif not depth_in_object_band(gripper_depth_median, object_depth_stats):
            if gripper_depth_median > object_depth_stats["band_high"]:
                reasons.append("thin finger behind object depth band")
            else:
                reasons.append("thin finger in front of object depth band")

    accepted = len(reasons) == 0 and score >= 0.42

    x_values = [p1[0], p2[0]]
    y_values = [p1[1], p2[1]]
    bbox = (
        max(0, int(min(x_values)) - 4),
        max(0, int(min(y_values)) - 4),
        min(w, int(max(x_values)) + 5),
        min(h, int(max(y_values)) + 5),
    )
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        1,
        thickness=3,
    )

    candidate.update({
        "bbox": bbox,
        "mask": mask,
        "score": float(score),
        "confidence": float(score),
        "accepted": accepted,
        "reasons": reasons,
        "face_distance": float(face_distance),
        "tip_yx": tip_yx.astype(float),
        "tail_yx": tail_yx.astype(float),
        "object_contact_yx": object_contact_yx.astype(float),
        "away_direction_xy": away_xy.astype(float),
        "line_length": length,
        "edge_strength": edge_strength,
        "object_overlap": float(object_overlap),
        "depth_confidence": float(depth_confidence),
        "candidate_depth_difference_mm": float(depth_diff) if depth_diff is not None else None,
        "candidate_depth_median_mm": float(gripper_depth_median) if gripper_depth_median is not None else None,
        "candidate_depth_valid_ratio": float(gripper_valid_ratio),
        "candidate_depth_difference_from_object_median_mm": (
            float(depth_difference_from_object_median)
            if depth_difference_from_object_median is not None
            else None
        ),
        "object_depth_band_low": object_depth_stats.get("band_low") if object_depth_stats else None,
        "object_depth_band_high": object_depth_stats.get("band_high") if object_depth_stats else None,
        "source": "thin_finger",
    })
    return candidate


def detect_thin_finger_body(image_rgb, side, roi, object_mask, object_box, object_points, depth_img=None, object_depth_stats=None):
    x1, y1, x2, y2 = roi
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_crop = cv2.Canny(blur, 35, 115)
    object_crop = object_mask[y1:y2, x1:x2].astype(np.uint8)
    object_dilated = cv2.dilate(object_crop, np.ones((5, 5), np.uint8), iterations=1)
    edges_crop[object_dilated > 0] = 0

    edges_full = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    edges_full[y1:y2, x1:x2] = edges_crop

    min_line_length = max(18, int(max(object_box[2] - object_box[0], object_box[3] - object_box[1]) * 0.18))
    lines = cv2.HoughLinesP(
        edges_crop,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=min_line_length,
        maxLineGap=12,
    )

    candidates = []
    if lines is not None:
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = line
            candidate = {
                "p1_xy": np.array([lx1 + x1, ly1 + y1], dtype=float),
                "p2_xy": np.array([lx2 + x1, ly2 + y1], dtype=float),
            }
            scored = score_thin_finger_candidate(
                candidate,
                side,
                image_rgb.shape,
                object_box,
                object_mask,
                object_points,
                edges_full,
                depth_img,
                object_depth_stats,
            )
            if scored is not None:
                candidates.append(scored)

    contours, _ = cv2.findContours(edges_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 8:
            continue
        points = contour.reshape(-1, 2).astype(float)
        vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        projections = (points[:, 0] - x0) * vx + (points[:, 1] - y0) * vy
        p_min = np.array([x0 + vx * np.min(projections), y0 + vy * np.min(projections)])
        p_max = np.array([x0 + vx * np.max(projections), y0 + vy * np.max(projections)])
        candidate = {
            "p1_xy": p_min + np.array([x1, y1], dtype=float),
            "p2_xy": p_max + np.array([x1, y1], dtype=float),
        }
        scored = score_thin_finger_candidate(
            candidate,
            side,
            image_rgb.shape,
            object_box,
            object_mask,
            object_points,
            edges_full,
            depth_img,
            object_depth_stats,
        )
        if scored is not None:
            candidates.append(scored)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    if not accepted:
        return {
            "side": side,
            "roi": roi,
            "prompt": None,
            "mask": None,
            "bbox": None,
            "rejected_bbox": candidates[0].get("bbox"),
            "candidates": candidates[:10],
            "rejected_candidates": candidates[:10],
            "confidence": candidates[0].get("confidence", 0.0),
            "mode": "thin_finger_failed",
            "warning": "Automatic gripper detection uncertain. Manual selection required.",
        }

    best = accepted[0]
    return {
        "side": side,
        "roi": roi,
        "prompt": None,
        "mask": best["mask"],
        "bbox": best["bbox"],
        "rejected_bbox": None,
        "candidates": candidates[:10],
        "rejected_candidates": [candidate for candidate in candidates[:10] if not candidate.get("accepted")],
        "confidence": best["confidence"],
        "mode": "thin_finger",
        "thin_line": (
            best["p1_xy"].astype(float).tolist(),
            best["p2_xy"].astype(float).tolist(),
        ),
        "thin_tip_yx": best["tip_yx"].astype(float).tolist(),
        "thin_tail_yx": best["tail_yx"].astype(float).tolist(),
        "gripper_direction_xy": best["away_direction_xy"].astype(float).tolist(),
        "warning": None if best["confidence"] >= 0.50 else "Automatic gripper detection uncertain.",
    }


def component_at_prompt(mask, roi, prompt):
    if mask is None:
        return None

    x1, y1, x2, y2 = roi
    restricted = np.zeros_like(mask, dtype=np.uint8)
    restricted[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(restricted, 8)
    if num_labels <= 1:
        return restricted if np.any(restricted) else None

    label = None
    if prompt is not None:
        px, py = int(prompt[0]), int(prompt[1])
        if 0 <= py < labels.shape[0] and 0 <= px < labels.shape[1]:
            prompt_label = labels[py, px]
            if prompt_label > 0:
                label = prompt_label

    if label is None:
        candidates = [
            (i, stats[i, cv2.CC_STAT_AREA])
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > 100
        ]
        if not candidates:
            return None
        label = max(candidates, key=lambda item: item[1])[0]

    return (labels == label).astype(np.uint8)


def detect_gripper_body(
    predictor,
    image_rgb,
    side,
    roi,
    manual_boxes=None,
    min_area=250,
    exclude_mask=None,
    object_box=None,
    object_points=None,
    depth_img=None,
    object_depth_stats=None,
):
    manual_boxes = manual_boxes or {}
    if side in manual_boxes:
        mask = rect_mask(image_rgb.shape, manual_boxes[side])
        bbox = mask_bbox(mask)
        ok, reason = gripper_body_bbox_status(side, bbox, image_rgb.shape, object_box)
        if not ok:
            print(reason)
            return {
                "side": side,
                "roi": manual_boxes[side],
                "prompt": None,
                "mask": None,
                "bbox": None,
                "rejected_bbox": bbox,
                "candidates": [],
                "rejected_candidates": [{"bbox": bbox, "reasons": [reason], "score": 0.0}],
                "confidence": 0.0,
                "mode": "manual_rejected",
                "warning": "Fallback rejected: likely support/base detection.",
            }
        return {
            "side": side,
            "roi": manual_boxes[side],
            "prompt": None,
            "mask": mask,
            "bbox": bbox,
            "rejected_bbox": None,
            "candidates": [],
            "rejected_candidates": [],
            "confidence": 0.75,
            "mode": "manual_box",
            "warning": "Manual gripper bounding box used.",
        }

    object_points = object_points if object_points is not None else np.empty((0, 2), dtype=int)
    scored_candidates = []
    raw_candidates = generate_gripper_candidates(
        image_rgb,
        roi,
        exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
        depth_img,
        object_depth_stats,
    )
    for candidate in raw_candidates:
        if box_area(candidate["bbox"]) < min_area:
            continue
        scored = score_gripper_candidate(
            candidate,
            side,
            image_rgb.shape,
            object_box,
            exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            object_points,
            depth_img,
            object_depth_stats,
        )
        if scored is not None:
            scored_candidates.append(scored)

    accepted = [c for c in scored_candidates if c.get("accepted")]
    if accepted:
        best = max(accepted, key=lambda item: item["score"])
        refined = refine_candidate_by_depth(
            best,
            side,
            image_rgb.shape,
            object_box,
            exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            object_points,
            depth_img,
            roi,
            object_depth_stats,
        )
        if refined is not best:
            scored_candidates.append(refined)
            best = refined
        trimmed = trim_candidate_to_inner_band(
            best,
            side,
            image_rgb.shape,
            object_box,
            exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            object_points,
            depth_img,
            object_depth_stats,
        )
        if trimmed is not best:
            scored_candidates.append(trimmed)
            best = trimmed
        thin_body = detect_thin_finger_body(
            image_rgb,
            side,
            roi,
            exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            object_box,
            object_points,
            depth_img,
            object_depth_stats,
        )
        if thin_body is not None and thin_body.get("mode") == "thin_finger" and (
            thin_body.get("confidence", 0.0) >= 0.58 and
            not reliable_block_candidate(best) and
            (best["confidence"] < 0.58 or thin_body.get("confidence", 0.0) > best["confidence"] + 0.08)
        ):
            thin_body["block_candidates"] = scored_candidates
            thin_body["block_confidence"] = best.get("confidence", 0.0)
            thin_body["thin_finger_confidence"] = thin_body.get("confidence", 0.0)
            thin_body["selected_mode_reason"] = "thin_finger selected: block confidence low or block depth/solidness unreliable"
            return thin_body
        if thin_body is not None and thin_body.get("bbox") is not None and reliable_block_candidate(best):
            thin_body["reasons"] = thin_body.get("reasons", []) + ["rejected thin_finger: reliable block candidate exists"]
        return {
            "side": side,
            "roi": roi,
            "prompt": None,
            "mask": best["mask"],
            "bbox": best["bbox"],
            "rejected_bbox": None,
            "candidates": scored_candidates,
            "rejected_candidates": [c for c in scored_candidates if not c.get("accepted")],
            "confidence": best["confidence"],
            "mode": "block",
            "block_confidence": best.get("confidence", 0.0),
            "thin_finger_confidence": thin_body.get("confidence", 0.0) if thin_body else 0.0,
            "selected_mode_reason": mode_reason_for_block(best, thin_body),
            "warning": None if best["confidence"] >= 0.55 else "Automatic gripper detection uncertain.",
        }

    prompt = find_prompt_in_roi(image_rgb, roi, min_area, exclude_mask=exclude_mask)
    sam_candidate = None
    if prompt is not None:
        raw_mask = sam_predict(predictor, prompt)
        body_mask = component_at_prompt(raw_mask, roi, prompt)
        if body_mask is not None and exclude_mask is not None:
            body_mask = np.logical_and(body_mask > 0, exclude_mask == 0).astype(np.uint8)
        body_bbox = mask_bbox(body_mask) if body_mask is not None else None
        if body_bbox is not None:
            sam_candidate = {
                "bbox": body_bbox,
                "mask": body_mask,
                "solidness": min(1.0, np.count_nonzero(body_mask) / max(box_area(body_bbox), 1)),
                "source": "sam",
            }
            sam_candidate = score_gripper_candidate(
                sam_candidate,
                side,
                image_rgb.shape,
                object_box,
                exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
                object_points,
                depth_img,
                object_depth_stats,
            )
            if sam_candidate is not None:
                scored_candidates.append(sam_candidate)
                if sam_candidate.get("accepted"):
                    refined = refine_candidate_by_depth(
                        sam_candidate,
                        side,
                        image_rgb.shape,
                        object_box,
                        exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
                        object_points,
                        depth_img,
                        roi,
                        object_depth_stats,
                    )
                    if refined is not sam_candidate:
                        scored_candidates.append(refined)
                        sam_candidate = refined
                    trimmed = trim_candidate_to_inner_band(
                        sam_candidate,
                        side,
                        image_rgb.shape,
                        object_box,
                        exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
                        object_points,
                        depth_img,
                        object_depth_stats,
                    )
                    if trimmed is not sam_candidate:
                        scored_candidates.append(trimmed)
                        sam_candidate = trimmed
                    thin_body = detect_thin_finger_body(
                        image_rgb,
                        side,
                        roi,
                        exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
                        object_box,
                        object_points,
                        depth_img,
                        object_depth_stats,
                    )
                    if thin_body is not None and thin_body.get("mode") == "thin_finger" and (
                        thin_body.get("confidence", 0.0) >= 0.58 and
                        not reliable_block_candidate(sam_candidate) and
                        (sam_candidate["confidence"] < 0.58 or thin_body.get("confidence", 0.0) > sam_candidate["confidence"] + 0.08)
                    ):
                        thin_body["block_candidates"] = scored_candidates
                        thin_body["block_confidence"] = sam_candidate.get("confidence", 0.0)
                        thin_body["thin_finger_confidence"] = thin_body.get("confidence", 0.0)
                        thin_body["selected_mode_reason"] = "thin_finger selected: SAM block confidence low or depth/solidness unreliable"
                        return thin_body
                    return {
                        "side": side,
                        "roi": roi,
                        "prompt": prompt,
                        "mask": sam_candidate["mask"],
                        "bbox": sam_candidate["bbox"],
                        "rejected_bbox": None,
                        "candidates": scored_candidates,
                        "rejected_candidates": [c for c in scored_candidates if not c.get("accepted")],
                        "confidence": sam_candidate["confidence"],
                        "mode": "block_sam",
                        "block_confidence": sam_candidate.get("confidence", 0.0),
                        "thin_finger_confidence": thin_body.get("confidence", 0.0) if thin_body else 0.0,
                        "selected_mode_reason": mode_reason_for_block(sam_candidate, thin_body),
                        "warning": None if sam_candidate["confidence"] >= 0.55 else "Automatic gripper detection uncertain.",
                    }

    thin_body = detect_thin_finger_body(
        image_rgb,
        side,
        roi,
        exclude_mask if exclude_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
        object_box,
        object_points,
        depth_img,
        object_depth_stats,
    )
    if thin_body is not None and thin_body.get("mode") == "thin_finger":
        best_block = max(scored_candidates, key=lambda item: item.get("confidence", 0.0), default={})
        if best_block and thin_body.get("bbox") is not None and best_block.get("bbox") is not None:
            if best_block.get("confidence", 0.0) >= 0.45 and box_overlap_fraction(thin_body["bbox"], best_block["bbox"]) > 0.55:
                thin_body["rejected_candidates"] = thin_body.get("rejected_candidates", []) + [{
                    "bbox": thin_body.get("bbox"),
                    "reasons": ["rejected thin_finger: slot/gap inside block-like candidate"],
                    "score": thin_body.get("confidence", 0.0),
                }]
                rejected = sorted(scored_candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:8]
                rejected.extend(thin_body.get("rejected_candidates", [])[:4])
                return {
                    "side": side,
                    "roi": roi,
                    "prompt": prompt,
                    "mask": None,
                    "bbox": None,
                    "rejected_bbox": thin_body.get("bbox"),
                    "candidates": scored_candidates,
                    "rejected_candidates": rejected,
                    "confidence": best_block.get("confidence", 0.0),
                    "mode": "failed",
                    "block_confidence": best_block.get("confidence", 0.0),
                    "thin_finger_confidence": thin_body.get("confidence", 0.0),
                    "selected_mode_reason": "thin_finger rejected as likely slot/gap inside block; no reliable block accepted",
                    "warning": "Automatic gripper detection uncertain. Manual selection required.",
                }
        thin_body["block_candidates"] = scored_candidates
        thin_body["block_confidence"] = best_block.get("confidence", 0.0)
        thin_body["thin_finger_confidence"] = thin_body.get("confidence", 0.0)
        thin_body["selected_mode_reason"] = "thin_finger selected: no accepted block candidate"
        return thin_body

    rejected = sorted(scored_candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:8]
    if thin_body is not None:
        rejected.extend(thin_body.get("rejected_candidates", [])[:4])
    warning = "Automatic gripper detection uncertain."
    if rejected:
        best_rejected = rejected[0]
        reasons = best_rejected.get("reasons") or []
        if any("support/base" in reason or "below" in reason for reason in reasons):
            print("Rejected gripper fallback: detected support/base instead of metal jaw.")

    return {
        "side": side,
        "roi": roi,
        "prompt": prompt,
        "mask": None,
        "bbox": None,
        "rejected_bbox": rejected[0]["bbox"] if rejected else None,
        "candidates": scored_candidates,
        "rejected_candidates": rejected,
        "confidence": rejected[0].get("confidence", 0.0) if rejected else 0.0,
        "mode": "failed",
        "warning": f"{warning} Manual selection required.",
    }


def gripper_inner_face_points(body_mask, side, object_box=None, object_points=None):
    pts = mask_boundary(body_mask)
    if len(pts) == 0:
        return pts

    if object_box is not None:
        ox1, oy1, ox2, oy2 = object_box
        y_pad = 40
        near_object_y = pts[
            (pts[:, 0] >= oy1 - y_pad) &
            (pts[:, 0] <= oy2 + y_pad)
        ]
        if len(near_object_y) >= 5:
            pts = near_object_y

    if side == "left":
        x_cut = np.percentile(pts[:, 1], 92)
        face = pts[pts[:, 1] >= x_cut]
    else:
        x_cut = np.percentile(pts[:, 1], 8)
        face = pts[pts[:, 1] <= x_cut]

    if len(face) < 3:
        face = pts

    if object_points is not None and len(object_points) > 0 and len(face) > 0:
        nearest_dists = []
        for face_p in face:
            nearest_dists.append(float(np.min(np.linalg.norm(object_points - face_p, axis=1))))
        nearest_dists = np.asarray(nearest_dists)
        min_dist = float(np.min(nearest_dists))
        close_face = face[nearest_dists <= min_dist + 12.0]
        if len(close_face) >= 2:
            face = close_face
    return face


def block_face_overlap_contact(side, body_info, object_mask, object_points, object_box):
    if body_info is None or body_info.get("bbox") is None or object_box is None or object_mask is None:
        return None

    x1, y1, x2, y2 = body_info["bbox"]
    ox1, oy1, ox2, oy2 = object_box
    face_x = x2 - 1 if side == "left" else x1

    overlap_y1 = max(y1, oy1)
    overlap_y2 = min(y2, oy2)
    if overlap_y2 <= overlap_y1:
        return None

    rows = []
    for y in range(int(overlap_y1), int(overlap_y2)):
        xs = np.where(object_mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        side_x = int(np.min(xs)) if side == "left" else int(np.max(xs))
        if side == "left" and face_x <= side_x + 45:
            rows.append((y, side_x))
        elif side == "right" and face_x >= side_x - 45:
            rows.append((y, side_x))

    if len(rows) < 4:
        return None

    ys = np.asarray([row[0] for row in rows], dtype=int)
    side_xs = np.asarray([row[1] for row in rows], dtype=int)

    # Avoid the lower slot/gap region for side-approaching block jaws.
    low_cut = int(np.percentile(ys, 85))
    keep = ys <= low_cut
    if np.count_nonzero(keep) >= 4:
        ys = ys[keep]
        side_xs = side_xs[keep]

    target_y = int(np.median(ys))
    nearest_row = int(np.argmin(np.abs(ys - target_y)))
    obj_x = int(side_xs[nearest_row])
    obj_p = np.array([target_y, obj_x], dtype=float)
    if side == "left" and face_x >= obj_x - MIN_OBJECT_GRIPPER_SAMPLE_DISTANCE_PX:
        face_x = max(0, obj_x - int(MIN_OBJECT_GRIPPER_SAMPLE_DISTANCE_PX + 2))
    elif side == "right" and face_x <= obj_x + MIN_OBJECT_GRIPPER_SAMPLE_DISTANCE_PX:
        face_x = min(object_mask.shape[1] - 1, obj_x + int(MIN_OBJECT_GRIPPER_SAMPLE_DISTANCE_PX + 2))
    face_p = np.array([target_y, face_x], dtype=float)

    face_ys = np.arange(int(np.min(ys)), int(np.max(ys)) + 1)
    face_points = np.column_stack([face_ys, np.full_like(face_ys, int(face_x))]).astype(float)
    return {
        "object_point": obj_p,
        "face_point": face_p,
        "face_points": face_points,
        "overlap_segment": ((int(face_x), int(np.min(ys))), (int(face_x), int(np.max(ys)))),
        "selected_overlap_midpoint": (int(face_x), int(target_y)),
    }


def associate_face_to_object(
    object_points,
    face_points,
    depth_img=None,
    object_mask=None,
    gripper_direction_xy=None,
    object_depth_stats=None,
):
    if len(object_points) == 0 or len(face_points) == 0:
        return None, None, None, None, None, None, 0.0, False, "Missing object or gripper face points.", None

    evaluated = []
    for face_p in face_points:
        dists = np.linalg.norm(object_points - face_p, axis=1)
        idx = int(np.argmin(dists))
        obj_p = object_points[idx]
        candidate = evaluate_contact_candidate(
            obj_p,
            face_p,
            depth_img=depth_img,
            object_mask=object_mask,
            gripper_direction_xy=gripper_direction_xy,
            object_depth_stats=object_depth_stats,
        )
        evaluated.append(candidate)

    valid_candidates = [candidate for candidate in evaluated if candidate["valid"]]
    if valid_candidates:
        best = max(valid_candidates, key=lambda item: item["score"])
    else:
        best = max(evaluated, key=lambda item: item["score"])

    obj_p = best["object_point"]
    face_p = best["face_point"]
    pixel_dist = best["pixel_dist"]
    obj_depth = best["object_depth_mm"]
    face_depth = best["face_depth_mm"]
    depth_diff = best["depth_difference_mm"]
    depth_confidence = best["depth_confidence"]
    valid = best["valid"]
    if valid:
        warning = None
    elif best["depth_available"]:
        reason = "; ".join(best["reasons"]) if best["reasons"] else "no RGB-D valid contact candidate"
        warning = f"RGB-D contact uncertain/invalid: {reason}."
    else:
        warning = "Depth unavailable, geometry-only contact selection."

    return obj_p, face_p, pixel_dist, obj_depth, face_depth, depth_diff, depth_confidence, valid, warning, best


def build_gripper_contact(side, body_info, object_points, object_box, depth_img=None, object_mask=None, object_depth_stats=None):
    if body_info is None or body_info.get("mask") is None:
        return {
            "side": side,
            "object_point": None,
            "face_point": None,
            "distance_px": None,
            "face_points": np.empty((0, 2), dtype=int),
            "body": body_info,
            "valid": False,
            "warning": "No gripper body detected.",
            "selection_source": "failed",
            "depth_affected_selection": False,
        }

    ray_points = np.empty((0, 2), dtype=float)
    if body_info.get("mode") == "thin_finger" and body_info.get("thin_tip_yx") is not None:
        face_points = np.asarray([body_info["thin_tip_yx"]], dtype=float)
        direction_xy = np.asarray(body_info.get("gripper_direction_xy", [0.0, 0.0]), dtype=float)
        toward_object_xy = -direction_xy
        norm = np.linalg.norm(toward_object_xy)
        if norm > 1e-6:
            toward_object_xy = toward_object_xy / norm
            tip_yx = np.asarray(body_info["thin_tip_yx"], dtype=float)
            tip_xy = np.array([tip_yx[1], tip_yx[0]], dtype=float)
            ray = []
            h = int(np.max(object_points[:, 0]) + 2) if len(object_points) else 0
            w = int(np.max(object_points[:, 1]) + 2) if len(object_points) else 0
            for step in range(1, 45):
                sample_xy = tip_xy + toward_object_xy * step
                sample_yx = np.array([sample_xy[1], sample_xy[0]], dtype=float)
                ray.append(sample_yx)
                if len(object_points) > 0:
                    nearest_dist = float(np.min(np.linalg.norm(object_points - sample_yx, axis=1)))
                    if nearest_dist <= 2.0:
                        idx = int(np.argmin(np.linalg.norm(object_points - sample_yx, axis=1)))
                        face_p = tip_yx
                        obj_p = object_points[idx]
                        dist = float(np.linalg.norm(obj_p - face_p))
                        candidate = evaluate_contact_candidate(
                            obj_p,
                            face_p,
                            depth_img=depth_img,
                            object_mask=object_mask,
                            gripper_direction_xy=body_info.get("gripper_direction_xy"),
                            object_depth_stats=object_depth_stats,
                        )
                        obj_depth = candidate["object_depth_mm"]
                        face_depth = candidate["face_depth_mm"]
                        depth_diff = candidate["depth_difference_mm"]
                        depth_confidence = candidate["depth_confidence"]
                        valid = candidate["valid"]
                        warning = None if valid else (
                            "RGB-D contact uncertain/invalid: "
                            + ("; ".join(candidate["reasons"]) if candidate["reasons"] else "no RGB-D valid contact candidate")
                            + "."
                        )
                        ray_points = np.asarray(ray, dtype=float)
                        return {
                            "side": side,
                            "object_point": candidate["object_point"],
                            "face_point": candidate["face_point"],
                            "raw_face_point": candidate["raw_face_point"],
                            "distance_px": candidate["pixel_dist"],
                            "object_depth_mm": obj_depth,
                            "face_depth_mm": face_depth,
                            "depth_difference_mm": depth_diff,
                            "depth_confidence": depth_confidence,
                            "object_depth_valid_ratio": candidate["object_depth_valid_ratio"],
                            "gripper_depth_valid_ratio": candidate["gripper_depth_valid_ratio"],
                            "gripper_depth_difference_from_object_median_mm": candidate["gripper_depth_difference_from_object_median_mm"],
                            "object_depth_band_low": candidate["object_depth_band_low"],
                            "object_depth_band_high": candidate["object_depth_band_high"],
                            "gripper_sample_inside_object": candidate["gripper_sample_inside_object"],
                            "near_same_sample": candidate["near_same_sample"],
                            "gripper_depth_sample_shifted": candidate["sample_shifted"],
                            "face_points": face_points,
                            "ray_points": ray_points,
                            "body": body_info,
                            "valid": valid,
                            "warning": warning,
                            "selection_source": candidate["selection_source"],
                            "depth_affected_selection": candidate["depth_affected_selection"],
                        }
            ray_points = np.asarray(ray, dtype=float)
    else:
        face_points = gripper_inner_face_points(body_info["mask"], side, object_box, object_points)
        if body_info.get("mode") in ("block", "block_sam", "manual_box"):
            overlap = block_face_overlap_contact(side, body_info, object_mask, object_points, object_box)
            if overlap is not None:
                o_obj, o_face, o_dist, o_obj_depth, o_face_depth, o_depth_diff, o_depth_conf, o_valid, o_warning, candidate = associate_face_to_object(
                    object_points,
                    overlap["face_points"],
                    depth_img,
                    object_mask,
                    body_info.get("gripper_direction_xy"),
                    object_depth_stats,
                )
                if candidate is not None and (o_valid or o_dist <= CONTACT_PIXEL_THRESHOLD_PX):
                    overlap_warning = None if o_valid else (
                        "Block overlap segment selected, but RGB-D validation is uncertain: "
                        + ("; ".join(candidate["reasons"]) if candidate.get("reasons") else "check debug view")
                        + "."
                    )
                    return {
                        "side": side,
                        "object_point": o_obj,
                        "face_point": o_face,
                        "raw_face_point": candidate.get("raw_face_point"),
                        "distance_px": o_dist,
                        "object_depth_mm": o_obj_depth,
                        "face_depth_mm": o_face_depth,
                        "depth_difference_mm": o_depth_diff,
                        "depth_confidence": o_depth_conf,
                        "object_depth_valid_ratio": candidate["object_depth_valid_ratio"],
                        "gripper_depth_valid_ratio": candidate["gripper_depth_valid_ratio"],
                        "gripper_depth_difference_from_object_median_mm": candidate["gripper_depth_difference_from_object_median_mm"],
                        "object_depth_band_low": candidate["object_depth_band_low"],
                        "object_depth_band_high": candidate["object_depth_band_high"],
                        "gripper_sample_inside_object": candidate["gripper_sample_inside_object"],
                        "near_same_sample": candidate["near_same_sample"],
                        "gripper_depth_sample_shifted": candidate["sample_shifted"],
                        "face_points": overlap["face_points"],
                        "overlap_segment": overlap["overlap_segment"],
                        "selected_overlap_midpoint": overlap["selected_overlap_midpoint"],
                        "ray_points": ray_points,
                        "body": body_info,
                        "valid": bool(o_valid),
                        "warning": overlap_warning,
                        "selection_source": "RGB-D validated block overlap segment" if o_valid else "block overlap segment, RGB-D uncertain",
                        "depth_affected_selection": candidate["depth_affected_selection"],
                    }
    obj_p, face_p, dist, obj_depth, face_depth, depth_diff, depth_confidence, valid, warning, candidate = associate_face_to_object(
        object_points,
        face_points,
        depth_img,
        object_mask,
        body_info.get("gripper_direction_xy"),
        object_depth_stats,
    )
    if body_info.get("warning"):
        warning = body_info["warning"] if warning is None else f"{body_info['warning']} {warning}"

    return {
        "side": side,
        "object_point": obj_p,
        "face_point": face_p,
        "raw_face_point": candidate.get("raw_face_point") if candidate else None,
        "distance_px": dist,
        "object_depth_mm": obj_depth,
        "face_depth_mm": face_depth,
        "depth_difference_mm": depth_diff,
        "depth_confidence": depth_confidence,
        "object_depth_valid_ratio": candidate.get("object_depth_valid_ratio") if candidate else 0.0,
        "gripper_depth_valid_ratio": candidate.get("gripper_depth_valid_ratio") if candidate else 0.0,
        "gripper_depth_difference_from_object_median_mm": candidate.get("gripper_depth_difference_from_object_median_mm") if candidate else None,
        "object_depth_band_low": candidate.get("object_depth_band_low") if candidate else None,
        "object_depth_band_high": candidate.get("object_depth_band_high") if candidate else None,
        "gripper_sample_inside_object": candidate.get("gripper_sample_inside_object") if candidate else False,
        "near_same_sample": candidate.get("near_same_sample") if candidate else False,
        "gripper_depth_sample_shifted": candidate.get("sample_shifted") if candidate else False,
        "face_points": face_points,
        "ray_points": ray_points,
        "body": body_info,
        "valid": valid,
        "warning": warning,
        "selection_source": candidate.get("selection_source") if candidate else "failed",
        "depth_affected_selection": candidate.get("depth_affected_selection") if candidate else False,
    }


def fingertip_boundary(mask, side):
    pts = mask_boundary(mask)

    if len(pts) == 0:
        return pts

    ys = pts[:, 0]

    # Use only lower 25% of gripper boundary
    y_cut = np.percentile(ys, 75)
    pts = pts[pts[:, 0] >= y_cut]

    if len(pts) == 0:
        return mask_boundary(mask)

    if side == "left":
        # left gripper inner side faces right
        x_cut = np.percentile(pts[:, 1], 70)
        pts = pts[pts[:, 1] >= x_cut]
    else:
        # right gripper inner side faces left
        x_cut = np.percentile(pts[:, 1], 30)
        pts = pts[pts[:, 1] <= x_cut]

    return pts


def closest_points_depth_aware(object_points, gripper_points, depth_img=None, top_k=30):
    if len(object_points) == 0 or len(gripper_points) == 0:
        return None, None, None

    candidates = []

    for obj_p in object_points:
        dists = np.linalg.norm(gripper_points - obj_p, axis=1)
        nearest_indices = np.argsort(dists)[:top_k]

        for idx in nearest_indices:
            grip_p = gripper_points[idx]
            pixel_dist = float(dists[idx])

            obj_depth = get_depth_mm(depth_img, obj_p)
            grip_depth = get_depth_mm(depth_img, grip_p)

            if obj_depth is not None and grip_depth is not None:
                depth_diff = abs(obj_depth - grip_depth)
            else:
                depth_diff = 999.0

            # weighted score: prioritize close pixels, then matching depth
            score = pixel_dist + 0.05 * depth_diff

            candidates.append((score, obj_p, grip_p, pixel_dist, depth_diff))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0])

    best_score, best_obj, best_grip, best_pixel_dist, best_depth_diff = candidates[0]

    return best_obj, best_grip, best_pixel_dist


def save_contact_debug(image_rgb, object_mask, left_mask, right_mask, contacts):
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)

    plt.imshow(object_mask, alpha=0.25)

    if left_mask is not None:
        plt.imshow(left_mask, alpha=0.20)

    if right_mask is not None:
        plt.imshow(right_mask, alpha=0.20)

    for name, obj_p, grip_p, dist in contacts:
        if obj_p is None or grip_p is None:
            continue

        color = "red" if name == "left" else "blue"

        plt.scatter(obj_p[1], obj_p[0], c=color, s=150, marker="o", label=f"{name} object contact")
        plt.scatter(grip_p[1], grip_p[0], c=color, s=100, marker="x", label=f"{name} gripper point")
        plt.plot([obj_p[1], grip_p[1]], [obj_p[0], grip_p[0]], color=color, linewidth=2)

    plt.title("Contact Debug View")
    plt.legend(fontsize=8)
    plt.axis("off")
    plt.savefig("output/contact_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, box, color, label=None, linewidth=2, linestyle="-"):
    if box is None:
        return
    x1, y1, x2, y2 = box
    ax.plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


def draw_gripper_contact_annotations(ax, gripper_results, include_text=True):
    if not gripper_results:
        return

    for side, result in gripper_results.items():
        if not result:
            continue

        color = "red" if side == "left" else "blue"
        body = result.get("body") or {}
        for candidate in body.get("block_candidates", [])[:8]:
            draw_box(ax, candidate.get("bbox"), "darkgreen", None, linewidth=1, linestyle=":")
        for candidate in body.get("candidates", [])[:12]:
            if candidate.get("source") == "thin_finger" and candidate.get("p1_xy") is not None:
                p1 = candidate["p1_xy"]
                p2 = candidate["p2_xy"]
                candidate_color = color if candidate.get("accepted") else "gray"
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=candidate_color, linewidth=1, linestyle=":")
            else:
                candidate_color = color if candidate.get("accepted") else "gray"
                linestyle = "-" if candidate.get("accepted") else ":"
                draw_box(
                    ax,
                    candidate.get("bbox"),
                    candidate_color,
                    None,
                    linewidth=1,
                    linestyle=linestyle,
                )
        draw_box(ax, body.get("bbox"), color, f"{side} gripper body", linewidth=2.5)
        if body.get("thin_line") is not None:
            p1, p2 = body["thin_line"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3, label=f"{side} thin finger")
            tip_yx = body.get("thin_tip_yx")
            if tip_yx is not None:
                ax.scatter(tip_yx[1], tip_yx[0], c=color, s=180, marker="*", label=f"{side} finger tip")
        draw_box(
            ax,
            body.get("rejected_bbox"),
            "orange",
            f"rejected {side} body",
            linewidth=2,
            linestyle=":",
        )

        face_points = result.get("face_points")
        if face_points is not None and len(face_points) > 0:
            stride = max(1, len(face_points) // 80)
            shown = face_points[::stride]
            ax.scatter(
                shown[:, 1],
                shown[:, 0],
                c=color,
                s=10,
                marker=".",
                alpha=0.9,
                label=f"{side} inner face",
            )

        obj_p = result.get("object_point")
        face_p = result.get("face_point")
        ray_points = result.get("ray_points")
        overlap_segment = result.get("overlap_segment")
        if overlap_segment is not None:
            (sx, sy1), (_, sy2) = overlap_segment
            ax.plot([sx, sx], [sy1, sy2], color="magenta", linewidth=4, alpha=0.85, label=f"{side} face overlap")
            midpoint = result.get("selected_overlap_midpoint")
            if midpoint is not None:
                ax.scatter(midpoint[0], midpoint[1], c="magenta", s=160, marker="s", label=f"{side} overlap midpoint")
        if ray_points is not None and len(ray_points) > 1:
            ax.plot(
                ray_points[:, 1],
                ray_points[:, 0],
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"{side} contact ray",
            )
        if obj_p is not None and face_p is not None:
            ax.scatter(obj_p[1], obj_p[0], c=color, s=160, marker="o")
            ax.scatter(face_p[1], face_p[0], c=color, s=120, marker="x")
            ax.plot(
                [obj_p[1], face_p[1]],
                [obj_p[0], face_p[0]],
                color=color,
                linewidth=2.2,
            )

            if include_text:
                state = "valid" if result.get("valid") else "uncertain"
                dist = result.get("distance_px")
                depth_diff = result.get("depth_difference_mm")
                dist_text = f"{dist:.1f}px" if dist is not None else "n/a"
                text = f"{side}: {state}\n{dist_text}"
                if depth_diff is not None:
                    text += f", {depth_diff:.1f}mm"
                confidence = body.get("confidence")
                mode = body.get("mode")
                if confidence is not None:
                    text += f"\nconf {confidence:.2f}"
                if mode:
                    text += f"\n{mode}"
                ax.text(
                    face_p[1] + 8,
                    face_p[0],
                    text,
                    color=color,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": color},
                )


def save_gripper_detection_debug(
    image_rgb,
    object_mask,
    object_box,
    left_roi,
    right_roi,
    gripper_results,
    object_depth_stats=None,
    path="output/gripper_detection_debug.png",
):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(image_rgb)
    ax.imshow(object_mask, alpha=0.26)

    draw_box(ax, object_box, "yellow", "object bbox", linewidth=2)
    draw_box(ax, left_roi, "cyan", "left ROI", linewidth=2, linestyle="--")
    draw_box(ax, right_roi, "lime", "right ROI", linewidth=2, linestyle="--")

    for side, result in (gripper_results or {}).items():
        if not result:
            continue
        body = result.get("body") or {}
        for candidate in body.get("block_candidates", [])[:8]:
            draw_box(ax, candidate.get("bbox"), "darkgreen", None, linewidth=1, linestyle=":")
        for candidate in body.get("candidates", [])[:16]:
            candidate_color = "green" if candidate.get("accepted") else "gray"
            if candidate.get("source") == "thin_finger" and candidate.get("p1_xy") is not None:
                p1 = candidate["p1_xy"]
                p2 = candidate["p2_xy"]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color=candidate_color,
                    linewidth=1.2,
                    linestyle="-" if candidate.get("accepted") else ":",
                    label=f"{side} thin candidate" if candidate.get("accepted") else None,
                )
            else:
                draw_box(
                    ax,
                    candidate.get("bbox"),
                    candidate_color,
                    f"{side} candidate" if candidate.get("accepted") else None,
                    linewidth=1,
                    linestyle="-" if candidate.get("accepted") else ":",
                )
            if candidate.get("bbox") is not None and candidate.get("candidate_depth_difference_from_object_median_mm") is not None:
                x1, y1, _, _ = candidate["bbox"]
                ax.text(
                    x1,
                    y1 - 3,
                    f"dZobj {candidate['candidate_depth_difference_from_object_median_mm']:.0f}mm",
                    color=candidate_color,
                    fontsize=6.5,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": candidate_color},
                )
        mask = body.get("mask")
        if mask is not None:
            ax.imshow(mask, alpha=0.18)
        draw_box(
            ax,
            body.get("rejected_bbox"),
            "orange",
            f"rejected {side} box",
            linewidth=2.5,
            linestyle=":",
        )

    draw_gripper_contact_annotations(ax, gripper_results, include_text=True)

    warnings = []
    if object_depth_stats and object_depth_stats.get("available"):
        warnings.append(
            "Object depth band: "
            f"{object_depth_stats['band_low']:.1f}-{object_depth_stats['band_high']:.1f} mm "
            f"(median {object_depth_stats['median']:.1f}, std {object_depth_stats['std']:.1f})"
        )
    elif object_depth_stats:
        warnings.append(object_depth_stats.get("warning", "geometry-only: depth unreliable"))
    warnings.extend([
        result.get("warning")
        for result in (gripper_results or {}).values()
        if result and result.get("warning")
    ])
    if warnings:
        ax.text(
            0.02,
            0.98,
            "\n".join(warnings),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "orange"},
        )

    ax.set_title("Gripper Body Detection and Inner Contact Faces")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def closest_points(object_points, gripper_points):
    if len(object_points) == 0 or len(gripper_points) == 0:
        return None, None, None

    min_dist = float("inf")
    best_obj = None
    best_grip = None

    for p in object_points:
        dists = np.linalg.norm(gripper_points - p, axis=1)
        idx = np.argmin(dists)

        if dists[idx] < min_dist:
            min_dist = dists[idx]
            best_obj = p
            best_grip = gripper_points[idx]

    return best_obj, best_grip, min_dist


def get_object_orientation(mask):
    pts = np.column_stack(np.where(mask > 0))  # [y, x]

    if len(pts) < 10:
        return 0.0, None, None

    mean = np.mean(pts, axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    main_vec = eigvecs[:, np.argmax(eigvals)]  # [y, x]
    main_vec = main_vec / np.linalg.norm(main_vec)

    angle_rad = np.arctan2(main_vec[0], main_vec[1])
    angle_deg = np.degrees(angle_rad)

    return angle_deg, mean, main_vec


def ordered_mask_contour_yx(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=float)

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)  # [x, y]
    return contour[:, ::-1].astype(float)  # [y, x]


def smooth_closed_points(points, window=9):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    half = window // 2
    smoothed = []
    for i in range(len(points)):
        idx = [(i + offset) % len(points) for offset in range(-half, half + 1)]
        smoothed.append(np.mean(points[idx], axis=0))
    return np.asarray(smoothed)


def contour_roughness(mask):
    contour = ordered_mask_contour_yx(mask)
    if len(contour) < 12:
        return 0.0

    smooth = smooth_closed_points(contour, 11)
    residual = np.linalg.norm(contour - smooth, axis=1)
    perimeter = cv2.arcLength(contour[:, ::-1].astype(np.float32), True)
    scale = max(perimeter / len(contour), 1.0)
    return float(np.mean(residual) / scale)


def contact_normal_info(object_mask, contact_point, object_centroid=None, window=18):
    contour = ordered_mask_contour_yx(object_mask)

    if contact_point is None or len(contour) < 8:
        inward = np.array([0.0, 1.0])
        return {
            "inward": inward,
            "outward": -inward,
            "confidence": 0.0,
            "index": None,
        }

    smooth = smooth_closed_points(contour, 9)
    dists = np.linalg.norm(smooth - contact_point, axis=1)
    idx = int(np.argmin(dists))
    half = max(3, int(window // 2))
    local_indices = [(idx + offset) % len(smooth) for offset in range(-half, half + 1)]
    local_pts = smooth[local_indices]

    if len(local_pts) < 5:
        inward = np.array([0.0, 1.0])
        return {
            "inward": inward,
            "outward": -inward,
            "confidence": 0.0,
            "index": idx,
        }

    mean = np.mean(local_pts, axis=0)
    centered = local_pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)

    tangent = eigvecs[:, order[-1]]
    tangent = tangent / np.linalg.norm(tangent)

    normal = np.array([-tangent[1], tangent[0]])

    if np.linalg.norm(normal) == 0:
        inward = np.array([0.0, 1.0])
        return {
            "inward": inward,
            "outward": -inward,
            "confidence": 0.0,
            "index": idx,
        }

    normal = normal / np.linalg.norm(normal)

    # Make normal point inward toward object center.
    if object_centroid is not None:
        inward_to_center = object_centroid - contact_point
        if np.dot(normal, inward_to_center) < 0:
            normal = -normal

    eigvals = np.real(eigvals)
    major = max(float(np.max(eigvals)), 1e-9)
    minor = max(float(np.min(eigvals)), 0.0)
    confidence = float(np.clip(1.0 - minor / major, 0.0, 1.0))

    return {
        "inward": normal,
        "outward": -normal,
        "confidence": confidence,
        "index": idx,
    }


def contact_normal(object_mask, contact_point, object_centroid=None, window=18):
    return contact_normal_info(object_mask, contact_point, object_centroid, window)["outward"]


def contact_normal_info_with_gripper(
    object_mask,
    contact_point,
    object_centroid=None,
    body_info=None,
    window=18,
):
    contour_info = contact_normal_info(object_mask, contact_point, object_centroid, window)

    if not body_info or body_info.get("mode") != "thin_finger" or body_info.get("gripper_direction_xy") is None:
        contour_info["source"] = "contour"
        return contour_info

    direction_xy = np.asarray(body_info["gripper_direction_xy"], dtype=float)
    if np.linalg.norm(direction_xy) < 1e-6:
        contour_info["source"] = "contour"
        return contour_info

    # Stored direction points from the contact tip back along the finger.
    # The contact force direction points the opposite way, from gripper tip into the object.
    gripper_inward_xy = -direction_xy / np.linalg.norm(direction_xy)
    gripper_inward_yx = np.array([gripper_inward_xy[1], gripper_inward_xy[0]], dtype=float)

    if object_centroid is not None:
        to_center = object_centroid - contact_point
        if np.dot(gripper_inward_yx, to_center) < 0:
            gripper_inward_yx = -gripper_inward_yx

    contour_inward = np.asarray(contour_info["inward"], dtype=float)
    if np.dot(contour_inward, gripper_inward_yx) < 0:
        contour_inward = -contour_inward

    contour_conf = float(contour_info.get("confidence", 0.0))
    gripper_conf = float(np.clip(body_info.get("confidence", 0.65), 0.35, 0.95))

    if contour_conf < 0.35:
        inward = gripper_inward_yx
        confidence = gripper_conf
        source = "gripper_direction"
    else:
        inward = contour_inward * contour_conf + gripper_inward_yx * gripper_conf
        inward = unit_vector(inward)
        confidence = float(np.clip((contour_conf + gripper_conf) / 2.0, 0.0, 1.0))
        source = "blended"

    return {
        "inward": inward,
        "outward": -inward,
        "confidence": confidence,
        "index": contour_info.get("index"),
        "source": source,
        "contour_confidence": contour_conf,
        "gripper_direction_confidence": gripper_conf,
    }


def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def rotate_2d(v, angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        c * v[0] - s * v[1],
        s * v[0] + c * v[1]
    ])


def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm


def object_radius_from_mask(mask, centroid_yx):
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) == 0:
        return 1.0
    dists = np.linalg.norm(pts - centroid_yx, axis=1)
    radius = float(np.percentile(dists, 95))
    return max(radius, 1.0)


def contact_yx_to_epsilon_contact(contact_yx, normal_yx, normal_is_inward=False):
    contact_yx = np.asarray(contact_yx, dtype=float)
    normal_yx = np.asarray(normal_yx, dtype=float)

    contact_xy = np.array([contact_yx[1], contact_yx[0]], dtype=float)
    normal_xy = np.array([normal_yx[1], normal_yx[0]], dtype=float)
    inward_normal_xy = normal_xy if normal_is_inward else -normal_xy

    return {
        "point": contact_xy,
        "inward_normal": inward_normal_xy
    }


def candidate_points_near_contact(object_mask, contact_yx, count=7, spacing=4):
    contour = ordered_mask_contour_yx(object_mask)
    if contact_yx is None or len(contour) == 0:
        return []

    smooth = smooth_closed_points(contour, 9)
    idx = int(np.argmin(np.linalg.norm(smooth - contact_yx, axis=1)))
    half = count // 2
    candidates = []
    used = set()
    for offset in range(-half, half + 1):
        cand_idx = (idx + offset * spacing) % len(smooth)
        key = int(cand_idx)
        if key in used:
            continue
        used.add(key)
        candidates.append(smooth[cand_idx])
    return candidates


def depth_variance_near_points(depth_img, points_yx, window=7):
    if depth_img is None:
        return None

    values = []
    h, w = depth_img.shape[:2]
    for point in points_yx:
        if point is None:
            continue
        y, x = int(point[0]), int(point[1])
        x1 = max(0, x - window)
        x2 = min(w, x + window + 1)
        y1 = max(0, y - window)
        y2 = min(h, y + window + 1)
        patch = depth_img[y1:y2, x1:x2]
        valid = patch[(patch > 0) & (patch < 10000)]
        if len(valid) > 0:
            values.extend(valid.astype(float).tolist())

    if len(values) < 2:
        return None
    return float(np.var(values))


def select_best_epsilon_pair(object_mask, left_contact, right_contact, object_centroid, object_radius, mu):
    left_candidates = candidate_points_near_contact(object_mask, left_contact)
    right_candidates = candidate_points_near_contact(object_mask, right_contact)

    if not left_candidates:
        left_candidates = [left_contact]
    if not right_candidates:
        right_candidates = [right_contact]

    object_centroid_xy = np.array([object_centroid[1], object_centroid[0]])
    best = None

    for left_candidate in left_candidates:
        left_info = contact_normal_info(object_mask, left_candidate, object_centroid)
        for right_candidate in right_candidates:
            right_info = contact_normal_info(object_mask, right_candidate, object_centroid)
            epsilon_contacts = [
                contact_yx_to_epsilon_contact(left_candidate, left_info["inward"], True),
                contact_yx_to_epsilon_contact(right_candidate, right_info["inward"], True),
            ]
            result = compute_epsilon_quality(
                epsilon_contacts,
                object_centroid_xy,
                object_radius,
                mu=mu,
                samples_per_contact=12,
            )
            original_distance = (
                np.linalg.norm(left_candidate - left_contact) +
                np.linalg.norm(right_candidate - right_contact)
            )
            score = (
                1 if result.get("force_closure") else 0,
                float(result.get("epsilon", 0.0)),
                -float(original_distance),
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "left_contact": np.asarray(left_candidate, dtype=float),
                    "right_contact": np.asarray(right_candidate, dtype=float),
                    "left_info": left_info,
                    "right_info": right_info,
                    "epsilon_contacts": epsilon_contacts,
                    "epsilon_result": result,
                }

    return best


def save_best_grasp_overlay(
    image_rgb,
    object_mask,
    contacts,
    metrics,
    gripper_results=None,
    path="output/best_grasp_overlay.png",
):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.imshow(image_rgb)
    ax.imshow(object_mask, alpha=0.22)

    draw_gripper_contact_annotations(ax, gripper_results, include_text=False)

    for name, obj_p, grip_p, dist in contacts:
        if obj_p is None:
            continue
        color = "red" if name == "left" else "blue"
        plt.scatter(obj_p[1], obj_p[0], c=color, s=120, marker="o", label=f"original {name}")
        if grip_p is not None:
            plt.scatter(grip_p[1], grip_p[0], c=color, s=90, marker="x")
            plt.plot([obj_p[1], grip_p[1]], [obj_p[0], grip_p[0]], color=color, linewidth=1.5, alpha=0.8)

    best_left = metrics.get("best_left_contact")
    best_right = metrics.get("best_right_contact")
    if best_left is not None and best_right is not None:
        plt.scatter(best_left[1], best_left[0], c="lime", s=170, marker="o", label="best epsilon left")
        plt.scatter(best_right[1], best_right[0], c="cyan", s=170, marker="o", label="best epsilon right")
        plt.plot(
            [best_left[1], best_right[1]],
            [best_left[0], best_right[0]],
            color="white",
            linewidth=2.5,
            linestyle="--",
            label="best epsilon pair"
        )
        best_left_normal = metrics.get("best_left_normal")
        best_right_normal = metrics.get("best_right_normal")
        arrow_len = 38
        if best_left_normal is not None:
            plt.arrow(
                best_left[1],
                best_left[0],
                best_left_normal[1] * arrow_len,
                best_left_normal[0] * arrow_len,
                color="lime",
                width=1.4,
                head_width=7,
            )
        if best_right_normal is not None:
            plt.arrow(
                best_right[1],
                best_right[0],
                best_right_normal[1] * arrow_len,
                best_right_normal[0] * arrow_len,
                color="cyan",
                width=1.4,
                head_width=7,
            )

    warning = metrics.get("epsilon_warning")
    if warning:
        plt.text(
            0.02,
            0.98,
            warning,
            transform=plt.gca().transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "orange"},
        )

    plt.title("Original Contacts and Best Epsilon Contact Pair")
    plt.axis("off")
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_epsilon_2d(
    left_contact,
    right_contact,
    object_centroid,
    left_normal,
    right_normal,
    mu=0.5,
    num_dirs=720
):
    """
    Approximate 2D Ferrari-Canny epsilon grasp metric.

    Contact points are [y, x].
    Normals are outward object normals [y, x].
    Wrench format:
        [Fx, Fy, Tau]

    Returns:
        epsilon value
        force closure boolean
        primitive wrench list
    """

    contacts = [
        (left_contact, left_normal),
        (right_contact, right_normal)
    ]

    friction_angle = np.arctan(mu)

    # Use grasp width to normalize torque scale
    grasp_width = np.linalg.norm(right_contact - left_contact)
    if grasp_width < 1e-9:
        return {
            "epsilon": 0.0,
            "epsilon_force_closure": False,
            "wrenches": []
        }

    wrenches = []

    for contact_yx, outward_normal_yx in contacts:
        # Convert [y, x] to [x, y]
        contact_xy = np.array([contact_yx[1], contact_yx[0]], dtype=float)
        centroid_xy = np.array([object_centroid[1], object_centroid[0]], dtype=float)

        # r is contact position relative to object center
        r = (contact_xy - centroid_xy) / grasp_width

        # Convert normal [y, x] to [x, y]
        outward_xy = np.array([outward_normal_yx[1], outward_normal_yx[0]], dtype=float)
        outward_xy = unit_vector(outward_xy)

        # Contact force on object points inward
        inward_xy = -outward_xy
        inward_xy = unit_vector(inward_xy)

        # Friction cone edge forces
        f1 = rotate_2d(inward_xy, friction_angle)
        f2 = rotate_2d(inward_xy, -friction_angle)

        for f in [f1, f2]:
            f = unit_vector(f)

            fx = f[0]
            fy = f[1]

            # 2D torque: tau = r_x * F_y - r_y * F_x
            tau = r[0] * fy - r[1] * fx

            wrench = np.array([fx, fy, tau], dtype=float)
            wrenches.append(wrench)

    wrenches = np.array(wrenches)

    if len(wrenches) == 0:
        return {
            "epsilon": 0.0,
            "epsilon_force_closure": False,
            "wrenches": []
        }

    # Approximate epsilon by sampling directions on the unit sphere in wrench space.
    # epsilon = min over directions of max support in that direction.
    min_support = float("inf")

    for i in range(num_dirs):
        theta = 2 * np.pi * i / num_dirs

        # sample circle in Fx-Fy plane
        d1 = np.array([np.cos(theta), np.sin(theta), 0.0])

        # sample circle involving torque
        d2 = np.array([np.cos(theta), 0.0, np.sin(theta)])

        # sample another mixed plane
        d3 = np.array([0.0, np.cos(theta), np.sin(theta)])

        for direction in [d1, d2, d3]:
            direction = unit_vector(direction)
            support = np.max(wrenches @ direction)
            min_support = min(min_support, support)

    epsilon = max(0.0, float(min_support))
    epsilon_force_closure = epsilon > 1e-4

    return {
        "epsilon": epsilon,
        "epsilon_force_closure": epsilon_force_closure,
        "wrenches": wrenches.tolist()
    }



def force_closure_2d(left_contact, right_contact, left_normal, right_normal, mu=0.5):
    """
    Simplified 2D force-closure approximation.
    Checks whether the line connecting contacts lies inside both friction cones.
    """
    friction_angle = np.degrees(np.arctan(mu))

    lr_vec = right_contact - left_contact
    rl_vec = left_contact - right_contact

    lr_vec = lr_vec / (np.linalg.norm(lr_vec) + 1e-9)
    rl_vec = rl_vec / (np.linalg.norm(rl_vec) + 1e-9)

    # inward directions are opposite outward normals
    left_inward = -left_normal
    right_inward = -right_normal

    left_angle = angle_between(left_inward, lr_vec)
    right_angle = angle_between(right_inward, rl_vec)

    closure_left = left_angle <= friction_angle
    closure_right = right_angle <= friction_angle

    force_closure = closure_left and closure_right

    return {
        "friction_angle": friction_angle,
        "left_cone_angle": left_angle,
        "right_cone_angle": right_angle,
        "force_closure": force_closure
    }


def compute_quality_metrics(object_mask, contacts, depth_img=None, gripper_results=None):
    metrics = {}

    valid_contacts = [
        contact for contact in contacts
        if contact[1] is not None and contact[2] is not None and contact[3] is not None
    ]

    if len(valid_contacts) < 2:
        metrics["final_quality"] = "Poor"
        metrics["reason"] = "Missing contacts"
        empty_result = {
            "epsilon": 0.0,
            "force_closure": False,
            "wrench_points": [],
            "hull_equations": [],
            "weakest_direction": None,
            "error": "Need at least two valid contacts.",
            "mu": FRICTION_COEFFICIENT,
            "num_wrench_points": 0,
        }
        save_wrench_space_projection(
            empty_result,
            "output/wrench_space_projection.png",
            FRICTION_COEFFICIENT,
        )
        save_all_wrench_space_projections(
            empty_result,
            "output",
            FRICTION_COEFFICIENT,
        )
        return metrics

    left_contact = valid_contacts[0][1]
    right_contact = valid_contacts[1][1]

    ys, xs = np.where(object_mask > 0)
    object_centroid = np.array([np.mean(ys), np.mean(xs)])  # [y, x]
    object_centroid_xy = np.array([object_centroid[1], object_centroid[0]])
    object_radius = object_radius_from_mask(object_mask, object_centroid)

    grasp_midpoint = (left_contact + right_contact) / 2.0
    grasp_width = np.linalg.norm(right_contact - left_contact)
    centroid_offset = np.linalg.norm(grasp_midpoint - object_centroid)

    left_dist = valid_contacts[0][3]
    right_dist = valid_contacts[1][3]

    object_angle, _, object_axis = get_object_orientation(object_mask)

    grasp_vec = right_contact - left_contact
    grasp_angle = np.degrees(np.arctan2(grasp_vec[0], grasp_vec[1]))

    angle_difference = abs(grasp_angle - object_angle)
    angle_difference = angle_difference % 180
    if angle_difference > 90:
        angle_difference = 180 - angle_difference

    left_to_center = np.linalg.norm(left_contact - object_centroid)
    right_to_center = np.linalg.norm(right_contact - object_centroid)
    symmetry_error = abs(left_to_center - right_to_center)

    gripper_results = gripper_results or {}
    left_body_info = (gripper_results.get("left", {}) or {}).get("body")
    right_body_info = (gripper_results.get("right", {}) or {}).get("body")
    left_normal_info = contact_normal_info_with_gripper(object_mask, left_contact, object_centroid, left_body_info)
    right_normal_info = contact_normal_info_with_gripper(object_mask, right_contact, object_centroid, right_body_info)
    left_normal = left_normal_info["outward"]
    right_normal = right_normal_info["outward"]

    contour_roughness_value = contour_roughness(object_mask)
    contact_depth_variance = depth_variance_near_points(
        depth_img,
        [left_contact, right_contact],
    )
    normal_confidence = min(
        left_normal_info["confidence"],
        right_normal_info["confidence"],
    )

    closure = force_closure_2d(
        left_contact,
        right_contact,
        left_normal,
        right_normal,
        FRICTION_COEFFICIENT
    )

    uses_thin_finger = (
        (left_body_info or {}).get("mode") == "thin_finger" or
        (right_body_info or {}).get("mode") == "thin_finger"
    )
    best_pair = None if uses_thin_finger else select_best_epsilon_pair(
            object_mask,
            left_contact,
            right_contact,
            object_centroid,
            object_radius,
            FRICTION_COEFFICIENT,
        )

    if best_pair is None:
        epsilon_contacts = [
            contact_yx_to_epsilon_contact(left_contact, left_normal_info["inward"], True),
            contact_yx_to_epsilon_contact(right_contact, right_normal_info["inward"], True),
        ]
        epsilon_result = compute_epsilon_quality(
            epsilon_contacts,
            object_centroid_xy,
            object_radius,
            mu=FRICTION_COEFFICIENT,
            samples_per_contact=12,
        )
        best_left_contact = left_contact
        best_right_contact = right_contact
        best_left_info = left_normal_info
        best_right_info = right_normal_info
    else:
        epsilon_contacts = best_pair["epsilon_contacts"]
        epsilon_result = best_pair["epsilon_result"]
        best_left_contact = best_pair["left_contact"]
        best_right_contact = best_pair["right_contact"]
        best_left_info = best_pair["left_info"]
        best_right_info = best_pair["right_info"]

    save_all_wrench_space_projections(
        epsilon_result,
        "output",
        FRICTION_COEFFICIENT,
    )
    save_epsilon_result_json(
        epsilon_result,
        epsilon_contacts,
        object_centroid_xy,
        object_radius,
        "output/epsilon_result.json",
    )

    epsilon_value = epsilon_result["epsilon"]
    epsilon_force_closure = epsilon_result["force_closure"]

    warnings = []
    if normal_confidence < 0.45:
        warnings.append("Warning: low normal confidence. Epsilon may be unreliable.")
    if contour_roughness_value > 1.15 or (
        contact_depth_variance is not None and contact_depth_variance > 400.0
    ):
        warnings.append("Warning: deformable/irregular object. Rigid-body epsilon metric is approximate.")
 
    # Score contact from the active gripper faces, not from gripper body centers.
    avg_contact_distance = (left_dist + right_dist) / 2.0
    contact_score = max(0, 100 - avg_contact_distance * 1.2)

    if grasp_width > 1:
        balance_score = max(0, 100 - (centroid_offset / grasp_width) * 100)
        symmetry_score = max(0, 100 - (symmetry_error / grasp_width) * 150)
    else:
        balance_score = 0
        symmetry_score = 0

    angle_score = max(0, 100 - angle_difference * 3)

    if closure["force_closure"]:
        closure_score = 100
    else:
        avg_cone_error = (
            max(0, closure["left_cone_angle"] - closure["friction_angle"]) +
            max(0, closure["right_cone_angle"] - closure["friction_angle"])
        ) / 2
        closure_score = max(35, 100 - avg_cone_error * 2)
    
    epsilon_score = min(100, epsilon_value * 1000)

    final_score = (
        0.30 * contact_score +
        0.20 * balance_score +
        0.15 * symmetry_score +
        0.10 * angle_score +
        0.10 * closure_score +
        0.15 * epsilon_score
)

    if final_score >= 80:
        final_quality = "Good"
    elif final_score >= 55:
        final_quality = "Medium"
    else:
        final_quality = "Poor"

    metrics["left_distance"] = left_dist
    metrics["right_distance"] = right_dist
    metrics["grasp_width"] = grasp_width
    metrics["object_centroid"] = object_centroid
    metrics["object_radius"] = object_radius
    metrics["grasp_midpoint"] = grasp_midpoint
    metrics["centroid_offset"] = centroid_offset

    metrics["object_angle"] = object_angle
    metrics["object_axis"] = object_axis
    metrics["grasp_angle"] = grasp_angle
    metrics["angle_difference"] = angle_difference
    metrics["symmetry_error"] = symmetry_error

    metrics["left_normal"] = left_normal
    metrics["right_normal"] = right_normal
    metrics["left_inward_normal"] = left_normal_info["inward"]
    metrics["right_inward_normal"] = right_normal_info["inward"]
    metrics["left_normal_confidence"] = left_normal_info["confidence"]
    metrics["right_normal_confidence"] = right_normal_info["confidence"]
    metrics["normal_confidence"] = normal_confidence
    metrics["left_normal_source"] = left_normal_info.get("source", "contour")
    metrics["right_normal_source"] = right_normal_info.get("source", "contour")
    metrics["contour_roughness"] = contour_roughness_value
    metrics["contact_depth_variance"] = contact_depth_variance

    metrics["best_left_contact"] = best_left_contact
    metrics["best_right_contact"] = best_right_contact
    metrics["best_left_normal"] = best_left_info["outward"]
    metrics["best_right_normal"] = best_right_info["outward"]

    metrics["friction_angle"] = closure["friction_angle"]
    metrics["left_cone_angle"] = closure["left_cone_angle"]
    metrics["right_cone_angle"] = closure["right_cone_angle"]
    metrics["force_closure"] = closure["force_closure"]

    metrics["epsilon"] = epsilon_value
    metrics["epsilon_force_closure"] = epsilon_force_closure
    metrics["epsilon_score"] = epsilon_score
    metrics["epsilon_num_wrenches"] = epsilon_result.get("num_wrench_points", 0)
    metrics["epsilon_error"] = epsilon_result.get("error")
    metrics["epsilon_weakest_direction"] = epsilon_result.get("weakest_direction")
    metrics["epsilon_closest_face_distance"] = epsilon_result.get("closest_face_distance")
    metrics["epsilon_closest_hull_point"] = epsilon_result.get("closest_hull_point")
    metrics["epsilon_warning"] = "\n".join(warnings)

    metrics["contact_score"] = contact_score
    metrics["balance_score"] = balance_score
    metrics["angle_score"] = angle_score
    metrics["symmetry_score"] = symmetry_score
    metrics["closure_score"] = closure_score
    metrics["final_score"] = final_score
    metrics["final_quality"] = final_quality

    return metrics


# ---------------- MAIN ----------------

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Could not find {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]

object_roi = (int(w * 0.20), int(h * 0.20), int(w * 0.80), int(h * 0.75))

object_prompt = find_prompt_in_roi(image_rgb, object_roi, 500)

if object_prompt is None:
    raise RuntimeError("Could not find object prompt.")

model = build_sam2(CONFIG, CHECKPOINT, device="cpu")
predictor = SAM2ImagePredictor(model)
predictor.set_image(image_rgb)

object_mask = sam_predict(predictor, object_prompt)
object_box = mask_bbox(object_mask)
left_roi, right_roi = object_guided_gripper_rois(image_rgb.shape, object_box)
manual_gripper_boxes = load_manual_gripper_boxes()
obj_pts = mask_boundary(object_mask)
depth_img = load_depth_optional("output/depth_frame.png")
object_depth_stats = compute_object_depth_stats(object_mask, depth_img)

left_body = detect_gripper_body(
    predictor,
    image_rgb,
    "left",
    left_roi,
    manual_gripper_boxes,
    exclude_mask=object_mask,
    object_box=object_box,
    object_points=obj_pts,
    depth_img=depth_img,
    object_depth_stats=object_depth_stats,
)
right_body = detect_gripper_body(
    predictor,
    image_rgb,
    "right",
    right_roi,
    manual_gripper_boxes,
    exclude_mask=object_mask,
    object_box=object_box,
    object_points=obj_pts,
    depth_img=depth_img,
    object_depth_stats=object_depth_stats,
)

left_mask = left_body.get("mask") if left_body else None
right_mask = right_body.get("mask") if right_body else None
left_prompt = left_body.get("prompt") if left_body else None
right_prompt = right_body.get("prompt") if right_body else None

contacts = []

left_result = build_gripper_contact(
    "left",
    left_body,
    obj_pts,
    object_box,
    depth_img,
    object_mask,
    object_depth_stats,
)
right_result = build_gripper_contact(
    "right",
    right_body,
    obj_pts,
    object_box,
    depth_img,
    object_mask,
    object_depth_stats,
)
gripper_results = {
    "left": left_result,
    "right": right_result,
}

for side, result in gripper_results.items():
    contacts.append((
        side,
        result.get("object_point"),
        result.get("face_point"),
        result.get("distance_px"),
    ))
import json

contact_data = {}
os.makedirs("output", exist_ok=True)

for name, obj_p, grip_p, dist in contacts:
    if obj_p is None or grip_p is None:
        continue

    result = gripper_results.get(name, {})
    body = result.get("body") or {}
    depth_diff = result.get("depth_difference_mm")
    contact_data[name] = {
        "object_point_xy": [int(obj_p[1]), int(obj_p[0])],
        "gripper_point_xy": [int(grip_p[1]), int(grip_p[0])],
        "gripper_face_point_xy": [int(grip_p[1]), int(grip_p[0])],
        "gripper_body_bbox_xyxy": list(body.get("bbox")) if body.get("bbox") else None,
        "distance_px": float(dist),
        "object_depth_mm": result.get("object_depth_mm"),
        "gripper_face_depth_mm": result.get("face_depth_mm"),
        "depth_difference_mm": float(depth_diff) if depth_diff is not None else None,
        "depth_confidence": result.get("depth_confidence"),
        "object_depth_valid_ratio": result.get("object_depth_valid_ratio"),
        "gripper_depth_valid_ratio": result.get("gripper_depth_valid_ratio"),
        "object_depth_median_mm": object_depth_stats.get("median"),
        "object_depth_mean_mm": object_depth_stats.get("mean"),
        "object_depth_std_mm": object_depth_stats.get("std"),
        "object_depth_valid_ratio_global": object_depth_stats.get("valid_ratio"),
        "object_depth_band_mm": [object_depth_stats.get("band_low"), object_depth_stats.get("band_high")],
        "gripper_depth_difference_from_object_median_mm": result.get("gripper_depth_difference_from_object_median_mm"),
        "gripper_sample_inside_object": result.get("gripper_sample_inside_object"),
        "near_same_sample": result.get("near_same_sample"),
        "gripper_depth_sample_shifted": result.get("gripper_depth_sample_shifted"),
        "selected_contact_source": result.get("selection_source"),
        "depth_affected_selection": result.get("depth_affected_selection"),
        "object_mask_path": "output/object_mask.png",
        "gripper_confidence": body.get("confidence"),
        "gripper_detection_mode": body.get("mode"),
        "block_confidence": body.get("block_confidence", body.get("confidence") if body.get("mode") in ("block", "block_sam") else 0.0),
        "thin_finger_confidence": body.get("thin_finger_confidence", body.get("confidence") if body.get("mode") == "thin_finger" else 0.0),
        "selected_mode_reason": body.get("selected_mode_reason"),
        "gripper_direction_xy": body.get("gripper_direction_xy"),
        "thin_finger_line_xy": body.get("thin_line"),
        "normal_source": metrics.get(f"{name}_normal_source") if "metrics" in globals() else None,
        "normal_confidence": metrics.get(f"{name}_normal_confidence") if "metrics" in globals() else None,
        "candidate_count": len(body.get("candidates", [])),
        "rejected_candidate_count": len(body.get("rejected_candidates", [])),
        "contact_valid": bool(result.get("valid", False)),
        "block_overlap_segment": result.get("overlap_segment"),
        "selected_overlap_midpoint": result.get("selected_overlap_midpoint"),
        "warning": result.get("warning"),
    }

with open("output/contact_points.json", "w") as f:
    json.dump(contact_data, f, indent=4)


metrics = compute_quality_metrics(object_mask, contacts, depth_img, gripper_results)
for side in ["left", "right"]:
    if side in contact_data:
        contact_data[side]["normal_source"] = metrics.get(f"{side}_normal_source")
        contact_data[side]["normal_confidence"] = metrics.get(f"{side}_normal_confidence")
with open("output/contact_points.json", "w") as f:
    json.dump(contact_data, f, indent=4)
save_contact_debug(image_rgb, object_mask, left_mask, right_mask, contacts)
save_gripper_detection_debug(
    image_rgb,
    object_mask,
    object_box,
    left_roi,
    right_roi,
    gripper_results,
    object_depth_stats,
)
save_best_grasp_overlay(image_rgb, object_mask, contacts, metrics, gripper_results)

os.makedirs("output", exist_ok=True)

# Scene
cv2.imwrite("output/scene.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("output/object_mask.png", (object_mask.astype(np.uint8) * 255))

# Segmentation
fig = plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.imshow(object_mask, alpha=0.35)

if left_mask is not None:
    plt.imshow(left_mask, alpha=0.35)

if right_mask is not None:
    plt.imshow(right_mask, alpha=0.35)

for roi, color in [
    (left_roi, "cyan"),
    (object_roi, "yellow"),
    (right_roi, "lime")
]:
    x1, y1, x2, y2 = roi
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=2)

ax = plt.gca()
draw_box(ax, object_box, "orange", "object bbox", linewidth=2)
draw_box(ax, left_body.get("bbox") if left_body else None, "red", "left body", linewidth=2.5)
draw_box(ax, right_body.get("bbox") if right_body else None, "blue", "right body", linewidth=2.5)

if left_prompt:
    plt.scatter(left_prompt[0], left_prompt[1], c="cyan", s=80)
if object_prompt:
    plt.scatter(object_prompt[0], object_prompt[1], c="yellow", s=80)
if right_prompt:
    plt.scatter(right_prompt[0], right_prompt[1], c="lime", s=80)

plt.title("Segmentation View")
plt.axis("off")
plt.savefig("output/segmentation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Contact Model
fig = plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.imshow(image_rgb)
ax.imshow(object_mask, alpha=0.25)
draw_box(ax, object_box, "yellow", "object bbox", linewidth=2)
draw_gripper_contact_annotations(ax, gripper_results, include_text=True)

for name, obj_p, grip_p, dist in contacts:
    if obj_p is None or grip_p is None:
        continue

    color = "red" if name == "left" else "blue"

    plt.scatter(obj_p[1], obj_p[0], c=color, s=150, label=f"{name} contact")
    plt.scatter(grip_p[1], grip_p[0], c=color, s=90, marker="x")
    plt.plot([obj_p[1], grip_p[1]], [obj_p[0], grip_p[0]], color=color, linewidth=2)

    n = metrics.get("left_normal") if name == "left" else metrics.get("right_normal")
    if n is None:
        continue
    arrow_len = 40

    plt.arrow(
        obj_p[1],
        obj_p[0],
        n[1] * arrow_len,
        n[0] * arrow_len,
        color=color,
        width=1.5,
        head_width=8
    )

if "object_centroid" in metrics:
    oc = metrics["object_centroid"]
    gm = metrics["grasp_midpoint"]

    plt.scatter(oc[1], oc[0], c="yellow", s=130, marker="o", label="object centroid")
    plt.scatter(gm[1], gm[0], c="magenta", s=130, marker="D", label="grasp midpoint")

    axis = metrics.get("object_axis")
    if axis is not None:
        axis_len = 90
        plt.plot(
            [oc[1] - axis[1] * axis_len, oc[1] + axis[1] * axis_len],
            [oc[0] - axis[0] * axis_len, oc[0] + axis[0] * axis_len],
            color="yellow",
            linewidth=2,
            label="object axis"
        )

    # grasp axis
    valid_contact_points = [contact[1] for contact in contacts if contact[1] is not None]
    if len(valid_contact_points) >= 2:
        p1 = valid_contact_points[0]
        p2 = valid_contact_points[1]
        plt.plot(
            [p1[1], p2[1]],
            [p1[0], p2[0]],
            color="magenta",
            linewidth=2,
            linestyle="--",
            label="grasp axis"
        )

plt.title("Contact Model")
plt.axis("off")
plt.legend(loc="upper right", fontsize=7)
plt.savefig("output/contact_model.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Epsilon Quality Report
fig = plt.figure(figsize=(9.5, 9.0))
plt.axis("off")

force_closure_text = "Yes" if metrics.get("force_closure", False) else "No"
epsilon_force_closure_text = "Yes" if metrics.get("epsilon_force_closure", False) else "No"
weak_direction_text = metrics.get("epsilon_weakest_direction")
warning_text = metrics.get("epsilon_warning", "")
gripper_warning_text = "\n".join(
    result.get("warning")
    for result in gripper_results.values()
    if result and result.get("warning")
)
if gripper_warning_text:
    warning_text = "\n".join(part for part in [warning_text, gripper_warning_text] if part)
if object_depth_stats and not object_depth_stats.get("available"):
    warning_text = "\n".join(part for part in [warning_text, object_depth_stats.get("warning", "geometry-only: depth unreliable")] if part)
left_body_info = (gripper_results.get("left", {}).get("body") or {})
right_body_info = (gripper_results.get("right", {}).get("body") or {})
left_result_info = gripper_results.get("left", {})
right_result_info = gripper_results.get("right", {})

warning_display = warning_text if warning_text else "No critical warnings."
if not metrics.get("epsilon_force_closure", False):
    warning_display = "\n".join(part for part in [warning_display, "Origin outside wrench hull or force closure false."] if part)
if metrics.get("normal_confidence", 0) < 0.45:
    warning_display = "\n".join(part for part in [warning_display, "Low normal confidence."] if part)

quality_text = f"""Epsilon Quality Report

Warnings
{warning_display}

1. Detection
Object detected: {"Yes" if object_box is not None else "No"}
Left gripper: {left_body_info.get('mode', 'unknown')}  conf {left_body_info.get('confidence', 0):.2f}
Right gripper: {right_body_info.get('mode', 'unknown')}  conf {right_body_info.get('confidence', 0):.2f}
Left block/thin confidence: {left_body_info.get('block_confidence', 0):.2f} / {left_body_info.get('thin_finger_confidence', 0):.2f}
Right block/thin confidence: {right_body_info.get('block_confidence', 0):.2f} / {right_body_info.get('thin_finger_confidence', 0):.2f}
Left mode reason: {left_body_info.get('selected_mode_reason', 'n/a')}
Right mode reason: {right_body_info.get('selected_mode_reason', 'n/a')}
Normal sources: L={metrics.get('left_normal_source', 'unknown')} R={metrics.get('right_normal_source', 'unknown')}
Normal confidence: L={metrics.get('left_normal_confidence', 0):.2f} R={metrics.get('right_normal_confidence', 0):.2f}

2. RGB-D Contact
Object median depth: {object_depth_stats.get('median') if object_depth_stats.get('median') is not None else 'unknown'} mm
Object mean/std depth: {object_depth_stats.get('mean') if object_depth_stats.get('mean') is not None else 'unknown'} / {object_depth_stats.get('std') if object_depth_stats.get('std') is not None else 'unknown'} mm
Object depth valid ratio: {object_depth_stats.get('valid_ratio', 0):.2f}
Object depth band: {object_depth_stats.get('band_low') if object_depth_stats.get('band_low') is not None else 'unknown'} to {object_depth_stats.get('band_high') if object_depth_stats.get('band_high') is not None else 'unknown'} mm
Left source: {left_result_info.get('selection_source', 'unknown')}
Right source: {right_result_info.get('selection_source', 'unknown')}
Depth affected selection: L={left_result_info.get('depth_affected_selection', False)} R={right_result_info.get('depth_affected_selection', False)}
Left contact distance: {metrics.get('left_distance', 999):.2f} px
Right contact distance: {metrics.get('right_distance', 999):.2f} px
Left depth difference: {left_result_info.get('depth_difference_mm') if left_result_info.get('depth_difference_mm') is not None else 'unknown'} mm
Right depth difference: {right_result_info.get('depth_difference_mm') if right_result_info.get('depth_difference_mm') is not None else 'unknown'} mm
Left gripper depth: {left_result_info.get('face_depth_mm') if left_result_info.get('face_depth_mm') is not None else 'unknown'} mm
Right gripper depth: {right_result_info.get('face_depth_mm') if right_result_info.get('face_depth_mm') is not None else 'unknown'} mm
Left gripper vs object median: {left_result_info.get('gripper_depth_difference_from_object_median_mm') if left_result_info.get('gripper_depth_difference_from_object_median_mm') is not None else 'unknown'} mm
Right gripper vs object median: {right_result_info.get('gripper_depth_difference_from_object_median_mm') if right_result_info.get('gripper_depth_difference_from_object_median_mm') is not None else 'unknown'} mm
Left depth confidence: {left_result_info.get('depth_confidence', 0):.2f}
Right depth confidence: {right_result_info.get('depth_confidence', 0):.2f}
Left valid depth ratio: obj={left_result_info.get('object_depth_valid_ratio', 0):.2f} grip={left_result_info.get('gripper_depth_valid_ratio', 0):.2f}
Right valid depth ratio: obj={right_result_info.get('object_depth_valid_ratio', 0):.2f} grip={right_result_info.get('gripper_depth_valid_ratio', 0):.2f}
Grasp width: {metrics.get('grasp_width', 0):.2f} px

3. Epsilon Metric
Friction coefficient: {FRICTION_COEFFICIENT:.2f}
Epsilon value: {metrics.get('epsilon', 0):.4f}
Epsilon force closure: {epsilon_force_closure_text}
Wrench points: {metrics.get('epsilon_num_wrenches', 0)}
Weak direction: {weak_direction_text}
Closest hull point: {metrics.get('epsilon_closest_hull_point')}
Closest hull face distance: {metrics.get('epsilon_closest_face_distance') if metrics.get('epsilon_closest_face_distance') is not None else 0:.4f}

4. Scores
Contact: {metrics.get('contact_score', 0):.1f}/100
Balance: {metrics.get('balance_score', 0):.1f}/100
Angle: {metrics.get('angle_score', 0):.1f}/100
Symmetry: {metrics.get('symmetry_score', 0):.1f}/100
Closure: {metrics.get('closure_score', 0):.1f}/100
Normalized epsilon: {metrics.get('epsilon_score', 0):.1f}/100

Final Result
Final score: {metrics.get('final_score', 0):.1f}/100
Final quality: {metrics.get('final_quality', 'Poor')}"""

plt.text(0.04, 0.98, quality_text, fontsize=12.6, va="top", linespacing=1.35)
plt.savefig("output/quality_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Combined dashboard summary
dashboard_items = [
    ("Object detected", "Yes" if object_box is not None else "No", "green" if object_box is not None else "red"),
    ("Left gripper", f"{left_body_info.get('mode', 'failed')}\nconf {left_body_info.get('confidence', 0):.2f}", "green" if left_result_info.get("valid") else "orange"),
    ("Right gripper", f"{right_body_info.get('mode', 'failed')}\nconf {right_body_info.get('confidence', 0):.2f}", "green" if right_result_info.get("valid") else "orange"),
    ("RGB-D contact", "See RGB-D panel", "orange"),
    ("Epsilon closure", str(bool(metrics.get("epsilon_force_closure", False))), "green" if metrics.get("epsilon_force_closure", False) else "red"),
    ("Final quality", metrics.get("final_quality", "Poor"), "green" if metrics.get("final_quality") == "Good" else ("orange" if metrics.get("final_quality") == "Medium" else "red")),
    ("Final score", f"{metrics.get('final_score', 0):.1f}/100", "green" if metrics.get("final_score", 0) >= 80 else ("orange" if metrics.get("final_score", 0) >= 55 else "red")),
]
color_map = {
    "green": "#2e7d32",
    "orange": "#ef6c00",
    "red": "#c62828",
}
fig, axes = plt.subplots(1, len(dashboard_items), figsize=(18, 2.8))
for ax, (title, value, status) in zip(axes, dashboard_items):
    ax.set_facecolor(color_map[status])
    ax.text(0.5, 0.68, title, ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text(0.5, 0.32, value, ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("Grasp Quality Dashboard", fontsize=18, weight="bold")
fig.tight_layout()
fig.savefig("output/dashboard_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Full wrench-space technical dashboard
fig = plt.figure(figsize=(16, 10))
grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.12)
wrench_panels = [
    ("Wrench Fx/Fy", "output/wrench_space_fx_fy.png", "Fx/Fy: force balance projection."),
    ("Wrench Fx/Tau", "output/wrench_space_fx_tau.png", "Fx/Tau: horizontal force vs torque resistance."),
    ("Wrench Fy/Tau", "output/wrench_space_fy_tau.png", "Fy/Tau: vertical force vs torque resistance."),
]
for index, (title, image_path, caption) in enumerate(wrench_panels):
    ax = fig.add_subplot(grid[index // 2, index % 2])
    ax.set_title(title, fontsize=15, weight="bold")
    if os.path.exists(image_path):
        ax.imshow(plt.imread(image_path))
    else:
        ax.text(0.5, 0.5, f"Missing image:\n{image_path}", ha="center", va="center", fontsize=13, color="red")
    ax.axis("off")
    ax.text(
        0.5,
        -0.06,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#333333",
    )

analysis_ax = fig.add_subplot(grid[1, 1])
force_closure = bool(metrics.get("epsilon_force_closure", False))
analysis_ax.set_facecolor("#e7f5ea" if force_closure else "#fdeaea")
analysis_ax.set_xticks([])
analysis_ax.set_yticks([])
for spine in analysis_ax.spines.values():
    spine.set_color("#1b5e20" if force_closure else "#b91c1c")
    spine.set_linewidth(2.0)
origin_status = "inside" if force_closure else "outside"
closure_note = (
    "Origin inside wrench hull. Grasp has planar force closure."
    if force_closure
    else "Origin outside 3D wrench hull. Grasp cannot resist all planar disturbances."
)
analysis_text = f"""Wrench-Space Analysis

{closure_note}

Epsilon value: {metrics.get('epsilon', 0):.4f}
Force closure: {force_closure}
Origin: {origin_status} hull
Weak direction vector: {metrics.get('epsilon_weakest_direction')}
Closest hull point: {metrics.get('epsilon_closest_hull_point')}
Closest hull face distance: {metrics.get('epsilon_closest_face_distance') if metrics.get('epsilon_closest_face_distance') is not None else 0:.4f}
Friction coefficient: {FRICTION_COEFFICIENT:.2f}
Wrench points: {metrics.get('epsilon_num_wrenches', 0)}

The green circle is only a 2D projection of epsilon.
The green marker/line shows the projected closest hull face touch point.
Actual epsilon is the 3D distance from origin to the closest hull face in [Fx, Fy, tau].

Can the grasp resist force in different directions? {"Yes" if force_closure else "No"}
Can it resist torque? {"Yes" if force_closure else "Not in all planar directions"}
Weakest disturbance direction: {metrics.get('epsilon_weakest_direction')}"""
analysis_ax.text(
    0.04,
    0.96,
    analysis_text,
    transform=analysis_ax.transAxes,
    va="top",
    fontsize=13,
    color="#1b5e20" if force_closure else "#b91c1c",
    linespacing=1.35,
    weight="bold",
)
fig.suptitle("Technical Wrench Debug Dashboard", fontsize=18, weight="bold")
fig.savefig("output/wrench_debug_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Phase 1 complete.")
print("Saved:")
print("output/scene.png")
print("output/segmentation.png")
print("output/contact_model.png")
print("output/gripper_detection_debug.png")
print("output/quality_matrix.png")
print("output/wrench_space_projection.png")
print("output/wrench_space_fx_fy.png")
print("output/wrench_space_fx_tau.png")
print("output/wrench_space_fy_tau.png")
print("output/best_grasp_overlay.png")
print("output/dashboard_summary.png")
print("output/wrench_debug_dashboard.png")

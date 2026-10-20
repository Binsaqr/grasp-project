"""Camera-free object profile schema, descriptors, and matching.

Profiles are reusable priors for the existing grasp pipeline. They do not
replace live RGB-D detection, depth validation, contact validation, quality
staleness checks, or safety gates.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from grasp_project.common.file_io import json_safe

PROFILE_VERSION = "1.0"
GENERIC_OBJECT_ID = "generic_graspable_object"
LEGACY_BLUE_BOTTLE_ID = "blue_bottle"


def utc_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def slugify(value: str, fallback: str = "object") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or fallback


def _num(value, default=None):
    try:
        if value is None:
            return default
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def _as_mask(mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if mask is None:
        if shape is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def _rgb_array(rgb: np.ndarray | None) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return arr[..., :3].astype(np.uint8, copy=False)


def normalized_contour_from_mask(mask: np.ndarray, max_points: int = 80) -> list[list[float]]:
    mask = _as_mask(mask)
    ys, xs = np.where(mask > 0)
    if xs.size < 3:
        return []
    try:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    except Exception:
        contour = np.column_stack([xs, ys]).astype(float)
    if contour.shape[0] > max_points:
        idx = np.linspace(0, contour.shape[0] - 1, max_points).astype(int)
        contour = contour[idx]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    pts = np.column_stack([(contour[:, 0] - x0) / w, (contour[:, 1] - y0) / h])
    return np.clip(pts, 0.0, 1.0).round(4).tolist()


def shape_descriptors(mask: np.ndarray | None) -> dict[str, Any]:
    mask = _as_mask(mask)
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if xs.size < 3:
        return {
            "mask_area_ratio": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "principal_axis": [0.0, 1.0],
            "symmetry_score": 0.0,
            "cylindrical_score": 0.0,
            "flatness_score": 1.0,
            "bbox_xyxy": None,
            "aspect_ratio": None,
            "oriented_aspect_ratio": None,
        }
    area = float(xs.size)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    extent = area / float(max(1, bw * bh))
    pts = np.column_stack([xs, ys]).astype(float)
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    except Exception:
        axis = np.array([0.0, 1.0], dtype=float)
    if axis[1] < 0:
        axis = -axis
    projections = centered @ axis
    lateral_axis = np.array([-axis[1], axis[0]], dtype=float)
    lateral = centered @ lateral_axis
    long_span = max(1.0, float(np.percentile(projections, 98) - np.percentile(projections, 2)))
    short_span = max(1.0, float(np.percentile(lateral, 98) - np.percentile(lateral, 2)))
    oriented_aspect = float(max(long_span, short_span) / max(1.0, min(long_span, short_span)))
    left = mask[:, : w // 2]
    right = np.fliplr(mask[:, w - left.shape[1] :])
    if left.size and right.size:
        symmetry = 1.0 - float(np.mean(np.abs(left.astype(float) - right.astype(float))))
    else:
        symmetry = 0.0
    try:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        contour_area = float(cv2.contourArea(contour)) if contour is not None else area
        hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None and len(contour) >= 3 else contour_area
        solidity = contour_area / max(1.0, hull_area)
    except Exception:
        solidity = min(1.0, extent + 0.1)
    cylindrical_score = float(np.clip(0.45 * symmetry + 0.35 * min(oriented_aspect / 2.5, 1.0) + 0.20 * solidity, 0.0, 1.0))
    flatness_score = float(np.clip(1.0 - min(oriented_aspect / 4.0, 1.0), 0.0, 1.0))
    return {
        "mask_area_ratio": round(area / float(max(1, h * w)), 5),
        "solidity": round(float(np.clip(solidity, 0.0, 1.0)), 4),
        "extent": round(float(np.clip(extent, 0.0, 1.0)), 4),
        "principal_axis": np.round(axis.astype(float), 5).tolist(),
        "symmetry_score": round(float(np.clip(symmetry, 0.0, 1.0)), 4),
        "cylindrical_score": round(cylindrical_score, 4),
        "flatness_score": round(flatness_score, 4),
        "bbox_xyxy": bbox,
        "aspect_ratio": round(float(max(bw, bh) / max(1, min(bw, bh))), 4),
        "oriented_aspect_ratio": round(oriented_aspect, 4),
    }


def color_descriptors(rgb: np.ndarray | None, mask: np.ndarray | None = None) -> dict[str, Any]:
    rgb = _rgb_array(rgb)
    if rgb is None:
        return {"dominant_colors": [], "color_histogram": [], "appearance_confidence": 0.0}
    mask = _as_mask(mask, rgb.shape[:2])
    pixels = rgb[mask > 0] if np.any(mask > 0) else rgb.reshape(-1, 3)
    if pixels.size == 0:
        return {"dominant_colors": [], "color_histogram": [], "appearance_confidence": 0.0}
    bins = 4
    quant = np.clip((pixels.astype(int) * bins) // 256, 0, bins - 1)
    flat = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    counts = np.bincount(flat, minlength=bins ** 3).astype(float)
    hist = counts / max(1.0, counts.sum())
    top = np.argsort(hist)[-3:][::-1]
    colors = []
    for idx in top:
        if hist[idx] <= 0:
            continue
        r = (idx // (bins * bins)) % bins
        g = (idx // bins) % bins
        b = idx % bins
        colors.append({
            "rgb": [int((r + 0.5) * 256 / bins), int((g + 0.5) * 256 / bins), int((b + 0.5) * 256 / bins)],
            "fraction": round(float(hist[idx]), 4),
        })
    confidence = float(np.clip(np.count_nonzero(mask) / max(1, mask.size), 0.0, 1.0)) if mask.size else 0.0
    return {
        "dominant_colors": colors,
        "color_histogram": np.round(hist, 5).tolist(),
        "appearance_confidence": round(confidence, 4),
    }


def geometry_from_mask_depth(mask: np.ndarray | None, depth: np.ndarray | None = None, intrinsics: dict[str, Any] | None = None) -> dict[str, Any]:
    shape = shape_descriptors(mask)
    bbox = shape.get("bbox_xyxy")
    depth_values = []
    if depth is not None and bbox:
        mask_u8 = _as_mask(mask, np.asarray(depth).shape[:2])
        arr = np.asarray(depth)
        values = arr[mask_u8 > 0]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            if values.dtype == np.uint16 or float(np.nanmax(values)) > 50.0:
                depth_values = values.astype(float).tolist()
            else:
                depth_values = (values.astype(float) * 1000.0).tolist()
    median_depth_mm = float(np.median(depth_values)) if depth_values else None
    expected_width_mm = None
    expected_height_mm = None
    if median_depth_mm and bbox and intrinsics:
        fx = _num(intrinsics.get("fx"))
        fy = _num(intrinsics.get("fy"))
        if fx and fy:
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            expected_width_mm = float(bw * median_depth_mm / fx)
            expected_height_mm = float(bh * median_depth_mm / fy)
    extents = [v for v in (expected_width_mm, expected_height_mm) if v is not None]
    expected_diameter_mm = min(extents) if extents else None
    geometry_confidence = 0.35
    if median_depth_mm:
        geometry_confidence += 0.25
    if expected_width_mm and expected_height_mm:
        geometry_confidence += 0.25
    geometry_confidence += 0.15 * float(shape.get("solidity") or 0.0)
    return {
        "expected_bbox_aspect_range": _range_around(shape.get("aspect_ratio"), 0.25, fallback=[0.4, 5.0]),
        "expected_oriented_aspect_range": _range_around(shape.get("oriented_aspect_ratio"), 0.25, fallback=[0.4, 5.0]),
        "expected_height_mm": _range_around(expected_height_mm, 0.20),
        "expected_width_mm": _range_around(expected_width_mm, 0.20),
        "expected_depth_mm": None,
        "expected_diameter_mm": _range_around(expected_diameter_mm, 0.22),
        "expected_grip_width_mm": _range_around(expected_diameter_mm or expected_width_mm, 0.25),
        "expected_depth_range_mm": _range_around(median_depth_mm, 0.15),
        "geometry_confidence": round(float(np.clip(geometry_confidence, 0.0, 1.0)), 4),
        "bbox_xyxy": bbox,
    }


def _range_around(value, frac: float, fallback=None):
    value = _num(value)
    if value is None or value <= 0:
        return fallback
    return {"min": round(value * (1.0 - frac), 3), "max": round(value * (1.0 + frac), 3)}


def classify_object_type(shape: dict[str, Any], geometry: dict[str, Any]) -> str:
    aspect = float(shape.get("oriented_aspect_ratio") or 0.0)
    cyl = float(shape.get("cylindrical_score") or 0.0)
    flat = float(shape.get("flatness_score") or 0.0)
    grip = geometry.get("expected_grip_width_mm") or {}
    grip_max = _num(grip.get("max") if isinstance(grip, dict) else None)
    if grip_max and grip_max > 160:
        return "unsupported"
    if cyl >= 0.55 and aspect >= 1.6:
        return "cylindrical_bottle"
    if flat >= 0.75 and aspect < 1.5:
        return "cup_like"
    if aspect < 1.4:
        return "rectangular_object"
    return "unknown_graspable"


def default_grasp_profile(shape: dict[str, Any], geometry: dict[str, Any], object_type: str) -> dict[str, Any]:
    grip = geometry.get("expected_grip_width_mm") or {}
    expected_contacts = 2
    limitations = []
    if object_type == "unsupported":
        limitations.append("Object may not be suitable for parallel-jaw grasp.")
    if not grip:
        limitations.append("Metric grip width unavailable; requires stronger depth/contact validation.")
    return {
        "preferred_grasp_regions": [{"body_fraction": 0.50, "score": 0.80}, {"body_fraction": 0.58, "score": 0.70}],
        "forbidden_regions": [{"body_fraction_min": 0.0, "body_fraction_max": 0.15, "reason": "avoid top/end cap"}, {"body_fraction_min": 0.85, "body_fraction_max": 1.0, "reason": "avoid bottom/base"}],
        "preferred_body_fraction": 0.50,
        "contact_patch_height_px": 34,
        "contact_patch_width_px": 0,
        "expected_contact_count": expected_contacts,
        "compatible_embodiments": ["parallel_jaw_two_finger"],
        "expected_grip_width_mm": grip,
        "grasp_notes": "; ".join(limitations) if limitations else "Profile-derived conservative side grasp region.",
    }


def create_object_profile(
    *,
    object_id: str | None = None,
    display_name: str = "Scanned Object",
    object_type: str | None = None,
    rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    intrinsics: dict[str, Any] | None = None,
    pointcloud_status: dict[str, Any] | None = None,
    artifact_paths: dict[str, str] | None = None,
    source: str = "object_scan",
    notes: str = "",
) -> dict[str, Any]:
    object_id = slugify(object_id or display_name)
    mask_u8 = _as_mask(mask, _rgb_array(rgb).shape[:2] if _rgb_array(rgb) is not None else None)
    colors = color_descriptors(rgb, mask_u8)
    shape = shape_descriptors(mask_u8)
    geometry = geometry_from_mask_depth(mask_u8, depth=depth, intrinsics=intrinsics)
    object_type = object_type or classify_object_type(shape, geometry)
    pointcloud = _pointcloud_profile(pointcloud_status or {})
    grasp = default_grasp_profile(shape, geometry, object_type)
    now = utc_iso()
    scan_quality = _scan_quality(colors, geometry, shape, pointcloud)
    limitations = []
    if object_type == "unsupported":
        limitations.append("Object may not be suitable for the current parallel-jaw gripper.")
    if scan_quality < 0.45:
        limitations.append("Scan confidence is low; rescan or use Generic mode.")
    profile = {
        "object_id": object_id,
        "display_name": display_name,
        "object_type": object_type,
        "created_at": now,
        "updated_at": now,
        "profile_version": PROFILE_VERSION,
        "source": source,
        "active_target_allowed": object_type != "unsupported",
        "notes": notes,
        "representative_rgb_snapshot_path": (artifact_paths or {}).get("rgb_snapshot"),
        "dominant_colors": colors["dominant_colors"],
        "color_histogram": colors["color_histogram"],
        "grayscale_template_path": (artifact_paths or {}).get("grayscale_template"),
        "edge_template_path": (artifact_paths or {}).get("edge_template"),
        "contour_descriptors": {
            "normalized_contour_points": normalized_contour_from_mask(mask_u8),
            "aspect_ratio": shape.get("aspect_ratio"),
            "oriented_aspect_ratio": shape.get("oriented_aspect_ratio"),
        },
        "keypoint_descriptors": None,
        "appearance_confidence": colors["appearance_confidence"],
        **geometry,
        "geometry_confidence": geometry["geometry_confidence"],
        "contour_points_normalized": normalized_contour_from_mask(mask_u8),
        "mask_area_ratio": shape["mask_area_ratio"],
        "solidity": shape["solidity"],
        "extent": shape["extent"],
        "principal_axis": shape["principal_axis"],
        "symmetry_score": shape["symmetry_score"],
        "cylindrical_score": shape["cylindrical_score"],
        "flatness_score": shape["flatness_score"],
        **pointcloud,
        **grasp,
        "initial_quality_score": round(scan_quality * 100.0, 2),
        "epsilon_applicable": object_type != "unsupported",
        "quality_metric_applicability": {
            "two_contact_epsilon": object_type in {"cylindrical_bottle", "rectangular_object", "unknown_graspable"},
            "pointcloud_contact_validation": True,
            "davis_motion": "post_grasp_only",
        },
        "limitations": limitations,
        "profile_confidence": round(scan_quality, 4),
    }
    return json_safe(profile)


def _scan_quality(colors: dict[str, Any], geometry: dict[str, Any], shape: dict[str, Any], pointcloud: dict[str, Any]) -> float:
    return float(np.clip(
        0.20 * float(colors.get("appearance_confidence") or 0.0)
        + 0.30 * float(geometry.get("geometry_confidence") or 0.0)
        + 0.25 * float(shape.get("solidity") or 0.0)
        + 0.15 * float(shape.get("cylindrical_score") or shape.get("symmetry_score") or 0.0)
        + 0.10 * float(pointcloud.get("pointcloud_quality") or 0.0),
        0.0,
        1.0,
    ))


def _pointcloud_profile(status: dict[str, Any]) -> dict[str, Any]:
    extents = status.get("oriented_extents_mm") or status.get("bottle_pointcloud_extents_mm") or status.get("axis_aligned_extents_mm")
    quality = 0.0
    if status.get("cloud_valid_for_dimension_estimation") or status.get("bottle_dimensions_valid"):
        quality += 0.45
    if status.get("cloud_valid_for_contact_validation") or status.get("contact_points_on_bottle_cloud"):
        quality += 0.25
    valid_depth = _num(status.get("depth_valid_ratio") or status.get("bottle_pointcloud_valid_depth_ratio"), 0.0)
    quality += 0.30 * min(valid_depth or 0.0, 1.0)
    return {
        "representative_pointcloud_path": status.get("latest_realsense_pointcloud_ply"),
        "bottle_or_object_cloud_path": status.get("latest_bottle_pointcloud_ply"),
        "centroid_mm": status.get("robust_centroid_xyz_mm") or status.get("centroid_xyz_mm") or status.get("bottle_pointcloud_centroid_mm"),
        "extents_mm": extents,
        "principal_axis_3d": status.get("principal_axis") or status.get("bottle_principal_axis_m"),
        "point_count": status.get("bottle_filtered_point_count") or status.get("bottle_pointcloud_point_count") or status.get("point_count"),
        "valid_depth_ratio": valid_depth,
        "pointcloud_quality": round(float(np.clip(quality, 0.0, 1.0)), 4),
    }


def generic_object_profile() -> dict[str, Any]:
    now = utc_iso()
    return {
        "object_id": GENERIC_OBJECT_ID,
        "display_name": "Generic Graspable Object",
        "object_type": "unknown_graspable",
        "created_at": now,
        "updated_at": now,
        "profile_version": PROFILE_VERSION,
        "source": "generic_mode",
        "active_target_allowed": True,
        "notes": "Generic mode uses live RGB-D geometry only and does not claim known-object confidence.",
        "representative_rgb_snapshot_path": None,
        "dominant_colors": [],
        "color_histogram": [],
        "grayscale_template_path": None,
        "edge_template_path": None,
        "contour_descriptors": {},
        "keypoint_descriptors": None,
        "appearance_confidence": 0.0,
        "expected_bbox_aspect_range": [0.35, 6.0],
        "expected_oriented_aspect_range": [0.35, 6.0],
        "expected_height_mm": None,
        "expected_width_mm": None,
        "expected_depth_mm": None,
        "expected_diameter_mm": None,
        "expected_grip_width_mm": {"min": 25.0, "max": 130.0},
        "expected_depth_range_mm": None,
        "geometry_confidence": 0.35,
        "contour_points_normalized": [],
        "mask_area_ratio": None,
        "solidity": None,
        "extent": None,
        "principal_axis": [0.0, 1.0],
        "symmetry_score": None,
        "cylindrical_score": None,
        "flatness_score": None,
        "representative_pointcloud_path": None,
        "bottle_or_object_cloud_path": None,
        "centroid_mm": None,
        "extents_mm": None,
        "principal_axis_3d": None,
        "point_count": None,
        "valid_depth_ratio": None,
        "pointcloud_quality": 0.0,
        "preferred_grasp_regions": [{"body_fraction": 0.50, "score": 0.55}],
        "forbidden_regions": [],
        "preferred_body_fraction": 0.50,
        "contact_patch_height_px": 34,
        "contact_patch_width_px": 0,
        "expected_contact_count": 2,
        "compatible_embodiments": ["parallel_jaw_two_finger"],
        "grasp_notes": "Conservative generic side grasp; requires stronger live depth/contact validation.",
        "initial_quality_score": 35.0,
        "epsilon_applicable": True,
        "quality_metric_applicability": {"two_contact_epsilon": True, "pointcloud_contact_validation": True},
        "limitations": ["No saved object-specific dimensions; require stronger live validation."],
        "profile_confidence": 0.35,
    }


def legacy_blue_bottle_profile(legacy_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    legacy_profile = legacy_profile or {}
    now = utc_iso()
    diameter = legacy_profile.get("expected_diameter_mm") or {"min": 45, "max": 85}
    height = legacy_profile.get("expected_height_mm") or {"min": 140, "max": 280}
    return {
        **generic_object_profile(),
        "object_id": LEGACY_BLUE_BOTTLE_ID,
        "display_name": "Blue Bottle",
        "object_type": "cylindrical_bottle",
        "created_at": now,
        "updated_at": now,
        "source": "legacy_known_bottle_profile",
        "notes": "Backward-compatible profile for the original known bottle.",
        "expected_height_mm": height,
        "expected_width_mm": diameter,
        "expected_diameter_mm": diameter,
        "expected_grip_width_mm": diameter,
        "preferred_grasp_regions": [
            {"body_fraction": f, "score": 0.90}
            for f in (legacy_profile.get("preferred_grasp_body_fractions") or [0.45, 0.52, 0.60])
        ],
        "preferred_body_fraction": (legacy_profile.get("preferred_grasp_body_fractions") or [0.52])[0],
        "contact_patch_height_px": 34,
        "profile_confidence": 0.80,
        "initial_quality_score": 80.0,
        "limitations": [],
        "grasp_notes": "Legacy known-bottle side grasp priors preserved for the production reference path.",
    }


def profile_to_legacy_grasp_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(profile or generic_object_profile())
    grip = profile.get("expected_grip_width_mm") or profile.get("expected_diameter_mm") or {"min": 30, "max": 130}
    height = profile.get("expected_height_mm") or {"min": 80, "max": 320}
    preferred = [r.get("body_fraction") for r in profile.get("preferred_grasp_regions") or [] if r.get("body_fraction") is not None]
    if profile.get("preferred_body_fraction") is not None and profile.get("preferred_body_fraction") not in preferred:
        preferred.insert(0, profile.get("preferred_body_fraction"))
    preferred = [float(x) for x in preferred if 0.0 <= float(x) <= 1.0] or [0.50]
    forbidden = profile.get("forbidden_regions") or []
    avoid_top = 0.15
    avoid_bottom = 0.15
    for region in forbidden:
        if _num(region.get("body_fraction_min")) == 0.0:
            avoid_top = max(avoid_top, _num(region.get("body_fraction_max"), avoid_top))
        if _num(region.get("body_fraction_max")) == 1.0:
            avoid_bottom = max(avoid_bottom, 1.0 - _num(region.get("body_fraction_min"), 1.0 - avoid_bottom))
    return {
        "object_name": profile.get("object_id") or GENERIC_OBJECT_ID,
        "display_name": profile.get("display_name") or "Generic Graspable Object",
        "object_type": profile.get("object_type") or "unknown_graspable",
        "expected_diameter_mm": grip,
        "expected_height_mm": height,
        "preferred_grasp_body_fractions": preferred[:3],
        "fallback_grasp_body_fractions": preferred[3:] or [0.35, 0.65],
        "avoid_top_fraction": avoid_top,
        "avoid_bottom_fraction": avoid_bottom,
        "preferred_patch_height_mm": profile.get("preferred_patch_height_mm") or 50,
        "assumed_friction_coefficient": 0.5,
        "profile_path": profile.get("profile_path"),
        "profile_type": "object_profile",
        "profile_confidence": profile.get("profile_confidence"),
        "active_target_allowed": profile.get("active_target_allowed", True),
        "notes": profile.get("notes") or profile.get("grasp_notes"),
    }


def observation_descriptor(rgb: np.ndarray | None = None, mask: np.ndarray | None = None, depth: np.ndarray | None = None, intrinsics: dict[str, Any] | None = None, pointcloud_status: dict[str, Any] | None = None) -> dict[str, Any]:
    mask_u8 = _as_mask(mask, _rgb_array(rgb).shape[:2] if _rgb_array(rgb) is not None else None)
    return {
        **color_descriptors(rgb, mask_u8),
        **shape_descriptors(mask_u8),
        **geometry_from_mask_depth(mask_u8, depth=depth, intrinsics=intrinsics),
        **_pointcloud_profile(pointcloud_status or {}),
    }


def match_profiles(observation: dict[str, Any], profiles: Iterable[dict[str, Any]], strong_threshold: float = 0.72, ambiguous_delta: float = 0.08) -> dict[str, Any]:
    candidates = []
    for profile in profiles:
        if not profile or profile.get("active_target_allowed") is False:
            continue
        score_parts = {
            "color_similarity": _hist_similarity(observation.get("color_histogram"), profile.get("color_histogram")),
            "contour_similarity": _scalar_similarity(observation.get("oriented_aspect_ratio"), (profile.get("contour_descriptors") or {}).get("oriented_aspect_ratio") or _mid(profile.get("expected_oriented_aspect_range"))),
            "aspect_similarity": _scalar_similarity(observation.get("aspect_ratio"), _mid(profile.get("expected_bbox_aspect_range"))),
            "dimension_similarity": _dimension_similarity(observation, profile),
            "mask_area_ratio_similarity": _scalar_similarity(observation.get("mask_area_ratio"), profile.get("mask_area_ratio")),
        }
        score = (
            0.30 * score_parts["color_similarity"]
            + 0.25 * score_parts["contour_similarity"]
            + 0.20 * score_parts["aspect_similarity"]
            + 0.15 * score_parts["dimension_similarity"]
            + 0.10 * score_parts["mask_area_ratio_similarity"]
        )
        candidates.append({
            "object_id": profile.get("object_id"),
            "display_name": profile.get("display_name"),
            "profile_match_score": round(float(np.clip(score, 0.0, 1.0)), 4),
            **{k: round(float(v), 4) for k, v in score_parts.items()},
        })
    candidates.sort(key=lambda item: item["profile_match_score"], reverse=True)
    if not candidates:
        status = "NO_SAVED_PROFILES"
        selected = None
    elif candidates[0]["profile_match_score"] < strong_threshold:
        status = "NO_MATCH"
        selected = None
    elif len(candidates) > 1 and candidates[0]["profile_match_score"] - candidates[1]["profile_match_score"] <= ambiguous_delta:
        status = "AMBIGUOUS"
        selected = None
    else:
        status = "MATCHED"
        selected = candidates[0]["object_id"]
    return {
        "profile_match_status": status,
        "active_profile_match_score": candidates[0]["profile_match_score"] if candidates else 0.0,
        "matched_object_id": selected,
        "profile_match_candidates": candidates[:5],
    }


def _hist_similarity(a, b) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.0
    n = min(len(a), len(b))
    av = np.asarray(a[:n], dtype=float)
    bv = np.asarray(b[:n], dtype=float)
    denom = max(float(np.linalg.norm(av) * np.linalg.norm(bv)), 1e-9)
    return float(np.clip(float(av @ bv) / denom, 0.0, 1.0))


def _mid(value) -> float | None:
    if isinstance(value, dict):
        lo = _num(value.get("min"))
        hi = _num(value.get("max"))
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lo = _num(value[0])
        hi = _num(value[1])
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0
    return _num(value)


def _scalar_similarity(a, b) -> float:
    a = _num(a)
    b = _num(b)
    if a is None or b is None or b == 0:
        return 0.0
    return float(np.clip(1.0 - abs(a - b) / max(abs(b), 1.0), 0.0, 1.0))


def _dimension_similarity(observation: dict[str, Any], profile: dict[str, Any]) -> float:
    obs_w = _mid(observation.get("expected_width_mm")) or _mid(observation.get("expected_grip_width_mm"))
    prof_w = _mid(profile.get("expected_width_mm")) or _mid(profile.get("expected_grip_width_mm"))
    obs_h = _mid(observation.get("expected_height_mm"))
    prof_h = _mid(profile.get("expected_height_mm"))
    scores = [_scalar_similarity(obs_w, prof_w), _scalar_similarity(obs_h, prof_h)]
    scores = [s for s in scores if s > 0]
    return float(sum(scores) / len(scores)) if scores else 0.0


def artifact_paths_ok(profile: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or Path.cwd()
    keys = [
        "representative_rgb_snapshot_path",
        "grayscale_template_path",
        "edge_template_path",
        "representative_pointcloud_path",
        "bottle_or_object_cloud_path",
    ]
    missing = []
    for key in keys:
        value = profile.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            missing.append({"field": key, "path": str(value)})
    return {
        "artifact_paths_valid": not missing,
        "missing_artifacts": missing,
    }

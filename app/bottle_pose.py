import cv2
import json
import numpy as np
from pathlib import Path

from depth_utils import depth_stats, valid_depth_mask


MIN_COMPONENT_AREA = 800
DETECTOR_REFERENCE_SHAPE = (270, 480)
CAP_LIKE_MIN_CONFIDENCE = 0.55
CAP_LIKE_MIN_END_WINDOW_SUPPORT_FRACTION = 0.80
PROJECT_ROOT = Path(__file__).resolve().parent
KNOWN_BOTTLE_PROFILE_PATH = PROJECT_ROOT / "known_bottle_grasp_profile.json"


def detector_resolution_scale(image_shape):
    """Return the linear detector scale relative to the live 480x270 stream.

    The real-time search deliberately runs this detector at half resolution.
    Pixel-area and morphology constants must shrink with that image; otherwise
    a valid 2,800 px full-resolution bottle becomes a 700 px component and is
    discarded by the fixed 800 px threshold before any RGB-D checks run.
    Values are capped at one so existing full/high-resolution behavior is not
    broadened.
    """
    try:
        height, width = [float(value) for value in image_shape[:2]]
    except (TypeError, ValueError):
        return 1.0
    reference_height, reference_width = DETECTOR_REFERENCE_SHAPE
    area_ratio = (height * width) / max(
        float(reference_height * reference_width),
        1.0,
    )
    return float(np.clip(np.sqrt(max(area_ratio, 0.0)), 0.25, 1.0))


def scaled_detector_area(base_area, image_shape):
    scale = detector_resolution_scale(image_shape)
    return max(32, int(round(float(base_area) * scale * scale)))


def scaled_detector_length(base_length, image_shape, minimum=3):
    return max(int(minimum), int(round(float(base_length) * detector_resolution_scale(image_shape))))


def scaled_detector_kernel(base_size, image_shape):
    size = scaled_detector_length(base_size, image_shape, minimum=3)
    if size % 2 == 0:
        size += 1
    return size


def load_known_bottle_profile(path=KNOWN_BOTTLE_PROFILE_PATH):
    """Load the known water-bottle profile without making detection depend on it."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def score_from_range(value, good, bad):
    if value is None:
        return 0.0
    value = abs(float(value))
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(np.clip(1.0 - (value - good) / max(bad - good, 1e-6), 0.0, 1.0))


def depth_geometry_evidence(mask, depth, exclusion_mask=None):
    """Estimate whether a candidate is a coherent 3-D foreground surface.

    This is object-identity evidence from aligned depth, not semantic bottle
    recognition and not a physical dimension measurement.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    unavailable = {
        "status": "NOT_TESTED_MISSING_ALIGNED_DEPTH",
        "surface_coherence_score": None,
        "surface_depth_mad_mm": None,
        "surface_depth_interdecile_range_mm": None,
        "foreground_separation_mm": None,
        "foreground_separation_score": None,
        "foreground_separated": None,
        "far_background_reference_mm": None,
        "far_background_separation_mm": None,
        "far_background_support_fraction": 0.0,
        "far_background_support_pixel_count": 0,
        "far_background_min_required_pixel_count": 32,
        "far_background_sector_support_counts": [0] * 8,
        "far_background_supported_sector_count": 0,
        "far_background_opposite_sector_pair_count": 0,
        "far_background_distributed": False,
        "mixed_boundary_foreground_supported": False,
        "surrounding_depth_valid_ratio": 0.0,
        "surrounding_depth_mad_mm": None,
        "depth_geometry_score": None,
        "physically_validated": False,
    }
    if depth is None or np.asarray(depth).shape != mask_u8.shape or not np.any(mask_u8):
        return unavailable

    depth_arr = np.asarray(depth, dtype=float)
    inside_valid = (mask_u8 > 0) & valid_depth_mask(depth_arr)
    inside_values = depth_arr[inside_valid]
    if inside_values.size < max(80, int(round(np.count_nonzero(mask_u8) * 0.18))):
        unavailable["status"] = "NOT_TESTED_INSUFFICIENT_CANDIDATE_DEPTH"
        return unavailable

    ys, xs = np.where(mask_u8 > 0)
    minor_extent = max(1, min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    ring_radius = int(np.clip(round(0.14 * minor_extent), 4, 14))
    kernel_size = 2 * ring_radius + 1
    dilated = cv2.dilate(mask_u8, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    ring = (dilated > 0) & (mask_u8 == 0)
    if exclusion_mask is not None and np.asarray(exclusion_mask).shape == mask_u8.shape:
        ring &= np.asarray(exclusion_mask) == 0
    ring_valid = ring & valid_depth_mask(depth_arr)
    ring_values = depth_arr[ring_valid]
    ring_pixels = int(np.count_nonzero(ring))
    ring_valid_ratio = float(ring_values.size / max(ring_pixels, 1))

    inside_median = float(np.median(inside_values))
    inside_mad = float(np.median(np.abs(inside_values - inside_median)))
    inside_interdecile_range = float(
        np.percentile(inside_values, 90.0) - np.percentile(inside_values, 10.0)
    )
    coherence_band_mm = max(35.0, 0.08 * inside_median)
    surface_coherence = float(np.mean(np.abs(inside_values - inside_median) <= coherence_band_mm))

    foreground_separation = None
    foreground_score = None
    foreground_separated = None
    far_background_reference = None
    far_background_separation = None
    far_background_support_fraction = 0.0
    far_background_support_pixel_count = 0
    far_background_min_required_pixel_count = 32
    far_background_sector_support_counts = [0] * 8
    far_background_supported_sector_count = 0
    far_background_opposite_sector_pair_count = 0
    far_background_distributed = False
    mixed_boundary_foreground_supported = False
    surrounding_mad = None
    if ring_values.size >= max(80, int(round(0.30 * ring_pixels))):
        surrounding_median = float(np.median(ring_values))
        surrounding_mad = float(np.median(np.abs(ring_values - surrounding_median)))
        foreground_separation = float(surrounding_median - inside_median)
        # A foreground object must be closer than its surrounding ring.  Using
        # the absolute separation here would incorrectly promote a coloured
        # recess or a shape drawn on a nearer support plane as foreground.
        foreground_score = float(np.clip((foreground_separation - 8.0) / 72.0, 0.0, 1.0))
        foreground_separated = bool(foreground_separation >= 15.0)
        # Dark, curved bottles can produce stereo edge-fill at the bottle depth
        # over most of this narrow ring even when the physical background is
        # much farther away.  Preserve the conservative median decision, but
        # also report whether a material, coherent part of the same ring sees
        # a distinctly farther surface.  The demo identity policy may use this
        # only together with strong known-blue/profile evidence; it is not a
        # generic replacement for the median foreground gate.
        far_threshold_mm = max(40.0, 6.0 * max(inside_mad, 1.0))
        far_flags = ring_values >= inside_median + far_threshold_mm
        far_values = ring_values[far_flags]
        far_background_support_pixel_count = int(far_values.size)
        far_background_support_fraction = float(far_values.size / max(ring_values.size, 1))
        if far_values.size:
            far_background_reference = float(np.median(far_values))
            far_background_separation = float(far_background_reference - inside_median)

            # A few far pixels in one corner are commonly produced by holes,
            # a support-plane edge, or stereo speckle. A freestanding bottle
            # must expose farther background around distributed boundary
            # sectors, including at least one pair of opposite sectors. Scale
            # x/y before angular binning so a tall bottle does not collapse all
            # of its side support into only the left/right bins.
            ring_y, ring_x = np.where(ring_valid)
            center_x = float(np.mean(xs))
            center_y = float(np.mean(ys))
            half_width = max(0.5 * float(xs.max() - xs.min() + 1), 1.0)
            half_height = max(0.5 * float(ys.max() - ys.min() + 1), 1.0)
            angles = np.mod(
                np.arctan2(
                    (ring_y - center_y) / half_height,
                    (ring_x - center_x) / half_width,
                ),
                2.0 * np.pi,
            )
            sectors = (
                np.floor((angles + np.pi / 8.0) / (np.pi / 4.0)).astype(int)
                % 8
            )
            far_sectors = sectors[far_flags]
            far_background_sector_support_counts = [
                int(np.count_nonzero(far_sectors == sector_index))
                for sector_index in range(8)
            ]
            minimum_per_sector = max(6, int(np.ceil(0.01 * ring_values.size)))
            supported_sectors = {
                sector_index
                for sector_index, count in enumerate(far_background_sector_support_counts)
                if count >= minimum_per_sector
            }
            far_background_supported_sector_count = len(supported_sectors)
            far_background_opposite_sector_pair_count = sum(
                1
                for sector_index in range(4)
                if sector_index in supported_sectors
                and sector_index + 4 in supported_sectors
            )
            far_background_distributed = bool(
                far_background_supported_sector_count >= 2
                and far_background_opposite_sector_pair_count >= 1
            )

        minimum_surface_range_mm = max(6.0, 0.015 * inside_median)
        mixed_boundary_foreground_supported = bool(
            surface_coherence >= 0.85
            and far_background_support_fraction >= 0.08
            and far_background_support_pixel_count
            >= far_background_min_required_pixel_count
            and far_background_separation is not None
            and far_background_separation >= far_threshold_mm
            and far_background_distributed
            and inside_interdecile_range >= minimum_surface_range_mm
        )

    if foreground_score is None:
        geometry_score = surface_coherence
        status = "ESTIMATED_FROM_CANDIDATE_ALIGNED_DEPTH_ONLY"
    else:
        geometry_score = 0.45 * surface_coherence + 0.55 * foreground_score
        status = "ESTIMATED_FROM_ALIGNED_DEPTH_NEIGHBORHOOD"
    return {
        "status": status,
        "surface_coherence_score": surface_coherence,
        "surface_depth_mad_mm": inside_mad,
        "surface_depth_interdecile_range_mm": inside_interdecile_range,
        "foreground_separation_mm": foreground_separation,
        "foreground_separation_score": foreground_score,
        "foreground_separated": foreground_separated,
        "far_background_reference_mm": far_background_reference,
        "far_background_separation_mm": far_background_separation,
        "far_background_support_fraction": far_background_support_fraction,
        "far_background_support_pixel_count": far_background_support_pixel_count,
        "far_background_min_required_pixel_count": far_background_min_required_pixel_count,
        "far_background_sector_support_counts": far_background_sector_support_counts,
        "far_background_supported_sector_count": far_background_supported_sector_count,
        "far_background_opposite_sector_pair_count": far_background_opposite_sector_pair_count,
        "far_background_distributed": far_background_distributed,
        "mixed_boundary_foreground_supported": mixed_boundary_foreground_supported,
        "surrounding_depth_valid_ratio": ring_valid_ratio,
        "surrounding_depth_mad_mm": surrounding_mad,
        "depth_geometry_score": float(np.clip(geometry_score, 0.0, 1.0)),
        "physically_validated": False,
    }


def contour_quality_metrics(mask, contour, bbox_xywh, image_shape, rgb=None, depth=None, depth_exclusion_mask=None):
    """Compute lightweight multi-cue shape/mask/depth scores for a candidate."""
    img_h, img_w = image_shape[:2]
    x, y, w, h = bbox_xywh
    area = float(np.count_nonzero(mask))
    bbox_area = float(max(w * h, 1))
    contour_area = float(cv2.contourArea(contour)) if contour is not None else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None and len(contour) >= 3 else 0.0
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    fill_ratio = float(area / bbox_area)
    solidity = float(contour_area / max(hull_area, 1.0))
    extent = float(contour_area / bbox_area)
    aspect = float(max(w, h) / max(min(w, h), 1))
    rect = cv2.minAreaRect(contour) if contour is not None and len(contour) >= 5 else ((0, 0), (w, h), 0)
    rw, rh = rect[1]
    oriented_aspect = float(max(rw, rh) / max(min(rw, rh), 1.0)) if rw > 0 and rh > 0 else aspect
    axis, angle_deg, _axis_length = pca_axis_from_mask(mask)
    verticality = float(abs(axis[1]))
    color_fraction = bottle_color_fraction(rgb, mask)
    chromatic_fraction = chromatic_color_fraction(rgb, mask)
    ds = depth_stats(depth, mask.astype(np.uint8)) if depth is not None else {"median_mm": None, "valid_ratio": 0.0, "std_mm": None}
    depth_geometry = depth_geometry_evidence(mask, depth, exclusion_mask=depth_exclusion_mask)
    edge_support = 0.0
    if contour is not None and perimeter > 0:
        edge_support = float(np.clip(contour_area / max(perimeter * perimeter, 1.0) * 20.0, 0.0, 1.0))
    circularity = float(np.clip(4.0 * np.pi * contour_area / max(perimeter * perimeter, 1.0), 0.0, 1.0))
    rgb_boundary_score = None
    if rgb is not None and np.asarray(rgb).shape[:2] == mask.shape:
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        boundary_kernel = np.ones((5, 5), np.uint8)
        inner_band = (mask_u8 > 0) & (cv2.erode(mask_u8, boundary_kernel) == 0)
        outer_band = (cv2.dilate(mask_u8, boundary_kernel) > 0) & (mask_u8 == 0)
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
        if np.count_nonzero(inner_band) >= 20 and np.count_nonzero(outer_band) >= 20:
            contrast = abs(float(np.median(gray[inner_band])) - float(np.median(gray[outer_band])))
            rgb_boundary_score = float(np.clip(contrast / 64.0, 0.0, 1.0))
    depth_consistency = score_from_range(ds.get("std_mm"), 25.0, 95.0)
    border_touch = bool(x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2)
    contamination = 0.0
    if border_touch:
        contamination += 0.25
    if fill_ratio > 0.92 or fill_ratio < 0.18:
        contamination += 0.25
    if color_fraction < 0.15 and area / max(img_h * img_w, 1) > 0.20:
        contamination += 0.35
    return {
        "mask_area_score": float(np.clip((area / max(img_h * img_w, 1)) / 0.08, 0.0, 1.0)),
        "contour_quality_score": float(np.clip(0.45 * solidity + 0.35 * extent + 0.20 * edge_support, 0.0, 1.0)),
        "solidity_score": float(np.clip(solidity, 0.0, 1.0)),
        "extent_score": float(np.clip(extent, 0.0, 1.0)),
        "oriented_aspect_score": float(np.clip(1.0 - abs(oriented_aspect - 3.2) / 3.2, 0.0, 1.0)),
        "verticality_score": verticality,
        "edge_support_score": edge_support,
        "circularity_score": circularity,
        "rgb_boundary_score": rgb_boundary_score,
        "RGB_contrast_score": float(np.clip(max(color_fraction / 0.55, chromatic_fraction / 0.70), 0.0, 1.0)),
        "depth_support_score": float(np.clip(ds.get("valid_ratio", 0.0), 0.0, 1.0)),
        "depth_consistency_score": depth_consistency,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect,
        "oriented_aspect_ratio": oriented_aspect,
        "bottle_color_fraction": color_fraction,
        "chromatic_color_fraction": chromatic_fraction,
        "depth_valid_ratio": ds.get("valid_ratio", 0.0),
        "depth_median_mm": ds.get("median_mm"),
        "depth_std_mm": ds.get("std_mm"),
        "depth_geometry_status": depth_geometry["status"],
        "depth_surface_coherence_score": depth_geometry["surface_coherence_score"],
        "depth_surface_mad_mm": depth_geometry["surface_depth_mad_mm"],
        "depth_surface_interdecile_range_mm": depth_geometry[
            "surface_depth_interdecile_range_mm"
        ],
        "depth_foreground_separation_mm": depth_geometry["foreground_separation_mm"],
        "depth_foreground_separation_score": depth_geometry["foreground_separation_score"],
        "depth_foreground_separated": depth_geometry["foreground_separated"],
        "depth_far_background_reference_mm": depth_geometry["far_background_reference_mm"],
        "depth_far_background_separation_mm": depth_geometry["far_background_separation_mm"],
        "depth_far_background_support_fraction": depth_geometry["far_background_support_fraction"],
        "depth_far_background_support_pixel_count": depth_geometry[
            "far_background_support_pixel_count"
        ],
        "depth_far_background_min_required_pixel_count": depth_geometry[
            "far_background_min_required_pixel_count"
        ],
        "depth_far_background_sector_support_counts": depth_geometry[
            "far_background_sector_support_counts"
        ],
        "depth_far_background_supported_sector_count": depth_geometry[
            "far_background_supported_sector_count"
        ],
        "depth_far_background_opposite_sector_pair_count": depth_geometry[
            "far_background_opposite_sector_pair_count"
        ],
        "depth_far_background_distributed": depth_geometry[
            "far_background_distributed"
        ],
        "depth_mixed_boundary_foreground_supported": depth_geometry["mixed_boundary_foreground_supported"],
        "depth_surrounding_valid_ratio": depth_geometry["surrounding_depth_valid_ratio"],
        "depth_surrounding_mad_mm": depth_geometry["surrounding_depth_mad_mm"],
        "depth_geometry_score": depth_geometry["depth_geometry_score"],
        "mask_contamination_score": float(np.clip(contamination, 0.0, 1.0)),
        "border_touch": border_touch,
        "axis_angle_degrees": float(angle_deg),
    }


def known_profile_agreement(candidate_metrics, profile=None):
    """Score agreement with the known-bottle profile using available image cues."""
    profile = profile or load_known_bottle_profile()
    failures = []
    preferred = profile.get("preferred_grasp_body_fractions") or []
    avoid_top = float(profile.get("avoid_top_fraction", 0.20))
    avoid_bottom = float(profile.get("avoid_bottom_fraction", 0.15))
    aspect = float(candidate_metrics.get("oriented_aspect_ratio") or candidate_metrics.get("aspect_ratio") or 0.0)
    color_fraction = float(candidate_metrics.get("bottle_color_fraction") or 0.0)
    chromatic_fraction = float(candidate_metrics.get("chromatic_color_fraction") or 0.0)
    depth_valid = float(candidate_metrics.get("depth_valid_ratio") or 0.0)
    solidity = float(candidate_metrics.get("solidity_score") or 0.0)
    extent = float(candidate_metrics.get("extent_score") or 0.0)
    contamination = float(candidate_metrics.get("mask_contamination_score") or 0.0)
    profile_aspect_score = float(np.clip(1.0 - abs(aspect - 3.2) / 3.5, 0.0, 1.0))
    body_fraction_score = 1.0 if preferred and 0.0 <= avoid_top < 0.45 and 0.0 <= avoid_bottom < 0.45 else 0.75
    color_score = float(np.clip(max(color_fraction / 0.45, chromatic_fraction / 0.65), 0.0, 1.0))
    depth_score = float(np.clip(depth_valid / 0.45, 0.0, 1.0))
    shape_score = float(np.clip(0.50 * profile_aspect_score + 0.25 * solidity + 0.25 * extent, 0.0, 1.0))
    if profile_aspect_score < 0.20:
        failures.append("oriented_aspect_outside_known_bottle_profile")
    if color_score < 0.15 and depth_score < 0.35:
        failures.append("weak_rgb_and_depth_support_for_known_bottle")
    if contamination > 0.50:
        failures.append("mask_contamination_too_high")
    agreement = float(np.clip(0.40 * shape_score + 0.25 * color_score + 0.25 * depth_score + 0.10 * body_fraction_score - 0.25 * contamination, 0.0, 1.0))
    return {
        "bottle_profile_agreement_score": agreement,
        "bottle_profile_failures": failures,
        "bottle_profile_valid": bool(agreement >= 0.35 and not failures),
        "profile_source": str(KNOWN_BOTTLE_PROFILE_PATH),
        "metric_dimension_status": "not_evaluated_missing_camera_intrinsics",
        "expected_height_range_mm": profile.get("expected_height_mm"),
        "expected_diameter_range_mm": profile.get("expected_diameter_mm"),
        "preferred_grasp_body_fractions": preferred,
        "fallback_grasp_body_fractions": profile.get("fallback_grasp_body_fractions") or [],
        "avoid_top_fraction": profile.get("avoid_top_fraction"),
        "avoid_bottom_fraction": profile.get("avoid_bottom_fraction"),
        "preferred_patch_height_mm": profile.get("preferred_patch_height_mm"),
    }


def fill_holes(mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape[:2]
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((mask_u8 | holes) > 0).astype(np.uint8)


def clean_mask(mask, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    if (
        np.any(cleaned[0, :] > 0)
        or np.any(cleaned[-1, :] > 0)
        or np.any(cleaned[:, 0] > 0)
        or np.any(cleaned[:, -1] > 0)
    ):
        return (cleaned > 0).astype(np.uint8)
    return fill_holes(cleaned)


def clean_mask_without_fill(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (cleaned > 0).astype(np.uint8)


def bottle_sized_rgb_components(mask, kernel_size=5, min_area=MIN_COMPONENT_AREA):
    """Preserve adjacent RGB instances before closing can bridge their gap."""
    kernel_size = scaled_detector_kernel(kernel_size, mask.shape)
    min_area = scaled_detector_area(min_area, mask.shape)
    minimum_extent = scaled_detector_length(25, mask.shape, minimum=10)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area < int(min_area) or w < minimum_extent or h < minimum_extent:
            continue
        component = cv2.morphologyEx(
            (labels == label).astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        if np.count_nonzero(component) >= int(min_area):
            components.append(component)
    if not components:
        cleaned = clean_mask_without_fill(mask, kernel_size=kernel_size)
        return [cleaned] if np.count_nonzero(cleaned) >= int(min_area) else []

    parents = list(range(len(components)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    bboxes = [cv2.boundingRect(component) for component in components]
    for left_index, left_bbox in enumerate(bboxes):
        lx, ly, lw, lh = left_bbox
        left_aspect = max(lw, lh) / max(min(lw, lh), 1)
        for right_index in range(left_index + 1, len(bboxes)):
            rx, ry, rw, rh = bboxes[right_index]
            right_aspect = max(rw, rh) / max(min(rw, rh), 1)
            if left_aspect >= 1.55 or right_aspect >= 1.55:
                continue
            horizontal_overlap = max(0, min(lx + lw, rx + rw) - max(lx, rx))
            vertical_overlap = max(0, min(ly + lh, ry + rh) - max(ly, ry))
            horizontal_ratio = horizontal_overlap / max(min(lw, rw), 1)
            vertical_ratio = vertical_overlap / max(min(lh, rh), 1)
            vertical_gap = max(0, max(ly, ry) - min(ly + lh, ry + rh))
            horizontal_gap = max(0, max(lx, rx) - min(lx + lw, rx + rw))
            aligned_fragments = (
                horizontal_ratio >= 0.55 and vertical_gap <= max(12.0, 0.25 * max(lh, rh))
            ) or (
                vertical_ratio >= 0.55 and horizontal_gap <= max(12.0, 0.25 * max(lw, rw))
            )
            if aligned_fragments:
                join(left_index, right_index)

    grouped = {}
    for index, component in enumerate(components):
        grouped.setdefault(root(index), []).append(component)
    merged_components = []
    for group in grouped.values():
        merged = np.zeros_like(opened, dtype=np.uint8)
        for component in group:
            merged |= component
        if len(group) > 1:
            merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)
        merged_components.append((merged > 0).astype(np.uint8))
    return merged_components


def split_overwide_rgb_bottle_cluster(mask, rgb, depth=None, min_area=MIN_COMPONENT_AREA):
    """Split a touching low-aspect cluster only at a strong supported RGB edge."""
    component = (np.asarray(mask) > 0).astype(np.uint8)
    if rgb is None or np.asarray(rgb).shape[:2] != component.shape:
        return [component]
    min_area = scaled_detector_area(min_area, component.shape)
    minimum_extent = scaled_detector_length(25, component.shape, minimum=10)
    split_extent = scaled_detector_length(50, component.shape, minimum=20)
    x, y, w, h = cv2.boundingRect(component)
    if w < split_extent or h < split_extent or max(w, h) / max(min(w, h), 1) >= 1.75:
        return [component]

    vertical_cut = h >= w
    span_start = x if vertical_cut else y
    span = w if vertical_cut else h
    cut_start = span_start + max(10, int(round(0.25 * span)))
    cut_end = span_start + min(span - 10, int(round(0.75 * span)))
    best = None
    rgb_arr = np.asarray(rgb, dtype=float)
    for cut in range(cut_start, cut_end + 1):
        first = component.copy()
        second = component.copy()
        if vertical_cut:
            first[:, cut:] = 0
            second[:, :cut] = 0
            support = np.any(component[:, max(0, cut - 2) : min(component.shape[1], cut + 2)] > 0, axis=1)
            gradient = np.mean(np.abs(rgb_arr[:, cut] - rgb_arr[:, cut - 1]), axis=1)
        else:
            first[cut:, :] = 0
            second[:cut, :] = 0
            support = np.any(component[max(0, cut - 2) : min(component.shape[0], cut + 2), :] > 0, axis=0)
            gradient = np.mean(np.abs(rgb_arr[cut, :] - rgb_arr[cut - 1, :]), axis=1)
        parts = []
        valid = True
        for part in (first, second):
            part_area = int(np.count_nonzero(part))
            px, py, pw, ph = cv2.boundingRect(part)
            part_aspect = max(pw, ph) / max(min(pw, ph), 1)
            if (
                part_area < int(min_area)
                or pw < minimum_extent
                or ph < minimum_extent
                or part_aspect < 1.55
            ):
                valid = False
                break
            if depth is not None and np.asarray(depth).shape == component.shape:
                if float(depth_stats(depth, part).get("valid_ratio", 0.0) or 0.0) < 0.18:
                    valid = False
                    break
            parts.append(part)
        if not valid or not np.any(support):
            continue
        edge_strength = float(np.percentile(gradient[support], 75))
        first_area = int(np.count_nonzero(parts[0]))
        second_area = int(np.count_nonzero(parts[1]))
        balance = min(first_area, second_area) / max(first_area + second_area, 1)
        score = edge_strength + 20.0 * balance
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "edge_strength": edge_strength,
                "parts": parts,
            }
    if best is None or best["edge_strength"] < 24.0:
        return [component]
    return [(part > 0).astype(np.uint8) for part in best["parts"]]


def largest_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def pca_axis_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size < 5:
        return np.array([0.0, 1.0]), 90.0, 0.0
    points = np.column_stack([xs, ys]).astype(float)
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    if axis[1] < 0:
        axis = -axis
    projections = points @ axis
    axis_length = float(np.max(projections) - np.min(projections))
    angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return axis, angle_deg, axis_length


def depth_supported_achromatic_endcap(mask, rgb, depth, exclusion_mask=None, axis_hint=None):
    """Extend an elongated RGB mask with one supported achromatic end region.

    This is a conservative RGB-D geometry estimator for a *cap-like* region,
    not a semantic bottle-cap classifier.  Only low-saturation pixels directly
    beyond the long-axis end of the source mask may be added, and every added
    pixel must have aligned depth close to the source surface.  The caller must
    keep its original body mask for grasp/contact reasoning.
    """
    source = (np.asarray(mask) > 0).astype(np.uint8)
    cap_mask = np.zeros_like(source, dtype=np.uint8)
    evidence = {
        "status": "NOT_TESTED_MISSING_ALIGNED_RGBD",
        "detected": False,
        "confidence": None,
        "minimum_confidence": CAP_LIKE_MIN_CONFIDENCE,
        "minimum_end_window_support_fraction": CAP_LIKE_MIN_END_WINDOW_SUPPORT_FRACTION,
        "minimum_area_px": None,
        "sibling_exclusion_applied": False,
        "sibling_exclusion_pixel_count": 0,
        "axis_source": None,
        "semantic_identity": "CAP_LIKE_END_REGION_NOT_SEMANTICALLY_CLASSIFIED",
        "measurement_type": "ESTIMATED_FROM_ALIGNED_RGBD_GEOMETRY",
        "physically_validated": False,
        "added_pixel_count": 0,
        "candidate_count": 0,
        "unbounded_surface_reject_count": 0,
        "axis_end_sign": None,
        "source_depth_median_mm": None,
        "end_region_depth_median_mm": None,
        "depth_delta_mm": None,
        "depth_valid_ratio": None,
        "end_reference_depth_median_mm": None,
        "end_reference_depth_source": None,
        "limitations": [
            "Low-saturation end geometry is not semantic cap recognition.",
            "Reflective metal may have insufficient RealSense depth and remain undetected.",
        ],
    }
    if (
        rgb is None
        or depth is None
        or np.asarray(rgb).shape[:2] != source.shape
        or np.asarray(depth).shape != source.shape
        or np.count_nonzero(source) < 25
    ):
        return source, cap_mask, evidence

    depth_arr = np.asarray(depth, dtype=float)
    ys, xs = np.where(source > 0)
    source_depth_samples = depth_arr[ys, xs]
    source_depth_values = source_depth_samples[valid_depth_mask(source_depth_samples)]
    source_pixel_count = int(xs.size)
    minimum_source_depth = max(20, int(round(source_pixel_count * 0.15)))
    if source_depth_values.size < minimum_source_depth:
        evidence["status"] = "NOT_TESTED_INSUFFICIENT_SOURCE_DEPTH"
        return source, cap_mask, evidence

    source_depth_median = float(np.median(source_depth_values))
    evidence["source_depth_median_mm"] = source_depth_median
    try:
        axis = np.asarray(axis_hint, dtype=float).reshape(-1)[:2]
    except (TypeError, ValueError):
        axis = np.empty(0, dtype=float)
    if axis.size == 2 and np.all(np.isfinite(axis)) and float(np.linalg.norm(axis)) > 1e-9:
        axis = axis / float(np.linalg.norm(axis))
        if axis[1] < 0:
            axis = -axis
        evidence["axis_source"] = "CALLER_REUSED_PCA_AXIS"
    else:
        axis, _angle_deg, _axis_length = pca_axis_from_mask(source)
        evidence["axis_source"] = "PCA_FROM_SOURCE_MASK"
    width_axis = np.array([-axis[1], axis[0]], dtype=float)
    points = np.column_stack([xs, ys]).astype(float)
    axial = points @ axis
    lateral = points @ width_axis
    axial_low, axial_high = [float(value) for value in np.percentile(axial, (1.0, 99.0))]
    lateral_low, lateral_high = [float(value) for value in np.percentile(lateral, (5.0, 95.0))]
    minor_width = max(1.0, lateral_high - lateral_low)
    major_length = max(1.0, axial_high - axial_low)
    if major_length / minor_width < 1.35:
        evidence["status"] = "NOT_APPLICABLE_NON_ELONGATED_SOURCE_MASK"
        return source, cap_mask, evidence

    height, width = source.shape
    lateral_center = 0.5 * (lateral_low + lateral_high)
    corridor_half_width = 0.72 * minor_width
    extension_length = float(np.clip(1.25 * minor_width, 6.0, 52.0))
    overlap_length = float(np.clip(0.12 * minor_width, 2.0, 6.0))
    adjacency_radius = int(np.clip(round(0.08 * minor_width), 2, 5))
    search_margin = int(np.ceil(extension_length + overlap_length + adjacency_radius + 3.0))
    roi_x1 = max(0, int(xs.min()) - search_margin)
    roi_y1 = max(0, int(ys.min()) - search_margin)
    roi_x2 = min(width, int(xs.max()) + search_margin + 1)
    roi_y2 = min(height, int(ys.max()) + search_margin + 1)
    source_crop = source[roi_y1:roi_y2, roi_x1:roi_x2]
    depth_crop = depth_arr[roi_y1:roi_y2, roi_x1:roi_x2]
    exclusion_crop = np.zeros_like(source_crop, dtype=bool)
    if exclusion_mask is not None and np.asarray(exclusion_mask).shape == source.shape:
        exclusion_radius = int(np.clip(round(0.45 * minor_width), 3, 24))
        exclusion_crop = cv2.dilate(
            (np.asarray(exclusion_mask)[roi_y1:roi_y2, roi_x1:roi_x2] > 0).astype(np.uint8),
            np.ones((2 * exclusion_radius + 1, 2 * exclusion_radius + 1), np.uint8),
            iterations=1,
        ) > 0
        evidence["sibling_exclusion_applied"] = True
        evidence["sibling_exclusion_pixel_count"] = int(np.count_nonzero(exclusion_crop))
    grid_y, grid_x = np.indices(source_crop.shape, dtype=float)
    grid_y += float(roi_y1)
    grid_x += float(roi_x1)
    grid_axial = grid_x * axis[0] + grid_y * axis[1]
    grid_lateral = grid_x * width_axis[0] + grid_y * width_axis[1]
    corridor = np.abs(grid_lateral - lateral_center) <= corridor_half_width

    hsv = cv2.cvtColor(
        np.asarray(rgb, dtype=np.uint8)[roi_y1:roi_y2, roi_x1:roi_x2],
        cv2.COLOR_RGB2HSV,
    )
    achromatic = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 45)
    global_depth_tolerance_mm = float(np.clip(0.08 * source_depth_median, 20.0, 45.0))
    outside_source = source_crop == 0
    adjacency_kernel = np.ones((2 * adjacency_radius + 1, 2 * adjacency_radius + 1), np.uint8)
    source_neighborhood = cv2.dilate(source_crop, adjacency_kernel, iterations=1) > 0
    minimum_area = max(8, int(round(0.01 * source_pixel_count)))
    evidence["minimum_area_px"] = minimum_area
    source_depth_valid_flags = valid_depth_mask(source_depth_samples)
    evidence["search_roi_xyxy"] = [roi_x1, roi_y1, roi_x2, roi_y2]
    candidates = []

    for end_sign in (-1, 1):
        if end_sign < 0:
            axial_window = (
                (grid_axial >= axial_low - extension_length)
                & (grid_axial < axial_low + overlap_length)
            )
            terminal_points = axial <= axial_low + 0.18 * major_length
        else:
            axial_window = (
                (grid_axial > axial_high - overlap_length)
                & (grid_axial <= axial_high + extension_length)
            )
            terminal_points = axial >= axial_high - 0.18 * major_length
        terminal_depth_values = source_depth_samples[terminal_points & source_depth_valid_flags]
        minimum_terminal_depth = max(8, int(round(0.03 * source_pixel_count)))
        if terminal_depth_values.size >= minimum_terminal_depth:
            end_reference_depth = float(np.median(terminal_depth_values))
            end_reference_source = "LOCAL_TERMINAL_BODY_DEPTH"
        else:
            end_reference_depth = source_depth_median
            end_reference_source = "GLOBAL_BODY_DEPTH_FALLBACK"
        depth_tolerance_mm = float(np.clip(0.08 * end_reference_depth, 20.0, 45.0))
        close_depth = valid_depth_mask(depth_crop) & (
            np.abs(depth_crop - end_reference_depth) <= depth_tolerance_mm
        )
        # Label the complete achromatic depth surface before restricting it to
        # the end neighborhood.  Otherwise a broad desk/support plane is
        # cropped by the corridor into a short cap-shaped component.
        supported_surface = (
            achromatic
            & close_depth
            & outside_source
            & ~exclusion_crop
        ).astype(np.uint8)
        if not np.any(supported_surface):
            continue
        connected = cv2.morphologyEx(
            supported_surface,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            connected,
            connectivity=8,
        )
        for label in range(1, component_count):
            component = labels == label
            # Do not turn morphology-filled or invalid-depth pixels into
            # evidence: the returned mask remains strictly depth-supported.
            full_supported_component = component & (supported_surface > 0)
            supported_component = full_supported_component & corridor & axial_window
            area = int(np.count_nonzero(supported_component))
            if area < minimum_area or not np.any(supported_component & source_neighborhood):
                continue
            full_surface_area = int(np.count_nonzero(full_supported_component))
            end_window_support_fraction = float(area / max(full_surface_area, 1))
            full_y_local, full_x_local = np.where(full_supported_component)
            touches_search_roi_edge = bool(
                full_x_local.size
                and (
                    full_x_local.min() <= 0
                    or full_y_local.min() <= 0
                    or full_x_local.max() >= source_crop.shape[1] - 1
                    or full_y_local.max() >= source_crop.shape[0] - 1
                )
            )
            if (
                touches_search_roi_edge
                or end_window_support_fraction < CAP_LIKE_MIN_END_WINDOW_SUPPORT_FRACTION
            ):
                evidence["unbounded_surface_reject_count"] += 1
                continue
            component_y_local, component_x_local = np.where(supported_component)
            component_x = component_x_local + roi_x1
            component_y = component_y_local + roi_y1
            if (
                component_x.min() <= 1
                or component_y.min() <= 1
                or component_x.max() >= width - 2
                or component_y.max() >= height - 2
            ):
                continue
            component_points = np.column_stack([component_x, component_y]).astype(float)
            component_axial = component_points @ axis
            component_lateral = component_points @ width_axis
            axial_span = float(np.ptp(component_axial)) if area > 1 else 0.0
            lateral_span = float(np.ptp(component_lateral)) if area > 1 else 0.0
            if (
                axial_span < 0.15 * minor_width
                or axial_span > 0.75 * minor_width
                or lateral_span < 0.50 * minor_width
                or lateral_span > 1.45 * minor_width
                or area > 0.55 * source_pixel_count
            ):
                continue
            component_depth_values = depth_crop[supported_component]
            component_depth_median = float(np.median(component_depth_values))
            depth_delta = abs(component_depth_median - end_reference_depth)
            lateral_ratio = lateral_span / minor_width
            axial_ratio = axial_span / minor_width
            shape_score = float(np.clip(
                0.55 * (1.0 - abs(lateral_ratio - 0.85) / 0.85)
                + 0.45 * (1.0 - axial_ratio / 0.75),
                0.0,
                1.0,
            ))
            depth_score = float(np.clip(1.0 - depth_delta / max(depth_tolerance_mm, 1.0), 0.0, 1.0))
            area_score = float(np.clip(area / max(0.16 * minor_width * minor_width, 1.0), 0.0, 1.0))
            confidence = float(np.clip(0.50 * depth_score + 0.30 * shape_score + 0.20 * area_score, 0.0, 1.0))
            candidates.append(
                {
                    "mask_crop": supported_component.astype(np.uint8),
                    "axis_end_sign": int(end_sign),
                    "area": area,
                    "depth_median_mm": component_depth_median,
                    "depth_delta_mm": depth_delta,
                    "end_reference_depth_median_mm": end_reference_depth,
                    "end_reference_depth_source": end_reference_source,
                    "depth_tolerance_mm": depth_tolerance_mm,
                    "depth_valid_ratio": 1.0,
                    "full_surface_area_px": full_surface_area,
                    "end_window_support_fraction": end_window_support_fraction,
                    "touches_search_roi_edge": touches_search_roi_edge,
                    "axial_span_px": axial_span,
                    "lateral_span_px": lateral_span,
                    "confidence": confidence,
                }
            )

    evidence["candidate_count"] = len(candidates)
    if not candidates:
        evidence["status"] = (
            "NOT_DETECTED_UNBOUNDED_ACHROMATIC_SURFACE"
            if evidence["unbounded_surface_reject_count"]
            else "NOT_DETECTED_NO_SUPPORTED_ACHROMATIC_END_REGION"
        )
        return source, cap_mask, evidence

    selected = max(candidates, key=lambda item: (item["confidence"], item["area"]))
    if selected["confidence"] < CAP_LIKE_MIN_CONFIDENCE:
        evidence.update(
            {
                "status": "NOT_DETECTED_LOW_CONFIDENCE_CAP_LIKE_REGION",
                "candidate_confidence": selected["confidence"],
                "axis_end_sign": selected["axis_end_sign"],
                "end_reference_depth_median_mm": selected["end_reference_depth_median_mm"],
                "end_reference_depth_source": selected["end_reference_depth_source"],
            }
        )
        return source, cap_mask, evidence
    selected_mask_crop = selected.pop("mask_crop")
    cap_mask[roi_y1:roi_y2, roi_x1:roi_x2] = selected_mask_crop
    extended = ((source > 0) | (cap_mask > 0)).astype(np.uint8)
    evidence.update(
        {
            "status": "ESTIMATED_FROM_ALIGNED_RGBD_CAP_LIKE_END_GEOMETRY",
            "detected": True,
            "confidence": selected["confidence"],
            "added_pixel_count": selected["area"],
            "axis_end_sign": selected["axis_end_sign"],
            "end_region_depth_median_mm": selected["depth_median_mm"],
            "depth_delta_mm": selected["depth_delta_mm"],
            "depth_valid_ratio": selected["depth_valid_ratio"],
            "full_surface_area_px": selected["full_surface_area_px"],
            "end_window_support_fraction": selected["end_window_support_fraction"],
            "touches_search_roi_edge": selected["touches_search_roi_edge"],
            "end_reference_depth_median_mm": selected["end_reference_depth_median_mm"],
            "end_reference_depth_source": selected["end_reference_depth_source"],
            "axial_span_px": selected["axial_span_px"],
            "lateral_span_px": selected["lateral_span_px"],
            "depth_tolerance_mm": selected["depth_tolerance_mm"],
            "global_depth_tolerance_mm": global_depth_tolerance_mm,
        }
    )
    return extended, cap_mask, evidence


def candidate_masks_from_depth(depth):
    valid = valid_depth_mask(depth)
    if np.count_nonzero(valid) < MIN_COMPONENT_AREA:
        return []
    values = depth[valid]
    nearest = float(np.min(values))
    percentiles = [
        nearest + 45.0,
        nearest + 90.0,
        float(np.percentile(values, 5)) + 25.0,
        float(np.percentile(values, 10)) + 30.0,
        float(np.percentile(values, 15)) + 20.0,
        float(np.percentile(values, 25)) + 35.0,
        float(np.percentile(values, 35)) + 50.0,
        float(np.percentile(values, 45)) + 70.0,
    ]
    masks = []
    for limit in sorted(set(round(v, 1) for v in percentiles)):
        masks.append(clean_mask(valid & (depth <= limit), kernel_size=5))
    return masks


def depth_complete_shadowed_rgb_mask(mask, depth, rgb=None):
    """Recover a shadowed side only when it is connected at the same depth.

    RGB remains the seed for bottle identity.  Aligned depth may complete that
    seed inside a bounded neighborhood, but cannot introduce a disconnected
    nearby object or an arbitrarily large same-depth surface.
    """
    seed = (np.asarray(mask) > 0).astype(np.uint8)
    if depth is None or np.asarray(depth).shape != seed.shape:
        return seed
    seed_area = int(np.count_nonzero(seed))
    if seed_area < scaled_detector_area(MIN_COMPONENT_AREA, seed.shape):
        return seed

    depth_arr = np.asarray(depth, dtype=float)
    seed_valid = (seed > 0) & valid_depth_mask(depth_arr)
    seed_values = depth_arr[seed_valid]
    if seed_values.size < max(40, int(round(0.18 * seed_area))):
        return seed
    median_mm = float(np.median(seed_values))
    mad_mm = float(np.median(np.abs(seed_values - median_mm)))
    depth_band_mm = float(np.clip(max(25.0, 4.0 * mad_mm, 0.08 * median_mm), 25.0, 75.0))
    same_depth = valid_depth_mask(depth_arr) & (np.abs(depth_arr - median_mm) <= depth_band_mm)

    x, y, width, height = cv2.boundingRect(seed)
    major = max(width, height)
    minor = max(1, min(width, height))
    padding = max(5, int(round(max(0.75 * minor, 0.18 * major))))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(seed.shape[1], x + width + padding)
    y2 = min(seed.shape[0], y + height + padding)
    bounded_depth = np.zeros_like(seed, dtype=np.uint8)
    bounded_depth[y1:y2, x1:x2] = same_depth[y1:y2, x1:x2].astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        bounded_depth,
        connectivity=8,
    )
    best_label = None
    best_overlap = 0
    for label in range(1, count):
        overlap = int(np.count_nonzero((labels == label) & (seed > 0)))
        if overlap > best_overlap:
            best_label = label
            best_overlap = overlap
    if best_label is None or best_overlap < max(16, int(round(0.08 * seed_area))):
        return seed

    completed = ((labels == best_label) | (seed > 0)).astype(np.uint8)
    completed_area = int(np.count_nonzero(completed))
    if completed_area > int(round(2.8 * seed_area)):
        return seed
    axis, _angle_deg, _axis_length = pca_axis_from_mask(seed)
    seed_y, seed_x = np.where(seed > 0)
    completed_y, completed_x = np.where(completed > 0)
    seed_projection = np.column_stack([seed_x, seed_y]).astype(float) @ axis
    completed_projection = np.column_stack([completed_x, completed_y]).astype(float) @ axis
    seed_length = float(np.max(seed_projection) - np.min(seed_projection))
    axial_tolerance = max(3.0, 0.04 * seed_length)
    if (
        float(np.min(completed_projection)) < float(np.min(seed_projection)) - axial_tolerance
        or float(np.max(completed_projection)) > float(np.max(seed_projection)) + axial_tolerance
    ):
        return seed
    if rgb is not None and np.asarray(rgb).shape[:2] == seed.shape:
        hsv = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2HSV)
        chromatic = (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 35)
        added = (completed > 0) & (seed == 0)
        added_pixels = int(np.count_nonzero(added))
        added_chromatic = int(np.count_nonzero(added & chromatic))
        # A second colored object at the same depth is not a shadow. Reject
        # completion instead of undoing the existing instance separation.
        if added_pixels and added_chromatic / added_pixels > 0.25:
            return seed
    completed_x, completed_y, completed_w, completed_h = cv2.boundingRect(completed)
    if (
        completed_x < x1
        or completed_y < y1
        or completed_x + completed_w > x2
        or completed_y + completed_h > y2
    ):
        return seed
    return completed


def candidate_masks_from_rgb(
    rgb,
    depth=None,
    allow_border=False,
    include_central_near_recovery=True,
):
    """Find known-blue and depth-consistent generic chromatic bottle regions."""
    if rgb is None:
        return []
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)

    # The demo bottle is dark navy. Under the D405's auto exposure its body can
    # fall below the old bright-blue thresholds even though blue still clearly
    # dominates the RGB channels. Keep hue bounded, but accept that low-light
    # navy evidence and let aligned depth/shape validation reject background.
    low_light_blue_dominance = (blue > red + 10) & (blue > green + 3)
    blue_hue = (
        (hue >= 82)
        & (hue <= 132)
        & (sat >= 35)
        & (val >= 20)
        & ((val >= 35) | low_light_blue_dominance)
    )
    blue_channel = (blue > red + 10) & (blue > green + 3) & (sat >= 25) & (val >= 18)
    mask = blue_hue | blue_channel
    minimum_area = scaled_detector_area(MIN_COMPONENT_AREA, mask.shape)
    minimum_width = scaled_detector_length(25, mask.shape, minimum=10)
    minimum_dark_height = scaled_detector_length(40, mask.shape, minimum=16)
    masks = []
    for component in bottle_sized_rgb_components(mask, kernel_size=5):
        for instance in split_overwide_rgb_bottle_cluster(component, rgb, depth):
            depth_components = depth_mode_components(instance, depth, min_area=MIN_COMPONENT_AREA)
            selected = depth_components if len(depth_components) > 1 else [instance]
            masks.extend(depth_complete_shadowed_rgb_mask(component, depth, rgb) for component in selected)
    # A television/monitor behind the demo bottle can be blue and physically
    # touch the bottle silhouette in RGB, creating one enormous connected
    # component.  Recover the real dark-navy body as its own conservative
    # central candidate before generic chromatic fusion.  The tighter channel
    # dominance/value gate is specific to the known demo bottle; normal RGB-D
    # shape, dimensions, and depth checks still decide whether it is usable.
    if (
        include_central_near_recovery
        and depth is not None
        and np.asarray(depth).shape == mask.shape
    ):
        height, width = mask.shape
        x1, x2 = int(round(0.32 * width)), int(round(0.68 * width))
        y1, y2 = int(round(0.16 * height)), int(round(0.97 * height))
        central_dark_navy = (
            (blue > red + 6)
            & (blue > green + 2)
            & (sat >= 18)
            & (val >= 12)
            & (val <= 145)
        ).astype(np.uint8)
        central_mask = np.zeros_like(central_dark_navy, dtype=np.uint8)
        central_mask[y1:y2, x1:x2] = central_dark_navy[y1:y2, x1:x2]
        central_mask = cv2.morphologyEx(
            central_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            central_mask,
            connectivity=8,
        )
        for label in range(1, num_labels):
            cx, cy, cw, ch, area = [int(value) for value in stats[label]]
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            component = (labels == label).astype(np.uint8)
            ds = depth_stats(depth, component)
            touches_search_edge = bool(
                cx <= x1 + 1
                or cx + cw >= x2 - 1
                or cy <= y1 + 1
                or cy + ch >= y2 - 1
            )
            if (
                area >= minimum_area
                and cw >= minimum_width
                and ch >= minimum_dark_height
                and aspect >= 1.45
                and not touches_search_edge
                and float(ds.get("valid_ratio") or 0.0) >= 0.45
            ):
                masks.append(component)
        # Depth may fill holes only after a strong-blue component owns the
        # candidate (see depth_complete_shadowed_rgb_mask above). Never anchor
        # a bottle candidate from the nearest depth percentile of the whole
        # center region: a hand or fixture can be closer and borrow a sparse
        # navy overlap, producing a false full-bottle mask.
    generic_masks = generic_chromatic_candidate_masks(rgb, depth, allow_border=allow_border)
    masks.extend(depth_complete_shadowed_rgb_mask(mask, depth, rgb) for mask in generic_masks)
    return masks


def bottle_color_fraction(rgb, mask):
    if rgb is None or np.count_nonzero(mask) == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    blue_like = (hue >= 82) & (hue <= 132) & (sat >= 35) & (val >= 30)
    return float(np.count_nonzero(blue_like & (mask > 0)) / max(np.count_nonzero(mask), 1))


def chromatic_color_fraction(rgb, mask):
    if rgb is None or np.count_nonzero(mask) == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    chromatic = (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 35)
    return float(np.count_nonzero(chromatic & (mask > 0)) / max(np.count_nonzero(mask), 1))


def generic_chromatic_candidate_masks(rgb, depth=None, allow_border=False):
    """Fuse differently colored bottle sections using image geometry and depth."""
    if rgb is None:
        return []
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    color_valid = (saturation >= 40) & (value >= 35)
    bands = (
        (hue <= 20) & color_valid,
        (hue > 20) & (hue <= 80) & color_valid,
        (hue > 80) & (hue <= 132) & color_valid,
        (hue > 132) & color_valid,
    )
    fragments = []
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    for band_index, band in enumerate(bands):
        opened = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        for label in range(1, num_labels):
            x, y, w, h, area = [int(value) for value in stats[label]]
            if area < 250 or w < 12 or h < 12:
                continue
            component = cv2.morphologyEx(
                (labels == label).astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel_close,
                iterations=2,
            )
            for depth_component in depth_mode_components(component, depth, min_area=250):
                ys, xs = np.where(depth_component > 0)
                if xs.size < 250:
                    continue
                component_x = int(xs.min())
                component_y = int(ys.min())
                component_w = int(xs.max() - xs.min() + 1)
                component_h = int(ys.max() - ys.min() + 1)
                ds = depth_stats(depth, depth_component) if depth is not None else {}
                fragments.append(
                    {
                        "band": band_index,
                        "mask": depth_component,
                        "bbox": (component_x, component_y, component_w, component_h),
                        "area": int(xs.size),
                        "depth_median_mm": ds.get("median_mm"),
                        "depth_valid_ratio": float(ds.get("valid_ratio", 0.0) or 0.0),
                    }
                )
    if not fragments:
        return []

    parents = list(range(len(fragments)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(fragments):
        for right_index in range(left_index + 1, len(fragments)):
            right = fragments[right_index]
            if left["band"] == right["band"] or not chromatic_fragments_compatible(left, right):
                continue
            join(left_index, right_index)

    groups = {}
    for index, fragment in enumerate(fragments):
        groups.setdefault(root(index), []).append(fragment)

    height, width = rgb.shape[:2]
    candidate_masks = []
    for group in groups.values():
        if len(group) >= 2:
            candidate = np.zeros((height, width), dtype=np.uint8)
            for fragment in group:
                candidate |= fragment["mask"]
            overlap_left = max(fragment["bbox"][0] for fragment in group)
            overlap_right = min(fragment["bbox"][0] + fragment["bbox"][2] for fragment in group)
            overlap_top = max(fragment["bbox"][1] for fragment in group)
            overlap_bottom = min(fragment["bbox"][1] + fragment["bbox"][3] for fragment in group)
            min_width = min(fragment["bbox"][2] for fragment in group)
            min_height = min(fragment["bbox"][3] for fragment in group)
            if overlap_right > overlap_left:
                corridor_pad = max(4, int(round(0.18 * min_width)))
                corridor_left = max(0, overlap_left - corridor_pad)
                corridor_right = min(width, overlap_right + corridor_pad)
                candidate[:, :corridor_left] = 0
                candidate[:, corridor_right:] = 0
                join_kernel = np.ones((19, 5), np.uint8)
            elif overlap_bottom > overlap_top:
                corridor_pad = max(4, int(round(0.18 * min_height)))
                corridor_top = max(0, overlap_top - corridor_pad)
                corridor_bottom = min(height, overlap_bottom + corridor_pad)
                candidate[:corridor_top, :] = 0
                candidate[corridor_bottom:, :] = 0
                join_kernel = np.ones((5, 19), np.uint8)
            else:
                # Groups only form through one of the overlap rules above.
                # Keep this fail-closed if malformed fragment metadata reaches
                # the merge path.
                continue
            candidate = cv2.morphologyEx(
                candidate,
                cv2.MORPH_CLOSE,
                join_kernel,
                iterations=1,
            )
            ys, xs = np.where(candidate > 0)
            if xs.size:
                bbox_x = int(xs.min())
                bbox_y = int(ys.min())
                bbox_w = int(xs.max() - xs.min() + 1)
                bbox_h = int(ys.max() - ys.min() + 1)
                combined_aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
                border_touch = (
                    bbox_x <= 2
                    or bbox_y <= 2
                    or bbox_x + bbox_w >= width - 2
                    or bbox_y + bbox_h >= height - 2
                )
                if (
                    np.count_nonzero(candidate) >= MIN_COMPONENT_AREA
                    and combined_aspect >= 1.45
                    and (allow_border or not border_touch)
                ):
                    candidate_masks.append(candidate)
        for fragment in group:
            x, y, w, h = fragment["bbox"]
            aspect = max(w, h) / max(min(w, h), 1)
            border_touch = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
            if fragment["area"] >= MIN_COMPONENT_AREA and aspect >= 2.0 and (allow_border or not border_touch):
                candidate_masks.append(fragment["mask"])
    return candidate_masks


def chromatic_fragments_compatible(left, right):
    lx, ly, lw, lh = left["bbox"]
    rx, ry, rw, rh = right["bbox"]
    horizontal_overlap = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    vertical_overlap = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    horizontal_overlap_ratio = horizontal_overlap / max(min(lw, rw), 1)
    vertical_overlap_ratio = vertical_overlap / max(min(lh, rh), 1)
    vertical_gap = max(0, max(ly, ry) - min(ly + lh, ry + rh))
    horizontal_gap = max(0, max(lx, rx) - min(lx + lw, rx + rw))
    max_vertical_gap = max(28.0, 0.28 * max(lh, rh))
    max_horizontal_gap = max(28.0, 0.28 * max(lw, rw))
    left_depth, right_depth = left.get("depth_median_mm"), right.get("depth_median_mm")
    depth_compatible = bool(
        left_depth is not None
        and right_depth is not None
        and left.get("depth_valid_ratio", 0.0) >= 0.18
        and right.get("depth_valid_ratio", 0.0) >= 0.18
        and abs(float(left_depth) - float(right_depth)) <= 100.0
    )
    # A hand-mounted camera may be rolled relative to raw image-up.  Keep the
    # existing vertical-fragment rule and accept its 90-degree counterpart so
    # a shadowed bottle body remains one RGB-D candidate in either image roll.
    vertical_stack = bool(
        horizontal_overlap_ratio >= 0.50 and vertical_gap <= max_vertical_gap
    )
    horizontal_stack = bool(
        vertical_overlap_ratio >= 0.50 and horizontal_gap <= max_horizontal_gap
    )
    return bool((vertical_stack or horizontal_stack) and depth_compatible)


def depth_mode_components(mask, depth, min_area=250, bin_width_mm=50.0):
    """Split touching color regions only when aligned depth has supported modes."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    min_area = scaled_detector_area(min_area, mask_u8.shape)
    if depth is None or np.asarray(depth).shape != mask_u8.shape:
        return [mask_u8]
    depth_arr = np.asarray(depth, dtype=float)
    valid = (mask_u8 > 0) & valid_depth_mask(depth_arr)
    values = depth_arr[valid]
    if values.size < min_area:
        return [mask_u8]
    lower = float(np.floor(np.min(values) / bin_width_mm) * bin_width_mm)
    upper = float(np.ceil(np.max(values) / bin_width_mm) * bin_width_mm + bin_width_mm)
    edges = np.arange(lower, upper + 0.5 * bin_width_mm, bin_width_mm)
    if edges.size < 3:
        return [mask_u8]
    counts, _ = np.histogram(values, bins=edges)
    supported = counts >= max(int(min_area), int(round(values.size * 0.015)))
    groups = []
    start = None
    for index, active in enumerate(supported):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(supported) - 1):
            end = index if active and index == len(supported) - 1 else index - 1
            groups.append((start, end))
            start = None
    if len(groups) <= 1:
        return [mask_u8]

    components = []
    for start, end in groups:
        low = edges[start] - 20.0
        high = edges[end + 1] + 20.0
        band = ((mask_u8 > 0) & valid & (depth_arr >= low) & (depth_arr <= high)).astype(np.uint8)
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(band, connectivity=8)
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
                components.append((labels == label).astype(np.uint8))
    return components or [mask_u8]


def score_component(
    component,
    labels,
    label,
    stats,
    centroids,
    depth,
    image_shape,
    rgb=None,
    source="depth",
    depth_exclusion_mask=None,
):
    img_h, img_w = image_shape[:2]
    x, y, w, h, area = stats[label]
    minimum_area = scaled_detector_area(MIN_COMPONENT_AREA, image_shape)
    minimum_extent = scaled_detector_length(25, image_shape, minimum=10)
    if area < minimum_area or w < minimum_extent or h < minimum_extent:
        return None
    mask = labels == label
    ds = depth_stats(depth, mask.astype(np.uint8))
    if ds["median_mm"] is None or ds["valid_ratio"] < 0.18:
        return None
    contour = largest_contour(mask.astype(np.uint8))
    metrics = contour_quality_metrics(
        mask.astype(np.uint8),
        contour,
        (int(x), int(y), int(w), int(h)),
        image_shape,
        rgb=rgb,
        depth=depth,
        depth_exclusion_mask=depth_exclusion_mask,
    )
    # Pre-lock bottle identity requires a complete image boundary.  Live
    # keyboard/mouse and monitor depth surfaces repeatedly entered from the
    # frame edge; downstream already treats such candidates as unsafe, while
    # their raw depth remains available to the separate obstacle corridors.
    if metrics["border_touch"]:
        return None
    profile = known_profile_agreement(metrics)

    cx, cy = centroids[label]
    area_ratio = float(area / max(img_h * img_w, 1))
    area_score = metrics["mask_area_score"]
    center_offset = abs(cx - img_w / 2.0) / max(img_w / 2.0, 1.0)
    center_score = float(np.clip(1.0 - center_offset, 0.0, 1.0))
    near_score = float(np.clip(1.0 - (ds["median_mm"] / 500.0), 0.0, 1.0))
    fill_ratio = metrics["fill_ratio"]
    fill_score = float(np.clip(1.0 - abs(fill_ratio - 0.60) / 0.60, 0.0, 1.0))
    aspect = metrics["aspect_ratio"]
    aspect_score = metrics["oriented_aspect_score"]
    color_fraction = metrics["bottle_color_fraction"]
    oriented_aspect = float(metrics.get("oriented_aspect_ratio", aspect) or 0.0)
    top_down_shape = bool(
        1.0 <= oriented_aspect < 1.35
        and float(metrics.get("circularity_score") or 0.0) >= 0.82
        and float(metrics.get("solidity_score") or 0.0) >= 0.78
        and metrics.get("depth_foreground_separated") is True
    )
    if oriented_aspect < 1.35 and not top_down_shape:
        return None
    if color_fraction < 0.80 and (
        float(metrics.get("solidity_score", 0.0) or 0.0) < 0.65
        or float(metrics.get("extent_score", 0.0) or 0.0) < 0.45
        or float(metrics.get("contour_quality_score", 0.0) or 0.0) < 0.55
    ):
        return None
    if (
        source == "rgb"
        and metrics.get("depth_foreground_separated") is False
        and float(metrics.get("depth_foreground_separation_mm") or 0.0) <= 8.0
        and float(metrics.get("depth_surface_mad_mm") or 0.0) <= 2.0
        and float(metrics.get("depth_surrounding_mad_mm") or 0.0) <= 2.0
        and not (
            metrics.get("depth_mixed_boundary_foreground_supported") is True
            and color_fraction >= 0.55
            and float(metrics.get("depth_valid_ratio") or 0.0) >= 0.90
            and profile.get("bottle_profile_valid") is True
        )
    ):
        return None
    if w > img_w * 0.90:
        return None
    if h > img_h * 0.92 and color_fraction < 0.35:
        return None
    if w > img_w * 0.55 and h > img_h * 0.55 and color_fraction < 0.35:
        return None
    if metrics["mask_contamination_score"] > 0.65:
        return None
    if profile["bottle_profile_agreement_score"] < 0.18 and color_fraction < 0.20:
        return None
    border_penalty = 0.18 if x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2 else 0.0
    bottom_penalty = 0.12 if y + h > img_h * 0.92 else 0.0

    score = (
        1.3 * area_score
        + 0.8 * center_score
        + 0.8 * near_score
        + 0.8 * metrics["depth_support_score"]
        + 0.8 * metrics["depth_consistency_score"]
        + 0.8 * float(metrics.get("depth_geometry_score") or 0.0)
        + 0.5 * fill_score
        + 1.3 * metrics["RGB_contrast_score"]
        + 1.1 * aspect_score
        + 0.9 * metrics["contour_quality_score"]
        + 0.9 * profile["bottle_profile_agreement_score"]
        - border_penalty
        - bottom_penalty
        - 0.7 * metrics["mask_contamination_score"]
    )
    if source == "rgb":
        score += 3.0
    return {
        "score": float(score),
        "bbox": (int(x), int(y), int(w), int(h)),
        "centroid": (float(cx), float(cy)),
        "depth_stats": ds,
        "fill_ratio": fill_ratio,
        "aspect_ratio": float(aspect),
        "bottle_color_fraction": color_fraction,
        "area_ratio": area_ratio,
        "candidate_evidence_scores": metrics,
        "known_profile_agreement": profile,
        "component": mask.astype(np.uint8),
        "depth_exclusion_mask": depth_exclusion_mask,
        "source": source,
        "top_down_view_candidate": top_down_shape,
    }


def refine_by_depth_band(mask, depth):
    ds = depth_stats(depth, mask)
    if ds["median_mm"] is None:
        return mask
    valid_values = depth[(mask > 0) & valid_depth_mask(depth)]
    mad = float(np.median(np.abs(valid_values - ds["median_mm"]))) if valid_values.size else 0.0
    band = max(35.0, 3.5 * mad, ds["median_mm"] * 0.18)
    refined = mask & (np.abs(depth.astype(float) - ds["median_mm"]) <= band)
    refined = clean_mask(
        refined,
        kernel_size=scaled_detector_kernel(5, mask.shape),
    )
    if np.count_nonzero(refined) < scaled_detector_area(MIN_COMPONENT_AREA, mask.shape):
        return mask.astype(np.uint8)
    return refined.astype(np.uint8)


def body_region_from_axis(mask, axis, trim_low=0.18, trim_high=0.18):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return mask.copy()
    points = np.column_stack([xs, ys]).astype(float)
    projections = points @ axis
    low = float(np.percentile(projections, trim_low * 100.0))
    high = float(np.percentile(projections, (1.0 - trim_high) * 100.0))
    keep = (projections >= low) & (projections <= high)
    body = np.zeros_like(mask, dtype=np.uint8)
    body[ys[keep], xs[keep]] = 1
    return clean_mask(
        body,
        kernel_size=scaled_detector_kernel(5, mask.shape),
    )


def _scored_bottle_components(rgb, depth):
    candidates = []
    rgb_masks = candidate_masks_from_rgb(rgb, depth)
    mask_sources = [("rgb", mask) for mask in rgb_masks]
    mask_sources.extend(("depth", mask) for mask in candidate_masks_from_depth(depth))
    for source, base_mask in mask_sources:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            base_mask.astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8)
            depth_exclusion_mask = None
            if source == "rgb":
                sibling_masks = []
                component_area = int(np.count_nonzero(component_mask))
                for rgb_mask in rgb_masks:
                    rgb_area = int(np.count_nonzero(rgb_mask))
                    overlap = int(np.count_nonzero((component_mask > 0) & (rgb_mask > 0)))
                    if overlap / max(min(component_area, rgb_area), 1) < 0.20:
                        sibling_masks.append(rgb_mask)
                if sibling_masks:
                    depth_exclusion_mask = np.zeros_like(component_mask, dtype=np.uint8)
                    for sibling_mask in sibling_masks:
                        depth_exclusion_mask |= (np.asarray(sibling_mask) > 0).astype(np.uint8)
            if source == "depth":
                supported_rgb_instances = 0
                component_area = int(np.count_nonzero(component_mask))
                for rgb_mask in rgb_masks:
                    rgb_area = int(np.count_nonzero(rgb_mask))
                    overlap = int(np.count_nonzero((component_mask > 0) & (rgb_mask > 0)))
                    if overlap / max(min(component_area, rgb_area), 1) >= 0.55:
                        supported_rgb_instances += 1
                if supported_rgb_instances >= 2:
                    continue
            scored = score_component(
                base_mask,
                labels,
                label,
                stats,
                centroids,
                depth,
                depth.shape,
                rgb=rgb,
                source=source,
                depth_exclusion_mask=depth_exclusion_mask,
            )
            if scored is not None:
                candidates.append(scored)
    return candidates


def _pose_from_scored_component(rgb, depth, best, intrinsics=None):
    raw_mask = best["component"].astype(np.uint8)
    pre_extension_body = refine_by_depth_band(best["component"], depth)
    depth_refinement_extent_recovered = False
    raw_contour = largest_contour(raw_mask)
    refined_contour = largest_contour(pre_extension_body)
    raw_extent_bbox = cv2.boundingRect(raw_contour) if raw_contour is not None else None
    refined_extent_bbox = (
        cv2.boundingRect(refined_contour) if refined_contour is not None else None
    )
    raw_extent_differs = bool(
        raw_extent_bbox is not None
        and refined_extent_bbox is not None
        and (
            raw_extent_bbox != refined_extent_bbox
            or np.count_nonzero(raw_mask) != np.count_nonzero(pre_extension_body)
        )
    )

    # The raw RGB component is useful for explaining reflective/curved depth
    # loss, but it cannot replace the depth-refined mask used for bottle pose,
    # dimensions, tracking, or 3-D lock. Otherwise a connected blue hand,
    # fixture, or display region can silently become authoritative geometry.
    # Depth-supported cap/extent estimation below remains diagnostic-only.
    pre_extension_axis, _pre_extension_angle, _pre_extension_length = pca_axis_from_mask(pre_extension_body)
    body_mask = body_region_from_axis(pre_extension_body, pre_extension_axis)
    estimated_whole_mask, cap_like_mask, cap_like_evidence = depth_supported_achromatic_endcap(
        pre_extension_body,
        rgb,
        depth,
        exclusion_mask=best.get("depth_exclusion_mask"),
        axis_hint=pre_extension_axis,
    )
    # The pre-extension RGB-D body is the authoritative identity and tracking
    # geometry.  A reflective cap-like region may be intermittently measurable,
    # so its union is retained only as explicitly estimated extent evidence.
    contour = largest_contour(pre_extension_body)
    if contour is None:
        return None, "Bottle contour could not be recovered after depth cleanup."
    whole_contours, _ = cv2.findContours(
        estimated_whole_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if cap_like_evidence.get("detected") is True and len(whole_contours) > 1:
        estimated_whole_contour = cv2.convexHull(np.vstack(whole_contours))
    else:
        estimated_whole_contour = (
            max(whole_contours, key=cv2.contourArea)
            if whole_contours
            else contour
        )

    x, y, w, h = cv2.boundingRect(contour)
    minimum_extent = scaled_detector_length(25, depth.shape, minimum=10)
    if w < minimum_extent or h < minimum_extent:
        return None, "Bottle candidate is too small for a reliable pre-grasp plan."

    ds = depth_stats(depth, pre_extension_body)
    if ds["median_mm"] is None:
        return None, "Bottle candidate has no reliable depth."
    rect = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(rect)
    whole_x, whole_y, whole_w, whole_h = cv2.boundingRect(estimated_whole_contour)
    cap_like_evidence = dict(cap_like_evidence or {})
    cap_like_evidence.update({
        "estimated_whole_extent_status": (
            "ESTIMATED_FROM_PRE_EXTENSION_BODY_PLUS_CAP_LIKE_REGION"
            if cap_like_evidence.get("detected") is True
            else "SAME_AS_AUTHORITATIVE_PRE_EXTENSION_BODY_NO_CAP_LIKE_REGION"
        ),
        "estimated_whole_extent_bbox_xywh": [int(whole_x), int(whole_y), int(whole_w), int(whole_h)],
        "estimated_whole_extent_role": "DIAGNOSTIC_EXTENT_ONLY_NOT_TRACKING_GEOMETRY",
        "estimated_whole_extent_source": "PRE_EXTENSION_RGBD_BODY_PLUS_DEPTH_SUPPORTED_CAP_LIKE_MASK",
        "coordinate_frame": "detector_input_pixels",
        "applied_to_tracking_geometry": False,
        "tracking_geometry_source": "pre_extension_rgbd_body",
        "physically_validated": False,
    })
    approximate_dimensions = approximate_bbox_dimensions_mm(
        (int(x), int(y), int(w), int(h)),
        ds["median_mm"],
        intrinsics,
        oriented_bbox=box_points,
    )
    top_down_view_candidate = bool(best.get("top_down_view_candidate"))
    if approximate_dimensions and not (
        approximate_dimensions["broad_bottle_dimensions_valid"]
        or (
            top_down_view_candidate
            and approximate_dimensions.get("broad_top_down_bottle_diameter_valid")
        )
    ):
        return None, "Bottle candidate failed broad RGB-D dimension plausibility."

    axis, angle_deg, axis_length = pca_axis_from_mask(pre_extension_body)
    body_contour = largest_contour(body_mask)
    ys, xs = np.where(pre_extension_body > 0)
    center = (float(np.mean(xs)), float(np.mean(ys)))
    box = box_points.astype(int).tolist()
    fill_ratio = float(np.count_nonzero(pre_extension_body) / max(w * h, 1))
    area_ratio = float(np.count_nonzero(pre_extension_body) / max(depth.shape[0] * depth.shape[1], 1))
    color_fraction = bottle_color_fraction(rgb, pre_extension_body)
    candidate_evidence_scores = contour_quality_metrics(
        pre_extension_body,
        contour,
        (int(x), int(y), int(w), int(h)),
        depth.shape,
        rgb=rgb,
        depth=depth,
        depth_exclusion_mask=best.get("depth_exclusion_mask"),
    )
    elongated_shape_score = float(candidate_evidence_scores.get("oriented_aspect_score") or 0.0)
    top_down_shape_score = float(candidate_evidence_scores.get("circularity_score") or 0.0)
    candidate_evidence_scores.update(
        {
            "contour_score": candidate_evidence_scores.get("contour_quality_score"),
            "shape_score": top_down_shape_score if top_down_view_candidate else elongated_shape_score,
            "profile_score": None,
            "temporal_score": None,
            "temporal_score_status": "NOT_TESTED_SINGLE_FRAME_CANDIDATE",
            "pointcloud_score": None,
            "pointcloud_score_status": "NOT_APPLICABLE_ASYNC_POINTCLOUD_NOT_BOUND_TO_CANDIDATE",
            "clutter_separation_score": candidate_evidence_scores.get("depth_foreground_separation_score"),
            "clutter_separation_status": "ESTIMATED_FROM_ALIGNED_DEPTH_BOUNDARY_RING",
            "depth_refinement_extent_recovered": depth_refinement_extent_recovered,
            "depth_refinement_extent_recovery_status": (
                "RAW_RGB_EXTENT_RETAINED_AS_DIAGNOSTIC_ONLY"
                if raw_extent_differs
                else "NOT_REQUIRED_OR_NOT_ELIGIBLE"
            ),
            "depth_refinement_raw_extent_bbox_xywh": (
                [int(value) for value in raw_extent_bbox]
                if raw_extent_bbox is not None
                else None
            ),
            "depth_refinement_refined_extent_bbox_xywh": (
                [int(value) for value in refined_extent_bbox]
                if refined_extent_bbox is not None
                else None
            ),
            "depth_refinement_raw_extent_role": "DIAGNOSTIC_ONLY_NOT_TRACKING_GEOMETRY",
            "depth_refinement_raw_extent_applied_to_tracking_geometry": False,
        }
    )
    chromatic_fraction = float(candidate_evidence_scores.get("chromatic_color_fraction", 0.0) or 0.0)
    profile_agreement = known_profile_agreement(candidate_evidence_scores)
    candidate_evidence_scores["profile_score"] = profile_agreement["bottle_profile_agreement_score"]
    if not profile_agreement["bottle_profile_valid"] and profile_agreement["bottle_profile_agreement_score"] < 0.20:
        return None, "Bottle candidate failed known-profile agreement: " + ",".join(profile_agreement["bottle_profile_failures"])
    image_center_score = float(np.clip(1.0 - abs(center[0] - depth.shape[1] / 2.0) / max(depth.shape[1] / 2.0, 1.0), 0.0, 1.0))
    size_score = float(np.clip(area_ratio / 0.08, 0.0, 1.0))
    fill_score = float(np.clip(1.0 - abs(fill_ratio - 0.60) / 0.60, 0.0, 1.0))
    rgb_boundary_or_color = max(
        float(candidate_evidence_scores.get("RGB_contrast_score") or 0.0),
        float(candidate_evidence_scores.get("rgb_boundary_score") or 0.0),
    )
    confidence = float(np.clip(
        0.22 * candidate_evidence_scores["depth_support_score"]
        + 0.14 * candidate_evidence_scores["depth_consistency_score"]
        + 0.08 * float(candidate_evidence_scores.get("depth_geometry_score") or 0.0)
        + 0.10 * size_score
        + 0.06 * image_center_score
        + 0.10 * fill_score
        + 0.12 * rgb_boundary_or_color
        + 0.10 * candidate_evidence_scores["contour_quality_score"]
        + 0.08 * profile_agreement["bottle_profile_agreement_score"],
        0.0,
        1.0,
    ))
    # A single RGB-D frame is a candidate.  Temporal target-lock stability is
    # what may later verify/lock an identity.
    detection_status = "CANDIDATE"
    detection_rejection_reasons = []
    if confidence < 0.45:
        detection_status = "LOW_CONFIDENCE"
        detection_rejection_reasons.append("final_candidate_confidence_below_verified_threshold")
    if ds["valid_ratio"] < 0.30:
        detection_status = "DEGRADED_DEPTH"
        detection_rejection_reasons.append("depth_support_ratio_low")
    if not profile_agreement["bottle_profile_valid"]:
        detection_status = "LOW_CONFIDENCE"
        detection_rejection_reasons.extend(profile_agreement["bottle_profile_failures"])
    raw_area = float(np.count_nonzero(raw_mask))
    cleaned_area = float(np.count_nonzero(pre_extension_body))
    body_area = float(np.count_nonzero(body_mask))
    working_minimum_area = scaled_detector_area(MIN_COMPONENT_AREA, raw_mask.shape)
    raw_mask_confidence = float(np.clip(raw_area / max(working_minimum_area * 3.0, 1.0), 0.0, 1.0))
    cleaned_mask_confidence = float(np.clip(cleaned_area / max(raw_area, 1.0), 0.0, 1.0))
    body_mask_confidence = float(np.clip(body_area / max(cleaned_area, 1.0), 0.0, 1.0))
    segmentation_rejection_reasons = []
    if candidate_evidence_scores["mask_contamination_score"] > 0.35:
        segmentation_rejection_reasons.append("possible_background_or_hand_contamination")
    if cleaned_mask_confidence < 0.45:
        segmentation_rejection_reasons.append("cleaned_mask_lost_too_much_area")
    if body_mask_confidence < 0.35:
        segmentation_rejection_reasons.append("body_mask_too_small")
    candidate_score_breakdown = {
        "selected_source": best.get("source", "unknown"),
        "component_score": best.get("score"),
        "component_score_formula": (
            "multi-cue: area, center, depth support/consistency, RGB contrast, "
            "oriented aspect, contour quality, known-profile agreement, contamination penalty"
        ),
        "detection_confidence_formula": (
            "0.22*depth_support + 0.14*depth_consistency + 0.08*depth_geometry + "
            "0.10*size + 0.06*center + 0.10*fill + 0.12*max(RGB_boundary,color) + "
            "0.10*contour_quality + 0.08*known_profile"
        ),
        "area_ratio": area_ratio,
        "size_score": size_score,
        "image_center_score": image_center_score,
        "fill_ratio": fill_ratio,
        "fill_score": fill_score,
        "aspect_ratio": float(max(w, h) / max(min(w, h), 1)),
        "bottle_color_fraction": color_fraction,
        "chromatic_color_fraction": chromatic_fraction,
        "color_evidence_mode": "known_blue_or_generic_chromatic_depth_fused",
        "depth_valid_ratio": ds["valid_ratio"],
        "depth_median_mm": ds["median_mm"],
        "depth_geometry_evidence": {
            key: candidate_evidence_scores.get(key)
            for key in (
                "depth_geometry_status",
                "depth_surface_coherence_score",
                "depth_surface_mad_mm",
                "depth_surface_interdecile_range_mm",
                "depth_foreground_separation_mm",
                "depth_foreground_separation_score",
                "depth_foreground_separated",
                "depth_far_background_reference_mm",
                "depth_far_background_separation_mm",
                "depth_far_background_support_fraction",
                "depth_far_background_support_pixel_count",
                "depth_far_background_min_required_pixel_count",
                "depth_far_background_sector_support_counts",
                "depth_far_background_supported_sector_count",
                "depth_far_background_opposite_sector_pair_count",
                "depth_far_background_distributed",
                "depth_mixed_boundary_foreground_supported",
                "depth_surrounding_valid_ratio",
                "depth_surrounding_mad_mm",
                "depth_geometry_score",
            )
        },
        "rgb_source_bonus_applied": best.get("source") == "rgb",
        "candidate_evidence_scores": candidate_evidence_scores,
        "cap_like_end_region": cap_like_evidence,
        "known_profile_agreement": profile_agreement,
        "final_candidate_confidence": confidence,
        "detection_status": detection_status,
        "detection_rejection_reasons": detection_rejection_reasons,
    }

    return {
        "detected": True,
        "mask": pre_extension_body.astype(np.uint8),
        "raw_mask": raw_mask,
        "cleaned_mask": pre_extension_body.astype(np.uint8),
        "body_mask": body_mask.astype(np.uint8),
        "cap_like_mask": cap_like_mask.astype(np.uint8),
        "contour": contour,
        "body_contour": body_contour,
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "oriented_bbox": box,
        "center_px": [float(center[0]), float(center[1])],
        "axis_vector": [float(axis[0]), float(axis[1])],
        "axis_angle_degrees": angle_deg,
        "axis_length_px": axis_length,
        "top_down_view_candidate": top_down_view_candidate,
        "cap_view_candidate": bool(
            top_down_view_candidate
            and approximate_dimensions
            and float(approximate_dimensions.get("major_extent_mm") or 0.0) <= 55.0
        ),
        "cap_like_end_region": cap_like_evidence,
        "depth_median_mm": ds["median_mm"],
        "depth_mean_mm": ds["mean_mm"],
        "depth_std_mm": ds["std_mm"],
        "depth_confidence": ds["valid_ratio"],
        "depth_geometry_evidence": candidate_score_breakdown["depth_geometry_evidence"],
        "approximate_bbox_dimensions_mm": approximate_dimensions,
        "fill_ratio": fill_ratio,
        "bottle_color_fraction": color_fraction,
        "chromatic_color_fraction": chromatic_fraction,
        "color_evidence_mode": "known_blue_or_generic_chromatic_depth_fused",
        "area_ratio": area_ratio,
        "detection_confidence": confidence,
        "bottle_detection_status": detection_status,
        "bottle_detection_confidence": confidence,
        "bottle_profile_agreement_score": profile_agreement["bottle_profile_agreement_score"],
        "bottle_profile_failures": profile_agreement["bottle_profile_failures"],
        "bottle_profile_valid": profile_agreement["bottle_profile_valid"],
        "known_bottle_profile": profile_agreement,
        "segmentation_source": best.get("source", "unknown"),
        "segmentation_source_hierarchy": [
            "fast_opencv_aligned_depth" if best.get("source") == "depth" else "fast_opencv_rgb_depth",
            "temporally_propagated_locked_mask",
            "optional_sam2_reacquisition",
            "failure",
        ],
        "raw_mask_confidence": raw_mask_confidence,
        "cleaned_mask_confidence": cleaned_mask_confidence,
        "body_mask_confidence": body_mask_confidence,
        "depth_support_ratio": ds["valid_ratio"],
        "contour_quality": candidate_evidence_scores["contour_quality_score"],
        "mask_temporal_iou": None,
        "mask_contamination_score": candidate_evidence_scores["mask_contamination_score"],
        "segmentation_rejection_reasons": segmentation_rejection_reasons,
        "candidate_evidence_scores": candidate_evidence_scores,
        "candidate_score_breakdown": candidate_score_breakdown,
    }, None


def _bbox_iou_xywh(left, right):
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(lw * lh + rw * rh - intersection, 1.0)
    return intersection / union


def _bbox_overlap_over_smaller_xywh(left, right):
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / max(min(lw * lh, rw * rh), 1.0)


def _same_physical_candidate(left, right, dedup_iou):
    left_depth = left.get("depth_median_mm")
    right_depth = right.get("depth_median_mm")
    depth_close = (
        left_depth is None
        or right_depth is None
        or abs(float(left_depth) - float(right_depth)) <= 100.0
    )
    return bool(
        depth_close
        and (
            _bbox_iou_xywh(left["bbox_xywh"], right["bbox_xywh"]) >= float(dedup_iou)
            or _bbox_overlap_over_smaller_xywh(left["bbox_xywh"], right["bbox_xywh"]) >= 0.72
        )
    )


def detect_bottle_poses(rgb, depth, max_candidates=8, dedup_iou=0.45, intrinsics=None):
    """Return distinct bottle poses from the existing RGB-D component detector.

    Target IDs are frame-local and ordered left-to-right. Target lock remains the
    persistent identity mechanism once an operator begins an approach.
    """
    candidates = sorted(_scored_bottle_components(rgb, depth), key=lambda item: item["score"], reverse=True)
    if not candidates:
        return [], "No stable RGB-D foreground component found for the known bottle."

    built_poses = []
    rejection_reasons = []
    for candidate in candidates:
        pose, error = _pose_from_scored_component(rgb, depth, candidate, intrinsics=intrinsics)
        if pose is None:
            if error:
                rejection_reasons.append(error)
            continue
        built_poses.append(pose)

    rgb_supported = any(pose.get("segmentation_source") == "rgb" for pose in built_poses)
    if rgb_supported:
        built_poses = [
            pose
            for pose in built_poses
            if pose.get("segmentation_source") == "rgb"
            or _strong_standalone_depth_bottle_pose(pose)
        ]
    poses = []
    for pose in built_poses:
        if any(_same_physical_candidate(pose, existing, dedup_iou) for existing in poses):
            continue
        poses.append(pose)
        if len(poses) >= max(1, int(max_candidates)):
            break

    if not poses:
        reason = rejection_reasons[0] if rejection_reasons else "Bottle candidates failed pose validation."
        return [], reason

    horizontal_order = {
        id(pose): index + 1
        for index, pose in enumerate(sorted(poses, key=lambda item: float(item["center_px"][0])))
    }
    for pose in poses:
        pose["target_id"] = f"bottle_{horizontal_order[id(pose)]:02d}"
        pose["target_id_scope"] = "FRAME_LOCAL_UNTIL_TARGET_LOCK"
        pose["detected_bottle_count"] = len(poses)
    return poses, None


def detect_bottle_pose(rgb, depth, intrinsics=None):
    """Compatibility wrapper returning the strongest distinct RGB-D bottle."""
    poses, error = detect_bottle_poses(rgb, depth, max_candidates=8, intrinsics=intrinsics)
    return (poses[0], None) if poses else (None, error)


def _strong_standalone_depth_bottle_pose(pose):
    """Keep a non-blue depth instance only when shape/depth evidence is strong.

    This lets a dark or clear bottle coexist with an RGB-supported bottle while
    avoiding the previous unsafe rule that treated every depth rectangle as a
    bottle.  It is still geometric evidence, not semantic verification.
    """
    metrics = pose.get("candidate_evidence_scores") or {}
    dimensions = pose.get("approximate_bbox_dimensions_mm") or {}
    aspect = float(metrics.get("oriented_aspect_ratio") or 0.0)
    return bool(
        2.40 <= aspect <= 6.50
        and float(metrics.get("depth_valid_ratio") or 0.0) >= 0.45
        and float(metrics.get("surface_coherence_score") or metrics.get("depth_surface_coherence_score") or 0.0) >= 0.75
        and float(metrics.get("solidity_score") or 0.0) >= 0.65
        and float(metrics.get("contour_quality_score") or 0.0) >= 0.55
        and float(metrics.get("mask_contamination_score") or 0.0) <= 0.35
        and float(pose.get("bottle_profile_agreement_score") or 0.0) >= 0.35
        and (not dimensions or bool(dimensions.get("broad_bottle_dimensions_valid")))
    )


def approximate_bbox_dimensions_mm(bbox_xywh, depth_median_mm, intrinsics, oriented_bbox=None):
    if not intrinsics:
        return None
    try:
        fx = float(intrinsics.get("fx"))
        fy = float(intrinsics.get("fy"))
        depth_mm = float(depth_median_mm)
        _x, _y, width_px, height_px = [float(value) for value in bbox_xywh]
    except (AttributeError, TypeError, ValueError):
        return None
    if fx <= 0.0 or fy <= 0.0 or depth_mm <= 0.0:
        return None
    width_mm = width_px * depth_mm / fx
    height_mm = height_px * depth_mm / fy
    dimension_source = "AXIS_ALIGNED_BBOX"
    if oriented_bbox is not None:
        try:
            points = np.asarray(oriented_bbox, dtype=float).reshape(-1, 2)
            if points.shape[0] == 4:
                edge_vectors = np.roll(points, -1, axis=0) - points
                edge_lengths_mm = np.sqrt(
                    np.square(edge_vectors[:, 0] * depth_mm / fx)
                    + np.square(edge_vectors[:, 1] * depth_mm / fy)
                )
                unique_edges = [float(edge_lengths_mm[0]), float(edge_lengths_mm[1])]
                if min(unique_edges) > 0.0:
                    width_mm, height_mm = unique_edges
                    dimension_source = "ORIENTED_MIN_AREA_RECT"
        except (TypeError, ValueError):
            pass
    minor_mm = min(width_mm, height_mm)
    major_mm = max(width_mm, height_mm)
    return {
        "status": "ESTIMATED_FROM_ALIGNED_DEPTH_AND_CAMERA_INTRINSICS",
        "dimension_source": dimension_source,
        "width_extent_mm": round(width_mm, 2),
        "height_extent_mm": round(height_mm, 2),
        "minor_extent_mm": round(minor_mm, 2),
        "major_extent_mm": round(major_mm, 2),
        "broad_bottle_dimensions_valid": bool(35.0 <= minor_mm <= 125.0 and 110.0 <= major_mm <= 360.0),
        "broad_top_down_bottle_diameter_valid": bool(35.0 <= minor_mm <= 125.0 and 35.0 <= major_mm <= 125.0),
        "broad_minor_extent_range_mm": [35.0, 125.0],
        "broad_major_extent_range_mm": [110.0, 360.0],
        "not_a_physical_caliper_measurement": True,
    }

"""Pure geometry helpers for approximate hand-mounted camera guidance.

The bottle grasp band remains object-relative after lock.  These helpers
project the configured camera-to-hand contact-point offset into the current
camera image so initial planning can choose a reachable safe bottle band and
operator movement can then be expressed from the hand reach point to that band.
"""

from __future__ import annotations

from math import isfinite


DEFAULT_CAMERA_TO_HAND_OFFSET_MM = {"x": -70.0, "y": 80.0, "z": 170.0}
DEFAULT_CAMERA_TO_HAND_UNCERTAINTY_MM = {"x": 30.0, "y": 10.0, "z": 10.0}
DEFAULT_DESIRED_CONTACT_GAP_MM = 10.0
HAND_OFFSET_LIMITS_MM = {"x": (-300.0, 300.0), "y": (-300.0, 300.0), "z": (0.0, 600.0)}


def _finite_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def normalized_offset_mm(value, default=None):
    """Return a finite x/y/z mapping, falling back per component."""
    source = value if isinstance(value, dict) else {}
    fallback = default if isinstance(default, dict) else DEFAULT_CAMERA_TO_HAND_OFFSET_MM
    return {
        axis: _finite_float(source.get(axis), _finite_float(fallback.get(axis), 0.0))
        for axis in ("x", "y", "z")
    }


def validate_hand_offset_update(value, desired_contact_gap_mm=None):
    """Validate dashboard/runtime tuning without silently accepting bad input."""
    if not isinstance(value, dict):
        return False, None, "camera_to_hand_offset_mm must be an x/y/z object"
    offset = {}
    for axis, limits in HAND_OFFSET_LIMITS_MM.items():
        number = _finite_float(value.get(axis))
        if number is None:
            return False, None, f"camera_to_hand_offset_mm.{axis} must be finite"
        if not limits[0] <= number <= limits[1]:
            return False, None, f"camera_to_hand_offset_mm.{axis} must be within {limits[0]:g}..{limits[1]:g} mm"
        offset[axis] = number
    gap = _finite_float(desired_contact_gap_mm, DEFAULT_DESIRED_CONTACT_GAP_MM)
    if not 0.0 <= gap <= 30.0:
        return False, None, "desired_contact_gap_mm must be within 0..30 mm"
    return True, {"camera_to_hand_offset_mm": offset, "desired_contact_gap_mm": gap}, None


def project_hand_reach_point(
    intrinsics,
    target_depth_mm,
    camera_to_hand_offset_mm,
    *,
    enabled=True,
    image_shape=None,
):
    """Project the fixed camera-to-finger contact point into the image.

    The configured x/y/z values describe one rigid point in the camera frame,
    so its image position uses the contact point's own z coordinate.  Bottle
    depth is returned separately and is used by approach-error conversion, not
    to make the mounted finger marker slide as the bottle approaches.  The
    returned point remains an uncalibrated estimate, not a detected finger.
    """
    intrinsics = intrinsics if isinstance(intrinsics, dict) else {}
    depth = _finite_float(target_depth_mm)
    fx = _finite_float(intrinsics.get("fx"))
    fy = _finite_float(intrinsics.get("fy"))
    cx = _finite_float(intrinsics.get("cx"))
    cy = _finite_float(intrinsics.get("cy"))
    if cx is None and image_shape is not None:
        cx = float(image_shape[1]) * 0.5
    if cy is None and image_shape is not None:
        cy = float(image_shape[0]) * 0.5
    offset = normalized_offset_mm(camera_to_hand_offset_mm)
    if not enabled:
        offset = {"x": 0.0, "y": 0.0, "z": 0.0}
    if depth is None or depth <= 0.0:
        return {
            "valid": False,
            "point_px": None,
            "offset_px": None,
            "reason": "invalid_target_depth",
            "offset_mm": offset,
        }
    if not fx or not fy or cx is None or cy is None:
        return {
            "valid": False,
            "point_px": None,
            "offset_px": None,
            "reason": "missing_intrinsics",
            "offset_mm": offset,
        }
    lateral_offset_present = bool(
        abs(float(offset["x"])) > 1e-9 or abs(float(offset["y"])) > 1e-9
    )
    projection_depth = _finite_float(offset.get("z"))
    if enabled and lateral_offset_present and (
        projection_depth is None or projection_depth <= 0.0
    ):
        return {
            "valid": False,
            "point_px": None,
            "offset_px": None,
            "visible": False,
            "reason": "invalid_hand_contact_point_depth",
            "offset_mm": offset,
            "projection_depth_mm": projection_depth,
            "target_depth_mm": depth,
        }
    if not enabled or not lateral_offset_present:
        offset_px = [0.0, 0.0]
    else:
        offset_px = [
            fx * offset["x"] / projection_depth,
            fy * offset["y"] / projection_depth,
        ]
    point = [cx + offset_px[0], cy + offset_px[1]]
    visible = True
    if image_shape is not None:
        height, width = image_shape[:2]
        visible = bool(0.0 <= point[0] < float(width) and 0.0 <= point[1] < float(height))
    return {
        "valid": True,
        "point_px": point,
        "offset_px": offset_px,
        "visible": visible,
        "reason": "projected_fixed_camera_to_hand_contact_point",
        "offset_mm": offset,
        "projection_depth_mm": projection_depth if enabled else None,
        "target_depth_mm": depth,
    }


def hand_to_locked_band_error(
    locked_band_center_px,
    hand_reach_point_px,
    target_depth_mm,
    intrinsics,
    camera_to_hand_offset_mm,
    desired_contact_gap_mm=DEFAULT_DESIRED_CONTACT_GAP_MM,
):
    """Return camera-frame movement error from projected reach to bottle band."""
    band = locked_band_center_px or []
    reach = hand_reach_point_px or []
    depth = _finite_float(target_depth_mm)
    fx = _finite_float((intrinsics or {}).get("fx"))
    fy = _finite_float((intrinsics or {}).get("fy"))
    gap = _finite_float(desired_contact_gap_mm, DEFAULT_DESIRED_CONTACT_GAP_MM)
    offset = normalized_offset_mm(camera_to_hand_offset_mm)
    if len(band) < 2 or len(reach) < 2 or depth is None or depth <= 0 or not fx or not fy:
        return {
            "valid": False,
            "error_px": None,
            "error_mm": None,
            "hand_to_bottle_z_error_mm": None,
            "reason": "missing_current_depth_intrinsics_or_band",
        }
    error_px = [float(band[0]) - float(reach[0]), float(band[1]) - float(reach[1])]
    error_mm = [error_px[0] * depth / fx, error_px[1] * depth / fy]
    z_error = depth - offset["z"] - gap
    return {
        "valid": True,
        "error_px": error_px,
        "error_mm": error_mm,
        "hand_to_bottle_z_error_mm": z_error,
        # Existing operator sign convention: negative means move closer.
        "recommended_move_z_mm": -z_error,
        "reason": "locked_band_minus_projected_hand_reach",
    }


__all__ = [
    "DEFAULT_CAMERA_TO_HAND_OFFSET_MM",
    "DEFAULT_CAMERA_TO_HAND_UNCERTAINTY_MM",
    "DEFAULT_DESIRED_CONTACT_GAP_MM",
    "hand_to_locked_band_error",
    "normalized_offset_mm",
    "project_hand_reach_point",
    "validate_hand_offset_update",
]

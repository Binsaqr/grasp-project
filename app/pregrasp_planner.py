import json
import math
from pathlib import Path

import numpy as np

from depth_utils import depth_patch_stats, pixel_to_3d, valid_depth_mask
from epsilon_quality import compute_epsilon_quality
from gripper_workspace import evaluate_workspace_reachability


GRIP_WIDTH_MM_RANGE = (30.0, 130.0)
GRIP_WIDTH_PX_RANGE = (35.0, 320.0)
GRIP_WIDTH_LIMITS_SOURCE = "default"
BODY_FRACTION_MIN = 0.22
BODY_FRACTION_MAX = 0.78
BODY_FRACTION_EQUIVALENCE_TOLERANCE = 0.015
CENTER_GRASP_BODY_FRACTION = 0.50
ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "known_bottle_grasp_profile.json"
DEFAULT_KNOWN_BOTTLE_PROFILE = {
    "object_name": "known_water_bottle",
    "expected_diameter_mm": {"min": 45, "max": 85},
    "expected_height_mm": {"min": 140, "max": 280},
    "preferred_grasp_body_fractions": [0.45, 0.52, 0.60],
    "fallback_grasp_body_fractions": [0.32, 0.38, 0.68, 0.74],
    "avoid_top_fraction": 0.20,
    "avoid_bottom_fraction": 0.15,
    "preferred_patch_height_mm": 50,
    "preferred_close_distance_mm": 200,
    "min_close_distance_mm": 150,
    "max_close_distance_mm": 260,
    "assumed_friction_coefficient": 0.5,
    "notes": "Known-object profile fallback; hand-authored, not a learned full grasp model.",
}
CONTACT_CANDIDATE_FRACTIONS = (0.45, 0.52, 0.60, 0.32, 0.38, 0.68, 0.74)
FRICTION_COEFFICIENT = 0.5


def active_object_profile_override():
    """Return planner-compatible active object priors when explicitly selected."""
    try:
        from grasp_project.objects.object_registry import active_grasp_profile
    except Exception:
        return None
    try:
        profile = active_grasp_profile()
    except Exception:
        return None
    object_name = str(profile.get("object_name") or "").strip()
    if object_name in {"", "blue_bottle", "known_water_bottle"}:
        return None
    return profile


def load_known_bottle_profile(path=PROFILE_PATH):
    try:
        with open(path, "r") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError):
        profile = dict(DEFAULT_KNOWN_BOTTLE_PROFILE)
        profile["profile_load_warning"] = f"Could not load {path}; using built-in defaults."
    profile.setdefault("object_name", "known_water_bottle")
    profile.setdefault("preferred_grasp_body_fractions", list(CONTACT_CANDIDATE_FRACTIONS[:3]))
    profile.setdefault("fallback_grasp_body_fractions", list(CONTACT_CANDIDATE_FRACTIONS[3:]))
    profile.setdefault("avoid_top_fraction", 0.20)
    profile.setdefault("avoid_bottom_fraction", 0.15)
    profile.setdefault("preferred_patch_height_mm", 50)
    profile.setdefault("assumed_friction_coefficient", FRICTION_COEFFICIENT)
    active_profile = active_object_profile_override()
    if active_profile:
        legacy_path = str(path)
        profile.update({key: value for key, value in active_profile.items() if value is not None})
        profile["legacy_profile_path"] = legacy_path
        profile["active_object_profile_applied"] = True
    if not profile.get("active_object_profile_applied"):
        profile["profile_path"] = str(path)
    profile["profile_type"] = "object_profile_active_target" if profile.get("active_object_profile_applied") else "known_object_hand_authored_grasp_profile"
    return profile


def profile_candidate_fractions(profile):
    fractions = []
    for key in ("preferred_grasp_body_fractions", "fallback_grasp_body_fractions"):
        for value in profile.get(key, []):
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= fraction <= 1.0 and fraction not in fractions:
                fractions.append(fraction)
    return tuple(fractions or CONTACT_CANDIDATE_FRACTIONS)


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def perpendicular(vector):
    vector = unit(vector)
    return np.array([-vector[1], vector[0]], dtype=float)


def closest_mask_point(mask, point):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.asarray(point, dtype=float)
    pts = np.column_stack([xs, ys]).astype(float)
    distances = np.linalg.norm(pts - np.asarray(point, dtype=float), axis=1)
    return pts[int(np.argmin(distances))]


def point_inside(mask, point):
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return bool(mask[y, x] > 0)


def boundary_along_direction(mask, start, direction, max_steps=None):
    h, w = mask.shape[:2]
    if max_steps is None:
        max_steps = int(math.hypot(w, h))
    direction = unit(direction)
    current = np.asarray(start, dtype=float)
    last_inside = None
    for step in range(max_steps):
        p = current + direction * step
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if mask[y, x] > 0:
            last_inside = np.array([float(x), float(y)])
        elif last_inside is not None:
            break
    return last_inside


def boundary_contacts_from_axis_band(mask, center, axis, direction, axis_length):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None, None
    pts = np.column_stack([xs, ys]).astype(float)
    center = np.asarray(center, dtype=float)
    axis = unit(axis)
    direction = unit(direction)
    band_half_width = max(8.0, float(axis_length) * 0.035)
    axial_offsets = (pts - center) @ axis
    band = pts[np.abs(axial_offsets) <= band_half_width]
    if band.shape[0] < 8:
        band_half_width = max(16.0, float(axis_length) * 0.07)
        band = pts[np.abs(axial_offsets) <= band_half_width]
    if band.shape[0] < 2:
        return None, None
    lateral = (band - center) @ direction
    axial = (band - center) @ axis
    low = float(np.percentile(lateral, 5.0))
    high = float(np.percentile(lateral, 95.0))
    negative_indexes = np.where(lateral <= low)[0]
    positive_indexes = np.where(lateral >= high)[0]
    if negative_indexes.size == 0 or positive_indexes.size == 0:
        return band[int(np.argmin(lateral))], band[int(np.argmax(lateral))]
    negative = int(negative_indexes[np.argmin(np.abs(axial[negative_indexes]))])
    positive = int(positive_indexes[np.argmin(np.abs(axial[positive_indexes]))])
    return band[negative], band[positive]


def projection_fraction(point, pose):
    axis = np.asarray(pose["axis_vector"], dtype=float)
    mask = pose["cleaned_mask"]
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.5
    projections = np.column_stack([xs, ys]).astype(float) @ axis
    p = float(np.asarray(point, dtype=float) @ axis)
    return float((p - np.min(projections)) / max(np.max(projections) - np.min(projections), 1e-6))


def center_for_body_fraction(pose, fraction):
    mask = pose["cleaned_mask"]
    axis = unit(np.asarray(pose["axis_vector"], dtype=float))
    center = np.asarray(pose["center_px"], dtype=float)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return center
    pts = np.column_stack([xs, ys]).astype(float)
    projections = pts @ axis
    target_projection = float(np.min(projections) + np.clip(fraction, 0.0, 1.0) * (np.max(projections) - np.min(projections)))
    center_projection = float(center @ axis)
    target = center + axis * (target_projection - center_projection)
    if not point_inside(pose.get("body_mask", mask), target):
        return closest_mask_point(pose.get("body_mask", mask), target)
    return target


def estimate_width_mm(left, right, depth, intrinsics):
    left_stats = depth_patch_stats(depth, left)
    right_stats = depth_patch_stats(depth, right)
    left_3d = pixel_to_3d(left, left_stats["depth_mm"], intrinsics)
    right_3d = pixel_to_3d(right, right_stats["depth_mm"], intrinsics)
    width_mm = None
    if left_3d is not None and right_3d is not None:
        width_mm = float(np.linalg.norm(np.asarray(right_3d) - np.asarray(left_3d)) * 1000.0)
    elif intrinsics is not None:
        z = depth_patch_stats(depth, (np.asarray(left) + np.asarray(right)) / 2.0)["depth_mm"]
        if z is not None:
            width_mm = float(np.linalg.norm(np.asarray(right) - np.asarray(left)) * z / float(intrinsics["fx"]))
    return width_mm, left_stats, right_stats, left_3d, right_3d


def score_width(width_px, width_mm):
    if width_mm is not None:
        low, high = GRIP_WIDTH_MM_RANGE
        value = float(width_mm)
    else:
        low, high = GRIP_WIDTH_PX_RANGE
        value = float(width_px)
    center = (low + high) / 2.0
    half = (high - low) / 2.0
    if value < low or value > high:
        return 0.0, False
    return float(max(0.0, 100.0 * (1.0 - abs(value - center) / max(half, 1e-6)))), True


def score_from_abs_error(value, good_limit, fail_limit):
    value = abs(float(value))
    good_limit = max(float(good_limit), 0.0)
    fail_limit = max(float(fail_limit), good_limit + 1e-6)
    if value <= good_limit:
        return 100.0
    if value >= fail_limit:
        return 0.0
    return float(np.clip(100.0 * (1.0 - (value - good_limit) / (fail_limit - good_limit)), 0.0, 100.0))


def evaluate_candidate(pose, depth, intrinsics, center, axis, grasp_direction, fraction, profile=None, candidate_index=None):
    profile = profile or load_known_bottle_profile()
    mask = pose["body_mask"] if pose.get("body_mask") is not None else pose["cleaned_mask"]
    neg, pos = boundary_contacts_from_axis_band(
        mask,
        center,
        axis,
        grasp_direction,
        pose.get("axis_length_px", 0.0),
    )
    if neg is None or pos is None:
        neg = boundary_along_direction(mask, center, -grasp_direction)
        pos = boundary_along_direction(mask, center, grasp_direction)
    if neg is None or pos is None:
        return None, {"body_fraction": float(fraction), "reasons": ["Could not find left/right bottle boundary contacts."]}

    contacts = sorted([neg, pos], key=lambda p: p[0])
    left = contacts[0]
    right = contacts[1]
    grip_width_px = float(np.linalg.norm(right - left))
    grasp_center = (left + right) / 2.0
    body_fraction = projection_fraction(grasp_center, pose)
    avoid_top = float(profile.get("avoid_top_fraction", 0.20) or 0.20)
    avoid_bottom = float(profile.get("avoid_bottom_fraction", 0.15) or 0.15)
    on_body = BODY_FRACTION_MIN <= body_fraction <= BODY_FRACTION_MAX
    in_profile_body_band = avoid_top <= body_fraction <= (1.0 - avoid_bottom)
    width_mm, left_stats, right_stats, left_3d, right_3d = estimate_width_mm(left, right, depth, intrinsics)
    width_score, grip_width_in_range = score_width(grip_width_px, width_mm)
    workspace = evaluate_workspace_reachability(left, right, depth.shape)
    workspace_score = float(workspace["workspace_score"])
    reachability_score = float(width_score)

    left_normal = unit(grasp_center - left)
    right_normal = unit(grasp_center - right)
    contacts_for_epsilon = [
        {"point": left.astype(float).tolist(), "inward_normal": left_normal.astype(float).tolist()},
        {"point": right.astype(float).tolist(), "inward_normal": right_normal.astype(float).tolist()},
    ]
    epsilon_result = compute_epsilon_quality(
        contacts_for_epsilon,
        object_centroid=pose["center_px"],
        object_radius=max(grip_width_px / 2.0, 1.0),
        mu=FRICTION_COEFFICIENT,
        samples_per_contact=8,
    )
    epsilon_score = 0.0
    if epsilon_result.get("force_closure"):
        # Pre-grasp candidate-ranking scale: full score requires epsilon >= 0.20,
        # deliberately MORE conservative than the operator-facing display scale
        # (0.05, aligned with the GOOD/OK/WEAK labels in realtime_runner). This
        # is an internal ranking signal at the planning stage, not the operator
        # epsilon score; do not "unify" it to 0.05 -- that would weaken pre-grasp
        # candidate selection. See docs/final_epsilon_consistency_report.md.
        epsilon_score = float(np.clip((float(epsilon_result.get("epsilon", 0.0) or 0.0) / 0.20) * 100.0, 0.0, 100.0))

    contact_quality = float(np.clip(
        0.50 * left_stats["valid_ratio"] * 100.0
        + 0.50 * right_stats["valid_ratio"] * 100.0,
        0.0,
        100.0,
    ))
    balance_px = float(np.linalg.norm(grasp_center - np.asarray(pose["center_px"], dtype=float)))
    balance_score = float(np.clip(100.0 * (1.0 - balance_px / max(grip_width_px, 1.0)), 0.0, 100.0))
    body_region_score = score_from_abs_error(float(body_fraction) - 0.50, 0.12, 0.32)
    slip_score = float(np.clip(0.45 * body_region_score + 0.35 * contact_quality + 0.20 * width_score, 0.0, 100.0))
    rotation_score = float(np.clip(0.55 * balance_score + 0.45 * epsilon_score, 0.0, 100.0))
    profile_preferred = profile.get("preferred_grasp_body_fractions", [])
    preference_error = min([abs(float(body_fraction) - float(f)) for f in profile_preferred] or [0.0])
    profile_preference_score = score_from_abs_error(preference_error, 0.035, 0.22)
    candidate_score = float(np.clip(
        0.18 * body_region_score
        + 0.14 * balance_score
        + 0.14 * width_score
        + 0.14 * workspace_score
        + 0.14 * contact_quality
        + 0.10 * slip_score
        + 0.10 * rotation_score
        + 0.04 * epsilon_score
        + 0.02 * profile_preference_score,
        0.0,
        100.0,
    ))

    rejected_reasons = []
    if not on_body:
        rejected_reasons.append("Candidate contacts are outside the valid bottle body band.")
    if not in_profile_body_band:
        rejected_reasons.append("Candidate overlaps top/bottom avoidance region from known bottle profile.")
    if not grip_width_in_range:
        rejected_reasons.append("Candidate grip width is outside allowed limits.")
    if contact_quality < 45.0:
        rejected_reasons.append("Candidate contact depth confidence is too low.")
    if not epsilon_result.get("force_closure", False):
        rejected_reasons.append("Candidate epsilon force closure is false.")

    candidate_id = f"known_water_bottle_fraction_{float(fraction):.3f}"
    if candidate_index is not None:
        candidate_id = f"cand_{int(candidate_index):02d}_fraction_{float(fraction):.3f}"
    candidate = {
        "candidate_id": candidate_id,
        "candidate_source": "known_bottle_grasp_profile",
        "body_fraction": float(body_fraction),
        "requested_body_fraction": float(fraction),
        "patch_height_mm": float(profile.get("preferred_patch_height_mm", 50.0) or 50.0),
        "grip_width_mm": width_mm,
        "depth_confidence": contact_quality,
        "candidate_score": candidate_score,
        "ranking_score": candidate_score,
        "body_region_score": body_region_score,
        "profile_preference_score": profile_preference_score,
        "profile_body_band_valid": bool(in_profile_body_band),
        "left_right_balance_score": balance_score,
        "grip_width_feasibility_score": width_score,
        "workspace_reachability_score": workspace_score,
        "positioning_guidance_score": workspace_score,
        "workspace_guidance_only": True,
        "depth_confidence_score": contact_quality,
        "slip_score": slip_score,
        "slip_risk": float(np.clip(100.0 - slip_score, 0.0, 100.0)),
        "rotation_score": rotation_score,
        "rotation_risk": float(np.clip(100.0 - rotation_score, 0.0, 100.0)),
        "epsilon_value": float(epsilon_result.get("epsilon", 0.0) or 0.0),
        "epsilon_score": epsilon_score,
        "force_closure": bool(epsilon_result.get("force_closure", False)),
        "contacts_on_body": bool(on_body),
        "left_contact_px": [int(round(left[0])), int(round(left[1]))],
        "right_contact_px": [int(round(right[0])), int(round(right[1]))],
        "grasp_center_px": [float(grasp_center[0]), float(grasp_center[1])],
        "line_left_px": [float(left[0]), float(left[1])],
        "line_right_px": [float(right[0]), float(right[1])],
        "grip_width_px": grip_width_px,
        "grip_width_in_range": bool(grip_width_in_range),
        "workspace_both_reachable": bool(workspace.get("both_reachable")),
        "left_reachable": bool(workspace["left_reachable"]),
        "right_reachable": bool(workspace["right_reachable"]),
        "target_contact_quality": contact_quality,
        "patch_quality_score": contact_quality,
        "pair_quality_score": candidate_score,
        "rejection_reasons": rejected_reasons,
        "rejected": bool(rejected_reasons),
        "rejected_reasons": rejected_reasons,
    }

    selected_fields = {
        "left": left,
        "right": right,
        "left_stats": left_stats,
        "right_stats": right_stats,
        "left_3d": left_3d,
        "right_3d": right_3d,
        "workspace": workspace,
        "contacts_for_epsilon": contacts_for_epsilon,
        "candidate": candidate,
        "grasp_center": grasp_center,
        "body_fraction": body_fraction,
        "grip_width_px": grip_width_px,
        "grip_width_mm": width_mm,
        "width_score": width_score,
        "grip_width_in_range": grip_width_in_range,
        "reachability_score": reachability_score,
        "contact_quality": contact_quality,
        "balance_score": balance_score,
    }
    return selected_fields, None


def candidate_center_distance(candidate):
    fraction = candidate.get("requested_body_fraction")
    if fraction is None:
        fraction = candidate.get("body_fraction", CENTER_GRASP_BODY_FRACTION)
    try:
        return abs(float(fraction) - CENTER_GRASP_BODY_FRACTION)
    except (TypeError, ValueError):
        return float("inf")


def normalized_contact_point(point):
    try:
        values = np.asarray(point, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        return None
    return values[:2]


def candidate_contact_distance(candidate, preferred_contact_point_px):
    preferred = normalized_contact_point(preferred_contact_point_px)
    center = normalized_contact_point(candidate.get("grasp_center_px"))
    if preferred is None or center is None:
        return None
    return float(np.linalg.norm(center - preferred))


def candidate_ranking_table(candidates, selected_candidate_id=None, preferred_contact_point_px=None):
    preferred = normalized_contact_point(preferred_contact_point_px)

    def ranking_key(candidate):
        safety = (
            not bool(candidate.get("selection_eligible")),
            bool(candidate.get("rejected")),
        )
        if preferred is not None:
            return safety + (
                candidate_contact_distance(candidate, preferred),
                str(candidate.get("grasp_preference_tier") or "FALLBACK") != "PREFERRED",
                candidate_center_distance(candidate),
                -float(candidate.get("candidate_score", 0.0) or 0.0),
            )
        return safety + (
            str(candidate.get("grasp_preference_tier") or "FALLBACK") != "PREFERRED",
            candidate_center_distance(candidate),
            -float(candidate.get("candidate_score", 0.0) or 0.0),
        )

    rows = []
    for rank, candidate in enumerate(
        sorted(candidates, key=ranking_key),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "selected": candidate.get("candidate_id") == selected_candidate_id,
                "body_fraction": candidate.get("body_fraction"),
                "requested_body_fraction": candidate.get("requested_body_fraction"),
                "grasp_preference_tier": candidate.get("grasp_preference_tier"),
                "clearance_allowed": bool(candidate.get("clearance_allowed", True)),
                "selection_eligible": bool(candidate.get("selection_eligible")),
                "hand_reach_distance_px": candidate.get("hand_reach_distance_px"),
                "candidate_score": candidate.get("candidate_score"),
                "valid": not bool(candidate.get("rejected")),
                "grip_width_mm": candidate.get("grip_width_mm"),
                "depth_confidence": candidate.get("depth_confidence"),
                "force_closure": candidate.get("force_closure"),
                "rejection_reasons": list(candidate.get("rejection_reasons", candidate.get("rejected_reasons", [])) or []),
            }
        )
    return rows


def robustness_check_for_selected(pose, depth, intrinsics, selected, axis, grasp_direction, profile):
    if not selected:
        return {
            "robustness_trials": [],
            "robustness_pass_count": 0,
            "robustness_total_count": 0,
            "robustness_score": 0.0,
            "robustness_failure_reasons": ["no_selected_candidate"],
        }
    base_fraction = float(selected["candidate"].get("body_fraction", selected.get("body_fraction", 0.5)) or 0.5)
    trial_specs = [
        ("center_nominal", base_fraction),
        ("body_fraction_minus_0.03", base_fraction - 0.03),
        ("body_fraction_plus_0.03", base_fraction + 0.03),
    ]
    trials = []
    failures = []
    for name, fraction in trial_specs:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        center = center_for_body_fraction(pose, fraction)
        result, rejection = evaluate_candidate(
            pose,
            depth,
            intrinsics,
            center,
            axis,
            grasp_direction,
            fraction,
            profile=profile,
        )
        if result is None:
            reasons = list((rejection or {}).get("reasons", ["candidate_generation_failed"]))
            passed = False
            score = 0.0
        else:
            candidate = result["candidate"]
            reasons = list(candidate.get("rejection_reasons", candidate.get("rejected_reasons", [])) or [])
            passed = bool(not candidate.get("rejected") and candidate.get("candidate_score", 0.0) >= 55.0)
            score = float(candidate.get("candidate_score", 0.0) or 0.0)
        if not passed:
            failures.extend(reasons or [name])
        trials.append(
            {
                "trial": name,
                "body_fraction": fraction,
                "pass": passed,
                "candidate_score": score,
                "failure_reasons": reasons,
            }
        )
    pass_count = sum(1 for trial in trials if trial["pass"])
    total = len(trials)
    return {
        "robustness_trials": trials,
        "robustness_pass_count": pass_count,
        "robustness_total_count": total,
        "robustness_score": float(100.0 * pass_count / max(total, 1)),
        "robustness_failure_reasons": list(dict.fromkeys(failures)),
        "robustness_note": "Lightweight known-object robustness check over small body-fraction perturbations. Epsilon math is unchanged.",
    }


def plan_pregrasp(
    pose,
    depth,
    intrinsics=None,
    required_body_fraction=None,
    allowed_body_fractions=None,
    preferred_contact_point_px=None,
):
    profile = load_known_bottle_profile()
    mask = pose["body_mask"] if pose.get("body_mask") is not None else pose["cleaned_mask"]
    axis = unit(np.asarray(pose["axis_vector"], dtype=float))
    grasp_direction = perpendicular(axis)
    candidates = []
    rejected_candidates = []
    evaluated_results = []
    preferred_contact_point = normalized_contact_point(preferred_contact_point_px)
    fractions = list(profile_candidate_fractions(profile))
    preferred_fractions = []
    for value in profile.get("preferred_grasp_body_fractions", []):
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            continue
        if BODY_FRACTION_MIN <= fraction <= BODY_FRACTION_MAX and fraction not in preferred_fractions:
            preferred_fractions.append(fraction)
    allowed_fractions = None
    if allowed_body_fractions is not None:
        allowed_fractions = []
        for value in allowed_body_fractions:
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                continue
            if BODY_FRACTION_MIN <= fraction <= BODY_FRACTION_MAX and fraction not in allowed_fractions:
                allowed_fractions.append(fraction)
        if not allowed_fractions:
            return None, "No bottle-relative grasp band passed the current clearance gates."
    required_fraction = None
    if required_body_fraction is not None:
        try:
            required_fraction = float(required_body_fraction)
        except (TypeError, ValueError):
            required_fraction = None
    if required_fraction is not None and BODY_FRACTION_MIN <= required_fraction <= BODY_FRACTION_MAX:
        fractions = [required_fraction] + [
            fraction for fraction in fractions if abs(float(fraction) - required_fraction) > 1e-6
        ]
    else:
        required_fraction = None
    if (
        required_fraction is not None
        and allowed_fractions is not None
        and not any(
            abs(required_fraction - value) <= BODY_FRACTION_EQUIVALENCE_TOLERANCE
            for value in allowed_fractions
        )
    ):
        return None, "Remembered bottle-relative grasp band is not allowed by the current clearance gates."
    required_result = None
    required_rejection = None
    for index, fraction in enumerate(fractions):
        center = center_for_body_fraction(pose, fraction)
        if not point_inside(mask, center):
            center = closest_mask_point(mask, center)
        result, rejection = evaluate_candidate(
            pose,
            depth,
            intrinsics,
            center,
            axis,
            grasp_direction,
            fraction,
            profile=profile,
            candidate_index=index,
        )
        if result is None:
            rejected_candidates.append(rejection)
            if required_fraction is not None and abs(float(fraction) - required_fraction) <= 1e-6:
                required_rejection = rejection
            continue
        preference_tier = (
            "PREFERRED"
            if any(
                abs(float(fraction) - value) <= BODY_FRACTION_EQUIVALENCE_TOLERANCE
                for value in preferred_fractions
            )
            else "FALLBACK"
        )
        clearance_allowed = bool(
            allowed_fractions is None
            or any(
                abs(float(fraction) - value) <= BODY_FRACTION_EQUIVALENCE_TOLERANCE
                for value in allowed_fractions
            )
        )
        result["candidate"]["grasp_preference_tier"] = preference_tier
        result["candidate"]["clearance_allowed"] = clearance_allowed
        result["candidate"]["selection_eligible"] = bool(
            clearance_allowed and not result["candidate"].get("rejected")
        )
        result["candidate"]["hand_reach_distance_px"] = candidate_contact_distance(
            result["candidate"],
            preferred_contact_point,
        )
        candidates.append(result["candidate"])
        evaluated_results.append(result)
        if required_fraction is not None and abs(float(fraction) - required_fraction) <= 1e-6:
            required_result = result
        if result["candidate"]["rejected"]:
            rejected_candidates.append({
                "candidate_id": result["candidate"].get("candidate_id"),
                "body_fraction": result["candidate"]["body_fraction"],
                "candidate_score": result["candidate"]["candidate_score"],
                "reasons": result["candidate"]["rejection_reasons"],
            })
    if required_fraction is not None and required_result is None:
        reasons = list((required_rejection or {}).get("reasons", []))
        return None, (
            "Remembered/clearance-constrained bottle-relative grasp band could not produce valid boundary contacts."
            + (f" {'; '.join(map(str, reasons))}" if reasons else "")
        )
    if required_result is not None and not required_result["candidate"].get("selection_eligible"):
        reasons = list(required_result["candidate"].get("rejection_reasons") or [])
        return None, (
            "Remembered bottle-relative grasp band did not pass the current planner validity gates."
            + (f" {'; '.join(map(str, reasons))}" if reasons else "")
        )
    if required_result is not None:
        selected = required_result
        selection_mode = "REMEMBERED_BOTTLE_LOCAL_BAND"
    else:
        selection_eligible = [
            result for result in evaluated_results
            if result["candidate"].get("selection_eligible")
        ]
        if selection_eligible:
            if preferred_contact_point is not None:
                selected = min(
                    selection_eligible,
                    key=lambda result: (
                        candidate_contact_distance(result["candidate"], preferred_contact_point),
                        result["candidate"].get("grasp_preference_tier") != "PREFERRED",
                        candidate_center_distance(result["candidate"]),
                        -float(result["candidate"].get("candidate_score", 0.0) or 0.0),
                    ),
                )
                selection_mode = "HAND_REACH_ALIGNED_BOTTLE_LOCAL_BAND"
            else:
                selected = max(
                    selection_eligible,
                    key=lambda result: (
                        result["candidate"].get("grasp_preference_tier") == "PREFERRED",
                        -candidate_center_distance(result["candidate"]),
                        float(result["candidate"].get("candidate_score", 0.0) or 0.0),
                    ),
                )
                selection_mode = "BOTTLE_CENTER_PREFERRED"
        elif allowed_fractions is not None:
            return None, "No planner-valid candidate also passed the current grasp-band clearance gates."
        elif evaluated_results:
            selected = max(
                evaluated_results,
                key=lambda result: (
                    not bool(result["candidate"].get("rejected")),
                    result["candidate"].get("grasp_preference_tier") == "PREFERRED",
                    -candidate_center_distance(result["candidate"]),
                    float(result["candidate"].get("candidate_score", 0.0) or 0.0),
                ),
            )
            selection_mode = "BEST_AVAILABLE_INVALID_CANDIDATE"
        else:
            selected = None
    if selected is None:
        return None, "Could not find two planned boundary contacts on the bottle body."
    ranking_table = candidate_ranking_table(
        candidates,
        selected["candidate"].get("candidate_id"),
        preferred_contact_point,
    )
    robustness = robustness_check_for_selected(pose, depth, intrinsics, selected, axis, grasp_direction, profile)

    left = selected["left"]
    right = selected["right"]
    grasp_center = selected["grasp_center"]
    grip_width_px = selected["grip_width_px"]
    width_mm = selected["grip_width_mm"]
    workspace = selected["workspace"]

    depth_valid = valid_depth_mask(depth)
    local_depth = depth[depth_valid & (pose["cleaned_mask"] > 0)]
    planned_depth_mm = float(np.median(local_depth)) if local_depth.size else None
    grasp_angle_degrees = float(np.degrees(np.arctan2(grasp_direction[1], grasp_direction[0])))

    return {
        "grasp_center_px": [float(grasp_center[0]), float(grasp_center[1])],
        "grasp_direction": [float(grasp_direction[0]), float(grasp_direction[1])],
        "grasp_angle_degrees": grasp_angle_degrees,
        "body_fraction": selected["body_fraction"],
        "left_contact_px": [int(round(left[0])), int(round(left[1]))],
        "right_contact_px": [int(round(right[0])), int(round(right[1]))],
        "left_contact_3d_m": selected["left_3d"],
        "right_contact_3d_m": selected["right_3d"],
        "left_depth_stats": selected["left_stats"],
        "right_depth_stats": selected["right_stats"],
        "planned_depth_mm": planned_depth_mm,
        "grip_width_px": grip_width_px,
        "grip_width_mm": width_mm,
        "raw_grip_width_score": selected["width_score"],
        "grip_width_in_range": bool(selected["grip_width_in_range"]),
        "workspace_reachability_score": workspace["workspace_score"],
        "positioning_guidance_score": workspace["workspace_score"],
        "workspace_guidance_only": True,
        "grip_width_score": selected["reachability_score"],
        "workspace_both_reachable": bool(workspace["both_reachable"]),
        "width_reachable": bool(selected["grip_width_in_range"]),
        "left_reachable": bool(workspace["left_reachable"]),
        "right_reachable": bool(workspace["right_reachable"]),
        "gripper_workspace": workspace,
        "contacts_on_body": bool(selected["candidate"]["contacts_on_body"]),
        "target_contact_quality": selected["contact_quality"],
        "left_right_balance_score": selected["balance_score"],
        "epsilon_contacts": selected["contacts_for_epsilon"],
        "contact_model_ready": bool(candidates and not selected["candidate"]["rejected"]),
        "contact_model_ready_reasons": [] if not selected["candidate"]["rejected"] else selected["candidate"]["rejection_reasons"],
        "valid_contact_band": {
            "body_fraction_min": BODY_FRACTION_MIN,
            "body_fraction_max": BODY_FRACTION_MAX,
            "sampled_fractions": list(fractions),
            "avoid_top_fraction": profile.get("avoid_top_fraction"),
            "avoid_bottom_fraction": profile.get("avoid_bottom_fraction"),
        },
        "grasp_profile_used": profile,
        "candidate_count": len(candidates),
        "valid_candidate_count": sum(1 for candidate in candidates if not candidate.get("rejected")),
        "rejected_candidate_count": len(rejected_candidates),
        "selected_candidate_id": selected["candidate"].get("candidate_id"),
        "selected_candidate_score": selected["candidate"].get("candidate_score"),
        "selected_body_fraction": selected["candidate"].get("body_fraction"),
        "selected_requested_body_fraction": selected["candidate"].get("requested_body_fraction"),
        "grasp_band_selection_mode": selection_mode,
        "hand_reach_alignment_applied": selection_mode == "HAND_REACH_ALIGNED_BOTTLE_LOCAL_BAND",
        "hand_reach_selection_anchor_px": (
            None if preferred_contact_point is None else preferred_contact_point.tolist()
        ),
        "selected_hand_reach_distance_px": selected["candidate"].get("hand_reach_distance_px"),
        "candidate_ranking_table": ranking_table,
        "candidate_generation_method": "known_object_profile_generate_filter_rank_lock",
        "grasp_body_fraction_constraint": {
            "status": (
                "APPLIED_BOTTLE_RELATIVE"
                if required_fraction is not None
                else (
                    "APPLIED_CURRENT_CLEARANCE_ALLOWED_SET"
                    if allowed_fractions is not None
                    else "NOT_REQUESTED"
                )
            ),
            "requested_body_fraction": required_fraction,
            "allowed_body_fractions": None if allowed_fractions is None else list(allowed_fractions),
            "selected_body_fraction": selected["candidate"].get("requested_body_fraction"),
            "measured_contact_body_fraction": selected["candidate"].get("body_fraction"),
            "selected_preference_tier": selected["candidate"].get("grasp_preference_tier"),
            "candidate_safety_gates_bypassed": False,
        },
        "grasp_memory_applied": bool(required_fraction is not None),
        "clearance_allowed_body_fractions": None if allowed_fractions is None else list(allowed_fractions),
        "selected_grasp_preference_tier": selected["candidate"].get("grasp_preference_tier"),
        "grasp_band_fallback_used": selected["candidate"].get("grasp_preference_tier") == "FALLBACK",
        **robustness,
        "candidate_contacts": candidates,
        "selected_contact_pair": selected["candidate"],
        "rejected_candidates": rejected_candidates,
    }, None

"""Object-state-aware grasp strategy selection (planning only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from grasp_project.realsense.object_state import ObjectState


@dataclass(frozen=True)
class GraspPlan:
    strategy: str
    approach: str
    target_pose: Optional[dict[str, Any]]
    force_profile: dict[str, Any]
    allow_rotation: bool
    rotation_target_deg: Optional[float]
    stop_conditions: list[str]
    applicability: str
    limitations: list[str]
    selected_candidate: Optional[dict[str, Any]] = None
    epsilon_evidence: Optional[dict[str, Any]] = None
    physically_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BottleTargetAssessment:
    target_id: str
    rank: int
    selected: bool
    eligible: bool
    feasibility_score: float
    detection_confidence: float
    bbox_xywh: Optional[list[int]]
    center_px: Optional[list[float]]
    depth_median_mm: Optional[float]
    depth_geometry_status: str
    depth_geometry_score: Optional[float]
    depth_foreground_separation_mm: Optional[float]
    depth_foreground_separated: Optional[bool]
    depth_foreground_override_eligible: bool
    depth_foreground_usable: bool
    metric_dimensions_valid: Optional[bool]
    orientation: str
    standing_confidence: float
    available_for_pickup: bool
    occluded: bool
    strategy: str
    approach: str
    grasp_clearance_status: str
    selected_grasp_body_fraction: Optional[float]
    preferred_grasp_body_fraction: Optional[float]
    grasp_band_fallback_used: bool
    clear_grasp_body_fractions: list[float]
    preferred_grasp_body_fractions: list[float]
    fallback_grasp_body_fractions: list[float]
    grasp_band_clearance_table: list[dict[str, Any]]
    left_clearance_px: Optional[float]
    right_clearance_px: Optional[float]
    required_clearance_px: Optional[float]
    blocking_target_ids: list[str]
    left_depth_obstacle_ratio: Optional[float]
    right_depth_obstacle_ratio: Optional[float]
    depth_corridor_status: str
    distance_priority_score: float
    top_exposure_score: float
    reasons: list[str]
    evidence_status: str = "ESTIMATED_FROM_CURRENT_ALIGNED_RGBD"
    physically_validated: bool = False


@dataclass(frozen=True)
class MultiBottleSelection:
    selected_target_id: Optional[str]
    selection_status: str
    candidate_count: int
    eligible_count: int
    candidates: list[BottleTargetAssessment]
    grasp_feasibility_score: float
    grasp_feasibility_label: str
    grasp_feasibility_reason: str
    suggested_operator_action: str
    suggested_robot_action: str
    limitations: list[str]
    physically_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_task_grasp(
    object_state: ObjectState | Mapping[str, Any],
    *,
    pregrasp_plan: Optional[Mapping[str, Any]] = None,
    epsilon_result: Optional[Mapping[str, Any]] = None,
    target_pose: Optional[Mapping[str, Any]] = None,
) -> GraspPlan:
    """Select a symbolic task policy without recomputing candidates or epsilon."""
    state = object_state.to_dict() if isinstance(object_state, ObjectState) else dict(object_state)
    orientation = str(state.get("orientation") or "UNKNOWN").upper()
    pose_class = str(state.get("pose_class") or "UNKNOWN").upper()
    reasons = [str(reason) for reason in state.get("reasons") or []]
    available = bool(state.get("available_for_pickup"))
    occluded = bool(state.get("occluded"))
    limitations = [
        "PLANNING_ONLY_NO_ACTUATION",
        "NOT_PHYSICALLY_VALIDATED",
        "force_profile_is_symbolic_not_measured_force",
    ]
    stop_conditions = [
        "target_lost_or_tracking_stale",
        "depth_or_contact_geometry_invalid",
        "object_motion_invalid_or_global_disturbance",
        "gross_slip_or_unregulated_rotation",
    ]
    if occluded or not available:
        blockers = reasons or (["object_occluded"] if occluded else ["object_not_available_for_pickup"])
        return GraspPlan(
            strategy="DO_NOT_ATTEMPT",
            approach="NONE",
            target_pose=None,
            force_profile=_symbolic_force_profile("HOLD_ZERO"),
            allow_rotation=False,
            rotation_target_deg=None,
            stop_conditions=blockers + stop_conditions,
            applicability="PLANNING_ONLY_BLOCKED_BY_OBJECT_STATE",
            limitations=limitations,
            epsilon_evidence={"status": "NOT_APPLICABLE", "reason": "grasp_attempt_blocked"},
        )

    selected = _selected_candidate(pregrasp_plan)
    epsilon = _reuse_epsilon_evidence(selected, epsilon_result)
    if pose_class in {"TOP_DOWN_CAP", "TOP_DOWN_BODY"}:
        limitations.append("top_down_contact_geometry_not_generated_by_existing_side_candidate_planner")
        return GraspPlan(
            strategy="TOP_DOWN_PICK",
            approach="TOP",
            target_pose=None if target_pose is None else dict(target_pose),
            force_profile=_symbolic_force_profile("MINIMAL_TOP_PICK"),
            allow_rotation=False,
            rotation_target_deg=None,
            stop_conditions=stop_conditions,
            applicability="PLANNING_ONLY_VALIDATED_WITH_SYNTHETIC_STATE",
            limitations=limitations,
            selected_candidate=None,
            epsilon_evidence={"status": "NOT_APPLICABLE", "reason": "top_down_contact_geometry_not_available"},
        )
    if orientation == "STANDING":
        if selected is None:
            limitations.append("existing_pregrasp_candidate_not_available")
        elif any(selected.get(key) is not None for key in ("epsilon_value", "epsilon_score", "force_closure")):
            selected["epsilon_applicability"] = "PLANNING_ONLY_ANALYTIC_NOT_CURRENT_CONTACT_QUALITY"
            selected["epsilon_physically_validated"] = False
            limitations.append("pregrasp_epsilon_is_analytic_planning_evidence_not_confirmed_contact_quality")
        return GraspPlan(
            strategy="SIDE_ANTIPODAL",
            approach="SIDE",
            target_pose=None if target_pose is None else dict(target_pose),
            force_profile=_symbolic_force_profile("NOMINAL_SIDE_HOLD"),
            allow_rotation=False,
            rotation_target_deg=0.0,
            stop_conditions=stop_conditions,
            applicability="PLANNING_ONLY_VALIDATED_WITH_SYNTHETIC_STATE",
            limitations=limitations,
            selected_candidate=selected,
            epsilon_evidence=epsilon,
        )
    if orientation == "LYING":
        limitations.append("top_grasp_contact_geometry_not_generated_by_existing_side_candidate_planner")
        return GraspPlan(
            strategy="TOP_GRASP_ROTATE_UPRIGHT",
            approach="TOP",
            target_pose=None if target_pose is None else dict(target_pose),
            force_profile=_symbolic_force_profile("MINIMAL_ROTATION_REGULATED"),
            allow_rotation=True,
            rotation_target_deg=0.0,
            stop_conditions=stop_conditions + ["stop_rotation_when_tilt_near_zero"],
            applicability="PLANNING_ONLY_VALIDATED_WITH_SYNTHETIC_STATE",
            limitations=limitations,
            selected_candidate=None,
            epsilon_evidence={"status": "NOT_APPLICABLE", "reason": "top_grasp_contact_geometry_not_available"},
        )
    return GraspPlan(
        strategy="DO_NOT_ATTEMPT",
        approach="NONE",
        target_pose=None,
        force_profile=_symbolic_force_profile("HOLD_ZERO"),
        allow_rotation=False,
        rotation_target_deg=None,
        stop_conditions=[f"unsupported_orientation_{orientation.lower()}"] + stop_conditions,
        applicability="PLANNING_ONLY_BLOCKED_BY_OBJECT_STATE",
        limitations=limitations,
        epsilon_evidence={"status": "NOT_APPLICABLE", "reason": "orientation_not_supported_by_task_policy"},
    )


def select_bottle_target(
    targets: Iterable[Mapping[str, Any]],
    *,
    image_shape: Optional[tuple[int, ...]] = None,
    intrinsics: Optional[Mapping[str, Any]] = None,
    depth_image: Any = None,
) -> MultiBottleSelection:
    """Rank frame-local detections by safety gates, clearance, confidence, and distance."""
    targets = list(targets)
    target_depths = [
        depth
        for target in targets
        if (depth := _optional_float((target.get("pose") or {}).get("depth_median_mm"))) is not None
    ]
    scored = []
    for index, target in enumerate(targets):
        pose = dict(target.get("pose") or {})
        state_value = target.get("object_state") or {}
        state = state_value.to_dict() if isinstance(state_value, ObjectState) else dict(state_value)
        task_plan = plan_task_grasp(state)
        target_id = str(target.get("target_id") or pose.get("target_id") or f"bottle_{index + 1:02d}")
        detection_confidence = _bounded(pose.get("detection_confidence"))
        availability = (state.get("evidence") or {}).get("availability") or {}
        mask_completeness = _bounded(availability.get("mask_completeness"))
        depth_valid_ratio = _bounded(availability.get("depth_valid_ratio"))
        profile_score = _bounded(pose.get("bottle_profile_agreement_score"))
        depth_geometry = pose.get("depth_geometry_evidence") or {}
        depth_foreground_separated = depth_geometry.get("depth_foreground_separated")
        depth_foreground_override = bool(
            pose.get("demo_depth_foreground_override_eligible")
        )
        depth_foreground_usable = bool(
            depth_foreground_separated is not False or depth_foreground_override
        )
        approximate_dimensions = pose.get("approximate_bbox_dimensions_mm")
        metric_dimensions_valid = (
            bool(approximate_dimensions.get("broad_bottle_dimensions_valid"))
            if isinstance(approximate_dimensions, Mapping)
            else None
        )
        border_touch = bool((pose.get("candidate_evidence_scores") or {}).get("border_touch"))
        clearance = evaluate_grasp_band_clearance(
            index,
            targets,
            image_shape=image_shape,
            intrinsics=intrinsics,
            depth_image=depth_image,
        )
        side_clearance_ok = bool(
            task_plan.approach != "SIDE"
            or clearance["grasp_clearance_status"] == "ESTIMATED_CLEAR"
        )
        eligible = bool(
            state.get("available_for_pickup")
            and not state.get("occluded")
            and not border_touch
            and depth_foreground_usable
            and task_plan.strategy != "DO_NOT_ATTEMPT"
            and side_clearance_ok
        )
        approach_score = 1.0 if task_plan.approach == "SIDE" else (0.75 if task_plan.approach == "TOP" else 0.0)
        standing_confidence = _bounded(state.get("standing_confidence"))
        distance_priority = _distance_priority_score(pose.get("depth_median_mm"), target_depths)
        top_exposure = _top_exposure_score(index, targets)
        feasibility = (
            40.0 * float(eligible)
            + 12.0 * detection_confidence
            + 8.0 * standing_confidence
            + 10.0 * distance_priority
            + 7.0 * mask_completeness
            + 7.0 * depth_valid_ratio
            + 6.0 * approach_score
            + 5.0 * profile_score
            + 5.0 * top_exposure
        )
        if not eligible:
            feasibility = min(feasibility, 39.0)
        reasons = list(state.get("reasons") or [])
        if border_touch:
            reasons.append("candidate_touches_image_border")
        if depth_foreground_separated is False and not depth_foreground_override:
            reasons.append("candidate_not_separated_from_surrounding_depth_surface")
        elif depth_foreground_override:
            reasons.append("known_bottle_supported_by_mixed_depth_boundary")
        if task_plan.approach == "SIDE" and not side_clearance_ok:
            reasons.append("no_collision_free_side_grasp_band")
        elif task_plan.approach == "SIDE" and clearance["grasp_band_fallback_used"]:
            reasons.append("less_optimal_clear_grasp_band_selected")
        if task_plan.approach == "TOP":
            reasons.append("top_approach_is_planning_only_without_top_contact_geometry")
        if eligible:
            reasons.append(f"{task_plan.approach.lower()}_approach_supported_by_task_policy")
        scored.append(
            {
                "target_id": target_id,
                "eligible": eligible,
                "feasibility_score": round(feasibility, 2),
                "detection_confidence": round(detection_confidence, 4),
                "bbox_xywh": _number_list(pose.get("bbox_xywh"), int),
                "center_px": _number_list(pose.get("center_px"), float),
                "depth_median_mm": _optional_float(pose.get("depth_median_mm")),
                "depth_geometry_status": str(
                    depth_geometry.get("depth_geometry_status") or "NOT_TESTED"
                ),
                "depth_geometry_score": _optional_float(depth_geometry.get("depth_geometry_score")),
                "depth_foreground_separation_mm": _optional_float(
                    depth_geometry.get("depth_foreground_separation_mm")
                ),
                "depth_foreground_separated": (
                    bool(depth_foreground_separated)
                    if depth_foreground_separated is not None
                    else None
                ),
                "depth_foreground_override_eligible": depth_foreground_override,
                "depth_foreground_usable": depth_foreground_usable,
                "metric_dimensions_valid": metric_dimensions_valid,
                "orientation": str(state.get("orientation") or "UNKNOWN"),
                "standing_confidence": round(standing_confidence, 4),
                "available_for_pickup": bool(state.get("available_for_pickup")),
                "occluded": bool(state.get("occluded")),
                "strategy": task_plan.strategy,
                "approach": task_plan.approach,
                "grasp_clearance_status": clearance["grasp_clearance_status"],
                "selected_grasp_body_fraction": clearance["selected_grasp_body_fraction"],
                "preferred_grasp_body_fraction": clearance["preferred_grasp_body_fraction"],
                "grasp_band_fallback_used": clearance["grasp_band_fallback_used"],
                "clear_grasp_body_fractions": clearance["clear_grasp_body_fractions"],
                "preferred_grasp_body_fractions": clearance["preferred_grasp_body_fractions"],
                "fallback_grasp_body_fractions": clearance["fallback_grasp_body_fractions"],
                "grasp_band_clearance_table": clearance["grasp_band_clearance_table"],
                "left_clearance_px": clearance["left_clearance_px"],
                "right_clearance_px": clearance["right_clearance_px"],
                "required_clearance_px": clearance["required_clearance_px"],
                "blocking_target_ids": clearance["blocking_target_ids"],
                "left_depth_obstacle_ratio": clearance["left_depth_obstacle_ratio"],
                "right_depth_obstacle_ratio": clearance["right_depth_obstacle_ratio"],
                "depth_corridor_status": clearance["depth_corridor_status"],
                "distance_priority_score": round(distance_priority, 4),
                "top_exposure_score": round(top_exposure, 4),
                "reasons": list(dict.fromkeys(map(str, reasons))),
            }
        )

    scored.sort(
        key=lambda item: (
            not item["eligible"],
            -item["feasibility_score"],
            -item["standing_confidence"],
            -item["distance_priority_score"],
            -item["detection_confidence"],
            (item["center_px"] or [float("inf")])[0],
            item["target_id"],
        )
    )
    selected_id = next((item["target_id"] for item in scored if item["eligible"]), None)
    assessments = [
        BottleTargetAssessment(
            rank=rank,
            selected=item["target_id"] == selected_id,
            **item,
        )
        for rank, item in enumerate(scored, start=1)
    ]
    feasibility = _selection_feasibility(assessments, selected_id)
    return MultiBottleSelection(
        selected_target_id=selected_id,
        selection_status="TARGET_SELECTED" if selected_id else ("NO_FEASIBLE_TARGET" if assessments else "NO_BOTTLES_DETECTED"),
        candidate_count=len(assessments),
        eligible_count=sum(int(item.eligible) for item in assessments),
        candidates=assessments,
        **feasibility,
        limitations=[
            "PLANNING_ONLY_NO_ACTUATION",
            "FRAME_LOCAL_IDS_UNTIL_TARGET_LOCK",
            "ORIENTATION_IS_CAMERA_FRAME_NOT_GRAVITY_CALIBRATED",
            "PER_CANDIDATE_POINTCLOUD_NOT_EVALUATED",
            "INTER_BOTTLE_CLEARANCE_IS_IMAGE_PLANE_PLUS_ALIGNED_DEPTH_ESTIMATE",
            "NOT_PHYSICALLY_VALIDATED",
        ],
    )


def evaluate_grasp_band_clearance(
    target_index: int,
    targets: list[Mapping[str, Any]],
    *,
    image_shape: Optional[tuple[int, ...]] = None,
    intrinsics: Optional[Mapping[str, Any]] = None,
    depth_image: Any = None,
    finger_clearance_mm: float = 20.0,
    patch_height_mm: float = 50.0,
    depth_neighbor_margin_mm: float = 100.0,
    depth_contact_plane_tolerance_mm: float = 20.0,
) -> dict[str, Any]:
    """Estimate whether both side fingers fit at a bottle-relative grasp band."""
    if target_index < 0 or target_index >= len(targets):
        return _clearance_result("NOT_TESTED", None, None, False, None, None, None, [], None, None, "NOT_TESTED")
    target = targets[target_index]
    pose = target.get("pose") or {}
    bbox = _bbox_xyxy(pose)
    if bbox is None:
        return _clearance_result("NOT_TESTED_NO_BBOX", None, None, False, None, None, None, [], None, None, "NOT_TESTED")

    x1, y1, x2, y2 = bbox
    depth_mm = _optional_float(pose.get("depth_median_mm"))
    fx = _optional_float((intrinsics or {}).get("fx"))
    fy = _optional_float((intrinsics or {}).get("fy"))
    required_px = max(6.0, finger_clearance_mm * fx / depth_mm) if fx and depth_mm else max(6.0, 0.25 * (x2 - x1))
    half_band_px = max(6.0, 0.5 * patch_height_mm * fy / depth_mm) if fy and depth_mm else max(6.0, 0.09 * (y2 - y1))
    preferred = _preferred_grasp_fractions(pose)
    fallback = _fallback_grasp_fractions(pose)
    fractions = _grasp_fractions(pose)
    image_h = float(image_shape[0]) if image_shape else None
    image_w = float(image_shape[1]) if image_shape else None
    candidates = []
    for fraction in fractions:
        band_y = y1 + float(fraction) * (y2 - y1)
        band_top = band_y - half_band_px
        band_bottom = band_y + half_band_px
        left_gap = x1 if image_w is not None else float("inf")
        right_gap = image_w - x2 if image_w is not None else float("inf")
        blockers = []
        for neighbor_index, neighbor in enumerate(targets):
            if neighbor_index == target_index:
                continue
            neighbor_pose = neighbor.get("pose") or {}
            neighbor_bbox = _bbox_xyxy(neighbor_pose)
            if neighbor_bbox is None:
                continue
            nx1, ny1, nx2, ny2 = neighbor_bbox
            neighbor_depth = _optional_float(neighbor_pose.get("depth_median_mm"))
            if (
                depth_mm is not None
                and neighbor_depth is not None
                and neighbor_depth > depth_mm + float(depth_neighbor_margin_mm)
            ):
                continue
            if min(band_bottom, ny2) <= max(band_top, ny1):
                continue
            neighbor_id = str(
                neighbor.get("target_id")
                or neighbor_pose.get("target_id")
                or f"bottle_{neighbor_index + 1:02d}"
            )
            neighbor_center_x = 0.5 * (nx1 + nx2)
            target_center_x = 0.5 * (x1 + x2)
            if neighbor_center_x <= target_center_x:
                gap = x1 - nx2
                left_gap = min(left_gap, gap)
            else:
                gap = nx1 - x2
                right_gap = min(right_gap, gap)
            if gap < required_px:
                blockers.append(neighbor_id)
        depth_corridor = _depth_corridor_obstacles(
            depth_image,
            pose.get("cleaned_mask") if pose.get("cleaned_mask") is not None else pose.get("body_mask"),
            (x1, y1, x2, y2),
            band_top,
            band_bottom,
            required_px,
            depth_mm,
            depth_contact_plane_tolerance_mm,
        )
        if depth_corridor["left_blocked"]:
            blockers.append("depth_obstacle_left")
        if depth_corridor["right_blocked"]:
            blockers.append("depth_obstacle_right")
        clear = bool(
            left_gap >= required_px
            and right_gap >= required_px
            and not depth_corridor["left_blocked"]
            and not depth_corridor["right_blocked"]
        )
        candidates.append(
            {
                "fraction": float(fraction),
                "preference_tier": (
                    "PREFERRED"
                    if _fraction_in(float(fraction), preferred)
                    else "FALLBACK"
                ),
                "left_gap": left_gap,
                "right_gap": right_gap,
                "clear": clear,
                "blockers": blockers,
                "left_depth_obstacle_ratio": depth_corridor["left_ratio"],
                "right_depth_obstacle_ratio": depth_corridor["right_ratio"],
                "depth_corridor_status": depth_corridor["status"],
            }
        )
    clear_candidates = [candidate for candidate in candidates if candidate["clear"]]
    clear_preferred = [
        candidate for candidate in clear_candidates
        if candidate["preference_tier"] == "PREFERRED"
    ]
    selected = min(
        clear_preferred or clear_candidates,
        key=lambda candidate: abs(candidate["fraction"] - 0.50),
        default=None,
    )
    preferred_center = min(preferred, key=lambda fraction: abs(fraction - 0.50), default=None)
    if selected is None:
        blockers = list(dict.fromkeys(blocker for candidate in candidates for blocker in candidate["blockers"]))
        best = max(candidates, key=lambda candidate: min(candidate["left_gap"], candidate["right_gap"]))
        return _clearance_result(
            "BLOCKED_BY_NEIGHBOR_OR_IMAGE_EDGE",
            None,
            preferred_center,
            False,
            best["left_gap"],
            best["right_gap"],
            required_px,
            blockers,
            best["left_depth_obstacle_ratio"],
            best["right_depth_obstacle_ratio"],
            best["depth_corridor_status"],
            clear_fractions=[],
            preferred_fractions=preferred,
            fallback_fractions=fallback,
            candidates=candidates,
        )
    return _clearance_result(
        "ESTIMATED_CLEAR",
        selected["fraction"],
        preferred_center,
        selected["preference_tier"] == "FALLBACK",
        selected["left_gap"],
        selected["right_gap"],
        required_px,
        selected["blockers"],
        selected["left_depth_obstacle_ratio"],
        selected["right_depth_obstacle_ratio"],
        selected["depth_corridor_status"],
        clear_fractions=[candidate["fraction"] for candidate in clear_candidates],
        preferred_fractions=preferred,
        fallback_fractions=fallback,
        candidates=candidates,
    )


def finalize_planned_grasp_selection(
    selection: dict[str, Any],
    selected_target_id: Optional[str],
    pregrasp_plan: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Make the planner winner authoritative in the target-selection report."""
    if not isinstance(selection, dict) or not selected_target_id or not pregrasp_plan:
        return selection
    selected_pair = pregrasp_plan.get("selected_contact_pair") or {}
    selected_band_fraction = _optional_float(
        selected_pair.get("requested_body_fraction")
        if selected_pair.get("requested_body_fraction") is not None
        else pregrasp_plan.get("selected_requested_body_fraction")
    )
    measured_contact_fraction = _optional_float(
        selected_pair.get("body_fraction")
        if selected_pair.get("body_fraction") is not None
        else pregrasp_plan.get("body_fraction")
    )
    if selected_band_fraction is None:
        return selection
    candidates = selection.get("candidates") or []
    target = next(
        (
            candidate for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("target_id") == selected_target_id
        ),
        None,
    )
    if target is None:
        return selection
    allowed = target.get("clear_grasp_body_fractions") or []
    if allowed and not _fraction_in(selected_band_fraction, allowed):
        return selection
    selected_tier = str(pregrasp_plan.get("selected_grasp_preference_tier") or "").upper()
    preferred = target.get("preferred_grasp_body_fractions") or []
    fallback = bool(
        selected_tier == "FALLBACK"
        or (selected_tier != "PREFERRED" and not _fraction_in(selected_band_fraction, preferred))
    )
    target["selected_grasp_body_fraction"] = round(selected_band_fraction, 4)
    target["measured_contact_body_fraction"] = (
        None if measured_contact_fraction is None else round(measured_contact_fraction, 4)
    )
    target["grasp_band_fallback_used"] = fallback
    target["grasp_band_selection_source"] = "PLANNER_RANKED_WITHIN_CURRENT_CLEARANCE_ALLOWED_SET"
    target["selected_grasp_candidate_id"] = pregrasp_plan.get("selected_candidate_id")
    target["selected_grasp_candidate_score"] = pregrasp_plan.get("selected_candidate_score")
    target["selected_grasp_preference_tier"] = "FALLBACK" if fallback else "PREFERRED"
    clearance_row = next(
        (
            row for row in target.get("grasp_band_clearance_table") or []
            if _fraction_in(selected_band_fraction, [row.get("body_fraction")])
        ),
        None,
    )
    if clearance_row is not None:
        for source, destination in (
            ("left_clearance_px", "left_clearance_px"),
            ("right_clearance_px", "right_clearance_px"),
            ("blocking_target_ids", "blocking_target_ids"),
            ("left_depth_obstacle_ratio", "left_depth_obstacle_ratio"),
            ("right_depth_obstacle_ratio", "right_depth_obstacle_ratio"),
            ("depth_corridor_status", "depth_corridor_status"),
        ):
            target[destination] = clearance_row.get(source)
    reasons = [
        reason for reason in target.get("reasons") or []
        if reason != "less_optimal_clear_grasp_band_selected"
    ]
    if fallback:
        reasons.append("less_optimal_clear_grasp_band_selected")
    target["reasons"] = list(dict.fromkeys(map(str, reasons)))
    if target.get("eligible") and target.get("approach") == "SIDE":
        selection.update(
            _selected_side_feasibility(
                float(target.get("feasibility_score", 0.0) or 0.0),
                fallback,
            )
        )
    return selection


def mark_planned_grasp_selection_failed(
    selection: dict[str, Any],
    selected_target_id: Optional[str],
    reason: Optional[str],
) -> dict[str, Any]:
    """Prevent a provisional clearance result from claiming an optimal plan."""
    if not isinstance(selection, dict):
        return selection
    target = next(
        (
            candidate for candidate in selection.get("candidates") or []
            if isinstance(candidate, dict) and candidate.get("target_id") == selected_target_id
        ),
        None,
    )
    if target is not None:
        target["selected_grasp_body_fraction"] = None
        target["grasp_band_fallback_used"] = False
        target["grasp_band_selection_source"] = "NO_PLANNER_VALID_CLEAR_BAND"
        target["grasp_plan_status"] = "BLOCKED_BY_PLANNER_VALIDITY"
        target["grasp_plan_failure_reason"] = str(reason or "no_planner_valid_clear_band")
        target["reasons"] = list(
            dict.fromkeys(
                [
                    item for item in target.get("reasons") or []
                    if item != "less_optimal_clear_grasp_band_selected"
                ]
                + ["no_planner_valid_clear_band"]
            )
        )
    selection.update(
        {
            "grasp_feasibility_label": "CANNOT_GRASP_PLANNER_INVALID",
            "grasp_feasibility_reason": "no_planner_valid_candidate_in_current_clearance_allowed_set",
            "suggested_operator_action": "Adjust the view or finger clearance until a planner-valid grasp band is available.",
            "suggested_robot_action": "HOLD_NO_ACTUATION",
        }
    )
    return selection


def _selection_feasibility(
    assessments: list[BottleTargetAssessment],
    selected_target_id: Optional[str],
) -> dict[str, Any]:
    """Summarize the existing safety-gated target ranking for the operator."""
    selected = next(
        (item for item in assessments if item.target_id == selected_target_id),
        None,
    )
    if selected is not None:
        if selected.approach != "SIDE":
            return {
                "grasp_feasibility_score": selected.feasibility_score,
                "grasp_feasibility_label": "HOLD_AND_ADJUST",
                "grasp_feasibility_reason": "top_approach_is_planning_only_without_validated_top_contact_geometry",
                "suggested_operator_action": "Keep the target visible and obtain a view/contact model suitable for the planned top approach.",
                "suggested_robot_action": "HOLD_NO_ACTUATION",
            }
        return _selected_side_feasibility(
            selected.feasibility_score,
            selected.grasp_band_fallback_used,
        )

    if not assessments:
        return {
            "grasp_feasibility_score": 0.0,
            "grasp_feasibility_label": "HOLD_AND_ADJUST",
            "grasp_feasibility_reason": "no_current_bottle_candidate",
            "suggested_operator_action": "Move to a clear view where the full bottle and surrounding grasp space are visible.",
            "suggested_robot_action": "HOLD_NO_TARGET",
        }

    score = max(item.feasibility_score for item in assessments)
    blocked_by_neighbor = any(
        item.blocking_target_ids
        and any(not blocker.startswith("depth_obstacle_") for blocker in item.blocking_target_ids)
        for item in assessments
    )
    blocked_clearance = any(
        item.grasp_clearance_status == "BLOCKED_BY_NEIGHBOR_OR_IMAGE_EDGE"
        for item in assessments
    )
    depth_invalid = all(
        item.depth_median_mm is None
        or item.depth_geometry_status in {"NOT_AVAILABLE", "INVALID"}
        for item in assessments
    )
    availability_blocked = all(not item.available_for_pickup for item in assessments)
    unsupported_pose = all(item.strategy == "DO_NOT_ATTEMPT" for item in assessments)
    if blocked_by_neighbor and len(assessments) > 1:
        label = "CANNOT_GRASP_MULTIPLE_OBJECTS"
        reason = "adjacent_object_blocks_required_parallel_jaw_clearance"
        operator_action = "Separate the bottles or expose a clear contact band before grasping."
    elif blocked_clearance:
        label = "CANNOT_GRASP_CLEARANCE"
        reason = "no_collision_free_grasp_band_for_current_parallel_jaw_geometry"
        operator_action = "Reposition the hand or clear the blocked finger corridor."
    elif depth_invalid:
        label = "CANNOT_GRASP_DEPTH_INVALID"
        reason = "current_aligned_depth_support_is_not_valid_for_grasp_guidance"
        operator_action = "Adjust the view until the bottle has stable, complete aligned depth."
    elif availability_blocked:
        label = "CANNOT_GRASP_AVAILABILITY"
        reason = "current_object_state_availability_evidence_failed"
        operator_action = "Inspect the displayed availability reasons and expose the bottle boundary or clear the occluder."
    elif unsupported_pose:
        label = "CANNOT_GRASP_POSE"
        reason = "current_pose_is_not_supported_by_the_available_contact_model"
        operator_action = "Obtain a clearer side view or safely reorient the bottle outside this planner."
    else:
        label = "HOLD_AND_ADJUST"
        reason = "no_candidate_passed_all_current_availability_and_safety_gates"
        operator_action = "Inspect the displayed blockers and improve visibility or clearance."
    return {
        "grasp_feasibility_score": score,
        "grasp_feasibility_label": label,
        "grasp_feasibility_reason": reason,
        "suggested_operator_action": operator_action,
        "suggested_robot_action": "HOLD_NO_ACTUATION",
    }


def _selected_side_feasibility(score, fallback):
    if fallback:
        return {
            "grasp_feasibility_score": score,
            "grasp_feasibility_label": "LESS_OPTIMAL_BUT_ACCEPTABLE",
            "grasp_feasibility_reason": "preferred_contact_bands_blocked_clear_fallback_band_selected",
            "suggested_operator_action": "Use the displayed fallback band and keep both finger corridors clear.",
            "suggested_robot_action": "PLAN_ONLY_USE_FALLBACK_CONTACT_BAND",
        }
    return {
        "grasp_feasibility_score": score,
        "grasp_feasibility_label": "OPTIMAL_GRASP",
        "grasp_feasibility_reason": "highest_ranked_valid_clear_preferred_side_contact_band_selected",
        "suggested_operator_action": "Follow the camera-frame distance guidance while keeping the selected bottle unobstructed.",
        "suggested_robot_action": "PLAN_ONLY_APPROACH_SELECTED_TARGET",
    }


def _clearance_result(
    status,
    selected,
    preferred,
    fallback,
    left,
    right,
    required,
    blockers,
    left_depth_ratio=None,
    right_depth_ratio=None,
    depth_corridor_status="NOT_TESTED",
    clear_fractions=None,
    preferred_fractions=None,
    fallback_fractions=None,
    candidates=None,
):
    finite = lambda value: None if value is None or value == float("inf") else round(float(value), 2)
    candidate_rows = []
    for candidate in candidates or []:
        candidate_rows.append(
            {
                "body_fraction": round(float(candidate["fraction"]), 4),
                "preference_tier": str(candidate.get("preference_tier") or "UNKNOWN"),
                "clear": bool(candidate.get("clear")),
                "left_clearance_px": finite(candidate.get("left_gap")),
                "right_clearance_px": finite(candidate.get("right_gap")),
                "blocking_target_ids": list(dict.fromkeys(map(str, candidate.get("blockers") or []))),
                "left_depth_obstacle_ratio": finite(candidate.get("left_depth_obstacle_ratio")),
                "right_depth_obstacle_ratio": finite(candidate.get("right_depth_obstacle_ratio")),
                "depth_corridor_status": str(candidate.get("depth_corridor_status") or "NOT_TESTED"),
            }
        )
    return {
        "grasp_clearance_status": status,
        "selected_grasp_body_fraction": None if selected is None else round(float(selected), 4),
        "preferred_grasp_body_fraction": None if preferred is None else round(float(preferred), 4),
        "grasp_band_fallback_used": bool(fallback),
        "clear_grasp_body_fractions": [round(float(value), 4) for value in clear_fractions or []],
        "preferred_grasp_body_fractions": [round(float(value), 4) for value in preferred_fractions or []],
        "fallback_grasp_body_fractions": [round(float(value), 4) for value in fallback_fractions or []],
        "grasp_band_clearance_table": candidate_rows,
        "left_clearance_px": finite(left),
        "right_clearance_px": finite(right),
        "required_clearance_px": finite(required),
        "blocking_target_ids": list(dict.fromkeys(map(str, blockers or []))),
        "left_depth_obstacle_ratio": finite(left_depth_ratio),
        "right_depth_obstacle_ratio": finite(right_depth_ratio),
        "depth_corridor_status": str(depth_corridor_status),
        "evidence_status": "ESTIMATED_IMAGE_PLANE_WITH_ALIGNED_DEPTH_SCALE",
    }


def _depth_corridor_obstacles(
    depth_image,
    target_mask,
    bbox,
    band_top,
    band_bottom,
    required_px,
    target_depth_mm,
    contact_plane_tolerance_mm,
):
    """Measure non-target occupancy at or in front of the contact plane.

    A surface safely behind the observed bottle surface is background, not a
    side-finger collision.  Closer pixels and pixels within the conservative
    contact-plane tolerance still block.  This keeps the safety gate active
    for adjacent bottles without treating a nearby back plane as an obstacle.
    """
    unavailable = {
        "left_ratio": None,
        "right_ratio": None,
        "left_blocked": False,
        "right_blocked": False,
        "status": "NOT_TESTED_NO_ALIGNED_DEPTH",
    }
    if depth_image is None or target_depth_mm is None:
        return unavailable
    depth = np.asarray(depth_image)
    if depth.ndim != 2:
        return unavailable
    height, width = depth.shape
    x1, _y1, x2, _y2 = bbox
    top = max(0, int(np.floor(band_top)))
    bottom = min(height, int(np.ceil(band_bottom)))
    corridor_width = max(1, int(np.ceil(required_px)))
    if bottom <= top:
        return unavailable
    mask = None
    if target_mask is not None and np.asarray(target_mask).shape == depth.shape:
        mask = np.asarray(target_mask) > 0

    def occupancy(left, right):
        left = max(0, int(np.floor(left)))
        right = min(width, int(np.ceil(right)))
        if right <= left:
            return 1.0, 0
        corridor = np.zeros(depth.shape, dtype=bool)
        corridor[top:bottom, left:right] = True
        if mask is not None:
            corridor &= ~mask
        corridor_pixels = int(np.count_nonzero(corridor))
        if corridor_pixels == 0:
            return 1.0, 0
        valid = corridor & np.isfinite(depth) & (depth >= 50.0) & (depth <= 2000.0)
        contact_plane_upper_mm = float(target_depth_mm) + max(
            0.0,
            float(contact_plane_tolerance_mm),
        )
        near = valid & (depth <= contact_plane_upper_mm)
        return float(np.count_nonzero(near) / corridor_pixels), int(np.count_nonzero(valid))

    left_ratio, left_valid = occupancy(x1 - required_px, x1)
    right_ratio, right_valid = occupancy(x2, x2 + required_px)
    minimum_support = max(6, int(round((bottom - top) * corridor_width * 0.10)))
    left_blocked = bool(left_valid >= minimum_support and left_ratio >= 0.18)
    right_blocked = bool(right_valid >= minimum_support and right_ratio >= 0.18)
    return {
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "left_blocked": left_blocked,
        "right_blocked": right_blocked,
        "status": "ESTIMATED_FROM_CURRENT_ALIGNED_DEPTH_CORRIDORS",
    }


def _grasp_fractions(pose):
    values = (
        _preferred_grasp_fractions(pose)
        + _fallback_grasp_fractions(pose)
    )
    return _valid_grasp_fractions(values) or [0.5]


def _preferred_grasp_fractions(pose):
    return _valid_grasp_fractions(
        pose.get("preferred_grasp_body_fractions") or [0.45, 0.52, 0.60]
    )


def _fallback_grasp_fractions(pose):
    return _valid_grasp_fractions(
        pose.get("fallback_grasp_body_fractions") or [0.32, 0.38, 0.68, 0.74]
    )


def _valid_grasp_fractions(values):
    fractions = []
    for value in values:
        number = _optional_float(value)
        if number is not None and 0.22 <= number <= 0.78 and number not in fractions:
            fractions.append(number)
    return fractions


def _fraction_in(value, fractions, tolerance=1e-6):
    return any(abs(float(value) - float(candidate)) <= tolerance for candidate in fractions or [])


def _bbox_xyxy(pose):
    bbox = pose.get("bbox_xyxy") or pose.get("bottle_extent_bbox_xyxy")
    if bbox is None:
        xywh = pose.get("bbox_xywh")
        if xywh and len(xywh) >= 4:
            x, y, width, height = map(float, xywh[:4])
            bbox = [x, y, x + width, y + height]
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = map(float, bbox[:4])
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _distance_priority_score(depth, depths):
    depth = _optional_float(depth)
    depths = [float(value) for value in depths if value is not None]
    if depth is None or not depths:
        return 0.0
    low, high = min(depths), max(depths)
    if high - low <= 1e-6:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (depth - low) / (high - low))))


def _top_exposure_score(target_index, targets):
    pose = targets[target_index].get("pose") or {}
    bbox = _bbox_xyxy(pose)
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    center_y = 0.5 * (y1 + y2)
    depth = _optional_float(pose.get("depth_median_mm"))
    for index, neighbor in enumerate(targets):
        if index == target_index:
            continue
        neighbor_pose = neighbor.get("pose") or {}
        other = _bbox_xyxy(neighbor_pose)
        if other is None:
            continue
        nx1, ny1, nx2, ny2 = other
        neighbor_depth = _optional_float(neighbor_pose.get("depth_median_mm"))
        similar_depth = depth is None or neighbor_depth is None or neighbor_depth <= depth + 100.0
        horizontal_overlap = min(x2, nx2) > max(x1, nx1)
        if similar_depth and horizontal_overlap and 0.5 * (ny1 + ny2) < center_y:
            return 0.25
    return 1.0


def _selected_candidate(pregrasp_plan):
    if not pregrasp_plan:
        return None
    candidate = pregrasp_plan.get("selected_contact_pair") or pregrasp_plan.get("selected_candidate")
    return dict(candidate) if isinstance(candidate, Mapping) else None


def _reuse_epsilon_evidence(candidate, epsilon_result):
    if candidate is None or not _candidate_has_contact_geometry(candidate):
        return {"status": "NOT_APPLICABLE", "reason": "valid_contact_geometry_unavailable"}
    quality_applicability = str(
        (epsilon_result or {}).get("quality_applicability")
        or (epsilon_result or {}).get("applicability")
        or ""
    ).upper()
    if quality_applicability.startswith(("NOT_APPLICABLE", "NOT_TESTED")):
        return {
            "status": "NOT_APPLICABLE",
            "reason": (
                (epsilon_result or {}).get("quality_not_applicable_reason")
                or "current_valid_contact_geometry_unavailable"
            ),
            "planning_candidate_available": True,
            "physically_validated": False,
        }
    if not epsilon_result:
        embedded = candidate.get("epsilon_value")
        if embedded is None:
            return {"status": "NOT_TESTED", "reason": "epsilon_result_not_supplied"}
        return {
            "status": "ESTIMATED_PLANNING_ONLY_FROM_VISUAL_CANDIDATE",
            "epsilon_value": embedded,
            "epsilon_score": candidate.get("epsilon_score"),
            "force_closure": candidate.get("force_closure"),
            "recomputed": False,
            "applicable_to_live_contact_quality": False,
            "physically_validated": False,
        }
    return {
        "status": "ESTIMATED_FROM_CURRENT_VALID_CONTACT_GEOMETRY",
        "epsilon_value": epsilon_result.get("epsilon_value", epsilon_result.get("epsilon_closure")),
        "epsilon_score": epsilon_result.get("epsilon_score"),
        "force_closure": epsilon_result.get("force_closure"),
        "recomputed": False,
        "physically_validated": False,
    }


def _candidate_has_contact_geometry(candidate):
    left = candidate.get("left_contact_px") or candidate.get("left_visual_clamp_patch")
    right = candidate.get("right_contact_px") or candidate.get("right_visual_clamp_patch")
    return bool(left and right and candidate.get("contacts_on_body", True))


def _symbolic_force_profile(name):
    return {
        "name": name,
        "nominal": "SYMBOLIC",
        "measurement_status": "NOT_MEASURED_NO_GRIPPER",
        "units": "NOT_APPLICABLE",
    }


def _bounded(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number_list(value, cast):
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [cast(item) for item in value]
    except (TypeError, ValueError):
        return None

import numpy as np


MAX_CENTER_OFFSET_RATIO = 0.24
WARNING_CENTER_OFFSET_RATIO = 0.06
MIN_DETECTION_CONFIDENCE = 45.0
WARNING_DETECTION_CONFIDENCE = 70.0
MIN_DEPTH_CONFIDENCE = 45.0
MIN_TARGET_CONTACT_QUALITY = 45.0
MIN_ROTATION_SCORE_WARNING = 65.0
MIN_SLIP_SCORE_WARNING = 70.0
MIN_BALANCE_SCORE_WARNING = 65.0
MAX_AXIS_TILT_DEG = 55.0
GOOD_AXIS_TILT_DEG = 18.0
READY_DECISION = "READY TO CLOSE"
ADJUST_DECISION = "ADJUST POSITION"
BLOCK_DECISION = "DO NOT CLOSE"
LEGACY_READY_DECISION = "SAFE TO GRIP/LIFT"
LEGACY_BLOCK_DECISION = "DO NOT LIFT"


SCORE_WEIGHTS = {
    "bottle_detection_confidence": 0.20,
    "alignment_score": 0.15,
    "target_contact_quality": 0.15,
    "grip_width_reachability": 0.15,
    "rotation_score": 0.15,
    "slip_score": 0.10,
    "epsilon_score": 0.10,
}


def clamp_score(value):
    return float(np.clip(float(value or 0.0), 0.0, 100.0))


def score_from_abs_error(value, good_limit, fail_limit):
    value = abs(float(value))
    good_limit = max(float(good_limit), 0.0)
    fail_limit = max(float(fail_limit), good_limit + 1e-6)
    if value <= good_limit:
        return 100.0
    if value >= fail_limit:
        return 0.0
    return clamp_score(100.0 * (1.0 - (value - good_limit) / (fail_limit - good_limit)))


def epsilon_score(epsilon):
    if not epsilon.get("force_closure", False):
        return 0.0
    return clamp_score((float(epsilon.get("epsilon", 0.0) or 0.0) / 0.20) * 100.0)


def classify_axis_tilt(axis_angle_degrees):
    # Axis angle is measured from image +x. A standing bottle should be close to +/-90 deg.
    angle = float(axis_angle_degrees)
    vertical_error = abs(((angle - 90.0 + 90.0) % 180.0) - 90.0)
    return angle, float(vertical_error)


def metric_record(name, value, formula, inputs, units, thresholds, higher_is_better=True, hard_fail=False, passed=True, weight=0.0):
    score = clamp_score(value)
    return {
        "name": name,
        "value": score,
        "formula": formula,
        "inputs": inputs,
        "units": units,
        "range": [0.0, 100.0],
        "clamped_0_100": True,
        "higher_is_better": bool(higher_is_better),
        "thresholds": thresholds,
        "can_trigger_hard_fail": bool(hard_fail),
        "passed": bool(passed),
        "weight": float(weight),
        "weighted_contribution": float(score * weight),
    }


def build_metric_audit(pose, plan, epsilon, image_shape, values, gate_status):
    img_h, img_w = image_shape[:2]
    return [
        metric_record(
            "bottle_detection_confidence",
            values["detection_score"],
            "pose.detection_confidence * 100",
            {"detection_confidence": pose.get("detection_confidence")},
            "score",
            {"hard_fail_below": MIN_DETECTION_CONFIDENCE, "warning_below": WARNING_DETECTION_CONFIDENCE},
            hard_fail=True,
            passed=gate_status["mask_confidence"],
            weight=SCORE_WEIGHTS["bottle_detection_confidence"],
        ),
        metric_record(
            "alignment_score",
            values["alignment_score"],
            "100 inside good band; linearly decreases to 0 at fail band",
            {
                "alignment_error_px": values["alignment_error_px"],
                "good_limit_px": img_w * WARNING_CENTER_OFFSET_RATIO,
                "fail_limit_px": img_w * MAX_CENTER_OFFSET_RATIO,
            },
            "score",
            {"hard_fail_abs_error_px_above": img_w * MAX_CENTER_OFFSET_RATIO},
            hard_fail=True,
            passed=gate_status["pose_alignment"],
            weight=SCORE_WEIGHTS["alignment_score"],
        ),
        metric_record(
            "positioning_guidance_score",
            values["workspace_score"],
            "minimum score of left/right planned contact inside preferred camera-frame positioning zones",
            {"workspace_reachability_score": plan.get("workspace_reachability_score")},
            "score",
            {"guidance_only": True, "reposition_warning_below": 55.0},
            hard_fail=False,
            passed=gate_status["positioning_guidance"],
            weight=0.0,
        ),
        metric_record(
            "grip_width_score",
            values["raw_width_score"],
            "100 at middle of allowed grip width range; 0 outside range",
            {
                "grip_width_px": plan.get("grip_width_px"),
                "grip_width_mm": plan.get("grip_width_mm"),
                "allowed_mm": [30.0, 130.0],
                "allowed_px_when_mm_unknown": [35.0, 320.0],
            },
            "score",
            {"hard_fail_if_outside_allowed_range": True},
            hard_fail=True,
            passed=gate_status["grip_width"],
            weight=0.0,
        ),
        metric_record(
            "grip_width_reachability",
            values["grip_width_score"],
            "raw physical grip width feasibility score; fixed image workspace is guidance only",
            {
                "raw_grip_width_score": values["raw_width_score"],
                "positioning_guidance_score": values["workspace_score"],
            },
            "score",
            {"weighted_score_component": True, "workspace_guidance_only": True},
            hard_fail=False,
            passed=gate_status["grip_width"],
            weight=SCORE_WEIGHTS["grip_width_reachability"],
        ),
        metric_record(
            "grip_height_score",
            values["grip_height_score"],
            "score_from_abs_error(body_fraction - 0.50, good=0.12, fail=0.32)",
            {"body_fraction": plan.get("body_fraction")},
            "score",
            {"hard_fail_if_contacts_not_on_body": True},
            hard_fail=True,
            passed=gate_status["contacts_on_body"],
            weight=0.0,
        ),
        metric_record(
            "target_contact_quality",
            values["target_contact_quality"],
            "0.5 * left_depth_valid_ratio * 100 + 0.5 * right_depth_valid_ratio * 100",
            {
                "left_valid_ratio": plan.get("left_depth_stats", {}).get("valid_ratio"),
                "right_valid_ratio": plan.get("right_depth_stats", {}).get("valid_ratio"),
            },
            "score",
            {"hard_fail_below": MIN_TARGET_CONTACT_QUALITY},
            hard_fail=True,
            passed=gate_status["target_contact_quality"],
            weight=SCORE_WEIGHTS["target_contact_quality"],
        ),
        metric_record(
            "depth_confidence_score",
            values["depth_score"],
            "pose.depth_confidence * 100",
            {"depth_confidence": pose.get("depth_confidence")},
            "score",
            {"hard_fail_below": MIN_DEPTH_CONFIDENCE},
            hard_fail=True,
            passed=gate_status["depth_confidence"],
            weight=0.0,
        ),
        metric_record(
            "balance_score",
            values["balance_score"],
            "100 * (1 - distance(grasp_center, bottle_center) / grip_width_px)",
            {"left_right_balance_score": plan.get("left_right_balance_score")},
            "score",
            {"warning_below": MIN_BALANCE_SCORE_WARNING},
            hard_fail=False,
            passed=values["balance_score"] >= MIN_BALANCE_SCORE_WARNING,
            weight=0.0,
        ),
        metric_record(
            "rotation_score",
            values["rotation_score"],
            "0.45 * balance_score + 0.35 * epsilon_score + 0.20 * tilt_score",
            {
                "balance_score": values["balance_score"],
                "epsilon_score": values["epsilon_score"],
                "tilt_score": values["tilt_score"],
                "axis_vertical_error_deg": values["axis_vertical_error_deg"],
            },
            "score",
            {"warning_below": MIN_ROTATION_SCORE_WARNING, "axis_hard_fail_deg_above": MAX_AXIS_TILT_DEG},
            hard_fail=True,
            passed=gate_status["axis_pose"],
            weight=SCORE_WEIGHTS["rotation_score"],
        ),
        metric_record(
            "slip_score",
            values["slip_score"],
            "0.45 * grip_height_score + 0.35 * target_contact_quality + 0.20 * raw_width_score",
            {
                "grip_height_score": values["grip_height_score"],
                "target_contact_quality": values["target_contact_quality"],
                "raw_width_score": values["raw_width_score"],
            },
            "score",
            {"warning_below": MIN_SLIP_SCORE_WARNING},
            hard_fail=False,
            passed=values["slip_score"] >= MIN_SLIP_SCORE_WARNING,
            weight=SCORE_WEIGHTS["slip_score"],
        ),
        metric_record(
            "epsilon_score",
            values["epsilon_score"],
            "0 if force_closure false; else clamp((epsilon / 0.20) * 100)",
            {"epsilon": epsilon.get("epsilon"), "force_closure": epsilon.get("force_closure")},
            "score",
            {"hard_fail_if_force_closure_false": True},
            hard_fail=True,
            passed=gate_status["epsilon_force_closure"],
            weight=SCORE_WEIGHTS["epsilon_score"],
        ),
    ]


def evaluate_pregrasp_safety(pose, plan, epsilon, image_shape):
    img_h, img_w = image_shape[:2]
    failures = []
    reposition_reasons = []
    warnings = []

    if pose is None or not pose.get("detected"):
        failures.append("Bottle not detected.")
        return {
            "decision": BLOCK_DECISION,
            "pregrasp_decision": BLOCK_DECISION,
            "legacy_decision": LEGACY_BLOCK_DECISION,
            "final_score": 0.0,
            "raw_weighted_score": 0.0,
            "decision_score": 0.0,
            "capped_decision_score": 0.0,
            "cap_reason": "hard_gate_failure",
            "decision_stage": "pre_close",
            "failure_reasons": failures,
            "warnings": warnings,
            "reposition_reasons": [],
            "scores": {},
            "risk_scores": {},
            "metric_audit": [],
            "hard_gates": {"passed": False, "failed": failures},
            "decision_policy": {
                "hard_fail": BLOCK_DECISION,
                "reposition": ADJUST_DECISION,
                "ready_threshold": 80.0,
                "adjust_threshold": 60.0,
            },
        }
    if plan is None:
        failures.append("Planned target contacts could not be found.")
        return {
            "decision": BLOCK_DECISION,
            "pregrasp_decision": BLOCK_DECISION,
            "legacy_decision": LEGACY_BLOCK_DECISION,
            "final_score": 0.0,
            "raw_weighted_score": 0.0,
            "decision_score": 0.0,
            "capped_decision_score": 0.0,
            "cap_reason": "hard_gate_failure",
            "decision_stage": "pre_close",
            "failure_reasons": failures,
            "warnings": warnings,
            "reposition_reasons": [],
            "scores": {},
            "risk_scores": {},
            "metric_audit": [],
            "hard_gates": {"passed": False, "failed": failures},
            "decision_policy": {
                "hard_fail": BLOCK_DECISION,
                "reposition": ADJUST_DECISION,
                "ready_threshold": 80.0,
                "adjust_threshold": 60.0,
            },
        }

    alignment_error_px = float(pose["center_px"][0] - img_w / 2.0)
    detection_score = clamp_score(float(pose.get("detection_confidence", 0.0) or 0.0) * 100.0)
    alignment_score = score_from_abs_error(alignment_error_px, img_w * WARNING_CENTER_OFFSET_RATIO, img_w * MAX_CENTER_OFFSET_RATIO)
    grip_height_score = score_from_abs_error(float(plan.get("body_fraction", 0.5) or 0.5) - 0.50, 0.12, 0.32)
    target_contact_quality = clamp_score(plan.get("target_contact_quality", 0.0))
    raw_width_score = clamp_score(plan.get("raw_grip_width_score", plan.get("grip_width_score", 0.0)))
    workspace_score = clamp_score(plan.get("positioning_guidance_score", plan.get("workspace_reachability_score", 0.0)))
    grip_width_score = clamp_score(raw_width_score)
    balance_score = clamp_score(plan.get("left_right_balance_score", 0.0))
    depth_score = clamp_score(float(pose.get("depth_confidence", 0.0) or 0.0) * 100.0)
    eps_score = epsilon_score(epsilon)
    try:
        axis_angle = float(pose.get("axis_angle_degrees"))
    except (TypeError, ValueError):
        axis_angle = None
    axis_evidence_available = axis_angle is not None and bool(np.isfinite(axis_angle))
    if axis_evidence_available:
        _, axis_vertical_error = classify_axis_tilt(axis_angle)
        tilt_score = score_from_abs_error(axis_vertical_error, GOOD_AXIS_TILT_DEG, MAX_AXIS_TILT_DEG)
    else:
        axis_angle = None
        axis_vertical_error = None
        tilt_score = 0.0
    rotation_score = clamp_score(0.45 * balance_score + 0.35 * eps_score + 0.20 * tilt_score)
    slip_score = clamp_score(0.45 * grip_height_score + 0.35 * target_contact_quality + 0.20 * raw_width_score)
    miss_score = clamp_score(0.55 * alignment_score + 0.30 * grip_width_score + 0.15 * target_contact_quality)

    gate_status = {
        "mask_confidence": detection_score >= MIN_DETECTION_CONFIDENCE,
        "depth_confidence": depth_score >= MIN_DEPTH_CONFIDENCE,
        "full_bottle_visible": bool(plan.get("full_bottle_visible", False)),
        "contact_model_ready": bool(plan.get("contact_model_ready", False)),
        "planned_contacts": bool(plan.get("left_contact_px") is not None and plan.get("right_contact_px") is not None),
        "contacts_on_body": bool(plan.get("contacts_on_body")),
        "positioning_guidance": workspace_score >= 55.0,
        "workspace_guidance_only": bool(plan.get("workspace_guidance_only", True)),
        "grip_width": bool(plan.get("grip_width_in_range", False)),
        "both_sides_reachable_guidance": bool(plan.get("left_reachable") and plan.get("right_reachable")),
        "epsilon_force_closure": bool(epsilon.get("force_closure", False)),
        "pose_alignment": abs(alignment_error_px) <= img_w * MAX_CENTER_OFFSET_RATIO,
        "target_contact_quality": target_contact_quality >= MIN_TARGET_CONTACT_QUALITY,
        "axis_pose": axis_evidence_available and axis_vertical_error <= MAX_AXIS_TILT_DEG,
    }

    if not gate_status["mask_confidence"]:
        failures.append("Bottle/mask confidence is too low.")
    if not gate_status["depth_confidence"]:
        failures.append("Depth confidence is too low.")
    if not gate_status["full_bottle_visible"]:
        if plan.get("full_bottle_visible") is None:
            failures.append("Full-bottle visibility evidence is missing; close decision is disabled.")
        else:
            failures.append("Full bottle is not visible; contact modeling and close decision are disabled.")
    if not gate_status["contact_model_ready"]:
        if plan.get("contact_model_ready") is None:
            reasons = ["Contact-model readiness evidence is missing; close decision is disabled."]
        else:
            reasons = plan.get("contact_model_ready_reasons") or ["Contact model is not ready."]
        failures.extend(reasons)
    if not gate_status["planned_contacts"]:
        failures.append("Planned target contacts are missing.")
    if not gate_status["contacts_on_body"]:
        failures.append("Planned contacts are not on the bottle body.")
    if not gate_status["grip_width"]:
        failures.append("Grip width is outside the allowed range.")
    if not gate_status["epsilon_force_closure"]:
        failures.append("Epsilon force closure is false for the planned contacts.")
    if not gate_status["pose_alignment"]:
        failures.append("Bottle pose/alignment is too poor.")
    if not gate_status["target_contact_quality"]:
        failures.append("Planned target contact quality is too low.")
    if not gate_status["axis_pose"]:
        if not axis_evidence_available:
            failures.append("Bottle axis/pose evidence is missing or invalid; close decision is disabled.")
        else:
            failures.append("Bottle axis/pose is too tilted for this pre-grasp.")

    if detection_score < WARNING_DETECTION_CONFIDENCE:
        warnings.append("Bottle detection confidence is moderate.")
    if alignment_score < 75.0 and gate_status["pose_alignment"]:
        reposition_reasons.append("Bottle is not well centered between the grippers.")
    if workspace_score < 55.0:
        reposition_reasons.append("Planned contacts are outside the preferred image positioning guide; reposition the camera-gripper unit.")
    elif not gate_status["both_sides_reachable_guidance"]:
        reposition_reasons.append("One planned contact is outside the preferred image positioning guide; reposition the camera-gripper unit.")
    if rotation_score < MIN_ROTATION_SCORE_WARNING:
        warnings.append("Rotation-out risk is elevated.")
    if slip_score < MIN_SLIP_SCORE_WARNING:
        warnings.append("Slip-down risk is elevated.")
    if balance_score < MIN_BALANCE_SCORE_WARNING:
        warnings.append("Planned contacts are not balanced around the bottle center.")

    scores = {
        "bottle_detection_confidence": detection_score,
        "alignment": alignment_score,
        "alignment_score": alignment_score,
        "workspace_score": workspace_score,
        "positioning_guidance_score": workspace_score,
        "target_contact_quality": target_contact_quality,
        "grip_width_reachability": grip_width_score,
        "grip_width_score": raw_width_score,
        "rotation_stability": rotation_score,
        "rotation_score": rotation_score,
        "slip_risk_score": slip_score,
        "slip_score": slip_score,
        "miss_score": miss_score,
        "epsilon": eps_score,
        "epsilon_score": eps_score,
        "depth_confidence": depth_score,
        "grip_height": grip_height_score,
        "grip_height_score": grip_height_score,
        "left_right_balance": balance_score,
        "balance_score": balance_score,
        "axis_tilt_score": tilt_score,
    }
    risk_scores = {
        "miss_risk": clamp_score(100.0 - miss_score),
        "slip_risk": clamp_score(100.0 - slip_score),
        "rotation_risk": clamp_score(100.0 - rotation_score),
    }

    weighted_score = sum(scores[name] * weight for name, weight in SCORE_WEIGHTS.items())
    raw_weighted_score = clamp_score(weighted_score)
    final_score = raw_weighted_score
    values = {
        "detection_score": detection_score,
        "alignment_score": alignment_score,
        "alignment_error_px": alignment_error_px,
        "workspace_score": workspace_score,
        "raw_width_score": raw_width_score,
        "grip_width_score": grip_width_score,
        "grip_height_score": grip_height_score,
        "target_contact_quality": target_contact_quality,
        "depth_score": depth_score,
        "balance_score": balance_score,
        "rotation_score": rotation_score,
        "slip_score": slip_score,
        "miss_score": miss_score,
        "epsilon_score": eps_score,
        "tilt_score": tilt_score,
        "axis_vertical_error_deg": axis_vertical_error,
    }
    metric_audit = build_metric_audit(pose, plan, epsilon, image_shape, values, gate_status)

    cap_reason = "none"
    if failures:
        decision = BLOCK_DECISION
        cap_reason = "hard_gate_failure"
        final_score = min(raw_weighted_score, 49.0)
    elif reposition_reasons:
        decision = ADJUST_DECISION
        cap_reason = "repositioning_needed"
        final_score = min(raw_weighted_score, 79.0)
    elif final_score >= 80.0:
        decision = READY_DECISION
    elif final_score >= 60.0:
        decision = ADJUST_DECISION
    else:
        decision = BLOCK_DECISION

    legacy_decision = {
        READY_DECISION: LEGACY_READY_DECISION,
        ADJUST_DECISION: ADJUST_DECISION,
        BLOCK_DECISION: LEGACY_BLOCK_DECISION,
    }[decision]

    return {
        "decision": decision,
        "pregrasp_decision": decision,
        "legacy_decision": legacy_decision,
        "final_score": final_score,
        "raw_weighted_score": raw_weighted_score,
        "decision_score": final_score,
        "capped_decision_score": final_score,
        "cap_reason": cap_reason,
        "decision_stage": "pre_close",
        "failure_reasons": failures,
        "warnings": warnings,
        "reposition_reasons": reposition_reasons,
        "scores": scores,
        "risk_scores": risk_scores,
        "alignment_error_px": alignment_error_px,
        "axis_angle_degrees": axis_angle,
        "axis_vertical_error_degrees": axis_vertical_error,
        "axis_tilt_score": tilt_score,
        "metric_audit": metric_audit,
        "hard_gates": {"passed": not failures, "status": gate_status, "failed": failures},
        "true_hard_gate_reasons": failures,
        "contact_model_ready": gate_status["contact_model_ready"],
        "contact_model_ready_reasons": plan.get("contact_model_ready_reasons", []),
        "positioning_guidance_score": workspace_score,
        "workspace_guidance_only": True,
        "alignment_guidance": {
            "alignment_error_px": alignment_error_px,
            "alignment_score": alignment_score,
            "positioning_guidance_score": workspace_score,
            "workspace_both_reachable_guidance": bool(plan.get("workspace_both_reachable", False)),
            "left_in_preferred_zone": bool(plan.get("left_reachable", False)),
            "right_in_preferred_zone": bool(plan.get("right_reachable", False)),
        },
        "selected_preclose_decision": decision,
        "score_weights": SCORE_WEIGHTS,
        "decision_policy": {
            "hard_fail": BLOCK_DECISION,
            "reposition": ADJUST_DECISION,
            "ready_threshold": 80.0,
            "adjust_threshold": 60.0,
            "hard_gate_score_cap": 49.0,
            "reposition_score_cap": 79.0,
            "rule_order": [
                "hard gates",
                "reposition reasons",
                "weighted score thresholds",
            ],
        },
    }

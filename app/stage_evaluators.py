def _score(report):
    return float(report.get("final_score", report.get("safety_score", 0.0)) or 0.0)


def _active_action(report):
    active = report.get("active_positioning", {})
    return active.get("recommended_action") or report.get("decision") or "UNKNOWN"


def _bottle_detected(report):
    return bool(report.get("bottle_detected"))


def _failure_reasons(report):
    return list(report.get("failure_reasons", []))


def _warnings(report):
    return list(report.get("warnings", []))


def _epsilon_closed(report):
    return bool(report.get("epsilon_result", {}).get("force_closure", False))


def _depth_confidence(report):
    return float(report.get("scores", {}).get("depth_confidence", 0.0) or 0.0)


def _target_quality(report):
    if report.get("pair_quality_score") is not None:
        return float(report.get("pair_quality_score", 0.0) or 0.0)
    if report.get("visual_contact_quality_score") is not None:
        return float(report.get("visual_contact_quality_score", 0.0) or 0.0)
    return float(report.get("scores", {}).get("target_contact_quality", 0.0) or 0.0)


def _alignment_score(report):
    return float(report.get("scores", {}).get("alignment", 0.0) or 0.0)


def _grip_width_stable(report):
    contacts = report.get("planned_contacts", {})
    return bool(
        contacts.get("grip_width_in_range")
        and contacts.get("left_contact_px") is not None
        and contacts.get("right_contact_px") is not None
    )


def _full_bottle_visible(report):
    return bool(report.get("full_bottle_visible", False))


def _contact_model_ready(report):
    if "visual_contact_ready" in report:
        return bool(report.get("visual_contact_ready"))
    return bool(report.get("contact_model_ready", bool(report.get("planned_contacts"))))


def _base_result(stage, decision, action, report, reasons=None, warnings=None, extra=None):
    return {
        "stage": stage,
        "stage_decision": decision,
        "recommended_action": action,
        "safety_score": _score(report),
        "reasons": reasons or [],
        "warnings": warnings or [],
        "metrics": {
            "bottle_detected": _bottle_detected(report),
            "full_bottle_visible": _full_bottle_visible(report),
            "contact_model_ready": _contact_model_ready(report),
            "visual_contact_ready": bool(report.get("visual_contact_ready", False)),
            "contact_model_type": report.get("contact_model_type"),
            "no_force_sensor": bool(report.get("no_force_sensor", True)),
            "alignment_score": _alignment_score(report),
            "depth_confidence_score": _depth_confidence(report),
            "target_contact_quality": _target_quality(report),
            "grip_width_stable": _grip_width_stable(report),
            "epsilon_force_closure": _epsilon_closed(report),
        },
        "source_decision": report.get("decision"),
        "source_recommended_action": _active_action(report),
        "extra": extra or {},
    }


def evaluate_full_bottle_view_stage(report):
    if not _bottle_detected(report):
        return _base_result(
            "FULL_BOTTLE_VIEW_CHECK",
            "SEARCH",
            "SEARCH",
            report,
            ["Bottle not detected; acquire a view with the full known bottle in frame."],
            _warnings(report),
        )
    if not _full_bottle_visible(report):
        visibility = report.get("visibility", {})
        return _base_result(
            "FULL_BOTTLE_VIEW_CHECK",
            "ADJUST_VIEW",
            report.get("recommended_view_action", "ADJUST_VIEW"),
            report,
            ["Full bottle is not visible; contact modeling is disabled."] + list(visibility.get("missing_region_reason", [])),
            _warnings(report),
            {"visibility": visibility},
        )
    return _base_result(
        "FULL_BOTTLE_VIEW_CHECK",
        "VIEW_OK",
        "CONTACT_MODELING",
        report,
        ["Full bottle is visible with adequate border margins and depth support."],
        _warnings(report),
        {"visibility": report.get("visibility", {})},
    )


def evaluate_contact_modeling_stage(report):
    if not _full_bottle_visible(report):
        return evaluate_full_bottle_view_stage(report)
    if not _contact_model_ready(report):
        return _base_result(
            "CONTACT_MODELING",
            "CONTACT_MODEL_DISABLED",
            "ADJUST_VIEW",
            report,
            ["Contact model is not ready even though the bottle is visible."],
            _warnings(report),
            {"candidate_count": len(report.get("candidate_contacts", []))},
        )
    return _base_result(
        "CONTACT_MODELING",
        "CONTACT_MODEL_READY",
        "PRE_GRASP_EVALUATION",
        report,
        ["Candidate contact pairs were generated and a selected pair is available."],
        _warnings(report),
        {
            "candidate_count": len(report.get("candidate_contacts", [])),
            "selected_contact_pair": report.get("selected_contact_pair"),
        },
    )


def evaluate_pregrasp_stage(report):
    if not _bottle_detected(report):
        return _base_result(
            "PRE_GRASP_EVALUATION",
            "SEARCH",
            "SEARCH",
            report,
            ["Bottle not detected from the wider pre-grasp view."],
            _warnings(report),
        )
    if not _full_bottle_visible(report):
        return _base_result(
            "PRE_GRASP_EVALUATION",
            "ADJUST_VIEW",
            report.get("recommended_view_action", "ADJUST_VIEW"),
            report,
            ["Full bottle must be visible before pre-grasp safety scoring can authorize closing."],
            _warnings(report),
        )
    if not _contact_model_ready(report):
        return _base_result(
            "PRE_GRASP_EVALUATION",
            "CONTACT_MODEL_DISABLED",
            "ADJUST_VIEW",
            report,
            ["Contact modeling is not ready; improve view or depth confidence before closing."],
            _warnings(report),
        )

    failures = _failure_reasons(report)
    if failures:
        return _base_result(
            "PRE_GRASP_EVALUATION",
            "DO_NOT_ATTEMPT",
            "DO_NOT_ATTEMPT",
            report,
            failures,
            _warnings(report),
        )

    active = report.get("active_positioning", {})
    if active.get("state") == "POSITIONING" or report.get("reposition_reasons"):
        return _base_result(
            "PRE_GRASP_EVALUATION",
            "ADJUST_POSITION",
            active.get("recommended_action", "ADJUST POSITION"),
            report,
            report.get("reposition_reasons", []),
            _warnings(report),
        )

    return _base_result(
        "PRE_GRASP_EVALUATION",
        "READY_TO_APPROACH",
        "APPROACH",
        report,
        ["Bottle is detected and the current plan is good enough to approach; do not close yet."],
        _warnings(report),
    )


def evaluate_approach_stage(report):
    if not _bottle_detected(report):
        return _base_result(
            "APPROACH_EVALUATION",
            "BACK_UP",
            "BACK_UP",
            report,
            ["Bottle target was lost in close-up approach."],
            _warnings(report),
        )

    reasons = []
    if _alignment_score(report) < 70.0:
        reasons.append("Close-up alignment score is too low.")
    if _depth_confidence(report) < 60.0:
        reasons.append("Close-up depth confidence is too low.")
    if _target_quality(report) < 60.0:
        reasons.append("Planned target contact visibility/depth support is too low.")
    if not _grip_width_stable(report):
        reasons.append("Grip width estimate is not stable or outside configured limits.")
    if not _epsilon_closed(report):
        reasons.append("Epsilon force closure is false for the planned contacts.")

    failures = _failure_reasons(report)
    if failures:
        return _base_result(
            "APPROACH_EVALUATION",
            "DO_NOT_CLOSE",
            "DO_NOT_CLOSE",
            report,
            failures,
            _warnings(report),
        )
    if reasons:
        action = "BACK_UP" if _depth_confidence(report) < 60.0 else "ADJUST POSITION"
        decision = "BACK_UP" if action == "BACK_UP" else "ADJUST_POSITION"
        return _base_result("APPROACH_EVALUATION", decision, action, report, reasons, _warnings(report))

    return _base_result(
        "APPROACH_EVALUATION",
        "READY_TO_CLOSE",
        "READY_TO_CLOSE",
        report,
        ["Close-up planned grasp is aligned, visible, depth-supported, and epsilon-valid."],
        _warnings(report),
    )


def evaluate_postgrasp_stage(report, pre_close_report=None):
    reasons = []
    warnings = _warnings(report)

    if not _bottle_detected(report):
        return _base_result(
            "POST_GRASP_EVALUATION",
            "DO_NOT_LIFT",
            "DO_NOT_LIFT",
            report,
            ["Bottle is not detected after closing."],
            warnings,
        )

    if pre_close_report:
        before = pre_close_report.get("bottle_geometry", {}).get("center_px")
        after = report.get("bottle_geometry", {}).get("center_px")
        if before and after:
            dx = float(after[0] - before[0])
            dy = float(after[1] - before[1])
            displacement = (dx * dx + dy * dy) ** 0.5
            if displacement > 45.0:
                reasons.append(f"Bottle shifted {displacement:.1f} px after closing.")

    if _depth_confidence(report) < 60.0:
        reasons.append("Post-close depth consistency is too low.")
    if _target_quality(report) < 60.0:
        reasons.append("Post-close contact depth evidence is too weak.")
    if report.get("axis_vertical_error_degrees", 0.0) and float(report.get("axis_vertical_error_degrees", 0.0)) > 55.0:
        reasons.append("Bottle tilt/rotation is too high after closing.")
    if not _epsilon_closed(report):
        reasons.append("Post-close epsilon force closure is false.")

    if _failure_reasons(report):
        reasons.extend(_failure_reasons(report))

    if reasons:
        return _base_result(
            "POST_GRASP_EVALUATION",
            "REGRIP",
            "REGRIP",
            report,
            reasons,
            warnings,
            {"requires_physical_contact_validation": True},
        )

    return _base_result(
        "POST_GRASP_EVALUATION",
        "READY_TO_LIFT",
        "READY_TO_LIFT",
        report,
        ["Post-close RGB-D checks are acceptable for a lift decision."],
        warnings,
        {"requires_physical_contact_validation": True},
    )


def evaluate_lift_monitor_stage(report):
    return _base_result(
        "LIFT_MONITOR",
        "MONITOR_ONLY",
        "MONITOR",
        report,
        ["Lift monitoring is reserved for future continuous tracking during lift."],
        _warnings(report),
    )

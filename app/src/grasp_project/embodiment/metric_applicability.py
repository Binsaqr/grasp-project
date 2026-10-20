"""Metric applicability rules by grasping embodiment."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.base import canonical_embodiment_id

APPLICABLE = "APPLICABLE"
PARTIALLY_APPLICABLE = "PARTIALLY_APPLICABLE"
EXPERIMENTAL = "EXPERIMENTAL"
NOT_APPLICABLE = "NOT_APPLICABLE"
UNAVAILABLE = "UNAVAILABLE"

METRICS = [
    "two_contact_epsilon",
    "generalized_wrench_closure",
    "grip_width",
    "opposition",
    "support_polygon",
    "support_triangle",
    "contact_coverage",
    "wrap_coverage",
    "suction_seal",
    "surface_flatness",
    "normal_alignment",
    "3d_surface_support",
    "davis_slip",
    "davis_rotation",
    "davis_vibration",
    "stability",
    "contact_count_validity",
]


def metric_applicability(embodiment_id: str | None) -> dict[str, Any]:
    emb = canonical_embodiment_id(embodiment_id)
    rules = {metric: {"status": UNAVAILABLE, "reason": "metric_not_configured"} for metric in METRICS}
    primary = "contact_count_validity"
    secondary: list[str] = []
    hidden: list[str] = []
    if emb == "parallel_jaw_two_finger":
        primary = "two_contact_epsilon"
        secondary = ["grip_width", "opposition", "contact_coverage", "3d_surface_support", "davis_slip", "stability"]
        _mark(rules, ["two_contact_epsilon", "grip_width", "opposition", "contact_coverage", "3d_surface_support", "davis_slip", "davis_rotation", "davis_vibration", "stability", "contact_count_validity"], APPLICABLE, "parallel_jaw_reference_metric")
        _mark(rules, ["generalized_wrench_closure"], EXPERIMENTAL, "not_validated_for_live_pipeline")
        hidden = ["suction_seal", "wrap_coverage", "support_triangle"]
    elif emb == "articulated_two_finger":
        primary = "opposition"
        secondary = ["contact_coverage", "3d_surface_support", "davis_slip", "stability"]
        _mark(rules, ["opposition", "contact_coverage", "3d_surface_support", "davis_slip", "davis_rotation", "davis_vibration", "stability", "contact_count_validity"], APPLICABLE, "articulated_two_finger_contact_metric")
        _mark(rules, ["two_contact_epsilon", "grip_width"], PARTIALLY_APPLICABLE, "fingers_may_not_be_parallel")
        hidden = ["suction_seal", "wrap_coverage", "support_triangle"]
    elif emb == "three_finger_gripper":
        primary = "support_triangle"
        secondary = ["contact_coverage", "3d_surface_support", "davis_slip", "stability"]
        _mark(rules, ["support_triangle", "contact_coverage", "3d_surface_support", "davis_slip", "davis_rotation", "davis_vibration", "stability", "contact_count_validity"], APPLICABLE, "three_contact_support_metric")
        _mark(rules, ["two_contact_epsilon"], NOT_APPLICABLE, "two_contact_epsilon_requires_exactly_two_opposing_contacts")
        _mark(rules, ["generalized_wrench_closure"], EXPERIMENTAL, "multi_contact_wrench_metric_not_validated")
        hidden = ["grip_width", "suction_seal", "wrap_coverage"]
    elif emb in {"multi_finger_robotic_hand", "human_hand"}:
        primary = "support_polygon"
        secondary = ["contact_coverage", "3d_surface_support", "davis_slip", "davis_rotation", "stability"]
        _mark(rules, ["support_polygon", "contact_coverage", "3d_surface_support", "davis_slip", "davis_rotation", "davis_vibration", "stability", "contact_count_validity"], APPLICABLE, "variable_multi_contact_metric")
        _mark(rules, ["two_contact_epsilon", "grip_width"], NOT_APPLICABLE, "variable_contact_hand_not_parallel_two_point_grasp")
        hidden = ["suction_seal", "support_triangle"]
    elif emb == "vacuum_suction":
        primary = "suction_seal"
        secondary = ["surface_flatness", "normal_alignment", "3d_surface_support", "davis_slip", "stability"]
        _mark(rules, ["suction_seal", "surface_flatness", "normal_alignment", "3d_surface_support", "davis_slip", "davis_rotation", "davis_vibration", "stability", "contact_count_validity"], APPLICABLE, "suction_area_metric")
        _mark(rules, ["two_contact_epsilon", "grip_width", "opposition"], NOT_APPLICABLE, "suction_is_area_contact_not_two_opposing_contacts")
        hidden = ["two_contact_epsilon", "grip_width", "opposition", "support_triangle", "wrap_coverage"]
    elif emb == "soft_gripper":
        primary = "wrap_coverage"
        secondary = ["contact_coverage", "3d_surface_support", "davis_slip", "stability"]
        _mark(rules, ["wrap_coverage", "contact_coverage", "3d_surface_support", "davis_slip", "davis_rotation", "davis_vibration", "stability", "contact_count_validity"], APPLICABLE, "distributed_wrap_metric")
        _mark(rules, ["two_contact_epsilon", "grip_width", "opposition"], NOT_APPLICABLE, "soft_gripper_uses_distributed_contact_not_two_opposing_contacts")
        hidden = ["two_contact_epsilon", "grip_width", "opposition", "suction_seal", "support_triangle"]
    else:
        _mark(rules, ["contact_count_validity"], UNAVAILABLE, "unknown_embodiment")
        hidden = ["two_contact_epsilon", "grip_width", "suction_seal", "wrap_coverage", "support_triangle"]
    for metric in hidden:
        rules.setdefault(metric, {"status": NOT_APPLICABLE, "reason": "hidden_for_this_embodiment"})
    return {
        "embodiment_id": emb,
        "primary_metric": primary,
        "secondary_metrics": secondary,
        "hidden_metrics": sorted(set(hidden)),
        "metric_applicability": rules,
        "metric_applicability_reasons": {metric: data["reason"] for metric, data in rules.items()},
    }


def epsilon_applicability(embodiment_id: str | None) -> dict[str, str]:
    data = metric_applicability(embodiment_id)
    item = data["metric_applicability"].get("two_contact_epsilon", {"status": UNAVAILABLE, "reason": "missing"})
    return {"epsilon_applicability": item["status"], "reason": item["reason"]}


def _mark(rules: dict[str, dict[str, str]], metrics: list[str], status: str, reason: str) -> None:
    for metric in metrics:
        rules[metric] = {"status": status, "reason": reason}

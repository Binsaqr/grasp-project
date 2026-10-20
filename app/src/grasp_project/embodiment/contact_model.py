"""Metric applicability rules for embodiment-aware contact interpretation."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.base import canonical_embodiment_id
from grasp_project.embodiment.gripper_model import load_active_embodiment
from grasp_project.embodiment.metric_applicability import metric_applicability
from grasp_project.embodiment.registry import legacy_embodiment_id


def quality_metric_applicability(embodiment_type: str | None = None) -> dict[str, Any]:
    model = load_active_embodiment()
    if embodiment_type:
        if embodiment_type != model.embodiment_type:
            from grasp_project.embodiment.gripper_model import DEFAULT_MODELS

            canonical_candidate = canonical_embodiment_id(embodiment_type)
            legacy_candidate = legacy_embodiment_id(canonical_candidate)
            model = (
                DEFAULT_MODELS.get(embodiment_type)
                or DEFAULT_MODELS.get(legacy_candidate)
                or DEFAULT_MODELS.get(canonical_candidate)
                or DEFAULT_MODELS["unknown"]
            )
    canonical = canonical_embodiment_id(model.embodiment_type)
    applicability = metric_applicability(canonical)
    epsilon_status = applicability["metric_applicability"].get("two_contact_epsilon", {})
    if canonical == "parallel_jaw_two_finger":
        return {
            "quality_metric_applicability": "epsilon_primary",
            "primary_quality_metric": "epsilon_force_closure",
            "secondary_quality_metrics": ["visual_contact", "depth_consistency", "grip_width"],
            "quality_not_applicable_reason": None,
            "canonical_embodiment_id": canonical,
            "metric_applicability": applicability["metric_applicability"],
            "hidden_metrics": applicability["hidden_metrics"],
            "epsilon_applicability": epsilon_status.get("status"),
        }
    return {
        "quality_metric_applicability": "epsilon_not_primary",
        "primary_quality_metric": applicability["primary_metric"] or (model.allowed_quality_metrics[0] if model.allowed_quality_metrics else "not_configured"),
        "secondary_quality_metrics": applicability["secondary_metrics"] or model.allowed_quality_metrics[1:],
        "quality_not_applicable_reason": f"Two-contact epsilon is not the primary metric for {model.embodiment_type}.",
        "canonical_embodiment_id": canonical,
        "metric_applicability": applicability["metric_applicability"],
        "hidden_metrics": applicability["hidden_metrics"],
        "epsilon_applicability": epsilon_status.get("status"),
    }


def contact_source_status(contact_source: str, contact_points=None, contact_patches=None, contact_hull=None) -> dict[str, Any]:
    model = load_active_embodiment()
    points = list(contact_points or [])
    patches = list(contact_patches or [])
    expected = model.expected_contact_count
    if isinstance(expected, int):
        valid_count = len(points) >= expected or len(patches) >= expected
    else:
        valid_count = bool(points or patches or contact_hull)
    return {
        "contact_source": contact_source,
        "contact_source_confidence": 100.0 if valid_count else 0.0,
        "contact_source_reason": "matches_expected_contact_model" if valid_count else "waiting_for_expected_contacts",
        "embodiment_type": model.embodiment_type,
        "contact_points": points,
        "contact_patches": patches,
        "contact_hull": contact_hull,
        "contact_model_valid": bool(valid_count and model.embodiment_type != "unknown"),
        **quality_metric_applicability(model.embodiment_type),
    }

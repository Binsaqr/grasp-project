"""Shared helpers for embodiment plugins."""

from __future__ import annotations

from typing import Any

import numpy as np

from grasp_project.embodiment.base import EmbodimentQualityResult
from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.metric_applicability import metric_applicability
from grasp_project.embodiment.registry import load_registry


def valid_contacts(contact_set: ContactSet) -> list[ContactObservation]:
    return [contact for contact in contact_set.contacts if contact.valid]


def score_from_contacts(contact_set: ContactSet, expected_min: int = 1) -> float:
    contacts = valid_contacts(contact_set)
    count_score = min(1.0, len(contacts) / max(float(expected_min), 1.0))
    confidences = [float(contact.confidence or 0.0) for contact in contacts]
    confidence_score = float(np.mean(confidences) / 100.0) if confidences and max(confidences) > 1.0 else float(np.mean(confidences) if confidences else 0.0)
    return round(float(np.clip(100.0 * (0.45 * count_score + 0.55 * confidence_score), 0.0, 100.0)), 2)


def label(score: float | None, invalid: bool = False) -> str:
    if invalid or score is None:
        return "INVALID"
    if score >= 75:
        return "GOOD"
    if score >= 50:
        return "OK"
    if score > 0:
        return "WEAK"
    return "INVALID"


def make_result(
    embodiment_id: str,
    contact_set: ContactSet,
    *,
    score: float | None,
    metric_name: str | None = None,
    decision: str | None = None,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    explanation: str = "",
) -> EmbodimentQualityResult:
    definition = load_registry().get(embodiment_id)
    applicability = metric_applicability(embodiment_id)
    invalid = bool(blockers or not contact_set.contact_set_valid)
    metric = metric_name or definition.primary_quality_metric
    return EmbodimentQualityResult(
        embodiment_id=definition.embodiment_id,
        contact_model_type=",".join(definition.contact_geometry_types),
        expected_contacts=definition.expected_contact_count_min if definition.expected_contact_count_min == definition.expected_contact_count_max else f"{definition.expected_contact_count_min}+",
        detected_contacts=contact_set.detected_contact_count,
        contact_set_valid=contact_set.contact_set_valid,
        primary_metric_name=metric,
        primary_metric_score=score,
        primary_metric_label=label(score, invalid=invalid),
        secondary_scores={name: None for name in definition.secondary_quality_metrics},
        analytic_quality_score=score,
        live_operational_quality=score,
        metric_applicability=applicability,
        hard_blockers=list(blockers or contact_set.blockers),
        warnings=list(warnings or []),
        confidence=0.0 if score is None else float(np.clip(score / 100.0, 0.0, 1.0)),
        decision=decision or (definition.engagement_action if not invalid else "ACQUIRE_CONTACT_MODEL"),
        explanation=explanation,
        limitations=definition.limitations,
    )


def contact_set_from_observations(embodiment_id: str, contacts: list[ContactObservation], expected: int | str | None) -> ContactSet:
    contact_set = ContactSet(embodiment_id=embodiment_id, expected_contact_count=expected, contacts=list(contacts))
    contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=expected if isinstance(expected, int) else 1)
    return contact_set

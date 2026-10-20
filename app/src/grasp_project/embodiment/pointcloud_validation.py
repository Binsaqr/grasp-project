"""Embodiment-aware point-cloud contact validation.

This module validates point-cloud contact evidence without changing the
RealSense point-cloud worker. It uses the universal ContactSet and the selected
embodiment's contact-count rules instead of assuming two contacts.
"""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.base import canonical_embodiment_id
from grasp_project.embodiment.contact_primitives import ContactSet
from grasp_project.embodiment.registry import load_registry


CURRENT_STATES = {"CURRENT", "FRESH", "ACTIVE", "OK"}
STALE_STATES = {"STALE", "AGING"}
UNAVAILABLE_STATES = {"", "UNAVAILABLE", "MISSING", "ERROR", "NOT_READY"}


def validate_pointcloud_contact_set(
    contact_set: ContactSet,
    pointcloud_status: dict[str, Any] | None = None,
    embodiment_id: str | None = None,
) -> dict[str, Any]:
    """Validate whether fresh point-cloud evidence supports a ContactSet."""

    pointcloud_status = pointcloud_status or {}
    canonical = canonical_embodiment_id(embodiment_id or contact_set.embodiment_id)
    definition = load_registry().get(canonical)
    freshness = _freshness(pointcloud_status)
    detected = int(contact_set.detected_contact_count)
    required = _required_contact_count(definition.expected_contact_count_min)
    max_expected = definition.expected_contact_count_max
    contact_3d_valid = _bool_or_none(pointcloud_status.get("contact_3d_valid"))
    off_surface = _bool_or_none(pointcloud_status.get("contact_points_on_bottle_cloud")) is False
    cloud_valid = _bool_or_none(pointcloud_status.get("cloud_valid_for_contact_validation"))
    reason = []

    if freshness in STALE_STATES:
        return _payload(canonical, "STALE_EXCLUDED", contact_set, required, max_expected, freshness, False, "stale_pointcloud_excluded")
    if freshness in UNAVAILABLE_STATES:
        return _payload(canonical, "UNAVAILABLE", contact_set, required, max_expected, freshness, False, "pointcloud_unavailable")
    if cloud_valid is False:
        reason.append("pointcloud_not_trusted_for_contact_validation")
    if detected < required:
        reason.append("missing_required_pointcloud_contacts")
    if contact_3d_valid is False:
        reason.append("contact_3d_invalid")
    if off_surface:
        reason.append("contacts_not_on_bottle_cloud")
    if not contact_set.contact_set_valid:
        reason.extend(contact_set.blockers or ["contact_set_invalid"])

    valid = not reason
    return _payload(
        canonical,
        "CURRENT_VALID" if valid else "CURRENT_INVALID",
        contact_set,
        required,
        max_expected,
        freshness,
        valid,
        "valid_3d_contact_support" if valid else ",".join(_unique(reason)),
    )


def _payload(
    embodiment_id: str,
    status: str,
    contact_set: ContactSet,
    required: int,
    max_expected: int | None,
    freshness: str,
    valid: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "embodiment_id": embodiment_id,
        "pointcloud_contact_validation_status": status,
        "pointcloud_contact_validation_applicable": status not in {"UNAVAILABLE"},
        "pointcloud_contact_set_valid": bool(valid),
        "pointcloud_contact_reason": reason,
        "required_contact_count": required,
        "expected_contact_count_max": max_expected,
        "detected_contact_count": contact_set.detected_contact_count,
        "freshness": freshness or "UNAVAILABLE",
        "contact_3d_valid": bool(valid),
        "contact_set_valid": contact_set.contact_set_valid,
        "contact_set_confidence": contact_set.contact_set_confidence,
    }


def _freshness(pointcloud_status: dict[str, Any]) -> str:
    raw = (
        pointcloud_status.get("freshness")
        or pointcloud_status.get("pointcloud_freshness")
        or pointcloud_status.get("status")
        or ""
    )
    text = str(raw).strip().upper()
    if text == "CURRENT":
        return "CURRENT"
    if text in CURRENT_STATES:
        return text
    if text in STALE_STATES:
        return text
    if text in UNAVAILABLE_STATES:
        return "UNAVAILABLE" if text == "" else text
    return text


def _required_contact_count(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, number)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "valid", "current"}:
        return True
    if text in {"0", "false", "no", "invalid", "stale"}:
        return False
    return None


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["validate_pointcloud_contact_set"]

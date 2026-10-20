"""Compatibility helpers from legacy gripper code to embodiment plugins."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.plugin_interface import compute_quality_for_evidence, load_builtin_plugin


def contact_set_from_gripper_evidence(embodiment_id: str | None, evidence: dict[str, Any] | None = None):
    """Return a universal ContactSet for legacy gripper evidence."""

    loaded = load_builtin_plugin(embodiment_id)
    if not loaded.ok or loaded.plugin is None:
        loaded = load_builtin_plugin("custom")
    contact_set = loaded.plugin.estimate_contacts(evidence or {})
    return contact_set


def quality_from_gripper_evidence(embodiment_id: str | None, evidence: dict[str, Any] | None = None):
    """Return an embodiment-specific quality result for gripper evidence."""

    return compute_quality_for_evidence(embodiment_id, evidence or {})


__all__ = ["contact_set_from_gripper_evidence", "quality_from_gripper_evidence"]

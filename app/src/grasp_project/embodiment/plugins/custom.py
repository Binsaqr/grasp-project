"""Custom gripper plugin fallback.

This plugin is intentionally conservative. It accepts configured contact
observations for display and benchmarking experiments, but it does not make
production grasp decisions.
"""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.plugins.common import contact_set_from_observations, make_result, score_from_contacts, valid_contacts


class CustomPlugin:
    """Generic configurable gripper model.

    A project-specific plugin can replace this class through
    ``config/custom_gripper_plugin.yaml``. Until that happens, custom mode is
    display/config support only.
    """

    embodiment_id = "custom"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        contacts_payload = (
            evidence.get("contacts")
            or evidence.get("distributed_contacts")
            or evidence.get("contact_observations")
            or []
        )
        if isinstance(contacts_payload, dict):
            contacts_payload = list(contacts_payload.values())
        contacts = [_contact(i, item, self.embodiment_id) for i, item in enumerate(contacts_payload)]
        expected = evidence.get("expected_contact_count") or evidence.get("expected_contacts") or "configured"
        return self.validate_contact_set(contact_set_from_observations(self.embodiment_id, contacts, expected))

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        contacts = valid_contacts(contact_set)
        if not contacts:
            contact_set.blockers.append("missing_custom_contacts")
        contact_set.contact_set_valid = bool(contacts) and not contact_set.blockers
        contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=1)
        contact_set.limitations.extend([
            "custom contact model is configuration/display support only by default",
            "custom gripper must pass embodiment-specific release gates before production use",
        ])
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        return make_result(
            self.embodiment_id,
            contact_set,
            score=score_from_contacts(contact_set, expected_min=1),
            metric_name="contact_count_validity",
            decision="DISPLAY_ONLY" if contact_set.contact_set_valid else "ACQUIRE_CONTACT_MODEL",
            warnings=[
                "custom_plugin_display_only",
                "production_validation_required",
                "no_force_or_pressure_claims",
            ],
            explanation="Custom gripper contact validity is configurable and not production-ready by default.",
        )


def _contact(index: int, item: dict[str, Any], embodiment_id: str) -> ContactObservation:
    item = item or {}
    return ContactObservation(
        contact_id=str(item.get("contact_id") or f"custom_contact_{index + 1}"),
        embodiment_id=embodiment_id,
        source=str(item.get("source") or "configured_contact"),
        geometry_type=str(item.get("geometry_type") or "patch"),
        associated_finger_or_element=item.get("associated_finger_or_element") or item.get("element_id"),
        image_polygon_px=item.get("image_polygon_px") or item.get("polygon") or [],
        image_center_px=item.get("image_center_px") or item.get("center"),
        pointcloud_points_xyz=item.get("pointcloud_points_xyz") or [],
        center_xyz_mm=item.get("center_xyz_mm"),
        local_normal=item.get("local_normal"),
        area_px=item.get("area_px"),
        area_mm2=item.get("area_mm2"),
        depth_valid_ratio=item.get("depth_valid_ratio"),
        visibility=item.get("visibility"),
        confidence=float(item.get("confidence") or item.get("quality_score") or 0.0),
        temporal_stability=item.get("temporal_stability"),
        observation_count=int(item.get("observation_count") or 1),
        on_object_surface=item.get("on_object_surface"),
        occluded=bool(item.get("occluded", False)),
        valid=bool(item.get("valid", True) and (item.get("image_center_px") or item.get("center") or item.get("image_polygon_px") or item.get("polygon") or item.get("center_xyz_mm"))),
        validation_reasons=list(item.get("validation_reasons") or ["custom_contact_observation"]),
    )

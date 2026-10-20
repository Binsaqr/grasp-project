"""Soft-gripper plugin for distributed contact and wrap evidence."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.plugins.common import contact_set_from_observations, make_result, score_from_contacts, valid_contacts


class SoftGripperPlugin:
    embodiment_id = "soft_gripper"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        contacts = []
        for i, item in enumerate(evidence.get("distributed_contacts") or evidence.get("contacts") or []):
            contacts.append(
                ContactObservation(
                    contact_id=str(item.get("contact_id") or f"soft_region_{i + 1}"),
                    embodiment_id=self.embodiment_id,
                    source=str(item.get("source") or "visual_contact"),
                    geometry_type=str(item.get("geometry_type") or "distributed_patch"),
                    associated_finger_or_element=item.get("associated_finger_or_element") or f"lobe_{i + 1}",
                    image_polygon_px=item.get("image_polygon_px") or item.get("polygon") or [],
                    image_center_px=item.get("image_center_px") or item.get("center"),
                    area_px=item.get("area_px"),
                    confidence=float(item.get("confidence") or item.get("quality_score") or 0.0),
                    valid=bool(item.get("valid", True)),
                    validation_reasons=list(item.get("validation_reasons") or ["soft_distributed_contact"]),
                )
            )
        return self.validate_contact_set(contact_set_from_observations(self.embodiment_id, contacts, "distributed"))

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        contacts = valid_contacts(contact_set)
        if not contacts:
            contact_set.blockers.append("missing_distributed_contact")
        total_area = sum(float(c.area_px or 0.0) for c in contacts)
        contact_set.contact_coverage = min(1.0, total_area / 1000.0) if total_area > 0 else min(1.0, len(contacts) / 3.0)
        if contact_set.contact_coverage < 0.25:
            contact_set.blockers.append("insufficient_wrap_coverage")
        contact_set.contact_set_valid = bool(contacts) and not contact_set.blockers
        contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=1)
        contact_set.limitations.append("distributed contact is visual/depth support, not tactile pressure")
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        score = score_from_contacts(contact_set, expected_min=1)
        if contact_set.contact_coverage is not None:
            score = round(0.6 * score + 40.0 * contact_set.contact_coverage, 2)
        return make_result(
            self.embodiment_id,
            contact_set,
            score=score,
            metric_name="wrap_coverage",
            decision="READY_TO_WRAP" if contact_set.contact_set_valid else "ACQUIRE_CONTACT_MODEL",
            warnings=["two_contact_epsilon_not_applicable"],
            explanation="Soft-gripper quality uses distributed contact and wrap coverage.",
        )

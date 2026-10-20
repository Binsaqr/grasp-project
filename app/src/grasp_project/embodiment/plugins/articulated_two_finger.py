"""Articulated two-finger plugin."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.plugins.common import contact_set_from_observations, make_result, score_from_contacts, valid_contacts


class ArticulatedTwoFingerPlugin:
    embodiment_id = "articulated_two_finger"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        contacts = [
            _contact(i, item)
            for i, item in enumerate(evidence.get("contacts") or [])
        ]
        return self.validate_contact_set(contact_set_from_observations(self.embodiment_id, contacts, 2))

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        contacts = valid_contacts(contact_set)
        if len(contacts) < 2:
            contact_set.blockers.append("need_two_articulated_finger_contacts")
        contact_set.contact_set_valid = len(contacts) >= 2 and not contact_set.blockers
        contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=2)
        contact_set.limitations.append("contact angles are visual estimates only")
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        return make_result(
            self.embodiment_id,
            contact_set,
            score=score_from_contacts(contact_set, expected_min=2),
            metric_name="opposition",
            decision="READY_TO_ENGAGE" if contact_set.contact_set_valid else "WAIT_FOR_CONTACTS",
            explanation="Articulated two-finger model allows non-parallel fingertip or link contacts.",
        )


def _contact(index: int, item: dict[str, Any]) -> ContactObservation:
    return ContactObservation(
        contact_id=str(item.get("contact_id") or f"articulated_contact_{index + 1}"),
        embodiment_id="articulated_two_finger",
        source=str(item.get("source") or "visual_contact"),
        geometry_type=str(item.get("geometry_type") or "fingertip"),
        associated_finger_or_element=item.get("associated_finger_or_element") or f"finger_{index + 1}",
        image_polygon_px=item.get("image_polygon_px") or [],
        image_center_px=item.get("image_center_px"),
        confidence=float(item.get("confidence") or 0.0),
        valid=bool(item.get("valid", True)),
        validation_reasons=list(item.get("validation_reasons") or ["articulated_two_finger_observation"]),
    )

"""Three-finger gripper plugin."""

from __future__ import annotations

from typing import Any

import numpy as np

from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.plugins.common import contact_set_from_observations, make_result, score_from_contacts, valid_contacts


class ThreeFingerPlugin:
    embodiment_id = "three_finger_gripper"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        contacts = [_contact(i, item) for i, item in enumerate(evidence.get("contacts") or [])]
        return self.validate_contact_set(contact_set_from_observations(self.embodiment_id, contacts, 3))

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        contacts = valid_contacts(contact_set)
        if len(contacts) < 3:
            contact_set.blockers.append("missing_three_finger_contact")
            contact_set.missing_contacts = [f"finger_{i}" for i in range(len(contacts) + 1, 4)]
        contact_set.contact_spread = _triangle_area(contacts)
        if len(contacts) >= 3 and (contact_set.contact_spread or 0.0) < 1.0:
            contact_set.blockers.append("poor_triangular_coverage")
        contact_set.contact_set_valid = len(contacts) >= 3 and not contact_set.blockers
        contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=3)
        contact_set.limitations.append("current two-contact epsilon is not applicable to this three-contact set")
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        score = score_from_contacts(contact_set, expected_min=3)
        if contact_set.contact_spread is not None:
            score = min(100.0, score * 0.7 + min(30.0, contact_set.contact_spread / 10.0))
        return make_result(
            self.embodiment_id,
            contact_set,
            score=round(score, 2),
            metric_name="support_triangle",
            decision="READY_TO_ENGAGE" if contact_set.contact_set_valid else "WAIT_FOR_CONTACTS",
            warnings=["two_contact_epsilon_not_applied"],
            explanation="Three-finger quality uses support-triangle/contact-coverage evidence, not two-contact epsilon.",
        )


def _contact(index: int, item: dict[str, Any]) -> ContactObservation:
    return ContactObservation(
        contact_id=str(item.get("contact_id") or f"finger_{index + 1}"),
        embodiment_id="three_finger_gripper",
        source=str(item.get("source") or "visual_contact"),
        geometry_type=str(item.get("geometry_type") or "patch"),
        associated_finger_or_element=f"finger_{index + 1}",
        image_polygon_px=item.get("image_polygon_px") or [],
        image_center_px=item.get("image_center_px"),
        confidence=float(item.get("confidence") or 0.0),
        valid=bool(item.get("valid", True)),
        validation_reasons=list(item.get("validation_reasons") or ["three_finger_observation"]),
    )


def _triangle_area(contacts: list[ContactObservation]) -> float | None:
    centers = [c.image_center_px for c in contacts[:3] if c.image_center_px]
    if len(centers) < 3:
        return None
    arr = np.asarray(centers, dtype=float)
    # scalar 2D cross product; np.cross no longer accepts 2D vectors on NumPy >= 2.0
    u = arr[1] - arr[0]
    v = arr[2] - arr[0]
    return float(abs(u[0] * v[1] - u[1] * v[0]) * 0.5)

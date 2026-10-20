"""Parallel two-finger gripper plugin: the current regression reference."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.contact_primitives import ContactSet, contact_from_legacy_patch
from grasp_project.embodiment.plugins.common import contact_set_from_observations, make_result, score_from_contacts, valid_contacts


class ParallelJawPlugin:
    embodiment_id = "parallel_jaw_two_finger"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        left = evidence.get("left_contact") or evidence.get("left_clamp_patch") or {}
        right = evidence.get("right_contact") or evidence.get("right_clamp_patch") or {}
        contacts = [
            contact_from_legacy_patch("left_contact", self.embodiment_id, left, associated_finger_or_element="left_inner_jaw"),
            contact_from_legacy_patch("right_contact", self.embodiment_id, right, associated_finger_or_element="right_inner_jaw"),
        ]
        return self.validate_contact_set(contact_set_from_observations(self.embodiment_id, contacts, 2))

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        contacts = valid_contacts(contact_set)
        if len(contacts) < 2:
            contact_set.blockers.append("missing_left_or_right_contact")
            contact_set.missing_contacts = ["left_contact", "right_contact"][len(contacts):]
        contact_set.opposition = _opposition_score(contact_set)
        if contact_set.opposition is not None and contact_set.opposition < 0.35:
            contact_set.blockers.append("contacts_not_opposed")
        contact_set.contact_set_valid = len(contacts) == 2 and not contact_set.blockers
        contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=2)
        contact_set.limitations.append("visual/depth contacts are not force or pressure measurements")
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        score = score_from_contacts(contact_set, expected_min=2)
        return make_result(
            self.embodiment_id,
            contact_set,
            score=score,
            metric_name="two_contact_epsilon",
            decision="READY_TO_CLOSE" if contact_set.contact_set_valid else "WAIT_FOR_CONTACTS",
            explanation="Reference parallel-jaw model using two opposing contact patches.",
        )


def _opposition_score(contact_set: ContactSet) -> float | None:
    left = contact_set.left_contact
    right = contact_set.right_contact
    if not left or not right or not left.image_center_px or not right.image_center_px:
        return None
    dx = abs(float(right.image_center_px[0]) - float(left.image_center_px[0]))
    dy = abs(float(right.image_center_px[1]) - float(left.image_center_px[1]))
    if dx + dy <= 1e-6:
        return 0.0
    return float(max(0.0, min(1.0, dx / (dx + dy))))

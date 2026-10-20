"""Human-hand observation plugin."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.contact_primitives import ContactSet
from grasp_project.embodiment.plugins.common import make_result
from grasp_project.embodiment.plugins.multi_finger_hand import MultiFingerHandPlugin


class HumanHandPlugin(MultiFingerHandPlugin):
    embodiment_id = "human_hand"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        contact_set = super().estimate_contacts(evidence)
        contact_set.embodiment_id = self.embodiment_id
        for contact in contact_set.contacts:
            contact.embodiment_id = self.embodiment_id
        contact_set.limitations.append("human-hand mode is research/observation only and must not control the robot")
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        return make_result(
            self.embodiment_id,
            contact_set,
            score=contact_set.contact_set_confidence,
            metric_name="support_polygon",
            decision="HOLD" if contact_set.contact_set_valid else "ACQUIRE_CONTACT_MODEL",
            warnings=["research_observation_mode", "no_force_claims"],
            explanation="Human-hand mode reports observation quality only.",
        )

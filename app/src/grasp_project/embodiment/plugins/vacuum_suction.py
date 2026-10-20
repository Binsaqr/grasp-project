"""Vacuum/suction gripper plugin."""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.plugins.common import contact_set_from_observations, make_result, score_from_contacts, valid_contacts


class VacuumSuctionPlugin:
    embodiment_id = "vacuum_suction"

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        patch = evidence.get("suction_area") or evidence.get("contact_patch") or {}
        contact = ContactObservation(
            contact_id="suction_area",
            embodiment_id=self.embodiment_id,
            source=str(patch.get("source") or "visual_contact"),
            geometry_type="suction_area",
            associated_finger_or_element="suction_cup",
            image_polygon_px=patch.get("image_polygon_px") or patch.get("polygon") or [],
            image_center_px=patch.get("image_center_px") or patch.get("center"),
            center_xyz_mm=patch.get("center_xyz_mm"),
            local_normal=patch.get("local_normal"),
            area_px=patch.get("area_px"),
            area_mm2=patch.get("area_mm2"),
            depth_valid_ratio=patch.get("depth_valid_ratio"),
            confidence=float(patch.get("confidence") or patch.get("quality_score") or 0.0),
            valid=bool(patch.get("valid", True) and (patch.get("image_polygon_px") or patch.get("polygon") or patch.get("image_center_px") or patch.get("center"))),
            validation_reasons=list(patch.get("validation_reasons") or ["suction_area_visual_geometry"]),
        )
        return self.validate_contact_set(contact_set_from_observations(self.embodiment_id, [contact], 1))

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        contacts = valid_contacts(contact_set)
        if len(contacts) < 1:
            contact_set.blockers.append("missing_suction_area")
        contact = contacts[0] if contacts else None
        if contact and contact.depth_valid_ratio is not None and contact.depth_valid_ratio < 0.6:
            contact_set.blockers.append("missing_depth_for_seal_area")
        if contact and contact.area_px is not None and contact.area_px < 10:
            contact_set.blockers.append("suction_patch_too_small")
        contact_set.contact_set_valid = len(contacts) == 1 and not contact_set.blockers
        contact_set.contact_set_confidence = score_from_contacts(contact_set, expected_min=1)
        contact_set.limitations.append("visual/depth seal suitability is not suction pressure")
        return contact_set

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None):
        return make_result(
            self.embodiment_id,
            contact_set,
            score=score_from_contacts(contact_set, expected_min=1),
            metric_name="suction_seal",
            decision="READY_TO_APPLY_SUCTION" if contact_set.contact_set_valid else "ACQUIRE_CONTACT_MODEL",
            warnings=["two_contact_epsilon_hidden", "no_suction_pressure_measurement"],
            explanation="Vacuum quality uses seal area, flatness, normal alignment, and geometric support.",
        )

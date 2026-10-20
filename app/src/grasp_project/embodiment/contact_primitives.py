"""Universal contact primitives for embodiment-aware grasp perception."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CONTACT_GEOMETRY_TYPES = {
    "point",
    "line",
    "patch",
    "distributed_patch",
    "suction_area",
    "wrap_region",
    "fingertip",
    "finger_link",
    "palm_region",
}

CONTACT_SOURCES = {
    "planned_contact",
    "visual_contact",
    "depth_supported_contact",
    "pointcloud_supported_contact",
    "detected_gripper_contact",
    "configured_contact",
    "unavailable",
}


@dataclass
class ContactObservation:
    contact_id: str
    embodiment_id: str
    source: str
    geometry_type: str
    associated_finger_or_element: str | None = None
    image_polygon_px: list[list[float]] = field(default_factory=list)
    image_center_px: list[float] | None = None
    pointcloud_points_xyz: list[list[float]] = field(default_factory=list)
    center_xyz_mm: list[float] | None = None
    local_normal: list[float] | None = None
    tangent_basis: list[list[float]] = field(default_factory=list)
    area_px: float | None = None
    area_mm2: float | None = None
    depth_valid_ratio: float | None = None
    visibility: float | None = None
    confidence: float = 0.0
    temporal_stability: float | None = None
    observation_count: int = 1
    on_object_surface: bool | None = None
    occluded: bool = False
    valid: bool = False
    validation_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.geometry_type not in CONTACT_GEOMETRY_TYPES:
            self.valid = False
            self.validation_reasons.append(f"unsupported_geometry_type:{self.geometry_type}")
        if self.source not in CONTACT_SOURCES:
            self.source = "unavailable"
            self.valid = False
            self.validation_reasons.append("unsupported_contact_source")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContactSet:
    embodiment_id: str
    expected_contact_count: int | str | None
    contacts: list[ContactObservation] = field(default_factory=list)
    contact_hull_2d: list[list[float]] = field(default_factory=list)
    contact_hull_3d: list[list[float]] = field(default_factory=list)
    support_polygon: list[list[float]] = field(default_factory=list)
    support_volume: Any = None
    contact_coverage: float | None = None
    contact_spread: float | None = None
    symmetry: float | None = None
    opposition: float | None = None
    contact_set_confidence: float = 0.0
    contact_set_valid: bool = False
    missing_contacts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @property
    def detected_contact_count(self) -> int:
        return len([contact for contact in self.contacts if contact.valid])

    @property
    def left_contact(self) -> ContactObservation | None:
        for contact in self.contacts:
            label = str(contact.associated_finger_or_element or contact.contact_id).lower()
            if "left" in label:
                return contact
        return self.contacts[0] if len(self.contacts) >= 1 else None

    @property
    def right_contact(self) -> ContactObservation | None:
        for contact in self.contacts:
            label = str(contact.associated_finger_or_element or contact.contact_id).lower()
            if "right" in label:
                return contact
        return self.contacts[1] if len(self.contacts) >= 2 else None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contacts"] = [contact.to_dict() for contact in self.contacts]
        payload["detected_contact_count"] = self.detected_contact_count
        payload["left_contact"] = None if self.left_contact is None else self.left_contact.to_dict()
        payload["right_contact"] = None if self.right_contact is None else self.right_contact.to_dict()
        return payload


def contact_from_legacy_patch(
    contact_id: str,
    embodiment_id: str,
    patch: dict[str, Any] | None,
    *,
    source: str = "planned_contact",
    associated_finger_or_element: str | None = None,
) -> ContactObservation:
    """Adapt a current left/right patch dict into a generic contact observation."""

    patch = patch or {}
    polygon = patch.get("polygon") or patch.get("patch_polygon_px") or []
    center = patch.get("center") or patch.get("center_px") or patch.get("contact_px")
    confidence = float(patch.get("quality_score") or patch.get("confidence") or 0.0)
    depth_ratio = patch.get("valid_depth_ratio")
    return ContactObservation(
        contact_id=contact_id,
        embodiment_id=embodiment_id,
        source=source,
        geometry_type="patch",
        associated_finger_or_element=associated_finger_or_element,
        image_polygon_px=_points(polygon),
        image_center_px=None if center is None else [float(center[0]), float(center[1])],
        area_px=_polygon_area(polygon),
        depth_valid_ratio=None if depth_ratio is None else float(depth_ratio),
        confidence=confidence,
        valid=bool(polygon or center),
        validation_reasons=["legacy_parallel_patch_adapter"] if (polygon or center) else ["missing_legacy_patch_geometry"],
    )


def _points(points) -> list[list[float]]:
    try:
        return [[float(p[0]), float(p[1])] for p in points]
    except Exception:
        return []


def _polygon_area(points) -> float | None:
    pts = _points(points)
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

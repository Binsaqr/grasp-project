"""Base types for embodiment-aware grasp perception.

The classes here are data/adapter contracts only. They do not change the
existing RealSense tracker, epsilon math, DAVIS scoring, or robot commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


EMBODIMENT_IDS = [
    "parallel_jaw_two_finger",
    "articulated_two_finger",
    "three_finger_gripper",
    "multi_finger_robotic_hand",
    "human_hand",
    "vacuum_suction",
    "soft_gripper",
    "custom",
    "unknown",
]

LEGACY_EMBODIMENT_ALIASES = {
    "two_point_parallel_gripper": "parallel_jaw_two_finger",
    "parallel_jaw": "parallel_jaw_two_finger",
    "three_finger_grip": "three_finger_gripper",
    "vacuum_grip": "vacuum_suction",
}


def canonical_embodiment_id(embodiment_id: str | None) -> str:
    """Return a known canonical embodiment id, falling back to ``unknown``."""

    key = str(embodiment_id or "unknown").strip()
    key = LEGACY_EMBODIMENT_ALIASES.get(key, key)
    return key if key in EMBODIMENT_IDS else "unknown"


@dataclass
class EmbodimentDefinition:
    embodiment_id: str
    display_name: str
    category: str
    expected_contact_count_min: int
    expected_contact_count_max: int | None
    contact_geometry_types: list[str] = field(default_factory=list)
    required_perception_features: list[str] = field(default_factory=list)
    optional_perception_features: list[str] = field(default_factory=list)
    supported_quality_metrics: list[str] = field(default_factory=list)
    primary_quality_metric: str = "not_configured"
    secondary_quality_metrics: list[str] = field(default_factory=list)
    unsupported_metrics: list[str] = field(default_factory=list)
    readiness_rules: list[str] = field(default_factory=list)
    safety_rules: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    release_status: str = "CONFIG_ONLY"
    stable_status: str = "experimental"
    engagement_action: str = "READY_TO_ENGAGE"
    legacy_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbodimentSelectionResult:
    embodiment_id: str
    embodiment_source: str = "unknown"
    embodiment_confidence: float = 0.0
    embodiment_locked: bool = False
    embodiment_lock_timestamp: float | None = None
    embodiment_detection_reasons: list[str] = field(default_factory=list)
    competing_embodiments: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_embodiment: bool = False
    state: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["embodiment_id"] = canonical_embodiment_id(payload["embodiment_id"])
        return payload


@dataclass
class EmbodimentQualityResult:
    embodiment_id: str
    contact_model_type: str
    expected_contacts: int | str | None
    detected_contacts: int
    contact_set_valid: bool
    primary_metric_name: str
    primary_metric_score: float | None
    primary_metric_label: str
    secondary_scores: dict[str, Any] = field(default_factory=dict)
    analytic_quality_score: float | None = None
    live_operational_quality: float | None = None
    metric_applicability: dict[str, Any] = field(default_factory=dict)
    hard_blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    decision: str = "ACQUIRE_CONTACT_MODEL"
    explanation: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbodimentDetector(Protocol):
    def detect_embodiment(self, evidence: dict[str, Any]) -> EmbodimentSelectionResult:
        ...


class ContactModelPlugin(Protocol):
    embodiment_id: str

    def estimate_contacts(self, evidence: dict[str, Any]):
        ...

    def validate_contact_set(self, contact_set):
        ...


class QualityMetricPlugin(Protocol):
    embodiment_id: str

    def compute_quality(self, contact_set, evidence: dict[str, Any] | None = None) -> EmbodimentQualityResult:
        ...

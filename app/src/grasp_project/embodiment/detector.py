"""Conservative embodiment selection and locking helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from grasp_project.common.config import config_path, load_config_file
from grasp_project.embodiment.base import EmbodimentSelectionResult, canonical_embodiment_id
from grasp_project.embodiment.registry import active_embodiment_id, load_registry

CONFIGURED = "CONFIGURED"
DETECTED = "DETECTED"
AMBIGUOUS = "AMBIGUOUS"
UNKNOWN = "UNKNOWN"
OCCLUDED = "OCCLUDED"
UNSUPPORTED = "UNSUPPORTED"


@dataclass
class EmbodimentLockState:
    embodiment_id: str = "unknown"
    locked: bool = False
    lock_timestamp: float | None = None
    lock_source: str = "unknown"

    def lock(self, selection: EmbodimentSelectionResult) -> EmbodimentSelectionResult:
        self.embodiment_id = canonical_embodiment_id(selection.embodiment_id)
        self.locked = True
        self.lock_timestamp = time.time()
        self.lock_source = selection.embodiment_source
        selection.embodiment_id = self.embodiment_id
        selection.embodiment_locked = True
        selection.embodiment_lock_timestamp = self.lock_timestamp
        return selection

    def reset(self) -> None:
        self.embodiment_id = "unknown"
        self.locked = False
        self.lock_timestamp = None
        self.lock_source = "unknown"


def select_embodiment(evidence: dict[str, Any] | None = None, config: dict[str, Any] | None = None, lock_state: EmbodimentLockState | None = None) -> EmbodimentSelectionResult:
    """Select an embodiment without switching during an active grasp."""

    evidence = evidence or {}
    config = config or load_config_file(config_path("embodiment.yaml"), default={}) or {}
    if lock_state and lock_state.locked and config.get("lock_embodiment_during_grasp", True):
        return EmbodimentSelectionResult(
            embodiment_id=lock_state.embodiment_id,
            embodiment_source=lock_state.lock_source,
            embodiment_confidence=1.0,
            embodiment_locked=True,
            embodiment_lock_timestamp=lock_state.lock_timestamp,
            embodiment_detection_reasons=["embodiment_locked_during_active_grasp"],
            state=CONFIGURED,
        )
    mode = str(config.get("selection_mode") or "configured")
    configured = canonical_embodiment_id(config.get("configured_embodiment") or config.get("active_embodiment"))
    if mode == "configured" and configured != "unknown":
        return EmbodimentSelectionResult(
            embodiment_id=configured,
            embodiment_source="configured",
            embodiment_confidence=1.0,
            embodiment_detection_reasons=["configured_embodiment_selected"],
            state=CONFIGURED,
        )
    detected = detect_embodiment_from_evidence(evidence, minimum_confidence=float(config.get("minimum_detection_confidence", 0.75)))
    if detected.state == DETECTED and config.get("require_user_confirmation", False):
        detected.embodiment_source = "automatically_detected_waiting_for_user_confirmation"
        detected.state = AMBIGUOUS
        detected.ambiguous_embodiment = True
    if detected.state == DETECTED and config.get("allow_automatic_detection", False):
        return detected
    if configured != "unknown":
        return EmbodimentSelectionResult(
            embodiment_id=configured,
            embodiment_source="configured_fallback",
            embodiment_confidence=0.85,
            embodiment_detection_reasons=["automatic_detection_disabled_or_uncertain"],
            competing_embodiments=[detected.to_dict()],
            state=CONFIGURED,
        )
    return detected


def detect_embodiment_from_evidence(evidence: dict[str, Any] | None = None, minimum_confidence: float = 0.75) -> EmbodimentSelectionResult:
    """Conservatively classify only persistent, consistent embodiment evidence."""

    evidence = evidence or {}
    registry = load_registry()
    reasons = []
    competing = []
    if evidence.get("occluded"):
        return EmbodimentSelectionResult("unknown", "automatically_detected", 0.0, state=OCCLUDED, ambiguous_embodiment=True, embodiment_detection_reasons=["embodiment_occluded"])
    if evidence.get("configured_hardware_embodiment"):
        emb = canonical_embodiment_id(evidence.get("configured_hardware_embodiment"))
        return EmbodimentSelectionResult(emb, "hardware_metadata", 0.9, state=DETECTED, embodiment_detection_reasons=["configured_hardware_metadata"])
    persistence = int(evidence.get("temporal_persistence_frames") or 0)
    if persistence < 3:
        return EmbodimentSelectionResult("unknown", "automatically_detected", 0.0, state=UNKNOWN, embodiment_detection_reasons=["insufficient_temporal_persistence"])
    visible = int(evidence.get("visible_finger_like_elements") or evidence.get("detected_contact_count") or 0)
    symmetric = bool(evidence.get("parallel_jaw_symmetry") or evidence.get("opposing_parallel_edges"))
    suction = bool(evidence.get("circular_suction_area") or evidence.get("configured_suction_footprint"))
    soft = bool(evidence.get("broad_deformable_regions") or evidence.get("wrap_region_visible"))
    hand = bool(evidence.get("hand_like_multi_link_geometry"))
    if suction:
        competing.append({"embodiment_id": "vacuum_suction", "confidence": 0.82})
        reasons.append("suction_area_like_geometry")
    if soft:
        competing.append({"embodiment_id": "soft_gripper", "confidence": 0.78})
        reasons.append("broad_wrap_or_deformable_region")
    if hand:
        competing.append({"embodiment_id": "human_hand", "confidence": 0.65})
        reasons.append("hand_like_geometry_not_robot_gripper_by_default")
    if visible == 2 and symmetric:
        competing.append({"embodiment_id": "parallel_jaw_two_finger", "confidence": 0.80})
        reasons.append("two_symmetric_parallel_finger_like_elements")
    elif visible == 2:
        competing.append({"embodiment_id": "articulated_two_finger", "confidence": 0.62})
        reasons.append("two_finger_like_elements_without_parallel_consistency")
    elif visible == 3:
        competing.append({"embodiment_id": "three_finger_gripper", "confidence": 0.78})
        reasons.append("three_finger_like_elements")
    elif visible > 3:
        competing.append({"embodiment_id": "multi_finger_robotic_hand", "confidence": 0.58})
        reasons.append("variable_multi_contact_count")
    if not competing:
        return EmbodimentSelectionResult("unknown", "automatically_detected", 0.0, state=UNKNOWN, embodiment_detection_reasons=["no_conservative_embodiment_evidence"])
    competing.sort(key=lambda item: item["confidence"], reverse=True)
    best = competing[0]
    ambiguous = len(competing) > 1 and (best["confidence"] - competing[1]["confidence"]) < 0.15
    if ambiguous or best["confidence"] < minimum_confidence:
        return EmbodimentSelectionResult(
            best["embodiment_id"],
            "automatically_detected",
            best["confidence"],
            state=AMBIGUOUS,
            ambiguous_embodiment=True,
            embodiment_detection_reasons=reasons + ["confidence_below_threshold_or_conflicting_interpretation"],
            competing_embodiments=competing,
        )
    if registry.get(best["embodiment_id"]).release_status == "NOT_IMPLEMENTED":
        state = UNSUPPORTED
    else:
        state = DETECTED
    return EmbodimentSelectionResult(
        best["embodiment_id"],
        "automatically_detected",
        best["confidence"],
        state=state,
        embodiment_detection_reasons=reasons,
        competing_embodiments=competing,
    )


def current_selection_status(evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    selection = select_embodiment(evidence=evidence)
    definition = load_registry().get(selection.embodiment_id)
    return {**selection.to_dict(), "definition": definition.to_dict(), "active_embodiment": active_embodiment_id()}

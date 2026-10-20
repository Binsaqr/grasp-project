"""Simulation-only slip/rotation force adjustment policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


GLOBAL_DISTURBANCES = {"LIGHTING_CHANGE", "LIGHTING CHANGE", "CAMERA_SHAKE", "CAMERA SHAKE", "BACKGROUND_MOTION", "BACKGROUND MOTION"}


@dataclass(frozen=True)
class ForceCommand:
    grip_force_delta: float
    torque_delta: float
    reason: str
    confidence: float
    source_cues: dict[str, Any]
    physically_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SlipForceController:
    """Emit normalized symbolic adjustments; never sends actuator commands."""

    def __init__(self, *, precursor_threshold: float = 35.0, gross_slip_threshold: float = 35.0, rotation_threshold: float = 25.0):
        self.precursor_threshold = float(precursor_threshold)
        self.gross_slip_threshold = float(gross_slip_threshold)
        self.rotation_threshold = float(rotation_threshold)

    def update(self, boundary_field, object_state, grasp_plan, motion_report=None) -> ForceCommand:
        field = _mapping(boundary_field)
        state = _mapping(object_state)
        plan = _mapping(grasp_plan)
        motion = _mapping(motion_report)
        precursor = float(field.get("incipient_slip_precursor_score") or 0.0)
        slip = float(field.get("slip_cue") or 0.0)
        rotation = float(field.get("rotation_cue") or 0.0)
        disturbance = str(motion.get("raw_motion_class") or motion.get("disturbance_type") or "").upper()
        confidence = _confidence(field, motion)
        sources = {
            "incipient_slip_precursor_score": precursor,
            "slip_cue": slip,
            "rotation_cue": rotation,
            "object_motion_valid": field.get("object_motion_valid"),
            "davis_motion_usable_for_grasp": motion.get("davis_motion_usable_for_grasp"),
            "current_stage": motion.get("current_stage"),
            "disturbance": disturbance or None,
            "measurement_status": field.get("measurement_status"),
        }
        if field.get("object_motion_valid") is not True:
            return ForceCommand(0.0, 0.0, "NO_ACTION_OBJECT_MOTION_INVALID", 0.0, sources)
        if (
            motion.get("davis_motion_usable_for_grasp") is False
            or str(motion.get("current_stage") or "").upper() == "PRE_CLOSE"
        ):
            return ForceCommand(0.0, 0.0, "NO_ACTION_NOT_POST_CLOSE_USABLE", 0.0, sources)
        if disturbance in GLOBAL_DISTURBANCES:
            return ForceCommand(0.0, 0.0, "NO_ACTION_GLOBAL_DISTURBANCE", 0.0, sources)
        if str(plan.get("strategy") or "") == "DO_NOT_ATTEMPT":
            return ForceCommand(0.0, 0.0, "NO_ACTION_GRASP_PLAN_BLOCKED", confidence, sources)

        grip_delta = 0.0
        reasons = []
        tilt = abs(float(state.get("tilt_angle_deg") or 0.0))
        if precursor >= self.precursor_threshold:
            grip_delta = max(grip_delta, (0.05 + precursor * 0.002) * confidence)
            reasons.append("ESTIMATED_INCIPIENT_SLIP_CUE_PREEMPTIVE_POLICY")
        if slip >= self.gross_slip_threshold:
            grip_delta = max(grip_delta, (0.15 + slip * 0.004 + min(tilt, 90.0) / 900.0) * confidence)
            reasons.append("ESTIMATED_GROSS_SLIP_CUE_POLICY_ESCALATION")

        torque_delta = 0.0
        if rotation >= self.rotation_threshold:
            rotation_delta = float(motion.get("rotation_delta_deg") or field.get("tangential_circulation_px") or 0.0)
            direction = 1.0 if rotation_delta >= 0.0 else -1.0
            if bool(plan.get("allow_rotation")) and tilt > 5.0:
                torque_delta = min(1.0, (tilt / 90.0) * (rotation / 100.0) * confidence)
                reasons.append("REGULATE_ALLOWED_ROTATION_TOWARD_UPRIGHT")
            else:
                torque_delta = -direction * min(1.0, rotation / 100.0 * confidence)
                reasons.append("SUPPRESS_UNWANTED_ROTATION" if not plan.get("allow_rotation") else "UPRIGHT_ROTATION_LOCK")

        if not reasons:
            reasons.append("NO_ADJUSTMENT_BELOW_POLICY_THRESHOLDS")
        return ForceCommand(
            grip_force_delta=round(grip_delta, 4),
            torque_delta=round(torque_delta, 4),
            reason=";".join(reasons),
            confidence=round(confidence, 3),
            source_cues=sources,
        )


def simulate_force_policy(
    event_patterns: Sequence[Mapping[str, Any]],
    object_states,
    grasp_plans,
    motion_reports=None,
    *,
    controller: SlipForceController | None = None,
) -> list[ForceCommand]:
    """Run a deterministic sequence for camera-free policy testing."""
    controller = controller or SlipForceController()
    commands = []
    for index, pattern in enumerate(event_patterns):
        state = _sequence_value(object_states, index)
        plan = _sequence_value(grasp_plans, index)
        motion = {} if motion_reports is None else _sequence_value(motion_reports, index)
        commands.append(controller.update(pattern, state, plan, motion))
    return commands


def _mapping(value):
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def _sequence_value(value, index):
    if isinstance(value, (list, tuple)):
        return value[min(index, len(value) - 1)]
    return value


def _confidence(field, motion):
    supplied = motion.get("motion_estimation_confidence")
    if supplied is None:
        supplied = motion.get("motion_confidence")
    if supplied is None:
        support_count = field.get("assigned_event_count")
        if support_count is None:
            support_count = field.get("event_count")
        supplied = min(1.0, float(support_count or 0.0) / 50.0)
    return max(0.0, min(1.0, float(supplied)))

"""Live operational grasp-quality monitor.

The monitor keeps two values separate:

* analytic quality: immutable quality from the locked RealSense epsilon/wrench
  computation
* live quality: operational confidence after current RGB-D, point-cloud, DAVIS,
  and synchronization evidence applies penalties or hard gates

This module is pure Python and camera-free. It reads dictionaries produced by
the existing status pipeline and never changes baseline epsilon math.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from grasp_project.common.config import OUTPUT_DIR, load_named_config
from grasp_project.common.file_io import append_jsonl_bounded, read_json, write_json_atomic
from grasp_project.embodiment.base import canonical_embodiment_id
from grasp_project.embodiment.registry import load_registry


LIVE_QUALITY_TIMELINE_MAX_BYTES = max(
    1024 * 1024,
    int(os.environ.get("LIVE_QUALITY_TIMELINE_MAX_BYTES", str(16 * 1024 * 1024))),
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "score_scale": 100.0,
    "weights": {
        "alignment": 12.0,
        "angle": 10.0,
        "depth": 10.0,
        "grip_width": 8.0,
        "contact": 20.0,
        "band_drift": 18.0,
        "band_visibility": 15.0,
        "pointcloud": 12.0,
        "incipient_slip": 12.0,
        "gross_slip": 35.0,
        "rotation": 18.0,
        "vibration": 12.0,
        "target_loss": 100.0,
        "uncertainty": 10.0,
    },
    "thresholds": {
        "alignment_px": 24.0,
        "angle_deg": 18.0,
        "depth_mm": 35.0,
        "grip_width_mm": 18.0,
        "band_drift_px": 24.0,
        "band_visibility_ratio": 0.50,
        "frame_age_ms": 120.0,
        "sync_delta_sec": 0.15,
        "pointcloud_stale_sec": 2.0,
        "incipient_slip_score": 70.0,
        "gross_slip_score": 65.0,
        "severe_rotation_score": 70.0,
        "vibration_score": 75.0,
        "davis_incipient_slip_px": 4.0,
        "davis_gross_slip_px": 10.0,
        "davis_rotation_deg": 8.0,
        "davis_vibration_rms_px": 5.0,
        "davis_motion_confidence": 0.55,
    },
    "hard_gates": {
        "target_lost_score_cap": 5.0,
        "gross_slip_score_cap": 12.0,
        "missing_contacts_score_cap": 18.0,
        "severe_rotation_score_cap": 20.0,
        "invalid_baseline_score_cap": 0.0,
        "off_surface_score_cap": 15.0,
        "band_relock_score_cap": 15.0,
        "sync_score_cap": 25.0,
    },
    "smoothing": {
        "alpha_normal": 0.30,
        "alpha_fast_drop": 0.75,
        "alpha_recovery": 0.18,
        "fast_drop_threshold": 12.0,
        "trend_deadband": 2.0,
        "severe_recovery_frames": 6,
    },
    "required_contacts": 2,
}


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_live_quality_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load live-quality config with defaults for missing fields."""
    if path is not None:
        loaded = read_json(path, default={}) if str(path).endswith(".json") else {}
        if not loaded and path.exists():
            try:
                import yaml  # type: ignore

                loaded = yaml.safe_load(path.read_text()) or {}
            except Exception:
                loaded = {}
    else:
        loaded = load_named_config("live_quality.yaml", default={})
    return _deep_merge(DEFAULT_CONFIG, loaded if isinstance(loaded, Mapping) else {})


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _norm(value: Optional[float], threshold: float) -> float:
    if value is None or threshold <= 0:
        return 0.0
    return _clamp(abs(value) / threshold, 0.0, 1.0)


def _score_label(score: Optional[float], hard_gates: Iterable[str] = ()) -> str:
    if score is None:
        return "INVALID"
    gates = set(hard_gates)
    if "invalid_baseline" in gates:
        return "INVALID"
    if gates:
        return "BLOCKED"
    if score >= 75:
        return "GOOD"
    if score >= 50:
        return "OK"
    if score >= 25:
        return "WEAK"
    return "RISK"


def _baseline_score_from_matrix(matrix: Mapping[str, Any]) -> Optional[float]:
    # These named fields are authored by the quality worker on a 0..100 scale.
    # Never infer units from the numeric value: 0.5 and 1.0 are legitimate
    # exceptionally weak scores, not normalized fractions.
    raw = _first_number(
        matrix.get("normalized_quality_score"),
        matrix.get("epsilon_score"),
        matrix.get("quality_score"),
    )
    if raw is None:
        return None
    return _clamp(raw)


def _mean_numbers(values: Iterable[Any]) -> Optional[float]:
    nums = [_first_number(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


@dataclass(frozen=True)
class BaselineGraspQuality:
    """Immutable analytic quality from the locked RealSense snapshot."""

    valid: bool
    analytic_score: Optional[float]
    analytic_label: str
    embodiment_id: str = "parallel_jaw_two_finger"
    quality_compute_id: Optional[str] = None
    writer_session_id: Optional[str] = None
    lock_frame_id: Optional[Any] = None
    input_source: Optional[str] = None
    contact_source: Optional[str] = None
    raw_epsilon: Optional[float] = None
    force_closure: Optional[Any] = None
    expected_contacts: int = 2
    required_contacts: int = 2
    locked_depth_mm: Optional[float] = None
    locked_grip_width_mm: Optional[float] = None
    locked_angle_deg: Optional[float] = None
    timestamp: Optional[float] = None
    failure_reason: Optional[str] = None

    @property
    def identity_key(self) -> str:
        return "|".join(
            str(v)
            for v in (
                self.writer_session_id or "writer",
                self.lock_frame_id or "lock",
                self.quality_compute_id or "compute",
            )
        )


@dataclass
class LiveGraspObservation:
    """Current operational evidence used to update the live score."""

    timestamp: float = field(default_factory=time.time)
    embodiment_id: str = "parallel_jaw_two_finger"
    writer_session_id: Optional[str] = None
    lock_frame_id: Optional[Any] = None
    lock_attempt_id: Optional[Any] = None
    bottle_target_id: Optional[Any] = None
    frame_id: Optional[Any] = None
    target_status: Optional[str] = None
    ready_to_close: Optional[bool] = None
    detected_contacts: Optional[int] = None
    expected_contacts: Optional[int] = None
    visual_contact_ready: Optional[bool] = None
    alignment_error_px: Optional[float] = None
    angle_error_deg: Optional[float] = None
    depth_delta_mm: Optional[float] = None
    grip_width_delta_mm: Optional[float] = None
    current_depth_mm: Optional[float] = None
    current_grip_width_mm: Optional[float] = None
    distance_confidence: Optional[float] = None
    depth_valid_ratio: Optional[float] = None
    distance_measurement_valid: Optional[bool] = None
    band_drift_px: Optional[float] = None
    band_visibility: Optional[float] = None
    band_drift_state: Optional[str] = None
    band_relock_required: bool = False
    pointcloud_freshness: Optional[str] = None
    pointcloud_age_sec: Optional[float] = None
    pointcloud_contact_valid: Optional[bool] = None
    pointcloud_off_surface: bool = False
    pointcloud_status: Optional[str] = None
    incipient_slip_score: float = 0.0
    gross_slip_score: float = 0.0
    rotation_score: float = 0.0
    vibration_score: float = 0.0
    davis_displacement_px: Optional[float] = None
    davis_cumulative_slip_px: Optional[float] = None
    davis_downward_slip_px: Optional[float] = None
    davis_rotation_delta_deg: Optional[float] = None
    davis_cumulative_rotation_deg: Optional[float] = None
    davis_vibration_rms_px: Optional[float] = None
    davis_motion_confidence: Optional[float] = None
    davis_disturbance_type: Optional[str] = None
    davis_motion_class: Optional[str] = None
    davis_event_freshness: Optional[str] = None
    davis_event_age_ms: Optional[float] = None
    davis_events_valid_for_live_quality: Optional[bool] = None
    davis_report_heartbeat_current: Optional[bool] = None
    davis_report_heartbeat_age_sec: Optional[float] = None
    davis_motion_usable_for_grasp: Optional[bool] = None
    davis_motion_excluded_reason: Optional[str] = None
    davis_target_lock_provenance_valid: Optional[bool] = None
    davis_target_lock_source_writer_session_id: Optional[Any] = None
    davis_target_lock_source_lock_frame_id: Optional[Any] = None
    davis_target_lock_source_attempt_id: Optional[Any] = None
    davis_target_lock_source_bottle_target_id: Optional[Any] = None
    davis_closeup_window: Optional[str] = None
    davis_object_motion_valid: Optional[bool] = None
    davis_grasp_zone_motion_valid: Optional[bool] = None
    grasp_stage: str = "PRE_CLOSE"
    post_grasp_active: bool = False
    boundary_incipient_slip_score: float = 0.0
    boundary_slip_score: float = 0.0
    boundary_rotation_score: float = 0.0
    boundary_vibration_score: float = 0.0
    boundary_event_count: int = 0
    boundary_active_band_count: int = 0
    boundary_measurement_status: Optional[str] = None
    boundary_alignment_status: Optional[str] = None
    boundary_suppression_reason: Optional[str] = None
    aps_state: Optional[str] = None
    event_state: Optional[str] = None
    sync_quality: Optional[str] = None
    davis_realsense_sync_quality: Optional[str] = None
    sync_delta_sec: Optional[float] = None
    sync_age_sec: Optional[float] = None
    frame_age_ms: Optional[float] = None
    missing_optional_sensors: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityDeviation:
    alignment: float = 0.0
    angle: float = 0.0
    depth: float = 0.0
    grip_width: float = 0.0
    contact: float = 0.0
    band_drift: float = 0.0
    band_visibility: float = 0.0
    pointcloud: float = 0.0
    incipient_slip: float = 0.0
    gross_slip: float = 0.0
    rotation: float = 0.0
    vibration: float = 0.0
    target_loss: float = 0.0
    uncertainty: float = 0.0

    @property
    def total_penalty(self) -> float:
        return sum(float(v) for v in asdict(self).values())

    def primary(self) -> str:
        values = asdict(self)
        name, value = max(values.items(), key=lambda item: item[1])
        return name if value > 0.0 else "none"


@dataclass
class SensorContribution:
    sensor: str
    status: str
    available: bool
    confidence: float
    penalty: float
    details: str


@dataclass
class LiveQualityResult:
    timestamp: float
    analytic_quality_score: Optional[float]
    analytic_quality_label: str
    live_quality_score: Optional[float]
    live_quality_label: str
    delta_from_analytic: Optional[float]
    trend: str
    primary_risk: str
    decision: str
    next_action: str
    hard_gates: List[str]
    deviations: QualityDeviation
    sensor_contributions: List[SensorContribution]
    baseline: BaselineGraspQuality
    observation: LiveGraspObservation
    smoothing_active: bool
    smoothing_alpha_used: Optional[float]
    smoothing_bypassed_reason: Optional[str]
    formula: str
    event_marker: Optional[str] = None
    davis_motion_used_for_live_quality: Optional[bool] = None
    davis_motion_live_quality_reason: Optional[str] = None
    davis_closeup_used_for_live_quality: Optional[bool] = None
    davis_closeup_exclusion_reason: Optional[str] = None
    davis_closeup_confidence: Optional[float] = None
    davis_closeup_motion_class: Optional[str] = None
    davis_closeup_window: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["deviations"]["total_penalty"] = round(self.deviations.total_penalty, 3)
        payload["sensor_contributions"] = [asdict(item) for item in self.sensor_contributions]
        analytic_stale = bool(
            self.observation.notes.get("analytic_quality_stale")
            or self.observation.notes.get("matrix_stale_due_to_motion")
        )
        raw_davis_penalty = round(
            float(self.deviations.incipient_slip)
            + float(self.deviations.gross_slip)
            + float(self.deviations.rotation)
            + float(self.deviations.vibration),
            3,
        )
        davis_used = self.davis_motion_used_for_live_quality is True
        davis_penalty = raw_davis_penalty if davis_used else 0.0
        if (
            davis_used
            and _post_grasp_scoring_active(self.observation)
            and self.analytic_quality_score is not None
            and self.live_quality_score is not None
        ):
            # The presentation equation must reconcile exactly even while the
            # operational score is smoothed or a DAVIS hard gate is active.
            davis_penalty = round(
                max(0.0, float(self.analytic_quality_score) - float(self.live_quality_score)),
                3,
            )
        payload.update({
            "live_quality_trend": self.trend,
            "live_quality_primary_risk": self.primary_risk,
            "live_quality_action": self.decision,
            "live_quality_explanation": self.next_action,
            "analytic_quality_status": (
                "INVALID"
                if not self.baseline.valid
                else "STALE"
                if analytic_stale
                else "CURRENT"
            ),
            "analytic_quality_last_label": self.analytic_quality_label,
            "analytic_quality_stale_reason": (
                self.observation.notes.get("analytic_quality_stale_reason")
                if analytic_stale
                else None
            ),
            "live_quality_band_drift_penalty": round(
                float(self.deviations.band_drift) + float(self.deviations.band_visibility), 3
            ),
            "live_quality_depth_penalty": round(
                float(self.deviations.depth) + float(self.deviations.grip_width), 3
            ),
            "live_quality_davis_penalty": davis_penalty if davis_used else 0.0,
            "davis_adjustment_points": -davis_penalty if davis_used else 0.0,
            "davis_raw_penalty_points": raw_davis_penalty if davis_used else 0.0,
            "davis_adjustment_status": "INCLUDED" if davis_used else "EXCLUDED",
            "davis_adjustment_reason": (
                self.davis_motion_live_quality_reason
                or self.davis_closeup_exclusion_reason
                or ("object_motion_valid" if davis_used else "not_current_or_not_object_specific")
            ),
            "live_quality_pointcloud_penalty": round(float(self.deviations.pointcloud), 3),
        })
        payload["display"] = {
            "analytic_quality": self._format_score(self.analytic_quality_score, self.analytic_quality_label),
            "live_quality": self._format_score(self.live_quality_score, self.live_quality_label),
            "delta": None if self.delta_from_analytic is None else round(self.delta_from_analytic, 1),
            "trend": self.trend,
            "primary_risk": self.primary_risk,
            "decision": self.decision,
            "next_action": self.next_action,
        }
        return payload

    @staticmethod
    def _format_score(score: Optional[float], label: str) -> str:
        if score is None:
            return label
        return f"{label} {score:.1f}"


class LiveGraspQualityMonitor:
    """Stateful live monitor with smoothing, hysteresis, and hard gates."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, recording_sink=None):
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})
        self.recording_sink = recording_sink
        self._recording_stream_id = uuid.uuid4().hex
        self._recording_sequence = 0
        self._recording_state_epoch = 0
        self._recording_epoch_input_index = 0
        self._last_baseline_key: Optional[str] = None
        self._last_score: Optional[float] = None
        self._last_result: Optional[LiveQualityResult] = None
        self._last_decision: Optional[str] = None
        self._last_event_key: Optional[str] = None
        self._severe_recovery_remaining = 0
        self._last_post_grasp_phase: Optional[bool] = None

    def reset(self, reason: str = "manual") -> None:
        self._recording_state_epoch += 1
        self._recording_epoch_input_index = 0
        self._last_baseline_key = None
        self._last_score = None
        self._last_result = None
        self._last_decision = None
        self._last_event_key = None
        self._severe_recovery_remaining = 0
        self._last_post_grasp_phase = None
        self._reset_reason = reason

    def update(self, baseline: BaselineGraspQuality, observation: LiveGraspObservation) -> LiveQualityResult:
        if self.recording_sink is not None:
            self.recording_sink.record(
                "live_quality_inputs", {},
                observation_id=f"quality-{self._recording_stream_id}-{self._recording_sequence}",
                host_arrival_monotonic_ns=time.monotonic_ns(),
                host_arrival_epoch_ns=time.time_ns(),
                native_clock={"domain": "application_observation_unspecified", "unit": "s", "timestamp": observation.timestamp},
                metadata={"baseline": asdict(baseline), "observation": asdict(observation), "effective_config": self.config, "state_epoch": self._recording_state_epoch, "epoch_input_index": self._recording_epoch_input_index, "initial_state": ("constructor_reset" if self._recording_state_epoch == 0 else "epoch_reset") if self._recording_epoch_input_index == 0 else "continuation"},
                source_sequence=self._recording_sequence,
            )
            self._recording_sequence += 1
        self._recording_epoch_input_index += 1
        post_grasp_phase = _post_grasp_scoring_active(observation)
        if baseline.identity_key != self._last_baseline_key:
            self._last_baseline_key = baseline.identity_key
            self._last_score = None
            self._last_result = None
            self._last_decision = None
            self._last_event_key = None
            self._severe_recovery_remaining = 0
        elif (
            self._last_post_grasp_phase is not None
            and post_grasp_phase != self._last_post_grasp_phase
        ):
            # Approach penalties must not bleed through smoothing into the
            # post-grasp DAVIS-only score (or vice versa).
            self._last_score = None
            self._last_result = None
            self._last_decision = None
            self._last_event_key = None
            self._severe_recovery_remaining = 0
        self._last_post_grasp_phase = post_grasp_phase

        deviations, contributions = self._compute_deviations(baseline, observation)
        hard_gates = self._hard_gates(baseline, observation)
        analytic_score = baseline.analytic_score if baseline.valid else None
        raw_score = 0.0 if analytic_score is None else _clamp(analytic_score - deviations.total_penalty)
        if hard_gates:
            cap = min(self._hard_gate_cap(gate) for gate in hard_gates)
            raw_score = min(raw_score, cap)

        smoothing_alpha = None
        smoothing_bypass = None
        smoothing_active = False
        if hard_gates:
            smoothed = raw_score
            smoothing_bypass = "hard_gate"
            self._severe_recovery_remaining = int(self.config["smoothing"].get("severe_recovery_frames", 6))
        elif self._last_score is None:
            smoothed = raw_score
            smoothing_bypass = "first_sample"
        else:
            smoothing_active = True
            score_delta = raw_score - self._last_score
            smoothing_alpha = self._alpha_for_delta(score_delta)
            smoothed = self._last_score + smoothing_alpha * score_delta

        smoothed = _clamp(smoothed)
        trend = self._trend(smoothed)
        live_label = _score_label(smoothed, hard_gates)
        primary_risk = hard_gates[0] if hard_gates else deviations.primary()
        decision, next_action = self._decision(live_label, primary_risk, hard_gates, observation)
        event_marker = self._event_marker(decision, primary_risk, hard_gates)
        delta = None if analytic_score is None else round(smoothed - analytic_score, 3)
        davis_current = _davis_events_current(observation)
        davis_motion_exclusion = self._davis_motion_exclusion(observation, baseline)
        davis_used = bool(baseline.valid and davis_current and davis_motion_exclusion is None)
        if not baseline.valid:
            davis_closeup_exclusion = "analytic_baseline_unavailable"
        elif davis_motion_exclusion:
            davis_closeup_exclusion = davis_motion_exclusion
        elif not davis_current:
            freshness = str(observation.davis_event_freshness or "").upper()
            davis_closeup_exclusion = "stale_events" if freshness in {"STALE", "AGING"} else "events_not_current"
        else:
            davis_closeup_exclusion = None
        result = LiveQualityResult(
            timestamp=observation.timestamp,
            analytic_quality_score=analytic_score,
            analytic_quality_label=baseline.analytic_label,
            live_quality_score=round(smoothed, 3),
            live_quality_label=live_label,
            delta_from_analytic=delta,
            trend=trend,
            primary_risk=primary_risk,
            decision=decision,
            next_action=next_action,
            hard_gates=hard_gates,
            deviations=deviations,
            sensor_contributions=contributions,
            baseline=baseline,
            observation=observation,
            smoothing_active=smoothing_active,
            smoothing_alpha_used=None if smoothing_alpha is None else round(smoothing_alpha, 3),
            smoothing_bypassed_reason=smoothing_bypass,
            formula=(
                "live_score = smooth(immutable_analytic_score - validated_DAVIS_motion_penalty); epsilon is not recomputed"
                if post_grasp_phase
                else "live_score = smooth(analytic_score - sum(weighted_current_approach_risk_penalties)); epsilon is not recomputed"
            ),
            event_marker=event_marker,
            davis_motion_used_for_live_quality=davis_used,
            davis_motion_live_quality_reason=(
                (
                    "object_motion_valid"
                    if observation.davis_object_motion_valid is True
                    else "post_grasp_fixed_grasp_zone_valid"
                )
                if davis_used
                else f"excluded: {davis_closeup_exclusion}"
            ),
            davis_closeup_used_for_live_quality=davis_used,
            davis_closeup_exclusion_reason=davis_closeup_exclusion,
            davis_closeup_confidence=observation.davis_motion_confidence,
            davis_closeup_motion_class=observation.davis_motion_class,
            davis_closeup_window=observation.davis_closeup_window,
        )
        self._last_score = smoothed
        self._last_result = result
        self._last_decision = decision
        self._last_event_key = self._event_key(decision, primary_risk, hard_gates)
        return result

    def _compute_deviations(
        self,
        baseline: BaselineGraspQuality,
        obs: LiveGraspObservation,
    ) -> tuple[QualityDeviation, List[SensorContribution]]:
        weights = self.config["weights"]
        thresholds = self.config["thresholds"]
        deviations = QualityDeviation()
        deviations.alignment = weights["alignment"] * _norm(obs.alignment_error_px, thresholds["alignment_px"])
        deviations.angle = weights["angle"] * _norm(obs.angle_error_deg, thresholds["angle_deg"])
        depth_risk = _norm(obs.depth_delta_mm, thresholds["depth_mm"])
        confidence_values = [
            value
            for value in (obs.distance_confidence, obs.depth_valid_ratio)
            if value is not None
        ]
        if confidence_values:
            normalized_confidence = min(
                _clamp(float(value) / 100.0 if float(value) > 1.0 else float(value), 0.0, 1.0)
                for value in confidence_values
            )
            depth_risk = max(depth_risk, 1.0 - normalized_confidence)
        if obs.distance_measurement_valid is False:
            depth_risk = max(depth_risk, 0.75)
        deviations.depth = weights["depth"] * depth_risk
        deviations.grip_width = weights["grip_width"] * _norm(obs.grip_width_delta_mm, thresholds["grip_width_mm"])
        band_state = str(obs.band_drift_state or "").upper()
        band_drift_risk = _norm(obs.band_drift_px, thresholds["band_drift_px"])
        if band_state == "LOW_CONFIDENCE":
            band_drift_risk = max(band_drift_risk, 0.4)
        elif band_state == "STALE":
            band_drift_risk = max(band_drift_risk, 0.75)
        elif band_state == "RELOCK_REQUIRED" or obs.band_relock_required:
            band_drift_risk = 1.0
        deviations.band_drift = weights["band_drift"] * band_drift_risk
        if obs.band_visibility is not None:
            visibility_threshold = max(float(thresholds["band_visibility_ratio"]), 1e-6)
            visibility_risk = _clamp(
                (visibility_threshold - float(obs.band_visibility)) / visibility_threshold,
                0.0,
                1.0,
            )
            deviations.band_visibility = weights["band_visibility"] * visibility_risk
        davis_current = _davis_events_current(obs)
        motion_conf = _clamp(obs.davis_motion_confidence if obs.davis_motion_confidence is not None else 1.0, 0.0, 1.0)
        motion_exclusion = self._davis_motion_exclusion(obs, baseline)
        measured_slip_px = max(
            abs(obs.davis_displacement_px or 0.0),
            abs(obs.davis_downward_slip_px or 0.0),
            abs(obs.davis_cumulative_slip_px or 0.0) * 0.7,
        )
        measured_rotation_deg = max(abs(obs.davis_rotation_delta_deg or 0.0), abs(obs.davis_cumulative_rotation_deg or 0.0) * 0.5)
        measured_vibration_px = abs(obs.davis_vibration_rms_px or 0.0)
        if davis_current and motion_exclusion is None:
            incipient_risk = max(
                _norm(obs.incipient_slip_score, thresholds["incipient_slip_score"]),
                _norm(obs.boundary_incipient_slip_score, thresholds["incipient_slip_score"]),
                _norm(measured_slip_px, thresholds["davis_incipient_slip_px"]) * motion_conf,
            )
            gross_risk = max(
                _norm(obs.gross_slip_score, thresholds["gross_slip_score"]),
                _norm(obs.boundary_slip_score, thresholds["gross_slip_score"]),
                _norm(obs.davis_cumulative_slip_px, thresholds["davis_gross_slip_px"]) * motion_conf,
            )
            rotation_risk = max(
                _norm(obs.rotation_score, thresholds["severe_rotation_score"]),
                _norm(obs.boundary_rotation_score, thresholds["severe_rotation_score"]),
                _norm(measured_rotation_deg, thresholds["davis_rotation_deg"]) * motion_conf,
            )
            vibration_risk = max(
                _norm(obs.vibration_score, thresholds["vibration_score"]),
                _norm(obs.boundary_vibration_score, thresholds["vibration_score"]),
                _norm(measured_vibration_px, thresholds["davis_vibration_rms_px"]) * motion_conf,
            )
        else:
            # a global disturbance (lighting change / camera shake / background
            # motion) or backend-invalidated object motion must not be scored
            # as if the object slipped; it surfaces as sensor uncertainty only
            incipient_risk = gross_risk = rotation_risk = vibration_risk = 0.0
        deviations.incipient_slip = weights["incipient_slip"] * incipient_risk
        deviations.gross_slip = weights["gross_slip"] * gross_risk
        deviations.rotation = weights["rotation"] * rotation_risk
        deviations.vibration = weights["vibration"] * vibration_risk

        target = str(obs.target_status or "").upper()
        if target and target not in {"TRACKING", "LOCKED", "TARGET_LOCKED", "TRACKING_UNCERTAIN"}:
            deviations.target_loss = weights["target_loss"]

        required = int(baseline.required_contacts if baseline.required_contacts is not None else self.config.get("required_contacts", 2))
        detected = obs.detected_contacts
        if detected is not None and detected < required:
            deviations.contact = weights["contact"] * (1.0 - max(0.0, detected / max(required, 1)))
        elif obs.visual_contact_ready is False:
            deviations.contact = weights["contact"] * 0.6

        pointcloud_penalty = 0.0
        pc_fresh = str(obs.pointcloud_freshness or obs.pointcloud_status or "").upper()
        if obs.pointcloud_contact_valid is False and pc_fresh not in {"STALE", "UNAVAILABLE", "AGING"}:
            pointcloud_penalty = weights["pointcloud"]
        # Point cloud is an asynchronous diagnostic/advisory source in the
        # hand demo. Its nominal 0.5 FPS cadence can legitimately exceed the
        # freshness window, so missing/aging/stale data must not make the live
        # quality graph move. Only a fresh, current off-surface contact result
        # is allowed to contribute a penalty.
        deviations.pointcloud = pointcloud_penalty

        uncertainty = 0.0
        if obs.frame_age_ms is not None:
            uncertainty += weights["uncertainty"] * 0.4 * _norm(obs.frame_age_ms, thresholds["frame_age_ms"])
        # The hand demo freezes the RealSense analytic baseline before DAVIS
        # monitoring starts. DAVIS then measures later motion in its own pixel
        # frame, so cross-camera skew is diagnostic rather than a quality
        # penalty. Current DAVIS event freshness is still enforced below.
        scored_missing_optional = [
            sensor
            for sensor in obs.missing_optional_sensors
            if str(sensor).lower() not in {"point_cloud", "point_cloud_stale", "pointcloud", "pointcloud_stale"}
        ]
        if scored_missing_optional:
            uncertainty += min(weights["uncertainty"] * 0.4, len(scored_missing_optional) * 1.5)
        if (
            davis_current
            and motion_exclusion is not None
            and not _davis_applicability_exclusion(motion_exclusion)
        ):
            # excluded DAVIS motion surfaces as sensor uncertainty, not as slip
            uncertainty += min(weights["uncertainty"] * 0.3, 1.5)
        deviations.uncertainty = _clamp(uncertainty, 0.0, weights["uncertainty"] * 1.5)

        if _post_grasp_scoring_active(obs):
            # After the operator declares a grasp, hand/finger occlusion makes
            # RGB-D alignment, band visibility, contact count, and distance
            # unsuitable as live stability penalties. The immutable planned
            # baseline is now adjusted only by validated DAVIS motion. Sensor
            # availability remains visible through exclusion/readiness state.
            for name in (
                "alignment",
                "angle",
                "depth",
                "grip_width",
                "contact",
                "band_drift",
                "band_visibility",
                "pointcloud",
                "target_loss",
                "uncertainty",
            ):
                setattr(deviations, name, 0.0)

        contributions = self._sensor_contributions(baseline, obs, deviations)
        return deviations, contributions

    def _sensor_contributions(
        self,
        baseline: BaselineGraspQuality,
        obs: LiveGraspObservation,
        deviations: QualityDeviation,
    ) -> List[SensorContribution]:
        rgb_penalty = (
            deviations.alignment
            + deviations.angle
            + deviations.band_drift
            + deviations.band_visibility
            + deviations.contact * 0.35
            + deviations.target_loss
        )
        depth_penalty = deviations.depth + deviations.grip_width + deviations.contact * 0.35
        pc_penalty = deviations.pointcloud
        aps_penalty = deviations.uncertainty * 0.15
        event_penalty = deviations.incipient_slip + deviations.gross_slip + deviations.rotation + deviations.vibration
        sync_penalty = 0.0
        davis_event_status = obs.davis_motion_class or obs.event_state or obs.davis_event_freshness or "OPTIONAL"
        davis_event_details = (
            "object-local DAVIS pixel motion plus boundary bands "
            f"(precursor {obs.boundary_incipient_slip_score:.1f}, slip {obs.boundary_slip_score:.1f}, "
            f"rotation {obs.boundary_rotation_score:.1f}, vibration {obs.boundary_vibration_score:.1f}); "
            "epsilon is not recomputed"
        )
        motion_exclusion = self._davis_motion_exclusion(obs, baseline)
        if not _davis_events_current(obs):
            davis_event_details = "DAVIS not used because event data is stale, unavailable, or outside the accepted software sync window"
        elif motion_exclusion is not None:
            davis_event_details = f"DAVIS motion excluded from live quality: {motion_exclusion} (global disturbance / non-object motion is not scored as slip)"
        return [
            SensorContribution("RGB", self._rgb_status(obs), True, _confidence(rgb_penalty), round(rgb_penalty, 2), "target alignment, locked-band drift/visibility, angle, and contact evidence"),
            SensorContribution("depth", self._depth_status(obs), obs.current_depth_mm is not None, _confidence(depth_penalty), round(depth_penalty, 2), "depth shift, masked-depth/distance confidence, and grip-width consistency"),
            SensorContribution("point cloud", self._pointcloud_status(obs), str(obs.pointcloud_freshness or "").upper() not in {"", "UNAVAILABLE", "STALE", "AGING"}, _confidence(pc_penalty), round(pc_penalty, 2), "advisory 3D contact consistency; only fresh invalid contacts affect quality"),
            SensorContribution("DAVIS APS", obs.aps_state or "OPTIONAL", bool(obs.aps_state), _confidence(aps_penalty), round(aps_penalty, 2), "APS is context only unless validated as current"),
            SensorContribution("DAVIS events", davis_event_status, bool(obs.event_state or obs.davis_motion_class), _confidence(event_penalty), round(event_penalty, 2), davis_event_details),
            SensorContribution("synchronization", _effective_live_sync_quality(obs) or "N/A", False, 1.0, 0.0, f"diagnostic only; RealSense baseline is frozen before asynchronous DAVIS monitoring (global bundle quality {obs.sync_quality or 'N/A'})"),
        ]

    def _hard_gates(self, baseline: BaselineGraspQuality, obs: LiveGraspObservation) -> List[str]:
        thresholds = self.config["thresholds"]
        gates: List[str] = []
        if not baseline.valid:
            gates.append("invalid_baseline")
        post_grasp_phase = _post_grasp_scoring_active(obs)
        if not post_grasp_phase:
            if obs.band_relock_required or str(obs.band_drift_state or "").upper() == "RELOCK_REQUIRED":
                gates.append("band_relock_required")
            target = str(obs.target_status or "").upper()
            if target and target not in {"TRACKING", "LOCKED", "TARGET_LOCKED", "TRACKING_UNCERTAIN"}:
                gates.append("target_lost")
            required = int(baseline.required_contacts if baseline.required_contacts is not None else self.config.get("required_contacts", 2))
            if obs.detected_contacts is not None and obs.detected_contacts < required:
                gates.append("missing_required_contacts")
        if _davis_events_current(obs) and self._davis_motion_exclusion(obs, baseline) is None:
            # global disturbances (lighting change / camera shake) are not
            # object slip and must not trip the slip/rotation hard gates
            if max(obs.gross_slip_score, obs.boundary_slip_score) >= float(thresholds["gross_slip_score"]):
                gates.append("gross_slip")
            if obs.davis_cumulative_slip_px is not None and obs.davis_cumulative_slip_px >= float(thresholds["davis_gross_slip_px"]) * 1.4:
                gates.append("gross_slip")
            if max(obs.rotation_score, obs.boundary_rotation_score) >= float(thresholds["severe_rotation_score"]):
                gates.append("severe_rotation")
            if obs.davis_cumulative_rotation_deg is not None and abs(obs.davis_cumulative_rotation_deg) >= float(thresholds["davis_rotation_deg"]) * 1.5:
                gates.append("severe_rotation")
        if not post_grasp_phase:
            pc_fresh = str(obs.pointcloud_freshness or obs.pointcloud_status or "").upper()
            if obs.pointcloud_off_surface and pc_fresh not in {"STALE", "UNAVAILABLE", "AGING"}:
                gates.append("clearly_off_surface_3d_contacts")
        return _unique(gates)

    def _hard_gate_cap(self, gate: str) -> float:
        caps = self.config["hard_gates"]
        mapping = {
            "invalid_baseline": "invalid_baseline_score_cap",
            "target_lost": "target_lost_score_cap",
            "missing_required_contacts": "missing_contacts_score_cap",
            "gross_slip": "gross_slip_score_cap",
            "severe_rotation": "severe_rotation_score_cap",
            "clearly_off_surface_3d_contacts": "off_surface_score_cap",
            "band_relock_required": "band_relock_score_cap",
            "unacceptable_synchronization": "sync_score_cap",
        }
        return float(caps.get(mapping.get(gate, ""), 0.0))

    def _davis_motion_exclusion(
        self,
        obs: LiveGraspObservation,
        baseline: Optional[BaselineGraspQuality] = None,
    ) -> Optional[str]:
        threshold = float(
            self.config.get("thresholds", {}).get(
                "davis_motion_confidence",
                DEFAULT_CONFIG["thresholds"]["davis_motion_confidence"],
            )
        )
        return _davis_motion_exclusion(
            obs,
            baseline=baseline,
            min_confidence=threshold,
        )

    def _alpha_for_delta(self, score_delta: float) -> float:
        smoothing = self.config["smoothing"]
        if self._severe_recovery_remaining > 0 and score_delta > 0:
            self._severe_recovery_remaining -= 1
            return float(smoothing["alpha_recovery"])
        if score_delta < -float(smoothing["fast_drop_threshold"]):
            return float(smoothing["alpha_fast_drop"])
        return float(smoothing["alpha_normal"])

    def _trend(self, score: float) -> str:
        if self._last_score is None:
            return "stable"
        deadband = float(self.config["smoothing"].get("trend_deadband", 2.0))
        delta = score - self._last_score
        if delta > deadband:
            return "rising"
        if delta < -deadband:
            return "falling"
        return "stable"

    @staticmethod
    def _decision(label: str, risk: str, gates: List[str], obs: LiveGraspObservation) -> tuple[str, str]:
        if "invalid_baseline" in gates:
            return "WAIT_FOR_ANALYTIC_QUALITY", "Run Fresh Lock + Quality before using live quality."
        if "target_lost" in gates:
            return "DO_NOT_CLOSE", "Recover or reacquire the target."
        if "band_relock_required" in gates:
            return "HOLD_AND_RELOCK", "The locked band is stale; run Fresh Lock + Quality before closing."
        if "gross_slip" in gates:
            return "REGRIP", "Gross slip-like motion estimated; hold, then release or regrip."
        if "missing_required_contacts" in gates:
            return "HOLD_AND_RECHECK", "Wait for both required contacts before closing further."
        if "severe_rotation" in gates:
            return "REGRIP", "Rotation risk is severe; correct orientation."
        if "clearly_off_surface_3d_contacts" in gates:
            return "HOLD_AND_RECHECK", "3D contacts appear off surface; validate point cloud/contact association."
        if "unacceptable_synchronization" in gates:
            return "HOLD_AND_RECHECK", "Sensor timing is not trustworthy enough for live quality."
        if _post_grasp_scoring_active(obs) and not _davis_stream_active(obs):
            return "WAIT_FOR_DAVIS", "DAVIS event evidence is not current; keep holding and wait for the monitor."
        if not _post_grasp_scoring_active(obs) and obs.distance_measurement_valid is False:
            return "HOLD_AND_RECHECK", "Masked RGB-D distance is not trustworthy enough for movement or closing guidance."
        motion = str(obs.davis_motion_class or "").upper()
        if risk == "incipient_slip" or motion in {"INCIPIENT_SLIP", "MICRO_SLIP"}:
            return "HOLD_AND_RECHECK", "Incipient slip is rising; hold and verify stability."
        if label in {"GOOD", "OK"}:
            if _post_grasp_scoring_active(obs):
                return "MONITOR", "Post-grasp quality is using the frozen baseline and current validated DAVIS motion."
            if obs.ready_to_close:
                return "READY_TO_CLOSE", "Analytic grasp is acceptable and live evidence is stable."
            return "MONITOR", "Live quality is acceptable; continue monitoring current grasp state."
        if label == "WEAK":
            return "ADJUST", "Operational quality is weak; improve alignment/contact before lift."
        return "HOLD_AND_RECHECK", "Operational quality is low."

    def _event_marker(self, decision: str, risk: str, gates: List[str]) -> Optional[str]:
        current = self._event_key(decision, risk, gates)
        last = self._last_event_key
        return current if current != last else None

    @staticmethod
    def _event_key(decision: str, risk: str, gates: List[str]) -> str:
        marker = ",".join(gates) if gates else risk
        return f"{decision}:{marker}"

    @staticmethod
    def _rgb_status(obs: LiveGraspObservation) -> str:
        if obs.target_status:
            return str(obs.target_status)
        return "active"

    @staticmethod
    def _depth_status(obs: LiveGraspObservation) -> str:
        if obs.current_depth_mm is None:
            return "UNAVAILABLE"
        if obs.distance_measurement_valid is False:
            return f"UNRELIABLE ({obs.current_depth_mm:.0f} mm)"
        return f"{obs.current_depth_mm:.0f} mm"

    @staticmethod
    def _pointcloud_status(obs: LiveGraspObservation) -> str:
        if obs.pointcloud_freshness:
            return str(obs.pointcloud_freshness)
        return obs.pointcloud_status or "UNAVAILABLE"


def _confidence(penalty: float) -> float:
    return round(_clamp(1.0 - penalty / 100.0, 0.0, 1.0), 3)


def _davis_motion_exclusion(
    obs: LiveGraspObservation,
    *,
    baseline: Optional[BaselineGraspQuality] = None,
    min_confidence: float = DEFAULT_CONFIG["thresholds"]["davis_motion_confidence"],
) -> Optional[str]:
    """Reason to exclude DAVIS motion from slip/rotation/vibration penalties.

    Global disturbances (lighting change, camera shake, background motion) and
    backend-invalidated object motion must not be scored as if the object
    slipped."""
    if obs.notes.get("analytic_quality_stale"):
        return "analytic_baseline_stale"
    if not obs.post_grasp_active or str(obs.grasp_stage or "PRE_CLOSE").upper() == "PRE_CLOSE":
        return "pre_close_monitoring_only"
    if (
        obs.davis_object_motion_valid is not True
        and obs.davis_grasp_zone_motion_valid is not True
    ):
        return "object_motion_not_confirmed"
    if obs.davis_motion_usable_for_grasp is not True:
        return obs.davis_motion_excluded_reason or "object_motion_invalid"
    confidence = _first_number(obs.davis_motion_confidence)
    required_confidence = float(min_confidence)
    if confidence is None:
        return "motion_confidence_unavailable"
    if confidence < required_confidence:
        return "motion_confidence_below_threshold"
    disturbance = str(obs.davis_disturbance_type or "").upper().replace(" ", "_")
    motion_class = str(obs.davis_motion_class or "").upper().replace(" ", "_")
    for name in (disturbance, motion_class):
        if name in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"}:
            return name.lower()
    if not _davis_events_current(obs):
        return "stale_events"
    # The APS contour and event estimate are DAVIS-native. Cross-camera lock
    # identity is intentionally not required because these cameras are not
    # calibrated and the post-grasp signal is only relative bottle motion.
    return None


def _davis_target_lock_identity_exclusion(
    obs: LiveGraspObservation,
    baseline: Optional[BaselineGraspQuality],
) -> Optional[str]:
    """Require DAVIS object motion to name the same analytic bottle lock."""
    if obs.davis_target_lock_provenance_valid is not True:
        return "davis_target_lock_provenance_not_valid"
    if baseline is None:
        return "davis_target_lock_analytic_identity_unavailable"

    identity_pairs = (
        (
            "writer_session",
            obs.davis_target_lock_source_writer_session_id,
            baseline.writer_session_id,
        ),
        (
            "lock_frame",
            obs.davis_target_lock_source_lock_frame_id,
            baseline.lock_frame_id,
        ),
        (
            "attempt",
            obs.davis_target_lock_source_attempt_id,
            obs.lock_attempt_id,
        ),
        (
            "bottle_target",
            obs.davis_target_lock_source_bottle_target_id,
            obs.bottle_target_id,
        ),
    )
    for name, source_value, expected_value in identity_pairs:
        if source_value is None or expected_value is None:
            return f"davis_target_lock_{name}_identity_unavailable"
        if str(source_value) != str(expected_value):
            return f"davis_target_lock_{name}_mismatch"

    current_analytic_pairs = (
        ("writer_session", obs.writer_session_id, baseline.writer_session_id),
        ("lock_frame", obs.lock_frame_id, baseline.lock_frame_id),
    )
    for name, current_value, analytic_value in current_analytic_pairs:
        if current_value is None or analytic_value is None:
            return f"davis_target_lock_current_{name}_unavailable"
        if str(current_value) != str(analytic_value):
            return f"davis_target_lock_current_{name}_mismatch"
    return None


def _davis_applicability_exclusion(reason: Optional[str]) -> bool:
    """Return true for exclusions that are not sensor-quality uncertainty."""
    text = str(reason or "")
    return text in {"analytic_baseline_stale", "pre_close_monitoring_only"} or text.startswith(
        "davis_target_lock_"
    )


def _davis_events_current(obs: LiveGraspObservation) -> bool:
    if obs.davis_events_valid_for_live_quality is not True:
        return False
    freshness = str(obs.davis_event_freshness or "").upper()
    if freshness in {"STALE", "UNAVAILABLE", "AGING"}:
        return False
    return True


def _davis_stream_active(obs: LiveGraspObservation) -> bool:
    if obs.davis_report_heartbeat_current is False:
        return False
    state = str(obs.event_state or "").upper()
    if not state:
        return False
    return not any(
        marker in state
        for marker in (
            "UNAVAILABLE",
            "NOT_RUNNING",
            "NOT RUNNING",
            "DEVICE_NOT_FOUND",
            "STREAM_NOT_STARTED",
            "ERROR",
            "STOPPED",
        )
    )


def _post_grasp_scoring_active(obs: LiveGraspObservation) -> bool:
    return bool(
        obs.post_grasp_active
        and str(obs.grasp_stage or "PRE_CLOSE").upper() != "PRE_CLOSE"
    )


def _effective_live_sync_quality(obs: LiveGraspObservation) -> str:
    return str(obs.davis_realsense_sync_quality or obs.sync_quality or "")


def quality_applicability_from_status(
    realtime: Mapping[str, Any],
    matrix: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe whether analytic quality applies without changing epsilon math."""
    realtime = realtime or {}
    matrix = matrix or {}
    object_state = realtime.get("object_state") if isinstance(realtime.get("object_state"), Mapping) else {}
    selection = (
        realtime.get("multi_bottle_target_selection")
        if isinstance(realtime.get("multi_bottle_target_selection"), Mapping)
        else {}
    )
    selected_id = selection.get("selected_target_id")
    candidates = selection.get("candidates") if isinstance(selection.get("candidates"), list) else []
    selected = next(
        (
            item
            for item in candidates
            if isinstance(item, Mapping) and str(item.get("target_id")) == str(selected_id)
        ),
        {},
    )
    pose_class = _first_text(
        object_state.get("pose_class"),
        realtime.get("object_pose_class"),
        default="UNKNOWN",
    ).upper()
    pose_valid_value = object_state.get("pose_valid_for_grasp")
    pose_valid = bool(pose_valid_value) if pose_valid_value is not None else pose_class not in {
        "UNKNOWN",
        "UNKNOWN_ORIENTATION",
        "PARTIAL_VIEW",
        "TOP_DOWN_CAP",
        "TOP_DOWN_BODY",
    }
    clearance = (
        realtime.get("inter_bottle_grasp_clearance")
        if isinstance(realtime.get("inter_bottle_grasp_clearance"), Mapping)
        else {}
    )
    clearance_status = _first_text(
        selected.get("grasp_clearance_status"),
        clearance.get("grasp_clearance_status"),
        default="NOT_TESTED",
    ).upper()
    clearance_valid = clearance_status in {"ESTIMATED_CLEAR", "CLEAR_GRASP_AVAILABLE"}
    fallback_used = bool(
        selected.get("grasp_band_fallback_used")
        or clearance.get("grasp_band_fallback_used")
    )
    selection_blocked = bool(selection and selection.get("candidate_count", 0) and not selected_id)
    clutter_valid = bool(clearance_valid and not selection_blocked)
    stale = bool(
        matrix.get("quality_label_stale")
        or matrix.get("matrix_stale_due_to_motion")
        or str(matrix.get("quality_matrix_status") or "").upper() == "STALE_NEEDS_RECOMPUTE"
    )
    contact_source = _first_text(
        matrix.get("contact_source"),
        realtime.get("contact_source"),
        default="unavailable",
    )
    contact_geometry_valid = bool(
        matrix.get("contact_3d_inputs_valid")
        or matrix.get("contact_quality_gate_pass")
    )

    not_applicable_reason = None
    cannot_grasp_reason = None
    if stale:
        applicability = "STALE_NOT_APPLICABLE_TO_CURRENT_POSE"
        not_applicable_reason = "quality_matrix_stale_for_current_pose_or_lock"
    elif selection_blocked or (selection and not clutter_valid):
        applicability = "BLOCKED_BY_CURRENT_GRASP_FEASIBILITY"
        candidate_reasons = [
            reason
            for item in candidates
            if isinstance(item, Mapping)
            for reason in (item.get("reasons") or [])
            if not any(
                positive in str(reason)
                for positive in (
                    "graspable_region_clear",
                    "approach_supported",
                    "clear_grasp_band_selected",
                )
            )
        ]
        cannot_grasp_reason = _first_text(
            selection.get("grasp_feasibility_reason"),
            *candidate_reasons,
            *(selection.get("limitations") or []),
            default="no_current_feasible_grasp_clearance",
        )
        not_applicable_reason = cannot_grasp_reason
    elif not pose_valid:
        applicability = "NOT_APPLICABLE_UNSUPPORTED_OR_UNVERIFIED_POSE"
        not_applicable_reason = f"pose_not_valid_for_current_contact_model:{pose_class.lower()}"
    elif not clearance_valid and selection:
        applicability = "NOT_APPLICABLE_CLEARANCE_NOT_VALIDATED"
        not_applicable_reason = f"clearance_status:{clearance_status.lower()}"
    elif not contact_geometry_valid:
        applicability = "NOT_TESTED_MISSING_VALID_CONTACT_GEOMETRY"
        not_applicable_reason = "valid_current_contact_geometry_unavailable"
    else:
        applicability = "APPLICABLE_ESTIMATED_ANALYTIC"

    return {
        "quality_applicability": applicability,
        "quality_pose_valid": pose_valid,
        "quality_clearance_valid": clearance_valid,
        "quality_clutter_valid": clutter_valid,
        "quality_contact_source": contact_source,
        "quality_estimated_vs_confirmed": "ESTIMATED_ANALYTIC_NOT_PHYSICALLY_CONFIRMED",
        "quality_not_applicable_reason": not_applicable_reason,
        "less_optimal_grasp_reason": (
            "preferred_contact_band_blocked_clear_fallback_band_selected"
            if fallback_used
            else None
        ),
        "cannot_grasp_reason": cannot_grasp_reason,
        "quality_physically_validated": False,
    }


def _unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def baseline_from_quality_matrix(
    matrix: Mapping[str, Any],
    realtime: Optional[Mapping[str, Any]] = None,
    embodiment: Optional[Mapping[str, Any]] = None,
) -> BaselineGraspQuality:
    """Build an immutable baseline object from quality-matrix/dashboard fields."""
    realtime = realtime or {}
    embodiment = embodiment or {}
    embodiment_id = _embodiment_id_from_status(embodiment, matrix, realtime)
    expected_contacts, required_contacts = _contact_counts_for_embodiment(
        embodiment_id,
        matrix=matrix,
        embodiment=embodiment,
        default_required=int(DEFAULT_CONFIG.get("required_contacts", 2)),
    )
    score = _baseline_score_from_matrix(matrix)
    label = _first_text(matrix.get("quality_label"), matrix.get("quality_label_display"), default="INVALID").replace(" (STALE)", "")
    status = str(matrix.get("quality_matrix_status") or "").lower()
    valid = bool(score is not None and status not in {"failed", "invalid"})
    failure_reason = None if valid else _first_text(
        matrix.get("quality_matrix_error"),
        matrix.get("failure_reason"),
        matrix.get("quality_stale_reason"),
        matrix.get("quality_matrix_message"),
        default="analytic_quality_unavailable",
    )
    contact_depths = matrix.get("contact_depths_mm") or {}
    locked_depth = _first_number(
        matrix.get("object_center_depth_mm"),
        matrix.get("locked_depth_mm"),
        _mean_numbers(contact_depths.values()) if isinstance(contact_depths, Mapping) else None,
        realtime.get("bottle_distance_mm"),
    )
    return BaselineGraspQuality(
        valid=valid,
        analytic_score=score,
        analytic_label=label.upper(),
        embodiment_id=embodiment_id,
        quality_compute_id=_first_text(matrix.get("quality_compute_id"), matrix.get("matrix_compute_id"), default="") or None,
        writer_session_id=_first_text(matrix.get("writer_session_id"), realtime.get("writer_session_id"), default="") or None,
        lock_frame_id=matrix.get("lock_frame_id") or realtime.get("lock_frame_id"),
        input_source=_first_text(matrix.get("quality_input_source"), realtime.get("quality_input_source"), default="planned_contact"),
        contact_source=_first_text(matrix.get("contact_source"), realtime.get("contact_source"), default="planned_contact"),
        raw_epsilon=_first_number(matrix.get("raw_epsilon"), matrix.get("epsilon"), matrix.get("epsilon_score_raw")),
        force_closure=matrix.get("force_closure_estimate") if "force_closure_estimate" in matrix else matrix.get("force_closure"),
        expected_contacts=expected_contacts,
        required_contacts=required_contacts,
        locked_depth_mm=locked_depth,
        locked_grip_width_mm=_first_number(matrix.get("locked_grip_width_mm"), matrix.get("grip_width_mm"), realtime.get("grip_width_mm")),
        locked_angle_deg=_first_number(matrix.get("locked_bottle_angle_deg"), matrix.get("raw_bottle_angle_deg")),
        timestamp=_first_number(matrix.get("timestamp"), realtime.get("timestamp")),
        failure_reason=failure_reason,
    )


def baseline_from_post_grasp_snapshot(
    realtime: Mapping[str, Any],
    embodiment: Optional[Mapping[str, Any]] = None,
) -> Optional[BaselineGraspQuality]:
    """Return the exact immutable close-time baseline when its lineage is current.

    The quality-matrix artifact is intentionally not consulted here.  It may be
    temporarily absent or may contain a late result from another lock while the
    post-grasp monitor is active.
    """
    realtime = realtime or {}
    if realtime.get("post_grasp_active") is not True:
        return None
    snapshot = realtime.get("post_grasp_baseline")
    if not isinstance(snapshot, Mapping):
        return None
    target_lock = realtime.get("target_lock")
    target_lock = target_lock if isinstance(target_lock, Mapping) else {}
    locked_target = target_lock.get("locked_target")
    locked_target = locked_target if isinstance(locked_target, Mapping) else {}
    writer_session_id = realtime.get("writer_session_id")
    lock_frame_id = (
        realtime.get("lock_frame_id")
        or target_lock.get("lock_frame_id")
        or locked_target.get("lock_frame_id")
    )
    attempt_id = (
        realtime.get("lock_attempt_id")
        or realtime.get("attempt_id")
        or target_lock.get("attempt_id")
        or locked_target.get("attempt_id")
    )
    contract_ok = bool(
        snapshot.get("baseline_frozen") is True
        and snapshot.get("quality_baseline_role") == "IMMUTABLE_LOCK_BASELINE"
        and snapshot.get("quality_input_source") == "planned_contact"
        and snapshot.get("quality_contact_source_forced") is True
        and writer_session_id is not None
        and snapshot.get("writer_session_id") is not None
        and str(writer_session_id) == str(snapshot.get("writer_session_id"))
        and lock_frame_id is not None
        and snapshot.get("lock_frame_id") is not None
        and str(lock_frame_id) == str(snapshot.get("lock_frame_id"))
        and attempt_id is not None
        and snapshot.get("attempt_id") is not None
        and str(attempt_id) == str(snapshot.get("attempt_id"))
        and snapshot.get("quality_compute_id") is not None
    )
    score = _first_number(snapshot.get("quality_score"))
    epsilon = _first_number(snapshot.get("quality_epsilon"))
    if not contract_ok or score is None or epsilon is None:
        return None
    score = _clamp(score)
    embodiment = embodiment or {}
    embodiment_id = _embodiment_id_from_status(embodiment, snapshot, realtime)
    expected_contacts, required_contacts = _contact_counts_for_embodiment(
        embodiment_id,
        matrix=snapshot,
        embodiment=embodiment,
        default_required=int(DEFAULT_CONFIG.get("required_contacts", 2)),
    )
    return BaselineGraspQuality(
        valid=True,
        analytic_score=score,
        analytic_label=_first_text(snapshot.get("quality_label"), default=_score_label(score)).upper(),
        embodiment_id=embodiment_id,
        quality_compute_id=str(snapshot.get("quality_compute_id")),
        writer_session_id=str(snapshot.get("writer_session_id")),
        lock_frame_id=snapshot.get("lock_frame_id"),
        input_source="planned_contact",
        contact_source="planned_contact",
        raw_epsilon=epsilon,
        force_closure=snapshot.get("force_closure"),
        expected_contacts=expected_contacts,
        required_contacts=required_contacts,
        locked_depth_mm=_first_number(snapshot.get("depth_mm")),
        locked_grip_width_mm=_first_number(snapshot.get("grip_width_mm")),
        locked_angle_deg=_first_number(snapshot.get("axis_angle_degrees")),
        timestamp=_first_number(snapshot.get("timestamp")),
        failure_reason=None,
    )


def observation_from_status(
    realtime: Mapping[str, Any],
    matrix: Optional[Mapping[str, Any]] = None,
    davis: Optional[Mapping[str, Any]] = None,
    pointcloud: Optional[Mapping[str, Any]] = None,
    sync: Optional[Mapping[str, Any]] = None,
    baseline: Optional[BaselineGraspQuality] = None,
    embodiment: Optional[Mapping[str, Any]] = None,
) -> LiveGraspObservation:
    """Create a live observation from current dashboard status dictionaries."""
    matrix = matrix or {}
    davis = davis or {}
    pointcloud = pointcloud or {}
    sync = sync or {}
    embodiment = embodiment or {}
    target_lock = realtime.get("target_lock") if isinstance(realtime.get("target_lock"), Mapping) else {}
    locked_target = (
        target_lock.get("locked_target")
        if isinstance(target_lock.get("locked_target"), Mapping)
        else {}
    )
    lock_frame_id = next(
        (
            value
            for value in (
                realtime.get("lock_frame_id"),
                target_lock.get("lock_frame_id"),
                locked_target.get("lock_frame_id"),
            )
            if value is not None
        ),
        None,
    )
    lock_attempt_id = next(
        (
            value
            for value in (
                realtime.get("lock_attempt_id"),
                realtime.get("attempt_id"),
                target_lock.get("attempt_id"),
                locked_target.get("attempt_id"),
            )
            if value is not None
        ),
        None,
    )
    bottle_target_id = next(
        (
            value
            for value in (
                realtime.get("locked_bottle_target_id"),
                realtime.get("bottle_target_id"),
                target_lock.get("bottle_target_id"),
                locked_target.get("bottle_target_id"),
            )
            if value is not None
        ),
        None,
    )
    embodiment_id = _embodiment_id_from_status(embodiment, matrix, realtime, baseline=baseline)
    expected_contacts, _required_contacts = _contact_counts_for_embodiment(
        embodiment_id,
        matrix=matrix,
        embodiment=embodiment,
        default_required=(baseline.required_contacts if baseline else int(DEFAULT_CONFIG.get("required_contacts", 2))),
    )
    contact_source = _first_text(
        matrix.get("contact_source"),
        matrix.get("quality_input_source"),
        realtime.get("contact_source"),
        baseline.contact_source if baseline else None,
        default="planned_contact",
    ).lower()
    detected_contact_source = contact_source in {
        "detected_gripper_contact",
        "detected_finger_contact",
        "measured_gripper_contact",
    }
    detected_contacts = (
        _detected_contacts(realtime, matrix, expected_contact_count=expected_contacts)
        if detected_contact_source
        else None
    )
    current_depth = _first_number(realtime.get("bottle_distance_mm"), realtime.get("depth_mm"))
    current_grip = _first_number(realtime.get("grip_width_mm"))
    depth_delta = _first_number(
        matrix.get("current_vs_locked_depth_delta_mm"),
        (None if not baseline or baseline.locked_depth_mm is None or current_depth is None else current_depth - baseline.locked_depth_mm),
    )
    grip_delta = _first_number(
        matrix.get("current_vs_locked_grip_width_delta_mm"),
        (None if not baseline or baseline.locked_grip_width_mm is None or current_grip is None else current_grip - baseline.locked_grip_width_mm),
    )
    pc_reason = str(pointcloud.get("contact_3d_reason") or pointcloud.get("contact_3d_quality") or "").lower()
    pc_freshness = _first_text(pointcloud.get("freshness"), pointcloud.get("pointcloud_freshness"), pointcloud.get("status"), default="")
    sync_freshness = sync.get("sensor_freshness") if isinstance(sync.get("sensor_freshness"), Mapping) else {}
    sync_pc = sync_freshness.get("pointcloud") if isinstance(sync_freshness.get("pointcloud"), Mapping) else {}
    sync_de = sync_freshness.get("davis_events") if isinstance(sync_freshness.get("davis_events"), Mapping) else {}
    sync_pointcloud_freshness = _first_text(sync.get("pointcloud_freshness"), sync_pc.get("freshness"), default="")
    if sync_pointcloud_freshness:
        pc_freshness = sync_pointcloud_freshness
    pointcloud_valid_for_live_quality = _bool_or_none(sync.get("pointcloud_valid_for_live_quality"))
    davis_event_freshness = _first_text(sync.get("davis_event_freshness"), sync_de.get("freshness"), default="")
    davis_events_valid_for_live_quality = _bool_or_none(sync.get("davis_events_valid_for_live_quality"))
    if davis_events_valid_for_live_quality is None:
        davis_stream_active = str(
            davis.get("stream") or davis.get("event_state") or ""
        ).upper() not in {"", "DEVICE_NOT_FOUND", "STREAM_NOT_STARTED", "UNAVAILABLE"}
        davis_events_valid_for_live_quality = bool(
            davis_stream_active
            and davis_event_freshness.upper() not in {"", "STALE", "AGING", "UNAVAILABLE"}
        )
    missing_optional = []
    if not pointcloud or pc_freshness.upper() in {"", "UNAVAILABLE"}:
        missing_optional.append("point_cloud")
    elif pointcloud_valid_for_live_quality is False or pc_freshness.upper() in {"STALE", "AGING"}:
        missing_optional.append("point_cloud_stale")
    if not davis or str(davis.get("stream") or davis.get("event_state") or "").upper() in {"", "DEVICE_NOT_FOUND", "STREAM_NOT_STARTED"}:
        missing_optional.append("davis_events")
    elif davis_events_valid_for_live_quality is False or davis_event_freshness.upper() in {"STALE", "AGING"}:
        missing_optional.append("davis_events_stale")
    sync_offset_ms = _first_number(sync.get("max_sensor_time_offset_ms"), sync.get("max_offset_ms"))
    davis_realsense_sync_quality = _first_text(sync.get("davis_realsense_sync_quality"), default="") or None
    davis_realsense_offset_ms = _first_number(sync.get("davis_realsense_offset_ms"))
    if davis_realsense_offset_ms is not None:
        sync_delta_sec = davis_realsense_offset_ms / 1000.0
    else:
        live_quality_offset_ms = _first_number(
            sync.get("live_quality_max_sensor_offset_ms"),
            sync_offset_ms,
        )
        sync_delta_sec = (
            live_quality_offset_ms / 1000.0
            if live_quality_offset_ms is not None
            else _first_number(sync.get("max_sensor_time_delta_sec"), sync.get("max_delta_sec"), sync.get("sensor_time_delta_sec"))
        )
    davis_notes = {}
    if davis_events_valid_for_live_quality is False or davis_event_freshness.upper() in {"STALE", "AGING", "UNAVAILABLE"}:
        davis_notes["davis_not_used_reason"] = "DAVIS not used because event data is stale"
    pc_contact_valid = _bool_or_none(pointcloud.get("contact_3d_valid"))
    pc_off_surface = ("off_surface" in pc_reason or "off surface" in pc_reason)
    if pointcloud_valid_for_live_quality is False or pc_freshness.upper() in {"STALE", "AGING", "UNAVAILABLE"}:
        pc_contact_valid = None
        pc_off_surface = False
    boundary = davis.get("boundary_event_field") if isinstance(davis.get("boundary_event_field"), Mapping) else {}
    post_grasp_active = realtime.get("post_grasp_active") is True
    grasp_stage = "POST_GRASP_MONITORING" if post_grasp_active else "PRE_CLOSE"
    matrix_status = str(matrix.get("quality_matrix_status") or "").upper()
    frozen_post_grasp_baseline = baseline_from_post_grasp_snapshot(
        realtime,
        embodiment=embodiment,
    )
    frozen_post_grasp_baseline_current = bool(
        frozen_post_grasp_baseline is not None
        and baseline is not None
        and frozen_post_grasp_baseline.identity_key == baseline.identity_key
        and frozen_post_grasp_baseline.raw_epsilon == baseline.raw_epsilon
        and frozen_post_grasp_baseline.analytic_score == baseline.analytic_score
    )
    analytic_quality_stale = False if frozen_post_grasp_baseline_current else bool(
        matrix.get("quality_label_stale")
        or matrix.get("matrix_stale_due_to_motion")
        or matrix_status in {"STALE", "STALE_NEEDS_RECOMPUTE"}
    )
    observation = LiveGraspObservation(
        timestamp=time.time(),
        embodiment_id=embodiment_id,
        writer_session_id=_first_text(realtime.get("writer_session_id"), matrix.get("writer_session_id"), default="") or None,
        lock_frame_id=lock_frame_id,
        lock_attempt_id=lock_attempt_id,
        bottle_target_id=bottle_target_id,
        frame_id=realtime.get("monotonic_frame_counter") or realtime.get("overlay_frame_id") or matrix.get("current_hull_frame_id"),
        target_status=_first_text(realtime.get("target_status"), realtime.get("state"), default=""),
        ready_to_close=bool(realtime.get("ready_to_close")),
        detected_contacts=detected_contacts,
        expected_contacts=expected_contacts,
        visual_contact_ready=(
            _bool_or_none(realtime.get("visual_contact_ready"))
            if detected_contact_source
            else None
        ),
        alignment_error_px=_first_number(matrix.get("current_vs_locked_center_delta_px"), realtime.get("display_center_lag_px")),
        angle_error_deg=_first_number(realtime.get("magenta_angle_error_to_bottle_deg"), realtime.get("display_angle_delta_deg"), matrix.get("current_vs_locked_angle_delta_deg")),
        depth_delta_mm=depth_delta,
        grip_width_delta_mm=grip_delta,
        current_depth_mm=current_depth,
        current_grip_width_mm=current_grip,
        distance_confidence=_first_number(
            realtime.get("distance_confidence"),
            realtime.get("distance_confidence_percent"),
        ),
        depth_valid_ratio=_first_number(
            realtime.get("depth_valid_ratio"),
            realtime.get("valid_depth_ratio"),
        ),
        distance_measurement_valid=_bool_or_none(realtime.get("distance_measurement_valid")),
        band_drift_px=_first_number(
            realtime.get("grasp_band_drift_px"),
            target_lock.get("grasp_band_drift_px"),
        ),
        band_visibility=_first_number(
            realtime.get("grasp_band_visibility"),
            target_lock.get("grasp_band_visibility"),
        ),
        band_drift_state=_first_text(
            realtime.get("band_drift_state"),
            target_lock.get("band_drift_state"),
            default="",
        ) or None,
        band_relock_required=bool(
            realtime.get("grasp_band_relock_required")
            or target_lock.get("grasp_band_relock_required")
        ),
        pointcloud_freshness=pc_freshness or None,
        pointcloud_age_sec=_first_number(pointcloud.get("age_sec"), pointcloud.get("pointcloud_age_sec")),
        pointcloud_contact_valid=pc_contact_valid,
        pointcloud_off_surface=pc_off_surface,
        pointcloud_status=_first_text(pointcloud.get("bottle_crop_status"), pointcloud.get("status"), default="") or None,
        incipient_slip_score=_first_number(davis.get("incipient_slip"), davis.get("incipient_slip_score"), davis.get("micro_deviation_score"), 0.0) or 0.0,
        gross_slip_score=_first_number(davis.get("gross_slip"), davis.get("gross_slip_score"), davis.get("final_slip_risk_score"), 0.0) or 0.0,
        rotation_score=_first_number(davis.get("rotation"), davis.get("rotation_score"), davis.get("rotation_risk_score"), 0.0) or 0.0,
        vibration_score=_first_number(davis.get("vibration"), davis.get("vibration_score"), davis.get("vibration_risk_score"), 0.0) or 0.0,
        davis_displacement_px=_first_number(davis.get("displacement_px"), (davis.get("motion_estimate") or {}).get("displacement_px")),
        davis_cumulative_slip_px=_first_number(davis.get("cumulative_slip_px"), (davis.get("motion_estimate") or {}).get("cumulative_slip_px")),
        davis_downward_slip_px=_first_number(davis.get("downward_slip_px"), (davis.get("motion_estimate") or {}).get("downward_slip_px")),
        davis_rotation_delta_deg=_first_number(davis.get("rotation_delta_deg"), (davis.get("motion_estimate") or {}).get("rotation_delta_deg")),
        davis_cumulative_rotation_deg=_first_number(davis.get("cumulative_rotation_deg"), (davis.get("motion_estimate") or {}).get("cumulative_rotation_deg")),
        davis_vibration_rms_px=_first_number(davis.get("vibration_amplitude_px_rms"), (davis.get("motion_estimate") or {}).get("vibration_amplitude_px_rms")),
        davis_motion_confidence=_first_number(davis.get("motion_estimation_confidence"), (davis.get("motion_estimate") or {}).get("motion_estimation_confidence")),
        davis_disturbance_type=_first_text(davis.get("disturbance_type"), (davis.get("motion_estimate") or {}).get("disturbance_type"), default="") or None,
        davis_motion_class=_first_text(davis.get("raw_motion_class"), davis.get("motion_class"), davis.get("davis_safety_class"), default="") or None,
        davis_motion_usable_for_grasp=davis.get("davis_motion_usable_for_grasp"),
        davis_motion_excluded_reason=_first_text(davis.get("davis_motion_excluded_reason"), davis.get("exclusion_reason"), default="") or None,
        davis_target_lock_provenance_valid=_bool_or_none(
            davis.get("davis_target_lock_provenance_valid")
        ),
        davis_target_lock_source_writer_session_id=davis.get(
            "davis_target_lock_source_writer_session_id"
        ),
        davis_target_lock_source_lock_frame_id=davis.get(
            "davis_target_lock_source_lock_frame_id"
        ),
        davis_target_lock_source_attempt_id=davis.get(
            "davis_target_lock_source_attempt_id"
        ),
        davis_target_lock_source_bottle_target_id=davis.get(
            "davis_target_lock_source_bottle_target_id"
        ),
        davis_closeup_window=_first_text(davis.get("davis_closeup_window"), davis.get("active_window"), default="") or None,
        davis_object_motion_valid=_bool_or_none(davis.get("object_motion_valid")),
        davis_grasp_zone_motion_valid=_bool_or_none(davis.get("grasp_zone_motion_valid")),
        grasp_stage=grasp_stage,
        post_grasp_active=post_grasp_active,
        boundary_incipient_slip_score=_first_number(boundary.get("incipient_slip_precursor_score"), 0.0) or 0.0,
        boundary_slip_score=_first_number(boundary.get("slip_cue"), 0.0) or 0.0,
        boundary_rotation_score=_first_number(boundary.get("rotation_cue"), 0.0) or 0.0,
        boundary_vibration_score=_first_number(boundary.get("vibration_cue"), 0.0) or 0.0,
        boundary_event_count=int(_first_number(boundary.get("event_count"), 0.0) or 0),
        boundary_active_band_count=int(_first_number(boundary.get("active_band_count"), 0.0) or 0),
        boundary_measurement_status=_first_text(boundary.get("measurement_status"), default="") or None,
        boundary_alignment_status=_first_text(boundary.get("alignment_status"), default="") or None,
        boundary_suppression_reason=_first_text(boundary.get("suppression_reason"), default="") or None,
        davis_event_freshness=davis_event_freshness or None,
        davis_event_age_ms=_first_number(sync.get("davis_event_age_ms"), sync_de.get("age_ms")),
        davis_events_valid_for_live_quality=davis_events_valid_for_live_quality,
        davis_report_heartbeat_current=_bool_or_none(
            davis.get("report_heartbeat_current")
        ),
        davis_report_heartbeat_age_sec=_first_number(
            davis.get("report_heartbeat_age_sec")
        ),
        aps_state=_first_text(davis.get("aps_state"), davis.get("aps_stream_status"), default="") or None,
        event_state=_first_text(davis.get("event_state"), davis.get("stream"), default="") or None,
        sync_quality=_first_text(sync.get("sync_quality"), sync.get("status"), sync.get("sync_status"), default="") or None,
        davis_realsense_sync_quality=davis_realsense_sync_quality,
        sync_delta_sec=sync_delta_sec,
        sync_age_sec=_first_number(sync.get("bundle_age_sec"), sync.get("age_sec")),
        frame_age_ms=_first_number(realtime.get("frame_age_ms"), (realtime.get("processed_overlay_timing") or {}).get("frame_age_ms")),
        missing_optional_sensors=missing_optional,
        notes={
            "matrix_stale_due_to_motion": (
                False
                if frozen_post_grasp_baseline_current
                else bool(matrix.get("matrix_stale_due_to_motion"))
            ),
            "analytic_quality_stale": analytic_quality_stale,
            "analytic_quality_stale_reason": (
                None
                if frozen_post_grasp_baseline_current
                else _first_text(
                    matrix.get("quality_stale_reason"),
                    ", ".join(str(reason) for reason in (matrix.get("matrix_stale_reasons") or [])),
                    default="",
                ) or None
            ),
            "immutable_post_grasp_baseline_current": frozen_post_grasp_baseline_current,
            "pointcloud_roi_source": pointcloud.get("bottle_roi_source"),
            "global_sync_quality": _first_text(sync.get("sync_quality"), sync.get("status"), sync.get("sync_status"), default="") or None,
            "davis_realsense_sync_quality": davis_realsense_sync_quality,
            "boundary_evidence_used": bool(
                boundary
                and boundary.get("object_motion_valid") is True
                and davis_events_valid_for_live_quality is True
                and davis.get("davis_motion_usable_for_grasp") is True
            ),
            "contact_evidence_mode": (
                "detected_gripper_contact"
                if detected_contact_source
                else "planned_contact_estimate_no_finger_detection_required"
            ),
            **davis_notes,
        },
    )
    provenance_exclusion = _davis_target_lock_identity_exclusion(observation, baseline)
    observation.notes["davis_target_lock_identity_valid"] = provenance_exclusion is None
    observation.notes["davis_target_lock_identity_reason"] = provenance_exclusion
    observation.notes["boundary_evidence_used"] = bool(
        observation.notes.get("boundary_evidence_used")
        and provenance_exclusion is None
    )
    return observation


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "ready", "valid"}:
        return True
    if text in {"false", "no", "0", "not_ready", "invalid"}:
        return False
    return None


def _embodiment_id_from_status(
    embodiment: Mapping[str, Any],
    matrix: Mapping[str, Any],
    realtime: Mapping[str, Any],
    *,
    baseline: Optional[BaselineGraspQuality] = None,
) -> str:
    nested = realtime.get("embodiment_status") if isinstance(realtime.get("embodiment_status"), Mapping) else {}
    return canonical_embodiment_id(
        embodiment.get("canonical_embodiment_id")
        or embodiment.get("active_embodiment")
        or embodiment.get("embodiment_type")
        or matrix.get("canonical_embodiment_id")
        or matrix.get("embodiment_id")
        or nested.get("canonical_embodiment_id")
        or nested.get("active_embodiment")
        or (baseline.embodiment_id if baseline else None)
        or "parallel_jaw_two_finger"
    )


def _contact_counts_for_embodiment(
    embodiment_id: str,
    *,
    matrix: Mapping[str, Any],
    embodiment: Mapping[str, Any],
    default_required: int,
) -> tuple[int, int]:
    definition = load_registry().get(embodiment_id)
    explicit_expected = _first_number(
        matrix.get("expected_contact_count"),
        matrix.get("expected_contacts"),
        embodiment.get("expected_contact_count"),
        embodiment.get("expected_contacts"),
    )
    explicit_required = _first_number(
        matrix.get("required_contact_count"),
        matrix.get("required_contacts"),
        embodiment.get("required_contact_count"),
        embodiment.get("required_contacts"),
    )
    if definition.embodiment_id == "custom" and explicit_expected is None:
        expected = 0
    else:
        expected = int(explicit_expected) if explicit_expected is not None else int(definition.expected_contact_count_min)
    required = int(explicit_required) if explicit_required is not None else expected
    if definition.embodiment_id == "custom" and explicit_required is None:
        required = max(0, required)
    elif explicit_expected is None and definition.expected_contact_count_min <= 0:
        expected = int(default_required)
        required = int(default_required)
    return max(0, expected), max(0, required)


def _detected_contacts(realtime: Mapping[str, Any], matrix: Mapping[str, Any], *, expected_contact_count: int = 2) -> Optional[int]:
    explicit = _first_number(matrix.get("detected_contact_count"), matrix.get("detected_contacts"), realtime.get("detected_contact_count"))
    if explicit is not None:
        return int(explicit)
    left = _first_number(matrix.get("contact_confidence_left"), realtime.get("left_finger_confidence"))
    right = _first_number(matrix.get("contact_confidence_right"), realtime.get("right_finger_confidence"))
    if left is not None or right is not None:
        return int((left or 0.0) > 0.35) + int((right or 0.0) > 0.35)
    visual = _bool_or_none(realtime.get("visual_contact_ready"))
    if visual is True:
        return int(max(1, expected_contact_count))
    if visual is False:
        return 0
    return None


def write_live_quality_outputs(
    result: LiveQualityResult,
    output_dir: Path = OUTPUT_DIR / "live_quality",
    timeline_limit: int = 20,
    timeline_min_interval_sec: float = 1.0,
) -> Dict[str, Any]:
    """Write latest status and append a compact timeline row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    latest_path = output_dir / "latest_status.json"
    timeline_path = output_dir / "timeline.jsonl"
    compact = {
        "timestamp": result.timestamp,
        "analytic_quality_score": result.analytic_quality_score,
        "live_quality_score": result.live_quality_score,
        "live_quality_label": result.live_quality_label,
        "delta_from_analytic": result.delta_from_analytic,
        "trend": result.trend,
        "primary_risk": result.primary_risk,
        "decision": result.decision,
        "event_marker": result.event_marker,
        "hard_gates": result.hard_gates,
        "frame_id": result.observation.frame_id,
        "lock_frame_id": result.baseline.lock_frame_id,
        "writer_session_id": result.baseline.writer_session_id,
        "davis_closeup_used_for_live_quality": result.davis_closeup_used_for_live_quality,
        "davis_closeup_exclusion_reason": result.davis_closeup_exclusion_reason,
        "davis_closeup_confidence": result.davis_closeup_confidence,
        "davis_closeup_motion_class": result.davis_closeup_motion_class,
        "davis_closeup_window": result.davis_closeup_window,
        "davis_adjustment_points": payload["davis_adjustment_points"],
        "davis_adjustment_status": payload["davis_adjustment_status"],
        "davis_adjustment_reason": payload["davis_adjustment_reason"],
    }
    recent_before = read_timeline_tail(timeline_path, limit=1)
    should_append = True
    if recent_before and not result.event_marker:
        last = recent_before[-1]
        last_time = _first_number(last.get("timestamp")) or 0.0
        same_state = last.get("decision") == result.decision and last.get("primary_risk") == result.primary_risk
        should_append = not (same_state and result.timestamp - last_time < timeline_min_interval_sec)
    if should_append:
        append_jsonl_bounded(
            timeline_path,
            compact,
            max_bytes=LIVE_QUALITY_TIMELINE_MAX_BYTES,
        )
    payload["timeline_recent"] = read_timeline_tail(timeline_path, limit=timeline_limit)
    write_json_atomic(latest_path, payload, indent=2)
    return payload


def read_timeline_tail(path: Path = OUTPUT_DIR / "live_quality" / "timeline.jsonl", limit: int = 20) -> List[Dict[str, Any]]:
    """Read the last N timeline rows without requiring pandas or shell tools."""
    if limit <= 0:
        return []
    try:
        path = Path(path)
        chunk_size = 8192
        chunks = []
        line_count = 0
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            while position > 0 and line_count <= limit:
                read_size = min(chunk_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                line_count += chunk.count(b"\n")
        text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
        lines = text.splitlines()[-limit:]
    except Exception:
        return []
    rows = []
    for line in lines:
        try:
            import json

            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


__all__ = [
    "BaselineGraspQuality",
    "LiveGraspObservation",
    "LiveGraspQualityMonitor",
    "LiveQualityResult",
    "QualityDeviation",
    "SensorContribution",
    "baseline_from_quality_matrix",
    "load_live_quality_config",
    "observation_from_status",
    "read_timeline_tail",
    "write_live_quality_outputs",
]

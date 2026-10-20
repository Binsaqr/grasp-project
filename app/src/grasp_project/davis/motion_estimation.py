"""Quantitative DAVIS event-motion estimation.

This module is camera-free: it accepts event arrays already collected by the
canonical DAVIS backend and estimates pixel translation, rotation, vibration,
and disturbance context without opening hardware or changing RealSense state.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from grasp_project.common.config import load_named_config


DEFAULT_DAVIS_MOTION_CONFIG: dict[str, float] = {
    "micro_window_ms": 5.0,
    "short_window_ms": 25.0,
    "motion_window_ms": 100.0,
    "stability_window_ms": 300.0,
    "post_grasp_window_ms": 1200.0,
    "incipient_slip_px_threshold": 2.0,
    "gross_slip_px_threshold": 8.0,
    "cumulative_slip_px_threshold": 12.0,
    "rotation_deg_threshold": 3.0,
    "gross_rotation_deg_threshold": 10.0,
    "vibration_rms_px_threshold": 2.5,
    "min_event_count_for_motion_estimate": 50.0,
    "min_motion_confidence": 0.55,
    "phase_correlation_min_response": 0.18,
    "feature_min_points": 8.0,
}


@dataclass
class DavisMotionEstimate:
    """Quantitative motion result for one DAVIS event window."""

    estimate_id: str
    host_timestamp: float
    event_window_start: Optional[float]
    event_window_end: Optional[float]
    event_window_ms: float
    roi_source: str
    roi_xywh: list[int]
    coordinate_frame: str
    event_count: int
    event_rate: float
    active_pixel_count: int
    active_pixel_ratio: float
    positive_event_count: int
    negative_event_count: int
    event_density: float
    event_support_valid: bool
    dx_px: Optional[float]
    dy_px: Optional[float]
    displacement_px: Optional[float]
    displacement_direction_deg: Optional[float]
    velocity_x_px_s: Optional[float]
    velocity_y_px_s: Optional[float]
    speed_px_s: Optional[float]
    cumulative_dx_px: float
    cumulative_dy_px: float
    cumulative_slip_px: float
    cumulative_motion_path_px: float
    slip_axis_dx_px: Optional[float]
    slip_axis_dy_px: Optional[float]
    slip_along_bottle_axis_px: Optional[float]
    slip_across_bottle_axis_px: Optional[float]
    downward_slip_px: Optional[float]
    cumulative_downward_slip_px: float
    slip_confidence: float
    rotation_delta_deg: Optional[float]
    cumulative_rotation_deg: float
    angular_velocity_deg_s: Optional[float]
    rotation_center_px: Optional[list[float]]
    rotation_confidence: float
    vibration_amplitude_px_rms: Optional[float]
    vibration_frequency_hz: Optional[float]
    repeated_burst_score: float
    vibration_confidence: float
    motion_estimation_method: str
    motion_estimation_confidence: float
    estimate_valid: bool
    invalid_reason: Optional[str]
    disturbance_type: str
    motion_class: str
    impact_score: float
    incipient_slip_score: float
    gross_slip_score: float
    rotation_score: float
    vibration_score: float
    stability_score: float
    translation_method: str
    translation_confidence: float
    translation_valid: bool
    translation_invalid_reason: Optional[str]
    rotation_method: str
    rotation_valid: bool
    rotation_invalid_reason: Optional[str]
    bottle_axis_available: bool
    bottle_axis_source: str
    bottle_axis_vector_px: Optional[list[float]]
    cross_axis_vector_px: Optional[list[float]]
    slip_axis_valid: bool
    slip_axis_reason: str
    roi_confidence: float
    roi_projection_status: str
    roi_projection_error: Optional[str]
    roi_used_for_motion: bool
    global_event_ratio: float
    local_to_global_event_ratio: float
    outside_roi_event_rate: float
    global_centroid_shift: Optional[float]
    full_frame_uniformity: float
    lighting_flicker_score: float
    camera_shake_score: float
    background_motion_score: float
    object_motion_score: float
    disturbance_confidence: float
    disturbance_rejection_applied: bool
    displacement_above_baseline_px: Optional[float]
    rotation_above_baseline_deg: Optional[float]
    vibration_above_baseline_px: Optional[float]
    reason: str
    quantitative_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_davis_motion_config(path: Optional[Path] = None) -> dict[str, float]:
    """Load tuneable DAVIS motion thresholds with conservative defaults."""
    loaded: Mapping[str, Any] = {}
    if path is not None and path.exists():
        try:
            import yaml  # type: ignore

            parsed = yaml.safe_load(path.read_text()) or {}
            loaded = parsed.get("davis_motion", parsed) if isinstance(parsed, Mapping) else {}
        except Exception:
            loaded = {}
    else:
        parsed = load_named_config("davis.yaml", default={})
        loaded = parsed.get("davis_motion", {}) if isinstance(parsed, Mapping) else {}
    if not isinstance(loaded, Mapping):
        # the no-PyYAML flat parser returns nested sections as scalar strings
        loaded = {}
    cfg = dict(DEFAULT_DAVIS_MOTION_CONFIG)
    for key, value in loaded.items():
        if key in cfg:
            try:
                cfg[key] = float(value)
            except (TypeError, ValueError):
                pass
    return cfg


class DavisMotionEstimator:
    """Stateful quantitative event-motion estimator.

    The state only stores baseline noise and cumulative pixel motion. It does
    not own a camera, poll events, or control production grasp decisions.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, frame_size=(346, 260)):
        self.config = dict(DEFAULT_DAVIS_MOTION_CONFIG)
        self.config.update({k: float(v) for k, v in (config or {}).items() if k in self.config})
        self.frame_size = tuple(frame_size)
        self.baseline: dict[str, Any] = self.empty_baseline()
        self.cumulative_dx_px = 0.0
        self.cumulative_dy_px = 0.0
        self.cumulative_slip_px = 0.0
        self.cumulative_motion_path_px = 0.0
        self.cumulative_downward_slip_px = 0.0
        self.cumulative_rotation_deg = 0.0
        self.last_integration_end: Optional[float] = None
        self.last_estimate: Optional[DavisMotionEstimate] = None

    @staticmethod
    def empty_baseline() -> dict[str, Any]:
        return {
            "baseline_valid": False,
            "baseline_event_rate": 0.0,
            "baseline_centroid_jitter_px": 0.0,
            "baseline_translation_noise_px": 0.0,
            "baseline_rotation_noise_deg": 0.0,
            "baseline_vibration_rms_px": 0.0,
            "baseline_timestamp": None,
            "baseline_window_count": 0,
            "baseline_reason": "not_calibrated",
        }

    def reset(self) -> None:
        self.cumulative_dx_px = 0.0
        self.cumulative_dy_px = 0.0
        self.cumulative_slip_px = 0.0
        self.cumulative_motion_path_px = 0.0
        self.cumulative_downward_slip_px = 0.0
        self.cumulative_rotation_deg = 0.0
        self.last_integration_end = None
        self.last_estimate = None

    def calibrate_baseline(self, events: Any, *, now: Optional[float] = None, roi_xywh=None, frame_size=None) -> dict[str, Any]:
        """Estimate quiet-scene event motion noise for later suppression."""
        frame_size = tuple(frame_size or self.frame_size)
        roi = _clip_roi(roi_xywh, frame_size)
        arr = _as_event_array(events)
        if arr.shape[0] < 2:
            self.baseline = {**self.empty_baseline(), "baseline_reason": "no_events", "baseline_timestamp": time.time()}
            return self.baseline
        end = float(now if now is not None else np.max(arr[:, 0]))
        start = float(np.min(arr[:, 0]))
        window_ms = float(self.config["short_window_ms"])
        starts = np.arange(start, end, window_ms / 1000.0)
        centroids = []
        angles = []
        rates = []
        for t0 in starts:
            win = _events_between(arr, t0, min(end, t0 + window_ms / 1000.0), roi)
            if win.shape[0] < 3:
                continue
            centroids.append(_centroid(win))
            angle, conf = _pca_orientation(win)
            if conf > 0.1 and angle is not None:
                angles.append(angle)
            rates.append(win.shape[0] / max(window_ms / 1000.0, 1e-6))
        if len(centroids) < 2:
            self.baseline = {
                **self.empty_baseline(),
                "baseline_reason": "insufficient_quiet_windows",
                "baseline_event_rate": float(np.mean(rates)) if rates else 0.0,
                "baseline_timestamp": time.time(),
            }
            return self.baseline
        c = np.asarray(centroids, dtype=float)
        centroid_jitter = float(np.mean(np.linalg.norm(c - np.mean(c, axis=0), axis=1)))
        translation_noise = float(np.percentile(np.linalg.norm(np.diff(c, axis=0), axis=1), 90))
        rotation_noise = _angle_std_deg(angles)
        vibration = _vibration_rms(c)
        self.baseline = {
            "baseline_valid": True,
            "baseline_event_rate": float(np.mean(rates)) if rates else 0.0,
            "baseline_centroid_jitter_px": round(centroid_jitter, 3),
            "baseline_translation_noise_px": round(translation_noise, 3),
            "baseline_rotation_noise_deg": round(rotation_noise, 3),
            "baseline_vibration_rms_px": round(vibration, 3),
            "baseline_timestamp": time.time(),
            "baseline_window_count": len(centroids),
            "baseline_reason": "ok",
        }
        return self.baseline

    def estimate(
        self,
        events: Any,
        *,
        now: Optional[float] = None,
        analysis_end: Optional[float] = None,
        motion_window_ms: Optional[float] = None,
        roi_xywh=None,
        roi_source: str = "full_frame",
        frame_size=None,
        bottle_axis_vector_px: Optional[list[float] | tuple[float, float]] = None,
        bottle_axis_source: str = "unavailable",
        baseline: Optional[Mapping[str, Any]] = None,
    ) -> DavisMotionEstimate:
        frame_size = tuple(frame_size or self.frame_size)
        roi = _clip_roi(roi_xywh, frame_size)
        arr = _as_event_array(events)
        end = float(
            analysis_end
            if analysis_end is not None
            else (now if now is not None else (np.max(arr[:, 0]) if arr.shape[0] else time.monotonic()))
        )
        motion_ms = float(
            self.config["motion_window_ms"]
            if motion_window_ms is None
            else motion_window_ms
        )
        if not np.isfinite(end):
            raise ValueError("analysis_end must be finite")
        if not np.isfinite(motion_ms) or motion_ms <= 0.0:
            raise ValueError("motion_window_ms must be positive and finite")
        current = _events_between(arr, end - motion_ms / 1000.0, end, roi)
        previous = _events_between(arr, end - 2.0 * motion_ms / 1000.0, end - motion_ms / 1000.0, roi)
        full_current = _events_between(arr, end - motion_ms / 1000.0, end, None)
        baseline_data = dict(self.baseline)
        baseline_data.update(baseline or {})
        support = self._event_support(current, full_current, roi, frame_size, motion_ms)
        if current.shape[0] == 0:
            return self._invalid_estimate("no_events", support, roi, roi_source, frame_size, end, motion_ms, baseline_data)
        if current.shape[0] < int(self.config["min_event_count_for_motion_estimate"]):
            return self._invalid_estimate("insufficient_events", support, roi, roi_source, frame_size, end, motion_ms, baseline_data)

        translation = self._translation(current, previous, roi, frame_size)
        rotation = self._rotation(current, previous, roi, frame_size)
        vibration = self._vibration(arr, end, roi)
        disturbance = self._disturbance(current, full_current, roi, frame_size, translation, motion_ms)
        dx = translation["dx_px"]
        dy = translation["dy_px"]
        displacement = None if dx is None or dy is None else float(math.hypot(dx, dy))
        direction = None if dx is None or dy is None else float(math.degrees(math.atan2(dy, dx)))
        duration = max(motion_ms / 1000.0, 1e-6)
        velocity_x = None if dx is None else dx / duration
        velocity_y = None if dy is None else dy / duration
        speed = None if displacement is None else displacement / duration
        axis = _normalize_axis(bottle_axis_vector_px)
        cross = None if axis is None else [-axis[1], axis[0]]
        if axis is None:
            along = None
            across = None
            slip_axis_dx = dx
            slip_axis_dy = dy
            slip_axis_valid = False
            slip_axis_reason = "no_bottle_axis_available"
        else:
            along = None if dx is None or dy is None else float(dx * axis[0] + dy * axis[1])
            across = None if dx is None or dy is None else float(dx * cross[0] + dy * cross[1])
            slip_axis_dx = None if along is None else float(axis[0] * along)
            slip_axis_dy = None if along is None else float(axis[1] * along)
            slip_axis_valid = True
            slip_axis_reason = "bottle_axis_available"
        downward = None if dy is None else max(0.0, float(dy))
        displacement_above = None if displacement is None else max(0.0, displacement - float(baseline_data.get("baseline_translation_noise_px") or 0.0))
        rotation_delta = rotation["rotation_delta_deg"]
        rotation_above = None if rotation_delta is None else max(0.0, abs(rotation_delta) - float(baseline_data.get("baseline_rotation_noise_deg") or 0.0))
        vibration_rms = vibration["vibration_amplitude_px_rms"]
        vibration_above = None if vibration_rms is None else max(0.0, vibration_rms - float(baseline_data.get("baseline_vibration_rms_px") or 0.0))
        confidence = self._estimate_confidence(support, translation, rotation, disturbance)
        valid = bool(translation["translation_valid"] and confidence >= float(self.config["min_motion_confidence"]))
        invalid_reason = None if valid else (translation["translation_invalid_reason"] or "low_confidence")

        # The translation estimate spans ``motion_ms`` even when estimate() is
        # called faster than that. Scale integration by newly elapsed time so
        # overlapping analysis windows cannot double-count cumulative motion.
        if self.last_integration_end is None:
            integration_fraction = 1.0
        else:
            integration_fraction = float(np.clip((end - self.last_integration_end) / duration, 0.0, 1.0))
        self.last_integration_end = end
        if valid and disturbance["disturbance_type"] not in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"}:
            increment_dx = float(dx or 0.0) * integration_fraction
            increment_dy = float(dy or 0.0) * integration_fraction
            increment_magnitude = float(math.hypot(increment_dx, increment_dy))
            noise = float(baseline_data.get("baseline_translation_noise_px") or 0.0) * integration_fraction
            effective_magnitude = max(0.0, increment_magnitude - noise)
            if increment_magnitude > 1e-9:
                keep = effective_magnitude / increment_magnitude
                increment_dx *= keep
                increment_dy *= keep
            else:
                increment_dx = increment_dy = 0.0
            self.cumulative_dx_px += increment_dx
            self.cumulative_dy_px += increment_dy
            # Persistent slip is net directional drift. Total travelled path is
            # retained separately as a non-slip motion/oscillation diagnostic.
            self.cumulative_slip_px = float(math.hypot(self.cumulative_dx_px, self.cumulative_dy_px))
            self.cumulative_motion_path_px += effective_magnitude
            self.cumulative_downward_slip_px = max(0.0, self.cumulative_dy_px)
            if rotation_delta is not None and rotation["rotation_valid"]:
                self.cumulative_rotation_deg += rotation_delta * integration_fraction

        vibration_for_class = vibration_above
        if vibration_for_class is not None and vibration_for_class > 0.0:
            if float(vibration.get("repeated_burst_score") or 0.0) < 0.12 and float(vibration.get("vibration_frequency_hz") or 0.0) < 3.0:
                vibration_for_class = 0.0
        classes = self._classify(
            valid=valid,
            disturbance=disturbance,
            displacement_above=displacement_above,
            downward=downward,
            rotation_above=rotation_above,
            vibration_above=vibration_for_class,
            impact_score=support["impact_score"],
            confidence=confidence,
            rotation_valid=rotation["rotation_valid"],
        )
        estimate = DavisMotionEstimate(
            estimate_id=uuid.uuid4().hex,
            host_timestamp=time.time(),
            event_window_start=end - motion_ms / 1000.0,
            event_window_end=end,
            event_window_ms=motion_ms,
            roi_source=roi_source,
            roi_xywh=roi,
            coordinate_frame="davis_pixel",
            event_count=int(current.shape[0]),
            event_rate=support["event_rate"],
            active_pixel_count=support["active_pixel_count"],
            active_pixel_ratio=support["active_pixel_ratio"],
            positive_event_count=support["positive_event_count"],
            negative_event_count=support["negative_event_count"],
            event_density=support["event_density"],
            event_support_valid=support["event_support_valid"],
            dx_px=_round_or_none(dx),
            dy_px=_round_or_none(dy),
            displacement_px=_round_or_none(displacement),
            displacement_direction_deg=_round_or_none(direction),
            velocity_x_px_s=_round_or_none(velocity_x),
            velocity_y_px_s=_round_or_none(velocity_y),
            speed_px_s=_round_or_none(speed),
            cumulative_dx_px=round(self.cumulative_dx_px, 3),
            cumulative_dy_px=round(self.cumulative_dy_px, 3),
            cumulative_slip_px=round(self.cumulative_slip_px, 3),
            cumulative_motion_path_px=round(self.cumulative_motion_path_px, 3),
            slip_axis_dx_px=_round_or_none(slip_axis_dx),
            slip_axis_dy_px=_round_or_none(slip_axis_dy),
            slip_along_bottle_axis_px=_round_or_none(along),
            slip_across_bottle_axis_px=_round_or_none(across),
            downward_slip_px=_round_or_none(downward),
            cumulative_downward_slip_px=round(self.cumulative_downward_slip_px, 3),
            slip_confidence=round(classes["slip_confidence"], 3),
            rotation_delta_deg=_round_or_none(rotation_delta),
            cumulative_rotation_deg=round(self.cumulative_rotation_deg, 3),
            angular_velocity_deg_s=_round_or_none(None if rotation_delta is None else rotation_delta / duration),
            rotation_center_px=_centroid(current),
            rotation_confidence=round(rotation["rotation_confidence"], 3),
            vibration_amplitude_px_rms=_round_or_none(vibration_rms),
            vibration_frequency_hz=_round_or_none(vibration["vibration_frequency_hz"]),
            repeated_burst_score=round(vibration["repeated_burst_score"], 3),
            vibration_confidence=round(vibration["vibration_confidence"], 3),
            motion_estimation_method=translation["translation_method"],
            motion_estimation_confidence=round(confidence, 3),
            estimate_valid=valid,
            invalid_reason=invalid_reason,
            disturbance_type=disturbance["disturbance_type"],
            motion_class=classes["motion_class"],
            impact_score=round(classes["impact_score"], 3),
            incipient_slip_score=round(classes["incipient_slip_score"], 3),
            gross_slip_score=round(classes["gross_slip_score"], 3),
            rotation_score=round(classes["rotation_score"], 3),
            vibration_score=round(classes["vibration_score"], 3),
            stability_score=round(classes["stability_score"], 3),
            translation_method=translation["translation_method"],
            translation_confidence=round(translation["translation_confidence"], 3),
            translation_valid=translation["translation_valid"],
            translation_invalid_reason=translation["translation_invalid_reason"],
            rotation_method=rotation["rotation_method"],
            rotation_valid=rotation["rotation_valid"],
            rotation_invalid_reason=rotation["rotation_invalid_reason"],
            bottle_axis_available=axis is not None,
            bottle_axis_source=bottle_axis_source if axis is not None else "unavailable",
            bottle_axis_vector_px=None if axis is None else [round(axis[0], 6), round(axis[1], 6)],
            cross_axis_vector_px=None if cross is None else [round(cross[0], 6), round(cross[1], 6)],
            slip_axis_valid=slip_axis_valid,
            slip_axis_reason=slip_axis_reason,
            roi_confidence=disturbance["roi_confidence"],
            roi_projection_status="not_calibrated" if roi_source == "full_frame" else "provided_by_existing_context",
            roi_projection_error=None if roi_source != "full_frame" else "Full-frame DAVIS motion, not object-specific",
            roi_used_for_motion=roi_source != "full_frame",
            global_event_ratio=disturbance["global_event_ratio"],
            local_to_global_event_ratio=disturbance["local_to_global_event_ratio"],
            outside_roi_event_rate=disturbance["outside_roi_event_rate"],
            global_centroid_shift=disturbance["global_centroid_shift"],
            full_frame_uniformity=disturbance["full_frame_uniformity"],
            lighting_flicker_score=disturbance["lighting_flicker_score"],
            camera_shake_score=disturbance["camera_shake_score"],
            background_motion_score=disturbance["background_motion_score"],
            object_motion_score=disturbance["object_motion_score"],
            disturbance_confidence=disturbance["disturbance_confidence"],
            disturbance_rejection_applied=disturbance["disturbance_rejection_applied"],
            displacement_above_baseline_px=_round_or_none(displacement_above),
            rotation_above_baseline_deg=_round_or_none(rotation_above),
            vibration_above_baseline_px=_round_or_none(vibration_above),
            reason=classes["reason"],
            quantitative_evidence=classes["evidence"],
        )
        self.last_estimate = estimate
        return estimate

    def _invalid_estimate(self, reason, support, roi, roi_source, frame_size, end, motion_ms, baseline_data) -> DavisMotionEstimate:
        full_count = int(support.get("full_event_count") or 0)
        local_count = int(support.get("event_count") or 0)
        background_only = full_count >= int(self.config["min_event_count_for_motion_estimate"]) and local_count == 0 and roi != _clip_roi(None, frame_size)
        disturbance = {
            "disturbance_type": "BACKGROUND_MOTION" if background_only else ("none" if reason == "no_events" else "UNKNOWN_MOTION"),
            "roi_confidence": 0.2 if roi_source == "full_frame" else 1.0,
            "global_event_ratio": 0.0,
            "local_to_global_event_ratio": 0.0,
            "outside_roi_event_rate": full_count / max(float(motion_ms) / 1000.0, 1e-6) if background_only else 0.0,
            "global_centroid_shift": None,
            "full_frame_uniformity": 0.0,
            "lighting_flicker_score": 0.0,
            "camera_shake_score": 0.0,
            "background_motion_score": 100.0 if background_only else 0.0,
            "object_motion_score": 0.0,
            "disturbance_confidence": 1.0 if background_only else 0.0,
            "disturbance_rejection_applied": background_only,
        }
        motion_class = "BACKGROUND_MOTION" if background_only else ("STABLE" if reason == "no_events" else "UNKNOWN_MOTION")
        return DavisMotionEstimate(
            estimate_id=uuid.uuid4().hex,
            host_timestamp=time.time(),
            event_window_start=end - motion_ms / 1000.0,
            event_window_end=end,
            event_window_ms=motion_ms,
            roi_source=roi_source,
            roi_xywh=roi,
            coordinate_frame="davis_pixel",
            event_count=int(support.get("event_count", 0)),
            event_rate=float(support.get("event_rate", 0.0)),
            active_pixel_count=int(support.get("active_pixel_count", 0)),
            active_pixel_ratio=float(support.get("active_pixel_ratio", 0.0)),
            positive_event_count=int(support.get("positive_event_count", 0)),
            negative_event_count=int(support.get("negative_event_count", 0)),
            event_density=float(support.get("event_density", 0.0)),
            event_support_valid=False,
            dx_px=None,
            dy_px=None,
            displacement_px=None,
            displacement_direction_deg=None,
            velocity_x_px_s=None,
            velocity_y_px_s=None,
            speed_px_s=None,
            cumulative_dx_px=round(self.cumulative_dx_px, 3),
            cumulative_dy_px=round(self.cumulative_dy_px, 3),
            cumulative_slip_px=round(self.cumulative_slip_px, 3),
            cumulative_motion_path_px=round(self.cumulative_motion_path_px, 3),
            slip_axis_dx_px=None,
            slip_axis_dy_px=None,
            slip_along_bottle_axis_px=None,
            slip_across_bottle_axis_px=None,
            downward_slip_px=None,
            cumulative_downward_slip_px=round(self.cumulative_downward_slip_px, 3),
            slip_confidence=0.0,
            rotation_delta_deg=None,
            cumulative_rotation_deg=round(self.cumulative_rotation_deg, 3),
            angular_velocity_deg_s=None,
            rotation_center_px=None,
            rotation_confidence=0.0,
            vibration_amplitude_px_rms=None,
            vibration_frequency_hz=None,
            repeated_burst_score=0.0,
            vibration_confidence=0.0,
            motion_estimation_method="unavailable",
            motion_estimation_confidence=0.0,
            estimate_valid=False,
            invalid_reason=reason,
            disturbance_type=disturbance["disturbance_type"],
            motion_class=motion_class,
            impact_score=0.0,
            incipient_slip_score=0.0,
            gross_slip_score=0.0,
            rotation_score=0.0,
            vibration_score=0.0,
            stability_score=100.0 if reason == "no_events" else 70.0,
            translation_method="unavailable",
            translation_confidence=0.0,
            translation_valid=False,
            translation_invalid_reason=reason,
            rotation_method="unavailable",
            rotation_valid=False,
            rotation_invalid_reason=reason,
            bottle_axis_available=False,
            bottle_axis_source="unavailable",
            bottle_axis_vector_px=None,
            cross_axis_vector_px=None,
            slip_axis_valid=False,
            slip_axis_reason="no_motion_estimate",
            roi_confidence=disturbance["roi_confidence"],
            roi_projection_status="not_calibrated" if roi_source == "full_frame" else "provided_by_existing_context",
            roi_projection_error=None if roi_source != "full_frame" else "Full-frame DAVIS motion, not object-specific",
            roi_used_for_motion=roi_source != "full_frame",
            global_event_ratio=0.0,
            local_to_global_event_ratio=0.0,
            outside_roi_event_rate=0.0,
            global_centroid_shift=None,
            full_frame_uniformity=0.0,
            lighting_flicker_score=0.0,
            camera_shake_score=0.0,
            background_motion_score=disturbance["background_motion_score"],
            object_motion_score=0.0,
            disturbance_confidence=0.0,
            disturbance_rejection_applied=False,
            displacement_above_baseline_px=None,
            rotation_above_baseline_deg=None,
            vibration_above_baseline_px=None,
            reason="background_motion_outside_roi" if background_only else reason,
            quantitative_evidence=("DAVIS events were outside the target ROI; not object slip." if background_only else f"No precise DAVIS motion estimate: {reason}."),
        )

    def _event_support(self, current, full_current, roi, frame_size, window_ms) -> dict[str, Any]:
        width, height = frame_size
        frame_area = max(float(width * height), 1.0)
        roi_area = max(float(roi[2] * roi[3]), 1.0)
        count = int(current.shape[0])
        duration = max(float(window_ms) / 1000.0, 1e-6)
        if count:
            pixels = set(zip(current[:, 1].astype(int).tolist(), current[:, 2].astype(int).tolist()))
            positive = int(np.count_nonzero(current[:, 3] > 0))
        else:
            pixels = set()
            positive = 0
        full_pixels = set(zip(full_current[:, 1].astype(int).tolist(), full_current[:, 2].astype(int).tolist())) if full_current.shape[0] else set()
        return {
            "event_count": count,
            "event_rate": count / duration,
            "active_pixel_count": len(pixels),
            "active_pixel_ratio": len(pixels) / roi_area,
            "positive_event_count": positive,
            "negative_event_count": count - positive,
            "event_density": count / roi_area,
            "event_support_valid": count >= int(self.config["min_event_count_for_motion_estimate"]),
            "full_active_pixel_ratio": len(full_pixels) / frame_area,
            "full_event_count": int(full_current.shape[0]),
            "impact_score": _score(count / duration, 3500.0, 18000.0),
        }

    def _translation(self, current, previous, roi, frame_size) -> dict[str, Any]:
        if previous.shape[0] >= int(self.config["min_event_count_for_motion_estimate"]):
            prev_img = _event_image(previous, roi)
            curr_img = _event_image(current, roi)
            response = 0.0
            dx = dy = None
            if prev_img.size and curr_img.size and np.count_nonzero(prev_img) and np.count_nonzero(curr_img):
                try:
                    (dx, dy), response = cv2.phaseCorrelate(prev_img.astype(np.float32), curr_img.astype(np.float32))
                    dx = float(dx)
                    dy = float(dy)
                    response = float(np.clip(response, 0.0, 1.0))
                except Exception:
                    dx = dy = None
            if dx is not None and response >= float(self.config["phase_correlation_min_response"]):
                return {
                    "dx_px": dx,
                    "dy_px": dy,
                    "translation_method": "event_image_phase_correlation",
                    "translation_confidence": response,
                    "translation_valid": True,
                    "translation_invalid_reason": None,
                }
        cur_c = _centroid(current)
        prev_c = _centroid(previous) if previous.shape[0] else None
        if cur_c is not None and prev_c is not None:
            support_conf = float(np.clip(min(current.shape[0], previous.shape[0]) / max(float(self.config["min_event_count_for_motion_estimate"]), 1.0), 0.0, 1.0))
            return {
                "dx_px": cur_c[0] - prev_c[0],
                "dy_px": cur_c[1] - prev_c[1],
                "translation_method": "event_centroid",
                "translation_confidence": max(0.55, support_conf * 0.75),
                "translation_valid": True,
                "translation_invalid_reason": None,
            }
        return {
            "dx_px": None,
            "dy_px": None,
            "translation_method": "event_centroid",
            "translation_confidence": 0.0,
            "translation_valid": False,
            "translation_invalid_reason": "previous_window_not_available",
        }

    def _rotation(self, current, previous, roi, frame_size) -> dict[str, Any]:
        affine_angle, affine_conf = _feature_affine_rotation(previous, current, roi)
        if affine_angle is not None and affine_conf >= 0.60:
            return {
                "rotation_delta_deg": affine_angle,
                "rotation_method": "feature_affine_ransac",
                "rotation_confidence": affine_conf,
                "rotation_valid": True,
                "rotation_invalid_reason": None,
            }
        cur_angle, cur_conf = _pca_orientation(current)
        prev_angle, prev_conf = _pca_orientation(previous)
        if cur_angle is not None and prev_angle is not None:
            conf = min(cur_conf, prev_conf)
            if conf >= 0.25:
                return {
                    "rotation_delta_deg": _angle_diff_180(cur_angle, prev_angle),
                    "rotation_method": "event_cloud_pca_orientation",
                    "rotation_confidence": conf,
                    "rotation_valid": True,
                    "rotation_invalid_reason": None,
                }
        asym = _side_asymmetry(current, roi)
        return {
            "rotation_delta_deg": None,
            "rotation_method": "left_right_asymmetry_risk_only",
            "rotation_confidence": min(0.45, asym),
            "rotation_valid": False,
            "rotation_invalid_reason": "rotation_unavailable_risk_only",
        }

    def _vibration(self, arr, end, roi) -> dict[str, Any]:
        window_sec = float(self.config["stability_window_ms"]) / 1000.0
        starts = np.linspace(end - window_sec, end - float(self.config["short_window_ms"]) / 1000.0, 6)
        centroids = []
        rates = []
        for t0 in starts:
            win = _events_between(arr, t0, t0 + float(self.config["short_window_ms"]) / 1000.0, roi)
            if win.shape[0] >= 3:
                centroids.append(_centroid(win))
                rates.append(win.shape[0] / max(float(self.config["short_window_ms"]) / 1000.0, 1e-6))
        if len(centroids) < 3:
            return {
                "vibration_amplitude_px_rms": None,
                "vibration_frequency_hz": None,
                "repeated_burst_score": 0.0,
                "vibration_confidence": 0.0,
            }
        c = np.asarray(centroids, dtype=float)
        rms = _vibration_rms(c)
        diffs = np.diff(c, axis=0)
        signs = np.sign(diffs[:, 1] + 0.35 * diffs[:, 0])
        reversals = int(np.count_nonzero(signs[1:] * signs[:-1] < 0))
        frequency = reversals / max(2.0 * window_sec, 1e-6)
        burst = float(np.clip((np.std(rates) / max(np.mean(rates), 1.0)) if rates else 0.0, 0.0, 1.0))
        confidence = float(np.clip(len(centroids) / 6.0, 0.0, 1.0))
        return {
            "vibration_amplitude_px_rms": rms,
            "vibration_frequency_hz": frequency,
            "repeated_burst_score": burst,
            "vibration_confidence": confidence,
        }

    def _disturbance(self, current, full_current, roi, frame_size, translation, motion_window_ms=None) -> dict[str, Any]:
        width, height = frame_size
        full_count = int(full_current.shape[0])
        local_count = int(current.shape[0])
        outside = max(0, full_count - local_count)
        duration = max(float(motion_window_ms or self.config["motion_window_ms"]) / 1000.0, 1e-6)
        local_ratio = local_count / max(full_count, 1)
        uniformity = _spatial_grid_uniformity(full_current, frame_size)
        polarity_balance = abs((int(np.count_nonzero(full_current[:, 3] > 0)) - int(np.count_nonzero(full_current[:, 3] <= 0))) / max(full_count, 1)) if full_count else 0.0
        lighting = float(np.clip(uniformity * 0.85 + (polarity_balance * 0.30 if uniformity > 0.15 else polarity_balance * 0.15), 0.0, 1.0))
        camera_shake = float(np.clip(uniformity * (1.0 - min(1.0, local_ratio * 1.5)), 0.0, 1.0))
        background = float(np.clip((outside / max(full_count, 1)) if roi != _clip_roi(None, frame_size) else 0.0, 0.0, 1.0))
        object_motion = float(np.clip(local_ratio * (translation.get("translation_confidence") or 0.0), 0.0, 1.0))
        if lighting >= 0.65:
            dtype = "LIGHTING_CHANGE"
            dconf = lighting
        elif camera_shake >= 0.55:
            dtype = "CAMERA_SHAKE"
            dconf = camera_shake
        elif background >= 0.70 and local_ratio < 0.30:
            dtype = "BACKGROUND_MOTION"
            dconf = background
        elif object_motion >= 0.35:
            dtype = "OBJECT_MOTION"
            dconf = object_motion
        else:
            dtype = "UNKNOWN_MOTION"
            dconf = max(object_motion, background, camera_shake, lighting)
        return {
            "disturbance_type": dtype,
            "roi_confidence": 0.2 if roi == _clip_roi(None, frame_size) else 1.0,
            "global_event_ratio": full_count / max(full_count + local_count, 1),
            "local_to_global_event_ratio": local_ratio,
            "outside_roi_event_rate": outside / duration,
            "global_centroid_shift": None,
            "full_frame_uniformity": round(uniformity, 3),
            "lighting_flicker_score": round(lighting * 100.0, 3),
            "camera_shake_score": round(camera_shake * 100.0, 3),
            "background_motion_score": round(background * 100.0, 3),
            "object_motion_score": round(object_motion * 100.0, 3),
            "disturbance_confidence": round(dconf, 3),
            "disturbance_rejection_applied": dtype in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"},
        }

    def _estimate_confidence(self, support, translation, rotation, disturbance) -> float:
        support_conf = min(1.0, support["event_count"] / max(float(self.config["min_event_count_for_motion_estimate"]), 1.0))
        method_conf = float(translation.get("translation_confidence") or 0.0)
        object_conf = float(disturbance.get("object_motion_score") or 0.0) / 100.0
        if disturbance.get("disturbance_type") in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"}:
            object_conf *= 0.25
            method_conf *= 0.65
        return float(np.clip(0.45 * support_conf + 0.35 * method_conf + 0.20 * max(object_conf, 0.2), 0.0, 1.0))

    def _classify(self, *, valid, disturbance, displacement_above, downward, rotation_above, vibration_above, impact_score, confidence, rotation_valid) -> dict[str, Any]:
        cfg = self.config
        incipient = 0.0
        gross = 0.0
        rotation = 0.0
        vibration = 0.0
        motion_class = "STABLE"
        reason = "No motion above baseline."
        if disturbance["disturbance_type"] in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"}:
            motion_class = disturbance["disturbance_type"]
            reason = f"{motion_class}: disturbance score {disturbance['disturbance_confidence']:.2f}; object slip confidence reduced."
        elif valid:
            disp = float(displacement_above or 0.0)
            down = float(downward or 0.0)
            rot = float(rotation_above or 0.0)
            vib = float(vibration_above or 0.0)
            incipient = max(_score(disp, cfg["incipient_slip_px_threshold"], cfg["gross_slip_px_threshold"]), _score(down, cfg["incipient_slip_px_threshold"], cfg["gross_slip_px_threshold"]) * 0.85)
            incipient = max(incipient, _score(self.cumulative_downward_slip_px, cfg["incipient_slip_px_threshold"], cfg["cumulative_slip_px_threshold"]) * 0.85)
            gross = max(
                _score(disp, cfg["gross_slip_px_threshold"], cfg["cumulative_slip_px_threshold"]),
                _score(self.cumulative_slip_px, cfg["cumulative_slip_px_threshold"], cfg["cumulative_slip_px_threshold"] * 1.8),
            )
            rotation = _score(rot, cfg["rotation_deg_threshold"], cfg["gross_rotation_deg_threshold"]) if rotation_valid else 0.0
            vibration = _score(vib, cfg["vibration_rms_px_threshold"], cfg["vibration_rms_px_threshold"] * 3.0)
            if impact_score > 65.0 and disp < cfg["incipient_slip_px_threshold"] and vib < cfg["vibration_rms_px_threshold"]:
                motion_class = "IMPACT_ONLY"
                reason = f"Impact burst score {impact_score:.1f}; no persistent displacement above baseline."
            elif gross >= 55.0:
                motion_class = "GROSS_SLIP"
                reason = f"Persistent displacement {disp:.1f} px, cumulative net slip {self.cumulative_slip_px:.1f} px."
            elif rotation >= 45.0:
                motion_class = "SPIN_OR_ROTATION"
                reason = f"Estimated rotation {rot:.1f} deg above baseline."
            elif vibration >= 35.0 and gross < 45.0 and incipient < 30.0:
                motion_class = "SHAKE_OR_VIBRATION"
                reason = f"Vibration RMS {vib:.1f} px above baseline without persistent slip."
            elif incipient >= 30.0:
                motion_class = "INCIPIENT_SLIP"
                reason = f"Small displacement {disp:.1f} px above baseline; downward component {down:.1f} px."
            elif disp > 0.5 or rot > 0.5 or vib > 0.5:
                motion_class = "UNKNOWN_MOTION"
                reason = f"Motion evidence below tuned thresholds: displacement {disp:.1f} px, rotation {rot:.1f} deg, vibration {vib:.1f} px."
        else:
            motion_class = "UNKNOWN_MOTION"
            reason = "Motion estimate invalid or low confidence."
        slip_conf = float(np.clip(confidence * (max(incipient, gross) / 100.0 if max(incipient, gross) else 0.2), 0.0, 1.0))
        stability = float(np.clip(100.0 - max(impact_score * 0.35, incipient, gross, rotation, vibration), 0.0, 100.0))
        evidence = (
            f"dx/dy are measured in DAVIS pixels; no millimetre slip is claimed. "
            f"class={motion_class}, confidence={confidence:.2f}, reason={reason}"
        )
        return {
            "motion_class": motion_class,
            "reason": reason,
            "evidence": evidence,
            "impact_score": impact_score,
            "incipient_slip_score": incipient,
            "gross_slip_score": gross,
            "rotation_score": rotation,
            "vibration_score": vibration,
            "stability_score": stability,
            "slip_confidence": slip_conf,
        }


def _as_event_array(events: Any) -> np.ndarray:
    if events is None:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    return arr[:, :4]


def _clip_roi(roi_xywh, frame_size) -> list[int]:
    width, height = [int(v) for v in frame_size]
    if roi_xywh is None:
        return [0, 0, width, height]
    x, y, w, h = [int(round(float(v))) for v in roi_xywh]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return [x, y, w, h]


def _events_between(arr: np.ndarray, start: float, end: float, roi) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    mask = (arr[:, 0] >= float(start)) & (arr[:, 0] < float(end))
    if roi is not None:
        x, y, w, h = roi
        mask &= (arr[:, 1] >= x) & (arr[:, 1] < x + w) & (arr[:, 2] >= y) & (arr[:, 2] < y + h)
    return arr[mask]


def _event_image(events: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, w, h = roi
    img = np.zeros((h, w), dtype=np.float32)
    if events.shape[0] == 0:
        return img
    xs = np.clip(np.round(events[:, 1] - x).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(events[:, 2] - y).astype(np.int32), 0, h - 1)
    weights = np.where(events[:, 3] > 0, 1.0, 0.75).astype(np.float32)
    np.add.at(img, (ys, xs), weights)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _centroid(events: np.ndarray) -> Optional[list[float]]:
    if events.shape[0] == 0:
        return None
    return [float(np.mean(events[:, 1])), float(np.mean(events[:, 2]))]


def _pca_orientation(events: np.ndarray) -> tuple[Optional[float], float]:
    if events.shape[0] < 6:
        return None, 0.0
    pts = events[:, 1:3].astype(np.float64)
    centered = pts - np.mean(pts, axis=0)
    cov = np.cov(centered.T)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return None, 0.0
    order = np.argsort(vals)
    major = vecs[:, order[-1]]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    denom = max(float(vals[order[-1]] + vals[order[-2]]), 1e-6)
    confidence = float(np.clip((float(vals[order[-1]]) - float(vals[order[-2]])) / denom, 0.0, 1.0))
    return angle, confidence


def _feature_affine_rotation(previous: np.ndarray, current: np.ndarray, roi: list[int]) -> tuple[Optional[float], float]:
    if previous.shape[0] < 20 or current.shape[0] < 20:
        return None, 0.0
    prev_img = _event_image(previous, roi).astype(np.uint8)
    curr_img = _event_image(current, roi).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(prev_img, maxCorners=80, qualityLevel=0.01, minDistance=3)
    if pts is None or len(pts) < 8:
        return None, 0.0
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, pts, None)
    if next_pts is None or status is None:
        return None, 0.0
    good_prev = pts[status.reshape(-1) == 1].reshape(-1, 2)
    good_next = next_pts[status.reshape(-1) == 1].reshape(-1, 2)
    if good_prev.shape[0] < 8:
        return None, 0.0
    affine, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine is None:
        return None, 0.0
    angle = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    confidence = float(np.mean(inliers)) if inliers is not None and inliers.size else 0.0
    return angle, confidence


def _side_asymmetry(events: np.ndarray, roi: list[int]) -> float:
    if events.shape[0] == 0:
        return 0.0
    x, _y, w, _h = roi
    left = int(np.count_nonzero(events[:, 1] < x + w / 2.0))
    right = int(events.shape[0] - left)
    return abs(right - left) / max(events.shape[0], 1)


def _spatial_grid_uniformity(events: np.ndarray, frame_size, grid_shape=(8, 6)) -> float:
    """Return broad spatial occupancy/evenness, not raw active-pixel count.

    A dense local object edge may activate hundreds of unique pixels without
    resembling a full-frame lighting change. Coarse-grid occupancy multiplied
    by normalized entropy distinguishes that case from activity distributed
    across the sensor.
    """
    if events.shape[0] == 0:
        return 0.0
    width, height = [max(int(value), 1) for value in frame_size]
    grid_x, grid_y = [max(int(value), 1) for value in grid_shape]
    xs = np.clip((events[:, 1] * grid_x / width).astype(np.int32), 0, grid_x - 1)
    ys = np.clip((events[:, 2] * grid_y / height).astype(np.int32), 0, grid_y - 1)
    counts = np.bincount(ys * grid_x + xs, minlength=grid_x * grid_y).astype(float)
    occupied = counts > 0.0
    occupied_ratio = float(np.count_nonzero(occupied) / counts.size)
    probabilities = counts[counts > 0.0] / max(float(np.sum(counts)), 1.0)
    if counts.size <= 1:
        entropy = 1.0
    else:
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(counts.size))
    return float(np.clip(occupied_ratio * entropy, 0.0, 1.0))


def _angle_diff_180(current: float, previous: float) -> float:
    diff = (float(current) - float(previous) + 90.0) % 180.0 - 90.0
    return float(diff)


def _angle_std_deg(angles: list[float]) -> float:
    if len(angles) < 2:
        return 0.0
    doubled = np.deg2rad(np.asarray(angles, dtype=float) * 2.0)
    mean = math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled)))) / 2.0
    diffs = [abs(_angle_diff_180(a, math.degrees(mean))) for a in angles]
    return float(np.std(diffs))


def _vibration_rms(centroids: np.ndarray) -> float:
    if len(centroids) < 2:
        return 0.0
    c = np.asarray(centroids, dtype=float)
    return float(np.sqrt(np.mean(np.sum((c - np.mean(c, axis=0)) ** 2, axis=1))))


def _normalize_axis(axis) -> Optional[list[float]]:
    if axis is None:
        return None
    try:
        x = float(axis[0])
        y = float(axis[1])
    except Exception:
        return None
    norm = math.hypot(x, y)
    if norm <= 1e-6:
        return None
    if y < 0:
        x, y = -x, -y
    return [x / norm, y / norm]


def _score(value: float, low: float, high: float) -> float:
    return float(np.clip(100.0 * (float(value) - float(low)) / max(float(high) - float(low), 1e-6), 0.0, 100.0))


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)


__all__ = [
    "DEFAULT_DAVIS_MOTION_CONFIG",
    "DavisMotionEstimate",
    "DavisMotionEstimator",
    "load_davis_motion_config",
]

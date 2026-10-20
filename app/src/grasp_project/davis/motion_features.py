"""Camera-free DAVIS event feature extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from grasp_project.davis.event_stream import sensitivity_thresholds


@dataclass
class EventBaseline:
    """Quiet-scene baseline used to normalize event feature values."""

    baseline_calibrated: bool
    baseline_event_rate: float = 0.0
    baseline_polarity_balance: float = 0.0
    baseline_event_density: float = 0.0
    baseline_spatial_spread: float = 0.0
    baseline_left_right_asymmetry: float = 0.0
    baseline_top_bottom_asymmetry: float = 0.0
    calibration_quality: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_calibrated": self.baseline_calibrated,
            "baseline_event_rate": self.baseline_event_rate,
            "baseline_polarity_balance": self.baseline_polarity_balance,
            "baseline_event_density": self.baseline_event_density,
            "baseline_spatial_spread": self.baseline_spatial_spread,
            "baseline_left_right_asymmetry": self.baseline_left_right_asymmetry,
            "baseline_top_bottom_asymmetry": self.baseline_top_bottom_asymmetry,
            "calibration_quality": self.calibration_quality,
        }


def _feature(value, *, confidence=1.0, window_ms=100.0, baseline=None, valid=True, invalid_reason=None):
    base_value = None if baseline is None else float(baseline)
    if base_value is None:
        normalized = None
    else:
        normalized = float(value) - base_value
    return {
        "value": _safe_float(value),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "window_ms": float(window_ms),
        "baseline_normalized": None if normalized is None else _safe_float(normalized),
        "valid": bool(valid),
        "invalid_reason": invalid_reason,
    }


def _safe_float(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None:
        return None
    return float(value)


def _empty_feature_set(window_ms: float, reason: str) -> dict[str, Any]:
    names = [
        "event_count",
        "event_rate",
        "event_density",
        "active_pixel_ratio",
        "positive_event_count",
        "negative_event_count",
        "polarity_balance",
        "centroid_x",
        "centroid_y",
        "centroid_shift_x",
        "centroid_shift_y",
        "spatial_spread_x",
        "spatial_spread_y",
        "vertical_bias",
        "horizontal_bias",
        "left_right_asymmetry",
        "top_bottom_asymmetry",
        "downward_motion_score",
        "lateral_motion_score",
        "burst_score",
        "persistence_score",
        "decay_ratio",
        "repeated_burst_score",
        "event_frequency_estimate",
        "local_to_global_ratio",
        "camera_shake_score",
        "lighting_flicker_score",
        "background_motion_score",
        "object_motion_score",
    ]
    return {
        "valid": False,
        "invalid_reason": reason,
        "window_ms": float(window_ms),
        "event_motion_class": "STREAM_ACTIVE_NO_MOTION" if reason == "no_events" else "SENSOR_UNAVAILABLE",
        "motion_bbox": None,
        **{name: _feature(0.0, window_ms=window_ms, confidence=0.0, valid=False, invalid_reason=reason) for name in names},
    }


def event_features_for_window(
    events: Any,
    *,
    window_ms: float = 100.0,
    frame_size=(346, 260),
    baseline: dict[str, Any] | EventBaseline | None = None,
    roi_xywh=None,
) -> dict[str, Any]:
    """Return raw DAVIS event features with validity and baseline metadata.

    ``events`` must be an Nx4 array ordered as ``[timestamp, x, y, polarity]``.
    The helper is intentionally camera-free so it can be unit tested and reused
    by the dashboard, synchronization layer, and future benchmark code.
    """

    arr = np.asarray(events, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return _empty_feature_set(window_ms, "invalid_event_array")
    if arr.shape[0] == 0:
        return _empty_feature_set(window_ms, "no_events")
    arr = arr[:, :4]
    duration_sec = max(float(window_ms) / 1000.0, 1e-6)
    width, height = [float(v) for v in frame_size]
    frame_area = max(width * height, 1.0)
    xs = np.clip(arr[:, 1], 0.0, width - 1.0)
    ys = np.clip(arr[:, 2], 0.0, height - 1.0)
    pol = arr[:, 3] > 0
    count = int(arr.shape[0])
    positive = int(np.count_nonzero(pol))
    negative = int(count - positive)
    event_rate = count / duration_sec
    active_pixels = len(set(zip(xs.astype(int).tolist(), ys.astype(int).tolist())))
    event_density = count / frame_area
    active_pixel_ratio = active_pixels / frame_area
    centroid_x = float(np.mean(xs))
    centroid_y = float(np.mean(ys))
    spread_x = float(np.std(xs))
    spread_y = float(np.std(ys))
    bbox = [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]
    left = int(np.count_nonzero(xs < width / 2.0))
    right = int(count - left)
    top = int(np.count_nonzero(ys < height / 2.0))
    bottom = int(count - top)
    polarity_balance = (positive - negative) / max(count, 1)
    horizontal_bias = (right - left) / max(count, 1)
    vertical_bias = (bottom - top) / max(count, 1)
    timestamps = arr[:, 0]
    burst_score = _burst_score(timestamps, duration_sec)
    persistence_score = float(np.clip(len(np.unique(np.floor((timestamps - np.min(timestamps)) * 1000.0 / 10.0))) / max(window_ms / 10.0, 1.0), 0.0, 1.0))
    decay_ratio = _decay_ratio(timestamps)
    repeated_burst_score = float(np.clip(burst_score * persistence_score, 0.0, 1.0))
    frequency = _frequency_estimate(timestamps)
    roi_stats = _roi_stats(arr, roi_xywh, frame_size)
    baseline_dict = baseline.to_dict() if isinstance(baseline, EventBaseline) else (baseline or {})
    baseline_rate = baseline_dict.get("baseline_event_rate")
    baseline_density = baseline_dict.get("baseline_event_density")
    confidence = float(np.clip(min(1.0, count / 80.0), 0.05, 1.0))
    lighting_score = float(np.clip(abs(polarity_balance) * active_pixel_ratio * 20.0, 0.0, 1.0))
    shake_score = float(np.clip(active_pixel_ratio * 120.0 * (1.0 - min(1.0, roi_stats["local_to_global_ratio"])), 0.0, 1.0))
    object_motion_score = float(np.clip(roi_stats["local_to_global_ratio"] * min(1.0, event_rate / 1000.0), 0.0, 1.0))
    features = {
        "valid": True,
        "invalid_reason": None,
        "window_ms": float(window_ms),
        "event_motion_class": "EVENT_MOTION",
        "motion_bbox": bbox,
        "event_count": _feature(count, window_ms=window_ms, confidence=1.0, baseline=0.0),
        "event_rate": _feature(event_rate, window_ms=window_ms, confidence=confidence, baseline=baseline_rate),
        "event_density": _feature(event_density, window_ms=window_ms, confidence=confidence, baseline=baseline_density),
        "active_pixel_ratio": _feature(active_pixel_ratio, window_ms=window_ms, confidence=confidence),
        "positive_event_count": _feature(positive, window_ms=window_ms, confidence=confidence),
        "negative_event_count": _feature(negative, window_ms=window_ms, confidence=confidence),
        "polarity_balance": _feature(polarity_balance, window_ms=window_ms, confidence=confidence, baseline=baseline_dict.get("baseline_polarity_balance")),
        "centroid_x": _feature(centroid_x, window_ms=window_ms, confidence=confidence),
        "centroid_y": _feature(centroid_y, window_ms=window_ms, confidence=confidence),
        "centroid_shift_x": _feature(0.0, window_ms=window_ms, confidence=0.0, valid=False, invalid_reason="previous_window_not_supplied"),
        "centroid_shift_y": _feature(0.0, window_ms=window_ms, confidence=0.0, valid=False, invalid_reason="previous_window_not_supplied"),
        "spatial_spread_x": _feature(spread_x, window_ms=window_ms, confidence=confidence),
        "spatial_spread_y": _feature(spread_y, window_ms=window_ms, confidence=confidence),
        "vertical_bias": _feature(vertical_bias, window_ms=window_ms, confidence=confidence, baseline=baseline_dict.get("baseline_top_bottom_asymmetry")),
        "horizontal_bias": _feature(horizontal_bias, window_ms=window_ms, confidence=confidence, baseline=baseline_dict.get("baseline_left_right_asymmetry")),
        "left_right_asymmetry": _feature(abs(horizontal_bias), window_ms=window_ms, confidence=confidence, baseline=abs(float(baseline_dict.get("baseline_left_right_asymmetry", 0.0)))),
        "top_bottom_asymmetry": _feature(abs(vertical_bias), window_ms=window_ms, confidence=confidence, baseline=abs(float(baseline_dict.get("baseline_top_bottom_asymmetry", 0.0)))),
        "downward_motion_score": _feature(max(0.0, vertical_bias), window_ms=window_ms, confidence=confidence),
        "lateral_motion_score": _feature(abs(horizontal_bias), window_ms=window_ms, confidence=confidence),
        "burst_score": _feature(burst_score, window_ms=window_ms, confidence=confidence),
        "persistence_score": _feature(persistence_score, window_ms=window_ms, confidence=confidence),
        "decay_ratio": _feature(decay_ratio, window_ms=window_ms, confidence=confidence),
        "repeated_burst_score": _feature(repeated_burst_score, window_ms=window_ms, confidence=confidence),
        "event_frequency_estimate": _feature(frequency, window_ms=window_ms, confidence=confidence),
        "target_roi_event_rate": _feature(roi_stats["target_roi_event_rate"], window_ms=window_ms, confidence=roi_stats["roi_confidence"]),
        "outside_roi_event_rate": _feature(roi_stats["outside_roi_event_rate"], window_ms=window_ms, confidence=roi_stats["roi_confidence"]),
        "local_to_global_ratio": _feature(roi_stats["local_to_global_ratio"], window_ms=window_ms, confidence=roi_stats["roi_confidence"]),
        "camera_shake_score": _feature(shake_score, window_ms=window_ms, confidence=confidence),
        "lighting_flicker_score": _feature(lighting_score, window_ms=window_ms, confidence=confidence),
        "background_motion_score": _feature(max(0.0, 1.0 - roi_stats["local_to_global_ratio"]), window_ms=window_ms, confidence=roi_stats["roi_confidence"]),
        "object_motion_score": _feature(object_motion_score, window_ms=window_ms, confidence=roi_stats["roi_confidence"]),
        "disturbance_type": _classify_disturbance(shake_score, lighting_score, object_motion_score),
    }
    return features


def calibrate_event_baseline(events: Any, *, duration_sec: float, frame_size=(346, 260)) -> EventBaseline:
    arr = np.asarray(events, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return EventBaseline(False, calibration_quality="no_events")
    window_ms = max(float(duration_sec) * 1000.0, 1.0)
    features = event_features_for_window(arr, window_ms=window_ms, frame_size=frame_size)
    rate = features["event_rate"]["value"]
    density = features["event_density"]["value"]
    polarity = features["polarity_balance"]["value"]
    spread = float(np.hypot(features["spatial_spread_x"]["value"], features["spatial_spread_y"]["value"]))
    lr = features["horizontal_bias"]["value"]
    tb = features["vertical_bias"]["value"]
    quality = "quiet" if rate < 200.0 else ("active_scene" if rate < 1000.0 else "too_much_motion")
    return EventBaseline(
        baseline_calibrated=quality != "too_much_motion",
        baseline_event_rate=rate,
        baseline_polarity_balance=polarity,
        baseline_event_density=density,
        baseline_spatial_spread=spread,
        baseline_left_right_asymmetry=lr,
        baseline_top_bottom_asymmetry=tb,
        calibration_quality=quality,
    )


def _burst_score(timestamps: np.ndarray, duration_sec: float) -> float:
    if timestamps.size < 2:
        return 0.0
    bins = max(1, int(np.ceil(duration_sec / 0.005)))
    hist, _ = np.histogram(timestamps, bins=bins)
    return float(np.clip(np.max(hist) / max(np.mean(hist), 1e-6) / 10.0, 0.0, 1.0))


def _decay_ratio(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return 0.0
    midpoint = (float(np.min(timestamps)) + float(np.max(timestamps))) / 2.0
    early = int(np.count_nonzero(timestamps <= midpoint))
    late = int(np.count_nonzero(timestamps > midpoint))
    return float(late / max(early, 1))


def _frequency_estimate(timestamps: np.ndarray) -> float:
    if timestamps.size < 3:
        return 0.0
    deltas = np.diff(np.sort(timestamps))
    median_delta = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
    return 0.0 if median_delta <= 0.0 else float(1.0 / median_delta)


def _roi_stats(arr: np.ndarray, roi_xywh, frame_size) -> dict[str, float]:
    duration = max(float(np.max(arr[:, 0]) - np.min(arr[:, 0])), 1e-3)
    if not roi_xywh:
        return {
            "target_roi_event_rate": float(arr.shape[0] / duration),
            "outside_roi_event_rate": 0.0,
            "local_to_global_ratio": 1.0,
            "roi_confidence": 0.2,
        }
    x, y, w, h = [float(v) for v in roi_xywh]
    xs = arr[:, 1]
    ys = arr[:, 2]
    inside = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
    inside_count = int(np.count_nonzero(inside))
    outside_count = int(arr.shape[0] - inside_count)
    return {
        "target_roi_event_rate": inside_count / duration,
        "outside_roi_event_rate": outside_count / duration,
        "local_to_global_ratio": inside_count / max(arr.shape[0], 1),
        "roi_confidence": 1.0,
    }


def _classify_disturbance(shake_score: float, lighting_score: float, object_motion_score: float) -> str:
    if shake_score > 0.65:
        return "CAMERA_SHAKE"
    if lighting_score > 0.65:
        return "LIGHTING_CHANGE"
    if object_motion_score > 0.2:
        return "OBJECT_MOTION"
    return "UNKNOWN_MOTION"


__all__ = [
    "EventBaseline",
    "calibrate_event_baseline",
    "event_features_for_window",
    "sensitivity_thresholds",
]

"""Camera-free spatio-temporal event search along an object boundary.

The search consumes an already-mapped boundary and an existing DAVIS event
buffer.  It does not open hardware or claim that the RealSense-to-DAVIS
mapping is calibrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

import numpy as np

from grasp_project.davis.event_stream import _clip_roi


@dataclass(frozen=True)
class BoundaryBandDescriptor:
    band_index: int
    center_px: list[float]
    tangent: list[float]
    normal: list[float]
    lower_band: bool
    contact_band: bool
    event_count: int
    density_events_px2: float
    polarity_balance: float
    centroid_px: Optional[list[float]]
    tangential_flow_px: float
    normal_flow_px: float
    vertical_flow_px: float
    velocity_px_s: float
    acceleration_px_s2: float
    oscillation_energy_px: float
    temporal_event_counts: list[int]
    temporal_centroids_px: list[Optional[list[float]]]


@dataclass(frozen=True)
class BoundaryEventField:
    bands: list[BoundaryBandDescriptor]
    boundary_source: str
    alignment_status: str
    time_window_sec: float
    temporal_subwindows: int
    event_count: int
    assigned_event_count: int
    raw_window_event_count: int
    active_band_count: int
    contact_band_count: int
    contact_band_active_count: int
    contact_band_event_count: int
    contact_tangential_flow_px: float
    contact_normal_flow_px: float
    contact_vertical_drift_px: float
    contact_oscillation_energy_px: float
    contact_incipient_slip_precursor_score: float
    contact_band_measurement_status: str
    tangential_circulation_px: float
    lower_vertical_drift_px: float
    oscillation_energy_px: float
    raw_rotation_cue: float
    raw_slip_cue: float
    raw_vibration_cue: float
    raw_incipient_slip_precursor_score: float
    rotation_cue: float
    slip_cue: float
    vibration_cue: float
    incipient_slip_precursor_score: float
    object_motion_valid: Optional[bool]
    suppression_reason: Optional[str]
    measurement_status: str
    physically_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def boundary_from_roi(roi_xywh, frame_size=(346, 260)) -> np.ndarray:
    """Return the clipped rectangular boundary for an existing mapped ROI."""
    x, y, w, h = _clip_roi(roi_xywh, frame_size)
    return np.asarray(
        [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
        dtype=float,
    )


def apply_object_motion_validity(
    field: BoundaryEventField,
    object_motion_valid: bool,
    reason: Optional[str] = None,
) -> BoundaryEventField:
    """Suppress actionable cues when the canonical object-motion gate rejects."""
    if object_motion_valid:
        return replace(
            field,
            object_motion_valid=True,
            suppression_reason=None,
            measurement_status="ESTIMATED_OBJECT_LOCAL_UNCALIBRATED",
        )
    return replace(
        field,
        rotation_cue=0.0,
        slip_cue=0.0,
        vibration_cue=0.0,
        incipient_slip_precursor_score=0.0,
        contact_tangential_flow_px=0.0,
        contact_normal_flow_px=0.0,
        contact_vertical_drift_px=0.0,
        contact_oscillation_energy_px=0.0,
        contact_incipient_slip_precursor_score=0.0,
        contact_band_measurement_status="SUPPRESSED_BY_OBJECT_MOTION_VALIDITY",
        object_motion_valid=False,
        suppression_reason=reason or "object_motion_invalid",
        measurement_status="SUPPRESSED_BY_OBJECT_MOTION_VALIDITY",
    )


def search_boundary_events(
    boundary,
    events,
    time_window_sec: float,
    *,
    now: Optional[float] = None,
    band_count: int = 24,
    temporal_subwindows: int = 4,
    normal_half_width_px: float = 5.0,
    boundary_source: str = "mapped_boundary",
    alignment_status: str = "APPROXIMATE_UNCALIBRATED",
    vertical_slip_sign: float = 1.0,
    object_motion_valid: Optional[bool] = None,
    suppression_reason: Optional[str] = None,
    contact_band_polygon=None,
) -> BoundaryEventField:
    """Estimate local boundary motion using oriented spatial and time bands."""
    samples, tangents, normals, arc_step = _sample_boundary(boundary, band_count)
    arr = _event_array(events)
    window = max(float(time_window_sec), 1e-3)
    end = float(now if now is not None else (np.max(arr[:, 0]) if arr.size else 0.0))
    arr = arr[(arr[:, 0] >= end - window) & (arr[:, 0] <= end)] if arr.size else arr
    assignments = _assign_events(arr, samples, tangents, arc_step * 0.75, normal_half_width_px)
    lower_threshold = float(np.quantile(samples[:, 1], 0.65))
    contact_selection = _points_in_polygon(samples, contact_band_polygon)
    descriptors = []
    for index, (center, tangent, normal) in enumerate(zip(samples, tangents, normals)):
        band_events = arr[assignments == index] if arr.size else arr
        descriptors.append(
            _describe_band(
                index,
                center,
                tangent,
                normal,
                band_events,
                end,
                window,
                temporal_subwindows,
                arc_step * 1.5 * normal_half_width_px * 2.0,
                bool(center[1] >= lower_threshold),
                bool(contact_selection[index]),
            )
        )

    counts = np.asarray([band.event_count for band in descriptors], dtype=float)
    weights = np.maximum(counts, 1.0)
    active = counts > 0
    tangential = _weighted_mean([band.tangential_flow_px for band in descriptors], weights, active)
    lower = np.asarray([band.lower_band and band.event_count > 0 for band in descriptors], dtype=bool)
    lower_vertical = _weighted_mean([band.vertical_flow_px for band in descriptors], weights, lower) * float(vertical_slip_sign)
    oscillation = _weighted_mean([band.oscillation_energy_px for band in descriptors], weights, active)
    contact = np.asarray([band.contact_band for band in descriptors], dtype=bool)
    contact_active = contact & active
    contact_descriptors = [band for band in descriptors if band.contact_band]
    contact_tangential = _weighted_mean([band.tangential_flow_px for band in descriptors], weights, contact_active)
    contact_normal = _weighted_mean([band.normal_flow_px for band in descriptors], weights, contact_active)
    contact_vertical = _weighted_mean([band.vertical_flow_px for band in descriptors], weights, contact_active) * float(vertical_slip_sign)
    contact_oscillation = _weighted_mean([band.oscillation_energy_px for band in descriptors], weights, contact_active)
    contact_precursor = _precursor_score(contact_descriptors) if contact_descriptors else 0.0

    raw_rotation = float(np.clip(abs(tangential) * 25.0, 0.0, 100.0))
    raw_slip = float(np.clip(max(0.0, lower_vertical) * 40.0, 0.0, 100.0))
    raw_vibration = float(np.clip(oscillation * 22.0, 0.0, 100.0))
    raw_precursor = _precursor_score(descriptors)
    assigned_event_count = int(np.sum(counts))
    field = BoundaryEventField(
        bands=descriptors,
        boundary_source=str(boundary_source),
        alignment_status=str(alignment_status),
        time_window_sec=window,
        temporal_subwindows=max(2, int(temporal_subwindows)),
        # ``event_count`` remains the support count used by existing
        # consumers, but now honestly means events assigned to a boundary
        # neighborhood.  Full-window/global activity is reported separately.
        event_count=assigned_event_count,
        assigned_event_count=assigned_event_count,
        raw_window_event_count=int(arr.shape[0]),
        active_band_count=int(np.count_nonzero(active)),
        contact_band_count=int(np.count_nonzero(contact)),
        contact_band_active_count=int(np.count_nonzero(contact_active)),
        contact_band_event_count=int(np.sum(counts[contact])) if np.any(contact) else 0,
        contact_tangential_flow_px=round(contact_tangential, 4),
        contact_normal_flow_px=round(contact_normal, 4),
        contact_vertical_drift_px=round(contact_vertical, 4),
        contact_oscillation_energy_px=round(contact_oscillation, 4),
        contact_incipient_slip_precursor_score=round(contact_precursor, 3),
        contact_band_measurement_status="ESTIMATED_APPROXIMATE_UNCALIBRATED" if np.any(contact) else "NOT_AVAILABLE",
        tangential_circulation_px=round(tangential, 4),
        lower_vertical_drift_px=round(lower_vertical, 4),
        oscillation_energy_px=round(oscillation, 4),
        raw_rotation_cue=round(raw_rotation, 3),
        raw_slip_cue=round(raw_slip, 3),
        raw_vibration_cue=round(raw_vibration, 3),
        raw_incipient_slip_precursor_score=round(raw_precursor, 3),
        rotation_cue=round(raw_rotation, 3),
        slip_cue=round(raw_slip, 3),
        vibration_cue=round(raw_vibration, 3),
        incipient_slip_precursor_score=round(raw_precursor, 3),
        object_motion_valid=None,
        suppression_reason=None,
        measurement_status="ESTIMATED_NOT_GATED",
    )
    if object_motion_valid is not None:
        field = apply_object_motion_validity(field, object_motion_valid, suppression_reason)
    return field


def _event_array(events) -> np.ndarray:
    arr = np.asarray(events, dtype=float)
    if arr.size == 0:
        return np.empty((0, 4), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("events must have columns (t, x, y, polarity)")
    return arr[:, :4]


def _sample_boundary(boundary, band_count: int):
    points = np.asarray(boundary, dtype=float).reshape(-1, 2)
    if points.shape[0] < 3:
        raise ValueError("boundary requires at least three points")
    if np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    closed = np.vstack([points, points[0]])
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-6
    if np.count_nonzero(keep) < 3:
        raise ValueError("boundary requires at least three distinct points")
    points = points[keep]
    closed = np.vstack([points, points[0]])
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    perimeter = float(cumulative[-1])
    n = max(8, int(band_count))
    distances = np.arange(n, dtype=float) * perimeter / n
    samples = []
    for distance in distances:
        idx = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(lengths) - 1)
        fraction = (distance - cumulative[idx]) / max(lengths[idx], 1e-9)
        samples.append(closed[idx] + fraction * segments[idx])
    samples = np.asarray(samples, dtype=float)
    tangents = np.roll(samples, -1, axis=0) - np.roll(samples, 1, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return samples, tangents, normals, perimeter / n


def _points_in_polygon(points, polygon):
    """Vectorized ray casting; keeps this camera-free module NumPy-only."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if polygon is None:
        return np.zeros(points.shape[0], dtype=bool)
    try:
        poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return np.zeros(points.shape[0], dtype=bool)
    if poly.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=bool)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    previous = poly[-1]
    for current in poly:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= crosses & (x <= x_cross)
        previous = current
    return inside


def _assign_events(events, samples, tangents, tangent_half_width, normal_half_width):
    if events.size == 0:
        return np.empty(0, dtype=int)
    delta = events[:, None, 1:3] - samples[None, :, :]
    along = np.abs(np.sum(delta * tangents[None, :, :], axis=2))
    across = np.abs(np.sum(delta * np.column_stack([-tangents[:, 1], tangents[:, 0]])[None, :, :], axis=2))
    cost = (along / max(tangent_half_width, 1e-6)) ** 2 + (across / max(normal_half_width, 1e-6)) ** 2
    nearest = np.argmin(cost, axis=1)
    best = cost[np.arange(cost.shape[0]), nearest]
    nearest[best > 1.0] = -1
    return nearest


def _describe_band(index, center, tangent, normal, events, end, window, subwindows, area, lower_band, contact_band):
    k = max(2, int(subwindows))
    edges = np.linspace(end - window, end, k + 1)
    counts = []
    centroids = []
    for start, stop in zip(edges[:-1], edges[1:]):
        selected = events[(events[:, 0] >= start) & (events[:, 0] <= stop)]
        counts.append(int(selected.shape[0]))
        centroids.append(np.mean(selected[:, 1:3], axis=0) if selected.shape[0] else None)
    flows = []
    for previous, current in zip(centroids[:-1], centroids[1:]):
        if previous is not None and current is not None:
            flows.append(np.asarray(current) - np.asarray(previous))
    flow_arr = np.asarray(flows, dtype=float) if flows else np.empty((0, 2), dtype=float)
    mean_flow = np.mean(flow_arr, axis=0) if flow_arr.size else np.zeros(2, dtype=float)
    bin_sec = window / k
    velocities = flow_arr / max(bin_sec, 1e-6) if flow_arr.size else flow_arr
    speed = float(np.linalg.norm(np.mean(velocities, axis=0))) if velocities.size else 0.0
    acceleration = float(np.linalg.norm(velocities[-1] - velocities[0]) / max(bin_sec * max(len(velocities) - 1, 1), 1e-6)) if len(velocities) >= 2 else 0.0
    residual = flow_arr - np.mean(flow_arr, axis=0) if flow_arr.size else flow_arr
    oscillation = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))) if residual.size else 0.0
    polarity = float(np.mean(np.where(events[:, 3] > 0, 1.0, -1.0))) if events.shape[0] else 0.0
    centroid = np.mean(events[:, 1:3], axis=0).tolist() if events.shape[0] else None
    return BoundaryBandDescriptor(
        band_index=index,
        center_px=[round(float(v), 3) for v in center],
        tangent=[round(float(v), 5) for v in tangent],
        normal=[round(float(v), 5) for v in normal],
        lower_band=lower_band,
        contact_band=contact_band,
        event_count=int(events.shape[0]),
        density_events_px2=round(float(events.shape[0]) / max(float(area), 1.0), 5),
        polarity_balance=round(polarity, 4),
        centroid_px=None if centroid is None else [round(float(v), 3) for v in centroid],
        tangential_flow_px=round(float(np.dot(mean_flow, tangent)), 4),
        normal_flow_px=round(float(np.dot(mean_flow, normal)), 4),
        vertical_flow_px=round(float(mean_flow[1]), 4),
        velocity_px_s=round(speed, 3),
        acceleration_px_s2=round(acceleration, 3),
        oscillation_energy_px=round(oscillation, 4),
        temporal_event_counts=counts,
        temporal_centroids_px=[None if value is None else [round(float(v), 3) for v in value] for value in centroids],
    )


def _weighted_mean(values, weights, selection):
    values = np.asarray(values, dtype=float)
    selection = np.asarray(selection, dtype=bool)
    if not np.any(selection):
        return 0.0
    return float(np.average(values[selection], weights=np.asarray(weights, dtype=float)[selection]))


def _precursor_score(bands):
    active = [band for band in bands if band.event_count > 0]
    if not active or len(active) / max(len(bands), 1) > 0.45:
        return 0.0
    candidates = []
    total_events = sum(band.event_count for band in active)
    for band in active:
        counts = np.asarray(band.temporal_event_counts, dtype=float)
        if counts.size < 3 or counts[-1] <= counts[0] or np.count_nonzero(counts) < 2:
            continue
        growth = (counts[-1] - counts[0]) / max(float(np.max(counts)), 1.0)
        monotonic = float(np.mean(np.diff(counts) >= 0.0))
        concentration = band.event_count / max(total_events, 1)
        candidates.append(100.0 * growth * monotonic * min(1.0, concentration * 2.0))
    return float(np.clip(max(candidates, default=0.0), 0.0, 100.0))

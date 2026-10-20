"""Software timestamp synchronization for RealSense RGB-D, point cloud, and DAVIS events."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grasp_project.common.config import OUTPUT_DIR, config_path, ensure_output_dir, get_env_float, load_config_file
from grasp_project.common.file_io import read_json, write_json_atomic
from grasp_project.sync.calibration_status import calibration_status

SYNC_ENABLED = True
SYNC_MODE_DISPLAY_LATEST = "display_latest"
SYNC_MODE_SOFTWARE_TIMESTAMP = "software_timestamp"
SYNC_MODE_ACTION_ANCHORED_WINDOW = "action_anchored_window"
SYNC_MODES = {SYNC_MODE_DISPLAY_LATEST, SYNC_MODE_SOFTWARE_TIMESTAMP, SYNC_MODE_ACTION_ANCHORED_WINDOW}
SYNC_MAX_OFFSET_MS = get_env_float("SYNC_MAX_OFFSET_MS", 50.0)
DAVIS_EVENT_WINDOW_MS = get_env_float("DAVIS_EVENT_WINDOW_MS", 100.0)
SYNC_TARGET_FPS = get_env_float("SYNC_TARGET_FPS", 10.0)
SYNC_RING_BUFFER_SEC = get_env_float("SYNC_RING_BUFFER_SEC", 5.0)
REALSENSE_MAX_AGE_MS = get_env_float("REALSENSE_MAX_AGE_MS", 750.0)
DAVIS_EVENT_MAX_AGE_MS = get_env_float("DAVIS_EVENT_MAX_AGE_MS", 100.0)
DAVIS_APS_MAX_AGE_MS = get_env_float("DAVIS_APS_MAX_AGE_MS", 250.0)
POINTCLOUD_MAX_AGE_SEC = get_env_float("POINTCLOUD_MAX_AGE_SEC", 2.0)
BENCHMARK_MAX_SYNC_OFFSET_MS = get_env_float("BENCHMARK_MAX_SYNC_OFFSET_MS", 200.0)
LIVE_QUALITY_MAX_SYNC_OFFSET_MS = get_env_float("LIVE_QUALITY_MAX_SYNC_OFFSET_MS", 150.0)

SYNC_STATUS_PATH = OUTPUT_DIR / "sync_status.json"
LATEST_BUNDLE_PATH = OUTPUT_DIR / "latest_sensor_bundle.json"
SYNC_REPORT_PATH = OUTPUT_DIR / "sensor_sync_report.json"


@dataclass
class SensorFrameBundle:
    bundle_id: str
    system_timestamp: float
    bundle_timestamp: float | None = None
    realsense_rgb_frame_id: int | None = None
    realsense_rgb_timestamp: float | None = None
    realsense_depth_frame_id: int | None = None
    realsense_depth_timestamp: float | None = None
    pointcloud_timestamp: float | None = None
    davis_aps_timestamp: float | None = None
    davis_event_window_start: float | None = None
    davis_event_window_end: float | None = None
    davis_event_count: int = 0
    davis_event_rate: float = 0.0
    sync_latency_ms: float | None = None
    max_sensor_time_offset_ms: float | None = None
    sync_quality: str = "unknown"
    davis_realsense_sync_quality: str = "unknown"
    davis_realsense_valid_for_live_quality: bool = False
    sync_mode: str = "software_timestamp"
    stale_modalities: list[str] = field(default_factory=list)
    unavailable_modalities: list[str] = field(default_factory=list)
    bundle_valid_for_benchmark: bool = False
    bundle_valid_for_live_quality: bool = False
    sensor_freshness: dict[str, Any] = field(default_factory=dict)
    freshness_gates: dict[str, Any] = field(default_factory=dict)
    sync_warnings: list[str] = field(default_factory=list)
    event_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncStats:
    bundle_count: int = 0
    bundle_fps: float = 0.0
    average_offset_ms: float | None = None
    max_offset_ms: float | None = None
    last_bundle_id: str | None = None
    sync_quality: str = "unknown"


def event_features_for_window(events: np.ndarray, duration_sec: float) -> dict[str, Any]:
    """Compute simple DAVIS event-window features from Nx4 events [t,x,y,p]."""
    arr = np.asarray(events, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return {
            "event_count": 0,
            "event_rate": 0.0,
            "event_centroid": None,
            "event_motion_bbox": None,
            "event_vertical_bias": 0.0,
            "event_horizontal_bias": 0.0,
            "event_locality_score": 0.0,
            "event_motion_class": "STREAM_ACTIVE_NO_MOTION",
        }
    xs = arr[:, 1]
    ys = arr[:, 2]
    count = int(arr.shape[0])
    event_rate = float(count / max(duration_sec, 1e-6))
    x_mid = float(np.mean(xs))
    y_mid = float(np.mean(ys))
    left = int(np.count_nonzero(xs <= x_mid))
    right = int(count - left)
    upper = int(np.count_nonzero(ys <= y_mid))
    lower = int(count - upper)
    spread = float(np.hypot(np.std(xs), np.std(ys)))
    locality = float(np.clip(100.0 / max(spread, 1.0), 0.0, 100.0))
    motion_class = "EVENT_MOTION" if event_rate > 0 else "STREAM_ACTIVE_NO_MOTION"
    return {
        "event_count": count,
        "event_rate": event_rate,
        "event_centroid": [round(x_mid, 2), round(y_mid, 2)],
        "event_motion_bbox": [round(float(np.min(xs)), 2), round(float(np.min(ys)), 2), round(float(np.max(xs)), 2), round(float(np.max(ys)), 2)],
        "event_vertical_bias": float((lower - upper) / count),
        "event_horizontal_bias": float((right - left) / count),
        "event_locality_score": round(locality, 2),
        "event_motion_class": motion_class,
    }


def _davis_event_contribution(davis_report: dict[str, Any], event_time: float | None, now: float) -> dict[str, Any]:
    """Use the DAVIS analyzer's selected window as event-freshness authority.

    The analyzer deliberately expands its observation window in low-rate scenes
    (for example to 800 ms).  Applying the synchronizer's old fixed 100 ms gate
    to that same report made valid post-grasp evidence alternate CURRENT/STALE.
    Cross-camera offset remains a separate diagnostic/fusion gate below.
    """
    declared_status = str(davis_report.get("event_timestamp_status") or "").upper()
    declared_lag_ms = _first_number(davis_report.get("event_timestamp_lag_ms"))
    selected_window_ms = _first_number(
        davis_report.get("event_window_ms"),
        davis_report.get("event_window_fallback_ms"),
        davis_report.get("scoring_window_ms"),
        DAVIS_EVENT_MAX_AGE_MS,
    )
    accepted_window_ms = max(float(selected_window_ms or DAVIS_EVENT_MAX_AGE_MS), DAVIS_EVENT_MAX_AGE_MS)
    contribution = _sensor_contribution(
        event_time,
        now,
        accepted_window_ms,
        available=bool(davis_report),
        confidence_source=_first_number(
            davis_report.get("motion_estimation_confidence"),
            (davis_report.get("motion_estimate") or {}).get("motion_estimation_confidence"),
        ),
    )
    if declared_status in {"CURRENT", "AGING", "STALE", "UNAVAILABLE"}:
        # The analyzer may tighten a timestamp verdict, but a cached CURRENT
        # string must never make an old/frozen report current again.  Wall-clock
        # age from the event timestamp is the upper authority.
        computed_status = str(contribution.get("freshness") or "UNAVAILABLE").upper()
        if declared_status != "CURRENT" or computed_status in {"CURRENT", "AGING"}:
            severity = {"CURRENT": 0, "AGING": 1, "STALE": 2, "UNAVAILABLE": 3}
            final_status = max(
                (computed_status, declared_status),
                key=lambda value: severity.get(value, 3),
            )
            contribution["freshness"] = final_status
            contribution["valid"] = final_status == "CURRENT"
        if declared_lag_ms is not None:
            computed_age_ms = _first_number(contribution.get("age_ms"), 0.0) or 0.0
            authoritative_age_ms = max(float(computed_age_ms), float(declared_lag_ms))
            contribution["age_ms"] = round(authoritative_age_ms, 2)
            contribution["age_sec"] = round(authoritative_age_ms / 1000.0, 3)
            if authoritative_age_ms > accepted_window_ms:
                contribution["freshness"] = "STALE"
                contribution["valid"] = False
        contribution["freshness_source"] = "wall_clock_event_age_with_analyzer_bound"
    else:
        contribution["freshness_source"] = "software_timestamp_fallback"
    contribution["accepted_window_ms"] = round(accepted_window_ms, 3)
    contribution["analyzer_declared_status"] = declared_status or None
    contribution["event_window_mode"] = davis_report.get("event_window_mode")
    return contribution


class SensorSynchronizer:
    """Create software-timestamp bundles from current status/report files."""

    def __init__(self, max_offset_ms: float = SYNC_MAX_OFFSET_MS, event_window_ms: float = DAVIS_EVENT_WINDOW_MS):
        self.max_offset_ms = float(max_offset_ms)
        self.event_window_ms = float(event_window_ms)
        self.bundle_times: list[float] = []
        self.offsets: list[float] = []

    def make_bundle(
        self,
        *,
        realsense_status: dict[str, Any] | None = None,
        pointcloud_status: dict[str, Any] | None = None,
        davis_report: dict[str, Any] | None = None,
        events: np.ndarray | None = None,
        now: float | None = None,
        sync_mode: str = SYNC_MODE_SOFTWARE_TIMESTAMP,
        action_timestamp: float | None = None,
        event_window_ms: float | None = None,
    ) -> SensorFrameBundle:
        now = time.time() if now is None else float(now)
        realsense_status = realsense_status or {}
        pointcloud_status = pointcloud_status or {}
        davis_report = davis_report or {}
        sync_mode = sync_mode if sync_mode in SYNC_MODES else SYNC_MODE_SOFTWARE_TIMESTAMP
        bundle_time = action_timestamp if sync_mode == SYNC_MODE_ACTION_ANCHORED_WINDOW and action_timestamp is not None else now
        rs_time = _timestamp_from_status(
            realsense_status,
            keys=("rgb_timestamp", "color_timestamp", "realsense_rgb_timestamp", "timestamp", "system_timestamp"),
            default=None,
        )
        depth_time = _timestamp_from_status(
            realsense_status,
            keys=("depth_timestamp", "realsense_depth_timestamp", "timestamp", "system_timestamp"),
            default=rs_time,
        )
        pc_time = _timestamp_from_status(
            pointcloud_status,
            keys=("pointcloud_timestamp", "timestamp", "realsense_pointcloud_last_update_time", "system_timestamp"),
            default=None,
        )
        aps_time = _davis_aps_timestamp(davis_report)
        event_time = _davis_event_timestamp(davis_report)
        requested_window_ms = float(event_window_ms if event_window_ms is not None else self.event_window_ms)
        window_sec = requested_window_ms / 1000.0
        action_window_planned_end = None
        if sync_mode == SYNC_MODE_ACTION_ANCHORED_WINDOW:
            anchor = float(action_timestamp if action_timestamp is not None else _first_number(
                davis_report.get("close_command_time"),
                davis_report.get("post_grasp_start_time"),
                davis_report.get("action_timestamp"),
                now,
            ))
            start = anchor
            action_window_planned_end = anchor + 1.2
            end = min(max(now, start), action_window_planned_end)
        else:
            start = now - window_sec
            end = now
        if events is not None:
            event_arr = np.asarray(events, dtype=float)
            if event_arr.ndim == 2 and event_arr.shape[1] >= 4:
                event_arr = event_arr[(event_arr[:, 0] >= start) & (event_arr[:, 0] <= end)]
            else:
                event_arr = np.empty((0, 4), dtype=float)
            features = event_features_for_window(event_arr, max(end - start, 1e-6))
            features["event_window_source"] = "raw_events"
            features["event_window_materialized"] = True
            if event_arr.size:
                event_time = float(np.max(event_arr[:, 0]))
            else:
                event_time = None
        else:
            cached_count = davis_report.get("event_count")
            if cached_count is None:
                cached_count = davis_report.get("post_grasp_event_count")
            cached_rate = davis_report.get("event_rate")
            if cached_rate is None:
                cached_rate = davis_report.get("davis_event_rate")
            count = int(cached_count or 0)
            rate = float(cached_rate or 0.0)
            features = {
                "event_count": 0,
                "event_rate": 0.0,
                "event_centroid": None,
                "event_motion_bbox": None,
                "event_vertical_bias": 0.0,
                "event_horizontal_bias": 0.0,
                "event_locality_score": 0.0,
                "event_motion_class": "NOT_COMPUTED_NO_RAW_EVENTS",
                "event_window_source": "cached_report_diagnostic_only",
                "event_window_materialized": False,
                "cached_report_event_count": count,
                "cached_report_event_rate": rate,
                "cached_report_event_centroid": davis_report.get("event_centroid"),
                "cached_report_event_motion_bbox": davis_report.get("event_motion_bbox"),
                "cached_report_motion_class": davis_report.get("motion_class"),
            }
        if sync_mode == SYNC_MODE_ACTION_ANCHORED_WINDOW:
            features["action_event_windows"] = {
                "impact": [round(start, 6), round(start + 0.3, 6)],
                "settle": [round(start + 0.3, 6), round(start + 0.7, 6)],
                "stability": [round(start + 0.7, 6), round(start + 1.2, 6)],
            }
            features["action_anchor_timestamp"] = round(start, 6)
            features["action_event_window_planned_end"] = round(action_window_planned_end, 6)
            features["action_event_window_complete"] = now >= action_window_planned_end
        else:
            features["recent_event_windows_ms"] = [50, 100, 300]

        sensor_freshness = {
            "realsense_rgb": _sensor_contribution(rs_time, now, REALSENSE_MAX_AGE_MS, available=bool(realsense_status), confidence_source=realsense_status.get("tracking_confidence")),
            "realsense_depth": _sensor_contribution(depth_time, now, REALSENSE_MAX_AGE_MS, available=bool(realsense_status), confidence_source=realsense_status.get("depth_confidence")),
            "pointcloud": _sensor_contribution(pc_time, now, POINTCLOUD_MAX_AGE_SEC * 1000.0, available=bool(pointcloud_status), confidence_source=pointcloud_status.get("bottle_crop_confidence")),
            "davis_aps": _sensor_contribution(aps_time, now, DAVIS_APS_MAX_AGE_MS, available=bool(davis_report.get("aps_state") or davis_report.get("aps_stream_status") or davis_report.get("aps_fps"))),
            "davis_events": (
                _davis_event_contribution(davis_report, event_time, now)
                if events is None
                else _sensor_contribution(
                    event_time,
                    now,
                    DAVIS_EVENT_MAX_AGE_MS,
                    available=True,
                    confidence_source=_first_number(
                        davis_report.get("motion_estimation_confidence"),
                        (davis_report.get("motion_estimate") or {}).get("motion_estimation_confidence"),
                    ),
                )
            ),
        }
        sync_warnings = _freshness_warnings(sensor_freshness)
        if bool(realsense_status) and rs_time is None:
            sync_warnings.append("missing_timestamp:realsense")
        if bool(pointcloud_status) and pc_time is None:
            sync_warnings.append("missing_timestamp:pointcloud")
        if bool(davis_report.get("aps_state") or davis_report.get("aps_stream_status")) and aps_time is None:
            sync_warnings.append("missing_timestamp:davis_aps")
        if bool(davis_report or events is not None) and event_time is None:
            sync_warnings.append("missing_timestamp:davis_events")

        stale = [name for name, info in sensor_freshness.items() if info["freshness"] == "STALE"]
        unavailable = [name for name, info in sensor_freshness.items() if info["freshness"] == "UNAVAILABLE"]
        timestamps = [t for t in (rs_time, depth_time, pc_time, aps_time, event_time) if t is not None]
        offsets_ms = [(abs(bundle_time - t) * 1000.0) for t in timestamps]
        max_offset = max(offsets_ms) if offsets_ms else None
        offset_too_large = max_offset is not None and max_offset > self.max_offset_ms
        benchmark_offset_ok = max_offset is not None and max_offset <= BENCHMARK_MAX_SYNC_OFFSET_MS
        live_offset_ok = max_offset is not None and max_offset <= LIVE_QUALITY_MAX_SYNC_OFFSET_MS
        realsense_current = sensor_freshness["realsense_rgb"]["freshness"] == "CURRENT" and sensor_freshness["realsense_depth"]["freshness"] in {"CURRENT", "AGING"}
        pointcloud_current = sensor_freshness["pointcloud"]["freshness"] == "CURRENT"
        davis_events_current = sensor_freshness["davis_events"]["freshness"] == "CURRENT"
        rgbd_times = [value for value in (rs_time, depth_time) if value is not None]
        davis_realsense_offset_ms = max((abs(event_time - value) * 1000.0 for value in rgbd_times), default=None) if event_time is not None else None
        pointcloud_realsense_offset_ms = max((abs(pc_time - value) * 1000.0 for value in rgbd_times), default=None) if pc_time is not None else None
        davis_live_offset_ok = davis_realsense_offset_ms is not None and davis_realsense_offset_ms <= LIVE_QUALITY_MAX_SYNC_OFFSET_MS
        pointcloud_live_offset_ok = pointcloud_realsense_offset_ms is not None and pointcloud_realsense_offset_ms <= LIVE_QUALITY_MAX_SYNC_OFFSET_MS
        live_quality_offsets = []
        if davis_events_current and davis_realsense_offset_ms is not None:
            live_quality_offsets.append(davis_realsense_offset_ms)
        if pointcloud_current and pointcloud_realsense_offset_ms is not None:
            live_quality_offsets.append(pointcloud_realsense_offset_ms)
        live_quality_max_sensor_offset_ms = max(live_quality_offsets, default=None)
        if offset_too_large:
            sync_warnings.append("offset_too_large")
        if sync_mode == SYNC_MODE_DISPLAY_LATEST:
            quality = "display_latest"
            valid_for_benchmark = False
            valid_for_live_quality = False
            sync_warnings.append("display_latest_no_strict_fusion_claim")
        elif not realsense_status:
            quality = "waiting_for_realsense"
            valid_for_benchmark = False
            valid_for_live_quality = False
        elif not timestamps:
            quality = "missing_timestamp"
            valid_for_benchmark = False
            valid_for_live_quality = False
        elif not realsense_current:
            quality = "stale_realsense"
            valid_for_benchmark = False
            valid_for_live_quality = False
        elif max_offset is not None and max_offset > BENCHMARK_MAX_SYNC_OFFSET_MS:
            quality = "offset_too_large"
            valid_for_benchmark = False
            valid_for_live_quality = False
        elif stale:
            quality = "stale_modalities"
            valid_for_benchmark = benchmark_offset_ok
            valid_for_live_quality = live_offset_ok
        elif max_offset is not None and max_offset <= self.max_offset_ms:
            quality = "good"
            valid_for_benchmark = True
            valid_for_live_quality = live_offset_ok
        else:
            quality = "software_limited"
            valid_for_benchmark = benchmark_offset_ok
            valid_for_live_quality = live_offset_ok

        if sync_mode == SYNC_MODE_DISPLAY_LATEST:
            davis_realsense_sync_quality = "display_latest"
        elif not realsense_status:
            davis_realsense_sync_quality = "waiting_for_realsense"
        elif rs_time is None or depth_time is None:
            davis_realsense_sync_quality = "missing_timestamp"
        elif not realsense_current:
            davis_realsense_sync_quality = "stale_realsense"
        elif not davis_report and events is None:
            davis_realsense_sync_quality = "davis_events_unavailable"
        elif event_time is None or davis_realsense_offset_ms is None:
            davis_realsense_sync_quality = "missing_timestamp"
        elif not davis_events_current:
            davis_realsense_sync_quality = "stale_davis_events"
        elif not davis_live_offset_ok:
            davis_realsense_sync_quality = "offset_too_large"
        elif davis_realsense_offset_ms <= self.max_offset_ms:
            davis_realsense_sync_quality = "good"
        else:
            davis_realsense_sync_quality = "software_limited"
        davis_realsense_valid_for_live_quality = davis_realsense_sync_quality in {"good", "software_limited"}

        features["davis_realsense_sync_quality"] = davis_realsense_sync_quality
        # Post-grasp DAVIS quality is asynchronous and object-local.  Event
        # freshness is authoritative for that path; RGB-D/DAVIS offset remains
        # a separate cross-modal fusion claim and benchmark diagnostic.
        features["davis_events_valid_for_live_quality"] = bool(davis_events_current)
        features["pointcloud_valid_for_live_quality"] = bool(realsense_current and pointcloud_current and pointcloud_live_offset_ok and sync_mode != SYNC_MODE_DISPLAY_LATEST)
        features["software_sync_event_window_ms"] = round((end - start) * 1000.0, 3)
        freshness_gates = {
            "realsense_max_age_ms": REALSENSE_MAX_AGE_MS,
            "davis_event_max_age_ms": sensor_freshness["davis_events"].get(
                "accepted_window_ms", DAVIS_EVENT_MAX_AGE_MS
            ),
            "davis_event_freshness_source": sensor_freshness["davis_events"].get(
                "freshness_source"
            ),
            "davis_aps_max_age_ms": DAVIS_APS_MAX_AGE_MS,
            "pointcloud_max_age_sec": POINTCLOUD_MAX_AGE_SEC,
            "benchmark_max_sync_offset_ms": BENCHMARK_MAX_SYNC_OFFSET_MS,
            "live_quality_max_sync_offset_ms": LIVE_QUALITY_MAX_SYNC_OFFSET_MS,
            "davis_realsense_offset_ms": None if davis_realsense_offset_ms is None else round(davis_realsense_offset_ms, 3),
            "pointcloud_realsense_offset_ms": None if pointcloud_realsense_offset_ms is None else round(pointcloud_realsense_offset_ms, 3),
            "live_quality_max_sensor_offset_ms": None if live_quality_max_sensor_offset_ms is None else round(live_quality_max_sensor_offset_ms, 3),
            "davis_realsense_sync_quality": davis_realsense_sync_quality,
            "davis_realsense_valid_for_live_quality": davis_realsense_valid_for_live_quality,
            "davis_events_valid_for_live_quality": features["davis_events_valid_for_live_quality"],
            "pointcloud_valid_for_live_quality": features["pointcloud_valid_for_live_quality"],
            "davis_events_valid_for_benchmark": bool(davis_events_current and benchmark_offset_ok and sync_mode != SYNC_MODE_DISPLAY_LATEST),
            "pointcloud_valid_for_benchmark": bool(pointcloud_current and benchmark_offset_ok and sync_mode != SYNC_MODE_DISPLAY_LATEST),
        }
        bundle = SensorFrameBundle(
            bundle_id=f"sync-{int(now * 1000)}-{uuid.uuid4().hex[:8]}",
            system_timestamp=now,
            bundle_timestamp=bundle_time,
            realsense_rgb_frame_id=realsense_status.get("frame_id") or realsense_status.get("camera_frame_id"),
            realsense_rgb_timestamp=rs_time,
            realsense_depth_frame_id=realsense_status.get("frame_id") or realsense_status.get("camera_frame_id"),
            realsense_depth_timestamp=depth_time,
            pointcloud_timestamp=pc_time,
            davis_aps_timestamp=aps_time,
            davis_event_window_start=start if events is not None else None,
            davis_event_window_end=end if events is not None else None,
            davis_event_count=int(features.get("event_count") or 0),
            davis_event_rate=float(features.get("event_rate") or 0.0),
            sync_latency_ms=round((time.time() - now) * 1000.0, 2),
            max_sensor_time_offset_ms=None if max_offset is None else round(max_offset, 2),
            sync_quality=quality,
            davis_realsense_sync_quality=davis_realsense_sync_quality,
            davis_realsense_valid_for_live_quality=davis_realsense_valid_for_live_quality,
            sync_mode=sync_mode,
            stale_modalities=stale,
            unavailable_modalities=unavailable,
            bundle_valid_for_benchmark=bool(valid_for_benchmark),
            bundle_valid_for_live_quality=bool(valid_for_live_quality),
            sensor_freshness=sensor_freshness,
            freshness_gates=freshness_gates,
            sync_warnings=_unique(sync_warnings),
            event_features=features,
        )
        self.bundle_times.append(now)
        self.bundle_times = self.bundle_times[-120:]
        if max_offset is not None:
            self.offsets.append(max_offset)
            self.offsets = self.offsets[-120:]
        return bundle

    def stats(self, bundle: SensorFrameBundle | None = None) -> SyncStats:
        if len(self.bundle_times) >= 2:
            fps = (len(self.bundle_times) - 1) / max(self.bundle_times[-1] - self.bundle_times[0], 1e-6)
        else:
            fps = 0.0
        avg = float(np.mean(self.offsets)) if self.offsets else None
        max_offset = float(np.max(self.offsets)) if self.offsets else None
        return SyncStats(
            bundle_count=len(self.bundle_times),
            bundle_fps=round(fps, 2),
            average_offset_ms=None if avg is None else round(avg, 2),
            max_offset_ms=None if max_offset is None else round(max_offset, 2),
            last_bundle_id=None if bundle is None else bundle.bundle_id,
            sync_quality="unknown" if bundle is None else bundle.sync_quality,
        )


def _timestamp_from_status(
    status: dict[str, Any],
    default: float | None = None,
    *,
    keys: tuple[str, ...] = ("timestamp", "realsense_pointcloud_last_update_time", "system_timestamp"),
) -> float | None:
    value = None
    for key in keys:
        if key in status and status.get(key) not in (None, ""):
            value = status.get(key)
            break
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime

            text = value.replace("Z", "+00:00")
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            return default
    return default


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _davis_aps_timestamp(report: dict[str, Any]) -> float | None:
    aps = report.get("aps") if isinstance(report.get("aps"), dict) else {}
    return _timestamp_from_status(
        {**aps, **report},
        keys=(
            "davis_aps_timestamp",
            "aps_timestamp",
            "aps_host_timestamp",
            "aps_frame_timestamp",
            "last_aps_timestamp",
            "last_aps_host_timestamp",
        ),
        default=None,
    )


def _davis_event_timestamp(report: dict[str, Any]) -> float | None:
    motion = report.get("motion_estimate") if isinstance(report.get("motion_estimate"), dict) else {}
    if report.get("event_timestamp_status") is not None:
        return _timestamp_from_status(
            report,
            keys=(
                "event_host_timestamp",
                "davis_event_timestamp",
                "event_timestamp",
                "last_event_host_timestamp",
                "last_event_timestamp",
            ),
            default=None,
        )
    return _timestamp_from_status(
        {**motion, **report},
        keys=(
            "davis_event_timestamp",
            "event_timestamp",
            "event_host_timestamp",
            "host_timestamp",
            "last_event_timestamp",
            "last_event_host_timestamp",
            "motion_estimate_timestamp",
            "timestamp",
        ),
        default=None,
    )


def _freshness_for_age(age_ms: float | None, max_age_ms: float, *, available: bool) -> str:
    if not available or age_ms is None:
        return "UNAVAILABLE"
    if age_ms <= max_age_ms:
        return "CURRENT"
    if age_ms <= max_age_ms * 2.0:
        return "AGING"
    return "STALE"


def _sensor_contribution(
    timestamp: float | None,
    now: float,
    max_age_ms: float,
    *,
    available: bool,
    confidence_source: Any = None,
) -> dict[str, Any]:
    age_ms = None if timestamp is None else max(0.0, (now - timestamp) * 1000.0)
    freshness = _freshness_for_age(age_ms, max_age_ms, available=available)
    confidence = _first_number(confidence_source)
    if confidence is None:
        confidence = 1.0 if freshness == "CURRENT" else (0.5 if freshness == "AGING" else 0.0)
    elif confidence > 1.0:
        confidence = max(0.0, min(1.0, confidence / 100.0))
    else:
        confidence = max(0.0, min(1.0, confidence))
    return {
        "timestamp": timestamp,
        "age_ms": None if age_ms is None else round(age_ms, 2),
        "age_sec": None if age_ms is None else round(age_ms / 1000.0, 3),
        "freshness": freshness,
        "valid": freshness == "CURRENT",
        "confidence": round(confidence, 3),
    }


def _freshness_warnings(sensor_freshness: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name, info in sensor_freshness.items():
        freshness = str(info.get("freshness") or "").upper()
        if freshness == "STALE":
            warnings.append(f"{name}_stale")
        elif freshness == "UNAVAILABLE":
            warnings.append(f"{name}_unavailable")
    return warnings


def build_status(bundle: SensorFrameBundle, stats: SyncStats) -> dict[str, Any]:
    cal = calibration_status()
    mount = load_config_file(config_path("camera_mount.yaml"), default={}) or {}
    freshness = bundle.sensor_freshness or {}
    warnings = list(bundle.sync_warnings or [])
    sync_reason = ", ".join(warnings) if warnings else "software_timestamp_freshness_ok"
    pointcloud_info = freshness.get("pointcloud", {})
    davis_event_info = freshness.get("davis_events", {})
    davis_aps_info = freshness.get("davis_aps", {})
    rs_info = freshness.get("realsense_rgb", {})
    status = {
        "sync_enabled": SYNC_ENABLED,
        "sync_mode": bundle.sync_mode,
        "hardware_sync_available": False,
        "sync_quality": bundle.sync_quality,
        "davis_realsense_sync_quality": bundle.davis_realsense_sync_quality,
        "davis_realsense_valid_for_live_quality": bundle.davis_realsense_valid_for_live_quality,
        "max_sensor_time_offset_ms": bundle.max_sensor_time_offset_ms,
        "davis_realsense_offset_ms": bundle.freshness_gates.get("davis_realsense_offset_ms"),
        "pointcloud_realsense_offset_ms": bundle.freshness_gates.get("pointcloud_realsense_offset_ms"),
        "live_quality_max_sensor_offset_ms": bundle.freshness_gates.get("live_quality_max_sensor_offset_ms"),
        "average_sensor_time_offset_ms": stats.average_offset_ms,
        "last_bundle_id": bundle.bundle_id,
        "bundle_timestamp": bundle.bundle_timestamp,
        "bundle_age_sec": None if bundle.bundle_timestamp is None else round(max(0.0, time.time() - bundle.bundle_timestamp), 3),
        "synchronized_bundle_available": bundle.bundle_valid_for_benchmark or bundle.bundle_valid_for_live_quality,
        "sync_warning": "Software timestamp sync only; no shared hardware trigger detected.",
        "sync_warnings": warnings,
        "sync_reason": sync_reason,
        "stale_modalities": list(bundle.stale_modalities),
        "unavailable_modalities": list(bundle.unavailable_modalities),
        "bundle_valid_for_benchmark": bundle.bundle_valid_for_benchmark,
        "bundle_valid_for_live_quality": bundle.bundle_valid_for_live_quality,
        "valid_for_benchmark": bundle.bundle_valid_for_benchmark,
        "valid_for_live_quality": bundle.bundle_valid_for_live_quality,
        "sensor_freshness": freshness,
        "freshness_gates": bundle.freshness_gates,
        "realsense_age_ms": rs_info.get("age_ms"),
        "realsense_freshness": rs_info.get("freshness"),
        "davis_event_age_ms": davis_event_info.get("age_ms"),
        "davis_event_freshness": davis_event_info.get("freshness"),
        "davis_aps_age_ms": davis_aps_info.get("age_ms"),
        "davis_aps_freshness": davis_aps_info.get("freshness"),
        "pointcloud_age_sec": pointcloud_info.get("age_sec"),
        "pointcloud_freshness": pointcloud_info.get("freshness"),
        "davis_events_valid_for_live_quality": bundle.freshness_gates.get("davis_events_valid_for_live_quality"),
        "pointcloud_valid_for_live_quality": bundle.freshness_gates.get("pointcloud_valid_for_live_quality"),
        "davis_events_valid_for_benchmark": bundle.freshness_gates.get("davis_events_valid_for_benchmark"),
        "pointcloud_valid_for_benchmark": bundle.freshness_gates.get("pointcloud_valid_for_benchmark"),
        "bundle_fps": stats.bundle_fps,
        "davis_event_window_count": bundle.davis_event_count,
        "davis_event_rate": bundle.davis_event_rate,
        "davis_event_window_start": bundle.davis_event_window_start,
        "davis_event_window_end": bundle.davis_event_window_end,
        "davis_event_window_source": bundle.event_features.get("event_window_source"),
        "davis_event_window_materialized": bundle.event_features.get("event_window_materialized"),
        "cached_report_event_count": bundle.event_features.get("cached_report_event_count"),
        "cached_report_event_rate": bundle.event_features.get("cached_report_event_rate"),
        "action_event_window_planned_end": bundle.event_features.get("action_event_window_planned_end"),
        "action_event_window_complete": bundle.event_features.get("action_event_window_complete"),
        "mount_layout": mount.get("mount_layout", "unknown"),
        "davis_realsense_alignment_status": cal.get("alignment_status"),
        "homography_available": cal.get("homography_available"),
        "extrinsics_available": cal.get("extrinsics_available"),
        "approximate_alignment_only": not bool(cal.get("usable_for_overlay")),
    }
    return status


def run_file_sync_once(
    syncer: SensorSynchronizer | None = None,
    *,
    davis_report: dict[str, Any] | None = None,
) -> tuple[SensorFrameBundle, dict[str, Any]]:
    ensure_output_dir()
    syncer = syncer or SensorSynchronizer()
    rs = read_json(OUTPUT_DIR / "latest_realtime_status.json", default={}) or {}
    pc = read_json(OUTPUT_DIR / "pointcloud_status.json", default={}) or {}
    davis = davis_report or read_json(OUTPUT_DIR / "davis" / "latest_motion_report.json", default={}) or read_json(OUTPUT_DIR / "davis_motion_report.json", default={}) or {}
    bundle = syncer.make_bundle(realsense_status=rs, pointcloud_status=pc, davis_report=davis)
    stats = syncer.stats(bundle)
    bundle_payload = asdict(bundle)
    status = build_status(bundle, stats)
    write_json_atomic(LATEST_BUNDLE_PATH, bundle_payload)
    write_json_atomic(SYNC_STATUS_PATH, status)
    write_json_atomic(SYNC_REPORT_PATH, {"status": status, "latest_bundle": bundle_payload, "stats": asdict(stats)})
    return bundle, status


def diagnose() -> dict[str, Any]:
    bundle, status = run_file_sync_once()
    return {
        "sync_enabled": SYNC_ENABLED,
        "sync_mode": "software_timestamp",
        "hardware_sync_available": False,
        "latest_bundle_id": bundle.bundle_id,
        "sync_quality": status.get("sync_quality"),
        "max_sensor_time_offset_ms": status.get("max_sensor_time_offset_ms"),
        "status_path": str(SYNC_STATUS_PATH),
    }


def test_live(duration: float) -> dict[str, Any]:
    syncer = SensorSynchronizer()
    end = time.time() + max(0.1, float(duration))
    period = 1.0 / max(SYNC_TARGET_FPS, 1e-6)
    latest_status = None
    while time.time() < end:
        _bundle, latest_status = run_file_sync_once(syncer)
        time.sleep(period)
    return {"status": "complete", **(latest_status or {}), "duration_sec": duration}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSense + DAVIS software timestamp sync diagnostics.")
    parser.add_argument("--diagnose", action="store_true", help="Build one sync bundle from current status files.")
    parser.add_argument("--test-live", action="store_true", help="Build sync bundles from live status files for a duration.")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args(argv)
    payload = test_live(args.duration) if args.test_live else diagnose()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical DAVIS camera facade for APS frames and event packets.

This module is safe to import without DAVIS hardware attached. It does not open
the camera until :meth:`DavisCamera.start` is called, and it reuses the existing
``EventPostGraspMonitor`` backend so the motion GUI, dashboard, and future
benchmarking code do not create competing DAVIS hardware connections.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from grasp_project.davis.aps_stream import (
    DavisApsFrame,
    aps_features,
    latest_aps_array,
    normalize_backend_aps_frame,
)
from grasp_project.davis.event_stream import DAVIS_FRAME_SIZE, EventPostGraspMonitor
from grasp_project.davis.motion_features import event_features_for_window
from grasp_project.hardware.device_discovery import (
    DavisDeviceInfo,
    DEVICE_NOT_CONNECTED,
    DEVICE_DISCOVERED,
    ERROR,
    OPENING,
    PERMISSION_DENIED,
    select_davis_device,
)

DEVICE_NOT_FOUND = "DEVICE_NOT_FOUND"
STREAMING = "STREAMING"
APS_ONLY = "APS_ONLY"
EVENTS_ONLY = "EVENTS_ONLY"
STREAMING_BOTH = "STREAMING_BOTH"
DISCONNECTED = "DISCONNECTED"
RECONNECTING = "RECONNECTING"


@dataclass
class DavisEventBatch:
    """One polled DAVIS event packet in normalized Nx4 ``[t, x, y, polarity]`` form."""

    events: Any = field(repr=False)
    host_timestamp: float
    device_timestamp_start: float | None = None
    device_timestamp_end: float | None = None
    event_count: int = 0
    duration_ms: float = 0.0
    monotonic_timestamps: bool = True

    def to_status(self) -> dict[str, Any]:
        return {
            "host_timestamp": self.host_timestamp,
            "device_timestamp_start": self.device_timestamp_start,
            "device_timestamp_end": self.device_timestamp_end,
            "event_count": self.event_count,
            "duration_ms": self.duration_ms,
            "monotonic_timestamps": self.monotonic_timestamps,
        }


@dataclass
class DavisStreamStatus:
    """Current DAVIS stream state for dashboard/status consumers."""

    state: str = DEVICE_NOT_CONNECTED
    backend: str | None = None
    device_status: str | None = None
    reason: str | None = None
    aps_mode: str = "DAVIS APS Unavailable"
    aps_frame_id: int | None = None
    aps_fps: float = 0.0
    event_count_total: int = 0
    event_rate_hz: float = 0.0
    dropped_event_count: int = 0
    last_event_count: int = 0
    last_poll_time: float | None = None
    ring_duration_sec: float = 0.0
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DavisEventRingBuffer:
    """Bounded in-memory event ring for multiple DAVIS consumers."""

    def __init__(self, max_duration_sec: float = 2.0, max_event_count: int = 50_000, frame_size=DAVIS_FRAME_SIZE):
        self.max_duration_sec = float(max_duration_sec)
        self.max_event_count = int(max_event_count)
        self.frame_size = tuple(frame_size)
        self._events = np.empty((0, 4), dtype=np.float64)
        self.dropped_event_count = 0
        self.monotonic_timestamps = True

    @property
    def events(self) -> np.ndarray:
        return self._events.copy()

    @property
    def event_count(self) -> int:
        return int(self._events.shape[0])

    @property
    def duration_sec(self) -> float:
        if self._events.shape[0] < 2:
            return 0.0
        return float(max(0.0, self._events[-1, 0] - self._events[0, 0]))

    def add(self, events: Any) -> DavisEventBatch:
        now = time.time()
        arr = np.asarray(events, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return DavisEventBatch(
                events=np.empty((0, 4), dtype=np.float64),
                host_timestamp=now,
                event_count=0,
            )
        arr = arr[:, :4]
        monotonic = bool(np.all(np.diff(arr[:, 0]) >= -1e-9))
        self.monotonic_timestamps = bool(self.monotonic_timestamps and monotonic)
        if self._events.size:
            before = self._events.shape[0]
            self._events = np.vstack([self._events, arr])
            added = self._events.shape[0] - before
        else:
            self._events = arr.copy()
            added = arr.shape[0]
        self._prune()
        duration = float(max(0.0, arr[-1, 0] - arr[0, 0]) * 1000.0) if arr.shape[0] >= 2 else 0.0
        return DavisEventBatch(
            events=arr,
            host_timestamp=now,
            device_timestamp_start=float(arr[0, 0]),
            device_timestamp_end=float(arr[-1, 0]),
            event_count=int(added),
            duration_ms=duration,
            monotonic_timestamps=monotonic,
        )

    def _prune(self) -> None:
        if self._events.size == 0:
            return
        latest = float(self._events[-1, 0])
        keep_after = latest - self.max_duration_sec
        mask = self._events[:, 0] >= keep_after
        dropped = int(np.count_nonzero(~mask))
        if dropped:
            self.dropped_event_count += dropped
            self._events = self._events[mask]
        if self._events.shape[0] > self.max_event_count:
            overflow = self._events.shape[0] - self.max_event_count
            self.dropped_event_count += int(overflow)
            self._events = self._events[overflow:]

    def get_events_between(self, start_time: float, end_time: float) -> np.ndarray:
        if self._events.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        start = float(start_time)
        end = float(end_time)
        mask = (self._events[:, 0] >= start) & (self._events[:, 0] <= end)
        return self._events[mask].copy()

    def get_recent_events(self, window_ms: float = 100.0, now: float | None = None) -> np.ndarray:
        if self._events.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        end = float(now if now is not None else self._events[-1, 0])
        start = end - float(window_ms) / 1000.0
        return self.get_events_between(start, end)

    def get_event_count(self, window_ms: float = 100.0) -> int:
        return int(self.get_recent_events(window_ms).shape[0])

    def get_event_features(self, window_ms: float = 100.0, baseline: dict[str, Any] | None = None, roi_xywh=None) -> dict[str, Any]:
        events = self.get_recent_events(window_ms)
        return event_features_for_window(events, window_ms=window_ms, frame_size=self.frame_size, baseline=baseline, roi_xywh=roi_xywh)

    def get_event_image(self, window_ms: float = 100.0) -> np.ndarray:
        events = self.get_recent_events(window_ms)
        width, height = self.frame_size
        heat = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        if events.size == 0:
            return heat
        xs = np.clip(events[:, 1].astype(int), 0, int(width) - 1)
        ys = np.clip(events[:, 2].astype(int), 0, int(height) - 1)
        pol = events[:, 3] > 0
        heat[ys[pol], xs[pol], 1] = 255
        heat[ys[~pol], xs[~pol], 2] = 255
        return cv2.GaussianBlur(heat, (3, 3), 0)


class DavisCamera:
    """Single-producer DAVIS stream manager for standalone test-mode consumers."""

    def __init__(
        self,
        *,
        enable_aps: bool = True,
        enable_events: bool = True,
        max_event_duration_sec: float = 2.0,
        max_event_count: int = 50_000,
        aps_config: dict[str, Any] | None = None,
    ):
        self.enable_aps = bool(enable_aps)
        self.enable_events = bool(enable_events)
        self.aps_config = aps_config or {}
        self.monitor: EventPostGraspMonitor | None = None
        self.ring = DavisEventRingBuffer(max_duration_sec=max_event_duration_sec, max_event_count=max_event_count)
        self.latest_aps_frame: DavisApsFrame | None = None
        self.latest_event_batch: DavisEventBatch | None = None
        self.previous_aps_image = None
        self.status_record = DavisStreamStatus()
        self._frame_id = 0
        self._aps_times: list[float] = []
        self._event_total = 0
        self._event_times: list[tuple[float, int]] = []

    def start(self) -> DavisStreamStatus:
        self.status_record = DavisStreamStatus(state=OPENING, active=False)
        selection = select_davis_device()
        if not selection.selected:
            state = PERMISSION_DENIED if selection.status == PERMISSION_DENIED else DEVICE_NOT_FOUND
            self.status_record = DavisStreamStatus(
                state=state,
                device_status=selection.status,
                reason=selection.reason or selection.error or "DAVIS device not found.",
                active=False,
            )
            return self.status_record
        self.monitor = EventPostGraspMonitor(enabled=True)
        report = self.monitor.start()
        if not self.monitor.active:
            self.status_record = DavisStreamStatus(
                state=ERROR,
                backend=self.monitor.backend_name,
                device_status=DEVICE_DISCOVERED,
                reason=report.get("davis_unavailable_reason") or self.monitor.device_open_error,
                active=False,
            )
            return self.status_record
        self.status_record = DavisStreamStatus(
            state=STREAMING,
            backend=self.monitor.backend_name,
            device_status=DEVICE_DISCOVERED,
            active=True,
            reason=None,
        )
        return self.status_record

    def poll_once(self) -> DavisStreamStatus:
        if not self.monitor or not self.monitor.active:
            self.status_record.state = DISCONNECTED if self.monitor else DEVICE_NOT_CONNECTED
            self.status_record.active = False
            return self.status_record
        events, frames = self.monitor.poll_event_and_frame_arrays()
        event_count = int(events.shape[0]) if events is not None and getattr(events, "ndim", 0) == 2 else 0
        if self.enable_events and event_count:
            self.latest_event_batch = self.ring.add(events)
            self._event_total += event_count
            self._event_times.append((time.time(), event_count))
            self._event_times = self._event_times[-120:]
        elif self.enable_events:
            self.latest_event_batch = DavisEventBatch(
                events=np.empty((0, 4), dtype=np.float64),
                host_timestamp=time.time(),
                event_count=0,
            )
        latest_frame = latest_aps_array(frames)
        if self.enable_aps and latest_frame is not None:
            self._frame_id += 1
            self.latest_aps_frame = normalize_backend_aps_frame(
                latest_frame,
                backend=self.monitor.backend_name,
                frame_id=self._frame_id,
                device_timestamp_us=self.monitor.latest_aps_device_timestamp_us,
                config=self.aps_config,
            )
            self._aps_times.append(time.time())
            self._aps_times = self._aps_times[-120:]
        self._refresh_status()
        return self.status_record

    def _refresh_status(self) -> None:
        aps_present = bool(self.latest_aps_frame and self.latest_aps_frame.aps_frame_valid)
        event_present = bool(self.ring.event_count > 0)
        if aps_present and event_present:
            state = STREAMING_BOTH
        elif aps_present:
            state = APS_ONLY
        elif self.enable_events and (event_present or self.monitor and self.monitor.active):
            state = EVENTS_ONLY
        else:
            state = STREAMING if self.monitor and self.monitor.active else DISCONNECTED
        self.status_record = DavisStreamStatus(
            state=state,
            backend=None if self.monitor is None else self.monitor.backend_name,
            device_status=DEVICE_DISCOVERED,
            reason=None if self.monitor and self.monitor.active else "DAVIS monitor inactive.",
            aps_mode="DAVIS APS Unavailable" if not self.latest_aps_frame else self.latest_aps_frame.aps_mode,
            aps_frame_id=None if not self.latest_aps_frame else self.latest_aps_frame.frame_id,
            aps_fps=_fps_from_times(self._aps_times),
            event_count_total=self._event_total,
            event_rate_hz=_event_rate_from_counts(self._event_times),
            dropped_event_count=self.ring.dropped_event_count,
            last_event_count=0 if self.latest_event_batch is None else self.latest_event_batch.event_count,
            last_poll_time=time.time(),
            ring_duration_sec=self.ring.duration_sec,
            active=bool(self.monitor and self.monitor.active),
        )

    def get_recent_events(self, window_ms: float = 100.0) -> np.ndarray:
        return self.ring.get_recent_events(window_ms)

    def get_event_features(self, window_ms: float = 100.0, baseline=None, roi_xywh=None) -> dict[str, Any]:
        return self.ring.get_event_features(window_ms=window_ms, baseline=baseline, roi_xywh=roi_xywh)

    def aps_features(self) -> dict[str, Any]:
        image = None if self.latest_aps_frame is None else self.latest_aps_frame.image
        features = aps_features(image, previous_frame=self.previous_aps_image)
        if image is not None:
            self.previous_aps_image = image.copy()
        return features

    def status(self) -> dict[str, Any]:
        payload = self.status_record.to_dict()
        payload["event_ring"] = {
            "event_count": self.ring.event_count,
            "duration_sec": round(self.ring.duration_sec, 4),
            "dropped_event_count": self.ring.dropped_event_count,
            "monotonic_timestamps": self.ring.monotonic_timestamps,
        }
        if self.latest_aps_frame is not None:
            payload["aps"] = self.latest_aps_frame.to_status()
        if self.latest_event_batch is not None:
            payload["latest_event_batch"] = self.latest_event_batch.to_status()
        return payload

    def stop(self) -> None:
        if self.monitor is not None:
            self.monitor.stop()
        self.monitor = None
        self.status_record.state = DISCONNECTED
        self.status_record.active = False


def _fps_from_times(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    return round((len(times) - 1) / max(times[-1] - times[0], 1e-6), 2)


def _event_rate_from_counts(samples: list[tuple[float, int]]) -> float:
    if len(samples) < 2:
        return 0.0
    duration = max(samples[-1][0] - samples[0][0], 1e-6)
    return round(sum(count for _t, count in samples[1:]) / duration, 2)


__all__ = [
    "APS_ONLY",
    "DAVIS_FRAME_SIZE",
    "DEVICE_NOT_FOUND",
    "DISCONNECTED",
    "DavisApsFrame",
    "DavisCamera",
    "DavisDeviceInfo",
    "DavisEventBatch",
    "DavisEventRingBuffer",
    "DavisStreamStatus",
    "ERROR",
    "EVENTS_ONLY",
    "OPENING",
    "PERMISSION_DENIED",
    "RECONNECTING",
    "STREAMING",
    "STREAMING_BOTH",
]

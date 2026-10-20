"""Common lightweight timing/profile helpers."""

import threading
import time
from collections import deque

profile_lock = threading.Lock()
profile = {}
profile_history = {}


class RateTracker:
    """Small rolling FPS/rate helper for scripts and status panels."""

    def __init__(self, maxlen=120, clock=time.time):
        self.times = deque(maxlen=int(maxlen))
        self.clock = clock

    def tick(self, timestamp=None):
        """Record one event and return the timestamp used."""
        ts = self.clock() if timestamp is None else float(timestamp)
        self.times.append(ts)
        return ts

    def fps(self):
        """Return the current rolling event rate."""
        return fps_from_timestamps(self.times)

    def clear(self):
        """Clear all recorded timestamps."""
        self.times.clear()


def profile_set(stage, value_ms):
    """Record a timing sample for a named stage."""
    value = float(value_ms)
    now = time.time()
    with profile_lock:
        history = profile_history.setdefault(stage, deque(maxlen=120))
        history.append(value)
        profile[stage] = {
            "ms": round(value, 2),
            "avg_ms": round(float(sum(history) / max(len(history), 1)), 2),
            "max_ms": round(float(max(history)), 2),
            "samples": len(history),
            "timestamp": now,
        }


def profile_snapshot():
    """Return the latest profile summary by stage."""
    with profile_lock:
        return dict(profile)


def fps_from_timestamps(timestamps):
    """Estimate FPS from a sequence of monotonic/wall-clock timestamps."""
    values = [float(t) for t in timestamps]
    if len(values) < 2:
        return 0.0
    elapsed = values[-1] - values[0]
    if elapsed <= 1e-9:
        return 0.0
    return float((len(values) - 1) / elapsed)


__all__ = [
    "RateTracker",
    "fps_from_timestamps",
    "profile",
    "profile_history",
    "profile_lock",
    "profile_set",
    "profile_snapshot",
]

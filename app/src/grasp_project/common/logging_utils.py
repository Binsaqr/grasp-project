"""Small logging/status helpers with no hardware dependencies."""

import time


def utc_iso(timestamp=None):
    """Return a compact UTC ISO-8601 timestamp with seconds precision."""
    ts = time.time() if timestamp is None else float(timestamp)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def format_status_line(status, message=None, **fields):
    """Format a short human-readable status line for scripts and diagnostics."""
    parts = [str(status)]
    if message:
        parts.append(str(message))
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def compact_exception(exc):
    """Return a stable one-line exception description."""
    return f"{exc.__class__.__name__}: {exc}"


__all__ = ["compact_exception", "format_status_line", "utc_iso"]

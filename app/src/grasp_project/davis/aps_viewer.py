"""DAVIS APS/live viewer compatibility module.

The experimental viewer is loaded lazily so import tests do not require DAVIS
hardware or optional viewer dependencies.
"""

from importlib import import_module


def load_legacy_viewer():
    """Return the current experimental DAVIS live viewer module."""
    try:
        return import_module("davis_live_viewer")
    except Exception as exc:
        raise RuntimeError(f"DAVIS live viewer is unavailable: {exc}") from exc

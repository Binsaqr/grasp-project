"""RealSense camera utility compatibility imports.

The transport-health helpers are optional in non-ROS environments. This module
must remain importable when RealSense/ROS hardware is unavailable.
"""

try:
    from scripts.realsense_transport_health import *  # noqa: F401,F403
    OPTIONAL_IMPORT_ERROR = None
except Exception as exc:
    OPTIONAL_IMPORT_ERROR = exc


def optional_import_error():
    """Return the deferred import error for RealSense helpers, if any."""
    return OPTIONAL_IMPORT_ERROR

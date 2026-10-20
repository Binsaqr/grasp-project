"""Hardware discovery helpers for camera identity and permissions."""

__all__ = [
    "DavisDeviceInfo",
    "DeviceDiscoveryResult",
    "DiscoveredDevice",
    "RealSenseDeviceInfo",
    "check_usb_access",
    "discover_davis_devices",
    "discover_realsense_devices",
    "find_usb_node_by_vid_pid",
    "monitor_reconnect",
    "select_davis_device",
    "select_realsense_device",
    "wait_for_device",
]


def __getattr__(name):
    if name in __all__:
        from . import device_discovery

        return getattr(device_discovery, name)
    raise AttributeError(name)

"""Stable hardware discovery for RealSense and DAVIS cameras.

This module is safe to import without cameras attached. It never treats a
transient ``/dev/bus/usb`` path as a persistent camera identifier; bus/device
nodes are reported only for diagnostics and permission checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from grasp_project.common.config import load_named_config


DAVIS_VENDOR_ID = "152a"
DAVIS_PRODUCT_ID = "841a"
REALSENSE_VENDOR_ID = "8086"
DEVICE_NOT_CONNECTED = "DEVICE_NOT_CONNECTED"
DEVICE_DISCOVERED = "DEVICE_DISCOVERED"
PERMISSION_DENIED = "PERMISSION_DENIED"
OPENING = "OPENING"
STREAMING = "STREAMING"
DISCONNECTED = "DISCONNECTED"
RECONNECTING = "RECONNECTING"
MULTIPLE_DEVICES = "MULTIPLE_DEVICES"
BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
ERROR = "ERROR"


@dataclass
class DiscoveredDevice:
    """Common hardware record for a discovered camera-like device."""

    sensor_type: str
    vendor_id: str | None = None
    product_id: str | None = None
    serial_number: str | None = None
    model_name: str | None = None
    usb_speed: str | None = None
    bus: str | None = None
    device_number: str | None = None
    device_node: str | None = None
    permission_status: str = "unknown"
    backend_available: bool | None = None
    backend_name: str | None = None
    status: str = DEVICE_DISCOVERED
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealSenseDeviceInfo(DiscoveredDevice):
    """RealSense-specific device record."""

    sensor_type: str = "realsense"


@dataclass
class DavisDeviceInfo(DiscoveredDevice):
    """DAVIS-specific device record."""

    sensor_type: str = "davis"


@dataclass
class DeviceDiscoveryResult:
    """Result of discovering and optionally selecting one camera."""

    sensor_type: str
    devices: list[DiscoveredDevice] = field(default_factory=list)
    selected: DiscoveredDevice | None = None
    status: str = DEVICE_NOT_CONNECTED
    reason: str = ""
    selection_mode: str = "auto"
    configured_serial: str | None = None
    preferred_model: str | None = None
    require_usb3: bool | None = None
    backend_available: bool | None = None
    backend_name: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.selected is not None and self.status in {DEVICE_DISCOVERED, STREAMING, OPENING}

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _run_text(command: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as exc:
        return 127, "", str(exc)


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower_hex(value: str | None) -> str | None:
    text = _norm(value)
    return text.lower() if text else None


def _hardware_config() -> dict:
    config = load_named_config("hardware.yaml", default={}) or {}
    realsense = dict(config.get("realsense") or {})
    davis = dict(config.get("davis") or {})
    mode = os.environ.get("CAMERA_SELECTION_MODE")
    if mode:
        realsense["selection"] = mode
        davis["selection"] = mode
    if os.environ.get("REALSENSE_SERIAL"):
        realsense["serial"] = os.environ.get("REALSENSE_SERIAL")
    if os.environ.get("DAVIS_SERIAL"):
        davis["serial"] = os.environ.get("DAVIS_SERIAL")
    config["realsense"] = realsense
    config["davis"] = davis
    return config


def parse_lsusb_lines(text: str) -> list[dict]:
    """Parse ``lsusb`` output into structured USB records."""

    pattern = re.compile(r"Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\s*(.*)")
    records = []
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        bus, device, vendor, product, description = match.groups()
        node = Path("/dev/bus/usb") / bus / device
        records.append(
            {
                "line": line.strip(),
                "bus": bus,
                "device_number": device,
                "vendor_id": vendor.lower(),
                "product_id": product.lower(),
                "description": description.strip(),
                "device_node": str(node),
            }
        )
    return records


def _lsusb_records() -> tuple[list[dict], str | None]:
    code, stdout, stderr = _run_text(["lsusb"])
    if code != 0:
        return [], stderr or stdout or "lsusb failed"
    return parse_lsusb_lines(stdout), None


def check_usb_access(path_or_device) -> dict:
    """Return read/write diagnostics for a transient USB device node."""

    if isinstance(path_or_device, DiscoveredDevice):
        path = path_or_device.device_node
    elif isinstance(path_or_device, dict):
        path = path_or_device.get("device_node") or path_or_device.get("device_path")
    else:
        path = path_or_device
    if not path:
        return {"device_node": None, "exists": False, "readable": False, "writable": False, "permission_ok": False}
    node = Path(path)
    result = {
        "device_node": str(node),
        "exists": node.exists(),
        "readable": os.access(node, os.R_OK),
        "writable": os.access(node, os.W_OK),
        "permission_ok": bool(node.exists() and os.access(node, os.R_OK | os.W_OK)),
    }
    if node.exists():
        try:
            stat = node.stat()
            result.update({"mode_octal": oct(stat.st_mode & 0o777), "uid": stat.st_uid, "gid": stat.st_gid})
        except OSError as exc:
            result["stat_error"] = str(exc)
    return result


def find_usb_node_by_vid_pid(vendor_id: str, product_id: str) -> list[dict]:
    """Find current USB nodes for a VID/PID pair.

    The returned node path is diagnostic only. It must not be stored as camera
    identity.
    """

    vendor = _lower_hex(vendor_id)
    product = _lower_hex(product_id)
    records, error = _lsusb_records()
    matches = [r for r in records if r["vendor_id"] == vendor and r["product_id"] == product]
    for record in matches:
        record.update(check_usb_access(record["device_node"]))
    if error and not matches:
        return []
    return matches


def _backend_available(module_name: str) -> tuple[bool, str | None]:
    try:
        __import__(module_name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _pyrealsense_devices() -> tuple[list[RealSenseDeviceInfo], str | None]:
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        return [], f"pyrealsense2 unavailable: {exc}"
    devices = []
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            def info(kind):
                try:
                    return dev.get_info(kind) if dev.supports(kind) else None
                except Exception:
                    return None

            serial = info(rs.camera_info.serial_number)
            model = info(rs.camera_info.name) or info(rs.camera_info.product_line)
            usb_speed = info(rs.camera_info.usb_type_descriptor)
            product_id = info(rs.camera_info.product_id)
            devices.append(
                RealSenseDeviceInfo(
                    vendor_id=REALSENSE_VENDOR_ID,
                    product_id=_lower_hex(product_id),
                    serial_number=_norm(serial),
                    model_name=_norm(model),
                    usb_speed=_norm(usb_speed),
                    backend_available=True,
                    backend_name="pyrealsense2",
                    raw={"source": "pyrealsense2"},
                )
            )
        return devices, None
    except Exception as exc:
        return [], f"pyrealsense2 query failed: {exc}"


def _parse_rs_enumerate(text: str) -> list[RealSenseDeviceInfo]:
    devices = []
    current: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("device info"):
            if current:
                devices.append(_realsense_from_enumerate(current))
            current = {}
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_l = key.lower().replace(" ", "_")
        if key_l in {"name", "serial_number", "firmware_version", "usb_type_descriptor", "product_id", "product_line"}:
            current[key_l] = value
    if current:
        devices.append(_realsense_from_enumerate(current))
    return devices


def _realsense_from_enumerate(info: dict) -> RealSenseDeviceInfo:
    return RealSenseDeviceInfo(
        vendor_id=REALSENSE_VENDOR_ID,
        product_id=_lower_hex(info.get("product_id")),
        serial_number=_norm(info.get("serial_number")),
        model_name=_norm(info.get("name") or info.get("product_line")),
        usb_speed=_norm(info.get("usb_type_descriptor")),
        backend_available=True,
        backend_name="rs-enumerate-devices",
        raw={"source": "rs-enumerate-devices", **info},
    )


def _rs_enumerate_devices() -> tuple[list[RealSenseDeviceInfo], str | None]:
    code, stdout, stderr = _run_text(["rs-enumerate-devices"], timeout=8.0)
    if code != 0:
        return [], stderr or stdout or "rs-enumerate-devices failed"
    devices = _parse_rs_enumerate(stdout)
    return devices, None if devices else "rs-enumerate-devices found no RealSense devices"


def discover_realsense_devices() -> DeviceDiscoveryResult:
    """Enumerate connected RealSense devices by SDK identity."""

    devices, error = _pyrealsense_devices()
    if not devices:
        devices, second_error = _rs_enumerate_devices()
        error = second_error or error
    config = _hardware_config().get("realsense") or {}
    result = DeviceDiscoveryResult(
        sensor_type="realsense",
        devices=devices,
        status=DEVICE_DISCOVERED if devices else DEVICE_NOT_CONNECTED,
        reason="" if devices else (error or "No compatible RealSense device detected."),
        selection_mode=str(config.get("selection") or "auto"),
        configured_serial=_norm(config.get("serial")),
        preferred_model=_norm(config.get("preferred_model")),
        require_usb3=bool(config.get("require_usb3", True)),
        backend_available=True if devices else None,
        backend_name=devices[0].backend_name if devices else None,
        error=error if not devices else None,
    )
    return result


def discover_davis_devices() -> DeviceDiscoveryResult:
    """Enumerate connected DAVIS devices by VID/PID and backend availability."""

    config = _hardware_config().get("davis") or {}
    vendor = _lower_hex(config.get("vendor_id") or DAVIS_VENDOR_ID)
    product = _lower_hex(config.get("product_id") or DAVIS_PRODUCT_ID)
    backend_ok, backend_error = _backend_available("pyaer")
    matches = find_usb_node_by_vid_pid(vendor, product)
    devices: list[DavisDeviceInfo] = []
    for record in matches:
        access = check_usb_access(record["device_node"])
        devices.append(
            DavisDeviceInfo(
                vendor_id=record.get("vendor_id"),
                product_id=record.get("product_id"),
                serial_number=None,
                model_name=record.get("description") or "DAVIS/iniVation event camera",
                bus=record.get("bus"),
                device_number=record.get("device_number"),
                device_node=record.get("device_node"),
                permission_status="ok" if access.get("permission_ok") else "denied",
                backend_available=backend_ok,
                backend_name="pyaer" if backend_ok else None,
                status=PERMISSION_DENIED if not access.get("permission_ok") else DEVICE_DISCOVERED,
                raw={"source": "lsusb_vid_pid", "lsusb_line": record.get("line"), **access},
            )
        )
    status = DEVICE_DISCOVERED if devices else DEVICE_NOT_CONNECTED
    reason = "" if devices else f"No DAVIS device detected with VID:PID {vendor}:{product}."
    if devices and not backend_ok:
        status = BACKEND_UNAVAILABLE
        reason = f"pyaer backend unavailable: {backend_error}"
    result = DeviceDiscoveryResult(
        sensor_type="davis",
        devices=devices,
        status=status,
        reason=reason,
        selection_mode=str(config.get("selection") or "auto"),
        configured_serial=_norm(config.get("serial")),
        backend_available=backend_ok,
        backend_name="pyaer" if backend_ok else None,
        error=backend_error if not backend_ok else None,
    )
    return result


def _select_device(result: DeviceDiscoveryResult, *, serial: str | None = None, preferred_model: str | None = None) -> DeviceDiscoveryResult:
    serial = _norm(serial) or result.configured_serial
    preferred_model = _norm(preferred_model) or result.preferred_model
    devices = list(result.devices)
    if not devices:
        result.selected = None
        result.status = DEVICE_NOT_CONNECTED if result.status != BACKEND_UNAVAILABLE else result.status
        result.reason = result.reason or f"No {result.sensor_type} devices detected."
        return result
    if serial:
        for device in devices:
            if str(device.serial_number or "") == str(serial):
                result.selected = device
                result.status = device.status if device.status == PERMISSION_DENIED else DEVICE_DISCOVERED
                result.reason = f"Selected configured serial {serial}."
                return result
        result.selected = None
        result.status = DEVICE_NOT_CONNECTED
        result.reason = f"Configured {result.sensor_type} serial {serial} was not found."
        return result
    if len(devices) == 1:
        device = devices[0]
        result.selected = device
        result.status = device.status if device.status == PERMISSION_DENIED else DEVICE_DISCOVERED
        result.reason = "Only compatible device connected; selected automatically."
        return result
    if preferred_model:
        matching = [d for d in devices if preferred_model.lower() in str(d.model_name or "").lower()]
        if len(matching) == 1:
            result.selected = matching[0]
            result.status = DEVICE_DISCOVERED
            result.reason = f"Selected preferred model {preferred_model}."
            return result
    result.selected = None
    result.status = MULTIPLE_DEVICES
    serials = [d.serial_number or d.model_name or d.device_node or "unknown" for d in devices]
    result.reason = (
        f"Multiple {result.sensor_type} devices detected ({', '.join(map(str, serials))}). "
        f"Set {result.sensor_type.upper()}_SERIAL or config/hardware.yaml serial."
    )
    return result


def select_realsense_device(serial: str | None = None, preferred_model: str | None = None) -> DeviceDiscoveryResult:
    result = discover_realsense_devices()
    return _select_device(result, serial=serial, preferred_model=preferred_model)


def select_davis_device(serial: str | None = None) -> DeviceDiscoveryResult:
    result = discover_davis_devices()
    selected = _select_device(result, serial=serial)
    if selected.backend_available is False:
        selected.status = BACKEND_UNAVAILABLE
        selected.reason = selected.reason or selected.error or "DAVIS backend unavailable."
    if selected.selected and selected.selected.permission_status == "denied":
        selected.status = PERMISSION_DENIED
        selected.reason = (
            f"DAVIS detected at {selected.selected.device_node} but access is denied. "
            "Run: ./scripts/setup_davis_udev.sh"
        )
    return selected


def wait_for_device(
    selector: Callable[[], DeviceDiscoveryResult],
    timeout_sec: float = 10.0,
    interval_sec: float = 0.5,
) -> DeviceDiscoveryResult:
    """Poll a selector until a device is selected or timeout expires."""

    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    last = selector()
    while time.monotonic() < deadline:
        if last.selected is not None or last.status in {MULTIPLE_DEVICES, PERMISSION_DENIED}:
            return last
        time.sleep(max(0.05, float(interval_sec)))
        last = selector()
        if last.status == DEVICE_NOT_CONNECTED:
            last.status = RECONNECTING
    return last


def monitor_reconnect(
    selector: Callable[[], DeviceDiscoveryResult],
    timeout_sec: float = 10.0,
    interval_sec: float = 1.0,
) -> Iterable[DeviceDiscoveryResult]:
    """Yield bounded reconnect status samples for UI/diagnostic loops."""

    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while time.monotonic() <= deadline:
        result = selector()
        if result.selected is None and result.status == DEVICE_NOT_CONNECTED:
            result.status = RECONNECTING
        yield result
        if result.selected is not None:
            return
        time.sleep(max(0.1, float(interval_sec)))


def discovery_summary() -> dict:
    return {
        "realsense": select_realsense_device().to_dict(),
        "davis": select_davis_device().to_dict(),
        "selection_policy": {
            "persistent_identity": "serial/model/VID_PID",
            "transient_usb_node": "diagnostic_only",
        },
    }


def _print_human(result: DeviceDiscoveryResult) -> None:
    prefix = (
        "PASS"
        if result.selected and result.status != BACKEND_UNAVAILABLE
        else ("WARN" if result.status in {DEVICE_NOT_CONNECTED, BACKEND_UNAVAILABLE} else "FAIL")
    )
    print(f"{prefix} {result.sensor_type}: {result.status} - {result.reason or 'OK'}")
    for device in result.devices:
        selected = " selected" if result.selected is device else ""
        serial = f" serial={device.serial_number}" if device.serial_number else ""
        node = f" node={device.device_node}" if device.device_node else ""
        access = f" permission={device.permission_status}" if device.permission_status else ""
        print(f"  - {device.model_name or 'unknown model'}{serial}{node}{access}{selected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stable camera discovery diagnostics.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--summary", action="store_true", help="Discover both camera types.")
    parser.add_argument("--realsense-check", action="store_true", help="Select RealSense camera.")
    parser.add_argument("--davis-check", action="store_true", help="Select DAVIS camera and check permissions.")
    args = parser.parse_args(argv)
    if not (args.summary or args.realsense_check or args.davis_check):
        args.summary = True
    payload = {}
    exit_code = 0
    if args.summary or args.realsense_check:
        rs = select_realsense_device()
        payload["realsense"] = rs.to_dict()
        if not rs.selected:
            exit_code = max(exit_code, 2)
    if args.summary or args.davis_check:
        davis = select_davis_device()
        payload["davis"] = davis.to_dict()
        if not davis.selected or davis.status == PERMISSION_DENIED:
            exit_code = max(exit_code, 2)
    payload["selection_policy"] = {
        "persistent_identity": "serial/model/VID_PID",
        "transient_usb_node": "diagnostic_only",
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if "realsense" in payload:
            _print_human(select_realsense_device())
        if "davis" in payload:
            _print_human(select_davis_device())
        print("Device selection mode: auto unless serial is configured.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

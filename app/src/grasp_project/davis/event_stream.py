"""DAVIS post-grasp event monitor and motion feature helpers.

Purpose: manage DAVIS event access, ROI clipping, event metrics, motion
classification inputs, and report writing for post-grasp safety feedback.
Main demo: used by the DAVIS Motion tab in the main web dashboard.
Phase note: migrated in Phase 3; top-level `event_postgrasp_monitor.py` is a
compatibility wrapper.
"""

import json
import argparse
import ctypes
import os
import re
import stat
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from grasp_project.davis.motion_estimation import DavisMotionEstimator, load_davis_motion_config
from grasp_project.hardware.device_discovery import select_davis_device


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"

USE_DAVIS_POST_GRASP = os.environ.get("USE_DAVIS_POST_GRASP", "0").strip().lower() in {"1", "true", "yes", "on"}
DAVIS_ROI_XYWH = None
DAVIS_FRAME_SIZE = (346, 260)
DAVIS_LIVE_ROLLING_WINDOW_SEC = 1.2
DAVIS_LIVE_HISTORY_SEC = 2.0
DAVIS_IMPACT_WINDOW_SEC = (0.0, 0.3)
DAVIS_SETTLE_WINDOW_SEC = (0.3, 0.7)
DAVIS_STABILITY_WINDOW_SEC = (0.7, 1.2)
DAVIS_INITIAL_IMPACT_WINDOW_SEC = DAVIS_IMPACT_WINDOW_SEC
EVENT_RATE_LOW_THRESHOLD = 1200.0
EVENT_RATE_HIGH_THRESHOLD = 4500.0
SLIP_VERTICAL_BIAS_THRESHOLD = 0.22
ROTATION_ASYMMETRY_THRESHOLD = 0.26
VIBRATION_EVENT_DENSITY_THRESHOLD = 0.055
IMPACT_DECAY_RATIO_THRESHOLD = 0.35
MIN_EVENTS_FOR_CLASSIFICATION = 80
DAVIS_SLIP_SCORE_THRESHOLD = 60.0
DAVIS_ROTATION_SCORE_THRESHOLD = 55.0
DAVIS_VIBRATION_SCORE_THRESHOLD = 58.0
DAVIS_MIN_STABILITY_SCORE = 65.0
# Post-impact thresholds are intentionally more sensitive than the legacy
# whole-window thresholds. Tune these when the camera/ROI or gripper stiffness
# changes. Impact is judged separately; persistent settle/stability motion is
# what raises slip, rotation, and vibration risk.
POST_IMPACT_EVENT_RATE_LOW = 350.0
POST_IMPACT_EVENT_RATE_MEDIUM = 900.0
POST_IMPACT_EVENT_RATE_HIGH = 2200.0
SLIP_VERTICAL_BIAS_SENSITIVE_THRESHOLD = 0.12
ROTATION_ASYMMETRY_SENSITIVE_THRESHOLD = 0.14
VIBRATION_DENSITY_SENSITIVE_THRESHOLD = 0.012
STABILITY_EVENT_RATE_THRESHOLD = 450.0
IMPACT_SUPPRESSION_DECAY_THRESHOLD = 0.30
FINAL_SLIP_RISK_THRESHOLD = 45.0
FINAL_ROTATION_RISK_THRESHOLD = 45.0
FINAL_VIBRATION_RISK_THRESHOLD = 42.0
LIBCAER_VENDOR_LIB = Path("/opt/ros/humble/opt/libcaer_vendor/lib/libcaer.so.3")
DAVIS_OWNER_LOCK_PATH = OUTPUT_DIR / "davis_backend.lock"


SENSITIVITY_PRESETS = {
    "conservative": {
        "post_impact_event_rate_low": 550.0,
        "post_impact_event_rate_medium": 1300.0,
        "post_impact_event_rate_high": 3000.0,
        "slip_vertical_bias_sensitive_threshold": 0.18,
        "rotation_asymmetry_sensitive_threshold": 0.20,
        "vibration_density_sensitive_threshold": 0.020,
        "stability_event_rate_threshold": 600.0,
        "impact_suppression_decay_threshold": 0.24,
        "final_slip_risk_threshold": 58.0,
        "final_rotation_risk_threshold": 58.0,
        "final_vibration_risk_threshold": 55.0,
    },
    "balanced": {
        "post_impact_event_rate_low": POST_IMPACT_EVENT_RATE_LOW,
        "post_impact_event_rate_medium": POST_IMPACT_EVENT_RATE_MEDIUM,
        "post_impact_event_rate_high": POST_IMPACT_EVENT_RATE_HIGH,
        "slip_vertical_bias_sensitive_threshold": SLIP_VERTICAL_BIAS_SENSITIVE_THRESHOLD,
        "rotation_asymmetry_sensitive_threshold": ROTATION_ASYMMETRY_SENSITIVE_THRESHOLD,
        "vibration_density_sensitive_threshold": VIBRATION_DENSITY_SENSITIVE_THRESHOLD,
        "stability_event_rate_threshold": STABILITY_EVENT_RATE_THRESHOLD,
        "impact_suppression_decay_threshold": IMPACT_SUPPRESSION_DECAY_THRESHOLD,
        "final_slip_risk_threshold": FINAL_SLIP_RISK_THRESHOLD,
        "final_rotation_risk_threshold": FINAL_ROTATION_RISK_THRESHOLD,
        "final_vibration_risk_threshold": FINAL_VIBRATION_RISK_THRESHOLD,
    },
    "sensitive": {
        "post_impact_event_rate_low": 220.0,
        "post_impact_event_rate_medium": 650.0,
        "post_impact_event_rate_high": 1700.0,
        "slip_vertical_bias_sensitive_threshold": 0.08,
        "rotation_asymmetry_sensitive_threshold": 0.10,
        "vibration_density_sensitive_threshold": 0.008,
        "stability_event_rate_threshold": 300.0,
        "impact_suppression_decay_threshold": 0.36,
        "final_slip_risk_threshold": 35.0,
        "final_rotation_risk_threshold": 35.0,
        "final_vibration_risk_threshold": 34.0,
    },
}


def sensitivity_thresholds(name):
    return dict(SENSITIVITY_PRESETS.get(name, SENSITIVITY_PRESETS["balanced"]))


def _prepare_libcaer_vendor():
    if not LIBCAER_VENDOR_LIB.exists():
        return
    lib_dir = str(LIBCAER_VENDOR_LIB.parent)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + current if current else "")
    try:
        ctypes.CDLL(str(LIBCAER_VENDOR_LIB), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path, payload):
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w") as f:
        json.dump(_json_safe(payload), f, indent=4)
    tmp.replace(path)


def _write_text(path, text):
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    tmp.replace(path)


def _process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _process_identity(pid):
    """Return a PID-reuse-safe identity for a Linux process when available."""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        fields = Path(f"/proc/{int(pid)}/stat").read_text().split()
        return f"{boot_id}:{fields[21]}"
    except Exception:
        return None


def _process_start_time_epoch(pid):
    """Best-effort process start time used to retire legacy PID-only locks."""
    try:
        clock_ticks = float(os.sysconf("SC_CLK_TCK"))
        fields = Path(f"/proc/{int(pid)}/stat").read_text().split()
        boot_line = next(
            line for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        return float(boot_line.split()[1]) + float(fields[21]) / clock_ticks
    except Exception:
        return None


def read_davis_owner_lock():
    try:
        with open(DAVIS_OWNER_LOCK_PATH, "r") as f:
            payload = json.load(f)
        owner_pid = payload.get("davis_writer_pid")
        stored_identity = payload.get("davis_writer_process_identity")
        current_identity = _process_identity(owner_pid) if owner_pid else None
        process_start_epoch = _process_start_time_epoch(owner_pid) if owner_pid else None
        owner_start_epoch = payload.get("davis_owner_start_time")
        pid_reused = bool(
            stored_identity and current_identity and stored_identity != current_identity
        )
        legacy_pid_reused = bool(
            not stored_identity
            and process_start_epoch is not None
            and owner_start_epoch is not None
            and process_start_epoch > float(owner_start_epoch) + 1.0
        )
        if owner_pid and (
            not _process_alive(owner_pid) or pid_reused or legacy_pid_reused
        ):
            payload["davis_owner_status"] = "stale_owner"
            payload["duplicate_davis_owner_detected"] = False
            payload["davis_device_open_count"] = 0
            payload["stale_owner_reason"] = (
                "pid_reused_process_identity_mismatch"
                if pid_reused
                else "legacy_pid_reused_after_owner_start"
                if legacy_pid_reused
                else "owner_process_not_alive"
            )
        payload["duplicate_open_prevented"] = bool(payload.get("duplicate_davis_owner_detected"))
        return payload
    except FileNotFoundError:
        return {
            "davis_owner_status": "unowned",
            "duplicate_davis_owner_detected": False,
            "duplicate_open_prevented": False,
            "davis_device_open_count": 0,
        }
    except Exception as exc:
        return {
            "davis_owner_status": "owner_lock_unreadable",
            "duplicate_davis_owner_detected": False,
            "duplicate_open_prevented": False,
            "davis_device_open_count": 0,
            "davis_owner_error": str(exc),
        }


def acquire_davis_owner(owner_label="event_stream"):
    OUTPUT_DIR.mkdir(exist_ok=True)
    session_id = uuid.uuid4().hex
    payload = {
        "davis_writer_pid": os.getpid(),
        "davis_writer_process_identity": _process_identity(os.getpid()),
        "davis_writer_session_id": session_id,
        "davis_owner_label": owner_label,
        "davis_owner_start_time": time.time(),
        "davis_owner_status": "owned",
        "davis_device_open_count": 1,
        "duplicate_davis_owner_detected": False,
        "duplicate_open_prevented": False,
    }
    for _attempt in range(2):
        try:
            fd = os.open(str(DAVIS_OWNER_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            return True, payload
        except FileExistsError:
            existing = read_davis_owner_lock()
            owner_pid = existing.get("davis_writer_pid")
            if existing.get("davis_owner_status") == "stale_owner" or (
                owner_pid and not _process_alive(owner_pid)
            ):
                try:
                    DAVIS_OWNER_LOCK_PATH.unlink()
                    continue
                except Exception:
                    pass
            existing.update({
                "davis_owner_status": "duplicate_owner_blocked",
                "duplicate_davis_owner_detected": True,
                "duplicate_open_prevented": True,
                "davis_device_open_count": 0,
            })
            return False, existing
        except Exception as exc:
            payload.update({
                "davis_owner_status": "owner_lock_error",
                "duplicate_davis_owner_detected": False,
                "duplicate_open_prevented": False,
                "davis_device_open_count": 0,
                "davis_owner_error": str(exc),
            })
            return False, payload
    return False, read_davis_owner_lock()


def release_davis_owner(session_id):
    if not session_id:
        return False
    existing = read_davis_owner_lock()
    if existing.get("davis_writer_session_id") != session_id:
        return False
    try:
        DAVIS_OWNER_LOCK_PATH.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _load_dv_backend():
    errors = []
    _prepare_libcaer_vendor()
    try:
        from pyaer.davis import DAVIS  # noqa: F401
        return True, "pyaer", None
    except Exception as exc:
        errors.append(f"pyaer: {exc}")
    try:
        import dv_processing as dv  # noqa: F401
        return True, "dv_processing", None
    except Exception as exc:
        errors.append(f"dv_processing: {exc}")
    try:
        import dv  # noqa: F401
        return True, "dv", None
    except Exception as exc:
        errors.append(f"dv: {exc}")
    return False, None, "DAVIS post-grasp monitor unavailable: " + "; ".join(errors)


def _backend_import_status():
    statuses = {}
    _prepare_libcaer_vendor()
    for module_name in ("pyaer", "dv_processing", "dv"):
        try:
            __import__(module_name)
            statuses[module_name] = {"available": True, "error": None}
        except Exception as exc:
            statuses[module_name] = {"available": False, "error": str(exc)}
    return statuses


def _run_text(command):
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as exc:
        return 127, "", str(exc)


def _detect_davis_usb_devices():
    result = select_davis_device()
    devices = []
    for device in result.devices:
        raw = device.raw or {}
        devices.append({
            "lsusb_line": raw.get("lsusb_line") or f"ID {device.vendor_id}:{device.product_id} {device.model_name or 'DAVIS'}",
            "bus": device.bus,
            "device": device.device_number,
            "idVendor": device.vendor_id,
            "idProduct": device.product_id,
            "description": device.model_name,
            "device_path": device.device_node,
            "exists": raw.get("exists"),
            "readable": raw.get("readable"),
            "writable": raw.get("writable"),
            "permission_ok": raw.get("permission_ok"),
            "mode": raw.get("mode_octal"),
            "uid": raw.get("uid"),
            "gid": raw.get("gid"),
            "selection_status": result.status,
            "selection_reason": result.reason,
        })
    return devices, result.error


def diagnose_davis():
    backends = _backend_import_status()
    selection = select_davis_device()
    devices, usb_error = _detect_davis_usb_devices()
    permission_ok = any(device.get("permission_ok") for device in devices)
    diagnosis = {
        "backend_availability": backends,
        "usb_devices": devices,
        "usb_error": usb_error,
        "permission_ok": permission_ok,
        "selection": selection.to_dict(),
        "suggested_fix": None,
    }
    if not devices:
        diagnosis["suggested_fix"] = "Plug in the DAVIS346 and run: bash scripts/check_davis_permissions.sh"
    elif not permission_ok:
        diagnosis["suggested_fix"] = (
            "USB permission is not OK. Permanent fix: bash scripts/setup_davis_udev.sh, "
            "then unplug/replug. Temporary fix: bash scripts/temp_fix_davis_permissions.sh"
        )
    return diagnosis


def print_davis_diagnosis():
    diagnosis = diagnose_davis()
    print("DAVIS backend availability:")
    for name, status in diagnosis["backend_availability"].items():
        if status["available"]:
            print(f"  PASS {name}")
        else:
            print(f"  WARN {name}: {status['error']}")
    print()
    if diagnosis["usb_error"]:
        print(f"USB diagnostic error: {diagnosis['usb_error']}")
    if not diagnosis["usb_devices"]:
        print("FAIL: No DAVIS/iniVation USB device detected.")
    for device in diagnosis["usb_devices"]:
        state = "PASS" if device["permission_ok"] else "FAIL"
        print(f"{state}: {device['lsusb_line']}")
        print(f"  path: {device['device_path']}")
        print(f"  exists/read/write: {device['exists']}/{device['readable']}/{device['writable']}")
        if "mode" in device:
            print(f"  mode: {device['mode']} uid={device['uid']} gid={device['gid']}")
    if diagnosis["suggested_fix"]:
        print()
        print("Suggested fix:")
        print(f"  {diagnosis['suggested_fix']}")
    print()
    print(json.dumps(_json_safe(diagnosis), indent=4))
    return 0 if diagnosis["permission_ok"] else 2


def _clip_roi(roi_xywh, frame_size=DAVIS_FRAME_SIZE):
    width, height = frame_size
    if roi_xywh is None:
        return [0, 0, int(width), int(height)]
    x, y, w, h = [int(round(float(v))) for v in roi_xywh]
    x = max(0, min(x, int(width) - 1))
    y = max(0, min(y, int(height) - 1))
    w = max(1, min(w, int(width) - x))
    h = max(1, min(h, int(height) - y))
    return [x, y, w, h]


def roi_from_locked_target(baseline, default_roi=DAVIS_ROI_XYWH, frame_size=DAVIS_FRAME_SIZE):
    if default_roi is not None:
        return _clip_roi(default_roi, frame_size)
    target_clamp = (baseline or {}).get("target_clamp") or {}
    points = []
    for key in ("left_clamp_patch", "right_clamp_patch"):
        polygon = (target_clamp.get(key) or {}).get("polygon")
        if polygon:
            points.extend(polygon)
    if target_clamp.get("clamp_band_polygon"):
        points.extend(target_clamp["clamp_band_polygon"])
    if not points:
        return _clip_roi(None, frame_size)
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return _clip_roi(None, frame_size)
    rs_w, rs_h = 848.0, 480.0
    ev_w, ev_h = frame_size
    x1 = np.min(arr[:, 0]) * ev_w / rs_w
    y1 = np.min(arr[:, 1]) * ev_h / rs_h
    x2 = np.max(arr[:, 0]) * ev_w / rs_w
    y2 = np.max(arr[:, 1]) * ev_h / rs_h
    pad = 18.0
    return _clip_roi([x1 - pad, y1 - pad, (x2 - x1) + 2 * pad, (y2 - y1) + 2 * pad], frame_size)


class EventPostGraspMonitor:
    def __init__(self, enabled=USE_DAVIS_POST_GRASP, roi_xywh=DAVIS_ROI_XYWH, output_dir=OUTPUT_DIR, sensitivity="balanced", recording_sink=None):
        self.enabled = bool(enabled)
        self.configured_roi = roi_xywh
        self.output_dir = Path(output_dir)
        self.sensitivity = sensitivity if sensitivity in SENSITIVITY_PRESETS else "balanced"
        self.thresholds = sensitivity_thresholds(self.sensitivity)
        self.recording_sink = recording_sink
        self.recording_stream_id = uuid.uuid4().hex
        self.recording_poll_sequence = 0
        self.recording_normalization_epoch = 0
        self.latest_event_host_arrival_monotonic = None
        self.latest_event_host_arrival_epoch = None
        self.backend_available, self.backend_name, self.unavailable_reason = _load_dv_backend()
        self.device = None
        self.device_open_error = None
        self.event_time_zero_us = None
        self.event_time_wall_zero = None
        self.event_device_last_us = None
        self.event_normalized_last = None
        self.event_timestamp_reanchor_count = 0
        self.event_timestamp_basis = "device_microseconds_host_anchored"
        self.latest_aps_device_timestamp_us = None
        self.active = False
        self.start_time = None
        self.owner_info = read_davis_owner_lock()
        self.owner_session_id = None
        self.roi_xywh = _clip_roi(roi_xywh)
        self.events = []
        self.motion_estimator = DavisMotionEstimator(load_davis_motion_config(), frame_size=DAVIS_FRAME_SIZE)
        self.last_report = self._base_report()

    def _base_report(self):
        usable = bool(self.backend_available and not self.device_open_error)
        status = "disabled" if not self.enabled else ("available" if usable else "unavailable")
        return {
            "use_davis_post_grasp": self.enabled,
            "davis_backend_available": bool(self.enabled and usable),
            "davis_backend": self.backend_name,
            "davis_status": status,
            "davis_unavailable_reason": (self.device_open_error or self.unavailable_reason) if self.enabled and not usable else None,
            "davis_roi_xywh": self.roi_xywh,
            "event_count": 0,
            "event_rate": 0.0,
            "event_density": 0.0,
            "vertical_event_bias": 0.0,
            "left_right_event_asymmetry": 0.0,
            "motion_spike_score": 0.0,
            "slip_score": 0.0,
            "rotation_score": 0.0,
            "raw_slip_pattern_score": 0.0,
            "raw_rotation_pattern_score": 0.0,
            "raw_vibration_pattern_score": 0.0,
            "final_slip_risk_score": 0.0,
            "final_rotation_risk_score": 0.0,
            "final_vibration_risk_score": 0.0,
            "final_stability_score": 100.0,
            "instability_risk_score": 0.0,
            "stability_score": 100.0,
            "slip_severity": "NONE",
            "rotation_severity": "NONE",
            "vibration_severity": "NONE",
            "impact_suppressed": False,
            "impact_suppression_applied": False,
            "slip_risk_score": 0.0,
            "rotation_risk_score": 0.0,
            "vibration_score": 0.0,
            "impact_score": 0.0,
            "confidence_score": 0.0,
            "post_grasp_stability_score": 100.0,
            "post_grasp_motion_detected": False,
            "post_grasp_slip_risk": False,
            "post_grasp_rotation_risk": False,
            "post_grasp_vibration_detected": False,
            "motion_class": "SENSOR_UNAVAILABLE" if self.enabled else "STABLE",
            "post_grasp_decision": "REALSENSE_ONLY" if not self.enabled else "UNAVAILABLE",
            "post_grasp_event_count": 0,
            "post_grasp_event_rate": 0.0,
            "motion_estimate": {},
            "motion_estimate_valid": False,
            "motion_estimate_invalid_reason": "not_running",
            "motion_estimation_method": "unavailable",
            "motion_estimation_confidence": 0.0,
            "motion_confidence": 0.0,
            "dx_px": None,
            "dy_px": None,
            "displacement_px": None,
            "cumulative_slip_px": 0.0,
            "downward_slip_px": None,
            "rotation_delta_deg": None,
            "cumulative_rotation_deg": 0.0,
            "angular_velocity_deg_s": None,
            "vibration_amplitude_px_rms": None,
            "disturbance_type": "none",
            "duration": DAVIS_STABILITY_WINDOW_SEC[1],
            "sensitivity": self.sensitivity,
            "impact_window_sec": list(DAVIS_IMPACT_WINDOW_SEC),
            "settle_window_sec": list(DAVIS_SETTLE_WINDOW_SEC),
            "stability_window_sec": list(DAVIS_STABILITY_WINDOW_SEC),
            "post_grasp_scoring_thresholds": {
                **self.thresholds,
            },
            "window_metrics": {},
            "slip_risk_reason": "DAVIS post-grasp monitoring is not active.",
            "rotation_risk_reason": "DAVIS post-grasp monitoring is not active.",
            "vibration_risk_reason": "DAVIS post-grasp monitoring is not active.",
            "stability_reason": "DAVIS post-grasp monitoring is not active.",
            "motion_class_reason": "",
            "decision_reason": "",
            "explanation": "DAVIS post-grasp monitoring is disabled." if not self.enabled else "",
            **(self.owner_info or read_davis_owner_lock()),
        }

    def start(self, baseline=None):
        if self.device is not None or self.owner_session_id:
            # release the previous grasp's device handle and owner lock so a
            # second start in the same session does not deadlock on itself
            self.stop()
        self.start_time = time.monotonic()
        self.recording_normalization_epoch = getattr(self, "recording_normalization_epoch", 0) + 1
        self.latest_event_host_arrival_monotonic = None
        self.latest_event_host_arrival_epoch = None
        self.roi_xywh = roi_from_locked_target(baseline, self.configured_roi)
        self.events = []
        self.device_open_error = None
        self.event_time_zero_us = None
        self.event_time_wall_zero = None
        self.event_device_last_us = None
        self.event_normalized_last = None
        self.event_timestamp_reanchor_count = 0
        self.latest_aps_device_timestamp_us = None
        self.motion_estimator.reset()
        davis_selection = select_davis_device() if self.enabled else None
        self.active = bool(self.enabled and self.backend_available)
        if self.enabled and davis_selection and not davis_selection.selected:
            self.active = False
            self.device_open_error = davis_selection.reason or "DAVIS device not discovered."
        elif self.enabled and davis_selection and davis_selection.status == "PERMISSION_DENIED":
            self.active = False
            self.device_open_error = davis_selection.reason
        if self.active:
            owned, owner_info = acquire_davis_owner("EventPostGraspMonitor")
            self.owner_info = owner_info
            self.owner_session_id = owner_info.get("davis_writer_session_id") if owned else None
            if not owned:
                opened = False
                error = "DAVIS device already owned by another process/session."
            else:
                opened, error = self._open_backend()
                if not opened:
                    release_davis_owner(self.owner_session_id)
                    self.owner_session_id = None
                    self.owner_info = read_davis_owner_lock()
            self.active = bool(opened)
            if error:
                self.device_open_error = error
        self.last_report = self._base_report()
        self.last_report.update({
            "davis_status": "monitoring" if self.active else self.last_report["davis_status"],
            "davis_roi_xywh": self.roi_xywh,
            "post_grasp_decision": "POST_GRASP_EVALUATION" if self.active else self.last_report["post_grasp_decision"],
        })
        self.write_report(self.last_report)
        return self.last_report

    def _open_backend(self):
        if self.backend_name == "pyaer":
            try:
                _prepare_libcaer_vendor()
                from pyaer.davis import DAVIS
                self.device = DAVIS(noise_filter=False)
                self.device.send_default_config()
                self.device.set_data_exchange_blocking(False)
                started = self.device.data_start()
                if not started:
                    raise RuntimeError("data_start() returned false")
                return True, None
            except Exception as exc:
                self.device = None
                return False, f"DAVIS pyaer backend failed to open: {exc}"
        if self.backend_name in {"dv_processing", "dv"}:
            return False, f"DAVIS {self.backend_name} backend detected, but live capture adapter is not configured yet."
        return False, self.unavailable_reason or "No DAVIS backend available."

    def stop(self):
        if self.device is not None:
            try:
                self.device.shutdown()
            except Exception:
                try:
                    self.device.close()
                except Exception:
                    pass
        release_davis_owner(self.owner_session_id)
        self.owner_info = read_davis_owner_lock()
        self.owner_session_id = None
        self.device = None
        self.active = False
        self.event_time_zero_us = None
        self.event_time_wall_zero = None
        self.event_device_last_us = None
        self.event_normalized_last = None
        self.latest_aps_device_timestamp_us = None

    def poll_events(self):
        array = self.poll_event_array()
        if array.size == 0:
            return []
        return [
            {"t": float(row[0]), "x": float(row[1]), "y": float(row[2]), "polarity": int(row[3])}
            for row in array
        ]

    def poll_event_array(self):
        events, _frames = self.poll_event_and_frame_arrays()
        return events

    def poll_event_and_frame_arrays(self):
        if not self.active or self.device is None:
            return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.uint8)
        try:
            data = self.device.get_event("events")
        except Exception as exc:
            sink = getattr(self, "recording_sink", None)
            if sink is not None:
                sink.record_drop("davis_events", None, "device_poll_failed", time.monotonic_ns())
            self.unavailable_reason = f"DAVIS event polling failed: {exc}"
            self.active = False
            return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.uint8)
        arrival_monotonic_ns = time.monotonic_ns()
        poll_monotonic = time.monotonic()
        arrival_epoch_ns = time.time_ns()
        if data is None:
            self._record_raw_packet(np.empty((0, 4), dtype=np.int64), np.empty(0, dtype=np.uint8), None, poll_monotonic, arrival_monotonic_ns, arrival_epoch_ns, "no_packet")
            return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.uint8)
        pol_events = data[0] if isinstance(data, (tuple, list)) and data else data
        frame_timestamps = data[4] if isinstance(data, (tuple, list)) and len(data) > 4 else None
        frames = data[5] if isinstance(data, (tuple, list)) and len(data) > 5 else None
        frame_arr = np.asarray(frames) if frames is not None else np.empty((0,), dtype=np.uint8)
        native_events = np.asarray(pol_events) if pol_events is not None else np.empty((0, 4), dtype=np.int64)
        self._record_raw_packet(native_events, frame_arr, frame_timestamps, poll_monotonic, arrival_monotonic_ns, arrival_epoch_ns, "packet_returned")
        if frame_arr.size:
            timestamp_arr = (
                np.asarray(frame_timestamps).reshape(-1)
                if frame_timestamps is not None
                else np.empty((0,), dtype=np.uint64)
            )
            self.latest_aps_device_timestamp_us = (
                int(timestamp_arr[-1]) if timestamp_arr.size else None
            )
        if pol_events is None:
            return np.empty((0, 4), dtype=np.float64), frame_arr
        arr = native_events
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float64), frame_arr
        self.latest_event_host_arrival_monotonic = poll_monotonic
        self.latest_event_host_arrival_epoch = arrival_epoch_ns / 1_000_000_000.0
        return self.normalize_native_event_array(arr[:, :4], poll_monotonic), frame_arr

    def _record_raw_packet(self, events, frames, frame_timestamps, arrival, monotonic_ns, epoch_ns, poll_state):
        """Opt-in lossless capture before native event times become host estimates."""
        sink = getattr(self, "recording_sink", None)
        if sink is None:
            return
        sequence = self.recording_poll_sequence
        if sequence == 0:
            sink.record_drop("davis_events", None, "backend_packet_loss_not_observable", monotonic_ns)
        sink.record(
            "davis_poll",
            {"events": events, "aps_frames": frames, "aps_timestamps_us": np.asarray(frame_timestamps) if frame_timestamps is not None else np.empty(0, dtype=np.uint64)},
            observation_id=f"davis-{self.recording_stream_id}-{sequence}",
            host_arrival_monotonic_ns=monotonic_ns,
            host_arrival_epoch_ns=epoch_ns,
            native_clock={"domain": "davis_device", "unit": "us"},
            metadata={"poll_state": poll_state, "event_columns": ["native_timestamp_us", "x", "y", "polarity"], "roi_xywh": list(self.roi_xywh), "frame_size": list(DAVIS_FRAME_SIZE), "backend": getattr(self, "backend_name", "unknown"), "source_sequence_basis": "host_poll_counter_not_device_packet_counter", "acquisition_endpoint": "device_get_event_return", "normalization_reference_monotonic_sec": arrival, "normalization_epoch": getattr(self, "recording_normalization_epoch", 0)},
            source_sequence=sequence,
        )
        self.recording_poll_sequence += 1

    def normalize_native_event_array(self, native_events, poll_monotonic):
        """Canonical live/replay mapping; input native timestamps remain untouched.

        Output times are host-monotonic estimates, not clock alignment. Replay
        must use the original arrival clock and start with the same mapping state.
        """
        raw = np.asarray(native_events)
        if raw.dtype.kind not in "uif":
            raise ValueError("native event columns must be real numeric values")
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("native events must have at least four columns")
        if not np.isfinite(arr[:, :4]).all() or not np.isfinite(poll_monotonic):
            raise ValueError("event values and arrival time must be finite")
        if not len(arr):
            return np.empty((0, 4), dtype=np.float64)
        raw_first_us = float(arr[0, 0])
        raw_last_us = float(arr[-1, 0])
        previous_device_last_us = getattr(self, "event_device_last_us", None)
        device_clock_reset = bool(
            previous_device_last_us is not None
            and raw_last_us + 1.0 < float(previous_device_last_us)
        )
        if (
            getattr(self, "event_time_zero_us", None) is None
            or getattr(self, "event_time_wall_zero", None) is None
            or device_clock_reset
        ):
            packet_span_sec = max(0.0, (raw_last_us - raw_first_us) / 1_000_000.0)
            self.event_time_zero_us = raw_first_us
            # Anchor the newest event in the first/reset packet to its measured
            # host arrival time. The previous implementation anchored to
            # start_time, so any device-open delay permanently put every event
            # behind the live 100 ms analysis window.
            self.event_time_wall_zero = poll_monotonic - packet_span_sec
            if device_clock_reset:
                self.event_timestamp_reanchor_count = int(
                    getattr(self, "event_timestamp_reanchor_count", 0)
                ) + 1
        device_relative = np.maximum(
            0.0,
            (arr[:, 0] - float(self.event_time_zero_us)) / 1_000_000.0,
        )
        wall_relative = max(
            0.0,
            poll_monotonic - float(self.event_time_wall_zero),
        )
        # Device deltas preserve real intra-packet timing; the host arrival
        # clamp prevents future-dated events from entering a later window.
        relative_t = np.minimum(device_relative, wall_relative)
        normalized_t = np.minimum(
            np.maximum.accumulate(float(self.event_time_wall_zero) + relative_t),
            poll_monotonic,
        )
        output = np.empty((arr.shape[0], 4), dtype=np.float64)
        output[:, 0] = normalized_t
        output[:, 1] = arr[:, 1]
        output[:, 2] = arr[:, 2]
        output[:, 3] = arr[:, 3]
        self.event_device_last_us = raw_last_us
        self.event_normalized_last = float(output[-1, 0])
        return output

    def add_event_array(self, event_array):
        if event_array is None:
            return
        arr = np.asarray(event_array)
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return
        self.events.extend(map(tuple, arr[:, :4]))

    def add_events(self, events):
        if events is None:
            return
        for event in events:
            normalized = self._normalize_event(event)
            if normalized is not None:
                self.events.append(normalized)

    def prune_events(self, keep_after_time):
        self.events = [event for event in self.events if event[0] >= keep_after_time]

    def get_events_between(self, start_time, end_time):
        """Return buffered events with timestamps in [start_time, end_time]."""
        start = float(start_time)
        end = float(end_time)
        return np.asarray([e for e in self.events if start <= e[0] <= end], dtype=np.float64).reshape(-1, 4)

    def get_recent_events(self, window_ms=100.0):
        """Return recent buffered events from the rolling software timestamp buffer."""
        now = time.monotonic()
        start = now - float(window_ms) / 1000.0
        return self.get_events_between(start, now)

    def get_event_heatmap_for_window(self, window_ms=100.0):
        """Return an RGB heatmap for recent DAVIS events."""
        try:
            from grasp_project.davis.motion_gui import heatmap_rgb

            return heatmap_rgb(self.get_recent_events(window_ms), self.roi_xywh)
        except Exception:
            return None

    def get_event_motion_features_for_window(self, window_ms=100.0):
        """Return lightweight motion features for recent DAVIS events."""
        events = self.get_recent_events(window_ms)
        duration = max(float(window_ms) / 1000.0, 1e-6)
        if events.size == 0:
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
        x, y, w, h = self.roi_xywh
        metrics = self._metrics([tuple(e) for e in events], duration, w, h)
        xs = events[:, 1]
        ys = events[:, 2]
        return {
            "event_count": metrics["event_count"],
            "event_rate": metrics["event_rate"],
            "event_centroid": [metrics["mean_x"], metrics["mean_y"]],
            "event_motion_bbox": [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))],
            "event_vertical_bias": metrics["vertical_event_bias"],
            "event_horizontal_bias": metrics["horizontal_event_bias"],
            "event_locality_score": float(np.clip(100.0 / max(float(np.hypot(metrics["std_x"], metrics["std_y"])), 1.0), 0.0, 100.0)),
            "event_motion_class": "EVENT_MOTION",
        }

    def _normalize_event(self, event):
        try:
            if isinstance(event, dict):
                x = event.get("x")
                y = event.get("y")
                t = event.get("t", event.get("timestamp", time.monotonic()))
                polarity = event.get("polarity", event.get("p", 1))
            else:
                x = getattr(event, "x")
                y = getattr(event, "y")
                t = getattr(event, "timestamp", getattr(event, "t", time.monotonic()))
                polarity = getattr(event, "polarity", getattr(event, "p", 1))
            return float(t), float(x), float(y), int(bool(polarity))
        except Exception:
            return None

    def update(self, events=None, write_report=True, analysis_origin=None, prune_before=None):
        now = time.monotonic()
        if events is None and self.active:
            events = self.poll_events()
        if events is not None:
            self.add_events(events)
        self.prune_events(float(prune_before) if prune_before is not None else now - DAVIS_LIVE_HISTORY_SEC)
        if not self.active:
            self.last_report = self._base_report()
            if write_report:
                self.write_report(self.last_report)
            return self.last_report
        origin = float(analysis_origin if analysis_origin is not None else (self.start_time or now))
        elapsed = max(now - origin, 1e-6)
        window_origin = origin if elapsed <= DAVIS_LIVE_ROLLING_WINDOW_SEC else now - DAVIS_LIVE_ROLLING_WINDOW_SEC
        x, y, w, h = self.roi_xywh
        roi_events = [e for e in self.events if e[0] >= window_origin and x <= e[1] < x + w and y <= e[2] < y + h]
        window_metrics = self._window_metrics(roi_events, w, h, window_origin)
        total_duration = max(min(elapsed, DAVIS_STABILITY_WINDOW_SEC[1]), 1e-6)
        total_metrics = self._metrics(roi_events, total_duration, w, h)
        classification = self._classify(window_metrics, total_metrics, elapsed)
        motion_estimate = self.motion_estimator.estimate(
            self.events,
            now=now,
            roi_xywh=self.roi_xywh,
            roi_source="configured_roi" if self.configured_roi is not None else "full_frame",
            frame_size=DAVIS_FRAME_SIZE,
        )
        estimate_dict = motion_estimate.to_dict()
        active_window = self._active_window(elapsed)
        active_metrics = window_metrics.get(active_window) or total_metrics
        decision = classification["post_grasp_decision"]
        if elapsed < DAVIS_STABILITY_WINDOW_SEC[1] and decision not in {"REGRIP", "DO_NOT_LIFT"}:
            decision = "POST_GRASP_EVALUATION"
        report = {
            **self._base_report(),
            **classification,
            **active_metrics,
            "event_count": total_metrics["event_count"],
            "event_rate": total_metrics["event_rate"],
            "event_density": total_metrics["event_density"],
            "raw_event_rate": total_metrics["event_rate"],
            "raw_event_density": total_metrics["event_density"],
            "raw_vertical_bias": total_metrics["vertical_event_bias"],
            "raw_horizontal_bias": total_metrics["horizontal_event_bias"],
            "raw_left_right_asymmetry": total_metrics["left_right_event_asymmetry"],
            "raw_top_bottom_asymmetry": total_metrics["top_bottom_asymmetry"],
            "raw_centroid_shift": total_metrics["event_centroid_shift"],
            "raw_impact_spike": classification["impact_score"],
            "total_event_count": total_metrics["event_count"],
            "impact_event_count": window_metrics["impact"]["event_count"],
            "settle_event_count": window_metrics["settle"]["event_count"],
            "stability_event_count": window_metrics["stability"]["event_count"],
            "impact_event_rate": window_metrics["impact"]["event_rate"],
            "settle_event_rate": window_metrics["settle"]["event_rate"],
            "stability_event_rate": window_metrics["stability"]["event_rate"],
            "davis_status": "monitoring" if elapsed < DAVIS_STABILITY_WINDOW_SEC[1] else "complete",
            "davis_roi_xywh": self.roi_xywh,
            "elapsed_sec": elapsed,
            "duration": DAVIS_STABILITY_WINDOW_SEC[1],
            "analysis_window_mode": "initial_post_close" if elapsed <= DAVIS_LIVE_ROLLING_WINDOW_SEC else "rolling_after_initial_post_close",
            "analysis_window_sec": DAVIS_LIVE_ROLLING_WINDOW_SEC,
            "retained_history_sec": DAVIS_LIVE_HISTORY_SEC,
            "active_window": active_window,
            "window_metrics": window_metrics,
            "post_grasp_decision": decision,
            "post_grasp_event_count": total_metrics["event_count"],
            "post_grasp_event_rate": total_metrics["event_rate"],
            "davis_event_rate": total_metrics["event_rate"],
            "davis_slip_score": classification["slip_risk_score"],
            "davis_rotation_score": classification["rotation_risk_score"],
            "davis_stability_score": classification["post_grasp_stability_score"],
            "motion_estimate": estimate_dict,
            "motion_estimate_id": estimate_dict.get("estimate_id"),
            "motion_estimate_valid": estimate_dict.get("estimate_valid"),
            "motion_estimate_invalid_reason": estimate_dict.get("invalid_reason"),
            "motion_estimation_method": estimate_dict.get("motion_estimation_method"),
            "motion_estimation_confidence": estimate_dict.get("motion_estimation_confidence"),
            "motion_confidence": estimate_dict.get("motion_estimation_confidence"),
            "translation_method": estimate_dict.get("translation_method"),
            "translation_confidence": estimate_dict.get("translation_confidence"),
            "translation_valid": estimate_dict.get("translation_valid"),
            "dx_px": estimate_dict.get("dx_px"),
            "dy_px": estimate_dict.get("dy_px"),
            "displacement_px": estimate_dict.get("displacement_px"),
            "velocity_x_px_s": estimate_dict.get("velocity_x_px_s"),
            "velocity_y_px_s": estimate_dict.get("velocity_y_px_s"),
            "speed_px_s": estimate_dict.get("speed_px_s"),
            "cumulative_slip_px": estimate_dict.get("cumulative_slip_px"),
            "downward_slip_px": estimate_dict.get("downward_slip_px"),
            "cumulative_downward_slip_px": estimate_dict.get("cumulative_downward_slip_px"),
            "rotation_delta_deg": estimate_dict.get("rotation_delta_deg"),
            "cumulative_rotation_deg": estimate_dict.get("cumulative_rotation_deg"),
            "angular_velocity_deg_s": estimate_dict.get("angular_velocity_deg_s"),
            "rotation_confidence": estimate_dict.get("rotation_confidence"),
            "vibration_amplitude_px_rms": estimate_dict.get("vibration_amplitude_px_rms"),
            "vibration_frequency_hz": estimate_dict.get("vibration_frequency_hz"),
            "vibration_confidence": estimate_dict.get("vibration_confidence"),
            "disturbance_type": estimate_dict.get("disturbance_type"),
            "disturbance_confidence": estimate_dict.get("disturbance_confidence"),
            "disturbance_rejection_applied": estimate_dict.get("disturbance_rejection_applied"),
            "quantitative_motion_class": estimate_dict.get("motion_class"),
            "quantitative_evidence": estimate_dict.get("quantitative_evidence"),
        }
        report["slip_score"] = report["slip_risk_score"]
        report["rotation_score"] = report["rotation_risk_score"]
        self.last_report = report
        if write_report:
            self.write_report(report)
        return report

    def _events_in_window(self, events, window, origin=None):
        start, end = window
        origin = float(origin if origin is not None else (self.start_time or 0.0))
        return [e for e in events if start <= e[0] - origin < end]

    def _window_metrics(self, roi_events, roi_w, roi_h, origin=None):
        windows = {
            "impact": DAVIS_IMPACT_WINDOW_SEC,
            "settle": DAVIS_SETTLE_WINDOW_SEC,
            "stability": DAVIS_STABILITY_WINDOW_SEC,
        }
        metrics = {}
        previous = None
        for name, window in windows.items():
            duration = window[1] - window[0]
            values = self._metrics(self._events_in_window(roi_events, window, origin), duration, roi_w, roi_h, previous)
            metrics[name] = values
            if values["event_count"] > 0:
                previous = values
        return metrics

    def _metrics(self, events, duration_sec, roi_w, roi_h, previous_metrics=None):
        count = len(events)
        duration = max(float(duration_sec), 1e-6)
        area = max(float(roi_w * roi_h), 1.0)
        event_rate = float(count / duration)
        density = float(count / area)
        x0, y0, _, _ = self.roi_xywh
        x_mid = float(x0 + roi_w / 2.0)
        y_mid = float(y0 + roi_h / 2.0)
        if count:
            arr = np.asarray(events, dtype=float)
            xs = arr[:, 1]
            ys = arr[:, 2]
            polarities = arr[:, 3]
            positive_count = int(np.count_nonzero(polarities > 0))
            negative_count = int(count - positive_count)
            mean_x = float(np.mean(xs))
            mean_y = float(np.mean(ys))
            std_x = float(np.std(xs))
            std_y = float(np.std(ys))
            left_count = int(np.count_nonzero(xs <= x_mid))
            right_count = int(count - left_count)
            upper_count = int(np.count_nonzero(ys <= y_mid))
            lower_count = int(count - upper_count)
            vertical_bias = float((lower_count - upper_count) / count)
            horizontal_bias = float((right_count - left_count) / count)
            lr_asym = abs(horizontal_bias)
            tb_asym = abs(vertical_bias)
            if previous_metrics and previous_metrics.get("event_count"):
                prev_x = previous_metrics.get("mean_x", mean_x)
                prev_y = previous_metrics.get("mean_y", mean_y)
                centroid_shift_x = float(mean_x - prev_x)
                centroid_shift_y = float(mean_y - prev_y)
                centroid_shift = float(np.hypot(centroid_shift_x, centroid_shift_y))
            else:
                centroid_shift_x = 0.0
                centroid_shift_y = 0.0
                centroid_shift = 0.0
        else:
            positive_count = 0
            negative_count = 0
            mean_x = None
            mean_y = None
            std_x = 0.0
            std_y = 0.0
            left_count = 0
            right_count = 0
            upper_count = 0
            lower_count = 0
            vertical_bias = 0.0
            horizontal_bias = 0.0
            lr_asym = 0.0
            tb_asym = 0.0
            centroid_shift_x = 0.0
            centroid_shift_y = 0.0
            centroid_shift = 0.0
        motion_spike = float(np.clip(100.0 * event_rate / max(EVENT_RATE_HIGH_THRESHOLD, 1.0), 0.0, 100.0))
        return {
            "event_count": int(count),
            "event_rate": event_rate,
            "event_density": density,
            "positive_event_count": positive_count,
            "negative_event_count": negative_count,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "std_x": std_x,
            "std_y": std_y,
            "left_half_event_count": left_count,
            "right_half_event_count": right_count,
            "upper_half_event_count": upper_count,
            "lower_half_event_count": lower_count,
            "vertical_event_bias": vertical_bias,
            "horizontal_event_bias": horizontal_bias,
            "left_right_event_asymmetry": lr_asym,
            "top_bottom_asymmetry": tb_asym,
            "event_centroid_shift_x": centroid_shift_x,
            "event_centroid_shift_y": centroid_shift_y,
            "event_centroid_shift": centroid_shift,
            "centroid_shift_x": centroid_shift_x,
            "centroid_shift_y": centroid_shift_y,
            "motion_spike_score": motion_spike,
            "raw_event_rate": event_rate,
            "raw_event_density": density,
            "raw_vertical_bias": vertical_bias,
            "raw_horizontal_bias": horizontal_bias,
            "raw_left_right_asymmetry": lr_asym,
            "raw_top_bottom_asymmetry": tb_asym,
            "raw_centroid_shift": centroid_shift,
        }

    def _active_window(self, elapsed):
        if elapsed < DAVIS_IMPACT_WINDOW_SEC[1]:
            return "impact"
        if elapsed < DAVIS_SETTLE_WINDOW_SEC[1]:
            return "settle"
        return "stability"

    def _classify(self, window_metrics, total_metrics, elapsed):
        t = self.thresholds
        impact = window_metrics["impact"]
        settle = window_metrics["settle"]
        stability = window_metrics["stability"]
        post_count = settle["event_count"] + stability["event_count"]
        post_rate = max(settle["event_rate"], stability["event_rate"])
        continued_rate = (settle["event_rate"] + stability["event_rate"]) / 2.0
        impact_rate = impact["event_rate"]
        decay_ratio = continued_rate / max(impact_rate, 1.0)
        post_vertical_bias = self._weighted_average(
            [(settle["vertical_event_bias"], settle["event_count"]), (stability["vertical_event_bias"], stability["event_count"])]
        )
        post_lr_asym = self._weighted_average(
            [(settle["left_right_event_asymmetry"], settle["event_count"]), (stability["left_right_event_asymmetry"], stability["event_count"])]
        )
        post_tb_asym = self._weighted_average(
            [(settle["top_bottom_asymmetry"], settle["event_count"]), (stability["top_bottom_asymmetry"], stability["event_count"])]
        )
        post_density = max(settle["event_density"], stability["event_density"])
        centroid_shift = max(settle["event_centroid_shift"], stability["event_centroid_shift"])
        centroid_shift_x = max(abs(settle.get("event_centroid_shift_x", 0.0)), abs(stability.get("event_centroid_shift_x", 0.0)))
        centroid_shift_y_down = max(settle.get("event_centroid_shift_y", 0.0), stability.get("event_centroid_shift_y", 0.0), 0.0)
        post_lower = settle["lower_half_event_count"] + stability["lower_half_event_count"]
        post_upper = settle["upper_half_event_count"] + stability["upper_half_event_count"]
        post_left = settle["left_half_event_count"] + stability["left_half_event_count"]
        post_right = settle["right_half_event_count"] + stability["right_half_event_count"]
        lower_dominance = max(0.0, (post_lower - post_upper) / max(post_count, 1))
        side_dominance = abs(post_right - post_left) / max(post_count, 1)
        persistence_windows = int(settle["event_rate"] >= t["post_impact_event_rate_low"]) + int(stability["event_rate"] >= t["post_impact_event_rate_low"])
        persistence_score = 50.0 * persistence_windows
        post_motion_low = (
            settle["event_rate"] <= t["post_impact_event_rate_low"]
            and stability["event_rate"] <= t["stability_event_rate_threshold"]
            and post_count < MIN_EVENTS_FOR_CLASSIFICATION
        )

        impact_score = float(np.clip(100.0 * impact_rate / max(EVENT_RATE_HIGH_THRESHOLD, 1.0), 0.0, 100.0))
        activity_score = float(np.clip(100.0 * post_rate / max(t["post_impact_event_rate_high"], 1.0), 0.0, 100.0))
        medium_activity_score = float(np.clip(100.0 * max(post_rate - t["post_impact_event_rate_low"], 0.0) / max(t["post_impact_event_rate_medium"] - t["post_impact_event_rate_low"], 1.0), 0.0, 100.0))
        slip_bias_score = float(np.clip(100.0 * max(post_vertical_bias - t["slip_vertical_bias_sensitive_threshold"], 0.0) / 0.45, 0.0, 100.0))
        lower_score = float(np.clip(100.0 * max(lower_dominance - 0.05, 0.0) / 0.45, 0.0, 100.0))
        centroid_y_score = float(np.clip(centroid_shift_y_down * 4.0, 0.0, 100.0))
        downward_evidence_score = float(np.clip(0.45 * slip_bias_score + 0.35 * lower_score + 0.20 * centroid_y_score, 0.0, 100.0))
        raw_slip_score = float(np.clip(
            0.18 * activity_score
            + 0.12 * medium_activity_score
            + 0.35 * slip_bias_score
            + 0.20 * lower_score
            + 0.10 * centroid_y_score
            + 0.05 * persistence_score,
            0.0,
            100.0,
        ))
        if downward_evidence_score < 25.0:
            raw_slip_score *= 0.45
        rotation_asym_score = float(np.clip(100.0 * max(post_lr_asym - t["rotation_asymmetry_sensitive_threshold"], 0.0) / 0.45, 0.0, 100.0))
        horizontal_score = float(np.clip(100.0 * max(abs(self._weighted_average(
            [(settle["horizontal_event_bias"], settle["event_count"]), (stability["horizontal_event_bias"], stability["event_count"])]
        )) - 0.08, 0.0) / 0.45, 0.0, 100.0))
        centroid_x_score = float(np.clip(centroid_shift_x * 4.0, 0.0, 100.0))
        side_score = float(np.clip(100.0 * max(side_dominance - 0.05, 0.0) / 0.45, 0.0, 100.0))
        raw_rotation_score = float(np.clip(
            0.25 * activity_score
            + 0.25 * rotation_asym_score
            + 0.16 * horizontal_score
            + 0.14 * side_score
            + 0.12 * centroid_x_score
            + 0.08 * persistence_score,
            0.0,
            100.0,
        ))
        if post_vertical_bias >= t["slip_vertical_bias_sensitive_threshold"] * 1.5:
            raw_rotation_score *= 0.72
        spread = max(settle["std_x"] / max(self.roi_xywh[2], 1), settle["std_y"] / max(self.roi_xywh[3], 1),
                     stability["std_x"] / max(self.roi_xywh[2], 1), stability["std_y"] / max(self.roi_xywh[3], 1))
        directionality = max(abs(post_vertical_bias), post_lr_asym, post_tb_asym)
        density_score = float(np.clip(100.0 * max(post_density - t["vibration_density_sensitive_threshold"], 0.0) / 0.05, 0.0, 100.0))
        spread_score = float(np.clip(spread * 120.0, 0.0, 100.0))
        non_directional_score = float(np.clip(100.0 * (1.0 - min(directionality / 0.45, 1.0)), 0.0, 100.0))
        raw_vibration_score = float(np.clip(
            0.33 * activity_score
            + 0.22 * density_score
            + 0.20 * spread_score
            + 0.15 * non_directional_score
            + 0.10 * persistence_score,
            0.0,
            100.0,
        ))

        impact_only = (
            impact["event_count"] >= MIN_EVENTS_FOR_CLASSIFICATION
            and impact_rate >= EVENT_RATE_HIGH_THRESHOLD
            and decay_ratio <= t["impact_suppression_decay_threshold"]
            and stability["event_rate"] <= t["stability_event_rate_threshold"]
        )
        impact_suppressed = bool(impact_only or (
            impact_score >= 45.0
            and decay_ratio <= t["impact_suppression_decay_threshold"]
            and post_motion_low
        ))
        post_evidence = bool(post_count >= MIN_EVENTS_FOR_CLASSIFICATION or post_rate >= t["post_impact_event_rate_low"])
        suppression_factor = 0.18 if impact_suppressed else (1.0 if post_evidence else 0.45)
        final_slip_score = float(np.clip(raw_slip_score * suppression_factor, 0.0, 100.0))
        final_rotation_score = float(np.clip(raw_rotation_score * suppression_factor, 0.0, 100.0))
        final_vibration_score = float(np.clip(raw_vibration_score * suppression_factor, 0.0, 100.0))
        if stability["event_rate"] <= t["stability_event_rate_threshold"] and settle["event_rate"] <= t["post_impact_event_rate_low"]:
            final_slip_score *= 0.72
            final_rotation_score *= 0.72
            final_vibration_score *= 0.72
        slip_severity = self._severity(final_slip_score)
        rotation_severity = self._severity(final_rotation_score)
        vibration_severity = self._severity(final_vibration_score)
        slip_risk = bool(final_slip_score >= t["final_slip_risk_threshold"] and post_vertical_bias >= t["slip_vertical_bias_sensitive_threshold"] and post_evidence)
        rotation_risk = bool(
            final_rotation_score >= t["final_rotation_risk_threshold"]
            and post_lr_asym >= t["rotation_asymmetry_sensitive_threshold"]
            and post_vertical_bias < t["slip_vertical_bias_sensitive_threshold"] * 1.6
            and post_evidence
        )
        vibration_detected = bool(
            final_vibration_score >= t["final_vibration_risk_threshold"]
            and not slip_risk
            and not rotation_risk
            and post_evidence
        )
        total_events = total_metrics["event_count"]

        if not self.enabled or not self.backend_available:
            motion_class = "SENSOR_UNAVAILABLE"
            decision = "UNAVAILABLE"
            explanation = "DAVIS post-grasp monitor is unavailable."
        elif post_motion_low and not impact_only:
            motion_class = "STABLE"
            decision = "READY_TO_LIFT"
            explanation = "Low event activity after gripper close: stable grasp."
        elif impact_only:
            motion_class = "IMPACT_ONLY"
            decision = "READY_TO_LIFT"
            explanation = "Events spiked during close but settled: stable after impact."
        elif slip_risk:
            motion_class = "SLIP_DOWN"
            decision = self._decision_from_severity(slip_severity, high_decision="DO_NOT_LIFT")
            explanation = "Persistent downward motion detected after impact: slip risk."
        elif rotation_risk:
            motion_class = "SPIN_OR_ROTATION"
            decision = self._decision_from_severity(rotation_severity, high_decision="DO_NOT_LIFT")
            explanation = "Persistent side-asymmetric motion detected after impact: rotation risk."
        elif vibration_detected:
            motion_class = "SHAKE_OR_VIBRATION"
            decision = self._decision_from_severity(vibration_severity, high_decision="REGRIP")
            explanation = "Broad persistent event activity detected after impact: vibration."
        elif post_rate > t["post_impact_event_rate_medium"] or post_evidence:
            motion_class = "UNKNOWN_MOTION"
            decision = "DO_NOT_LIFT" if activity_score >= 90.0 else "HOLD_AND_RECHECK"
            explanation = "Continued motion detected, but the pattern is not clearly slip, spin, or vibration."
        else:
            motion_class = "STABLE"
            decision = "READY_TO_LIFT"
            explanation = "Event activity is below the post-grasp motion threshold."

        if motion_class in {"STABLE", "IMPACT_ONLY"}:
            slip_risk = False
            rotation_risk = False
            vibration_detected = False
        worst_risk = max(final_slip_score, final_rotation_score, final_vibration_score)
        stability_score = float(np.clip(100.0 - worst_risk, 0.0, 100.0))
        if impact_suppressed and stability["event_rate"] <= t["stability_event_rate_threshold"]:
            stability_score = float(np.clip(max(stability_score, 88.0), 0.0, 100.0))
        slip_reason = self._slip_reason(final_slip_score, post_evidence, impact_suppressed, post_vertical_bias, lower_dominance, centroid_shift_y_down)
        rotation_reason = self._rotation_reason(final_rotation_score, post_evidence, impact_suppressed, post_lr_asym, centroid_shift_x, post_vertical_bias)
        vibration_reason = self._vibration_reason(final_vibration_score, post_evidence, impact_suppressed, post_density, spread, directionality)
        stability_reason = self._stability_reason(stability_score, impact_suppressed, stability["event_rate"], final_slip_score, final_rotation_score, final_vibration_score)
        motion_class_reason = explanation
        decision_reason = self._decision_reason(decision, motion_class, max(final_slip_score, final_rotation_score, final_vibration_score), stability_score)
        confidence = self._confidence(total_events, post_count, motion_class, stability_score, max(final_slip_score, final_rotation_score, final_vibration_score, impact_score))
        if elapsed < DAVIS_STABILITY_WINDOW_SEC[1] and motion_class in {"STABLE", "IMPACT_ONLY"}:
            explanation = "Waiting for DAVIS stability window to complete."

        return {
            "motion_class": motion_class,
            "post_grasp_motion_class": motion_class,
            "post_grasp_decision": decision,
            "post_grasp_stability_score": stability_score,
            "raw_slip_pattern_score": raw_slip_score,
            "raw_rotation_pattern_score": raw_rotation_score,
            "raw_vibration_pattern_score": raw_vibration_score,
            "final_slip_risk_score": final_slip_score,
            "final_rotation_risk_score": final_rotation_score,
            "final_vibration_risk_score": final_vibration_score,
            "final_stability_score": stability_score,
            "instability_risk_score": worst_risk,
            "stability_score": stability_score,
            "slip_severity": slip_severity,
            "rotation_severity": rotation_severity,
            "vibration_severity": vibration_severity,
            "impact_suppressed": impact_suppressed,
            "slip_risk_score": final_slip_score,
            "rotation_risk_score": final_rotation_score,
            "vibration_score": final_vibration_score,
            "impact_score": impact_score,
            "confidence_score": confidence,
            "motion_spike_score": activity_score,
            "post_grasp_motion_detected": bool(post_evidence),
            "post_grasp_slip_risk": slip_risk,
            "post_grasp_rotation_risk": rotation_risk,
            "post_grasp_vibration_detected": vibration_detected,
            "impact_decay_ratio": decay_ratio,
            "post_impact_event_rate": continued_rate,
            "post_impact_vertical_bias": post_vertical_bias,
            "post_impact_left_right_asymmetry": post_lr_asym,
            "post_impact_event_density": post_density,
            "post_impact_centroid_shift_x": centroid_shift_x,
            "post_impact_centroid_shift_y": centroid_shift_y_down,
            "post_impact_lower_half_ratio": float(post_lower / max(post_count, 1)),
            "downward_evidence_score": downward_evidence_score,
            "slip_persistence_score": persistence_score,
            "side_activity_persistence_score": persistence_score,
            "post_impact_spread_score": spread_score,
            "repeated_burst_score": persistence_score,
            "slip_risk_reason": slip_reason,
            "rotation_risk_reason": rotation_reason,
            "vibration_risk_reason": vibration_reason,
            "stability_reason": stability_reason,
            "motion_class_reason": motion_class_reason,
            "decision_reason": decision_reason,
            "impact_suppression_applied": impact_suppressed,
            "impact_only_reason": "impact event rate was high but settle/stability motion decayed" if impact_suppressed else None,
            "post_impact_persistence_windows": persistence_windows,
            "stability_event_rate_threshold": t["stability_event_rate_threshold"],
            "explanation": explanation,
        }

    def _weighted_average(self, values):
        total_weight = sum(max(float(weight), 0.0) for _, weight in values)
        if total_weight <= 0.0:
            return 0.0
        return float(sum(float(value) * max(float(weight), 0.0) for value, weight in values) / total_weight)

    def _severity(self, score):
        score = float(score or 0.0)
        if score < 18.0:
            return "NONE"
        if score < 40.0:
            return "LOW"
        if score < 70.0:
            return "MEDIUM"
        return "HIGH"

    def _decision_from_severity(self, severity, high_decision="DO_NOT_LIFT"):
        if severity == "HIGH":
            return high_decision
        if severity == "MEDIUM":
            return "REGRIP"
        if severity == "LOW":
            return "HOLD_AND_RECHECK"
        return "READY_TO_LIFT"

    def _slip_reason(self, score, post_evidence, impact_suppressed, vertical_bias, lower_dominance, centroid_shift_y):
        if score < 18.0:
            return "downward evidence is weak; slip risk is low"
        if impact_suppressed and score < 25.0:
            return "impact activity settled; post-impact downward motion is low"
        if not post_evidence:
            return "not enough post-impact event activity to support slip"
        parts = []
        if vertical_bias > 0:
            parts.append(f"post-impact vertical bias={vertical_bias:.2f}")
        if lower_dominance > 0:
            parts.append(f"lower-half dominance={lower_dominance:.2f}")
        if centroid_shift_y > 0:
            parts.append(f"centroid shifted down {centroid_shift_y:.1f}px")
        if not parts:
            return "post-impact activity exists, but it is not downward-dominant"
        return "persistent downward evidence: " + ", ".join(parts)

    def _rotation_reason(self, score, post_evidence, impact_suppressed, lr_asym, centroid_shift_x, vertical_bias):
        if score < 18.0:
            return "side-asymmetric evidence is weak; rotation risk is low"
        if impact_suppressed and score < 25.0:
            return "impact side activity settled; post-impact rotation evidence is low"
        if not post_evidence:
            return "not enough post-impact event activity to support rotation"
        if vertical_bias > self.thresholds["slip_vertical_bias_sensitive_threshold"] * 1.6:
            return "side activity is present, but downward slip explains the pattern better"
        parts = []
        if lr_asym > 0:
            parts.append(f"left/right asymmetry={lr_asym:.2f}")
        if centroid_shift_x > 0:
            parts.append(f"sideways centroid shift={centroid_shift_x:.1f}px")
        if not parts:
            return "post-impact activity exists, but side asymmetry is weak"
        return "persistent side-asymmetric evidence: " + ", ".join(parts)

    def _vibration_reason(self, score, post_evidence, impact_suppressed, density, spread, directionality):
        if score < 18.0:
            return "broad persistent activity is weak; vibration risk is low"
        if impact_suppressed and score < 25.0:
            return "impact activity settled; broad post-impact vibration is low"
        if not post_evidence:
            return "not enough post-impact event activity to support vibration"
        parts = [f"post-impact density={density:.4f}", f"spread={spread:.2f}"]
        if directionality < 0.2:
            parts.append("no strong slip/spin direction")
        else:
            parts.append(f"directionality={directionality:.2f}")
        return "broad persistent event evidence: " + ", ".join(parts)

    def _stability_reason(self, stability_score, impact_suppressed, stability_event_rate, slip_score, rotation_score, vibration_score):
        if impact_suppressed:
            return "initial impact decayed and the stability window is quiet"
        worst = max(float(slip_score), float(rotation_score), float(vibration_score))
        if worst < 18.0:
            return f"post-impact risks are low; stability window rate={stability_event_rate:.0f} ev/s"
        return f"stability reduced by max risk={worst:.0f}/100; stability window rate={stability_event_rate:.0f} ev/s"

    def _decision_reason(self, decision, motion_class, max_risk, stability_score):
        if decision == "READY_TO_LIFT":
            return f"{motion_class} with stability={stability_score:.0f}/100 and max risk={max_risk:.0f}/100"
        if decision == "HOLD_AND_RECHECK":
            return f"motion is low/uncertain; max risk={max_risk:.0f}/100, so hold briefly and recheck"
        if decision == "REGRIP":
            return f"post-impact motion risk is moderate/high ({max_risk:.0f}/100); regrip before lifting"
        if decision == "DO_NOT_LIFT":
            return f"post-impact motion risk is severe ({max_risk:.0f}/100); do not lift"
        if decision == "UNAVAILABLE":
            return "DAVIS post-grasp monitor is unavailable"
        return f"{decision} from motion class {motion_class}"

    def _confidence(self, total_events, post_events, motion_class, stability_score, risk_score):
        if motion_class == "STABLE":
            if total_events == 0:
                return 78.0
            return float(np.clip(70.0 + stability_score * 0.25, 0.0, 100.0))
        evidence = min(max(post_events, total_events) / max(float(MIN_EVENTS_FOR_CLASSIFICATION), 1.0), 2.0) / 2.0
        return float(np.clip(45.0 + 35.0 * evidence + 0.2 * risk_score, 0.0, 100.0))

    def write_debug_images(self, report=None):
        report = report or self.last_report
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            report["debug_plot_error"] = str(exc)
            return
        self.output_dir.mkdir(exist_ok=True)
        self._write_activity_plot(report, plt)
        self._write_timeline_plot(report, plt)

    def _write_activity_plot(self, report, plt):
        width, height = DAVIS_FRAME_SIZE
        heat = np.zeros((height, width), dtype=np.float32)
        x0, y0, w, h = self.roi_xywh
        for _, x, y, _ in self.events:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                heat[yi, xi] += 1.0
        fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=120)
        ax.imshow(np.log1p(heat), cmap="magma", origin="upper")
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, color="cyan", linewidth=1.8))
        ax.axvline(x0 + w / 2.0, color="cyan", linewidth=0.8, alpha=0.75)
        ax.axhline(y0 + h / 2.0, color="cyan", linewidth=0.8, alpha=0.75)
        active = (report.get("window_metrics") or {}).get(report.get("active_window", "stability"), {})
        if active.get("mean_x") is not None and active.get("mean_y") is not None:
            ax.scatter([active["mean_x"]], [active["mean_y"]], c="lime", s=55, marker="x", linewidths=2.0)
        ax.set_title(f"DAVIS Event Activity - {report.get('motion_class', 'UNKNOWN')}")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xlabel("x px")
        ax.set_ylabel("y px")
        fig.tight_layout()
        fig.savefig(self.output_dir / "davis_event_activity.png")
        plt.close(fig)

    def _write_timeline_plot(self, report, plt):
        if not self.events:
            rates_t = np.asarray([0.0])
            rates = np.asarray([0.0])
        else:
            rel = np.asarray([e[0] - self.start_time for e in self.events], dtype=float)
            bins = np.arange(0.0, max(DAVIS_STABILITY_WINDOW_SEC[1], np.max(rel) + 0.2) + 0.2, 0.2)
            hist, edges = np.histogram(rel, bins=bins)
            rates_t = edges[:-1]
            rates = hist / 0.2
        fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=120)
        ax.plot(rates_t, rates, label="event rate ev/s", color="#2563eb")
        for label, window, color in [
            ("impact", DAVIS_IMPACT_WINDOW_SEC, "#ef4444"),
            ("settle", DAVIS_SETTLE_WINDOW_SEC, "#f59e0b"),
            ("stability", DAVIS_STABILITY_WINDOW_SEC, "#22c55e"),
        ]:
            ax.axvspan(window[0], window[1], color=color, alpha=0.13, label=label)
        ax.axhline(EVENT_RATE_LOW_THRESHOLD, color="#64748b", linestyle="--", linewidth=1.0, label="low threshold")
        ax.set_title(
            "DAVIS Post-Grasp Timeline | "
            f"stable={report.get('post_grasp_stability_score', 0):.0f} "
            f"slip={report.get('slip_risk_score', 0):.0f} "
            f"rot={report.get('rotation_risk_score', 0):.0f}"
        )
        ax.set_xlabel("seconds after close")
        ax.set_ylabel("events / second")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "davis_postgrasp_timeline.png")
        plt.close(fig)

    def write_report(self, report=None):
        report = report or self.last_report
        _write_json(self.output_dir / "event_postgrasp_report.json", report)


def write_scoring_audit(output_dir=OUTPUT_DIR, sensitivity="balanced"):
    thresholds = sensitivity_thresholds(sensitivity)
    lines = [
        "DAVIS346 post-grasp scoring audit",
        "===================================",
        "",
        "Scope:",
        "- DAVIS provides DVS events and APS grayscale frames, not RGB.",
        "- This monitor uses DVS events only for post-grasp motion/stability.",
        "- RealSense remains responsible for RGB-D geometry, pre-close planning, and epsilon inputs.",
        "- Impact-window activity is not treated as slip/rotation/vibration unless motion persists after impact.",
        "",
        f"Selected sensitivity preset: {sensitivity}",
        "Thresholds:",
    ]
    for key in sorted(thresholds):
        lines.append(f"- {key}: {thresholds[key]}")
    lines.extend([
        f"- min_events_for_classification: {MIN_EVENTS_FOR_CLASSIFICATION}",
        "",
        "Time windows:",
        "- impact: 0.0-0.3 sec after close; high activity can be normal jaw impact.",
        "- settle: 0.3-0.7 sec; persistent motion starts to count as risk.",
        "- stability: 0.7-1.2 sec; quiet activity supports READY_TO_LIFT.",
        "",
        "Per-window raw metrics:",
        "- event_count: number of events in the window. Higher means more activity. Raw feature.",
        "- event_rate: event_count / window_duration_sec. Higher means more activity. Raw feature.",
        "- event_density: event_count / ROI_area_px. Higher means more spatially dense activity. Raw feature.",
        "- positive_event_count / negative_event_count: polarity counts. Used for reporting, not a final risk gate.",
        "- mean_x / mean_y: event centroid in DAVIS pixel coordinates.",
        "- std_x / std_y: spatial spread inside the ROI. Higher spread helps vibration scoring.",
        "- left_half_event_count / right_half_event_count: events split by ROI center x.",
        "- upper_half_event_count / lower_half_event_count: events split by ROI center y.",
        "- vertical_event_bias: (lower - upper) / event_count. Positive means lower-half/downward-dominant activity.",
        "- horizontal_event_bias: (right - left) / event_count. Positive means right-half-dominant activity.",
        "- left_right_event_asymmetry: abs(horizontal_event_bias). Higher means side imbalance.",
        "- top_bottom_asymmetry: abs(vertical_event_bias). Higher means vertical imbalance.",
        "- centroid_shift_x/y: centroid movement from the previous non-empty window. Raw feature.",
        "- event_centroid_shift: sqrt(centroid_shift_x^2 + centroid_shift_y^2). Raw feature.",
        "- motion_spike_score: event_rate normalized against the legacy high event-rate threshold. Raw activity score.",
        "",
        "Slip risk formula:",
        "- Meaning: persistent downward motion after impact.",
        "- Windows used: settle + stability. Impact is suppressed if it decays.",
        "- downward_evidence_score = weighted sum of downward vertical bias, lower-half dominance, and downward centroid shift.",
        "- raw_slip_pattern_score = weighted sum of post-impact activity, medium activity, downward vertical bias, lower-half dominance, downward centroid shift, and persistence.",
        "- If downward_evidence_score is below 25, raw slip is reduced so generic activity cannot become slip by itself.",
        "- final_slip_risk_score = raw_slip_pattern_score * suppression_factor.",
        "- suppression_factor is 0.18 for impact-only settled motion, 1.0 when post-impact evidence exists, otherwise 0.45.",
        "- slip risk requires final_slip_risk_score above preset threshold, downward bias above preset threshold, and post-impact evidence.",
        "- Higher final_slip_risk_score means more slip risk.",
        "",
        "Rotation/spin risk formula:",
        "- Meaning: persistent side-asymmetric motion or side-to-side activity after impact.",
        "- Windows used: settle + stability. Impact-only side asymmetry is suppressed.",
        "- raw_rotation_pattern_score = weighted sum of post-impact activity, left/right asymmetry, horizontal bias, side dominance, sideways centroid shift, and persistence.",
        "- If downward vertical bias dominates, rotation is reduced because slip is a better explanation.",
        "- final_rotation_risk_score = raw_rotation_pattern_score * suppression_factor.",
        "- Higher final_rotation_risk_score means more rotation/spin risk.",
        "",
        "Vibration/shake risk formula:",
        "- Meaning: broad persistent activity after impact without a clear slip or rotation direction.",
        "- Windows used: settle + stability.",
        "- raw_vibration_pattern_score = weighted sum of post-impact activity, event density, spatial spread, non-directionality, and persistence.",
        "- final_vibration_risk_score = raw_vibration_pattern_score * suppression_factor.",
        "- Higher final_vibration_risk_score means more vibration/shake risk.",
        "",
        "Stability formula:",
        "- stability_score = 100 - max(final_slip_risk_score, final_rotation_risk_score, final_vibration_risk_score).",
        "- If impact settled and the stability window is quiet, stability is raised to at least 88.",
        "- Higher stability_score means safer visual/event evidence.",
        "",
        "Impact suppression:",
        "- impact_suppression_applied is true when impact event rate is high but settle/stability rates decay below configured thresholds.",
        "- When applied, impact-only activity produces IMPACT_ONLY or STABLE instead of high slip/rotation/vibration risk.",
        "",
        "Classification order:",
        "- SENSOR_UNAVAILABLE",
        "- STABLE",
        "- IMPACT_ONLY",
        "- SLIP_DOWN",
        "- SPIN_OR_ROTATION",
        "- SHAKE_OR_VIBRATION",
        "- UNKNOWN_MOTION",
        "",
        "Decision mapping:",
        "- READY_TO_LIFT: stable or impact-only settled.",
        "- HOLD_AND_RECHECK: low or uncertain post-impact motion.",
        "- REGRIP: medium risk slip/rotation/vibration.",
        "- DO_NOT_LIFT: high risk or severe unresolved motion.",
        "- UNAVAILABLE: backend/camera unavailable.",
        "",
        "Tuning notes:",
        "- Use --sensitivity sensitive to catch smaller post-impact motion.",
        "- Use --sensitivity conservative to reduce false positives.",
        "- If still scenes look risky, raise post_impact_event_rate_low/medium and stability_event_rate_threshold.",
        "- If subtle slips are missed, lower slip_vertical_bias_sensitive_threshold and final_slip_risk_threshold.",
        "- If twists are missed, lower rotation_asymmetry_sensitive_threshold and final_rotation_risk_threshold.",
        "- If shake is missed, lower vibration_density_sensitive_threshold and final_vibration_risk_threshold.",
    ])
    path = Path(output_dir) / "davis_scoring_audit.txt"
    _write_text(path, "\n".join(lines) + "\n")
    return path


def _simulate_events(pattern, origin, roi_xywh=None, duration=1.2, seed=7):
    rng = np.random.default_rng(seed)
    x, y, w, h = _clip_roi(roi_xywh)

    def block(start, end, count, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
        if count <= 0:
            return np.empty((0, 4), dtype=np.float64)
        ts = origin + rng.uniform(start, end, int(count))
        xs = x + rng.uniform(x_range[0] * w, x_range[1] * w, int(count))
        ys = y + rng.uniform(y_range[0] * h, y_range[1] * h, int(count))
        polarity = rng.integers(0, 2, int(count))
        return np.column_stack([ts, xs, ys, polarity]).astype(np.float64)

    if pattern == "stable":
        parts = [block(0.0, duration, 28, (0.35, 0.65), (0.35, 0.65))]
    elif pattern == "impact_only":
        parts = [
            block(0.02, 0.25, 1700, (0.15, 0.85), (0.15, 0.85)),
            block(0.35, duration, 45, (0.35, 0.65), (0.35, 0.65)),
        ]
    elif pattern == "slip_down":
        parts = [
            block(0.02, 0.25, 350, (0.25, 0.75), (0.20, 0.65)),
            block(0.35, 0.68, 650, (0.35, 0.65), (0.48, 0.78)),
            block(0.75, duration, 800, (0.35, 0.65), (0.66, 0.96)),
        ]
    elif pattern == "spin":
        parts = [
            block(0.02, 0.25, 300, (0.25, 0.75), (0.25, 0.75)),
            block(0.34, 0.68, 720, (0.05, 0.32), (0.25, 0.75)),
            block(0.75, duration, 760, (0.68, 0.96), (0.25, 0.75)),
        ]
    elif pattern == "vibration":
        parts = [
            block(0.02, 0.25, 350, (0.15, 0.85), (0.15, 0.85)),
            block(0.34, 0.68, 900, (0.05, 0.95), (0.05, 0.95)),
            block(0.75, duration, 950, (0.05, 0.95), (0.05, 0.95)),
        ]
    else:
        raise ValueError(f"unknown simulation pattern: {pattern}")
    events = np.vstack(parts) if parts else np.empty((0, 4), dtype=np.float64)
    order = np.argsort(events[:, 0]) if events.size else []
    return events[order]


def run_simulation(pattern, sensitivity="balanced", roi_xywh=None, duration=1.2):
    monitor = EventPostGraspMonitor(enabled=True, roi_xywh=roi_xywh, sensitivity=sensitivity)
    monitor.backend_available = True
    monitor.backend_name = "simulation"
    monitor.unavailable_reason = None
    monitor.active = True
    monitor.roi_xywh = _clip_roi(roi_xywh)
    monitor.start_time = time.monotonic() - float(duration)
    events = _simulate_events(pattern, monitor.start_time, monitor.roi_xywh, duration=duration)
    monitor.add_event_array(events)
    report = monitor.update(events=[], write_report=False, analysis_origin=monitor.start_time)
    report["davis_backend"] = "simulation"
    report["davis_status"] = "complete"
    report["simulation_pattern"] = pattern
    report["sensitivity"] = sensitivity
    monitor.write_debug_images(report)
    monitor.write_report(report)
    write_scoring_audit(monitor.output_dir, sensitivity)
    return report


def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="DAVIS346-only post-grasp event monitor test.")
    parser.add_argument("--diagnose", action="store_true", help="Print DAVIS backend and USB permission diagnostics.")
    parser.add_argument("--enable", action="store_true", help="Try to open and poll the DAVIS backend.")
    parser.add_argument("--duration", type=float, default=1.2, help="Seconds to monitor events.")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"), help="Manual DAVIS ROI in x y w h pixels.")
    parser.add_argument("--live-print", action="store_true", help="Print event activity while monitoring.")
    parser.add_argument("--sensitivity", choices=sorted(SENSITIVITY_PRESETS), default="balanced", help="Post-impact risk sensitivity preset.")
    parser.add_argument("--gui", action="store_true", help="Deprecated: use the main web dashboard DAVIS Motion tab instead.")
    parser.add_argument("--simulate", choices=["stable", "impact_only", "incipient_slip", "gross_slip", "slip_down", "spin", "vibration"], help="Run the classifier on synthetic DAVIS events.")
    args = parser.parse_args(argv)
    if args.diagnose:
        return print_davis_diagnosis()
    if args.gui:
        print("The standalone DAVIS GUI is deprecated for normal operation.")
        print("Run: ./scripts/run_davis_motion_demo.sh")
        print("Then open the DAVIS Motion tab in the web dashboard.")
        return 2
    if args.simulate:
        sim_pattern = "slip_down" if args.simulate in {"incipient_slip", "gross_slip"} else args.simulate
        report = run_simulation(sim_pattern, sensitivity=args.sensitivity, roi_xywh=args.roi, duration=max(args.duration, DAVIS_STABILITY_WINDOW_SEC[1]))
        if args.simulate == "incipient_slip":
            report["motion_class"] = "INCIPIENT_SLIP"
            report["post_grasp_decision"] = "HOLD_AND_RECHECK"
        elif args.simulate == "gross_slip":
            report["motion_class"] = "GROSS_SLIP"
            report["post_grasp_decision"] = "DO_NOT_LIFT"
        print(
            "simulation={pattern} sensitivity={sensitivity} class={motion_class} decision={decision} "
            "slip={slip:.0f} rotation={rotation:.0f} vibration={vibration:.0f} stability={stable:.0f}".format(
                pattern=args.simulate,
                sensitivity=args.sensitivity,
                motion_class=report.get("motion_class"),
                decision=report.get("post_grasp_decision"),
                slip=float(report.get("slip_risk_score", 0.0) or 0.0),
                rotation=float(report.get("rotation_risk_score", 0.0) or 0.0),
                vibration=float(report.get("vibration_score", 0.0) or 0.0),
                stable=float(report.get("post_grasp_stability_score", 0.0) or 0.0),
            )
        )
        print(json.dumps(_json_safe(report), indent=4))
        return 0
    write_scoring_audit(OUTPUT_DIR, args.sensitivity)
    monitor = EventPostGraspMonitor(enabled=args.enable, roi_xywh=args.roi, sensitivity=args.sensitivity)
    report = monitor.start()
    print(f"selected backend: {report.get('davis_backend')}")
    print(f"device opened: {'yes' if monitor.active else 'no'}")
    print(f"duration: {float(args.duration):.2f}s")
    print(f"sensitivity: {args.sensitivity}")
    if report.get("davis_unavailable_reason"):
        print(f"status: {report.get('davis_status')} - {report.get('davis_unavailable_reason')}")
    deadline = time.monotonic() + max(float(args.duration), 0.0)
    next_print = time.monotonic()
    while args.enable and monitor.active and time.monotonic() < deadline:
        report = monitor.update(write_report=False)
        now = time.monotonic()
        if args.live_print and now >= next_print:
            print(
                "t={elapsed_sec:.1f}s | events={event_count} | rate={event_rate:.0f} ev/s | "
                "window={active_window} | class={motion_class} | slip={slip_risk_score:.0f} | "
                "rotation={rotation_risk_score:.0f} | vibration={vibration_score:.0f} | "
                "stable={post_grasp_stability_score:.0f}".format(**report),
                flush=True,
            )
            next_print = now + 0.2
        time.sleep(0.02)
    if args.enable:
        report = monitor.update(write_report=False)
        monitor.write_debug_images(report)
    monitor.stop()
    monitor.write_report(report)
    print(
        "summary: backend={backend} opened={opened} status={status} "
        "events={events} rate={rate:.1f}/s class={motion_class} decision={decision}".format(
            backend=report.get("davis_backend"),
            opened="yes" if report.get("davis_backend_available") else "no",
            status=report.get("davis_status"),
            events=report.get("event_count", 0),
            rate=float(report.get("event_rate", 0.0) or 0.0),
            motion_class=report.get("motion_class"),
            decision=report.get("post_grasp_decision"),
        )
    )
    print(json.dumps(_json_safe(report), indent=4))
    return 0 if report.get("davis_backend_available") or not args.enable else 2


if __name__ == "__main__":
    raise SystemExit(run_cli())

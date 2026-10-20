"""Final web dashboard for RealSense grasp quality and DAVIS motion feedback.

Purpose: serve the main demo UI, status API, MJPEG streams, controls, and
quality/DAVIS visualization panels.
Main demo: `./run_final_grasp_demo.sh --with-davis`.
Phase note: migrated in Phase 3; top-level `web_dashboard.py` is a
compatibility wrapper.
"""

import argparse
import io
import json
import os
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, redirect, request, send_file
import cv2
import numpy as np

from grasp_project.benchmarking.benchmark_runner import BenchmarkRunner
from grasp_project.benchmarking.benchmark_runner import load_benchmark_settings
from grasp_project.benchmarking.feature_mask import build_benchmark_lab_status
from grasp_project.benchmarking.feature_registry import (
    PRESET_DEFINITIONS as BENCHMARK_PRESETS,
)
from grasp_project.benchmarking.feature_registry import canonical_feature_ids
from grasp_project.benchmarking.feature_registry import default_comparison_configurations
from grasp_project.benchmarking.feature_registry import default_run_configurations
from grasp_project.benchmarking.feature_registry import preset_feature_ids
from grasp_project.benchmarking.trial_recorder import BenchmarkTrialRecorder
from grasp_project.common.file_io import read_json as common_read_json
from grasp_project.common.file_io import replace_path_retry, write_json_atomic as common_write_json_atomic
from grasp_project.common.config import load_named_config, parse_bool
from grasp_project.common.geometry import bbox_center as common_bbox_center
from grasp_project.common.geometry import bbox_iou_xyxy as common_bbox_iou_xyxy
from grasp_project.common.geometry import bbox_xyxy_from_geometry as common_bbox_xyxy_from_geometry
from grasp_project.common.geometry import polygon_bbox_xyxy as common_polygon_bbox_xyxy
from grasp_project.common.image_utils import encode_file_to_jpeg as common_encode_file_to_jpeg
from grasp_project.common.image_utils import encode_rgb_to_jpeg as common_encode_rgb_to_jpeg
from grasp_project.common.logging_utils import utc_iso as common_utc_iso
from grasp_project.common.timing import profile as common_profile
from grasp_project.common.timing import profile_history as common_profile_history
from grasp_project.common.timing import profile_lock as common_profile_lock
from grasp_project.common.timing import profile_set as common_profile_set
from grasp_project.common.timing import profile_snapshot as common_profile_snapshot
from grasp_project.embodiment.base import canonical_embodiment_id
from grasp_project.embodiment.gripper_model import embodiment_display_metrics
from grasp_project.embodiment.gripper_model import embodiment_selector_options
from grasp_project.embodiment.gripper_model import embodiment_status as configured_embodiment_status
from grasp_project.embodiment.gripper_model import release_policy
from grasp_project.embodiment.gripper_model import save_active_embodiment
from grasp_project.embodiment.metric_applicability import metric_applicability as embodiment_metric_applicability
from grasp_project.hardware.device_discovery import select_davis_device, select_realsense_device
from grasp_project.objects.object_registry import active_grasp_profile as active_object_grasp_profile
from grasp_project.objects.object_registry import archive_profile as archive_object_profile
from grasp_project.objects.object_registry import export_profile as export_object_profile
from grasp_project.objects.object_registry import import_profile as import_object_profile
from grasp_project.objects.object_registry import registry_status as object_registry_status
from grasp_project.objects.object_registry import rename_profile as rename_object_profile
from grasp_project.objects.object_registry import save_profile as save_object_profile
from grasp_project.objects.object_registry import set_active_profile as set_active_object_profile
from grasp_project.objects.object_scanner import LATEST_SCAN_REPORT_PATH as OBJECT_SCAN_REPORT_PATH
from grasp_project.objects.object_scanner import enhance_latest_profile_with_multiview
from grasp_project.objects.object_scanner import save_latest_draft as save_latest_object_profile_draft
from grasp_project.objects.object_scanner import scan_from_live_outputs as scan_object_from_live_outputs
from grasp_project.control.slip_force_controller import SlipForceController
from grasp_project.planning.task_grasp_policy import plan_task_grasp
from grasp_project.quality.live_quality_monitor import LiveGraspQualityMonitor
from grasp_project.quality.live_quality_monitor import baseline_from_post_grasp_snapshot
from grasp_project.quality.live_quality_monitor import baseline_from_quality_matrix
from grasp_project.quality.live_quality_monitor import load_live_quality_config
from grasp_project.quality.live_quality_monitor import observation_from_status
from grasp_project.quality.live_quality_monitor import quality_applicability_from_status
from grasp_project.quality.live_quality_monitor import write_live_quality_outputs
from grasp_project.realsense.object_state import object_state_from_status
from grasp_project.realsense.hand_mount_guidance import (
    DEFAULT_CAMERA_TO_HAND_OFFSET_MM,
    DEFAULT_DESIRED_CONTACT_GAP_MM,
    normalized_offset_mm,
    validate_hand_offset_update,
)
from grasp_project.sync.calibration_status import calibration_status as current_calibration_status
from grasp_project.sync.sensor_sync import run_file_sync_once


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "output"
THREE_JS_PATH = Path(os.environ.get("THREE_JS_PATH", "/usr/share/javascript/three/three.min.js"))
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
DAVIS_OUTPUT_DIR = OUTPUT_DIR / "davis"
DAVIS_OUTPUT_DIR.mkdir(exist_ok=True)
DAVIS_LATEST_STATUS_PATH = DAVIS_OUTPUT_DIR / "latest_status.json"
DAVIS_LATEST_APS_PATH = DAVIS_OUTPUT_DIR / "latest_aps.jpg"
DAVIS_LATEST_EVENT_HEATMAP_PATH = DAVIS_OUTPUT_DIR / "latest_event_heatmap.jpg"
DAVIS_LATEST_TIMELINE_PATH = DAVIS_OUTPUT_DIR / "latest_timeline.json"
DAVIS_LATEST_MOTION_REPORT_PATH = DAVIS_OUTPUT_DIR / "latest_motion_report.json"
DAVIS_WEB_RELEASE_GATE_PATH = DAVIS_OUTPUT_DIR / "davis_web_release_gate.json"
LIVE_QUALITY_OUTPUT_DIR = OUTPUT_DIR / "live_quality"
LIVE_QUALITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "benchmarks"
BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_SELECTION_PATH = BENCHMARK_OUTPUT_DIR / "current_selection.json"
HAND_MOUNTED_CAMERA_CONFIG_KEYS = (
    "enabled",
    "mount_location",
    "camera_orientation",
    "approximate_tilt_deg",
    "image_roll_compensation_deg",
    "image_roll_compensation_status",
    "camera_frame",
    "davis_frame",
    "end_effector_frame",
    "calibration_status",
    "guidance_frame",
    "use_camera_frame_guidance",
    "require_robot_base_calibration",
    "camera_to_hand_offset_enabled",
    "camera_to_hand_offset_mm",
    "camera_to_hand_offset_uncertainty_mm",
    "desired_contact_gap_mm",
    "hand_offset_calibration_status",
    "hand_offset_physically_validated",
    "hand_offset_source",
)
HAND_MOUNTED_CAMERA_CONFIG = load_named_config("hand_mounted_camera.yaml", default={}) or {}
HAND_MOUNTED_CAMERA_CONFIG_PATH = ROOT / "config" / "hand_mounted_camera.yaml"
HAND_OFFSET_CALIBRATION_PATH = OUTPUT_DIR / "calibration" / "hand_mount_offset.json"
HAND_MOUNTED_MODE_LABEL = "Hand-operated RealSense + DAVIS grasp-quality monitoring demo"


def hand_offset_reset_defaults():
    """Return the one canonical reset contract loaded from the demo YAML."""
    offset = normalized_offset_mm(
        HAND_MOUNTED_CAMERA_CONFIG.get("camera_to_hand_offset_mm"),
        DEFAULT_CAMERA_TO_HAND_OFFSET_MM,
    )
    gap = float(
        HAND_MOUNTED_CAMERA_CONFIG.get(
            "desired_contact_gap_mm", DEFAULT_DESIRED_CONTACT_GAP_MM
        )
    )
    desired = float(HAND_MOUNTED_CAMERA_CONFIG.get("desired_close_distance_mm", offset["z"] + gap))
    minimum = float(HAND_MOUNTED_CAMERA_CONFIG.get("min_close_distance_mm", desired - 20.0))
    maximum = float(HAND_MOUNTED_CAMERA_CONFIG.get("max_close_distance_mm", desired + 40.0))
    return {
        "camera_to_hand_offset_mm": offset,
        "desired_contact_gap_mm": gap,
        "desired_close_distance_mm": desired,
        "min_close_distance_mm": minimum,
        "max_close_distance_mm": maximum,
        "min_delta_mm": desired - minimum,
        "max_delta_mm": maximum - desired,
    }


HAND_OFFSET_RESET_DEFAULTS = hand_offset_reset_defaults()


def effective_hand_offset_close_range(offset_mm, desired_contact_gap_mm):
    """Return the runtime reach target while preserving the demo range deltas."""
    offset = normalized_offset_mm(offset_mm, HAND_OFFSET_RESET_DEFAULTS["camera_to_hand_offset_mm"])
    desired = float(offset["z"]) + float(desired_contact_gap_mm)
    return {
        "desired_close_distance_mm": desired,
        "min_close_distance_mm": desired - float(HAND_OFFSET_RESET_DEFAULTS["min_delta_mm"]),
        "max_close_distance_mm": desired + float(HAND_OFFSET_RESET_DEFAULTS["max_delta_mm"]),
        "close_distance_range_source": "camera_to_hand_offset_plus_contact_gap_estimate",
        "close_distance_range_validation_status": "APPROXIMATE_NOT_PHYSICALLY_CALIBRATED",
        "close_distance_range_physically_validated": False,
    }

app = Flask(__name__)

processes = {}
process_logs = {}
process_lock = threading.Lock()
process_starts_in_flight = set()
desired_processes = {"realsense": False, "davis_motion": False}
reconnect_thread = None
reconnect_stop = threading.Event()
reconnect_failures = {"realsense": 0, "davis_motion": 0}
reconnect_retry_after = {"realsense": 0.0, "davis_motion": 0.0}
RECONNECT_BASE_DELAY_SEC = 15.0
RECONNECT_MAX_DELAY_SEC = 120.0
RECONNECT_CIRCUIT_BREAKER_FAILURES = 3
REALSENSE_D405_USB_ID = ("8086", "0b5b")
realsense_usb_rearm = {"waiting": False, "last_present": False}
route_timing_lock = threading.Lock()
route_timings = {}
hardware_status_lock = threading.Lock()
hardware_status_cache = {"timestamp": 0.0, "payload": None}
hardware_status_refresh_thread = None
realtime_status_cache_lock = threading.Lock()
realtime_status_cache = {
    "payload": None,
    "read_monotonic": 0.0,
    "source_age_sec": None,
}
live_quality_monitor = LiveGraspQualityMonitor(load_live_quality_config())
runtime_recording = None
operator_status_snapshot_lock = threading.Lock()
operator_status_snapshot_cache = {"timestamp": 0.0, "payload": None}
derived_status_lock = threading.Lock()
derived_status_thread = None
derived_status_stop = threading.Event()
derived_status_health = {"last_success_time": 0.0}
benchmark_lock = threading.Lock()
benchmark_recorder = BenchmarkTrialRecorder(BENCHMARK_OUTPUT_DIR)
benchmark_runner = None
benchmark_runner_lock = threading.Lock()

davis_lock = threading.Lock()
davis_lifecycle_lock = threading.Lock()
davis_thread = None
davis_stop = threading.Event()
davis_analyzer = None
davis_last_error = None
davis_recording = False
DAVIS_DEBUG_HEATMAP = os.environ.get("DAVIS_DEBUG_HEATMAP", "0").strip().lower() in {"1", "true", "yes", "on"}
DAVIS_ROI_SCALE_X = float(os.environ.get("DAVIS_ROI_SCALE_X", "1.0"))
DAVIS_ROI_SCALE_Y = float(os.environ.get("DAVIS_ROI_SCALE_Y", "1.0"))
DAVIS_ROI_OFFSET_X = float(os.environ.get("DAVIS_ROI_OFFSET_X", "0.0"))
DAVIS_ROI_OFFSET_Y = float(os.environ.get("DAVIS_ROI_OFFSET_Y", "0.0"))
DAVIS_ROI_PADDING = float(os.environ.get("DAVIS_ROI_PADDING", "18.0"))
DAVIS_ROI_FRAME_WIDTH = int(os.environ.get("DAVIS_ROI_FRAME_WIDTH", "346"))
DAVIS_ROI_FRAME_HEIGHT = int(os.environ.get("DAVIS_ROI_FRAME_HEIGHT", "260"))


def start_dashboard_recording():
    """Attach before the first quality update; defaults off and opens no sensor."""
    global runtime_recording
    from grasp_project.recording.runtime import create_runtime_recording, runtime_recording_requested

    if not runtime_recording_requested():
        return None
    if runtime_recording is not None or live_quality_monitor._recording_epoch_input_index:
        raise ValueError("dashboard recording must start before the first quality update")
    from grasp_project.davis.motion_estimation import load_davis_motion_config
    from grasp_project.davis.motion_gui import (
        FALLBACK_EVENT_WINDOW_SEC, MIN_PRIMARY_WINDOW_EVENTS, PRIMARY_EVENT_WINDOW_SEC,
    )

    runtime_recording = create_runtime_recording("dashboard", {
        "live_quality": live_quality_monitor.config,
        "davis_motion": load_davis_motion_config(),
        "event_window_selection": {"primary_window_sec": PRIMARY_EVENT_WINDOW_SEC,
                                   "fallback_window_sec": FALLBACK_EVENT_WINDOW_SEC,
                                   "min_primary_events": MIN_PRIMARY_WINDOW_EVENTS},
    })
    live_quality_monitor.recording_sink = runtime_recording
    return runtime_recording


def finish_dashboard_recording(success):
    """Finalize only after producer and DAVIS teardown; failed teardown aborts."""
    if runtime_recording is None:
        return
    reconnect_stop.set()
    set_desired_process("davis_motion", False)
    set_desired_process("realsense", False)
    if reconnect_thread is not None and reconnect_thread.is_alive():
        reconnect_thread.join(timeout=4)
    derived_status_stop.set()
    if derived_status_thread is not None and derived_status_thread.is_alive():
        derived_status_thread.join(timeout=4)
    stopped, _message = stop_davis_motion()
    if (not stopped or (derived_status_thread is not None and derived_status_thread.is_alive())
            or (reconnect_thread is not None and reconnect_thread.is_alive())):
        runtime_recording.abort("dashboard_workers_did_not_stop")
        raise RuntimeError("recording remains incomplete: dashboard workers did not stop")
    with derived_status_lock:
        live_quality_monitor.recording_sink = None
        if success:
            runtime_recording.close()
        else:
            runtime_recording.abort("dashboard_runtime_failed_or_interrupted")


def _realsense_frame_shape_hw():
    """RealSense (height, width) for RealSense->DAVIS ROI mapping.

    The realtime status does not publish a frame shape, so fall back to the
    configured color profile (WxHxFPS), then a 640x480 default."""
    profile = os.environ.get("REALSENSE_COLOR_PROFILE") or os.environ.get("REALSENSE_DEPTH_PROFILE") or ""
    parts = str(profile).lower().split("x")
    try:
        w = int(parts[0])
        h = int(parts[1])
        if w > 0 and h > 0:
            return float(h), float(w)
    except (IndexError, ValueError):
        pass
    return 480.0, 640.0
STALE_FILE_THRESHOLD_SEC = 2.0
STREAM_DEBUG_RGB = os.environ.get("STREAM_DEBUG_RGB", "0").strip().lower() in {"1", "true", "yes", "on"}
STREAM_DEBUG_DEPTH = os.environ.get("STREAM_DEBUG_DEPTH", "0").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_STREAM_FPS = float(os.environ.get("DEBUG_STREAM_FPS", "1"))
DASHBOARD_POLL_INTERVAL_MS = int(os.environ.get("DASHBOARD_POLL_INTERVAL_MS", "500"))
REALTIME_STATUS_CACHE_GRACE_SEC = max(1.0, 3.0 * DASHBOARD_POLL_INTERVAL_MS / 1000.0)
DEBUG_STREAM_STATE_PATH = OUTPUT_DIR / "web_debug_streams.json"
BACKEND_COMMAND_PATH = OUTPUT_DIR / "realsense_backend_command.json"
BACKEND_COMMAND_STATUS_PATH = OUTPUT_DIR / "realsense_backend_command_status.json"
FRESH_LOCK_QUALITY_STATUS_PATH = OUTPUT_DIR / "fresh_lock_quality_status.json"
MATRIX_STALE_CENTER_SHIFT_PX = float(os.environ.get("MATRIX_STALE_CENTER_SHIFT_PX", "20"))
MATRIX_STALE_IOU_THRESHOLD = float(os.environ.get("MATRIX_STALE_IOU_THRESHOLD", "0.75"))
MATRIX_STALE_DEPTH_DELTA_MM = float(os.environ.get("MATRIX_STALE_DEPTH_DELTA_MM", "30"))
FINAL_DEMO_FILES = {
    "rgb": OUTPUT_DIR / "latest_rgb_live.jpg",
    "depth": OUTPUT_DIR / "latest_depth_live.jpg",
    "overlay": OUTPUT_DIR / "latest_overlay_live.jpg",
    "realtime_status": OUTPUT_DIR / "latest_realtime_status.json",
    "epsilon_result": OUTPUT_DIR / "realtime_epsilon_result.json",
    "quality_matrix_status": OUTPUT_DIR / "realtime_wrench_matrix_status.json",
    "current_quality_bundle": OUTPUT_DIR / "current_quality_bundle.json",
    "grasp_hulls": OUTPUT_DIR / "realtime_grasp_hulls.json",
    "grasp_hulls_overlay": OUTPUT_DIR / "latest_grasp_hulls_overlay.jpg",
    "bottle_quality_overlay": OUTPUT_DIR / "bottle_quality_overlay.jpg",
    "bottle_quality_context": OUTPUT_DIR / "bottle_quality_context.jpg",
    "bottle_quality_closeup": OUTPUT_DIR / "bottle_quality_closeup.jpg",
    "contact_quality_overlay": OUTPUT_DIR / "contact_quality_overlay.jpg",
    "grasp_quality_weak_direction": OUTPUT_DIR / "grasp_quality_bottle_mask_weak_direction.png",
    "contact_hull_overlay": OUTPUT_DIR / "contact_hull_overlay.jpg",
    "contact_patch_debug": OUTPUT_DIR / "contact_patch_debug.jpg",
    "contact_hull_debug": OUTPUT_DIR / "contact_hull_debug.json",
    "wrench_space_convex_hull_debug": OUTPUT_DIR / "wrench_space_convex_hull_debug.jpg",
    "wrench_space_3d_hull": OUTPUT_DIR / "wrench_space_3d_hull.png",
    "wrench_space_projection": OUTPUT_DIR / "wrench_space_projection.png",
    "wrench_space_fx_fy": OUTPUT_DIR / "wrench_space_fx_fy.png",
    "wrench_space_fx_tau": OUTPUT_DIR / "wrench_space_fx_tau.png",
    "wrench_space_fy_tau": OUTPUT_DIR / "wrench_space_fy_tau.png",
    "davis_motion_report": OUTPUT_DIR / "davis_motion_report.json",
    "davis_motion_overlay": OUTPUT_DIR / "davis_motion_overlay.jpg",
    "davis_adjusted_quality_overlay": OUTPUT_DIR / "davis_adjusted_quality_overlay.jpg",
    "davis_realsense_confirmation": OUTPUT_DIR / "davis_realsense_confirmation.jpg",
    "davis_performance_profile": OUTPUT_DIR / "davis_performance_profile.json",
    "pointcloud_status": OUTPUT_DIR / "pointcloud_status.json",
    "pointcloud_preview": OUTPUT_DIR / "latest_pointcloud_preview.jpg",
    "bottle_pointcloud_preview": OUTPUT_DIR / "latest_bottle_pointcloud_preview.jpg",
    "pointcloud_context": OUTPUT_DIR / "latest_pointcloud_context.jpg",
    "pointcloud_live_json": OUTPUT_DIR / "latest_pointcloud_live.json",
    "latest_realsense_pointcloud_ply": OUTPUT_DIR / "latest_realsense_pointcloud.ply",
    "latest_bottle_pointcloud_ply": OUTPUT_DIR / "latest_bottle_pointcloud.ply",
    "multiview_reconstruction_status": OUTPUT_DIR / "multiview" / "latest_status.json",
    "multiview_reconstruction_preview": OUTPUT_DIR / "multiview" / "latest_preview.json",
    "multiview_accumulated_scene_ply": OUTPUT_DIR / "multiview" / "accumulated_scene.ply",
    "multiview_accumulated_bottle_ply": OUTPUT_DIR / "multiview" / "accumulated_bottle.ply",
    "multiview_reconstructed_mesh_ply": OUTPUT_DIR / "multiview" / "reconstructed_bottle_mesh.ply",
    "sync_status": OUTPUT_DIR / "sync_status.json",
    "latest_sensor_bundle": OUTPUT_DIR / "latest_sensor_bundle.json",
    "embodiment_status": OUTPUT_DIR / "embodiment_status.json",
}

DAVIS_TARGET_DISPLAY_FPS = float(os.environ.get("DAVIS_TARGET_DISPLAY_FPS", "30"))
DAVIS_EVENT_ANALYSIS_FPS = float(os.environ.get("DAVIS_EVENT_ANALYSIS_FPS", "10"))
DAVIS_REPORT_INTERVAL_SEC = float(os.environ.get("DAVIS_REPORT_INTERVAL_SEC", "2.0"))
DAVIS_SAVE_DEBUG_EVERY_FRAME = os.environ.get("DAVIS_SAVE_DEBUG_EVERY_FRAME", "0").strip().lower() in {"1", "true", "yes", "on"}
DAVIS_CONFIRMATION_INTERVAL_SEC = float(os.environ.get("DAVIS_CONFIRMATION_INTERVAL_SEC", "0.5"))
DAVIS_REALSENSE_CONTEXT_MAX_AGE_SEC = float(os.environ.get("DAVIS_REALSENSE_CONTEXT_MAX_AGE_SEC", "1.0"))
DAVIS_APS_TARGET_CONTEXT_MAX_AGE_SEC = 0.75
DAVIS_PRE_CLOSE_DIAGNOSTIC_HOLD_SEC = float(os.environ.get("DAVIS_PRE_CLOSE_DIAGNOSTIC_HOLD_SEC", "5.0"))
DAVIS_HOMOGRAPHY_PATH = ROOT / "config" / "davis_to_realsense_homography.json"
davis_homography_cache_lock = threading.Lock()
davis_homography_cache = {"key": None, "payload": None}

profile_lock = common_profile_lock
profile = common_profile
profile_history = common_profile_history
debug_stream_lock = threading.Lock()
debug_stream_state = {
    "rgb": bool(STREAM_DEBUG_RGB),
    "depth": bool(STREAM_DEBUG_DEPTH),
    "updated_at": time.time(),
}
writer_session_lock = threading.Lock()
last_writer_session_id = None
last_writer_session_warning = None
last_writer_session_warning_time = 0.0
fresh_lock_quality_lock = threading.RLock()
fresh_lock_quality_thread = None
fresh_lock_quality_status = {
    "fresh_lock_quality_status": "idle",
    "fresh_lock_quality_step": "Idle",
    "fresh_lock_quality_error": None,
    "fresh_lock_quality_timestamp": None,
    "last_fresh_lock_quality_result": None,
    "fresh_lock_quality_requested": False,
    "fresh_lock_quality_request_received": False,
    "fresh_lock_quality_worker_busy": False,
    "fresh_lock_quality_target_valid": None,
    "fresh_lock_quality_lock_succeeded": False,
    "fresh_lock_quality_matrix_started": False,
    "fresh_lock_quality_matrix_completed": False,
    "fresh_lock_quality_failure_reason": None,
    "fresh_lock_quality_failure_code": None,
}


def normalize_benchmark_configurations(configurations):
    normalized = []
    for idx, config in enumerate(configurations or []):
        if not isinstance(config, dict):
            continue
        preset_id = config.get("preset_id")
        feature_ids = config.get("feature_ids")
        if preset_id and (not feature_ids) and preset_id in BENCHMARK_PRESETS:
            feature_ids = preset_feature_ids(preset_id)
        feature_ids = canonical_feature_ids(feature_ids or [])
        display_name = config.get("display_name")
        if not display_name and preset_id in BENCHMARK_PRESETS:
            display_name = BENCHMARK_PRESETS[preset_id]["display_name"]
        normalized.append({
            "configuration_id": config.get("configuration_id") or preset_id or f"custom_{idx + 1}",
            "display_name": display_name or f"Custom {idx + 1}",
            "preset_id": preset_id,
            "feature_ids": feature_ids,
        })
        if len(normalized) >= 4:
            break
    return normalized or default_comparison_configurations()


def benchmark_selection_snapshot():
    with benchmark_lock:
        payload = common_read_json(BENCHMARK_SELECTION_PATH, {})
    return normalize_benchmark_configurations((payload or {}).get("configurations"))


def save_benchmark_selection(configurations):
    normalized = normalize_benchmark_configurations(configurations)
    payload = {
        "configurations": normalized,
        "updated_at": time.time(),
        "benchmark_only": True,
        "production_control": False,
    }
    with benchmark_lock:
        common_write_json_atomic(BENCHMARK_SELECTION_PATH, payload)
    return normalized


def _benchmark_check(ok, reason="ok"):
    return {"ok": bool(ok), "reason": "ok" if ok else reason}


def benchmark_sensor_availability(realsense_summary, davis_summary, pointcloud_summary, sync_summary, quality_matrix):
    rs = realsense_summary or {}
    dm = davis_summary or {}
    pc = pointcloud_summary or {}
    sync = sync_summary or {}
    qm = quality_matrix or {}
    davis_event_state = str(dm.get("event_state") or dm.get("stream") or "").upper()
    aps_state = str(dm.get("aps_state") or dm.get("aps_stream_status") or "").upper()
    baseline_ok = str(qm.get("quality_matrix_status") or "").lower() in {"updated", "ok", "valid", "current"}
    return {
        "realsense_active": bool(rs.get("realsense_rgb_active") or rs.get("realsense_depth_active")),
        "rgb_active": bool(rs.get("realsense_rgb_active")),
        "depth_active": bool(rs.get("realsense_depth_active")),
        "bottle_detected": str(rs.get("target_status") or "").upper() in {"TRACKING", "LOCKED", "DETECTED"},
        "target_stable": str(rs.get("target_status") or "").upper() in {"TRACKING", "LOCKED"} and float(rs.get("tracking_confidence") or 0.0) >= 40.0,
        "dashboard_fresh": not bool(rs.get("overlay_stale") or rs.get("overlay_health") == "STALE"),
        "pointcloud_active": bool(pc.get("active")) and str(pc.get("freshness") or "").upper() in {"CURRENT", "AGING", ""},
        "davis_aps_active": aps_state in {
            "APS_COLOR_VERIFIED", "APS_RAW_BAYER", "APS_GRAYSCALE",
            "APS_UNKNOWN", "STREAM_ACTIVE", "ACTIVE",
        },
        "davis_events_active": davis_event_state in {"STREAM_ACTIVE_EVENTS", "STREAM_ACTIVE_NO_MOTION", "ACTIVE", "RUNNING"},
        "baseline_quality_available": bool(baseline_ok or qm.get("quality_label")),
        "sync_available": bool(sync.get("sync_quality") or sync.get("last_bundle_id") or sync.get("bundle_id")),
        "multiview_available": False,
    }


def benchmark_precheck_snapshot():
    source = benchmark_source_snapshot(configurations=[])
    rs = source["realsense_summary"]
    availability = source["availability"]
    with process_lock:
        production_running = is_process_running("realsense")
    required = {
        "production_pipeline_running": _benchmark_check(production_running, "production_pipeline_not_running"),
        "realsense_active": _benchmark_check(availability["realsense_active"], "realsense_inactive"),
        "rgb_active": _benchmark_check(availability["rgb_active"], "rgb_stream_inactive"),
        "depth_active": _benchmark_check(availability["depth_active"], "depth_stream_inactive"),
        "bottle_detected": _benchmark_check(availability["bottle_detected"], "bottle_not_detected"),
        "target_stable": _benchmark_check(availability["target_stable"], "target_not_stable"),
        "dashboard_fresh": _benchmark_check(availability["dashboard_fresh"], "dashboard_status_or_overlay_stale"),
    }
    optional = {
        "pointcloud_active": _benchmark_check(availability["pointcloud_active"], "pointcloud_unavailable_or_stale"),
        "davis_aps_active": _benchmark_check(availability["davis_aps_active"], "davis_aps_unavailable"),
        "davis_events_active": _benchmark_check(availability["davis_events_active"], "davis_events_unavailable"),
        "baseline_quality_available": _benchmark_check(availability["baseline_quality_available"], "baseline_quality_unavailable"),
        "sync_available": _benchmark_check(availability["sync_available"], "sync_status_unavailable"),
        "multiview_available": _benchmark_check(availability["multiview_available"], "multiview_status_unavailable"),
    }
    failed = [name for name, item in required.items() if not item["ok"]]
    return {
        "ok": not failed,
        "reason": ",".join(failed) if failed else "ok",
        "required": required,
        "optional": optional,
        "target_status": rs.get("target_status"),
        "tracking_confidence": rs.get("tracking_confidence"),
        "overlay_health": rs.get("overlay_health"),
        "benchmark_only": True,
    }


def current_benchmark_status(
    *,
    realsense_summary=None,
    davis_summary=None,
    pointcloud_summary=None,
    sync_summary=None,
    quality_matrix=None,
    live_quality_status=None,
    embodiment_status=None,
    configurations=None,
    include_runner=True,
):
    status = build_benchmark_lab_status(
        realsense=realsense_summary or {},
        davis=davis_summary or {},
        pointcloud=pointcloud_summary or {},
        sync=sync_summary or {},
        quality=quality_matrix or {},
        live_quality=live_quality_status or {},
        embodiment=embodiment_status or {},
        configurations=configurations if configurations is not None else benchmark_selection_snapshot(),
        trial_state=benchmark_recorder.current_state(),
    )
    status["source_status"] = {
        "sensor_availability": benchmark_sensor_availability(
            realsense_summary or {},
            davis_summary or {},
            pointcloud_summary or {},
            sync_summary or {},
            quality_matrix or {},
        ),
        "realsense": realsense_summary or {},
        "davis": davis_summary or {},
        "pointcloud": pointcloud_summary or {},
        "sync": sync_summary or {},
        "quality": quality_matrix or {},
        "live_quality": live_quality_status or {},
        "embodiment": embodiment_status or {},
    }
    status["default_run_configurations"] = default_run_configurations(
        duration_sec=(load_benchmark_settings().get("duration_sec") or 20.0)
    )
    if include_runner:
        status["run"] = get_benchmark_runner().snapshot()
    return status


def _same_dashboard_origin(url):
    """Compare browser origins, including default ports, without trusting a prefix."""
    try:
        source = urlsplit(url)
        target = urlsplit(request.host_url)
        return (
            source.scheme in {"http", "https"}
            and source.scheme == target.scheme
            and source.hostname == target.hostname
            and source.username is None
            and source.password is None
            and (source.port if source.port is not None else (443 if source.scheme == "https" else 80))
            == (target.port if target.port is not None else (443 if target.scheme == "https" else 80))
        )
    except ValueError:
        return False


@app.before_request
def _reject_nonlocal_hosts():
    """Reject DNS-rebinding hosts even when their browser Origin matches Host."""
    try:
        hostname = urlsplit(request.host_url).hostname
    except ValueError:
        hostname = None
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        return jsonify({"ok": False, "message": "This dashboard supports loopback hosts only."}), 403
    return None


@app.before_request
def _reject_cross_origin_controls():
    """Block browser CSRF while preserving local command-line launcher requests.

    This is not authentication; changing HOST cannot secure remote deployment.
    """
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return None
    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")
    source = origin if origin is not None else referer
    if (
        request.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}
        or (source is not None and not _same_dashboard_origin(source))
    ):
        return jsonify({"ok": False, "message": "Cross-origin control requests are not allowed."}), 403
    return None


@app.before_request
def _route_timer_start():
    request._demo_route_t0 = time.perf_counter()


@app.after_request
def _route_timer_stop(response):
    started = getattr(request, "_demo_route_t0", None)
    if started is None:
        return response
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    with route_timing_lock:
        route_timings[request.path] = {
            "last_ms": elapsed_ms,
            "updated_at": time.time(),
        }
    response.headers["X-Demo-Route-Ms"] = str(elapsed_ms)
    return response


def route_timing_snapshot():
    with route_timing_lock:
        return {key: dict(value) for key, value in route_timings.items()}


def write_debug_stream_state():
    with debug_stream_lock:
        state = dict(debug_stream_state)
    tmp_path = DEBUG_STREAM_STATE_PATH.with_name(f".{DEBUG_STREAM_STATE_PATH.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(DEBUG_STREAM_STATE_PATH)


def set_debug_streams(enabled):
    with debug_stream_lock:
        debug_stream_state["rgb"] = bool(STREAM_DEBUG_RGB or enabled)
        debug_stream_state["depth"] = bool(STREAM_DEBUG_DEPTH or enabled)
        debug_stream_state["updated_at"] = time.time()
    write_debug_stream_state()


def debug_stream_enabled(name):
    with debug_stream_lock:
        return bool(debug_stream_state.get(name))


def debug_stream_snapshot():
    with debug_stream_lock:
        return dict(debug_stream_state)


def update_writer_session_warning(realtime):
    global last_writer_session_id, last_writer_session_warning, last_writer_session_warning_time
    session_id = (realtime or {}).get("writer_session_id")
    if not session_id:
        return last_writer_session_warning
    with writer_session_lock:
        if last_writer_session_id and session_id != last_writer_session_id:
            last_writer_session_warning = (
                f"writer_session_id changed from {last_writer_session_id} to {session_id}; "
                "live status may include mixed backend snapshots"
            )
            last_writer_session_warning_time = time.time()
        elif last_writer_session_warning and time.time() - last_writer_session_warning_time > 10.0:
            last_writer_session_warning = None
        last_writer_session_id = session_id
        return last_writer_session_warning


def utc_iso():
    return common_utc_iso()


def fresh_status_snapshot():
    persisted = read_json(FRESH_LOCK_QUALITY_STATUS_PATH, {})
    with fresh_lock_quality_lock:
        status = dict(fresh_lock_quality_status)
    if persisted:
        status.update({k: v for k, v in persisted.items() if v is not None})
    return status


def update_fresh_status(status=None, step=None, error=None, result=None, **extra):
    with fresh_lock_quality_lock:
        if status is not None:
            fresh_lock_quality_status["fresh_lock_quality_status"] = status
        if step is not None:
            fresh_lock_quality_status["fresh_lock_quality_step"] = step
        fresh_lock_quality_status["fresh_lock_quality_error"] = error
        fresh_lock_quality_status["fresh_lock_quality_timestamp"] = utc_iso()
        if result is not None:
            fresh_lock_quality_status["last_fresh_lock_quality_result"] = result
        fresh_lock_quality_status.update(extra)
        snapshot = dict(fresh_lock_quality_status)
    write_json_atomic(FRESH_LOCK_QUALITY_STATUS_PATH, snapshot)
    return snapshot


def profile_set(stage, value_ms):
    return common_profile_set(stage, value_ms)


def profile_snapshot():
    return common_profile_snapshot()


class LatestJpegBuffer:
    """Single-slot latest-frame buffer. Producers overwrite; consumers never queue."""

    def __init__(self, name):
        self.name = name
        self.cond = threading.Condition()
        self.frame_id = 0
        self.jpeg = None
        self.meta = {}
        self.last_sent_id = 0
        self.overwrite_drops = 0
        self.produced_times = deque(maxlen=120)
        self.sent_times = deque(maxlen=120)
        self.last_latency_ms = None

    def update(self, jpeg, **meta):
        with self.cond:
            if self.jpeg is not None and self.frame_id > self.last_sent_id:
                self.overwrite_drops += 1
            self.frame_id += 1
            self.jpeg = jpeg
            self.meta = {"frame_id": self.frame_id, "timestamp": time.time(), **meta}
            self.produced_times.append(self.meta["timestamp"])
            self.cond.notify_all()

    def wait_latest(self, last_id=0, timeout=1.0):
        with self.cond:
            if self.jpeg is None or self.frame_id == last_id:
                self.cond.wait(timeout=timeout)
            return self.frame_id, self.jpeg, dict(self.meta)

    def mark_sent(self, frame_id, meta):
        with self.cond:
            self.last_sent_id = max(self.last_sent_id, int(frame_id))
            now = time.time()
            self.sent_times.append(now)
            ts = meta.get("timestamp")
            self.last_latency_ms = None if ts is None else (now - float(ts)) * 1000.0

    @staticmethod
    def _fps(times):
        if len(times) < 2:
            return 0.0
        span = max(times[-1] - times[0], 1e-6)
        return float((len(times) - 1) / span)

    def stats(self):
        with self.cond:
            age_ms = None
            if self.meta.get("timestamp") is not None:
                age_ms = (time.time() - float(self.meta["timestamp"])) * 1000.0
            source_age_ms = None
            if self.meta.get("source_mtime_ns") is not None:
                source_age_ms = (time.time() - float(self.meta["source_mtime_ns"]) / 1e9) * 1000.0
            return {
                "frame_id": self.frame_id,
                "queue_size": 1 if self.jpeg is not None else 0,
                "overwrite_drops": self.overwrite_drops,
                "producer_fps": round(self._fps(self.produced_times), 2),
                "send_fps": round(self._fps(self.sent_times), 2),
                "frame_age_ms": None if age_ms is None else round(age_ms, 1),
                "source_file_age_ms": None if source_age_ms is None else round(source_age_ms, 1),
                "send_latency_ms": None if self.last_latency_ms is None else round(self.last_latency_ms, 1),
                "source": self.meta.get("source"),
            }


stream_buffers = {
    "rgb": LatestJpegBuffer("rgb"),
    "depth": LatestJpegBuffer("depth"),
    "overlay": LatestJpegBuffer("overlay"),
    "dashboard": LatestJpegBuffer("dashboard"),
    "epsilon": LatestJpegBuffer("epsilon"),
    "davis_heatmap": LatestJpegBuffer("davis_heatmap"),
    "davis_aps": LatestJpegBuffer("davis_aps"),
}

stream_files = {
    "rgb": OUTPUT_DIR / "latest_rgb_live.jpg",
    "depth": OUTPUT_DIR / "latest_depth_live.jpg",
    "overlay": OUTPUT_DIR / "latest_overlay_live.jpg",
}

poller_threads = []
poller_stop = threading.Event()


def encode_file_to_jpeg(path, quality=78):
    return common_encode_file_to_jpeg(path, quality=quality)


def encode_rgb_to_jpeg(rgb, quality=75):
    return common_encode_rgb_to_jpeg(rgb, quality=quality)


def file_poller_loop(name, path, interval=0.03, quality=78):
    last_mtime = None
    while not poller_stop.is_set():
        try:
            if name in {"rgb", "depth"} and not debug_stream_enabled(name):
                last_mtime = None
                time.sleep(max(interval, 0.25))
                continue
            if path.exists():
                mtime = path.stat().st_mtime_ns
                if mtime != last_mtime:
                    last_mtime = mtime
                    jpeg, read_ms, encode_ms = encode_file_to_jpeg(path, quality=quality)
                    if jpeg:
                        profile_set(f"{name}_file_read", read_ms)
                        profile_set(f"{name}_encoding", encode_ms)
                        stream_buffers[name].update(jpeg, source=str(path), mtime_ns=mtime, source_mtime_ns=mtime)
        except Exception as exc:
            profile_set(f"{name}_poll_error", 0.0)
        time.sleep(interval)


def start_file_pollers():
    if poller_threads:
        return
    for name, path in stream_files.items():
        interval = max(1.0 / max(DEBUG_STREAM_FPS, 0.1), 0.25) if name in {"rgb", "depth"} else 0.03
        quality = 76 if name in {"rgb", "depth", "overlay", "davis_heatmap"} else 82
        thread = threading.Thread(target=file_poller_loop, args=(name, path, interval, quality), daemon=True)
        thread.start()
        poller_threads.append(thread)


COMMANDS = {
    "realsense": {
        "title": "RealSense RGB-D Grasp Planner",
        "cmd": ["./run_bottle_realtime.sh", "--headless-web"],
        "log": LOG_DIR / "realsense_backend.log",
    },
}


def realsense_d405_enumerated(sysfs_root=Path("/sys/bus/usb/devices")):
    """Check the passive USB identity files without opening the camera."""
    vendor_id, product_id = REALSENSE_D405_USB_ID
    try:
        device_paths = tuple(Path(sysfs_root).iterdir())
    except OSError:
        return False
    for device_path in device_paths:
        try:
            if (device_path / "idVendor").read_text().strip().lower() != vendor_id:
                continue
            if (device_path / "idProduct").read_text().strip().lower() == product_id:
                return True
        except OSError:
            continue
    return False


def observe_realsense_usb_for_rearm(present):
    """Consume an absent-to-present transition after the recovery circuit opens."""
    present = bool(present)
    transitioned = bool(
        realsense_usb_rearm["waiting"]
        and not realsense_usb_rearm["last_present"]
        and present
    )
    realsense_usb_rearm["last_present"] = present
    if transitioned:
        realsense_usb_rearm["waiting"] = False
    return transitioned


def stop_realsense_recovery_processes():
    """Stop a recovery-owned camera driver after timeout/disconnect."""
    pid_path = OUTPUT_DIR / "realsense_camera_launch.pid"
    try:
        driver_pid = int(pid_path.read_text().strip())
    except (FileNotFoundError, TypeError, ValueError):
        driver_pid = None
    if driver_pid:
        try:
            os.killpg(driver_pid, signal.SIGTERM)
            time.sleep(0.25)
        except (ProcessLookupError, PermissionError):
            pass
    for pattern in (
        "realsense_transport_health.py",
        "realsense2_camera_node",
        "ros2 launch realsense2_camera rs_launch.py",
    ):
        subprocess.run(
            ["pkill", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    pid_path.unlink(missing_ok=True)


def run_realsense_transport_recovery():
    """Restart/check RealSense ROS transport before launching grasp mode."""
    script = ROOT / "scripts" / "recover_realsense_ros_transport.sh"
    log_path = LOG_DIR / "realsense_backend.log"
    if not script.exists():
        return False, f"CAMERA STREAM ERROR: recovery script missing: {script}"
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
    env.setdefault("ROS_DOMAIN_ID", "67")
    env.setdefault("REALSENSE_FPS", "30")
    env.setdefault("REALSENSE_COLOR_PROFILE", f"480x270x{env.get('REALSENSE_FPS', '30')}")
    env.setdefault("REALSENSE_DEPTH_PROFILE", f"480x270x{env.get('REALSENSE_FPS', '30')}")
    with open(log_path, "ab", buffering=0) as log:
        log.write(b"\n=== RealSense transport recovery/health gate ===\n")
        try:
            proc = subprocess.run(
                [str(script)],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=90,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(b"CAMERA STREAM ERROR: RealSense recovery timed out.\n")
            stop_realsense_recovery_processes()
            return False, "CAMERA STREAM ERROR: RealSense recovery timed out. Suggested fix: restart camera driver / lower resolution / restart ROS daemon."
    profile_set("realsense_transport_recovery", (time.perf_counter() - started) * 1000.0)
    health = read_json(OUTPUT_DIR / "realsense_camera_health.json", {})
    if proc.returncode != 0 or not health.get("pass"):
        if health.get("status") == "DEVICE_NOT_CONNECTED":
            return (
                False,
                "CAMERA STREAM ERROR: RealSense D405 is not enumerated. "
                "Reconnect it, confirm USB enumeration, then retry Start RealSense.",
            )
        last_error = health.get("last_error") or f"recovery exited {proc.returncode}"
        return (
            False,
            "CAMERA STREAM ERROR: RealSense ROS topic unreadable. "
            f"last error: {last_error}. "
            "suggested fix: restart camera driver / lower resolution / restart ROS daemon.",
        )
    return True, "RealSense ROS transport health PASS."


def read_json(path, default=None):
    return common_read_json(path, default=default)


def read_realtime_status_snapshot(path=None):
    """Read one coherent status snapshot across Windows bind-mount replaces.

    Docker Desktop can briefly remove the destination name while replacing the
    JSON file on a Windows bind mount.  A dashboard poll in that interval must
    not turn a healthy camera into an empty/offline payload or raise from an
    ``exists()``/``stat()`` race.  Retain only a short, bounded last-good
    snapshot; normal freshness gates still expire it.
    """
    path = Path(path or FINAL_DEMO_FILES["realtime_status"])
    now_monotonic = time.monotonic()
    payload = read_json(path, {})
    try:
        source_age_sec = max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        source_age_sec = None

    with realtime_status_cache_lock:
        if isinstance(payload, dict) and payload:
            realtime_status_cache.update({
                "payload": dict(payload),
                "read_monotonic": now_monotonic,
                "source_age_sec": source_age_sec,
            })
            return payload, source_age_sec, False

        cached = realtime_status_cache.get("payload")
        elapsed = max(
            0.0,
            now_monotonic - float(realtime_status_cache.get("read_monotonic") or 0.0),
        )
        if isinstance(cached, dict) and cached and elapsed <= REALTIME_STATUS_CACHE_GRACE_SEC:
            cached_source_age = realtime_status_cache.get("source_age_sec")
            estimated_age = (
                max(0.0, float(cached_source_age) + elapsed)
                if cached_source_age is not None
                else None
            )
            return dict(cached), estimated_age, True

    return {}, source_age_sec, False


def write_json_atomic(path, value):
    return common_write_json_atomic(path, value)


def write_bytes_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_bytes(payload)
    try:
        replace_path_retry(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_runtime_output_best_effort(writer, *args, **kwargs):
    """Keep camera analysis alive when a diagnostic output is momentarily locked."""
    try:
        writer(*args, **kwargs)
        return True
    except OSError:
        return False


def read_davis_report():
    with davis_lock:
        live_report = getattr(davis_analyzer, "report", None)
        if isinstance(live_report, dict) and live_report:
            return dict(live_report)
    report = read_json(DAVIS_LATEST_MOTION_REPORT_PATH)
    if report:
        return report
    return read_json(OUTPUT_DIR / "davis_motion_report.json")


def object_profile_dashboard_status():
    """Return object-profile state without making the dashboard depend on cameras."""
    try:
        status = object_registry_status()
        status["scan_report"] = read_json(OBJECT_SCAN_REPORT_PATH, {})
        status["active_grasp_profile"] = active_object_grasp_profile()
        status["dashboard_status"] = "ready"
        return status
    except Exception as exc:
        return {
            "dashboard_status": "error",
            "object_profile_status": "error",
            "message": str(exc),
            "profiles": [],
            "active_profile": {},
            "scan_report": read_json(OBJECT_SCAN_REPORT_PATH, {}),
            "production_safety_unchanged": True,
        }


def first_number(*values):
    for value in values:
        try:
            if value is None:
                continue
            number = float(value)
            if np.isfinite(number):
                return number
        except (TypeError, ValueError):
            continue
    return None


def bbox_xyxy_from_geometry(geometry):
    return common_bbox_xyxy_from_geometry(geometry)


def bbox_center(bbox):
    return common_bbox_center(bbox)


def bbox_iou_xyxy(a, b):
    return common_bbox_iou_xyxy(a, b)


def live_matrix_staleness(realtime, matrix, hulls):
    """Validate analytic identity while reporting approach motion separately.

    The RealSense backend owns the analytic matrix and its lock/compute
    provenance.  Image-space center, bbox IoU, and depth naturally change as
    the hand-mounted camera approaches; they are live observations, not proof
    that epsilon must be recomputed.
    """
    realtime = realtime or {}
    matrix = matrix or {}
    hulls = hulls or {}
    locked_file = read_json(OUTPUT_DIR / "locked_target.json", {})
    locked = (locked_file.get("locked_target") or {}).get("locked_target") or locked_file.get("locked_target") or {}
    locked_geometry = locked.get("locked_bottle_geometry") or {}
    current_geometry = realtime.get("bottle_geometry") or {}
    locked_bbox = (
        bbox_xyxy_from_geometry(locked_geometry)
        or (matrix.get("locked_quality_bbox_xyxy") if isinstance(matrix.get("locked_quality_bbox_xyxy"), list) else None)
        or (hulls.get("locked_quality_bbox_xyxy") if isinstance(hulls.get("locked_quality_bbox_xyxy"), list) else None)
        or (hulls.get("bottle_bbox_xyxy") if isinstance(hulls.get("bottle_bbox_xyxy"), list) else None)
    )
    current_bbox = (
        bbox_xyxy_from_geometry(current_geometry)
        or (realtime.get("raw_bottle_bbox") if isinstance(realtime.get("raw_bottle_bbox"), list) else None)
        or (realtime.get("accepted_current_bottle_bbox") if isinstance(realtime.get("accepted_current_bottle_bbox"), list) else None)
    )
    locked_center = locked_geometry.get("center_px") or bbox_center(locked_bbox)
    current_center = current_geometry.get("center_px") or realtime.get("raw_bottle_center") or bbox_center(current_bbox)
    center_delta = None
    if locked_center is not None and current_center is not None:
        try:
            center_delta = float(np.linalg.norm(np.asarray(current_center, dtype=float) - np.asarray(locked_center, dtype=float)))
        except (TypeError, ValueError):
            center_delta = None
    iou = bbox_iou_xyxy(locked_bbox, current_bbox)
    locked_depth = first_number(locked_geometry.get("depth_median_mm"), locked.get("depth_mm_at_lock"))
    current_depth = first_number(
        current_geometry.get("depth_median_mm"),
        realtime.get("bottle_depth_median_mm"),
        realtime.get("depth_mm"),
    )
    depth_delta = None if locked_depth is None or current_depth is None else abs(float(current_depth) - float(locked_depth))
    observation_reasons = []
    if center_delta is not None and center_delta > MATRIX_STALE_CENTER_SHIFT_PX:
        observation_reasons.append("center_shift_live_observation")
    if iou is not None and iou < MATRIX_STALE_IOU_THRESHOLD:
        observation_reasons.append("bbox_scale_or_viewpoint_live_observation")
    if depth_delta is not None and depth_delta > MATRIX_STALE_DEPTH_DELTA_MM:
        observation_reasons.append("depth_change_live_observation")
    current_frame = realtime.get("frame_id")
    matrix_frame = matrix.get("current_hull_frame_id") or matrix.get("quality_matrix_frame_id") or hulls.get("current_hull_frame_id")
    lock_frame = locked.get("lock_frame_id") or matrix.get("lock_frame_id") or hulls.get("lock_frame_id")
    writer_session = realtime.get("writer_session_id")
    matrix_writer = matrix.get("writer_session_id") or hulls.get("writer_session_id")
    target_lock = realtime.get("target_lock") if isinstance(realtime.get("target_lock"), dict) else {}
    locked_band = (
        realtime.get("locked_grasp_band")
        if isinstance(realtime.get("locked_grasp_band"), dict)
        else target_lock.get("locked_grasp_band")
        if isinstance(target_lock.get("locked_grasp_band"), dict)
        else {}
    )
    live_attempt = realtime.get("lock_attempt_id") or target_lock.get("attempt_id")
    matrix_attempt = matrix.get("lock_attempt_id") or matrix.get("attempt_id") or hulls.get("lock_attempt_id")
    live_compute = realtime.get("quality_compute_id") or locked_band.get("locked_quality_compute_id")
    matrix_compute = matrix.get("quality_compute_id") or hulls.get("quality_compute_id")
    same_lock_lineage = bool(
        writer_session
        and matrix_writer
        and str(writer_session) == str(matrix_writer)
        and lock_frame is not None
        and (realtime.get("lock_frame_id") or target_lock.get("lock_frame_id")) is not None
        and str(realtime.get("lock_frame_id") or target_lock.get("lock_frame_id")) == str(lock_frame)
        and live_attempt
        and matrix_attempt
        and str(live_attempt) == str(matrix_attempt)
    )
    relock_required = bool(
        realtime.get("grasp_band_relock_required")
        or realtime.get("band_relock_required")
        or target_lock.get("grasp_band_relock_required")
        or target_lock.get("band_relock_required")
    )
    same_lock_quality_refresh = False
    reasons = []
    if matrix_writer and writer_session and matrix_writer != writer_session:
        reasons.append("writer_session_mismatch")
    if lock_frame and current_frame is not None:
        live_lock_frame = realtime.get("lock_frame_id") or target_lock.get("lock_frame_id")
        if live_lock_frame is not None and str(live_lock_frame) != str(lock_frame):
            reasons.append("lock_frame_mismatch")
    if live_attempt and matrix_attempt and str(live_attempt) != str(matrix_attempt):
        reasons.append("lock_attempt_mismatch")
    if live_compute and matrix_compute and str(live_compute) != str(matrix_compute):
        if same_lock_lineage and not relock_required and realtime.get("post_grasp_active") is not True:
            same_lock_quality_refresh = True
            observation_reasons.append("same_lock_quality_refresh_in_progress")
        else:
            reasons.append("quality_compute_id_mismatch")
    if relock_required:
        reasons.append("grasp_band_relock_required")
    backend_stale = bool(matrix.get("quality_label_stale") or matrix.get("matrix_stale_due_to_motion"))
    if backend_stale:
        backend_reasons = list(
            matrix.get("quality_current_mismatch_reasons")
            or matrix.get("matrix_stale_reasons")
            or hulls.get("matrix_stale_reasons")
            or ["backend_marked_analytic_quality_stale"]
        )
        compute_refresh_reasons = {"quality_compute_id_changed", "quality_compute_id_mismatch"}
        if (
            same_lock_lineage
            and not relock_required
            and realtime.get("post_grasp_active") is not True
            and backend_reasons
            and set(str(reason) for reason in backend_reasons).issubset(compute_refresh_reasons)
        ):
            same_lock_quality_refresh = True
            observation_reasons.append("same_lock_quality_refresh_in_progress")
        else:
            reasons.extend(backend_reasons)
    reasons = list(dict.fromkeys(str(reason) for reason in reasons if reason))
    return {
        "lock_frame_id": lock_frame,
        "current_hull_frame_id": current_frame,
        "matrix_hull_frame_id": matrix_frame,
        "locked_quality_bbox_xyxy": locked_bbox,
        "current_live_bbox_xyxy": current_bbox,
        "current_vs_locked_center_delta_px": None if center_delta is None else round(center_delta, 2),
        "current_vs_locked_iou": None if iou is None else round(float(iou), 4),
        "current_vs_locked_depth_delta_mm": None if depth_delta is None else round(depth_delta, 2),
        "matrix_stale_due_to_motion": bool(reasons),
        "matrix_stale_reasons": reasons,
        "same_lock_quality_refresh_in_progress": same_lock_quality_refresh,
        "live_observation_degraded": bool(observation_reasons),
        "live_observation_reasons": observation_reasons,
        "analytic_staleness_source": "backend_identity_and_safety_semantics",
        "staleness_thresholds": {
            "center_shift_px": MATRIX_STALE_CENTER_SHIFT_PX,
            "iou": MATRIX_STALE_IOU_THRESHOLD,
            "depth_delta_mm": MATRIX_STALE_DEPTH_DELTA_MM,
        },
    }


def _xyxy_to_xywh(bbox):
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def _first_bbox_xywh(realtime):
    realtime = realtime or {}
    geometry = realtime.get("bottle_geometry") or {}
    if geometry.get("bbox_xywh"):
        return geometry.get("bbox_xywh")
    # the realtime status publishes the tracked bottle box as xyxy, not xywh;
    # accept the tracking-oriented boxes so the DAVIS ROI can follow the object
    for source in (
        geometry.get("contact_tracking_bbox_xyxy"),
        geometry.get("bottle_extent_bbox_xyxy"),
        geometry.get("bbox_xyxy"),
        realtime.get("display_bottle_bbox"),
        realtime.get("raw_bottle_bbox"),
    ):
        converted = _xyxy_to_xywh(source)
        if converted is not None:
            return converted
    target_lock = realtime.get("target_lock") or {}
    tracked = target_lock.get("current_tracked_target") or {}
    patch_bbox = tracked.get("patch_bbox_current_px") or target_lock.get("patch_bbox_current_px") or realtime.get("patch_bbox_current_px")
    converted = _xyxy_to_xywh(patch_bbox)
    if converted is not None:
        return converted
    return None


def davis_stage_from_realsense(realtime):
    """Map the existing RealSense state machine to DAVIS motion semantics."""
    realtime = realtime or {}
    state_machine = realtime.get("state_machine") or {}
    state = str(realtime.get("state") or state_machine.get("state") or "").upper()
    if state == "READY_TO_LIFT":
        return "LIFT_CHECK"
    if (
        state in {"POST_GRASP_EVALUATION", "REGRIP", "DO_NOT_LIFT"}
        or realtime.get("post_grasp_active") is True
        or realtime.get("gripper_closed_confirmed") is True
        or realtime.get("manual_gripper_closed") is True
    ):
        return "POST_CLOSE_STABILITY"
    if state in {"CLOSE_IMPACT", "GRIPPER_CLOSING"}:
        return "CLOSE_IMPACT"
    return "PRE_CLOSE"


def use_davis_native_grasp_zone(configured_roi, stage, target_context_status):
    """Use the configured fallback only when no higher-priority ROI is valid.

    A frozen calibrated RealSense band remains valid for local post-close
    motion, and a DAVIS-native APS target (when supplied) also outranks the
    configured diagnostic ROI.  Neither may be silently replaced just because
    the RealSense state crossed the close boundary.
    """
    if configured_roi is None:
        return False
    status = target_context_status or {}
    mode = str(status.get("davis_target_context_mode") or "none")
    priority = str(status.get("davis_target_roi_priority_selected") or "none")
    alignment = str(status.get("davis_target_alignment_status") or "none")
    context_valid = status.get("davis_target_context_valid") is True
    calibrated_locked_mapping = bool(
        context_valid
        and mode in {"live_realsense_mapping", "post_grasp_locked_mapping"}
        and priority.startswith("calibrated_realsense_")
        and alignment == "CALIBRATED_2D_REFERENCE_PLANE"
    )
    davis_aps_target = bool(
        context_valid
        and (
            mode == "davis_aps_target"
            or priority in {"davis_aps_target", "davis_aps_target_roi"}
        )
    )
    if calibrated_locked_mapping or davis_aps_target:
        return False
    return str(stage or "PRE_CLOSE").upper() != "PRE_CLOSE" or mode == "pre_close_held_target_diagnostic"


def calibrated_realsense_target_context(context, status):
    """Return whether priority-1 calibrated locked-band evidence is current."""
    context = context or {}
    status = status or {}
    priority = str(
        context.get("roi_priority_selected")
        or status.get("davis_target_roi_priority_selected")
        or ""
    )
    alignment = str(
        context.get("alignment_status")
        or status.get("davis_target_alignment_status")
        or ""
    )
    return bool(
        context
        and status.get("davis_target_context_valid") is True
        and priority.startswith("calibrated_realsense_")
        and alignment == "CALIBRATED_2D_REFERENCE_PLANE"
    )


def current_davis_aps_target_context(analyzer, *, now=None):
    """Return a fresh priority-2 DAVIS-native APS target, never stale evidence."""
    now = time.monotonic() if now is None else float(now)
    raw = getattr(analyzer, "latest_aps_target_context", None) if analyzer is not None else None
    if not isinstance(raw, dict) or raw.get("roi_source") != "aps_target":
        return None, "aps_target_unavailable"
    try:
        captured = float(raw.get("captured_monotonic"))
        age_sec = max(0.0, now - captured)
    except (TypeError, ValueError):
        return None, "aps_target_timestamp_unavailable"
    try:
        max_age_sec = float(
            raw.get("aps_target_context_max_age_sec")
            or DAVIS_APS_TARGET_CONTEXT_MAX_AGE_SEC
        )
    except (TypeError, ValueError):
        max_age_sec = DAVIS_APS_TARGET_CONTEXT_MAX_AGE_SEC
    max_age_sec = float(np.clip(max_age_sec, 0.05, 3.0))
    if not np.isfinite(age_sec) or age_sec > max_age_sec:
        return None, f"aps_target_stale_age_{age_sec:.3f}s"
    roi = raw.get("roi_xywh")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return None, "aps_target_roi_invalid"
    context = _copy_davis_target_context(raw)
    # latest_aps_target_context is populated only after the detector's
    # multi-frame confirmation (or its bounded post-grasp latch).
    context["aps_target_confirmed"] = True
    confidence = first_number(context.get("target_confidence"))
    reason = (
        f"fresh DAVIS APS bottle target age={age_sec:.3f}s"
        + ("" if confidence is None else f", confidence={confidence:.3f}")
    )
    context["mapping_reason"] = reason
    status = {
        "davis_target_context_mode": "davis_aps_target",
        "davis_target_context_valid": True,
        "davis_target_context_reason": reason,
        "davis_target_lock_active": False,
        "davis_target_lock_age_ms": None,
        "davis_target_lock_source_frame_id": None,
        "davis_target_lock_source_writer_session_id": None,
        "davis_target_lock_source_lock_frame_id": None,
        "davis_target_lock_source_attempt_id": None,
        "davis_target_lock_source_bottle_target_id": None,
        "davis_target_lock_provenance_valid": False,
        "davis_aps_target_frame_id": context.get("source_frame_id"),
        "davis_aps_target_age_ms": round(age_sec * 1000.0, 2),
        "davis_target_mapping_mode": "davis_aps_native_target_roi",
        "davis_target_roi_priority_selected": "davis_aps_target_roi",
        "davis_target_alignment_status": "DAVIS_NATIVE_APS",
        "davis_target_mapping_physically_validated": False,
        "davis_target_mapping_limitations": list(context.get("mapping_limitations") or []),
        "davis_target_calibration_reprojection_error_px": None,
    }
    return context, status


def bind_aps_target_context_to_realsense_lock(
    aps_context,
    aps_status,
    realsense_context,
    realsense_status,
):
    """Bind APS-local motion to a complete current/frozen lock identity only."""
    context = _copy_davis_target_context(aps_context)
    status = dict(aps_status or {})
    context["aps_target_confirmed"] = bool(
        context.get("aps_target_confirmed") is True
        or (
            status.get("davis_target_context_valid") is True
            and status.get("davis_target_context_mode") == "davis_aps_target"
        )
    )
    lock_context = realsense_context or {}
    lock_status = realsense_status or {}
    allowed_modes = {"live_realsense_mapping", "post_grasp_locked_mapping"}
    identity_keys = (
        "source_writer_session_id",
        "source_lock_frame_id",
        "source_attempt_id",
        "source_bottle_target_id",
    )
    if (
        lock_status.get("davis_target_context_valid") is not True
        or str(lock_status.get("davis_target_context_mode") or "") not in allowed_modes
        or any(lock_context.get(key) is None for key in identity_keys)
    ):
        status["davis_target_lock_provenance_valid"] = False
        return context, status
    for key in identity_keys:
        context[key] = lock_context.get(key)
    context["target_lock_provenance_valid"] = True
    context["source_lock_provenance_frame_id"] = lock_context.get("source_frame_id")
    context.setdefault("mapping_limitations", []).append(
        "The APS ROI is DAVIS-native; lock identity is inherited from the current/frozen RealSense lock and is not a cross-camera identity measurement."
    )
    status.update({
        "davis_target_lock_active": lock_status.get("davis_target_lock_active") is True,
        "davis_target_lock_age_ms": lock_status.get("davis_target_lock_age_ms"),
        "davis_target_lock_source_frame_id": lock_context.get("source_frame_id"),
        "davis_target_lock_source_writer_session_id": context.get("source_writer_session_id"),
        "davis_target_lock_source_lock_frame_id": context.get("source_lock_frame_id"),
        "davis_target_lock_source_attempt_id": context.get("source_attempt_id"),
        "davis_target_lock_source_bottle_target_id": context.get("source_bottle_target_id"),
        "davis_target_lock_provenance_valid": True,
        "davis_target_mapping_limitations": list(context.get("mapping_limitations") or []),
    })
    return context, status


def davis_realsense_mapping_context(realtime, status_age_sec):
    """Require a fresh, currently tracked target before object-local mapping."""
    realtime = realtime or {}
    try:
        age_sec = float(status_age_sec)
    except (TypeError, ValueError):
        return False, "realsense_status_age_unknown"
    if not np.isfinite(age_sec) or age_sec > DAVIS_REALSENSE_CONTEXT_MAX_AGE_SEC:
        return False, "realsense_status_stale"
    target_lock = realtime.get("target_lock") or {}
    tracked = target_lock.get("current_tracked_target") or {}
    band_relock_required = realtime.get("grasp_band_relock_required")
    if band_relock_required is None:
        band_relock_required = target_lock.get("grasp_band_relock_required")
    if band_relock_required is True:
        return False, "locked_grasp_band_relock_required"
    band_stale = realtime.get("grasp_band_stale")
    if band_stale is None:
        band_stale = target_lock.get("grasp_band_stale")
    if band_stale is True:
        return False, "locked_grasp_band_stale"
    band_locked = realtime.get("grasp_band_locked")
    if band_locked is None:
        band_locked = target_lock.get("grasp_band_locked")
    band_current = realtime.get("grasp_band_current")
    if band_current is None:
        band_current = target_lock.get("grasp_band_current")
    if band_locked is True and band_current is False:
        return False, "locked_grasp_band_not_current"
    band_projected = realtime.get("grasp_band_projected")
    if band_projected is None:
        band_projected = target_lock.get("grasp_band_projected")
    if band_locked is True and band_projected is False:
        return False, "locked_grasp_band_not_projected"
    target_status = str(realtime.get("target_status") or target_lock.get("target_status") or "").upper()
    if target_status not in {"LOCKED", "TRACKING"}:
        return False, f"target_not_current:{target_status or 'UNKNOWN'}"
    identity_tracking = realtime.get("identity_tracking_ok")
    if identity_tracking is None:
        identity_tracking = tracked.get("identity_tracking_ok")
    if identity_tracking is not True:
        return False, "target_identity_tracking_invalid"
    identity_measurement_current = realtime.get("identity_measurement_current")
    if identity_measurement_current is None:
        identity_measurement_current = tracked.get("identity_measurement_current")
    if identity_measurement_current is not True:
        return False, "target_identity_measurement_not_current"
    bottle_geometry_tracking_ok = realtime.get("bottle_geometry_tracking_ok")
    if bottle_geometry_tracking_ok is None:
        bottle_geometry_tracking_ok = tracked.get("bottle_geometry_tracking_ok")
    if bottle_geometry_tracking_ok is not True:
        return False, "target_bottle_geometry_tracking_invalid"
    if _first_bbox_xywh(realtime) is None:
        return False, "current_target_bbox_unavailable"
    return True, "fresh_current_tracked_target"


def _davis_target_identity(realtime):
    """Read one target identity from full or compact RealSense status."""
    realtime = realtime or {}
    target_lock = realtime.get("target_lock") or {}
    locked_target = target_lock.get("locked_target") or {}
    return {
        "source_writer_session_id": realtime.get("writer_session_id"),
        "source_lock_frame_id": next(
            (
                value
                for value in (
                    target_lock.get("lock_frame_id"),
                    locked_target.get("lock_frame_id"),
                    realtime.get("lock_frame_id"),
                )
                if value is not None
            ),
            None,
        ),
        "source_attempt_id": next(
            (
                value
                for value in (
                    target_lock.get("attempt_id"),
                    locked_target.get("attempt_id"),
                    realtime.get("lock_attempt_id"),
                    realtime.get("attempt_id"),
                )
                if value is not None
            ),
            None,
        ),
        "source_bottle_target_id": next(
            (
                value
                for value in (
                    target_lock.get("bottle_target_id"),
                    locked_target.get("bottle_target_id"),
                    realtime.get("locked_bottle_target_id"),
                )
                if value is not None
            ),
            None,
        ),
    }


def _read_davis_homography_payload():
    try:
        stat = DAVIS_HOMOGRAPHY_PATH.stat()
    except OSError:
        return None
    key = (str(DAVIS_HOMOGRAPHY_PATH), stat.st_mtime_ns, stat.st_size)
    with davis_homography_cache_lock:
        if davis_homography_cache.get("key") == key:
            return davis_homography_cache.get("payload")
    payload = read_json(DAVIS_HOMOGRAPHY_PATH, {})
    with davis_homography_cache_lock:
        davis_homography_cache["key"] = key
        davis_homography_cache["payload"] = payload
    return payload


def _realsense_to_davis_transform(realtime):
    """Return the best available RealSense-pixel -> DAVIS-pixel transform.

    A measured 2D homography is preferred only when its recorded reprojection
    error passes the existing calibration gate. Otherwise this returns the
    existing fixed linear side-by-side approximation.
    """
    realtime = realtime or {}
    shape = realtime.get("live_rgb_shape") or realtime.get("overlay_image_shape") or []
    if len(shape) >= 2 and float(shape[0]) > 0 and float(shape[1]) > 0:
        rs_h, rs_w = float(shape[0]), float(shape[1])
    else:
        rs_h, rs_w = _realsense_frame_shape_hw()
    if rs_w <= 0 or rs_h <= 0:
        return None, {
            "mapping_mode": "unavailable",
            "alignment_status": "INVALID_REALSENSE_SHAPE",
            "reason": "bad_realsense_shape",
            "physically_validated": False,
        }

    homography_reject_reason = "homography_not_configured"
    payload = _read_davis_homography_payload()
    if payload is not None:
        h_value = payload.get("homography") or payload.get("davis_to_realsense_homography")
        reprojection_error = first_number(payload.get("reprojection_error_px"))
        try:
            transform_direction = str(payload.get("transform_direction") or "davis_to_realsense").lower()
            if transform_direction != "davis_to_realsense":
                raise ValueError(f"unsupported_transform_direction:{transform_direction}")
            if reprojection_error is None:
                raise ValueError("missing_reprojection_error")
            if float(reprojection_error) > 8.0:
                raise ValueError("reprojection_error_above_8px_gate")
            davis_size_value = payload.get("davis_frame_size")
            realsense_size_value = payload.get("realsense_frame_size")
            if not isinstance(davis_size_value, list) or not isinstance(realsense_size_value, list):
                raise ValueError("missing_frame_size_metadata")
            davis_size = [float(v) for v in davis_size_value]
            realsense_size = [float(v) for v in realsense_size_value]
            if len(davis_size) != 2 or len(realsense_size) != 2:
                raise ValueError("missing_frame_size_metadata")
            if not np.allclose(davis_size, [DAVIS_ROI_FRAME_WIDTH, DAVIS_ROI_FRAME_HEIGHT], atol=0.5):
                raise ValueError("davis_frame_size_mismatch")
            if not np.allclose(realsense_size, [rs_w, rs_h], atol=0.5):
                raise ValueError("realsense_frame_size_mismatch")
            davis_to_realsense = np.asarray(h_value, dtype=float).reshape(3, 3)
            if not np.isfinite(davis_to_realsense).all():
                raise ValueError("non-finite homography")
            realsense_to_davis = np.linalg.inv(davis_to_realsense)
            if not np.isfinite(realsense_to_davis).all():
                raise ValueError("non-finite inverse homography")
            return realsense_to_davis, {
                "mapping_mode": "inverse_2d_homography",
                "alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
                "roi_source": "realsense_projected",
                "boundary_source": "realsense_projected_boundary",
                "reprojection_error_px": round(float(reprojection_error), 3),
                "calibration_path": str(DAVIS_HOMOGRAPHY_PATH),
                "davis_frame_size": [int(round(v)) for v in davis_size],
                "realsense_frame_size": [int(round(v)) for v in realsense_size],
                "reason": (
                    "inverse of measured DAVIS-to-RealSense 2D homography; "
                    f"reference-plane reprojection error {float(reprojection_error):.3f}px"
                ),
                "physically_validated": False,
                "limitations": [
                    "2D homography is valid near its calibration plane and camera pose only.",
                    "No physical grasp, contact, slip-direction, or force-control validation is implied.",
                ],
            }
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            homography_reject_reason = f"homography_rejected:{exc}"

    base_sx = DAVIS_ROI_FRAME_WIDTH / rs_w
    base_sy = DAVIS_ROI_FRAME_HEIGHT / rs_h
    transform = np.asarray(
        [
            [base_sx * DAVIS_ROI_SCALE_X, 0.0, DAVIS_ROI_OFFSET_X],
            [0.0, base_sy * DAVIS_ROI_SCALE_Y, DAVIS_ROI_OFFSET_Y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transform, {
        "mapping_mode": "fixed_linear_side_by_side",
        "alignment_status": "APPROXIMATE_UNCALIBRATED",
        "roi_source": "realsense_bbox_mapped",
        "boundary_source": "realsense_contour_mapped",
        "reprojection_error_px": None,
        "calibration_path": str(DAVIS_HOMOGRAPHY_PATH),
        "reason": (
            f"approximate (uncalibrated) map in {int(rs_w)}x{int(rs_h)} using base scale "
            f"{base_sx:.3f},{base_sy:.3f}; config scale {DAVIS_ROI_SCALE_X:.3f},{DAVIS_ROI_SCALE_Y:.3f}; "
            f"offset {DAVIS_ROI_OFFSET_X:.1f},{DAVIS_ROI_OFFSET_Y:.1f}; "
            f"{homography_reject_reason}"
        ),
        "physically_validated": False,
        "limitations": [
            "Fixed scale/offset is not RealSense-to-DAVIS camera calibration.",
            "Boundary and grasp-band alignment are approximate.",
        ],
    }


def _apply_realsense_to_davis_transform(points, transform):
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([arr, np.ones(arr.shape[0], dtype=float)]) @ np.asarray(transform, dtype=float).T
    denominator = homogeneous[:, 2]
    if np.any(~np.isfinite(homogeneous)) or np.any(np.abs(denominator) < 1e-9):
        return None
    mapped = homogeneous[:, :2] / denominator[:, None]
    mapped[:, 0] = np.clip(mapped[:, 0], 0.0, DAVIS_ROI_FRAME_WIDTH - 1.0)
    mapped[:, 1] = np.clip(mapped[:, 1], 0.0, DAVIS_ROI_FRAME_HEIGHT - 1.0)
    return mapped


def davis_roi_on_realsense_overlay(report, realtime):
    """Map an object-specific DAVIS ROI back onto RGB only with valid calibration."""
    report = report or {}
    source = str(report.get("roi_source") or "full_frame")
    roi = report.get("roi_xywh") or report.get("davis_roi_xywh")
    unavailable = {
        "valid": False,
        "source": source,
        "reason": "calibrated_object_specific_davis_roi_unavailable",
        "physically_validated": False,
    }
    if source not in {"aps_target", "realsense_contact_projected", "realsense_projected"}:
        unavailable["reason"] = "davis_roi_not_object_specific"
        return unavailable
    if report.get("davis_target_context_valid") is False:
        unavailable["reason"] = "davis_target_context_not_current"
        return unavailable
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        unavailable["reason"] = "davis_roi_missing"
        return unavailable
    realsense_to_davis, metadata = _realsense_to_davis_transform(realtime)
    if (
        realsense_to_davis is None
        or metadata.get("mapping_mode") != "inverse_2d_homography"
        or metadata.get("alignment_status") != "CALIBRATED_2D_REFERENCE_PLANE"
    ):
        unavailable["reason"] = metadata.get("reason") or "davis_realsense_calibration_unavailable"
        return unavailable
    try:
        x, y, width, height = [float(value) for value in roi]
        if not np.all(np.isfinite([x, y, width, height])) or width <= 0.0 or height <= 0.0:
            raise ValueError("invalid ROI")
        davis_points = np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=float,
        )
        davis_to_realsense = np.linalg.inv(np.asarray(realsense_to_davis, dtype=float))
        homogeneous = np.column_stack([davis_points, np.ones(4, dtype=float)]) @ davis_to_realsense.T
        denominator = homogeneous[:, 2]
        if np.any(~np.isfinite(homogeneous)) or np.any(np.abs(denominator) < 1e-9):
            raise ValueError("non-finite projected ROI")
        projected = homogeneous[:, :2] / denominator[:, None]
    except (TypeError, ValueError, np.linalg.LinAlgError):
        unavailable["reason"] = "davis_roi_reverse_projection_failed"
        return unavailable
    shape = (realtime or {}).get("live_rgb_shape") or (realtime or {}).get("overlay_image_shape") or []
    if len(shape) >= 2:
        frame_height, frame_width = float(shape[0]), float(shape[1])
    else:
        frame_height, frame_width = _realsense_frame_shape_hw()
    projected[:, 0] = np.clip(projected[:, 0], 0.0, max(frame_width - 1.0, 0.0))
    projected[:, 1] = np.clip(projected[:, 1], 0.0, max(frame_height - 1.0, 0.0))
    return {
        "valid": True,
        "source": source,
        "polygon_px": np.round(projected, 2).tolist(),
        "realsense_frame_size": [int(round(frame_width)), int(round(frame_height))],
        "alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
        "reprojection_error_px": metadata.get("reprojection_error_px"),
        "warning": "Reference-plane 2D overlay only; physical contact alignment is not validated.",
        "physically_validated": False,
    }


def _map_realsense_bbox_to_davis_roi_with_metadata(realtime, *, transform=None, metadata=None):
    bbox = _first_bbox_xywh(realtime)
    if not bbox:
        return None, {"reason": "no_realsense_bbox", "mapping_mode": "unavailable", "physically_validated": False}
    if transform is None or metadata is None:
        transform, metadata = _realsense_to_davis_transform(realtime)
    if transform is None:
        return None, metadata
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None, {**metadata, "reason": "realsense_bbox_is_not_numeric"}
    if not np.isfinite([x, y, w, h]).all() or w <= 0.0 or h <= 0.0:
        return None, {**metadata, "reason": "realsense_bbox_is_nonfinite_or_degenerate"}
    corners = _apply_realsense_to_davis_transform(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        transform,
    )
    if corners is None:
        return None, {**metadata, "reason": "mapped_bbox_has_invalid_homogeneous_coordinates"}
    x1 = max(0.0, min(float(np.min(corners[:, 0])) - DAVIS_ROI_PADDING, DAVIS_ROI_FRAME_WIDTH - 1.0))
    y1 = max(0.0, min(float(np.min(corners[:, 1])) - DAVIS_ROI_PADDING, DAVIS_ROI_FRAME_HEIGHT - 1.0))
    x2 = max(x1 + 1.0, min(float(np.max(corners[:, 0])) + DAVIS_ROI_PADDING, DAVIS_ROI_FRAME_WIDTH))
    y2 = max(y1 + 1.0, min(float(np.max(corners[:, 1])) + DAVIS_ROI_PADDING, DAVIS_ROI_FRAME_HEIGHT))
    roi = [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]
    return roi, {
        **metadata,
        "reason": f"{metadata.get('reason')}; bbox {bbox}; padding {DAVIS_ROI_PADDING:.1f}px",
    }


def map_realsense_bbox_to_davis_roi(realtime):
    roi, metadata = _map_realsense_bbox_to_davis_roi_with_metadata(realtime)
    return roi, metadata.get("reason")


def _map_realsense_points_to_davis(realtime, points, *, transform=None):
    if points is None:
        return None
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    if transform is None:
        transform, _metadata = _realsense_to_davis_transform(realtime)
    return None if transform is None else _apply_realsense_to_davis_transform(arr, transform)


def map_realsense_boundary_to_davis(realtime, *, transform=None, metadata=None):
    """Map the existing bottle outline with the same ROI transform."""
    geometry = (realtime or {}).get("bottle_geometry") or {}
    points = geometry.get("contour") or geometry.get("oriented_box_points") or geometry.get("oriented_bbox")
    if points is None or len(points) == 0:
        return None, "no_realsense_boundary"
    mapped = _map_realsense_points_to_davis(realtime, points, transform=transform)
    if mapped is None:
        return None, "bad_realsense_boundary"
    if metadata is None:
        _transform, metadata = _realsense_to_davis_transform(realtime)
    return mapped, f"{metadata.get('alignment_status')}: mapped RealSense bottle boundary; {metadata.get('reason')}"


def map_realsense_grasp_band_to_davis(realtime, *, transform=None, metadata=None):
    """Map the current gripper/contact band for read-only DAVIS diagnostics."""
    target_clamp = (realtime or {}).get("target_clamp") or ((realtime or {}).get("target_lock") or {}).get("target_clamp") or {}
    points = target_clamp.get("clamp_band_polygon") or (realtime or {}).get("magenta_band_polygon")
    if points is None or len(points) == 0:
        return None, "no_realsense_grasp_band"
    mapped = _map_realsense_points_to_davis(realtime, points, transform=transform)
    if mapped is None:
        return None, "bad_realsense_grasp_band"
    if metadata is None:
        _transform, metadata = _realsense_to_davis_transform(realtime)
    return mapped, f"{metadata.get('alignment_status')}: mapped RealSense grasp band; {metadata.get('reason')}"


def build_davis_target_context(realtime, status_age_sec, *, now=None):
    """Build one fresh, object-specific target handoff without opening a camera."""
    current, context_reason = davis_realsense_mapping_context(realtime, status_age_sec)
    if not current:
        return None, context_reason
    transform, mapping = _realsense_to_davis_transform(realtime)
    if transform is None:
        return None, mapping.get("reason") or "realsense_to_davis_transform_unavailable"
    roi, mapping = _map_realsense_bbox_to_davis_roi_with_metadata(
        realtime,
        transform=transform,
        metadata=mapping,
    )
    if roi is None:
        return None, mapping.get("reason") or "realsense_to_davis_mapping_failed"
    boundary, boundary_reason = map_realsense_boundary_to_davis(
        realtime,
        transform=transform,
        metadata=mapping,
    )
    grasp_band, grasp_band_reason = map_realsense_grasp_band_to_davis(
        realtime,
        transform=transform,
        metadata=mapping,
    )
    roi_source = mapping.get("roi_source") or "realsense_bbox_mapped"
    roi_priority = "calibrated_realsense_bottle_roi" if roi_source == "realsense_projected" else "uncalibrated_realsense_diagnostic_roi"
    if grasp_band is not None:
        calibrated = mapping.get("alignment_status") == "CALIBRATED_2D_REFERENCE_PLANE"
        contact_bbox = common_polygon_bbox_xyxy(
            grasp_band,
            image_shape=(DAVIS_ROI_FRAME_HEIGHT, DAVIS_ROI_FRAME_WIDTH),
            expand_px=DAVIS_ROI_PADDING if calibrated else 2.0 * DAVIS_ROI_PADDING,
        )
        contact_roi = _xyxy_to_xywh(contact_bbox)
        if contact_roi is not None:
            roi = [int(round(value)) for value in contact_roi]
            if calibrated:
                roi_source = "realsense_contact_projected"
                roi_priority = "calibrated_realsense_contact_roi"
            else:
                roi_source = "realsense_contact_band_mapped"
                roi_priority = "uncalibrated_realsense_contact_motion_diagnostic_roi"
    target_identity = _davis_target_identity(realtime)
    limitations = list(mapping.get("limitations") or [])
    if roi_source == "realsense_contact_band_mapped":
        limitations.append(
            "The focused contact-band ROI is an uncalibrated movement diagnostic and cannot attribute events to the bottle."
        )
    return {
        "roi_xywh": list(roi),
        "boundary_points": None if boundary is None else np.asarray(boundary, dtype=float).tolist(),
        "contact_band_points": None if grasp_band is None else np.asarray(grasp_band, dtype=float).tolist(),
        "roi_source": roi_source,
        "roi_priority_selected": roi_priority,
        "boundary_source": mapping.get("boundary_source") or "realsense_contour_mapped",
        "mapping_mode": mapping.get("mapping_mode"),
        "alignment_status": mapping.get("alignment_status"),
        "mapping_reason": mapping.get("reason"),
        "boundary_reason": boundary_reason,
        "grasp_band_reason": grasp_band_reason,
        "mapping_physically_validated": False,
        "mapping_limitations": limitations,
        "calibration_reprojection_error_px": mapping.get("reprojection_error_px"),
        "projection_metadata": {
            "mapping_mode": mapping.get("mapping_mode"),
            "alignment_status": mapping.get("alignment_status"),
            "reprojection_error_px": mapping.get("reprojection_error_px"),
        },
        "source_frame_id": (realtime or {}).get("frame_id"),
        **target_identity,
        "source_target_status": (realtime or {}).get("target_status") or ((realtime or {}).get("target_lock") or {}).get("target_status"),
        "source_depth_mm": first_number(
            (realtime or {}).get("bottle_depth_median_mm"),
            (realtime or {}).get("depth_mm"),
            ((realtime or {}).get("bottle_geometry") or {}).get("depth_median_mm"),
        ),
        "captured_monotonic": time.monotonic() if now is None else float(now),
    }, context_reason


def _copy_davis_target_context(context):
    if context is None:
        return None
    copied = dict(context)
    for key in ("roi_xywh", "boundary_points", "contact_band_points", "mapping_limitations"):
        value = context.get(key)
        copied[key] = None if value is None else [list(item) if isinstance(item, (list, tuple)) else item for item in value]
    return copied


class DavisTargetHandoff:
    """Latch the last fresh RealSense mapping across a declared grasp.

    The latch is intentionally fixed: it supports local DAVIS micro-motion
    around the lock-time target, not large object tracking or camera stitching.
    Returning to PRE_CLOSE clears the latch so a new test needs fresh evidence.
    """

    def __init__(
        self,
        max_capture_age_sec=DAVIS_REALSENSE_CONTEXT_MAX_AGE_SEC,
        preclose_diagnostic_hold_sec=DAVIS_PRE_CLOSE_DIAGNOSTIC_HOLD_SEC,
    ):
        self.max_capture_age_sec = float(max_capture_age_sec)
        self.preclose_diagnostic_hold_sec = float(preclose_diagnostic_hold_sec)
        self.last_live_context = None
        self.post_grasp_context = None
        self.previous_stage = "PRE_CLOSE"
        self.locked_at_monotonic = None

    def reset(self):
        self.last_live_context = None
        self.post_grasp_context = None
        self.previous_stage = "PRE_CLOSE"
        self.locked_at_monotonic = None

    def update(self, realtime, status_age_sec, stage, *, now=None):
        now = time.monotonic() if now is None else float(now)
        stage = str(stage or "PRE_CLOSE").upper()

        if stage == "PRE_CLOSE":
            live_context, live_reason = build_davis_target_context(realtime, status_age_sec, now=now)
            if self.previous_stage != "PRE_CLOSE":
                self.last_live_context = None
            self.post_grasp_context = None
            self.locked_at_monotonic = None
            if live_context is not None:
                self.last_live_context = _copy_davis_target_context(live_context)
            else:
                target_lock = (realtime or {}).get("target_lock") or {}
                target_status = str(
                    (realtime or {}).get("target_status") or target_lock.get("target_status") or ""
                ).upper()
                current_identity = _davis_target_identity(realtime)
                previous_identity = self.last_live_context or {}
                identity_changed = any(
                    previous_identity.get(key) is not None
                    and value is not None
                    and value != previous_identity.get(key)
                    for key, value in current_identity.items()
                )
                if target_status == "SEARCHING" or identity_changed:
                    self.last_live_context = None
                candidate = self.last_live_context
                if candidate is not None:
                    captured = candidate.get("captured_monotonic")
                    capture_age = max(0.0, now - float(now if captured is None else captured))
                    if capture_age <= self.preclose_diagnostic_hold_sec:
                        held = _copy_davis_target_context(candidate)
                        held["roi_source"] = "held_preclose_motion_diagnostic"
                        held["roi_priority_selected"] = "held_pre_close_contact_motion_diagnostic_roi"
                        held["mapping_mode"] = "held_preclose_motion_diagnostic"
                        held["alignment_status"] = "HELD_DIAGNOSTIC_NOT_CURRENT"
                        held["mapping_reason"] = (
                            f"held_last_pre_close_motion_roi_age_{capture_age:.3f}s; {live_reason}"
                        )
                        held["projection_metadata"] = {
                            "mapping_mode": "held_preclose_motion_diagnostic",
                            "alignment_status": "HELD_DIAGNOSTIC_NOT_CURRENT",
                            "reprojection_error_px": candidate.get("calibration_reprojection_error_px"),
                        }
                        held.setdefault("mapping_limitations", []).append(
                            "Held close-range ROI is diagnostic-only; the RealSense target is not current and object motion remains invalid."
                        )
                        self.previous_stage = stage
                        status = self._status(
                            held,
                            "pre_close_held_target_diagnostic",
                            held["mapping_reason"],
                            now,
                        )
                        return held, status
                    self.last_live_context = None
            self.previous_stage = stage
            status = self._status(live_context, "live_realsense_mapping" if live_context else "none", live_reason, now)
            return live_context, status

        _context_current, raw_context_reason = davis_realsense_mapping_context(realtime, status_age_sec)
        live_reason = f"post_grasp_live_mapping_not_used:{raw_context_reason}"
        if raw_context_reason in {
            "locked_grasp_band_relock_required",
            "locked_grasp_band_stale",
            "locked_grasp_band_not_current",
            "locked_grasp_band_not_projected",
        }:
            self.last_live_context = None
            self.post_grasp_context = None
            self.locked_at_monotonic = None
            self.previous_stage = stage
            reason = f"post_grasp_target_lock_invalidated:{raw_context_reason}"
            return None, self._status(None, "none", reason, now)
        try:
            status_fresh = np.isfinite(float(status_age_sec)) and float(status_age_sec) <= self.max_capture_age_sec
        except (TypeError, ValueError):
            status_fresh = False
        if not status_fresh:
            self.post_grasp_context = None
            self.locked_at_monotonic = None
            self.previous_stage = stage
            reason = f"post_grasp_target_lock_invalidated:{live_reason}"
            return None, self._status(None, "none", reason, now)

        entering_post_grasp = self.previous_stage == "PRE_CLOSE"
        if self.post_grasp_context is None and entering_post_grasp:
            candidate = self.last_live_context
            lock_reason = "locked_from_last_pre_close_mapping"
            if candidate is not None:
                captured = candidate.get("captured_monotonic")
                capture_age = max(0.0, now - float(now if captured is None else captured))
                if capture_age <= self.max_capture_age_sec:
                    lock_reason = f"locked_from_last_pre_close_mapping_age_{capture_age:.3f}s"
                else:
                    candidate = None
            if candidate is not None:
                required_identity = (
                    "source_writer_session_id",
                    "source_lock_frame_id",
                    "source_attempt_id",
                    "source_bottle_target_id",
                )
                missing_identity = [
                    key for key in required_identity if candidate.get(key) is None
                ]
                if missing_identity:
                    candidate = None
                    live_reason = (
                        "post_grasp_locked_mapping_identity_missing:"
                        f"{','.join(missing_identity)}; {live_reason}"
                    )
            if candidate is not None:
                self.post_grasp_context = _copy_davis_target_context(candidate)
                self.locked_at_monotonic = now
                self.post_grasp_context["post_grasp_lock_reason"] = lock_reason
                self.post_grasp_context.setdefault("mapping_limitations", []).append(
                    "Post-grasp ROI is frozen at lock time for local micro-motion; it does not track large translation."
                )

        self.previous_stage = stage
        if self.post_grasp_context is None:
            reason = f"post_grasp_target_lock_unavailable:{live_reason}"
            return None, self._status(None, "none", reason, now)
        current_identity = _davis_target_identity(realtime)
        mismatches = [
            key
            for key, current_value in current_identity.items()
            if self.post_grasp_context.get(key) is not None
            and current_value != self.post_grasp_context.get(key)
        ]
        if mismatches:
            self.post_grasp_context = None
            self.locked_at_monotonic = None
            reason = f"post_grasp_target_lock_invalidated:identity_mismatch:{','.join(mismatches)}"
            return None, self._status(None, "none", reason, now)
        reason = (
            f"{self.post_grasp_context.get('post_grasp_lock_reason')}; "
            "frozen lock-time ROI for local DAVIS micro-motion; "
            f"current RealSense context: {live_reason}"
        )
        locked = _copy_davis_target_context(self.post_grasp_context)
        return locked, self._status(locked, "post_grasp_locked_mapping", reason, now)

    def _status(self, context, mode, reason, now):
        lock_age_ms = (
            None
            if self.locked_at_monotonic is None
            else round(max(0.0, now - self.locked_at_monotonic) * 1000.0, 2)
        )
        provenance_keys = (
            "source_writer_session_id",
            "source_lock_frame_id",
            "source_attempt_id",
            "source_bottle_target_id",
        )
        provenance_valid = bool(
            context is not None
            and all(context.get(key) is not None for key in provenance_keys)
        )
        return {
            "davis_target_context_mode": mode,
            "davis_target_context_valid": context is not None,
            "davis_target_context_reason": reason,
            "davis_target_lock_active": mode == "post_grasp_locked_mapping",
            "davis_target_lock_age_ms": lock_age_ms,
            "davis_target_lock_source_frame_id": None if context is None else context.get("source_frame_id"),
            "davis_target_lock_source_writer_session_id": (
                None if context is None else context.get("source_writer_session_id")
            ),
            "davis_target_lock_source_lock_frame_id": (
                None if context is None else context.get("source_lock_frame_id")
            ),
            "davis_target_lock_source_attempt_id": (
                None if context is None else context.get("source_attempt_id")
            ),
            "davis_target_lock_source_bottle_target_id": (
                None if context is None else context.get("source_bottle_target_id")
            ),
            "davis_target_lock_provenance_valid": provenance_valid,
            "davis_target_mapping_mode": None if context is None else context.get("mapping_mode"),
            "davis_target_roi_priority_selected": None if context is None else context.get("roi_priority_selected"),
            "davis_target_alignment_status": None if context is None else context.get("alignment_status"),
            "davis_target_mapping_physically_validated": False,
            "davis_target_mapping_limitations": [] if context is None else list(context.get("mapping_limitations") or []),
            "davis_target_calibration_reprojection_error_px": (
                None if context is None else context.get("calibration_reprojection_error_px")
            ),
        }


def draw_davis_motion_hull(report, base_image=None):
    """Clean operator-facing DAVIS view with honest motion attribution."""
    width, height = DAVIS_ROI_FRAME_WIDTH, DAVIS_ROI_FRAME_HEIGHT
    if base_image is None:
        img = np.full((height, width, 3), (15, 23, 42), dtype=np.uint8)
    else:
        img = np.asarray(base_image, dtype=np.uint8)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        img = np.ascontiguousarray(img.copy())
    roi = report.get("davis_roi_xywh") or report.get("roi_xywh") or [0, 0, width, height]
    try:
        x, y, w, h = [int(round(float(v))) for v in roi]
    except Exception:
        x, y, w, h = 0, 0, width, height
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    metrics = report.get("lightweight_event_metrics") or report.get("event_window_metrics") or {}
    raw_class = str(report.get("raw_motion_class") or report.get("motion_class") or "")
    safety_class = report.get("davis_safety_class") or report.get("motion_class") or "N/A"
    object_motion_valid = bool(report.get("object_motion_valid"))
    grasp_zone_motion_valid = bool(report.get("grasp_zone_motion_valid"))
    presentation_motion_valid = object_motion_valid or grasp_zone_motion_valid
    hull_valid = bool(report.get("motion_hull_valid"))
    activity_label = report.get("motion_activity_label") or ("OBJECT_MOTION" if object_motion_valid else "EVENT_ACTIVITY")
    disturbance = raw_class.upper() in {"LIGHTING_CHANGE", "CAMERA_SHAKE", "BACKGROUND_MOTION"} or str(safety_class).upper() in {"LIGHTING CHANGE", "CAMERA SHAKE", "BACKGROUND MOTION"}

    # A single quiet grasp-zone frame replaces the old double ROI boxes and
    # multi-line diagnostic wall. Detailed provenance remains in telemetry.
    roi_color = (235, 205, 70) if presentation_motion_valid else (190, 145, 60)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), roi_color, 1, cv2.LINE_AA)

    if disturbance:
        cv2.rectangle(img, (1, 1), (width - 2, height - 2), (55, 150, 240), 2)

    if presentation_motion_valid and hull_valid:
        hull = report.get("motion_hull_points") or []
        if len(hull) >= 3:
            pts = np.asarray(hull, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=True, color=(90, 255, 120), thickness=2, lineType=cv2.LINE_AA)
            cv2.fillPoly(img, [pts], color=(20, 80, 45))
            cv2.polylines(img, [pts], isClosed=True, color=(130, 255, 160), thickness=2, lineType=cv2.LINE_AA)
        cx, cy = metrics.get("centroid_x"), metrics.get("centroid_y")
        dx = report.get("dx_px")
        dy = report.get("dy_px")
        if cx is not None and cy is not None:
            cpt = (int(round(float(cx))), int(round(float(cy))))
            cv2.circle(img, cpt, 4, (255, 230, 80), -1, lineType=cv2.LINE_AA)
            if dx is not None and dy is not None and (abs(float(dx)) + abs(float(dy))) > 0.05:
                scale = 6.0
                tip = (int(round(cpt[0] + float(dx) * scale)), int(round(cpt[1] + float(dy) * scale)))
                cv2.arrowedLine(img, cpt, tip, (80, 200, 255), 2, cv2.LINE_AA, tipLength=0.3)
    stage = str(report.get("current_stage") or report.get("davis_motion_stage") or "PRE_CLOSE").upper()
    if stage == "PRE_CLOSE":
        badge = "DAVIS  |  ARMED"
        badge_color = (170, 185, 205)
    elif disturbance:
        badge = "DISTURBANCE FILTERED"
        badge_color = (70, 180, 255)
    elif presentation_motion_valid:
        prefix = "OBJECT" if object_motion_valid else "GRASP ZONE"
        badge = f"{prefix}  |  {str(safety_class).upper()}"
        badge_color = (115, 240, 155)
    else:
        badge = "POST-GRASP  |  ACQUIRING"
        badge_color = (80, 205, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(badge, font, 0.44, 1)
    cv2.rectangle(img, (6, 6), (min(width - 6, tw + 22), 30), (8, 14, 23), -1)
    cv2.rectangle(img, (6, 6), (10, 30), badge_color, -1)
    cv2.putText(img, badge, (16, 23), font, 0.44, (240, 247, 255), 1, cv2.LINE_AA)
    rate = float(metrics.get("event_rate", 0.0) or 0.0)
    footer = "POST-GRASP ONLY" if stage == "PRE_CLOSE" else f"{rate:.0f} events/s"
    (fw, _), _ = cv2.getTextSize(footer, font, 0.36, 1)
    cv2.rectangle(img, (6, height - 27), (fw + 18, height - 6), (8, 14, 23), -1)
    cv2.putText(img, footer, (12, height - 12), font, 0.36, (205, 220, 235), 1, cv2.LINE_AA)
    return img


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    return value


def _write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.jpg")
    ok = cv2.imwrite(str(tmp), image)
    if not ok:
        return False
    try:
        replace_path_retry(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _fit_image(image, width, height, bg=(12, 18, 32)):
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    if image is None:
        return canvas
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return canvas
    scale = min(width / float(w), height / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _label_panel(image, title, subtitle=None, color=(235, 245, 255)):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 36), (8, 13, 26), -1)
    cv2.putText(out, str(title), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, str(title), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    if subtitle:
        y = out.shape[0] - 12
        cv2.putText(out, str(subtitle), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(subtitle), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def _read_quality_background():
    candidates = [
        FINAL_DEMO_FILES["bottle_quality_overlay"],
        FINAL_DEMO_FILES["grasp_quality_weak_direction"],
        FINAL_DEMO_FILES["contact_quality_overlay"],
        FINAL_DEMO_FILES["grasp_hulls_overlay"],
        FINAL_DEMO_FILES["overlay"],
    ]
    for path in candidates:
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                return img, path
    return np.full((480, 640, 3), (18, 24, 38), dtype=np.uint8), None


def _davis_alignment_status(realtime=None):
    """Report the same transform gate used by the live DAVIS ROI mapper."""
    if realtime is None:
        realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
    _transform, metadata = _realsense_to_davis_transform(realtime)
    if metadata.get("mapping_mode") == "inverse_2d_homography":
        return {
            "davis_realsense_alignment_status": "calibrated_2d_reference_plane",
            "davis_realsense_alignment_mode": "homography",
            "davis_realsense_homography_path": str(DAVIS_HOMOGRAPHY_PATH),
            "davis_realsense_reprojection_error_px": metadata.get("reprojection_error_px"),
            "davis_realsense_alignment_physically_validated": False,
        }
    return {
        "davis_realsense_alignment_status": (
            "invalid_calibration" if DAVIS_HOMOGRAPHY_PATH.exists() else "uncalibrated"
        ),
        "davis_realsense_alignment_mode": "side_by_side",
        "davis_realsense_homography_path": str(DAVIS_HOMOGRAPHY_PATH),
        "davis_realsense_alignment_reason": metadata.get("reason"),
        "davis_realsense_alignment_physically_validated": False,
    }


def davis_motion_compensation(report, alignment):
    report = report or {}
    metrics = (
        report.get("lightweight_event_metrics")
        or report.get("event_window_metrics")
        or report.get("current_metrics")
        or {}
    )
    event_count = int(metrics.get("event_count") or 0)
    event_rate = float(metrics.get("event_rate") or report.get("event_rate") or 0.0)
    dx = float(metrics.get("centroid_shift_x") or 0.0)
    dy = float(metrics.get("centroid_shift_y") or 0.0)
    hull = report.get("motion_hull_points") or []
    active = str(report.get("event_stream_status") or "").lower() == "active"
    has_motion = active and (event_count > 0 or event_rate > 1.0)
    confidence = 0.0
    source = "none"
    reason = "davis_event_stream_unavailable"
    if active and has_motion:
        confidence = min(100.0, 25.0 + event_count / 12.0 + event_rate / 60.0 + min(len(hull), 12) * 2.0)
        source = "davis_event_centroid_motion_bbox" if hull else "davis_event_centroid"
        reason = "davis_motion_estimate_available"
    elif active:
        reason = "stream_active_no_motion_events"
    alignment_status = alignment.get("davis_realsense_alignment_status")
    adjusted_available = bool(has_motion and confidence >= 25.0)
    adjusted_valid = bool(
        adjusted_available
        and alignment_status in {"calibrated", "calibrated_2d_reference_plane"}
    )
    if not adjusted_available:
        adjusted_reason = reason
    elif not adjusted_valid:
        adjusted_reason = "davis_realsense_alignment_not_calibrated_side_by_side_only"
    else:
        adjusted_reason = "davis_motion_compensated_preview"
    return {
        "davis_motion_compensation_enabled": True,
        "davis_motion_dx_px": round(dx, 2),
        "davis_motion_dy_px": round(dy, 2),
        "davis_motion_rotation_deg": 0.0,
        "davis_motion_scale": 1.0,
        "davis_motion_confidence": round(confidence, 1),
        "davis_motion_source": source,
        "davis_adjusted_overlay_available": adjusted_available,
        "davis_adjusted_overlay_valid": adjusted_valid,
        "davis_adjusted_overlay_reason": adjusted_reason,
    }


def draw_davis_motion_overlay(report, compensation):
    img = draw_davis_motion_hull(report or {})
    metrics = (report or {}).get("lightweight_event_metrics") or (report or {}).get("event_window_metrics") or {}
    hull = (report or {}).get("motion_hull_points") or []
    if len(hull) >= 2:
        arr = np.asarray(hull, dtype=np.int32).reshape(-1, 2)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 60), 2, cv2.LINE_AA)
    cx, cy = metrics.get("centroid_x"), metrics.get("centroid_y")
    if cx is not None and cy is not None:
        c = (int(round(float(cx))), int(round(float(cy))))
        cv2.drawMarker(img, c, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)
        cv2.circle(img, c, 5, (0, 200, 255), 1, cv2.LINE_AA)
    label = f"DAVIS preview motion dx {compensation.get('davis_motion_dx_px', 0)} dy {compensation.get('davis_motion_dy_px', 0)} conf {compensation.get('davis_motion_confidence', 0)}"
    cv2.putText(img, label, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 255), 1, cv2.LINE_AA)
    return img


def draw_davis_adjusted_preview(base_img, compensation, alignment):
    out = base_img.copy()
    reason = compensation.get("davis_adjusted_overlay_reason") or "not_available"
    if compensation.get("davis_adjusted_overlay_valid"):
        dx = float(compensation.get("davis_motion_dx_px") or 0.0)
        dy = float(compensation.get("davis_motion_dy_px") or 0.0)
        matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        out = cv2.warpAffine(out, matrix, (out.shape[1], out.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(18, 24, 38))
        title = "DAVIS-adjusted overlay preview only"
        subtitle = f"visual compensation dx={dx:.1f}, dy={dy:.1f}; RealSense matrix unchanged"
        color = (120, 255, 180)
    else:
        out = np.full_like(out, (18, 24, 38))
        title = "DAVIS adjustment unavailable"
        subtitle = "Event stream not available" if reason == "davis_event_stream_unavailable" else reason
        color = (120, 200, 255)
    return _label_panel(out, title, subtitle, color=color)


def make_davis_confirmation_image(quality_img, davis_img, adjusted_img, matrix, compensation, alignment):
    panel_w, panel_h = 520, 390
    matrix_state = "STALE" if matrix.get("quality_label_stale") or matrix.get("matrix_stale_due_to_motion") else (matrix.get("quality_matrix_status") or "idle")
    quality_panel = _label_panel(
        _fit_image(quality_img, panel_w, panel_h),
        f"RealSense matrix: {str(matrix_state).upper()}",
        f"compute {matrix.get('quality_compute_id') or 'N/A'} | source of truth",
        color=(255, 230, 120),
    )
    davis_panel = _label_panel(
        _fit_image(davis_img, panel_w, panel_h),
        "DAVIS event motion",
        f"{compensation.get('davis_motion_source')} | confidence {compensation.get('davis_motion_confidence')}",
        color=(120, 220, 255),
    )
    adjusted_panel = _label_panel(
        _fit_image(adjusted_img, panel_w, panel_h),
        "DAVIS-adjusted quality preview",
        f"{alignment.get('davis_realsense_alignment_mode')} | {compensation.get('davis_adjusted_overlay_reason')}",
        color=(180, 255, 180) if compensation.get("davis_adjusted_overlay_valid") else (120, 200, 255),
    )
    return np.concatenate([quality_panel, davis_panel, adjusted_panel], axis=1)


def static_image_info(final_files, name, url, stale=False, stale_label="", version=None):
    exists = bool((final_files.get(name) or {}).get("exists", False))
    return {
        "exists": exists,
        "url": url,
        "stale": bool(exists and stale),
        "stale_label": stale_label if exists and stale else "",
        "version": version,
    }


def ensure_davis_confirmation_outputs(davis_report, realsense_summary, quality_matrix, final_files):
    now = time.time()
    perf_path = FINAL_DEMO_FILES["davis_performance_profile"]
    existing = read_json(perf_path, {})
    last_update = float(existing.get("last_confirmation_update_time") or 0.0)
    if now - last_update < DAVIS_CONFIRMATION_INTERVAL_SEC:
        return existing
    quality_img, quality_path = _read_quality_background()
    alignment = _davis_alignment_status()
    compensation = davis_motion_compensation(davis_report, alignment)
    davis_img = draw_davis_motion_overlay(davis_report or {}, compensation)
    adjusted_img = draw_davis_adjusted_preview(quality_img, compensation, alignment)
    confirmation_img = make_davis_confirmation_image(quality_img, davis_img, adjusted_img, quality_matrix or {}, compensation, alignment)
    _write_image(FINAL_DEMO_FILES["davis_motion_overlay"], davis_img)
    _write_image(FINAL_DEMO_FILES["davis_adjusted_quality_overlay"], adjusted_img)
    _write_image(FINAL_DEMO_FILES["davis_realsense_confirmation"], confirmation_img)
    matrix_stale = bool((quality_matrix or {}).get("quality_label_stale") or (quality_matrix or {}).get("matrix_stale_due_to_motion"))
    matrix_status = (quality_matrix or {}).get("quality_matrix_status") or "idle"
    overlay_status = "stale" if matrix_stale else ("current" if str(matrix_status).lower() in {"updated", "ok", "valid", "stale_needs_recompute"} else str(matrix_status))
    perf = {
        "last_confirmation_update_time": now,
        "quality_overlay_frame_id": (quality_matrix or {}).get("current_hull_frame_id") or (quality_matrix or {}).get("lock_frame_id"),
        "quality_overlay_timestamp": now,
        "quality_overlay_status": overlay_status,
        "last_valid_quality_overlay_path": str(quality_path) if quality_path else None,
        "current_quality_overlay_path": str(quality_path) if quality_path else None,
        "stale_overlay_reason": (
            ", ".join((quality_matrix or {}).get("quality_current_mismatch_reasons") or (quality_matrix or {}).get("matrix_stale_reasons") or [])
            if matrix_stale else ""
        ),
        "realsense_quality_status": "STALE" if matrix_stale else str(matrix_status).upper(),
        "matrix_valid_for_current_pose": not matrix_stale and str(matrix_status).lower() == "updated",
        "matrix_stale_reason": (
            ", ".join((quality_matrix or {}).get("quality_current_mismatch_reasons") or (quality_matrix or {}).get("matrix_stale_reasons") or [])
            if matrix_stale else ""
        ),
        "davis_adjusted_overlay_status": "VALID_PREVIEW" if compensation.get("davis_adjusted_overlay_valid") else "UNAVAILABLE",
        "davis_motion_status": (davis_report or {}).get("motion_class") or (davis_report or {}).get("davis_safety_class") or "N/A",
        "recompute_needed": matrix_stale,
        "davis_display_fps": (davis_report or {}).get("davis_display_fps"),
        "davis_capture_fps": (davis_report or {}).get("davis_capture_fps") or (davis_report or {}).get("davis_update_fps"),
        "davis_event_analysis_fps": (davis_report or {}).get("davis_event_analysis_fps") or (davis_report or {}).get("davis_update_fps"),
        "davis_event_rate": ((davis_report or {}).get("current_metrics") or {}).get("event_rate") or (davis_report or {}).get("event_rate"),
        "davis_aps_frame_age_ms": (davis_report or {}).get("davis_aps_frame_age_ms"),
        "davis_event_frame_age_ms": (davis_report or {}).get("davis_event_frame_age_ms"),
        "davis_gui_draw_time_ms": (davis_report or {}).get("davis_gui_draw_time_ms"),
        "davis_analysis_time_ms": (davis_report or {}).get("davis_analysis_time_ms") or (davis_report or {}).get("davis_processing_latency_ms"),
        "davis_bottleneck": (davis_report or {}).get("davis_bottleneck") or "unknown",
        **alignment,
        **compensation,
    }
    write_json_atomic(perf_path, _json_compatible(perf))
    return perf


def file_info(name):
    path = OUTPUT_DIR / name
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False, "url": None, "mtime": None}
    return {
        "exists": True,
        "url": f"/image/{name}?t={stat.st_mtime_ns}",
        "mtime": stat.st_mtime,
    }


def final_file_status(path):
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return {
            "exists": False,
            "state": "MISSING",
            "age_sec": None,
            "mtime": None,
            "size": None,
        }
    age = time.time() - stat.st_mtime
    if stat.st_size <= 0:
        state = "BLANK"
    else:
        state = "LIVE" if age <= STALE_FILE_THRESHOLD_SEC else "STALE"
    return {
        "exists": True,
        "state": state,
        "age_sec": round(age, 2),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def final_files_status():
    return {name: final_file_status(path) for name, path in FINAL_DEMO_FILES.items()}


def overlay_health_info(status, final_files):
    status = status or {}
    item = (final_files or {}).get("overlay") or {}
    exists = bool(item.get("exists"))
    state = item.get("state") or "MISSING"
    if not exists:
        health = "MISSING"
    elif state == "BLANK":
        health = "BLANK"
    elif state == "STALE":
        health = "STALE"
    else:
        health = status.get("overlay_health") or "OK"
    return {
        "overlay_health": health,
        "overlay_last_updated": status.get("overlay_last_updated"),
        "overlay_frame_id": status.get("overlay_frame_id") or status.get("frame_id"),
        "overlay_source_image": status.get("overlay_source_image") or status.get("overlay_source"),
        "overlay_current_path": status.get("overlay_current_path") or "output/latest_overlay_live.jpg",
        "overlay_last_valid_path": status.get("overlay_last_valid_path") or ("output/latest_overlay_live.jpg" if exists else None),
        "overlay_stale": health == "STALE",
        "overlay_error": status.get("overlay_error"),
        "dashboard_overlay_path": status.get("dashboard_overlay_path") or "/stream/overlay",
        "dashboard_overlay_cache_buster": status.get("dashboard_overlay_cache_buster") or item.get("mtime"),
        "dashboard_overlay_file_age_sec": item.get("age_sec"),
    }


def realsense_backend_live(max_age_sec=2.0):
    status_path = OUTPUT_DIR / "latest_realtime_status.json"
    if not status_path.exists():
        return False
    try:
        if time.time() - status_path.stat().st_mtime > max_age_sec:
            return False
        status = read_json(status_path, {})
        return bool(status.get("writer_session_id") and status.get("writer_mode") == "backend_only")
    except Exception:
        return False


def is_process_running(name):
    if name == "realsense" and realsense_backend_live():
        return True
    proc = processes.get(name)
    return bool(proc and proc.poll() is None)


def start_process(name):
    if name not in COMMANDS:
        return False, f"{name} is not part of the final combined demo."
    with process_lock:
        if name in process_starts_in_flight:
            return True, f"{COMMANDS[name]['title']} start is already in progress."
        process_starts_in_flight.add(name)
    try:
        return _start_process_once(name)
    finally:
        with process_lock:
            process_starts_in_flight.discard(name)


def _start_process_once(name):
    spec = COMMANDS[name]
    transport_prevalidated = False
    if is_process_running(name):
        return True, f"{spec['title']} is already running."
    if name == "realsense" and realsense_backend_live():
        return True, "RealSense backend writer is already running."
    if name == "realsense":
        ok, message = run_realsense_transport_recovery()
        if not ok:
            return False, message
        transport_prevalidated = True
    log_path = spec["log"]
    log_path.parent.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
    env.setdefault("ROS_DOMAIN_ID", "67")
    if transport_prevalidated:
        env["REALSENSE_TRANSPORT_PREVALIDATED"] = "1"
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            spec["cmd"],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    processes[name] = proc
    process_logs[name] = log_path
    return True, f"Started {spec['title']}."


def stop_process(name):
    proc = processes.get(name)
    if not proc or proc.poll() is not None:
        return True, f"{name} is not running."
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=4)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    return True, f"Stopped {name}."


def davis_loop(stage="PRE_CLOSE", sensitivity="sensitive", roi=None):
    global davis_analyzer, davis_last_error
    davis_last_error = None
    analyzer = None
    try:
        from grasp_project.davis.motion_gui import (
            DavisMotionAnalyzer,
            FIXED_CENTER_THIRD_BOUNDARY_SOURCE,
            FIXED_CENTER_THIRD_MODE,
            FIXED_CENTER_THIRD_OBJECT_EVIDENCE_ALIGNMENT,
            LIVE_HEATMAP_WINDOW_SEC,
            STABILITY_WINDOW_SEC,
            boundary_from_roi,
            configured_center_third_roi,
            event_buffer_age_sec,
            heatmap_rgb,
            recent_events,
        )

        davis_config = load_named_config("davis.yaml", default={})
        events_config = (davis_config.get("events") or {}) if isinstance(davis_config, dict) else {}
        analysis_roi_mode = str(events_config.get("analysis_roi_mode") or "configured_or_mapped").strip().lower()
        fixed_center_third_mode = analysis_roi_mode == FIXED_CENTER_THIRD_MODE
        configured_diagnostic_roi = roi
        if fixed_center_third_mode:
            configured_diagnostic_roi = configured_center_third_roi()
        elif configured_diagnostic_roi is None:
            configured_diagnostic_roi = events_config.get("diagnostic_roi_xywh")
        analyzer = DavisMotionAnalyzer(
            roi_xywh=configured_diagnostic_roi,
            stage=stage,
            sensitivity=sensitivity,
            analysis_roi_mode=analysis_roi_mode,
            recording_sink=runtime_recording,
        )
        simulated_force_controller = SlipForceController()
        target_handoff = DavisTargetHandoff()
        with davis_lock:
            davis_analyzer = analyzer
        next_save = 0.0
        next_heatmap = 0.0
        next_hull = 0.0
        next_analysis = 0.0
        display_period = 1.0 / max(DAVIS_TARGET_DISPLAY_FPS, 1.0)
        analysis_period = 1.0 / max(DAVIS_EVENT_ANALYSIS_FPS, 1.0)
        target_period = min(display_period, analysis_period)
        latest_report = analyzer.report if isinstance(analyzer.report, dict) else {}
        display_times = deque(maxlen=120)
        analysis_times = deque(maxlen=120)
        realtime_status_path = OUTPUT_DIR / "latest_realtime_status.json"
        while not davis_stop.is_set():
            loop_start = time.perf_counter()
            now = time.monotonic()
            realtime = read_json(realtime_status_path)
            try:
                realtime_status_age_sec = max(0.0, time.time() - realtime_status_path.stat().st_mtime)
            except OSError:
                realtime_status_age_sec = float("inf")
            current_stage = davis_stage_from_realsense(realtime)
            analyzer.set_stage(current_stage)
            context_current, context_reason = davis_realsense_mapping_context(realtime, realtime_status_age_sec)
            target_context, target_context_status = target_handoff.update(
                realtime,
                realtime_status_age_sec,
                current_stage,
                now=now,
            )
            aps_target_context, aps_target_context_status = current_davis_aps_target_context(
                analyzer,
                now=now,
            )
            if (
                aps_target_context is not None
                and current_stage != "PRE_CLOSE"
                and (
                    fixed_center_third_mode
                    or not calibrated_realsense_target_context(
                        target_context,
                        target_context_status,
                    )
                )
            ):
                target_context, target_context_status = bind_aps_target_context_to_realsense_lock(
                    aps_target_context,
                    aps_target_context_status,
                    target_context,
                    target_context_status,
                )
            mapped_roi = None if target_context is None else target_context.get("roi_xywh")
            mapped_reason = (
                target_context_status.get("davis_target_context_reason")
                if target_context is None or target_context_status.get("davis_target_lock_active")
                else target_context.get("mapping_reason")
            )
            use_native_grasp_zone = use_davis_native_grasp_zone(
                configured_diagnostic_roi,
                current_stage,
                target_context_status,
            )
            if fixed_center_third_mode:
                analyzer.set_roi(configured_diagnostic_roi, source="configured_roi")
                analyzer.set_boundary(
                    boundary_from_roi(configured_diagnostic_roi),
                    source=FIXED_CENTER_THIRD_BOUNDARY_SOURCE,
                )
                analyzer.set_contact_band(None)
                object_evidence_active = bool(
                    current_stage != "PRE_CLOSE"
                    and
                    target_context is not None
                    and target_context.get("roi_source") == "aps_target"
                    and analyzer.set_object_evidence_context(target_context, now=now)
                )
                if not object_evidence_active:
                    analyzer.clear_object_evidence_context(
                        getattr(
                            analyzer,
                            "object_evidence_invalid_reason",
                            "fresh_confirmed_aps_object_evidence_unavailable",
                        )
                    )
                mapped_roi = None
                target_context_status = dict(target_context_status)
                if object_evidence_active:
                    mapped_reason = (
                        "The visible DAVIS acquisition region stays fixed at the middle third. "
                        "A fresh confirmed DAVIS APS contour inside it supplies separate "
                        "object-local post-grasp evidence without requiring cross-camera calibration."
                    )
                    grasp_band_reason = (
                        "APS-native estimated mid-body neighborhood; uncalibrated and not a "
                        "measured physical contact band."
                    )
                    limitations = list(
                        target_context_status.get("davis_target_mapping_limitations") or []
                    )
                    limitations.extend([
                        "The fixed middle-third acquisition ROI is not itself object-specific.",
                        "APS object association is image-space estimated, uncalibrated, and NOT_PHYSICALLY_VALIDATED.",
                    ])
                    target_context_status.update({
                        "davis_target_context_mode": "fixed_center_third_with_aps_object_evidence",
                        "davis_target_context_valid": True,
                        "davis_target_context_reason": mapped_reason,
                        "davis_target_mapping_mode": "fixed_center_third_acquisition_plus_davis_aps_object_evidence",
                        "davis_target_roi_priority_selected": "davis_aps_object_evidence_within_configured_center_third",
                        "davis_target_alignment_status": FIXED_CENTER_THIRD_OBJECT_EVIDENCE_ALIGNMENT,
                        "davis_target_mapping_physically_validated": False,
                        "davis_target_mapping_limitations": list(dict.fromkeys(limitations)),
                        "davis_target_calibration_reprojection_error_px": None,
                    })
                else:
                    mapped_reason = (
                        "Fixed middle-third DAVIS acquisition region selected; no current "
                        "confirmed APS contour with complete RealSense lock provenance is "
                        "available, so events remain diagnostic-only."
                    )
                    grasp_band_reason = "fixed_center_third_has_no_confirmed_object_local_contact_neighborhood"
                    target_context_status.update({
                        "davis_target_context_mode": FIXED_CENTER_THIRD_MODE,
                        "davis_target_context_valid": False,
                        "davis_target_context_reason": mapped_reason,
                        "davis_target_lock_active": False,
                        "davis_target_lock_age_ms": None,
                        "davis_target_lock_source_frame_id": None,
                        "davis_target_lock_source_writer_session_id": None,
                        "davis_target_lock_source_lock_frame_id": None,
                        "davis_target_lock_source_attempt_id": None,
                        "davis_target_lock_source_bottle_target_id": None,
                        "davis_target_lock_provenance_valid": False,
                        "davis_target_mapping_mode": FIXED_CENTER_THIRD_MODE,
                        "davis_target_roi_priority_selected": "configured_center_third_diagnostic",
                        "davis_target_alignment_status": FIXED_CENTER_THIRD_BOUNDARY_SOURCE,
                        "davis_target_mapping_physically_validated": False,
                        "davis_target_mapping_limitations": [
                            "The fixed middle-third ROI is uncalibrated and not object-specific; it cannot validate bottle slip or contact motion by itself."
                        ],
                        "davis_target_calibration_reprojection_error_px": None,
                    })
            elif use_native_grasp_zone:
                prior_handoff_reason = mapped_reason
                analyzer.set_roi(configured_diagnostic_roi, source="configured_roi")
                analyzer.set_contact_band(None)
                mapped_roi = None
                mapped_reason = (
                    "DAVIS-native fixed near-grasp movement ROI selected because the bottle is "
                    f"too close for a current cross-camera projection; prior handoff: {prior_handoff_reason}"
                )
                grasp_band_reason = "fixed_davis_near_grasp_roi_has_no_calibrated_contact_polygon"
                target_context_status = dict(target_context_status)
                target_context_status.update({
                    "davis_target_context_mode": "davis_native_close_grasp_zone_diagnostic",
                    "davis_target_context_valid": False,
                    "davis_target_context_reason": mapped_reason,
                    "davis_target_lock_active": False,
                    "davis_target_lock_age_ms": None,
                    "davis_target_lock_source_frame_id": None,
                    "davis_target_lock_source_writer_session_id": None,
                    "davis_target_lock_source_lock_frame_id": None,
                    "davis_target_lock_source_attempt_id": None,
                    "davis_target_lock_source_bottle_target_id": None,
                    "davis_target_lock_provenance_valid": False,
                    "davis_target_mapping_mode": "davis_native_fixed_near_grasp_roi",
                    "davis_target_roi_priority_selected": "davis_native_close_grasp_movement_diagnostic",
                    "davis_target_alignment_status": "DAVIS_NATIVE_FIXED_DIAGNOSTIC",
                    "davis_target_mapping_physically_validated": False,
                    "davis_target_mapping_limitations": [
                        "Fixed DAVIS near-grasp ROI is mount-specific and diagnostic-only; it is not a bottle boundary or camera calibration."
                    ],
                    "davis_target_calibration_reprojection_error_px": None,
                })
            elif mapped_roi is not None:
                analyzer.set_roi(
                    mapped_roi,
                    source=target_context.get("roi_source") or "realsense_bbox_mapped",
                    projection_metadata=target_context.get("projection_metadata"),
                )
                mapped_boundary = target_context.get("boundary_points")
                if mapped_boundary is not None:
                    analyzer.set_boundary(
                        mapped_boundary,
                        source=target_context.get("boundary_source") or "realsense_contour_mapped",
                    )
                else:
                    analyzer.set_boundary(
                        boundary_from_roi(mapped_roi),
                        source=f"{target_context.get('roi_source') or 'realsense_bbox_mapped'}_boundary",
                    )
                mapped_grasp_band = target_context.get("contact_band_points")
                grasp_band_reason = target_context.get("grasp_band_reason")
                analyzer.set_contact_band(mapped_grasp_band)
            else:
                grasp_band_reason = f"no_mapped_realsense_roi:{mapped_reason}"
                if configured_diagnostic_roi is not None:
                    analyzer.set_roi(configured_diagnostic_roi, source="configured_roi")
                    analyzer.set_contact_band(None)
                else:
                    analyzer.clear_object_context()
            report = latest_report
            if now >= next_analysis:
                t0 = time.perf_counter()
                report = analyzer.update()
                object_state = realtime.get("object_state") or {}
                grasp_plan = realtime.get("selected_target_task_plan") or plan_task_grasp(object_state).to_dict()
                simulated_force_command = simulated_force_controller.update(
                    report.get("boundary_event_field") or {},
                    object_state,
                    grasp_plan,
                    report,
                ).to_dict()
                report["simulated_force_command"] = simulated_force_command
                report["force_control_status"] = "NOT_PHYSICALLY_VALIDATED"
                report["force_control_actuator_connected"] = False
                analysis_ms = (time.perf_counter() - t0) * 1000.0
                analysis_times.append(now)
                profile_set("davis_event_capture_processing", analysis_ms)
                report.update(target_context_status)
                if mapped_roi is not None:
                    report["mapped_realsense_roi_reason"] = mapped_reason
                    report["mapped_realsense_grasp_band_reason"] = grasp_band_reason
                    report["roi_priority_selected"] = target_context.get("roi_priority_selected")
                    report["davis_roi_mapping_config"] = {
                        "DAVIS_ROI_SCALE_X": DAVIS_ROI_SCALE_X,
                        "DAVIS_ROI_SCALE_Y": DAVIS_ROI_SCALE_Y,
                        "DAVIS_ROI_OFFSET_X": DAVIS_ROI_OFFSET_X,
                        "DAVIS_ROI_OFFSET_Y": DAVIS_ROI_OFFSET_Y,
                        "DAVIS_ROI_PADDING": DAVIS_ROI_PADDING,
                    }
                else:
                    report["mapped_realsense_roi_reason"] = mapped_reason
                if len(analysis_times) >= 2:
                    report["davis_event_analysis_fps"] = round((len(analysis_times) - 1) / max(analysis_times[-1] - analysis_times[0], 1e-6), 2)
                report["davis_analysis_time_ms"] = round(analysis_ms, 2)
                report["davis_capture_fps"] = report.get("davis_update_fps")
                event_host_timestamp = first_number(report.get("event_host_timestamp"))
                report["davis_event_frame_age_ms"] = (
                    round(max(0.0, time.time() - event_host_timestamp) * 1000.0, 2)
                    if report.get("event_stream_status") == "active" and event_host_timestamp is not None
                    else None
                )
                report["realsense_mapping_context_current"] = (
                    target_context_status.get("davis_target_context_mode") == "live_realsense_mapping"
                )
                report["realsense_mapping_context_raw_gate"] = context_current
                report["realsense_mapping_context_reason"] = context_reason
                report["davis_aps_target_candidate_reason"] = (
                    aps_target_context_status.get("davis_target_context_reason")
                    if isinstance(aps_target_context_status, dict)
                    else aps_target_context_status
                )
                report["realsense_status_age_ms"] = (
                    round(realtime_status_age_sec * 1000.0, 2)
                    if np.isfinite(realtime_status_age_sec)
                    else None
                )
                analyzer.report = report
                write_runtime_output_best_effort(
                    write_json_atomic,
                    DAVIS_LATEST_MOTION_REPORT_PATH,
                    _json_compatible(report),
                )
                latest_report = report
                next_analysis = now + analysis_period
            report.update(target_context_status)
            if mapped_roi is not None:
                report["mapped_realsense_roi_reason"] = mapped_reason
                report["roi_priority_selected"] = target_context.get("roi_priority_selected")
                report["davis_roi_mapping_config"] = {
                    "DAVIS_ROI_SCALE_X": DAVIS_ROI_SCALE_X,
                    "DAVIS_ROI_SCALE_Y": DAVIS_ROI_SCALE_Y,
                    "DAVIS_ROI_OFFSET_X": DAVIS_ROI_OFFSET_X,
                    "DAVIS_ROI_OFFSET_Y": DAVIS_ROI_OFFSET_Y,
                    "DAVIS_ROI_PADDING": DAVIS_ROI_PADDING,
                }
            else:
                report["mapped_realsense_roi_reason"] = mapped_reason
            report["mapped_realsense_grasp_band_reason"] = grasp_band_reason
            if now >= next_save:
                t1 = time.perf_counter()
                report["davis_bottleneck"] = "analysis" if float(report.get("davis_analysis_time_ms") or 0.0) > display_period * 1000.0 else "display_or_idle"
                analyzer.report = report
                analyzer.save_outputs(include_images=bool(DAVIS_SAVE_DEBUG_EVERY_FRAME and DAVIS_DEBUG_HEATMAP))
                perf_snapshot = read_json(FINAL_DEMO_FILES["davis_performance_profile"], {})
                perf_snapshot.update({
                    "davis_display_fps": report.get("davis_display_fps"),
                    "davis_capture_fps": report.get("davis_capture_fps"),
                    "davis_event_analysis_fps": report.get("davis_event_analysis_fps"),
                    "davis_event_rate": (report.get("current_metrics") or {}).get("event_rate"),
                    "davis_aps_frame_age_ms": report.get("davis_aps_frame_age_ms"),
                    "davis_event_frame_age_ms": report.get("davis_event_frame_age_ms"),
                    "davis_gui_draw_time_ms": report.get("davis_gui_draw_time_ms"),
                    "davis_analysis_time_ms": report.get("davis_analysis_time_ms"),
                    "davis_bottleneck": report.get("davis_bottleneck"),
                })
                write_runtime_output_best_effort(
                    write_json_atomic,
                    FINAL_DEMO_FILES["davis_performance_profile"],
                    _json_compatible(perf_snapshot),
                )
                profile_set("davis_report_write", (time.perf_counter() - t1) * 1000.0)
                next_save = now + DAVIS_REPORT_INTERVAL_SEC
            aps_rgb = getattr(analyzer, "latest_aps_display_enhanced", None)
            aps_display_meta = getattr(analyzer, "latest_aps_display_meta", None) or {}
            if aps_rgb is None:
                aps_rgb = getattr(analyzer, "latest_aps_display_rgb", None)
            if aps_rgb is not None:
                aps_t0 = time.perf_counter()
                aps_out = aps_rgb
                label = aps_display_meta.get("aps_display_label")
                warning = report.get("aps_display_warning")
                if label or warning:
                    aps_out = np.ascontiguousarray(aps_rgb.copy())
                    if label:
                        cv2.putText(aps_out, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(aps_out, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 235, 160), 1, cv2.LINE_AA)
                    if warning:
                        yy = aps_out.shape[0] - 8
                        cv2.putText(aps_out, str(warning)[:64], (6, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(aps_out, str(warning)[:64], (6, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 200, 255), 1, cv2.LINE_AA)
                jpeg, _convert_ms, _encode_ms = encode_rgb_to_jpeg(aps_out, quality=82)
                if jpeg:
                    stream_buffers["davis_aps"].update(jpeg, source="davis_aps_frame")
                    write_runtime_output_best_effort(write_bytes_atomic, DAVIS_LATEST_APS_PATH, jpeg)
                profile_set("davis_aps_encoding", (time.perf_counter() - aps_t0) * 1000.0)
            if now >= next_hull:
                t2 = time.perf_counter()
                raw_heatmap_events = recent_events(
                    analyzer.events_arr,
                    LIVE_HEATMAP_WINDOW_SEC,
                    now=now,
                )
                raw_heatmap = heatmap_rgb(
                    raw_heatmap_events,
                    analyzer.roi_xywh,
                    background_rgb=aps_rgb,
                    neighborhood_filter=True,
                )
                hull_img = draw_davis_motion_hull(report, base_image=raw_heatmap)
                report["raw_event_heatmap_visible"] = True
                report["raw_event_heatmap_event_count"] = int(raw_heatmap_events.shape[0])
                report["event_heatmap_display_background"] = (
                    "CURRENT_DAVIS_APS_DISPLAY_ONLY" if aps_rgb is not None else "DARK_BACKGROUND_APS_UNAVAILABLE"
                )
                report["event_heatmap_display_neighborhood_filter"] = True
                report["event_heatmap_display_temporal_decay"] = True
                report["event_heatmap_analysis_events_unchanged"] = True
                draw_ms = (time.perf_counter() - t2) * 1000.0
                jpeg, convert_ms, encode_ms = encode_rgb_to_jpeg(hull_img, quality=72)
                if jpeg:
                    display_times.append(now)
                    if len(display_times) >= 2:
                        report["davis_display_fps"] = round((len(display_times) - 1) / max(display_times[-1] - display_times[0], 1e-6), 2)
                    report["davis_gui_draw_time_ms"] = round(draw_ms + convert_ms + encode_ms, 2)
                    profile_set("davis_hull_drawing", draw_ms + convert_ms)
                    profile_set("davis_hull_encoding", encode_ms)
                    stream_buffers["davis_heatmap"].update(jpeg, source="davis_recent_motion_hull")
                    write_runtime_output_best_effort(
                        write_bytes_atomic,
                        DAVIS_LATEST_EVENT_HEATMAP_PATH,
                        jpeg,
                    )
                next_hull = now + display_period
            if DAVIS_DEBUG_HEATMAP and now >= next_heatmap:
                t2 = time.perf_counter()
                heatmap_events = recent_events(analyzer.events_arr, LIVE_HEATMAP_WINDOW_SEC)
                heatmap = heatmap_rgb(
                    heatmap_events,
                    analyzer.roi_xywh,
                    background_rgb=aps_rgb,
                    neighborhood_filter=True,
                )
                current = report.get("current_metrics") or {}
                debug_lines = [
                    f"rate {float(current.get('event_rate', 0.0) or 0.0):.0f} ev/s",
                    f"heatmap {int(round(LIVE_HEATMAP_WINDOW_SEC * 1000.0))} ms | scoring {int(round(STABILITY_WINDOW_SEC * 1000.0))} ms",
                    f"rolling {event_buffer_age_sec(analyzer.events_arr):.2f}s | shown {len(heatmap_events)} ev",
                    f"drops win/tick/roll {report.get('dropped_event_windows', 0)}/{report.get('dropped_events_due_tick_cap', 0)}/{report.get('dropped_events_due_rolling_cap', 0)}",
                ]
                for idx, text in enumerate(debug_lines):
                    y = 18 + idx * 18
                    cv2.putText(heatmap, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(heatmap, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                draw_ms = (time.perf_counter() - t2) * 1000.0
                jpeg, convert_ms, encode_ms = encode_rgb_to_jpeg(heatmap, quality=74)
                if jpeg:
                    profile_set("davis_drawing", draw_ms + convert_ms)
                    profile_set("davis_encoding", encode_ms)
                    stream_buffers["davis_heatmap"].update(jpeg, source="davis_motion_thread")
                next_heatmap = now + display_period
            elapsed = time.perf_counter() - loop_start
            profile_set("davis_loop_total", elapsed * 1000.0)
            time.sleep(max(0.0, target_period - elapsed))
        analyzer.save_outputs(include_images=bool(DAVIS_DEBUG_HEATMAP))
    except Exception as exc:
        davis_last_error = str(exc)
        if runtime_recording is not None:
            runtime_recording.abort("dashboard_davis_loop_failed")
        try:
            with open(LOG_DIR / "davis_backend.log", "a", encoding="utf-8") as log:
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} error: {davis_last_error}\n")
        except Exception:
            pass
        error_report = {
            "davis_status": "error",
            "motion_class": "SENSOR_UNAVAILABLE",
            "decision": "UNAVAILABLE",
            "visual_safety_response": "UNAVAILABLE",
            "error": davis_last_error,
            "source": "web_dashboard_davis_loop",
            "timestamp": time.time(),
        }
        try:
            if analyzer is not None and isinstance(getattr(analyzer, "report", None), dict):
                error_report = {**analyzer.report, **error_report}
            write_json_atomic(DAVIS_LATEST_MOTION_REPORT_PATH, _json_compatible(error_report))
            write_json_atomic(DAVIS_LATEST_STATUS_PATH, _json_compatible({
                "state": "ERROR",
                "event_state": "ERROR",
                "error": davis_last_error,
                "timestamp": time.time(),
                "source": "web_dashboard_davis_loop",
            }))
            write_json_atomic(OUTPUT_DIR / "davis_motion_report.json", _json_compatible(error_report))
        except Exception:
            pass
    finally:
        if analyzer is not None:
            try:
                analyzer.monitor.stop()
            except Exception:
                if runtime_recording is not None:
                    runtime_recording.abort("dashboard_davis_teardown_failed")
        with davis_lock:
            if davis_analyzer is analyzer:
                davis_analyzer = None


def start_davis_motion():
    global davis_thread
    with davis_lifecycle_lock:
        if davis_thread and davis_thread.is_alive():
            return True, "DAVIS motion monitor is already running."
        with open(LOG_DIR / "davis_backend.log", "a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} starting dashboard-owned DAVIS backend\n")
        davis_stop.clear()
        davis_thread = threading.Thread(target=davis_loop, daemon=True)
        davis_thread.start()
        return True, "Started DAVIS motion monitor."


def stop_davis_motion():
    global davis_thread
    with davis_lifecycle_lock:
        thread = davis_thread
        davis_stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=4)
        if thread and thread.is_alive():
            return False, "DAVIS motion monitor did not stop within 4 seconds; duplicate restart blocked."
        if davis_thread is thread:
            davis_thread = None
        with open(LOG_DIR / "davis_backend.log", "a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} stopped dashboard-owned DAVIS backend\n")
        return True, "Stopped DAVIS motion monitor."


def reconnect_monitor_loop():
    while not reconnect_stop.is_set():
        try:
            now = time.monotonic()
            if observe_realsense_usb_for_rearm(realsense_d405_enumerated()):
                # One bounded attempt after a physical absent-to-present edge:
                # seed the existing failure counter one below the circuit limit.
                desired_processes["realsense"] = True
                reconnect_failures["realsense"] = RECONNECT_CIRCUIT_BREAKER_FAILURES - 1
                reconnect_retry_after["realsense"] = 0.0
                print(
                    "RealSense D405 re-enumerated; re-arming one bounded recovery attempt.",
                    flush=True,
                )
            candidates = (
                ("realsense", is_process_running("realsense"), start_process),
                ("davis_motion", bool(davis_thread and davis_thread.is_alive()), lambda _name: start_davis_motion()),
            )
            for name, running, starter in candidates:
                if not desired_processes.get(name) or running or now < reconnect_retry_after[name]:
                    continue
                ok, msg = starter(name)
                label = "RealSense" if name == "realsense" else "DAVIS"
                print(f"{label} reconnect attempt: {msg}", flush=True)
                if ok:
                    reconnect_failures[name] = 0
                    reconnect_retry_after[name] = 0.0
                    continue
                reconnect_failures[name] += 1
                failures = reconnect_failures[name]
                delay = min(
                    RECONNECT_BASE_DELAY_SEC * (2 ** max(0, failures - 1)),
                    RECONNECT_MAX_DELAY_SEC,
                )
                reconnect_retry_after[name] = time.monotonic() + delay
                if failures >= RECONNECT_CIRCUIT_BREAKER_FAILURES:
                    desired_processes[name] = False
                    if name == "realsense":
                        realsense_usb_rearm["waiting"] = True
                        realsense_usb_rearm["last_present"] = realsense_d405_enumerated()
                    print(
                        f"{label} reconnect circuit opened after {failures} failures; "
                        "automatic USB probing stopped. Use Start after checking the connection.",
                        flush=True,
                    )
        except Exception as exc:
            print(f"Hardware reconnect monitor warning: {exc}", flush=True)
        reconnect_stop.wait(5.0)


def start_reconnect_monitor():
    global reconnect_thread
    if reconnect_thread and reconnect_thread.is_alive():
        return
    reconnect_stop.clear()
    reconnect_thread = threading.Thread(target=reconnect_monitor_loop, daemon=True)
    reconnect_thread.start()


def set_desired_process(name, enabled):
    if name in desired_processes:
        if name == "realsense":
            # Explicit Start/Stop always cancels an older passive rearm wait.
            realsense_usb_rearm["waiting"] = False
        if enabled and not desired_processes[name]:
            reconnect_failures[name] = 0
            reconnect_retry_after[name] = 0.0
        desired_processes[name] = bool(enabled)


def latest_log(path, max_chars=5000):
    try:
        text = Path(path).read_text(errors="replace")
        return text[-max_chars:]
    except Exception:
        return ""


def hand_prototype_dashboard_status(status):
    """Expose the configured hand prototype and current locked-band state."""
    status = status if isinstance(status, dict) else {}
    target_lock = status.get("target_lock") if isinstance(status.get("target_lock"), dict) else {}
    live_config = status.get("hand_mounted_camera")
    live_config = live_config if isinstance(live_config, dict) else {}
    config = {
        key: live_config.get(key)
        if live_config.get(key) is not None
        else HAND_MOUNTED_CAMERA_CONFIG.get(key)
        for key in HAND_MOUNTED_CAMERA_CONFIG_KEYS
    }

    def band_value(key):
        value = status.get(key)
        return target_lock.get(key) if value is None else value

    configured_enabled = bool(config.get("enabled"))
    mode_enabled = status.get("hand_mounted_camera_mode")
    locked_band = band_value("locked_grasp_band")
    locked_band = locked_band if isinstance(locked_band, dict) else {}
    return {
        "hand_mounted_camera_mode": configured_enabled if mode_enabled is None else bool(mode_enabled),
        "hand_mounted_camera": config,
        "hand_mounted_mode_label": status.get("hand_mounted_mode_label") or HAND_MOUNTED_MODE_LABEL,
        "hand_mounted_stage": status.get("hand_mounted_stage"),
        "approach_mode_active": status.get("approach_mode_active"),
        "approach_tracking_active": status.get(
            "approach_tracking_active",
            status.get("approach_mode_active"),
        ),
        "approach_mode_requested": status.get("approach_mode_requested"),
        "approach_mode_requested_at": status.get("approach_mode_requested_at"),
        "lock_generation": status.get("lock_generation") if status.get("lock_generation") is not None else target_lock.get("lock_generation"),
        "last_lock_event": status.get("last_lock_event") or target_lock.get("last_lock_event"),
        "last_relock_event": status.get("last_relock_event") or target_lock.get("last_relock_event"),
        "close_proximity_mode_active": status.get("close_proximity_mode_active"),
        "full_bottle_required": status.get("full_bottle_required"),
        "partial_view_allowed": status.get("partial_view_allowed"),
        "locked_band_visible": status.get("locked_band_visible"),
        "locked_band_current_rgbd_supported": status.get(
            "locked_band_current_rgbd_supported"
        ),
        "locked_band_visibility_reason": status.get("locked_band_visibility_reason"),
        "operator_band_state": status.get("operator_band_state"),
        "operator_locked_fraction_visual_guide_current": status.get(
            "operator_locked_fraction_visual_guide_current"
        ),
        "operator_band_is_safety_measurement": status.get(
            "operator_band_is_safety_measurement", False
        ),
        "guidance_frame": status.get("guidance_frame") or (
            "camera_frame" if config.get("use_camera_frame_guidance") else "approximate_hand_frame"
        ),
        "guidance_label": status.get("guidance_label") or "Hand-reach guidance (camera frame, approximate)",
        "robot_base_calibrated": status.get("robot_base_calibrated", False),
        "robot_base_coordinates_available": status.get("robot_base_coordinates_available", False),
        "locked_band_depth_mm": status.get("locked_band_depth_mm"),
        "locked_body_fraction_center": locked_band.get("locked_body_fraction_center"),
        "target_error_x_px": status.get("target_error_x_px"),
        "target_error_y_px": status.get("target_error_y_px"),
        "hand_reach_projection_px": status.get("hand_reach_projection_px"),
        "hand_reach_projection_valid": status.get("hand_reach_projection_valid"),
        "hand_reach_projection_visible": status.get("hand_reach_projection_visible"),
        "hand_reach_projection_held": status.get("hand_reach_projection_held"),
        "hand_reach_projection_reason": status.get("hand_reach_projection_reason"),
        "hand_alignment_error_px": status.get("hand_alignment_error_px"),
        "hand_alignment_error_mm": status.get("hand_alignment_error_mm"),
        "hand_to_bottle_z_error_mm": status.get("hand_to_bottle_z_error_mm"),
        "guidance_basis": status.get("guidance_basis"),
        "grasp_band_locked": band_value("grasp_band_locked"),
        "grasp_band_current": band_value("grasp_band_current"),
        "grasp_band_held": band_value("grasp_band_held"),
        "grasp_band_projection_unavailable": band_value(
            "grasp_band_projection_unavailable"
        ),
        "grasp_band_state": band_value("grasp_band_state"),
        "sticky_single_bottle_lock_enabled": band_value(
            "sticky_single_bottle_lock_enabled"
        ),
        "sticky_lock_waiting_reacquisition": band_value(
            "sticky_lock_waiting_reacquisition"
        ),
        "sticky_lock_dropout_frames": band_value("sticky_lock_dropout_frames"),
        "sticky_lock_preserved_attempt_id": band_value(
            "sticky_lock_preserved_attempt_id"
        ),
        "grasp_band_stale": band_value("grasp_band_stale"),
        "grasp_band_projected": band_value("grasp_band_projected"),
        "grasp_band_drift_px": band_value("grasp_band_drift_px"),
        "grasp_band_scale_change": band_value("grasp_band_scale_change"),
        "grasp_band_angle_change_deg": band_value("grasp_band_angle_change_deg"),
        "grasp_band_visibility": band_value("grasp_band_visibility"),
        "grasp_band_relock_required": band_value("grasp_band_relock_required"),
        "grasp_band_relock_reason": band_value("grasp_band_relock_reason"),
        "grasp_band_projection_confidence": band_value("grasp_band_projection_confidence"),
        "locked_projection_mode": band_value("locked_projection_mode"),
        "projection_mode": band_value("projection_mode"),
        "projection_source": band_value("projection_source"),
        "projection_valid": band_value("projection_valid"),
        "projection_invalid_reason": band_value("projection_invalid_reason"),
        "two_d_fallback_used": band_value("two_d_fallback_used"),
        "band_drift_state": band_value("band_drift_state"),
        "band_drift_reason": band_value("band_drift_reason"),
    }


def realsense_operator_target_state(status):
    """Translate backend lock telemetry into mutually exclusive operator states."""
    status = status or {}
    target_lock = status.get("target_lock") or {}
    target_status = str(
        status.get("target_status")
        or target_lock.get("target_status")
        or status.get("state")
        or "SEARCHING"
    ).upper()
    reacquisition_status = str(
        status.get("lost_target_reacquisition_status") or ""
    ).upper()
    automatic_value = status.get("automatic_reacquisition_enabled")
    if automatic_value is None:
        automatic_value = target_lock.get("automatic_reacquisition_enabled")
    automatic_enabled = bool(automatic_value)
    active_reacquisition_statuses = {
        "ORIGINAL_TARGET_UNAMBIGUOUS_MATCH",
        "SINGLE_BOTTLE_AUTO_REACQUIRE_NEW_LOCK_STARTED",
        "SINGLE_BOTTLE_AUTO_REACQUIRE_STABILITY_PENDING",
    }
    reacquisition_active = bool(
        automatic_enabled and reacquisition_status in active_reacquisition_statuses
    )
    target_lost = target_status in {"LOST", "TARGET_LOST", "RELOCK_REQUIRED"}
    reacquisition_waiting = bool(
        automatic_enabled
        and not reacquisition_active
        and (
            target_lost
            or "AUTO_REACQUIRE" in reacquisition_status
            or reacquisition_status == "BLOCKED_MULTIPLE_OR_AMBIGUOUS"
        )
    )
    bottle_detected = bool(
        status.get("bottle_detected") is True
        or int(status.get("detected_bottle_count") or 0) > 0
    )
    if reacquisition_active:
        phase = "AUTO_REACQUIRING"
    elif reacquisition_waiting:
        phase = "WAITING_FOR_ONE_STABLE_CANDIDATE"
    elif target_lost:
        phase = "LOST"
    elif target_status in {
        "TRACKING",
        "LOCKED",
        "TARGET_LOCKED",
        "TRACKING_UNCERTAIN",
        "LOW_CONFIDENCE",
    }:
        phase = "TRACKING"
    elif bottle_detected:
        phase = "DETECTED"
    else:
        phase = "SEARCHING"
    return {
        "operator_target_phase": phase,
        "operator_reacquisition_active": reacquisition_active,
        "operator_reacquisition_waiting": reacquisition_waiting,
        "operator_reacquisition_status": reacquisition_status or None,
    }


def merge_full_grasp_hulls(matrix, hulls):
    """Attach full wrench geometry without crossing quality-compute identities."""
    matrix = matrix if isinstance(matrix, dict) else {}
    hulls = hulls if isinstance(hulls, dict) else {}
    if not hulls:
        return matrix
    matrix_id = matrix.get("quality_compute_id")
    hull_id = hulls.get("quality_compute_id")
    if matrix_id is not None and hull_id is not None and str(matrix_id) != str(hull_id):
        return matrix
    existing = matrix.get("grasp_hulls") if isinstance(matrix.get("grasp_hulls"), dict) else {}
    # realtime_wrench_matrix_status carries a deliberately compact hull summary.
    # Let the identity-matched artifact add its wrench points/simplices instead
    # of allowing the summary to shadow the graph geometry via setdefault().
    matrix["grasp_hulls"] = {**existing, **hulls}
    return matrix


def summarize_realsense(status):
    target_lock = status.get("target_lock") or {}
    target_selection = status.get("multi_bottle_target_selection") or {}
    finger_detection = status.get("finger_detection") or {}
    hand_prototype = hand_prototype_dashboard_status(status)
    operator_target = realsense_operator_target_state(status)
    move = {
        "x": first_number(status.get("recommended_move_x_mm"), status.get("move_x_mm"), status.get("dx_mm")),
        "y": first_number(status.get("recommended_move_y_mm"), status.get("move_y_mm"), status.get("dy_mm")),
        "z": first_number(status.get("recommended_move_z_mm"), status.get("move_z_mm"), status.get("dz_mm")),
    }
    matrix_path = OUTPUT_DIR / "realtime_wrench_matrix_status.json"
    matrix = read_json(matrix_path, {})
    if matrix:
        matrix.setdefault("quality_matrix_status", "updated" if matrix.get("epsilon") is not None else "idle")
        try:
            matrix["quality_matrix_file_age_sec"] = round(time.time() - matrix_path.stat().st_mtime, 2)
        except OSError:
            matrix["quality_matrix_file_age_sec"] = None
    hull_path = OUTPUT_DIR / "realtime_grasp_hulls.json"
    hulls = read_json(hull_path, {})
    if hulls:
        side_inputs = hulls.get("side_clamp_contact_inputs") or {}
        try:
            hulls["hull_file_age_sec"] = round(time.time() - hull_path.stat().st_mtime, 2)
        except OSError:
            hulls["hull_file_age_sec"] = None
        merge_full_grasp_hulls(matrix, hulls)
        matrix.setdefault("quality_compute_id", hulls.get("quality_compute_id"))
        matrix.setdefault("writer_session_id", hulls.get("writer_session_id"))
        matrix.setdefault("lock_frame_id", hulls.get("lock_frame_id"))
        matrix.setdefault("hull_status", hulls.get("status"))
        matrix.setdefault("hull_compute_time_ms", hulls.get("hull_compute_time_ms"))
        matrix.setdefault("hull_failure_reason", hulls.get("failure_reason"))
        matrix.setdefault("quality_label", hulls.get("quality_label"))
        matrix.setdefault("quality_label_stale", hulls.get("quality_label_stale"))
        matrix.setdefault("quality_label_display", hulls.get("quality_label_display"))
        matrix.setdefault("hull_lock_frame_id", hulls.get("lock_frame_id"))
        matrix.setdefault("quality_input_source", hulls.get("quality_input_source"))
        matrix.setdefault("contact_source", hulls.get("contact_source"))
        matrix.setdefault("contact_source_label", hulls.get("contact_source_label"))
        matrix.setdefault("contact_source_status", hulls.get("contact_source_status"))
        matrix.setdefault("contact_source_failure_reason", hulls.get("contact_source_failure_reason"))
        matrix.setdefault("left_contact_source", hulls.get("left_contact_source"))
        matrix.setdefault("right_contact_source", hulls.get("right_contact_source"))
        matrix.setdefault("contact_confidence_left", hulls.get("contact_confidence_left"))
        matrix.setdefault("contact_confidence_right", hulls.get("contact_confidence_right"))
        matrix.setdefault("contact_stable_frames", hulls.get("contact_stable_frames"))
        matrix.setdefault("contact_quality_gate_status", hulls.get("contact_quality_gate_status"))
        matrix.setdefault("locked_hull_frame_id", hulls.get("locked_hull_frame_id"))
        matrix.setdefault("current_hull_frame_id", hulls.get("current_hull_frame_id"))
        matrix.setdefault("hull_display_mode", hulls.get("hull_display_mode"))
        matrix.setdefault("locked_hull_age_ms", hulls.get("locked_hull_age_ms"))
        matrix.setdefault("current_vs_locked_center_delta_px", hulls.get("current_vs_locked_center_delta_px"))
        matrix.setdefault("current_vs_locked_iou", hulls.get("current_vs_locked_iou"))
        matrix.setdefault("matrix_stale_due_to_motion", hulls.get("matrix_stale_due_to_motion"))
        matrix.setdefault("occlusion_status", hulls.get("occlusion_status"))
        matrix.setdefault("contact_points_px", hulls.get("contact_points_px"))
        matrix.setdefault("contact_points_3d", hulls.get("contact_points_3d"))
        matrix.setdefault("contact_depths_mm", hulls.get("contact_depths_mm"))
        matrix.setdefault("contact_normals", hulls.get("contact_normals"))
        matrix.setdefault("intrinsics_available", hulls.get("intrinsics_available", side_inputs.get("intrinsics_available")))
        matrix.setdefault("intrinsics", hulls.get("intrinsics", side_inputs.get("intrinsics")))
        matrix.setdefault("intrinsics_source", hulls.get("intrinsics_source", side_inputs.get("intrinsics_source")))
        matrix.setdefault("contact_3d_inputs_valid", hulls.get("contact_3d_inputs_valid", side_inputs.get("contact_3d_inputs_valid")))
        matrix.setdefault("contact_3d_failure_reasons", hulls.get("contact_3d_failure_reasons", side_inputs.get("contact_3d_failure_reasons")))
        matrix.setdefault("quality_matrix_failure_reasons", hulls.get("quality_matrix_failure_reasons"))
        matrix.setdefault("force_closure_estimate", hulls.get("force_closure_estimate"))
        matrix.setdefault("force_closure", hulls.get("force_closure"))
        matrix.setdefault("raw_epsilon", hulls.get("raw_epsilon"))
        matrix.setdefault("normalized_quality_score", hulls.get("normalized_quality_score"))
        matrix.setdefault("epsilon_ball_radius", hulls.get("epsilon_ball_radius"))
        matrix.setdefault("weak_direction", hulls.get("weak_direction"))
        matrix.setdefault("wrench_hull_plot_path", hulls.get("wrench_hull_plot_path"))
        matrix.setdefault("wrench_hull_plot_status", hulls.get("wrench_hull_plot_status"))
        matrix.setdefault("wrench_hull_plot_failure_reason", hulls.get("wrench_hull_plot_failure_reason"))
        matrix.setdefault("plot_compute_time_ms", hulls.get("plot_compute_time_ms"))
        matrix.setdefault("weak_direction_overlay_path", hulls.get("weak_direction_overlay_path"))
        matrix.setdefault("weak_direction_overlay_status", hulls.get("weak_direction_overlay_status"))
        matrix.setdefault("weak_direction_overlay_failure_reason", hulls.get("weak_direction_overlay_failure_reason"))
        matrix.setdefault("weak_direction_wrench", hulls.get("weak_direction_wrench"))
        matrix.setdefault("weak_force_direction_px", hulls.get("weak_force_direction_px"))
        matrix.setdefault("weak_torque_component", hulls.get("weak_torque_component"))
        matrix.setdefault("bottle_mask_source", hulls.get("bottle_mask_source"))
        matrix.setdefault("wrench_point_count", hulls.get("wrench_point_count"))
        matrix.setdefault("hull_vertex_count", hulls.get("hull_vertex_count"))
        matrix.setdefault("hull_simplex_count", hulls.get("hull_simplex_count"))
        matrix.setdefault("finger_detection_status", hulls.get("finger_detection_status"))
        matrix.setdefault("finger_quality_gate_status", hulls.get("finger_quality_gate_status"))
        matrix.setdefault("finger_quality_gate_reasons", hulls.get("finger_quality_gate_reasons"))
        matrix.setdefault("left_finger_confidence", hulls.get("left_finger_confidence"))
        matrix.setdefault("right_finger_confidence", hulls.get("right_finger_confidence"))
        matrix.setdefault("finger_stable_frames", hulls.get("finger_stable_frames"))
    stale_live = live_matrix_staleness(status, matrix, hulls)
    matrix.update({
        "live_staleness_check": stale_live,
        "lock_frame_id": matrix.get("lock_frame_id") or stale_live.get("lock_frame_id"),
        "current_hull_frame_id": stale_live.get("current_hull_frame_id"),
        "current_vs_locked_center_delta_px": stale_live.get("current_vs_locked_center_delta_px"),
        "current_vs_locked_iou": stale_live.get("current_vs_locked_iou"),
        "current_vs_locked_depth_delta_mm": stale_live.get("current_vs_locked_depth_delta_mm"),
        "matrix_stale_reasons": stale_live.get("matrix_stale_reasons") or matrix.get("matrix_stale_reasons"),
        "matrix_stale_due_to_motion": bool(stale_live.get("matrix_stale_due_to_motion")),
        "live_observation_degraded": stale_live.get("live_observation_degraded"),
        "live_observation_reasons": stale_live.get("live_observation_reasons") or [],
        "analytic_staleness_source": stale_live.get("analytic_staleness_source"),
    })
    if stale_live.get("same_lock_quality_refresh_in_progress"):
        # The matrix/artifact files are published independently.  Keep the
        # last complete analytic baseline visible during the tiny same-lock
        # handoff window instead of flashing STALE.  Different sessions,
        # locks, attempts, relock requests, and post-grasp baselines remain
        # strict provenance failures.
        matrix["quality_label_stale"] = False
        matrix["matrix_stale_due_to_invalidation"] = False
        if str(matrix.get("quality_matrix_status") or "").upper() == "STALE_NEEDS_RECOMPUTE":
            matrix["quality_matrix_status"] = "updated"
        if matrix.get("quality_label"):
            matrix["quality_label_display"] = matrix.get("quality_label")
        matrix["quality_matrix_message"] = "Current same-lock baseline held during analytic refresh."
    live_writer_session_id = status.get("writer_session_id")
    matrix_writer_session_id = matrix.get("writer_session_id")
    live_lock_frame_id = (
        status.get("lock_frame_id")
        or (status.get("target_lock") or {}).get("lock_frame_id")
        or (status.get("target_lock_debug") or {}).get("lock_frame_id")
    )
    matrix_lock_frame_id = matrix.get("lock_frame_id") or matrix.get("hull_lock_frame_id")
    current_mismatch_reasons = []
    if live_writer_session_id and matrix_writer_session_id and live_writer_session_id != matrix_writer_session_id:
        current_mismatch_reasons.append("writer_session_mismatch")
    if live_lock_frame_id and matrix_lock_frame_id and live_lock_frame_id != matrix_lock_frame_id:
        current_mismatch_reasons.append("lock_frame_mismatch")
    if current_mismatch_reasons:
        matrix["quality_label_stale"] = True
        matrix["quality_current_mismatch_reasons"] = current_mismatch_reasons
        matrix["quality_matrix_message"] = "MATRIX STALE - previous lock/session"
        matrix["quality_label_display"] = f"{matrix.get('quality_label') or 'MATRIX'} (STALE)"
    if stale_live.get("matrix_stale_due_to_motion") and matrix.get("quality_matrix_status") in {"updated", "STALE_NEEDS_RECOMPUTE", "idle"}:
        base_label = matrix.get("quality_label") or "MATRIX"
        matrix["quality_matrix_status"] = "STALE_NEEDS_RECOMPUTE"
        matrix["quality_matrix_message"] = "LOCK BASELINE INVALID - RESET / NEW LOCK"
        matrix["quality_label_stale"] = True
        matrix["quality_label_display"] = f"{base_label} (STALE)"
    matrix.update(quality_applicability_from_status(status, matrix))
    blockers = status.get("ready_blockers", []) or []
    matrix_status = str(matrix.get("quality_matrix_status") or "idle").upper()
    matrix_stale = bool(matrix.get("quality_label_stale") or matrix.get("matrix_stale_due_to_motion"))
    quality_label = matrix.get("quality_label_display") or matrix.get("quality_label") or "N/A"
    if matrix_stale and "(STALE)" not in str(quality_label).upper():
        quality_label = f"{quality_label} (STALE)"
    primary_status = "READY" if status.get("ready_to_close") else (status.get("decision") or status.get("state") or "WAITING")
    primary_action = status.get("recommended_action") or status.get("approach_action") or "WAIT"
    quality_message = (
        matrix.get("quality_matrix_message")
        or matrix.get("quality_matrix_error")
        or (", ".join(matrix.get("matrix_stale_reasons") or []) if matrix_stale else None)
        or ("Ready to close" if status.get("ready_to_close") else (blockers[0] if blockers else "Waiting for valid grasp quality"))
    )
    ui_fields = {
        "ui_primary_status": primary_status,
        "ui_primary_action": primary_action,
        "ui_quality_status": "STALE" if matrix_stale else matrix_status,
        "ui_quality_message": quality_message,
    }
    realtime_audit = status.get("realtime_pipeline_audit", {}) or {}
    processed_overlay = status.get("processed_overlay_timing", {}) or {}
    performance = status.get("performance_profile", {}).get("timings_ms", {}) or {}
    overlay_fps = (
        realtime_audit.get("evaluation_output_fps")
        or realtime_audit.get("previous_processed_fps")
        or realtime_audit.get("evaluation_requested_fps")
        or realtime_audit.get("rgbd_latest_frame_fps")
    )
    live_fps = (
        realtime_audit.get("rgbd_latest_frame_fps")
        or realtime_audit.get("camera_emitted_fps")
        or realtime_audit.get("color_callback_fps")
    )
    numeric_timings = {
        key: value
        for key, value in performance.items()
        if isinstance(value, (int, float)) and not key.startswith("rs_output_")
    }
    slowest_stage = None
    if numeric_timings:
        slowest_name, slowest_ms = max(numeric_timings.items(), key=lambda item: item[1])
        slowest_stage = {"name": slowest_name, "ms": round(float(slowest_ms), 2)}
    return {
        "state": status.get("state"),
        "decision": status.get("decision"),
        "action": status.get("recommended_action") or status.get("approach_action"),
        "ready_to_close": status.get("ready_to_close"),
        "target_status": status.get("target_status") or (status.get("target_lock") or {}).get("target_status") or status.get("target_lock_debug", {}).get("target_status"),
        "tracking_confidence": status.get("tracking_confidence"),
        "frame_id": status.get("frame_id"),
        "post_grasp_active": status.get("post_grasp_active"),
        "post_grasp_baseline": status.get("post_grasp_baseline") if isinstance(status.get("post_grasp_baseline"), dict) else None,
        "post_grasp_baseline_ready": status.get("post_grasp_baseline_ready"),
        "post_grasp_baseline_reason": status.get("post_grasp_baseline_reason"),
        "post_grasp_baseline_reasons": status.get("post_grasp_baseline_reasons") or [],
        "gripper_closed_confirmed": status.get("gripper_closed_confirmed"),
        "manual_gripper_closed": status.get("manual_gripper_closed"),
        "gripper_closed_confirmation_source": status.get("gripper_closed_confirmation_source"),
        "physical_contact_validated": status.get("physical_contact_validated"),
        "bottle_distance_mm": first_number(
            status.get("distance_to_bottle_mm"),
            status.get("bottle_distance_mm"),
            status.get("distance_mm"),
            status.get("bottle_depth_median_mm"),
            status.get("depth_mm"),
        ),
        "object_depth_mm": first_number(status.get("object_depth_mm"), status.get("bottle_depth_median_mm")),
        "target_center_depth_mm": status.get("target_center_depth_mm"),
        "contact_band_depth_mm": status.get("contact_band_depth_mm"),
        "distance_confidence": status.get("distance_confidence"),
        "distance_confidence_percent": status.get("distance_confidence_percent"),
        "distance_confidence_ready_threshold": status.get("distance_confidence_ready_threshold"),
        "distance_measurement_valid": status.get("distance_measurement_valid"),
        "distance_validity_reason": status.get("distance_validity_reason"),
        "distance_mask_support_valid": status.get("distance_mask_support_valid"),
        "left_patch_bottle_overlap_ratio": status.get("left_patch_bottle_overlap_ratio"),
        "right_patch_bottle_overlap_ratio": status.get("right_patch_bottle_overlap_ratio"),
        "left_patch_depth_eligible_pixels": status.get("left_patch_depth_eligible_pixels"),
        "right_patch_depth_eligible_pixels": status.get("right_patch_depth_eligible_pixels"),
        "actionable_distance_mm": status.get("actionable_distance_mm"),
        "distance_ready": status.get("distance_ready"),
        "distance_status": status.get("distance_status"),
        "in_close_range": status.get("in_close_range"),
        "desired_close_distance_mm": status.get("desired_close_distance_mm"),
        "min_close_distance_mm": status.get("min_close_distance_mm"),
        "max_close_distance_mm": status.get("max_close_distance_mm"),
        "close_distance_range_source": status.get("close_distance_range_source"),
        "close_distance_range_validation_status": status.get("close_distance_range_validation_status"),
        "close_distance_range_physically_validated": status.get("close_distance_range_physically_validated"),
        "roi_depth_projection_scale": status.get("roi_depth_projection_scale"),
        "roi_projected_bbox_unclipped": status.get("roi_projected_bbox_unclipped"),
        "roi_projected_visible_fraction": status.get("roi_projected_visible_fraction"),
        "roi_projected_extent_partial": status.get("roi_projected_extent_partial"),
        "roi_candidate_skin_exclusion_applied": status.get("roi_candidate_skin_exclusion_applied"),
        "roi_candidate_skin_overlap_seed_pixels": status.get("roi_candidate_skin_overlap_seed_pixels"),
        "roi_candidate_hand_occlusion": status.get("roi_candidate_hand_occlusion"),
        "roi_candidate_hand_occlusion_state": status.get("roi_candidate_hand_occlusion_state"),
        "roi_candidate_hand_occlusion_validation_status": status.get(
            "roi_candidate_hand_occlusion_validation_status"
        ),
        "camera_to_target_xyz_mm": status.get("camera_to_target_xyz_mm"),
        "camera_optical_center_px": status.get("camera_optical_center_px"),
        "fx": status.get("fx"),
        "fy": status.get("fy"),
        "camera_to_hand_offset_enabled": status.get("camera_to_hand_offset_enabled"),
        "camera_to_hand_offset_active": status.get("camera_to_hand_offset_active"),
        "camera_to_hand_offset_mm": status.get("camera_to_hand_offset_mm"),
        "camera_to_hand_offset_uncertainty_mm": status.get("camera_to_hand_offset_uncertainty_mm"),
        "desired_contact_gap_mm": status.get("desired_contact_gap_mm"),
        "hand_reach_projection_px": status.get("hand_reach_projection_px"),
        "desired_camera_center_for_grasp_px": status.get("desired_camera_center_for_grasp_px"),
        "camera_to_hand_offset_px": status.get("camera_to_hand_offset_px"),
        "hand_reach_projection_valid": status.get("hand_reach_projection_valid"),
        "hand_projection_valid": status.get("hand_projection_valid"),
        "hand_projection_reason": status.get("hand_projection_reason"),
        "hand_projection_depth_mm": status.get("hand_projection_depth_mm"),
        "hand_projection_target_depth_mm": status.get(
            "hand_projection_target_depth_mm"
        ),
        "hand_reach_projection_visible": status.get("hand_reach_projection_visible"),
        "hand_reach_projection_held": status.get("hand_reach_projection_held"),
        "hand_reach_projection_reason": status.get("hand_reach_projection_reason"),
        "hand_alignment_error_px": status.get("hand_alignment_error_px"),
        "hand_alignment_error_mm": status.get("hand_alignment_error_mm"),
        "hand_to_grasp_error_px": status.get("hand_to_grasp_error_px"),
        "hand_to_grasp_error_mm": status.get("hand_to_grasp_error_mm"),
        "hand_to_bottle_z_error_mm": status.get("hand_to_bottle_z_error_mm"),
        "guidance_basis": status.get("guidance_basis"),
        "hand_offset_calibration_status": status.get("hand_offset_calibration_status"),
        "hand_offset_physically_validated": status.get("hand_offset_physically_validated"),
        "coordinate_frame": status.get("coordinate_frame"),
        "guidance_frame_status": status.get("guidance_frame_status"),
        "movement_convention": status.get("movement_convention") or {},
        "grip_width_mm": status.get("grip_width_mm"),
        "grasp_feasibility_score": first_number(
            status.get("grasp_feasibility_score"),
            target_selection.get("grasp_feasibility_score"),
        ),
        "grasp_feasibility_label": (
            status.get("grasp_feasibility_label")
            or target_selection.get("grasp_feasibility_label")
        ),
        "grasp_feasibility_reason": (
            status.get("grasp_feasibility_reason")
            or target_selection.get("grasp_feasibility_reason")
        ),
        "suggested_operator_action": (
            status.get("suggested_operator_action")
            or target_selection.get("suggested_operator_action")
        ),
        "suggested_robot_action": (
            status.get("suggested_robot_action")
            or target_selection.get("suggested_robot_action")
        ),
        "visual_contact_ready": (
            status.get("visual_contact_ready")
            if status.get("visual_contact_ready") is not None
            else status.get("contact_model_ready")
        ),
        "epsilon": status.get("epsilon_closure") or status.get("force_closure"),
        "move": move,
        "blockers": status.get("ready_blockers", []),
        "performance": performance,
        "processed_overlay_timing": processed_overlay,
        "realtime_pipeline_audit": realtime_audit,
        "live_fps": live_fps,
        "overlay_fps": overlay_fps,
        "frame_age_ms": processed_overlay.get("frame_age_ms") or realtime_audit.get("pending_frame_age_ms"),
        "overlay_render_ms": processed_overlay.get("overlay_drawing_ms") or performance.get("gui_overlay_render"),
        "slowest_stage": slowest_stage,
        "fps_counter_source": "evaluation_output_fps" if realtime_audit.get("evaluation_output_fps") else (
            "previous_processed_fps" if realtime_audit.get("previous_processed_fps") else (
                "evaluation_requested_fps" if realtime_audit.get("evaluation_requested_fps") else "rgbd_latest_frame_fps"
            )
        ),
        "overlay_health": status.get("overlay_health"),
        "overlay_last_updated": status.get("overlay_last_updated"),
        "overlay_frame_id": status.get("overlay_frame_id"),
        "overlay_source_image": status.get("overlay_source_image"),
        "overlay_current_path": status.get("overlay_current_path"),
        "overlay_last_valid_path": status.get("overlay_last_valid_path"),
        "overlay_stale": status.get("overlay_stale"),
        "overlay_error": status.get("overlay_error"),
        "dashboard_overlay_path": status.get("dashboard_overlay_path"),
        "dashboard_overlay_cache_buster": status.get("dashboard_overlay_cache_buster"),
        "quality_matrix": matrix,
        "finger_detection_status": status.get("finger_detection_status"),
        "left_finger_detected": status.get("left_finger_detected"),
        "right_finger_detected": status.get("right_finger_detected"),
        "left_inner_edge": status.get("left_inner_edge"),
        "right_inner_edge": status.get("right_inner_edge"),
        "left_finger_bbox": status.get("left_finger_bbox"),
        "right_finger_bbox": status.get("right_finger_bbox"),
        "left_finger_confidence": status.get("left_finger_confidence"),
        "right_finger_confidence": status.get("right_finger_confidence"),
        "left_finger_contact_px": status.get("left_finger_contact_px"),
        "right_finger_contact_px": status.get("right_finger_contact_px"),
        "left_finger_depth_mm": status.get("left_finger_depth_mm"),
        "right_finger_depth_mm": status.get("right_finger_depth_mm"),
        "left_finger_depth_verified": status.get("left_finger_depth_verified"),
        "right_finger_depth_verified": status.get("right_finger_depth_verified"),
        "left_finger_depth_delta_to_bottle_mm": status.get("left_finger_depth_delta_to_bottle_mm"),
        "right_finger_depth_delta_to_bottle_mm": status.get("right_finger_depth_delta_to_bottle_mm"),
        "finger_bottle_depth_reference_mm": status.get("finger_bottle_depth_reference_mm"),
        "finger_bottle_depth_reference_source": status.get("finger_bottle_depth_reference_source"),
        "finger_bottle_depth_tolerance_mm": status.get("finger_bottle_depth_tolerance_mm"),
        "finger_contact_estimate_type": (
            status.get("finger_contact_estimate_type")
            or finger_detection.get("contact_estimate_type")
        ),
        "finger_physical_contact_validated": (
            status.get("finger_physical_contact_validated")
            if status.get("finger_physical_contact_validated") is not None
            else finger_detection.get("physical_contact_validated")
        ),
        "finger_validation_status": (
            status.get("finger_validation_status")
            or finger_detection.get("validation_status")
        ),
        "finger_stable_frames": status.get("finger_stable_frames"),
        "finger_detection_source": status.get("finger_detection_source"),
        "quality_input_source": status.get("quality_input_source"),
        "realsense_rgb_active": status.get("realsense_rgb_active"),
        "realsense_depth_active": status.get("realsense_depth_active"),
        "realsense_intrinsics_available": status.get("realsense_intrinsics_available"),
        "realsense_pointcloud_active": status.get("realsense_pointcloud_active"),
        "realsense_pointcloud_backend": status.get("realsense_pointcloud_backend"),
        "realsense_pointcloud_fps": status.get("realsense_pointcloud_fps"),
        "realsense_pointcloud_last_update_time": status.get("realsense_pointcloud_last_update_time"),
        "realsense_pointcloud_point_count": status.get("realsense_pointcloud_point_count"),
        "realsense_pointcloud_quality": status.get("realsense_pointcloud_quality"),
        "yolo_enabled": status.get("yolo_enabled"),
        "yolo_available": status.get("yolo_available"),
        "yolo_detected": status.get("yolo_detected"),
        "yolo_model_backend": status.get("yolo_model_backend"),
        "yolo_inference_ms": status.get("yolo_inference_ms"),
        "yolo_detection_count": status.get("yolo_detection_count"),
        "yolo_runtime_state": status.get("yolo_runtime_state"),
        "yolo_rgbd_fusion_mode": status.get("yolo_rgbd_fusion_mode"),
        "rgbd_candidate_count": status.get("rgbd_candidate_count"),
        "yolo_matched_rgbd_count": status.get("yolo_matched_rgbd_count"),
        "yolo_unmatched_detection_count": status.get("yolo_unmatched_detection_count"),
        "yolo_roi_pose_count": status.get("yolo_roi_pose_count"),
        "yolo_ambiguous_merged_rgbd_count": status.get("yolo_ambiguous_merged_rgbd_count"),
        "yolo_split_replacement_count": status.get("yolo_split_replacement_count"),
        "fused_pose_count": status.get("fused_pose_count"),
        "rgbd_search_resolution": status.get("rgbd_search_resolution"),
        "single_bottle_demo_mode": status.get("single_bottle_demo_mode"),
        "automatic_reacquisition_enabled": status.get("automatic_reacquisition_enabled"),
        "lost_target_reacquisition_status": status.get("lost_target_reacquisition_status"),
        "detected_bottle_count": status.get("detected_bottle_count"),
        "eligible_bottle_count": status.get("eligible_bottle_count"),
        "selected_bottle_target_id": status.get("selected_bottle_target_id"),
        "grasp_memory_status": target_lock.get("grasp_memory_status"),
        "grasp_memory": target_lock.get("grasp_memory") or {},
        "locked_body_fraction": target_lock.get("locked_body_fraction"),
        "multi_bottle_target_selection": status.get("multi_bottle_target_selection") or {},
        "bottle_target_candidates": status.get("bottle_target_candidates") or [],
        "pointcloud_status": status.get("pointcloud_status") or {},
        "embodiment_status": status.get("embodiment_status") or {},
        "quality_metric_applicability": status.get("quality_metric_applicability"),
        "primary_quality_metric": status.get("primary_quality_metric"),
        "secondary_quality_metrics": status.get("secondary_quality_metrics"),
        "quality_not_applicable_reason": status.get("quality_not_applicable_reason"),
        "contact_source": status.get("contact_source"),
        "contact_source_label": status.get("contact_source_label"),
        "tracking_source": status.get("tracking_source"),
        "identity_tracking_ok": status.get("identity_tracking_ok"),
        "identity_tracking_source": status.get("identity_tracking_source"),
        "identity_measurement_current": status.get("identity_measurement_current"),
        "identity_measurement_age_frames": status.get("identity_measurement_age_frames"),
        "identity_occlusion_hold_active": status.get("identity_occlusion_hold_active"),
        "identity_occlusion_hold_frames": status.get("identity_occlusion_hold_frames"),
        "target_identity_preserved": status.get("target_identity_preserved"),
        "identity_claim_status": status.get("identity_claim_status"),
        "locked_bottle_target_id": status.get("locked_bottle_target_id"),
        "lock_attempt_id": status.get("lock_attempt_id"),
        "lock_frame_id": status.get("lock_frame_id"),
        "post_grasp_identity_tracking_ok": status.get("post_grasp_identity_tracking_ok"),
        "post_grasp_identity_occlusion_hold": status.get("post_grasp_identity_occlusion_hold"),
        "post_grasp_identity_status": status.get("post_grasp_identity_status"),
        "contact_patch_tracking_ok": status.get("contact_patch_tracking_ok"),
        "contact_patch_tracking_reason": status.get("contact_patch_tracking_reason"),
        "tracking_roi_source": status.get("tracking_roi_source"),
        "edge_source": status.get("edge_source"),
        "band_source": status.get("band_source"),
        "roi_escape_risk": status.get("roi_escape_risk"),
        "display_center_lag_px": status.get("display_center_lag_px"),
        "display_smoothing_mode": status.get("display_smoothing_mode"),
        "smoothing_mode": status.get("smoothing_mode"),
        "writer_pid": status.get("writer_pid"),
        "writer_session_id": status.get("writer_session_id"),
        "writer_start_time": status.get("writer_start_time"),
        "writer_mode": status.get("writer_mode"),
        "monotonic_frame_counter": status.get("monotonic_frame_counter"),
        **hand_prototype,
        **operator_target,
        **ui_fields,
    }


def summarize_davis(report):
    report_timestamp = first_number(
        report.get("report_timestamp"),
        report.get("timestamp"),
    )
    report_heartbeat_age_sec = (
        None
        if report_timestamp is None
        else max(0.0, time.time() - float(report_timestamp))
    )
    report_heartbeat_max_age_sec = max(3.0, 3.0 * DAVIS_REPORT_INTERVAL_SEC)
    report_heartbeat_current = bool(
        report_heartbeat_age_sec is not None
        and report_heartbeat_age_sec <= report_heartbeat_max_age_sec
    )
    metrics = report.get("lightweight_event_metrics") or {}
    estimate = report.get("motion_estimate") or {}
    boundary = report.get("boundary_event_field") or {}
    target_mode = str(report.get("davis_target_context_mode") or "").lower()
    target_priority = str(report.get("davis_target_roi_priority_selected") or "").lower()
    target_alignment = str(report.get("davis_target_alignment_status") or "").upper()
    roi_projection_method = str(report.get("roi_projection_method") or "").lower()
    fixed_center_third_acquisition = bool(
        report.get("acquisition_roi_fixed") is True
        or report.get("analysis_roi_fixed") is True
        or target_mode.startswith("fixed_center_third")
        or roi_projection_method == "fixed_center_third"
    )
    object_evidence_current = report.get("object_evidence_context_current") is True
    fixed_center_third_diagnostic = bool(
        fixed_center_third_acquisition
        and not object_evidence_current
        and (
            target_mode == "fixed_center_third"
            or target_priority == "configured_center_third_diagnostic"
            or target_alignment == "CONFIGURED_CENTER_THIRD_UNCALIBRATED"
            or roi_projection_method == "fixed_center_third"
        )
    )
    return {
        **{k: report.get(k) for k in (
            "aps_state", "aps_stream_status", "aps_fps",
            "aps_visibility_status", "aps_visibility_score", "aps_underexposed",
            "aps_overexposed", "aps_low_contrast", "aps_display_mode",
            "aps_display_enhanced", "aps_display_warning",
            "aps_exposure_control_available", "aps_exposure_configuration_applied",
            "aps_exposure_us", "aps_exposure_auto", "aps_exposure_source",
            "aps_gain_status",
            "aps_channel_order", "aps_channel_order_source",
            "aps_color_accuracy_status", "aps_device_timestamp_us",
            "aps_device_timestamp_basis", "aps_host_timestamp",
            "motion_hull_valid", "motion_hull_reason", "object_motion_valid",
            "object_motion_reason", "grasp_zone_motion_valid",
            "grasp_zone_motion_reason", "grasp_zone_motion_class",
            "motion_attribution", "motion_activity_label", "roi_object_specific",
            "local_to_global_event_ratio", "davis_motion_usable_for_grasp",
            "davis_motion_excluded_reason", "davis_live_quality_contribution",
            "current_stage", "cumulative_motion_path_px",
            "realsense_mapping_context_current", "realsense_mapping_context_reason",
            "realsense_status_age_ms",
            "raw_window_event_count", "object_roi_window_event_count",
            "event_analysis_status", "event_timestamp_basis",
            "event_timestamp_status", "event_timestamp_lag_ms",
            "event_timestamp_reanchor_count", "event_host_timestamp",
            "rolling_buffer_span_sec",
            "force_control_status", "force_control_actuator_connected",
            "event_health_valid", "active_pixel_ratio", "active_pixel_count",
            "positive_event_count", "negative_event_count", "polarity_balance",
            "roi_confidence", "roi_confidence_status", "roi_projection_method",
            "roi_projection_status", "roi_projection_warning",
            "roi_projection_physically_validated", "full_frame_fallback",
            "local_event_rate", "outside_roi_event_rate", "global_activity_score",
            "lighting_flicker_score", "camera_shake_score", "background_motion_score",
            "davis_closeup_used_for_live_quality", "davis_closeup_exclusion_reason",
            "davis_closeup_confidence", "davis_closeup_motion_class",
            "davis_closeup_window", "davis_closeup_monitoring_state",
            "davis_closeup_stability_decision", "closeup_action_elapsed_sec",
            "closeup_action_window_complete", "closeup_action_windows_sec",
            "raw_event_heatmap_visible", "raw_event_heatmap_event_count",
            "roi_priority_selected",
            "diagnostic_movement_pattern", "diagnostic_movement_pattern_score",
            "diagnostic_movement_pattern_status", "diagnostic_movement_pattern_reason",
            "diagnostic_movement_pattern_object_attributed",
            "diagnostic_movement_pattern_physically_validated",
            "diagnostic_movement_pattern_scores",
            "acquisition_roi_source", "acquisition_roi_xywh",
            "acquisition_roi_fixed", "object_evidence_context_current",
            "object_evidence_roi_source", "object_evidence_roi_xywh",
            "object_evidence_boundary_source", "object_evidence_status",
            "object_evidence_contact_band_source",
            "object_evidence_contact_band_physically_validated",
            "object_evidence_overlap_ratio", "object_evidence_physically_validated",
            "object_evidence_invalid_reason", "aps_target_used_as_object_evidence_roi",
        ) if report.get(k) is not None},
        "backend": report.get("backend") or report.get("davis_backend"),
        "report_timestamp": report_timestamp,
        "report_heartbeat_age_sec": (
            None
            if report_heartbeat_age_sec is None
            else round(report_heartbeat_age_sec, 3)
        ),
        "report_heartbeat_max_age_sec": report_heartbeat_max_age_sec,
        "report_heartbeat_current": report_heartbeat_current,
        "stream": report.get("event_stream_status") or report.get("davis_status"),
        "motion_class": report.get("davis_safety_class") or report.get("motion_class") or report.get("post_grasp_motion_class"),
        "raw_motion_class": report.get("raw_motion_class") or estimate.get("motion_class"),
        "decision": report.get("decision") or report.get("post_grasp_decision"),
        "event_rate": (report.get("current_metrics") or {}).get("event_rate") or report.get("event_rate"),
        "last_poll_event_count": report.get("last_poll_event_count"),
        "rolling_event_count": report.get("rolling_event_count"),
        "davis_processing_latency_ms": report.get("davis_processing_latency_ms"),
        "davis_update_fps": report.get("davis_update_fps"),
        "event_window_ms": report.get("event_window_ms"),
        "event_window_target_ms": report.get("event_window_target_ms"),
        "event_window_mode": report.get("event_window_mode"),
        "heatmap_window_ms": report.get("heatmap_window_ms") if DAVIS_DEBUG_HEATMAP else "debug disabled",
        "scoring_window_ms": report.get("scoring_window_ms"),
        "rolling_buffer_age_sec": report.get("rolling_buffer_age_sec"),
        "dropped_event_windows": report.get("dropped_event_windows"),
        "dropped_events_due_tick_cap": report.get("dropped_events_due_tick_cap"),
        "dropped_events_due_rolling_cap": report.get("dropped_events_due_rolling_cap"),
        "gross_slip": report.get("gross_slip_score") or report.get("final_slip_risk_score"),
        "incipient_slip": report.get("incipient_slip_score"),
        "rotation": report.get("rotation_risk_score"),
        "micro_deviation_score": report.get("incipient_slip_score"),
        "rotation_score": report.get("rotation_risk_score"),
        "vibration": report.get("vibration_risk_score") or report.get("vibration_score"),
        "stability": report.get("stability_score") or report.get("post_grasp_stability_score"),
        "event_count": metrics.get("event_count"),
        "centroid_shift_x": metrics.get("centroid_shift_x"),
        "centroid_shift_y": metrics.get("centroid_shift_y"),
        "left_right_event_imbalance": metrics.get("left_right_event_imbalance"),
        "rotation_like_event_pattern": metrics.get("rotation_like_event_pattern"),
        "sudden_spike_after_grasp": metrics.get("sudden_spike_after_grasp"),
        "roi_xywh": report.get("davis_roi_xywh") or report.get("roi_xywh"),
        "roi_source": report.get("roi_source"),
        "operator_roi_label": (
            "Fixed center-third diagnostic ROI - not object-specific"
            if fixed_center_third_diagnostic
            else "Fixed center-third view · APS-local object evidence (uncalibrated)"
            if fixed_center_third_acquisition and object_evidence_current
            else None
        ),
        "operator_roi_role": (
            "RAW_EVENT_DIAGNOSTIC_ONLY"
            if fixed_center_third_diagnostic
            else "ESTIMATED_OBJECT_LOCAL_NOT_PHYSICALLY_VALIDATED"
            if fixed_center_third_acquisition and object_evidence_current
            else "OBJECT_SPECIFIC" if report.get("roi_object_specific") is True else "NOT_OBJECT_SPECIFIC"
        ),
        "fixed_center_third_acquisition": fixed_center_third_acquisition,
        "fixed_center_third_diagnostic": fixed_center_third_diagnostic,
        "roi_projection_uncalibrated": report.get("roi_projection_uncalibrated"),
        "object_specific_motion_confidence": report.get("object_specific_motion_confidence"),
        "mapped_realsense_roi_reason": report.get("mapped_realsense_roi_reason"),
        "realsense_mapping_context_current": report.get("realsense_mapping_context_current"),
        "realsense_mapping_context_raw_gate": report.get("realsense_mapping_context_raw_gate"),
        "realsense_mapping_context_reason": report.get("realsense_mapping_context_reason"),
        "davis_target_context_mode": report.get("davis_target_context_mode"),
        "davis_target_context_valid": report.get("davis_target_context_valid"),
        "davis_target_context_reason": report.get("davis_target_context_reason"),
        "davis_target_lock_active": report.get("davis_target_lock_active"),
        "davis_target_lock_age_ms": report.get("davis_target_lock_age_ms"),
        "davis_target_lock_source_frame_id": report.get("davis_target_lock_source_frame_id"),
        "davis_target_lock_source_writer_session_id": report.get("davis_target_lock_source_writer_session_id"),
        "davis_target_lock_source_lock_frame_id": report.get("davis_target_lock_source_lock_frame_id"),
        "davis_target_lock_source_attempt_id": report.get("davis_target_lock_source_attempt_id"),
        "davis_target_lock_source_bottle_target_id": report.get("davis_target_lock_source_bottle_target_id"),
        "davis_target_lock_provenance_valid": report.get("davis_target_lock_provenance_valid"),
        "davis_target_mapping_mode": report.get("davis_target_mapping_mode"),
        "davis_target_roi_priority_selected": report.get("davis_target_roi_priority_selected"),
        "davis_target_alignment_status": report.get("davis_target_alignment_status"),
        "davis_target_mapping_physically_validated": report.get("davis_target_mapping_physically_validated"),
        "davis_target_mapping_limitations": report.get("davis_target_mapping_limitations") or [],
        "davis_target_calibration_reprojection_error_px": report.get("davis_target_calibration_reprojection_error_px"),
        "motion_hull_point_count": report.get("motion_hull_point_count") or metrics.get("motion_hull_point_count"),
        "motion_hull_area_px2": report.get("motion_hull_area_px2") or metrics.get("motion_hull_area_px2"),
        "motion_hull_explanation": report.get("motion_hull_explanation"),
        "debug_heatmap_enabled": DAVIS_DEBUG_HEATMAP,
        "reason": report.get("response_reason") or report.get("explanation"),
        "error": davis_last_error,
        "motion_estimate": estimate,
        "motion_estimate_valid": report.get("motion_estimate_valid") if report.get("motion_estimate_valid") is not None else estimate.get("estimate_valid"),
        "motion_estimate_invalid_reason": report.get("motion_estimate_invalid_reason") or estimate.get("invalid_reason"),
        "motion_estimation_method": report.get("motion_estimation_method") or estimate.get("motion_estimation_method"),
        "motion_estimation_confidence": report.get("motion_estimation_confidence") or estimate.get("motion_estimation_confidence"),
        "motion_confidence": report.get("motion_confidence") or report.get("motion_estimation_confidence") or estimate.get("motion_estimation_confidence"),
        "translation_method": report.get("translation_method") or estimate.get("translation_method"),
        "translation_confidence": report.get("translation_confidence") or estimate.get("translation_confidence"),
        "translation_valid": report.get("translation_valid") if report.get("translation_valid") is not None else estimate.get("translation_valid"),
        "dx_px": report.get("dx_px") if report.get("dx_px") is not None else estimate.get("dx_px"),
        "dy_px": report.get("dy_px") if report.get("dy_px") is not None else estimate.get("dy_px"),
        "displacement_px": report.get("displacement_px") if report.get("displacement_px") is not None else estimate.get("displacement_px"),
        "displacement_direction_deg": report.get("displacement_direction_deg") if report.get("displacement_direction_deg") is not None else estimate.get("displacement_direction_deg"),
        "velocity_x_px_s": report.get("velocity_x_px_s") if report.get("velocity_x_px_s") is not None else estimate.get("velocity_x_px_s"),
        "velocity_y_px_s": report.get("velocity_y_px_s") if report.get("velocity_y_px_s") is not None else estimate.get("velocity_y_px_s"),
        "speed_px_s": report.get("speed_px_s") if report.get("speed_px_s") is not None else estimate.get("speed_px_s"),
        "cumulative_slip_px": report.get("cumulative_slip_px") if report.get("cumulative_slip_px") is not None else estimate.get("cumulative_slip_px"),
        "downward_slip_px": report.get("downward_slip_px") if report.get("downward_slip_px") is not None else estimate.get("downward_slip_px"),
        "cumulative_downward_slip_px": report.get("cumulative_downward_slip_px") if report.get("cumulative_downward_slip_px") is not None else estimate.get("cumulative_downward_slip_px"),
        "slip_along_bottle_axis_px": report.get("slip_along_bottle_axis_px") if report.get("slip_along_bottle_axis_px") is not None else estimate.get("slip_along_bottle_axis_px"),
        "slip_across_bottle_axis_px": report.get("slip_across_bottle_axis_px") if report.get("slip_across_bottle_axis_px") is not None else estimate.get("slip_across_bottle_axis_px"),
        "slip_confidence": report.get("slip_confidence") if report.get("slip_confidence") is not None else estimate.get("slip_confidence"),
        "rotation_method": report.get("rotation_method") or estimate.get("rotation_method"),
        "rotation_delta_deg": report.get("rotation_delta_deg") if report.get("rotation_delta_deg") is not None else estimate.get("rotation_delta_deg"),
        "cumulative_rotation_deg": report.get("cumulative_rotation_deg") if report.get("cumulative_rotation_deg") is not None else estimate.get("cumulative_rotation_deg"),
        "angular_velocity_deg_s": report.get("angular_velocity_deg_s") if report.get("angular_velocity_deg_s") is not None else estimate.get("angular_velocity_deg_s"),
        "rotation_confidence": report.get("rotation_confidence") if report.get("rotation_confidence") is not None else estimate.get("rotation_confidence"),
        "roll_valid": report.get("roll_valid", False),
        "roll_delta_deg": report.get("roll_delta_deg"),
        "roll_status": report.get("roll_status") or "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
        "roll_exclusion_reason": report.get("roll_exclusion_reason") or "No calibrated 3D object orientation source is available.",
        "vibration_amplitude_px_rms": report.get("vibration_amplitude_px_rms") if report.get("vibration_amplitude_px_rms") is not None else estimate.get("vibration_amplitude_px_rms"),
        "vibration_frequency_hz": report.get("vibration_frequency_hz") if report.get("vibration_frequency_hz") is not None else estimate.get("vibration_frequency_hz"),
        "vibration_confidence": report.get("vibration_confidence") if report.get("vibration_confidence") is not None else estimate.get("vibration_confidence"),
        "disturbance_type": report.get("disturbance_type") or estimate.get("disturbance_type"),
        "disturbance_confidence": report.get("disturbance_confidence") or estimate.get("disturbance_confidence"),
        "disturbance_rejection_applied": bool(report.get("disturbance_rejection_applied") or estimate.get("disturbance_rejection_applied")),
        "quantitative_evidence": report.get("quantitative_evidence") or estimate.get("quantitative_evidence"),
        "closeup_window_metrics": report.get("closeup_window_metrics") or {},
        "simulated_force_command": report.get("simulated_force_command") or {},
        "boundary_event_field": {
            key: boundary.get(key)
            for key in (
                "boundary_source", "alignment_status", "measurement_status", "event_count",
                "assigned_event_count", "raw_window_event_count",
                "active_band_count", "tangential_circulation_px", "lower_vertical_drift_px",
                "oscillation_energy_px", "rotation_cue", "slip_cue", "vibration_cue",
                "incipient_slip_precursor_score", "object_motion_valid", "suppression_reason",
                "contact_band_count", "contact_band_active_count", "contact_band_event_count",
                "contact_tangential_flow_px", "contact_normal_flow_px", "contact_vertical_drift_px",
                "contact_oscillation_energy_px", "contact_incipient_slip_precursor_score",
                "contact_band_measurement_status",
                "physically_validated",
            )
        } if boundary else {},
    }


def build_grasp_quality_diagram_payload(matrix, live_quality, davis, post_grasp_baseline=None):
    """Build the bounded, honest payload for the operator wrench-space diagram.

    Geometry is copied from the existing validated epsilon result.  The live
    radius is only a visual encoding of the operational score ratio; it is not
    a new epsilon value and never mutates the analytic result.
    """

    matrix = matrix if isinstance(matrix, dict) else {}
    live_quality = live_quality if isinstance(live_quality, dict) else {}
    davis = davis if isinstance(davis, dict) else {}
    post_grasp_baseline = post_grasp_baseline if isinstance(post_grasp_baseline, dict) else {}
    live_baseline = live_quality.get("baseline") if isinstance(live_quality.get("baseline"), dict) else {}
    frozen_baseline_current = bool(
        post_grasp_baseline
        and post_grasp_baseline.get("baseline_frozen") is True
        and post_grasp_baseline.get("quality_baseline_role") == "IMMUTABLE_LOCK_BASELINE"
        and post_grasp_baseline.get("quality_input_source") == "planned_contact"
        and post_grasp_baseline.get("quality_contact_source_forced") is True
        and post_grasp_baseline.get("writer_session_id") is not None
        and post_grasp_baseline.get("attempt_id") is not None
        and post_grasp_baseline.get("lock_frame_id") is not None
        and post_grasp_baseline.get("quality_compute_id") is not None
        and first_number(post_grasp_baseline.get("quality_epsilon")) is not None
        and first_number(post_grasp_baseline.get("quality_score")) is not None
        and str(live_quality.get("analytic_quality_status") or "").upper() == "CURRENT"
        and str(live_baseline.get("writer_session_id")) == str(post_grasp_baseline.get("writer_session_id"))
        and str(live_baseline.get("lock_frame_id")) == str(post_grasp_baseline.get("lock_frame_id"))
        and str(live_baseline.get("quality_compute_id")) == str(post_grasp_baseline.get("quality_compute_id"))
    )
    matrix_matches_frozen_baseline = bool(
        frozen_baseline_current
        and matrix.get("writer_session_id") is not None
        and matrix.get("lock_frame_id") is not None
        and (matrix.get("lock_attempt_id") or matrix.get("attempt_id")) is not None
        and matrix.get("quality_compute_id") is not None
        and str(matrix.get("writer_session_id")) == str(post_grasp_baseline.get("writer_session_id"))
        and str(matrix.get("lock_frame_id")) == str(post_grasp_baseline.get("lock_frame_id"))
        and str(matrix.get("lock_attempt_id") or matrix.get("attempt_id")) == str(post_grasp_baseline.get("attempt_id"))
        and str(matrix.get("quality_compute_id")) == str(post_grasp_baseline.get("quality_compute_id"))
    )
    grasp_hulls = matrix.get("grasp_hulls") if isinstance(matrix.get("grasp_hulls"), dict) else {}
    hull = grasp_hulls.get("wrench_space_hull") if isinstance(grasp_hulls.get("wrench_space_hull"), dict) else {}
    if not hull and isinstance(matrix.get("wrench_space_hull"), dict):
        hull = matrix.get("wrench_space_hull")

    raw_points = hull.get("wrench_points") or grasp_hulls.get("wrench_vectors") or matrix.get("wrench_vectors") or []
    points = []
    point_index_map = {}
    for source_index, point in enumerate(raw_points[:512] if isinstance(raw_points, list) else []):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        xyz = [first_number(point[index]) for index in range(3)]
        if all(value is not None for value in xyz):
            point_index_map[source_index] = len(points)
            points.append(xyz)

    raw_faces = hull.get("hull_simplices") or grasp_hulls.get("hull_simplices") or matrix.get("hull_simplices") or []
    faces = []
    for face in raw_faces[:1024] if isinstance(raw_faces, list) else []:
        if not isinstance(face, (list, tuple)) or len(face) < 3:
            continue
        try:
            indices = [int(face[index]) for index in range(3)]
        except (TypeError, ValueError):
            continue
        if all(index in point_index_map for index in indices):
            faces.append([point_index_map[index] for index in indices])

    if frozen_baseline_current and not matrix_matches_frozen_baseline:
        # Never draw a hull from an absent/late matrix under a different frozen
        # baseline. The score stays current, but this diagram is explicitly
        # geometry-unavailable until the matching artifact is present.
        points = []
        faces = []

    weak_direction = hull.get("weakest_direction") or matrix.get("weak_direction_wrench") or matrix.get("weak_direction")
    if isinstance(weak_direction, (list, tuple)) and len(weak_direction) >= 3:
        weak_direction = [first_number(weak_direction[index]) for index in range(3)]
        if any(value is None for value in weak_direction):
            weak_direction = None
    else:
        weak_direction = None
    if frozen_baseline_current and not matrix_matches_frozen_baseline:
        weak_direction = None

    matrix_status = str(matrix.get("quality_matrix_status") or grasp_hulls.get("status") or "idle").upper()
    no_current_lock = bool(
        not frozen_baseline_current
        and str(matrix.get("contact_quality_gate_status") or "").upper() == "NO_CURRENT_LOCK"
    )
    quality_bundle_status = str(matrix.get("quality_bundle_status") or "").upper()
    stale = bool(
        not frozen_baseline_current
        and not no_current_lock
        and (
            matrix.get("quality_label_stale")
            or matrix.get("matrix_stale_due_to_motion")
            or matrix_status in {"STALE", "STALE_NEEDS_RECOMPUTE"}
            or quality_bundle_status == "STALE"
            or matrix.get("matrix_valid_for_current_pose") is False
        )
    )
    epsilon = (
        first_number(post_grasp_baseline.get("quality_epsilon"))
        if frozen_baseline_current
        else first_number(matrix.get("raw_epsilon"), matrix.get("epsilon"), matrix.get("epsilon_ball_radius"), grasp_hulls.get("epsilon"))
    )
    analytic_score = (
        first_number(post_grasp_baseline.get("quality_score"))
        if frozen_baseline_current
        else first_number(
            live_quality.get("analytic_quality_score"),
            matrix.get("normalized_quality_score"),
            matrix.get("epsilon_score"),
        )
    )
    if no_current_lock:
        # Never expose geometry or values from an older lock merely because a
        # stale artifact is still present on disk for engineering provenance.
        points = []
        faces = []
        weak_direction = None
        epsilon = None
        analytic_score = None
    geometry_valid = len(points) >= 4 and len(faces) >= 4
    matrix_computed = bool(
        matrix_status in {"UPDATED", "OK", "VALID", "CURRENT", "STALE", "STALE_NEEDS_RECOMPUTE"}
        or (frozen_baseline_current and matrix_matches_frozen_baseline)
    )
    analytic_available = bool(geometry_valid and epsilon is not None and epsilon >= 0.0 and matrix_computed)

    failure_reasons = matrix.get("quality_matrix_failure_reasons") or matrix.get("matrix_stale_reasons") or []
    if not isinstance(failure_reasons, list):
        failure_reasons = [failure_reasons]
    if frozen_baseline_current and not matrix_matches_frozen_baseline:
        diagram_status = "BASELINE_CURRENT_GEOMETRY_UNAVAILABLE"
        reason = "immutable post-grasp baseline is current, but matching wrench geometry is unavailable"
        action = "Continue monitoring the current score; the diagram will remain hidden rather than show another lock's geometry."
    elif no_current_lock:
        diagram_status = "UNAVAILABLE"
        reason = "waiting for a current bottle lock and its one-time analytic baseline"
        action = "Place one full bottle in view; the baseline will be captured automatically after lock."
    elif stale:
        diagram_status = "STALE"
        reason = matrix.get("quality_stale_reason") or matrix.get("quality_matrix_message") or (
            ", ".join(str(item) for item in failure_reasons if item) or "analytic quality no longer matches the current grasp pose"
        )
        action = "Reset and acquire a new lock; the immutable baseline is never recomputed in place."
    elif not analytic_available:
        diagram_status = "UNAVAILABLE"
        reason = (
            matrix.get("quality_matrix_error")
            or matrix.get("hull_failure_reason")
            or matrix.get("quality_matrix_message")
            or (", ".join(str(item) for item in failure_reasons if item))
            or ("computed wrench hull is unavailable" if not geometry_valid else "analytic epsilon is unavailable")
        )
        action = "Hold exactly one bottle steady until current RGB-D geometry passes the automatic lock and baseline gates."
    else:
        diagram_status = "CURRENT"
        reason = "locked RealSense contact geometry and analytic epsilon are current"
        action = "Monitor live operational quality."

    live_score = first_number(live_quality.get("live_quality_score"))
    live_available = bool(analytic_available and not stale and live_score is not None and analytic_score is not None)
    score_ratio = None
    live_radius = None
    delta = None
    live_state = "UNAVAILABLE"
    if live_available:
        score_ratio = 0.0 if analytic_score <= 0.0 else float(np.clip(live_score / analytic_score, 0.0, 1.0))
        live_radius = float(epsilon) * score_ratio
        delta = live_score - analytic_score
        live_state = "DEGRADED" if delta < -0.5 else ("IMPROVED" if delta > 0.5 else "UNCHANGED")

    deviations = live_quality.get("deviations") if isinstance(live_quality.get("deviations"), dict) else {}
    observation = live_quality.get("observation") if isinstance(live_quality.get("observation"), dict) else {}
    object_motion_valid = davis.get("object_motion_valid") is True
    grasp_zone_motion_valid = davis.get("grasp_zone_motion_valid") is True
    presentation_motion_valid = object_motion_valid or grasp_zone_motion_valid
    davis_used = bool(
        presentation_motion_valid
        and (
            live_quality.get("davis_motion_used_for_live_quality") is True
            or live_quality.get("davis_closeup_used_for_live_quality") is True
        )
    )
    exclusion_reason = (
        live_quality.get("davis_closeup_exclusion_reason")
        or live_quality.get("davis_motion_live_quality_reason")
        or davis.get("davis_motion_excluded_reason")
        or davis.get("object_motion_reason")
    )
    disturbance = str(davis.get("disturbance_type") or observation.get("davis_disturbance_type") or "NONE").upper()
    davis_stage = str(davis.get("current_stage") or davis.get("davis_motion_stage") or "PRE_CLOSE").upper()
    if not davis_used and disturbance in {"LIGHTING_CHANGE", "CAMERA_SHAKE", "BACKGROUND_MOTION"}:
        exclusion_reason = disturbance.lower()
    if davis_stage == "PRE_CLOSE":
        exclusion_reason = "pre_close_monitoring_only"

    return {
        "status": diagram_status,
        "reason": reason,
        "action": action,
        "coordinate_frame": (
            "unavailable_no_matching_frozen_hull"
            if frozen_baseline_current and not matrix_matches_frozen_baseline
            else hull.get("coordinate_frame") or "planar_wrench_space_fx_fy_tau"
        ),
        "axes": ["Fx", "Fy", "tau_z"],
        "geometry_source": (
            None
            if frozen_baseline_current and not matrix_matches_frozen_baseline
            else hull.get("source") or grasp_hulls.get("wrench_hull_source") or "epsilon_quality.compute_epsilon_quality"
        ),
        "analytic": {
            "available": analytic_available,
            "baseline_current": frozen_baseline_current,
            "geometry_matches_frozen_baseline": matrix_matches_frozen_baseline if frozen_baseline_current else None,
            "stale": stale,
            "quality_compute_id": (
                post_grasp_baseline.get("quality_compute_id")
                if frozen_baseline_current
                else matrix.get("quality_compute_id") or grasp_hulls.get("quality_compute_id")
            ),
            "quality_label": (
                live_quality.get("analytic_quality_label") or post_grasp_baseline.get("quality_label")
                if frozen_baseline_current
                else matrix.get("quality_label")
            ),
            "quality_score": analytic_score,
            "epsilon": epsilon,
            "epsilon_region_radius": epsilon if analytic_available else None,
            "force_closure": (
                None
                if frozen_baseline_current and not matrix_matches_frozen_baseline
                else matrix.get("force_closure_estimate", matrix.get("epsilon_force_closure", matrix.get("force_closure")))
            ),
            "contact_source": (
                "planned_contact"
                if frozen_baseline_current
                else matrix.get("contact_source_label") or matrix.get("contact_source") or matrix.get("quality_input_source")
            ),
            "wrench_points": points if analytic_available else [],
            "hull_faces": faces if analytic_available else [],
            "weak_direction": weak_direction if analytic_available else None,
            "weak_direction_semantics": "analytic weakest grasp-resistance direction in normalized planar wrench space",
            "measurement_status": "ESTIMATED_FROM_RGBD_CONTACT_GEOMETRY_NOT_PHYSICALLY_VALIDATED",
        },
        "live": {
            "available": live_available,
            "quality_score": live_score if live_available else None,
            "quality_label": live_quality.get("live_quality_label") if live_available else "UNAVAILABLE",
            "trend": live_quality.get("trend") or live_quality.get("live_quality_trend"),
            "state": live_state,
            "delta_from_analytic": delta,
            "operational_radius": live_radius,
            "operational_radius_ratio": score_ratio,
            "radius_semantics": "visual encoding of live_score / analytic_score; not analytic epsilon",
            "epsilon_recomputed": False,
            "primary_risk": live_quality.get("primary_risk") or live_quality.get("live_quality_primary_risk"),
            "decision": live_quality.get("decision") or live_quality.get("live_quality_action"),
            "formula": live_quality.get("formula") or "analytic score minus gated current-risk penalties; epsilon is not recomputed",
            "penalties": {
                "slip": round(float(first_number(deviations.get("incipient_slip"), 0.0) or 0.0) + float(first_number(deviations.get("gross_slip"), 0.0) or 0.0), 3) if davis_used else 0.0,
                "rotation": (first_number(deviations.get("rotation"), 0.0) or 0.0) if davis_used else 0.0,
                "vibration": (first_number(deviations.get("vibration"), 0.0) or 0.0) if davis_used else 0.0,
                "disturbance_uncertainty": first_number(deviations.get("uncertainty"), 0.0) or 0.0,
            },
        },
        "davis": {
            "used_for_live_quality": davis_used,
            "current_stage": davis_stage,
            "object_motion_valid": object_motion_valid,
            "grasp_zone_motion_valid": grasp_zone_motion_valid,
            "motion_attribution": davis.get("motion_attribution"),
            "exclusion_reason": None if davis_used else (exclusion_reason or "object motion evidence is not valid/current"),
            "motion_class": davis.get("raw_motion_class") or davis.get("motion_class"),
            "disturbance_type": disturbance,
            "confidence": first_number(live_quality.get("davis_closeup_confidence"), davis.get("motion_confidence")),
            "displacement_px": first_number(observation.get("davis_displacement_px"), davis.get("displacement_px")) if presentation_motion_valid else None,
            "rotation_deg": first_number(observation.get("davis_rotation_delta_deg"), davis.get("rotation_delta_deg")) if presentation_motion_valid else None,
            "roll_valid": davis.get("roll_valid") is True,
            "roll_deg": first_number(davis.get("roll_delta_deg")) if davis.get("roll_valid") is True else None,
            "roll_status": davis.get("roll_status") or "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
            "vibration_rms_px": first_number(observation.get("davis_vibration_rms_px"), davis.get("vibration_amplitude_px_rms")) if presentation_motion_valid else None,
            "wrench_direction_available": False,
            "wrench_direction_reason": "No calibrated event-motion-to-physical-wrench mapping; the diagram shows the analytic weak direction only.",
            "physically_validated": False,
        },
    }


def davis_web_state(report, running=False):
    report = report or {}
    reason = str(report.get("davis_unavailable_reason") or report.get("error") or "")
    stream = report.get("event_stream_status") or report.get("davis_status")
    aps_status = report.get("aps_stream_status")
    raw_event_activity = bool(
        int(report.get("raw_window_event_count") or 0) > 0
        or int(report.get("last_poll_event_count") or 0) > 0
    )
    if report.get("duplicate_davis_owner_detected"):
        return "ERROR"
    if running and not report:
        return "OPENING"
    if reason:
        lower = reason.lower()
        if "permission" in lower or "udev" in lower:
            return "PERMISSION_DENIED"
        if "not found" in lower or "not detected" in lower or "not connected" in lower or "no davis" in lower:
            return "DEVICE_NOT_FOUND"
        if "backend" in lower or "pyaer" in lower or "dv_processing" in lower:
            return "BACKEND_UNAVAILABLE"
        return "ERROR"
    event_active = stream in {"active", "monitoring", "complete"}
    aps_active = aps_status == "active" or bool((report.get("aps_status") or {}).get("aps_frame_valid"))
    if event_active and aps_active:
        return "STREAM_ACTIVE_EVENTS" if raw_event_activity else "STREAM_ACTIVE_NO_MOTION"
    if event_active:
        return "STREAM_ACTIVE_EVENTS" if raw_event_activity else "EVENTS_ONLY"
    if aps_active:
        return "APS_ONLY"
    if running:
        return "OPENING"
    return "NOT_CONNECTED"


def davis_event_state(report, running=False):
    report = report or {}
    state = davis_web_state(report, running=running)
    if state == "NOT_CONNECTED":
        return "DEVICE_NOT_FOUND"
    if state == "OPENING":
        return "STREAM_NOT_STARTED"
    if state == "APS_ONLY":
        return "STREAM_NOT_STARTED"
    if state in {"STREAM_ACTIVE_NO_MOTION", "STREAM_ACTIVE_EVENTS", "PERMISSION_DENIED", "BACKEND_UNAVAILABLE", "ERROR"}:
        return state
    if state == "EVENTS_ONLY":
        return (
            "STREAM_ACTIVE_EVENTS"
            if int(report.get("raw_window_event_count") or 0) > 0
            or int(report.get("last_poll_event_count") or 0) > 0
            else "STREAM_ACTIVE_NO_MOTION"
        )
    return "ERROR" if running else "DEVICE_NOT_FOUND"


def davis_aps_state(report):
    report = report or {}
    aps_state = report.get("aps_state")
    if aps_state in {"APS_COLOR_VERIFIED", "APS_RAW_BAYER", "APS_GRAYSCALE", "APS_UNKNOWN", "APS_UNAVAILABLE"}:
        return aps_state
    aps_status = report.get("aps_status") or {}
    mode = str(aps_status.get("aps_mode") or report.get("aps_mode") or "")
    if aps_status.get("aps_color_verified") or report.get("aps_color_verified"):
        return "APS_COLOR_VERIFIED"
    if "Raw Bayer" in mode:
        return "APS_RAW_BAYER"
    if "Grayscale" in mode:
        return "APS_GRAYSCALE"
    if "Unavailable" in mode or not report.get("aps_stream_status"):
        return "APS_UNAVAILABLE"
    return "APS_UNKNOWN"


def davis_status_payload():
    report = read_davis_report()
    running = bool(davis_thread and davis_thread.is_alive())
    summary = summarize_davis(report)
    try:
        report_age_sec = max(0.0, time.time() - DAVIS_LATEST_MOTION_REPORT_PATH.stat().st_mtime)
    except OSError:
        report_age_sec = None
    report_current = bool(
        running
        and report_age_sec is not None
        and report_age_sec <= max(3.0, 3.0 * DAVIS_REPORT_INTERVAL_SEC)
    )
    try:
        from grasp_project.davis.event_stream import read_davis_owner_lock
        owner = read_davis_owner_lock()
    except Exception as exc:
        owner = {"davis_owner_status": "owner_status_unavailable", "davis_owner_error": str(exc)}
    owner_status = owner.get("davis_owner_status")
    owner_stale = owner_status == "stale_owner" and not running
    state = (
        "DISCONNECTED"
        if owner_stale
        else (davis_web_state(report, running=running) if report_current else "STREAM_NOT_STARTED")
    )
    object_state = object_state_from_status(
        read_json(OUTPUT_DIR / "latest_realtime_status.json", {}),
        read_json(OUTPUT_DIR / "pointcloud_status.json", {}),
    ).to_dict()
    payload = {
        "ok": True,
        "state": state,
        "running": bool(running and not owner_stale),
        "desired": bool(desired_processes.get("davis_motion")),
        "report": report,
        "summary": summary,
        "report_current": report_current,
        "report_age_sec": report_age_sec,
        "report_status": "CURRENT" if report_current else "HISTORICAL_NOT_CURRENT",
        "object_state": object_state,
        "owner": owner,
        "davis_writer_pid": owner.get("davis_writer_pid") or report.get("davis_writer_pid"),
        "davis_writer_session_id": owner.get("davis_writer_session_id") or report.get("davis_writer_session_id"),
        "davis_device_open_count": owner.get("davis_device_open_count") if owner.get("davis_device_open_count") is not None else (report.get("davis_device_open_count") or 0),
        "davis_owner_status": owner_status or report.get("davis_owner_status"),
        "duplicate_davis_owner_detected": bool(owner.get("duplicate_davis_owner_detected") or report.get("duplicate_davis_owner_detected")),
        "duplicate_open_prevented": bool(owner.get("duplicate_open_prevented") or report.get("duplicate_open_prevented")),
        "backend": (report.get("backend") or report.get("davis_backend")) if report_current else None,
        "davis_backend": (report.get("davis_backend") or report.get("backend")) if report_current else None,
        "event_stream_status": "stale_owner" if owner_stale else ((report.get("event_stream_status") or report.get("davis_status")) if report_current else "not_running"),
        "event_state": "DISCONNECTED" if owner_stale else (davis_event_state(report, running=running) if report_current else "STREAM_NOT_STARTED"),
        "aps_stream_status": report.get("aps_stream_status") if report_current else "not_running",
        "aps_state": davis_aps_state(report) if report_current else "APS_UNAVAILABLE",
        "aps_fps": report.get("aps_fps") if report_current else None,
        "event_rate": (((report.get("current_metrics") or {}).get("event_rate") or report.get("event_rate")) if report_current else None),
        "motion_class": report.get("motion_class") if report_current else None,
        "motion_estimate_valid": summary.get("motion_estimate_valid") if report_current else False,
        "motion_estimate_invalid_reason": summary.get("motion_estimate_invalid_reason") if report_current else "stale_events",
        "motion_estimation_method": summary.get("motion_estimation_method") if report_current else None,
        "motion_estimation_confidence": summary.get("motion_estimation_confidence") if report_current else None,
        "motion_confidence": summary.get("motion_confidence") if report_current else None,
        "dx_px": summary.get("dx_px") if report_current else None,
        "dy_px": summary.get("dy_px") if report_current else None,
        "displacement_px": summary.get("displacement_px") if report_current else None,
        "cumulative_slip_px": summary.get("cumulative_slip_px") if report_current else None,
        "downward_slip_px": summary.get("downward_slip_px") if report_current else None,
        "rotation_delta_deg": summary.get("rotation_delta_deg") if report_current else None,
        "cumulative_rotation_deg": summary.get("cumulative_rotation_deg") if report_current else None,
        "angular_velocity_deg_s": summary.get("angular_velocity_deg_s") if report_current else None,
        "vibration_amplitude_px_rms": summary.get("vibration_amplitude_px_rms") if report_current else None,
        "disturbance_type": summary.get("disturbance_type") if report_current else None,
        "decision": (report.get("decision") or report.get("post_grasp_decision")) if report_current else "UNAVAILABLE",
        "error": "DAVIS owner lock is stale." if owner_stale else (report.get("error") or davis_last_error),
        "api_source": "main_web_dashboard",
    }
    return payload


def summarize_combined_quality(
    matrix,
    davis,
    *,
    post_grasp_active=False,
    live_quality=None,
    post_grasp_baseline=None,
):
    matrix = matrix or {}
    davis = davis or {}
    live_quality = live_quality if isinstance(live_quality, dict) else {}
    post_grasp_baseline = post_grasp_baseline if isinstance(post_grasp_baseline, dict) else {}
    live_baseline = live_quality.get("baseline") if isinstance(live_quality.get("baseline"), dict) else {}
    frozen_baseline_current = bool(
        post_grasp_active
        and post_grasp_baseline.get("baseline_frozen") is True
        and post_grasp_baseline.get("quality_baseline_role") == "IMMUTABLE_LOCK_BASELINE"
        and post_grasp_baseline.get("quality_input_source") == "planned_contact"
        and post_grasp_baseline.get("quality_contact_source_forced") is True
        and post_grasp_baseline.get("quality_compute_id") is not None
        and first_number(post_grasp_baseline.get("quality_score")) is not None
        and str(live_quality.get("analytic_quality_status") or "").upper() == "CURRENT"
        and str(live_baseline.get("writer_session_id")) == str(post_grasp_baseline.get("writer_session_id"))
        and str(live_baseline.get("lock_frame_id")) == str(post_grasp_baseline.get("lock_frame_id"))
        and str(live_baseline.get("quality_compute_id")) == str(post_grasp_baseline.get("quality_compute_id"))
    )
    epsilon_score = (
        first_number(post_grasp_baseline.get("quality_score"))
        if frozen_baseline_current
        else first_number(matrix.get("normalized_quality_score"), matrix.get("epsilon_score"))
    )
    matrix_status = "updated" if frozen_baseline_current else matrix.get("quality_matrix_status", "idle")
    reported_motion_class = (
        live_quality.get("davis_closeup_motion_class")
        or davis.get("raw_motion_class")
        or davis.get("motion_class")
        or davis.get("post_grasp_motion_class")
        or davis.get("davis_safety_class")
    )
    davis_motion_usable = bool(
        post_grasp_active
        and (
            live_quality.get("davis_motion_used_for_live_quality") is True
            if frozen_baseline_current
            else davis.get("davis_motion_usable_for_grasp") is True
        )
    )
    motion_class = reported_motion_class if davis_motion_usable else None
    slip = float(davis.get("gross_slip") or davis.get("slip_risk_score") or davis.get("final_slip_risk_score") or 0.0) if davis_motion_usable else 0.0
    rotation = float(davis.get("rotation") or davis.get("rotation_risk_score") or 0.0) if davis_motion_usable else 0.0
    vibration = float(davis.get("vibration") or davis.get("vibration_risk_score") or 0.0) if davis_motion_usable else 0.0
    penalty = max(slip, rotation, vibration)
    if (
        not frozen_baseline_current
        and (
            matrix_status == "STALE_NEEDS_RECOMPUTE"
            or matrix.get("matrix_stale_due_to_motion")
        )
    ):
        status = "MATRIX_STALE_RECOMPUTE"
        reason = matrix.get("quality_matrix_message") or "locked quality matrix is stale relative to current bottle geometry"
    elif not frozen_baseline_current and matrix_status == "failed":
        status = "INITIAL_QUALITY_FAILED"
        reason = matrix.get("quality_matrix_error") or "quality matrix failed"
    elif epsilon_score is None:
        status = "WAITING_FOR_INITIAL_QUALITY"
        reason = "initial epsilon matrix not available yet"
    elif motion_class in {"GROSS_SLIP", "SLIP_DOWN"} or slip >= 70.0:
        status = "SLIP_RISK"
        reason = "DAVIS post-grasp motion indicates slip risk"
    elif motion_class == "SPIN_OR_ROTATION" or rotation >= 65.0:
        status = "ROTATION_RISK"
        reason = "DAVIS post-grasp motion indicates rotation risk"
    elif motion_class in {"SHAKE_OR_VIBRATION", "UNKNOWN_MOTION"} or vibration >= 70.0:
        status = "UNSAFE_REGRASP"
        reason = "DAVIS post-grasp motion is unstable"
    elif motion_class in {"MICRO_SLIP", "INCIPIENT_SLIP"} or penalty >= 35.0:
        status = "MICRO_ADJUST_NEEDED"
        reason = "small event motion detected after initial grasp"
    elif matrix_status == "updated" and float(epsilon_score) >= 50.0:
        status = "STABLE_AFTER_GRASP" if motion_class in {"STABLE", "IMPACT_ONLY"} else "INITIAL_GOOD"
        reason = "initial RealSense epsilon is available; DAVIS does not currently show unsafe motion"
    else:
        status = "INITIAL_GOOD"
        reason = "initial quality is available but below preferred score"
    return {
        "combined_quality_status": status,
        "initial_epsilon_score": epsilon_score,
        "analytic_quality_status": "CURRENT" if frozen_baseline_current else str(matrix_status).upper(),
        "quality_compute_id": (
            post_grasp_baseline.get("quality_compute_id")
            if frozen_baseline_current
            else matrix.get("quality_compute_id")
        ),
        "baseline_source": "IMMUTABLE_POST_GRASP_SNAPSHOT" if frozen_baseline_current else "QUALITY_MATRIX_ARTIFACT",
        "davis_motion_class": reported_motion_class,
        "davis_motion_used_for_quality": davis_motion_usable,
        "davis_motion_excluded_reason": None if davis_motion_usable else (
            davis.get("davis_motion_excluded_reason")
            or davis.get("object_motion_reason")
            or ("pre_close_monitoring_only" if not post_grasp_active else "object_motion_not_validated")
        ),
        "davis_motion_penalty": round(penalty, 1),
        "suggested_correction_direction": davis.get("suggested_correction_direction") or davis.get("movement_direction") or "N/A",
        "reason": reason,
    }


def quality_plots_are_stale(matrix):
    matrix = matrix or {}
    status = str(matrix.get("quality_matrix_status") or "").lower()
    label = str(matrix.get("quality_label") or matrix.get("quality_label_display") or "").lower()
    if status in {"failed", "invalid"} or "invalid" in label:
        return True, "PLOT STALE - CURRENT MATRIX FAILED"
    if matrix.get("quality_label_stale") or matrix.get("matrix_stale_due_to_motion"):
        return True, "PLOT INVALID - RESET / NEW LOCK"
    return False, ""


def validate_quality_bundle_for_live_status(bundle, status, status_age_sec=None):
    """Return a bundle that cannot claim CURRENT without its originating live lock."""
    bundle = dict(bundle or {})
    status = status or {}
    if not bundle:
        return bundle
    live_lock = status.get("target_lock") or {}
    live_writer_session_id = status.get("writer_session_id")
    live_lock_frame_id = status.get("lock_frame_id") or live_lock.get("lock_frame_id")
    live_lock_attempt_id = (
        status.get("lock_attempt_id")
        or status.get("attempt_id")
        or live_lock.get("attempt_id")
    )
    live_locked_band = status.get("locked_grasp_band") or live_lock.get("locked_grasp_band") or {}
    live_quality_compute_id = status.get("quality_compute_id") or live_locked_band.get("locked_quality_compute_id")
    live_target_status = str(status.get("target_status") or live_lock.get("target_status") or "").upper()
    bundle_writer_session_id = bundle.get("writer_session_id")
    bundle_lock_frame_id = bundle.get("lock_frame_id")
    bundle_lock_attempt_id = bundle.get("lock_attempt_id") or bundle.get("attempt_id")
    bundle_quality_compute_id = bundle.get("quality_compute_id")
    post_grasp_baseline = (
        status.get("post_grasp_baseline")
        if isinstance(status.get("post_grasp_baseline"), dict)
        else {}
    )
    post_grasp_frozen_baseline = bool(
        status.get("post_grasp_active") is True
        and post_grasp_baseline
        and post_grasp_baseline.get("writer_session_id")
        and str(post_grasp_baseline.get("writer_session_id")) == str(live_writer_session_id)
        and post_grasp_baseline.get("lock_frame_id") is not None
        and str(post_grasp_baseline.get("lock_frame_id")) == str(live_lock_frame_id)
        and post_grasp_baseline.get("attempt_id")
        and str(post_grasp_baseline.get("attempt_id")) == str(live_lock_attempt_id)
        and (
            bundle_writer_session_id is None
            or str(post_grasp_baseline.get("writer_session_id")) == str(bundle_writer_session_id)
        )
        and (
            bundle_lock_frame_id is None
            or str(post_grasp_baseline.get("lock_frame_id")) == str(bundle_lock_frame_id)
        )
        and (
            bundle_lock_attempt_id is None
            or str(post_grasp_baseline.get("attempt_id")) == str(bundle_lock_attempt_id)
        )
        and (
            not post_grasp_baseline.get("quality_compute_id")
            or not bundle_quality_compute_id
            or str(post_grasp_baseline.get("quality_compute_id")) == str(bundle_quality_compute_id)
        )
    )
    relock_required = bool(
        status.get("grasp_band_relock_required")
        or status.get("band_relock_required")
        or live_lock.get("grasp_band_relock_required")
        or live_lock.get("band_relock_required")
    )
    same_lock_lineage = bool(
        live_writer_session_id
        and bundle_writer_session_id
        and str(live_writer_session_id) == str(bundle_writer_session_id)
        and live_lock_frame_id is not None
        and bundle_lock_frame_id is not None
        and str(live_lock_frame_id) == str(bundle_lock_frame_id)
        and live_lock_attempt_id
        and bundle_lock_attempt_id
        and str(live_lock_attempt_id) == str(bundle_lock_attempt_id)
    )
    original_bundle_status = str(bundle.get("quality_bundle_status") or "").upper()
    claims_current_lock = original_bundle_status in {"CURRENT", "PARTIAL"}
    retained_low_confidence_lock = bool(
        live_target_status in {"LOW_CONFIDENCE", "TRACKING_UNCERTAIN"}
        and (
            status.get("grasp_band_locked") is True
            or live_lock.get("grasp_band_locked") is True
        )
        and (
            status.get("grasp_band_projected") is True
            or status.get("grasp_band_held") is True
            or live_lock.get("grasp_band_projected") is True
            or live_lock.get("grasp_band_held") is True
        )
        and not relock_required
    )
    reasons = list(bundle.get("stale_reasons") or [])
    if status_age_sec is None or not np.isfinite(float(status_age_sec)):
        reasons.append("realtime_status_missing")
    elif float(status_age_sec) > STALE_FILE_THRESHOLD_SEC:
        reasons.append("realtime_status_stale")
    if claims_current_lock:
        if not live_writer_session_id:
            reasons.append("live_writer_session_missing")
        elif not bundle_writer_session_id:
            reasons.append("writer_session_provenance_missing")
        elif live_writer_session_id != bundle_writer_session_id:
            reasons.append("writer_session_mismatch")
        if live_lock_frame_id is None or (
            live_target_status not in {"LOCKED", "TRACKING"}
            and not retained_low_confidence_lock
            and not post_grasp_frozen_baseline
        ):
            reasons.append("no_active_lock_for_quality")
        elif bundle_lock_frame_id is None:
            reasons.append("lock_frame_provenance_missing")
        elif live_lock_frame_id != bundle_lock_frame_id:
            reasons.append("lock_frame_mismatch")
        if bundle_lock_attempt_id and not live_lock_attempt_id:
            reasons.append("live_lock_attempt_missing")
        elif live_lock_attempt_id and bundle_lock_attempt_id is None:
            reasons.append("lock_attempt_provenance_missing")
        elif (
            live_lock_attempt_id
            and bundle_lock_attempt_id
            and live_lock_attempt_id != bundle_lock_attempt_id
        ):
            reasons.append("lock_attempt_mismatch")
        if (
            live_quality_compute_id
            and bundle_quality_compute_id
            and str(live_quality_compute_id) != str(bundle_quality_compute_id)
        ):
            if (
                same_lock_lineage
                and not relock_required
                and status.get("post_grasp_active") is not True
            ):
                bundle["same_lock_quality_refresh_in_progress"] = True
            else:
                reasons.append("quality_compute_id_mismatch")
        if relock_required:
            reasons.append("grasp_band_relock_required")
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        if original_bundle_status != "FAILED":
            bundle["quality_bundle_status"] = "STALE"
        bundle["dashboard_artifact_fresh"] = False
        bundle["matrix_valid_for_current_pose"] = False
        bundle["stale_reasons"] = reasons
        bundle["quality_stale_reason"] = "; ".join(reasons)
    bundle["post_grasp_frozen_baseline_active"] = post_grasp_frozen_baseline
    return bundle


def quality_image_info(final_files, name, url, matrix, bundle=None, *, advanced=False):
    bundle = bundle or {}
    exists = bool((final_files.get(name) or {}).get("exists", False))
    compute_id = bundle.get("quality_compute_id") or (matrix or {}).get("quality_compute_id")
    bundle_status = str(bundle.get("quality_bundle_status") or "").upper()
    artifact = (bundle.get("artifact_paths") or {}).get(name) or {}
    if (
        bundle_status not in {"CURRENT", "PARTIAL"}
        or artifact.get("generated_for_current_compute") is not True
    ):
        exists = False
    if bundle_status == "FAILED":
        exists = False
    stale = False
    label = ""
    if bundle_status == "STALE":
        if advanced:
            stale = True
            label = "PREVIOUS LOCK - NOT CURRENT"
        else:
            exists = False
    elif bundle_status == "PARTIAL" and advanced:
        stale = True
        label = "RENDER PARTIAL"
    return {
        "exists": exists,
        "url": url,
        "stale": bool(exists and stale),
        "stale_label": label if exists and stale else "",
        "quality_compute_id": compute_id,
        "quality_bundle_status": bundle.get("quality_bundle_status"),
        "version": compute_id,
        "writer_session_id": (matrix or {}).get("writer_session_id"),
        "lock_frame_id": (matrix or {}).get("lock_frame_id") or (matrix or {}).get("hull_lock_frame_id"),
    }


def summarize_profile(realtime):
    rs_perf = realtime.get("performance_profile", {}).get("timings_ms", {}) if realtime else {}
    rs_audit = realtime.get("realtime_pipeline_audit", {}) if realtime else {}
    rs_output_timing = read_json(OUTPUT_DIR / "rs_output_timing_breakdown.json", {})
    return {
        "realsense_audit": rs_audit,
        "realsense_capture": rs_perf.get("camera_capture") or rs_perf.get("capture"),
        "realsense_preprocessing": rs_perf.get("preprocessing") or rs_perf.get("depth_stats_mask_region") or rs_perf.get("visibility_check"),
        "realsense_segmentation": (
            rs_perf.get("pose_detection")
            or rs_perf.get("yolo_trigger_pose")
            or rs_perf.get("bottle_detection_yolo_rgbd_fusion")
            or rs_perf.get("bottle_detection_rgbd_yolo_no_match")
            or rs_perf.get("bottle_detection_yolo_roi_opencv_depth")
            or rs_perf.get("bottle_detection_opencv_depth")
            or rs_perf.get("bottle_detection_full_frame")
            or rs_perf.get("bottle_detection_roi")
            or rs_perf.get("bottle_detection_roi_scaled")
        ),
        "realsense_depth_processing": rs_perf.get("depth_smoothing"),
        "realsense_contact_modeling": rs_perf.get("contact_modeling") or rs_perf.get("visual_contact_model") or rs_perf.get("candidate_generation_planning") or rs_perf.get("clamp_patch_update"),
        "realsense_drawing": rs_perf.get("overlay_drawing") or rs_perf.get("debug_overlay") or rs_perf.get("gui_overlay_render"),
        "realsense_epsilon": rs_perf.get("epsilon_calculation"),
        "realsense_safety_scoring": rs_perf.get("safety_scoring"),
        "realsense_output_write": rs_perf.get("image_saving"),
        "realsense_output_breakdown": rs_output_timing.get("breakdown_ms", {}),
        "realsense_output_files_written": rs_output_timing.get("files_written", []),
        "realsense_live_jpeg_output": rs_perf.get("web_live_jpeg_output"),
        "realsense_total": rs_perf.get("total_with_io") or rs_perf.get("total"),
        "dashboard_routes": route_timing_snapshot(),
        "web": profile_snapshot(),
        "streams": {name: buffer.stats() for name, buffer in stream_buffers.items()},
    }


def summarize_pointcloud(status, realtime=None, backend_running=None):
    status = status or {}
    realtime = realtime or {}
    active = bool(status.get("realsense_pointcloud_active"))
    last_update = status.get("pointcloud_timestamp") or status.get("realsense_pointcloud_last_update_time")
    try:
        age_sec = None if not active or last_update is None else round(max(0.0, time.time() - float(last_update)), 2)
    except Exception:
        age_sec = None
    if age_sec is None:
        freshness = "UNAVAILABLE"
    elif age_sec <= 2.0:
        freshness = "CURRENT"
    elif age_sec <= 5.0:
        freshness = "AGING"
    else:
        freshness = "STALE"
    writer_session = status.get("backend_writer_session_id")
    current_writer = realtime.get("writer_session_id")
    session_matches = None
    if writer_session and current_writer:
        session_matches = writer_session == current_writer
        if not session_matches:
            freshness = "ORPHANED"
    if backend_running is False and age_sec is not None and age_sec > 5.0:
        freshness = "ORPHANED"
    pointcloud_current = active and freshness == "CURRENT" and (session_matches is not False)
    scene_ply = OUTPUT_DIR / "latest_realsense_pointcloud.ply"
    bottle_ply = OUTPUT_DIR / "latest_bottle_pointcloud.ply"
    scene_preview = OUTPUT_DIR / "latest_pointcloud_preview.jpg"
    bottle_preview = OUTPUT_DIR / "latest_bottle_pointcloud_preview.jpg"
    context_preview = OUTPUT_DIR / "latest_pointcloud_context.jpg"
    live_json = OUTPUT_DIR / "latest_pointcloud_live.json"
    return {
        "active": active,
        "backend": status.get("realsense_pointcloud_backend") or status.get("selected_pointcloud_backend"),
        "fps": status.get("realsense_pointcloud_fps"),
        "last_update_time": status.get("realsense_pointcloud_last_update_time"),
        "age_sec": age_sec,
        "freshness": freshness,
        "pointcloud_current": pointcloud_current,
        "backend_writer_session_id": writer_session,
        "backend_running": bool(backend_running) if backend_running is not None else status.get("backend_running"),
        "backend_session_matches": session_matches,
        "viewer_watermark": (
            "Point cloud unavailable - no current cloud"
            if freshness == "UNAVAILABLE"
            else (
                "Last known point cloud - not current"
                if freshness in {"STALE", "ORPHANED"}
                else ("Point cloud aging - refresh soon" if freshness == "AGING" else None)
            )
        ),
        "coordinate_frame": status.get("coordinate_frame") or "realsense_camera_m",
        "coordinate_frame_type": status.get("coordinate_frame_type") or "current_camera_frame",
        "coordinate_frame_stable": status.get("coordinate_frame_stable", False),
        "world_registered": status.get("world_registered", False),
        "multiview_registered": status.get("multiview_registered", False),
        "pointcloud_frame_id": status.get("pointcloud_frame_id"),
        "pointcloud_session_id": status.get("pointcloud_session_id"),
        "pointcloud_generation_id": status.get("pointcloud_generation_id"),
        "pointcloud_timestamp": status.get("pointcloud_timestamp"),
        "pointcloud_source_rgb_frame_id": status.get("pointcloud_source_rgb_frame_id"),
        "pointcloud_source_depth_frame_id": status.get("pointcloud_source_depth_frame_id"),
        "rolling_window_active": bool(status.get("rolling_window_active")),
        "rolling_cloud_age_sec": status.get("rolling_cloud_age_sec"),
        "rolling_cloud_window_sec": status.get("rolling_cloud_window_sec"),
        "rolling_cloud_point_count": status.get("rolling_cloud_point_count"),
        "rolling_observation_count": status.get("rolling_observation_count"),
        "rolling_reset_reason": status.get("rolling_reset_reason"),
        "rolling_valid_for_geometry": bool(status.get("rolling_valid_for_geometry")),
        "rolling_valid_for_contact": bool(status.get("rolling_valid_for_contact")),
        "rolling_valid_for_benchmark": bool(status.get("rolling_valid_for_benchmark")),
        "rolling_pointcloud_active": status.get("rolling_pointcloud_active"),
        "rolling_pointcloud_status": status.get("rolling_pointcloud_status"),
        "rolling_pointcloud_mode": status.get("rolling_pointcloud_mode"),
        "rolling_pointcloud_age_sec": status.get("rolling_pointcloud_age_sec"),
        "rolling_pointcloud_window_sec": status.get("rolling_pointcloud_window_sec"),
        "rolling_pointcloud_observation_span_sec": status.get("rolling_pointcloud_observation_span_sec"),
        "rolling_pointcloud_point_count": status.get("rolling_pointcloud_point_count"),
        "rolling_pointcloud_merged_point_count": status.get("rolling_pointcloud_merged_point_count"),
        "rolling_pointcloud_observation_count": status.get("rolling_pointcloud_observation_count"),
        "rolling_pointcloud_latest_frame_id": status.get("rolling_pointcloud_latest_frame_id"),
        "rolling_pointcloud_target_id": status.get("rolling_pointcloud_target_id"),
        "rolling_pointcloud_target_id_scope": status.get("rolling_pointcloud_target_id_scope"),
        "rolling_pointcloud_registration_status": status.get("rolling_pointcloud_registration_status"),
        "rolling_pointcloud_last_reset_reason": status.get("rolling_pointcloud_last_reset_reason"),
        "rolling_pointcloud_last_reset_timestamp": status.get("rolling_pointcloud_last_reset_timestamp"),
        "rolling_pointcloud_last_rejection_reason": status.get("rolling_pointcloud_last_rejection_reason"),
        "rolling_pointcloud_valid_for_geometry": bool(status.get("rolling_pointcloud_valid_for_geometry")),
        "rolling_pointcloud_valid_for_contact": bool(status.get("rolling_pointcloud_valid_for_contact")),
        "rolling_pointcloud_valid_for_benchmark": bool(status.get("rolling_pointcloud_valid_for_benchmark")),
        "rolling_pointcloud_physically_validated": bool(status.get("rolling_pointcloud_physically_validated")),
        "point_count": status.get("realsense_pointcloud_point_count") or status.get("point_count"),
        "quality": status.get("realsense_pointcloud_quality") or status.get("pointcloud_quality_score"),
        "valid_depth_ratio": status.get("valid_depth_ratio"),
        "rgb_available": status.get("pointcloud_rgb_available"),
        "depth_available": status.get("pointcloud_depth_available"),
        "intrinsics_available": status.get("pointcloud_intrinsics_available"),
        "raw_point_count": status.get("pointcloud_raw_point_count"),
        "filtered_point_count": status.get("pointcloud_filtered_point_count"),
        "preview_point_count": status.get("pointcloud_preview_point_count"),
        "scene_raw_point_count": status.get("scene_raw_point_count") or status.get("pointcloud_raw_point_count"),
        "scene_filtered_point_count": status.get("scene_filtered_point_count") or status.get("pointcloud_filtered_point_count"),
        "scene_display_point_count": status.get("scene_display_point_count") or status.get("pointcloud_json_point_count") or status.get("pointcloud_preview_point_count"),
        "depth_min_mm": status.get("pointcloud_depth_min_mm"),
        "depth_max_mm": status.get("pointcloud_depth_max_mm"),
        "depth_valid_ratio": status.get("pointcloud_depth_valid_ratio"),
        "worker_busy": status.get("pointcloud_worker_busy"),
        "last_duration_ms": status.get("pointcloud_last_duration_ms"),
        "skipped_updates": status.get("pointcloud_skipped_updates"),
        "bottleneck": status.get("pointcloud_bottleneck"),
        "bottle_roi_available": status.get("bottle_roi_available"),
        "bottle_cloud_valid": status.get("bottle_cloud_valid"),
        "bottle_point_count": status.get("bottle_point_count"),
        "bottle_pointcloud_valid": status.get("bottle_pointcloud_valid"),
        "bottle_pointcloud_reason": status.get("bottle_pointcloud_reason"),
        "bottle_crop_reason": status.get("bottle_crop_reason"),
        "bottle_crop_status": status.get("bottle_crop_status"),
        "bottle_crop_confidence": status.get("bottle_crop_confidence"),
        "bottle_crop_quality": status.get("bottle_crop_quality"),
        "bottle_crop_degraded": status.get("bottle_crop_degraded"),
        "bottle_crop_safe_for_dimensions": status.get("bottle_crop_safe_for_dimensions"),
        "bottle_crop_safe_for_contacts": status.get("bottle_crop_safe_for_contacts"),
        "bottle_crop_safe_for_quality_support": status.get("bottle_crop_safe_for_quality_support"),
        "bottle_mask_pixel_count": status.get("bottle_mask_pixel_count"),
        "bottle_mask_available": status.get("bottle_mask_available"),
        "bottle_mask_source": status.get("bottle_mask_source"),
        "bottle_mask_frame_id": status.get("bottle_mask_frame_id"),
        "bottle_mask_matches_rgb_frame": status.get("bottle_mask_matches_rgb_frame"),
        "bottle_mask_shape_valid": status.get("bottle_mask_shape_valid"),
        "bottle_mask_rejection_reason": status.get("bottle_mask_rejection_reason"),
        "bottle_depth_valid_pixel_count": status.get("bottle_depth_valid_pixel_count"),
        "bottle_crop_fraction_of_scene": status.get("bottle_crop_fraction_of_scene"),
        "bottle_connected_component_count": status.get("bottle_connected_component_count"),
        "bottle_pointcloud_roi_source": status.get("bottle_pointcloud_roi_source") or status.get("bottle_roi_source"),
        "bottle_roi_source": status.get("bottle_roi_source") or status.get("bottle_pointcloud_roi_source"),
        "bottle_pointcloud_point_count": status.get("bottle_pointcloud_point_count") or status.get("bottle_point_count"),
        "bottle_raw_point_count": status.get("bottle_raw_point_count"),
        "bottle_filtered_point_count": status.get("bottle_filtered_point_count"),
        "bottle_display_point_count": status.get("bottle_display_point_count") or status.get("pointcloud_json_bottle_point_count"),
        "bottle_pointcloud_valid_depth_ratio": status.get("bottle_pointcloud_valid_depth_ratio"),
        "bottle_pointcloud_centroid_mm": status.get("bottle_pointcloud_centroid_mm"),
        "bottle_pointcloud_extents_mm": status.get("bottle_pointcloud_extents_mm"),
        "bottle_3d_centroid_m": status.get("bottle_3d_centroid_m"),
        "bottle_3d_extents_m": status.get("bottle_3d_extents_m"),
        "centroid_xyz_mm": status.get("centroid_xyz_mm"),
        "robust_centroid_xyz_mm": status.get("robust_centroid_xyz_mm"),
        "axis_aligned_extents_mm": status.get("axis_aligned_extents_mm"),
        "oriented_extents_mm": status.get("oriented_extents_mm"),
        "bottle_principal_axis_m": status.get("bottle_principal_axis_m"),
        "principal_axis": status.get("principal_axis"),
        "geometry_profile_score": status.get("geometry_profile_score"),
        "geometry_valid": status.get("geometry_valid"),
        "geometry_validation_reason": status.get("geometry_validation_reason"),
        "bottle_point_density": status.get("bottle_point_density"),
        "bottle_compactness": status.get("bottle_compactness"),
        "bottle_dimensions_valid": status.get("bottle_dimensions_valid"),
        "bottle_dimensions_warning": status.get("bottle_dimensions_warning"),
        "left_contact_xyz_mm": status.get("left_contact_xyz_mm"),
        "right_contact_xyz_mm": status.get("right_contact_xyz_mm"),
        "contact_width_3d_mm": status.get("contact_width_3d_mm"),
        "contact_depth_valid": status.get("contact_depth_valid"),
        "contact_points_on_bottle_cloud": status.get("contact_points_on_bottle_cloud"),
        "contacts_on_filtered_bottle_cloud": status.get("contacts_on_filtered_bottle_cloud"),
        "contacts_on_opposite_sides": status.get("contacts_on_opposite_sides"),
        "left_contact_cloud_distance_mm": status.get("left_contact_cloud_distance_mm"),
        "right_contact_cloud_distance_mm": status.get("right_contact_cloud_distance_mm"),
        "contact_height_difference_mm": status.get("contact_height_difference_mm"),
        "contact_3d_valid": status.get("contact_3d_valid") if pointcloud_current else False,
        "contact_3d_reason": status.get("contact_3d_reason"),
        "contact_3d_quality": status.get("contact_3d_quality"),
        "cloud_valid_for_visualization": status.get("cloud_valid_for_visualization"),
        "cloud_valid_for_dimension_estimation": bool(status.get("cloud_valid_for_dimension_estimation") and pointcloud_current),
        "cloud_valid_for_contact_validation": bool(status.get("cloud_valid_for_contact_validation") and pointcloud_current),
        "cloud_valid_for_multiview_input": bool(status.get("cloud_valid_for_multiview_input") and pointcloud_current),
        "pointcloud_snapshot_consistent": status.get("pointcloud_snapshot_consistent"),
        "pointcloud_snapshot_max_frame_delta": status.get("pointcloud_snapshot_max_frame_delta"),
        "pointcloud_snapshot_rejection_reason": status.get("pointcloud_snapshot_rejection_reason"),
        "status": status.get("pointcloud_status") or ("active" if status.get("realsense_pointcloud_active") else "waiting"),
        "error": status.get("pointcloud_error") or status.get("pointcloud_preview_error"),
        "open3d_available": status.get("open3d_available"),
        "preview_enabled": status.get("pointcloud_preview_enabled"),
        "preview_status": status.get("pointcloud_preview_status"),
        "preview_updated": status.get("pointcloud_preview_updated"),
        "preview_fps": status.get("pointcloud_preview_fps"),
        "preview_processing_ms": status.get("pointcloud_preview_processing_ms"),
        "render_ms": status.get("pointcloud_render_ms") or status.get("pointcloud_preview_processing_ms"),
        "preview_visible": status.get("pointcloud_preview_visible"),
        "preview_reason": status.get("pointcloud_preview_reason"),
        "preview_color_mode": status.get("pointcloud_color_mode") or status.get("pointcloud_preview_color_mode"),
        "color_mode": status.get("pointcloud_color_mode") or status.get("pointcloud_preview_color_mode"),
        "preview_outlier_filter": status.get("pointcloud_preview_outlier_filter"),
        "view_bounds": status.get("pointcloud_view_bounds"),
        "scene_display_max_points": status.get("pointcloud_scene_display_max_points"),
        "bottle_display_max_points": status.get("pointcloud_bottle_display_max_points"),
        "point_size": status.get("pointcloud_preview_point_size"),
        "brightness_boost": status.get("pointcloud_preview_brightness_boost"),
        "remove_outliers": status.get("pointcloud_preview_remove_outliers"),
        "preview_depth_min_m": status.get("pointcloud_preview_depth_min_m"),
        "preview_depth_max_m": status.get("pointcloud_preview_depth_max_m"),
        "preview_exists": scene_preview.exists(),
        "preview_url": "/image/latest_pointcloud_preview.jpg" if scene_preview.exists() else None,
        "bottle_preview_exists": bottle_preview.exists(),
        "bottle_preview_url": "/image/latest_bottle_pointcloud_preview.jpg" if bottle_preview.exists() else None,
        "context_preview_exists": context_preview.exists(),
        "context_preview_url": "/image/latest_pointcloud_context.jpg" if context_preview.exists() else None,
        "context_status": status.get("pointcloud_context_status"),
        "context_processing_ms": status.get("pointcloud_context_processing_ms"),
        "context_error": status.get("pointcloud_context_error"),
        "live_json_exists": live_json.exists(),
        "live_json_url": "/api/pointcloud-live" if live_json.exists() else None,
        "live_json_status": status.get("pointcloud_json_status"),
        "live_json_point_count": status.get("pointcloud_json_point_count"),
        "live_json_bottle_point_count": status.get("pointcloud_json_bottle_point_count"),
        "live_json_processing_ms": status.get("pointcloud_json_processing_ms"),
        "live_json_error": status.get("pointcloud_json_error"),
        "scene_ply_exists": scene_ply.exists(),
        "scene_ply_url": "/image/latest_realsense_pointcloud.ply" if scene_ply.exists() else None,
        "bottle_ply_exists": bottle_ply.exists(),
        "bottle_ply_url": "/image/latest_bottle_pointcloud.ply" if bottle_ply.exists() else None,
        "processing_ms": status.get("pointcloud_processing_ms"),
        "crop_processing_ms": status.get("crop_processing_ms"),
        "geometry_processing_ms": status.get("geometry_processing_ms"),
        "contact_validation_ms": status.get("contact_validation_ms"),
        "total_processing_ms": status.get("total_processing_ms"),
    }


def summarize_multiview(status):
    status = status or {}
    preview = FINAL_DEMO_FILES["multiview_reconstruction_preview"]
    bottle_ply = FINAL_DEMO_FILES["multiview_accumulated_bottle_ply"]
    scene_ply = FINAL_DEMO_FILES["multiview_accumulated_scene_ply"]
    mesh_ply = FINAL_DEMO_FILES["multiview_reconstructed_mesh_ply"]
    return {
        "scan_status": status.get("scan_status") or "idle",
        "session_id": status.get("reconstruction_session_id"),
        "session_start_timestamp": status.get("session_start_timestamp"),
        "output_session_dir": status.get("output_session_dir"),
        "pose_source": status.get("pose_source") or "registration_unavailable",
        "pose_confidence": status.get("pose_confidence"),
        "pose_valid": status.get("pose_valid"),
        "pose_failure_reason": status.get("pose_failure_reason"),
        "accepted_keyframes": status.get("accepted_keyframe_count") or 0,
        "rejected_keyframes": status.get("rejected_keyframe_count") or 0,
        "candidate_frames": status.get("candidate_frame_count") or 0,
        "skipped_candidates": status.get("reconstruction_skipped_candidates") or 0,
        "queue_size": status.get("reconstruction_queue_size") or 0,
        "model_points": status.get("filtered_accumulated_point_count") or status.get("fused_voxel_count") or 0,
        "scene_model_points": status.get("accumulated_scene_point_count") or 0,
        "bottle_model_points": status.get("accumulated_bottle_point_count") or status.get("filtered_accumulated_point_count") or 0,
        "raw_accumulated_point_count": status.get("raw_accumulated_point_count") or 0,
        "fused_voxel_count": status.get("fused_voxel_count") or 0,
        "coverage_percent": status.get("estimated_surface_coverage_percent") or status.get("model_coverage_score") or 0,
        "coverage_confidence": status.get("coverage_confidence"),
        "covered_view_angle_deg": status.get("covered_view_angle_deg"),
        "missing_view_directions": status.get("missing_view_directions") or [],
        "recommended_camera_motion": status.get("recommended_camera_motion"),
        "registration_fitness": status.get("registration_fitness"),
        "registration_inlier_rmse_m": status.get("registration_inlier_rmse_m"),
        "registration_ms": status.get("registration_ms"),
        "fusion_ms": status.get("fusion_ms"),
        "preview_ms": status.get("preview_ms"),
        "bottleneck": status.get("reconstruction_bottleneck"),
        "current_view_angle_deg": status.get("current_view_angle_deg"),
        "reconstruction_age_sec": status.get("reconstruction_age_sec"),
        "worker_busy": bool(status.get("worker_busy")),
        "last_rejection_reason": status.get("last_rejection_reason"),
        "multiview_input_valid": status.get("multiview_input_valid"),
        "multiview_input_reject_reason": status.get("multiview_input_reject_reason"),
        "multiview_input_roi_source": status.get("multiview_input_roi_source"),
        "multiview_input_scene_point_count": status.get("multiview_input_scene_point_count"),
        "multiview_input_bottle_point_count": status.get("multiview_input_bottle_point_count"),
        "multiview_input_pointcloud_age_sec": status.get("multiview_input_pointcloud_age_sec"),
        "multiview_input_snapshot_consistent": status.get("multiview_input_snapshot_consistent"),
        "multiview_input_intrinsics_available": status.get("multiview_input_intrinsics_available"),
        "contact_support_status": status.get("contact_support_status"),
        "contact_support_left_distance_mm": status.get("contact_support_left_distance_mm"),
        "contact_support_right_distance_mm": status.get("contact_support_right_distance_mm"),
        "contact_width_3d_mm": status.get("contact_width_3d_mm"),
        "bottle_static": status.get("bottle_static"),
        "requires_reset": bool(status.get("reconstruction_requires_reset")),
        "preview_exists": preview.exists(),
        "preview_url": "/api/multiview-live" if preview.exists() else None,
        "scene_ply_exists": scene_ply.exists(),
        "scene_ply_url": "/image/multiview/accumulated_scene.ply" if scene_ply.exists() else None,
        "bottle_ply_exists": bottle_ply.exists(),
        "bottle_ply_url": "/image/multiview/accumulated_bottle.ply" if bottle_ply.exists() else None,
        "mesh_ply_exists": mesh_ply.exists(),
        "mesh_ply_url": "/image/multiview/reconstructed_bottle_mesh.ply" if mesh_ply.exists() else None,
    }


def summarize_sync_status(sync_status, bundle, calibration, embodiment):
    sync_status = sync_status or {}
    bundle = bundle or {}
    event_features = bundle.get("event_features") if isinstance(bundle.get("event_features"), dict) else {}
    calibration = calibration or {}
    embodiment = embodiment or {}
    return {
        "sync_enabled": sync_status.get("sync_enabled"),
        "sync_mode": sync_status.get("sync_mode") or "software_timestamp",
        "sync_quality": sync_status.get("sync_quality"),
        "davis_realsense_sync_quality": sync_status.get("davis_realsense_sync_quality"),
        "davis_realsense_valid_for_live_quality": sync_status.get("davis_realsense_valid_for_live_quality"),
        "bundle_valid_for_live_quality": sync_status.get("bundle_valid_for_live_quality"),
        "valid_for_live_quality": sync_status.get("valid_for_live_quality"),
        "max_sensor_time_offset_ms": sync_status.get("max_sensor_time_offset_ms"),
        "davis_realsense_offset_ms": sync_status.get("davis_realsense_offset_ms"),
        "pointcloud_realsense_offset_ms": sync_status.get("pointcloud_realsense_offset_ms"),
        "live_quality_max_sensor_offset_ms": sync_status.get("live_quality_max_sensor_offset_ms"),
        "average_sensor_time_offset_ms": sync_status.get("average_sensor_time_offset_ms"),
        "last_bundle_id": sync_status.get("last_bundle_id") or bundle.get("bundle_id"),
        "synchronized_bundle_available": sync_status.get("synchronized_bundle_available"),
        "sync_warning": sync_status.get("sync_warning"),
        "bundle_fps": sync_status.get("bundle_fps"),
        "bundle_age_sec": sync_status.get("bundle_age_sec"),
        "sensor_freshness": sync_status.get("sensor_freshness") or {},
        "realsense_age_ms": sync_status.get("realsense_age_ms"),
        "realsense_freshness": sync_status.get("realsense_freshness"),
        "davis_event_age_ms": sync_status.get("davis_event_age_ms"),
        "davis_event_freshness": sync_status.get("davis_event_freshness"),
        "davis_events_valid_for_live_quality": sync_status.get("davis_events_valid_for_live_quality"),
        "davis_events_valid_for_benchmark": sync_status.get("davis_events_valid_for_benchmark"),
        "pointcloud_age_sec": sync_status.get("pointcloud_age_sec"),
        "pointcloud_freshness": sync_status.get("pointcloud_freshness"),
        "pointcloud_valid_for_live_quality": sync_status.get("pointcloud_valid_for_live_quality"),
        "pointcloud_valid_for_benchmark": sync_status.get("pointcloud_valid_for_benchmark"),
        "sync_warnings": sync_status.get("sync_warnings") or [],
        "sync_reason": sync_status.get("sync_reason"),
        "davis_event_window_count": sync_status.get("davis_event_window_count") if "davis_event_window_count" in sync_status else bundle.get("davis_event_count"),
        "davis_event_rate": sync_status.get("davis_event_rate") if "davis_event_rate" in sync_status else bundle.get("davis_event_rate"),
        "davis_event_window_start": sync_status.get("davis_event_window_start") if "davis_event_window_start" in sync_status else bundle.get("davis_event_window_start"),
        "davis_event_window_end": sync_status.get("davis_event_window_end") if "davis_event_window_end" in sync_status else bundle.get("davis_event_window_end"),
        "davis_event_window_source": sync_status.get("davis_event_window_source") if "davis_event_window_source" in sync_status else event_features.get("event_window_source"),
        "davis_event_window_materialized": sync_status.get("davis_event_window_materialized") if "davis_event_window_materialized" in sync_status else event_features.get("event_window_materialized"),
        "cached_report_event_count": sync_status.get("cached_report_event_count") if "cached_report_event_count" in sync_status else event_features.get("cached_report_event_count"),
        "cached_report_event_rate": sync_status.get("cached_report_event_rate") if "cached_report_event_rate" in sync_status else event_features.get("cached_report_event_rate"),
        "action_event_window_planned_end": sync_status.get("action_event_window_planned_end") if "action_event_window_planned_end" in sync_status else event_features.get("action_event_window_planned_end"),
        "action_event_window_complete": sync_status.get("action_event_window_complete") if "action_event_window_complete" in sync_status else event_features.get("action_event_window_complete"),
        "mount_layout": sync_status.get("mount_layout") or calibration.get("mount_layout"),
        "calibration_status": sync_status.get("davis_realsense_alignment_status") or calibration.get("alignment_status"),
        "homography_available": sync_status.get("homography_available") if sync_status.get("homography_available") is not None else calibration.get("homography_available"),
        "extrinsics_available": sync_status.get("extrinsics_available") if sync_status.get("extrinsics_available") is not None else calibration.get("extrinsics_available"),
        "approximate_alignment_only": sync_status.get("approximate_alignment_only") if sync_status.get("approximate_alignment_only") is not None else calibration.get("approximate_alignment_only"),
        "embodiment": embodiment.get("active_embodiment") or embodiment.get("embodiment_type"),
        "expected_contact_count": embodiment.get("expected_contact_count"),
        "contact_model_type": embodiment.get("contact_model_type"),
        "quality_metric_applicability": embodiment.get("quality_metric_applicability"),
        "limitations": embodiment.get("contact_model_limitations") or embodiment.get("limitations"),
    }


def _refresh_hardware_discovery_status():
    status = {}
    for name, selector in (("realsense", select_realsense_device), ("davis", select_davis_device)):
        try:
            status[name] = selector().to_dict()
        except Exception as exc:
            status[name] = {
                "sensor_type": name,
                "devices": [],
                "selected": None,
                "status": "ERROR",
                "reason": str(exc),
                "ok": False,
            }
    status["selection_policy"] = {
        "persistent_identity": "serial/model/VID_PID",
        "transient_usb_node": "diagnostic_only",
    }
    with hardware_status_lock:
        hardware_status_cache["timestamp"] = time.time()
        hardware_status_cache["payload"] = status


def hardware_discovery_status():
    """Return cached discovery immediately and refresh it off the request path.

    Some camera SDK calls can block while a USB device is recovering.  Hardware
    discovery is diagnostic, so it must never freeze the live dashboard/status
    endpoint or prevent the other camera from starting.
    """
    global hardware_status_refresh_thread
    now = time.time()
    with hardware_status_lock:
        payload = hardware_status_cache["payload"]
        realsense_transport_owned = bool(desired_processes.get("realsense"))
        fresh = payload is not None and (
            realsense_transport_owned
            or now - hardware_status_cache["timestamp"] < 5.0
        )
        refresh_running = bool(
            hardware_status_refresh_thread
            and hardware_status_refresh_thread.is_alive()
        )
        if not fresh and not refresh_running:
            hardware_status_refresh_thread = threading.Thread(
                target=_refresh_hardware_discovery_status,
                name="hardware-discovery-refresh",
                daemon=True,
            )
            hardware_status_refresh_thread.start()
        if payload is not None:
            if realsense_transport_owned:
                return {
                    **payload,
                    "selection_policy": {
                        **(payload.get("selection_policy") or {}),
                        "realsense_refresh": "paused_while_ros_transport_owns_camera",
                    },
                }
            return payload
    return {
        "realsense": {
            "sensor_type": "realsense",
            "devices": [],
            "selected": None,
            "status": "OPENING",
            "reason": "Hardware discovery is running in the background.",
            "ok": False,
        },
        "davis": {
            "sensor_type": "davis",
            "devices": [],
            "selected": None,
            "status": "OPENING",
            "reason": "Hardware discovery is running in the background.",
            "ok": False,
        },
        "selection_policy": {
            "persistent_identity": "serial/model/VID_PID",
            "transient_usb_node": "diagnostic_only",
        },
    }


def embodiment_change_locked(realtime):
    """Return True while changing embodiment could reinterpret an active grasp."""

    realtime = realtime or {}
    target_status = str(realtime.get("target_status") or "").upper()
    state = str(realtime.get("state") or "").upper()
    lock_frame = (
        realtime.get("lock_frame_id")
        or (realtime.get("target_lock") or {}).get("lock_frame_id")
        or (realtime.get("target_lock_debug") or {}).get("lock_frame_id")
    )
    return bool(
        realtime.get("ready_to_close")
        or target_status in {"TRACKING", "LOCKED"}
        or state in {"APPROACH_TRACKING", "TRACKING", "READY_TO_CLOSE"}
        or lock_frame is not None
    )


def _count_contacts_from_status(realtime, matrix):
    for value in (
        (matrix or {}).get("detected_contact_count"),
        (matrix or {}).get("contact_pair_count"),
        (realtime or {}).get("detected_contact_count"),
        (realtime or {}).get("contact_pair_count"),
    ):
        if value is not None:
            return value
    points = (matrix or {}).get("contact_points_px") or (realtime or {}).get("contact_points_px") or {}
    if isinstance(points, dict):
        return len([v for v in points.values() if v is not None])
    if isinstance(points, list):
        return len(points)
    count = 0
    for key in ("left_finger_contact_px", "right_finger_contact_px", "left_contact_px", "right_contact_px"):
        if (realtime or {}).get(key) is not None or (matrix or {}).get(key) is not None:
            count += 1
    return count if count else None


def build_embodiment_dashboard_status(embodiment, realtime=None, matrix=None, selector_locked=None):
    """Merge configured embodiment metadata with live contact counts for the UI."""

    embodiment = dict(embodiment or configured_embodiment_status())
    realtime = realtime or {}
    matrix = matrix or {}
    canonical = canonical_embodiment_id(
        embodiment.get("canonical_embodiment_id")
        or embodiment.get("configured_embodiment")
        or embodiment.get("active_embodiment")
        or embodiment.get("embodiment_type")
    )
    policy = release_policy(canonical)
    applicability = embodiment.get("metric_applicability") or embodiment_metric_applicability(canonical)
    display_metrics = embodiment.get("dashboard_metrics") or embodiment_display_metrics(canonical)
    displayed_ids = {item.get("metric_id") for item in display_metrics}
    hidden = list(embodiment.get("hidden_metrics") or applicability.get("hidden_metrics") or [])
    detected_contacts = _count_contacts_from_status(realtime, matrix)
    expected = embodiment.get("expected_contacts", embodiment.get("expected_contact_count"))
    active_grasp_locked = embodiment_change_locked(realtime)
    selector_locked = active_grasp_locked if selector_locked is None else bool(selector_locked)
    status = {
        **embodiment,
        **policy,
        "active_embodiment": embodiment.get("active_embodiment") or embodiment.get("embodiment_type") or canonical,
        "canonical_embodiment_id": canonical,
        "expected_contacts": expected,
        "expected_contact_count": expected,
        "detected_contacts": detected_contacts,
        "detected_contact_count": detected_contacts,
        "contact_set": {
            "expected_contacts": expected,
            "detected_contacts": detected_contacts,
            "contact_source": matrix.get("contact_source") or realtime.get("contact_source"),
            "contact_status": matrix.get("contact_quality_gate_status") or realtime.get("finger_detection_status"),
        },
        "dashboard_metrics": display_metrics,
        "dashboard_metric_ids": sorted([metric for metric in displayed_ids if metric]),
        "hidden_metrics": hidden,
        "invalid_metrics_hidden": all(metric not in displayed_ids for metric in hidden),
        "metric_applicability": applicability.get("metric_applicability") or {},
        "selector_options": embodiment.get("selector_options") or embodiment_selector_options(),
        "selector_locked": selector_locked,
        "selector_lock_reason": "active_grasp_locked" if selector_locked else None,
        "display_only_notice": policy["release_note"] if not policy["production_decisions_allowed"] else None,
    }
    return status


def _pick_fields(value, fields):
    value = value if isinstance(value, dict) else {}
    return {key: value.get(key) for key in fields if key in value}


COMPACT_REALSENSE_FIELDS = {
    "state", "decision", "action", "ready_to_close", "target_status",
    "tracking_confidence", "frame_id", "post_grasp_active",
    "post_grasp_baseline_ready", "post_grasp_baseline_reason",
    "gripper_closed_confirmed", "manual_gripper_closed", "bottle_detected",
    "detected_bottle_count", "eligible_bottle_count", "bottle_distance_mm",
    "object_depth_mm", "contact_band_depth_mm", "actionable_distance_mm",
    "distance_measurement_valid", "distance_validity_reason", "distance_status",
    "distance_confidence", "distance_confidence_percent", "distance_ready",
    "in_close_range", "desired_close_distance_mm", "min_close_distance_mm",
    "max_close_distance_mm", "close_distance_range_validation_status",
    "close_distance_range_physically_validated", "grip_width_mm", "move",
    "guidance_label", "guidance_basis", "camera_to_hand_offset_enabled",
    "camera_to_hand_offset_active", "camera_to_hand_offset_mm",
    "camera_to_hand_offset_uncertainty_mm", "desired_contact_gap_mm",
    "hand_offset_calibration_status", "hand_offset_physically_validated",
    "hand_reach_projection_valid", "hand_reach_projection_visible",
    "hand_projection_depth_mm", "hand_projection_target_depth_mm",
    "camera_optical_center_px", "fx", "fy",
    "left_finger_detected", "right_finger_detected",
    "left_inner_edge", "right_inner_edge",
    "hand_alignment_error_mm", "hand_to_bottle_z_error_mm",
    "grasp_band_locked", "grasp_band_projected", "grasp_band_current",
    "grasp_band_held", "grasp_band_stale", "grasp_band_projection_unavailable",
    "grasp_band_relock_required", "grasp_band_relock_reason",
    "grasp_band_visibility", "locked_projection_mode", "projection_mode",
    "projection_valid", "projection_invalid_reason", "locked_body_fraction_center",
    "band_drift_state", "band_drift_reason", "grasp_band_drift_px",
    "grasp_band_scale_change", "grasp_band_angle_change_deg",
    "sticky_lock_waiting_reacquisition", "operator_target_phase",
    "operator_reacquisition_active", "operator_reacquisition_waiting",
    "operator_band_state", "operator_locked_fraction_visual_guide_current",
    "operator_band_is_safety_measurement", "identity_measurement_current",
    "target_identity_preserved", "lock_attempt_id", "lock_frame_id",
    "locked_bottle_target_id", "writer_session_id", "writer_mode",
    "realsense_rgb_active", "realsense_depth_active", "rgbd_search_resolution",
    "live_fps", "overlay_fps", "frame_age_ms", "overlay_health", "overlay_stale",
    "ui_primary_status", "ui_primary_action", "ui_quality_status",
    "ui_quality_message", "finger_detection_status", "contact_source",
    "contact_source_label", "quality_input_source", "hand_mounted_camera",
}

COMPACT_QUALITY_FIELDS = {
    "quality_matrix_status", "quality_matrix_message", "quality_label",
    "quality_label_display", "quality_label_stale", "normalized_quality_score",
    "epsilon_score", "raw_epsilon", "epsilon", "epsilon_score_raw",
    "force_closure", "epsilon_force_closure", "force_closure_estimate",
    "matrix_stale_due_to_motion", "matrix_valid_for_current_pose",
    "quality_bundle_status", "quality_compute_id", "quality_input_source",
    "contact_source", "contact_source_label", "contact_quality_gate_status",
    "quality_current_mismatch_reasons", "matrix_stale_reasons",
    "baseline_frozen", "quality_baseline_role", "quality_contact_source_forced",
}

COMPACT_DAVIS_FIELDS = {
    "stream", "current_stage", "davis_motion_stage", "event_state",
    "event_count", "event_rate", "motion_class", "raw_motion_class",
    "davis_safety_class", "object_motion_valid", "object_motion_reason",
    "object_motion_class", "grasp_zone_motion_valid", "grasp_zone_motion_reason",
    "grasp_zone_motion_class", "motion_attribution", "motion_activity_label",
    "rotation_delta_deg", "vibration_amplitude_px_rms", "downward_slip_px",
    "motion_estimation_confidence", "event_health_valid", "event_timestamp_status",
    "event_timestamp_lag_ms", "event_window_ms", "event_window_mode",
    "report_timestamp", "report_heartbeat_age_sec", "report_heartbeat_max_age_sec",
    "report_heartbeat_current",
    "disturbance_type", "disturbance_rejection_applied", "roi_source",
    "roi_object_specific", "analysis_roi_fixed", "acquisition_roi_fixed",
    "operator_roi_role", "operator_roi_label", "fixed_center_third_diagnostic",
    "fixed_center_third_acquisition", "full_frame_fallback",
    "roi_projection_physically_validated", "roi_projection_uncalibrated",
    "roi_projection_method", "davis_target_context_mode",
    "davis_target_roi_priority_selected", "davis_motion_excluded_reason",
    "davis_closeup_used_for_live_quality", "davis_closeup_exclusion_reason",
    "davis_live_quality_contribution", "davis_writer_session_id", "backend",
}

COMPACT_LIVE_QUALITY_FIELDS = {
    "live_quality_status", "analytic_quality_score", "analytic_quality_label",
    "live_quality_score", "live_quality_label", "delta_from_analytic", "trend",
    "primary_risk", "decision", "next_action", "explanation", "live_quality_explanation",
    "davis_adjustment_points", "davis_adjustment_reason",
    "davis_motion_used_for_live_quality", "davis_closeup_used_for_live_quality",
    "davis_closeup_exclusion_reason", "live_quality_davis_penalty", "display",
    "baseline_frozen", "quality_compute_id", "timestamp", "timeline_recent",
}


def compact_operator_status_payload(payload):
    """Return the small presentation contract; diagnostics stay on ``full=1``."""
    if not isinstance(payload, dict):
        return payload
    compact = {
        key: payload.get(key)
        for key in (
            "time", "running", "desired_processes", "writer_warning", "images",
            "combined_quality", "grasp_quality_diagram", "davis_roi_on_realsense_overlay",
            "realtime_status_read_from_cache", "realtime_status_age_sec",
            "realtime_status_age_unknown", "operator_snapshot_age_sec",
            "operator_snapshot_stale",
        )
        if key in payload
    }
    realsense = _pick_fields(payload.get("realsense"), COMPACT_REALSENSE_FIELDS)
    realsense["quality_matrix"] = _pick_fields(
        (payload.get("realsense") or {}).get("quality_matrix"),
        COMPACT_QUALITY_FIELDS,
    )
    compact["realsense"] = realsense
    compact["hand_prototype"] = _pick_fields(
        payload.get("hand_prototype"),
        COMPACT_REALSENSE_FIELDS | {
            "hand_mounted_camera_mode", "hand_mounted_mode_label", "hand_mounted_stage",
            "approach_mode_active", "approach_mode_requested", "guidance_frame",
            "sticky_single_bottle_lock_enabled", "locked_band_visible",
            "locked_band_current_rgbd_supported", "locked_band_visibility_reason",
        },
    )
    compact["davis_motion"] = _pick_fields(payload.get("davis_motion"), COMPACT_DAVIS_FIELDS)
    live_quality = _pick_fields(payload.get("live_quality"), COMPACT_LIVE_QUALITY_FIELDS)
    if isinstance(live_quality.get("timeline_recent"), list):
        live_quality["timeline_recent"] = live_quality["timeline_recent"][-3:]
    compact["live_quality"] = live_quality
    compact["quality_bundle"] = _pick_fields(
        payload.get("quality_bundle"),
        {"quality_bundle_status", "quality_compute_id", "quality_stale_reason"},
    )
    compact["pointcloud"] = _pick_fields(
        payload.get("pointcloud"),
        {"active", "status", "freshness", "age_sec", "contact_width_3d_mm"},
    )
    compact["sensor_sync"] = _pick_fields(
        payload.get("sensor_sync"),
        {"status", "sync_status", "sync_quality", "sync_warning"},
    )
    compact["object_state"] = _pick_fields(
        payload.get("object_state"),
        {"orientation", "available_for_pickup", "availability_status"},
    )
    compact["multi_bottle_target_selection"] = _pick_fields(
        payload.get("multi_bottle_target_selection"),
        {"selection_status", "candidate_count", "eligible_count", "selected_target_id"},
    )
    compact["embodiment"] = _pick_fields(
        payload.get("embodiment"),
        {"canonical_embodiment_id", "active_embodiment", "display_name", "dashboard_metrics"},
    )
    return compact


def build_compact_operator_status_snapshot():
    """Build a read-only operator snapshot from producer-owned latest artifacts."""
    realtime, status_age, from_cache = read_realtime_status_snapshot(
        FINAL_DEMO_FILES["realtime_status"]
    )
    realsense_summary = summarize_realsense(realtime)
    overlay_files = {"overlay": final_file_status(FINAL_DEMO_FILES["overlay"])}
    realsense_summary.update(overlay_health_info(realtime, overlay_files))
    davis_report = read_davis_report()
    davis_summary = summarize_davis(davis_report)
    quality_bundle = validate_quality_bundle_for_live_status(
        read_json(OUTPUT_DIR / "current_quality_bundle.json", {}),
        realtime,
        status_age,
    )
    quality_matrix = realsense_summary.get("quality_matrix") or {}
    if quality_bundle:
        quality_matrix.setdefault("quality_compute_id", quality_bundle.get("quality_compute_id"))
        quality_matrix["quality_bundle_status"] = quality_bundle.get("quality_bundle_status")
    live_quality = read_json(LIVE_QUALITY_OUTPUT_DIR / "latest_status.json", {})
    pointcloud_status = read_json(OUTPUT_DIR / "pointcloud_status.json", {})
    sync_status = read_json(OUTPUT_DIR / "sync_status.json", {})
    embodiment = build_embodiment_dashboard_status(
        read_json(OUTPUT_DIR / "embodiment_status.json") or configured_embodiment_status(),
        realtime=realtime,
        matrix=quality_matrix,
    )
    with process_lock:
        running = {
            "realsense": is_process_running("realsense"),
            "davis_motion": bool(davis_thread and davis_thread.is_alive()),
        }
    if from_cache and realtime.get("writer_session_id") and status_age is not None:
        running["realsense"] = running["realsense"] or status_age <= REALTIME_STATUS_CACHE_GRACE_SEC
    pointcloud = summarize_pointcloud(
        pointcloud_status,
        realtime=realtime,
        backend_running=running["realsense"],
    )
    sync = summarize_sync_status(
        sync_status,
        read_json(OUTPUT_DIR / "latest_sensor_bundle.json", {}),
        current_calibration_status(),
        embodiment,
    )
    combined = summarize_combined_quality(
        quality_matrix,
        davis_summary,
        post_grasp_active=realtime.get("post_grasp_active") is True,
        live_quality=live_quality,
        post_grasp_baseline=(
            realtime.get("post_grasp_baseline")
            if realtime.get("post_grasp_active") is True
            else None
        ),
    )
    diagram = build_grasp_quality_diagram_payload(
        quality_matrix,
        live_quality,
        davis_summary,
        realtime.get("post_grasp_baseline") if realtime.get("post_grasp_active") is True else None,
    )
    payload = {
        "time": time.time(),
        "running": running,
        "desired_processes": dict(desired_processes),
        "writer_warning": None,
        "realsense": realsense_summary,
        "hand_prototype": hand_prototype_dashboard_status(realtime),
        "davis_motion": davis_summary,
        "live_quality": live_quality,
        "combined_quality": combined,
        "quality_bundle": quality_bundle,
        "grasp_quality_diagram": diagram,
        "pointcloud": pointcloud,
        "sensor_sync": sync,
        "object_state": realtime.get("object_state") or {},
        "multi_bottle_target_selection": realtime.get("multi_bottle_target_selection") or {},
        "embodiment": embodiment,
        "davis_roi_on_realsense_overlay": {
            "valid": False,
            "reason": "uncalibrated_cameras_operator_view_separate",
        },
        "images": {
            "overlay": {
                "exists": bool(overlay_files["overlay"].get("exists")),
                "url": "/stream/overlay?writer=" + str(realsense_summary.get("writer_session_id") or "starting"),
                "stale": realsense_summary.get("overlay_stale"),
                "health": realsense_summary.get("overlay_health"),
            },
            "davis_heatmap": {
                "exists": True,
                "url": "/stream/davis_heatmap?writer=" + str(davis_summary.get("davis_writer_session_id") or "starting"),
            },
        },
        "realtime_status_read_from_cache": from_cache,
        "realtime_status_age_sec": status_age,
        "realtime_status_age_unknown": status_age is None,
    }
    return compact_operator_status_payload(payload)


def cached_compact_operator_status_snapshot():
    now = time.time()
    producer_at = float(derived_status_health.get("last_success_time") or 0.0)
    producer_age_sec = max(0.0, now - producer_at) if producer_at else None
    producer_current = bool(
        producer_age_sec is not None
        and producer_age_sec <= REALTIME_STATUS_CACHE_GRACE_SEC
    )
    with operator_status_snapshot_lock:
        cached = operator_status_snapshot_cache.get("payload")
        cached_at = float(operator_status_snapshot_cache.get("timestamp") or 0.0)
        age_sec = max(0.0, now - cached_at) if cached_at else None
        if (
            isinstance(cached, dict)
            and age_sec is not None
            and age_sec <= REALTIME_STATUS_CACHE_GRACE_SEC
            and producer_current
        ):
            response = dict(cached)
            response["operator_snapshot_age_sec"] = round(age_sec, 3)
            response["operator_snapshot_stale"] = False
            return response
    try:
        snapshot = build_compact_operator_status_snapshot()
    except Exception:
        # Never preserve green/current claims from a producer that stopped.
        snapshot = json.loads(json.dumps(cached)) if isinstance(cached, dict) else {}
        snapshot["running"] = {"realsense": False, "davis_motion": False}
        realsense = snapshot.setdefault("realsense", {})
        realsense.update({
            "overlay_stale": True,
            "overlay_health": "STALE",
            "ready_to_close": False,
            "distance_measurement_valid": False,
            "grasp_band_current": False,
            "grasp_band_projected": False,
            "operator_locked_fraction_visual_guide_current": False,
        })
        images = snapshot.setdefault("images", {})
        images.setdefault("overlay", {}).update({"stale": True, "health": "STALE"})
        snapshot["operator_snapshot_age_sec"] = None if age_sec is None else round(age_sec, 3)
        snapshot["operator_snapshot_stale"] = True
        return snapshot
    if not producer_current:
        snapshot["operator_snapshot_age_sec"] = (
            None if producer_age_sec is None else round(producer_age_sec, 3)
        )
        snapshot["operator_snapshot_stale"] = True
        snapshot["live_quality"] = {
            "live_quality_status": "UNAVAILABLE",
            "live_quality_label": "UNAVAILABLE",
            "decision": "HOLD_AND_RECHECK",
            "primary_risk": "derived_status_producer_stale",
            "next_action": "Wait for current derived quality status.",
            "davis_motion_used_for_live_quality": False,
            "davis_closeup_used_for_live_quality": False,
            "davis_closeup_exclusion_reason": "derived_status_producer_stale",
        }
        snapshot["combined_quality"] = {
            "combined_quality_status": "UNAVAILABLE",
            "reason": "derived_status_producer_stale",
        }
        snapshot["sensor_sync"] = {
            "status": "STALE",
            "sync_status": "STALE",
            "sync_quality": "stale",
            "sync_warning": "derived_status_producer_stale",
        }
        davis = snapshot.setdefault("davis_motion", {})
        davis.update({
            "davis_closeup_used_for_live_quality": False,
            "davis_live_quality_contribution": "excluded",
            "davis_closeup_exclusion_reason": "derived_status_producer_stale",
        })
        return snapshot
    snapshot["operator_snapshot_age_sec"] = 0.0
    snapshot["operator_snapshot_stale"] = False
    with operator_status_snapshot_lock:
        operator_status_snapshot_cache.update({"timestamp": time.time(), "payload": snapshot})
    return dict(snapshot)


def refresh_derived_status_outputs_once():
    """Advance derived demo artifacts without depending on a browser GET."""
    if not derived_status_lock.acquire(blocking=False):
        return False
    try:
        realtime, status_age, _from_cache = read_realtime_status_snapshot(
            FINAL_DEMO_FILES["realtime_status"]
        )
        davis_report = read_davis_report()
        _bundle, sync_status = run_file_sync_once(davis_report=davis_report)
        realsense_summary = summarize_realsense(realtime)
        quality_matrix = realsense_summary.get("quality_matrix") or {}
        final_files = final_files_status()
        confirmation = ensure_davis_confirmation_outputs(
            davis_report,
            realsense_summary,
            quality_matrix,
            final_files,
        )
        quality_matrix.update({
            "davis_motion_compensation_enabled": confirmation.get("davis_motion_compensation_enabled"),
            "davis_adjusted_overlay_status": confirmation.get("davis_adjusted_overlay_status"),
            "davis_motion_confidence": confirmation.get("davis_motion_confidence"),
        })
        davis_summary = summarize_davis(davis_report)
        pointcloud_status = read_json(OUTPUT_DIR / "pointcloud_status.json", {})
        with process_lock:
            backend_running = is_process_running("realsense")
        pointcloud = summarize_pointcloud(
            pointcloud_status,
            realtime=realtime,
            backend_running=backend_running,
        )
        embodiment = build_embodiment_dashboard_status(
            read_json(OUTPUT_DIR / "embodiment_status.json") or configured_embodiment_status(),
            realtime=realtime,
            matrix=quality_matrix,
        )
        sync = summarize_sync_status(
            sync_status,
            read_json(OUTPUT_DIR / "latest_sensor_bundle.json", {}),
            current_calibration_status(),
            embodiment,
        )
        baseline = baseline_from_post_grasp_snapshot(
            realtime,
            embodiment=embodiment,
        ) or baseline_from_quality_matrix(
            quality_matrix,
            realtime=realsense_summary,
            embodiment=embodiment,
        )
        observation = observation_from_status(
            realsense_summary,
            matrix=quality_matrix,
            davis=davis_summary,
            pointcloud=pointcloud,
            sync=sync,
            baseline=baseline,
            embodiment=embodiment,
        )
        try:
            live_result = live_quality_monitor.update(baseline, observation)
        except Exception:
            if runtime_recording is not None:
                runtime_recording.abort("dashboard_quality_update_failed")
            raise
        write_live_quality_outputs(live_result, output_dir=LIVE_QUALITY_OUTPUT_DIR)
        derived_status_health["last_success_time"] = time.time()
        return True
    finally:
        derived_status_lock.release()


def derived_status_loop():
    interval_sec = max(0.2, DASHBOARD_POLL_INTERVAL_MS / 1000.0)
    while not derived_status_stop.is_set():
        started = time.monotonic()
        try:
            refresh_derived_status_outputs_once()
        except Exception as exc:
            print(f"Derived status producer warning: {exc}", flush=True)
        try:
            snapshot = build_compact_operator_status_snapshot()
            with operator_status_snapshot_lock:
                operator_status_snapshot_cache.update({
                    "timestamp": time.time(),
                    "payload": snapshot,
                })
        except Exception as exc:
            print(f"Operator status snapshot warning: {exc}", flush=True)
        elapsed = time.monotonic() - started
        derived_status_stop.wait(max(0.02, interval_sec - elapsed))


def start_derived_status_producer():
    global derived_status_thread
    if derived_status_thread and derived_status_thread.is_alive():
        return
    derived_status_stop.clear()
    derived_status_thread = threading.Thread(target=derived_status_loop, daemon=True)
    derived_status_thread.start()


@app.get("/")
def index():
    html = INDEX_HTML.replace("__DASHBOARD_POLL_INTERVAL_MS__", str(int(DASHBOARD_POLL_INTERVAL_MS)))
    html = html.replace(
        "__HAND_OFFSET_RESET_DEFAULTS__",
        json.dumps(HAND_OFFSET_RESET_DEFAULTS, separators=(",", ":")),
    )
    return Response(html, mimetype="text/html")


@app.get("/vendor/three.min.js")
def three_js():
    if not THREE_JS_PATH.is_file():
        return Response("Three.js is not installed.", status=404, mimetype="text/plain")
    return send_file(THREE_JS_PATH, mimetype="application/javascript")


@app.get("/api/status")
def api_status():
    compact_arg = request.args.get("compact")
    compact_status = True if compact_arg is None else compact_arg.strip().lower() not in {"0", "false", "no", "off"}
    if request.args.get("full", "").strip().lower() in {"1", "true", "yes", "on"}:
        compact_status = False
    if compact_status:
        # This high-rate route is intentionally read-only. Derived sync,
        # confirmation, and live-quality files are owned by the background
        # producer, never by browser polling.
        return jsonify(cached_compact_operator_status_snapshot())
    realtime, realtime_status_age_sec, realtime_status_read_from_cache = (
        read_realtime_status_snapshot(FINAL_DEMO_FILES["realtime_status"])
    )
    writer_warning = update_writer_session_warning(realtime)
    davis_report = read_davis_report()
    camera_health = read_json(OUTPUT_DIR / "realsense_camera_health.json")
    pointcloud_status = read_json(OUTPUT_DIR / "pointcloud_status.json")
    object_state = realtime.get("object_state") or object_state_from_status(realtime, pointcloud_status).to_dict()
    multiview_status = read_json(FINAL_DEMO_FILES["multiview_reconstruction_status"], {})
    quality_bundle = read_json(OUTPUT_DIR / "current_quality_bundle.json", {})
    quality_bundle = validate_quality_bundle_for_live_status(
        quality_bundle,
        realtime,
        realtime_status_age_sec,
    )
    embodiment_status = read_json(OUTPUT_DIR / "embodiment_status.json") or configured_embodiment_status()
    sync_status = read_json(OUTPUT_DIR / "sync_status.json", {})
    latest_bundle = read_json(OUTPUT_DIR / "latest_sensor_bundle.json")
    calibration = current_calibration_status()
    final_files = final_files_status()
    hardware = hardware_discovery_status()
    realsense_summary = summarize_realsense(realtime)
    realsense_summary.update({
        "realtime_status_read_from_cache": realtime_status_read_from_cache,
        "realtime_status_age_sec": realtime_status_age_sec,
        "realtime_status_age_unknown": realtime_status_age_sec is None,
    })
    hand_prototype = hand_prototype_dashboard_status(realtime)
    if pointcloud_status:
        realsense_summary.update({
            "realsense_pointcloud_active": bool(pointcloud_status.get("realsense_pointcloud_active")),
            "realsense_pointcloud_backend": pointcloud_status.get("realsense_pointcloud_backend"),
            "realsense_pointcloud_fps": pointcloud_status.get("realsense_pointcloud_fps"),
            "realsense_pointcloud_point_count": pointcloud_status.get("realsense_pointcloud_point_count"),
            "realsense_pointcloud_quality": pointcloud_status.get("realsense_pointcloud_quality"),
            "pointcloud_status": pointcloud_status,
        })
    if embodiment_status:
        realsense_summary["embodiment_status"] = embodiment_status
    fresh_lock_quality = fresh_status_snapshot()
    realsense_summary.update({k: v for k, v in fresh_lock_quality.items() if str(k).startswith("fresh_lock_quality_")})
    davis_summary = summarize_davis(davis_report)
    quality_matrix = realsense_summary.get("quality_matrix") or {}
    grasp_plan = plan_task_grasp(
        object_state,
        pregrasp_plan=read_json(OUTPUT_DIR / "candidate_grasp_report.json", {}),
        epsilon_result=quality_matrix,
    ).to_dict()
    multi_bottle_selection = realtime.get("multi_bottle_target_selection") or {}
    realsense_summary["object_state"] = object_state
    realsense_summary["task_grasp_policy"] = grasp_plan
    davis_summary["object_state"] = object_state
    if quality_bundle:
        quality_matrix.setdefault("quality_compute_id", quality_bundle.get("quality_compute_id"))
        quality_matrix["quality_bundle_status"] = quality_bundle.get("quality_bundle_status")
        quality_matrix["quality_bundle_manifest_path"] = "output/current_quality_bundle.json"
        quality_matrix["matrix_compute_id"] = quality_bundle.get("matrix_compute_id")
        quality_matrix["bottle_overlay_compute_id"] = quality_bundle.get("bottle_overlay_compute_id")
        quality_matrix["contact_hull_compute_id"] = quality_bundle.get("contact_hull_compute_id")
        quality_matrix["wrench_plot_compute_id"] = quality_bundle.get("wrench_plot_compute_id")
        quality_matrix["fx_fy_compute_id"] = quality_bundle.get("fx_fy_compute_id")
        quality_matrix["fx_tau_compute_id"] = quality_bundle.get("fx_tau_compute_id")
        quality_matrix["fy_tau_compute_id"] = quality_bundle.get("fy_tau_compute_id")
        quality_matrix["artifact_render_status"] = quality_bundle.get("artifact_render_status")
        quality_matrix["dashboard_artifact_fresh"] = quality_bundle.get("dashboard_artifact_fresh")
        quality_matrix["quality_stale_reason"] = quality_bundle.get("quality_stale_reason")
        if quality_bundle.get("quality_bundle_status") == "STALE":
            quality_matrix["matrix_valid_for_current_pose"] = False
            quality_matrix["quality_label_stale"] = True
            quality_matrix["quality_current_mismatch_reasons"] = quality_bundle.get("stale_reasons") or []
    embodiment_status = build_embodiment_dashboard_status(
        embodiment_status,
        realtime=realtime,
        matrix=quality_matrix,
    )
    realsense_summary["embodiment_status"] = embodiment_status
    davis_confirmation = read_json(FINAL_DEMO_FILES["davis_performance_profile"], {})
    quality_matrix.update({
        "davis_motion_compensation_enabled": davis_confirmation.get("davis_motion_compensation_enabled"),
        "davis_adjusted_overlay_status": davis_confirmation.get("davis_adjusted_overlay_status"),
        "davis_motion_confidence": davis_confirmation.get("davis_motion_confidence"),
    })
    davis_summary.update({
        "davis_display_fps": davis_confirmation.get("davis_display_fps"),
        "davis_event_analysis_fps": davis_confirmation.get("davis_event_analysis_fps"),
        "davis_event_rate": davis_confirmation.get("davis_event_rate"),
        "davis_bottleneck": davis_confirmation.get("davis_bottleneck"),
        "davis_motion_compensation_enabled": davis_confirmation.get("davis_motion_compensation_enabled"),
        "davis_motion_confidence": davis_confirmation.get("davis_motion_confidence"),
        "davis_adjusted_overlay_status": davis_confirmation.get("davis_adjusted_overlay_status"),
    })
    realsense_summary.update({
        "realsense_quality_status": davis_confirmation.get("realsense_quality_status"),
        "matrix_valid_for_current_pose": davis_confirmation.get("matrix_valid_for_current_pose"),
        "matrix_stale_reason": davis_confirmation.get("matrix_stale_reason"),
        "davis_motion_compensation_enabled": davis_confirmation.get("davis_motion_compensation_enabled"),
        "davis_adjusted_overlay_status": davis_confirmation.get("davis_adjusted_overlay_status"),
        "davis_motion_confidence": davis_confirmation.get("davis_motion_confidence"),
        "davis_display_fps": davis_confirmation.get("davis_display_fps"),
        "davis_event_analysis_fps": davis_confirmation.get("davis_event_analysis_fps"),
        "davis_event_rate": davis_confirmation.get("davis_event_rate"),
        "davis_bottleneck": davis_confirmation.get("davis_bottleneck"),
    })
    final_files = final_files_status()
    overlay_health = overlay_health_info(realtime, final_files)
    realsense_summary.update(overlay_health)
    object_profiles = object_profile_dashboard_status()
    active_object = object_profiles.get("active_profile") or {}
    realsense_summary.update({
        "active_object_id": active_object.get("object_id"),
        "active_object_name": active_object.get("display_name"),
        "active_object_type": active_object.get("object_type"),
        "object_profile_confidence": active_object.get("profile_confidence"),
        "object_profile_source": active_object.get("source"),
    })
    include_diagnostics = request.args.get("diagnostics", "0").strip().lower() in {"1", "true", "yes", "on"}
    profile_summary = summarize_profile(realtime)
    with process_lock:
        running = {
            "realsense": is_process_running("realsense"),
            "davis_motion": bool(davis_thread and davis_thread.is_alive()),
        }
    if (
        realtime_status_read_from_cache
        and realtime.get("writer_session_id")
        and realtime.get("writer_mode") == "backend_only"
        and realtime_status_age_sec is not None
        and realtime_status_age_sec <= REALTIME_STATUS_CACHE_GRACE_SEC
    ):
        running["realsense"] = True
    if not running["davis_motion"]:
        davis_summary.update({
            "report_current": False,
            "historical_report_available": bool(davis_report),
            "stream": "not_running",
            "event_health_valid": False,
            "event_rate": None,
            "event_count": None,
            "motion_class": None,
            "raw_motion_class": None,
            "object_motion_valid": False,
            "object_motion_reason": "stale_events",
            "davis_closeup_used_for_live_quality": False,
            "davis_closeup_exclusion_reason": "stale_events",
        })
    pointcloud_summary = summarize_pointcloud(pointcloud_status, realtime=realtime, backend_running=running["realsense"])
    sync_summary = summarize_sync_status(sync_status, latest_bundle, calibration, embodiment_status)
    live_quality_status = read_json(LIVE_QUALITY_OUTPUT_DIR / "latest_status.json", {})
    combined_quality = summarize_combined_quality(
        realsense_summary.get("quality_matrix"),
        davis_summary,
        post_grasp_active=realtime.get("post_grasp_active") is True,
        live_quality=live_quality_status,
        post_grasp_baseline=(
            realtime.get("post_grasp_baseline")
            if realtime.get("post_grasp_active") is True
            else None
        ),
    )
    davis_summary.update({
        "davis_closeup_used_for_live_quality": live_quality_status.get("davis_closeup_used_for_live_quality"),
        "davis_closeup_exclusion_reason": live_quality_status.get("davis_closeup_exclusion_reason"),
        "davis_closeup_confidence": live_quality_status.get("davis_closeup_confidence"),
        "davis_closeup_motion_class": live_quality_status.get("davis_closeup_motion_class"),
        "davis_closeup_window": live_quality_status.get("davis_closeup_window") or davis_summary.get("davis_closeup_window"),
        "davis_live_quality_contribution": (
            "included"
            if live_quality_status.get("davis_closeup_used_for_live_quality") is True
            else "excluded"
        ),
    })
    realsense_summary["live_quality"] = live_quality_status
    grasp_quality_diagram = build_grasp_quality_diagram_payload(
        quality_matrix,
        live_quality_status,
        davis_summary,
        realtime.get("post_grasp_baseline") if realtime.get("post_grasp_active") is True else None,
    )
    davis_roi_overlay = (
        davis_roi_on_realsense_overlay(davis_report, realtime)
        if running["davis_motion"]
        else {
            "valid": False,
            "source": None,
            "reason": "davis_monitor_not_running",
            "physically_validated": False,
        }
    )
    benchmark_status = None
    if not compact_status:
        benchmark_status = current_benchmark_status(
            realsense_summary=realsense_summary,
            davis_summary=davis_summary,
            pointcloud_summary=pointcloud_summary,
            sync_summary=sync_summary,
            quality_matrix=quality_matrix,
            live_quality_status=live_quality_status,
            embodiment_status=embodiment_status,
        )
    payload = {
        "time": time.time(),
        "running": running,
        "desired_processes": dict(desired_processes),
        "realsense": realsense_summary,
        "davis_motion": davis_summary,
        "davis_roi_on_realsense_overlay": davis_roi_overlay,
        "object_state": object_state,
        "task_grasp_policy": grasp_plan,
        "multi_bottle_target_selection": multi_bottle_selection,
        "davis_quality_confirmation": davis_confirmation,
        "pointcloud": pointcloud_summary,
        "multiview": summarize_multiview(multiview_status),
        "sensor_sync": sync_summary,
        "hardware": hardware,
        "calibration": calibration,
        "embodiment": embodiment_status,
        "combined_quality": combined_quality,
        "live_quality": live_quality_status,
        "grasp_quality_diagram": grasp_quality_diagram,
        "object_profiles": object_profiles,
        "fresh_lock_quality": fresh_lock_quality,
        "hand_prototype": hand_prototype,
        "quality_bundle": quality_bundle,
        "final_files": final_files,
        "stale_threshold_sec": STALE_FILE_THRESHOLD_SEC,
        "images": {
            "rgb": {"exists": True, "url": "/stream/rgb"},
            "depth": {"exists": True, "url": "/stream/depth"},
            "overlay": {
                "exists": bool((final_files.get("overlay") or {}).get("exists")),
                # Frame ids restart at one whenever the RGB-D backend restarts.
                # Give each MJPEG source a writer-specific URL so a browser that
                # survived the restart cannot keep showing frames from the old
                # connection/cache while the status cards describe the new run.
                "url": (
                    "/stream/overlay?writer="
                    + str(realsense_summary.get("writer_session_id") or "starting")
                ),
                "stale": overlay_health.get("overlay_stale"),
                "stale_label": "STALE - LIVE OVERLAY NOT UPDATING" if overlay_health.get("overlay_stale") else "",
                "health": overlay_health.get("overlay_health"),
            },
            "davis_heatmap": {
                "exists": True,
                "url": (
                    "/stream/davis_heatmap?writer="
                    + str(davis_summary.get("davis_writer_session_id") or "starting")
                ),
            },
            "davis_aps": {"exists": True, "url": "/stream/davis/aps"},
            "davis_motion_overlay": static_image_info(final_files, "davis_motion_overlay", "/image/davis_motion_overlay.jpg"),
            "davis_adjusted_quality_overlay": static_image_info(
                final_files,
                "davis_adjusted_quality_overlay",
                "/image/davis_adjusted_quality_overlay.jpg",
                stale=not bool(davis_confirmation.get("davis_adjusted_overlay_valid")),
                stale_label=davis_confirmation.get("davis_adjusted_overlay_reason") or "DAVIS PREVIEW UNAVAILABLE",
            ),
            "davis_realsense_confirmation": static_image_info(final_files, "davis_realsense_confirmation", "/image/davis_realsense_confirmation.jpg"),
            "pointcloud_preview": static_image_info(final_files, "pointcloud_preview", "/image/latest_pointcloud_preview.jpg"),
            "bottle_pointcloud_preview": static_image_info(final_files, "bottle_pointcloud_preview", "/image/latest_bottle_pointcloud_preview.jpg"),
            "pointcloud_context": static_image_info(final_files, "pointcloud_context", "/image/latest_pointcloud_context.jpg"),
            "bottle_quality_overlay": quality_image_info(final_files, "bottle_quality_overlay", "/image/bottle_quality_overlay.jpg", quality_matrix, quality_bundle),
            "bottle_quality_context": quality_image_info(final_files, "bottle_quality_context", "/image/bottle_quality_context.jpg", quality_matrix, quality_bundle),
            "bottle_quality_closeup": quality_image_info(final_files, "bottle_quality_closeup", "/image/bottle_quality_closeup.jpg", quality_matrix, quality_bundle),
            "contact_quality_overlay": quality_image_info(final_files, "contact_quality_overlay", "/image/contact_quality_overlay.jpg", quality_matrix, quality_bundle),
            "grasp_quality_weak_direction": quality_image_info(final_files, "grasp_quality_weak_direction", "/image/grasp_quality_bottle_mask_weak_direction.png", quality_matrix, quality_bundle, advanced=True),
            "grasp_hulls_overlay": quality_image_info(final_files, "grasp_hulls_overlay", "/image/latest_grasp_hulls_overlay.jpg", quality_matrix, quality_bundle, advanced=True),
            "contact_hull_overlay": quality_image_info(final_files, "contact_hull_overlay", "/image/contact_hull_overlay.jpg", quality_matrix, quality_bundle),
            "contact_patch_debug": quality_image_info(final_files, "contact_patch_debug", "/image/contact_patch_debug.jpg", quality_matrix, quality_bundle, advanced=True),
            "wrench_space_convex_hull_debug": quality_image_info(final_files, "wrench_space_convex_hull_debug", "/image/wrench_space_convex_hull_debug.jpg", quality_matrix, quality_bundle, advanced=True),
            "wrench_space_3d_hull": quality_image_info(final_files, "wrench_space_3d_hull", "/image/wrench_space_3d_hull.png", quality_matrix, quality_bundle, advanced=True),
            "wrench_space_projection": quality_image_info(final_files, "wrench_space_projection", "/image/wrench_space_projection.png", quality_matrix, quality_bundle, advanced=True),
            "wrench_space_fx_fy": quality_image_info(final_files, "wrench_space_fx_fy", "/image/wrench_space_fx_fy.png", quality_matrix, quality_bundle, advanced=True),
            "wrench_space_fx_tau": quality_image_info(final_files, "wrench_space_fx_tau", "/image/wrench_space_fx_tau.png", quality_matrix, quality_bundle, advanced=True),
            "wrench_space_fy_tau": quality_image_info(final_files, "wrench_space_fy_tau", "/image/wrench_space_fy_tau.png", quality_matrix, quality_bundle, advanced=True),
        },
        "debug_streams": {**debug_stream_snapshot(), "debug_stream_fps": DEBUG_STREAM_FPS},
        "writer_warning": writer_warning,
        "profile": profile_summary,
        "logs": {
            "realsense": latest_log(LOG_DIR / "realsense_backend.log") if include_diagnostics else "",
            "davis": latest_log(LOG_DIR / "davis_backend.log") if include_diagnostics else "",
        },
        "camera_health": camera_health,
        "realtime_status_read_from_cache": realtime_status_read_from_cache,
        "realtime_status_age_sec": realtime_status_age_sec,
        "realtime_status_age_unknown": realtime_status_age_sec is None,
    }
    if benchmark_status is not None:
        payload["benchmark"] = benchmark_status
    if compact_status:
        compact_operator_status_payload(payload)
    return jsonify(payload)


@app.get("/api/profile")
def api_profile():
    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json")
    return jsonify(summarize_profile(realtime))


@app.get("/api/pointcloud-live")
def api_pointcloud_live():
    path = FINAL_DEMO_FILES["pointcloud_live_json"]
    if not path.exists():
        return jsonify({
            "status": "missing",
            "message": "Point cloud JSON has not been produced yet.",
            "timestamp": time.time(),
        }), 404
    payload = read_json(path, {})
    pointcloud_status = read_json(OUTPUT_DIR / "pointcloud_status.json")
    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json")
    with process_lock:
        backend_running = is_process_running("realsense")
    summary = summarize_pointcloud(pointcloud_status, realtime=realtime, backend_running=backend_running)
    payload["pointcloud_freshness"] = summary.get("freshness")
    payload["pointcloud_age_sec"] = summary.get("age_sec")
    payload["pointcloud_current"] = summary.get("pointcloud_current")
    payload["viewer_watermark"] = summary.get("viewer_watermark")
    payload["coordinate_frame_type"] = summary.get("coordinate_frame_type")
    payload["coordinate_frame_stable"] = summary.get("coordinate_frame_stable")
    payload["world_registered"] = summary.get("world_registered")
    payload["multiview_registered"] = summary.get("multiview_registered")
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/api/multiview-live")
def api_multiview_live():
    path = FINAL_DEMO_FILES["multiview_reconstruction_preview"]
    if not path.exists():
        return jsonify({
            "status": "missing",
            "message": "Multi-view reconstruction preview has not been produced yet.",
            "timestamp": time.time(),
        }), 404
    payload = read_json(path, {})
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/api/object-profiles")
def api_object_profiles():
    return jsonify(object_profile_dashboard_status())


@app.post("/api/object-profiles/scan")
def api_object_profiles_scan():
    payload = request.get_json(silent=True) or {}
    name = payload.get("display_name") or payload.get("name") or "Scanned Object"
    try:
        duration = float(payload.get("duration_sec") or 3.0)
    except (TypeError, ValueError):
        duration = 3.0
    result = scan_object_from_live_outputs(display_name=name, duration_sec=duration)
    return jsonify({**result, "object_profiles": object_profile_dashboard_status()}), (200 if result.get("ok") else 409)


@app.post("/api/object-profiles/save")
def api_object_profiles_save():
    payload = request.get_json(silent=True) or {}
    try:
        saved = save_latest_object_profile_draft(
            display_name=payload.get("display_name") or payload.get("name"),
            notes=payload.get("notes") or "",
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "object_profiles": object_profile_dashboard_status()}), 400
    return jsonify({
        "ok": True,
        "message": f"Object profile saved: {saved.get('display_name') or saved.get('object_id')}",
        "profile": saved,
        "object_profiles": object_profile_dashboard_status(),
    })


def _active_realsense_grasp_locked():
    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
    target_status = str(realtime.get("target_status") or "").upper()
    state = str(realtime.get("state") or "").upper()
    return target_status in {"TRACKING", "LOCKED"} or state in {"TRACKING", "LOCKED", "READY_TO_CLOSE"}


@app.post("/api/object-profiles/set-active")
def api_object_profiles_set_active():
    payload = request.get_json(silent=True) or {}
    object_id = payload.get("object_id") or payload.get("active_object_id")
    if not object_id:
        return jsonify({"ok": False, "message": "Missing object_id."}), 400
    current = object_profile_dashboard_status().get("active_object_id")
    if _active_realsense_grasp_locked() and object_id != current:
        return jsonify({
            "ok": False,
            "message": "Object profile changes are locked during an active grasp. Reset/clear the lock before switching.",
            "object_profiles": object_profile_dashboard_status(),
        }), 409
    try:
        active = set_active_object_profile(object_id)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "object_profiles": object_profile_dashboard_status()}), 400
    return jsonify({
        "ok": True,
        "message": f"Active target set to {active.get('display_name') or active.get('object_id')}.",
        "active_profile": active,
        "object_profiles": object_profile_dashboard_status(),
    })


@app.post("/api/object-profiles/rename")
def api_object_profiles_rename():
    payload = request.get_json(silent=True) or {}
    object_id = payload.get("object_id")
    display_name = payload.get("display_name") or payload.get("name")
    if not object_id or not display_name:
        return jsonify({"ok": False, "message": "Missing object_id or display_name."}), 400
    try:
        profile = rename_object_profile(object_id, display_name)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "object_profiles": object_profile_dashboard_status()}), 400
    return jsonify({
        "ok": True,
        "message": "Object profile renamed.",
        "profile": profile,
        "object_profiles": object_profile_dashboard_status(),
    })


@app.post("/api/object-profiles/archive")
def api_object_profiles_archive():
    payload = request.get_json(silent=True) or {}
    object_id = payload.get("object_id")
    if not object_id:
        return jsonify({"ok": False, "message": "Missing object_id."}), 400
    try:
        profile = archive_object_profile(object_id)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "object_profiles": object_profile_dashboard_status()}), 400
    return jsonify({
        "ok": True,
        "message": "Object profile archived. It was not deleted.",
        "profile": profile,
        "object_profiles": object_profile_dashboard_status(),
    })


@app.post("/api/object-profiles/import")
def api_object_profiles_import():
    try:
        if request.files:
            upload = next(iter(request.files.values()))
            dst = OUTPUT_DIR / "object_profiles" / f"import_{int(time.time())}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            upload.save(dst)
            profile = import_object_profile(dst)
        else:
            payload = request.get_json(silent=True) or {}
            if payload.get("profile"):
                profile = save_object_profile(payload["profile"])
            elif payload.get("path"):
                # only allow imports from the known profile directories; a bare
                # client-supplied path would otherwise read arbitrary files
                requested = Path(str(payload["path"])).resolve()
                allowed_roots = [
                    (OUTPUT_DIR / "object_profiles").resolve(),
                    (ROOT / "config" / "object_profiles").resolve(),
                ]
                if not any(requested.is_relative_to(root) for root in allowed_roots):
                    return jsonify({"ok": False, "message": "Path imports are restricted to the object profile directories."}), 400
                profile = import_object_profile(requested)
            else:
                return jsonify({"ok": False, "message": "Provide an uploaded file, profile JSON, or path."}), 400
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "object_profiles": object_profile_dashboard_status()}), 400
    return jsonify({
        "ok": True,
        "message": f"Imported object profile: {profile.get('display_name') or profile.get('object_id')}",
        "profile": profile,
        "object_profiles": object_profile_dashboard_status(),
    })


@app.get("/api/object-profiles/export/<object_id>")
def api_object_profiles_export(object_id):
    try:
        path = export_object_profile(object_id)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 404
    return send_file(path, as_attachment=True)


@app.post("/api/object-profiles/enhance-multiview")
def api_object_profiles_enhance_multiview():
    payload = request.get_json(silent=True) or {}
    try:
        profile = enhance_latest_profile_with_multiview(payload.get("object_id"))
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "object_profiles": object_profile_dashboard_status()}), 400
    return jsonify({
        "ok": True,
        "message": profile.get("multiview_enhancement_reason") or "Multi-view enhancement status updated.",
        "profile": profile,
        "object_profiles": object_profile_dashboard_status(),
    })


@app.post("/api/debug-streams")
def api_debug_streams():
    payload = request.get_json(silent=True) or {}
    set_debug_streams(bool(payload.get("enabled")))
    return jsonify({"ok": True, "debug_streams": debug_stream_snapshot(), "debug_stream_fps": DEBUG_STREAM_FPS})


@app.post("/api/embodiment")
def api_set_embodiment():
    payload = request.get_json(silent=True) or {}
    requested = canonical_embodiment_id(payload.get("embodiment_id") or payload.get("active_embodiment"))
    if requested == "unknown":
        return jsonify({"ok": False, "message": "Unknown or unsupported embodiment selection."}), 400
    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
    current = build_embodiment_dashboard_status(
        read_json(OUTPUT_DIR / "embodiment_status.json") or configured_embodiment_status(),
        realtime=realtime,
        matrix=read_json(OUTPUT_DIR / "realtime_wrench_matrix_status.json", {}),
    )
    current_id = canonical_embodiment_id(current.get("canonical_embodiment_id") or current.get("active_embodiment"))
    if current.get("selector_locked") and requested != current_id:
        return jsonify({
            "ok": False,
            "message": "Embodiment changes are locked during an active grasp. Reset/clear the lock before switching.",
            "embodiment": current,
        }), 409
    try:
        status = save_active_embodiment(requested)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    status = build_embodiment_dashboard_status(status, realtime=realtime)
    return jsonify({
        "ok": True,
        "message": f"Active embodiment set to {status.get('display_name') or requested}.",
        "embodiment": status,
    })


def benchmark_source_snapshot(configurations=None):
    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
    realsense_summary = summarize_realsense(realtime)
    realsense_summary.update(overlay_health_info(realtime, final_files_status()))
    # summarize_realsense drops the raw perception block; the masked benchmark
    # FeaturePacket needs it for RGB bottle center/angle (detection-table jitter
    # columns). Thread it through for the benchmark source only — this does not
    # change tracking, DAVIS scoring, epsilon, point-cloud, or production output.
    if realtime.get("realsense_perception") is not None and "realsense_perception" not in realsense_summary:
        realsense_summary["realsense_perception"] = realtime.get("realsense_perception")
    # Keep the current structured-label prediction inputs in the benchmark
    # snapshot. Feature masking ignores these fields; they are only recorded
    # when an operator explicitly marks the label evidence current.
    for field in (
        "object_state",
        "target_clamp",
        "selected_target_task_plan",
        "selected_band_id",
        "selected_candidate_id",
        "selected_grasp_preference_tier",
        "selected_grasp_body_fraction",
        "displayed_magenta_center",
        "display_magenta_center",
    ):
        if realtime.get(field) is not None:
            realsense_summary.setdefault(field, realtime.get(field))
    pointcloud_status = read_json(OUTPUT_DIR / "pointcloud_status.json", {})
    with process_lock:
        backend_running = is_process_running("realsense")
    pointcloud_summary = summarize_pointcloud(pointcloud_status, realtime=realtime, backend_running=backend_running)
    davis_summary = summarize_davis(read_davis_report())
    davis_summary.update({
        key: value
        for key, value in davis_status_payload().items()
        if key in {"event_state", "aps_stream_status", "aps_state", "aps_fps", "event_rate", "motion_class", "decision"}
    })
    quality_matrix = realsense_summary.get("quality_matrix") or read_json(OUTPUT_DIR / "realtime_wrench_matrix_status.json", {})
    live_quality_status = read_json(LIVE_QUALITY_OUTPUT_DIR / "latest_status.json", {})
    embodiment_status = build_embodiment_dashboard_status(
        read_json(OUTPUT_DIR / "embodiment_status.json") or configured_embodiment_status(),
        realtime=realtime,
        matrix=quality_matrix,
    )
    sync_summary = read_json(OUTPUT_DIR / "sync_status.json", {}) or {}
    availability = benchmark_sensor_availability(
        realsense_summary,
        davis_summary,
        pointcloud_summary,
        sync_summary,
        quality_matrix,
    )
    benchmark = current_benchmark_status(
        realsense_summary=realsense_summary,
        davis_summary=davis_summary,
        pointcloud_summary=pointcloud_summary,
        sync_summary=sync_summary,
        quality_matrix=quality_matrix,
        live_quality_status=live_quality_status,
        embodiment_status=embodiment_status,
        configurations=configurations,
        include_runner=False,
    )
    benchmark["source_status"]["sensor_availability"] = availability
    return {
        "benchmark": benchmark,
        "availability": availability,
        "realsense_summary": realsense_summary,
        "davis_summary": davis_summary,
        "pointcloud_summary": pointcloud_summary,
        "sync_summary": sync_summary,
        "quality_matrix": quality_matrix,
        "live_quality_status": live_quality_status,
        "embodiment_status": embodiment_status,
    }


def benchmark_sample_status(configurations):
    return benchmark_source_snapshot(configurations=configurations)["benchmark"]


def get_benchmark_runner():
    global benchmark_runner
    with benchmark_runner_lock:
        if benchmark_runner is None:
            benchmark_runner = BenchmarkRunner(
                BENCHMARK_OUTPUT_DIR,
                sample_callback=benchmark_sample_status,
                precheck_callback=benchmark_precheck_snapshot,
                settings_path=ROOT / "config" / "benchmarking.yaml",
            )
        return benchmark_runner


def lightweight_current_benchmark_status(configurations=None):
    source = benchmark_source_snapshot(configurations=configurations)
    benchmark = source["benchmark"]
    benchmark["run"] = get_benchmark_runner().snapshot()
    return benchmark


@app.get("/api/benchmark/status")
def api_benchmark_status():
    return jsonify(lightweight_current_benchmark_status())


@app.post("/api/benchmark/selection")
def api_benchmark_selection():
    payload = request.get_json(silent=True) or {}
    configurations = payload.get("configurations")
    if payload.get("preset_id"):
        preset_id = payload["preset_id"]
        if preset_id not in BENCHMARK_PRESETS:
            return jsonify({"ok": False, "message": "Unknown benchmark preset."}), 400
        configurations = [{
            "configuration_id": preset_id,
            "display_name": BENCHMARK_PRESETS[preset_id]["display_name"],
            "preset_id": preset_id,
            "feature_ids": preset_feature_ids(preset_id),
        }]
    normalized = save_benchmark_selection(configurations)
    return jsonify({
        "ok": True,
        "message": "Benchmark feature selection updated. Production decisions are unchanged.",
        "configurations": normalized,
        "benchmark": lightweight_current_benchmark_status(),
    })


@app.post("/api/benchmark/start")
def api_benchmark_start():
    payload = request.get_json(silent=True) or {}
    settings = load_benchmark_settings(ROOT / "config" / "benchmarking.yaml")
    try:
        duration = float(payload.get("duration_sec") or settings.get("duration_sec") or 20.0)
    except (TypeError, ValueError):
        duration = float(settings.get("duration_sec") or 20.0)
    duration = max(1.0, min(300.0, duration))
    try:
        sample_rate = float(payload.get("sample_rate_hz") or settings.get("sample_rate_hz") or 5.0)
    except (TypeError, ValueError):
        sample_rate = float(settings.get("sample_rate_hz") or 5.0)
    sample_rate = max(0.5, min(10.0, sample_rate))
    if payload.get("use_current_selection"):
        configurations = benchmark_selection_snapshot()
    else:
        configurations = payload.get("configurations") or default_run_configurations(duration_sec=duration)
    precheck = benchmark_precheck_snapshot()
    if not precheck.get("ok"):
        return jsonify({
            "ok": False,
            "message": f"Benchmark precheck failed: {precheck.get('reason')}",
            "precheck": precheck,
            "run": get_benchmark_runner().snapshot(),
            "benchmark": lightweight_current_benchmark_status(),
        }), 409
    state = get_benchmark_runner().start(
        configurations=configurations,
        duration_sec=duration,
        mode=str(payload.get("mode") or "sequential_live"),
        sample_rate_hz=sample_rate,
    )
    return jsonify({
        "ok": bool(state.get("start_accepted")),
        "message": state.get("message") or "Benchmark run command processed.",
        "run": state,
        "benchmark": lightweight_current_benchmark_status(),
    })


@app.post("/api/benchmark/stop")
def api_benchmark_stop():
    state = get_benchmark_runner().stop()
    return jsonify({"ok": True, "message": "Benchmark stop requested.", "run": state})


@app.post("/api/benchmark/pause")
def api_benchmark_pause():
    state = get_benchmark_runner().pause()
    return jsonify({"ok": True, "message": "Benchmark paused.", "run": state})


@app.post("/api/benchmark/resume")
def api_benchmark_resume():
    state = get_benchmark_runner().resume()
    return jsonify({"ok": True, "message": "Benchmark resumed.", "run": state})


@app.post("/api/benchmark/skip-current")
def api_benchmark_skip_current():
    state = get_benchmark_runner().skip_current()
    return jsonify({"ok": True, "message": "Skipping current benchmark configuration.", "run": state})


@app.post("/api/benchmark/repeat-current")
def api_benchmark_repeat_current():
    state = get_benchmark_runner().repeat_current()
    return jsonify({"ok": True, "message": "Repeating current benchmark configuration.", "run": state})


@app.get("/api/benchmark/results")
def api_benchmark_results():
    return jsonify(get_benchmark_runner().results())


@app.get("/api/benchmark/download-report")
def api_benchmark_download_report():
    fmt = request.args.get("format", "html").strip().lower()
    outputs = get_benchmark_runner().results().get("outputs") or {}
    key_by_format = {
        "html": "report_html",
        "md": "report_md",
        "json": "summary",
        "csv": "results_csv",
        "contributions": "feature_contributions_csv",
        "detection_table": "detection_comparison_md",
        "geometry_table": "geometry_comparison_md",
        "davis_motion_table": "davis_motion_comparison_md",
        "overall_table": "overall_recommendation_md",
        "detection_csv": "detection_comparison_csv",
        "geometry_csv": "geometry_comparison_csv",
        "davis_motion_csv": "davis_motion_comparison_csv",
        "overall_csv": "overall_recommendation_csv",
    }
    if fmt in {"tables_zip", "charts_zip"}:
        import zipfile

        run_dir = Path(outputs.get("run_dir") or "").resolve()
        if not run_dir or BENCHMARK_OUTPUT_DIR.resolve() not in run_dir.parents:
            return jsonify({"ok": False, "message": "No completed benchmark run directory is available yet."}), 404
        bundle_dir = run_dir / ("tables" if fmt == "tables_zip" else "charts")
        if not bundle_dir.exists():
            return jsonify({"ok": False, "message": f"Benchmark {fmt.replace('_zip', '')} directory is missing."}), 404
        zip_path = run_dir / f"benchmark_{fmt}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(bundle_dir.iterdir()):
                if item.is_file():
                    zf.write(item, arcname=f"{bundle_dir.name}/{item.name}")
        return send_file(zip_path, as_attachment=True)
    path = outputs.get(key_by_format.get(fmt, "report_html"))
    if not path:
        return jsonify({"ok": False, "message": "No completed benchmark report is available yet."}), 404
    resolved = Path(path).resolve()
    if BENCHMARK_OUTPUT_DIR.resolve() not in resolved.parents and resolved != BENCHMARK_OUTPUT_DIR.resolve():
        return jsonify({"ok": False, "message": "Benchmark report path is outside the output directory."}), 400
    if not resolved.exists():
        return jsonify({"ok": False, "message": "Benchmark report file is missing."}), 404
    return send_file(resolved, as_attachment=True)


@app.post("/api/benchmark/trial/start")
def api_benchmark_trial_start():
    benchmark = lightweight_current_benchmark_status()
    state = benchmark_recorder.start_trial(
        configurations=benchmark.get("configurations") or [],
        metadata={"source": "web_dashboard", "benchmark_only": True},
    )
    return jsonify({"ok": True, "message": "Benchmark trial started.", "trial": state})


@app.post("/api/benchmark/trial/stop")
def api_benchmark_trial_stop():
    benchmark = lightweight_current_benchmark_status()
    state = benchmark_recorder.stop_trial(summary={"last_benchmark": benchmark.get("production_context") or {}})
    return jsonify({"ok": True, "message": "Benchmark trial stopped.", "trial": state})


@app.post("/api/benchmark/label")
def api_benchmark_label():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "Benchmark label payload must be a JSON object."}), 400
    raw_label = payload.get("label")
    structured_keys = ("ground_truth", "prediction", "evidence")
    invalid_structured_key = next(
        (
            key
            for key in structured_keys
            if key in payload and payload[key] is not None and not isinstance(payload[key], dict)
        ),
        None,
    )
    if invalid_structured_key:
        return jsonify({"ok": False, "message": f"{invalid_structured_key} must be a JSON object."}), 400
    if isinstance(raw_label, dict):
        label_input = dict(raw_label)
        for key in structured_keys:
            if key in payload:
                label_input[key] = payload[key]
        has_content = bool(label_input)
        label_name = str(label_input.get("label") or "structured_scene_label").strip()
        label_input["label"] = label_name
        has_label = bool(label_name and has_content)
    elif any(key in payload for key in structured_keys):
        label_name = str(raw_label or "structured_scene_label").strip()
        label_input = {
            "label": label_name,
            **{key: payload[key] for key in structured_keys if key in payload},
        }
        has_label = bool(label_name)
    else:
        label_name = str(raw_label or "").strip()
        label_input = label_name
        has_label = bool(label_name)
    if not has_label:
        return jsonify({"ok": False, "message": "Missing ground-truth label."}), 400
    benchmark = lightweight_current_benchmark_status()
    state = benchmark_recorder.record_label(
        label_input,
        benchmark_status=benchmark,
        note=payload.get("note"),
        evidence_current=(
            payload.get("evidence_current")
            if "evidence_current" in payload
            else None
        ),
    )
    return jsonify({"ok": True, "message": f"Ground-truth label recorded: {label_name}", "trial": state})


@app.post("/api/benchmark/label/undo")
def api_benchmark_label_undo():
    state = benchmark_recorder.undo_last_label()
    return jsonify({"ok": True, "message": "Last benchmark ground-truth label removed.", "trial": state})


@app.get("/api/benchmark/export")
def api_benchmark_export():
    benchmark = lightweight_current_benchmark_status()
    exported = benchmark_recorder.export_report(benchmark)
    fmt = request.args.get("format", "json").strip().lower()
    if fmt == "html":
        return send_file(exported["benchmark_report_html"], as_attachment=True)
    if fmt == "csv":
        return send_file(exported["configuration_results_csv"], as_attachment=True)
    return send_file(exported["report_json"], as_attachment=True)


def send_backend_command(command, source="web_dashboard", **command_fields):
    command_id = f"{int(time.time() * 1000)}-{os.getpid()}"
    payload = {
        "command": command,
        "command_id": command_id,
        "requested_at": time.time(),
        "requested_at_iso": utc_iso(),
        "source": source,
        **command_fields,
    }
    write_json_atomic(BACKEND_COMMAND_PATH, payload)
    return command_id


def persist_hand_camera_offset(
    offset_mm,
    desired_contact_gap_mm,
    *,
    enabled=True,
    source="dashboard_live_tuning",
    path=HAND_MOUNTED_CAMERA_CONFIG_PATH,
    calibration_path=HAND_OFFSET_CALIBRATION_PATH,
):
    """Persist the small calibration block while preserving YAML comments/order."""
    ok, update, reason = validate_hand_offset_update(offset_mm, desired_contact_gap_mm)
    if not ok:
        raise ValueError(reason)
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    offset = update["camera_to_hand_offset_mm"]
    replacements = {
        "camera_to_hand_offset_enabled": "true" if enabled else "false",
        "desired_contact_gap_mm": f"{update['desired_contact_gap_mm']:.1f}",
        "hand_offset_calibration_status": "APPROXIMATE_USER_TUNED_NOT_PHYSICALLY_VALIDATED",
        "hand_offset_physically_validated": "false",
        "hand_offset_source": str(source),
    }
    found = set()
    in_offset = False
    found_offset_axes = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "camera_to_hand_offset_mm:":
            in_offset = True
            continue
        if in_offset:
            if line.startswith("  ") and ":" in stripped:
                axis = stripped.split(":", 1)[0]
                if axis in offset:
                    lines[index] = f"  {axis}: {offset[axis]:.1f}"
                    found_offset_axes.add(axis)
                continue
            in_offset = False
        if not line.startswith((" ", "\t")) and ":" in stripped:
            key = stripped.split(":", 1)[0]
            if key in replacements:
                lines[index] = f"{key}: {replacements[key]}"
                found.add(key)
    missing_lines = []
    if len(found_offset_axes) != 3:
        missing_lines.extend([
            "camera_to_hand_offset_mm:",
            *(f"  {axis}: {offset[axis]:.1f}" for axis in ("x", "y", "z")),
        ])
    missing_lines.extend(f"{key}: {value}" for key, value in replacements.items() if key not in found)
    if missing_lines:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(missing_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)
    HAND_MOUNTED_CAMERA_CONFIG.update(update)
    HAND_MOUNTED_CAMERA_CONFIG.update({
        "camera_to_hand_offset_enabled": bool(enabled),
        "hand_offset_calibration_status": "APPROXIMATE_USER_TUNED_NOT_PHYSICALLY_VALIDATED",
        "hand_offset_physically_validated": False,
        "hand_offset_source": str(source),
    })
    calibration = {
        **update,
        **effective_hand_offset_close_range(
            update["camera_to_hand_offset_mm"],
            update["desired_contact_gap_mm"],
        ),
        "camera_to_hand_offset_enabled": bool(enabled),
        "camera_to_hand_offset_uncertainty_mm": normalized_offset_mm(
            HAND_MOUNTED_CAMERA_CONFIG.get("camera_to_hand_offset_uncertainty_mm"),
            {"x": 30.0, "y": 10.0, "z": 10.0},
        ),
        "calibration_status": "APPROXIMATE_USER_TUNED_NOT_PHYSICALLY_VALIDATED",
        "physically_validated": False,
        "source": str(source),
        "coordinate_frame": "realsense_color_optical_frame",
        "coordinate_convention": {"+x": "camera right", "+y": "camera down", "+z": "forward away from camera"},
        "saved_at": utc_iso(),
    }
    write_json_atomic(calibration_path, calibration)
    return calibration


def wait_backend_command(command_id, timeout_sec=5.0):
    deadline = time.time() + float(timeout_sec)
    last = {}
    while time.time() < deadline:
        status = read_json(BACKEND_COMMAND_STATUS_PATH, {})
        if status.get("command_id") == command_id:
            return bool(status.get("ok")), status
        if status:
            last = status
        time.sleep(0.12)
    return False, {
        "ok": False,
        "command_id": command_id,
        "message": "Timed out waiting for backend command acknowledgement.",
        "last_backend_status": last,
    }


def current_target_valid(status):
    if not status:
        return False, "No RealSense status yet."
    status_path = OUTPUT_DIR / "latest_realtime_status.json"
    if not status_path.exists() or time.time() - status_path.stat().st_mtime > 2.0:
        return False, "RealSense status is stale."
    if status.get("target_status") in {"LOST", "NO_TARGET"}:
        return False, "Target is lost."
    if status.get("bottle_detected") is False:
        return False, "Bottle is not detected."
    if status.get("full_bottle_visible") is False and status.get("target_status") not in {"TRACKING", "LOCKED"}:
        return False, "Full bottle is not visible."
    confidence = first_number(status.get("tracking_confidence"))
    if confidence is not None and confidence < 45.0 and status.get("target_status") in {"TRACKING", "LOCKED"}:
        return False, f"Tracking confidence too low ({confidence:.0f}%)."
    return True, "Target valid."


def wait_for_valid_target(timeout_sec=8.0):
    deadline = time.time() + float(timeout_sec)
    last_reason = "Waiting for RealSense status."
    while time.time() < deadline:
        status = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
        ok, reason = current_target_valid(status)
        if ok:
            return True, status, reason
        last_reason = reason
        time.sleep(0.2)
    return False, read_json(OUTPUT_DIR / "latest_realtime_status.json", {}), last_reason


def status_lock_frame(status):
    lock = (status or {}).get("target_lock") or {}
    locked = lock.get("locked_target") or {}
    locked_file = read_json(OUTPUT_DIR / "locked_target.json", {})
    locked_file_target = locked_file.get("locked_target") or locked_file
    return (
        lock.get("lock_frame_id")
        or locked.get("lock_frame_id")
        or locked_file_target.get("lock_frame_id")
    )


def lock_transaction_evidence(
    payload,
    *,
    expected_command_id=None,
    expected_writer_session_id=None,
    previous_lock_frame=None,
):
    """Validate one current backend lock, not a coincidental tracking state."""
    root = payload or {}
    lock = root.get("target_lock") or root
    locked = lock.get("locked_target") or {}
    lock_frame = lock.get("lock_frame_id") or locked.get("lock_frame_id")
    command_id = lock.get("lock_command_id") or locked.get("lock_command_id")
    writer_session_id = (
        root.get("writer_session_id")
        or lock.get("writer_session_id")
        or locked.get("writer_session_id")
    )
    target_status = lock.get("target_status") or root.get("target_status")
    lock_state = lock.get("lock_state") or root.get("lock_state")
    band_locked = lock.get("grasp_band_locked")
    if band_locked is None:
        band_locked = root.get("grasp_band_locked")
    reasons = []
    if expected_command_id is not None and command_id != expected_command_id:
        reasons.append("lock_command_mismatch")
    if (
        expected_writer_session_id is not None
        and writer_session_id != expected_writer_session_id
    ):
        reasons.append("writer_session_mismatch")
    if previous_lock_frame is not None and lock_frame == previous_lock_frame:
        reasons.append("lock_frame_not_advanced")
    if lock_frame is None:
        reasons.append("lock_frame_missing")
    if lock_state != "LOCKED":
        reasons.append("lock_state_not_locked")
    if target_status not in {"LOCKED", "TRACKING"}:
        reasons.append("target_not_current")
    if band_locked is not True:
        reasons.append("grasp_band_not_locked")
    if locked.get("valid") is not True:
        reasons.append("locked_target_not_valid")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "lock_frame_id": lock_frame,
        "lock_attempt_id": lock.get("attempt_id") or locked.get("attempt_id"),
        "locked_target_id": (
            lock.get("bottle_target_id") or locked.get("bottle_target_id")
        ),
        "lock_command_id": command_id,
        "writer_session_id": writer_session_id,
        "target_status": target_status,
        "lock_state": lock_state,
    }


def wait_for_locked_target(
    previous_lock_frame=None,
    timeout_sec=8.0,
    *,
    expected_command_id=None,
    expected_writer_session_id=None,
):
    deadline = time.time() + float(timeout_sec)
    last_status = {}
    last_evidence = {}
    while time.time() < deadline:
        status = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
        last_status = status
        locked_file = read_json(OUTPUT_DIR / "locked_target.json", {})
        for source, payload in (
            ("latest_realtime_status", status),
            ("locked_target", locked_file),
        ):
            evidence = lock_transaction_evidence(
                payload,
                expected_command_id=expected_command_id,
                expected_writer_session_id=expected_writer_session_id,
                previous_lock_frame=previous_lock_frame,
            )
            evidence["source"] = source
            last_evidence = evidence
            if evidence["valid"]:
                matched = dict(status or payload)
                matched["fresh_lock_transaction"] = evidence
                return True, matched, evidence["lock_frame_id"]
        time.sleep(0.2)
    failed = dict(last_status)
    failed["fresh_lock_transaction"] = last_evidence
    return False, failed, status_lock_frame(last_status)


def quality_result_evidence(
    matrix,
    *,
    expected_lock_frame=None,
    expected_lock_attempt_id=None,
    expected_writer_session_id=None,
):
    """Validate that analytic quality belongs to the fresh current lock."""
    matrix = matrix or {}
    status = str(matrix.get("quality_matrix_status") or "").lower()
    input_source = matrix.get("quality_input_source") or matrix.get("contact_source")
    reasons = []
    if status != "updated":
        reasons.append(
            "analytic_quality_stale"
            if status == "stale_needs_recompute"
            else "analytic_quality_not_updated"
        )
    if (
        expected_lock_frame is not None
        and matrix.get("lock_frame_id") != expected_lock_frame
    ):
        reasons.append("quality_lock_frame_mismatch")
    if (
        expected_lock_attempt_id is not None
        and matrix.get("lock_attempt_id") != expected_lock_attempt_id
    ):
        reasons.append("quality_lock_attempt_mismatch")
    if (
        expected_writer_session_id is not None
        and matrix.get("writer_session_id") != expected_writer_session_id
    ):
        reasons.append("quality_writer_session_mismatch")
    if not matrix.get("quality_compute_id"):
        reasons.append("quality_compute_id_missing")
    if matrix.get("epsilon") is None:
        reasons.append("analytic_epsilon_missing")
    if matrix.get("epsilon_score") is None:
        reasons.append("analytic_quality_score_missing")
    if matrix.get("epsilon_force_closure") is None and matrix.get("force_closure") is None:
        reasons.append("force_closure_estimate_missing")
    if input_source not in {
        "planned_contact",
        "detected_finger",
        "detected_gripper",
        "detected_gripper_contact",
    }:
        reasons.append("quality_contact_source_unavailable")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "quality_matrix_status": status,
        "quality_compute_id": matrix.get("quality_compute_id"),
        "lock_frame_id": matrix.get("lock_frame_id"),
        "lock_attempt_id": matrix.get("lock_attempt_id"),
        "writer_session_id": matrix.get("writer_session_id"),
        "quality_input_source": input_source,
    }


def wait_for_quality_result(
    start_time,
    timeout_sec=18.0,
    *,
    expected_lock_frame=None,
    expected_lock_attempt_id=None,
    expected_writer_session_id=None,
):
    deadline = time.time() + float(timeout_sec)
    last_matrix = {}
    while time.time() < deadline:
        matrix_path = OUTPUT_DIR / "realtime_wrench_matrix_status.json"
        matrix = read_json(matrix_path, {})
        last_matrix = matrix
        if matrix_path.exists() and matrix_path.stat().st_mtime >= start_time:
            status = str(matrix.get("quality_matrix_status") or "").lower()
            if status == "failed":
                return False, matrix
            evidence = quality_result_evidence(
                matrix,
                expected_lock_frame=expected_lock_frame,
                expected_lock_attempt_id=expected_lock_attempt_id,
                expected_writer_session_id=expected_writer_session_id,
            )
            if evidence["valid"]:
                accepted = dict(matrix)
                accepted["fresh_quality_transaction"] = evidence
                return True, accepted
            last_matrix = dict(matrix)
            last_matrix["fresh_quality_transaction"] = evidence
        time.sleep(0.25)
    if last_matrix:
        evidence = last_matrix.get("fresh_quality_transaction") or {}
        reasons = evidence.get("reasons") or []
        last_matrix.setdefault(
            "quality_matrix_error",
            "Quality result did not match the fresh lock: " + ", ".join(reasons)
            if reasons
            else "Timed out waiting for quality matrix result.",
        )
        return False, last_matrix
    return False, {"quality_matrix_error": "Timed out waiting for quality matrix result."}


def fresh_lock_failure_code(message, *, stage=None, worker_busy=False):
    """Map operator-facing prose to a stable machine-readable failure code."""
    text = str(message or "").lower()
    stage = str(stage or "").lower()
    if worker_busy or "already running" in text or "worker" in text and "busy" in text:
        return "quality_worker_busy"
    if "intrinsic" in text:
        return "missing_intrinsics"
    if "depth" in text:
        return "invalid_contact_depth" if "contact" in text else "missing_depth"
    if "bottle is not detected" in text or "no bottle" in text or "no target" in text:
        return "no_bottle"
    if stage == "target" or any(
        phrase in text
        for phrase in ("target is lost", "full bottle", "tracking confidence", "target not valid")
    ):
        return "unstable_bottle"
    if stage == "lock" or "locked target" in text or "lock command" in text:
        return "invalid_band"
    if stage == "quality" or "quality" in text:
        return "quality_compute_failed"
    if "backend" in text:
        return "backend_unavailable"
    return "quality_compute_failed"


def fresh_lock_quality_worker():
    result = None
    try:
        if not realsense_backend_live(max_age_sec=3.0):
            update_fresh_status(
                "failed",
                "Backend unavailable",
                error="RealSense backend is not running.",
                fresh_lock_quality_failure_reason="RealSense backend is not running.",
                fresh_lock_quality_failure_code="backend_unavailable",
            )
            raise RuntimeError("RealSense backend is not running.")
        update_fresh_status(
            "running",
            "Clearing old lock",
            fresh_lock_quality_request_received=True,
            fresh_lock_quality_worker_busy=False,
            fresh_lock_quality_target_valid=None,
            fresh_lock_quality_lock_succeeded=False,
            fresh_lock_quality_matrix_started=False,
            fresh_lock_quality_matrix_completed=False,
            fresh_lock_quality_failure_reason=None,
            fresh_lock_quality_failure_code=None,
            last_fresh_lock_quality_result=None,
        )
        reset_id = send_backend_command("reset_lock_clear_matrix", source="fresh_lock_quality")
        ok, ack = wait_backend_command(reset_id, timeout_sec=5.0)
        if not ok:
            raise RuntimeError(ack.get("message") or "Could not clear old lock.")

        update_fresh_status("running", "Waiting for valid target")
        ok, target_status, reason = wait_for_valid_target(timeout_sec=15.0)
        if not ok:
            message = f"Cannot compute quality: target not valid. {reason}"
            update_fresh_status(
                "failed",
                "Target not valid",
                error=message,
                fresh_lock_quality_target_valid=False,
                fresh_lock_quality_failure_reason=message,
                fresh_lock_quality_failure_code=fresh_lock_failure_code(message, stage="target"),
            )
            raise RuntimeError(message)
        update_fresh_status("running", "Target valid", fresh_lock_quality_target_valid=True)
        previous_lock = status_lock_frame(target_status)

        update_fresh_status("running", "Locking grasp")
        lock_id = send_backend_command("lock_grasp", source="fresh_lock_quality")
        ok, ack = wait_backend_command(lock_id, timeout_sec=5.0)
        if not ok:
            message = ack.get("message") or "Lock command was not accepted."
            update_fresh_status(
                "failed",
                "Lock request rejected",
                error=message,
                fresh_lock_quality_failure_reason=message,
                fresh_lock_quality_failure_code=fresh_lock_failure_code(message, stage="lock"),
            )
            raise RuntimeError(message)
        ok, locked_status, lock_frame = wait_for_locked_target(
            previous_lock_frame=previous_lock,
            timeout_sec=15.0,
            expected_command_id=lock_id,
            expected_writer_session_id=ack.get("writer_session_id"),
        )
        if not ok:
            message = "Lock command accepted, but no locked target became available."
            update_fresh_status(
                "failed",
                "Lock unavailable",
                error=message,
                fresh_lock_quality_failure_reason=message,
                fresh_lock_quality_failure_code="invalid_band",
            )
            raise RuntimeError(message)
        lock_transaction = locked_status.get("fresh_lock_transaction") or {}
        update_fresh_status(
            "running",
            "Lock succeeded",
            fresh_lock_quality_lock_succeeded=True,
            fresh_lock_quality_lock_command_id=lock_id,
            fresh_lock_quality_lock_attempt_id=lock_transaction.get("lock_attempt_id"),
            fresh_lock_quality_locked_target_id=lock_transaction.get("locked_target_id"),
            fresh_lock_quality_writer_session_id=lock_transaction.get("writer_session_id"),
        )

        update_fresh_status("running", "Computing quality")
        quality_start = time.time()
        quality_id = send_backend_command("compute_quality", source="fresh_lock_quality")
        ok, ack = wait_backend_command(quality_id, timeout_sec=5.0)
        if not ok:
            message = ack.get("message") or "Quality compute command was not accepted."
            worker_busy = bool(ack.get("quality_matrix_worker_busy", "busy" in str(message).lower()))
            update_fresh_status(
                "failed",
                "Quality request rejected",
                error=message,
                fresh_lock_quality_worker_busy=worker_busy,
                fresh_lock_quality_target_valid=ack.get("quality_matrix_target_valid"),
                fresh_lock_quality_matrix_started=False,
                fresh_lock_quality_failure_reason=message,
                fresh_lock_quality_failure_code=fresh_lock_failure_code(
                    message,
                    stage="quality",
                    worker_busy=worker_busy,
                ),
            )
            raise RuntimeError(message)
        update_fresh_status(
            "running",
            "Quality matrix started",
            fresh_lock_quality_worker_busy=False,
            fresh_lock_quality_target_valid=ack.get("quality_matrix_target_valid", True),
            fresh_lock_quality_matrix_started=bool(ack.get("quality_matrix_started", True)),
            fresh_lock_quality_failure_reason=None,
            fresh_lock_quality_failure_code=None,
        )
        ok, matrix = wait_for_quality_result(
            quality_start,
            timeout_sec=20.0,
            expected_lock_frame=lock_frame,
            expected_lock_attempt_id=lock_transaction.get("lock_attempt_id"),
            expected_writer_session_id=lock_transaction.get("writer_session_id"),
        )
        if not ok:
            message = matrix.get("quality_matrix_error") or matrix.get("failure_reason") or "Quality compute failed."
            result = {
                "ok": False,
                "message": message,
                "lock_frame_id": lock_frame,
                "quality_compute_id": matrix.get("quality_compute_id"),
            }
            update_fresh_status(
                "failed",
                "Quality matrix failed",
                error=message,
                result=result,
                fresh_lock_quality_matrix_completed=False,
                fresh_lock_quality_failure_reason=message,
                fresh_lock_quality_failure_code=fresh_lock_failure_code(message, stage="quality"),
            )
            return

        result = {
            "ok": True,
            "message": "Fresh lock and quality computed.",
            "lock_frame_id": lock_frame or matrix.get("lock_frame_id"),
            "quality_compute_id": matrix.get("quality_compute_id"),
            "quality_label": matrix.get("quality_label_display") or matrix.get("quality_label"),
        }
        update_fresh_status(
            "done",
            "Done",
            error=None,
            result=result,
            fresh_lock_quality_matrix_completed=True,
            fresh_lock_quality_failure_reason=None,
            fresh_lock_quality_failure_code=None,
        )
    except Exception as exc:
        message = str(exc)
        result = result or {"ok": False, "message": message}
        update_fresh_status(
            "failed",
            "Failed",
            error=message,
            result=result,
            fresh_lock_quality_failure_reason=message,
            fresh_lock_quality_failure_code=fresh_lock_failure_code(message),
        )


@app.post("/api/realsense-command/<command>")
def api_realsense_command(command):
    allowed = {
        "lock_grasp": "Lock current RealSense grasp on the next valid evaluation.",
        "start_approach": "Manual approach phase acknowledged for monitoring and camera-frame guidance only; no actuation is commanded.",
        "mark_gripper_closed": "Manual operator declaration recorded; post-grasp estimation will start without claiming sensor-confirmed contact.",
        "mark_gripper_open": "Manual release declaration recorded; return to pre-close monitoring without claiming sensor-confirmed release.",
        "compute_quality": "Compute grasp quality from the current locked snapshot.",
        "recompute_quality": "Recompute grasp quality from the current locked snapshot.",
        "reset_lock_clear_matrix": "Reset the RealSense lock and clear matrix status.",
        "refresh_pointcloud": "Refresh RealSense point cloud preview and PLY outputs on the next RGB-D update.",
    }
    if command not in allowed:
        return jsonify({"ok": False, "message": f"Unknown RealSense command: {command}"}), 404
    command_id = send_backend_command(command, source="web_dashboard")
    ok, backend_status = wait_backend_command(command_id)
    message = backend_status.get("message") or allowed[command]
    response = {
        "ok": bool(ok),
        "message": message,
        "command_id": command_id,
        "backend_acknowledged": backend_status.get("command_id") == command_id,
        "backend_status": backend_status,
    }
    return jsonify(response), 200 if ok else 409


@app.post("/api/hand-camera-offset")
def api_hand_camera_offset():
    payload = request.get_json(silent=True) or {}
    offset = payload.get("camera_to_hand_offset_mm")
    if offset is None:
        offset = {axis: payload.get(axis) for axis in ("x", "y", "z")}
    desired_gap = payload.get("desired_contact_gap_mm", DEFAULT_DESIRED_CONTACT_GAP_MM)
    enabled = parse_bool(payload.get("camera_to_hand_offset_enabled"), default=True)
    ok, update, reason = validate_hand_offset_update(offset, desired_gap)
    if not ok:
        return jsonify({"ok": False, "message": reason}), 400
    source = str(payload.get("hand_offset_source") or "dashboard_live_tuning")
    try:
        calibration = persist_hand_camera_offset(
            update["camera_to_hand_offset_mm"],
            update["desired_contact_gap_mm"],
            enabled=enabled,
            source=source,
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Could not save hand offset: {exc}"}), 500
    command_id = send_backend_command(
        "set_hand_camera_offset",
        source="web_dashboard",
        camera_to_hand_offset_mm=update["camera_to_hand_offset_mm"],
        desired_contact_gap_mm=update["desired_contact_gap_mm"],
        camera_to_hand_offset_enabled=enabled,
        hand_offset_source=source,
    )
    backend_ok, backend_status = wait_backend_command(command_id)
    message = (
        "Approximate hand offset saved and applied live. It is not physically calibrated."
        if backend_ok
        else "Approximate hand offset saved; restart RealSense to apply it because the live backend did not acknowledge the update."
    )
    return jsonify({
        "ok": bool(backend_ok),
        "saved": True,
        "message": message,
        "calibration": calibration,
        "effective_guidance": effective_hand_offset_close_range(
            update["camera_to_hand_offset_mm"],
            update["desired_contact_gap_mm"],
        ),
        "command_id": command_id,
        "backend_status": backend_status,
    }), 200 if backend_ok else 409


@app.post("/api/multiview-command/<command>")
def api_multiview_command(command):
    allowed = {
        "start_multiview_scan": "Start multi-view bottle reconstruction from accepted RGB-D keyframes.",
        "pause_multiview_scan": "Pause multi-view reconstruction.",
        "resume_multiview_scan": "Resume multi-view reconstruction.",
        "finish_multiview_scan": "Finish the current multi-view reconstruction session.",
        "reset_multiview_scan": "Reset only the multi-view reconstruction session.",
        "save_multiview_ply": "Save accumulated multi-view PLY outputs.",
        "save_accumulated_bottle_ply": "Save the accumulated bottle PLY output.",
        "save_scene_map_ply": "Save the accumulated scene map PLY output.",
        "save_multiview_report": "Save the current multi-view reconstruction report.",
        "save_multiview_mesh": "Save reconstructed bottle mesh/placeholder PLY.",
        "rebuild_multiview_preview": "Rebuild the browser preview for the accumulated model.",
    }
    if command not in allowed:
        return jsonify({"ok": False, "message": f"Unknown multi-view command: {command}"}), 404
    command_id = send_backend_command(command, source="web_dashboard_multiview")
    return jsonify({"ok": True, "message": allowed[command], "command_id": command_id})


@app.post("/api/fresh_lock_quality")
def api_fresh_lock_quality():
    global fresh_lock_quality_thread
    with fresh_lock_quality_lock:
        running = bool(fresh_lock_quality_thread and fresh_lock_quality_thread.is_alive())
        if running:
            update_fresh_status(
                "running",
                "Busy",
                error="Fresh Lock + Quality is already running.",
                fresh_lock_quality_requested=True,
                fresh_lock_quality_request_received=True,
                fresh_lock_quality_worker_busy=True,
                fresh_lock_quality_failure_reason="Fresh Lock + Quality is already running.",
                fresh_lock_quality_failure_code="quality_worker_busy",
            )
            return jsonify({"ok": False, "message": "Fresh Lock + Quality is already running.", **dict(fresh_lock_quality_status)}), 409
        update_fresh_status(
            "running",
            "Request received",
            error=None,
            result=None,
            fresh_lock_quality_requested=True,
            fresh_lock_quality_request_received=True,
            fresh_lock_quality_worker_busy=False,
            fresh_lock_quality_target_valid=None,
            fresh_lock_quality_lock_succeeded=False,
            fresh_lock_quality_matrix_started=False,
            fresh_lock_quality_matrix_completed=False,
            fresh_lock_quality_failure_reason=None,
            fresh_lock_quality_failure_code=None,
        )
        fresh_lock_quality_thread = threading.Thread(target=fresh_lock_quality_worker, daemon=True)
        fresh_lock_quality_thread.start()
    return jsonify({"ok": True, "message": "Fresh Lock + Quality started.", **fresh_status_snapshot()})


DEMO_TIMELINE_MAX_LINES = 120
DEMO_TIMELINE_MAX_BYTES = 8 * 1024 * 1024
DEMO_STAGE_SCREENSHOTS = (
    "operator_start.png",
    "locked_band.png",
    "approach.png",
    "davis_closeup.png",
    "post_grasp.png",
)
DEMO_EXPORT_SCREENSHOTS = (
    "hand_grasp_demo_screen.png",
    "operator_snapshot.png",
    "davis_snapshot.png",
    "quality_snapshot.png",
)


def _bounded_tail_lines(path, max_lines=DEMO_TIMELINE_MAX_LINES, max_bytes=DEMO_TIMELINE_MAX_BYTES):
    """Return complete trailing lines without reading an unbounded trace."""
    path = Path(path)
    if not path.is_file() or max_lines <= 0 or max_bytes <= 0:
        return [], 0
    source_size = path.stat().st_size
    if source_size == 0:
        return [], 0
    position = source_size
    chunks = []
    bytes_read = 0
    newline_count = 0
    with path.open("rb") as handle:
        while position > 0 and bytes_read < max_bytes and newline_count <= max_lines:
            read_size = min(64 * 1024, position, max_bytes - bytes_read)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            bytes_read += len(chunk)
            newline_count += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    if position > 0:
        first_newline = data.find(b"\n")
        data = b"" if first_newline < 0 else data[first_newline + 1 :]
    return [line for line in data.splitlines() if line.strip()][-max_lines:], source_size


def _demo_first_present(*values):
    return next((value for value in values if value is not None), None)


def _demo_evidence_identity(payload, *, kind="realsense"):
    """Extract the lock provenance carried by one structured artifact."""
    payload = payload if isinstance(payload, dict) else {}
    target_lock = payload.get("target_lock") if isinstance(payload.get("target_lock"), dict) else {}
    locked_target = (
        target_lock.get("locked_target")
        if isinstance(target_lock.get("locked_target"), dict)
        else {}
    )
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    if kind == "davis":
        return {
            "writer_session_id": payload.get("davis_target_lock_source_writer_session_id"),
            "lock_frame_id": payload.get("davis_target_lock_source_lock_frame_id"),
            "attempt_id": payload.get("davis_target_lock_source_attempt_id"),
            "bottle_target_id": payload.get("davis_target_lock_source_bottle_target_id"),
        }
    return {
        "writer_session_id": _demo_first_present(
            payload.get("writer_session_id"),
            target_lock.get("writer_session_id"),
            locked_target.get("writer_session_id"),
            baseline.get("writer_session_id"),
        ),
        "lock_frame_id": _demo_first_present(
            payload.get("lock_frame_id"),
            target_lock.get("lock_frame_id"),
            locked_target.get("lock_frame_id"),
            baseline.get("lock_frame_id"),
        ),
        "attempt_id": _demo_first_present(
            payload.get("lock_attempt_id"),
            payload.get("attempt_id"),
            target_lock.get("attempt_id"),
            locked_target.get("attempt_id"),
            baseline.get("attempt_id"),
        ),
        "bottle_target_id": _demo_first_present(
            payload.get("locked_bottle_target_id"),
            payload.get("bottle_target_id"),
            target_lock.get("bottle_target_id"),
            locked_target.get("bottle_target_id"),
            baseline.get("bottle_target_id"),
        ),
        "quality_compute_id": _demo_first_present(
            payload.get("quality_compute_id"),
            baseline.get("quality_compute_id"),
        ),
    }


def _demo_current_lock_identity(realtime):
    """Treat latest_realtime_status as the sole authority for an export."""
    identity = _demo_evidence_identity(realtime, kind="realsense")
    realtime = realtime if isinstance(realtime, dict) else {}
    target_lock = realtime.get("target_lock") if isinstance(realtime.get("target_lock"), dict) else {}
    locked_target = target_lock.get("locked_target") if isinstance(target_lock.get("locked_target"), dict) else {}
    locked_band = (
        realtime.get("locked_grasp_band")
        if isinstance(realtime.get("locked_grasp_band"), dict)
        else target_lock.get("locked_grasp_band")
        if isinstance(target_lock.get("locked_grasp_band"), dict)
        else locked_target.get("locked_grasp_band")
        if isinstance(locked_target.get("locked_grasp_band"), dict)
        else {}
    )
    post_grasp_baseline = (
        realtime.get("post_grasp_baseline")
        if realtime.get("post_grasp_active") is True
        and isinstance(realtime.get("post_grasp_baseline"), dict)
        else {}
    )
    identity["quality_compute_id"] = _demo_first_present(
        post_grasp_baseline.get("quality_compute_id"),
        locked_band.get("locked_quality_compute_id"),
        locked_target.get("locked_quality_compute_id"),
        realtime.get("quality_compute_id"),
        identity.get("quality_compute_id"),
    )
    return identity


def _demo_identity_validation(payload, authority, *, kind="realsense"):
    evidence = _demo_evidence_identity(payload, kind=kind)
    authority = dict(authority or {})
    reasons = []
    for key in ("writer_session_id", "lock_frame_id"):
        expected = authority.get(key)
        actual = evidence.get(key)
        if expected is None:
            reasons.append(f"authoritative_{key}_missing")
        elif actual is None:
            reasons.append(f"evidence_{key}_missing")
        elif str(actual) != str(expected):
            reasons.append(f"{key}_mismatch:expected={expected}:actual={actual}")
    for key in ("attempt_id", "bottle_target_id"):
        expected = authority.get(key)
        actual = evidence.get(key)
        if expected is not None and actual is not None and str(actual) != str(expected):
            reasons.append(f"{key}_mismatch:expected={expected}:actual={actual}")
    if kind == "quality" and authority.get("quality_compute_id") is not None:
        expected = authority.get("quality_compute_id")
        actual = evidence.get("quality_compute_id")
        if actual is None:
            reasons.append("evidence_quality_compute_id_missing")
        elif str(actual) != str(expected):
            reasons.append(
                f"quality_compute_id_mismatch:expected={expected}:actual={actual}"
            )
    return {
        "valid": not reasons,
        "kind": kind,
        "authoritative_identity": authority,
        "evidence_identity": evidence,
        "reasons": reasons or ["current_lock_identity_match"],
    }


def _demo_authority_validation(realtime, authority):
    reasons = []
    if not isinstance(realtime, dict) or not realtime:
        reasons.append("latest_realtime_status_unavailable")
    for key in ("writer_session_id", "lock_frame_id"):
        if (authority or {}).get(key) is None:
            reasons.append(f"authoritative_{key}_missing")
    return {
        "valid": not reasons,
        "kind": "authoritative_realtime",
        "authoritative_identity": dict(authority or {}),
        "evidence_identity": dict(authority or {}),
        "reasons": reasons or ["authoritative_current_lock_identity_complete"],
    }


def capture_hand_demo_stage_screenshot(stage, png_payload, *, realtime=None):
    """Save a real browser viewport capture with explicit lock provenance."""
    filename = f"{str(stage or '').strip().lower()}.png"
    if filename not in {*DEMO_STAGE_SCREENSHOTS, *DEMO_EXPORT_SCREENSHOTS}:
        return {"ok": False, "message": f"Unknown hand-demo stage: {stage}"}
    payload = bytes(png_payload or b"")
    if len(payload) < 8 or len(payload) > 12 * 1024 * 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return {"ok": False, "message": "Stage capture must be a PNG smaller than 12 MiB."}
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim < 2 or decoded.shape[0] < 90 or decoded.shape[1] < 160:
        return {"ok": False, "message": "Stage capture is not a valid dashboard-sized PNG."}

    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {}) if realtime is None else realtime
    realtime = realtime if isinstance(realtime, dict) else {}
    authority = _demo_current_lock_identity(realtime)
    authority_validation = _demo_authority_validation(realtime, authority)
    if filename != "operator_start.png" and not authority_validation["valid"]:
        return {
            "ok": False,
            "message": "A complete current lock identity is required before this stage can be captured.",
            "identity_validation": authority_validation,
        }
    if filename == "operator_start.png" and authority.get("writer_session_id") is None:
        return {"ok": False, "message": "Current RealSense writer session is unavailable."}

    stage_dir = OUTPUT_DIR / "hand_demo_stage_screenshots"
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = stage_dir / filename
    write_bytes_atomic(destination, payload)
    captured_identity = dict(authority)
    provenance_identity = (
        {
            "writer_session_id": authority.get("writer_session_id"),
            "lock_frame_id": None,
            "attempt_id": None,
            "bottle_target_id": None,
        }
        if filename == "operator_start.png"
        else dict(authority)
    )
    provenance = {
        "schema_version": "hand_demo_stage_capture_v1",
        "artifact_name": filename,
        "artifact_mtime_ns": destination.stat().st_mtime_ns,
        **provenance_identity,
        "capture_stage": filename.removesuffix(".png"),
        "capture_kind": "browser_viewport_png",
        "captured_at": utc_iso(),
        "captured_at_unix": time.time(),
        "viewport_width_px": int(decoded.shape[1]),
        "viewport_height_px": int(decoded.shape[0]),
        "capture_identity_at_time": captured_identity,
        "pending_lock_link": filename == "operator_start.png",
        "lock_link_method": None,
    }
    write_json_atomic(destination.with_name(destination.name + ".json"), provenance)

    linked_operator_start = False
    if filename != "operator_start.png":
        operator_path = stage_dir / "operator_start.png"
        operator_sidecar = operator_path.with_name(operator_path.name + ".json")
        prior = read_json(operator_sidecar, {}) if operator_path.is_file() else {}
        same_writer = (
            isinstance(prior, dict)
            and prior.get("writer_session_id") is not None
            and str(prior.get("writer_session_id")) == str(authority.get("writer_session_id"))
        )
        if same_writer and prior.get("pending_lock_link") is True:
            prior.update({
                **authority,
                "pending_lock_link": False,
                "lock_link_method": "next_current_lock_same_writer_session",
                "lock_linked_at": utc_iso(),
                "linked_by_stage": filename,
            })
            write_json_atomic(operator_sidecar, prior)
            linked_operator_start = True
    return {
        "ok": True,
        "message": f"Captured hand-demo stage: {filename}",
        "stage": filename.removesuffix(".png"),
        "path": str(destination),
        "identity": provenance_identity,
        "operator_start_linked_to_current_lock": linked_operator_start,
    }


def _demo_current_post_grasp_action():
    """Read the in-process DAVIS action marker, not a possibly stale file."""
    with davis_lock:
        analyzer = davis_analyzer
        return {
            "post_grasp_monitoring_action_id": getattr(analyzer, "close_action_id", None),
            "post_grasp_monitoring_started_at": getattr(
                analyzer, "close_action_started_at", None
            ),
        }


def _demo_post_grasp_action_validation(payload, action):
    payload = payload if isinstance(payload, dict) else {}
    action = dict(action or {})
    expected_id = action.get("post_grasp_monitoring_action_id")
    expected_start = action.get("post_grasp_monitoring_started_at")
    actual_id = payload.get("post_grasp_monitoring_action_id")
    actual_start = payload.get("post_grasp_monitoring_started_at")
    reasons = []
    if expected_id is None:
        reasons.append("authoritative_post_grasp_action_id_unavailable")
    elif actual_id is None:
        reasons.append("evidence_post_grasp_action_id_missing")
    elif str(actual_id) != str(expected_id):
        reasons.append(
            f"post_grasp_action_id_mismatch:expected={expected_id}:actual={actual_id}"
        )
    if expected_start is not None:
        if actual_start is None:
            reasons.append("evidence_post_grasp_started_at_missing")
        else:
            try:
                matches = abs(float(actual_start) - float(expected_start)) <= 1e-6
            except (TypeError, ValueError):
                matches = str(actual_start) == str(expected_start)
            if not matches:
                reasons.append(
                    f"post_grasp_started_at_mismatch:expected={expected_start}:actual={actual_start}"
                )
    return {
        "valid": not reasons,
        "authoritative_action": action,
        "evidence_action": {
            "post_grasp_monitoring_action_id": actual_id,
            "post_grasp_monitoring_started_at": actual_start,
        },
        "reasons": reasons or ["current_post_grasp_action_identity_match"],
    }


def _demo_sidecar_provenance(path):
    """Read explicit image provenance without accepting filename age as identity."""
    path = Path(path)
    candidates = (path.with_name(path.name + ".json"), path.with_suffix(".json"))
    for candidate in candidates:
        value = read_json(candidate, None) if candidate.is_file() else None
        if isinstance(value, dict):
            return candidate, value
    directory_manifest = path.parent / "provenance.json"
    manifest = read_json(directory_manifest, None) if directory_manifest.is_file() else None
    if isinstance(manifest, dict):
        entries = manifest.get("screenshots") or manifest.get("artifacts") or manifest
        value = entries.get(path.name) if isinstance(entries, dict) else None
        if isinstance(value, dict):
            return directory_manifest, value
    return None, {}


def _demo_image_validation(path, authority, realtime, *, require_sidecar=False):
    """Validate an image through a sidecar or the current overlay status link."""
    path = Path(path)
    sidecar, provenance = _demo_sidecar_provenance(path)
    if sidecar is not None:
        validation = _demo_identity_validation(provenance, authority)
        named = provenance.get("artifact_name") or provenance.get("screenshot_name")
        if named is None:
            validation["valid"] = False
            validation["reasons"].append("image_provenance_artifact_name_missing")
        elif str(named) != path.name:
            validation["valid"] = False
            validation["reasons"].append(
                f"image_provenance_artifact_name_mismatch:expected={path.name}:actual={named}"
            )
        expected_mtime_ns = provenance.get("artifact_mtime_ns")
        actual_mtime_ns = None
        if expected_mtime_ns is None:
            validation["valid"] = False
            validation["reasons"].append("image_provenance_artifact_mtime_ns_missing")
        else:
            try:
                actual_mtime_ns = path.stat().st_mtime_ns
            except OSError:
                actual_mtime_ns = None
            if actual_mtime_ns is None or str(actual_mtime_ns) != str(expected_mtime_ns):
                validation["valid"] = False
                validation["reasons"].append(
                    f"image_provenance_artifact_mtime_ns_mismatch:expected={expected_mtime_ns}:actual={actual_mtime_ns}"
                )
        validation.update({
            "source": str(path),
            "provenance_source": str(sidecar),
            "artifact_mtime_ns": actual_mtime_ns,
        })
        return validation
    if not require_sidecar and path.name == "latest_overlay_live.jpg":
        overlay_path = str((realtime or {}).get("overlay_current_path") or "")
        overlay_frame_id = (realtime or {}).get("overlay_frame_id")
        current_frame_id = (realtime or {}).get("frame_id")
        reasons = []
        if Path(overlay_path).name != path.name:
            reasons.append("latest_realtime_status_overlay_path_mismatch")
        if overlay_frame_id is None or current_frame_id is None:
            reasons.append("latest_realtime_status_overlay_frame_provenance_missing")
        elif str(overlay_frame_id) != str(current_frame_id):
            reasons.append(
                f"overlay_frame_id_mismatch:expected={current_frame_id}:actual={overlay_frame_id}"
            )
        if (realtime or {}).get("overlay_stale") is not False:
            reasons.append("latest_realtime_status_overlay_not_explicitly_current")
        expected_mtime = first_number(
            (realtime or {}).get("dashboard_overlay_cache_buster")
        )
        try:
            actual_mtime = path.stat().st_mtime
        except OSError:
            actual_mtime = None
        if expected_mtime is None:
            reasons.append("latest_realtime_status_overlay_file_version_missing")
        elif actual_mtime is None or abs(float(actual_mtime) - float(expected_mtime)) > 1e-4:
            reasons.append(
                f"overlay_file_version_mismatch:expected={expected_mtime}:actual={actual_mtime}"
            )
        authority_status = _demo_authority_validation(realtime, authority)
        if not authority_status["valid"]:
            reasons.extend(authority_status["reasons"])
        return {
            "valid": not reasons,
            "kind": "realtime_overlay_link",
            "authoritative_identity": dict(authority or {}),
            "evidence_identity": dict(authority or {}),
            "source": str(path),
            "provenance_source": "latest_realtime_status.json",
            "artifact_mtime": actual_mtime,
            "reasons": reasons or ["current_realtime_overlay_link_match"],
        }
    return {
        "valid": False,
        "kind": "image",
        "authoritative_identity": dict(authority or {}),
        "evidence_identity": {},
        "source": str(path),
        "provenance_source": None,
        "reasons": ["image_identity_provenance_missing"],
    }


def _demo_copied_image_version_reason(validation, copied_path):
    """Recheck the copied inode version to close a source-update race."""
    copied_path = Path(copied_path)
    try:
        copied_stat = copied_path.stat()
    except OSError:
        return "copied_image_unavailable_after_copy"
    if validation.get("kind") == "realtime_overlay_link":
        expected = validation.get("artifact_mtime")
        if expected is None or abs(float(copied_stat.st_mtime) - float(expected)) > 1e-4:
            return (
                f"copied_overlay_file_version_mismatch:expected={expected}:"
                f"actual={copied_stat.st_mtime}"
            )
    elif validation.get("provenance_source"):
        expected = validation.get("artifact_mtime_ns")
        if expected is None or str(copied_stat.st_mtime_ns) != str(expected):
            return (
                f"copied_image_artifact_mtime_ns_mismatch:expected={expected}:"
                f"actual={copied_stat.st_mtime_ns}"
            )
    return None


def _copy_bounded_timeline(
    source_candidates,
    destination,
    *,
    authority=None,
    identity_kind=None,
    post_grasp_action=None,
    require_post_grasp_action=False,
):
    source = next((Path(path) for path in source_candidates if Path(path).is_file()), None)
    lines, source_size = _bounded_tail_lines(source) if source is not None else ([], 0)
    accepted = []
    rejected_reasons = {}
    if identity_kind is None:
        accepted = lines
    else:
        for line in lines:
            try:
                row = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                row = None
            if not isinstance(row, dict):
                reasons = ["timeline_row_not_valid_json_object"]
            else:
                validation = _demo_identity_validation(row, authority, kind=identity_kind)
                reasons = [] if validation["valid"] else validation["reasons"]
                if not reasons and require_post_grasp_action:
                    action_validation = _demo_post_grasp_action_validation(
                        row,
                        post_grasp_action,
                    )
                    if not action_validation["valid"]:
                        reasons = action_validation["reasons"]
            if reasons:
                for reason in reasons:
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            else:
                accepted.append(line)
    write_bytes_atomic(destination, b"\n".join(accepted) + (b"\n" if accepted else b""))
    validation_reasons = (
        [f"{reason} ({count} rows)" for reason, count in sorted(rejected_reasons.items())]
        if rejected_reasons
        else (["all_captured_rows_match_current_lock_identity"] if accepted else ["no_timeline_rows_available"])
    )
    return {
        "available": source is not None,
        "source": None if source is None else str(source),
        "captured_as": str(Path(destination).name),
        "captured_line_count": len(accepted),
        "rejected_line_count": len(lines) - len(accepted),
        "source_size_bytes": source_size,
        "tail_line_limit": DEMO_TIMELINE_MAX_LINES,
        "tail_byte_read_limit": DEMO_TIMELINE_MAX_BYTES,
        "identity_validation": {
            "valid": bool(accepted),
            "kind": identity_kind,
            "reasons": validation_reasons,
        },
        "source_had_rejected_rows": bool(rejected_reasons),
    }, accepted


def _jsonl_rows(lines):
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _first_json_source(paths):
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        value = read_json(path, None)
        if isinstance(value, dict):
            return path, value
    return None, {}


def _demo_read_validated_json(path, authority, *, kind="realsense"):
    path = Path(path)
    if not path.is_file():
        return {}, {
            "valid": False,
            "kind": kind,
            "source": str(path),
            "authoritative_identity": dict(authority or {}),
            "evidence_identity": {},
            "reasons": ["source_unavailable"],
        }
    value = read_json(path, None)
    if not isinstance(value, dict):
        return {}, {
            "valid": False,
            "kind": kind,
            "source": str(path),
            "authoritative_identity": dict(authority or {}),
            "evidence_identity": {},
            "reasons": ["source_not_valid_json_object"],
        }
    validation = _demo_identity_validation(value, authority, kind=kind)
    validation["source"] = str(path)
    return (value if validation["valid"] else {}), validation


def _demo_final_summary(
    realtime,
    locked_target_status,
    quality,
    live_quality,
    davis,
    approach_rows,
    *,
    authoritative_identity=None,
    evidence_validation=None,
):
    target_lock = realtime.get("target_lock") or locked_target_status or {}
    locked_target = target_lock.get("locked_target") or locked_target_status.get("locked_target") or {}
    target_status = str(realtime.get("target_status") or target_lock.get("target_status") or "").upper()
    bottle_detected = realtime.get("bottle_detected")
    if bottle_detected is None and target_status:
        bottle_detected = target_status in {"LOCKED", "TRACKING", "LOW_CONFIDENCE"}

    distances = []
    approach_distances = []
    band_drifts = []
    band_visibilities = []
    guidance_samples = []
    for row in approach_rows:
        if row.get("distance_measurement_valid") is True:
            distance = first_number(
                row.get("distance_to_bottle_mm"),
                row.get("bottle_distance_mm"),
                row.get("actionable_distance_mm"),
                row.get("contact_band_depth_mm"),
                row.get("depth_median"),
            )
            if distance is not None:
                distances.append(distance)
                approach_distances.append(distance)
        drift = first_number(row.get("grasp_band_drift_px"), row.get("band_drift_px"))
        if drift is not None:
            band_drifts.append(abs(drift))
        visibility = first_number(
            row.get("grasp_band_visibility_ratio"),
            row.get("grasp_band_visibility"),
            row.get("locked_band_visibility_ratio"),
        )
        if visibility is not None:
            band_visibilities.append(float(visibility))
        if isinstance(row.get("guidance_valid"), bool):
            guidance_samples.append(bool(row["guidance_valid"]))
    current_distance = None
    if realtime.get("distance_measurement_valid") is True:
        current_distance = first_number(
            realtime.get("distance_to_bottle_mm"),
            realtime.get("bottle_distance_mm"),
            realtime.get("actionable_distance_mm"),
            realtime.get("contact_band_depth_mm"),
        )
        if current_distance is not None:
            distances.append(current_distance)

    object_motion_valid = davis.get("object_motion_valid")
    if not isinstance(object_motion_valid, bool):
        object_motion_valid = None
    grasp_zone_motion_valid = davis.get("grasp_zone_motion_valid") is True
    presentation_motion_valid = object_motion_valid is True or grasp_zone_motion_valid
    motion_attribution = (
        davis.get("motion_attribution")
        or ("OBJECT_SPECIFIC" if object_motion_valid is True else None)
        or (
            "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED"
            if grasp_zone_motion_valid
            else "NOT_CURRENT_OR_NOT_ATTRIBUTABLE"
        )
    )
    motion_class = (
        davis.get("object_motion_class")
        if object_motion_valid is True
        else davis.get("grasp_zone_motion_class")
        if grasp_zone_motion_valid
        else None
    ) or davis.get("motion_class") or davis.get("davis_safety_class")
    slip_detected = None
    if presentation_motion_valid:
        slip_detected = "SLIP" in str(motion_class or "").upper()
    analytic_score = first_number(
        live_quality.get("analytic_quality_score"),
        quality.get("normalized_quality_score"),
        quality.get("epsilon_score"),
    )
    live_score = first_number(live_quality.get("live_quality_score"))
    davis_adjustment_used = bool(
        live_quality.get("davis_motion_used_for_live_quality") is True
        or live_quality.get("davis_closeup_used_for_live_quality") is True
    )
    davis_adjustment_points = first_number(live_quality.get("davis_adjustment_points"))
    if davis_adjustment_points is None:
        davis_penalty = first_number(live_quality.get("live_quality_davis_penalty"))
        davis_adjustment_points = -abs(davis_penalty) if davis_adjustment_used and davis_penalty is not None else 0.0
    if not davis_adjustment_used:
        davis_adjustment_points = 0.0
    davis_adjustment_reason = (
        live_quality.get("davis_adjustment_reason")
        or live_quality.get("davis_motion_live_quality_reason")
        or live_quality.get("davis_closeup_exclusion_reason")
        or davis.get("davis_motion_excluded_reason")
        or davis.get("object_motion_reason")
        or ("object_motion_valid" if davis_adjustment_used else "not_current_or_not_object_specific")
    )
    post_grasp_baseline = (
        realtime.get("post_grasp_baseline")
        if realtime.get("post_grasp_active") is True
        and isinstance(realtime.get("post_grasp_baseline"), dict)
        else {}
    )
    live_baseline = live_quality.get("baseline") if isinstance(live_quality.get("baseline"), dict) else {}
    frozen_baseline_current = bool(
        post_grasp_baseline.get("baseline_frozen") is True
        and post_grasp_baseline.get("quality_baseline_role") == "IMMUTABLE_LOCK_BASELINE"
        and post_grasp_baseline.get("quality_input_source") == "planned_contact"
        and post_grasp_baseline.get("quality_contact_source_forced") is True
        and post_grasp_baseline.get("quality_compute_id") is not None
        and first_number(post_grasp_baseline.get("quality_epsilon")) is not None
        and first_number(post_grasp_baseline.get("quality_score")) is not None
        and str(live_quality.get("analytic_quality_status") or "").upper() == "CURRENT"
        and str(live_baseline.get("writer_session_id")) == str(post_grasp_baseline.get("writer_session_id"))
        and str(live_baseline.get("lock_frame_id")) == str(post_grasp_baseline.get("lock_frame_id"))
        and str(live_baseline.get("quality_compute_id")) == str(post_grasp_baseline.get("quality_compute_id"))
    )
    analytic_quality_summary = {
        "score": (
            first_number(post_grasp_baseline.get("quality_score"))
            if frozen_baseline_current
            else analytic_score
        ),
        "label": (
            live_quality.get("analytic_quality_label") or post_grasp_baseline.get("quality_label")
            if frozen_baseline_current
            else live_quality.get("analytic_quality_label") or quality.get("quality_label_display") or quality.get("quality_label")
        ),
        "status": "CURRENT" if frozen_baseline_current else quality.get("quality_matrix_status"),
        "quality_compute_id": (
            post_grasp_baseline.get("quality_compute_id")
            if frozen_baseline_current
            else quality.get("quality_compute_id")
        ),
        "epsilon": (
            first_number(post_grasp_baseline.get("quality_epsilon"))
            if frozen_baseline_current
            else first_number(quality.get("epsilon"), quality.get("raw_epsilon"))
        ),
        "baseline_source": (
            "IMMUTABLE_POST_GRASP_SNAPSHOT"
            if frozen_baseline_current
            else "QUALITY_MATRIX_ARTIFACT"
        ),
    }
    live_quality_summary = {
        "score": live_score,
        "label": live_quality.get("live_quality_label"),
        "trend": live_quality.get("trend"),
        "primary_risk": live_quality.get("primary_risk"),
        "decision": live_quality.get("decision"),
        "davis_adjustment_points": davis_adjustment_points,
        "davis_adjustment_status": "INCLUDED" if davis_adjustment_used else "EXCLUDED",
        "davis_adjustment_reason": davis_adjustment_reason,
    }
    band_lock_confidence = first_number(
        realtime.get("grasp_band_lock_confidence"),
        target_lock.get("grasp_band_lock_confidence"),
        locked_target.get("lock_confidence"),
        (locked_target.get("locked_grasp_band_3d") or {}).get("locked_confidence"),
    )
    distance_start_mm = approach_distances[0] if approach_distances else current_distance
    distance_end_mm = current_distance if current_distance is not None else (
        approach_distances[-1] if approach_distances else None
    )
    guidance_valid_rate = (
        None
        if not guidance_samples
        else round(sum(guidance_samples) / len(guidance_samples), 4)
    )
    limitations = [
        "Hand-operated prototype export; it does not claim autonomous control or physical grasp success.",
        "Guidance remains in the camera frame unless a calibrated end-effector transform is explicitly available.",
        "Fixed-zone DAVIS motion is an uncalibrated post-grasp diagnostic and is not claimed as calibrated object-specific motion.",
    ]
    if not approach_rows:
        limitations.append("No approach timeline source was available at export time.")
    if not distances:
        limitations.append(
            "No current valid depth measurements were available for an approach distance range."
        )
    if object_motion_valid is not True and not grasp_zone_motion_valid:
        limitations.append(
            f"DAVIS activity was not object-specific: {davis.get('object_motion_reason') or davis.get('davis_closeup_exclusion_reason') or 'no valid object-motion reason reported'}."
        )
    authority_complete = bool(
        (authoritative_identity or {}).get("writer_session_id") is not None
        and (authoritative_identity or {}).get("lock_frame_id") is not None
    )
    if not authority_complete:
        limitations.append(
            "Current lock identity is incomplete; lock-scoped evidence is excluded from this summary."
        )
    for evidence_name, validation in (evidence_validation or {}).items():
        if evidence_name == "authoritative_realtime" or validation.get("valid"):
            continue
        limitations.append(
            f"Excluded {evidence_name} from current evidence: "
            + "; ".join(str(reason) for reason in validation.get("reasons") or ["identity validation failed"])
            + "."
        )
    return {
        "schema_version": "hand_demo_final_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bottle_detected": bottle_detected,
        "band_locked": authority_complete and target_status in {"LOCKED", "TRACKING"},
        "locked_band": {
            "target_status": target_status or None,
            "target_id": realtime.get("locked_bottle_target_id") or target_lock.get("bottle_target_id") or locked_target.get("bottle_target_id"),
            "lock_frame_id": target_lock.get("lock_frame_id") or locked_target.get("lock_frame_id"),
            "body_fraction": first_number(
                realtime.get("locked_body_fraction"),
                target_lock.get("locked_body_fraction"),
                locked_target.get("locked_body_fraction"),
            ),
            "identity_valid": authority_complete,
        },
        "band_lock_confidence": band_lock_confidence,
        "analytic_quality": analytic_quality_summary,
        "initial_analytic_quality": dict(analytic_quality_summary),
        "approach_distance_range_mm": None if not distances else {
            "min": round(min(distances), 3),
            "max": round(max(distances), 3),
            "sample_count": len(distances),
        },
        "band_drift_max_px": None if not band_drifts else round(max(band_drifts), 3),
        "max_band_drift_px": None if not band_drifts else round(max(band_drifts), 3),
        "min_band_visibility_ratio": None if not band_visibilities else round(min(band_visibilities), 4),
        "distance_start_mm": None if distance_start_mm is None else round(distance_start_mm, 3),
        "distance_end_mm": None if distance_end_mm is None else round(distance_end_mm, 3),
        "guidance_valid_rate": guidance_valid_rate,
        "davis_event_health": {
            "valid": davis.get("event_health_valid"),
            "event_count": (
                davis.get("event_count")
                if davis.get("event_count") is not None
                else (davis.get("current_metrics") or {}).get("event_count")
            ),
            "event_rate": first_number(davis.get("event_rate"), (davis.get("current_metrics") or {}).get("event_rate")),
        },
        "object_motion_valid": object_motion_valid,
        "davis_object_motion_valid": object_motion_valid,
        "grasp_zone_motion_valid": grasp_zone_motion_valid,
        "motion_attribution": motion_attribution,
        "davis_motion_class": motion_class,
        "slip_like_motion": {
            "detected": slip_detected,
            "motion_class": motion_class,
            "status": (
                "VALID_OBJECT_SPECIFIC"
                if object_motion_valid is True
                else "VALID_FIXED_GRASP_ZONE_UNCALIBRATED"
                if grasp_zone_motion_valid
                else "NOT_ATTRIBUTABLE"
            ),
            "motion_attribution": motion_attribution,
            "downward_slip_px": first_number(davis.get("downward_slip_px")) if presentation_motion_valid else None,
        },
        "rotation": {
            "valid": presentation_motion_valid and first_number(davis.get("rotation_delta_deg")) is not None,
            "delta_deg": first_number(davis.get("rotation_delta_deg")) if presentation_motion_valid else None,
            "motion_attribution": motion_attribution,
            "object_specific": object_motion_valid is True,
        },
        "roll": {
            "valid": davis.get("roll_valid") is True,
            "delta_deg": first_number(davis.get("roll_delta_deg")) if davis.get("roll_valid") is True else None,
            "status": davis.get("roll_status") or "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
            "exclusion_reason": davis.get("roll_exclusion_reason") or "No calibrated 3D object orientation source is available.",
        },
        "vibration": {
            "valid": presentation_motion_valid and first_number(davis.get("vibration_amplitude_px_rms")) is not None,
            "rms_px": first_number(davis.get("vibration_amplitude_px_rms")) if presentation_motion_valid else None,
            "motion_attribution": motion_attribution,
            "object_specific": object_motion_valid is True,
        },
        "disturbance": {
            "type": davis.get("disturbance_type") or "none",
            "excluded_from_object_motion": object_motion_valid is not True,
            "exclusion_reason": davis.get("davis_closeup_exclusion_reason") or davis.get("object_motion_reason"),
        },
        "davis_live_quality_adjustment": {
            "status": "INCLUDED" if davis_adjustment_used else "EXCLUDED",
            "points": davis_adjustment_points,
            "reason": davis_adjustment_reason,
            "epsilon_recomputed": False,
        },
        "final_live_quality": live_quality_summary,
        "live_quality_final": dict(live_quality_summary),
        "final_decision": live_quality.get("decision") or davis.get("decision") or davis.get("post_grasp_decision") or realtime.get("decision"),
        "authoritative_lock_identity": dict(authoritative_identity or {}),
        "evidence_validation": dict(evidence_validation or {}),
        "limitations": limitations,
    }


def record_demo_snapshot():
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{time.time_ns() % 1000000:06d}"
    run_dir = OUTPUT_DIR / "demo_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    realtime_path = OUTPUT_DIR / "latest_realtime_status.json"
    realtime = read_json(realtime_path, {})
    realtime = realtime if isinstance(realtime, dict) else {}
    authority = _demo_current_lock_identity(realtime)
    post_grasp_action = _demo_current_post_grasp_action()
    evidence_validation = {
        "authoritative_realtime": _demo_authority_validation(realtime, authority)
    }
    names = [
        "latest_overlay_live.jpg",
        "latest_realtime_status.json",
        "realtime_grasp_hulls.json",
        "latest_grasp_hulls_overlay.jpg",
        "realtime_epsilon_result.json",
        "realtime_wrench_matrix_status.json",
        "davis_motion_report.json",
        "grasp_quality_bottle_mask_weak_direction.png",
        "wrench_space_projection.png",
        "wrench_space_3d_hull.png",
        "wrench_space_fx_fy.png",
        "wrench_space_fx_tau.png",
        "wrench_space_fy_tau.png",
    ]
    copied = []
    missing = []
    excluded = []
    structured_kinds = {
        "realtime_grasp_hulls.json": "realsense",
        "realtime_epsilon_result.json": "realsense",
        "realtime_wrench_matrix_status.json": "realsense",
        "davis_motion_report.json": "davis",
    }
    for name in names:
        src = OUTPUT_DIR / name
        if not src.exists():
            missing.append(name)
            continue
        validation = None
        if name == "latest_realtime_status.json":
            validation = evidence_validation["authoritative_realtime"]
        elif name in structured_kinds:
            _value, validation = _demo_read_validated_json(
                src,
                authority,
                kind=structured_kinds[name],
            )
        else:
            validation = _demo_image_validation(src, authority, realtime)
        evidence_validation[f"bundle_artifact:{name}"] = validation
        if not validation["valid"] and name != "latest_realtime_status.json":
            excluded.append({"name": name, "reasons": validation["reasons"]})
            continue
        destination = run_dir / name
        shutil.copy2(src, destination)
        if src.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            copy_reason = _demo_copied_image_version_reason(validation, destination)
            if copy_reason:
                destination.unlink(missing_ok=True)
                validation["valid"] = False
                validation["reasons"].append(copy_reason)
                excluded.append({"name": name, "reasons": validation["reasons"]})
                continue
        copied.append(name)
    locked_snapshot_source = None
    locked_snapshot_validation = {
        "valid": False,
        "kind": "image",
        "authoritative_identity": dict(authority),
        "evidence_identity": {},
        "reasons": ["source_unavailable"],
    }
    locked_snapshot_rejections = []
    for path in (OUTPUT_DIR / "locked_band_snapshot.jpg",):
        if not path.is_file():
            continue
        validation = _demo_image_validation(path, authority, realtime)
        if validation["valid"]:
            locked_snapshot_source = path
            locked_snapshot_validation = validation
            break
        locked_snapshot_rejections.append(
            {"source": str(path), "reasons": validation["reasons"]}
        )
    if locked_snapshot_source is None and locked_snapshot_rejections:
        locked_snapshot_validation = {
            **locked_snapshot_validation,
            "reasons": [
                f"{item['source']}: {'; '.join(item['reasons'])}"
                for item in locked_snapshot_rejections
            ],
        }
    evidence_validation["locked_band_snapshot"] = locked_snapshot_validation
    if locked_snapshot_source is not None:
        destination = run_dir / "locked_band_snapshot.jpg"
        shutil.copy2(locked_snapshot_source, destination)
        copy_reason = _demo_copied_image_version_reason(
            locked_snapshot_validation,
            destination,
        )
        if copy_reason:
            destination.unlink(missing_ok=True)
            locked_snapshot_validation["valid"] = False
            locked_snapshot_validation["reasons"].append(copy_reason)
            locked_snapshot_source = None

    projected_snapshot_source = None
    projected_snapshot_validation = {
        "valid": False,
        "kind": "image",
        "authoritative_identity": dict(authority),
        "evidence_identity": {},
        "reasons": ["current_projected_band_unavailable"],
    }
    realtime_lock = realtime.get("target_lock") or {}
    projected_current = bool(
        (realtime.get("grasp_band_projected") is True or realtime_lock.get("grasp_band_projected") is True)
        and not (realtime.get("grasp_band_stale") is True or realtime_lock.get("grasp_band_stale") is True)
        and not (
            realtime.get("grasp_band_relock_required") is True
            or realtime_lock.get("grasp_band_relock_required") is True
        )
    )
    if projected_current:
        for path in (
            OUTPUT_DIR / "latest_overlay_live.jpg",
            OUTPUT_DIR / "latest_grasp_hulls_overlay.jpg",
        ):
            if not path.is_file():
                continue
            validation = _demo_image_validation(path, authority, realtime)
            if validation["valid"]:
                projected_snapshot_source = path
                projected_snapshot_validation = validation
                break
    evidence_validation["projected_band_snapshot"] = projected_snapshot_validation
    if projected_snapshot_source is not None:
        destination = run_dir / "projected_band_snapshot.jpg"
        shutil.copy2(projected_snapshot_source, destination)
        copy_reason = _demo_copied_image_version_reason(
            projected_snapshot_validation,
            destination,
        )
        if copy_reason:
            destination.unlink(missing_ok=True)
            projected_snapshot_validation["valid"] = False
            projected_snapshot_validation["reasons"].append(copy_reason)
            projected_snapshot_source = None

    approach_meta, approach_lines = _copy_bounded_timeline(
        (
            OUTPUT_DIR / "approach_timeline.jsonl",
            OUTPUT_DIR / "target_lock_trace.jsonl",
            OUTPUT_DIR / "patch_coordinate_trace.jsonl",
        ),
        run_dir / "approach_timeline.jsonl",
        authority=authority,
        identity_kind="realsense",
    )
    davis_post_timeline_path = (
        OUTPUT_DIR / "davis" / "post_grasp_monitoring" / "timeline.jsonl"
    )
    davis_timeline_meta, _davis_lines = _copy_bounded_timeline(
        (davis_post_timeline_path, OUTPUT_DIR / "davis" / "motion_timeline.jsonl"),
        run_dir / "davis_event_timeline.jsonl",
        authority=authority,
        identity_kind="davis",
        post_grasp_action=post_grasp_action,
        require_post_grasp_action=bool(
            davis_post_timeline_path.is_file()
            or post_grasp_action.get("post_grasp_monitoring_action_id") is not None
        ),
    )
    evidence_validation["approach_timeline"] = approach_meta["identity_validation"]
    evidence_validation["davis_event_timeline"] = davis_timeline_meta["identity_validation"]
    live_quality_timeline_meta, _live_quality_lines = _copy_bounded_timeline(
        (OUTPUT_DIR / "live_quality" / "timeline.jsonl",),
        run_dir / "live_quality_timeline.jsonl",
        authority=authority,
        identity_kind="realsense",
    )
    evidence_validation["live_quality_timeline"] = live_quality_timeline_meta[
        "identity_validation"
    ]

    post_grasp_source = None
    post_grasp_report = {}
    post_grasp_rejections = []
    for candidate in (
        OUTPUT_DIR / "davis" / "post_grasp_monitoring" / "latest_report.json",
        OUTPUT_DIR / "event_postgrasp_report.json",
        OUTPUT_DIR / "davis" / "latest_motion_report.json",
        OUTPUT_DIR / "davis_motion_report.json",
    ):
        value, validation = _demo_read_validated_json(candidate, authority, kind="davis")
        if validation["valid"]:
            action_validation = _demo_post_grasp_action_validation(
                value,
                post_grasp_action,
            )
            validation["post_grasp_action_validation"] = action_validation
            if not action_validation["valid"]:
                validation["valid"] = False
                validation["reasons"] = action_validation["reasons"]
        if validation["valid"]:
            post_grasp_source = candidate
            post_grasp_report = value
            evidence_validation["post_grasp_report"] = validation
            break
        if candidate.is_file():
            post_grasp_rejections.append(validation)
    if post_grasp_source is None:
        post_reasons = [
            f"{validation['source']}: {'; '.join(validation['reasons'])}"
            for validation in post_grasp_rejections
        ] or ["source_unavailable"]
        evidence_validation["post_grasp_report"] = {
            "valid": False,
            "kind": "davis",
            "authoritative_identity": dict(authority),
            "evidence_identity": {},
            "reasons": post_reasons,
        }
        post_grasp_report = {
            "status": "UNAVAILABLE",
            "reason": (
                "No post-grasp report matching the current lock identity was available at export time: "
                + "; ".join(post_reasons)
            ),
        }
    write_json_atomic(run_dir / "post_grasp_report.json", post_grasp_report)

    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir()
    screenshot_sources = (
        OUTPUT_DIR / "hand_demo_stage_screenshots",
        OUTPUT_DIR / "hand_mounted_demo" / "screenshots",
        OUTPUT_DIR / "demo_stage_screenshots",
    )
    stage_screenshots = []
    missing_stage_screenshots = []
    stage_screenshot_validation = {}
    for name in DEMO_STAGE_SCREENSHOTS:
        source = None
        rejection_reasons = []
        accepted_validation = None
        for directory in screenshot_sources:
            candidate = directory / name
            if not candidate.is_file():
                continue
            validation = _demo_image_validation(
                candidate,
                authority,
                realtime,
                require_sidecar=True,
            )
            if validation["valid"]:
                source = candidate
                accepted_validation = validation
                break
            rejection_reasons.extend(validation["reasons"])
        if source is None:
            missing_stage_screenshots.append(name)
            stage_screenshot_validation[name] = {
                "valid": False,
                "reasons": rejection_reasons or ["source_unavailable"],
            }
            continue
        destination = screenshots_dir / name
        shutil.copy2(source, destination)
        copy_reason = _demo_copied_image_version_reason(
            accepted_validation,
            destination,
        )
        if copy_reason:
            destination.unlink(missing_ok=True)
            accepted_validation["valid"] = False
            accepted_validation["reasons"].append(copy_reason)
            missing_stage_screenshots.append(name)
            stage_screenshot_validation[name] = accepted_validation
            continue
        stage_screenshot_validation[name] = accepted_validation
        stage_screenshots.append({
            "name": name,
            "source": str(source),
            "provenance_source": accepted_validation["provenance_source"],
            "captured_as": f"screenshots/{name}",
        })

    export_screenshots = []
    missing_export_screenshots = []
    export_screenshot_validation = {}
    export_capture_dir = OUTPUT_DIR / "hand_demo_stage_screenshots"
    for name in DEMO_EXPORT_SCREENSHOTS:
        source = export_capture_dir / name
        validation = _demo_image_validation(
            source,
            authority,
            realtime,
            require_sidecar=True,
        )
        export_screenshot_validation[name] = validation
        if not validation["valid"]:
            missing_export_screenshots.append(name)
            continue
        destination = run_dir / name
        shutil.copy2(source, destination)
        copy_reason = _demo_copied_image_version_reason(validation, destination)
        if copy_reason:
            destination.unlink(missing_ok=True)
            validation["valid"] = False
            validation["reasons"].append(copy_reason)
            missing_export_screenshots.append(name)
            continue
        export_screenshots.append({
            "name": name,
            "source": str(source),
            "provenance_source": validation["provenance_source"],
            "captured_as": name,
        })

    locked_target_status, evidence_validation["locked_target"] = _demo_read_validated_json(
        OUTPUT_DIR / "locked_target.json",
        authority,
    )
    quality, evidence_validation["analytic_quality"] = _demo_read_validated_json(
        OUTPUT_DIR / "realtime_wrench_matrix_status.json",
        authority,
        kind="quality",
    )
    live_quality, evidence_validation["live_quality"] = _demo_read_validated_json(
        OUTPUT_DIR / "live_quality" / "latest_status.json",
        authority,
        kind="quality",
    )
    davis, evidence_validation["davis_motion_report"] = _demo_read_validated_json(
        OUTPUT_DIR / "davis" / "latest_motion_report.json",
        authority,
        kind="davis",
    )
    if davis and (
        davis.get("post_grasp_monitoring_action_id") is not None
        or post_grasp_action.get("post_grasp_monitoring_action_id") is not None
    ):
        action_validation = _demo_post_grasp_action_validation(
            davis,
            post_grasp_action,
        )
        evidence_validation["davis_motion_report"][
            "post_grasp_action_validation"
        ] = action_validation
        if not action_validation["valid"]:
            evidence_validation["davis_motion_report"]["valid"] = False
            evidence_validation["davis_motion_report"]["reasons"] = action_validation[
                "reasons"
            ]
            davis = {}
    if not davis and post_grasp_source is not None:
        davis = post_grasp_report
        evidence_validation["davis_motion_report"] = dict(
            evidence_validation["post_grasp_report"]
        )
    final_summary = _demo_final_summary(
        realtime,
        locked_target_status,
        quality,
        live_quality,
        davis,
        _jsonl_rows(approach_lines),
        authoritative_identity=authority,
        evidence_validation={
            key: evidence_validation[key]
            for key in (
                "authoritative_realtime",
                "locked_target",
                "analytic_quality",
                "live_quality",
                "davis_motion_report",
                "approach_timeline",
                "davis_event_timeline",
                "live_quality_timeline",
                "post_grasp_report",
                "locked_band_snapshot",
                "projected_band_snapshot",
            )
        },
    )
    if locked_snapshot_source is None:
        final_summary["limitations"].append("No locked-band image source was available at export time.")
    if projected_snapshot_source is None:
        final_summary["limitations"].append(
            "No current projected-band image was available at export time."
        )
    if missing_stage_screenshots:
        final_summary["limitations"].append(
            "Stage screenshots missing at export: " + ", ".join(missing_stage_screenshots) + "."
        )
    if missing_export_screenshots:
        final_summary["limitations"].append(
            "Browser demo captures missing at export: "
            + ", ".join(missing_export_screenshots)
            + "."
        )
    write_json_atomic(run_dir / "final_summary.json", final_summary)
    write_json_atomic(run_dir / "demo_summary.json", final_summary)

    manifest = {
        "schema_version": "hand_demo_bundle_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "copied": copied,
        "missing": missing,
        "excluded": excluded,
        "created": [
            "approach_timeline.jsonl",
            "davis_event_timeline.jsonl",
            "live_quality_timeline.jsonl",
            "post_grasp_report.json",
            "final_summary.json",
            "demo_summary.json",
            "demo_manifest.json",
            "manifest.json",
        ]
        + (["locked_band_snapshot.jpg"] if locked_snapshot_source is not None else [])
        + (["projected_band_snapshot.jpg"] if projected_snapshot_source is not None else [])
        + [item["name"] for item in export_screenshots],
        "writer_session_id": authority.get("writer_session_id"),
        "lock_frame_id": authority.get("lock_frame_id"),
        "attempt_id": authority.get("attempt_id"),
        "bottle_target_id": authority.get("bottle_target_id"),
        "authoritative_lock_identity": authority,
        "authoritative_post_grasp_action": post_grasp_action,
        "evidence_validation": evidence_validation,
        "locked_band_snapshot": {
            "available": locked_snapshot_source is not None,
            "source": None if locked_snapshot_source is None else str(locked_snapshot_source),
            "captured_as": "locked_band_snapshot.jpg" if locked_snapshot_source is not None else None,
            "validation": locked_snapshot_validation,
        },
        "projected_band_snapshot": {
            "available": projected_snapshot_source is not None,
            "source": None if projected_snapshot_source is None else str(projected_snapshot_source),
            "captured_as": "projected_band_snapshot.jpg" if projected_snapshot_source is not None else None,
            "validation": projected_snapshot_validation,
        },
        "approach_timeline": approach_meta,
        "davis_event_timeline": davis_timeline_meta,
        "live_quality_timeline": live_quality_timeline_meta,
        "post_grasp_report": {
            "available": post_grasp_source is not None,
            "source": None if post_grasp_source is None else str(post_grasp_source),
            "captured_as": "post_grasp_report.json",
        },
        "stage_screenshots": stage_screenshots,
        "stage_screenshot_validation": stage_screenshot_validation,
        "missing_stage_screenshots": missing_stage_screenshots,
        "export_screenshots": export_screenshots,
        "export_screenshot_validation": export_screenshot_validation,
        "missing_export_screenshots": missing_export_screenshots,
        "image_provenance_contract": {
            "sidecar_locations": [
                "<image filename>.json",
                "<image stem>.json",
                "provenance.json screenshots/artifacts entry",
            ],
            "required_fields": [
                "artifact_name",
                "artifact_mtime_ns",
                "writer_session_id",
                "lock_frame_id",
            ],
            "optional_identity_fields": ["attempt_id", "bottle_target_id"],
        },
        "evidence_policy": (
            "latest_realtime_status defines the current lock identity; structured evidence and bounded "
            "timeline rows must match writer_session_id and lock_frame_id (plus attempt/target when "
            "both are available); images require explicit provenance or the current non-stale overlay "
            "link; mismatches are excluded and images are never fabricated."
        ),
    }
    write_json_atomic(run_dir / "demo_manifest.json", manifest)
    write_json_atomic(run_dir / "manifest.json", manifest)
    return manifest


@app.post("/api/record-snapshot")
def api_record_snapshot():
    manifest = record_demo_snapshot()
    return jsonify({"ok": True, "message": f"Recorded demo snapshot: {manifest['run_dir']}", "manifest": manifest})


@app.post("/api/hand-demo/stage-screenshot/<stage>")
def api_hand_demo_stage_screenshot(stage):
    result = capture_hand_demo_stage_screenshot(stage, request.get_data(cache=False))
    return jsonify(result), 200 if result.get("ok") else 409


def mjpeg_generator(name):
    buffer = stream_buffers[name]
    last_id = 0
    while True:
        if name in {"rgb", "depth"} and not debug_stream_enabled(name):
            return
        frame_id, jpeg, meta = buffer.wait_latest(last_id=last_id, timeout=1.0)
        if jpeg is None or frame_id == last_id:
            continue
        last_id = frame_id
        t0 = time.perf_counter()
        payload = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"X-Frame-Id: {frame_id}\r\n".encode("ascii")
            + b"\r\n"
            + jpeg
            + b"\r\n"
        )
        profile_set(f"{name}_browser_send_prepare", (time.perf_counter() - t0) * 1000.0)
        buffer.mark_sent(frame_id, meta)
        yield payload


@app.get("/stream/<name>")
def stream(name):
    if name not in stream_buffers:
        return "unknown stream", 404
    if name in {"rgb", "depth"} and not debug_stream_enabled(name):
        return "debug stream disabled", 409
    return Response(
        mjpeg_generator(name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.post("/api/start/<name>")
def api_start(name):
    if name == "davis_motion":
        set_desired_process("davis_motion", True)
        ok, msg = start_davis_motion()
    elif name in COMMANDS:
        set_desired_process(name, True)
        ok, msg = start_process(name)
    else:
        return jsonify({"ok": False, "message": f"Unknown demo: {name}"}), 404
    return jsonify({"ok": ok, "message": msg})


@app.post("/api/stop/<name>")
def api_stop(name):
    if name == "davis_motion":
        set_desired_process("davis_motion", False)
        ok, msg = stop_davis_motion()
    elif name in COMMANDS:
        set_desired_process(name, False)
        ok, msg = stop_process(name)
    else:
        return jsonify({"ok": False, "message": f"Unknown demo: {name}"}), 404
    return jsonify({"ok": ok, "message": msg})


@app.get("/api/davis/status")
def api_davis_status():
    return jsonify(davis_status_payload())


@app.get("/api/davis/features")
def api_davis_features():
    report = read_davis_report()
    return jsonify({
        "ok": True,
        "state": davis_web_state(report, running=bool(davis_thread and davis_thread.is_alive())),
        "current_metrics": report.get("current_metrics") or {},
        "event_window_metrics": report.get("event_window_metrics") or {},
        "impact_window_metrics": report.get("impact_window_metrics") or {},
        "settle_window_metrics": report.get("settle_window_metrics") or {},
        "stability_window_metrics": report.get("stability_window_metrics") or {},
        "closeup_monitoring": {
            key: report.get(key)
            for key in (
                "davis_closeup_monitoring_state", "davis_closeup_window",
                "closeup_action_elapsed_sec", "closeup_action_window_complete",
                "davis_closeup_motion_class", "davis_closeup_stability_decision",
                "davis_closeup_used_for_live_quality", "davis_closeup_exclusion_reason",
                "davis_closeup_confidence",
            )
        },
        "boundary_event_field": report.get("boundary_event_field") or {},
        "scores": {
            "impact_score": report.get("impact_score"),
            "incipient_slip_score": report.get("incipient_slip_score"),
            "gross_slip_score": report.get("gross_slip_score"),
            "rotation_risk_score": report.get("rotation_risk_score"),
            "vibration_risk_score": report.get("vibration_risk_score") or report.get("vibration_score"),
            "stability_score": report.get("stability_score") or report.get("post_grasp_stability_score"),
            "confidence_score": report.get("confidence_score"),
        },
        "motion_class": report.get("motion_class"),
        "decision": report.get("decision") or report.get("post_grasp_decision"),
        "owner": {
            "davis_writer_pid": report.get("davis_writer_pid"),
            "davis_writer_session_id": report.get("davis_writer_session_id"),
            "davis_owner_status": report.get("davis_owner_status"),
            "duplicate_davis_owner_detected": bool(report.get("duplicate_davis_owner_detected")),
        },
    })


@app.get("/api/davis/timeline")
def api_davis_timeline():
    report = read_davis_report()
    return jsonify({
        "ok": True,
        "motion_class_history": report.get("motion_class_history") or [],
        "decision_history": report.get("decision_history") or [],
        "history": report.get("history") or [],
        "event_rate": ((report.get("current_metrics") or {}).get("event_rate") or report.get("event_rate") or 0.0),
    })


@app.post("/api/davis/start")
def api_davis_start():
    set_desired_process("davis_motion", True)
    ok, msg = start_davis_motion()
    return jsonify({**davis_status_payload(), "ok": ok, "message": msg})


@app.post("/api/davis/stop")
def api_davis_stop():
    set_desired_process("davis_motion", False)
    ok, msg = stop_davis_motion()
    return jsonify({**davis_status_payload(), "ok": ok, "message": msg})


@app.post("/api/davis/restart")
def api_davis_restart():
    set_desired_process("davis_motion", True)
    stop_davis_motion()
    ok, msg = start_davis_motion()
    return jsonify({**davis_status_payload(), "ok": ok, "message": f"Restarted DAVIS motion monitor. {msg}"})


@app.post("/api/davis/calibrate")
def api_davis_calibrate():
    with davis_lock:
        analyzer = davis_analyzer
        if analyzer is not None:
            analyzer.baseline_metrics = None
            analyzer.baseline_rejected_count = 0
            analyzer.start_time = time.monotonic()
            analyzer.history = []
            analyzer.motion_class_history = []
            analyzer.decision_history = []
            analyzer.report["baseline_calibrated"] = False
            analyzer.report["calibration_command"] = "baseline_reset_from_web"
            ok = True
            msg = "DAVIS event baseline reset; hold the scene still for recalibration."
        else:
            ok = False
            msg = "DAVIS monitor is not running."
    return jsonify({**davis_status_payload(), "ok": ok, "message": msg})


@app.post("/api/davis/record/start")
def api_davis_record_start():
    global davis_recording
    davis_recording = True
    write_json_atomic(OUTPUT_DIR / "davis_recording_status.json", {"recording": True, "started_at": time.time(), "source": "web_dashboard"})
    return jsonify({"ok": True, "message": "DAVIS trial recording marker started.", "recording": True})


@app.post("/api/davis/record/stop")
def api_davis_record_stop():
    global davis_recording
    davis_recording = False
    write_json_atomic(OUTPUT_DIR / "davis_recording_status.json", {"recording": False, "stopped_at": time.time(), "source": "web_dashboard"})
    return jsonify({"ok": True, "message": "DAVIS trial recording marker stopped.", "recording": False})


@app.post("/api/davis/snapshot")
def api_davis_snapshot():
    """Save the current dashboard DAVIS report/images without opening hardware."""
    snapshot_ts = time.time()
    snapshot_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(snapshot_ts))
    snapshot_dir = OUTPUT_DIR / "davis_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    copy_errors = []

    report_path = DAVIS_LATEST_MOTION_REPORT_PATH if DAVIS_LATEST_MOTION_REPORT_PATH.exists() else OUTPUT_DIR / "davis_motion_report.json"
    if report_path.exists():
        dst = snapshot_dir / f"{snapshot_id}_davis_motion_report.json"
        try:
            shutil.copy2(report_path, dst)
            saved["report"] = str(dst.relative_to(ROOT))
        except OSError as exc:
            copy_errors.append(f"report:{type(exc).__name__}")

    heatmap_path = DAVIS_LATEST_EVENT_HEATMAP_PATH if DAVIS_LATEST_EVENT_HEATMAP_PATH.exists() else OUTPUT_DIR / "davis_event_heatmap.png"
    if heatmap_path.exists():
        dst = snapshot_dir / f"{snapshot_id}_davis_event_heatmap.png"
        try:
            shutil.copy2(heatmap_path, dst)
            saved["event_heatmap"] = str(dst.relative_to(ROOT))
        except OSError as exc:
            # The live heatmap writer may atomically replace the file between
            # exists() and copy2(). A snapshot request should still return its
            # manifest instead of becoming a dashboard 500.
            copy_errors.append(f"event_heatmap:{type(exc).__name__}")

    aps_jpeg = None
    aps_meta = {}
    with stream_buffers["davis_aps"].cond:
        aps_jpeg = stream_buffers["davis_aps"].jpeg
        aps_meta = dict(stream_buffers["davis_aps"].meta)
    if aps_jpeg:
        dst = snapshot_dir / f"{snapshot_id}_davis_aps.jpg"
        dst.write_bytes(aps_jpeg)
        saved["aps_frame"] = str(dst.relative_to(ROOT))

    manifest = {
        "ok": True,
        "snapshot_id": snapshot_id,
        "timestamp": snapshot_ts,
        "source": "web_dashboard",
        "saved": saved,
        "copy_errors": copy_errors,
        "davis_status": davis_status_payload(),
        "aps_meta": aps_meta,
    }
    manifest_path = snapshot_dir / f"{snapshot_id}_manifest.json"
    write_json_atomic(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path.relative_to(ROOT))
    return jsonify({**manifest, "message": "DAVIS snapshot saved." if saved else "DAVIS snapshot manifest saved; no current images were available."})


@app.get("/api/davis/export")
def api_davis_export():
    report_path = DAVIS_LATEST_MOTION_REPORT_PATH if DAVIS_LATEST_MOTION_REPORT_PATH.exists() else OUTPUT_DIR / "davis_motion_report.json"
    if not report_path.exists():
        return jsonify({"ok": False, "message": "DAVIS report is unavailable."}), 404
    return send_file(report_path, as_attachment=True, download_name="davis_motion_report.json")


@app.post("/api/davis/history/clear")
def api_davis_history_clear():
    with davis_lock:
        analyzer = davis_analyzer
        if analyzer is not None:
            analyzer.history = []
            analyzer.motion_class_history = []
            analyzer.decision_history = []
            analyzer.report["history_clear_command"] = "web_dashboard"
            ok = True
            msg = "DAVIS motion history cleared."
        else:
            ok = False
            msg = "DAVIS monitor is not running."
    return jsonify({"ok": ok, "message": msg})


@app.post("/api/davis/aps-diagnostics")
def api_davis_aps_diagnostics():
    from grasp_project.davis.aps_stream import save_aps_visibility_diagnostics

    with davis_lock:
        analyzer = davis_analyzer
        record = getattr(analyzer, "latest_aps_record", None) if analyzer is not None else None
        frame = getattr(record, "image", None) if record is not None else None
        config = dict(getattr(analyzer, "aps_config", {}) or {}) if analyzer is not None else {}
        metadata = {
            "color_sensor": bool(getattr(record, "aps_color_verified", False)),
            "channel_order": getattr(record, "aps_channel_order", None),
            "channel_order_source": getattr(record, "aps_channel_order_source", None),
            "raw_format": getattr(record, "raw_format", None),
        } if record is not None else {}
    if frame is None:
        return jsonify({"ok": False, "message": "No APS frame available for diagnostics."}), 409
    report = save_aps_visibility_diagnostics(
        frame,
        output_dir=DAVIS_OUTPUT_DIR,
        config=config,
        metadata=metadata,
    )
    return jsonify({"ok": True, "message": "APS visibility diagnostics saved to output/davis/.", "report": _json_compatible(report)})


@app.post("/api/davis/aps-display-mode")
def api_davis_aps_display_mode():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    with davis_lock:
        analyzer = davis_analyzer
        if analyzer is None:
            return jsonify({"ok": False, "message": "DAVIS motion backend is not running."}), 409
        if hasattr(analyzer, "set_aps_display_mode") and analyzer.set_aps_display_mode(mode):
            return jsonify({"ok": True, "message": f"APS display mode set to {mode}. Display only; raw APS statistics are unchanged.", "aps_display_mode": mode})
    return jsonify({"ok": False, "message": f"Unknown APS display mode: {mode}"}), 400


@app.get("/stream/davis/aps")
def stream_davis_aps():
    return Response(
        mjpeg_generator("davis_aps"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/image/davis/event_heatmap")
def image_davis_event_heatmap():
    path = DAVIS_LATEST_EVENT_HEATMAP_PATH if DAVIS_LATEST_EVENT_HEATMAP_PATH.exists() else OUTPUT_DIR / "davis_event_heatmap.png"
    if path.exists():
        return send_file(path)
    return "missing", 404


@app.get("/image/<path:name>")
def image(name):
    path = (OUTPUT_DIR / name).resolve()
    if OUTPUT_DIR.resolve() not in path.parents and path != OUTPUT_DIR.resolve():
        return "bad path", 400
    if not path.exists():
        return "missing", 404
    return send_file(path)


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" href="data:," />
  <title>RGB-D + DAVIS Grasp Quality Demo</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      --ink:#172033;
      --muted:#667085;
      --line:#d7dee8;
      --panel:#ffffff;
      --page:#eef2f5;
      --good:#138a43;
      --warn:#c56a18;
      --bad:#c33131;
      --blue:#2563eb;
    }
    * { box-sizing:border-box; }
    html { scroll-padding-top:128px; }
    body { margin:0; background:var(--page); color:var(--ink); overflow-x:hidden; }
    header {
      background:#152033;
      color:white;
      padding:18px 28px;
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:18px;
      border-bottom:4px solid #22a06b;
    }
    h1 { font-size:24px; margin:0 0 5px; letter-spacing:0; }
    h2 { margin:0; font-size:18px; letter-spacing:0; }
    h3 { margin:0; font-size:14px; color:#344054; letter-spacing:0; }
    main { padding:18px 22px 28px; display:grid; gap:18px; max-width:1920px; margin:0 auto; }
    .subtitle { color:#d5e3f0; font-size:14px; max-width:760px; line-height:1.35; }
    .clock { color:#c8d3df; font-size:13px; white-space:nowrap; padding-top:3px; }
    .header-side { display:grid; justify-items:end; gap:8px; }
    .mode-chips { display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; }
    .chip {
      display:inline-flex;
      align-items:center;
      gap:6px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.25);
      background:rgba(255,255,255,.08);
      color:#e6edf7;
      padding:5px 9px;
      font-size:12px;
      font-weight:800;
      white-space:nowrap;
    }
    .chip.ok { background:rgba(19,138,67,.18); border-color:rgba(72,187,120,.5); color:#d9fbe6; }
    .chip.warn { background:rgba(197,106,24,.18); border-color:rgba(251,191,36,.45); color:#fff2cf; }
    .chip.bad { background:rgba(195,49,49,.22); border-color:rgba(248,113,113,.5); color:#ffe0e0; }
    .chip.info { background:rgba(37,99,235,.20); border-color:rgba(96,165,250,.45); color:#e2ecff; }
    section, details {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
    }
    section { padding:14px; }
    details { padding:0; }
    details > summary {
      cursor:pointer;
      list-style:none;
      padding:13px 15px;
      font-size:16px;
      font-weight:800;
      display:flex;
      align-items:center;
      justify-content:space-between;
    }
    details > summary::after { content:'Open'; color:var(--muted); font-size:12px; font-weight:800; text-transform:uppercase; }
    details[open] > summary::after { content:'Close'; }
    details > summary::-webkit-details-marker { display:none; }
    .details-body { border-top:1px solid #e5eaf0; padding:14px; display:grid; gap:14px; }
    .section-head { display:flex; align-items:flex-end; justify-content:space-between; gap:12px; margin-bottom:12px; }
    .section-kicker { color:var(--muted); font-size:13px; line-height:1.35; }
    .hero { display:grid; grid-template-columns:minmax(0, 1.85fr) minmax(300px, .72fr); gap:16px; align-items:stretch; }
    .main-overlay { display:grid; gap:10px; min-width:0; }
    .main-overlay img {
      width:100%;
      min-height:560px;
      max-height:calc(100vh - 172px);
      background:#0a0d14;
      border:1px solid #1d2735;
      border-radius:8px;
      object-fit:contain;
    }
	    .stream-status-panel {
	      display:grid;
	      grid-template-columns:repeat(4, minmax(0, 1fr));
	      gap:8px;
	    }
	    .sensor-grid {
	      display:grid;
	      grid-template-columns:repeat(4, minmax(0, 1fr));
	      gap:10px;
	      align-items:stretch;
	    }
	    .sensor-card {
	      border:1px solid #dfe7f0;
	      border-radius:8px;
	      background:#fbfdff;
	      padding:10px;
	      min-width:0;
	      display:grid;
	      gap:8px;
	    }
	    .sensor-title {
	      display:flex;
	      align-items:center;
	      justify-content:space-between;
	      gap:8px;
	    }
	    .sensor-title h3 { font-size:14px; }
	    .sensor-badge {
	      border-radius:999px;
	      background:#eef4ff;
	      color:#1d4ed8;
	      border:1px solid #bfdbfe;
	      font-size:11px;
	      font-weight:900;
	      padding:3px 7px;
	      white-space:nowrap;
	    }
	    .sensor-badge.ok { background:#ecfdf3; border-color:#bbf7d0; color:var(--good); }
	    .sensor-badge.warn { background:#fff7ed; border-color:#fed7aa; color:var(--warn); }
	    .sensor-badge.bad { background:#fff1f2; border-color:#fecdd3; color:var(--bad); }
	    .sensor-card img {
	      aspect-ratio:4 / 3;
	      width:100%;
	      min-height:190px;
	      max-height:300px;
	      object-fit:contain;
	      background:#0a0d14;
	      border:1px solid #d9e2ec;
	      border-radius:6px;
	    }
	    .pointcloud-card {
	      align-content:start;
	      background:linear-gradient(180deg, #fbfdff, #f6f9fc);
	    }
	    .pointcloud-actions {
	      display:flex;
	      gap:8px;
	      flex-wrap:wrap;
	      margin-top:4px;
	    }
	    .pointcloud-layout {
	      display:grid;
	      grid-template-columns:minmax(0, 1.45fr) minmax(420px, .7fr);
	      gap:14px;
	      align-items:start;
	    }
	    .pointcloud-preview-grid {
	      display:grid;
	      grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));
	      gap:12px;
	    }
	    .pointcloud-preview-card {
	      border:1px solid #e2e8f0;
	      border-radius:8px;
	      background:#fbfdff;
	      padding:10px;
	      display:grid;
	      gap:8px;
	      min-width:0;
	      cursor:pointer;
	    }
	    .pointcloud-preview-card img {
	      aspect-ratio:3 / 2;
	      width:100%;
	      max-height:520px;
	      object-fit:contain;
	      background:#0a0d14;
	      border:1px solid #d9e2ec;
	      border-radius:6px;
	    }
	    .pointcloud-viewer-card {
	      cursor:default;
	    }
	    .pointcloud-canvas-shell {
	      position:relative;
	      aspect-ratio:3 / 2;
	      width:100%;
	      max-height:620px;
	      min-height:360px;
	      background:#080d16;
	      border:1px solid #d9e2ec;
	      border-radius:6px;
	      overflow:hidden;
	    }
	    .pointcloud-canvas-shell canvas {
	      width:100%;
	      height:100%;
	      display:block;
	    }
	    .pointcloud-fallback {
	      position:absolute;
	      inset:0;
	      display:none;
	      place-items:center;
	      background:#0a0d14;
	      color:#dbe7f3;
	      text-align:center;
	      padding:14px;
	      font-size:13px;
	      line-height:1.35;
	    }
	    .pointcloud-fallback.visible { display:grid; }
	    .pointcloud-watermark {
	      position:absolute;
	      left:14px;
	      right:14px;
	      bottom:14px;
	      display:none;
	      padding:8px 10px;
	      border-radius:6px;
	      background:rgba(127, 29, 29, 0.82);
	      color:#fff7ed;
	      font-size:12px;
	      font-weight:900;
	      letter-spacing:0;
	      text-align:center;
	      pointer-events:none;
	    }
	    .pointcloud-watermark.visible { display:block; }
	    .pointcloud-fallback img {
	      position:absolute;
	      inset:0;
	      width:100%;
	      height:100%;
	      object-fit:contain;
	      opacity:.72;
	    }
	    .pointcloud-fallback span {
	      position:relative;
	      z-index:1;
	      background:rgba(8,13,22,.72);
	      border:1px solid rgba(255,255,255,.16);
	      border-radius:6px;
	      padding:8px 10px;
	    }
	    .pointcloud-fallback strong { color:#f5f8ff; }
	    .pointcloud-fallback small { color:#8fbaff; font-weight:900; }
	    .pointcloud-viewer-toolbar {
	      display:flex;
	      align-items:center;
	      justify-content:space-between;
	      gap:8px;
	      flex-wrap:wrap;
	    }
	    .pointcloud-viewer-toolbar button {
	      padding:6px 9px;
	      font-size:12px;
	    }
	    .pointcloud-viewer-toolbar select {
	      border:1px solid #cbd5e1;
	      border-radius:6px;
	      background:white;
	      color:#172033;
	      padding:6px 8px;
	      font-size:12px;
	      font-weight:800;
	    }
	    .toggle-pill {
	      display:inline-flex;
	      align-items:center;
	      gap:5px;
	      border:1px solid #cbd5e1;
	      border-radius:6px;
	      background:#fff;
	      color:#172033;
	      padding:5px 8px;
	      font-size:12px;
	      font-weight:800;
	      line-height:1;
	    }
	    .toggle-pill input {
	      width:13px;
	      height:13px;
	      margin:0;
	    }
	    .pointcloud-side {
	      display:grid;
	      gap:10px;
	      min-width:0;
	    }
	    .pointcloud-summary-grid {
	      grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
	    }
	    .pointcloud-diagnostics {
	      background:#fbfdff;
	    }
	    .object-profile-layout {
	      display:grid;
	      grid-template-columns:minmax(300px, .9fr) minmax(0, 1.1fr);
	      gap:14px;
	      align-items:start;
	    }
	    .object-profile-panel {
	      border:1px solid #dfe7f0;
	      border-radius:8px;
	      background:#fbfdff;
	      padding:12px;
	      display:grid;
	      gap:10px;
	      min-width:0;
	    }
	    .object-profile-library {
	      display:grid;
	      grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
	      gap:10px;
	    }
	    .object-profile-card {
	      border:1px solid #e2e8f0;
	      border-radius:8px;
	      background:#ffffff;
	      padding:10px;
	      display:grid;
	      gap:8px;
	      min-width:0;
	    }
	    .object-profile-card.active {
	      border-color:#36d483;
	      box-shadow:0 0 0 1px rgba(54,212,131,.22);
	    }
	    .object-profile-card h3 {
	      display:flex;
	      justify-content:space-between;
	      gap:8px;
	      align-items:center;
	    }
	    .object-profile-card small,
	    .object-scan-result small {
	      color:#667085;
	      line-height:1.35;
	    }
	    .object-profile-form {
	      display:flex;
	      gap:8px;
	      flex-wrap:wrap;
	      align-items:center;
	    }
	    .object-profile-form input {
	      flex:1 1 220px;
	      border:1px solid #cbd5e1;
	      border-radius:6px;
	      background:#ffffff;
	      color:#172033;
	      padding:8px 10px;
	      font-weight:800;
	    }
	    .object-scan-result {
	      display:grid;
	      gap:8px;
	    }
	    .file-link {
	      display:inline-flex;
	      align-items:center;
	      justify-content:center;
	      border:1px solid #cbd5e1;
	      border-radius:6px;
	      background:white;
	      color:#172033;
	      padding:7px 9px;
	      font-size:12px;
	      font-weight:900;
	      text-decoration:none;
	    }
	    .file-link.disabled {
	      pointer-events:none;
	      opacity:.45;
	    }
	    .status-group {
	      border:1px solid #e2e8f0;
	      border-radius:8px;
      background:#fbfdff;
      padding:9px;
      min-width:0;
      display:grid;
      gap:7px;
    }
    .status-group h3 { color:#172033; font-size:13px; }
    .status-list { display:grid; gap:5px; }
    .status-list div {
      display:grid;
      grid-template-columns:minmax(72px, .72fr) minmax(0, 1fr);
      gap:8px;
      align-items:start;
      font-size:12px;
      line-height:1.2;
    }
    .status-list b {
      color:#667085;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.02em;
    }
    .status-list span { font-weight:800; overflow-wrap:anywhere; color:#243047; }
    .status-panel { display:grid; gap:10px; align-content:start; }
    .grid { display:grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap:10px; }
    .status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px; }
    .summary-grid { grid-template-columns: repeat(6, minmax(0, 1fr)); gap:8px; margin-bottom:14px; }
    .card {
      border:1px solid #e3e8ef;
      border-left:4px solid #98a2b3;
      border-radius:8px;
      padding:8px 9px;
      background:#fcfdff;
      min-height:48px;
    }
    .status-grid .card { min-height:46px; }
    .summary-grid .card { min-height:44px; }
    .card b {
      display:block;
      color:#667085;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.02em;
      margin-bottom:4px;
      line-height:1.1;
    }
    .card span {
      display:block;
      font-size:15px;
      font-weight:800;
      line-height:1.15;
      overflow-wrap:anywhere;
    }
    .status-grid .card span { font-size:14px; }
    .ok { border-left-color:var(--good); background:#f3fbf6; }
    .ok span { color:var(--good); }
    .warn { border-left-color:var(--warn); background:#fff8ee; }
    .warn span { color:var(--warn); }
    .bad { border-left-color:var(--bad); background:#fff5f5; }
    .bad span { color:var(--bad); }
    .info { border-left-color:var(--blue); background:#f5f8ff; }
    .info span { color:#1d4ed8; }
    .panel { grid-column:span 6; }
    .panel.small { grid-column:span 4; }
    .panel.wide { grid-column:span 12; }
    .quality-layout {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(420px, 1fr));
      gap:14px;
      align-items:start;
    }
    .quality-card {
      border:1px solid #dfe7f0;
      border-radius:8px;
      background:#fbfdff;
      padding:12px;
      display:grid;
      gap:10px;
      min-width:0;
    }
    .quality-card h3 { font-size:15px; }
    .quality-hero {
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:12px;
      border-bottom:1px solid #e8edf3;
      padding-bottom:9px;
    }
    .quality-result {
      font-size:28px;
      font-weight:900;
      line-height:1;
      letter-spacing:0;
    }
    .quality-result.good { color:var(--good); }
    .quality-result.ok { color:#15803d; }
    .quality-result.weak { color:var(--warn); }
    .quality-result.invalid { color:var(--bad); }
    .quality-result.stale { color:#7c3aed; }
    .quality-row {
      display:grid;
      grid-template-columns:minmax(120px, .8fr) minmax(0, 1fr);
      gap:10px;
      padding:6px 0;
      border-bottom:1px solid #eef2f6;
      align-items:start;
    }
    .quality-row:last-child { border-bottom:0; }
    .quality-row b {
      color:#667085;
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.02em;
    }
    .quality-row span { font-weight:800; overflow-wrap:anywhere; }
    .quality-message {
      border-radius:8px;
      padding:10px;
      background:#f8fafc;
      border:1px solid #e2e8f0;
      font-size:14px;
      line-height:1.35;
      color:#344054;
    }
    .quality-message.warn { background:#fff8ee; border-color:#fed7aa; color:#9a3412; }
    .quality-message.bad { background:#fff5f5; border-color:#fecaca; color:#991b1b; }
    .quality-message.ok { background:#f3fbf6; border-color:#bbf7d0; color:#166534; }
    .embodiment-selector {
      display:flex;
      align-items:center;
      gap:8px;
      flex-wrap:wrap;
      justify-content:flex-end;
    }
    .embodiment-selector label {
      color:#94a3b8;
      font-size:11px;
      font-weight:900;
      text-transform:uppercase;
      letter-spacing:.02em;
    }
    .embodiment-selector select {
      border:1px solid #cbd5e1;
      border-radius:6px;
      background:white;
      color:#172033;
      padding:7px 9px;
      font-size:12px;
      font-weight:800;
      max-width:260px;
    }
    .embodiment-selector span {
      color:#667085;
      font-size:12px;
      font-weight:800;
      max-width:260px;
      overflow-wrap:anywhere;
    }
    .plot-caption {
      color:#475467;
      font-size:13px;
      line-height:1.35;
      margin:0;
    }
    .plot-explain {
      margin-top:10px;
      display:grid;
      gap:6px;
      color:#475467;
      font-size:13px;
      line-height:1.35;
    }
    .plot-explain p { margin:0; }
    .advanced-plots {
      grid-column:span 12;
      background:transparent;
      border:0;
    }
    .advanced-plots > summary {
      padding:10px 0;
      color:#344054;
      font-size:14px;
    }
    .advanced-plots .details-body {
      border-top:1px solid #e5eaf0;
      padding:12px 0 0;
    }
    .plot-grid { grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); }
    .quality-plot-grid { grid-template-columns:minmax(0, 1fr); }
    .plot-card {
      border:1px solid #e2e8f0;
      border-radius:8px;
      background:#fbfdff;
      padding:10px;
      display:grid;
      gap:8px;
      cursor:pointer;
      min-width:0;
    }
    .plot-card.empty img { display:none; }
    .plot-empty-state {
      display:none;
      min-height:260px;
      place-items:center;
      text-align:center;
      border:1px dashed rgba(148,163,184,.36);
      border-radius:8px;
      background:rgba(8,13,22,.84);
      color:#dbe7f3;
      padding:18px;
      line-height:1.35;
    }
    .plot-card.empty .plot-empty-state { display:grid; }
    .plot-empty-state strong {
      display:block;
      color:#f5f8ff;
      font-size:15px;
      margin-bottom:5px;
    }
    .plot-empty-state span {
      display:block;
      color:#a9b7c8;
      max-width:520px;
      margin:0 auto;
    }
    .plot-empty-state small {
      display:block;
      color:#7fb0ff;
      font-weight:900;
      margin-top:8px;
    }
    .plot-card img {
      aspect-ratio:4 / 3;
      width:100%;
      max-height:420px;
      object-fit:contain;
      background:#0a0d14;
      border:1px solid #d9e2ec;
      border-radius:6px;
    }
    .plot-card.primary-plot { grid-column:span 2; }
    .plot-card.primary-plot img { aspect-ratio:16 / 9; max-height:470px; }
    .plot-card.hero-plot { grid-column:1 / -1; }
    .plot-card.hero-plot img { aspect-ratio:16 / 10; max-height:620px; }
    .plot-card.quality-main img { aspect-ratio:16 / 9; max-height:520px; object-fit:cover; }
    .global-quality-banner {
      display:none;
      margin:12px 0 0;
      padding:10px 12px;
      border-radius:8px;
      border:1px solid #fed7aa;
      background:#fff7ed;
      color:#9a3412;
      font-weight:900;
    }
    .global-quality-banner.visible { display:block; }
    .plot-card.stale {
      border-color:#f59e0b;
      background:#fffaf0;
    }
    .plot-card.stale img { opacity:.94; filter:none; }
    .plot-stale-label {
      display:none;
      color:#b42318;
      font-size:12px;
      font-weight:800;
      letter-spacing:.02em;
      margin:0;
    }
    .plot-card.stale .plot-stale-label { display:block; }
    img { max-width:100%; width:100%; object-fit:contain; }
    pre { white-space:pre-wrap; max-height:210px; overflow:auto; background:#101828; color:#dbeafe; padding:10px; border-radius:6px; font-size:12px; }
    .note { color:#667085; font-size:13px; line-height:1.35; margin:0; }
    .control-bar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
    .toolbar { display:flex; gap:8px; flex-wrap:wrap; }
    .operator-action-bar {
      margin:0;
      padding:10px 12px;
      border:1px solid rgba(127,150,179,.35);
      border-radius:8px;
      background:rgba(12,20,31,.92);
      display:grid;
      gap:8px;
      box-shadow:none;
    }
    .operator-action-bar .control-bar {
      display:grid;
      gap:8px;
    }
    .operator-action-bar .toolbar {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:7px;
    }
    .operator-action-bar button {
      width:100%;
      min-height:34px;
      padding:7px 8px;
      font-size:12px;
    }
    .operator-action-bar .advanced-controls {
      margin-top:4px;
      background:rgba(15,24,36,.92);
      border-color:rgba(111,132,158,.28);
      box-shadow:none;
    }
    .operator-action-bar .advanced-controls > summary {
      padding:8px 10px;
      font-size:13px;
      color:#aab8ca;
    }
    .advanced-controls { margin-top:12px; background:#f8fafc; }
    .advanced-controls > summary { font-size:14px; padding:10px 12px; }
    .fresh-status { margin-top:8px; font-weight:800; color:#344054; }
    button {
      border:1px solid #cbd5e1;
      background:white;
      color:#172033;
      border-radius:6px;
      padding:8px 10px;
      font-weight:800;
      cursor:pointer;
    }
    button.primary { background:#2563eb; color:white; border-color:#1d4ed8; }
    button.danger { background:#fff1f1; color:#a22222; border-color:#ffc8c8; }
    .debug-media img { min-height:260px; background:#0a0d14; border:1px solid #d9e2ec; border-radius:6px; }
    .hidden { display:none !important; }
    body {
      color-scheme:dark;
      --ink:#e6edf6;
      --muted:#9aa8ba;
      --line:#263243;
      --panel:#111923;
      --page:#071018;
      --good:#36d483;
      --warn:#f5a524;
      --bad:#ff5f64;
      --blue:#5fa8ff;
      --panel-2:#0d1520;
      --panel-3:#172232;
      --shadow:0 20px 54px rgba(0,0,0,.28);
      background:
        radial-gradient(circle at 18% -10%, rgba(58,127,255,.16), transparent 34%),
        linear-gradient(180deg, #08111a 0%, #071018 58%, #060b11 100%);
    }
    header {
      position:sticky;
      top:0;
      z-index:15;
      background:rgba(9,17,27,.94);
      backdrop-filter:blur(14px);
      border-bottom:1px solid rgba(122,148,179,.25);
      padding:14px 22px;
      align-items:center;
    }
    main { max-width:1880px; margin:0 auto; padding-bottom:28px; }
    h1 { font-size:22px; }
    h2 { color:#f4f8ff; }
    h3 { color:#d7e2ee; }
    section, details {
      background:linear-gradient(180deg, rgba(17,25,35,.96), rgba(11,18,27,.96));
      border-color:rgba(110,132,160,.28);
      box-shadow:var(--shadow);
    }
    details > summary { color:#e7eef8; }
    details > summary::after {
      color:#9eb0c2;
      background:#172232;
      border:1px solid #2b3a4d;
      border-radius:999px;
      padding:3px 8px;
    }
    .details-body { border-top-color:rgba(129,151,176,.18); }
    .card, .status-group, .sensor-card, .quality-card, .plot-card,
    .pointcloud-preview-card, .pointcloud-diagnostics, .object-profile-panel,
    .object-profile-card {
      background:rgba(15,24,36,.92);
      border-color:rgba(111,132,158,.28);
      color:#e8eef7;
    }
    .card { border-left-width:5px; }
    .card b, .quality-row b, .status-list b { color:#94a3b8; }
    .card span, .quality-row span, .status-list span { color:#f5f8fc; }
    .ok { background:rgba(22,101,52,.16); }
    .warn { background:rgba(146,64,14,.18); }
    .bad { background:rgba(127,29,29,.20); }
    .info { background:rgba(29,78,216,.17); }
    .note, .section-kicker, .plot-caption, .plot-explain { color:#9aa8ba; }
    button, .file-link, .pointcloud-viewer-toolbar select, .embodiment-selector select,
    .object-profile-form input {
      background:#162233;
      color:#e8f0fb;
      border-color:#33445c;
    }
    button:hover, .file-link:hover { border-color:#5fa8ff; background:#1b2c42; }
    button.primary { background:#1f6feb; border-color:#388bfd; }
    button.danger { background:#3a1820; color:#ffd9dd; border-color:#7f2b37; }
    pre { background:#050b12; border:1px solid #233247; }
    .modal-card { background:#101925; border-color:#33445c; }
    .modal { z-index:50; }
    .brand-stack { display:grid; gap:4px; }
    .header-title-row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
    .product-mark {
      width:12px;
      height:36px;
      border-radius:3px;
      background:linear-gradient(180deg, #36d483, #5fa8ff);
      box-shadow:0 0 24px rgba(54,212,131,.3);
    }
    .header-controls { display:flex; gap:8px; align-items:center; justify-content:flex-end; flex-wrap:wrap; }
    .mode-toggle {
      display:flex;
      padding:3px;
      background:#0b1320;
      border:1px solid #26364c;
      border-radius:999px;
    }
    .mode-toggle button {
      border:0;
      border-radius:999px;
      padding:7px 11px;
      background:transparent;
      color:#91a4b8;
      font-size:12px;
    }
    .mode-toggle button.active {
      background:#e8f0fb;
      color:#071018;
    }
    .sensor-strip {
      display:flex;
      gap:7px;
      flex-wrap:wrap;
      justify-content:flex-end;
    }
    .operator-shell { padding:0; overflow:hidden; }
    .hand-demo-grid {
      display:grid;
      grid-template-columns:minmax(0, 1fr) minmax(0, 1fr) minmax(280px, .82fr);
      gap:12px;
      padding:12px 14px 15px;
      background:radial-gradient(circle at 50% 0%, #111d2b 0, #080e16 48%, #060b12 100%);
      align-items:stretch;
    }
    .hand-demo-panel {
      min-width:0;
      border:1px solid rgba(130,161,197,.30);
      border-radius:12px;
      background:linear-gradient(180deg, rgba(18,29,43,.98), rgba(8,14,22,.99));
      box-shadow:0 12px 30px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.025);
      padding:11px;
      display:grid;
      gap:8px;
      align-content:start;
    }
    .hand-demo-panel-head {
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:8px;
    }
    .hand-demo-panel-head h3 { color:#f4f8ff; font-size:15px; letter-spacing:.01em; }
    .hand-demo-panel-head small {
      display:block;
      color:#8fa3b9;
      font-size:11px;
      line-height:1.25;
      margin-top:3px;
    }
    .hand-demo-panel-badge {
      border:1px solid rgba(87,174,255,.45);
      border-radius:999px;
      padding:3px 7px;
      color:#cfe2f7;
      background:rgba(12,27,42,.9);
      font-size:10px;
      font-weight:900;
      white-space:nowrap;
    }
    .hand-demo-visual {
      position:relative;
      display:grid;
      place-items:center;
      width:100%;
      min-height:0;
      aspect-ratio:4 / 3;
      overflow:hidden;
      border:1px solid rgba(96,145,196,.38);
      border-radius:10px;
      background:#02070d;
      box-shadow:inset 0 0 30px rgba(0,0,0,.34);
    }
    .hand-demo-visual img {
      display:block;
      width:100%;
      height:100%;
      min-height:0;
      object-fit:cover;
      background:#02070d;
    }
    #handDemoRealSenseVisual img { object-fit:contain; }
    .hand-demo-empty {
      position:absolute;
      inset:10px;
      display:none;
      place-items:center;
      text-align:center;
      padding:12px;
      border:1px dashed rgba(143,162,186,.35);
      border-radius:7px;
      background:rgba(5,10,17,.88);
      color:#b8c7d9;
      line-height:1.3;
    }
    .hand-demo-visual.empty img { display:none; }
    .hand-demo-visual.empty .hand-demo-empty { display:grid; }
    .hand-demo-empty strong,
    .hand-demo-empty span,
    .hand-demo-empty small { display:block; }
    .hand-demo-empty strong { color:#f5f8ff; margin-bottom:4px; }
    .hand-demo-empty small { color:#8fbaff; font-weight:900; margin-top:5px; }
    .hand-demo-status-grid {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:6px;
    }
    .hand-demo-status-grid .card { min-height:45px; padding:7px 8px; }
    .hand-demo-status-grid .card span { font-size:12px; }
    #handDemoDavisPanel .hand-demo-status-grid { grid-template-columns:repeat(3, minmax(0, 1fr)); }
    #handDemoDavisPanel .hand-demo-status-grid .card { min-height:32px; padding:4px 6px; }
    #handDemoDavisPanel .hand-demo-status-grid .card b { font-size:8px; }
    #handDemoDavisPanel .hand-demo-status-grid .card span { font-size:10px; }
    #handDemoQualityPanel { gap:6px; }
    #handDemoQualityPanel .hand-demo-status-grid { grid-template-columns:repeat(3, minmax(0, 1fr)); }
    #handDemoQualityPanel .hand-demo-status-grid .card { min-height:32px; padding:4px 6px; }
    #handDemoQualityPanel .hand-demo-status-grid .card b { font-size:8px; }
    #handDemoQualityPanel .hand-demo-status-grid .card span { font-size:10px; }
    #handDemoQualityPanel .hand-demo-timeline { display:none; }
    .quality-wrench-shell {
      position:relative;
      width:100%;
      height:105px;
      min-height:105px;
      overflow:hidden;
      border:1px solid #26384e;
      border-radius:7px;
      background:
        radial-gradient(circle at 50% 44%, rgba(40,75,112,.22), transparent 58%),
        #030911;
    }
    .quality-wrench-shell canvas { display:block; width:100%; height:100%; }
    .quality-wrench-fallback {
      position:absolute;
      inset:10px;
      display:none;
      place-items:center;
      padding:14px;
      text-align:center;
      border:1px dashed rgba(143,162,186,.35);
      border-radius:7px;
      background:rgba(5,10,17,.94);
      color:#b8c7d9;
      line-height:1.3;
      z-index:4;
    }
    .quality-wrench-fallback strong,
    .quality-wrench-fallback span,
    .quality-wrench-fallback small { display:block; }
    .quality-wrench-fallback strong { color:#f5f8ff; margin-bottom:4px; }
    .quality-wrench-fallback small { color:#8fbaff; font-weight:900; margin-top:5px; }
    .quality-wrench-shell.unavailable canvas { display:none; }
    .quality-wrench-shell.unavailable .quality-wrench-fallback { display:grid; }
    .quality-wrench-axis {
      position:absolute;
      z-index:3;
      color:#b9c9da;
      font-size:10px;
      font-weight:900;
      text-shadow:0 1px 3px #000;
      pointer-events:none;
    }
    .quality-wrench-axis.fx { right:7px; bottom:8px; color:#ff8b8f; }
    .quality-wrench-axis.fy { left:7px; top:50%; color:#77e8a8; }
    .quality-wrench-axis.tau { left:50%; top:6px; color:#76b8ff; }
    .quality-wrench-legend {
      position:absolute;
      left:7px;
      bottom:6px;
      z-index:3;
      display:grid;
      gap:3px;
      padding:5px 6px;
      border-radius:5px;
      background:rgba(3,9,17,.72);
      color:#b8c7d9;
      font-size:9px;
      line-height:1.15;
      pointer-events:none;
    }
    .quality-wrench-legend span::before {
      content:"";
      display:inline-block;
      width:10px;
      height:3px;
      margin-right:5px;
      border-radius:2px;
      vertical-align:middle;
      background:var(--legend-color);
    }
    .quality-wrench-caption {
      display:flex;
      justify-content:space-between;
      gap:8px;
      color:#9fb0c3;
      font-size:10px;
      line-height:1.25;
    }
    .quality-wrench-caption strong { color:#dce8f5; }
    .quality-risk-strip {
      display:grid;
      grid-template-columns:repeat(4, minmax(0, 1fr));
      gap:4px;
    }
    .quality-risk-strip div {
      min-width:0;
      border:1px solid rgba(111,132,158,.28);
      border-radius:5px;
      padding:4px 5px;
      background:#0d1622;
    }
    .quality-risk-strip b,
    .quality-risk-strip span { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .quality-risk-strip b { color:#8fa3b9; font-size:8px; text-transform:uppercase; }
    .quality-risk-strip span { color:#f1f6fc; font-size:10px; font-weight:900; margin-top:2px; }
    .quality-adjustment-flow {
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:5px;
    }
    .quality-adjustment-step {
      min-width:0;
      border:1px solid rgba(95,168,255,.30);
      border-radius:6px;
      padding:6px;
      background:#0b1623;
      text-align:center;
    }
    .quality-adjustment-step b,
    .quality-adjustment-step span,
    .quality-adjustment-step small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .quality-adjustment-step b { color:#91a4b8; font-size:8px; text-transform:uppercase; }
    .quality-adjustment-step span { color:#f4f8ff; font-size:13px; font-weight:900; margin-top:2px; }
    .quality-adjustment-step small { color:#8fa3b9; font-size:8px; margin-top:2px; }
    .quality-adjustment-step.included { border-color:rgba(245,165,36,.55); background:rgba(146,64,14,.16); }
    .quality-adjustment-step.excluded { border-color:rgba(143,162,186,.25); }
    .hand-demo-section-label {
      color:#91a4b8;
      font-size:10px;
      font-weight:900;
      letter-spacing:.06em;
      text-transform:uppercase;
      border-top:1px solid rgba(124,146,174,.18);
      padding-top:7px;
    }
    .hand-demo-decision {
      border:1px solid #2f4158;
      border-radius:8px;
      padding:10px;
      background:#0b1320;
      display:grid;
      gap:5px;
    }
    .hand-demo-decision strong { color:#f7fbff; font-size:22px; line-height:1.05; }
    .hand-demo-decision span { color:#b9c7d7; font-size:12px; line-height:1.3; }
    .hand-demo-decision.good { border-color:rgba(54,212,131,.48); }
    .hand-demo-decision.warn { border-color:rgba(245,165,36,.48); }
    .hand-demo-decision.bad { border-color:rgba(255,95,100,.52); }
    .hand-demo-timeline {
      display:grid;
      gap:4px;
      max-height:66px;
      overflow:hidden;
      color:#afbdd0;
      font-size:11px;
    }
    .hand-demo-timeline div {
      display:grid;
      grid-template-columns:48px minmax(0, 1fr);
      gap:6px;
    }
    .hand-mode-banner {
      display:flex;
      align-items:center;
      gap:10px;
      padding:8px 16px;
      border-bottom:1px solid rgba(95,168,255,.25);
      background:linear-gradient(90deg, rgba(37,99,235,.20), rgba(14,23,34,.78));
      color:#dcecff;
      font-size:12px;
    }
    .hand-mode-banner strong {
      color:#ffffff;
      font-size:13px;
      letter-spacing:.02em;
    }
    .hand-mode-banner span { font-weight:900; }
    .hand-mode-banner small {
      margin-left:auto;
      color:#9fb1c7;
      text-align:right;
    }
    .operator-topline {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:14px;
      padding:14px 16px;
      border-bottom:1px solid rgba(128,152,180,.18);
      background:rgba(11,18,27,.78);
    }
    .workflow-stepper {
      display:grid;
      grid-template-columns:repeat(5, minmax(0, 1fr));
      gap:6px;
      min-width:min(760px, 100%);
    }
    .workflow-step {
      position:relative;
      min-width:0;
      padding:8px 8px;
      border-radius:7px;
      background:#0b1320;
      border:1px solid #27364a;
      color:#94a3b8;
      font-size:11px;
      font-weight:900;
      text-align:center;
      text-transform:uppercase;
      letter-spacing:.03em;
      overflow:hidden;
      white-space:nowrap;
      text-overflow:ellipsis;
    }
    .workflow-step.done { color:#dffdea; border-color:rgba(54,212,131,.42); background:rgba(22,101,52,.20); }
    .workflow-step.active { color:#ffffff; border-color:#5fa8ff; background:rgba(37,99,235,.30); }
    .workflow-step.blocked { color:#ffe3e3; border-color:rgba(255,95,100,.42); background:rgba(127,29,29,.22); }
    .operator-visual {
      padding:14px;
      background:#090f17;
      border-right:1px solid rgba(128,152,180,.18);
    }
    .visual-tabs {
      display:flex;
      gap:7px;
      flex-wrap:wrap;
      align-items:center;
      margin-bottom:10px;
    }
    .visual-tab {
      border-radius:999px;
      font-size:12px;
      padding:7px 10px;
      color:#9fb0c4;
    }
    .visual-tab.active {
      color:#071018;
      background:#e7f0fb;
      border-color:#e7f0fb;
    }
    .release-nav {
      position:sticky;
      top:66px;
      z-index:14;
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      margin:0 0 2px;
      padding:10px 12px;
      border:1px solid rgba(134,150,171,.20);
      border-radius:8px;
      background:rgba(14,23,35,.76);
      box-shadow:var(--shadow-sm);
      overflow-x:auto;
      scrollbar-width:thin;
    }
    .release-nav a {
      color:#dbeafe;
      text-decoration:none;
      font-weight:800;
      font-size:12px;
      padding:7px 10px;
      border-radius:999px;
      border:1px solid rgba(125,151,181,.28);
      background:rgba(15,23,42,.78);
    }
    .release-nav a:hover {
      border-color:#7dd3fc;
      background:#0e3a5a;
    }
    .release-nav a.active {
      color:#04121f;
      background:#7dd3fc;
      border-color:#7dd3fc;
    }
    .release-summary-strip {
      display:grid;
      grid-template-columns:repeat(4, minmax(0, 1fr));
      gap:8px;
      padding:10px 14px;
      border-bottom:1px solid rgba(128,152,180,.18);
      background:rgba(7,13,21,.72);
    }
    .release-summary-card {
      border:1px solid rgba(124,146,174,.24);
      border-radius:8px;
      padding:8px 10px;
      background:#0b1320;
      min-width:0;
    }
    .release-summary-card b {
      display:block;
      color:#91a4b8;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.05em;
      margin-bottom:4px;
    }
    .release-summary-card span {
      display:block;
      color:#f4f8ff;
      font-size:14px;
      font-weight:900;
      overflow-wrap:anywhere;
    }
    .visual-stage {
      position:relative;
      min-height:0;
      border:1px solid #243246;
      border-radius:8px;
      background:#02070d;
      overflow:hidden;
      display:grid;
      align-items:start;
      justify-items:center;
      width:min(100%, 900px);
      justify-self:center;
    }
    .visual-panel {
      display:none;
      height:auto;
      min-height:0;
      width:100%;
      place-items:center;
      position:relative;
    }
    .visual-panel.active { display:grid; }
    .visual-panel img {
      border:0;
      border-radius:0;
      min-height:0;
      max-height:none;
      width:100%;
      height:auto;
      background:#02070d;
      cursor:zoom-in;
    }
    .davis-roi-on-overlay {
      position:absolute;
      inset:0;
      width:100%;
      height:100%;
      z-index:3;
      pointer-events:none;
    }
    .davis-roi-on-overlay.hidden { display:none; }
    .visual-empty {
      position:absolute;
      inset:auto 18px 18px;
      display:none;
      padding:10px 12px;
      border:1px solid rgba(143,162,186,.28);
      border-radius:7px;
      background:rgba(5,10,17,.82);
      color:#b8c7d9;
      font-size:13px;
      line-height:1.35;
      pointer-events:none;
    }
    .visual-empty strong {
      display:block;
      color:#f5f8ff;
      margin-bottom:3px;
    }
    .visual-empty span,
    .visual-empty small {
      display:block;
    }
    .visual-empty small {
      color:#8fbaff;
      font-weight:900;
      margin-top:3px;
    }
    .visual-panel.empty .visual-empty { display:block; }
    .decision-rail {
      padding:14px;
      display:grid;
      gap:12px;
      align-content:start;
      background:linear-gradient(180deg, rgba(14,23,34,.98), rgba(9,15,23,.98));
    }
    .decision-card {
      border:1px solid #2f4158;
      border-radius:8px;
      padding:14px;
      background:#0b1320;
      display:grid;
      gap:10px;
    }
    .decision-card.good { border-color:rgba(54,212,131,.48); background:linear-gradient(180deg, rgba(22,101,52,.22), #0b1320); }
    .decision-card.warn { border-color:rgba(245,165,36,.48); background:linear-gradient(180deg, rgba(146,64,14,.24), #0b1320); }
    .decision-card.bad { border-color:rgba(255,95,100,.52); background:linear-gradient(180deg, rgba(127,29,29,.26), #0b1320); }
    .hand-band-status {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:7px;
    }
    .hand-band-item {
      min-width:0;
      padding:8px 9px;
      border:1px solid #2b3b50;
      border-radius:7px;
      background:#0b1320;
    }
    .hand-band-item b {
      display:block;
      margin-bottom:3px;
      color:#91a4b8;
      font-size:10px;
      letter-spacing:.05em;
      text-transform:uppercase;
    }
    .hand-band-item span {
      display:block;
      color:#f4f8ff;
      font-size:12px;
      font-weight:900;
      line-height:1.25;
      overflow-wrap:anywhere;
    }
    .hand-band-item.ok { border-color:rgba(54,212,131,.42); }
    .hand-band-item.warn { border-color:rgba(245,165,36,.46); }
    .hand-band-item.bad { border-color:rgba(255,95,100,.52); }
    .decision-label {
      color:#94a3b8;
      text-transform:uppercase;
      font-size:11px;
      font-weight:900;
      letter-spacing:.08em;
    }
    .decision-state {
      font-size:34px;
      line-height:1;
      font-weight:950;
      letter-spacing:0;
      color:#f7fbff;
    }
    .decision-explain { color:#b9c7d7; line-height:1.35; font-size:14px; }
    .next-action {
      border-radius:7px;
      background:#111d2b;
      border:1px solid #2b3a4d;
      padding:10px;
      font-weight:900;
      color:#f3f7fb;
    }
    .metric-rail {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:8px;
    }
    .metric-pill {
      border:1px solid #26374e;
      border-radius:8px;
      background:#101926;
      padding:9px;
      min-width:0;
    }
    .metric-pill b {
      display:block;
      color:#93a4b7;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.04em;
      margin-bottom:4px;
    }
    .metric-pill span {
      display:block;
      color:#f3f8ff;
      font-size:15px;
      font-weight:900;
      overflow-wrap:anywhere;
    }
    .evidence-list {
      display:grid;
      gap:7px;
    }
    .evidence-row {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      border-bottom:1px solid rgba(124,146,174,.16);
      padding-bottom:7px;
      color:#c7d3e2;
      font-size:13px;
    }
    .evidence-row:last-child { border-bottom:0; padding-bottom:0; }
    .evidence-row b { color:#94a3b8; text-transform:uppercase; font-size:10px; letter-spacing:.05em; }
    .live-contribution-grid {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:7px;
    }
    .live-contribution {
      border:1px solid rgba(124,146,174,.22);
      border-radius:7px;
      padding:8px;
      background:rgba(12,20,31,.92);
      color:#d8e3f0;
      min-width:0;
    }
    .live-contribution b {
      display:block;
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.05em;
      color:#93a4b7;
      margin-bottom:4px;
    }
    .live-contribution span {
      display:block;
      font-weight:900;
      overflow-wrap:anywhere;
    }
    .live-contribution small {
      display:block;
      margin-top:4px;
      color:#a8b8cc;
      line-height:1.25;
    }
    .quality-timeline {
      display:grid;
      gap:5px;
      max-height:132px;
      overflow:auto;
      border-top:1px solid rgba(124,146,174,.18);
      padding-top:8px;
      color:#cbd7e6;
      font-size:12px;
    }
    .quality-timeline-row {
      display:grid;
      grid-template-columns:54px 1fr;
      gap:8px;
      align-items:start;
    }
    .quality-timeline-row time { color:#91a4b8; }
    .benchmark-banner {
      margin:10px 0 14px;
      padding:10px 12px;
      border:1px solid rgba(245,165,36,.48);
      border-radius:8px;
      background:rgba(146,64,14,.18);
      color:#ffdca8;
      font-weight:900;
      line-height:1.35;
    }
    .benchmark-layout {
      display:grid;
      grid-template-columns:minmax(280px, 340px) minmax(0, 1fr) minmax(260px, 320px);
      gap:12px;
      align-items:start;
    }
    .benchmark-panel {
      border:1px solid rgba(111,132,158,.28);
      border-radius:8px;
      background:rgba(15,24,36,.92);
      padding:12px;
      min-width:0;
    }
    .benchmark-panel-head {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      margin-bottom:10px;
    }
    .benchmark-presets, .compact-toolbar {
      display:flex;
      flex-wrap:wrap;
      gap:7px;
      margin-bottom:10px;
    }
    .benchmark-presets button, .compact-toolbar button {
      padding:7px 9px;
      font-size:12px;
    }
    .benchmark-feature-groups {
      display:grid;
      gap:9px;
      max-height:620px;
      overflow:auto;
      padding-right:3px;
    }
    .benchmark-feature-group {
      border:1px solid rgba(124,146,174,.18);
      border-radius:7px;
      padding:9px;
      background:rgba(10,17,26,.74);
    }
    .benchmark-feature-group h4 {
      margin:0 0 6px;
      color:#dbeafe;
      font-size:12px;
      text-transform:uppercase;
      letter-spacing:.04em;
    }
    .benchmark-check {
      display:flex;
      gap:8px;
      align-items:flex-start;
      padding:5px 0;
      color:#d7e4f2;
      font-size:13px;
      line-height:1.25;
    }
    .benchmark-check input { margin-top:2px; }
    .benchmark-check small {
      display:block;
      color:#93a4b7;
      margin-top:2px;
    }
    .benchmark-production {
      display:grid;
      grid-template-columns:repeat(4, minmax(0, 1fr));
      gap:8px;
      margin-bottom:10px;
    }
    .benchmark-production .metric-pill { background:#0b1320; }
    .benchmark-card-grid {
      display:grid;
      grid-template-columns:repeat(4, minmax(300px, 1fr));
      gap:10px;
      overflow-x:auto;
      padding-bottom:4px;
    }
    .benchmark-card {
      border:1px solid rgba(124,146,174,.26);
      border-radius:8px;
      background:#0b1320;
      padding:10px;
      display:grid;
      gap:8px;
      min-width:220px;
    }
    .benchmark-card h4 {
      margin:0;
      color:#f5f8ff;
      font-size:15px;
    }
    .benchmark-card .feature-count {
      color:#9fb0c4;
      font-size:12px;
      font-weight:800;
    }
    .benchmark-kv {
      display:grid;
      grid-template-columns:1fr auto;
      gap:8px;
      border-top:1px solid rgba(124,146,174,.14);
      padding-top:6px;
      color:#d9e6f4;
      font-size:12px;
    }
    .benchmark-kv b { color:#93a4b7; }
    .benchmark-limitation {
      padding:8px;
      border-radius:7px;
      background:rgba(245,165,36,.12);
      color:#ffdca8;
      font-size:12px;
      line-height:1.3;
    }
    .benchmark-trace {
      border-top:1px solid rgba(124,146,174,.14);
      padding-top:6px;
      color:#c9d7e6;
      font-size:12px;
    }
    .benchmark-trace summary {
      cursor:pointer;
      color:#9fc5ff;
      font-weight:800;
    }
    .benchmark-trace-row {
      display:grid;
      grid-template-columns:minmax(0, 1fr) minmax(0, 1.5fr);
      gap:8px;
      border-top:1px solid rgba(124,146,174,.1);
      padding:5px 0;
    }
    .benchmark-trace-row small {
      display:block;
      color:#8396aa;
      margin-top:2px;
    }
    .benchmark-label-grid {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:7px;
      margin-bottom:10px;
    }
    .benchmark-label-grid button {
      padding:8px;
      font-size:12px;
      min-height:38px;
    }
    .benchmark-bottom-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(420px, 1fr));
      gap:12px;
      margin-top:12px;
    }
    .benchmark-run-panel {
      display:grid;
      grid-template-columns:minmax(280px, 1.05fr) minmax(300px, 1.25fr) minmax(260px, .9fr);
      gap:12px;
      margin-bottom:12px;
    }
    .benchmark-run-card {
      border:1px solid rgba(124,146,174,.26);
      border-radius:8px;
      background:#0b1320;
      color:#dce8f7;
      padding:12px;
      min-width:0;
      display:grid;
      gap:9px;
    }
    .benchmark-run-card h3 { color:#f4f8ff; }
    .benchmark-run-controls {
      display:flex;
      flex-wrap:wrap;
      gap:7px;
      align-items:center;
    }
    .benchmark-run-controls label {
      color:#aebfd2;
      font-size:12px;
      font-weight:800;
      display:inline-flex;
      gap:6px;
      align-items:center;
    }
    .benchmark-run-controls select,
    .benchmark-run-controls input {
      border:1px solid rgba(148,163,184,.45);
      border-radius:6px;
      padding:7px 8px;
      color:#f8fbff;
      background:#111b2a;
      min-height:34px;
    }
    .benchmark-progress-shell {
      height:12px;
      border-radius:999px;
      overflow:hidden;
      background:#172337;
      border:1px solid rgba(148,163,184,.28);
    }
    .benchmark-progress-bar {
      height:100%;
      width:0%;
      background:linear-gradient(90deg, #38bdf8, #22c55e);
      transition:width .25s ease;
    }
    .benchmark-precheck-list {
      display:grid;
      gap:6px;
      font-size:12px;
    }
    .benchmark-precheck-list div {
      display:flex;
      justify-content:space-between;
      gap:8px;
      border-top:1px solid rgba(148,163,184,.14);
      padding-top:5px;
    }
    .benchmark-result-table {
      width:100%;
      border-collapse:collapse;
      font-size:12px;
      color:#223047;
    }
    .benchmark-result-table th,
    .benchmark-result-table td {
      border-bottom:1px solid rgba(148,163,184,.18);
      padding:6px;
      text-align:left;
      vertical-align:top;
    }
    .benchmark-result-table th {
      color:#5b6b82;
      font-size:11px;
      text-transform:uppercase;
    }
    .benchmark-finding {
      padding:7px 8px;
      border-radius:7px;
      background:rgba(56,189,248,.10);
      border:1px solid rgba(56,189,248,.22);
      color:#dff6ff;
      font-size:12px;
      line-height:1.35;
    }
    .presentation .engineering-only { display:none !important; }
    .engineering .presentation-only { display:none !important; }
    .presentation .release-nav { display:none !important; }
    .presentation .benchmark-trace,
    .presentation .pointcloud-diagnostics,
    .presentation .presentation-hidden {
      display:none !important;
    }
    .presentation .hero {
      grid-template-columns:minmax(0, 1.7fr) minmax(480px, .82fr);
      gap:0;
    }
    .presentation .operator-topline { padding:10px 14px; }
    .presentation .decision-rail {
      padding:10px;
      gap:8px;
    }
    .presentation .decision-card {
      padding:10px;
      gap:6px;
    }
    .presentation .decision-state { font-size:28px; }
    .presentation .decision-explain { font-size:13px; }
    .presentation .next-action { padding:8px; }
    .presentation .hand-band-status {
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:6px;
    }
    .presentation .hand-band-item { padding:7px; }
    .presentation .metric-rail {
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:6px;
    }
    .presentation .metric-pill { padding:7px; }
    .presentation .metric-pill span { font-size:13px; }
    .presentation .metric-guidance-confidence { grid-column:span 2; }
    .presentation .operator-action-bar { padding:8px 10px; gap:6px; }
    .presentation .operator-action-bar .toolbar { gap:6px; }
    .hand-demo-actions { margin:0 12px 12px; }
    .presentation .hand-demo-actions .toolbar {
      grid-template-columns:repeat(4, minmax(0, 1fr));
    }
    .operator-feedback { display:grid; gap:4px; }
    .operator-feedback .note:empty { display:none; }
    .hand-offset-tuning {
      border:1px solid rgba(95,168,255,.28);
      border-radius:8px;
      background:rgba(14,27,46,.62);
      padding:6px 9px;
    }
    .hand-offset-tuning summary { cursor:pointer; color:#d8ecff; font-size:12px; font-weight:900; }
    .hand-offset-grid { display:grid; grid-template-columns:repeat(4, minmax(130px, 1fr)); gap:8px; margin-top:8px; }
    .hand-offset-control { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:6px; color:#b8c8dc; font-size:11px; font-weight:800; }
    .hand-offset-control input[type="range"] { width:100%; min-width:70px; accent-color:#5fa8ff; }
    .hand-offset-control output { color:#f5f8ff; min-width:48px; text-align:right; }
    .hand-offset-actions { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-top:8px; }
    .hand-offset-actions button { padding:6px 9px; font-size:11px; }
    .hand-offset-honesty { color:#f4c95d; font-size:10px; }
    .operator-safety-note {
      padding-top:2px;
      color:#a8b8cc;
      font-size:12px;
    }
    .workspace-section { scroll-margin-top:0; }
    .sensor-card img, .debug-media img, .plot-card img, .pointcloud-preview-card img,
    .pointcloud-canvas-shell, .modal img {
      background:#02070d;
      border-color:#26364b;
    }
    .operator-action-bar {
      background:rgba(10,17,26,.92);
      border-color:rgba(127,150,179,.35);
    }
    .presentation main { padding-bottom:28px; }
    .toast-region {
      position:fixed;
      right:18px;
      top:92px;
      z-index:60;
      display:grid;
      gap:8px;
      width:min(360px, calc(100vw - 36px));
      pointer-events:none;
    }
    .toast {
      border:1px solid #31435a;
      background:rgba(12,20,30,.94);
      color:#edf4ff;
      border-radius:8px;
      padding:10px 12px;
      box-shadow:0 12px 28px rgba(0,0,0,.34);
      font-size:13px;
      font-weight:800;
      line-height:1.3;
    }
    .toast.bad { border-color:#7f2b37; color:#ffe1e5; }
    .toast.ok { border-color:#1f8f55; color:#defbe9; }
    .modal {
      position:fixed;
      inset:0;
      display:none;
      align-items:center;
      justify-content:center;
      padding:22px;
      background:rgba(13,18,30,.78);
      z-index:20;
    }
    .modal.open { display:flex; }
    .modal-card {
      width:min(1200px, 96vw);
      max-height:94vh;
      background:white;
      border-radius:8px;
      border:1px solid #d9e2ec;
      padding:14px;
      display:grid;
      gap:10px;
    }
    .modal-head { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .modal-actions { display:flex; gap:8px; align-items:center; justify-content:flex-end; flex-wrap:wrap; }
    .modal-actions a { text-decoration:none; }
    .modal img {
      max-height:82vh;
      width:100%;
      object-fit:contain;
      background:#0a0d14;
      border-radius:6px;
      border:1px solid #d9e2ec;
    }
    .section-unavailable {
      padding:12px;
      border:1px solid rgba(245,165,36,.38);
      border-radius:8px;
      background:rgba(146,64,14,.16);
      color:#ffe1af;
      font-weight:900;
      line-height:1.35;
      margin-bottom:12px;
    }
    .davis-visual-grid {
      display:grid;
      grid-template-columns:1.2fr .8fr;
      gap:12px;
      margin-top:12px;
    }
    .davis-visual-grid .plot-card img {
      min-height:360px;
      max-height:520px;
    }
    @media (min-width: 1500px) and (min-height: 900px) {
      header { padding:10px 22px; }
      main { padding:12px 18px 28px; gap:12px; }
      .hero { grid-template-columns:minmax(0, 2.35fr) minmax(380px, .9fr); gap:12px; }
      .hand-demo-grid { grid-template-columns:minmax(0, 1fr) minmax(0, 1fr) minmax(280px, .82fr); }
      .hand-demo-panel { max-height:660px; overflow:hidden; }
      .hand-demo-visual { aspect-ratio:16 / 10; }
      .visual-stage { width:min(100%, 900px); }
      .visual-panel, .visual-panel img { min-height:0; max-height:none; }
      .decision-rail { max-height:720px; overflow:auto; }
      .pointcloud-canvas-shell { min-height:620px; max-height:720px; }
      .plot-card.hero-plot img { max-height:720px; }
    }
    @media (max-width: 1180px) {
	      .hero { grid-template-columns:1fr; }
	      .presentation .hero { grid-template-columns:1fr; }
	      .hand-demo-grid { grid-template-columns:1fr; }
	      .hand-demo-actions .toolbar,
	      .presentation .hand-demo-actions .toolbar { grid-template-columns:repeat(2, minmax(0, 1fr)); }
	      .hand-offset-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
	      .operator-topline { align-items:flex-start; flex-direction:column; }
	      .workflow-stepper { grid-template-columns:repeat(5, minmax(0, 1fr)); min-width:0; width:100%; }
	      .main-overlay img { min-height:0; max-height:none; }
	      .visual-stage, .visual-panel, .visual-panel img { min-height:0; max-height:none; }
	      .stream-status-panel { grid-template-columns:repeat(2, minmax(0, 1fr)); }
	      .sensor-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
      .pointcloud-layout { grid-template-columns:1fr; }
	      .summary-grid { grid-template-columns:repeat(3, minmax(0, 1fr)); }
	      .quality-layout { grid-template-columns:1fr; }
	      .quality-plot-grid { grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); }
	      .plot-card.primary-plot, .plot-card.hero-plot { grid-column:span 1; }
	      .benchmark-layout { grid-template-columns:1fr; }
	      .benchmark-run-panel { grid-template-columns:1fr; }
	      .benchmark-card-grid { grid-template-columns:repeat(2, minmax(300px, 1fr)); }
	      .benchmark-bottom-grid, .benchmark-production { grid-template-columns:1fr; }
	      .release-summary-strip { grid-template-columns:repeat(2, minmax(0, 1fr)); }
	      .davis-visual-grid { grid-template-columns:1fr; }
      .panel, .panel.small { grid-column:span 12; }
    }
    @media (max-width: 720px) {
      header { padding:14px 16px; display:grid; }
      .header-side { justify-items:start; }
      .header-controls, .sensor-strip { justify-content:flex-start; }
      main { padding:10px; gap:12px; }
	      .hand-mode-banner { align-items:flex-start; flex-direction:column; }
	      .hand-mode-banner small { margin-left:0; text-align:left; }
	      .status-grid, .summary-grid { grid-template-columns:1fr; }
	      .main-overlay img { min-height:270px; }
	      .visual-stage, .visual-panel, .visual-panel img { min-height:0; }
	      .workflow-stepper { grid-template-columns:repeat(2, minmax(0, 1fr)); }
	      .metric-rail { grid-template-columns:1fr; }
	      .presentation .metric-rail,
	      .presentation .hand-band-status { grid-template-columns:1fr; }
	      .hand-demo-status-grid,
	      .hand-demo-actions .toolbar,
	      .presentation .hand-demo-actions .toolbar { grid-template-columns:1fr; }
	      .hand-offset-grid { grid-template-columns:1fr; }
	      .stream-status-panel { grid-template-columns:1fr; }
	      .sensor-grid { grid-template-columns:1fr; }
	      .plot-grid { grid-template-columns:1fr; }
	      .quality-plot-grid { grid-template-columns:1fr; }
	      .benchmark-card-grid { grid-template-columns:1fr; }
	      .benchmark-label-grid { grid-template-columns:1fr; }
	      .release-summary-strip { grid-template-columns:1fr; }
      .plot-card img, .plot-card.primary-plot img { aspect-ratio:4 / 3; }
    }
  </style>
</head>
<body class="presentation">
  <header>
    <div class="header-title-row">
      <div class="product-mark" aria-hidden="true"></div>
      <div class="brand-stack">
        <h1>RGB-D + DAVIS Grasp Quality Demo</h1>
        <div class="subtitle">RealSense estimates grasp quality; DAVIS monitors post-grasp micro-motion.</div>
      </div>
    </div>
    <div class="header-side">
      <div class="header-controls">
        <div class="mode-toggle" aria-label="Dashboard mode">
          <button id="presentationModeBtn" class="active" onclick="setDashboardMode('presentation')">Presentation</button>
          <button id="engineeringModeBtn" onclick="setDashboardMode('engineering')">Engineering</button>
        </div>
        <button onclick="toggleFullscreen()">Fullscreen</button>
        <div class="clock engineering-only" id="clock"></div>
      </div>
      <div class="sensor-strip">
        <span class="chip" id="realsenseChip">RealSense: waiting</span>
        <span class="chip" id="davisChip">DAVIS: optional</span>
        <span class="chip engineering-only" id="pointcloudChip">Point Cloud: waiting</span>
        <span class="chip engineering-only" id="syncChip">Sync: waiting</span>
      </div>
    </div>
  </header>
  <div class="toast-region" id="toastRegion" aria-live="polite"></div>
  <main>
    <nav class="release-nav" aria-label="Release candidate dashboard tabs">
      <a href="#operatorSection">Hand Grasp Demo</a>
      <a href="#davisMotionSection">DAVIS Motion</a>
      <a href="#qualitySection">Grasp Quality</a>
      <a href="#pointcloudSection">Point Cloud</a>
      <a href="#multiviewSection" class="engineering-only">Multi-View Scan</a>
      <a href="#benchmarkSection" class="engineering-only">Benchmark Lab</a>
      <a href="#embodimentSection" class="engineering-only">Embodiment</a>
      <a href="#diagnosticsDetails" class="engineering-only">Diagnostics</a>
    </nav>
    <section id="operatorSection" class="demo-section operator-shell">
      <div class="hand-mode-banner" id="handModeBanner">
        <strong class="engineering-only">Hand Prototype Demo</strong>
        <span id="handModeLabel">Hand-operated RealSense + DAVIS grasp-quality monitoring demo</span>
        <small>Manual hand movement only.</small>
      </div>
      <div class="operator-topline">
        <div>
          <h2><span class="presentation-only">Hand Grasp Demo</span><span class="engineering-only">Hand Prototype Demo · Operator Workspace — RealSense Overview</span></h2>
          <div class="section-kicker engineering-only">Live RealSense RGB-D grasp overview: detect → lock the bottle-relative grasp band → manually approach → monitor close events → monitor pickup.</div>
        </div>
        <div class="workflow-stepper" id="workflowStepper" aria-label="Five-stage hand prototype workflow"></div>
      </div>
      <div class="release-summary-strip engineering-only">
        <div class="release-summary-card"><b>Grasp State</b><span id="finalDemoStatus">Checking outputs</span></div>
        <div class="release-summary-card"><b>Quality</b><span id="finalQualityStatus">Waiting</span></div>
        <div class="release-summary-card"><b>Point Cloud</b><span id="finalBenchmarkStatus">Waiting</span></div>
        <div class="release-summary-card"><b>DAVIS</b><span id="finalPresentationStatus">Optional</span></div>
      </div>
      <div class="hand-demo-grid presentation-only" id="handGraspDemoView">
        <article class="hand-demo-panel" id="handDemoRealSensePanel">
          <div class="hand-demo-panel-head">
            <div>
              <h3>RealSense · Grasp Guidance</h3>
              <small>One bottle, one locked band, live approach distance</small>
            </div>
            <span class="hand-demo-panel-badge" id="handDemoRealSenseBadge">WAITING</span>
          </div>
          <div class="hand-demo-visual empty" id="handDemoRealSenseVisual">
            <img id="imgHandDemoRealSense" alt="Live RealSense RGB-D grasp overlay">
            <div class="hand-demo-empty"><div><strong>RealSense overlay unavailable</strong><span>Waiting for the current RGB-D grasp overlay.</span><small>Start RealSense or wait for the next frame.</small></div></div>
          </div>
          <div class="hand-demo-status-grid" id="handDemoRealSenseCards"></div>
        </article>

        <article class="hand-demo-panel" id="handDemoDavisPanel">
          <div class="hand-demo-panel-head">
            <div>
              <h3>DAVIS · Post-Grasp Motion</h3>
              <small>Armed before contact; slip, rotation, and disturbance after grasp</small>
            </div>
            <span class="hand-demo-panel-badge" id="handDemoDavisBadge">OPTIONAL</span>
          </div>
          <div class="hand-demo-visual empty" id="handDemoDavisVisual">
            <img id="imgHandDemoDavis" alt="Live DAVIS event heatmap">
            <div class="hand-demo-empty"><div><strong>DAVIS unavailable</strong><span>No current event heatmap is available.</span><small>Check USB permission, then start the close-up DAVIS monitor.</small></div></div>
          </div>
          <div class="hand-demo-section-label">Sensor Activity</div>
          <div class="hand-demo-status-grid" id="handDemoDavisRawCards"></div>
          <div class="hand-demo-section-label">Grasp-Zone Estimate</div>
          <div class="hand-demo-status-grid" id="handDemoDavisObjectCards"></div>
        </article>

        <article class="hand-demo-panel" id="handDemoQualityPanel">
          <div class="hand-demo-panel-head">
            <div>
              <h3>Live Grasp Quality Matrix</h3>
              <small>RealSense supplies the fixed baseline; gated DAVIS events adjust live stability without recomputing epsilon</small>
            </div>
            <span class="hand-demo-panel-badge">ESTIMATED</span>
          </div>
          <div class="hand-demo-section-label">Planar Wrench Space · Fx / Fy / τz</div>
          <div class="quality-wrench-shell unavailable" id="handDemoWrenchShell" data-status="UNAVAILABLE">
            <canvas id="handDemoWrenchCanvas" aria-label="Analytic wrench hull and live operational quality region"></canvas>
            <span class="quality-wrench-axis fx">+Fx</span>
            <span class="quality-wrench-axis fy">+Fy</span>
            <span class="quality-wrench-axis tau">+τz</span>
            <div class="quality-wrench-legend" aria-hidden="true">
              <span style="--legend-color:#5fa8ff">Analytic hull</span>
              <span style="--legend-color:#41d9d0">Analytic ε region</span>
              <span id="handDemoLiveLegend" style="--legend-color:#36d483">Live operational region</span>
              <span style="--legend-color:#d8f3ff">Analytic weak direction</span>
            </div>
            <div class="quality-wrench-fallback" id="handDemoWrenchFallback">
              <div><strong>Wrench-space quality unavailable</strong><span>No current analytic hull is available.</span><small>Hold exactly one bottle steady; automatic baseline pending.</small></div>
            </div>
          </div>
          <div class="quality-wrench-caption"><strong id="handDemoWrenchStatus">UNAVAILABLE</strong><span id="handDemoWrenchCaption">Waiting for locked RGB-D contact geometry.</span></div>
          <div class="hand-demo-section-label">Initial Analytic Quality</div>
          <div class="hand-demo-status-grid" id="handDemoAnalyticCards"></div>
          <div class="quality-adjustment-flow" aria-label="RealSense baseline plus DAVIS adjustment equals live quality">
            <div class="quality-adjustment-step"><b>RealSense Baseline</b><span id="handDemoMatrixBaseline">N/A</span><small>analytic, fixed</small></div>
            <div class="quality-adjustment-step excluded" id="handDemoMatrixDavisStep"><b>DAVIS Adjustment</b><span id="handDemoMatrixDavis">0.0 points</span><small id="handDemoMatrixDavisReason">excluded until valid</small></div>
            <div class="quality-adjustment-step"><b>Live Quality</b><span id="handDemoMatrixLive">N/A</span><small>operational estimate</small></div>
          </div>
          <div class="hand-demo-section-label">Live Operational Quality</div>
          <div class="hand-demo-status-grid" id="handDemoLiveCards"></div>
          <div class="quality-risk-strip" id="handDemoQualityRisks" aria-label="Current live quality risks"></div>
          <div class="hand-demo-section-label">Decision</div>
          <div class="hand-demo-decision" id="handDemoDecisionCard">
            <strong id="handDemoDecision">WAITING</strong>
            <span id="handDemoDecisionReason">Hold exactly one bottle steady while automatic lock and analytic baseline complete.</span>
          </div>
          <div class="hand-demo-timeline" id="handDemoTimeline"></div>
        </article>
      </div>
      <div class="hero engineering-only">
        <div class="main-overlay operator-visual">
          <div class="visual-tabs engineering-only" role="tablist" aria-label="Main evidence view">
            <button class="visual-tab active" data-visual-tab="overlay" onclick="setVisualTab('overlay')">Grasp Overlay</button>
            <button class="visual-tab" data-visual-tab="rgb" onclick="setVisualTab('rgb')">RGB</button>
            <button class="visual-tab" data-visual-tab="depth" onclick="setVisualTab('depth')">Depth</button>
            <button class="visual-tab" data-visual-tab="quality" onclick="setVisualTab('quality')">Quality</button>
            <button class="visual-tab" data-visual-tab="davis" onclick="setVisualTab('davis')">DAVIS</button>
            <button class="visual-tab" data-visual-tab="pointcloud" onclick="setVisualTab('pointcloud')">Point Cloud</button>
            <button class="visual-tab" data-visual-tab="multiview" onclick="setVisualTab('multiview')">Multi-View</button>
          </div>
          <div class="visual-stage">
            <div class="visual-panel active" data-visual-panel="overlay">
              <img id="imgOverlay" alt="Processed grasp overlay">
              <svg id="davisRoiOnOverlay" class="davis-roi-on-overlay hidden" preserveAspectRatio="xMidYMid meet" aria-label="Calibrated DAVIS event ROI on the RealSense overlay">
                <polygon id="davisRoiOnOverlayPolygon" fill="rgba(0,255,255,.06)" stroke="#00e5ff" stroke-width="2" stroke-dasharray="7 4" vector-effect="non-scaling-stroke"></polygon>
              </svg>
              <div class="visual-empty"><strong>Overlay unavailable</strong><span>Waiting for the processed grasp overlay.</span><small>Start RealSense or wait for the next frame.</small></div>
            </div>
            <div class="visual-panel" data-visual-panel="rgb">
              <img id="imgMainRgb" alt="RealSense RGB frame">
              <div class="visual-empty"><strong>RGB hidden</strong><span>RGB stream is loaded only in this view or Engineering mode.</span><small>Use Engineering mode for raw feeds.</small></div>
            </div>
            <div class="visual-panel" data-visual-panel="depth">
              <img id="imgMainDepth" alt="RealSense aligned depth frame">
              <div class="visual-empty"><strong>Depth hidden</strong><span>Depth stream is loaded only in this view or Engineering mode.</span><small>Use Engineering mode for raw feeds.</small></div>
            </div>
            <div class="visual-panel" data-visual-panel="quality">
              <img id="imgMainQuality" alt="Current quality close-up">
              <div class="visual-empty"><strong>Quality overlay unavailable</strong><span>No current quality graphic is available yet.</span><small>Hold exactly one bottle steady; automatic baseline pending.</small></div>
            </div>
            <div class="visual-panel" data-visual-panel="davis">
              <img id="imgMainDavis" alt="DAVIS motion overlay">
              <div class="visual-empty"><strong>DAVIS heatmap unavailable</strong><span>The event stream is not active or no heatmap is ready.</span><small>Start DAVIS from Advanced Controls.</small></div>
            </div>
            <div class="visual-panel" data-visual-panel="pointcloud">
              <img id="imgMainPointcloud" alt="RealSense point cloud preview">
              <div class="visual-empty"><strong>Point cloud loading</strong><span>No current point-cloud preview is available.</span><small>Use Refresh Point Cloud after RGB-D is active.</small></div>
            </div>
            <div class="visual-panel" data-visual-panel="multiview">
              <img id="imgMainMultiview" alt="Multi-view reconstruction preview">
              <div class="visual-empty"><strong>Multi-view unavailable</strong><span>No reconstruction preview is available yet.</span><small>Start a scan after the bottle is locked.</small></div>
            </div>
          </div>
          <div class="stream-status-panel engineering-only" style="margin-top:10px;">
            <div class="status-group">
              <h3>Main Status</h3>
              <div class="status-list" id="streamMainStatus"></div>
            </div>
            <div class="status-group">
              <h3>Geometry</h3>
              <div class="status-list" id="streamGeometryStatus"></div>
            </div>
            <div class="status-group">
              <h3>Contact</h3>
              <div class="status-list" id="streamContactStatus"></div>
            </div>
            <div class="status-group">
              <h3>Quality</h3>
              <div class="status-list" id="streamQualityStatus"></div>
            </div>
          </div>
        </div>
        <aside class="decision-rail">
          <div class="decision-card" id="overallDecisionCard">
            <div class="decision-label">Overall Decision</div>
            <div class="decision-state" id="decisionState">WAITING</div>
            <div class="decision-explain" id="decisionExplanation">Waiting for RealSense status.</div>
            <div class="next-action" id="decisionNextAction">Start or refresh the demo to populate the operator view.</div>
          </div>
          <div class="metric-rail" id="primaryMetricCards"></div>
          <div class="hand-band-status" id="handBandStatus" aria-label="Locked grasp band status">
            <div class="hand-band-item engineering-only" id="handStageCard"><b>Stage</b><span id="handPrototypeStage">Detect</span></div>
            <div class="hand-band-item" id="lockedBandCard"><b>Locked Band</b><span id="lockedBandStatus">Waiting for lock</span></div>
            <div class="hand-band-item" id="bandDriftCard"><b>Band Drift</b><span id="bandDriftStatus">N/A</span></div>
            <div class="hand-band-item" id="bandRelockCard"><b>Relock</b><span id="bandRelockStatus">Not required</span></div>
          </div>
          <div class="quality-card engineering-only">
            <h3>Sensor Support</h3>
            <div class="evidence-list" id="sensorEvidenceRows"></div>
          </div>
          <div class="quality-card engineering-only">
            <h3>Why This Score?</h3>
            <div class="live-contribution-grid" id="liveQualityContributionCards"></div>
            <div class="quality-timeline" id="liveQualityTimeline"></div>
          </div>
          <div class="quality-card engineering-only">
            <h3>Essential Status</h3>
            <div class="grid status-grid" id="statusCards"></div>
          </div>
        </aside>
      </div>
      <div class="operator-action-bar hand-demo-actions">
        <div class="control-bar">
          <div class="toolbar" aria-label="Operator actions">
            <button class="primary engineering-only" id="freshLockQualityBtn" onclick="freshLockQuality()" title="Optional manual recovery; normal single-bottle locking and baseline creation are automatic.">Fresh Lock + Quality</button>
            <button id="approachPhaseBtn" onclick="markApproachPhase()">Start Approach Mode</button>
            <button id="closeUpDavisBtn" onclick="startCloseUpDavisMonitor()">Start Close-Up DAVIS Monitor</button>
            <button id="graspStateBtn" onclick="toggleGraspDeclaration()" data-post-grasp-active="false" title="I Have Grasped — Monitor Motion is a manual operator declaration only; it does not confirm sensor-measured contact or physical grasp success.">Start Post-Grasp Monitor</button>
            <button onclick="recordSnapshot()">Export Demo Summary</button>
            <button class="danger engineering-only" id="resetHandPrototypeBtn" onclick="resetHandPrototype()" title="Optional engineering recovery; normal target loss uses bounded automatic reacquisition.">Reset Lock</button>
          </div>
        </div>
        <p class="note operator-safety-note">Single-bottle lock, analytic baseline, and approach tracking start automatically after current RGB-D gates pass. Hold one bottle steady and move the hand manually. Post-grasp/release controls only mark the physical phase; they do not confirm physical contact or grasp success.</p>
        <details class="hand-offset-tuning engineering-only" id="handOffsetTuning">
          <summary>Hand Reach Offset (approximate live tuning)</summary>
          <div class="hand-offset-grid">
            <label class="hand-offset-control">X <input id="handOffsetX" type="range" min="-300" max="300" step="1" value="-70" oninput="updateHandOffsetReadout()"><output id="handOffsetXValue">-70 mm</output></label>
            <label class="hand-offset-control">Y <input id="handOffsetY" type="range" min="-300" max="300" step="1" value="80" oninput="updateHandOffsetReadout()"><output id="handOffsetYValue">80 mm</output></label>
            <label class="hand-offset-control">Z <input id="handOffsetZ" type="range" min="0" max="600" step="1" value="170" oninput="updateHandOffsetReadout()"><output id="handOffsetZValue">170 mm</output></label>
            <label class="hand-offset-control">Gap <input id="handContactGap" type="range" min="0" max="30" step="1" value="10" oninput="updateHandOffsetReadout()"><output id="handContactGapValue">10 mm</output></label>
          </div>
          <div class="hand-offset-actions">
            <button onclick="beginHandOffsetCalibration()">Calibrate Hand Offset</button>
            <button onclick="setHandOffsetFromCurrentView()">Set From Current View</button>
            <button class="primary" onclick="saveHandCameraOffset()">Apply Live + Save</button>
            <button onclick="resetHandCameraOffset()">Reset to Demo Defaults</button>
            <span class="hand-offset-honesty" id="handOffsetStatus">Approximate, not physically calibrated</span>
            <span class="hand-offset-honesty" id="handOffsetTarget">Target 180 mm · range 160–220 mm</span>
          </div>
        </details>
        <div class="operator-feedback" aria-live="polite">
          <p class="note" id="controlMessage"></p>
          <p class="note fresh-status engineering-only" id="freshLockQualityMessage"></p>
        </div>
        <details class="advanced-controls engineering-only">
          <summary>Advanced Controls</summary>
          <div class="details-body">
            <div class="toolbar">
              <button class="primary" onclick="startDemo('realsense')">Start RealSense</button>
              <button class="danger" onclick="stopDemo('realsense')">Stop RealSense</button>
              <button class="primary" onclick="davisCommand('start')">Start DAVIS</button>
              <button class="danger" onclick="davisCommand('stop')">Stop DAVIS</button>
              <button onclick="davisCommand('restart')">Restart DAVIS</button>
              <button onclick="davisCommand('calibrate')">Calibrate Event Baseline</button>
              <button onclick="davisCommand('record/start')">Start Trial</button>
              <button onclick="davisCommand('record/stop')">Stop Recording</button>
              <button onclick="davisCommand('snapshot')">Save Snapshot</button>
              <button onclick="window.location='/api/davis/export'">Export DAVIS Report</button>
              <button onclick="davisCommand('history/clear')">Clear Motion History</button>
              <select id="handDemoCaptureStage" aria-label="Demo screenshot stage">
                <option value="operator_start">Operator Start</option>
                <option value="locked_band">Locked Band</option>
                <option value="approach">Approach</option>
                <option value="davis_closeup">DAVIS Close-Up</option>
                <option value="post_grasp">Post-Grasp</option>
              </select>
              <button onclick="captureSelectedHandDemoStage()">Capture Selected Stage</button>
              <button onclick="stopHandDemoCapture()">Stop Screenshot Sharing</button>
              <button onclick="realsenseCommand('lock_grasp')">Lock Grasp</button>
              <span class="hand-offset-honesty">Quality baseline is captured automatically once per lock</span>
              <button class="danger" onclick="realsenseCommand('reset_lock_clear_matrix')">Reset Lock / Clear Matrix</button>
            </div>
            <p class="note">Troubleshooting and optional stage captures only. Screen capture starts only when you press Capture Selected Stage or Export Demo Summary. Export asks for current-tab capture permission and saves the presentation screen plus panel crops when a current lock exists.</p>
          </div>
        </details>
      </div>
    </section>

    <section id="hardwareStatusSection" class="engineering-only workspace-section">
      <div class="section-head">
        <div>
          <h2>Hardware Status</h2>
          <div class="section-kicker">Cameras are selected by stable identity. USB bus/device paths are diagnostics only.</div>
        </div>
      </div>
      <div class="grid status-grid" id="hardwareCards"></div>
    </section>

	    <section id="sensorFeedsSection" class="engineering-only">
	      <div class="section-head">
	        <div>
	          <h2>Live Sensor Feeds</h2>
	          <div class="section-kicker">RealSense RGB, aligned depth, DAVIS events, and RealSense point-cloud output</div>
	        </div>
	      </div>
	      <div class="sensor-grid">
	        <div class="sensor-card">
	          <div class="sensor-title">
	            <h3>RealSense RGB</h3>
	            <span class="sensor-badge" id="rgbSensorState">waiting</span>
	          </div>
	          <img id="imgSensorRgb" alt="RealSense RGB stream">
	          <div class="status-list" id="rgbSensorRows"></div>
	        </div>
	        <div class="sensor-card">
	          <div class="sensor-title">
	            <h3>Aligned Depth</h3>
	            <span class="sensor-badge" id="depthSensorState">waiting</span>
	          </div>
	          <img id="imgSensorDepth" alt="RealSense aligned depth stream">
	          <div class="status-list" id="depthSensorRows"></div>
	        </div>
	        <div class="sensor-card">
	          <div class="sensor-title">
	            <h3>DAVIS Events</h3>
	            <span class="sensor-badge" id="davisSensorState">optional</span>
	          </div>
	          <img id="imgSensorDavis" alt="DAVIS event heatmap stream">
	          <div class="status-list" id="davisSensorRows"></div>
	        </div>
	        <div class="sensor-card pointcloud-card">
	          <div class="sensor-title">
	            <h3>3D Point Cloud</h3>
	            <span class="sensor-badge" id="pointcloudSensorState">waiting</span>
	          </div>
	          <div class="status-list" id="pointcloudRows"></div>
	          <div class="pointcloud-actions">
	            <a class="file-link disabled" id="scenePointcloudLink" href="#" target="_blank" rel="noopener">Scene PLY</a>
	            <a class="file-link disabled" id="bottlePointcloudLink" href="#" target="_blank" rel="noopener">Bottle PLY</a>
	          </div>
	          <p class="note">Point cloud is generated from aligned RealSense RGB-D and displayed here as live status plus downloadable PLY outputs.</p>
	        </div>
	      </div>
	    </section>

	    <section id="pointcloudSection" class="workspace-section">
	      <div class="section-head">
	        <div>
	          <h2>Current RealSense Point Cloud</h2>
	          <div class="section-kicker">Latest current-camera-frame RGB-D cloud; rolling diagnostics and registration-gated multi-view scans are separate.</div>
	        </div>
	        <div class="pointcloud-actions">
	          <button onclick="realsenseCommand('refresh_pointcloud')">Refresh Point Cloud</button>
	          <a class="file-link disabled" id="scenePointcloudLinkMain" href="#" target="_blank" rel="noopener">Download Full PLY</a>
	          <a class="file-link disabled" id="bottlePointcloudLinkMain" href="#" target="_blank" rel="noopener">Download Bottle PLY</a>
	        </div>
	      </div>
	      <div class="pointcloud-layout">
	        <div class="pointcloud-preview-grid">
	          <div class="pointcloud-preview-card pointcloud-viewer-card">
	            <div class="pointcloud-viewer-toolbar">
	              <h3>Interactive Current-Frame RGB-D Point Cloud</h3>
	              <div class="pointcloud-actions">
	                <select id="pointcloudColorMode" onchange="setPointcloudColorMode(this.value)" title="Point cloud display color">
	                  <option value="depth" selected>Depth colors</option>
	                  <option value="rgb">RGB colors</option>
	                  <option value="height">Height colors</option>
	                  <option value="white">White points</option>
	                </select>
	                <label class="toggle-pill"><input type="checkbox" id="pointcloudLayerScene" checked onchange="setPointcloudLayer('scene', this.checked)"> Scene</label>
	                <label class="toggle-pill"><input type="checkbox" id="pointcloudLayerBottle" checked onchange="setPointcloudLayer('bottle', this.checked)"> Bottle</label>
	                <label class="toggle-pill"><input type="checkbox" id="pointcloudLayerContacts" checked onchange="setPointcloudLayer('contacts', this.checked)"> Contacts</label>
	                <button onclick="resetPointcloudView()">Reset View</button>
	              </div>
	            </div>
	            <div class="pointcloud-canvas-shell" id="pointcloudThreeHost">
	              <canvas id="pointcloudThreeCanvas"></canvas>
	              <div class="pointcloud-fallback visible" id="pointcloudThreeFallback">
	                <img id="imgPointcloudPreview" alt="Projected RealSense RGB-D point cloud preview fallback">
	                <span><strong>Point cloud viewer loading</strong><br>Waiting for sampled RGB-D points.<br><small>Use Refresh Point Cloud if RGB-D is active.</small></span>
	              </div>
	              <div class="pointcloud-watermark" id="pointcloudWatermark"></div>
	            </div>
	            <p class="plot-caption" id="pointcloudThreeStatus">Waiting for sampled point-cloud JSON.</p>
	          </div>
	        </div>
	        <div class="pointcloud-side">
	          <div class="quality-card">
	            <h3>Point Cloud Status</h3>
	            <div class="grid status-grid pointcloud-summary-grid" id="pointcloudStatusRows"></div>
	            <p class="note">Current = latest camera frame. Rolling = unregistered temporal diagnostic only, not geometry/contact evidence. Multi-View = a separate registration-gated scan.</p>
	            <p class="note" id="pointcloudMessage">Waiting for RGB-D point cloud status.</p>
	          </div>
	          <details class="pointcloud-diagnostics">
	            <summary>Point Cloud Diagnostics</summary>
	            <div class="details-body">
	              <div class="plot-card pointcloud-preview-card" data-title="Bottle Point Cloud Crop" data-img="imgBottlePointcloudPreview">
	                <h3>Bottle Point Cloud Crop</h3>
	                <img id="imgBottlePointcloudPreview" alt="Projected bottle-only RealSense point cloud preview">
	                <p class="plot-caption">Bottle-only crop derived from the current RealSense bottle bbox/mask support. This is visualization only and does not change tracking or quality math.</p>
	              </div>
	              <div class="grid status-grid" id="pointcloudDiagnosticCards"></div>
	            </div>
	          </details>
	        </div>
	      </div>
	    </section>

	    <section id="multiviewSection" class="workspace-section">
	      <div class="section-head">
	        <div>
	          <h2>Registration-Gated Multi-View Bottle Reconstruction</h2>
	          <div class="section-kicker">Separate scan workspace for accepted RealSense keyframes; registration and coverage determine whether accumulation is usable.</div>
	        </div>
	        <div class="pointcloud-actions">
	          <button onclick="multiviewCommand('start_multiview_scan')">Start Multi-View Scan</button>
	          <button onclick="multiviewCommand('pause_multiview_scan')">Pause Scan</button>
	          <button onclick="multiviewCommand('resume_multiview_scan')">Resume Scan</button>
	          <button onclick="multiviewCommand('reset_multiview_scan')">Reset Scan</button>
	          <button onclick="multiviewCommand('save_multiview_ply')">Save Accumulated PLY</button>
	          <button onclick="multiviewCommand('save_multiview_mesh')">Save Bottle Mesh</button>
	          <button onclick="multiviewCommand('rebuild_multiview_preview')">Rebuild Preview</button>
	          <select id="multiviewDisplayMode" onchange="loadMultiviewModel()" title="Multi-view display mode">
	            <option value="current_accumulated" selected>Current Frame + Accepted Scan</option>
	            <option value="accumulated_bottle">Accepted Bottle Scan</option>
	            <option value="full_scene_map">Accepted Scene Scan</option>
	            <option value="current_frame">Current Frame Only</option>
	            <option value="reconstructed_mesh">Reconstructed Mesh</option>
	          </select>
	          <button onclick="loadMultiviewModel()">Show Accumulated Model</button>
	        </div>
	      </div>
	      <div class="grid status-grid" id="multiviewCards"></div>
	      <div class="section-unavailable" id="multiviewValidationBanner">Multi-view is a scan workspace. Treat reconstruction as not validated until accepted keyframes, coverage, and registration fitness are shown for the current scan.</div>
	      <div class="quality-card">
	        <h3>Reconstruction Guidance</h3>
	        <div class="status-list" id="multiviewRows"></div>
	        <p class="note" id="multiviewMessage">Start a scan after the bottle is locked, then move the camera slowly around the stationary bottle.</p>
	      </div>
	      <details class="pointcloud-diagnostics">
	        <summary>Multi-View Diagnostics</summary>
	        <div class="details-body">
	          <div class="grid status-grid" id="multiviewDiagnosticCards"></div>
	        </div>
	      </details>
	    </section>

	    <section id="objectProfilesSection" class="engineering-only workspace-section">
	      <div class="section-head">
	        <div>
	          <h2>Object Profiles</h2>
	          <div class="section-kicker">Enroll reusable object priors from existing RealSense RGB-D output without changing the production tracker.</div>
	        </div>
	      </div>
	      <div class="object-profile-layout">
	        <div class="object-profile-panel">
	          <h3>Active Target</h3>
	          <div class="grid status-grid" id="objectActiveCards"></div>
	          <div class="object-profile-form">
	            <input id="objectProfileName" value="New Scanned Object" title="Object profile display name">
	            <button class="primary" onclick="scanObjectProfile()">Scan Object</button>
	            <button onclick="saveObjectProfile()">Save Draft</button>
	            <button onclick="setActiveObjectProfile('generic_graspable_object')">Use Generic</button>
	          </div>
	          <p class="note">Object scans reuse current RGB-D, mask/bbox, and point-cloud outputs. They do not train a detector or bypass safety checks.</p>
	          <div class="object-scan-result" id="objectScanResult"></div>
	        </div>
	        <div class="object-profile-panel">
	          <div class="section-head" style="margin-bottom:0;">
	            <div>
	              <h3>Profile Library</h3>
	              <div class="section-kicker">Saved profiles can be active targets when the grasp is idle.</div>
	            </div>
	            <button onclick="enhanceObjectProfileWithMultiview()">Use Multi-View Model</button>
	          </div>
	          <div class="object-profile-library" id="objectProfileLibrary"></div>
	        </div>
	      </div>
	    </section>

	    <section id="syncSensorsSection" class="engineering-only">
	      <div class="section-head">
	        <div>
	          <h2>Synchronized Sensors</h2>
	          <div class="section-kicker">Software timestamp comparison for RealSense RGB-D, point cloud, and DAVIS events</div>
	        </div>
      </div>
      <div class="grid status-grid" id="syncStatusCards"></div>
    </section>

    <section id="qualitySection" class="workspace-section">
      <div class="section-head">
        <div>
          <h2>Quality Matrix</h2>
          <div class="section-kicker">Demo-readable force-closure summary, contact inputs, and current/stale plot status</div>
        </div>
        <div class="embodiment-selector">
          <span>Baseline captured automatically at lock</span>
          <label for="embodimentSelector">Embodiment</label>
          <select id="embodimentSelector" onchange="setEmbodimentFromSelector(this.value)" title="Select contact model for display and benchmarking">
            <option value="parallel_jaw_two_finger">Parallel Two-Finger Gripper</option>
          </select>
          <span id="embodimentSelectorState">parallel jaw reference</span>
        </div>
      </div>
      <div class="quality-layout">
        <div class="quality-card">
          <div class="quality-hero">
            <div>
              <h3>Quality Summary</h3>
              <p class="note" id="qualityTypeNote">Waiting for quality inputs.</p>
            </div>
            <div class="quality-result" id="qualityResult">N/A</div>
          </div>
          <div id="qualitySummaryRows"></div>
          <div class="quality-message" id="qualityAction">Waiting for a locked grasp and quality compute.</div>
        </div>
        <div class="quality-card engineering-only">
          <h3>Contact Inputs</h3>
          <div id="qualityContactRows"></div>
          <div class="quality-message" id="qualityFailureMessage">No failure reason reported.</div>
        </div>
      </div>
      <div class="global-quality-banner" id="qualityGlobalBanner"></div>
      <div class="grid plot-grid quality-plot-grid" style="margin-top:14px;">
        <div class="plot-card hero-plot quality-main" data-title="Bottle Quality Close-Up" data-img="imgQualityCloseup">
          <h3>Bottle Quality Close-Up</h3>
          <img id="imgQualityCloseup" alt="Cropped bottle quality overlay with contact patches and quality radius">
          <p class="plot-caption">Primary demo view: cropped around the locked bottle and contact region. Analytic direction arrows are omitted here because they are not detected bottle motion.</p>
        </div>
        <details class="advanced-plots">
          <summary>Contact Hull Overlay</summary>
          <div class="details-body">
            <div class="grid plot-grid">
              <div class="plot-card primary-plot" data-title="Bottle Quality Context" data-img="imgQualityContext">
                <h3>Bottle Quality Context</h3>
                <img id="imgQualityContext" alt="Full-scene bottle quality context">
                <p class="plot-caption">Full-scene quality overlay showing where the close-up is located.</p>
              </div>
              <div class="plot-card primary-plot" data-title="Contact Hull Overlay" data-img="imgContactHullOverlay">
                <h3>Contact Hull Overlay</h3>
                <img id="imgContactHullOverlay" alt="Bottle mask and contact patch hull overlay">
                <p class="plot-caption">Yellow: locked bottle mask hull. Magenta: side contact patches and image-space contact hull.</p>
              </div>
            </div>
          </div>
        </details>
        <details class="advanced-plots" id="wrenchSpaceDetails">
          <summary>RealSense / DAVIS Confirmation</summary>
          <div class="details-body">
            <div class="grid plot-grid">
              <div class="plot-card primary-plot" data-title="RealSense / DAVIS Confirmation" data-img="imgDavisConfirmation">
                <h3>RealSense / DAVIS Confirmation</h3>
                <img id="imgDavisConfirmation" alt="Side-by-side RealSense quality, DAVIS motion, and DAVIS-adjusted preview">
                <p class="plot-caption">RealSense quality remains the source of truth. DAVIS motion is confirmation/preview only.</p>
              </div>
              <div class="plot-card primary-plot" data-title="DAVIS Motion Overlay" data-img="imgDavisMotionOverlay">
                <h3>DAVIS Motion Overlay</h3>
                <img id="imgDavisMotionOverlay" alt="DAVIS event heatmap, motion bbox, and centroid">
                <p class="plot-caption">DAVIS event heatmap with recent motion bbox and centroid.</p>
              </div>
              <div class="plot-card primary-plot" data-title="DAVIS-Adjusted Quality Preview" data-img="imgDavisAdjustedOverlay">
                <h3>DAVIS-Adjusted Quality Preview</h3>
                <img id="imgDavisAdjustedOverlay" alt="DAVIS adjusted RealSense quality overlay preview">
                <p class="plot-caption">Shown only as a preview when DAVIS motion data and alignment are available.</p>
              </div>
            </div>
          </div>
        </details>
        <details class="advanced-plots">
          <summary>Advanced Wrench-Space Details</summary>
          <div class="details-body">
            <div class="grid plot-grid">
              <div class="plot-card primary-plot" data-title="Advanced Wrench-Space Plot" data-img="imgWrench3D">
                <h3>Advanced Wrench-Space Plot</h3>
                <img id="imgWrench3D" alt="3D grasp wrench-space convex hull">
                <p class="plot-caption">3D wrench-space hull and epsilon margin for the current quality compute.</p>
              </div>
              <div class="plot-card" data-title="Fx / Fy" data-img="imgWrenchFxFy"><h3>Fx / Fy</h3><img id="imgWrenchFxFy" alt="Fx Fy projection"><p class="plot-caption">Planar force projection.</p></div>
              <div class="plot-card" data-title="Fx / Tau" data-img="imgWrenchFxTau"><h3>Fx / Tau</h3><img id="imgWrenchFxTau" alt="Fx Tau projection"><p class="plot-caption">Horizontal force versus torque.</p></div>
              <div class="plot-card hidden" id="fyTauCard" data-title="Fy / Tau" data-img="imgWrenchFyTau"><h3>Fy / Tau</h3><img id="imgWrenchFyTau" alt="Fy Tau projection"><p class="plot-caption">Vertical force versus torque.</p></div>
            </div>
          </div>
        </details>
        <details class="advanced-plots">
          <summary>Quality Diagnostics</summary>
          <div class="details-body">
            <div class="grid plot-grid">
              <div class="plot-card" data-title="Bottle Quality Overlay" data-img="imgWeakDirection">
                <h3>Full Bottle Quality Overlay</h3>
                <img id="imgWeakDirection" alt="Full bottle quality overlay">
                <p class="plot-caption">The red arrow is the analytic weakest grasp-resistance direction, not detected bottle motion.</p>
              </div>
              <div class="plot-card" data-title="Legacy Hull Audit" data-img="imgHullOverlay">
                <h3>Legacy Hull Audit</h3>
                <img id="imgHullOverlay" alt="Legacy grasp hull overlay">
                <p class="plot-caption">Thin audit overlay for checking stored bottle/contact hulls.</p>
              </div>
              <div class="plot-card" data-title="Wrench Hull Debug" data-img="imgWrenchProjection">
                <h3>Wrench Projection Debug</h3>
                <img id="imgWrenchProjection" alt="Legacy wrench-space projection">
                <p class="plot-caption">Legacy projection variant retained for diagnostics.</p>
              </div>
            </div>
          </div>
        </details>
      </div>
    </section>

    <section id="davisMotionSection" class="workspace-section">
      <div class="section-head">
        <div>
          <h2>DAVIS Motion</h2>
          <div class="section-kicker">Post-grasp event-camera safety summary from the dashboard-owned DAVIS backend</div>
        </div>
      </div>
      <div id="davisUnavailableCard" class="section-unavailable hidden">DAVIS is unavailable. The final demo can still run RealSense-only, and the DAVIS tab will show the exact backend state when hardware is connected.</div>
      <p class="note">DAVIS is used only for post-grasp movement monitoring. After the operator declares a grasp, a fresh confirmed DAVIS APS bottle contour selects the event region directly in the DAVIS image. No RealSense-to-DAVIS calibration or cross-camera identity claim is required; the motion estimate remains DAVIS-pixel-relative and not physically calibrated.</p>
      <div class="grid status-grid" id="davisMotionSummaryCards"></div>
      <h3>Raw Event Activity</h3>
      <div class="grid status-grid" id="davisRawActivityCards"></div>
      <h3>Object Motion Estimate</h3>
      <div class="grid status-grid" id="davisObjectMotionCards"></div>
      <h3>Close-Up / Post-Grasp Monitoring</h3>
      <div class="grid status-grid" id="davisCloseupCards"></div>
      <h3>Disturbance / Exclusion</h3>
      <div class="grid status-grid" id="davisDisturbanceCards"></div>
      <div class="davis-visual-grid">
        <div class="plot-card hero-plot" data-title="DAVIS Event Heatmap" data-img="imgDavisMotionHeatmap">
          <h3>Event Heatmap</h3>
          <img id="imgDavisMotionHeatmap" alt="DAVIS event heatmap and motion evidence">
          <p class="plot-caption">Recent event activity, centroid, and motion evidence from the canonical DAVIS backend.</p>
        </div>
        <div class="plot-card" data-title="DAVIS APS Frame" data-img="imgDavisMotionAps">
          <h3>APS Frame</h3>
          <img id="imgDavisMotionAps" alt="DAVIS APS frame">
          <p class="plot-caption">APS channel encoding is reported separately from colour accuracy; DAVIS colour is not calibrated, and RealSense RGB remains the recognition source.</p>
          <p class="plot-caption" id="davisApsVisibility">APS visibility: waiting for frames...</p>
          <label class="plot-caption">Display
            <select id="davisApsDisplayModeSelect" onchange="setDavisApsDisplayMode(this.value)">
              <option value="auto_normalize" selected>Auto Normalize</option>
              <option value="raw">Raw</option>
              <option value="gamma">Gamma</option>
              <option value="clahe">CLAHE</option>
              <option value="false_color">False Color</option>
            </select>
          </label>
        </div>
      </div>
    </section>

    <section id="benchmarkSection" class="workspace-section">
      <div class="section-head">
        <div>
          <h2>Benchmark Lab</h2>
          <div class="section-kicker">Experimental comparison workspace. Production safety stays fully enabled and remains the only robot-control path.</div>
        </div>
      </div>
      <div class="benchmark-banner" id="benchmarkBanner">Benchmark results are experimental and do not control the robot.</div>
      <div class="benchmark-run-panel">
        <div class="benchmark-run-card">
          <div class="benchmark-panel-head">
            <h3>One-Click Run</h3>
            <span class="chip" id="benchmarkRunStateChip">IDLE</span>
          </div>
          <div class="benchmark-run-controls">
            <label>Duration
              <select id="benchmarkDurationSelect" onchange="toggleBenchmarkCustomDuration()">
                <option value="5">5 s</option>
                <option value="10">10 s</option>
                <option value="20" selected>20 s</option>
                <option value="30">30 s</option>
                <option value="custom">Custom</option>
              </select>
            </label>
            <input id="benchmarkCustomDuration" type="number" min="1" max="300" step="1" value="20" title="Custom duration in seconds" style="display:none;width:78px">
          </div>
          <div class="benchmark-run-controls">
            <button class="primary" onclick="runBenchmark('sequential_live')">Run Benchmark</button>
            <button disabled title="Use a verified recording with the offline replay and evaluation tools">Replay requires a recording</button>
            <button onclick="benchmarkRunCommand('pause')">Pause</button>
            <button onclick="benchmarkRunCommand('resume')">Resume</button>
            <button onclick="benchmarkRunCommand('stop')">Stop</button>
          </div>
          <div class="benchmark-run-controls">
            <button onclick="benchmarkRunCommand('skip-current')">Skip Current</button>
            <button onclick="benchmarkRunCommand('repeat-current')">Repeat Current</button>
            <button onclick="window.location='/api/benchmark/download-report?format=html'">Export Results</button>
          </div>
          <p class="note">Sequential Live samples current status. Verified recordings use the separate offline replay and evaluation tools; this runner does not perform fair replay.</p>
        </div>
        <div class="benchmark-run-card">
          <h3>Progress</h3>
          <div class="benchmark-progress-shell"><div class="benchmark-progress-bar" id="benchmarkProgressBar"></div></div>
          <div class="benchmark-production" id="benchmarkRunSummary"></div>
          <div class="quality-timeline" id="benchmarkRunTimeline"></div>
        </div>
        <div class="benchmark-run-card">
          <h3>Precheck</h3>
          <div class="benchmark-precheck-list" id="benchmarkPrecheckList"></div>
          <p class="note" id="benchmarkRunMessage">Ready to run when RealSense, RGB, depth, bottle, target, and dashboard freshness pass.</p>
        </div>
      </div>
      <div class="benchmark-layout">
        <details class="benchmark-panel benchmark-picker">
          <summary>Feature Selection</summary>
          <div class="details-body">
          <div class="benchmark-presets" id="benchmarkPresetButtons"></div>
          <div class="benchmark-feature-groups" id="benchmarkFeaturePicker"></div>
          <div class="toolbar compact-toolbar">
            <button class="primary" onclick="applyBenchmarkCustom()">Use Custom as Card</button>
            <button onclick="resetBenchmarkDefaults()">Default Compare</button>
            <button onclick="clearBenchmarkCards()">Clear Cards</button>
          </div>
          <p class="note" id="benchmarkSelectionReason">Select a preset or custom feature set.</p>
          </div>
        </details>
        <section class="benchmark-panel benchmark-live">
          <div class="benchmark-panel-head">
            <h3>Live Comparison</h3>
            <span class="chip" id="benchmarkTrialChip">No active trial</span>
          </div>
          <div class="benchmark-production" id="benchmarkProductionSummary"></div>
          <div class="benchmark-card-grid" id="benchmarkComparisonCards"></div>
        </section>
        <details class="benchmark-panel benchmark-trial">
          <summary>Ground Truth Controls</summary>
          <div class="details-body">
          <div class="benchmark-label-grid" id="benchmarkLabelButtons"></div>
          <div class="toolbar compact-toolbar">
            <button class="primary" onclick="benchmarkTrialCommand('start')">Start Trial</button>
            <button onclick="benchmarkTrialCommand('stop')">Stop Trial</button>
            <button onclick="benchmarkUndoLabel()">Undo Last Label</button>
          </div>
          <div class="toolbar compact-toolbar">
            <button onclick="window.location='/api/benchmark/export?format=json'">Export Trial JSON</button>
            <button onclick="window.location='/api/benchmark/export?format=csv'">Export Trial CSV</button>
            <button onclick="window.location='/api/benchmark/export?format=html'">Export Benchmark HTML</button>
          </div>
          <p class="note" id="benchmarkTrialMessage">Ground truth is stored separately from predictions.</p>
          </div>
        </details>
      </div>
      <div class="benchmark-bottom-grid">
        <div class="quality-card">
          <h3>Metrics / Accuracy</h3>
          <div id="benchmarkMetricRows"></div>
        </div>
        <div class="quality-card">
          <h3>Feature Contribution</h3>
          <div id="benchmarkContributionRows"></div>
        </div>
        <div class="quality-card">
          <h3>Quality Timeline</h3>
          <div id="benchmarkTimelineRows" class="quality-timeline"></div>
        </div>
      </div>
      <div class="benchmark-bottom-grid">
        <div class="quality-card">
          <h3>Generated Results</h3>
          <div id="benchmarkGeneratedResults"></div>
        </div>
        <div class="quality-card">
          <h3>Benchmark Winners</h3>
          <div id="benchmarkWinnerRows"></div>
        </div>
        <div class="quality-card">
          <h3>Ranking</h3>
          <div id="benchmarkRankingRows"></div>
        </div>
        <div class="quality-card">
          <h3>Findings</h3>
          <div id="benchmarkFindingRows"></div>
        </div>
      </div>
      <p class="note" id="benchmarkMessage">Benchmark Lab is display/report-only. Production decisions are unaffected.</p>
    </section>

    <section id="embodimentSection" class="workspace-section">
      <div class="section-head">
        <div>
          <h2>Embodiment</h2>
          <div class="section-kicker">Active contact model, release state, and metric applicability</div>
        </div>
      </div>
      <div class="grid status-grid" id="embodimentCards"></div>
    </section>

    <details id="debugViews" class="engineering-only">
      <summary>Debug Views</summary>
      <div class="details-body">
        <div class="grid debug-media">
          <div class="panel"><h3>Raw RGB</h3><img id="imgRgb" alt="Raw RGB stream"></div>
          <div class="panel"><h3>Depth</h3><img id="imgDepth" alt="Depth stream"></div>
        </div>
      </div>
    </details>

    <details id="diagnosticsDetails" class="engineering-only">
      <summary>Diagnostics</summary>
      <div class="details-body">
        <h3>RealSense / File Status</h3>
        <div class="grid" id="diagnosticCards"></div>
        <h3>DAVIS Motion Details</h3>
        <div class="grid" id="davisCards"></div>
        <div class="grid">
          <div class="panel"><h3>DAVIS APS Frame</h3><img id="imgDavisAps"><p class="note">Latest APS frame from the same dashboard-owned DAVIS backend.</p></div>
          <div class="panel"><h3>Recent Event Motion Hull</h3><img id="imgDavis"><p class="note">DAVIS hull from recent event-based motion around the grasped bottle region.</p></div>
          <div class="panel"><h3>Lightweight Event Metrics</h3><pre id="davisNumbers">DAVIS is a post-grasp event feedback sensor. It uses a mapped RealSense ROI and a short recent event window.</pre></div>
        </div>
        <h3>Performance Profile</h3>
        <div class="grid" id="profileCards"></div>
        <h3>Point Cloud Diagnostics</h3>
        <div class="grid">
          <div class="panel plot-card" data-title="RGB-D + Point Cloud Context" data-img="imgPointcloudContext">
            <h3>RGB-D + Point Cloud Context</h3>
            <img id="imgPointcloudContext" alt="RealSense RGB depth and point cloud context preview">
            <p class="note">Diagnostic composite: RGB, aligned depth, current scene cloud, and bottle crop. Hidden from the presentation view.</p>
          </div>
        </div>
        <h3>Latest-Frame Stream Health</h3>
        <div class="grid" id="streamCards"></div>
        <h3>Logs</h3>
        <div class="grid">
          <div class="panel"><h3>RealSense</h3><pre id="logRs"></pre></div>
        </div>
      </div>
    </details>
  </main>
  <div class="modal" id="imageModal" role="dialog" aria-modal="true" aria-label="Expanded plot">
    <div class="modal-card">
      <div class="modal-head">
        <h2 id="modalTitle">Plot</h2>
        <div class="modal-actions">
          <a class="file-link" id="modalDownload" href="#" download>Download Snapshot</a>
          <button onclick="closeModal()">Close</button>
        </div>
      </div>
      <img id="modalImg" alt="Expanded plot">
    </div>
  </div>
	  <script src="/vendor/three.min.js"></script>
	  <script>
	    const DASHBOARD_POLL_INTERVAL_MS = __DASHBOARD_POLL_INTERVAL_MS__;
	    const SENSOR_FEEDS_ALWAYS_ON = false;
	    const POINTCLOUD_JSON_REFRESH_MS = 1000;
	    const OPERATOR_UI_DISPLAY_POLICY = {
	      overall_decision: {label: 'Overall Decision', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'not_ready_or_unsafe'},
	      next_action: {label: 'Next Action', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'action_required'},
	      target_status: {label: 'Target Status', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'lost'},
	      object_pose: {label: 'Object Pose', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'unsupported_or_unknown'},
	      distance: {label: 'Distance', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'out_of_range'},
	      move_guidance: {label: 'Move X/Y/Z', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'invalid_depth'},
	      grasp_feasibility: {label: 'Grasp Feasibility', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'cannot_grasp'},
	      clutter_warning: {label: 'Clutter', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'blocked_or_ambiguous'},
	      grip_width: {label: 'Grip Width', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'missing'},
	      analytic_quality: {label: 'Analytic Quality', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'stale_or_invalid'},
	      quality_applicability: {label: 'Quality Applicability', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'not_applicable'},
	      live_quality: {label: 'Live Quality', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'unsafe'},
	      davis_state: {label: 'DAVIS State', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'unsafe_motion'},
	      pointcloud_state: {label: 'Point Cloud State', show_in_presentation: true, show_in_engineering: true, show_in_diagnostics: false, show_on_operator: true, critical_if: 'stale_contact_geometry'},
	      raw_backend_fields: {label: 'Backend Internals', show_in_presentation: false, show_in_engineering: true, show_in_diagnostics: true, show_on_operator: false, critical_if: 'writer_conflict'},
	      evaluator_trace: {label: 'Benchmark Evaluator Trace', show_in_presentation: false, show_in_engineering: true, show_in_diagnostics: true, show_on_operator: false, show_on_benchmark: true, critical_if: 'feature_mask_failure'},
	    };
	    const THREE_JS_CDN_URLS = [
	      '/vendor/three.min.js',
	      'https://unpkg.com/three@0.149.0/build/three.min.js',
	      'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
	    ];
	    let latestImages = {};
	    let latestOperatorStatus = {};
	    let operatorApproachPhaseMarked = false;
	    let debugViewsOpen = false;
	    const urlParams = new URLSearchParams(window.location.search);
	    let dashboardMode = urlParams.get('mode') || localStorage.getItem('dashboardMode') || 'presentation';
	    let activeVisualTab = urlParams.get('tab') || localStorage.getItem('activeVisualTab') || 'overlay';
	    let benchmarkCustomFeatures = new Set();
	    let benchmarkCurrentConfigurations = [];
	    let refreshInFlight = false;
	    let benchmarkRefreshInFlight = false;
	    let lastBenchmarkRefreshMs = 0;
	    const BENCHMARK_POLL_INTERVAL_MS = 3000;
	    let lastStaticImageRefreshMs = 0;
	    const STATIC_IMAGE_REFRESH_MS = 2000;
	    let refreshTick = 0;
	    const DIAGNOSTICS_REFRESH_DIVIDER = 4;
	    let threeLoadPromise = null;
	    const pointcloudViewer = {
	      initialized: false,
	      scene: null,
	      camera: null,
	      renderer: null,
	      group: null,
	      axes: null,
	      scenePoints: null,
	      bottlePoints: null,
	      contactMarkers: null,
	      lastTimestamp: null,
	      lastFetchMs: 0,
	      rotationX: -0.18,
	      rotationY: 0.52,
	      distance: 1.2,
	      viewFitted: false,
	      colorMode: 'depth',
	      showScene: true,
	      showBottle: true,
	      showContacts: true,
	      boundsCenter: [0, 0, 0],
	      dragging: false,
	      lastX: 0,
	      lastY: 0,
	    };
    const qualityWrenchViewer = {
      initialized: false,
      renderer: null,
      scene: null,
      camera: null,
      group: null,
      liveSphere: null,
      geometryKey: null,
      dragging: false,
      lastX: 0,
      lastY: 0,
      updating: false,
    };
    function currentSectionId() {
      const hash = decodeURIComponent(window.location.hash || '').replace(/^#/, '');
      return hash || 'operatorSection';
    }
    function isSectionActive(id) {
      return currentSectionId() === id || (!window.location.hash && id === 'operatorSection');
    }
    const PRIMARY_TAB_SECTIONS = ['operatorSection','davisMotionSection','qualitySection','pointcloudSection','multiviewSection','benchmarkSection','embodimentSection','diagnosticsDetails'];
    // Engineering-only detail sections that are not their own nav tab: show them
    // grouped under a primary tab instead of stacking them onto the Operator view.
    const EXTRA_SECTION_TAB = {hardwareStatusSection:'diagnosticsDetails', sensorFeedsSection:'diagnosticsDetails', syncSensorsSection:'diagnosticsDetails', debugViews:'diagnosticsDetails', objectProfilesSection:'embodimentSection'};
    function applyActiveSection() {
      let active = currentSectionId();
      if (!PRIMARY_TAB_SECTIONS.includes(active)) active = 'operatorSection';
      // Development/configuration workspaces stay out of the operator presentation.
      const engineeringOnlySections = ['multiviewSection','benchmarkSection','embodimentSection','diagnosticsDetails'];
      if (dashboardMode !== 'engineering' && engineeringOnlySections.includes(active)) active = 'operatorSection';
      PRIMARY_TAB_SECTIONS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = (id === active) ? '' : 'none';
      });
      Object.entries(EXTRA_SECTION_TAB).forEach(([id, parent]) => {
        const el = document.getElementById(id);
        if (el) el.style.display = (parent === active) ? '' : 'none';
      });
      document.querySelectorAll('.release-nav a').forEach(a => {
        const target = (a.getAttribute('href') || '').replace('#', '');
        a.classList.toggle('active', target === active);
      });
      syncDebugStreams();
    }
    function operatorVisualActive(tab) {
      return isSectionActive('operatorSection') && activeVisualTab === tab;
    }
    function heavyPanelVisible(name) {
      if (name === 'pointcloud') return isSectionActive('pointcloudSection') || operatorVisualActive('pointcloud');
      if (name === 'multiview') return isSectionActive('multiviewSection') || operatorVisualActive('multiview');
      if (name === 'davis') return isSectionActive('davisMotionSection') || operatorVisualActive('davis');
      if (name === 'quality') return isSectionActive('qualitySection') || operatorVisualActive('quality');
      if (name === 'benchmark') return isSectionActive('benchmarkSection');
      if (name === 'diagnostics') return dashboardMode === 'engineering' && isSectionActive('diagnosticsDetails');
      return false;
    }
    function setDashboardMode(mode) {
      dashboardMode = mode === 'engineering' ? 'engineering' : 'presentation';
      localStorage.setItem('dashboardMode', dashboardMode);
      document.body.classList.toggle('engineering', dashboardMode === 'engineering');
      document.body.classList.toggle('presentation', dashboardMode !== 'engineering');
      document.getElementById('presentationModeBtn')?.classList.toggle('active', dashboardMode !== 'engineering');
      document.getElementById('engineeringModeBtn')?.classList.toggle('active', dashboardMode === 'engineering');
      applyActiveSection();
    }
    function setVisualTab(tab) {
      activeVisualTab = tab || 'overlay';
      localStorage.setItem('activeVisualTab', activeVisualTab);
      document.querySelectorAll('.visual-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.visualTab === activeVisualTab);
      });
      document.querySelectorAll('.visual-panel').forEach(panel => {
        panel.classList.toggle('active', panel.dataset.visualPanel === activeVisualTab);
      });
      syncDebugStreams();
      setMainVisualImages({images: latestImages, final_files: {}});
    }
    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    }
    function showToast(message, kind='info') {
      const region = document.getElementById('toastRegion');
      if (!region || !message) return;
      const el = document.createElement('div');
      el.className = `toast ${kind}`;
      el.textContent = message;
      region.appendChild(el);
      window.setTimeout(() => el.remove(), 4200);
    }
    function markVisualAvailability(id, info) {
      const img = document.getElementById(id);
      if (!img) return;
      const available = Boolean(info && info.exists);
      const panel = img.closest('.visual-panel');
      if (panel) panel.classList.toggle('empty', !available);
      const card = img.closest('.plot-card');
      if (card) {
        card.classList.toggle('empty', !available);
        let empty = card.querySelector('.plot-empty-state');
        if (!empty) {
          empty = document.createElement('div');
          empty.className = 'plot-empty-state';
          img.insertAdjacentElement('afterend', empty);
        }
        const heading = card.querySelector('h3')?.textContent || card.dataset.title || 'Visual';
        const action = card.dataset.emptyAction || 'Refresh the relevant sensor or run the required compute step.';
        const reason = card.dataset.emptyReason || 'No current image is available for this view.';
        empty.innerHTML = `<div><strong>${esc(heading)} unavailable</strong><span>${esc(reason)}</span><small>${esc(action)}</small></div>`;
      }
    }
    function stateClass(v) {
      const s = String(v ?? '').toLowerCase();
      if (['computing','running','searching'].some(x => s.includes(x))) return 'info';
      if (['lost','unavailable','sensor_unavailable','do_not','cannot','blocked','not ready','not detected','fail','failed','stale','missing','invalid','unsafe','unstable','recompute required'].some(x => s.includes(x))) return 'bad';
      if (['true','ready_to_close','ready_to_lift','tracking','locked','detected','stable','initial_good','stable_after_grasp','pass'].some(x => s.includes(x))) return 'ok';
      return 'warn';
    }
    function cards(el, items) {
      if (!el) return;
      const html = items.map(([k,v]) => `<div class="card ${stateClass(v)}"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
      if (el.dataset.lastHtml !== html) {
        el.innerHTML = html;
        el.dataset.lastHtml = html;
      }
    }
    function statusRows(el, items) {
      if (!el) return;
      const html = items.map(([k, v]) => `<div><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
      if (el.dataset.lastHtml !== html) {
        el.innerHTML = html;
        el.dataset.lastHtml = html;
      }
    }
	    function setChip(id, label, value) {
	      const el = document.getElementById(id);
	      if (!el) return;
	      el.textContent = `${label}: ${value ?? 'N/A'}`;
	      el.classList.remove('ok', 'warn', 'bad', 'info');
	      el.classList.add('chip', stateClass(value));
	    }
	    function setSensorBadge(id, value) {
	      const el = document.getElementById(id);
	      if (!el) return;
	      el.textContent = value ?? 'N/A';
	      el.classList.remove('ok', 'warn', 'bad', 'info');
	      el.classList.add('sensor-badge', stateClass(value));
	    }
	    function setPointcloudLink(id, href, enabled) {
	      const el = document.getElementById(id);
	      if (!el) return;
	      if (enabled && href) {
	        el.href = href;
	        el.classList.remove('disabled');
	      } else {
	        el.href = '#';
	        el.classList.add('disabled');
	      }
	    }
    function compactBool(v) {
      if (v === true) return 'READY';
      if (v === false) return 'NOT READY';
      return v ?? 'N/A';
    }
    function fmtScore(v) {
      return v !== undefined && v !== null ? `${Math.round(Number(v))} / 100` : 'N/A';
    }
    function fmtNumber(v, digits=3) {
      if (v === undefined || v === null || Number.isNaN(Number(v))) return 'N/A';
      return Number(v).toFixed(digits);
    }
    function fmtFps(v) {
      if (v === undefined || v === null || Number.isNaN(Number(v))) return 'N/A';
      return `${Number(v).toFixed(1)} FPS`;
    }
    function davisHeadline(dm) {
      // Honest one-word DAVIS state for chips/pills: only show an object-motion
      // class (SLIP/ROTATION/...) when object motion is actually valid; when it
      // is global/event activity, show the activity label so event health is
      // never presented as object slip.
      dm = dm || {};
      const stage = String(dm.current_stage || dm.davis_motion_stage || 'PRE_CLOSE').toUpperCase();
      if (stage === 'PRE_CLOSE') return 'ARMED';
      const cls = dm.motion_class;
      if (dm.object_motion_valid === true) return cls || 'OBJECT MOTION';
      if (cls && /LIGHTING|SHAKE|BACKGROUND|STABLE|UNAVAILABLE|N\/A/i.test(cls)) return cls;
      return String(dm.motion_activity_label || 'EVENT ACTIVITY').replace(/_/g, ' ');
    }
    function fmtDepthPair(depths, rs) {
      const left = depths.left !== undefined ? Math.round(depths.left) : (rs.left_finger_depth_mm ? Math.round(rs.left_finger_depth_mm) : null);
      const right = depths.right !== undefined ? Math.round(depths.right) : (rs.right_finger_depth_mm ? Math.round(rs.right_finger_depth_mm) : null);
      if (left === null && right === null) return 'N/A';
      return `${left ?? 'N/A'} / ${right ?? 'N/A'} mm`;
    }
    function yesNo(v) {
      if (v === true) return 'yes';
      if (v === false) return 'no';
      return 'N/A';
    }
    function fmtContactConfidence(qm) {
      const left = qm.contact_confidence_left ?? qm.left_finger_confidence;
      const right = qm.contact_confidence_right ?? qm.right_finger_confidence;
      if (left === undefined && right === undefined) return 'N/A';
      return `${left !== undefined && left !== null ? Math.round(Number(left)) : 'N/A'} / ${right !== undefined && right !== null ? Math.round(Number(right)) : 'N/A'}`;
    }
    function qualityFailureReason(qm) {
      const reasons = qm.quality_matrix_failure_reasons || qm.contact_3d_failure_reasons || [];
      if (Array.isArray(reasons) && reasons.length) return reasons.join(', ');
      return qm.quality_matrix_error || qm.hull_failure_reason || qm.contact_source_failure_reason || 'none';
    }
    function humanReason(reason) {
      if (!reason) return 'N/A';
      const text = Array.isArray(reason) ? reason.join(', ') : String(reason);
      return text
        .replaceAll('_', ' ')
        .replace('center shift exceeds threshold', 'Bottle moved from locked quality pose')
        .replace('current vs locked iou below threshold', 'Bottle pose changed from locked quality pose')
        .replace('depth delta exceeds threshold', 'Bottle depth changed from locked quality pose')
        .replace('writer session mismatch', 'Previous backend session')
        .replace('lock frame mismatch', 'Previous lock frame')
        .replace(/\b\w/g, c => c.toUpperCase());
    }
    function fmtMm(v) {
      if (v === undefined || v === null || Number.isNaN(Number(v))) return 'N/A';
      return `${Math.round(Number(v))} mm`;
    }
    function fmtCloseDistance(rs) {
      const parts = [fmtMm(rs.bottle_distance_mm)];
      if (rs.distance_status) parts.push(humanReason(rs.distance_status));
      if (rs.desired_close_distance_mm !== undefined && rs.desired_close_distance_mm !== null) {
        parts.push(`target ${fmtMm(rs.desired_close_distance_mm)}`);
      }
      if (rs.min_close_distance_mm !== undefined && rs.min_close_distance_mm !== null
          && rs.max_close_distance_mm !== undefined && rs.max_close_distance_mm !== null) {
        parts.push(`range ${fmtMm(rs.min_close_distance_mm)}–${fmtMm(rs.max_close_distance_mm)}`);
      }
      const validation = rs.close_distance_range_validation_status
        || (rs.close_distance_range_physically_validated === false ? 'NOT_PHYSICALLY_VALIDATED' : null);
      if (validation) parts.push(`range ${humanReason(validation)}`);
      return parts.join(' · ');
    }
    function fmtOperatorDistance(rs) {
      const parts = [fmtMm(rs.bottle_distance_mm)];
      const hasRange = rs.min_close_distance_mm !== undefined && rs.min_close_distance_mm !== null
        && rs.max_close_distance_mm !== undefined && rs.max_close_distance_mm !== null;
      if (hasRange) {
        parts.push(`goal ${fmtMm(rs.min_close_distance_mm)}–${fmtMm(rs.max_close_distance_mm)}`);
      } else if (rs.desired_close_distance_mm !== undefined && rs.desired_close_distance_mm !== null) {
        parts.push(`goal ${fmtMm(rs.desired_close_distance_mm)}`);
      }
      const validation = String(rs.close_distance_range_validation_status || '').toUpperCase();
      if (rs.close_distance_range_physically_validated === false || validation.includes('NOT_PHYSICALLY_VALIDATED')) {
        parts.push('uncalibrated range');
      }
      return parts.join(' · ');
    }
    function fmtDepthRoiProjection(rs) {
      const scale = rs.roi_depth_projection_scale;
      const visible = rs.roi_projected_visible_fraction;
      const partial = rs.roi_projected_extent_partial;
      if ((scale === undefined || scale === null)
          && (visible === undefined || visible === null)
          && (partial === undefined || partial === null)) return 'N/A';
      const parts = ['Estimated RGB-D'];
      if (scale !== undefined && scale !== null) parts.push(`${Number(scale).toFixed(2)}× scale`);
      if (visible !== undefined && visible !== null) parts.push(`${Math.round(Number(visible) * 100)}% visible`);
      if (partial !== undefined && partial !== null) parts.push(partial ? 'partial extent' : 'full extent');
      return parts.join(' · ');
    }
    function fmtFingerBottleDepth(rs) {
      const values = [
        rs.left_finger_depth_mm, rs.right_finger_depth_mm,
        rs.left_finger_depth_delta_to_bottle_mm, rs.right_finger_depth_delta_to_bottle_mm,
        rs.finger_bottle_depth_reference_mm,
      ];
      if (!values.some(v => v !== undefined && v !== null)) return 'N/A';
      const left = rs.left_finger_depth_mm !== undefined && rs.left_finger_depth_mm !== null
        ? Math.round(Number(rs.left_finger_depth_mm)) : 'N/A';
      const right = rs.right_finger_depth_mm !== undefined && rs.right_finger_depth_mm !== null
        ? Math.round(Number(rs.right_finger_depth_mm)) : 'N/A';
      const leftDelta = rs.left_finger_depth_delta_to_bottle_mm;
      const rightDelta = rs.right_finger_depth_delta_to_bottle_mm;
      const parts = ['ESTIMATED RGB-D NEAR CONTACT', `L/R ${left} / ${right} mm`];
      if (rs.finger_bottle_depth_reference_mm !== undefined && rs.finger_bottle_depth_reference_mm !== null) {
        parts.push(`bottle ref ${Math.round(Number(rs.finger_bottle_depth_reference_mm))} mm`);
      }
      if ((leftDelta !== undefined && leftDelta !== null) || (rightDelta !== undefined && rightDelta !== null)) {
        const tolerance = rs.finger_bottle_depth_tolerance_mm;
        parts.push(`Δ ${leftDelta !== undefined && leftDelta !== null ? Math.round(Number(leftDelta)) : 'N/A'} / ${rightDelta !== undefined && rightDelta !== null ? Math.round(Number(rightDelta)) : 'N/A'} mm${tolerance !== undefined && tolerance !== null ? ` (≤ ${Math.round(Number(tolerance))} mm)` : ''}`);
      }
      parts.push(`depth match ${yesNo(rs.left_finger_depth_verified)} / ${yesNo(rs.right_finger_depth_verified)}`);
      parts.push(humanReason(rs.finger_validation_status || 'NOT_PHYSICALLY_VALIDATED'));
      return parts.join(' · ');
    }
    function fmtSkinOcclusion(rs) {
      if (rs.roi_candidate_hand_occlusion === undefined || rs.roi_candidate_hand_occlusion === null) return 'N/A';
      const state = rs.roi_candidate_hand_occlusion_state || (
        rs.roi_candidate_hand_occlusion
          ? 'POSSIBLE_HAND_SKIN_OCCLUSION_ESTIMATED'
          : 'NO_HAND_SKIN_OCCLUSION_OBSERVED_ESTIMATED'
      );
      const parts = [humanReason(state)];
      if (rs.roi_candidate_skin_overlap_seed_pixels !== undefined && rs.roi_candidate_skin_overlap_seed_pixels !== null) {
        parts.push(`${Math.round(Number(rs.roi_candidate_skin_overlap_seed_pixels))} seed px`);
      }
      parts.push(humanReason(rs.roi_candidate_hand_occlusion_validation_status || 'NOT_PHYSICALLY_VALIDATED'));
      return parts.join(' · ');
    }
    function fmtMmVector(v) {
      if (Array.isArray(v)) return v.map(x => Number(x).toFixed(1)).join(', ') + ' mm';
      if (v && typeof v === 'object') {
        return Object.entries(v).map(([k, x]) => `${k}: ${Number(x).toFixed(1)} mm`).join(', ');
      }
      return 'N/A';
    }
    function displaySource(source) {
      const s = String(source || '').toLowerCase();
      if (s === 'planned_contact' || s === 'planned') return 'Planned Preview';
      if (s === 'detected_finger_contacts') return 'Detected Finger Contacts';
      if (s === 'detected_gripper_contacts' || s === 'detected_gripper') return 'Detected Gripper Contacts';
      if (s === 'detected_finger') return 'Detected Fingers';
      if (s === 'detected_gripper') return 'Detected Gripper';
      if (!source) return 'N/A';
      return String(source).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
    function qualityDisplayLabel(qm) {
      const raw = String(qm.quality_label || qm.quality_label_display || 'N/A').replace(/\s*\(STALE\)\s*/i, '').toUpperCase();
      if (qm.quality_label_stale || qm.matrix_stale_due_to_motion) return 'BASELINE INVALID - RESET / NEW LOCK';
      return raw;
    }
    function qualityFriendlyLabel(label) {
      const s = String(label || '').toUpperCase();
      if (s.includes('STALE')) return 'Previous compute';
      if (s.includes('GOOD')) return 'Strong grasp margin';
      if (s.includes('OK')) return 'Acceptable grasp margin';
      if (s.includes('WEAK')) return 'Low grasp margin';
      if (s.includes('INVALID')) return 'Missing required inputs';
      if (s.includes('COMPUTING')) return 'Computing';
      return 'Waiting';
    }
    function qualityResultClass(label) {
      const s = String(label || '').toLowerCase();
      if (s.includes('stale')) return 'stale';
      if (s.includes('good')) return 'good';
      if (s.includes('ok')) return 'ok';
      if (s.includes('weak')) return 'weak';
      if (s.includes('invalid') || s.includes('failed')) return 'invalid';
      return '';
    }
    function qualityRows(el, rows) {
      el.innerHTML = rows.map(([k, v]) => `<div class="quality-row"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
    }
    function esc(v) {
      return String(v ?? 'N/A').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }
    function renderBenchmarkFeaturePicker(benchmark) {
      const picker = document.getElementById('benchmarkFeaturePicker');
      const presetHost = document.getElementById('benchmarkPresetButtons');
      if (!picker || !presetHost) return;
      const registry = benchmark.registry || {};
      const categories = registry.categories || [];
      const byCategory = registry.features_by_category || {};
      const firstConfig = (benchmark.configurations || [])[0] || {};
      if (!benchmarkCustomFeatures.size) {
        benchmarkCustomFeatures = new Set(firstConfig.feature_ids || []);
      }
      presetHost.innerHTML = (registry.presets || [])
        .filter(p => p.preset_id !== 'custom')
        .map(p => `<button onclick="addBenchmarkPreset('${esc(p.preset_id)}')" title="Add ${esc(p.display_name)} comparison card">${esc(p.display_name)}</button>`)
        .join('');
      picker.innerHTML = categories.map(category => {
        const features = byCategory[category] || [];
        const checks = features.map(feature => {
          const checked = benchmarkCustomFeatures.has(feature.feature_id) ? 'checked' : '';
          return `<label class="benchmark-check"><input class="benchmark-feature-checkbox" type="checkbox" value="${esc(feature.feature_id)}" ${checked} onchange="benchmarkCustomFeatures = new Set(selectedBenchmarkCustomFeatures())"><span>${esc(feature.display_name)}<small>${esc(feature.limitation || feature.description || '')}</small></span></label>`;
        }).join('');
        return `<div class="benchmark-feature-group"><h4>${esc(category)}</h4>${checks}</div>`;
      }).join('');
    }
    function renderBenchmarkCards(benchmark) {
      const host = document.getElementById('benchmarkComparisonCards');
      if (!host) return;
      const cardsData = (benchmark.comparison_cards || []).slice(0, 4);
      if (!cardsData.length) {
        host.innerHTML = '<div class="benchmark-card"><h4>No comparison cards</h4><div class="benchmark-limitation">Add a preset or custom feature set.</div></div>';
        return;
      }
      host.innerHTML = cardsData.map(card => {
        const used = card.used_features || [];
        const attempted = card.attempted_features || [];
        const disabledCount = (card.disabled_features || []).length;
        const trace = card.evaluator_trace || [];
        const compactList = items => {
          const values = (items || []).slice(0, 3).map(item => humanReason(item));
          const suffix = (items || []).length > 3 ? ` +${(items || []).length - 3}` : '';
          return values.length ? `${values.join(', ')}${suffix}` : 'none';
        };
        const enabledLabels = card.enabled_feature_labels || card.enabled_features || [];
        const geometryResult = card.geometry_validity || card.geometry_status || card.target_status || 'N/A';
        const validity = card.result_validity || (card.feature_selection_valid ? 'VALID' : 'DEPENDENCIES MISSING');
        const traceRows = trace.slice(0, 18).map(row =>
          `<div class="benchmark-trace-row"><b>${esc(row.requested_feature)}</b><span>${esc(row.status)} - ${esc(row.reason)}<small>${esc(row.source_field || '')}</small></span></div>`
        ).join('');
        return `<article class="benchmark-card ${stateClass(card.decision)}">
          <h4>${esc(card.display_name)}</h4>
          <div class="feature-count">${esc(validity)} | ${(card.enabled_features || []).length} enabled | ${used.length} used | ${disabledCount} disabled</div>
          <div class="benchmark-kv"><b>Enabled</b><span>${esc(compactList(enabledLabels))}</span></div>
          <div class="benchmark-kv"><b>Used</b><span>${esc(compactList(used))}</span></div>
          <div class="benchmark-kv"><b>Detection</b><span>${esc(card.detection_status)}</span></div>
          <div class="benchmark-kv"><b>Geometry</b><span>${esc(geometryResult)}</span></div>
          <div class="benchmark-kv"><b>Decision</b><span>${esc(card.decision)}</span></div>
          <div class="benchmark-kv"><b>Latency</b><span>${card.latency_ms !== null && card.latency_ms !== undefined ? `${Number(card.latency_ms).toFixed(1)} ms` : 'N/A'}</span></div>
          <div class="benchmark-limitation">${esc(card.main_limitation || 'No limitation reported.')}</div>
          <details class="benchmark-trace engineering-only"><summary>Evaluator Trace (${attempted.length})</summary>${traceRows || '<p class="note">No trace rows.</p>'}</details>
        </article>`;
      }).join('');
    }
    function toggleBenchmarkCustomDuration() {
      const select = document.getElementById('benchmarkDurationSelect');
      const custom = document.getElementById('benchmarkCustomDuration');
      if (!select || !custom) return;
      custom.style.display = select.value === 'custom' ? '' : 'none';
    }
    function benchmarkDurationSec() {
      const select = document.getElementById('benchmarkDurationSelect');
      const custom = document.getElementById('benchmarkCustomDuration');
      const value = select && select.value === 'custom' ? custom?.value : select?.value;
      const n = Number(value || 20);
      return Math.max(1, Math.min(300, Number.isFinite(n) ? n : 20));
    }
    function renderBenchmarkRun(benchmark) {
      const run = benchmark.run || {};
      const chip = document.getElementById('benchmarkRunStateChip');
      if (chip) {
        chip.textContent = run.state || 'IDLE';
        chip.className = `chip ${stateClass(run.state)}`;
      }
      const bar = document.getElementById('benchmarkProgressBar');
      if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(run.progress_percent || 0)))}%`;
      const summary = document.getElementById('benchmarkRunSummary');
      const summaryData = run.summary || {};
      const winners = summaryData.winner_cards || {};
      if (summary) {
        summary.innerHTML = [
          ['Run', run.benchmark_run_id ? String(run.benchmark_run_id).slice(0, 28) : 'not started'],
          ['Result Type', summaryData.metric_wording || 'Estimated Performance'],
          ['Trial Quality', summaryData.trial_quality || 'N/A'],
          ['Best Overall', (winners.best_overall || {}).display_name || 'N/A'],
          ['Configuration', run.current_config_name || 'N/A'],
          ['Progress', `${run.current_config_index || 0} / ${run.total_configs || 0}`],
          ['Remaining', run.config_remaining_sec !== undefined ? `${run.config_remaining_sec}s` : 'N/A'],
          ['Mode', run.run_mode || 'sequential_live'],
          ['Scoring', summaryData.ranking_mode_label || 'Safety First'],
          ['Sample Rate', run.benchmark_sample_rate_hz ? `${run.benchmark_sample_rate_hz} Hz` : 'N/A'],
          ['Eval / Log', `${run.benchmark_eval_ms || 0} / ${run.benchmark_log_ms || 0} ms`],
          ['Results', run.results_ready ? 'ready' : 'pending'],
        ].map(([k, v]) => `<div class="metric-pill"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
      }
      const timeline = document.getElementById('benchmarkRunTimeline');
      if (timeline) {
        const rows = run.timeline || [];
        timeline.innerHTML = rows.length
          ? rows.slice(-6).reverse().map(row => {
              const ts = row.timestamp ? new Date(Number(row.timestamp) * 1000).toLocaleTimeString() : '--';
              return `<div class="quality-timeline-row"><time>${esc(ts)}</time><span>${esc(row.configuration_id || 'config')}: ${esc(row.status || run.state || 'N/A')}${row.reason ? ` - ${esc(row.reason)}` : ''}</span></div>`;
            }).join('')
          : '<div class="quality-timeline-row"><time>--</time><span>No benchmark run events yet.</span></div>';
      }
      const availability = ((benchmark.source_status || {}).sensor_availability) || {};
      const precheck = document.getElementById('benchmarkPrecheckList');
      if (precheck) {
        const required = ['realsense_active', 'rgb_active', 'depth_active', 'bottle_detected', 'target_stable', 'dashboard_fresh'];
        const optional = ['pointcloud_active', 'davis_aps_active', 'davis_events_active', 'baseline_quality_available', 'sync_available'];
        precheck.innerHTML = [
          ...required.map(k => [k, availability[k], true]),
          ...optional.map(k => [k, availability[k], false]),
        ].map(([k, ok, req]) => `<div><b>${esc(humanReason(k))}${req ? ' *' : ''}</b><span class="${stateClass(ok)}">${ok ? 'ok' : 'unavailable'}</span></div>`).join('');
      }
      const message = document.getElementById('benchmarkRunMessage');
      if (message) {
        message.textContent = run.current_error
          ? `Benchmark error: ${run.current_error}`
          : (run.current_status_message || 'Benchmark runner idle.');
        message.className = `note ${stateClass(run.current_error || run.state)}`;
      }
      qualityRows(document.getElementById('benchmarkWinnerRows'), [
        ['Best Overall', (winners.best_overall || {}).display_name || 'N/A'],
        ['Best Detection', (winners.best_bottle_detection || {}).display_name || 'N/A'],
        ['Best Quality', (winners.best_grasp_quality || {}).display_name || 'N/A'],
        ['Safest Decision', (winners.best_safety_decision || {}).display_name || 'N/A'],
        ['Best Motion Safety', (winners.best_motion_detection || {}).display_name || 'N/A'],
        ['Most Stable', (winners.most_stable_quality || {}).display_name || 'N/A'],
        ['Fastest', (winners.fastest || {}).display_name || 'N/A'],
        ['Strongest Positive Score Association', winners.strongest_positive_score_association || 'N/A'],
        ['Most Unreliable', winners.most_unreliable_feature || 'N/A'],
      ]);
      const generatedHost = document.getElementById('benchmarkGeneratedResults');
      if (generatedHost) {
        const outputs = run.outputs || {};
        const generatedRanking = summaryData.ranking_table || [];
        const validCount = generatedRanking.filter(row => row.result_validity === 'VALID' || row.result_validity === 'VALID_WITH_LIMITATIONS').length;
        const invalidCount = generatedRanking.length ? generatedRanking.length - validCount : 0;
        const leakage = generatedRanking.some(row => row.feature_leakage_detected);
        const link = (label, fmt) => `<button onclick="window.location='/api/benchmark/download-report?format=${fmt}'">${label}</button>`;
        generatedHost.innerHTML = `
          <div class="quality-row"><b>Tested combinations</b><span>${generatedRanking.length || run.total_configs || 0}</span></div>
          <div class="quality-row"><b>Valid / limited</b><span>${validCount}</span></div>
          <div class="quality-row"><b>Invalid / unavailable</b><span>${invalidCount}</span></div>
          <div class="quality-row"><b>Feature leakage</b><span class="${leakage ? 'bad' : 'ok'}">${leakage ? 'detected' : 'none'}</span></div>
          <div class="quality-row"><b>Proof benchmark</b><span>${summaryData.proof_benchmark_status || 'see saved proof run'}</span></div>
          <div class="quality-row"><b>Final run ID</b><span>${esc(run.benchmark_run_id || (outputs.run_dir || '').split('/').pop() || 'N/A')}</span></div>
          <div class="toolbar compact-toolbar">
            ${link('Download Comparison Tables', 'tables_zip')}
            ${link('Download HTML Report', 'html')}
            ${link('Download CSV Results', 'csv')}
            ${link('Download Charts', 'charts_zip')}
          </div>
          <p class="note">${outputs.run_dir ? `Latest report directory: ${esc(outputs.run_dir)}` : 'Run a benchmark to enable report downloads.'}</p>
        `;
      }
      const rankingHost = document.getElementById('benchmarkRankingRows');
      const ranking = summaryData.ranking_table || [];
      if (rankingHost) {
        rankingHost.innerHTML = ranking.length
          ? `<table class="benchmark-result-table"><thead><tr><th>Rank</th><th>Configuration</th><th>Features</th><th>Detect</th><th>Track</th><th>Contact</th><th>Quality</th><th>Stability</th><th>Ready</th><th>Unsafe</th><th>Motion</th><th>Latency</th><th>FPS</th><th>Availability</th><th>Confidence</th><th>Validity</th><th>Limitation</th></tr></thead><tbody>${ranking.map(row => {
              const deltas = row.baseline_deltas || {};
              const latencyDelta = deltas.latency_delta !== null && deltas.latency_delta !== undefined ? ` (${Number(deltas.latency_delta).toFixed(1)} vs RGB-D)` : '';
              return `<tr>
                <td>${row.rank || '-'}</td>
                <td>${esc(row.display_name)}</td>
                <td>${esc((row.enabled_feature_labels || []).slice(0, 4).join(', ') || row.enabled_feature_count || 'N/A')}</td>
                <td>${row.detection_success_rate ?? 'N/A'}%</td>
                <td>${row.center_jitter_px !== null && row.center_jitter_px !== undefined ? `jitter ${Number(row.center_jitter_px).toFixed(1)} px` : (row.target_loss_rate !== undefined ? `loss ${row.target_loss_rate}%` : 'N/A')}</td>
                <td>${row.contact_valid_rate ?? 'N/A'}%</td>
                <td>${row.average_quality_score ?? 'N/A'}</td>
                <td>${row.quality_stability_score ?? 'N/A'}</td>
                <td>${row.ready_to_close_rate ?? 'N/A'}%</td>
                <td>${row.unsafe_blocker_count ?? 0}</td>
                <td>${row.motion_warning_count ?? 0} warnings</td>
                <td>${row.average_latency_ms ?? 'N/A'} ms${latencyDelta}</td>
                <td>${row.effective_fps ?? 'N/A'}</td>
                <td>${row.missing_sensor_count ? 'limited' : 'OK'}</td>
                <td>${row.confidence_mean ?? 'N/A'}</td>
                <td class="${stateClass(row.result_validity)}">${esc(row.result_validity || 'N/A')}</td>
                <td>${esc(row.main_failure_reason || row.result_validity_reason || row.plain_language_explanation || '')}</td>
              </tr>`;
            }).join('')}</tbody></table>`
          : '<p class="note">Run a benchmark to generate ranking results.</p>';
      }
      const findingsHost = document.getElementById('benchmarkFindingRows');
      if (findingsHost) {
        const findings = summaryData.plain_language_findings || [];
        const recommendations = summaryData.recommendations || [];
        const limitations = summaryData.limitations || summaryData.trial_notes || [];
        findingsHost.innerHTML = findings.length
          ? [
              ...findings.map(f => `<div class="benchmark-finding">${esc(f)}</div>`),
              ...recommendations.map(f => `<div class="benchmark-finding ok">${esc(f)}</div>`),
              ...limitations.map(f => `<div class="benchmark-finding warn">${esc(f)}</div>`),
            ].join('')
          : '<p class="note">Findings appear after a benchmark run completes.</p>';
      }
    }
    function renderBenchmarkLab(benchmark) {
      if (!benchmark) return;
      window.latestBenchmark = benchmark;
      benchmarkCurrentConfigurations = benchmark.configurations || [];
      const banner = document.getElementById('benchmarkBanner');
      if (banner) banner.textContent = benchmark.banner || 'Benchmark results are experimental and do not control the robot.';
      renderBenchmarkRun(benchmark);
      renderBenchmarkFeaturePicker(benchmark);
      renderBenchmarkCards(benchmark);
      const labelHost = document.getElementById('benchmarkLabelButtons');
      if (labelHost && !labelHost.dataset.ready) {
        const labels = [
          'bottle_detected_correctly', 'bottle_missed', 'false_bottle_detection',
          'good_grasp', 'bad_grasp', 'contact_valid', 'contact_invalid',
          'stable', 'impact_only', 'incipient_slip', 'gross_slip',
          'rotation', 'vibration', 'target_lost',
        ];
        labelHost.innerHTML = labels.map(label => `<button onclick="benchmarkLabel('${label}')">${esc(benchmarkLabelText(label))}</button>`).join('');
        labelHost.dataset.ready = '1';
      }
      const prod = benchmark.production_context || {};
      const prodEl = document.getElementById('benchmarkProductionSummary');
      if (prodEl) {
        prodEl.innerHTML = [
          ['Production Decision', prod.decision || 'N/A'],
          ['Action', prod.recommended_action || 'N/A'],
          ['Ready to Close', yesNo(prod.ready_to_close)],
          ['Frame', prod.frame_id || 'N/A'],
        ].map(([k, v]) => `<div class="metric-pill"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
      }
      const trial = benchmark.trial || {};
      const trialChip = document.getElementById('benchmarkTrialChip');
      if (trialChip) {
        trialChip.textContent = trial.active ? `Trial ${trial.trial_id || ''}` : 'No active trial';
        trialChip.className = `chip ${trial.active ? 'ok' : 'warn'}`;
      }
      const trialMsg = document.getElementById('benchmarkTrialMessage');
      if (trialMsg) trialMsg.textContent = `Labels: ${trial.label_count || 0}${trial.last_label ? ` | last: ${benchmarkLabelText(trial.last_label)}` : ''}`;
      qualityRows(document.getElementById('benchmarkMetricRows'), (benchmark.metrics_rows || []).map(row => [
        row.configuration,
        `det ${yesNo(row.bottle_detected)} | contact ${row.contact_validity === true ? 'valid' : row.contact_validity === false ? 'invalid' : 'N/A'} | eps ${row.epsilon} | decision ${row.decision}`,
      ]));
      const contributionRows = [];
      (benchmark.comparison_cards || []).forEach(card => {
        (card.contributions || []).forEach(c => contributionRows.push([`${card.display_name}: ${c.feature}`, c.message || c.effect || 'N/A']));
        if (card.missing_features && card.missing_features.length) contributionRows.push([`${card.display_name}: disabled`, card.missing_features.join(', ')]);
      });
      qualityRows(document.getElementById('benchmarkContributionRows'), contributionRows.slice(0, 12));
      const timeline = document.getElementById('benchmarkTimelineRows');
      if (timeline) {
        const now = new Date().toLocaleTimeString();
        timeline.innerHTML = (benchmark.comparison_cards || []).map(card => `<div class="quality-timeline-row"><time>${esc(now)}</time><span>${esc(card.display_name)}: ${esc(card.decision)} (${esc(card.main_limitation)})</span></div>`).join('');
      }
      const reason = document.getElementById('benchmarkSelectionReason');
      if (reason) {
        const invalid = (benchmark.comparison_cards || []).find(card => !card.feature_selection_valid);
        reason.textContent = invalid ? `${invalid.display_name}: ${invalid.feature_selection_reason}` : 'Selections valid. Benchmark masks remain display/report-only.';
      }
      const msg = document.getElementById('benchmarkMessage');
      if (msg) {
        const ms = ((benchmark.profiling || {}).benchmark_eval_ms);
        msg.textContent = `Production result is isolated. Benchmark eval ${ms !== undefined ? ms : 'N/A'} ms. No cameras are reopened per card.`;
      }
    }
    function renderObjectProfiles(objectProfiles) {
      const obj = objectProfiles || {};
      const active = obj.active_profile || {};
      const scan = obj.scan_report || {};
      const activeHost = document.getElementById('objectActiveCards');
      if (activeHost) {
        cards(activeHost, [
          ['Active Target', active.display_name || active.object_id || 'N/A'],
          ['Object ID', active.object_id || 'N/A'],
          ['Type', active.object_type || 'N/A'],
          ['Source', active.source || 'N/A'],
          ['Confidence', active.profile_confidence !== undefined && active.profile_confidence !== null ? `${Math.round(Number(active.profile_confidence) * 100)}%` : 'N/A'],
          ['Grasp Fractions', ((obj.active_grasp_profile || {}).preferred_grasp_body_fractions || []).join(', ') || 'N/A'],
          ['Profile Status', obj.object_profile_status || obj.dashboard_status || 'N/A'],
          ['Safety', active.active_target_allowed === false ? 'not allowed' : 'production gates unchanged'],
        ]);
      }
      const resultHost = document.getElementById('objectScanResult');
      if (resultHost) {
        const draft = scan.profile_draft || {};
        const artifacts = scan.artifacts || {};
        const limitations = draft.limitations || [];
        resultHost.innerHTML = `
          <h3>Scan Result</h3>
          <div class="status-list">
            ${[
              ['State', scan.object_scan_status || 'IDLE'],
              ['Message', scan.message || scan.object_scan_error || 'Waiting for Scan Object.'],
              ['Mask Source', scan.mask_source || 'N/A'],
              ['Confidence', scan.object_scan_confidence !== undefined && scan.object_scan_confidence !== null ? `${Math.round(Number(scan.object_scan_confidence) * 100)}%` : 'N/A'],
              ['Draft', draft.display_name || draft.object_id || 'N/A'],
              ['Artifacts', Object.keys(artifacts).length ? Object.keys(artifacts).join(', ') : 'N/A'],
            ].map(([k, v]) => `<div><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('')}
          </div>
          <small>${limitations.length ? esc(limitations.join('; ')) : 'Profiles are priors only; live RGB-D safety checks still decide grasp readiness.'}</small>
        `;
      }
      const library = document.getElementById('objectProfileLibrary');
      if (!library) return;
      const profiles = obj.profiles || [];
      if (!profiles.length) {
        library.innerHTML = '<div class="object-profile-card"><h3>No saved profiles</h3><small>The legacy blue bottle fallback will be created automatically.</small></div>';
        return;
      }
      const activeId = obj.active_object_id || active.object_id;
      library.innerHTML = profiles.map(profile => {
        const id = profile.object_id || '';
        const isActive = id && id === activeId;
        const canArchive = id && !['blue_bottle', 'generic_graspable_object'].includes(id);
        return `<article class="object-profile-card ${isActive ? 'active' : ''}">
          <h3><span>${esc(profile.display_name || id)}</span><span class="chip ${isActive ? 'ok' : 'info'}">${isActive ? 'ACTIVE' : 'SAVED'}</span></h3>
          <div class="status-list">
            <div><b>ID</b><span>${esc(id)}</span></div>
            <div><b>Type</b><span>${esc(profile.object_type || 'N/A')}</span></div>
            <div><b>Confidence</b><span>${profile.profile_confidence !== undefined && profile.profile_confidence !== null ? `${Math.round(Number(profile.profile_confidence) * 100)}%` : 'N/A'}</span></div>
            <div><b>Allowed</b><span>${profile.active_target_allowed === false ? 'no' : 'yes'}</span></div>
          </div>
          <div class="toolbar compact-toolbar">
            <button onclick="setActiveObjectProfile('${esc(id)}')" ${isActive ? 'disabled' : ''}>Set Active</button>
            <button onclick="renameObjectProfile('${esc(id)}')">Rename</button>
            <button onclick="window.location='/api/object-profiles/export/${encodeURIComponent(id)}'">Export</button>
            ${canArchive ? `<button class="danger" onclick="archiveObjectProfile('${esc(id)}')">Archive</button>` : ''}
          </div>
          <small>${esc(profile.profile_path || '')}</small>
        </article>`;
      }).join('');
    }
    function metricVisible(emb, metricId) {
      const hidden = emb.hidden_metrics || [];
      const ids = emb.dashboard_metric_ids || [];
      if (hidden.includes(metricId)) return false;
      return !ids.length || ids.includes(metricId);
    }
    function dashboardMetricValue(metricId, qm, rs, emb, rawEpsilon, forceClosure) {
      if (metricId === 'two_contact_epsilon') return `${fmtNumber(rawEpsilon, 3)}${forceClosure === true ? ' / closure' : ''}`;
      if (metricId === 'grip_width') return fmtMm(rs.grip_width_mm);
      if (metricId === 'opposition') return qm.opposition_score !== undefined && qm.opposition_score !== null ? fmtNumber(qm.opposition_score, 2) : (qm.contact_quality_gate_status || 'N/A');
      if (metricId === 'support_triangle') return qm.support_triangle_score ?? qm.support_triangle ?? 'benchmark display only';
      if (metricId === 'contact_coverage') return qm.contact_coverage !== undefined && qm.contact_coverage !== null ? fmtNumber(qm.contact_coverage, 2) : `${emb.detected_contacts ?? 'N/A'} contacts`;
      if (metricId === 'suction_seal') return qm.suction_seal_score ?? qm.seal_area ?? 'benchmark display only';
      if (metricId === 'surface_flatness') return qm.surface_flatness ?? 'benchmark display only';
      if (metricId === 'normal_alignment') return qm.normal_alignment ?? 'benchmark display only';
      if (metricId === 'wrap_coverage') return qm.wrap_coverage ?? 'benchmark display only';
      if (metricId === 'contact_count_validity') return `${emb.detected_contacts ?? 'N/A'} / ${emb.expected_contacts ?? 'N/A'}`;
      return qm[metricId] ?? 'N/A';
    }
    function embodimentMetricRows(emb, qm, rs, rawEpsilon, forceClosure) {
      const metrics = emb.dashboard_metrics || [];
      if (!metrics.length) return [['Relevant Metrics', 'N/A']];
      return metrics.map(m => [
        m.label || humanReason(m.metric_id),
        dashboardMetricValue(m.metric_id, qm, rs, emb, rawEpsilon, forceClosure),
      ]);
    }
    function liveQualityScoreText(score, label) {
      if (score === undefined || score === null || Number.isNaN(Number(score))) return label || 'N/A';
      return `${label || ''} ${Number(score).toFixed(1)}`.trim();
    }
    function updateLiveQualityPanel(lq) {
      const contributionEl = document.getElementById('liveQualityContributionCards');
      const timelineEl = document.getElementById('liveQualityTimeline');
      const contributions = lq.sensor_contributions || [];
      if (contributionEl) {
        contributionEl.innerHTML = contributions.length ? contributions.map(item => {
          const status = item.status ?? 'N/A';
          const confidence = item.confidence !== undefined && item.confidence !== null ? `${Math.round(Number(item.confidence) * 100)}%` : 'N/A';
          const penalty = item.penalty !== undefined && item.penalty !== null ? Number(item.penalty).toFixed(1) : '0.0';
          return `<div class="live-contribution ${stateClass(status)}"><b>${esc(item.sensor || 'Sensor')}</b><span>${esc(status)} · ${confidence}</span><small>Penalty ${penalty}. ${esc(item.details || '')}</small></div>`;
        }).join('') : '<div class="live-contribution warn"><b>Live Quality</b><span>Waiting</span><small>No live-quality contribution payload yet.</small></div>';
      }
      const rows = lq.timeline_recent || [];
      if (timelineEl) {
        timelineEl.innerHTML = rows.length ? rows.slice(-6).reverse().map(row => {
          const ts = row.timestamp ? new Date(Number(row.timestamp) * 1000).toLocaleTimeString() : '--:--:--';
          const score = row.live_quality_score !== undefined && row.live_quality_score !== null ? Number(row.live_quality_score).toFixed(1) : 'N/A';
          const marker = row.event_marker || row.primary_risk || row.decision || 'sample';
          return `<div class="quality-timeline-row"><time>${esc(ts)}</time><span><b>${esc(row.decision || 'N/A')}</b> · ${score} · ${esc(marker)}</span></div>`;
        }).join('') : '<div class="quality-timeline-row"><time>--</time><span>Waiting for live-quality samples.</span></div>';
      }
    }
    function updateEmbodimentSelector(emb) {
      const select = document.getElementById('embodimentSelector');
      const state = document.getElementById('embodimentSelectorState');
      if (!select) return;
      const options = emb.selector_options || [];
      const current = emb.canonical_embodiment_id || emb.active_embodiment || 'parallel_jaw_two_finger';
      const markup = options.map(option => {
        const label = `${option.display_name || option.embodiment_id} - ${option.release_status || 'CONFIG_ONLY'}`;
        return `<option value="${option.embodiment_id}">${label}</option>`;
      }).join('');
      if (markup && select.dataset.optionsMarkup !== markup) {
        select.innerHTML = markup;
        select.dataset.optionsMarkup = markup;
      }
      select.value = current;
      select.disabled = Boolean(emb.selector_locked);
      if (state) {
        const locked = emb.selector_locked ? 'locked during active grasp' : 'change allowed when idle';
        const mode = emb.release_mode || emb.release_status || 'CONFIG_ONLY';
        state.textContent = `${mode.replaceAll('_', ' ')}; ${locked}`;
        state.className = stateClass(emb.production_decisions_allowed ? 'ready' : mode);
      }
    }
    function forceClosureText(v) {
      if (v === true) return 'Yes';
      if (v === false) return 'No';
      return 'N/A';
    }
    function qualityIsPlanned(qm, inputSource, contactSource) {
      const value = `${inputSource || ''} ${contactSource || ''}`.toLowerCase();
      return value.includes('planned_contact') || value.includes('planned preview');
    }
    function recommendedQualityAction(qm, label, inputLabel, forceClosure, failureReason) {
      const status = String(qm.quality_matrix_status || '').toLowerCase();
      const staleReasons = qm.quality_current_mismatch_reasons || qm.matrix_stale_reasons || [];
      if (qm.quality_label_stale || qm.matrix_stale_due_to_motion) {
        const reason = Array.isArray(staleReasons) && staleReasons.length ? ` (${staleReasons.join(', ')})` : '';
        return `Current matrix is stale${reason}. Recompute quality for the current locked grasp.`;
      }
      if (status.includes('failed') || String(label).includes('INVALID')) {
        return `Quality is invalid: ${failureReason || 'missing required inputs'}.`;
      }
      if (inputLabel === 'Planned Preview') {
        if (forceClosure === true && String(label).includes('WEAK')) {
          return 'Planned grasp has force closure, but the margin is weak. Actual clamp contacts are not confirmed yet.';
        }
        return 'This is a planned-contact preview, not actual finger/gripper contact. Waiting for clamp contacts.';
      }
      if (forceClosure === true && String(label).includes('WEAK')) {
        return 'Force closure exists, but the margin against slip or rotation is small.';
      }
      if (forceClosure === true) return 'Actual contacts produce force closure. Larger epsilon means better margin.';
      if (forceClosure === false) return 'Contacts do not currently produce force closure. Adjust grasp before closing.';
      return 'Waiting for enough contact and quality inputs.';
    }
    async function setDavisApsDisplayMode(mode) {
      try {
        const r = await fetch('/api/davis/aps-display-mode', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({mode})
        });
        const j = await r.json();
        showToast(j.message || (j.ok ? 'APS display mode updated.' : 'APS display mode change failed.'), j.ok ? 'ok' : 'bad');
      } catch (err) {
        showToast('APS display mode change failed.', 'bad');
      }
    }
    async function post(path) {
      const r = await fetch(path, {method:'POST'});
      const j = await r.json();
      console.log(j.message);
      const msg = document.getElementById('controlMessage');
      if (msg) msg.textContent = j.message || (j.ok ? 'Command accepted.' : 'Command failed.');
      showToast(j.message || (j.ok ? 'Command accepted.' : 'Command failed.'), j.ok ? 'ok' : 'bad');
      await refresh();
      return j;
    }
    async function postJson(path, payload) {
      const r = await fetch(path, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload || {}),
      });
      const j = await r.json();
      const msg = document.getElementById('controlMessage');
      if (msg) msg.textContent = j.message || (j.ok ? 'Command accepted.' : 'Command failed.');
      showToast(j.message || (j.ok ? 'Command accepted.' : 'Command failed.'), j.ok ? 'ok' : 'bad');
      await refresh();
      return j;
    }
    const HAND_OFFSET_RESET_DEFAULTS = __HAND_OFFSET_RESET_DEFAULTS__;
    let handOffsetUiInitialized = false;
    function updateHandOffsetReadout(markDirty=true) {
      const controls = {
        handOffsetX: 'handOffsetXValue',
        handOffsetY: 'handOffsetYValue',
        handOffsetZ: 'handOffsetZValue',
        handContactGap: 'handContactGapValue',
      };
      for (const [inputId, outputId] of Object.entries(controls)) {
        const input = document.getElementById(inputId);
        const output = document.getElementById(outputId);
        if (input && output) output.textContent = `${Math.round(Number(input.value || 0))} mm`;
      }
      const z = Number(document.getElementById('handOffsetZ')?.value || 0);
      const gap = Number(document.getElementById('handContactGap')?.value || 0);
      const desired = z + gap;
      const minimum = desired - Number(HAND_OFFSET_RESET_DEFAULTS.min_delta_mm || 0);
      const maximum = desired + Number(HAND_OFFSET_RESET_DEFAULTS.max_delta_mm || 0);
      setText('handOffsetTarget', `Target ${Math.round(desired)} mm · range ${Math.round(minimum)}–${Math.round(maximum)} mm`);
      if (markDirty) document.getElementById('handOffsetTuning')?.setAttribute('data-dirty', 'true');
    }
    function syncHandOffsetControls(rs) {
      const tuning = document.getElementById('handOffsetTuning');
      if (!tuning || tuning.dataset.dirty === 'true') return;
      const config = rs.hand_mounted_camera || {};
      const offset = rs.camera_to_hand_offset_mm || config.camera_to_hand_offset_mm || HAND_OFFSET_RESET_DEFAULTS.camera_to_hand_offset_mm;
      const gap = rs.desired_contact_gap_mm ?? config.desired_contact_gap_mm ?? HAND_OFFSET_RESET_DEFAULTS.desired_contact_gap_mm;
      for (const [id, value] of Object.entries({handOffsetX:offset.x, handOffsetY:offset.y, handOffsetZ:offset.z, handContactGap:gap})) {
        const input = document.getElementById(id);
        if (input && value !== undefined && value !== null) input.value = String(Number(value));
      }
      updateHandOffsetReadout(false);
      handOffsetUiInitialized = true;
      const status = rs.hand_offset_calibration_status || config.hand_offset_calibration_status || 'APPROXIMATE_NOT_PHYSICALLY_CALIBRATED';
      setText('handOffsetStatus', humanReason(status));
    }
    async function saveHandCameraOffset(source='dashboard_live_tuning') {
      const numberValue = id => Number(document.getElementById(id)?.value);
      const payload = {
        camera_to_hand_offset_enabled: true,
        camera_to_hand_offset_mm: {
          x:numberValue('handOffsetX'),
          y:numberValue('handOffsetY'),
          z:numberValue('handOffsetZ'),
        },
        desired_contact_gap_mm:numberValue('handContactGap'),
        hand_offset_source:source,
      };
      const result = await postJson('/api/hand-camera-offset', payload);
      if (result.saved) {
        const tuning = document.getElementById('handOffsetTuning');
        if (tuning) tuning.dataset.dirty = 'false';
        setText('handOffsetStatus', 'Approximate user-tuned · not physically calibrated');
      }
      return result;
    }
    async function resetHandCameraOffset() {
      const offset = HAND_OFFSET_RESET_DEFAULTS.camera_to_hand_offset_mm;
      const defaults = {handOffsetX:offset.x, handOffsetY:offset.y, handOffsetZ:offset.z, handContactGap:HAND_OFFSET_RESET_DEFAULTS.desired_contact_gap_mm};
      for (const [id, value] of Object.entries(defaults)) {
        const input = document.getElementById(id);
        if (input) input.value = String(value);
      }
      updateHandOffsetReadout();
      return saveHandCameraOffset('dashboard_reset_to_user_starting_values');
    }
    function beginHandOffsetCalibration() {
      const tuning = document.getElementById('handOffsetTuning');
      if (tuning) tuning.open = true;
      setText('handOffsetStatus', 'Place both visible fingers at the locked band, then press Set From Current View. Approximate only.');
    }
    async function setHandOffsetFromCurrentView() {
      const rs = latestOperatorStatus.realsense || {};
      const left = rs.left_inner_edge;
      const right = rs.right_inner_edge;
      const center = rs.camera_optical_center_px;
      const depth = Number(rs.hand_projection_depth_mm);
      const fx = Number(rs.fx);
      const fy = Number(rs.fy);
      const evidenceValid = rs.left_finger_detected === true && rs.right_finger_detected === true
        && Array.isArray(left) && Array.isArray(right) && Array.isArray(center)
        && Number.isFinite(depth) && depth > 0 && Number.isFinite(fx) && fx > 0 && Number.isFinite(fy) && fy > 0;
      if (!evidenceValid) {
        showToast('Current-view calibration needs both detected fingers, fixed hand-contact projection depth, and intrinsics. Use the sliders instead.', 'bad');
        setText('handOffsetStatus', 'Current-view calibration unavailable · use manual sliders');
        return {ok:false};
      }
      const fingerCenter = [(Number(left[0]) + Number(right[0])) * 0.5, (Number(left[1]) + Number(right[1])) * 0.5];
      const xMm = Math.max(-300, Math.min(300, (fingerCenter[0] - Number(center[0])) * depth / fx));
      const yMm = Math.max(-300, Math.min(300, (fingerCenter[1] - Number(center[1])) * depth / fy));
      document.getElementById('handOffsetX').value = String(Math.round(xMm));
      document.getElementById('handOffsetY').value = String(Math.round(yMm));
      updateHandOffsetReadout();
      setText('handOffsetStatus', 'Estimated X/Y from detected fingers; Z retained from slider · not physically calibrated');
      return saveHandCameraOffset('detected_finger_midpoint_current_view_approximation');
    }
    let handDemoCaptureStream = null;
    async function captureHandDemoStage(stage, elementId=null) {
      try {
        const liveTrack = handDemoCaptureStream?.getVideoTracks?.().find(track => track.readyState === 'live');
        if (!liveTrack) {
          if (!navigator.mediaDevices?.getDisplayMedia) throw new Error('native current-tab capture is unavailable');
          handDemoCaptureStream = await navigator.mediaDevices.getDisplayMedia({
            video: {displaySurface: 'browser', frameRate: 1},
            audio: false,
            preferCurrentTab: true,
            selfBrowserSurface: 'include',
          });
        }
        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        video.srcObject = handDemoCaptureStream;
        await video.play();
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        if (!video.videoWidth || !video.videoHeight) throw new Error('shared tab has no current video frame');
        const canvas = document.createElement('canvas');
        const element = elementId ? document.getElementById(elementId) : null;
        if (element) {
          const rect = element.getBoundingClientRect();
          const scaleX = video.videoWidth / Math.max(1, window.innerWidth);
          const scaleY = video.videoHeight / Math.max(1, window.innerHeight);
          const sx = Math.max(0, Math.round(rect.left * scaleX));
          const sy = Math.max(0, Math.round(rect.top * scaleY));
          const sw = Math.max(1, Math.min(video.videoWidth - sx, Math.round(rect.width * scaleX)));
          const sh = Math.max(1, Math.min(video.videoHeight - sy, Math.round(rect.height * scaleY)));
          canvas.width = sw;
          canvas.height = sh;
          canvas.getContext('2d').drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
        } else {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
        }
        video.srcObject = null;
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        if (!blob) throw new Error('browser could not encode the stage PNG');
        const response = await fetch(`/api/hand-demo/stage-screenshot/${encodeURIComponent(stage)}`, {
          method: 'POST',
          headers: {'Content-Type': 'image/png'},
          body: blob,
        });
        const result = await response.json();
        if (!result.ok) throw new Error(result.message || 'stage capture rejected');
        showToast(result.message, 'ok');
        return true;
      } catch (err) {
        showToast(`Stage screenshot unavailable: ${err?.message || err}`, 'warn');
        return false;
      }
    }
    function stopHandDemoCapture() {
      for (const track of handDemoCaptureStream?.getTracks?.() || []) track.stop();
      handDemoCaptureStream = null;
    }
    async function captureSelectedHandDemoStage() {
      const stage = document.getElementById('handDemoCaptureStage')?.value;
      if (!stage) {
        showToast('Select a demo stage before capturing.', 'warn');
        return false;
      }
      return captureHandDemoStage(stage);
    }
    function benchmarkLabelText(label) {
      return String(label || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
    async function postBenchmark(path, payload) {
      const r = await fetch(path, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload || {}),
      });
      const j = await r.json();
      showToast(j.message || (j.ok ? 'Benchmark command accepted.' : 'Benchmark command failed.'), j.ok ? 'ok' : 'bad');
      await refreshBenchmarkStatus(true);
      await refresh();
      return j;
    }
    function selectedBenchmarkCustomFeatures() {
      return Array.from(document.querySelectorAll('.benchmark-feature-checkbox:checked')).map(cb => cb.value);
    }
    async function setBenchmarkConfigurations(configurations) {
      benchmarkCurrentConfigurations = (configurations || []).slice(0, 4);
      return postBenchmark('/api/benchmark/selection', {configurations: benchmarkCurrentConfigurations});
    }
    async function addBenchmarkPreset(presetId) {
      const bm = window.latestBenchmark || {};
      const preset = ((bm.registry || {}).presets || []).find(p => p.preset_id === presetId);
      if (!preset) return;
      const config = {
        configuration_id: preset.preset_id,
        display_name: preset.display_name,
        preset_id: preset.preset_id,
        feature_ids: preset.feature_ids || [],
      };
      const existing = (benchmarkCurrentConfigurations || []).filter(c => c.configuration_id !== config.configuration_id);
      existing.push(config);
      await setBenchmarkConfigurations(existing.slice(-4));
    }
    async function applyBenchmarkCustom() {
      const featureIds = selectedBenchmarkCustomFeatures();
      const configs = (benchmarkCurrentConfigurations || []).filter(c => c.configuration_id !== 'custom');
      configs.unshift({
        configuration_id: 'custom',
        display_name: 'Custom',
        preset_id: 'custom',
        feature_ids: featureIds,
      });
      await setBenchmarkConfigurations(configs.slice(0, 4));
    }
    async function resetBenchmarkDefaults() {
      const bm = window.latestBenchmark || {};
      const presets = ((bm.registry || {}).presets || []);
      const ids = ['rgb_only', 'rgb_depth', 'davis_events_only', 'full_multimodal'];
      const configs = ids.map(id => {
        const preset = presets.find(p => p.preset_id === id) || {};
        return {configuration_id:id, display_name:preset.display_name || benchmarkLabelText(id), preset_id:id, feature_ids:preset.feature_ids || []};
      });
      await setBenchmarkConfigurations(configs);
    }
    async function clearBenchmarkCards() {
      await setBenchmarkConfigurations([{configuration_id:'no_features', display_name:'No Features', preset_id:'custom', feature_ids:[]}]);
    }
    async function benchmarkRunCommand(command, payload={}) {
      return postBenchmark(`/api/benchmark/${command}`, payload);
    }
    async function runBenchmark(mode='sequential_live') {
      return benchmarkRunCommand('start', {
        mode,
        duration_sec: benchmarkDurationSec(),
        sample_rate_hz: 5,
      });
    }
    async function benchmarkTrialCommand(command) {
      return postBenchmark(`/api/benchmark/trial/${command}`, {});
    }
    async function benchmarkLabel(label) {
      return postBenchmark('/api/benchmark/label', {label});
    }
    async function benchmarkUndoLabel() {
      return postBenchmark('/api/benchmark/label/undo', {});
    }
    const startDemo = n => post(`/api/start/${n}`);
    const stopDemo = n => post(`/api/stop/${n}`);
    const davisCommand = n => post(`/api/davis/${n}`);
    const realsenseCommand = n => post(`/api/realsense-command/${n}`);
    async function markApproachPhase() {
      const rs = latestOperatorStatus.realsense || {};
      const hand = latestOperatorStatus.hand_prototype || rs;
      const bandLocked = hand.grasp_band_locked === true || rs.grasp_band_locked === true;
      const msg = document.getElementById('controlMessage');
      if (!bandLocked) {
        const text = 'Automatic lock and analytic baseline are pending. Keep exactly one bottle visible and hold it steady.';
        if (msg) msg.textContent = text;
        showToast(text, 'bad');
        return false;
      }
      const result = await realsenseCommand('start_approach');
      if (!result?.ok) return false;
      operatorApproachPhaseMarked = true;
      const button = document.getElementById('approachPhaseBtn');
      if (button) button.textContent = 'Approach Marked — Move Hand Manually';
      const text = 'Approach phase marked. Movement remains manual; tracking and all safety gates remain active.';
      if (msg) msg.textContent = text;
      showToast(text, 'ok');
      updateWorkflowStepper(rs, {}, latestOperatorStatus.davis_motion || {}, {});
      return true;
    }
    async function startCloseUpDavisMonitor() {
      return davisCommand('start');
    }
    async function resetHandPrototype() {
      operatorApproachPhaseMarked = false;
      const button = document.getElementById('approachPhaseBtn');
      if (button) button.textContent = 'Start Approach Mode';
      return realsenseCommand('reset_lock_clear_matrix');
    }
    async function toggleGraspDeclaration() {
      const button = document.getElementById('graspStateBtn');
      const postGraspActive = button?.dataset.postGraspActive === 'true';
      if (button) button.disabled = true;
      try {
        return await realsenseCommand(postGraspActive ? 'mark_gripper_open' : 'mark_gripper_closed');
      } finally {
        if (button) {
          button.disabled = button.dataset.postGraspActive !== 'true'
            && (button.dataset.bandLocked !== 'true' || button.dataset.postGraspBaselineReady !== 'true');
        }
      }
    }
    const multiviewCommand = n => post(`/api/multiview-command/${n}`);
    async function recordSnapshot() {
      const capturedScreen = await captureHandDemoStage('hand_grasp_demo_screen');
      if (capturedScreen) {
        await captureHandDemoStage('operator_snapshot', 'handDemoRealSensePanel');
        await captureHandDemoStage('davis_snapshot', 'handDemoDavisPanel');
        await captureHandDemoStage('quality_snapshot', 'handDemoQualityPanel');
      }
      stopHandDemoCapture();
      return post('/api/record-snapshot');
    }
    const setEmbodimentFromSelector = embodimentId => postJson('/api/embodiment', {embodiment_id: embodimentId});
    function objectProfileName() {
      return document.getElementById('objectProfileName')?.value || 'Scanned Object';
    }
    const scanObjectProfile = () => postJson('/api/object-profiles/scan', {display_name: objectProfileName(), duration_sec: 3.0});
    const saveObjectProfile = () => postJson('/api/object-profiles/save', {display_name: objectProfileName()});
    const setActiveObjectProfile = objectId => postJson('/api/object-profiles/set-active', {object_id: objectId});
    async function renameObjectProfile(objectId) {
      const displayName = window.prompt('New display name for this object profile:');
      if (!displayName) return;
      return postJson('/api/object-profiles/rename', {object_id: objectId, display_name: displayName});
    }
    async function archiveObjectProfile(objectId) {
      if (!window.confirm('Archive this object profile? It will not be deleted.')) return;
      return postJson('/api/object-profiles/archive', {object_id: objectId});
    }
    const enhanceObjectProfileWithMultiview = objectId => postJson('/api/object-profiles/enhance-multiview', objectId ? {object_id: objectId} : {});
    async function freshLockQuality() {
      const btn = document.getElementById('freshLockQualityBtn');
      const msg = document.getElementById('freshLockQualityMessage');
      if (btn) btn.disabled = true;
      if (msg) msg.textContent = 'Fresh Lock + Quality: starting...';
      try {
        const r = await fetch('/api/fresh_lock_quality', {method:'POST'});
        const j = await r.json();
        if (msg) msg.textContent = j.message || j.fresh_lock_quality_step || 'Fresh Lock + Quality command sent.';
        if (!j.ok) alert(j.message || 'Fresh Lock + Quality could not start.');
      } catch (err) {
        if (msg) msg.textContent = `Fresh Lock + Quality failed to start: ${err}`;
      }
      await refresh();
    }
    function setImg(id, info) {
      const img = document.getElementById(id);
      if (!info || !info.exists) {
        img.removeAttribute('src');
        return;
      }
      if (img.getAttribute('src') !== info.url) img.src = info.url;
    }
    function setStaticImg(id, info, fileInfo) {
      const img = document.getElementById(id);
      if (!img) return;
      const card = img.closest('.plot-card');
      let staleLabel = card ? card.querySelector('.plot-stale-label') : null;
      if (card && !staleLabel) {
        staleLabel = document.createElement('p');
        staleLabel.className = 'plot-stale-label';
        img.insertAdjacentElement('afterend', staleLabel);
      }
      if (!info || !info.exists) {
        img.removeAttribute('src');
        img.alt = 'not available';
        if (card) card.classList.remove('stale');
        if (staleLabel) staleLabel.textContent = '';
        markVisualAvailability(id, info);
        return;
      }
      const version = info.version || info.quality_compute_id || (fileInfo || {}).mtime || 0;
      const url = `${info.url}?v=${encodeURIComponent(version)}`;
      if (img.getAttribute('src') !== url) img.src = url;
      if (card) card.classList.toggle('stale', Boolean(info.stale));
      if (staleLabel) staleLabel.textContent = info.stale ? (info.stale_label || 'PLOT STALE') : '';
      markVisualAvailability(id, info);
    }
    function setMainVisualImages(data) {
      const images = data.images || latestImages || {};
      const files = data.final_files || {};
      const handDemoVisible = dashboardMode !== 'engineering' && isSectionActive('operatorSection');
      if (handDemoVisible) {
        setImg('imgHandDemoRealSense', images.overlay);
        setImg('imgHandDemoDavis', images.davis_heatmap);
      } else {
        document.getElementById('imgHandDemoRealSense')?.removeAttribute('src');
        document.getElementById('imgHandDemoDavis')?.removeAttribute('src');
      }
      setImg('imgOverlay', images.overlay);
      markVisualAvailability('imgOverlay', images.overlay);
      const davisRoi = data.davis_roi_on_realsense_overlay || {};
      const davisRoiSvg = document.getElementById('davisRoiOnOverlay');
      const davisRoiPolygon = document.getElementById('davisRoiOnOverlayPolygon');
      const roiPoints = Array.isArray(davisRoi.polygon_px) ? davisRoi.polygon_px : [];
      const roiFrame = Array.isArray(davisRoi.realsense_frame_size) ? davisRoi.realsense_frame_size : [];
      const showDavisRoi = davisRoi.valid === true && roiPoints.length >= 4 && roiFrame.length === 2;
      if (davisRoiSvg && davisRoiPolygon) {
        davisRoiSvg.classList.toggle('hidden', !showDavisRoi);
        if (showDavisRoi) {
          davisRoiSvg.setAttribute('viewBox', `0 0 ${Number(roiFrame[0])} ${Number(roiFrame[1])}`);
          davisRoiPolygon.setAttribute('points', roiPoints.map(point => `${Number(point[0])},${Number(point[1])}`).join(' '));
          davisRoiSvg.setAttribute('aria-label', `Calibrated ${davisRoi.source || 'DAVIS'} event ROI on the RealSense overlay`);
        } else {
          davisRoiPolygon.removeAttribute('points');
        }
      }
      if (operatorVisualActive('rgb')) {
        setImg('imgMainRgb', images.rgb);
      } else {
        document.getElementById('imgMainRgb')?.removeAttribute('src');
      }
      markVisualAvailability('imgMainRgb', images.rgb);
      if (operatorVisualActive('depth')) {
        setImg('imgMainDepth', images.depth);
      } else {
        document.getElementById('imgMainDepth')?.removeAttribute('src');
      }
      markVisualAvailability('imgMainDepth', images.depth);
      if (operatorVisualActive('quality')) {
        setStaticImg('imgMainQuality', images.bottle_quality_closeup || images.bottle_quality_overlay, files.bottle_quality_closeup || files.bottle_quality_overlay);
      } else {
        document.getElementById('imgMainQuality')?.removeAttribute('src');
      }
      if (operatorVisualActive('davis')) {
        setStaticImg('imgMainDavis', images.davis_motion_overlay || images.davis_heatmap, files.davis_motion_overlay || files.davis_heatmap);
      } else {
        document.getElementById('imgMainDavis')?.removeAttribute('src');
      }
      if (operatorVisualActive('pointcloud')) {
        setStaticImg('imgMainPointcloud', images.pointcloud_preview || images.bottle_pointcloud_preview, files.pointcloud_preview || files.bottle_pointcloud_preview);
      } else {
        document.getElementById('imgMainPointcloud')?.removeAttribute('src');
      }
      if (operatorVisualActive('multiview')) {
        setStaticImg('imgMainMultiview', images.multiview_reconstruction_preview || images.pointcloud_context, files.multiview_reconstruction_preview || files.pointcloud_context);
      } else {
        document.getElementById('imgMainMultiview')?.removeAttribute('src');
      }
    }
    function setHandDemoVisualState(id, available, heading, reason, action) {
      const visual = document.getElementById(id);
      if (!visual) return;
      visual.classList.toggle('empty', !available);
      const empty = visual.querySelector('.hand-demo-empty');
      if (!empty) return;
      empty.innerHTML = `<div><strong>${esc(heading)}</strong><span>${esc(reason)}</span><small>${esc(action)}</small></div>`;
    }
    function disposeQualityWrenchGroup() {
      const group = qualityWrenchViewer.group;
      if (!group) return;
      while (group.children.length) {
        const child = group.children.pop();
        child.traverse?.((node) => {
          if (node.geometry) node.geometry.dispose();
          if (Array.isArray(node.material)) node.material.forEach(material => material.dispose());
          else if (node.material) node.material.dispose();
        });
      }
      qualityWrenchViewer.liveSphere = null;
    }
    function renderQualityWrenchViewer() {
      const viewer = qualityWrenchViewer;
      const shell = document.getElementById('handDemoWrenchShell');
      if (!viewer.initialized || !shell) return;
      const width = Math.max(240, Math.round(shell.clientWidth));
      const height = Math.max(160, Math.round(shell.clientHeight));
      viewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      viewer.renderer.setSize(width, height, false);
      viewer.camera.aspect = width / height;
      viewer.camera.updateProjectionMatrix();
      viewer.renderer.render(viewer.scene, viewer.camera);
    }
    function initQualityWrenchViewer() {
      if (qualityWrenchViewer.initialized) return true;
      if (!window.THREE) return false;
      const canvas = document.getElementById('handDemoWrenchCanvas');
      if (!canvas) return false;
      const THREE = window.THREE;
      const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false, powerPreference:'low-power'});
      renderer.setClearColor(0x030911, 1);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(34, 2, 0.01, 100);
      const group = new THREE.Group();
      group.rotation.set(-0.28, 0.48, 0.04);
      scene.add(group);
      scene.add(new THREE.AmbientLight(0xbfd7f2, 0.72));
      const keyLight = new THREE.DirectionalLight(0xffffff, 0.72);
      keyLight.position.set(2, 3, 4);
      scene.add(keyLight);
      Object.assign(qualityWrenchViewer, {initialized:true, renderer, scene, camera, group});
      canvas.addEventListener('pointerdown', event => {
        qualityWrenchViewer.dragging = true;
        qualityWrenchViewer.lastX = event.clientX;
        qualityWrenchViewer.lastY = event.clientY;
        canvas.setPointerCapture?.(event.pointerId);
      });
      canvas.addEventListener('pointermove', event => {
        if (!qualityWrenchViewer.dragging) return;
        const dx = event.clientX - qualityWrenchViewer.lastX;
        const dy = event.clientY - qualityWrenchViewer.lastY;
        qualityWrenchViewer.lastX = event.clientX;
        qualityWrenchViewer.lastY = event.clientY;
        group.rotation.y += dx * 0.008;
        group.rotation.x += dy * 0.008;
        renderQualityWrenchViewer();
      });
      const stopDragging = () => { qualityWrenchViewer.dragging = false; };
      canvas.addEventListener('pointerup', stopDragging);
      canvas.addEventListener('pointercancel', stopDragging);
      return true;
    }
    function buildQualityWrenchScene(payload) {
      const THREE = window.THREE;
      const viewer = qualityWrenchViewer;
      const analytic = payload.analytic || {};
      const points = analytic.wrench_points || [];
      const faces = analytic.hull_faces || [];
      disposeQualityWrenchGroup();
      if (!points.length || !faces.length) return false;
      const positions = [];
      faces.forEach(face => face.forEach(index => {
        const point = points[index] || [0, 0, 0];
        positions.push(Number(point[0]), Number(point[1]), Number(point[2]));
      }));
      const hullGeometry = new THREE.BufferGeometry();
      hullGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      hullGeometry.computeVertexNormals();
      const stale = payload.status === 'STALE';
      const hullColor = stale ? 0xf5a524 : 0x5fa8ff;
      const hullMesh = new THREE.Mesh(
        hullGeometry,
        new THREE.MeshPhongMaterial({color:hullColor, transparent:true, opacity:0.16, side:THREE.DoubleSide, depthWrite:false, shininess:30})
      );
      viewer.group.add(hullMesh);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(hullGeometry, 2),
        new THREE.LineBasicMaterial({color:hullColor, transparent:true, opacity:0.92})
      );
      viewer.group.add(edges);
      const extent = Math.max(0.3, ...points.flat().map(value => Math.abs(Number(value) || 0)));
      const axisLength = extent * 1.18;
      viewer.group.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), axisLength, 0xff7379, axisLength * 0.07, axisLength * 0.035));
      viewer.group.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), axisLength, 0x55d991, axisLength * 0.07, axisLength * 0.035));
      viewer.group.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), axisLength, 0x5fa8ff, axisLength * 0.07, axisLength * 0.035));
      const epsilon = Number(analytic.epsilon_region_radius || 0);
      if (epsilon > 0) {
        const epsilonSphere = new THREE.Mesh(
          new THREE.SphereGeometry(1, 18, 12),
          new THREE.MeshBasicMaterial({color:0x41d9d0, wireframe:true, transparent:true, opacity:0.8, depthWrite:false})
        );
        epsilonSphere.scale.setScalar(epsilon);
        viewer.group.add(epsilonSphere);
      }
      const liveSphere = new THREE.Mesh(
        new THREE.SphereGeometry(1, 20, 14),
        new THREE.MeshPhongMaterial({color:0x36d483, transparent:true, opacity:0.34, depthWrite:false, shininess:50})
      );
      liveSphere.visible = false;
      viewer.group.add(liveSphere);
      viewer.liveSphere = liveSphere;
      const weak = analytic.weak_direction;
      if (Array.isArray(weak) && weak.length >= 3) {
        const direction = new THREE.Vector3(Number(weak[0]), Number(weak[1]), Number(weak[2]));
        if (direction.lengthSq() > 1e-8) {
          direction.normalize();
          viewer.group.add(new THREE.ArrowHelper(direction, new THREE.Vector3(0, 0, 0), axisLength * 0.78, 0xd8f3ff, axisLength * 0.09, axisLength * 0.045));
        }
      }
      viewer.camera.position.set(extent * 2.15, extent * 1.55, extent * 2.3);
      viewer.camera.near = Math.max(0.005, extent * 0.01);
      viewer.camera.far = extent * 20;
      viewer.camera.lookAt(0, 0, 0);
      viewer.camera.updateProjectionMatrix();
      return true;
    }
    function setQualityWrenchFallback(title, reason, action) {
      const fallback = document.getElementById('handDemoWrenchFallback');
      if (!fallback) return;
      const strong = fallback.querySelector('strong');
      const span = fallback.querySelector('span');
      const small = fallback.querySelector('small');
      if (strong) strong.textContent = title;
      if (span) span.textContent = reason;
      if (small) small.textContent = action;
    }
    async function updateQualityWrenchDiagram(payload) {
      payload = payload || {};
      const shell = document.getElementById('handDemoWrenchShell');
      if (!shell) return;
      const status = String(payload.status || 'UNAVAILABLE').toUpperCase();
      setText('handDemoWrenchStatus', status);
      if (status === 'UNAVAILABLE' || !(payload.analytic || {}).available) {
        shell.classList.add('unavailable');
        shell.dataset.status = 'UNAVAILABLE';
        setText('handDemoWrenchCaption', payload.reason || 'Waiting for locked RGB-D contact geometry.');
        setQualityWrenchFallback('Wrench-space quality unavailable', payload.reason || 'No current analytic hull is available.', payload.action || 'Hold exactly one bottle steady; automatic baseline pending.');
        return;
      }
      if (qualityWrenchViewer.updating) return;
      qualityWrenchViewer.updating = true;
      try {
        await loadThree();
        if (!initQualityWrenchViewer()) throw new Error('3D renderer could not initialize');
        const analytic = payload.analytic || {};
        const live = payload.live || {};
        const geometryKey = JSON.stringify([analytic.quality_compute_id, status, analytic.epsilon, (analytic.wrench_points || []).length, (analytic.hull_faces || []).length]);
        if (qualityWrenchViewer.geometryKey !== geometryKey) {
          if (!buildQualityWrenchScene(payload)) throw new Error('analytic wrench hull is empty');
          qualityWrenchViewer.geometryKey = geometryKey;
        }
        shell.classList.remove('unavailable');
        shell.dataset.status = status;
        const liveSphere = qualityWrenchViewer.liveSphere;
        if (liveSphere) {
          const score = Number(live.quality_score);
          const radius = Number(live.operational_radius);
          liveSphere.visible = live.available === true && Number.isFinite(radius) && radius > 0;
          if (liveSphere.visible) liveSphere.scale.setScalar(radius);
          const color = score >= 70 ? 0x36d483 : (score >= 40 ? 0xf5a524 : 0xff5f64);
          liveSphere.material.color.setHex(color);
          const legend = document.getElementById('handDemoLiveLegend');
          if (legend) legend.style.setProperty('--legend-color', `#${color.toString(16).padStart(6, '0')}`);
        }
        const epsilon = Number(analytic.epsilon);
        const liveText = live.available === true ? `Live ${Number(live.quality_score).toFixed(1)} · ${humanReason(live.state)}` : 'Live update unavailable';
        const davis = payload.davis || {};
        const davisPreClose = String(davis.current_stage || 'PRE_CLOSE').toUpperCase() === 'PRE_CLOSE';
        const davisText = davisPreClose
          ? 'DAVIS armed; scoring starts after grasp'
          : (davis.used_for_live_quality === true ? 'DAVIS object motion used' : `DAVIS excluded: ${humanReason(davis.exclusion_reason || 'not current')}`);
        const analyticPrefix = status === 'STALE' ? 'Last analytic ε' : 'Analytic ε';
        const staleNotice = status === 'STALE' ? ' · BASELINE INVALID - RESET / NEW LOCK' : '';
        setText('handDemoWrenchCaption', `${analyticPrefix} ${Number.isFinite(epsilon) ? epsilon.toFixed(3) : 'N/A'}${staleNotice} · ${liveText} · ${davisText}`);
        renderQualityWrenchViewer();
      } catch (error) {
        shell.classList.add('unavailable');
        shell.dataset.status = 'RENDERER_UNAVAILABLE';
        setQualityWrenchFallback('3D quality renderer unavailable', String(error.message || error), 'The numeric quality state remains available below.');
      } finally {
        qualityWrenchViewer.updating = false;
      }
    }
    function updateHandGraspDemo(data, rs, qm, dm, lq, pc, qualityLabel, rawEpsilon, forceClosure, contactSourceLabel) {
      const hand = data.hand_prototype || rs || {};
      const realsenseLive = Boolean(data.running?.realsense && rs.overlay_stale !== true);
      syncHandOffsetControls(rs);
      const qualityDiagram = data.grasp_quality_diagram || {};
      updateQualityWrenchDiagram(qualityDiagram);
      const targetStatus = String(rs.target_status || rs.state || '').toUpperCase();
      const bottleDetected = rs.bottle_detected === true || rs.detected_bottle_count > 0;
      const targetTracking = ['TRACKING', 'LOCKED', 'TARGET_LOCKED', 'TRACKING_UNCERTAIN'].includes(targetStatus);
      const targetLost = ['LOST', 'TARGET_LOST', 'RELOCK_REQUIRED'].includes(targetStatus);
      const operatorTargetPhase = String(rs.operator_target_phase || '').toUpperCase();
      const autoReacquiring = rs.operator_reacquisition_active === true || operatorTargetPhase === 'AUTO_REACQUIRING';
      const waitingForOneCandidate = rs.operator_reacquisition_waiting === true || operatorTargetPhase === 'WAITING_FOR_ONE_STABLE_CANDIDATE';
      const stickyWaiting = hand.sticky_lock_waiting_reacquisition === true
        || rs.sticky_lock_waiting_reacquisition === true
        || (rs.target_lock || {}).sticky_lock_waiting_reacquisition === true;
      const bottleState = stickyWaiting
        ? 'LOCKED · WAITING FOR BOTTLE'
        : autoReacquiring
        ? 'AUTO REACQUIRING'
        : (waitingForOneCandidate
          ? 'WAITING FOR ONE STABLE CANDIDATE'
          : (operatorTargetPhase || (targetLost ? 'LOST' : (targetTracking ? 'TRACKING' : (bottleDetected ? 'DETECTED' : 'SEARCHING')))));
      const bandLocked = hand.grasp_band_locked === true || rs.grasp_band_locked === true;
      const bandStale = hand.grasp_band_stale === true || rs.grasp_band_stale === true;
      const bandHeld = hand.grasp_band_held === true || rs.grasp_band_held === true;
      const relockRequired = hand.grasp_band_relock_required === true || rs.grasp_band_relock_required === true;
      const projectionMode = String(hand.locked_projection_mode || rs.locked_projection_mode || '').toUpperCase();
      let bandState = bandLocked
        ? (dashboardMode === 'engineering' ? 'LOCKED CURRENT 3D' : 'LOCKED · 3D GUIDANCE')
        : 'NOT LOCKED';
      if (projectionMode === '2D_FALLBACK') bandState = '2D_FALLBACK · NOT READY';
      else if (stickyWaiting) bandState = 'LOCKED · WAITING FOR BOTTLE · NOT READY';
      else if (bandHeld) bandState = dashboardMode === 'engineering'
        ? 'LOCKED · HELD 3D'
        : 'LOCKED · HELD GUIDANCE';
      else if (projectionMode === 'INVALID_3D') bandState = 'LOCKED · WAITING FOR DEPTH';
      else if (autoReacquiring) bandState = 'AUTO REACQUIRING';
      else if (waitingForOneCandidate) bandState = 'WAITING FOR ONE STABLE CANDIDATE';
      else if (relockRequired) bandState = 'RELOCK REQUIRED';
      else if (bandStale) bandState = 'LOW CONFIDENCE';
      const guidance = rs.move
        ? `X ${Math.round(Number(rs.move.x || 0))} · Y ${Math.round(Number(rs.move.y || 0))} · Z ${Math.round(Number(rs.move.z || 0))} mm`
        : (rs.guidance_label || 'Waiting for valid depth guidance');
      const diagramStatus = String(qualityDiagram.status || '').toUpperCase();
      const matrixStatus = String(qm.quality_matrix_status || '').toUpperCase();
      const bundleStatus = String(qm.quality_bundle_status || (data.quality_bundle || {}).quality_bundle_status || '').toUpperCase();
      const qualityArtifactStale = Boolean(
        qm.quality_label_stale
        || qm.matrix_stale_due_to_motion
        || qm.matrix_valid_for_current_pose === false
        || ['STALE', 'STALE_NEEDS_RECOMPUTE'].includes(matrixStatus)
        || bundleStatus === 'STALE'
        || diagramStatus === 'STALE'
        || qualityDiagram.analytic?.stale === true
      );
      const analyticInvalid = ['FAILED', 'INVALID'].includes(matrixStatus) || bundleStatus === 'FAILED';
      const analyticReady = bandLocked && !analyticInvalid && (qualityDiagram.analytic?.available === true || Boolean(
        rawEpsilon !== undefined && rawEpsilon !== null
        && ['UPDATED', 'OK', 'VALID', 'CURRENT', 'STALE', 'STALE_NEEDS_RECOMPUTE'].includes(matrixStatus)
      ));
      const qualityStale = bandLocked && qualityArtifactStale;
      const analyticState = !bandLocked ? 'WAITING FOR LOCK' : (!analyticReady ? 'NOT READY' : (qualityStale ? 'INVALID · RESET / NEW LOCK' : 'CURRENT'));
      const operatorBottleState = realsenseLive ? bottleState : 'OFFLINE';
      const operatorBandState = realsenseLive ? bandState : 'UNAVAILABLE';
      const operatorDistanceState = realsenseLive ? fmtOperatorDistance(rs) : 'N/A · REALSENSE OFFLINE';
      const operatorAnalyticState = realsenseLive ? analyticState : 'UNAVAILABLE';
      const operatorGuidance = realsenseLive ? guidance : 'N/A · REALSENSE OFFLINE';
      setText('handDemoRealSenseBadge', realsenseLive ? bottleState : 'OFFLINE');
      setHandDemoVisualState(
        'handDemoRealSenseVisual',
        Boolean(data.running?.realsense && (data.images?.overlay || {}).exists && rs.overlay_stale !== true),
        'RealSense overlay unavailable',
        rs.overlay_stale === true ? 'The last overlay is stale and is not shown as live.' : 'No current processed RGB-D overlay is available.',
        data.running?.realsense ? 'Wait for the next current frame.' : 'Start RealSense.'
      );
      cards(document.getElementById('handDemoRealSenseCards'), [
        ['Bottle', operatorBottleState],
        ['Grasp Band', operatorBandState],
        ['Distance', operatorDistanceState],
        ['Baseline Quality', operatorAnalyticState === 'CURRENT' ? qualityLabel : operatorAnalyticState],
        ['Guidance', operatorGuidance],
      ]);

      const eventStreamCurrent = Boolean(
        data.running?.davis_motion
        && dm.report_heartbeat_current !== false
        && !['NOT_RUNNING', 'UNAVAILABLE', 'DEVICE_NOT_FOUND', 'STREAM_NOT_STARTED'].includes(String(dm.stream || '').toUpperCase())
      );
      const eventCount = Number(dm.event_count || 0);
      const eventRate = Number(dm.event_rate || 0);
      const rawEventsActive = eventStreamCurrent && (eventCount > 0 || eventRate > 0);
      const objectMotionValid = dm.object_motion_valid === true;
      const graspZoneMotionValid = dm.grasp_zone_motion_valid === true;
      const presentationMotionValid = objectMotionValid || graspZoneMotionValid;
      const disturbance = String(dm.disturbance_type || 'none').toUpperCase();
      const objectMotionClass = String(dm.grasp_zone_motion_class || dm.object_motion_class || dm.motion_class || '').toUpperCase();
      const rawMotionClass = String(dm.raw_motion_class || dm.motion_class || '').toUpperCase();
      const rawMotionDisplay = presentationMotionValid
        ? humanReason(rawMotionClass || 'object motion')
        : (/LIGHTING|CAMERA|BACKGROUND|GLOBAL/.test(`${disturbance} ${rawMotionClass}`)
          ? humanReason(disturbance === 'NONE' ? rawMotionClass : disturbance)
          : (rawEventsActive ? 'EVENT ACTIVITY' : 'NO MOTION'));
      const roiSource = String(dm.roi_source || 'unavailable');
      const fullFrameOnly = roiSource.toLowerCase() === 'full_frame' || dm.full_frame_fallback === true;
      const uncalibratedRoi = dm.roi_projection_physically_validated === false || dm.roi_projection_uncalibrated === true;
      const davisUsed = presentationMotionValid && (lq.davis_closeup_used_for_live_quality === true || dm.davis_closeup_used_for_live_quality === true);
      const exclusionReason = dm.davis_closeup_exclusion_reason || dm.davis_motion_excluded_reason || dm.grasp_zone_motion_reason || dm.object_motion_reason || 'motion gate did not pass';
      const fixedCenterThirdDiagnostic = dm.fixed_center_third_diagnostic === true
        || String(dm.davis_target_context_mode || '').toLowerCase() === 'fixed_center_third'
        || String(dm.davis_target_roi_priority_selected || '').toLowerCase() === 'configured_center_third_diagnostic'
        || String(dm.roi_projection_method || '').toLowerCase() === 'fixed_center_third';
      const fixedCenterThirdView = dm.fixed_center_third_acquisition === true
        || dm.acquisition_roi_fixed === true
        || fixedCenterThirdDiagnostic;
      const roiLabel = fixedCenterThirdView
        ? (graspZoneMotionValid ? 'Fixed grasp zone · post-grasp active' : (dm.operator_roi_label || 'Fixed center grasp zone · armed'))
        : fullFrameOnly
        ? 'Full-frame event activity only'
        : (uncalibratedRoi ? `${humanReason(roiSource)} · object-specific ROI not calibrated` : humanReason(roiSource));
      const davisStage = String(dm.current_stage || dm.davis_motion_stage || 'PRE_CLOSE').toUpperCase();
      const presentationPreClose = dashboardMode !== 'engineering' && davisStage === 'PRE_CLOSE';
      setText('handDemoDavisBadge', !eventStreamCurrent ? 'UNAVAILABLE' : (davisStage === 'PRE_CLOSE' ? 'ARMED' : (presentationMotionValid ? 'MEASURING' : 'ACQUIRING')));
      setHandDemoVisualState(
        'handDemoDavisVisual',
        eventStreamCurrent && Boolean((data.images?.davis_heatmap || {}).exists),
        'DAVIS unavailable',
        eventStreamCurrent ? 'The stream is active but no current heatmap is ready.' : 'The canonical DAVIS event backend is not active.',
        eventStreamCurrent ? 'Move locally in view to produce events.' : 'Check USB permission, then start the close-up DAVIS monitor.'
      );
      cards(document.getElementById('handDemoDavisRawCards'), [
        ['Raw Events', presentationPreClose ? 'ARMED AFTER GRASP' : (rawEventsActive ? 'ACTIVE' : (eventStreamCurrent ? 'QUIET' : 'UNAVAILABLE'))],
        ['Event Rate', presentationPreClose ? 'POST-GRASP ONLY' : (eventStreamCurrent ? `${Math.round(eventRate)} ev/s` : 'N/A')],
        ['Disturbance', presentationPreClose ? 'NOT MONITORED YET' : humanReason(disturbance)],
      ]);
      const slipLikeText = presentationMotionValid
        ? (/SLIP/.test(objectMotionClass) ? `DETECTED · ${humanReason(objectMotionClass)}` : 'NO CURRENT CUE')
        : 'excluded';
      cards(document.getElementById('handDemoDavisObjectCards'), [
        ['Motion Signal', presentationPreClose ? 'ARMED' : (objectMotionValid ? 'OBJECT-LOCAL' : (graspZoneMotionValid ? 'GRASP-ZONE' : 'ACQUIRING'))],
        ['Slip-like Motion', presentationPreClose ? 'STARTS AFTER GRASP' : slipLikeText],
        ['Rotation', presentationPreClose ? 'STARTS AFTER GRASP' : (presentationMotionValid && dm.rotation_delta_deg !== undefined && dm.rotation_delta_deg !== null ? `${fmtNumber(dm.rotation_delta_deg, 1)} deg` : 'excluded')],
        ['Vibration', presentationPreClose ? 'STARTS AFTER GRASP' : (presentationMotionValid && dm.vibration_amplitude_px_rms !== undefined && dm.vibration_amplitude_px_rms !== null ? `${fmtNumber(dm.vibration_amplitude_px_rms, 1)} px RMS` : 'excluded')],
        ['Used for Live Quality', davisUsed ? 'YES' : 'NO'],
        ['Confidence', presentationPreClose ? 'N/A UNTIL GRASP' : (dm.motion_estimation_confidence !== undefined && dm.motion_estimation_confidence !== null ? `${Math.round(100 * Number(dm.motion_estimation_confidence))}%` : 'N/A')],
      ]);

      const analyticLabel = !bandLocked ? 'WAITING FOR LOCK' : (analyticReady ? String(qm.quality_label || qualityLabel || 'N/A').replace(/\s*\(STALE\)\s*/i, '') : 'NOT READY');
      const epsilonText = bandLocked && rawEpsilon !== undefined && rawEpsilon !== null ? fmtNumber(rawEpsilon, 3) : 'N/A';
      const closureText = !bandLocked ? 'N/A' : (forceClosure === true ? 'ESTIMATED YES' : (forceClosure === false ? 'ESTIMATED NO' : 'N/A'));
      cards(document.getElementById('handDemoAnalyticCards'), [
        ['Last Analytic Quality', analyticLabel],
        ['Status', analyticState],
        ['Epsilon', epsilonText],
        ['Force Closure', closureText],
        ['Quality Source', contactSourceLabel || 'planned bottle contact'],
        ['Action', !bandLocked ? 'Baseline will be computed once after lock' : (qualityStale ? 'Reset or acquire a new lock' : (analyticReady ? 'Monitor live stability' : 'Automatic baseline pending — hold the locked bottle steady'))],
      ]);
      const liveScore = lq.live_quality_score;
      const liveLabel = lq.live_quality_label || 'NOT READY';
      let davisContribution = 'DAVIS unavailable / excluded from live quality';
      if (eventStreamCurrent && davisUsed) {
        const risk = humanReason(lq.primary_risk || dm.motion_class || 'object motion');
        davisContribution = `DAVIS used for live quality: ${risk}`;
      } else if (eventStreamCurrent && /LIGHTING|CAMERA|BACKGROUND|GLOBAL/.test(`${disturbance} ${exclusionReason}`.toUpperCase())) {
        davisContribution = `DAVIS not used: global/lighting/camera disturbance`;
      } else if (eventStreamCurrent) {
        davisContribution = `DAVIS not used: ${humanReason(exclusionReason)}`;
      }
      const diagramPenalties = qualityDiagram.live?.penalties || {};
      const davisPenalty = Number(
        lq.live_quality_davis_penalty
        ?? ((Number(diagramPenalties.slip || 0)) + (Number(diagramPenalties.rotation || 0)) + (Number(diagramPenalties.vibration || 0)))
      );
      const davisAdjustmentText = presentationPreClose
        ? '0.0 points · ARMED'
        : (davisUsed ? `-${fmtNumber(davisPenalty, 1)} points` : '0.0 points · EXCLUDED');
      const baselineScore = lq.analytic_quality_score;
      const baselineDisplay = baselineScore !== undefined && baselineScore !== null ? fmtNumber(baselineScore, 1) : analyticLabel;
      setText('handDemoMatrixBaseline', !bandLocked ? 'WAITING FOR LOCK' : (qualityStale ? `INVALID · RESET / NEW LOCK · last ${baselineDisplay}` : baselineDisplay));
      setText('handDemoMatrixDavis', davisAdjustmentText);
      setText('handDemoMatrixDavisReason', presentationPreClose
        ? 'Scoring starts after grasp'
        : (davisUsed ? humanReason(lq.primary_risk || objectMotionClass || 'valid object motion') : humanReason(exclusionReason)));
      const operatorLiveQuality = !bandLocked
        ? 'WAITING FOR LOCK'
        : (liveScore !== undefined && liveScore !== null ? `${liveLabel} ${fmtNumber(liveScore, 1)}` : liveLabel);
      setText('handDemoMatrixLive', operatorLiveQuality);
      const davisStep = document.getElementById('handDemoMatrixDavisStep');
      if (davisStep) davisStep.className = `quality-adjustment-step ${davisUsed ? 'included' : 'excluded'}`;
      cards(document.getElementById('handDemoLiveCards'), [
        ['Live Quality', operatorLiveQuality],
        ['DAVIS Adjustment', davisAdjustmentText],
        ['Primary Risk', presentationPreClose ? 'Waiting for grasp' : humanReason(lq.primary_risk || 'none')],
        ['Trend', humanReason(lq.trend || 'N/A')],
        ['DAVIS Gate', presentationPreClose ? 'ARMED' : (davisUsed ? 'INCLUDED' : 'EXCLUDED')],
        ['DAVIS Contribution', presentationPreClose ? 'Scoring starts after grasp' : davisContribution],
      ]);
      const riskStrip = document.getElementById('handDemoQualityRisks');
      if (riskStrip) {
        const diagramLive = qualityDiagram.live || {};
        const diagramDavis = qualityDiagram.davis || {};
        const penalties = diagramLive.penalties || {};
        const eventsUsed = diagramDavis.used_for_live_quality === true;
        const riskCell = (label, value, title='') => `<div${title ? ` title="${esc(title)}"` : ''}><b>${esc(label)}</b><span>${esc(value)}</span></div>`;
        const slipRisk = presentationPreClose ? 'waiting for grasp' : (eventsUsed ? `${fmtNumber(penalties.slip, 1)} pts` : 'excluded');
        const rotationRisk = presentationPreClose ? 'waiting for grasp' : (eventsUsed ? `${fmtNumber(penalties.rotation, 1)} pts` : 'excluded');
        const rollText = diagramDavis.roll_valid === true ? `${fmtNumber(diagramDavis.roll_deg, 1)}°` : 'N/A';
        const vibrationRisk = presentationPreClose ? 'waiting for grasp' : (eventsUsed ? `${fmtNumber(penalties.vibration, 1)} pts` : 'excluded');
        const disturbanceText = presentationPreClose ? 'not monitored yet' : humanReason(diagramDavis.disturbance_type || 'none');
        riskStrip.innerHTML = [
          riskCell('Slip penalty', slipRisk, eventsUsed ? 'Gated live-quality penalty points from valid DAVIS object motion.' : diagramDavis.exclusion_reason),
          riskCell('Rotation / roll', `${rotationRisk} · ${rollText}`, diagramDavis.roll_status || 'Roll is not estimated without valid 3D orientation.'),
          riskCell('Vibration', vibrationRisk, eventsUsed ? 'Gated live-quality penalty points.' : diagramDavis.exclusion_reason),
          riskCell('Disturbance', disturbanceText, eventsUsed ? 'Object-local event evidence passed the existing gates.' : `Excluded from slip scoring: ${diagramDavis.exclusion_reason || 'not current'}`),
        ].join('');
      }
      const motionClass = String(dm.raw_motion_class || dm.motion_class || '').toUpperCase();
      let demoDecision = String(lq.decision || 'HOLD').replaceAll('_', ' ');
      if (!bandLocked) demoDecision = 'WAITING FOR LOCK';
      else if (!analyticReady || qualityStale) demoDecision = 'CANNOT CONFIRM';
      else if (!presentationPreClose && !objectMotionValid && /LIGHTING|CAMERA|BACKGROUND|GLOBAL/.test(`${disturbance} ${motionClass}`)) demoDecision = 'DISTURBANCE';
      else if (objectMotionValid && /SLIP/.test(motionClass)) demoDecision = 'SLIP-LIKE MOTION';
      else if (objectMotionValid && /ROTATION/.test(motionClass)) demoDecision = 'ROTATION WARNING';
      else if (objectMotionValid && /VIBRATION/.test(motionClass)) demoDecision = 'VIBRATION WARNING';
      else if (liveScore !== undefined && liveScore !== null && Number(liveScore) >= 70) demoDecision = 'STABLE';
      else if (liveScore !== undefined && liveScore !== null) demoDecision = 'DEGRADED';
      setText('handDemoDecision', demoDecision);
      setText('handDemoDecisionReason', !bandLocked
        ? 'Place one complete bottle in view and hold it steady for automatic lock confirmation.'
        : presentationPreClose
        ? 'DAVIS is armed; grasp the bottle before slip, rotation, and disturbance scoring begins.'
        : (lq.next_action || lq.live_quality_explanation || davisContribution));
      const decisionCard = document.getElementById('handDemoDecisionCard');
      if (decisionCard) decisionCard.className = `hand-demo-decision ${stateClass(demoDecision)}`;
      const timeline = document.getElementById('handDemoTimeline');
      const rows = Array.isArray(lq.timeline_recent) ? lq.timeline_recent.slice(-3).reverse() : [];
      if (timeline) timeline.innerHTML = rows.length
        ? rows.map(row => `<div><time>${esc(row.time_label || '')}</time><span>${esc(row.event_marker || row.decision || row.live_quality_label || 'update')}</span></div>`).join('')
        : '<div><time>Now</time><span>Waiting for live-quality history.</span></div>';
    }
    function updateHandPrototypeStatus(data, rs) {
      const hand = data.hand_prototype || rs || {};
      const enabled = hand.hand_mounted_camera_mode !== false;
      const banner = document.getElementById('handModeBanner');
      if (banner) banner.classList.toggle('hidden', !enabled);
      setText(
        'handModeLabel',
        dashboardMode === 'engineering'
          ? (hand.hand_mounted_mode_label || 'Hand-operated RealSense + DAVIS grasp-quality monitoring demo')
          : 'Hand-operated RealSense + DAVIS grasp-quality monitoring demo'
      );

      const stage = String(hand.hand_mounted_stage || rs.hand_mounted_stage || 'SEARCH_TARGET').toUpperCase();
      const bandLocked = hand.grasp_band_locked === true || rs.grasp_band_locked === true;
      const bandProjected = hand.grasp_band_projected === true || rs.grasp_band_projected === true;
      const bandStale = hand.grasp_band_stale === true || rs.grasp_band_stale === true;
      const bandHeld = hand.grasp_band_held === true || rs.grasp_band_held === true;
      const projectionUnavailable = hand.grasp_band_projection_unavailable === true || rs.grasp_band_projection_unavailable === true;
      const operatorBandState = String(hand.operator_band_state || rs.operator_band_state || '').toUpperCase();
      const currentBottleGuide = operatorBandState === 'CURRENT_BOTTLE_RELATIVE_GUIDE'
        || hand.operator_locked_fraction_visual_guide_current === true
        || rs.operator_locked_fraction_visual_guide_current === true;
      const relockRequired = hand.grasp_band_relock_required === true || rs.grasp_band_relock_required === true;
      const operatorTargetPhase = String(rs.operator_target_phase || '').toUpperCase();
      const autoReacquiring = rs.operator_reacquisition_active === true || operatorTargetPhase === 'AUTO_REACQUIRING';
      const waitingForOneCandidate = rs.operator_reacquisition_waiting === true || operatorTargetPhase === 'WAITING_FOR_ONE_STABLE_CANDIDATE';
      const stickyWaiting = hand.sticky_lock_waiting_reacquisition === true
        || rs.sticky_lock_waiting_reacquisition === true
        || (rs.target_lock || {}).sticky_lock_waiting_reacquisition === true;
      const visibility = hand.grasp_band_visibility ?? rs.grasp_band_visibility;
      const bodyFraction = hand.locked_body_fraction_center ?? rs.locked_body_fraction_center;
      const lockedParts = [bandLocked ? 'LOCKED' : 'NOT LOCKED'];
      if (currentBottleGuide) lockedParts.push('CURRENT BOTTLE GUIDE (DISPLAY)');
      if (bandProjected) lockedParts.push('PROJECTED');
      if (stickyWaiting) lockedParts.push('WAITING FOR BOTTLE');
      if (bandHeld) lockedParts.push('HELD ESTIMATE');
      if (projectionUnavailable) lockedParts.push('WAITING FOR DEPTH');
      if (bandStale) lockedParts.push('STALE');
      if (visibility !== undefined && visibility !== null) lockedParts.push(`${Math.round(Number(visibility) * 100)}% visible`);
      if (bodyFraction !== undefined && bodyFraction !== null) lockedParts.push(`body ${Math.round(Number(bodyFraction) * 100)}%`);

      const driftState = String(hand.band_drift_state || rs.band_drift_state || 'UNAVAILABLE').toUpperCase();
      const drift = hand.grasp_band_drift_px ?? rs.grasp_band_drift_px;
      const scale = hand.grasp_band_scale_change ?? rs.grasp_band_scale_change;
      const angle = hand.grasp_band_angle_change_deg ?? rs.grasp_band_angle_change_deg;
      const driftParts = [humanReason(driftState)];
      if (drift !== undefined && drift !== null) driftParts.push(`${Number(drift).toFixed(1)} px`);
      if (scale !== undefined && scale !== null) driftParts.push(`${Number(scale).toFixed(2)}× scale`);
      if (angle !== undefined && angle !== null) driftParts.push(`${Number(angle).toFixed(1)}°`);
      const relockReason = hand.grasp_band_relock_reason || rs.grasp_band_relock_reason;
      const lastRelock = hand.last_relock_event || rs.last_relock_event;

      setText('handPrototypeStage', humanReason(stage));
      setText('lockedBandStatus', lockedParts.join(' · '));
      setText('bandDriftStatus', driftParts.join(' · '));
      setText(
        'bandRelockStatus',
        autoReacquiring
          ? 'AUTOMATIC · keep one bottle visible and hold steady'
          : waitingForOneCandidate
          ? 'WAITING · keep exactly one stable bottle candidate visible'
          : relockRequired
          ? `REQUIRED · ${humanReason(relockReason || 'reason unavailable')}`
          : (lastRelock && lastRelock.event_type === 'RELOCK'
            ? `RELOCKED · generation ${lastRelock.lock_generation ?? hand.lock_generation ?? 'N/A'} · ${humanReason(lastRelock.reason || lastRelock.trigger || 'explicit transition')}`
            : (bandLocked ? 'NOT REQUIRED' : 'LOCK REQUIRED'))
      );

      const setCardClass = (id, cls) => {
        const card = document.getElementById(id);
        if (!card) return;
        card.classList.remove('ok', 'warn', 'bad', 'info');
        card.classList.add('hand-band-item', cls);
      };
      setCardClass('handStageCard', /RELOCK|LOST|ABORT/.test(stage) ? 'bad' : (stage === 'SEARCH_TARGET' ? 'warn' : 'ok'));
      setCardClass('lockedBandCard', relockRequired || bandStale ? 'bad' : (currentBottleGuide ? 'info' : (bandLocked ? 'ok' : 'warn')));
      setCardClass('bandDriftCard', /RELOCK|STALE/.test(driftState) ? 'bad' : (driftState === 'OK' ? 'ok' : 'warn'));
      setCardClass('bandRelockCard', relockRequired ? 'bad' : (bandLocked ? 'ok' : 'warn'));
    }
    function updateWorkflowStepper(rs, qm, dm, pc) {
      const steps = ['Detect','Lock Band','Approach','Close Event Monitor','Pickup Monitor'];
      const stage = String(rs.hand_mounted_stage || rs.state || 'SEARCH_TARGET').toUpperCase();
      const bandLocked = rs.grasp_band_locked === true;
      let activeIndex = 0;
      if (stage === 'POST_GRASP_MONITORING' || rs.post_grasp_active === true) activeIndex = 4;
      else if (stage === 'CLOSE_PROXIMITY_TRACKING' || stage === 'READY_TO_GRASP' || rs.close_proximity_mode_active === true) activeIndex = 3;
      else if (stage === 'APPROACH_TRACKING' || rs.approach_mode_active === true || (operatorApproachPhaseMarked && bandLocked)) activeIndex = 2;
      else if (stage === 'LOCK_GRASP_BAND' || bandLocked) activeIndex = 1;
      const blockedIndex = stage === 'RELOCK_REQUIRED' ? 1 : (stage === 'TARGET_LOST' ? 0 : (/ABORT|UNSAFE/.test(stage) ? activeIndex : -1));
      const el = document.getElementById('workflowStepper');
      if (!el) return;
      el.innerHTML = steps.map((step, i) => {
        let cls = i < activeIndex ? 'done' : (i === activeIndex ? 'active' : '');
        if (i === blockedIndex) cls = 'blocked';
        return `<div class="workflow-step ${cls}">${step}</div>`;
      }).join('');
    }
    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value ?? 'N/A';
    }
    function updateFinalDemoSummary(data, rs, qm, dm, pc, benchmark) {
      const graspState = rs.decision || rs.state || rs.target_status || (data.running?.realsense ? 'RealSense running' : 'RealSense stopped');
      const qualityStatus = qm.quality_label_stale || qm.matrix_stale_due_to_motion
        ? 'Recompute required'
        : (qm.quality_matrix_status || qm.quality_label || 'waiting');
      const pcStatus = pc.freshness || pc.status || (pc.active ? 'active' : 'waiting');
      const davisStatus = data.running?.davis_motion ? (dm.motion_class || dm.stream || 'active') : 'optional';
      setText('finalDemoStatus', humanReason(graspState));
      setText('finalQualityStatus', humanReason(qualityStatus));
      setText('finalBenchmarkStatus', humanReason(pcStatus));
      setText('finalPresentationStatus', humanReason(davisStatus));
    }
    function updateOperatorShell(data, rs, qm, dm, cq, pc, sync, lq, emb, qualityLabel, rawEpsilon, forceClosure, contactStatus, contactSourceLabel, liveFps, overlayFps, frameAge, pcDisplayStatus) {
      const lqDisplay = lq.display || {};
      const liveText = lqDisplay.live_quality || liveQualityScoreText(lq.live_quality_score, lq.live_quality_label);
      const objectState = data.object_state || rs.object_state || {};
      const targetSelection = data.multi_bottle_target_selection || rs.multi_bottle_target_selection || {};
      const selectedTarget = (targetSelection.candidates || []).find(candidate => candidate.selected) || {};
      const poseClass = objectState.pose_class || objectState.orientation || 'UNKNOWN_ORIENTATION';
      const poseConfidence = objectState.pose_confidence ?? objectState.standing_confidence;
      const poseText = `${humanReason(poseClass)}${poseConfidence !== undefined && poseConfidence !== null ? ` (${Math.round(Number(poseConfidence) * 100)}%, camera frame)` : ' (camera frame)'}`;
      const feasibilityLabel = rs.grasp_feasibility_label || targetSelection.grasp_feasibility_label || targetSelection.selection_status || 'NOT TESTED';
      const feasibilityReason = rs.grasp_feasibility_reason || targetSelection.grasp_feasibility_reason || 'current grasp feasibility has not been evaluated';
      const feasibilityBlocked = String(feasibilityLabel).includes('CANNOT') || String(feasibilityLabel).includes('HOLD');
      const qualityApplicability = qm.quality_applicability || 'NOT_TESTED_UNREPORTED';
      const qualityApplicable = String(qualityApplicability).startsWith('APPLICABLE_');
      const qualityReason = qm.quality_not_applicable_reason || 'current quality applicability has not been established';
      const operatorQuality = qm.quality_label_stale || qm.matrix_stale_due_to_motion
        ? 'RESET / NEW LOCK REQUIRED'
        : (qualityApplicable ? qualityLabel : `N/A — ${humanReason(qualityReason)}`);
      const signedMove = value => {
        if (value === undefined || value === null || !Number.isFinite(Number(value))) return 'N/A';
        const rounded = Math.round(Number(value));
        return `${rounded > 0 ? '+' : ''}${rounded} mm`;
      };
      const forwardBackMove = value => {
        if (value === undefined || value === null || !Number.isFinite(Number(value))) return 'N/A';
        const rounded = Math.round(Number(value));
        if (rounded === 0) return 'HOLD';
        return `${Math.abs(rounded)} mm ${rounded < 0 ? 'forward / closer' : 'back / farther'}`;
      };
      const distanceConfidence = rs.distance_confidence_percent ?? (
        rs.distance_confidence !== undefined && rs.distance_confidence !== null
          ? Number(rs.distance_confidence) * 100
          : null
      );
      const distanceEvidence = rs.distance_measurement_valid === false
        ? `UNRELIABLE · ${humanReason(rs.distance_validity_reason || 'not available')}`
        : distanceConfidence !== null
          ? `${Math.round(Number(distanceConfidence))}% · ${humanReason(rs.coordinate_frame || 'camera_frame')}`
          : humanReason(rs.distance_validity_reason || 'not available');
      const closeDistance = fmtCloseDistance(rs);
      const operatorDistance = fmtOperatorDistance(rs);
      const depthRoiProjection = fmtDepthRoiProjection(rs);
      const candidateCount = Number(targetSelection.candidate_count || 0);
      const clutterBlockers = selectedTarget.blocking_target_ids || [];
      const clutterText = clutterBlockers.length
        ? `BLOCKED: ${humanReason(clutterBlockers.join(', '))}`
        : (candidateCount > 1 ? `${candidateCount} candidates; selected clearance ${humanReason(selectedTarget.grasp_clearance_status || 'not validated')}` : 'No adjacent candidate blocker reported');
      const liveDecision = String(lq.decision || lqDisplay.decision || '').toUpperCase();
      const unsafeLiveRisk = Boolean(
        lq.primary_risk
        && lq.primary_risk !== 'none'
        && /HOLD|REGRIP|DO_NOT|STOP|UNSAFE|INVALID/.test(liveDecision)
      );
      const decisionValue = unsafeLiveRisk
        ? (lq.decision || lqDisplay.decision || 'HOLD_AND_RECHECK')
        : (rs.ready_to_close
          ? (lq.decision || lqDisplay.decision || rs.ui_primary_status || rs.decision || rs.state || cq.combined_quality_status || 'WAITING')
          : (rs.ui_primary_status || rs.decision || rs.state || cq.combined_quality_status || 'NOT READY'));
      const decisionCard = document.getElementById('overallDecisionCard');
      if (decisionCard) decisionCard.className = `decision-card ${stateClass(decisionValue)}`;
      document.getElementById('decisionState').textContent = String(decisionValue).replaceAll('_', ' ');
      const readyReason = unsafeLiveRisk
        ? `${lq.primary_risk}: ${lq.decision || 'monitor'}`
        : (feasibilityBlocked ? feasibilityReason : (rs.ready_reason || rs.not_ready_reason || rs.ui_ready_reason || rs.ui_quality_message || cq.reason || 'Waiting for live evidence.'));
      document.getElementById('decisionExplanation').textContent = humanReason(readyReason);
      document.getElementById('decisionNextAction').textContent = unsafeLiveRisk
        ? (lq.next_action || 'Hold and recheck motion evidence.')
        : (rs.suggested_operator_action || targetSelection.suggested_operator_action || rs.ui_primary_action || rs.action || dm.decision || 'Monitor current grasp state.');
      const gripWidth = rs.grip_width_mm ? `${Math.round(rs.grip_width_mm)} mm` : (pc.contact_width_3d_mm !== undefined && pc.contact_width_3d_mm !== null ? `${Math.round(Number(pc.contact_width_3d_mm))} mm` : 'N/A');
      const mainBlocker = humanReason(
        unsafeLiveRisk
          ? lq.primary_risk
          : (feasibilityBlocked ? feasibilityReason : (rs.not_ready_reason || rs.ready_reason || qm.quality_stale_reason || 'N/A'))
      );
      const primary = [
        ['Distance', operatorDistance, true],
        ['Move X', signedMove(rs.move?.x), true],
        ['Move Y', signedMove(rs.move?.y), true],
        ['Move Forward / Back', forwardBackMove(rs.move?.z), true],
        ['Guidance confidence', distanceEvidence, true],
        ['Grasp Feasibility', humanReason(feasibilityLabel), false],
        ['Grip Width', gripWidth, false],
        ['Analytic Quality', operatorQuality, false],
        ['Clutter', clutterText, false],
        ['Main Blocker', mainBlocker, false],
        ['Ready', compactBool(rs.ready_to_close), false],
        ['Target', rs.target_status || 'N/A', false],
        ['Object Pose', poseText, false],
        ['Close-Range ROI', depthRoiProjection, false],
        ['Contact', compactBool(rs.visual_contact_ready), false],
        ['Fingers', contactStatus, false],
        ['Quality Applicability', humanReason(qualityApplicability), false],
        ['Live Quality', liveText, false],
        ['Point Cloud', pcDisplayStatus, false],
        ['DAVIS', data.running.davis_motion ? `${davisHeadline(dm)} · ${humanReason(dm.current_stage || (rs.post_grasp_active ? 'POST_CLOSE_STABILITY' : 'PRE_CLOSE'))}` : 'OPTIONAL', false],
      ];
      const rail = document.getElementById('primaryMetricCards');
      if (rail) rail.innerHTML = primary.map(([k, v, showInPresentation]) => {
        const visibilityClass = showInPresentation ? '' : ' engineering-only';
        const keyClass = ` metric-${String(k).toLowerCase().replaceAll(/[^a-z0-9]+/g, '-').replaceAll(/^-|-$/g, '')}`;
        return `<div class="metric-pill ${stateClass(v)}${visibilityClass}${keyClass}"><b>${esc(k)}</b><span>${esc(v)}</span></div>`;
      }).join('');
      const evidence = [
        ['RealSense', data.running.realsense ? (rs.target_status || 'active') : 'stopped'],
        ['Grasp Contact', contactSourceLabel],
        ['Quality Matrix', qm.quality_label_stale || qm.matrix_stale_due_to_motion ? 'STALE' : (qm.quality_matrix_status || 'idle')],
        ['Point Cloud', pcDisplayStatus],
        ['DAVIS Motion', data.running.davis_motion ? (dm.motion_class || 'active') : 'optional'],
        ['Sync', sync.status || sync.sync_status || 'N/A'],
        ['FPS', `${fmtFps(overlayFps)} overlay / ${fmtFps(liveFps)} input`],
        ['Frame Age', frameAge !== undefined && frameAge !== null ? `${Number(frameAge).toFixed(1)} ms` : 'N/A'],
      ];
      const support = document.getElementById('sensorEvidenceRows');
      if (support) support.innerHTML = evidence.map(([k, v]) => `<div class="evidence-row"><b>${esc(k)}</b><span class="${stateClass(v)}">${esc(v)}</span></div>`).join('');
      updateLiveQualityPanel(lq);
      setChip('syncChip', 'Sync', sync.status || sync.sync_status || (pc && pc.active ? 'active' : 'waiting'));
      updateHandPrototypeStatus(data, rs);
      updateWorkflowStepper(rs, qm, dm, pc);
    }
    function openModalFromImg(id, title) {
      const img = document.getElementById(id);
      if (!img || !img.getAttribute('src')) return;
      document.getElementById('modalTitle').textContent = title || 'Plot';
      const src = img.getAttribute('src');
      document.getElementById('modalImg').src = src;
      const download = document.getElementById('modalDownload');
      if (download) download.href = src;
      document.getElementById('imageModal').classList.add('open');
    }
    function closeModal() {
      document.getElementById('imageModal').classList.remove('open');
      document.getElementById('modalImg').removeAttribute('src');
    }
    async function setDebugStreams(enabled) {
      await fetch('/api/debug-streams', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled})
      });
    }
    function clearDebugImages() {
      document.getElementById('imgRgb').removeAttribute('src');
      document.getElementById('imgDepth').removeAttribute('src');
    }
	    async function syncDebugStreams() {
	      const details = document.getElementById('debugViews');
	      debugViewsOpen = Boolean(details && details.open);
	      const rawViewSelected = isSectionActive('operatorSection') && (activeVisualTab === 'rgb' || activeVisualTab === 'depth');
	      const rawStreamsNeeded = Boolean(SENSOR_FEEDS_ALWAYS_ON || debugViewsOpen || rawViewSelected || heavyPanelVisible('diagnostics'));
	      await setDebugStreams(rawStreamsNeeded);
	      if (debugViewsOpen || heavyPanelVisible('diagnostics')) {
	        setImg('imgRgb', latestImages.rgb);
	        setImg('imgDepth', latestImages.depth);
	      } else {
	        clearDebugImages();
	      }
	    }
    function fmtMs(v) {
      if (v === null || v === undefined) return 'N/A';
      if (typeof v === 'object' && v.ms !== undefined) return `${v.ms} ms`;
      return `${Math.round(Number(v))} ms`;
    }
    function fileState(files, name) {
      const item = (files || {})[name] || {};
      if (!item.exists) return 'MISSING';
      const age = item.age_sec === null || item.age_sec === undefined ? 'N/A' : `${item.age_sec}s`;
      return `${item.state} (${age})`;
    }
	    function setPointcloudViewerStatus(text, cls='') {
	      const el = document.getElementById('pointcloudThreeStatus');
	      if (!el) return;
	      el.textContent = text;
	      el.className = `plot-caption ${cls}`;
	    }
	    function setPointcloudFallback(message, visible=true) {
	      const fallback = document.getElementById('pointcloudThreeFallback');
	      if (!fallback) return;
	      fallback.classList.toggle('visible', Boolean(visible));
	      const label = fallback.querySelector('span');
	      if (label && message) label.textContent = message;
	    }
	    function setPointcloudWatermark(message) {
	      const el = document.getElementById('pointcloudWatermark');
	      if (!el) return;
	      el.textContent = message || '';
	      el.classList.toggle('visible', Boolean(message));
	    }
	    function loadThree() {
	      if (window.THREE) return Promise.resolve(window.THREE);
	      if (threeLoadPromise) return threeLoadPromise;
	      threeLoadPromise = new Promise((resolve, reject) => {
	        const existing = document.querySelector('script[data-threejs-loader="1"]');
	        if (existing) {
	          existing.addEventListener('load', () => resolve(window.THREE));
	          existing.addEventListener('error', () => reject(new Error('Three.js could not load')));
	          return;
	        }
	        let idx = 0;
	        const tryNext = () => {
	          if (idx >= THREE_JS_CDN_URLS.length) {
	            reject(new Error('Three.js could not load'));
	            return;
	          }
	          const script = document.createElement('script');
	          script.src = THREE_JS_CDN_URLS[idx++];
	          script.async = true;
	          script.dataset.threejsLoader = '1';
	          script.onload = () => window.THREE ? resolve(window.THREE) : tryNext();
	          script.onerror = tryNext;
	          document.head.appendChild(script);
	        };
	        tryNext();
	        window.setTimeout(() => {
	          if (!window.THREE) reject(new Error('Three.js load timed out'));
	        }, 10000);
	      }).catch(err => {
	        threeLoadPromise = null;
	        throw err;
	      });
	      return threeLoadPromise;
	    }
	    function pointcloudCanvasSize() {
	      const host = document.getElementById('pointcloudThreeHost');
	      if (!host) return {width: 900, height: 600};
	      const rect = host.getBoundingClientRect();
	      return {
	        width: Math.max(320, Math.floor(rect.width || 900)),
	        height: Math.max(240, Math.floor(rect.height || 600)),
	      };
	    }
	    function resizePointcloudViewer() {
	      if (!pointcloudViewer.initialized || !pointcloudViewer.renderer || !pointcloudViewer.camera) return;
	      const {width, height} = pointcloudCanvasSize();
	      pointcloudViewer.renderer.setSize(width, height, false);
	      pointcloudViewer.camera.aspect = width / Math.max(1, height);
	      pointcloudViewer.camera.updateProjectionMatrix();
	      renderPointcloudViewer();
	    }
	    function resetPointcloudView() {
	      pointcloudViewer.rotationX = -0.18;
	      pointcloudViewer.rotationY = 0.52;
	      pointcloudViewer.distance = 1.2;
	      renderPointcloudViewer();
	    }
	    function setPointcloudColorMode(mode) {
	      pointcloudViewer.colorMode = mode || 'depth';
	      pointcloudViewer.lastTimestamp = null;
	      refreshPointcloudViewer({live_json_exists: true, live_json_url: '/api/pointcloud-live'});
	    }
	    function setPointcloudLayer(layer, visible) {
	      if (layer === 'scene') pointcloudViewer.showScene = Boolean(visible);
	      if (layer === 'bottle') pointcloudViewer.showBottle = Boolean(visible);
	      if (layer === 'contacts') pointcloudViewer.showContacts = Boolean(visible);
	      if (pointcloudViewer.scenePoints) pointcloudViewer.scenePoints.visible = pointcloudViewer.showScene;
	      if (pointcloudViewer.bottlePoints) pointcloudViewer.bottlePoints.visible = pointcloudViewer.showBottle;
	      if (pointcloudViewer.contactMarkers) pointcloudViewer.contactMarkers.visible = pointcloudViewer.showContacts;
	      if ((layer === 'scene' && visible && !pointcloudViewer.scenePoints) || (layer === 'bottle' && visible && !pointcloudViewer.bottlePoints)) {
	        pointcloudViewer.lastTimestamp = null;
	        pointcloudViewer.lastFetchMs = 0;
	        refreshPointcloudViewer({live_json_exists: true, live_json_url: '/api/pointcloud-live'});
	      }
	      renderPointcloudViewer();
	    }
	    function initPointcloudViewer() {
	      if (pointcloudViewer.initialized) return true;
	      if (!window.THREE) return false;
	      const canvas = document.getElementById('pointcloudThreeCanvas');
	      const host = document.getElementById('pointcloudThreeHost');
	      if (!canvas || !host) return false;
	      const THREE = window.THREE;
	      const {width, height} = pointcloudCanvasSize();
	      const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
	      renderer.setClearColor(0x080d16, 1);
	      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
	      renderer.setSize(width, height, false);
	      const scene = new THREE.Scene();
	      const camera = new THREE.PerspectiveCamera(45, width / Math.max(1, height), 0.01, 10);
	      camera.position.set(0, 0, pointcloudViewer.distance);
	      const group = new THREE.Group();
	      scene.add(group);
	      const ambient = new THREE.AmbientLight(0xffffff, 0.9);
	      scene.add(ambient);
	      const axes = new THREE.AxesHelper(0.18);
	      axes.material.depthTest = false;
	      axes.renderOrder = 10;
	      scene.add(axes);
	      pointcloudViewer.scene = scene;
	      pointcloudViewer.camera = camera;
	      pointcloudViewer.renderer = renderer;
	      pointcloudViewer.group = group;
	      pointcloudViewer.axes = axes;
	      pointcloudViewer.initialized = true;
	      host.addEventListener('pointerdown', (event) => {
	        pointcloudViewer.dragging = true;
	        pointcloudViewer.lastX = event.clientX;
	        pointcloudViewer.lastY = event.clientY;
	        host.setPointerCapture(event.pointerId);
	      });
	      host.addEventListener('pointermove', (event) => {
	        if (!pointcloudViewer.dragging) return;
	        const dx = event.clientX - pointcloudViewer.lastX;
	        const dy = event.clientY - pointcloudViewer.lastY;
	        pointcloudViewer.lastX = event.clientX;
	        pointcloudViewer.lastY = event.clientY;
	        pointcloudViewer.rotationY += dx * 0.008;
	        pointcloudViewer.rotationX += dy * 0.008;
	        pointcloudViewer.rotationX = Math.max(-1.45, Math.min(1.45, pointcloudViewer.rotationX));
	        renderPointcloudViewer();
	      });
	      host.addEventListener('pointerup', () => { pointcloudViewer.dragging = false; });
	      host.addEventListener('pointercancel', () => { pointcloudViewer.dragging = false; });
	      host.addEventListener('wheel', (event) => {
	        event.preventDefault();
	        const delta = Math.sign(event.deltaY) * 0.08;
	        pointcloudViewer.distance = Math.max(0.18, Math.min(4.0, pointcloudViewer.distance + delta));
	        renderPointcloudViewer();
	      }, {passive: false});
	      renderPointcloudViewer();
	      return true;
	    }
	    function renderPointcloudViewer() {
	      if (!pointcloudViewer.initialized) return;
	      const {camera, group, renderer, scene} = pointcloudViewer;
	      group.rotation.x = pointcloudViewer.rotationX;
	      group.rotation.y = pointcloudViewer.rotationY;
	      camera.position.set(0, 0, pointcloudViewer.distance);
	      camera.lookAt(0, 0, 0);
	      renderer.render(scene, camera);
	    }
	    function clamp01(v) { return Math.max(0, Math.min(1, Number(v) || 0)); }
	    function turboApprox(t) {
	      t = clamp01(t);
	      const r = clamp01(1.5 - Math.abs(4 * t - 3));
	      const g = clamp01(1.5 - Math.abs(4 * t - 2));
	      const b = clamp01(1.5 - Math.abs(4 * t - 1));
	      return [r, g, b];
	    }
	    function modeColor(mode, x, y, z, srcRgb, bounds) {
	      if (mode === 'rgb') {
	        const boost = 1.55;
	        return [
	          Math.max(0.16, clamp01((srcRgb[0] ?? 0.7) * boost)),
	          Math.max(0.16, clamp01((srcRgb[1] ?? 0.7) * boost)),
	          Math.max(0.16, clamp01((srcRgb[2] ?? 0.7) * boost)),
	        ];
	      }
	      if (mode === 'white') return [0.92, 0.95, 0.98];
	      const min = (bounds || {}).min || [-0.5, -0.5, 0.1];
	      const max = (bounds || {}).max || [0.5, 0.5, 1.2];
	      const axisVal = mode === 'height' ? -y : z;
	      const lo = mode === 'height' ? -(max[1] || 0.5) : (min[2] || 0.1);
	      const hi = mode === 'height' ? -(min[1] || -0.5) : (max[2] || 1.2);
	      return turboApprox((axisVal - lo) / Math.max(0.001, hi - lo));
	    }
	    function flatArrayToPointGeometry(positions, colors, center, useVertexColors=true, bounds=null) {
	      const THREE = window.THREE;
	      const count = Math.floor((positions || []).length / 3);
	      const xyz = new Float32Array(count * 3);
	      const rgb = new Float32Array(count * 3);
	      const cx = Number(center[0] || 0);
	      const cy = Number(center[1] || 0);
	      const cz = Number(center[2] || 0);
	      for (let i = 0; i < count; i += 1) {
	        const src = i * 3;
	        xyz[src] = Number(positions[src] || 0) - cx;
	        xyz[src + 1] = -(Number(positions[src + 1] || 0) - cy);
	        xyz[src + 2] = -(Number(positions[src + 2] || 0) - cz);
	        const rawX = Number(positions[src] || 0);
	        const rawY = Number(positions[src + 1] || 0);
	        const rawZ = Number(positions[src + 2] || 0);
	        const srcRgb = [Number((colors || [])[src] ?? 0.8), Number((colors || [])[src + 1] ?? 0.8), Number((colors || [])[src + 2] ?? 0.8)];
	        const c = useVertexColors
	          ? modeColor(pointcloudViewer.colorMode, rawX, rawY, rawZ, srcRgb, bounds)
	          : [1.0, 0.08, 0.86];
	        rgb[src] = c[0];
	        rgb[src + 1] = c[1];
	        rgb[src + 2] = c[2];
	      }
	      const geometry = new THREE.BufferGeometry();
	      geometry.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
	      geometry.setAttribute('color', new THREE.BufferAttribute(rgb, 3));
	      geometry.computeBoundingSphere();
	      return geometry;
	    }
	    function replacePointLayer(layerName, geometry, material) {
	      const group = pointcloudViewer.group;
	      const existing = pointcloudViewer[layerName];
	      if (existing) {
	        group.remove(existing);
	        existing.geometry.dispose();
	        existing.material.dispose();
	      }
	      const points = new window.THREE.Points(geometry, material);
	      pointcloudViewer[layerName] = points;
	      group.add(points);
	    }
	    function clearPointcloudObject(layerName) {
	      const existing = pointcloudViewer[layerName];
	      if (!existing) return;
	      pointcloudViewer.group.remove(existing);
	      existing.traverse?.((child) => {
	        if (child.geometry) child.geometry.dispose();
	        if (child.material) child.material.dispose();
	      });
	      pointcloudViewer[layerName] = null;
	    }
	    function addContactMarkers(payload, center) {
	      const THREE = window.THREE;
	      clearPointcloudObject('contactMarkers');
	      const contacts = (payload || {}).contacts || {};
	      const left = contacts.left_xyz_m;
	      const right = contacts.right_xyz_m;
	      if (!Array.isArray(left) || !Array.isArray(right) || left.length < 3 || right.length < 3) return;
	      const cx = Number(center[0] || 0);
	      const cy = Number(center[1] || 0);
	      const cz = Number(center[2] || 0);
	      function toVec(p) {
	        return new THREE.Vector3(Number(p[0] || 0) - cx, -(Number(p[1] || 0) - cy), -(Number(p[2] || 0) - cz));
	      }
	      const leftVec = toVec(left);
	      const rightVec = toVec(right);
	      const group = new THREE.Group();
	      const sphereGeom = new THREE.SphereGeometry(0.014, 14, 10);
	      const leftMat = new THREE.MeshBasicMaterial({color: 0xff32d6});
	      const rightMat = new THREE.MeshBasicMaterial({color: 0x38d9ff});
	      const leftSphere = new THREE.Mesh(sphereGeom.clone(), leftMat);
	      const rightSphere = new THREE.Mesh(sphereGeom.clone(), rightMat);
	      leftSphere.position.copy(leftVec);
	      rightSphere.position.copy(rightVec);
	      group.add(leftSphere);
	      group.add(rightSphere);
	      const lineGeom = new THREE.BufferGeometry().setFromPoints([leftVec, rightVec]);
	      const line = new THREE.Line(lineGeom, new THREE.LineBasicMaterial({color: 0xffffff, linewidth: 2}));
	      group.add(line);
	      pointcloudViewer.contactMarkers = group;
	      group.visible = pointcloudViewer.showContacts;
	      pointcloudViewer.group.add(group);
	    }
	    function updatePointcloudGeometry(payload) {
	      if (!pointcloudViewer.initialized || !window.THREE) return;
	      const THREE = window.THREE;
	      const center = ((payload || {}).bounds || {}).center || [0, 0, 0];
	      const bounds = (payload || {}).bounds || {};
	      const scene = (payload || {}).scene || {};
	      const bottle = (payload || {}).bottle || {};
	      const sceneCount = Number(scene.point_count || 0);
	      const bottleCount = Number(bottle.point_count || 0);
	      if (!sceneCount && !bottleCount) {
	        setPointcloudFallback('Point-cloud JSON is empty.', true);
	        setPointcloudViewerStatus('No points in sampled point-cloud JSON.', 'bad');
	        return;
	      }
	      if (sceneCount && pointcloudViewer.showScene) {
	        replacePointLayer(
	          'scenePoints',
	          flatArrayToPointGeometry(scene.positions || [], scene.colors || [], center, true, bounds),
	          new THREE.PointsMaterial({size: 0.011, vertexColors: true, transparent: true, opacity: 0.94, sizeAttenuation: true})
	        );
	      } else if (pointcloudViewer.scenePoints) {
	        pointcloudViewer.group.remove(pointcloudViewer.scenePoints);
	        pointcloudViewer.scenePoints.geometry.dispose();
	        pointcloudViewer.scenePoints.material.dispose();
	        pointcloudViewer.scenePoints = null;
	      }
	      if (bottleCount && pointcloudViewer.showBottle) {
	        replacePointLayer(
	          'bottlePoints',
	          flatArrayToPointGeometry(bottle.positions || [], bottle.colors || [], center, false, bounds),
	          new THREE.PointsMaterial({size: 0.024, vertexColors: true, transparent: true, opacity: 0.98, sizeAttenuation: true})
	        );
	      } else if (pointcloudViewer.bottlePoints) {
	        pointcloudViewer.group.remove(pointcloudViewer.bottlePoints);
	        pointcloudViewer.bottlePoints.geometry.dispose();
	        pointcloudViewer.bottlePoints.material.dispose();
	        pointcloudViewer.bottlePoints = null;
	      }
	      addContactMarkers(payload, center);
	      setPointcloudFallback('', false);
	      pointcloudViewer.lastTimestamp = payload.timestamp;
	      const span = (bounds || {}).max && (bounds || {}).min
	        ? Math.max(
	            Math.abs(Number(bounds.max[0] || 0) - Number(bounds.min[0] || 0)),
	            Math.abs(Number(bounds.max[1] || 0) - Number(bounds.min[1] || 0)),
	            Math.abs(Number(bounds.max[2] || 0) - Number(bounds.min[2] || 0))
	          )
	        : 0.7;
	      if (!pointcloudViewer.viewFitted) {
	        pointcloudViewer.distance = Math.max(0.32, Math.min(3.8, span * 2.4));
	        pointcloudViewer.viewFitted = true;
	      }
	      const src = (payload || {}).source || {};
	      const contacts = (payload || {}).contacts || {};
	      const rawScene = src.original_scene_point_count || scene.raw_point_count || sceneCount;
	      const rawBottle = src.original_bottle_point_count || bottle.raw_point_count || bottleCount;
	      const contactLabel = contacts.width_mm ? ` | grip ${contacts.width_mm} mm` : '';
	      const frameLabel = payload.coordinate_frame_type === 'current_camera_frame' ? 'Current RealSense camera frame' : (payload.coordinate_frame || 'camera frame');
	      const freshLabel = payload.pointcloud_freshness ? ` | ${payload.pointcloud_freshness}${payload.pointcloud_age_sec !== undefined && payload.pointcloud_age_sec !== null ? ` ${payload.pointcloud_age_sec}s` : ''}` : '';
	      setPointcloudWatermark(payload.viewer_watermark || '');
	      setPointcloudViewerStatus(`Scene ${sceneCount.toLocaleString()} shown (${Number(rawScene || 0).toLocaleString()} raw) | Bottle ${bottleCount.toLocaleString()} shown (${Number(rawBottle || 0).toLocaleString()} raw) | ${frameLabel} | ${pointcloudViewer.colorMode}${contactLabel}${freshLabel}`);
	      renderPointcloudViewer();
	    }
	    async function refreshPointcloudViewer(pc) {
	      const nowMs = Date.now();
	      if (nowMs - pointcloudViewer.lastFetchMs < POINTCLOUD_JSON_REFRESH_MS) return;
	      pointcloudViewer.lastFetchMs = nowMs;
	      if (!pc || !pc.live_json_exists || !pc.live_json_url) {
	        setPointcloudFallback('Waiting for sampled point-cloud JSON.', true);
	        setPointcloudViewerStatus('Waiting for sampled point-cloud JSON.');
	        return;
	      }
	      try {
	        await loadThree();
	        if (!initPointcloudViewer()) throw new Error('Point-cloud viewer could not initialize');
	        const response = await fetch(`${pc.live_json_url}?t=${Date.now()}`, {cache: 'no-store'});
	        if (!response.ok) throw new Error(`Point-cloud JSON HTTP ${response.status}`);
	        const payload = await response.json();
	        if (payload.timestamp === pointcloudViewer.lastTimestamp) {
	          renderPointcloudViewer();
	          return;
	        }
	        updatePointcloudGeometry(payload);
	      } catch (err) {
	        setPointcloudFallback('Three.js point cloud unavailable; showing preview image.', true);
	        setPointcloudViewerStatus(`Interactive point cloud unavailable: ${err.message || err}`, 'bad');
	      }
	    }
	    async function loadMultiviewModel() {
	      try {
	        const mode = (document.getElementById('multiviewDisplayMode') || {}).value || 'current_accumulated';
	        if (mode === 'current_frame') {
	          pointcloudViewer.lastTimestamp = null;
	          await refreshPointcloudViewer({live_json_exists: true, live_json_url: '/api/pointcloud-live'});
	          return;
	        }
	        await loadThree();
	        if (!initPointcloudViewer()) throw new Error('Point-cloud viewer could not initialize');
	        const response = await fetch(`/api/multiview-live?t=${Date.now()}`, {cache: 'no-store'});
	        if (!response.ok) throw new Error(`multi-view preview HTTP ${response.status}`);
	        const payload = await response.json();
	        payload.display_mode = mode;
	        if (mode === 'accumulated_bottle') payload.scene = {point_count: 0, positions: [], colors: []};
	        updatePointcloudGeometry(payload);
	        const recon = payload.reconstruction || {};
	        setPointcloudViewerStatus(`${mode.replaceAll('_', ' ')}: ${(recon.filtered_accumulated_point_count || 0).toLocaleString()} fused pts | keyframes ${recon.accepted_keyframe_count || 0}/${recon.rejected_keyframe_count || 0} | coverage ${recon.estimated_surface_coverage_percent || 0}%`);
	      } catch (err) {
	        setPointcloudViewerStatus(`Multi-view model unavailable: ${err.message || err}`, 'bad');
	      }
	    }
    async function refreshBenchmarkStatus(force=false) {
      const now = Date.now();
      if (!force && now - lastBenchmarkRefreshMs < BENCHMARK_POLL_INTERVAL_MS) return;
      if (benchmarkRefreshInFlight) return;
      benchmarkRefreshInFlight = true;
      try {
        const response = await fetch('/api/benchmark/status', {cache:'no-store'});
        if (!response.ok) throw new Error(`Benchmark status HTTP ${response.status}`);
        const benchmark = await response.json();
        lastBenchmarkRefreshMs = Date.now();
        renderBenchmarkLab(benchmark || {});
      } catch (err) {
        const msg = document.getElementById('benchmarkMessage');
        if (msg) msg.textContent = `Benchmark status unavailable: ${err.message || err}`;
        console.warn('Benchmark status refresh failed', err);
      } finally {
        benchmarkRefreshInFlight = false;
      }
    }
    async function refresh() {
      if (refreshInFlight) return;
      refreshInFlight = true;
      try {
      const statusUrl = dashboardMode === 'engineering'
        ? '/api/status?full=1'
        : '/api/status?compact=1';
      const data = await (await fetch(statusUrl, {cache:'no-store'})).json();
      latestOperatorStatus = data || {};
      refreshTick += 1;
      latestImages = data.images || {};
      document.getElementById('clock').textContent = new Date().toLocaleTimeString();
      const rs = data.realsense || {};
      const hand = data.hand_prototype || rs;
      const bandLocked = hand.grasp_band_locked === true || rs.grasp_band_locked === true;
      if (!bandLocked) operatorApproachPhaseMarked = false;
      if (hand.approach_mode_requested === true || rs.approach_mode_requested === true) {
        operatorApproachPhaseMarked = true;
      }
      const approachButton = document.getElementById('approachPhaseBtn');
      if (approachButton) {
        approachButton.textContent = operatorApproachPhaseMarked
          ? 'Approach Active — Move Hand Manually'
          : 'Start Approach Mode';
        approachButton.title = bandLocked
          ? 'Marks the operator approach phase; no automatic movement is commanded.'
          : 'Automatic lock and analytic baseline are pending; keep exactly one bottle visible and hold it steady.';
      }
      const graspStateButton = document.getElementById('graspStateBtn');
      const manualPostGraspActive = Boolean(
        rs.post_grasp_active
        || rs.gripper_closed_confirmed
        || rs.manual_gripper_closed
        || ['POST_GRASP_EVALUATION', 'READY_TO_LIFT', 'REGRIP', 'DO_NOT_LIFT'].includes(String(rs.state || '').toUpperCase())
      );
      if (graspStateButton) {
        const postGraspBaselineReady = rs.post_grasp_baseline_ready === true;
        graspStateButton.dataset.postGraspActive = manualPostGraspActive ? 'true' : 'false';
        graspStateButton.dataset.bandLocked = bandLocked ? 'true' : 'false';
        graspStateButton.dataset.postGraspBaselineReady = postGraspBaselineReady ? 'true' : 'false';
        graspStateButton.textContent = manualPostGraspActive
          ? 'Release / End Motion Test'
          : (!bandLocked ? 'Waiting for Bottle Lock' : (!postGraspBaselineReady ? 'Waiting for Baseline' : 'Start Post-Grasp Monitor'));
        graspStateButton.disabled = !manualPostGraspActive && (!bandLocked || !postGraspBaselineReady);
        graspStateButton.title = manualPostGraspActive
          ? 'End the current manually declared post-grasp motion test.'
          : (!bandLocked
            ? 'A current bottle lock is required first.'
            : (!postGraspBaselineReady
              ? `The immutable analytic baseline is still pending: ${humanReason(rs.post_grasp_baseline_reason || 'baseline pending')}`
              : 'Manual declaration only; this does not claim sensor-confirmed contact.'));
        graspStateButton.classList.toggle('danger', manualPostGraspActive);
      }
      const cq = data.combined_quality || {};
      const ra = ((data.profile || {}).realsense_audit) || {};
      const ot = rs.processed_overlay_timing || {};
      const qm = rs.quality_matrix || {};
      const qb = data.quality_bundle || {};
      const ch = data.camera_health || {};
      const files = data.final_files || {};
      const ds = data.debug_streams || {};
	      const fresh = data.fresh_lock_quality || {};
	      const dc = data.davis_quality_confirmation || {};
	      const pc = data.pointcloud || {};
	      const mv = data.multiview || {};
	      const sync = data.sensor_sync || {};
	      const lq = data.live_quality || {};
	      const emb = data.embodiment || rs.embodiment_status || {};
	      const obj = data.object_profiles || {};
	      const activeObj = obj.active_profile || {};
	      const dm = data.davis_motion || {};
	      const boundaryField = dm.boundary_event_field || {};
	      const simulatedForcePolicy = dm.simulated_force_command || {};
	      const objectState = data.object_state || dm.object_state || {};
	      const taskPlan = data.task_grasp_policy || {};
	      const bottleSelection = data.multi_bottle_target_selection || rs.multi_bottle_target_selection || {};
	      const selectedBottleCandidate = (bottleSelection.candidates || []).find(candidate => candidate.selected) || {};
	      const hw = data.hardware || {};
	      const desired = data.desired_processes || {};
	      const operatorVisible = isSectionActive('operatorSection');
	      const pointcloudVisible = heavyPanelVisible('pointcloud');
	      const multiviewVisible = heavyPanelVisible('multiview');
	      const davisVisible = heavyPanelVisible('davis');
	      const qualityVisible = heavyPanelVisible('quality');
	      const benchmarkVisible = heavyPanelVisible('benchmark');
	      const diagnosticsVisible = heavyPanelVisible('diagnostics');
	      const diagnosticsRefreshDue = diagnosticsVisible && (refreshTick % DIAGNOSTICS_REFRESH_DIVIDER === 0);
	      const pcAge = pc.age_sec;
	      const pcFreshness = String(pc.freshness || '').toUpperCase();
	      const pcFresh = pcFreshness ? !['STALE', 'UNAVAILABLE'].includes(pcFreshness) : (pcAge === undefined || pcAge === null ? Boolean(pc.active) : Number(pcAge) <= 5.0);
	      const pcDisplayStatus = pc.active ? (pcFresh ? (pcFreshness || pc.status || 'active') : 'STALE') : (pcFreshness || pc.status || 'waiting');
	      const qualityLabel = qualityDisplayLabel(qm);
      const rawQualityLabel = String(qm.quality_label || qm.quality_label_display || 'N/A').replace(/\s*\(STALE\)\s*/i, '').toUpperCase();
      const normalizedValue = qm.normalized_quality_score ?? qm.epsilon_score ?? null;
      const normalizedScore = normalizedValue !== undefined && normalizedValue !== null ? fmtScore(normalizedValue) : 'N/A';
      const rawEpsilon = qm.raw_epsilon ?? qm.epsilon ?? qm.epsilon_score_raw ?? null;
      const contactStatus = qm.contact_quality_gate_status || qm.contact_source_status || rs.finger_detection_status || 'WAITING_FOR_CONTACTS';
      const contactSource = qm.contact_source_label || qm.contact_source || qm.quality_input_source || rs.contact_source_label || rs.contact_source || rs.quality_input_source || 'planned_contact';
      const inputSource = qm.quality_input_source || contactSource;
      const inputLabel = displaySource(inputSource);
      const contactSourceLabel = displaySource(contactSource);
      const failureReason = qualityFailureReason(qm);
      const forceClosure = qm.force_closure_estimate ?? qm.epsilon_force_closure ?? qm.force_closure;
      const contactDepths = qm.contact_depths_mm || {};
      const liveFps = rs.live_fps ?? ra.rgbd_latest_frame_fps ?? ra.camera_emitted_fps ?? ra.color_callback_fps;
      const overlayFps = rs.overlay_fps ?? ra.evaluation_output_fps ?? ra.previous_processed_fps ?? ra.evaluation_requested_fps;
      const frameAge = rs.frame_age_ms ?? ot.frame_age_ms ?? ra.pending_frame_age_ms;
      const slowestStage = rs.slowest_stage ? `${rs.slowest_stage.name}: ${fmtMs(rs.slowest_stage.ms)}` : 'N/A';
      const epsilonMetricVisible = metricVisible(emb, 'two_contact_epsilon');
      setChip('realsenseChip', 'RealSense', data.running.realsense ? (rs.target_status || 'running') : 'stopped');
	      setChip('davisChip', 'DAVIS', data.running.davis_motion ? davisHeadline(dm) : 'optional');
      setChip('pointcloudChip', 'Point Cloud', pcDisplayStatus);
      updateFinalDemoSummary(data, rs, qm, dm, pc, data.benchmark || {});
      updateOperatorShell(data, rs, qm, dm, cq, pc, sync, lq, emb, qualityLabel, rawEpsilon, forceClosure, contactStatus, contactSourceLabel, liveFps, overlayFps, frameAge, pcDisplayStatus);
      updateHandGraspDemo(data, rs, qm, dm, lq, pc, qualityLabel, rawEpsilon, forceClosure, contactSourceLabel);
      updateEmbodimentSelector(emb);
      if (dashboardMode !== 'engineering') {
        // Presentation uses the small read-only operator contract. Hidden
        // engineering renderers and diagnostics are updated only on demand.
        setMainVisualImages(data);
        return;
      }
      const rsHw = hw.realsense || {};
      const rsSelected = rsHw.selected || {};
      const davisHw = hw.davis || {};
      const davisSelected = davisHw.selected || {};
      cards(document.getElementById('hardwareCards'), [
        ['RealSense Detected', rsSelected.model_name || rsHw.status || 'N/A'],
        ['RealSense Serial', rsSelected.serial_number || rsHw.configured_serial || 'auto'],
        ['RealSense USB', rsSelected.usb_speed || (rsHw.require_usb3 ? 'USB3 required' : 'N/A')],
        ['RealSense Desired', desired.realsense ? 'yes - reconnect enabled' : 'no'],
        ['RealSense Stream', data.running.realsense ? (rs.target_status || 'running') : (ch.pass ? 'topics available' : 'unavailable')],
        ['DAVIS Detected', davisSelected.model_name || davisHw.status || 'N/A'],
        ['DAVIS ID', davisSelected.serial_number || `${davisSelected.vendor_id || '152a'}:${davisSelected.product_id || '841a'}`],
        ['DAVIS Permission', davisSelected.permission_status || davisHw.status || 'N/A'],
        ['DAVIS Desired', desired.davis_motion ? 'yes - reconnect enabled' : 'no'],
        ['DAVIS Backend', davisHw.backend_name || dm.backend || 'N/A'],
      ]);
      const btnFresh = document.getElementById('freshLockQualityBtn');
      const freshRunning = fresh.fresh_lock_quality_status === 'running';
      if (btnFresh) btnFresh.disabled = freshRunning;
      const freshMsg = document.getElementById('freshLockQualityMessage');
      if (freshMsg) {
        const freshStatus = String(fresh.fresh_lock_quality_status || 'idle');
        freshMsg.textContent = freshRunning
          ? `Fresh Lock + Quality: ${fresh.fresh_lock_quality_step || 'running'}`
          : (fresh.fresh_lock_quality_error
            ? `Fresh Lock + Quality: ${freshStatus} - ${fresh.fresh_lock_quality_error}`
            : '');
      }
      statusRows(document.getElementById('streamMainStatus'), [
        ['State', rs.ui_primary_status || rs.state],
        ['Decision', rs.decision],
        ['Action', rs.ui_primary_action || rs.action],
        ['Ready', compactBool(rs.ready_to_close)],
        ['Active Target', activeObj.display_name || activeObj.object_id || rs.active_object_name || 'Blue Bottle'],
        ['Overlay', rs.overlay_health || ((files.overlay || {}).state) || 'N/A'],
        ['Fresh', fresh.fresh_lock_quality_status || 'idle'],
      ]);
      statusRows(document.getElementById('streamGeometryStatus'), [
        ['Distance', fmtCloseDistance(rs)],
        ['Close-Range ROI', fmtDepthRoiProjection(rs)],
        ['Move X/Y/Z', rs.move ? `${Math.round(rs.move.x || 0)}, ${Math.round(rs.move.y || 0)}, ${Math.round(rs.move.z || 0)} mm` : 'N/A'],
        ['Grip', fmtMm(rs.grip_width_mm)],
        ['Target', rs.target_status],
        ['Live FPS', fmtFps(liveFps)],
        ['Overlay FPS', fmtFps(overlayFps)],
        ['Frame Age', frameAge !== undefined && frameAge !== null ? `${Number(frameAge).toFixed(1)} ms` : 'N/A'],
      ]);
      statusRows(document.getElementById('streamContactStatus'), [
        ['Source', contactSourceLabel],
        ['Visual', rs.visual_contact_ready],
        ['Fingers', contactStatus],
        ['Finger RGB-D', fmtFingerBottleDepth(rs)],
        ['Skin Occlusion', fmtSkinOcclusion(rs)],
        ['Confidence', fmtContactConfidence(qm)],
        ['Stable', qm.contact_stable_frames ?? qm.finger_stable_frames ?? 'N/A'],
      ]);
      statusRows(document.getElementById('streamQualityStatus'), [
        ['Label', qualityLabel],
        ...embodimentMetricRows(emb, qm, rs, rawEpsilon, forceClosure).slice(0, 2),
        ...(epsilonMetricVisible ? [['Force', forceClosureText(forceClosure)]] : []),
        ['Current', qm.quality_label_stale || qm.matrix_stale_due_to_motion ? 'STALE' : (qm.quality_matrix_status || 'idle')],
        ['Reason', humanReason(qm.quality_current_mismatch_reasons || qm.matrix_stale_reasons || rs.ui_quality_message)],
      ]);
	      cards(document.getElementById('statusCards'), [
        ['State', rs.state],
        ['Decision', rs.decision],
        ['Action', rs.action],
        ['Active Target', activeObj.display_name || activeObj.object_id || rs.active_object_name || 'Blue Bottle'],
        ['Ready To Close', compactBool(rs.ready_to_close)],
        ['Target', rs.target_status],
        ['Distance', rs.bottle_distance_mm ? `${Math.round(rs.bottle_distance_mm)} mm` : 'N/A'],
        ['Visual Contact', rs.visual_contact_ready],
        ['Fingers', contactStatus],
        ['Analytic Quality', (lq.display || {}).analytic_quality || qualityLabel],
        ['Live Quality', (lq.display || {}).live_quality || 'N/A'],
        ['Live Delta', lq.delta_from_analytic !== undefined && lq.delta_from_analytic !== null ? Number(lq.delta_from_analytic).toFixed(1) : 'N/A'],
        ['Primary Risk', lq.primary_risk || 'N/A'],
        ['Quality Label', qualityLabel],
        ...embodimentMetricRows(emb, qm, rs, rawEpsilon, forceClosure).slice(0, 2),
        ['DAVIS Motion', (data.davis_motion || {}).motion_class || 'N/A'],
        ['Combined Quality', cq.combined_quality_status || 'N/A'],
	      ]);
	      setSensorBadge('rgbSensorState', rs.realsense_rgb_active ? ((files.rgb || {}).state || 'ACTIVE') : 'WAITING');
	      setSensorBadge('depthSensorState', rs.realsense_depth_active ? ((files.depth || {}).state || 'ACTIVE') : 'WAITING');
	      setSensorBadge('davisSensorState', data.running.davis_motion ? (dm.stream || dm.motion_class || 'ACTIVE') : 'OPTIONAL');
	      setSensorBadge('pointcloudSensorState', pcDisplayStatus);
	      statusRows(document.getElementById('rgbSensorRows'), [
	        ['Active', compactBool(rs.realsense_rgb_active)],
	        ['Topic FPS', fmtFps((data.camera_health || {}).color_fps)],
	        ['File', fileState(files, 'rgb')],
	        ['Stream', 'ON for demo'],
	      ]);
	      statusRows(document.getElementById('depthSensorRows'), [
	        ['Active', compactBool(rs.realsense_depth_active)],
	        ['Topic FPS', fmtFps((data.camera_health || {}).depth_fps)],
	        ['File', fileState(files, 'depth')],
	        ['Stream', 'ON for demo'],
	      ]);
	      statusRows(document.getElementById('davisSensorRows'), [
	        ['Running', compactBool(data.running.davis_motion)],
	        ['Backend', dm.backend || 'N/A'],
	        ['Stream', dm.stream || 'N/A'],
	        ['Event Rate', dm.event_rate ? `${Math.round(dm.event_rate)} ev/s` : '0 ev/s'],
	        ['Motion', dm.motion_class || 'N/A'],
	      ]);
	      statusRows(document.getElementById('pointcloudRows'), [
	        ['Freshness', pcDisplayStatus],
	        ['Coordinate Frame', pc.coordinate_frame_type === 'current_camera_frame' ? 'Current RealSense camera frame' : (pc.coordinate_frame || 'N/A')],
	        ['ROI Source', pc.bottle_roi_source || pc.bottle_pointcloud_roi_source || (pc.bottle_roi_available ? 'bbox' : 'unavailable')],
	        ['Crop Confidence', pc.bottle_crop_confidence !== undefined && pc.bottle_crop_confidence !== null ? `${Math.round(Number(pc.bottle_crop_confidence) * 100)}% (${pc.bottle_crop_status || 'N/A'})` : (pc.bottle_crop_status || 'N/A')],
	        ['Raw Scene Points', pc.scene_raw_point_count ?? pc.raw_point_count ?? 'N/A'],
	        ['Filtered Bottle Points', pc.bottle_filtered_point_count ?? pc.bottle_pointcloud_point_count ?? 'N/A'],
	        ['Valid Depth', pc.depth_valid_ratio !== undefined && pc.depth_valid_ratio !== null ? `${Math.round(Number(pc.depth_valid_ratio) * 100)}%` : 'N/A'],
	        ['Bottle Geometry', pc.cloud_valid_for_dimension_estimation ? fmtMmVector(pc.oriented_extents_mm || pc.bottle_pointcloud_extents_mm) : `not current/trusted (${pc.geometry_validation_reason || pc.bottle_crop_reason || 'N/A'})`],
	        ['3D Contacts', pc.cloud_valid_for_contact_validation ? (pc.contact_3d_quality || 'valid') : `not trusted (${pc.contact_3d_reason || pc.contact_3d_quality || 'N/A'})`],
	        ['Grip Width', pc.contact_width_3d_mm !== undefined && pc.contact_width_3d_mm !== null ? `${pc.contact_width_3d_mm} mm` : 'N/A'],
	      ]);
	      setPointcloudLink('scenePointcloudLink', pc.scene_ply_url, pc.scene_ply_exists);
	      setPointcloudLink('bottlePointcloudLink', pc.bottle_ply_url, pc.bottle_ply_exists);
	      setPointcloudLink('scenePointcloudLinkMain', pc.scene_ply_url, pc.scene_ply_exists);
	      setPointcloudLink('bottlePointcloudLinkMain', pc.bottle_ply_url, pc.bottle_ply_exists);
	      cards(document.getElementById('pointcloudStatusRows'), [
	        ['Freshness', pcDisplayStatus],
	        ['ROI Source', pc.bottle_roi_source || pc.bottle_pointcloud_roi_source || (pc.bottle_roi_available ? 'bbox' : 'unavailable')],
	        ['Scene Points', pc.scene_display_point_count ?? pc.scene_filtered_point_count ?? pc.point_count ?? 'N/A'],
	        ['Bottle Points', pc.bottle_display_point_count ?? pc.bottle_filtered_point_count ?? pc.bottle_pointcloud_point_count ?? 'N/A'],
	        ['Valid Depth', pc.depth_valid_ratio !== undefined && pc.depth_valid_ratio !== null ? `${Math.round(Number(pc.depth_valid_ratio) * 100)}%` : 'N/A'],
	        ['Bottle Dimensions', fmtMmVector(pc.bottle_pointcloud_extents_mm)],
	        ['3D Contact Validity', pc.cloud_valid_for_contact_validation ? (pc.contact_3d_quality || 'valid') : 'not current/trusted'],
	        ['Rolling Object', pc.rolling_pointcloud_mode || 'NOT_REGISTERED_DIAGNOSTIC'],
	        ['Rolling Window', pc.rolling_pointcloud_window_sec !== undefined && pc.rolling_pointcloud_window_sec !== null ? `${Number(pc.rolling_pointcloud_window_sec).toFixed(0)} s configured / ${Number(pc.rolling_pointcloud_observation_span_sec || 0).toFixed(1)} s retained` : 'N/A'],
	        ['Rolling Observations', pc.rolling_pointcloud_observation_count ?? 0],
	        ['Point Cloud FPS', fmtFps(pc.fps ?? rs.realsense_pointcloud_fps)],
	      ]);
	      const pcMessage = document.getElementById('pointcloudMessage');
	      if (pcMessage) {
	        pcMessage.textContent = !pcFresh
	          ? 'Point cloud stale - refresh required. Last image retained for reference.'
	          : (pc.error || pc.bottle_pointcloud_reason || (pc.active ? 'Point cloud preview is generated from aligned RealSense RGB-D.' : 'Point cloud unavailable: waiting for RealSense RGB-D and camera intrinsics.'));
	        pcMessage.className = `note ${stateClass(pc.error || pcDisplayStatus)}`;
	      }
	      cards(document.getElementById('pointcloudDiagnosticCards'), [
	        ['RGB Available', yesNo(pc.rgb_available)],
	        ['Depth Available', yesNo(pc.depth_available)],
	        ['Intrinsics Available', yesNo(pc.intrinsics_available)],
	        ['Preview Enabled', yesNo(pc.preview_enabled)],
	        ['Preview Status', pc.preview_status || 'N/A'],
	        ['Preview File', fileState(files, 'pointcloud_preview')],
	        ['Bottle Preview File', fileState(files, 'bottle_pointcloud_preview')],
	        ['Context File', fileState(files, 'pointcloud_context')],
	        ['Point JSON File', fileState(files, 'pointcloud_live_json')],
	        ['Frame ID', pc.pointcloud_frame_id ?? 'N/A'],
	        ['RGB Frame ID', pc.pointcloud_source_rgb_frame_id ?? 'N/A'],
	        ['Depth Frame ID', pc.pointcloud_source_depth_frame_id ?? 'N/A'],
	        ['Rolling Mode', pc.rolling_pointcloud_mode || 'NOT_REGISTERED_DIAGNOSTIC'],
	        ['Rolling Registration', pc.rolling_pointcloud_registration_status || 'NOT_REGISTERED_DIAGNOSTIC'],
	        ['Rolling Target', pc.rolling_pointcloud_target_id || 'N/A'],
	        ['Rolling Observations', pc.rolling_pointcloud_observation_count ?? 0],
	        ['Rolling Diagnostic Points', pc.rolling_pointcloud_point_count ?? 0],
	        ['Rolling Geometry Valid', yesNo(pc.rolling_pointcloud_valid_for_geometry)],
	        ['Rolling Contact Valid', yesNo(pc.rolling_pointcloud_valid_for_contact)],
	        ['Rolling Benchmark Valid', yesNo(pc.rolling_pointcloud_valid_for_benchmark)],
	        ['Rolling Last Reset', pc.rolling_pointcloud_last_reset_reason || 'N/A'],
	        ['Rolling Last Rejection', pc.rolling_pointcloud_last_rejection_reason || 'N/A'],
	        ['Raw Scene Points', pc.scene_raw_point_count ?? pc.raw_point_count ?? 'N/A'],
	        ['Filtered Scene Points', pc.scene_filtered_point_count ?? pc.filtered_point_count ?? pc.point_count ?? rs.realsense_pointcloud_point_count ?? 'N/A'],
	        ['Displayed Scene Points', pc.scene_display_point_count ?? pc.live_json_point_count ?? 'N/A'],
	        ['Raw Bottle Points', pc.bottle_raw_point_count ?? 'N/A'],
	        ['Filtered Bottle Points', pc.bottle_filtered_point_count ?? 'N/A'],
	        ['Displayed Bottle Points', pc.bottle_display_point_count ?? pc.live_json_bottle_point_count ?? 'N/A'],
	        ['Scene Max Points', pc.scene_display_max_points ?? 'N/A'],
	        ['Bottle Max Points', pc.bottle_display_max_points ?? 'N/A'],
	        ['Bottle Density', pc.bottle_pointcloud_density_mode || 'N/A'],
	        ['Bottle Crop Status', pc.bottle_crop_status || 'N/A'],
	        ['Bottle Crop Reason', pc.bottle_crop_reason || pc.bottle_pointcloud_reason || 'N/A'],
	        ['Mask Pixels', pc.bottle_mask_pixel_count ?? 'N/A'],
	        ['Valid Bottle Depth Pixels', pc.bottle_depth_valid_pixel_count ?? 'N/A'],
	        ['Crop Fraction', pc.bottle_crop_fraction_of_scene !== undefined && pc.bottle_crop_fraction_of_scene !== null ? Number(pc.bottle_crop_fraction_of_scene).toFixed(4) : 'N/A'],
	        ['Components', pc.bottle_connected_component_count ?? 'N/A'],
	        ['Dimensions Valid', yesNo(pc.bottle_dimensions_valid)],
	        ['Dimension Warning', pc.bottle_dimensions_warning || 'N/A'],
	        ['Point Density', pc.bottle_point_density ?? 'N/A'],
	        ['Compactness', pc.bottle_compactness ?? 'N/A'],
	        ['Left Contact XYZ', fmtMmVector(pc.left_contact_xyz_mm)],
	        ['Right Contact XYZ', fmtMmVector(pc.right_contact_xyz_mm)],
	        ['Contact On Cloud', yesNo(pc.contact_points_on_bottle_cloud)],
	        ['Visible', yesNo(pc.preview_visible)],
	        ['Preview Reason', pc.preview_reason || 'N/A'],
	        ['Outlier Filter', pc.preview_outlier_filter || 'N/A'],
	        ['Context Status', pc.context_status || 'N/A'],
	        ['Depth Range', pc.depth_min_mm !== undefined && pc.depth_min_mm !== null && pc.depth_max_mm !== undefined && pc.depth_max_mm !== null ? `${pc.depth_min_mm}-${pc.depth_max_mm} mm` : 'N/A'],
	        ['Preview Depth Window', pc.preview_depth_min_m !== undefined && pc.preview_depth_max_m !== undefined ? `${Number(pc.preview_depth_min_m).toFixed(2)}-${Number(pc.preview_depth_max_m).toFixed(2)} m` : 'N/A'],
	        ['Point Size', pc.point_size ?? 'N/A'],
	        ['Brightness', pc.brightness_boost ?? 'N/A'],
	        ['Preview ms', pc.preview_processing_ms !== undefined && pc.preview_processing_ms !== null ? `${pc.preview_processing_ms} ms` : 'N/A'],
	        ['Context ms', pc.context_processing_ms !== undefined && pc.context_processing_ms !== null ? `${pc.context_processing_ms} ms` : 'N/A'],
	        ['Point JSON ms', pc.live_json_processing_ms !== undefined && pc.live_json_processing_ms !== null ? `${pc.live_json_processing_ms} ms` : 'N/A'],
	        ['Processing ms', pc.processing_ms !== undefined && pc.processing_ms !== null ? `${pc.processing_ms} ms` : 'N/A'],
	        ['Worker Busy', yesNo(pc.worker_busy)],
	        ['Last Worker ms', pc.last_duration_ms !== undefined && pc.last_duration_ms !== null ? `${pc.last_duration_ms} ms` : 'N/A'],
	        ['Skipped Updates', pc.skipped_updates ?? 'N/A'],
	        ['Bottleneck', pc.bottleneck || 'N/A'],
	        ['Depth Ratio', pc.depth_valid_ratio !== undefined && pc.depth_valid_ratio !== null ? Number(pc.depth_valid_ratio).toFixed(3) : (pc.valid_depth_ratio !== undefined && pc.valid_depth_ratio !== null ? Number(pc.valid_depth_ratio).toFixed(3) : 'N/A')],
	        ['Bottle Depth Ratio', pc.bottle_pointcloud_valid_depth_ratio !== undefined && pc.bottle_pointcloud_valid_depth_ratio !== null ? Number(pc.bottle_pointcloud_valid_depth_ratio).toFixed(3) : 'N/A'],
	        ['Open3D Available', yesNo(pc.open3d_available)],
	        ['Scene PLY', fileState(files, 'latest_realsense_pointcloud_ply')],
	        ['Bottle PLY', fileState(files, 'latest_bottle_pointcloud_ply')],
	      ]);
	      cards(document.getElementById('multiviewCards'), [
	        ['Scan Status', mv.scan_status || 'idle'],
	        ['Pose Source', mv.pose_source || 'registration_unavailable'],
	        ['Accepted Keyframes', mv.accepted_keyframes ?? 0],
	        ['Rejected Keyframes', mv.rejected_keyframes ?? 0],
	        ['Model Points', mv.model_points ?? 0],
	        ['Coverage', mv.coverage_percent !== undefined && mv.coverage_percent !== null ? `${Number(mv.coverage_percent).toFixed(1)}%` : 'N/A'],
	        ['Registration Fitness', mv.registration_fitness !== undefined && mv.registration_fitness !== null ? Number(mv.registration_fitness).toFixed(3) : 'N/A'],
	        ['Reconstruction Age', mv.reconstruction_age_sec !== undefined && mv.reconstruction_age_sec !== null ? `${Number(mv.reconstruction_age_sec).toFixed(1)} s` : 'N/A'],
	      ]);
	      statusRows(document.getElementById('multiviewRows'), [
	        ['Current View Angle', mv.current_view_angle_deg !== undefined && mv.current_view_angle_deg !== null ? `${Number(mv.current_view_angle_deg).toFixed(1)} deg` : 'N/A'],
	        ['Missing Views', Array.isArray(mv.missing_view_directions) && mv.missing_view_directions.length ? mv.missing_view_directions.join(', ') : 'N/A'],
	        ['Recommended Motion', mv.recommended_camera_motion || 'Start scan after lock'],
	        ['Bottle Static', yesNo(mv.bottle_static)],
	        ['Requires Reset', yesNo(mv.requires_reset)],
	        ['Worker Busy', yesNo(mv.worker_busy)],
	        ['Last Rejection', mv.last_rejection_reason || 'N/A'],
	      ]);
	      const mvMessage = document.getElementById('multiviewMessage');
	      if (mvMessage) {
	        mvMessage.textContent = mv.requires_reset
	          ? 'Bottle moved - reset or start a new reconstruction.'
	          : (mv.last_rejection_reason || 'Accepted keyframes are fused into reconstruction_world. Use Show Accumulated Model to orbit previous views.');
	        mvMessage.className = `note ${stateClass(mv.requires_reset ? 'bad' : (mv.scan_status || 'idle'))}`;
	      }
	      cards(document.getElementById('multiviewDiagnosticCards'), [
	        ['Session', mv.session_id || 'N/A'],
	        ['Preview File', fileState(files, 'multiview_reconstruction_preview')],
	        ['Status File', fileState(files, 'multiview_reconstruction_status')],
	        ['Accumulated Scene PLY', fileState(files, 'multiview_accumulated_scene_ply')],
	        ['Accumulated Bottle PLY', fileState(files, 'multiview_accumulated_bottle_ply')],
	        ['Mesh PLY', fileState(files, 'multiview_reconstructed_mesh_ply')],
	        ['Raw Accumulated Points', mv.raw_accumulated_point_count ?? 'N/A'],
	        ['Fused Voxels', mv.fused_voxel_count ?? 'N/A'],
	        ['Covered Angle', mv.covered_view_angle_deg !== undefined && mv.covered_view_angle_deg !== null ? `${Number(mv.covered_view_angle_deg).toFixed(1)} deg` : 'N/A'],
	        ['RMSE', mv.registration_inlier_rmse_m !== undefined && mv.registration_inlier_rmse_m !== null ? `${Number(mv.registration_inlier_rmse_m).toFixed(4)} m` : 'N/A'],
	      ]);
      cards(document.getElementById('syncStatusCards'), [
        ['Sync Mode', sync.sync_mode || 'software_timestamp'],
        ['Sync Status', sync.sync_quality || 'waiting'],
        ['RealSense Age', sync.realsense_age_ms !== undefined && sync.realsense_age_ms !== null ? `${Number(sync.realsense_age_ms).toFixed(1)} ms (${sync.realsense_freshness || 'N/A'})` : 'N/A'],
        ['DAVIS Event Age', sync.davis_event_age_ms !== undefined && sync.davis_event_age_ms !== null ? `${Number(sync.davis_event_age_ms).toFixed(1)} ms (${sync.davis_event_freshness || 'N/A'})` : 'N/A'],
        ['DAVIS APS Age', sync.davis_aps_age_ms !== undefined && sync.davis_aps_age_ms !== null ? `${Number(sync.davis_aps_age_ms).toFixed(1)} ms (${sync.davis_aps_freshness || 'N/A'})` : 'N/A'],
        ['Point Cloud Age', sync.pointcloud_age_sec !== undefined && sync.pointcloud_age_sec !== null ? `${Number(sync.pointcloud_age_sec).toFixed(2)} s (${sync.pointcloud_freshness || 'N/A'})` : 'N/A'],
        ['Max Offset', sync.max_sensor_time_offset_ms !== undefined && sync.max_sensor_time_offset_ms !== null ? `${Number(sync.max_sensor_time_offset_ms).toFixed(1)} ms` : 'N/A'],
        ['Full Bundle For Live Quality', yesNo(sync.bundle_valid_for_live_quality || sync.valid_for_live_quality)],
        ['DAVIS For Live Quality', yesNo(sync.davis_events_valid_for_live_quality)],
        ['Point Cloud For Live Quality', yesNo(sync.pointcloud_valid_for_live_quality)],
        ['Used For Benchmark', yesNo(sync.bundle_valid_for_benchmark || sync.valid_for_benchmark)],
        ['Reason', sync.sync_reason || (Array.isArray(sync.sync_warnings) && sync.sync_warnings.length ? sync.sync_warnings.join(', ') : 'software sync latest-frame display')],
        ['RGB Active', compactBool(rs.realsense_rgb_active)],
        ['Depth Active', compactBool(rs.realsense_depth_active)],
        ['Point Cloud Active', compactBool(pc.active ?? rs.realsense_pointcloud_active)],
	        ['DAVIS Events Active', dm.stream || dm.backend || 'optional'],
        ['PointCloud Backend', pc.backend || rs.realsense_pointcloud_backend || 'N/A'],
        ['Point Count', pc.point_count ?? rs.realsense_pointcloud_point_count ?? 'N/A'],
        ['PointCloud FPS', fmtFps(pc.fps ?? rs.realsense_pointcloud_fps)],
        ['Bundle FPS', fmtFps(sync.bundle_fps)],
        ['Calibration', sync.calibration_status || 'approximate'],
        ['Mount', sync.mount_layout || 'unknown'],
      ]);
      // Object motion is only a trustworthy object-slip signal when the backend
      // gates say so. When it is global/event activity (lighting change, camera
      // shake, full-frame, too few events) the slip/rotation/vibration numbers
      // are event-activity RISK scores, not measured object motion, and must be
      // labelled as such so event health is never shown as object slip.
      const objMotionValid = dm.object_motion_valid === true;
      const activityLabel = String(dm.motion_activity_label || (objMotionValid ? 'OBJECT_MOTION' : 'EVENT_ACTIVITY')).replace(/_/g, ' ');
      const objMotionText = objMotionValid ? 'VALID' : `UNAVAILABLE — ${activityLabel}`;
      const riskNote = objMotionValid ? '' : ' risk';
      cards(document.getElementById('davisMotionSummaryCards'), [
        ['Backend', dm.backend || 'N/A'],
        ['APS State', dm.aps_state || dm.aps_stream_status || 'N/A'],
	        ['Event State', dm.event_state || dm.stream || 'N/A'],
	        ['Event Rate', dm.event_rate ? `${Math.round(dm.event_rate)} ev/s` : '0 ev/s'],
	        ['Grasp Stage', dm.current_stage || 'N/A'],
	        ['Quality Contribution', dm.davis_live_quality_contribution || 'excluded'],
	        ['RealSense Mapping Context', dm.realsense_mapping_context_current === true ? `current (${dm.realsense_status_age_ms ?? 'N/A'} ms)` : `suppressed: ${dm.realsense_mapping_context_reason || 'not current'}`],
	        ['Object Motion', objMotionText],
        ['Motion Class', dm.motion_class || 'N/A'],
        ['Object Displacement', (objMotionValid && dm.displacement_px !== undefined && dm.displacement_px !== null) ? `${Number(dm.displacement_px).toFixed(1)} px` : (objMotionValid ? (dm.motion_estimate_invalid_reason || 'N/A') : 'not object motion')],
        ['Object Rotation', (objMotionValid && dm.rotation_delta_deg !== undefined && dm.rotation_delta_deg !== null) ? `${Number(dm.rotation_delta_deg).toFixed(1)} deg` : (objMotionValid ? 'unavailable' : 'not object motion')],
        ['Object Roll', dm.roll_valid === true && dm.roll_delta_deg !== undefined && dm.roll_delta_deg !== null ? `${Number(dm.roll_delta_deg).toFixed(1)} deg` : String(dm.roll_status || 'NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION').replace(/_/g, ' ')],
        ['Object Vibration', (objMotionValid && dm.vibration_amplitude_px_rms !== undefined && dm.vibration_amplitude_px_rms !== null) ? `${Number(dm.vibration_amplitude_px_rms).toFixed(1)} px RMS` : (objMotionValid ? 'unavailable' : 'not object motion')],
        ['Confidence', dm.motion_estimation_confidence !== undefined && dm.motion_estimation_confidence !== null ? `${Math.round(Number(dm.motion_estimation_confidence) * 100)}%` : 'N/A'],
        ['Decision', dm.decision || 'N/A'],
        [`Incipient Slip${riskNote}`, dm.incipient_slip !== undefined && dm.incipient_slip !== null ? Number(dm.incipient_slip).toFixed(1) : 'N/A'],
        [`Gross Slip${riskNote}`, dm.gross_slip !== undefined && dm.gross_slip !== null ? Number(dm.gross_slip).toFixed(1) : 'N/A'],
        [`Rotation${riskNote}`, dm.rotation !== undefined && dm.rotation !== null ? Number(dm.rotation).toFixed(1) : 'N/A'],
        [`Vibration${riskNote}`, dm.vibration !== undefined && dm.vibration !== null ? Number(dm.vibration).toFixed(1) : 'N/A'],
        ['Stability', dm.stability !== undefined && dm.stability !== null ? Number(dm.stability).toFixed(1) : 'N/A'],
        ['ROI Source', dm.roi_source || 'N/A'],
      ]);
      const closeupMetrics = dm.closeup_window_metrics || {};
      const impactMetrics = closeupMetrics.impact || {};
      const settleMetrics = closeupMetrics.settle || {};
      const stabilityMetrics = closeupMetrics.stability || {};
      cards(document.getElementById('davisRawActivityCards'), [
        ['Stream Health', dm.event_health_valid === true ? 'VALID' : 'NOT CURRENT'],
        ['Raw Heatmap', dm.raw_event_heatmap_visible === true ? 'visible' : 'waiting'],
        ['Diagnostic Pattern', String(dm.diagnostic_movement_pattern || 'N/A').replace(/_/g, ' ')],
        ['Pattern Strength', dm.diagnostic_movement_pattern_score !== undefined && dm.diagnostic_movement_pattern_score !== null ? `${Number(dm.diagnostic_movement_pattern_score).toFixed(1)} / 100` : 'N/A'],
        ['Pattern Attribution', dm.diagnostic_movement_pattern_status || 'N/A'],
        ['Event Rate', `${Math.round(Number(dm.event_rate || 0))} ev/s`],
        ['Event Count', dm.event_count ?? 0],
        ['Active Pixels', dm.active_pixel_ratio !== undefined && dm.active_pixel_ratio !== null ? `${(100 * Number(dm.active_pixel_ratio)).toFixed(2)}%` : 'N/A'],
        ['Polarity + / -', `${dm.positive_event_count ?? 'N/A'} / ${dm.negative_event_count ?? 'N/A'}`],
        ['Polarity Balance', dm.polarity_balance !== undefined && dm.polarity_balance !== null ? Number(dm.polarity_balance).toFixed(3) : 'N/A'],
        ['Timeline Window', `${dm.event_window_ms ?? 'N/A'} ms rolling`],
      ]);
      cards(document.getElementById('davisObjectMotionCards'), [
        ['Validity', objMotionValid ? 'VALID' : 'INVALID'],
        ['Reason', objMotionValid ? 'ok' : (dm.object_motion_reason || dm.davis_closeup_exclusion_reason || 'no object evidence')],
        ['Fixed View', dm.fixed_center_third_acquisition === true ? 'middle third (constant)' : 'not configured'],
        ['Evidence ROI Source', dm.object_evidence_roi_source || dm.roi_source || 'N/A'],
        ['ROI Confidence', dm.roi_confidence !== undefined && dm.roi_confidence !== null ? Number(dm.roi_confidence).toFixed(3) : 'N/A'],
        ['Projection', dm.roi_projection_method || 'N/A'],
        ['dx / dy', objMotionValid ? `${dm.dx_px ?? 'N/A'} / ${dm.dy_px ?? 'N/A'} px` : 'excluded'],
        ['Rotation', objMotionValid && dm.rotation_delta_deg !== undefined && dm.rotation_delta_deg !== null ? `${Number(dm.rotation_delta_deg).toFixed(2)} deg` : 'excluded / risk-only'],
        ['Roll', dm.roll_valid === true && dm.roll_delta_deg !== undefined && dm.roll_delta_deg !== null ? `${Number(dm.roll_delta_deg).toFixed(2)} deg` : String(dm.roll_status || 'NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION').replace(/_/g, ' ')],
        ['Vibration', objMotionValid && dm.vibration_amplitude_px_rms !== undefined && dm.vibration_amplitude_px_rms !== null ? `${Number(dm.vibration_amplitude_px_rms).toFixed(2)} px RMS` : 'excluded'],
      ]);
      cards(document.getElementById('davisCloseupCards'), [
        ['Monitoring State', dm.davis_closeup_monitoring_state || 'not started'],
        ['Current Window', dm.davis_closeup_window || 'not started'],
        ['Elapsed', dm.closeup_action_elapsed_sec !== undefined && dm.closeup_action_elapsed_sec !== null ? `${Number(dm.closeup_action_elapsed_sec).toFixed(3)} s` : 'N/A'],
        ['Impact 0-300 ms', `${impactMetrics.event_count ?? 0} ev @ ${Math.round(Number(impactMetrics.event_rate || 0))}/s`],
        ['Settle 300-700 ms', `${settleMetrics.event_count ?? 0} ev @ ${Math.round(Number(settleMetrics.event_rate || 0))}/s`],
        ['Stability 700-1200 ms', `${stabilityMetrics.event_count ?? 0} ev @ ${Math.round(Number(stabilityMetrics.event_rate || 0))}/s`],
        ['Stability Decision', dm.davis_closeup_stability_decision || 'pending'],
        ['Used For Live Quality', dm.davis_closeup_used_for_live_quality === true ? 'yes' : `no: ${dm.davis_closeup_exclusion_reason || 'excluded'}`],
      ]);
      cards(document.getElementById('davisDisturbanceCards'), [
        ['Type', dm.disturbance_type || 'none'],
        ['Global Activity', dm.global_activity_score !== undefined && dm.global_activity_score !== null ? Number(dm.global_activity_score).toFixed(1) : 'N/A'],
        ['Lighting Change', dm.lighting_flicker_score !== undefined && dm.lighting_flicker_score !== null ? Number(dm.lighting_flicker_score).toFixed(1) : 'N/A'],
        ['Camera Shake', dm.camera_shake_score !== undefined && dm.camera_shake_score !== null ? Number(dm.camera_shake_score).toFixed(1) : 'N/A'],
        ['Local / Global Ratio', dm.local_to_global_event_ratio !== undefined && dm.local_to_global_event_ratio !== null ? Number(dm.local_to_global_event_ratio).toFixed(2) : 'N/A'],
        ['Excluded From Quality', dm.davis_closeup_used_for_live_quality === true ? 'no' : 'yes'],
        ['Exclusion Reason', dm.davis_closeup_exclusion_reason || 'none'],
        ['Projection Warning', dm.roi_projection_warning || 'none'],
      ]);
      const davisUnavailable = document.getElementById('davisUnavailableCard');
      if (davisUnavailable) {
        const davisActive = Boolean(data.running.davis_motion || dm.backend || dm.stream || dm.aps_state || dm.event_state);
        davisUnavailable.classList.toggle('hidden', davisActive);
        davisUnavailable.textContent = davisActive
          ? ''
          : 'DAVIS is unavailable. The final demo can still run RealSense-only, and this tab will show APS/events once the canonical DAVIS backend is active.';
      }
      renderObjectProfiles(obj);
      if (data.benchmark) renderBenchmarkLab(data.benchmark || {});
      if (benchmarkVisible) void refreshBenchmarkStatus(false);
      cards(document.getElementById('embodimentCards'), [
        ['Embodiment', emb.display_name || emb.active_embodiment || emb.embodiment_type || sync.embodiment || 'Parallel Two-Finger Gripper'],
        ['Release Status', emb.release_status || 'CONFIG_ONLY'],
        ['Release Mode', emb.release_mode || 'N/A'],
        ['Expected Contacts', emb.expected_contacts ?? emb.expected_contact_count ?? sync.expected_contact_count ?? 'N/A'],
        ['Detected Contacts', emb.detected_contacts ?? emb.detected_contact_count ?? 'N/A'],
        ['Contact Model', emb.contact_model_type || sync.contact_model_type || 'N/A'],
        ['Metric Applicability', emb.production_decisions_allowed ? 'production reference' : (emb.release_mode || 'display only')],
        ['Primary Metric', emb.primary_quality_metric || rs.primary_quality_metric || 'N/A'],
        ['Invalid Metrics Hidden', yesNo(emb.invalid_metrics_hidden)],
        ['Limitations', Array.isArray(emb.contact_model_limitations || emb.limitations) ? (emb.contact_model_limitations || emb.limitations).join('; ') : (emb.limitations || 'N/A')],
      ]);
      const resultEl = document.getElementById('qualityResult');
      const staleQuality = Boolean(qm.quality_label_stale || qm.matrix_stale_due_to_motion);
      const qualityStatusText = staleQuality ? 'BASELINE INVALID - RESET / NEW LOCK' : (qm.quality_matrix_status || qm.quality_bundle_status || qb.quality_bundle_status || 'CURRENT');
      resultEl.textContent = staleQuality ? (rawQualityLabel || 'LAST QUALITY') : (qualityLabel || 'N/A');
      resultEl.className = `quality-result ${staleQuality ? 'stale' : qualityResultClass(qualityLabel)}`;
      document.getElementById('qualityTypeNote').textContent = qualityIsPlanned(qm, inputSource, contactSource)
        ? 'Planned Grasp Preview - actual grasp quality is not confirmed yet.'
        : `${inputLabel} - contact-based quality estimate.`;
      const actionText = recommendedQualityAction(qm, rawQualityLabel, inputLabel, forceClosure, failureReason);
      const qualityMetricRows = embodimentMetricRows(emb, qm, rs, rawEpsilon, forceClosure);
      const gripMetric = qualityMetricRows.find(row => String(row[0] || '').toLowerCase().includes('grip width'));
      const contactMetric = qualityMetricRows.find(row => String(row[0] || '').toLowerCase().includes('opposing contacts'));
      qualityRows(document.getElementById('qualitySummaryRows'), [
        ['Last Quality', rawQualityLabel || 'N/A'],
        ['Status', qualityStatusText],
        ['Quality Meaning', qualityFriendlyLabel(qualityLabel)],
        ['Grip Width', gripMetric ? gripMetric[1] : 'N/A'],
        ['Opposing Contacts', contactMetric ? contactMetric[1] : 'N/A'],
        ...(epsilonMetricVisible ? [['Epsilon', normalizedScore]] : []),
        ...(epsilonMetricVisible ? [['Force Closure', forceClosureText(forceClosure)]] : []),
        ['Contact source', contactSourceLabel],
        ['Current action', actionText || rs.action || 'N/A'],
      ]);
      qualityRows(document.getElementById('qualityContactRows'), [
        ['Expected contacts', emb.expected_contacts ?? emb.expected_contact_count ?? qm.expected_contact_count ?? 'N/A'],
        ['Detected contacts', emb.detected_contacts ?? emb.detected_contact_count ?? qm.detected_contact_count ?? 'N/A'],
        ['Left contact source', displaySource(qm.left_contact_source || contactSource)],
        ['Right contact source', displaySource(qm.right_contact_source || contactSource)],
        ['Left / right depth', fmtDepthPair(contactDepths, rs)],
        ['Contact confidence', fmtContactConfidence(qm)],
        ['Stable frames', qm.contact_stable_frames ?? qm.finger_stable_frames ?? 'N/A'],
        ['Source confidence', qm.contact_source_confidence !== undefined && qm.contact_source_confidence !== null ? fmtNumber(qm.contact_source_confidence, 1) : 'N/A'],
        ['Contacts', `${qm.detected_contact_count ?? 'N/A'} / ${qm.expected_contact_count ?? 'N/A'}`],
        ['Source reason', qm.contact_source_validation_reason || qm.contact_source_failure_reason || 'N/A'],
        ['Intrinsics available', yesNo(qm.intrinsics_available)],
        ['3D contact inputs', qm.contact_3d_inputs_valid === true ? 'valid' : (qm.contact_3d_inputs_valid === false ? 'invalid' : 'N/A')],
        ['Release boundary', emb.production_decisions_allowed ? 'production reference' : (emb.display_only_notice || 'display/benchmark only')],
      ]);
      const actionEl = document.getElementById('qualityAction');
      actionEl.textContent = actionText;
      actionEl.className = `quality-message ${stateClass(actionText)}`;
      const failureEl = document.getElementById('qualityFailureMessage');
      failureEl.textContent = failureReason === 'none' ? 'No failure reason reported.' : `Failure reason: ${failureReason}`;
      failureEl.className = `quality-message ${failureReason === 'none' ? '' : 'bad'}`;
      const qualityBanner = document.getElementById('qualityGlobalBanner');
      if (qualityBanner) {
        const bundleStatus = String(qm.quality_bundle_status || qb.quality_bundle_status || '').toUpperCase();
        const staleReasons = qm.quality_current_mismatch_reasons || qm.matrix_stale_reasons || qb.stale_reasons || [];
        const staleReasonText = qm.quality_stale_reason || (Array.isArray(staleReasons) && staleReasons.length ? humanReason(staleReasons) : '');
        let bannerText = '';
        if (bundleStatus === 'STALE' || qm.quality_label_stale || qm.matrix_stale_due_to_motion) {
          bannerText = `Quality Stale - Recompute Required${staleReasonText ? ': ' + staleReasonText : ''}`;
        } else if (bundleStatus === 'FAILED') {
          bannerText = `Rendering Failed${failureReason && failureReason !== 'none' ? ': ' + failureReason : ''}`;
        } else if (bundleStatus === 'PARTIAL') {
          bannerText = 'Quality bundle partial - some artifacts failed to render';
        }
        qualityBanner.textContent = bannerText;
        qualityBanner.classList.toggle('visible', Boolean(bannerText));
      }
      document.getElementById('wrenchSpaceDetails')?.classList.toggle('hidden', !epsilonMetricVisible);
      cards(document.getElementById('diagnosticCards'), [
        ['Running', data.running.realsense],
        ['Camera Health', ch.pass === undefined ? 'N/A' : (ch.pass ? 'PASS' : 'FAIL')],
        ['Camera Error', ch.pass === false ? (ch.last_error || 'RealSense ROS topic unreadable') : 'none'],
        ['RealSense Discovery', `${rsHw.status || 'N/A'} - ${rsHw.reason || 'N/A'}`],
        ['DAVIS Discovery', `${davisHw.status || 'N/A'} - ${davisHw.reason || 'N/A'}`],
        ['DAVIS USB Node', davisSelected.device_node || 'diagnostic only / N/A'],
        ['Writer Session', rs.writer_session_id ? `${rs.writer_session_id.slice(0, 8)} #${rs.monotonic_frame_counter ?? 'N/A'}` : 'N/A'],
        ['Writer Warning', data.writer_warning || 'none'],
        ['Debug RGB Stream', ds.rgb ? `ON @ ${ds.debug_stream_fps || 1} fps` : 'OFF'],
        ['Debug Depth Stream', ds.depth ? `ON @ ${ds.debug_stream_fps || 1} fps` : 'OFF'],
        ['RGB File', fileState(files, 'rgb')],
        ['Depth File', fileState(files, 'depth')],
        ['Overlay File', fileState(files, 'overlay')],
        ['Overlay Health', rs.overlay_health || 'N/A'],
        ['Overlay Age', rs.dashboard_overlay_file_age_sec !== undefined && rs.dashboard_overlay_file_age_sec !== null ? `${rs.dashboard_overlay_file_age_sec}s` : 'N/A'],
        ['Overlay Error', rs.overlay_error || 'none'],
        ['Status File', fileState(files, 'realtime_status')],
        ['Grip Width', rs.grip_width_mm ? `${Math.round(rs.grip_width_mm)} mm` : 'N/A'],
        ['Move X/Y/Z', rs.move ? `${Math.round(rs.move.x || 0)}, ${Math.round(rs.move.y || 0)}, ${Math.round(rs.move.z || 0)} mm` : 'N/A'],
        ['Live FPS', fmtFps(liveFps)],
        ['Processed FPS', fmtFps(overlayFps)],
        ['FPS Source', rs.fps_counter_source || 'N/A'],
        ['Overlay Mode', ot.mode || ra.evaluation_branch || 'N/A'],
        ['Tracking Source', rs.tracking_source || 'N/A'],
        ['Bottle Identity Lock', rs.identity_tracking_ok === true ? 'CURRENT' : (rs.identity_tracking_ok === false ? 'LOST / UNCERTAIN' : 'N/A')],
        ['Identity Evidence', rs.identity_measurement_current === true ? `CURRENT (${rs.identity_tracking_source || 'RGB-D'})` : (rs.identity_measurement_current === false ? `HELD ${rs.identity_measurement_age_frames ?? '?'} frame(s) (${rs.identity_tracking_source || 'RGB-D'})` : (rs.identity_tracking_source || 'N/A'))],
        ['Original Target Hold', rs.identity_occlusion_hold_active === true ? `PRESERVED / READY BLOCKED (${rs.identity_occlusion_hold_frames ?? 0} frame(s))` : 'inactive'],
        ['Locked Bottle / Band', rs.locked_bottle_target_id ? `${rs.locked_bottle_target_id} @ ${rs.locked_body_fraction !== undefined && rs.locked_body_fraction !== null ? Number(rs.locked_body_fraction).toFixed(3) : 'N/A'}` : 'N/A'],
        ['Post-Grasp Identity', rs.post_grasp_identity_status || 'N/A'],
        ['Contact Band Tracking', rs.contact_patch_tracking_ok === true ? 'CURRENT' : (rs.contact_patch_tracking_ok === false ? `BLOCKED (${humanReason(rs.contact_patch_tracking_reason || 'low confidence')})` : 'N/A')],
        ['Tracking ROI Source', rs.tracking_roi_source || 'N/A'],
        ['Edge / Band Source', `${rs.edge_source || 'N/A'} / ${rs.band_source || 'N/A'}`],
        ['ROI Escape Risk', rs.roi_escape_risk ?? 'N/A'],
        ['Display Lag', rs.display_center_lag_px !== undefined && rs.display_center_lag_px !== null ? `${rs.display_center_lag_px} px` : 'N/A'],
        ['Smoothing Mode', rs.display_smoothing_mode || rs.smoothing_mode || 'N/A'],
        ['Eval Cycle', ra.evaluation_cycle_ms ? `${ra.evaluation_cycle_ms} ms` : 'N/A'],
        ['Frame Age', frameAge !== undefined && frameAge !== null ? `${Number(frameAge).toFixed(1)} ms` : 'N/A'],
        ['Slowest Stage', slowestStage],
        ['Detect ms', ot.bottle_detection_ms !== undefined ? `${ot.bottle_detection_ms} ms` : 'N/A'],
        ['Depth ms', ot.depth_processing_ms !== undefined ? `${ot.depth_processing_ms} ms` : 'N/A'],
        ['Tracking ms', ot.target_lock_update_ms !== undefined ? `${ot.target_lock_update_ms} ms` : 'N/A'],
        ['Overlay Draw ms', ot.overlay_drawing_ms !== undefined ? `${ot.overlay_drawing_ms} ms` : 'N/A'],
        ['JPEG/File ms', ot.jpeg_file_output_ms !== undefined ? `${ot.jpeg_file_output_ms} ms` : 'N/A'],
        ['Skipped Frames', ra.estimated_capture_frames_skipped ?? ra.skipped_since_last_eval ?? 'N/A'],
        ['Matrix Status', qm.quality_matrix_status || 'idle'],
        ['Matrix Current/Stale', qm.quality_label_stale || qm.matrix_stale_due_to_motion ? 'STALE' : 'CURRENT'],
        ['Matrix Stale Reason', humanReason(qm.quality_current_mismatch_reasons || qm.matrix_stale_reasons || 'none')],
        ['Matrix Compute ID', qm.quality_compute_id || 'N/A'],
        ['Matrix Lock Frame', qm.lock_frame_id || qm.hull_lock_frame_id || 'N/A'],
        ['Matrix JSON', fileState(files, 'quality_matrix_status')],
        ['Epsilon JSON', fileState(files, 'epsilon_result')],
        ['Bottle Quality Overlay', fileState(files, 'bottle_quality_overlay')],
        ['Contact Quality Overlay', fileState(files, 'contact_quality_overlay')],
        ['Legacy Weak Direction PNG', fileState(files, 'grasp_quality_weak_direction')],
        ['Contact Hull Overlay', fileState(files, 'contact_hull_overlay')],
        ['Contact Patch Debug', fileState(files, 'contact_patch_debug')],
        ['Contact Hull JSON', fileState(files, 'contact_hull_debug')],
        ['Wrench Hull Debug', fileState(files, 'wrench_space_convex_hull_debug')],
        ['3D Wrench Plot', fileState(files, 'wrench_space_3d_hull')],
        ['Matrix Compute', qm.quality_matrix_compute_time_ms ? `${qm.quality_matrix_compute_time_ms} ms` : 'N/A'],
        ['Matrix Epsilon', qm.epsilon_score ? `${Math.round(qm.epsilon_score)} / 100` : 'N/A'],
        ['Hull Status', qm.hull_status || ((qm.grasp_hulls || {}).status) || 'idle'],
        ['Hull Compute', qm.hull_compute_time_ms ? `${qm.hull_compute_time_ms} ms` : 'N/A'],
        ['Hull Lock', qm.hull_lock_frame_id ?? ((qm.grasp_hulls || {}).lock_frame_id) ?? 'N/A'],
        ['Hull Reason', qm.hull_failure_reason || ((qm.grasp_hulls || {}).failure_reason) || 'none'],
        ['Matrix Age', qm.quality_matrix_file_age_sec !== undefined ? `${qm.quality_matrix_file_age_sec}s` : 'N/A'],
        ['Combined Quality', cq.combined_quality_status || 'N/A'],
        ['Quality Reason', cq.reason || 'N/A'],
        ['DAVIS Preview', dc.davis_adjusted_overlay_status || 'N/A'],
        ['DAVIS Preview Reason', dc.davis_adjusted_overlay_reason || 'N/A'],
        ['DAVIS Alignment', dc.davis_realsense_alignment_status || 'N/A'],
        ['DAVIS Motion Confidence', dc.davis_motion_confidence !== undefined && dc.davis_motion_confidence !== null ? `${dc.davis_motion_confidence}` : 'N/A'],
        ['DAVIS Display FPS', dc.davis_display_fps ?? 'N/A'],
        ['DAVIS Analysis FPS', dc.davis_event_analysis_fps ?? 'N/A'],
        ['DAVIS Bottleneck', dc.davis_bottleneck || 'N/A'],
        ['Blockers', (rs.blockers || []).join(', ') || 'none'],
      ]);
	      cards(document.getElementById('davisCards'), [
        ['Running', data.running.davis_motion],
        ['DAVIS Report', fileState(files, 'davis_motion_report')],
        ['Backend', dm.backend],
        ['Stream', dm.stream],
	        ['Motion Class', dm.motion_class],
	        ['Decision', dm.decision],
	        ['Event Rate', dm.event_rate ? `${Math.round(dm.event_rate)} ev/s` : '0 ev/s'],
	        ['Event Count', dm.event_count ?? 'N/A'],
        ['Raw Poll Events', dm.last_poll_event_count ?? 'N/A'],
        ['Global / Object Window', `${dm.raw_window_event_count ?? 'N/A'} / ${dm.object_roi_window_event_count ?? 'N/A'}`],
        ['Event Analysis', dm.event_analysis_status || 'N/A'],
        ['Event Timestamp', `${dm.event_timestamp_status || 'N/A'} (${dm.event_timestamp_lag_ms ?? 'N/A'} ms lag)`],
        ['Estimate Valid', dm.motion_estimate_valid === true ? 'yes' : (dm.motion_estimate_valid === false ? `no: ${dm.motion_estimate_invalid_reason || 'unknown'}` : 'N/A')],
        ['Motion Method', dm.motion_estimation_method || 'N/A'],
        ['Motion Confidence', dm.motion_estimation_confidence !== undefined && dm.motion_estimation_confidence !== null ? `${Math.round(Number(dm.motion_estimation_confidence) * 100)}%` : 'N/A'],
        ['dx / dy', `${dm.dx_px !== undefined && dm.dx_px !== null ? Number(dm.dx_px).toFixed(1) : 'N/A'}, ${dm.dy_px !== undefined && dm.dy_px !== null ? Number(dm.dy_px).toFixed(1) : 'N/A'} px`],
        ['Displacement', dm.displacement_px !== undefined && dm.displacement_px !== null ? `${Number(dm.displacement_px).toFixed(1)} px` : 'N/A'],
        ['Cumulative Slip', dm.cumulative_slip_px !== undefined && dm.cumulative_slip_px !== null ? `${Number(dm.cumulative_slip_px).toFixed(1)} px` : 'N/A'],
        ['Downward Slip', dm.downward_slip_px !== undefined && dm.downward_slip_px !== null ? `${Number(dm.downward_slip_px).toFixed(1)} px` : 'N/A'],
        ['Rotation Delta', dm.rotation_delta_deg !== undefined && dm.rotation_delta_deg !== null ? `${Number(dm.rotation_delta_deg).toFixed(1)} deg` : 'unavailable'],
        ['Roll', dm.roll_valid === true && dm.roll_delta_deg !== undefined && dm.roll_delta_deg !== null ? `${Number(dm.roll_delta_deg).toFixed(1)} deg` : String(dm.roll_status || 'NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION').replace(/_/g, ' ')],
        ['Angular Velocity', dm.angular_velocity_deg_s !== undefined && dm.angular_velocity_deg_s !== null ? `${Number(dm.angular_velocity_deg_s).toFixed(1)} deg/s` : 'N/A'],
        ['Vibration RMS', dm.vibration_amplitude_px_rms !== undefined && dm.vibration_amplitude_px_rms !== null ? `${Number(dm.vibration_amplitude_px_rms).toFixed(1)} px` : 'N/A'],
        ['Disturbance', dm.disturbance_type || 'N/A'],
	        ['Centroid Shift', `${Math.round(dm.centroid_shift_x || 0)}, ${Math.round(dm.centroid_shift_y || 0)} px`],
	        ['Gross Slip', dm.gross_slip ? Math.round(dm.gross_slip) : 0],
        ['Incipient Slip', dm.incipient_slip ? Math.round(dm.incipient_slip) : 0],
        ['Rotation', dm.rotation ? Math.round(dm.rotation) : 0],
        ['Micro Deviation', dm.micro_deviation_score ? Math.round(dm.micro_deviation_score) : 0],
        ['Rotation Score', dm.rotation_score ? Math.round(dm.rotation_score) : 0],
        ['Vibration', dm.vibration ? Math.round(dm.vibration) : 0],
        ['Stability', dm.stability ? Math.round(dm.stability) : 0],
        ['Boundary Field', boundaryField.measurement_status || 'N/A'],
        ['Boundary Alignment', boundaryField.alignment_status || 'N/A'],
        ['Boundary Bands Active', boundaryField.active_band_count ?? 'N/A'],
        ['Boundary Events Assigned / Raw', `${boundaryField.assigned_event_count ?? boundaryField.event_count ?? 'N/A'} / ${boundaryField.raw_window_event_count ?? 'N/A'}`],
        ['Grasp-Band DAVIS Status', boundaryField.contact_band_measurement_status || 'N/A'],
        ['Grasp-Band Active / Events', `${boundaryField.contact_band_active_count ?? 'N/A'} / ${boundaryField.contact_band_event_count ?? 'N/A'}`],
        ['Grasp-Band Tangential Flow', boundaryField.contact_tangential_flow_px !== undefined && boundaryField.contact_tangential_flow_px !== null ? `${Number(boundaryField.contact_tangential_flow_px).toFixed(2)} px` : 'N/A'],
        ['Grasp-Band Normal Flow', boundaryField.contact_normal_flow_px !== undefined && boundaryField.contact_normal_flow_px !== null ? `${Number(boundaryField.contact_normal_flow_px).toFixed(2)} px` : 'N/A'],
        ['Grasp-Band Vertical Drift', boundaryField.contact_vertical_drift_px !== undefined && boundaryField.contact_vertical_drift_px !== null ? `${Number(boundaryField.contact_vertical_drift_px).toFixed(2)} px` : 'N/A'],
        ['Grasp-Band Oscillation', boundaryField.contact_oscillation_energy_px !== undefined && boundaryField.contact_oscillation_energy_px !== null ? `${Number(boundaryField.contact_oscillation_energy_px).toFixed(2)} px` : 'N/A'],
        ['Grasp-Band Precursor', boundaryField.contact_incipient_slip_precursor_score !== undefined && boundaryField.contact_incipient_slip_precursor_score !== null ? `${Math.round(boundaryField.contact_incipient_slip_precursor_score)} / 100` : 'N/A'],
        ['Boundary Slip Cue', boundaryField.slip_cue !== undefined && boundaryField.slip_cue !== null ? `${Math.round(boundaryField.slip_cue)} / 100` : 'N/A'],
        ['Boundary Rotation Cue', boundaryField.rotation_cue !== undefined && boundaryField.rotation_cue !== null ? `${Math.round(boundaryField.rotation_cue)} / 100` : 'N/A'],
        ['Boundary Vibration Cue', boundaryField.vibration_cue !== undefined && boundaryField.vibration_cue !== null ? `${Math.round(boundaryField.vibration_cue)} / 100` : 'N/A'],
        ['Boundary Precursor', boundaryField.incipient_slip_precursor_score !== undefined && boundaryField.incipient_slip_precursor_score !== null ? `${Math.round(boundaryField.incipient_slip_precursor_score)} / 100` : 'N/A'],
        ['Boundary Gate', boundaryField.object_motion_valid === true ? 'object-local' : `suppressed: ${boundaryField.suppression_reason || 'not valid'}`],
        ['Simulated Force Policy', simulatedForcePolicy.reason || 'NO_ACTION_NOT_AVAILABLE'],
        ['Simulated Grip Delta', simulatedForcePolicy.grip_force_delta !== undefined ? `${Number(simulatedForcePolicy.grip_force_delta).toFixed(3)} normalized` : 'N/A'],
        ['Simulated Torque Delta', simulatedForcePolicy.torque_delta !== undefined ? `${Number(simulatedForcePolicy.torque_delta).toFixed(3)} normalized` : 'N/A'],
        ['Force Policy Validation', dm.force_control_status || 'NOT_PHYSICALLY_VALIDATED'],
        ['Object State', objectState.orientation || 'UNKNOWN'],
        ['Object Tilt', objectState.tilt_angle_deg !== undefined && objectState.tilt_angle_deg !== null ? `${Number(objectState.tilt_angle_deg).toFixed(1)} deg` : 'N/A'],
        ['Pickup Available', objectState.available_for_pickup === true ? 'yes' : `no: ${(objectState.reasons || []).join(', ') || 'not tested'}`],
        ['Task Policy', taskPlan.strategy || 'N/A'],
        ['YOLO Runtime', rs.yolo_runtime_state || (rs.yolo_enabled ? 'not run' : 'disabled')],
        ['YOLO Backend', rs.yolo_model_backend || 'N/A'],
        ['YOLO Detections / RGB-D Matches', `${rs.yolo_detection_count ?? 0} / ${rs.yolo_matched_rgbd_count ?? 0}`],
        ['YOLO Inference', rs.yolo_inference_ms !== undefined && rs.yolo_inference_ms !== null ? `${Number(rs.yolo_inference_ms).toFixed(1)} ms` : 'skipped'],
        ['YOLO/RGB-D Fusion', rs.yolo_rgbd_fusion_mode || 'not active in locked tracking'],
        ['Bottles Detected / Feasible', `${bottleSelection.candidate_count ?? rs.detected_bottle_count ?? 0} / ${bottleSelection.eligible_count ?? rs.eligible_bottle_count ?? 0}`],
        ['Selected Bottle', bottleSelection.selected_target_id || rs.selected_bottle_target_id || 'none'],
        ['Selected Approach', selectedBottleCandidate.approach || 'NONE'],
        ['Selected Grasp Band', selectedBottleCandidate.selected_grasp_body_fraction !== undefined && selectedBottleCandidate.selected_grasp_body_fraction !== null ? `${Number(selectedBottleCandidate.selected_grasp_body_fraction).toFixed(2)} bottle fraction` : 'N/A'],
        ['Grasp Side Clearance', selectedBottleCandidate.grasp_clearance_status || 'N/A'],
        ['Fallback Band', selectedBottleCandidate.grasp_band_fallback_used ? 'yes - less optimal clear band' : 'no'],
        ['Blocking Bottles', (selectedBottleCandidate.blocking_target_ids || []).join(', ') || 'none'],
        ['Distance / Top Priority', `${selectedBottleCandidate.distance_priority_score ?? 'N/A'} / ${selectedBottleCandidate.top_exposure_score ?? 'N/A'}`],
        ['Grasp Memory', `${rs.grasp_memory_status || 'not locked'}${rs.locked_body_fraction !== undefined && rs.locked_body_fraction !== null ? ` @ ${Number(rs.locked_body_fraction).toFixed(2)}` : ''}`],
        ['Selected Depth Identity', selectedBottleCandidate.depth_geometry_score !== undefined && selectedBottleCandidate.depth_geometry_score !== null ? `${Math.round(100 * Number(selectedBottleCandidate.depth_geometry_score))} / 100 (${selectedBottleCandidate.depth_geometry_status || 'estimated'})` : 'N/A'],
        ['Selected Depth Boundary', selectedBottleCandidate.depth_foreground_separation_mm !== undefined && selectedBottleCandidate.depth_foreground_separation_mm !== null ? `${Number(selectedBottleCandidate.depth_foreground_separation_mm).toFixed(1)} mm signed` : 'N/A'],
        ['Selected Metric Size', selectedBottleCandidate.metric_dimensions_valid === true ? 'broad bottle range' : (selectedBottleCandidate.metric_dimensions_valid === false ? 'rejected' : 'not tested')],
        ['Target Selection', bottleSelection.selection_status || 'N/A'],
        ['Event Window', dm.event_window_ms ? `${dm.event_window_ms} ms (${dm.event_window_mode || 'recent'})` : 'N/A'],
        ['Scoring Window', dm.scoring_window_ms ? `${dm.scoring_window_ms} ms` : 'N/A'],
        ['DAVIS ROI', dm.roi_xywh ? dm.roi_xywh.join(', ') : 'full frame'],
        ['Target Handoff', dm.davis_target_context_mode || 'none'],
        ['Lock-Time ROI', dm.davis_target_lock_active ? `held ${Math.round(Number(dm.davis_target_lock_age_ms || 0))} ms` : 'not held'],
        ['Live RealSense Mapping', dm.realsense_mapping_context_current === true ? 'current' : 'not current'],
        ['Camera Alignment', dm.davis_target_alignment_status || 'NOT AVAILABLE'],
        ['Mapping Mode', dm.davis_target_mapping_mode || 'N/A'],
        ['Mapping Physically Validated', dm.davis_target_mapping_physically_validated === true ? 'yes' : 'no'],
        ['Hull Points', dm.motion_hull_point_count ?? 0],
        ['Update FPS', dm.davis_update_fps ?? 'N/A'],
        ['Display FPS', dm.davis_display_fps ?? dc.davis_display_fps ?? 'N/A'],
        ['Analysis FPS', dm.davis_event_analysis_fps ?? dc.davis_event_analysis_fps ?? 'N/A'],
        ['Compensation', dc.davis_motion_compensation_enabled ? 'enabled' : 'disabled'],
        ['Motion dx/dy', `${dc.davis_motion_dx_px ?? 'N/A'}, ${dc.davis_motion_dy_px ?? 'N/A'} px`],
        ['Adjusted Preview', dc.davis_adjusted_overlay_status || 'N/A'],
        ['Preview Reason', dc.davis_adjusted_overlay_reason || 'N/A'],
        ['Latency', dm.davis_processing_latency_ms ? `${dm.davis_processing_latency_ms} ms` : 'N/A'],
        ['Rolling Buffer Span', dm.rolling_buffer_span_sec !== undefined && dm.rolling_buffer_span_sec !== null ? `${dm.rolling_buffer_span_sec}s` : (dm.rolling_buffer_age_sec !== undefined && dm.rolling_buffer_age_sec !== null ? `${dm.rolling_buffer_age_sec}s` : 'N/A')],
        ['Dropped Events', `${dm.dropped_event_windows ?? 0} windows`],
        ['Reason', dm.reason || dm.error || 'N/A'],
      ]);
      document.getElementById('davisNumbers').textContent = [
        `Role: post-grasp event feedback only`,
        `Output class: ${dm.motion_class ?? 'N/A'}`,
        `Decision: ${dm.decision ?? 'N/A'}`,
        `Event count: ${dm.event_count ?? 'N/A'}`,
        `Event rate: ${dm.event_rate ? Math.round(dm.event_rate) : 0} ev/s`,
        `Raw poll/global window/object ROI: ${dm.last_poll_event_count ?? 'N/A'} / ${dm.raw_window_event_count ?? 'N/A'} / ${dm.object_roi_window_event_count ?? 'N/A'}`,
        `Event analysis: ${dm.event_analysis_status ?? 'N/A'}; timestamps=${dm.event_timestamp_status ?? 'N/A'} (${dm.event_timestamp_lag_ms ?? 'N/A'} ms lag, ${dm.event_timestamp_basis ?? 'N/A'})`,
        `Event window: ${dm.event_window_ms ?? 'N/A'} ms (${dm.event_window_mode ?? 'N/A'})`,
        `Motion estimate: ${dm.motion_estimate_valid === true ? 'valid' : 'invalid'} ${dm.motion_estimate_invalid_reason || ''}`,
        `dx/dy/displacement: ${dm.dx_px ?? 'N/A'}, ${dm.dy_px ?? 'N/A'}, ${dm.displacement_px ?? 'N/A'} px`,
        `Cumulative/downward slip: ${dm.cumulative_slip_px ?? 'N/A'}, ${dm.downward_slip_px ?? 'N/A'} px`,
        `Rotation/angular velocity: ${dm.rotation_delta_deg ?? 'N/A'} deg, ${dm.angular_velocity_deg_s ?? 'N/A'} deg/s`,
        `Vibration: ${dm.vibration_amplitude_px_rms ?? 'N/A'} px RMS at ${dm.vibration_frequency_hz ?? 'N/A'} Hz`,
        `Motion confidence/method: ${dm.motion_estimation_confidence ?? 'N/A'} / ${dm.motion_estimation_method ?? 'N/A'}`,
        `ROI bbox: ${dm.roi_xywh ? dm.roi_xywh.join(', ') : 'full frame'}`,
        `ROI source: ${dm.roi_source ?? 'N/A'}`,
        `Target handoff: ${dm.davis_target_context_mode ?? 'none'}; valid=${dm.davis_target_context_valid ?? false}; lock active=${dm.davis_target_lock_active ?? false}; lock age=${dm.davis_target_lock_age_ms ?? 'N/A'} ms`,
        `Live mapping current: ${dm.realsense_mapping_context_current ?? false}; raw current-target gate=${dm.realsense_mapping_context_raw_gate ?? false}`,
        `Alignment: ${dm.davis_target_alignment_status ?? 'N/A'}; mode=${dm.davis_target_mapping_mode ?? 'N/A'}; physically validated=${dm.davis_target_mapping_physically_validated ?? false}`,
        `Target handoff reason: ${dm.davis_target_context_reason ?? 'N/A'}`,
        `Disturbance: ${dm.disturbance_type ?? 'N/A'} (rejected=${dm.disturbance_rejection_applied ?? false})`,
        `Hull points: ${dm.motion_hull_point_count ?? 0}, hull area: ${dm.motion_hull_area_px2 ?? 0} px^2`,
        `Update FPS: ${dm.davis_update_fps ?? 'N/A'}, latency: ${dm.davis_processing_latency_ms ?? 'N/A'} ms`,
        `Centroid movement: ${Math.round(dm.centroid_shift_x || 0)}, ${Math.round(dm.centroid_shift_y || 0)} px`,
        `Left/right imbalance: ${dm.left_right_event_imbalance !== undefined ? Math.round(dm.left_right_event_imbalance * 100) + '%' : 'N/A'}`,
        `Rotation-like pattern: ${dm.rotation_like_event_pattern !== undefined ? Math.round(dm.rotation_like_event_pattern) + '/100' : 'N/A'}`,
        `Sudden spike after grasp: ${dm.sudden_spike_after_grasp !== undefined ? Math.round(dm.sudden_spike_after_grasp) + '/100' : 'N/A'}`,
        `Dropped events/windows: ${dm.dropped_event_windows ?? 0} windows, ${dm.dropped_events_due_tick_cap ?? 0} tick, ${dm.dropped_events_due_rolling_cap ?? 0} rolling`,
        `ROI mapping: ${dm.mapped_realsense_roi_reason ?? 'waiting for RealSense bbox'}`,
        `Quantitative evidence: ${dm.quantitative_evidence ?? 'N/A'}`,
        `Boundary diagnostic: ${boundaryField.measurement_status ?? 'N/A'}; alignment=${boundaryField.alignment_status ?? 'N/A'}; physically validated=${boundaryField.physically_validated ?? false}`,
        `Force policy diagnostic only: ${simulatedForcePolicy.reason ?? 'N/A'}; actuator connected=${dm.force_control_actuator_connected ?? false}; physically validated=${simulatedForcePolicy.physically_validated ?? false}`,
        `Object state: ${objectState.orientation ?? 'UNKNOWN'}; availability=${objectState.available_for_pickup ?? false}; evidence=${((objectState.evidence || {}).availability || {}).status ?? 'N/A'}`,
        `Task policy: ${taskPlan.strategy ?? 'N/A'} (${taskPlan.applicability ?? 'N/A'}); physically validated=${taskPlan.physically_validated ?? false}`,
        `Bottle selection: ${bottleSelection.selection_status ?? 'N/A'}; detected=${bottleSelection.candidate_count ?? 0}; feasible=${bottleSelection.eligible_count ?? 0}; selected=${bottleSelection.selected_target_id ?? 'none'}`,
        `Selected depth identity: ${selectedBottleCandidate.depth_geometry_status ?? 'N/A'}; score=${selectedBottleCandidate.depth_geometry_score ?? 'N/A'}; signed neighborhood separation=${selectedBottleCandidate.depth_foreground_separation_mm ?? 'N/A'} mm; metric size valid=${selectedBottleCandidate.metric_dimensions_valid ?? 'not tested'}`,
        `Reason: ${dm.reason || dm.error || 'N/A'}`,
      ].join('\n');
      const p = data.profile || {};
      const w = p.web || {};
      const ro = p.realsense_output_breakdown || {};
      cards(document.getElementById('profileCards'), [
        ['RS Capture', fmtMs(p.realsense_capture)],
        ['RS Preprocess', fmtMs(p.realsense_preprocessing)],
        ['RS Segmentation', fmtMs(p.realsense_segmentation)],
        ['RS Depth', fmtMs(p.realsense_depth_processing)],
        ['RS Contact', fmtMs(p.realsense_contact_modeling)],
        ['RS Epsilon', fmtMs(p.realsense_epsilon)],
        ['RS Safety', fmtMs(p.realsense_safety_scoring)],
        ['RS Drawing', fmtMs(p.realsense_drawing)],
        ['RS Live JPEG', fmtMs(p.realsense_live_jpeg_output)],
        ['RS Output', fmtMs(p.realsense_output_write)],
        ['RS Output Total', fmtMs(ro.total_rs_output_ms)],
        ['RS RGB JPEG Write', fmtMs(ro.rgb_live_jpeg_write_ms)],
        ['RS Depth JPEG Write', fmtMs(ro.depth_live_jpeg_write_ms)],
        ['RS Overlay JPEG Write', fmtMs(ro.overlay_jpeg_write_ms)],
        ['RS Status JSON', fmtMs(ro.status_json_write_ms)],
        ['RS Latest Pregrasp JSON', fmtMs(ro.latest_pregrasp_report_json_write_ms)],
        ['RS Pregrasp JSON', fmtMs(ro.pregrasp_report_json_write_ms)],
        ['RS Candidate JSON', fmtMs(ro.candidate_grasp_report_json_write_ms)],
        ['RS Gripper JSON', fmtMs(ro.gripper_command_status_json_write_ms)],
        ['RS Stage/Lock JSON', `${fmtMs(ro.stage_history_json_write_ms)} / ${fmtMs(ro.locked_target_json_write_ms)}`],
        ['RS JSON Safe', fmtMs(ro.json_safe_ms)],
        ['RS JSON Dump', fmtMs(ro.json_dump_ms)],
        ['RS File Replace', fmtMs(ro.file_replace_ms)],
        ['RS Dashboard Save', fmtMs(ro.dashboard_image_save_ms)],
        ['RS Debug Save', fmtMs(ro.debug_image_save_ms)],
        ['RS Matplotlib Savefig', fmtMs(ro.matplotlib_savefig_ms)],
        ['RS Epsilon JSON', fmtMs(ro.epsilon_json_write_ms)],
        ['RS Wrench Image Write', fmtMs(ro.wrench_image_write_ms)],
        ['RS Blocking Wait', fmtMs(ro.sleep_or_blocking_wait_ms)],
        ['RS Total', fmtMs(p.realsense_total)],
        ['DAVIS Capture/Process', fmtMs(w.davis_event_capture_processing)],
        ['DAVIS Hull Drawing', fmtMs(w.davis_hull_drawing)],
        ['DAVIS Hull Encoding', fmtMs(w.davis_hull_encoding)],
        ['RGB Encoding', fmtMs(w.rgb_encoding)],
      ]);
      const streams = p.streams || {};
      const streamItems = [];
      for (const [name, s] of Object.entries(streams)) {
        streamItems.push([`${name} FPS`, `${s.producer_fps ?? 0} prod / ${s.send_fps ?? 0} send`]);
        streamItems.push([`${name} Latency`, `${s.frame_age_ms ?? 'N/A'} ms age / ${s.send_latency_ms ?? 'N/A'} ms send`]);
        streamItems.push([`${name} Queue/Drops`, `${s.queue_size ?? 0} / ${s.overwrite_drops ?? 0}`]);
      }
      cards(document.getElementById('streamCards'), streamItems);
	      if (debugViewsOpen) {
	        setImg('imgRgb', data.images.rgb);
	        setImg('imgDepth', data.images.depth);
	      }
	      setMainVisualImages(data);
	      if (diagnosticsVisible) {
	        setImg('imgSensorRgb', data.images.rgb);
	        setImg('imgSensorDepth', data.images.depth);
	      } else {
	        document.getElementById('imgSensorRgb')?.removeAttribute('src');
	        document.getElementById('imgSensorDepth')?.removeAttribute('src');
	      }
	      if (diagnosticsRefreshDue) {
	        setImg('imgSensorDavis', data.images.davis_heatmap);
	        setImg('imgDavisAps', data.images.davis_aps);
	        setImg('imgDavis', data.images.davis_heatmap);
	      } else if (!diagnosticsVisible) {
	        document.getElementById('imgSensorDavis')?.removeAttribute('src');
	        document.getElementById('imgDavisAps')?.removeAttribute('src');
	        document.getElementById('imgDavis')?.removeAttribute('src');
	      }
	      const pointcloudDiagnosticsOpen = Boolean(document.querySelector('#pointcloudSection .pointcloud-diagnostics')?.open);
	      if (pointcloudVisible) {
	        setStaticImg('imgPointcloudPreview', data.images.pointcloud_preview, files.pointcloud_preview);
	      } else {
	        document.getElementById('imgPointcloudPreview')?.removeAttribute('src');
	      }
	      if (diagnosticsRefreshDue || (pointcloudVisible && pointcloudDiagnosticsOpen && dashboardMode === 'engineering')) {
	        setStaticImg('imgBottlePointcloudPreview', data.images.bottle_pointcloud_preview, files.bottle_pointcloud_preview);
	      } else {
	        document.getElementById('imgBottlePointcloudPreview')?.removeAttribute('src');
	      }
	      if (diagnosticsRefreshDue) {
	        setStaticImg('imgPointcloudContext', data.images.pointcloud_context, files.pointcloud_context);
	      } else {
	        document.getElementById('imgPointcloudContext')?.removeAttribute('src');
	      }
	      if (pointcloudVisible) refreshPointcloudViewer(pc);
      const staticImageRefreshDue = Date.now() - lastStaticImageRefreshMs >= STATIC_IMAGE_REFRESH_MS;
      if (staticImageRefreshDue) {
        lastStaticImageRefreshMs = Date.now();
        if (davisVisible || qualityVisible) {
          setStaticImg('imgDavisConfirmation', data.images.davis_realsense_confirmation, files.davis_realsense_confirmation);
          setStaticImg('imgDavisMotionOverlay', data.images.davis_motion_overlay, files.davis_motion_overlay);
          setStaticImg('imgDavisAdjustedOverlay', data.images.davis_adjusted_quality_overlay, files.davis_adjusted_quality_overlay);
        }
        if (davisVisible) {
          setStaticImg('imgDavisMotionHeatmap', data.images.davis_heatmap || data.images.davis_motion_overlay, files.davis_heatmap || files.davis_motion_overlay);
          setStaticImg('imgDavisMotionAps', data.images.davis_aps, files.davis_aps);
          const dm = data.davis_motion || {};
          const visEl = document.getElementById('davisApsVisibility');
          if (visEl) {
            const status = dm.aps_visibility_status || 'UNKNOWN';
            const score = (dm.aps_visibility_score === null || dm.aps_visibility_score === undefined) ? 'n/a' : dm.aps_visibility_score;
            let text = `APS visibility: ${status} (score ${score})`;
            if (dm.aps_display_warning) text += ` - ${dm.aps_display_warning}`;
            visEl.textContent = text;
            visEl.style.color = (status === 'GOOD') ? '' : '#e8a33d';
          }
        }
        if (qualityVisible) {
          setStaticImg('imgWeakDirection', data.images.bottle_quality_overlay || data.images.grasp_quality_weak_direction, files.bottle_quality_overlay || files.grasp_quality_weak_direction);
          setStaticImg('imgQualityCloseup', data.images.bottle_quality_closeup || data.images.bottle_quality_overlay, files.bottle_quality_closeup || files.bottle_quality_overlay);
          setStaticImg('imgQualityContext', data.images.bottle_quality_context || data.images.bottle_quality_overlay, files.bottle_quality_context || files.bottle_quality_overlay);
          setStaticImg('imgContactHullOverlay', data.images.contact_hull_overlay, files.contact_hull_overlay);
          setStaticImg('imgWrench3D', data.images.wrench_space_3d_hull, files.wrench_space_3d_hull);
          setStaticImg('imgHullOverlay', data.images.grasp_hulls_overlay, files.grasp_hulls_overlay);
          setStaticImg('imgWrenchProjection', data.images.wrench_space_projection, files.wrench_space_projection);
          setStaticImg('imgWrenchFxFy', data.images.wrench_space_fx_fy, files.wrench_space_fx_fy);
          setStaticImg('imgWrenchFxTau', data.images.wrench_space_fx_tau, files.wrench_space_fx_tau);
          setStaticImg('imgWrenchFyTau', data.images.wrench_space_fy_tau, files.wrench_space_fy_tau);
        }
      }
      const fyTauCard = document.getElementById('fyTauCard');
      if (fyTauCard) fyTauCard.classList.toggle('hidden', !(data.images.wrench_space_fy_tau && data.images.wrench_space_fy_tau.exists));
      document.getElementById('logRs').textContent = data.logs.realsense || '';
      } catch (err) {
        console.warn('Dashboard status refresh failed', err);
      } finally {
        refreshInFlight = false;
      }
    }
    document.querySelectorAll('.plot-card').forEach(card => {
      card.addEventListener('click', () => openModalFromImg(card.dataset.img, card.dataset.title));
    });
    document.getElementById('imageModal').addEventListener('click', (event) => {
      if (event.target.id === 'imageModal') closeModal();
    });
    document.querySelectorAll('.visual-panel img').forEach(img => {
      img.addEventListener('click', () => openModalFromImg(img.id, img.alt || 'Demo image'));
    });
	    window.addEventListener('keydown', (event) => {
	      if (event.key === 'Escape') closeModal();
	    });
	    window.addEventListener('resize', resizePointcloudViewer);
	    window.addEventListener('hashchange', () => {
	      syncDebugStreams();
	      refresh();
	    });
    document.getElementById('debugViews').addEventListener('toggle', syncDebugStreams);
    window.addEventListener('beforeunload', () => {
      navigator.sendBeacon('/api/debug-streams', new Blob([JSON.stringify({enabled:false})], {type: 'application/json'}));
    });
	    clearDebugImages();
	    document.body.classList.add('presentation');
	    setDashboardMode(dashboardMode);
	    setVisualTab(activeVisualTab);
	    applyActiveSection();
	    window.addEventListener('hashchange', applyActiveSection);
	    syncDebugStreams().finally(refresh);
	    setInterval(refresh, DASHBOARD_POLL_INTERVAL_MS);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Final web dashboard for RealSense RGB-D grasping plus optional DAVIS post-grasp metrics.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--start-realsense", action="store_true", help="Start the RealSense grasp pipeline immediately.")
    parser.add_argument("--start-davis", action="store_true", help="Start DAVIS motion monitor immediately.")
    args = parser.parse_args()
    start_dashboard_recording()
    previous_sigterm = None
    if runtime_recording is not None:
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def recording_stop_requested(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, recording_stop_requested)
    success = False
    try:
        write_debug_stream_state()
        start_file_pollers()
        start_reconnect_monitor()
        start_derived_status_producer()
        if args.start_realsense:
            set_desired_process("realsense", True)
            ok, message = start_process("realsense")
            print(message, flush=True)
            if not ok:
                print("RealSense did not start; dashboard will show the camera error.", flush=True)
        if args.start_davis:
            set_desired_process("davis_motion", True)
            start_davis_motion()
        try:
            app.run(host=args.host, port=args.port, threaded=True)
        except KeyboardInterrupt:
            if runtime_recording is None:
                raise
        success = True
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        finish_dashboard_recording(success)


if __name__ == "__main__":
    main()

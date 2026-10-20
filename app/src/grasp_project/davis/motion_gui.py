"""DAVIS event-camera motion analyzer and deprecated desktop presenter.

Purpose: read DAVIS events through the available backend, classify post-grasp
motion, render dashboard-ready event heatmaps/timelines, and write DAVIS motion
reports.
Main demo: the main web dashboard imports `DavisMotionAnalyzer`; standalone
desktop GUI launch is deprecated and env-gated.
Phase note: migrated in Phase 3; top-level `davis_motion_gui.py` is a
compatibility wrapper.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

QT_PLATFORM_PATH = Path("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
if QT_PLATFORM_PATH.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLATFORM_PATH)
    os.environ["QT_PLUGIN_PATH"] = str(QT_PLATFORM_PATH.parent)

import numpy as np
import cv2

if QT_PLATFORM_PATH.exists():
    # OpenCV can rewrite Qt plugin paths during import; restore the system Qt path
    # before PyQt creates QApplication.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLATFORM_PATH)
    os.environ["QT_PLUGIN_PATH"] = str(QT_PLATFORM_PATH.parent)

from grasp_project.davis.event_stream import (
    DAVIS_FRAME_SIZE,
    EventPostGraspMonitor,
    _clip_roi,
)
from grasp_project.davis.aps_stream import (
    APS_DISPLAY_MODES,
    APS_VISIBILITY_DEFAULTS,
    aps_display_image,
    classify_aps_visibility,
    compute_aps_stats,
    enhance_aps_display,
    latest_aps_array,
    normalize_backend_aps_frame,
)
from grasp_project.common.config import load_named_config
from grasp_project.common.file_io import append_jsonl_bounded, replace_path_retry
from grasp_project.common.latency import SoftwareLatencyTracker
from grasp_project.davis.boundary_event_search import boundary_from_roi, search_boundary_events
from grasp_project.davis.motion_estimation import DavisMotionEstimator, load_davis_motion_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"
DAVIS_OUTPUT_DIR = OUTPUT_DIR / "davis"
DAVIS_LATEST_STATUS_PATH = DAVIS_OUTPUT_DIR / "latest_status.json"
DAVIS_LATEST_APS_PATH = DAVIS_OUTPUT_DIR / "latest_aps.jpg"
DAVIS_LATEST_EVENT_HEATMAP_PATH = DAVIS_OUTPUT_DIR / "latest_event_heatmap.jpg"
DAVIS_LATEST_TIMELINE_PATH = DAVIS_OUTPUT_DIR / "latest_timeline.json"
DAVIS_LATEST_MOTION_REPORT_PATH = DAVIS_OUTPUT_DIR / "latest_motion_report.json"
DAVIS_LATEST_MOTION_ESTIMATE_PATH = DAVIS_OUTPUT_DIR / "latest_motion_estimate.json"
DAVIS_MOTION_TIMELINE_PATH = DAVIS_OUTPUT_DIR / "motion_timeline.jsonl"
DAVIS_MOTION_CALIBRATION_PATH = DAVIS_OUTPUT_DIR / "motion_calibration.json"
DAVIS_MOTION_DEBUG_SUMMARY_PATH = DAVIS_OUTPUT_DIR / "motion_debug_summary.json"
DAVIS_POST_GRASP_MONITORING_DIR = DAVIS_OUTPUT_DIR / "post_grasp_monitoring"
DAVIS_POST_GRASP_LATEST_REPORT_PATH = DAVIS_POST_GRASP_MONITORING_DIR / "latest_report.json"
DAVIS_POST_GRASP_TIMELINE_PATH = DAVIS_POST_GRASP_MONITORING_DIR / "timeline.jsonl"
DAVIS_TIMELINE_MAX_BYTES = max(
    1024 * 1024,
    int(os.environ.get("DAVIS_TIMELINE_MAX_BYTES", str(16 * 1024 * 1024))),
)
DAVIS_BASELINE_DURATION_SEC = 0.5
DAVIS_AUTO_BASELINE_ON_START = True
MICRO_WINDOW_SEC = 0.005
SHORT_WINDOW_SEC = 0.050
IMPACT_WINDOW_SEC = 0.300
SETTLE_WINDOW_SEC = 0.700
STABILITY_WINDOW_SEC = 1.200
CLOSEUP_ACTION_WINDOWS_SEC = {
    "impact": (0.0, IMPACT_WINDOW_SEC),
    "settle": (IMPACT_WINDOW_SEC, SETTLE_WINDOW_SEC),
    "stability": (SETTLE_WINDOW_SEC, STABILITY_WINDOW_SEC),
}
DAVIS_TARGET_DISPLAY_FPS = float(os.environ.get("DAVIS_TARGET_DISPLAY_FPS", "30"))
DAVIS_EVENT_ANALYSIS_FPS = float(os.environ.get("DAVIS_EVENT_ANALYSIS_FPS", "10"))
DAVIS_REPORT_INTERVAL_SEC = float(os.environ.get("DAVIS_REPORT_INTERVAL_SEC", "2.0"))
DAVIS_SAVE_DEBUG_EVERY_FRAME = os.environ.get("DAVIS_SAVE_DEBUG_EVERY_FRAME", "0").strip().lower() in {"1", "true", "yes", "on"}
LIVE_TIMER_MS = max(1, int(round(1000.0 / max(DAVIS_TARGET_DISPLAY_FPS, 1.0))))
LIVE_REPORT_INTERVAL_SEC = DAVIS_REPORT_INTERVAL_SEC
LIVE_HEATMAP_INTERVAL_SEC = 1.0 / max(DAVIS_TARGET_DISPLAY_FPS, 1.0)
LIVE_HEATMAP_WINDOW_SEC = max(0.005, float(os.environ.get("DAVIS_EVENT_HEATMAP_WINDOW_MS", "100")) / 1000.0)
PRIMARY_EVENT_WINDOW_SEC = max(0.005, float(os.environ.get("DAVIS_EVENT_WINDOW_MS", "100")) / 1000.0)
FALLBACK_EVENT_WINDOW_SEC = max(PRIMARY_EVENT_WINDOW_SEC, float(os.environ.get("DAVIS_EVENT_FALLBACK_WINDOW_MS", "800")) / 1000.0)
MIN_PRIMARY_WINDOW_EVENTS = int(os.environ.get("DAVIS_MIN_PRIMARY_WINDOW_EVENTS", "8"))
LIVE_TIMELINE_INTERVAL_SEC = 0.50
LIVE_HEATMAP_MAX_EVENTS = 20000
LIVE_HEATMAP_DECAY = 0.82
LIVE_BACKLOG_DRAIN_BUDGET_SEC = 0.012
LIVE_BACKLOG_DRAIN_MAX_POLLS = 12
LIVE_MAX_EVENTS_PER_TICK = 6000
LIVE_ROLLING_MAX_EVENTS = max(1000, int(25000 * max(float(os.environ.get("DAVIS_EVENT_RING_BUFFER_SEC", "2.0")), 0.1) / 2.0))
BASELINE_MAX_QUIET_RATE = 600.0
STABLE_ABSOLUTE_RATE_MAX = 450.0
GROSS_SLIP_ABSOLUTE_RATE_LOW = 900.0
GROSS_SLIP_DOWNWARD_BIAS_LOW = 0.10
GROSS_SLIP_CENTROID_Y_LOW = 2.0
# DAVIS image Y increases downward (top-left origin), matching the canonical
# event_stream.py slip_down convention and its own simulate/test suite. Keep
# this at 1.0 unless a controlled hardware slip test proves the mounting is
# inverted; do not recalibrate independently of event_stream.py again.
DAVIS_VERTICAL_SLIP_SIGN = 1.0
DAVIS_DEBUG_HEATMAP = os.environ.get("DAVIS_DEBUG_HEATMAP", "0").strip().lower() in {"1", "true", "yes", "on"}
DAVIS_APS_TARGET_DETECTION_INTERVAL_FRAMES = max(
    1,
    int(os.environ.get("DAVIS_APS_TARGET_DETECTION_INTERVAL_FRAMES", "1")),
)
DAVIS_APS_TARGET_MAX_AGE_SEC = 1.25
DAVIS_APS_TARGET_ROI_PADDING_PX = 8
FIXED_CENTER_THIRD_MODE = "fixed_center_third"
FIXED_CENTER_THIRD_BOUNDARY_SOURCE = "CONFIGURED_CENTER_THIRD_UNCALIBRATED"
FIXED_CENTER_THIRD_OBJECT_EVIDENCE_ALIGNMENT = (
    "DAVIS_NATIVE_APS_UNCALIBRATED_NOT_PHYSICALLY_VALIDATED"
)
FIXED_CENTER_THIRD_OBJECT_EVIDENCE_MIN_OVERLAP = 0.60

# object-motion visualization validity gates: a convex hull of recent events is
# only presented as OBJECT motion when it is local, object-specific, and the
# classifier does not report a global disturbance
MOTION_HULL_MAX_AREA_FRACTION = 0.35
MOTION_HULL_MAX_BORDER_TOUCH_RATIO = 0.30
MOTION_HULL_MIN_LOCAL_TO_GLOBAL_RATIO = 1.5
MOTION_HULL_MIN_EVENT_COUNT = 30
MIN_OBJECT_MOTION_CONFIDENCE = 0.55
MOTION_HULL_BORDER_MARGIN_PX = 3.0
GLOBAL_DISTURBANCE_CLASSES = {"LIGHTING_CHANGE", "CAMERA_SHAKE", "BACKGROUND_MOTION"}
OBJECT_SPECIFIC_ROI_SOURCES = {
    "aps_target",
    "realsense_contact_projected",
    "realsense_projected",
}
# RealSense->DAVIS bbox mapping is an uncalibrated side-by-side linear scale, so
# the ROI is object-centered but only approximately placed; motion from it is
# reported at reduced confidence, never as a calibrated projection.
UNCALIBRATED_ROI_SOURCES = {
    "realsense_bbox_mapped",
    "realsense_contour_mapped",
    "realsense_contact_band_mapped",
    "held_preclose_motion_diagnostic",
}
GRASP_ZONE_ROI_SOURCES = {"configured_roi"}


def configured_center_third_roi(frame_size=DAVIS_FRAME_SIZE):
    """Return the deterministic middle third of both DAVIS image axes.

    Remainder pixels stay in the middle partition, so the 346x260 DAVIS frame
    produces [115, 86, 116, 88].  This is a configured diagnostic region, not
    a localized bottle/contact ROI.
    """
    width, height = (max(1, int(value)) for value in frame_size)
    x0, y0 = width // 3, height // 3
    x1, y1 = width - x0, height - y0
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def closeup_window_for_elapsed(elapsed_sec):
    """Return the action-relative close/post-close DAVIS window name."""
    if elapsed_sec is None:
        return "not_started"
    elapsed = max(0.0, float(elapsed_sec))
    if elapsed < IMPACT_WINDOW_SEC:
        return "impact"
    if elapsed < SETTLE_WINDOW_SEC:
        return "settle"
    if elapsed < STABILITY_WINDOW_SEC:
        return "stability"
    return "post_window_rolling"


def closeup_action_window_metrics(events, action_origin, roi_xywh):
    """Compute non-overlapping action-relative metrics without owning a camera.

    The intervals intentionally reuse the canonical EventPostGraspMonitor
    boundaries: [0, 0.3), [0.3, 0.7), and [0.7, 1.2] seconds.
    """
    result = {name: _empty_metrics() for name in CLOSEUP_ACTION_WINDOWS_SEC}
    if action_origin is None:
        return result
    arr = np.asarray(events if events is not None else [], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return result
    x, y, w, h = _clip_roi(roi_xywh)
    inside = (
        (arr[:, 1] >= x) & (arr[:, 1] < x + w)
        & (arr[:, 2] >= y) & (arr[:, 2] < y + h)
    )
    local = arr[inside]
    previous = None
    for name, (start_sec, end_sec) in CLOSEUP_ACTION_WINDOWS_SEC.items():
        start = float(action_origin) + start_sec
        end = float(action_origin) + end_sec
        end_inclusive = name == "stability"
        time_mask = (local[:, 0] >= start) & (
            (local[:, 0] <= end) if end_inclusive else (local[:, 0] < end)
        )
        metrics = _metrics(local[time_mask], end_sec - start_sec, roi_xywh, previous)
        result[name] = metrics
        if metrics["event_count"]:
            previous = metrics
    return result


def _empty_closeup_accumulators():
    return {
        name: {
            "event_count": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_x2": 0.0,
            "sum_y2": 0.0,
            "left": 0,
            "right": 0,
            "upper": 0,
            "lower": 0,
        }
        for name in CLOSEUP_ACTION_WINDOWS_SEC
    }


def accumulate_closeup_action_events(accumulators, events, action_origin, roi_xywh):
    """Increment exact action-window statistics before rolling-buffer caps."""
    if action_origin is None:
        return
    arr = np.asarray(events if events is not None else [], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return
    x, y, w, h = _clip_roi(roi_xywh)
    local = arr[
        (arr[:, 1] >= x) & (arr[:, 1] < x + w)
        & (arr[:, 2] >= y) & (arr[:, 2] < y + h)
    ]
    if local.shape[0] == 0:
        return
    for name, (start_sec, end_sec) in CLOSEUP_ACTION_WINDOWS_SEC.items():
        start = float(action_origin) + start_sec
        end = float(action_origin) + end_sec
        mask = (local[:, 0] >= start) & (
            (local[:, 0] <= end) if name == "stability" else (local[:, 0] < end)
        )
        selected = local[mask]
        if selected.shape[0] == 0:
            continue
        xs = selected[:, 1]
        ys = selected[:, 2]
        row = accumulators[name]
        count = int(selected.shape[0])
        left = int(np.count_nonzero(xs <= x + w / 2.0))
        upper = int(np.count_nonzero(ys <= y + h / 2.0))
        row["event_count"] += count
        row["sum_x"] += float(np.sum(xs))
        row["sum_y"] += float(np.sum(ys))
        row["sum_x2"] += float(np.dot(xs, xs))
        row["sum_y2"] += float(np.dot(ys, ys))
        row["left"] += left
        row["right"] += count - left
        row["upper"] += upper
        row["lower"] += count - upper


def closeup_metrics_from_accumulators(accumulators, roi_xywh):
    x, y, w, h = _clip_roi(roi_xywh)
    area = max(float(w * h), 1.0)
    result = {}
    previous = None
    for name, (start_sec, end_sec) in CLOSEUP_ACTION_WINDOWS_SEC.items():
        row = accumulators[name]
        count = int(row["event_count"])
        if count == 0:
            metrics = _empty_metrics()
        else:
            cx = row["sum_x"] / count
            cy = row["sum_y"] / count
            std_x = float(np.sqrt(max(0.0, row["sum_x2"] / count - cx * cx)))
            std_y = float(np.sqrt(max(0.0, row["sum_y2"] / count - cy * cy)))
            prev_x = previous.get("centroid_x") if previous else None
            prev_y = previous.get("centroid_y") if previous else None
            metrics = {
                "event_count": count,
                "event_rate": float(count / max(end_sec - start_sec, 1e-6)),
                "event_density": float(count / area),
                "centroid_x": cx,
                "centroid_y": cy,
                "centroid_shift_x": 0.0 if prev_x is None else float(cx - prev_x),
                "centroid_shift_y": 0.0 if prev_y is None else float(cy - prev_y),
                "vertical_event_bias": float((row["lower"] - row["upper"]) / count),
                "horizontal_event_bias": float((row["right"] - row["left"]) / count),
                "left_right_asymmetry": float(abs(row["right"] - row["left"]) / count),
                "top_bottom_asymmetry": float(abs(row["lower"] - row["upper"]) / count),
                "event_cluster_spread": float(std_x / max(w, 1) + std_y / max(h, 1)),
                "event_persistence": float(min(100.0, count / 8.0)),
            }
            previous = metrics
        result[name] = metrics
    return result


def _reprojection_confidence(error):
    try:
        error = float(error)
    except (TypeError, ValueError):
        return 0.0, False
    if not np.isfinite(error) or error < 0.0:
        return 0.0, False
    return float(np.clip(1.0 - error / 8.0, 0.0, 1.0)), True


def roi_projection_diagnostics(roi_source, metadata=None):
    """Normalize ROI provenance without claiming unmeasured alignment accuracy."""
    source = str(roi_source or "full_frame")
    meta = dict(metadata or {})
    method = str(meta.get("mapping_mode") or "")
    status = str(meta.get("alignment_status") or "")
    warning = None
    if source == "realsense_contact_projected":
        method = method or "inverse_2d_homography_contact_band"
        status = status or "CALIBRATED_2D_REFERENCE_PLANE"
        confidence, confidence_available = _reprojection_confidence(meta.get("reprojection_error_px"))
        warning = (
            "Reference-plane reprojection confidence is unavailable; object motion is suppressed."
            if not confidence_available
            else "Reference-plane 2D calibration only; physical contact alignment is not validated."
        )
    elif source == "realsense_projected":
        method = method or "inverse_2d_homography_bottle_roi"
        status = status or "CALIBRATED_2D_REFERENCE_PLANE"
        confidence, confidence_available = _reprojection_confidence(meta.get("reprojection_error_px"))
        warning = (
            "Reference-plane reprojection confidence is unavailable; object motion is suppressed."
            if not confidence_available
            else "Reference-plane 2D calibration only; physical boundary alignment is not validated."
        )
    elif source == "aps_target":
        method = method or "davis_aps_native_target_roi"
        status = status or "DAVIS_NATIVE_ESTIMATED"
        try:
            confidence = float(np.clip(float(meta.get("target_confidence")), 0.0, 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        warning = "DAVIS APS target localization is estimated, not physical contact calibration."
    elif source in UNCALIBRATED_ROI_SOURCES:
        method = method or "fixed_linear_side_by_side"
        status = status or "APPROXIMATE_UNCALIBRATED"
        confidence = 0.0
        warning = "Uncalibrated RealSense-to-DAVIS mapping is diagnostic-only and cannot validate object motion."
    elif source == "configured_roi":
        method = method or "configured_diagnostic_roi"
        status = status or "DIAGNOSTIC_NOT_OBJECT_SPECIFIC"
        confidence = 0.0
        warning = "Configured ROI is diagnostic-only unless independently localized to the object."
    else:
        method = method or "full_frame_fallback"
        status = status or "NO_OBJECT_ROI"
        confidence = 0.0
        warning = "Full-frame events are event-health/disturbance evidence only."
    return {
        "roi_confidence": round(confidence, 3),
        "roi_confidence_status": "ESTIMATED_FROM_PROJECTION_EVIDENCE_NOT_PHYSICAL_ACCURACY",
        "roi_projection_method": method,
        "roi_projection_status": status,
        "roi_projection_warning": warning,
        "roi_projection_physically_validated": False,
    }


def _aps_native_midbody_band(detection):
    """Return an estimated APS image-space mid-body band, never physical contact."""
    selected = detection.get("selected_candidate") or {}
    center = detection.get("davis_bottle_center_px") or selected.get("center_px")
    axis = detection.get("davis_axis_vector_px") or selected.get("axis_vector_px")
    length = detection.get("davis_oriented_length_px") or selected.get("oriented_length_px")
    width = detection.get("davis_oriented_width_px") or selected.get("oriented_width_px")
    try:
        center = np.asarray(center, dtype=float).reshape(2)
        axis = np.asarray(axis, dtype=float).reshape(2)
        length = float(length)
        width = float(width)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(center).all() or not np.isfinite(axis).all() or norm <= 1e-9 or length <= 0.0 or width <= 0.0:
        return None
    axis /= norm
    normal = np.asarray([-axis[1], axis[0]], dtype=float)
    axial_half = max(3.0, 0.11 * length)
    radial_half = max(3.0, 0.55 * width)
    points = np.asarray([
        center - axis * axial_half - normal * radial_half,
        center - axis * axial_half + normal * radial_half,
        center + axis * axial_half + normal * radial_half,
        center + axis * axial_half - normal * radial_half,
    ])
    points[:, 0] = np.clip(points[:, 0], 0.0, DAVIS_FRAME_SIZE[0] - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, DAVIS_FRAME_SIZE[1] - 1.0)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def _bbox_iou_xyxy(first, second):
    if first is None or second is None:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def aps_target_context_from_detection(
    detection,
    *,
    frame_id,
    captured_monotonic,
    host_timestamp=None,
):
    """Convert one current, high-confidence APS bottle detection to a DAVIS ROI.

    The reused APS detector owns the confidence decision.  Low-confidence or
    malformed detections return no context, so full-frame/configured fallbacks
    remain health diagnostics rather than being mislabeled as object motion.
    """
    detection = detection or {}
    bbox = detection.get("davis_bbox_xyxy")
    if detection.get("davis_bottle_detected") is not True or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        confidence_percent = float(detection.get("davis_detection_confidence"))
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite([x1, y1, x2, y2, confidence_percent])) or x2 <= x1 or y2 <= y1:
        return None
    padding = float(DAVIS_APS_TARGET_ROI_PADDING_PX)
    roi_xywh = _clip_roi([
        x1 - padding,
        y1 - padding,
        (x2 - x1) + 2.0 * padding,
        (y2 - y1) + 2.0 * padding,
    ])
    x, y, width, height = roi_xywh
    confidence = float(np.clip(confidence_percent / 100.0, 0.0, 1.0))
    contour = detection.get("davis_contour_points")
    if contour is None:
        contour = (detection.get("selected_candidate") or {}).get("contour_points")
    try:
        contour_array = np.asarray(contour, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        contour_array = np.empty((0, 2), dtype=float)
    if contour_array.shape[0] >= 3 and np.isfinite(contour_array).all():
        contour_array[:, 0] = np.clip(contour_array[:, 0], 0.0, DAVIS_FRAME_SIZE[0] - 1.0)
        contour_array[:, 1] = np.clip(contour_array[:, 1], 0.0, DAVIS_FRAME_SIZE[1] - 1.0)
        boundary_points = contour_array.tolist()
        boundary_source = "davis_aps_target_contour"
    else:
        boundary_points = [
            [float(x), float(y)],
            [float(x + width), float(y)],
            [float(x + width), float(y + height)],
            [float(x), float(y + height)],
        ]
        boundary_source = "davis_aps_target_bbox_fallback"
    contact_band_points = _aps_native_midbody_band(detection)
    return {
        "roi_xywh": list(roi_xywh),
        "boundary_points": boundary_points,
        "contact_band_points": contact_band_points,
        "contact_band_source": (
            "aps_native_estimated_midbody_band"
            if contact_band_points is not None
            else "unavailable_without_oriented_contour"
        ),
        "contact_band_physically_validated": False,
        "roi_source": "aps_target",
        "roi_priority_selected": "davis_aps_target_roi",
        "boundary_source": boundary_source,
        "mapping_mode": "davis_aps_native_target_roi",
        "alignment_status": "DAVIS_NATIVE_APS",
        "mapping_reason": detection.get("detection_reason") or "current DAVIS APS bottle target",
        "mapping_physically_validated": False,
        "mapping_limitations": [
            "DAVIS APS localization is image-space object evidence, not physical contact or RealSense-to-DAVIS calibration.",
            "The APS-native mid-body band is an uncalibrated planned neighborhood, not a measured gripper contact band.",
        ],
        "calibration_reprojection_error_px": None,
        "projection_metadata": {
            "mapping_mode": "davis_aps_native_target_roi",
            "alignment_status": "DAVIS_NATIVE_APS",
            "target_confidence": confidence,
        },
        "source_frame_id": int(frame_id),
        "source_writer_session_id": None,
        "source_lock_frame_id": None,
        "source_attempt_id": None,
        "source_bottle_target_id": None,
        "source_target_status": detection.get("davis_detection_status"),
        "bottle_axis_vector_px": detection.get("davis_axis_vector_px"),
        "source_depth_mm": None,
        "target_confidence": confidence,
        "aps_detection_confidence": confidence_percent,
        "aps_detection_source": detection.get("aps_detection_source"),
        "aps_detection_backend": detection.get("davis_detection_backend"),
        "aps_host_timestamp": host_timestamp,
        "captured_monotonic": float(captured_monotonic),
    }


def aps_object_evidence_within_acquisition(
    context,
    acquisition_roi_xywh,
    *,
    now=None,
    min_overlap_fraction=FIXED_CENTER_THIRD_OBJECT_EVIDENCE_MIN_OVERLAP,
):
    """Validate and clip confirmed APS bottle evidence to a fixed acquisition ROI.

    The fixed center-third remains the user-visible acquisition region.  This
    helper creates a separate, smaller evidence ROI from a current confirmed
    DAVIS-native APS contour. It never claims cross-camera identity or upgrades
    the configured ROI itself to calibrated object evidence.
    """
    if not isinstance(context, dict) or context.get("roi_source") != "aps_target":
        return None, "aps_object_evidence_unavailable"
    if context.get("aps_target_confirmed") is not True:
        return None, "aps_object_evidence_not_confirmed"
    if str(context.get("boundary_source") or "") != "davis_aps_target_contour":
        return None, "aps_object_evidence_requires_confirmed_contour"

    current = time.monotonic() if now is None else float(now)
    try:
        captured = float(context.get("captured_monotonic"))
        max_age = float(
            context.get("aps_target_context_max_age_sec")
            or DAVIS_APS_TARGET_MAX_AGE_SEC
        )
    except (TypeError, ValueError):
        return None, "aps_object_evidence_timestamp_invalid"
    age = current - captured
    if not np.isfinite(age) or age < 0.0 or age > max(0.05, max_age):
        return None, "aps_object_evidence_stale"

    try:
        ax, ay, aw, ah = _clip_roi(acquisition_roi_xywh)
        ox, oy, ow, oh = _clip_roi(context.get("roi_xywh"))
    except (TypeError, ValueError):
        return None, "aps_object_evidence_roi_invalid"
    ix0, iy0 = max(ax, ox), max(ay, oy)
    ix1, iy1 = min(ax + aw, ox + ow), min(ay + ah, oy + oh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    overlap = float(iw * ih) / max(float(ow * oh), 1.0)
    if iw <= 0 or ih <= 0 or overlap < float(min_overlap_fraction):
        return None, f"aps_object_evidence_outside_acquisition_overlap_{overlap:.3f}"

    try:
        boundary = np.asarray(context.get("boundary_points"), dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None, "aps_object_evidence_boundary_invalid"
    if boundary.shape[0] < 3 or not np.isfinite(boundary).all():
        return None, "aps_object_evidence_boundary_invalid"
    boundary[:, 0] = np.clip(boundary[:, 0], float(ix0), float(ix1 - 1))
    boundary[:, 1] = np.clip(boundary[:, 1], float(iy0), float(iy1 - 1))
    if np.unique(np.round(boundary, 3), axis=0).shape[0] < 3:
        return None, "aps_object_evidence_boundary_collapsed_after_clip"

    contact = context.get("contact_band_points")
    if contact is not None:
        try:
            contact_array = np.asarray(contact, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            contact_array = np.empty((0, 2), dtype=float)
        if contact_array.shape[0] >= 3 and np.isfinite(contact_array).all():
            contact_array[:, 0] = np.clip(contact_array[:, 0], float(ix0), float(ix1 - 1))
            contact_array[:, 1] = np.clip(contact_array[:, 1], float(iy0), float(iy1 - 1))
            contact = (
                contact_array.tolist()
                if np.unique(np.round(contact_array, 3), axis=0).shape[0] >= 3
                else None
            )
        else:
            contact = None

    prepared = dict(context)
    prepared.update({
        "roi_xywh": [int(ix0), int(iy0), int(iw), int(ih)],
        "boundary_points": boundary.tolist(),
        "boundary_source": "davis_aps_target_contour_within_fixed_center_third_uncalibrated",
        "contact_band_points": contact,
        "alignment_status": FIXED_CENTER_THIRD_OBJECT_EVIDENCE_ALIGNMENT,
        "mapping_mode": "davis_aps_object_evidence_within_fixed_center_third",
        "mapping_physically_validated": False,
        "aps_object_evidence_overlap_ratio": round(overlap, 4),
        "aps_object_evidence_acquisition_roi_xywh": [int(ax), int(ay), int(aw), int(ah)],
        "aps_object_evidence_status": "ESTIMATED_OBJECT_LOCAL_UNCALIBRATED",
        "projection_metadata": {
            "mapping_mode": "davis_aps_object_evidence_within_fixed_center_third",
            "alignment_status": FIXED_CENTER_THIRD_OBJECT_EVIDENCE_ALIGNMENT,
            "target_confidence": context.get("target_confidence"),
        },
    })
    return prepared, "confirmed_davis_native_aps_contour_uncalibrated"


def roi_context_change_is_material(
    previous_roi,
    current_roi,
    previous_source,
    current_source,
    previous_projection=None,
    current_projection=None,
):
    """Return whether baseline/event history is unsafe to reuse for the ROI."""
    if str(previous_source or "full_frame") != str(current_source or "full_frame"):
        return True
    previous_projection = previous_projection or {}
    current_projection = current_projection or {}
    for key in ("roi_projection_method", "roi_projection_status"):
        if str(previous_projection.get(key) or "") != str(current_projection.get(key) or ""):
            return True

    old_x, old_y, old_w, old_h = _clip_roi(previous_roi)
    new_x, new_y, new_w, new_h = _clip_roi(current_roi)
    old_area = float(old_w * old_h)
    new_area = float(new_w * new_h)
    intersection_w = max(0.0, min(old_x + old_w, new_x + new_w) - max(old_x, new_x))
    intersection_h = max(0.0, min(old_y + old_h, new_y + new_h) - max(old_y, new_y))
    intersection = intersection_w * intersection_h
    union = max(old_area + new_area - intersection, 1.0)
    area_ratio = new_area / max(old_area, 1.0)
    return bool(intersection / union < 0.85 or area_ratio < 0.75 or area_ratio > 1.33)


def fuse_boundary_motion_evidence(scores, boundary_field, *, allow_classification=True):
    """Add canonical-gated boundary cues to the existing classifier result."""
    fused = dict(scores or {})
    field = boundary_field or {}
    if field.get("object_motion_valid") is not True:
        return fused
    fused["incipient_slip_score"] = max(float(fused.get("incipient_slip_score") or 0.0), float(field.get("incipient_slip_precursor_score") or 0.0))
    fused["gross_slip_score"] = max(float(fused.get("gross_slip_score") or 0.0), float(field.get("slip_cue") or 0.0))
    fused["rotation_risk_score"] = max(float(fused.get("rotation_risk_score") or 0.0), float(field.get("rotation_cue") or 0.0))
    fused["vibration_risk_score"] = max(float(fused.get("vibration_risk_score") or 0.0), float(field.get("vibration_cue") or 0.0))
    fused["stability_score"] = min(
        float(fused.get("stability_score") if fused.get("stability_score") is not None else 100.0),
        100.0 - max(
            fused["incipient_slip_score"], fused["gross_slip_score"],
            fused["rotation_risk_score"], fused["vibration_risk_score"],
        ),
    )
    current = str(fused.get("raw_motion_class") or fused.get("motion_class") or "STABLE")
    priority = {"STABLE": 0, "IMPACT_ONLY": 0, "INCIPIENT_SLIP": 1, "SHAKE_OR_VIBRATION": 2, "SPIN_OR_ROTATION": 3, "GROSS_SLIP": 4}
    candidate = "STABLE"
    if fused["gross_slip_score"] >= 55.0:
        candidate = "GROSS_SLIP"
    elif fused["rotation_risk_score"] >= 45.0:
        candidate = "SPIN_OR_ROTATION"
    elif fused["vibration_risk_score"] >= 45.0:
        candidate = "SHAKE_OR_VIBRATION"
    elif fused["incipient_slip_score"] >= 45.0:
        candidate = "INCIPIENT_SLIP"
    if allow_classification and priority.get(candidate, 0) > priority.get(current, 0):
        fused["raw_motion_class"] = candidate
        fused["motion_class"] = simplified_motion_class(candidate)
        fused["decision"] = {
            "INCIPIENT_SLIP": "HOLD_AND_RECHECK",
            "SHAKE_OR_VIBRATION": "HOLD_AND_RECHECK",
            "SPIN_OR_ROTATION": "REGRIP",
            "GROSS_SLIP": "DO_NOT_LIFT",
        }[candidate]
        fused["visual_safety_response"] = fused["decision"]
        fused["boundary_evidence_changed_classification"] = True
    return fused


def apply_object_motion_presentation_gate(scores, gates):
    """Keep raw event activity diagnostic but suppress object-motion claims."""
    gated = dict(scores or {})
    validity = gates or {}
    raw_class = str(gated.get("raw_motion_class") or gated.get("motion_class") or "UNKNOWN_MOTION")
    gated["event_activity_class"] = raw_class
    gated["raw_event_activity_scores"] = {
        key: float(gated.get(key) or 0.0)
        for key in (
            "incipient_slip_score",
            "gross_slip_score",
            "rotation_risk_score",
            "vibration_risk_score",
        )
    }
    stage = str(validity.get("davis_motion_stage") or "").upper()
    if (
        validity.get("object_motion_valid") is True
        or validity.get("grasp_zone_motion_valid") is True
    ) and stage != "PRE_CLOSE":
        safety_class = simplified_motion_class(raw_class)
        gated["object_motion_class"] = (
            safety_class
            if validity.get("object_motion_valid") is True
            else "NOT_OBJECT_SPECIFIC"
        )
        gated["grasp_zone_motion_class"] = safety_class
        gated["motion_class"] = safety_class
        gated["davis_safety_class"] = safety_class
        return gated

    for key in (
        "incipient_slip_score",
        "gross_slip_score",
        "rotation_risk_score",
        "vibration_risk_score",
    ):
        gated[key] = 0.0
    if validity.get("object_motion_valid") is True and stage == "PRE_CLOSE":
        display_class = "EVENT ACTIVITY"
        object_motion_class = "PRE-CLOSE MOTION"
        response_reason = (
            "DAVIS object-local motion is visible as an approach diagnostic, "
            "but PRE_CLOSE events are excluded from grasp-slip quality."
        )
    elif raw_class in GLOBAL_DISTURBANCE_CLASSES:
        display_class = simplified_motion_class(raw_class)
        object_motion_class = "NOT VALID"
        response_reason = (
            "DAVIS event activity is diagnostic only; object-motion validity rejected it: "
            f"{validity.get('object_motion_reason') or 'object_motion_invalid'}"
        )
    else:
        display_class = "EVENT ACTIVITY"
        object_motion_class = "NOT VALID"
        response_reason = (
            "DAVIS event activity is diagnostic only; object-motion validity rejected it: "
            f"{validity.get('object_motion_reason') or 'object_motion_invalid'}"
        )
    gated["object_motion_class"] = object_motion_class
    gated["motion_class"] = display_class
    gated["davis_safety_class"] = display_class
    gated["decision"] = "HOLD_AND_RECHECK"
    gated["visual_safety_response"] = "HOLD_AND_RECHECK"
    gated["response_reason"] = response_reason
    return gated


def simplified_motion_class(motion_class):
    mapping = {
        "STABLE": "STABLE",
        "IMPACT_ONLY": "STABLE",
        "INCIPIENT_SLIP": "MICRO SLIP",
        "GROSS_SLIP": "GROSS SLIP",
        "SPIN_OR_ROTATION": "ROTATION",
        "SHAKE_OR_VIBRATION": "VIBRATION",
        "CAMERA_SHAKE": "CAMERA SHAKE",
        "LIGHTING_CHANGE": "LIGHTING CHANGE",
        "BACKGROUND_MOTION": "BACKGROUND MOTION",
        "UNKNOWN_MOTION": "UNSAFE",
        "SENSOR_UNAVAILABLE": "UNSAFE",
    }
    return mapping.get(str(motion_class or "UNKNOWN_MOTION"), "UNSAFE")


def canonical_disturbance_type(raw_motion_class, provisional_type=None):
    """Keep the published disturbance label consistent with the gated class."""
    raw = str(raw_motion_class or "").upper()
    global_types = {
        "LIGHTING_CHANGE": "lighting_change",
        "CAMERA_SHAKE": "camera_shake",
        "BACKGROUND_MOTION": "background_motion",
    }
    if raw in global_types:
        return global_types[raw]
    provisional = str(provisional_type or "none").lower()
    if provisional in {"lighting_change", "camera_shake", "background_motion", "unknown_global_motion"}:
        return provisional
    return "none"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(_json_safe(payload), f, indent=4)
        replace_path_retry(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _append_jsonl(path, payload):
    return append_jsonl_bounded(
        path,
        _json_safe(payload),
        max_bytes=DAVIS_TIMELINE_MAX_BYTES,
    )


def _write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(payload)
        replace_path_retry(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _write_rgb_jpeg(path, rgb, quality=82):
    arr = np.asarray(rgb)
    if arr.size == 0:
        return False
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    _write_bytes(path, encoded.tobytes())
    return True


def _empty_metrics():
    return {
        "event_count": 0,
        "event_rate": 0.0,
        "event_density": 0.0,
        "centroid_x": None,
        "centroid_y": None,
        "centroid_shift_x": 0.0,
        "centroid_shift_y": 0.0,
        "vertical_event_bias": 0.0,
        "horizontal_event_bias": 0.0,
        "left_right_asymmetry": 0.0,
        "top_bottom_asymmetry": 0.0,
        "event_cluster_spread": 0.0,
        "event_persistence": 0.0,
    }


def _metrics(events, duration, roi_xywh, previous=None):
    x, y, w, h = _clip_roi(roi_xywh)
    area = max(float(w * h), 1.0)
    if events.size == 0:
        out = _empty_metrics()
        out["event_rate"] = 0.0
        return out
    xs = events[:, 1]
    ys = events[:, 2]
    count = int(events.shape[0])
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    left = int(np.count_nonzero(xs <= x + w / 2.0))
    right = count - left
    upper = int(np.count_nonzero(ys <= y + h / 2.0))
    lower = count - upper
    spread = float((np.std(xs) / max(w, 1)) + (np.std(ys) / max(h, 1)))
    prev_x = previous.get("centroid_x") if previous else None
    prev_y = previous.get("centroid_y") if previous else None
    return {
        "event_count": count,
        "event_rate": float(count / max(duration, 1e-6)),
        "event_density": float(count / area),
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_shift_x": 0.0 if prev_x is None else float(cx - prev_x),
        "centroid_shift_y": 0.0 if prev_y is None else float(cy - prev_y),
        "vertical_event_bias": float((lower - upper) / max(count, 1)),
        "horizontal_event_bias": float((right - left) / max(count, 1)),
        "left_right_asymmetry": float(abs(right - left) / max(count, 1)),
        "top_bottom_asymmetry": float(abs(lower - upper) / max(count, 1)),
        "event_cluster_spread": spread,
        "event_persistence": float(min(100.0, count / 8.0)),
    }


def _events_since(events, now, seconds, roi_xywh):
    if events is None:
        arr = np.empty((0, 4), dtype=np.float64)
    else:
        arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    x, y, w, h = _clip_roi(roi_xywh)
    recent = arr[(arr[:, 0] >= now - seconds) & (arr[:, 0] <= now)]
    in_roi = (recent[:, 1] >= x) & (recent[:, 1] < x + w) & (recent[:, 2] >= y) & (recent[:, 2] < y + h)
    return recent[in_roi]


def select_event_analysis_window(
    events, analysis_end, roi_xywh, *, primary_window_sec=None,
    fallback_window_sec=None, min_primary_events=None,
):
    """Select an adaptive window, optionally using a recorded replay config."""
    primary_sec = PRIMARY_EVENT_WINDOW_SEC if primary_window_sec is None else primary_window_sec
    fallback_sec = FALLBACK_EVENT_WINDOW_SEC if fallback_window_sec is None else fallback_window_sec
    minimum = MIN_PRIMARY_WINDOW_EVENTS if min_primary_events is None else min_primary_events
    for name, value in (
        ("primary_window_sec", primary_window_sec), ("fallback_window_sec", fallback_window_sec),
        ("min_primary_events", min_primary_events),
    ):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be positive and finite") from exc
        if isinstance(value, bool) or not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be positive and finite")
    primary_sec, fallback_sec = float(primary_sec), float(fallback_sec)
    if float(minimum) != int(float(minimum)):
        raise ValueError("min_primary_events must be a positive integer")
    minimum = int(float(minimum))
    end = float(analysis_end)
    primary = _events_since(events, end, primary_sec, roi_xywh)
    window_sec = primary_sec
    mode = f"primary_{round(primary_sec * 1000.0)}ms"
    if primary.shape[0] < minimum and fallback_sec > primary_sec:
        fallback = _events_since(events, end, fallback_sec, roi_xywh)
        if fallback.shape[0] > primary.shape[0]:
            primary = fallback
            window_sec = fallback_sec
            mode = f"fallback_{round(fallback_sec * 1000.0)}ms"
    return primary, window_sec, mode, end


def _score(value, low, high):
    return float(np.clip(100.0 * (float(value) - low) / max(high - low, 1e-6), 0.0, 100.0))


def classify_motion(micro, short, impact, settle, stability, baseline, roi_source):
    rate_delta = max(0.0, stability["event_rate"] - baseline.get("event_rate", 0.0))
    density_delta = max(0.0, stability["event_density"] - baseline.get("event_density", 0.0))
    vertical_delta = stability["vertical_event_bias"] - baseline.get("vertical_event_bias", 0.0)
    horizontal_delta = abs(stability["horizontal_event_bias"] - baseline.get("horizontal_event_bias", 0.0))
    asym_delta = max(
        abs(stability["left_right_asymmetry"] - baseline.get("left_right_asymmetry", 0.0)),
        abs(stability["top_bottom_asymmetry"] - baseline.get("top_bottom_asymmetry", 0.0)),
    )
    impact_score = _score(impact["event_rate"], 500.0, 7000.0)
    impact_decay = (settle["event_rate"] + stability["event_rate"]) / max(impact["event_rate"], 1.0)
    post_rate_abs = max(short["event_rate"], settle["event_rate"], stability["event_rate"])
    signed_short_v = DAVIS_VERTICAL_SLIP_SIGN * short["vertical_event_bias"]
    signed_settle_v = DAVIS_VERTICAL_SLIP_SIGN * settle["vertical_event_bias"]
    signed_stability_v = DAVIS_VERTICAL_SLIP_SIGN * stability["vertical_event_bias"]
    signed_short_dy = DAVIS_VERTICAL_SLIP_SIGN * short["centroid_shift_y"]
    signed_settle_dy = DAVIS_VERTICAL_SLIP_SIGN * settle["centroid_shift_y"]
    signed_stability_dy = DAVIS_VERTICAL_SLIP_SIGN * stability["centroid_shift_y"]
    vertical_motion_abs = max(abs(short["vertical_event_bias"]), abs(settle["vertical_event_bias"]), abs(stability["vertical_event_bias"]))
    vertical_centroid_abs = max(abs(short["centroid_shift_y"]), abs(settle["centroid_shift_y"]), abs(stability["centroid_shift_y"]))
    downward_bias_abs = max(signed_short_v, signed_settle_v, signed_stability_v, 0.0)
    downward_centroid_abs = max(signed_short_dy, signed_settle_dy, signed_stability_dy, 0.0)
    downward_persistence_windows = sum(
        1 for vb, dy, m in (
            (abs(short["vertical_event_bias"]), abs(short["centroid_shift_y"]), short),
            (abs(settle["vertical_event_bias"]), abs(settle["centroid_shift_y"]), settle),
            (abs(stability["vertical_event_bias"]), abs(stability["centroid_shift_y"]), stability),
        )
        if m["event_rate"] >= GROSS_SLIP_ABSOLUTE_RATE_LOW
        and (vb >= GROSS_SLIP_DOWNWARD_BIAS_LOW or dy >= GROSS_SLIP_CENTROID_Y_LOW)
    )
    localized = max(0.0, 100.0 - _score(stability["event_cluster_spread"], 0.45, 1.05))
    persistence = 0.5 * (short["event_persistence"] > 10.0) * 100.0 + 0.5 * (stability["event_persistence"] > 10.0) * 100.0
    incipient = np.clip(
        0.28 * _score(rate_delta, 40.0, 700.0)
        + 0.22 * localized
        + 0.22 * persistence
        + 0.18 * _score(max(signed_stability_dy, 0.0), 0.8, 8.0)
        + 0.10 * _score(micro["event_count"], 1.0, 20.0),
        0.0,
        100.0,
    )
    gross = np.clip(
        0.24 * _score(max(vertical_delta, 0.0), 0.08, 0.40)
        + 0.30 * _score(vertical_motion_abs, GROSS_SLIP_DOWNWARD_BIAS_LOW, 0.42)
        + 0.20 * _score(vertical_centroid_abs, GROSS_SLIP_CENTROID_Y_LOW, 24.0)
        + 0.14 * _score(rate_delta, 500.0, 3000.0)
        + 0.10 * _score(post_rate_abs, GROSS_SLIP_ABSOLUTE_RATE_LOW, 4500.0)
        + 0.06 * (100.0 if downward_persistence_windows >= 2 else 0.0),
        0.0,
        100.0,
    )
    if vertical_motion_abs < GROSS_SLIP_DOWNWARD_BIAS_LOW and vertical_centroid_abs < GROSS_SLIP_CENTROID_Y_LOW:
        gross *= 0.25
    rotation = np.clip(
        0.36 * _score(stability["left_right_asymmetry"], 0.10, 0.42)
        + 0.28 * _score(abs(stability["centroid_shift_x"]), 2.0, 24.0)
        + 0.20 * _score(horizontal_delta, 0.08, 0.36)
        + 0.16 * _score(rate_delta, 350.0, 3000.0),
        0.0,
        100.0,
    )
    vibration = np.clip(
        0.34 * _score(stability["event_cluster_spread"], 0.35, 1.05)
        + 0.28 * _score(density_delta, 0.004, 0.045)
        + 0.20 * _score(rate_delta, 600.0, 4500.0)
        + 0.18 * (100.0 - max(gross, rotation)),
        0.0,
        100.0,
    )
    broad_global = stability["event_cluster_spread"] > 0.85 and localized < 35.0
    if broad_global:
        incipient *= 0.45
        gross *= 0.55
    impact_only = impact_score > 55.0 and impact_decay < 0.30 and stability["event_rate"] < max(STABLE_ABSOLUTE_RATE_MAX, min(baseline.get("event_rate", 0.0) + 250.0, 900.0))
    if impact_only:
        incipient *= 0.35
        gross *= 0.35
        rotation *= 0.45
        vibration *= 0.45
    stability_score = float(np.clip(100.0 - max(incipient, gross, rotation, vibration), 0.0, 100.0))
    disturbance_type = "none"
    if broad_global and rate_delta > 500:
        disturbance_type = "camera_shake" if stability["event_cluster_spread"] > 1.0 else "lighting_change"
    elif rate_delta > 200 and localized >= 35:
        disturbance_type = "object_motion"
    elif rate_delta > 200:
        disturbance_type = "unknown_global_motion"
    stable_rate_gate = max(80.0, min(baseline.get("event_rate", 0.0) + 80.0, STABLE_ABSOLUTE_RATE_MAX))
    if stability["event_rate"] < stable_rate_gate and post_rate_abs < max(120.0, STABLE_ABSOLUTE_RATE_MAX):
        motion_class = "STABLE"
    elif impact_only:
        motion_class = "IMPACT_ONLY"
    elif gross >= 55:
        motion_class = "GROSS_SLIP"
    elif incipient >= 45:
        motion_class = "INCIPIENT_SLIP"
    elif rotation >= 45 and gross < 55:
        motion_class = "SPIN_OR_ROTATION"
    elif vibration >= 45:
        motion_class = "SHAKE_OR_VIBRATION"
    else:
        motion_class = "UNKNOWN_MOTION"
    decision_map = {
        "STABLE": "READY_TO_LIFT",
        "IMPACT_ONLY": "HOLD_AND_RECHECK" if stability_score < 75 else "READY_TO_LIFT",
        "INCIPIENT_SLIP": "HOLD_AND_RECHECK" if incipient < 70 else "REGRIP",
        "GROSS_SLIP": "DO_NOT_LIFT",
        "SPIN_OR_ROTATION": "REGRIP",
        "SHAKE_OR_VIBRATION": "HOLD_AND_RECHECK" if vibration < 70 else "REGRIP",
        "UNKNOWN_MOTION": "HOLD_AND_RECHECK",
        "SENSOR_UNAVAILABLE": "UNAVAILABLE",
    }
    return {
        "baseline_event_rate": baseline.get("event_rate", 0.0),
        "event_rate_delta": rate_delta,
        "post_rate_abs": post_rate_abs,
        "downward_bias_abs": downward_bias_abs,
        "downward_centroid_abs": downward_centroid_abs,
        "vertical_motion_abs": vertical_motion_abs,
        "vertical_centroid_abs": vertical_centroid_abs,
        "downward_persistence_windows": downward_persistence_windows,
        "davis_vertical_slip_sign": DAVIS_VERTICAL_SLIP_SIGN,
        "stable_rate_gate": stable_rate_gate,
        "event_density_delta": density_delta,
        "vertical_bias_delta": vertical_delta,
        "horizontal_bias_delta": horizontal_delta,
        "asymmetry_delta": asym_delta,
        "roi_source": roi_source,
        "local_to_global_event_ratio": 1.0 if roi_source == "full_frame" else None,
        "outside_roi_event_rate": None if roi_source == "full_frame" else 0.0,
        "global_event_ratio": 1.0 if roi_source == "full_frame" else None,
        "background_motion_score": 75.0 if disturbance_type in {"camera_shake", "lighting_change", "unknown_global_motion"} else 0.0,
        "camera_shake_score": 80.0 if disturbance_type == "camera_shake" else 0.0,
        "lighting_flicker_score": 80.0 if disturbance_type == "lighting_change" else 0.0,
        "disturbance_type": disturbance_type,
        "incipient_slip_score": float(incipient),
        "gross_slip_score": float(gross),
        "rotation_risk_score": float(rotation),
        "vibration_risk_score": float(vibration),
        "impact_score": float(impact_score),
        "stability_score": stability_score,
        "confidence_score": float(np.clip(max(rate_delta, impact["event_rate"]) / 1800.0 * 100.0, 20.0 if motion_class == "STABLE" else 0.0, 100.0)),
        "motion_class": motion_class,
        "visual_safety_response": decision_map[motion_class],
        "decision": decision_map[motion_class],
        "response_reason": f"{motion_class}: rate_delta={rate_delta:.0f}/s, disturbance={disturbance_type}, full-frame analysis={roi_source == 'full_frame'}",
    }


def heatmap_rgb(
    events,
    roi_xywh,
    frame_size=DAVIS_FRAME_SIZE,
    background_rgb=None,
    neighborhood_filter=True,
):
    """Render a readable event overlay without changing analysis events.

    Isolated display pixels are suppressed by a 3x3 neighborhood check and
    older events fade within the already bounded heatmap window. Event counts,
    motion estimation, and all safety gates continue to consume the raw array.
    """
    width, height = frame_size
    heat = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint16)
    if events is None:
        arr = np.empty((0, 4), dtype=np.float64)
    else:
        arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        arr = np.empty((0, 4), dtype=np.float64)
    if arr.size:
        if arr.shape[0] > LIVE_HEATMAP_MAX_EVENTS:
            arr = arr[-LIVE_HEATMAP_MAX_EVENTS:]
        xs = np.clip(arr[:, 1].astype(np.int32), 0, width - 1)
        ys = np.clip(arr[:, 2].astype(np.int32), 0, height - 1)
        latest_timestamp = float(np.max(arr[:, 0]))
        event_span = max(float(np.ptp(arr[:, 0])), 0.005)
        decay_steps = np.maximum(0.0, (latest_timestamp - arr[:, 0]) / max(event_span * 0.25, 0.001))
        weights = np.power(float(LIVE_HEATMAP_DECAY), decay_steps).astype(np.float32)
        np.add.at(heat, (ys, xs), weights)
        np.add.at(counts, (ys, xs), 1)
    if neighborhood_filter and np.any(counts):
        occupied = (counts > 0).astype(np.uint8)
        neighbors = cv2.filter2D(
            occupied,
            cv2.CV_16U,
            np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
        )
        keep = (counts >= 2) | (neighbors >= 2)
        heat = np.where(keep, heat, 0.0)
    t = np.log1p(heat)
    t = t / max(float(np.max(t)), 1e-6)
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    colors[..., 0] = np.clip(255 * np.minimum(1, 1.9 * t), 0, 255).astype(np.uint8)
    colors[..., 1] = np.clip(255 * np.maximum(0, 1.8 * t - 0.35), 0, 255).astype(np.uint8)
    colors[..., 2] = np.clip(255 * np.maximum(0, 2.8 * t - 1.5), 0, 255).astype(np.uint8)
    if background_rgb is None:
        img = colors
    else:
        background = np.asarray(background_rgb, dtype=np.uint8)
        if background.ndim == 2:
            background = np.repeat(background[..., None], 3, axis=2)
        if background.shape[:2] != (height, width):
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)
        background = np.ascontiguousarray(background[..., :3])
        alpha = np.clip(0.18 + 0.72 * t, 0.0, 0.90) * (t > 0.0)
        img = np.clip(
            background.astype(np.float32) * (1.0 - alpha[..., None])
            + colors.astype(np.float32) * alpha[..., None],
            0,
            255,
        ).astype(np.uint8)
    x, y, w, h = _clip_roi(roi_xywh)
    img[y:y + h, x:x + 2] = (0, 255, 255)
    img[y:y + h, x + w - 2:x + w] = (0, 255, 255)
    img[y:y + 2, x:x + w] = (0, 255, 255)
    img[y + h - 2:y + h, x:x + w] = (0, 255, 255)
    return img


def draw_motion_estimate_overlay(img, estimate):
    """Draw motion vector/ROI indicators on the dashboard heatmap."""
    if not estimate:
        return img
    est = estimate.to_dict() if hasattr(estimate, "to_dict") else dict(estimate)
    out = np.array(img, copy=True)
    x, y, w, h = _clip_roi(est.get("roi_xywh") or [0, 0, out.shape[1], out.shape[0]])
    center = (int(x + w / 2), int(y + h / 2))
    dx = est.get("dx_px")
    dy = est.get("dy_px")
    if dx is not None and dy is not None:
        end = (int(round(center[0] + float(dx) * 4.0)), int(round(center[1] + float(dy) * 4.0)))
        cv2.arrowedLine(out, center, end, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(out, end, 4, (255, 0, 255), -1, cv2.LINE_AA)
    rotation = est.get("rotation_delta_deg")
    if rotation is not None:
        radius = int(max(12, min(w, h) * 0.25))
        cv2.circle(out, center, radius, (0, 255, 255), 1, cv2.LINE_AA)
        label = f"{float(rotation):+.1f} deg"
        cv2.putText(out, label, (max(2, center[0] - radius), max(16, center[1] - radius - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (max(2, center[0] - radius), max(16, center[1] - radius - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    bbox = est.get("motion_bbox") or est.get("event_motion_bbox")
    if bbox and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
        except Exception:
            pass
    return out


def recent_events(events, window_sec=LIVE_HEATMAP_WINDOW_SEC, now=None):
    if events is None:
        arr = np.empty((0, 4), dtype=np.float64)
    else:
        arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    current = float(time.monotonic() if now is None else now)
    return arr[arr[:, 0] >= current - float(window_sec)]


def motion_hull(events, roi_xywh, frame_size=DAVIS_FRAME_SIZE):
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    frame_area = max(frame_w * frame_h, 1.0)
    arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] < 3:
        return {
            "motion_hull_points": [],
            "motion_hull_point_count": 0,
            "motion_hull_area_px2": 0.0,
            "motion_hull_area_fraction": 0.0,
            "motion_hull_border_touch_ratio": 0.0,
            "motion_hull_event_count": int(arr.shape[0]) if arr.ndim == 2 else 0,
        }
    points = arr[:, 1:3].astype(np.float32)
    hull = cv2.convexHull(points).reshape(-1, 2)
    area = float(cv2.contourArea(hull))
    x, y, w, h = _clip_roi(roi_xywh)
    roi_area = max(float(w * h), 1.0)
    if area > roi_area * 1.25:
        area = roi_area
    margin = MOTION_HULL_BORDER_MARGIN_PX
    touching = np.count_nonzero(
        (hull[:, 0] <= margin)
        | (hull[:, 1] <= margin)
        | (hull[:, 0] >= frame_w - 1.0 - margin)
        | (hull[:, 1] >= frame_h - 1.0 - margin)
    )
    return {
        "motion_hull_points": [[float(px), float(py)] for px, py in hull],
        "motion_hull_point_count": int(hull.shape[0]),
        "motion_hull_area_px2": round(area, 2),
        "motion_hull_area_fraction": round(area / frame_area, 4),
        "motion_hull_border_touch_ratio": round(float(touching) / max(int(hull.shape[0]), 1), 4),
        "motion_hull_event_count": int(arr.shape[0]),
    }


def local_to_global_event_density_ratio(all_events, roi_xywh, frame_size=DAVIS_FRAME_SIZE):
    """Ratio of in-ROI event density to outside-ROI event density.

    Returns None when the ROI effectively covers the frame (no 'outside' to
    compare against)."""
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    x, y, w, h = _clip_roi(roi_xywh, frame_size)
    roi_area = max(float(w * h), 1.0)
    frame_area = max(frame_w * frame_h, 1.0)
    outside_area = frame_area - roi_area
    if outside_area <= frame_area * 0.05:
        return None
    arr = np.asarray(all_events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return None
    xs, ys = arr[:, 1], arr[:, 2]
    inside = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
    inside_count = int(np.count_nonzero(inside))
    outside_count = int(arr.shape[0] - inside_count)
    inside_density = inside_count / roi_area
    outside_density = outside_count / outside_area
    if outside_density <= 0.0:
        return float("inf") if inside_count > 0 else None
    return float(inside_density / outside_density)


def object_motion_validity(
    hull,
    roi_source,
    event_count,
    local_global_ratio,
    raw_motion_class,
    motion_confidence,
    roi_confidence=None,
    baseline_compatible=True,
    allow_post_grasp_zone=False,
):
    """Camera-free gate deciding whether recent-event geometry may be presented
    as OBJECT motion (vs mere event activity)."""
    hull = hull or {}
    hull_reasons = []
    area_fraction = float(hull.get("motion_hull_area_fraction") or 0.0)
    border_ratio = float(hull.get("motion_hull_border_touch_ratio") or 0.0)
    disturbance = str(raw_motion_class or "").upper() in GLOBAL_DISTURBANCE_CLASSES
    if int(event_count or 0) < MOTION_HULL_MIN_EVENT_COUNT:
        hull_reasons.append("too_few_events")
    if area_fraction > MOTION_HULL_MAX_AREA_FRACTION:
        hull_reasons.append("hull_area_too_large")
    if border_ratio > MOTION_HULL_MAX_BORDER_TOUCH_RATIO:
        hull_reasons.append("border_touch_too_high")
    if disturbance:
        hull_reasons.append(f"disturbance_{str(raw_motion_class).lower()}")
    hull_valid = not hull_reasons

    object_reasons = list(hull_reasons)
    roi_object_specific = str(roi_source or "") in OBJECT_SPECIFIC_ROI_SOURCES
    try:
        projection_confidence = None if roi_confidence is None else float(roi_confidence)
    except (TypeError, ValueError):
        projection_confidence = float("nan")
    low_projection_confidence = bool(
        roi_confidence is not None
        and (
            not np.isfinite(projection_confidence)
            or projection_confidence < MIN_OBJECT_MOTION_CONFIDENCE
        )
    )
    if not roi_object_specific:
        object_reasons.append("full_frame_roi_only" if str(roi_source or "") == "full_frame" else f"roi_source_{roi_source or 'unknown'}_not_object_specific")
    if local_global_ratio is not None and float(local_global_ratio) < MOTION_HULL_MIN_LOCAL_TO_GLOBAL_RATIO:
        object_reasons.append("activity_not_local_to_roi")
    if motion_confidence is not None and float(motion_confidence) < MIN_OBJECT_MOTION_CONFIDENCE:
        object_reasons.append("low_motion_confidence")
    if roi_object_specific and low_projection_confidence:
        object_reasons.append("low_roi_projection_confidence")
    if baseline_compatible is not True:
        object_reasons.append("baseline_roi_context_not_ready")
    object_valid = not object_reasons
    # This fallback is explicitly grasp-zone motion, not object-specific
    # motion. It is enabled only after close and only when fixed-zone activity
    # is local, confident, and compared with a compatible quiet baseline.
    grasp_zone_reasons = list(hull_reasons)
    grasp_zone_candidate = bool(
        allow_post_grasp_zone
        and str(roi_source or "") in GRASP_ZONE_ROI_SOURCES
    )
    if not grasp_zone_candidate:
        grasp_zone_reasons.append("post_grasp_fixed_grasp_zone_not_active")
    if local_global_ratio is None:
        grasp_zone_reasons.append("local_to_global_ratio_unavailable")
    elif float(local_global_ratio) < MOTION_HULL_MIN_LOCAL_TO_GLOBAL_RATIO:
        grasp_zone_reasons.append("activity_not_local_to_grasp_zone")
    if motion_confidence is None or float(motion_confidence) < MIN_OBJECT_MOTION_CONFIDENCE:
        grasp_zone_reasons.append("low_motion_confidence")
    if baseline_compatible is not True:
        grasp_zone_reasons.append("baseline_roi_context_not_ready")
    grasp_zone_valid = not grasp_zone_reasons
    low_motion_confidence = bool(
        motion_confidence is not None
        and float(motion_confidence) < MIN_OBJECT_MOTION_CONFIDENCE
    )
    object_specific_confidence = (
        "low"
        if (
            not roi_object_specific
            or low_motion_confidence
            or low_projection_confidence
            or baseline_compatible is not True
        )
        else (
            "approximate_uncalibrated"
            if (
                str(roi_source or "") == "aps_target"
                or str(roi_source or "") in UNCALIBRATED_ROI_SOURCES
            )
            else "ok"
        )
    )

    label = "OBJECT_MOTION" if object_valid else ("GLOBAL_ACTIVITY" if (disturbance or "hull_area_too_large" in hull_reasons or "border_touch_too_high" in hull_reasons) else "EVENT_ACTIVITY")
    return {
        "motion_hull_valid": hull_valid,
        "motion_hull_reason": "ok" if hull_valid else ";".join(hull_reasons),
        "motion_hull_area_fraction": area_fraction,
        "motion_hull_border_touch_ratio": border_ratio,
        "motion_hull_roi_source": roi_source,
        "object_motion_valid": object_valid,
        "object_motion_reason": "ok" if object_valid else ";".join(dict.fromkeys(object_reasons)),
        "grasp_zone_motion_valid": grasp_zone_valid,
        "grasp_zone_motion_reason": "ok" if grasp_zone_valid else ";".join(dict.fromkeys(grasp_zone_reasons)),
        "motion_attribution": (
            "OBJECT_SPECIFIC"
            if object_valid
            else "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED"
            if grasp_zone_valid
            else "EVENT_ACTIVITY_ONLY"
        ),
        "local_to_global_event_ratio": None if local_global_ratio is None else (round(float(local_global_ratio), 3) if np.isfinite(local_global_ratio) else None),
        "roi_object_specific": roi_object_specific,
        "roi_projection_confidence": (
            None if projection_confidence is None or not np.isfinite(projection_confidence)
            else round(projection_confidence, 3)
        ),
        "baseline_roi_compatible": baseline_compatible is True,
        "roi_projection_uncalibrated": (
            str(roi_source or "") in {"configured_roi", "aps_target"}
            or str(roi_source or "") in UNCALIBRATED_ROI_SOURCES
        ),
        "object_specific_motion_confidence": object_specific_confidence,
        "motion_activity_label": label,
    }


def event_buffer_age_sec(events):
    if events is None:
        return 0.0
    try:
        arr = np.asarray(events, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return 0.0
        first = float(arr[0, 0])
        last = float(arr[-1, 0])
        return max(0.0, last - first)
    except Exception:
        return 0.0


def event_analysis_status_for_roi(primary_count, raw_count, last_poll_count, roi_source):
    """Describe available event evidence without upgrading diagnostic ROIs.

    An approximate or held RealSense ROI can make the movement display useful,
    but it is not calibrated object evidence.  Keep that distinction separate
    from the simpler question of whether events exist inside the selected ROI.
    """
    source = str(roi_source or "full_frame")
    if int(primary_count or 0) > 0:
        if source in OBJECT_SPECIFIC_ROI_SOURCES:
            return "OBJECT_ROI_EVENTS_AVAILABLE"
        if source != "full_frame":
            return "DIAGNOSTIC_ROI_EVENTS_AVAILABLE_NOT_OBJECT_ATTRIBUTED"
        return "GLOBAL_EVENTS_AVAILABLE_NO_OBJECT_ROI"
    if int(raw_count or 0) > 0:
        return "RAW_EVENTS_OUTSIDE_OBJECT_ROI"
    if int(last_poll_count or 0) > 0:
        return "RAW_EVENTS_OUTSIDE_ANALYSIS_WINDOW"
    return "NO_RECENT_RAW_EVENTS"


def diagnostic_movement_pattern(boundary_field):
    """Name the dominant raw neighborhood pattern without object attribution.

    The canonical object-motion gate remains authoritative. This helper only
    makes already-reported raw boundary cues readable when an uncalibrated or
    fixed close-grasp ROI prevents a bottle-specific classification.
    """
    field = dict(boundary_field or {})
    raw_count = int(field.get("raw_window_event_count") or 0)
    assigned_count = int(field.get("assigned_event_count") or 0)
    cue_values = {
        "VERTICAL_DRIFT_LIKE_ACTIVITY": float(field.get("raw_slip_cue") or 0.0),
        "ROTATION_LIKE_ACTIVITY": float(field.get("raw_rotation_cue") or 0.0),
        "VIBRATION_LIKE_ACTIVITY": float(field.get("raw_vibration_cue") or 0.0),
        "LOCALIZED_BURST_LIKE_ACTIVITY": float(field.get("raw_incipient_slip_precursor_score") or 0.0),
    }
    object_attributed = field.get("object_motion_valid") is True
    base = {
        "diagnostic_movement_pattern_scores": {
            key: round(float(np.clip(value, 0.0, 100.0)), 3)
            for key, value in cue_values.items()
        },
        "diagnostic_movement_pattern_object_attributed": object_attributed,
        "diagnostic_movement_pattern_physically_validated": False,
    }
    if raw_count <= 0:
        return {
            **base,
            "diagnostic_movement_pattern": "NO_RECENT_ACTIVITY",
            "diagnostic_movement_pattern_score": 0.0,
            "diagnostic_movement_pattern_status": "NOT_MEASURED_NO_EVENTS",
            "diagnostic_movement_pattern_reason": "No recent raw DAVIS events were available.",
        }
    if assigned_count <= 0:
        return {
            **base,
            "diagnostic_movement_pattern": "NO_LOCAL_NEIGHBORHOOD_SUPPORT",
            "diagnostic_movement_pattern_score": 0.0,
            "diagnostic_movement_pattern_status": "NOT_MEASURED_NO_ASSIGNED_BOUNDARY_EVENTS",
            "diagnostic_movement_pattern_reason": "Raw events exist, but none were assigned to boundary neighborhoods.",
        }
    pattern, score = max(cue_values.items(), key=lambda item: item[1])
    if score <= 0.0:
        pattern = "NO_DOMINANT_LOCAL_PATTERN"
    attribution = "ESTIMATED_OBJECT_LOCAL" if object_attributed else "DIAGNOSTIC_ONLY_NOT_OBJECT_ATTRIBUTED"
    suppression = str(field.get("suppression_reason") or "none")
    return {
        **base,
        "diagnostic_movement_pattern": pattern,
        "diagnostic_movement_pattern_score": round(float(np.clip(score, 0.0, 100.0)), 3),
        "diagnostic_movement_pattern_status": attribution,
        "diagnostic_movement_pattern_reason": (
            "Dominant raw spatio-temporal neighborhood cue; it does not bypass object-motion gates. "
            f"Suppression reason: {suppression}."
        ),
    }


def cap_event_packet_preserving_timestamps(events, max_events=LIVE_MAX_EVENTS_PER_TICK):
    """Copy/cap a normalized event packet without fabricating its time axis."""
    arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64), 0
    arr = np.array(arr[:, :4], dtype=np.float64, copy=True)
    dropped = max(0, int(arr.shape[0] - int(max_events)))
    if dropped:
        arr = arr[-int(max_events):]
    return arr, dropped


def aps_release_state(aps_status):
    mode = str((aps_status or {}).get("aps_mode") or "")
    if (aps_status or {}).get("aps_color_verified"):
        return "APS_COLOR_VERIFIED"
    if "Raw Bayer" in mode:
        return "APS_RAW_BAYER"
    if "Grayscale" in mode:
        return "APS_GRAYSCALE"
    if "Unavailable" in mode:
        return "APS_UNAVAILABLE"
    return "APS_UNKNOWN"


def davis_event_band_report_aliases(report):
    """Publish the requested event-band and close-up field names."""
    report = report or {}
    roi_source = str(report.get("roi_source") or "full_frame")
    object_specific = report.get("roi_object_specific") is True
    warning = report.get("roi_projection_warning")
    if not object_specific:
        warning = warning or report.get("object_motion_reason") or "ROI is not object-specific."
    return {
        "davis_event_band_active": bool(roi_source != "full_frame" and report.get("roi_xywh") is not None),
        "davis_event_band_source": roi_source,
        "davis_event_band_roi_xywh": report.get("roi_xywh"),
        "davis_event_band_object_specific": object_specific,
        "davis_event_band_confidence": report.get("roi_confidence"),
        "davis_event_band_warning": warning,
        "event_health": report.get("event_health_valid") is True,
        "vibration_rms_px": report.get("vibration_amplitude_px_rms"),
        # DAVIS supplies a 2D event plane. In-plane rotation is measurable, but
        # true bottle roll needs a calibrated 3D orientation source. Keep that
        # distinction explicit instead of aliasing rotation as roll.
        "roll_valid": False,
        "roll_delta_deg": None,
        "roll_status": "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
        "roll_exclusion_reason": "DAVIS 2D event ROI does not provide calibrated 3D bottle roll.",
        "davis_used_for_live_quality": report.get("davis_closeup_used_for_live_quality") is True,
        "davis_exclusion_reason": report.get("davis_closeup_exclusion_reason"),
    }


def post_grasp_recommendation(report):
    """Map the existing fail-closed decision to the manual pickup vocabulary."""
    report = report or {}
    if (
        report.get("roi_object_specific") is not True
        and report.get("grasp_zone_motion_valid") is not True
    ):
        return "NOT_OBJECT_SPECIFIC"
    if (
        report.get("object_motion_valid") is not True
        and report.get("grasp_zone_motion_valid") is not True
    ) or report.get("event_health_valid") is not True:
        return "HOLD"
    decision = str(report.get("decision") or report.get("visual_safety_response") or "").upper()
    return {
        "READY_TO_LIFT": "STABLE",
        "STABLE": "STABLE",
        "HOLD_AND_RECHECK": "HOLD",
        "WAIT_FOR_STABILITY": "HOLD",
        "REGRIP": "REGRIP",
        "DO_NOT_LIFT": "ABORT",
        "ABORT": "ABORT",
    }.get(decision, "HOLD")


def post_grasp_monitoring_payload(report, action_id=None, started_at=None):
    """Build the canonical latest manual-pickup report without a new backend."""
    payload = dict(report or {})
    payload.update(davis_event_band_report_aliases(payload))
    payload.update({
        "post_grasp_monitoring_active": True,
        "post_grasp_monitoring_action_id": action_id,
        "post_grasp_monitoring_started_at": started_at,
        "post_grasp_monitoring_saved_at": time.time(),
        "recommendation": post_grasp_recommendation(payload),
    })
    return payload


def post_grasp_timeline_entry(report, action_id=None, started_at=None):
    """Keep the post-grasp JSONL compact while retaining safety provenance."""
    payload = post_grasp_monitoring_payload(report, action_id, started_at)
    return {
        key: payload.get(key)
        for key in (
            "post_grasp_monitoring_saved_at",
            "post_grasp_monitoring_action_id",
            "post_grasp_monitoring_started_at",
            "current_stage",
            "closeup_action_elapsed_sec",
            "davis_closeup_window",
            "impact_window_metrics",
            "settle_window_metrics",
            "stability_window_metrics",
            "event_rate",
            "event_count",
            "event_health",
            "object_motion_valid",
            "dx_px",
            "dy_px",
            "displacement_px",
            "downward_slip_px",
            "rotation_delta_deg",
            "roll_valid",
            "roll_delta_deg",
            "roll_status",
            "roll_exclusion_reason",
            "vibration_rms_px",
            "motion_class",
            "disturbance_type",
            "stability_score",
            "davis_used_for_live_quality",
            "davis_exclusion_reason",
            "davis_event_band_source",
            "davis_event_band_confidence",
            "davis_target_lock_source_writer_session_id",
            "davis_target_lock_source_lock_frame_id",
            "davis_target_lock_source_attempt_id",
            "davis_target_lock_source_bottle_target_id",
            "recommendation",
        )
    }


class DavisMotionAnalyzer:
    def __init__(
        self,
        roi_xywh=None,
        stage="PRE_CLOSE",
        sensitivity="sensitive",
        analysis_roi_mode=None,
        latency_clock=None,
        recording_sink=None,
    ):
        self.analysis_roi_mode = str(analysis_roi_mode or "configured_or_mapped").strip().lower()
        self.fixed_center_third_mode = self.analysis_roi_mode == FIXED_CENTER_THIRD_MODE
        if self.fixed_center_third_mode:
            roi_xywh = configured_center_third_roi(DAVIS_FRAME_SIZE)
        self.roi_xywh = _clip_roi(roi_xywh)
        self.roi_source = "configured_roi" if roi_xywh else "full_frame"
        self.stage = stage
        self.monitor = EventPostGraspMonitor(enabled=True, roi_xywh=self.roi_xywh, sensitivity=sensitivity, recording_sink=recording_sink)
        self.report = self.monitor.start()
        self.start_time = time.monotonic()
        self.latency_tracker = SoftwareLatencyTracker(
            latency_clock,
            epoch_start=self.start_time if latency_clock is None else None,
        )
        self.baseline_metrics = None
        self.roi_context_revision = 0
        self.baseline_roi_context_revision = None
        self.history = deque(maxlen=500)
        self.motion_class_history = deque(maxlen=500)
        self.decision_history = deque(maxlen=500)
        self.slip_first_seen = None
        self.false_slip_warning_count = 0
        self.last_poll_event_count = 0
        self.drained_poll_count = 0
        self.latest_event_array = np.empty((0, 4), dtype=np.float64)
        # Rolling event buffer kept as one numpy array. The monitor's Python
        # list buffer costs ~7 ms per np.asarray() of 25k tuples, and update()
        # slices the buffer 7+ times per cycle; a single array keeps the whole
        # analysis under the display period.
        self.events_arr = np.empty((0, 4), dtype=np.float64)
        self.baseline_rejected_count = 0
        self.total_raw_events_polled = 0
        self.total_events_kept = 0
        self.dropped_events_due_tick_cap = 0
        self.dropped_events_due_rolling_cap = 0
        self.dropped_event_windows = 0
        self.update_times = []
        self.event_window_used_sec = PRIMARY_EVENT_WINDOW_SEC
        self.aps_frame_id = 0
        self.aps_times = []
        self.latest_aps_record = None
        self.latest_aps_display_rgb = None
        self.latest_aps_display_enhanced = None
        self.latest_aps_display_meta = {}
        self.latest_aps_stats = {}
        self.latest_aps_visibility = {}
        self.latest_aps_target_detection = None
        self.latest_aps_target_context = None
        self.latest_aps_target_failure_reason = "aps_target_not_evaluated"
        self.previous_aps_target_bbox = None
        self.confirmed_aps_target_bbox = None
        self.confirmed_aps_target_context = None
        self.pending_aps_target_bbox = None
        self.pending_aps_target_context = None
        self.pending_aps_target_count = 0
        self.aps_target_missed_detections = 0
        self.aps_target_confirmation_expired_reason = None
        self.post_grasp_aps_target_context = None
        self.post_grasp_aps_target_latch_until = None
        self.aps_exposure_info = {"aps_exposure_control_available": False}
        self.aps_config = dict(APS_VISIBILITY_DEFAULTS)
        self.baseline_duration_sec = DAVIS_BASELINE_DURATION_SEC
        self.baseline_max_quiet_rate = BASELINE_MAX_QUIET_RATE
        parsed = load_named_config("davis.yaml", default={})
        if isinstance(parsed, dict) and isinstance(parsed.get("aps"), dict):
            aps_cfg = parsed["aps"]
            for key in (
                "gamma",
                "clahe_enabled",
                "low_light_warn_p95_threshold",
                "low_contrast_warn_std_threshold",
                "exposure_auto",
                "exposure_us",
                "bayer_pattern",
                "channel_order",
                "glare_bright_threshold_u8",
                "glare_max_bright_fraction",
                "glare_max_interior_edge_density",
                "target_confirmation_frames",
                "target_switch_confirmation_frames",
                "target_same_iou_min",
                "target_pending_iou_min",
                "target_max_missed_detections",
                "target_confirmed_max_age_sec",
                "target_cadence_jitter_multiplier",
                "target_confirmed_age_cap_sec",
                "raw_adc_bit_depth",
            ):
                if key in aps_cfg and aps_cfg[key] is not None:
                    self.aps_config[key] = aps_cfg[key]
            # display_enhancement is the preview enhancement mode; the legacy
            # display_mode key (auto/gray/rgb) belongs to format classification
            enhancement = aps_cfg.get("display_enhancement")
            if enhancement is None and str(aps_cfg.get("display_mode") or "") in APS_DISPLAY_MODES:
                enhancement = aps_cfg.get("display_mode")
            if enhancement:
                self.aps_config["display_mode"] = str(enhancement)
        if isinstance(parsed, dict) and isinstance(parsed.get("events"), dict):
            configured_baseline = parsed["events"].get("baseline_duration_sec")
            if configured_baseline is not None:
                self.baseline_duration_sec = max(0.1, float(configured_baseline))
            configured_quiet_rate = parsed["events"].get("baseline_max_quiet_rate")
            if configured_quiet_rate is not None:
                self.baseline_max_quiet_rate = max(1.0, float(configured_quiet_rate))
        self.motion_estimator = DavisMotionEstimator(load_davis_motion_config(), frame_size=DAVIS_FRAME_SIZE)
        self.latest_motion_estimate = None
        self.boundary_points = boundary_from_roi(self.roi_xywh, DAVIS_FRAME_SIZE)
        self.boundary_source = (
            FIXED_CENTER_THIRD_BOUNDARY_SOURCE
            if self.fixed_center_third_mode
            else f"{self.roi_source}_boundary"
        )
        self.bottle_axis_vector_px = None
        self.bottle_axis_source = "unavailable"
        self.contact_band_polygon = None
        self.roi_projection = roi_projection_diagnostics(
            self.roi_source,
            {
                "mapping_mode": FIXED_CENTER_THIRD_MODE,
                "alignment_status": FIXED_CENTER_THIRD_BOUNDARY_SOURCE,
            } if self.fixed_center_third_mode else None,
        )
        # In fixed-center demo mode this remains separate from ``roi_xywh``:
        # roi_xywh is the constant acquisition/display view, while this
        # optional context is a confirmed APS contour used for object-local
        # validation only after complete RealSense lock provenance is bound.
        self.object_evidence_context = None
        self.object_evidence_roi_xywh = None
        self.object_evidence_roi_source = None
        self.object_evidence_boundary_points = None
        self.object_evidence_boundary_source = None
        self.object_evidence_contact_band_polygon = None
        self.object_evidence_projection = None
        self.object_evidence_invalid_reason = "aps_object_evidence_not_configured"
        self.close_action_origin = None
        self.close_action_id = None
        self.close_action_started_at = None
        self.closeup_action_accumulators = _empty_closeup_accumulators()
        self.closeup_action_metrics = closeup_metrics_from_accumulators(
            self.closeup_action_accumulators,
            self.roi_xywh,
        )
        self.closeup_action_snapshot_complete = False
        self.last_motion_timeline_write = 0.0
        self.last_post_grasp_timeline_write = 0.0

    def _refresh_aps_exposure_info(self):
        """Apply and read back APS exposure configuration via pyaer.

        Never raises: backends without exposure control simply report
        aps_exposure_control_available = False."""
        info = {
            "aps_exposure_control_available": False,
            "aps_exposure_configuration_applied": False,
            "aps_exposure_us": None,
            "aps_gain": None,
            "aps_gain_control_available": False,
            "aps_gain_status": "NOT_APPLICABLE_BACKEND_API_UNAVAILABLE",
            "aps_exposure_auto": None,
            "aps_exposure_source": None,
        }
        device = getattr(self.monitor, "device", None)
        if device is not None and getattr(self.monitor, "backend_name", "") == "pyaer":
            try:
                from pyaer import libcaer  # type: ignore

                mod = getattr(libcaer, "DAVIS_CONFIG_APS", None)
                exposure_key = getattr(libcaer, "DAVIS_CONFIG_APS_EXPOSURE", None)
                autoexp_key = getattr(libcaer, "DAVIS_CONFIG_APS_AUTOEXPOSURE", None)
                get_config = getattr(device, "get_config", None)
                set_config = getattr(device, "set_config", None)
                if mod is not None and exposure_key is not None and callable(get_config):
                    desired_auto = self.aps_config.get("exposure_auto")
                    desired_exposure = self.aps_config.get("exposure_us")
                    current_auto = bool(get_config(mod, autoexp_key)) if autoexp_key is not None else None
                    current_exposure = int(get_config(mod, exposure_key))
                    if callable(set_config):
                        if autoexp_key is not None and desired_auto is not None and current_auto != bool(desired_auto):
                            set_config(mod, autoexp_key, 1 if desired_auto else 0)
                        if desired_exposure is not None and desired_auto is False and current_exposure != int(desired_exposure):
                            set_config(mod, exposure_key, int(desired_exposure))
                    info["aps_exposure_us"] = int(get_config(mod, exposure_key))
                    if autoexp_key is not None:
                        info["aps_exposure_auto"] = bool(get_config(mod, autoexp_key))
                    auto_applied = desired_auto is None or info["aps_exposure_auto"] == bool(desired_auto)
                    exposure_applied = (
                        desired_exposure is None
                        or desired_auto is not False
                        or info["aps_exposure_us"] == int(desired_exposure)
                    )
                    info["aps_exposure_control_available"] = True
                    info["aps_exposure_configuration_applied"] = bool(auto_applied and exposure_applied)
                    info["aps_exposure_source"] = "pyaer_libcaer_config_readback"
            except Exception as exc:  # noqa: BLE001 - diagnostics must not break streaming
                info["aps_exposure_control_available"] = False
                info["aps_exposure_source"] = f"unavailable: {type(exc).__name__}"
        self.aps_exposure_info = info

    def set_aps_display_mode(self, mode):
        mode = str(mode or "").lower()
        if mode in APS_DISPLAY_MODES:
            self.aps_config["display_mode"] = mode
            return True
        return False

    def _aps_target_detector_cadence_sec(self):
        times = getattr(self, "aps_times", [])
        if len(times) < 2:
            return None
        span = float(times[-1]) - float(times[0])
        if not np.isfinite(span) or span <= 0.0:
            return None
        fps = (len(times) - 1) / span
        if not np.isfinite(fps) or fps <= 0.0:
            return None
        return float(self._aps_target_detection_interval_frames() / fps)

    def _aps_target_detection_interval_frames(self):
        try:
            configured = int(
                getattr(self, "aps_config", {}).get(
                    "target_detection_interval_frames",
                    DAVIS_APS_TARGET_DETECTION_INTERVAL_FRAMES,
                )
            )
        except (TypeError, ValueError):
            configured = DAVIS_APS_TARGET_DETECTION_INTERVAL_FRAMES
        return max(1, configured)

    def _aps_target_max_age_sec(self):
        nominal = float(
            getattr(self, "aps_config", {}).get(
                "target_confirmed_max_age_sec",
                DAVIS_APS_TARGET_MAX_AGE_SEC,
            )
        )
        multiplier = max(
            1.0,
            float(
                getattr(self, "aps_config", {}).get(
                    "target_cadence_jitter_multiplier",
                    1.5,
                )
            ),
        )
        cap = max(
            nominal,
            float(
                getattr(self, "aps_config", {}).get(
                    "target_confirmed_age_cap_sec",
                    3.0,
                )
            ),
        )
        cadence = self._aps_target_detector_cadence_sec()
        cadence_allowance = 0.0 if cadence is None else cadence * multiplier
        # Confirmation is refreshed only on detector cycles, not every APS
        # frame. Keep it alive through one measured cycle plus bounded jitter;
        # genuine stale gaps and configured misses still clear identity.
        return float(np.clip(max(nominal, cadence_allowance), 0.05, cap))

    def _post_grasp_aps_latch_active(self, now):
        context = getattr(self, "post_grasp_aps_target_context", None)
        latch_until = getattr(self, "post_grasp_aps_target_latch_until", None)
        return bool(context is not None and latch_until is not None and float(now) <= float(latch_until))

    def _reset_aps_target_confirmation(self, reason, *, now=None):
        """Forget stale identity so even the same-position target must reconfirm."""
        now = time.monotonic() if now is None else float(now)
        preserve_latch = self._post_grasp_aps_latch_active(now)
        self.confirmed_aps_target_bbox = None
        self.confirmed_aps_target_context = None
        self.previous_aps_target_bbox = None
        self.pending_aps_target_bbox = None
        self.pending_aps_target_context = None
        self.pending_aps_target_count = 0
        self.aps_target_confirmation_expired_reason = str(reason)
        self.latest_aps_target_failure_reason = str(reason)
        if not preserve_latch:
            self.latest_aps_target_context = None

    def _expire_stale_aps_target_confirmation(self, now):
        context = getattr(self, "confirmed_aps_target_context", None)
        if context is None:
            return False
        captured = context.get("captured_monotonic")
        try:
            age_sec = float(now) - float(captured)
        except (TypeError, ValueError):
            age_sec = float("inf")
        if not np.isfinite(age_sec) or age_sec < 0.0 or age_sec > self._aps_target_max_age_sec():
            self._reset_aps_target_confirmation("aps_target_confirmation_expired_age", now=now)
            return True
        return False

    def _record_aps_target_miss(self, reason, *, now):
        self.aps_target_missed_detections = int(
            getattr(self, "aps_target_missed_detections", 0)
        ) + 1
        maximum = max(
            1,
            int(getattr(self, "aps_config", {}).get("target_max_missed_detections", 2)),
        )
        if self.aps_target_missed_detections >= maximum:
            self._reset_aps_target_confirmation(
                f"aps_target_confirmation_expired_after_{self.aps_target_missed_detections}_misses",
                now=now,
            )
        else:
            self.latest_aps_target_failure_reason = str(reason)

    def _update_aps_target_detection(self):
        """Reuse the existing DAVIS APS bottle detector without another backend."""
        now = time.monotonic()
        self._expire_stale_aps_target_confirmation(now)
        if self.latest_aps_display_rgb is None:
            self._record_aps_target_miss("aps_display_unavailable", now=now)
            return
        try:
            from davis_grasp_planner import detect_bottle_from_aps

            gray = cv2.cvtColor(self.latest_aps_display_rgb, cv2.COLOR_RGB2GRAY)
            raw_aps = getattr(self.latest_aps_record, "image", None)
            raw_gray = None
            if raw_aps is not None:
                raw_array = np.asarray(raw_aps)
                if raw_array.ndim == 2:
                    raw_gray = raw_array
                elif raw_array.ndim == 3 and raw_array.shape[2] == 1:
                    raw_gray = raw_array[:, :, 0]
                elif raw_array.ndim == 3 and raw_array.shape[2] >= 3:
                    raw_gray = np.mean(raw_array[:, :, :3].astype(np.float64), axis=2)
            detection, _debug_images = detect_bottle_from_aps(
                gray,
                roi_xywh=None,
                aps_detection_input_mode=(
                    self.latest_aps_record.aps_mode
                    if self.latest_aps_record is not None
                    else "unknown"
                ),
                previous_bbox=self.previous_aps_target_bbox,
                detection_config=getattr(self, "aps_config", {}),
                raw_gray=raw_gray,
            )
            self.latest_aps_target_detection = detection
            host_timestamp = (
                None
                if self.latest_aps_record is None
                else self.latest_aps_record.host_timestamp
            )
            context = aps_target_context_from_detection(
                detection,
                frame_id=self.aps_frame_id,
                captured_monotonic=time.monotonic(),
                host_timestamp=host_timestamp,
            )
            if context is not None:
                context["aps_target_context_max_age_sec"] = self._aps_target_max_age_sec()
                context["aps_target_detector_cadence_sec"] = self._aps_target_detector_cadence_sec()
            now = time.monotonic()
            self._expire_stale_aps_target_confirmation(now)
            latch_until = getattr(self, "post_grasp_aps_target_latch_until", None)
            latched_context = getattr(self, "post_grasp_aps_target_context", None)
            if latched_context is not None and latch_until is not None and now > float(latch_until):
                # Do not let the normal context-age allowance silently extend
                # the configured post-grasp latch.
                if (self.latest_aps_target_context or {}).get("aps_post_grasp_latched"):
                    expired = dict(self.latest_aps_target_context)
                    expired["captured_monotonic"] = (
                        float(latch_until) - self._aps_target_max_age_sec() - 1e-3
                    )
                    expired["aps_post_grasp_latched"] = False
                    self.latest_aps_target_context = expired
                self.post_grasp_aps_target_context = None
                self.post_grasp_aps_target_latch_until = None
                latched_context = None
                latch_until = None
            if context is None:
                miss_reason = (
                    detection.get("detection_reason")
                    or detection.get("davis_detection_status")
                    or "aps_target_detection_not_current"
                )
                self._record_aps_target_miss(miss_reason, now=now)
            else:
                self.aps_target_missed_detections = 0
            if latched_context is not None and latch_until is not None and now <= float(latch_until):
                # Keep the confirmed object neighborhood frozen through the
                # manually declared motion test; never switch it to a
                # hand/glare contour while measuring relative bottle motion.
                latched_context = dict(latched_context)
                latched_context.update({
                    "captured_monotonic": now,
                    "aps_host_timestamp": host_timestamp,
                    "source_frame_id": int(self.aps_frame_id),
                    "aps_post_grasp_latched": True,
                    "aps_post_grasp_latch_remaining_sec": round(float(latch_until) - now, 3),
                })
                self.post_grasp_aps_target_context = latched_context
                self.latest_aps_target_context = latched_context
                self.latest_aps_target_failure_reason = None
                return
            if context is None:
                return

            bbox = list(detection.get("davis_bbox_xyxy"))
            confirmed_bbox = getattr(self, "confirmed_aps_target_bbox", None)
            aps_config = getattr(self, "aps_config", {})
            same_iou_min = float(aps_config.get("target_same_iou_min", 0.20))
            pending_iou_min = float(aps_config.get("target_pending_iou_min", 0.50))
            if confirmed_bbox is not None and _bbox_iou_xyxy(bbox, confirmed_bbox) >= same_iou_min:
                self.confirmed_aps_target_bbox = bbox
                self.confirmed_aps_target_context = context
                self.latest_aps_target_context = context
                self.previous_aps_target_bbox = bbox
                self.pending_aps_target_bbox = None
                self.pending_aps_target_context = None
                self.pending_aps_target_count = 0
                self.aps_target_missed_detections = 0
                self.aps_target_confirmation_expired_reason = None
                self.latest_aps_target_failure_reason = None
                return

            pending_bbox = getattr(self, "pending_aps_target_bbox", None)
            if pending_bbox is not None and _bbox_iou_xyxy(bbox, pending_bbox) >= pending_iou_min:
                self.pending_aps_target_count = int(getattr(self, "pending_aps_target_count", 0)) + 1
            else:
                self.pending_aps_target_count = 1
            self.pending_aps_target_bbox = bbox
            self.pending_aps_target_context = context
            frames_key = (
                "target_switch_confirmation_frames"
                if confirmed_bbox is not None
                else "target_confirmation_frames"
            )
            required = max(1, int(aps_config.get(frames_key, 3)))
            if self.pending_aps_target_count >= required:
                self.confirmed_aps_target_bbox = bbox
                self.confirmed_aps_target_context = context
                self.latest_aps_target_context = context
                self.previous_aps_target_bbox = bbox
                self.pending_aps_target_bbox = None
                self.pending_aps_target_context = None
                self.pending_aps_target_count = 0
                self.aps_target_missed_detections = 0
                self.aps_target_confirmation_expired_reason = None
                self.latest_aps_target_failure_reason = None
            else:
                action = "switch" if confirmed_bbox is not None else "initial"
                self.latest_aps_target_failure_reason = (
                    f"aps_target_{action}_pending_confirmation_"
                    f"{self.pending_aps_target_count}_of_{required}"
                )
        except Exception as exc:  # noqa: BLE001 - APS target is an optional safe fallback
            self.latest_aps_target_detection = None
            self._record_aps_target_miss(
                f"aps_target_detector_error:{type(exc).__name__}",
                now=time.monotonic(),
            )

    def set_roi(self, roi_xywh, source="realsense_bbox_mapped", projection_metadata=None):
        if self.fixed_center_third_mode:
            roi_xywh = configured_center_third_roi(DAVIS_FRAME_SIZE)
            source = "configured_roi"
            projection_metadata = {
                "mapping_mode": FIXED_CENTER_THIRD_MODE,
                "alignment_status": FIXED_CENTER_THIRD_BOUNDARY_SOURCE,
            }
        clipped = _clip_roi(roi_xywh)
        source = str(source or self.roi_source or "full_frame")
        projection = roi_projection_diagnostics(source, projection_metadata)
        material_change = roi_context_change_is_material(
            self.roi_xywh,
            clipped,
            self.roi_source,
            source,
            self.roi_projection,
            projection,
        )
        if clipped != self.roi_xywh:
            self.roi_xywh = clipped
            self.roi_source = source
            self.boundary_points = boundary_from_roi(clipped, DAVIS_FRAME_SIZE)
            self.boundary_source = (
                FIXED_CENTER_THIRD_BOUNDARY_SOURCE
                if self.fixed_center_third_mode
                else f"{source}_boundary"
            )
            self.bottle_axis_vector_px = None
            self.bottle_axis_source = "unavailable"
            self.monitor.roi_xywh = clipped
            self.monitor.configured_roi = clipped
        elif source and self.roi_source != source:
            self.roi_source = source
            self.boundary_source = (
                FIXED_CENTER_THIRD_BOUNDARY_SOURCE
                if self.fixed_center_third_mode
                else f"{source}_boundary"
            )
        self.roi_projection = projection
        if material_change:
            # Motion baselines and cumulative estimates have meaning only in
            # the ROI/projection context in which they were measured.
            self.roi_context_revision += 1
            self._reset_recent_motion(reset_baseline=True)

    def set_boundary(self, boundary, source="realsense_contour_mapped"):
        """Accept a pre-mapped contour without opening or owning either camera."""
        if self.fixed_center_third_mode:
            self.boundary_points = boundary_from_roi(
                configured_center_third_roi(DAVIS_FRAME_SIZE),
                DAVIS_FRAME_SIZE,
            )
            self.boundary_source = FIXED_CENTER_THIRD_BOUNDARY_SOURCE
            self.bottle_axis_vector_px = None
            self.bottle_axis_source = "unavailable"
            return
        points = np.asarray(boundary, dtype=float).reshape(-1, 2)
        if points.shape[0] < 3:
            raise ValueError("boundary requires at least three points")
        self.boundary_points = points
        self.boundary_source = str(source)
        centered = points - np.mean(points, axis=0, keepdims=True)
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        norm = float(np.linalg.norm(axis))
        if np.isfinite(axis).all() and norm > 1e-9:
            axis = axis / norm
            if axis[1] < 0.0:
                axis = -axis
            self.bottle_axis_vector_px = [float(axis[0]), float(axis[1])]
            self.bottle_axis_source = str(source)
        else:
            self.bottle_axis_vector_px = None
            self.bottle_axis_source = "unavailable"

    def set_contact_band(self, polygon):
        if self.fixed_center_third_mode:
            self.contact_band_polygon = None
            return
        if polygon is None:
            self.contact_band_polygon = None
            return
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        self.contact_band_polygon = points if points.shape[0] >= 3 else None

    def set_object_evidence_context(self, context, *, now=None):
        """Set confirmed APS-local evidence without moving the acquisition ROI."""
        if not self.fixed_center_third_mode:
            self.object_evidence_invalid_reason = "separate_object_evidence_requires_fixed_center_mode"
            return False
        prepared, reason = aps_object_evidence_within_acquisition(
            context,
            self.roi_xywh,
            now=now,
        )
        if prepared is None:
            self.clear_object_evidence_context(reason)
            return False
        projection = roi_projection_diagnostics(
            "aps_target",
            prepared.get("projection_metadata"),
        )
        prior = self.object_evidence_context or {}
        identity_keys = (
            "source_writer_session_id",
            "source_lock_frame_id",
            "source_attempt_id",
            "source_bottle_target_id",
        )
        identity_changed = any(
            str(prior.get(key)) != str(prepared.get(key))
            for key in identity_keys
        ) if prior else True
        material_change = identity_changed or roi_context_change_is_material(
            self.object_evidence_roi_xywh,
            prepared.get("roi_xywh"),
            self.object_evidence_roi_source,
            "aps_target",
            self.object_evidence_projection,
            projection,
        )
        self.object_evidence_context = prepared
        self.object_evidence_roi_xywh = list(prepared["roi_xywh"])
        self.object_evidence_roi_source = "aps_target"
        self.object_evidence_boundary_points = np.asarray(
            prepared["boundary_points"],
            dtype=float,
        ).reshape(-1, 2)
        self.object_evidence_boundary_source = str(prepared["boundary_source"])
        contact = prepared.get("contact_band_points")
        self.object_evidence_contact_band_polygon = (
            None
            if contact is None
            else np.asarray(contact, dtype=float).reshape(-1, 2)
        )
        self.object_evidence_projection = projection
        self.object_evidence_invalid_reason = None
        if material_change:
            self.roi_context_revision += 1
            self._reset_recent_motion(reset_baseline=True)
        return True

    def clear_object_evidence_context(self, reason="aps_object_evidence_unavailable"):
        """Fail closed and require a new quiet baseline after evidence loss."""
        changed = self.object_evidence_context is not None
        self.object_evidence_context = None
        self.object_evidence_roi_xywh = None
        self.object_evidence_roi_source = None
        self.object_evidence_boundary_points = None
        self.object_evidence_boundary_source = None
        self.object_evidence_contact_band_polygon = None
        self.object_evidence_projection = None
        self.object_evidence_invalid_reason = str(reason)
        if changed:
            self.roi_context_revision += 1
            self._reset_recent_motion(reset_baseline=True)
        return changed

    def _expire_object_evidence_if_stale(self, *, now=None):
        context = self.object_evidence_context
        if context is None:
            return False
        current = time.monotonic() if now is None else float(now)
        try:
            age = current - float(context.get("captured_monotonic"))
            max_age = max(
                0.05,
                float(
                    context.get("aps_target_context_max_age_sec")
                    or DAVIS_APS_TARGET_MAX_AGE_SEC
                ),
            )
        except (TypeError, ValueError):
            age = float("inf")
            max_age = DAVIS_APS_TARGET_MAX_AGE_SEC
        if not np.isfinite(age) or age < 0.0 or age > max_age:
            self.clear_object_evidence_context("aps_object_evidence_stale")
            return True
        return False

    def _active_analysis_context(self):
        if self.object_evidence_context is not None:
            return {
                "roi_xywh": self.object_evidence_roi_xywh,
                "roi_source": self.object_evidence_roi_source,
                "boundary_points": self.object_evidence_boundary_points,
                "boundary_source": self.object_evidence_boundary_source,
                "contact_band_polygon": self.object_evidence_contact_band_polygon,
                "roi_projection": self.object_evidence_projection,
                "bottle_axis_vector_px": self.object_evidence_context.get("bottle_axis_vector_px"),
                "bottle_axis_source": self.object_evidence_boundary_source,
            }
        return {
            "roi_xywh": self.roi_xywh,
            "roi_source": self.roi_source,
            "boundary_points": self.boundary_points,
            "boundary_source": self.boundary_source,
            "contact_band_polygon": self.contact_band_polygon,
            "roi_projection": self.roi_projection,
            "bottle_axis_vector_px": self.bottle_axis_vector_px,
            "bottle_axis_source": self.bottle_axis_source,
        }

    def _reset_recent_motion(self, *, reset_baseline=False):
        self.events_arr = np.empty((0, 4), dtype=np.float64)
        self.latest_event_array = np.empty((0, 4), dtype=np.float64)
        self.motion_estimator.reset()
        self.slip_first_seen = None
        if reset_baseline:
            self.baseline_metrics = None
            self.baseline_roi_context_revision = None
            self.motion_estimator.baseline = self.motion_estimator.empty_baseline()
            self.start_time = time.monotonic()
        self.latency_tracker.reset_epoch(
            reset_origin=reset_baseline,
        )

    def set_stage(self, stage):
        stage = str(stage or "PRE_CLOSE").upper()
        if stage not in {"PRE_CLOSE", "CLOSE_IMPACT", "POST_CLOSE_STABILITY", "LIFT_CHECK"}:
            stage = "PRE_CLOSE"
        previous = self.stage
        if previous == stage:
            return False
        self.stage = stage
        if (previous == "PRE_CLOSE") != (stage == "PRE_CLOSE"):
            # Do not let approach events/cumulative motion leak across the
            # close/contact boundary in either direction.
            self._reset_recent_motion(reset_baseline=False)
            if stage == "PRE_CLOSE":
                self.close_action_origin = None
                self.close_action_id = None
                self.close_action_started_at = None
                self.post_grasp_aps_target_context = None
                self.post_grasp_aps_target_latch_until = None
            else:
                self.close_action_origin = time.monotonic()
                self.close_action_id = f"davis-post-grasp-{time.time_ns()}"
                self.close_action_started_at = time.time()
                confirmed_context = getattr(self, "confirmed_aps_target_context", None)
                confirmed_at = (
                    None
                    if confirmed_context is None
                    else confirmed_context.get("captured_monotonic")
                )
                confirmed_current = bool(
                    confirmed_at is not None
                    and 0.0 <= self.close_action_origin - float(confirmed_at) <= self._aps_target_max_age_sec()
                )
                if confirmed_current:
                    latch_duration_sec = max(
                        STABILITY_WINDOW_SEC,
                        float(
                            getattr(self, "aps_config", {}).get(
                                "post_grasp_target_latch_sec",
                                120.0,
                            )
                        ),
                    )
                    self.post_grasp_aps_target_context = dict(confirmed_context)
                    self.post_grasp_aps_target_context.update({
                        "captured_monotonic": self.close_action_origin,
                        "aps_post_grasp_latched": True,
                        "aps_post_grasp_latch_duration_sec": latch_duration_sec,
                    })
                    self.post_grasp_aps_target_latch_until = (
                        self.close_action_origin + latch_duration_sec
                    )
                    self.latest_aps_target_context = self.post_grasp_aps_target_context
            self.closeup_action_accumulators = _empty_closeup_accumulators()
            self.closeup_action_metrics = closeup_metrics_from_accumulators(
                self.closeup_action_accumulators,
                self.roi_xywh,
            )
            self.closeup_action_snapshot_complete = False
        return True

    def clear_object_context(self):
        """Fail closed when the current RealSense target/ROI disappears."""
        if self.fixed_center_third_mode:
            evidence_changed = self.clear_object_evidence_context(
                "object_context_cleared",
            )
            fixed_roi = configured_center_third_roi(DAVIS_FRAME_SIZE)
            changed = self.roi_source != "configured_roi" or self.roi_xywh != fixed_roi
            self.roi_xywh = fixed_roi
            self.roi_source = "configured_roi"
            self.boundary_points = boundary_from_roi(fixed_roi, DAVIS_FRAME_SIZE)
            self.boundary_source = FIXED_CENTER_THIRD_BOUNDARY_SOURCE
            self.bottle_axis_vector_px = None
            self.bottle_axis_source = "unavailable"
            self.contact_band_polygon = None
            self.roi_projection = roi_projection_diagnostics(
                "configured_roi",
                {
                    "mapping_mode": FIXED_CENTER_THIRD_MODE,
                    "alignment_status": FIXED_CENTER_THIRD_BOUNDARY_SOURCE,
                },
            )
            self.monitor.roi_xywh = fixed_roi
            self.monitor.configured_roi = fixed_roi
            if changed:
                self.roi_context_revision += 1
                self._reset_recent_motion(reset_baseline=True)
            return changed or evidence_changed
        full_roi = _clip_roi(None)
        changed = self.roi_source != "full_frame" or self.roi_xywh != full_roi
        self.roi_xywh = full_roi
        self.roi_source = "full_frame"
        self.boundary_points = boundary_from_roi(full_roi, DAVIS_FRAME_SIZE)
        self.boundary_source = "full_frame_boundary"
        self.bottle_axis_vector_px = None
        self.bottle_axis_source = "unavailable"
        self.contact_band_polygon = None
        self.roi_projection = roi_projection_diagnostics("full_frame")
        self.monitor.roi_xywh = full_roi
        self.monitor.configured_roi = full_roi
        if changed:
            self.roi_context_revision += 1
            self._reset_recent_motion(reset_baseline=True)
        return changed

    def update(self):
        update_start = time.perf_counter()
        self.latency_tracker.begin_update()
        if not self.monitor.active:
            report = {"motion_class": "SENSOR_UNAVAILABLE", "decision": "UNAVAILABLE", "visual_safety_response": "UNAVAILABLE"}
            self.report = {**self.report, **report}
            self.report["software_timing"] = self.latency_tracker.report_payload_ready()
            return self.report
        self._expire_object_evidence_if_stale()
        analysis_context = self._active_analysis_context()
        analysis_roi_xywh = analysis_context["roi_xywh"]
        analysis_roi_source = analysis_context["roi_source"]
        analysis_boundary_points = analysis_context["boundary_points"]
        analysis_boundary_source = analysis_context["boundary_source"]
        analysis_contact_band_polygon = analysis_context["contact_band_polygon"]
        analysis_roi_projection = analysis_context["roi_projection"]
        analysis_axis_vector_px = analysis_context["bottle_axis_vector_px"]
        analysis_axis_source = analysis_context["bottle_axis_source"]
        packets = []
        latest_aps = None
        drain_start = time.monotonic()
        for _ in range(LIVE_BACKLOG_DRAIN_MAX_POLLS):
            packet, aps_frames = self.monitor.poll_event_and_frame_arrays()
            frame = latest_aps_array(aps_frames)
            if frame is not None:
                latest_aps = frame
            if getattr(packet, "ndim", 0) == 2 and packet.shape[0]:
                packets.append(packet)
                self.latency_tracker.observe_event_receipt(
                    getattr(self.monitor, "latest_event_host_arrival_monotonic", None),
                )
            else:
                break
            if time.monotonic() - drain_start >= LIVE_BACKLOG_DRAIN_BUDGET_SEC:
                break
        if latest_aps is not None:
            self.aps_frame_id += 1
            self.latest_aps_record = normalize_backend_aps_frame(
                latest_aps,
                backend=self.monitor.backend_name,
                frame_id=self.aps_frame_id,
                device_timestamp_us=getattr(self.monitor, "latest_aps_device_timestamp_us", None),
                config=self.aps_config,
            )
            self.latest_aps_display_rgb = aps_display_image(
                latest_aps,
                mode=self.latest_aps_record.aps_mode,
                bayer_pattern=self.latest_aps_record.aps_bayer_pattern,
                channel_order=self.latest_aps_record.aps_channel_order,
            )
            # raw stats + visibility at low rate (every 5th frame); display
            # enhancement is applied to the preview only and never feeds stats
            if self.aps_frame_id % 5 == 1 or not self.latest_aps_stats:
                self.latest_aps_stats = compute_aps_stats(latest_aps)
                self.latest_aps_visibility = classify_aps_visibility(self.latest_aps_stats, self.aps_config)
                self._refresh_aps_exposure_info()
            self.latest_aps_display_enhanced, self.latest_aps_display_meta = enhance_aps_display(
                self.latest_aps_display_rgb,
                mode=self.aps_config.get("display_mode", "auto_normalize"),
                gamma=self.aps_config.get("gamma", 0.7),
                clahe_enabled=bool(self.aps_config.get("clahe_enabled", False)),
            )
            if (self.aps_frame_id - 1) % self._aps_target_detection_interval_frames() == 0:
                self._update_aps_target_detection()
            self.aps_times.append(time.monotonic())
            self.aps_times = self.aps_times[-120:]
        drain_ms = (time.monotonic() - drain_start) * 1000.0
        self.drained_poll_count = len(packets)
        event_array = np.vstack(packets) if packets else np.empty((0, 4), dtype=np.float64)
        raw_event_count = int(event_array.shape[0]) if getattr(event_array, "ndim", 0) == 2 else 0
        self.last_poll_event_count = raw_event_count
        self.total_raw_events_polled += raw_event_count
        if raw_event_count and not self.closeup_action_snapshot_complete:
            accumulate_closeup_action_events(
                self.closeup_action_accumulators,
                event_array,
                self.close_action_origin,
                analysis_roi_xywh,
            )
        if self.last_poll_event_count:
            event_array, dropped = cap_event_packet_preserving_timestamps(event_array)
            if dropped:
                self.dropped_events_due_tick_cap += dropped
                self.dropped_event_windows += 1
            self.total_events_kept += int(event_array.shape[0])
            self.latest_event_array = event_array
        else:
            self.latest_event_array = np.empty((0, 4), dtype=np.float64)
        if event_array.shape[0]:
            self.events_arr = np.vstack([self.events_arr, event_array]) if self.events_arr.shape[0] else event_array
        # Polling can take several milliseconds. Analyze against a timestamp
        # captured after the packet was host-anchored so newly arrived events
        # are not incorrectly excluded as future-dated.
        now = time.monotonic()
        prune = None if self.baseline_metrics is None else now - 2.0
        if prune is not None and self.events_arr.shape[0]:
            self.events_arr = self.events_arr[self.events_arr[:, 0] >= prune]
        if self.events_arr.shape[0] > LIVE_ROLLING_MAX_EVENTS:
            dropped = int(self.events_arr.shape[0] - LIVE_ROLLING_MAX_EVENTS)
            self.dropped_events_due_rolling_cap += max(0, dropped)
            self.events_arr = self.events_arr[-LIVE_ROLLING_MAX_EVENTS:]
        if self.baseline_metrics is None and now - self.start_time >= self.baseline_duration_sec:
            base_events = _events_since(self.events_arr, now, self.baseline_duration_sec, analysis_roi_xywh)
            candidate_baseline = _metrics(base_events, self.baseline_duration_sec, analysis_roi_xywh)
            if candidate_baseline["event_rate"] <= self.baseline_max_quiet_rate:
                self.baseline_metrics = candidate_baseline
                self.baseline_roi_context_revision = self.roi_context_revision
                self.motion_estimator.calibrate_baseline(base_events, now=now, roi_xywh=analysis_roi_xywh, frame_size=DAVIS_FRAME_SIZE)
            else:
                self.baseline_rejected_count += 1
                self.start_time = now
        baseline = self.baseline_metrics or _empty_metrics()
        micro = _metrics(_events_since(self.events_arr, now, MICRO_WINDOW_SEC, analysis_roi_xywh), MICRO_WINDOW_SEC, analysis_roi_xywh)
        short = _metrics(_events_since(self.events_arr, now, SHORT_WINDOW_SEC, analysis_roi_xywh), SHORT_WINDOW_SEC, analysis_roi_xywh, micro)
        primary_events, event_window_sec, event_window_mode, analysis_end = select_event_analysis_window(
            self.events_arr,
            now,
            analysis_roi_xywh,
        )
        self.event_window_used_sec = event_window_sec
        event_window_metrics = _metrics(primary_events, event_window_sec, analysis_roi_xywh, short)
        hull = motion_hull(primary_events, analysis_roi_xywh)
        window_events_unfiltered = _events_since(self.events_arr, now, event_window_sec, None)
        raw_window_event_count = int(window_events_unfiltered.shape[0])
        latest_event_timestamp = (
            float(self.events_arr[-1, 0])
            if self.events_arr.ndim == 2 and self.events_arr.shape[0]
            else None
        )
        event_timestamp_lag_ms = (
            None
            if latest_event_timestamp is None
            else round((now - latest_event_timestamp) * 1000.0, 2)
        )
        if latest_event_timestamp is None:
            event_timestamp_status = "EMPTY"
        elif event_timestamp_lag_ms < -20.0:
            event_timestamp_status = "FUTURE"
        elif event_timestamp_lag_ms <= event_window_sec * 1000.0:
            event_timestamp_status = "CURRENT"
        elif event_timestamp_lag_ms <= 500.0:
            event_timestamp_status = "AGING"
        else:
            event_timestamp_status = "STALE"
        event_host_timestamp = (
            None
            if latest_event_timestamp is None
            else time.time() - max(0.0, float(event_timestamp_lag_ms) / 1000.0)
        )
        roi_object_specific_for_diagnostics = str(analysis_roi_source) in OBJECT_SPECIFIC_ROI_SOURCES
        object_roi_window_event_count = (
            int(primary_events.shape[0])
            if roi_object_specific_for_diagnostics
            else None
        )
        event_analysis_status = event_analysis_status_for_roi(
            primary_events.shape[0],
            raw_window_event_count,
            self.last_poll_event_count,
            analysis_roi_source,
        )
        local_global_ratio = local_to_global_event_density_ratio(window_events_unfiltered, analysis_roi_xywh)
        impact = _metrics(_events_since(self.events_arr, now, IMPACT_WINDOW_SEC, analysis_roi_xywh), IMPACT_WINDOW_SEC, analysis_roi_xywh, short)
        settle = _metrics(_events_since(self.events_arr, now, SETTLE_WINDOW_SEC, analysis_roi_xywh), SETTLE_WINDOW_SEC, analysis_roi_xywh, impact)
        stability = _metrics(_events_since(self.events_arr, now, STABILITY_WINDOW_SEC, analysis_roi_xywh), STABILITY_WINDOW_SEC, analysis_roi_xywh, settle)
        scores = classify_motion(micro, short, impact, settle, stability, baseline, analysis_roi_source)
        motion_estimate = self.motion_estimator.estimate(
            self.events_arr,
            now=now,
            # Match the exact wall-clock end used by _events_since() above;
            # timestamp freshness remains an independent fail-closed gate.
            analysis_end=analysis_end,
            motion_window_ms=event_window_sec * 1000.0,
            roi_xywh=analysis_roi_xywh,
            roi_source=analysis_roi_source,
            frame_size=DAVIS_FRAME_SIZE,
            bottle_axis_vector_px=analysis_axis_vector_px,
            bottle_axis_source=analysis_axis_source,
        )
        self.latest_motion_estimate = motion_estimate
        estimate_dict = motion_estimate.to_dict()
        closeup_elapsed_sec = (
            None
            if self.close_action_origin is None
            else max(0.0, now - self.close_action_origin)
        )
        closeup_window = closeup_window_for_elapsed(closeup_elapsed_sec)
        if self.close_action_origin is not None and not self.closeup_action_snapshot_complete:
            self.closeup_action_metrics = closeup_metrics_from_accumulators(
                self.closeup_action_accumulators,
                analysis_roi_xywh,
            )
            if closeup_elapsed_sec >= STABILITY_WINDOW_SEC:
                self.closeup_action_snapshot_complete = True
        action_window_metrics = self.closeup_action_metrics
        quantitative_motion_class = estimate_dict.get("motion_class")
        quantitative_scores = {
            "incipient_slip_score": estimate_dict.get("incipient_slip_score") or 0.0,
            "gross_slip_score": estimate_dict.get("gross_slip_score") or 0.0,
            "rotation_risk_score": estimate_dict.get("rotation_score") or 0.0,
            "vibration_risk_score": estimate_dict.get("vibration_score") or 0.0,
            "impact_score": estimate_dict.get("impact_score") or 0.0,
            "stability_score": estimate_dict.get("stability_score") if estimate_dict.get("stability_score") is not None else 100.0,
        }
        if estimate_dict.get("estimate_valid") or quantitative_motion_class in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"}:
            for key in ("incipient_slip_score", "gross_slip_score", "rotation_risk_score", "vibration_risk_score", "impact_score"):
                scores[key] = float(max(float(scores.get(key) or 0.0), float(quantitative_scores[key] or 0.0)))
            scores["stability_score"] = float(min(float(scores.get("stability_score") or 100.0), float(quantitative_scores["stability_score"] or 100.0)))
            if quantitative_motion_class in {"CAMERA_SHAKE", "LIGHTING_CHANGE", "BACKGROUND_MOTION"}:
                scores["motion_class"] = quantitative_motion_class
                scores["decision"] = "HOLD_AND_RECHECK"
                scores["visual_safety_response"] = "HOLD_AND_RECHECK"
            elif quantitative_motion_class in {"GROSS_SLIP", "SPIN_OR_ROTATION", "SHAKE_OR_VIBRATION", "INCIPIENT_SLIP"}:
                class_priority = {"INCIPIENT_SLIP": 1, "SHAKE_OR_VIBRATION": 2, "SPIN_OR_ROTATION": 3, "GROSS_SLIP": 4}
                if class_priority.get(quantitative_motion_class, 0) >= class_priority.get(scores.get("motion_class"), 0):
                    scores["motion_class"] = quantitative_motion_class
                    scores["decision"] = {
                        "INCIPIENT_SLIP": "HOLD_AND_RECHECK",
                        "SHAKE_OR_VIBRATION": "HOLD_AND_RECHECK",
                        "SPIN_OR_ROTATION": "REGRIP",
                        "GROSS_SLIP": "DO_NOT_LIFT",
                    }.get(quantitative_motion_class, scores.get("decision"))
                    scores["visual_safety_response"] = scores["decision"]
        scores["quantitative_motion_reason"] = estimate_dict.get("reason")
        scores["quantitative_evidence"] = estimate_dict.get("quantitative_evidence")
        if self.stage == "PRE_CLOSE" and scores["motion_class"] not in {"STABLE", "SENSOR_UNAVAILABLE"}:
            scores["stage_motion_label"] = "pre-close motion instability"
            scores["visual_safety_response"] = "WAIT_FOR_STABILITY"
            scores["decision"] = "HOLD_AND_RECHECK"
            scores["stage_decision_reason"] = "Before close, DAVIS motion means unstable view/object motion, not slip."
        else:
            scores["stage_motion_label"] = scores["motion_class"]
            scores["stage_decision_reason"] = scores["response_reason"]
        raw_motion_class = scores["motion_class"]
        davis_safety_class = simplified_motion_class(raw_motion_class)
        scores["raw_motion_class"] = raw_motion_class
        scores["davis_safety_class"] = davis_safety_class
        scores["motion_class"] = davis_safety_class

        # object-motion visualization/consumption gates: distinguish "the event
        # stream is alive" from "this is object-local motion usable for grasp"
        motion_confidence = estimate_dict.get("motion_estimation_confidence")
        baseline_roi_compatible = bool(
            self.baseline_metrics is not None
            and self.baseline_roi_context_revision == self.roi_context_revision
        )
        gates = object_motion_validity(
            hull,
            analysis_roi_source,
            hull.get("motion_hull_event_count") or event_window_metrics.get("event_count") or 0,
            local_global_ratio,
            raw_motion_class,
            motion_confidence,
            analysis_roi_projection.get("roi_confidence"),
            baseline_roi_compatible,
            allow_post_grasp_zone=(
                self.stage != "PRE_CLOSE"
                and analysis_roi_source == "configured_roi"
            ),
        )
        motion_evidence_valid = bool(
            gates["object_motion_valid"]
            or gates.get("grasp_zone_motion_valid")
        )
        disturbance = str(raw_motion_class or "").upper() in GLOBAL_DISTURBANCE_CLASSES
        gates["global_activity_score"] = float(
            max(
                float(estimate_dict.get("background_motion_score") or 0.0),
                float(estimate_dict.get("lighting_flicker_score") or 0.0),
                float(estimate_dict.get("camera_shake_score") or 0.0),
            )
        )
        gates["lighting_change_score"] = float(estimate_dict.get("lighting_flicker_score") or 0.0)
        gates["camera_shake_score"] = float(estimate_dict.get("camera_shake_score") or 0.0)
        exclusion_reason = None
        if disturbance:
            exclusion_reason = str(raw_motion_class).lower()
        elif not motion_evidence_valid:
            exclusion_reason = (
                gates.get("grasp_zone_motion_reason")
                or gates["object_motion_reason"]
            )
        elif self.stage == "PRE_CLOSE":
            exclusion_reason = "pre_close_motion_is_not_slip"
        gates["davis_motion_usable_for_grasp"] = bool(
            motion_evidence_valid
            and not disturbance
            and self.stage != "PRE_CLOSE"
        )
        gates["davis_motion_stage"] = self.stage
        gates["davis_motion_post_close_applicable"] = self.stage != "PRE_CLOSE"
        gates["davis_live_quality_contribution"] = "included" if gates["davis_motion_usable_for_grasp"] else "excluded"
        gates["davis_motion_used_for_live_quality"] = gates["davis_motion_usable_for_grasp"]
        gates["davis_motion_live_quality_reason"] = (
            "object_motion_valid"
            if gates["object_motion_valid"] and gates["davis_motion_usable_for_grasp"]
            else "post_grasp_fixed_grasp_zone_valid"
            if gates.get("grasp_zone_motion_valid") and gates["davis_motion_usable_for_grasp"]
            else f"excluded: {exclusion_reason}"
        )
        gates["davis_motion_excluded_reason"] = exclusion_reason
        gates["exclusion_reason"] = exclusion_reason
        estimate_invalid_reason = str(estimate_dict.get("invalid_reason") or "")
        if self.stage == "PRE_CLOSE":
            closeup_exclusion_reason = "pre_close_motion_is_not_slip"
        elif event_timestamp_status in {"STALE", "FUTURE"}:
            closeup_exclusion_reason = "stale_events"
        elif disturbance:
            closeup_exclusion_reason = str(raw_motion_class).lower()
        elif estimate_invalid_reason in {"no_events", "insufficient_events"}:
            closeup_exclusion_reason = estimate_invalid_reason
        elif analysis_roi_source == "full_frame":
            closeup_exclusion_reason = "full_frame_roi_only"
        elif not gates["roi_object_specific"] and not gates.get("grasp_zone_motion_valid"):
            closeup_exclusion_reason = "no_object_roi"
        elif estimate_invalid_reason == "low_confidence":
            closeup_exclusion_reason = "low_confidence"
        elif not motion_evidence_valid:
            closeup_exclusion_reason = (
                gates.get("grasp_zone_motion_reason")
                or gates["object_motion_reason"]
                or "low_confidence"
            )
        else:
            closeup_exclusion_reason = None
        event_health_valid = bool(
            self.monitor.active
            and event_timestamp_status not in {"STALE", "FUTURE"}
        )
        closeup_used_for_live_quality = bool(
            gates["davis_motion_usable_for_grasp"]
            and event_health_valid
            and closeup_exclusion_reason is None
        )
        if self.stage == "PRE_CLOSE":
            closeup_monitoring_state = "PRE_LOCK_EVENT_HEALTH_ONLY"
        elif closeup_elapsed_sec is not None and closeup_elapsed_sec < STABILITY_WINDOW_SEC:
            closeup_monitoring_state = "ACTION_WINDOW_MONITORING"
        else:
            closeup_monitoring_state = "POST_WINDOW_ROLLING_MONITORING"
        if closeup_elapsed_sec is None or closeup_elapsed_sec < STABILITY_WINDOW_SEC:
            closeup_stability_decision = "PENDING_1200MS_WINDOW"
        elif closeup_used_for_live_quality:
            closeup_stability_decision = scores.get("decision") or "HOLD_AND_RECHECK"
        else:
            closeup_stability_decision = "DAVIS_DID_NOT_PROVIDE_OBJECT_SPECIFIC_STABILITY_EVIDENCE"
        gates.update({
            "event_health_valid": event_health_valid,
            "davis_closeup_used_for_live_quality": closeup_used_for_live_quality,
            "davis_closeup_exclusion_reason": closeup_exclusion_reason,
            "davis_closeup_confidence": estimate_dict.get("motion_estimation_confidence"),
            "davis_closeup_motion_class": raw_motion_class,
            "davis_closeup_window": closeup_window,
            "davis_closeup_monitoring_state": closeup_monitoring_state,
            "davis_closeup_stability_decision": closeup_stability_decision,
        })
        try:
            boundary_field = search_boundary_events(
                analysis_boundary_points,
                self.events_arr,
                event_window_sec,
                now=now,
                boundary_source=analysis_boundary_source,
                alignment_status=analysis_roi_projection.get("roi_projection_status") or "NO_OBJECT_ROI",
                vertical_slip_sign=DAVIS_VERTICAL_SLIP_SIGN,
                object_motion_valid=gates["object_motion_valid"],
                suppression_reason=exclusion_reason or gates.get("object_motion_reason"),
                contact_band_polygon=analysis_contact_band_polygon,
            ).to_dict()
        except (TypeError, ValueError) as exc:
            boundary_field = {
                "measurement_status": "NOT_AVAILABLE",
                "suppression_reason": f"boundary_search_error: {exc}",
                "object_motion_valid": False,
                "physically_validated": False,
            }
        diagnostic_pattern = diagnostic_movement_pattern(boundary_field)
        estimate_dict["boundary_evidence"] = {
            key: boundary_field.get(key)
            for key in (
                "measurement_status", "alignment_status", "rotation_cue", "slip_cue",
                "vibration_cue", "incipient_slip_precursor_score", "suppression_reason",
                "physically_validated",
            )
        }
        scores = fuse_boundary_motion_evidence(scores, boundary_field, allow_classification=self.stage != "PRE_CLOSE")
        raw_motion_class = scores.get("raw_motion_class", raw_motion_class)
        scores["disturbance_type"] = canonical_disturbance_type(
            raw_motion_class,
            scores.get("disturbance_type"),
        )
        scores.update(gates)
        scores = apply_object_motion_presentation_gate(scores, gates)
        if (
            closeup_elapsed_sec is not None
            and closeup_elapsed_sec >= STABILITY_WINDOW_SEC
            and closeup_used_for_live_quality
        ):
            scores["davis_closeup_stability_decision"] = scores.get("decision") or "HOLD_AND_RECHECK"
        accepted_slip = gates["davis_motion_usable_for_grasp"] and raw_motion_class in {"INCIPIENT_SLIP", "GROSS_SLIP"}
        if accepted_slip and self.slip_first_seen is None:
            self.slip_first_seen = now
        self.latency_tracker.inference_complete(accepted_slip=accepted_slip)
        self.motion_class_history.append([time.time(), scores["motion_class"]])
        self.decision_history.append([time.time(), scores["decision"]])
        self.history.append([time.time(), stability["event_rate"], scores["incipient_slip_score"], scores["gross_slip_score"], scores["rotation_risk_score"], scores["vibration_risk_score"], scores["stability_score"]])
        update_ms = (time.perf_counter() - update_start) * 1000.0
        self.update_times.append(time.monotonic())
        self.update_times = self.update_times[-120:]
        update_fps = 0.0
        if len(self.update_times) >= 2:
            update_fps = (len(self.update_times) - 1) / max(self.update_times[-1] - self.update_times[0], 1e-6)
        aps_fps = 0.0
        if len(self.aps_times) >= 2:
            aps_fps = (len(self.aps_times) - 1) / max(self.aps_times[-1] - self.aps_times[0], 1e-6)
        aps_status = {} if self.latest_aps_record is None else self.latest_aps_record.to_status()
        aps_state = aps_release_state(aps_status)
        aps_target_context = self.latest_aps_target_context or {}
        aps_target_captured = aps_target_context.get("captured_monotonic")
        aps_target_age_sec = (
            None
            if aps_target_captured is None
            else max(0.0, now - float(aps_target_captured))
        )
        aps_target_max_age_sec = float(
            aps_target_context.get("aps_target_context_max_age_sec")
            or self._aps_target_max_age_sec()
        )
        aps_target_current = bool(
            aps_target_context
            and aps_target_age_sec is not None
            and aps_target_age_sec <= aps_target_max_age_sec
        )
        confirmed_aps_context = getattr(self, "confirmed_aps_target_context", None) or {}
        confirmed_aps_captured = confirmed_aps_context.get("captured_monotonic")
        try:
            confirmed_aps_age = now - float(confirmed_aps_captured)
        except (TypeError, ValueError):
            confirmed_aps_age = None
        confirmed_aps_current = bool(
            confirmed_aps_context
            and confirmed_aps_age is not None
            and 0.0 <= confirmed_aps_age <= self._aps_target_max_age_sec()
        )
        aps_target_detection = self.latest_aps_target_detection or {}
        positive_count = int(estimate_dict.get("positive_event_count") or 0)
        negative_count = int(estimate_dict.get("negative_event_count") or 0)
        polarity_total = positive_count + negative_count
        polarity_balance = (
            0.0
            if polarity_total == 0
            else float((positive_count - negative_count) / polarity_total)
        )
        report = {
            "report_timestamp": time.time(),
            "backend": self.monitor.backend_name,
            "davis_backend": self.monitor.backend_name,
            "davis_owner_status": self.monitor.owner_info.get("davis_owner_status") if self.monitor.owner_info else None,
            "davis_writer_pid": self.monitor.owner_info.get("davis_writer_pid") if self.monitor.owner_info else None,
            "davis_writer_session_id": self.monitor.owner_info.get("davis_writer_session_id") if self.monitor.owner_info else None,
            "davis_device_open_count": self.monitor.owner_info.get("davis_device_open_count") if self.monitor.owner_info else 0,
            "duplicate_davis_owner_detected": bool((self.monitor.owner_info or {}).get("duplicate_davis_owner_detected")),
            "duplicate_open_prevented": bool((self.monitor.owner_info or {}).get("duplicate_open_prevented")),
            "last_poll_event_count": self.last_poll_event_count,
            "drained_poll_count": self.drained_poll_count,
            "rolling_event_count": len(self.events_arr),
            "rolling_buffer_span_sec": round(event_buffer_age_sec(self.events_arr), 3),
            "raw_window_event_count": raw_window_event_count,
            "raw_event_heatmap_visible": True,
            "raw_event_heatmap_event_count": min(raw_window_event_count, LIVE_HEATMAP_MAX_EVENTS),
            "object_roi_window_event_count": object_roi_window_event_count,
            "event_analysis_status": event_analysis_status,
            "event_health_valid": event_health_valid,
            "event_host_timestamp": event_host_timestamp,
            "event_timestamp_basis": getattr(
                self.monitor,
                "event_timestamp_basis",
                "device_microseconds_host_anchored",
            ),
            "event_timestamp_status": event_timestamp_status,
            "event_timestamp_lag_ms": event_timestamp_lag_ms,
            "event_timestamp_reanchor_count": int(
                getattr(self.monitor, "event_timestamp_reanchor_count", 0)
            ),
            "total_raw_events_polled": self.total_raw_events_polled,
            "total_events_kept": self.total_events_kept,
            "dropped_events_due_tick_cap": self.dropped_events_due_tick_cap,
            "dropped_events_due_rolling_cap": self.dropped_events_due_rolling_cap,
            "dropped_event_windows": self.dropped_event_windows,
            "event_capture_drain_ms": round(drain_ms, 2),
            "davis_processing_latency_ms": round(update_ms, 2),
            "davis_update_fps": round(update_fps, 2),
            "aps_stream_status": "active" if self.latest_aps_record and self.latest_aps_record.aps_frame_valid else "unavailable",
            "aps_state": aps_state,
            "aps_fps": round(aps_fps, 2),
            **{k: v for k, v in (self.latest_aps_stats or {}).items() if k.startswith("aps_")},
            **(self.latest_aps_visibility or {}),
            **(self.latest_aps_display_meta or {}),
            **(self.aps_exposure_info or {}),
            "davis_aps_frame_age_ms": None if not self.latest_aps_record else round((time.time() - self.latest_aps_record.host_timestamp) * 1000.0, 1),
            "aps_status": aps_status,
            "aps_mode": aps_status.get("aps_mode") or "DAVIS APS Unavailable",
            "aps_color_verified": bool(aps_status.get("aps_color_verified")),
            "aps_channel_order": aps_status.get("aps_channel_order"),
            "aps_channel_order_source": aps_status.get("aps_channel_order_source"),
            "aps_color_accuracy_status": (
                "NOT_CALIBRATED"
                if aps_status.get("aps_color_verified")
                else "NOT_APPLICABLE_OR_NOT_TESTED"
            ),
            "aps_device_timestamp_us": aps_status.get("device_timestamp"),
            "aps_device_timestamp_basis": (
                "MID_EXPOSURE_DEVICE_CLOCK"
                if aps_status.get("device_timestamp") is not None
                else "NOT_AVAILABLE"
            ),
            "aps_host_timestamp": aps_status.get("host_timestamp"),
            "aps_target_context_current": aps_target_current,
            "aps_target_context_age_ms": (
                None if aps_target_age_sec is None else round(aps_target_age_sec * 1000.0, 2)
            ),
            "aps_target_context_max_age_ms": round(aps_target_max_age_sec * 1000.0, 2),
            "aps_target_detector_cadence_ms": (
                None
                if self._aps_target_detector_cadence_sec() is None
                else round(self._aps_target_detector_cadence_sec() * 1000.0, 2)
            ),
            "aps_target_roi_xywh": (
                list(aps_target_context.get("roi_xywh") or []) if aps_target_current else None
            ),
            "aps_target_detection_status": aps_target_detection.get("davis_detection_status"),
            "aps_target_detection_confidence": aps_target_detection.get("davis_detection_confidence"),
            "aps_target_detection_source": aps_target_detection.get("aps_detection_source"),
            "aps_target_confirmation_count": int(getattr(self, "pending_aps_target_count", 0)),
            "aps_target_confirmed": confirmed_aps_current,
            "aps_target_missed_detections": int(getattr(self, "aps_target_missed_detections", 0)),
            "aps_target_confirmation_expired_reason": getattr(
                self,
                "aps_target_confirmation_expired_reason",
                None,
            ),
            "aps_target_post_grasp_latched": bool(aps_target_context.get("aps_post_grasp_latched")),
            "aps_target_failure_reason": (
                None if aps_target_current else self.latest_aps_target_failure_reason or "aps_target_stale"
            ),
            "heatmap_window_ms": int(round(LIVE_HEATMAP_WINDOW_SEC * 1000.0)),
            "scoring_window_ms": int(round(STABILITY_WINDOW_SEC * 1000.0)),
            "event_window_ms": int(round(event_window_sec * 1000.0)),
            "event_window_target_ms": int(round(PRIMARY_EVENT_WINDOW_SEC * 1000.0)),
            "event_window_fallback_ms": int(round(FALLBACK_EVENT_WINDOW_SEC * 1000.0)),
            "event_window_mode": event_window_mode,
            "rolling_buffer_age_sec": round(event_buffer_age_sec(self.events_arr), 3),
            "event_stream_status": "active" if self.monitor.active else "unavailable",
            "current_stage": self.stage,
            "baseline_calibrated": self.baseline_metrics is not None,
            "roi_context_revision": self.roi_context_revision,
            "baseline_roi_context_revision": self.baseline_roi_context_revision,
            "baseline_roi_compatible": baseline_roi_compatible,
            "baseline_rejected_count": self.baseline_rejected_count,
            "baseline_duration_sec": self.baseline_duration_sec,
            "baseline_max_quiet_rate": self.baseline_max_quiet_rate,
            "baseline_metrics": baseline,
            "current_metrics": stability,
            "micro_window_metrics": micro,
            "short_window_metrics": short,
            "event_window_metrics": event_window_metrics,
            "event_count": event_window_metrics.get("event_count", 0),
            "event_rate": event_window_metrics.get("event_rate", 0.0),
            "event_density": event_window_metrics.get("event_density", 0.0),
            "impact_window_metrics": action_window_metrics["impact"],
            "settle_window_metrics": action_window_metrics["settle"],
            "stability_window_metrics": action_window_metrics["stability"],
            "rolling_impact_window_metrics": impact,
            "rolling_settle_window_metrics": settle,
            "rolling_stability_window_metrics": stability,
            "closeup_action_origin_monotonic": self.close_action_origin,
            "post_grasp_monitoring_action_id": self.close_action_id,
            "post_grasp_monitoring_started_at": self.close_action_started_at,
            "closeup_action_elapsed_sec": closeup_elapsed_sec,
            "closeup_action_window_complete": self.closeup_action_snapshot_complete,
            "closeup_action_windows_sec": {
                name: [start, end]
                for name, (start, end) in CLOSEUP_ACTION_WINDOWS_SEC.items()
            },
            "closeup_window_metrics": action_window_metrics,
            # roi_* describes the evidence used by the motion estimator;
            # davis_roi/acquisition_roi remains the constant user-visible view.
            "roi_source": analysis_roi_source,
            "roi_xywh": analysis_roi_xywh,
            "davis_roi_xywh": self.roi_xywh,
            "acquisition_roi_source": self.roi_source,
            "acquisition_roi_xywh": self.roi_xywh,
            "acquisition_roi_fixed": self.fixed_center_third_mode,
            "acquisition_roi_projection_method": self.roi_projection.get("roi_projection_method"),
            "acquisition_roi_projection_status": self.roi_projection.get("roi_projection_status"),
            "object_evidence_context_current": self.object_evidence_context is not None,
            "object_evidence_roi_source": self.object_evidence_roi_source,
            "object_evidence_roi_xywh": self.object_evidence_roi_xywh,
            "object_evidence_boundary_source": self.object_evidence_boundary_source,
            "object_evidence_contact_band_source": (
                None
                if self.object_evidence_context is None
                else self.object_evidence_context.get("contact_band_source")
            ),
            "object_evidence_contact_band_physically_validated": False,
            "object_evidence_status": (
                None
                if self.object_evidence_context is None
                else self.object_evidence_context.get("aps_object_evidence_status")
            ),
            "object_evidence_overlap_ratio": (
                None
                if self.object_evidence_context is None
                else self.object_evidence_context.get("aps_object_evidence_overlap_ratio")
            ),
            "object_evidence_physically_validated": False,
            "object_evidence_invalid_reason": self.object_evidence_invalid_reason,
            "analysis_roi_mode": self.analysis_roi_mode,
            "analysis_roi_fixed": self.fixed_center_third_mode,
            "analysis_roi_geometry": (
                "aps_object_evidence_within_fixed_middle_third"
                if self.object_evidence_context is not None
                else "middle_third_both_axes"
                if self.fixed_center_third_mode
                else "configured_or_context_driven"
            ),
            "analysis_roi_object_specific": gates.get("roi_object_specific"),
            "aps_target_used_as_analysis_roi": False if self.fixed_center_third_mode else None,
            "aps_target_used_as_object_evidence_roi": bool(
                self.object_evidence_context is not None
            ),
            "full_frame_fallback": analysis_roi_source == "full_frame",
            "local_event_rate": (
                event_window_metrics.get("event_rate")
                if analysis_roi_source != "full_frame"
                else None
            ),
            "active_pixel_ratio": estimate_dict.get("active_pixel_ratio"),
            "active_pixel_count": estimate_dict.get("active_pixel_count"),
            "positive_event_count": positive_count,
            "negative_event_count": negative_count,
            "polarity_balance": round(polarity_balance, 4),
            **analysis_roi_projection,
            "motion_estimate": estimate_dict,
            "boundary_event_field": boundary_field,
            **diagnostic_pattern,
            "boundary_rotation_cue": boundary_field.get("rotation_cue"),
            "boundary_slip_cue": boundary_field.get("slip_cue"),
            "boundary_vibration_cue": boundary_field.get("vibration_cue"),
            "boundary_incipient_slip_precursor_score": boundary_field.get("incipient_slip_precursor_score"),
            "motion_estimate_id": estimate_dict.get("estimate_id"),
            "motion_estimate_valid": estimate_dict.get("estimate_valid"),
            "motion_estimate_invalid_reason": estimate_dict.get("invalid_reason"),
            "motion_estimation_method": estimate_dict.get("motion_estimation_method"),
            "motion_estimation_confidence": estimate_dict.get("motion_estimation_confidence"),
            "motion_confidence": estimate_dict.get("motion_estimation_confidence"),
            "translation_method": estimate_dict.get("translation_method"),
            "translation_confidence": estimate_dict.get("translation_confidence"),
            "translation_valid": estimate_dict.get("translation_valid"),
            "translation_invalid_reason": estimate_dict.get("translation_invalid_reason"),
            "dx_px": estimate_dict.get("dx_px"),
            "dy_px": estimate_dict.get("dy_px"),
            "displacement_px": estimate_dict.get("displacement_px"),
            "displacement_direction_deg": estimate_dict.get("displacement_direction_deg"),
            "velocity_x_px_s": estimate_dict.get("velocity_x_px_s"),
            "velocity_y_px_s": estimate_dict.get("velocity_y_px_s"),
            "speed_px_s": estimate_dict.get("speed_px_s"),
            "cumulative_dx_px": estimate_dict.get("cumulative_dx_px"),
            "cumulative_dy_px": estimate_dict.get("cumulative_dy_px"),
            "cumulative_slip_px": estimate_dict.get("cumulative_slip_px"),
            "cumulative_motion_path_px": estimate_dict.get("cumulative_motion_path_px"),
            "downward_slip_px": estimate_dict.get("downward_slip_px"),
            "cumulative_downward_slip_px": estimate_dict.get("cumulative_downward_slip_px"),
            "slip_along_bottle_axis_px": estimate_dict.get("slip_along_bottle_axis_px"),
            "slip_across_bottle_axis_px": estimate_dict.get("slip_across_bottle_axis_px"),
            "slip_confidence": estimate_dict.get("slip_confidence"),
            "rotation_method": estimate_dict.get("rotation_method"),
            "rotation_delta_deg": estimate_dict.get("rotation_delta_deg"),
            "cumulative_rotation_deg": estimate_dict.get("cumulative_rotation_deg"),
            "angular_velocity_deg_s": estimate_dict.get("angular_velocity_deg_s"),
            "rotation_confidence": estimate_dict.get("rotation_confidence"),
            "rotation_valid": estimate_dict.get("rotation_valid"),
            "rotation_invalid_reason": estimate_dict.get("rotation_invalid_reason"),
            "vibration_amplitude_px_rms": estimate_dict.get("vibration_amplitude_px_rms"),
            "vibration_frequency_hz": estimate_dict.get("vibration_frequency_hz"),
            "vibration_confidence": estimate_dict.get("vibration_confidence"),
            "displacement_above_baseline_px": estimate_dict.get("displacement_above_baseline_px"),
            "rotation_above_baseline_deg": estimate_dict.get("rotation_above_baseline_deg"),
            "vibration_above_baseline_px": estimate_dict.get("vibration_above_baseline_px"),
            "bottle_axis_available": estimate_dict.get("bottle_axis_available"),
            "bottle_axis_source": estimate_dict.get("bottle_axis_source"),
            "slip_axis_valid": estimate_dict.get("slip_axis_valid"),
            "slip_axis_reason": estimate_dict.get("slip_axis_reason"),
            "local_to_global_event_ratio": estimate_dict.get("local_to_global_event_ratio"),
            "outside_roi_event_rate": estimate_dict.get("outside_roi_event_rate"),
            "global_event_ratio": estimate_dict.get("global_event_ratio"),
            "full_frame_uniformity": estimate_dict.get("full_frame_uniformity"),
            "lighting_flicker_score": estimate_dict.get("lighting_flicker_score"),
            "camera_shake_score": estimate_dict.get("camera_shake_score"),
            "background_motion_score": estimate_dict.get("background_motion_score"),
            "object_motion_score": estimate_dict.get("object_motion_score"),
            "disturbance_confidence": estimate_dict.get("disturbance_confidence"),
            "disturbance_rejection_applied": estimate_dict.get("disturbance_rejection_applied"),
            "motion_hull_explanation": "This DAVIS hull shows recent event-based motion around the grasped bottle region. Small changes suggest micro-deviation; larger/biased hull movement may suggest slip or rotation.",
            **hull,
            "slip_detection_latency_ms": None if self.slip_first_seen is None else round((self.slip_first_seen - self.start_time) * 1000.0, 1),
            "slip_detection_latency_basis": "analysis_epoch_start_to_first_accepted_slip_window_reference",
            "slip_detection_latency_clock_domain": "host_monotonic",
            "slip_detection_latency_physical_onset_measured": False,
            "incipient_slip_event_count": short["event_count"] if scores["incipient_slip_score"] >= 45 else 0,
            "gross_slip_event_count": stability["event_count"] if scores["gross_slip_score"] >= 55 else 0,
            "false_slip_warning_count": self.false_slip_warning_count,
            "stable_window_duration_sec": STABILITY_WINDOW_SEC if scores["motion_class"] == "STABLE" else 0.0,
            "post_grasp_stability_score": scores["stability_score"],
            "motion_class_history": list(self.motion_class_history)[-80:],
            "decision_history": list(self.decision_history)[-80:],
            "explanation": scores["response_reason"],
            "lightweight_event_metrics": {
                "event_count": event_window_metrics["event_count"],
                "event_rate": event_window_metrics["event_rate"],
                "event_density": event_window_metrics["event_density"],
                "centroid_x": event_window_metrics["centroid_x"],
                "centroid_y": event_window_metrics["centroid_y"],
                "centroid_shift_x": event_window_metrics["centroid_shift_x"],
                "centroid_shift_y": event_window_metrics["centroid_shift_y"],
                "left_right_event_imbalance": event_window_metrics["left_right_asymmetry"],
                "rotation_like_event_pattern": scores["rotation_risk_score"],
                "sudden_spike_after_grasp": scores["impact_score"],
                "motion_hull_point_count": hull["motion_hull_point_count"],
                "motion_hull_area_px2": hull["motion_hull_area_px2"],
                "event_window_ms": int(round(event_window_sec * 1000.0)),
            },
            **scores,
        }
        report.update(davis_event_band_report_aliases(report))
        report["software_timing"] = self.latency_tracker.report_payload_ready()
        self.report = report
        return report

    def save_outputs(self, include_images=True):
        OUTPUT_DIR.mkdir(exist_ok=True)
        DAVIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.report.update(davis_event_band_report_aliases(self.report))
        status = {
            "state": self.report.get("event_stream_status") or self.report.get("davis_status"),
            "event_state": "STREAM_ACTIVE_EVENTS"
            if (self.report.get("event_stream_status") == "active" and float(((self.report.get("current_metrics") or {}).get("event_rate") or self.report.get("event_rate") or 0.0) or 0.0) > 0.0)
            else ("STREAM_ACTIVE_NO_MOTION" if self.report.get("event_stream_status") == "active" else "STREAM_NOT_STARTED"),
            "backend": self.report.get("backend") or self.report.get("davis_backend"),
            "aps_state": self.report.get("aps_state"),
            "aps_stream_status": self.report.get("aps_stream_status"),
            **{k: self.report.get(k) for k in (
                "aps_min", "aps_p1", "aps_p50", "aps_p95", "aps_p99", "aps_max",
                "aps_mean", "aps_std", "aps_dtype", "aps_shape", "aps_channel_count",
                "aps_channel_means", "aps_channel_stds",
                "aps_visibility_status", "aps_visibility_score", "aps_underexposed",
                "aps_overexposed", "aps_low_contrast", "aps_display_mode",
                "aps_display_enhanced", "aps_display_warning",
                "aps_exposure_control_available", "aps_exposure_us", "aps_gain",
                "aps_exposure_auto", "aps_exposure_configuration_applied",
                "aps_exposure_source",
                "aps_gain_control_available", "aps_gain_status",
                "aps_channel_order", "aps_channel_order_source",
                "aps_color_accuracy_status", "aps_device_timestamp_us",
                "aps_device_timestamp_basis", "aps_host_timestamp",
            ) if self.report.get(k) is not None},
            "event_stream_status": self.report.get("event_stream_status"),
            "motion_class": self.report.get("motion_class"),
            "decision": self.report.get("decision") or self.report.get("post_grasp_decision"),
            "davis_writer_pid": self.report.get("davis_writer_pid"),
            "davis_writer_session_id": self.report.get("davis_writer_session_id"),
            "davis_device_open_count": self.report.get("davis_device_open_count"),
            "duplicate_open_prevented": self.report.get("duplicate_open_prevented"),
            "duplicate_davis_owner_detected": self.report.get("duplicate_davis_owner_detected"),
            "motion_estimate_valid": self.report.get("motion_estimate_valid"),
            "motion_estimate_invalid_reason": self.report.get("motion_estimate_invalid_reason"),
            "motion_estimation_confidence": self.report.get("motion_estimation_confidence"),
            "motion_confidence": self.report.get("motion_confidence") or self.report.get("motion_estimation_confidence"),
            "dx_px": self.report.get("dx_px"),
            "dy_px": self.report.get("dy_px"),
            "displacement_px": self.report.get("displacement_px"),
            "cumulative_slip_px": self.report.get("cumulative_slip_px"),
            "downward_slip_px": self.report.get("downward_slip_px"),
            "rotation_delta_deg": self.report.get("rotation_delta_deg"),
            "cumulative_rotation_deg": self.report.get("cumulative_rotation_deg"),
            "angular_velocity_deg_s": self.report.get("angular_velocity_deg_s"),
            "roll_valid": self.report.get("roll_valid", False),
            "roll_delta_deg": self.report.get("roll_delta_deg"),
            "roll_status": self.report.get("roll_status") or "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
            "roll_exclusion_reason": self.report.get("roll_exclusion_reason"),
            "vibration_amplitude_px_rms": self.report.get("vibration_amplitude_px_rms"),
            "disturbance_type": self.report.get("disturbance_type"),
            "event_health_valid": self.report.get("event_health_valid"),
            "object_motion_valid": self.report.get("object_motion_valid"),
            "object_motion_reason": self.report.get("object_motion_reason"),
            "roi_source": self.report.get("roi_source"),
            "roi_confidence": self.report.get("roi_confidence"),
            "roi_projection_method": self.report.get("roi_projection_method"),
            "roi_projection_warning": self.report.get("roi_projection_warning"),
            "full_frame_fallback": self.report.get("full_frame_fallback"),
            "davis_closeup_window": self.report.get("davis_closeup_window"),
            "davis_closeup_monitoring_state": self.report.get("davis_closeup_monitoring_state"),
            "davis_closeup_used_for_live_quality": self.report.get("davis_closeup_used_for_live_quality"),
            "davis_closeup_exclusion_reason": self.report.get("davis_closeup_exclusion_reason"),
            "timestamp": time.time(),
        }
        timeline = {
            "motion_class_history": self.report.get("motion_class_history") or [],
            "decision_history": self.report.get("decision_history") or [],
            "history": self.report.get("history") or list(self.history),
            "timestamp": time.time(),
        }
        _write_json(DAVIS_LATEST_MOTION_REPORT_PATH, self.report)
        _write_json(DAVIS_LATEST_STATUS_PATH, status)
        _write_json(DAVIS_LATEST_TIMELINE_PATH, timeline)
        estimate = self.report.get("motion_estimate") or {}
        _write_json(DAVIS_LATEST_MOTION_ESTIMATE_PATH, estimate)
        _write_json(DAVIS_MOTION_CALIBRATION_PATH, self.motion_estimator.baseline)
        _write_json(DAVIS_MOTION_DEBUG_SUMMARY_PATH, {
            "timestamp": time.time(),
            "motion_estimate_valid": self.report.get("motion_estimate_valid"),
            "invalid_reason": self.report.get("motion_estimate_invalid_reason"),
            "method": self.report.get("motion_estimation_method"),
            "confidence": self.report.get("motion_estimation_confidence"),
            "roi_source": self.report.get("roi_source"),
            "disturbance_type": self.report.get("disturbance_type"),
            "quantitative_evidence": self.report.get("quantitative_evidence"),
            "live_quality_contribution": "DAVIS measured pixel motion lowers live stability confidence; epsilon is not recomputed.",
        })
        if self.close_action_origin is not None:
            post_grasp = post_grasp_monitoring_payload(
                self.report,
                self.close_action_id,
                self.close_action_started_at,
            )
            _write_json(DAVIS_POST_GRASP_LATEST_REPORT_PATH, post_grasp)
            if time.time() - self.last_post_grasp_timeline_write >= 0.10:
                _append_jsonl(
                    DAVIS_POST_GRASP_TIMELINE_PATH,
                    post_grasp_timeline_entry(
                        self.report,
                        self.close_action_id,
                        self.close_action_started_at,
                    ),
                )
                self.last_post_grasp_timeline_write = time.time()
        if estimate and time.time() - self.last_motion_timeline_write >= 0.10:
            _append_jsonl(DAVIS_MOTION_TIMELINE_PATH, {
                "timestamp": time.time(),
                "estimate_id": estimate.get("estimate_id"),
                "motion_class": self.report.get("raw_motion_class") or estimate.get("motion_class"),
                "dx_px": estimate.get("dx_px"),
                "dy_px": estimate.get("dy_px"),
                "displacement_px": estimate.get("displacement_px"),
                "cumulative_slip_px": estimate.get("cumulative_slip_px"),
                "rotation_delta_deg": estimate.get("rotation_delta_deg"),
                "cumulative_rotation_deg": estimate.get("cumulative_rotation_deg"),
                "roll_valid": self.report.get("roll_valid", False),
                "roll_delta_deg": self.report.get("roll_delta_deg"),
                "roll_status": self.report.get("roll_status") or "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
                "vibration_amplitude_px_rms": estimate.get("vibration_amplitude_px_rms"),
                "confidence": estimate.get("motion_estimation_confidence"),
                "disturbance_type": estimate.get("disturbance_type"),
                "event_health_valid": self.report.get("event_health_valid"),
                "object_motion_valid": self.report.get("object_motion_valid"),
                "roi_source": self.report.get("roi_source"),
                "davis_closeup_window": self.report.get("davis_closeup_window"),
                "davis_closeup_used_for_live_quality": self.report.get("davis_closeup_used_for_live_quality"),
                "davis_closeup_exclusion_reason": self.report.get("davis_closeup_exclusion_reason"),
                "post_grasp_monitoring_action_id": self.report.get("post_grasp_monitoring_action_id"),
                "post_grasp_monitoring_started_at": self.report.get("post_grasp_monitoring_started_at"),
                "davis_target_lock_source_writer_session_id": self.report.get("davis_target_lock_source_writer_session_id"),
                "davis_target_lock_source_lock_frame_id": self.report.get("davis_target_lock_source_lock_frame_id"),
                "davis_target_lock_source_attempt_id": self.report.get("davis_target_lock_source_attempt_id"),
                "davis_target_lock_source_bottle_target_id": self.report.get("davis_target_lock_source_bottle_target_id"),
            })
            self.last_motion_timeline_write = time.time()
        # Compatibility mirrors for older dashboard cards and scripts.
        _write_json(OUTPUT_DIR / "davis_motion_report.json", self.report)
        _write_json(OUTPUT_DIR / "davis_motion_evaluation.json", {
            "slip_detection_latency_ms": self.report.get("slip_detection_latency_ms"),
            "slip_detection_latency_basis": self.report.get("slip_detection_latency_basis"),
            "slip_detection_latency_clock_domain": self.report.get("slip_detection_latency_clock_domain"),
            "slip_detection_latency_physical_onset_measured": False,
            "software_timing": self.report.get("software_timing"),
            "incipient_slip_event_count": self.report.get("incipient_slip_event_count"),
            "gross_slip_event_count": self.report.get("gross_slip_event_count"),
            "false_slip_warning_count": self.report.get("false_slip_warning_count"),
            "stable_window_duration_sec": self.report.get("stable_window_duration_sec"),
            "post_grasp_stability_score": self.report.get("post_grasp_stability_score"),
            "motion_class_history": self.report.get("motion_class_history"),
            "decision_history": self.report.get("decision_history"),
        })
        if not include_images:
            return
        try:
            from PIL import Image
            heatmap = heatmap_rgb(
                recent_events(self.events_arr),
                self.roi_xywh,
                background_rgb=self.latest_aps_display_enhanced
                if self.latest_aps_display_enhanced is not None
                else self.latest_aps_display_rgb,
                neighborhood_filter=True,
            )
            heatmap = draw_motion_estimate_overlay(heatmap, self.report.get("motion_estimate"))
            _write_rgb_jpeg(DAVIS_LATEST_EVENT_HEATMAP_PATH, heatmap, quality=82)
            Image.fromarray(heatmap).save(OUTPUT_DIR / "davis_event_heatmap.png")
        except Exception:
            pass
        try:
            if self.latest_aps_display_rgb is not None:
                _write_rgb_jpeg(DAVIS_LATEST_APS_PATH, self.latest_aps_display_rgb, quality=82)
        except Exception:
            pass
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            hist = np.asarray(list(self.history), dtype=float) if self.history else np.empty((0, 7))
            fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
            if hist.size:
                xs = hist[:, 0] - hist[0, 0]
                ax.plot(xs, hist[:, 1], label="event rate")
                ax.plot(xs, hist[:, 2], label="incipient")
                ax.plot(xs, hist[:, 3], label="gross")
                ax.plot(xs, hist[:, 4], label="rotation")
                ax.plot(xs, hist[:, 5], label="vibration")
                ax.plot(xs, hist[:, 6], label="stability")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "davis_motion_timeline.png")
            fig.savefig(OUTPUT_DIR / "davis_motion_dashboard.png")
            plt.close(fig)
        except Exception:
            pass


class DavisMotionWindow:
    def __init__(self, roi_xywh=None, stage="PRE_CLOSE", sensitivity="sensitive"):
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget
        self.Qt = Qt
        self.QImage = QImage
        self.QPixmap = QPixmap
        self.app = QApplication.instance() or QApplication([])
        self.analyzer = DavisMotionAnalyzer(roi_xywh=roi_xywh, stage=stage, sensitivity=sensitivity)
        self.last_save_time = 0.0
        self.last_heatmap_time = 0.0
        self.last_timeline_time = 0.0
        self.last_analysis_time = 0.0
        self.last_report = self.analyzer.report
        self.display_times = []
        self.analysis_times = []
        width, height = DAVIS_FRAME_SIZE
        self.heat_accum = np.zeros((height, width), dtype=np.float32)
        self.heat_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.window = QMainWindow()
        self.window.setWindowTitle("DAVIS Motion Safety Monitor")
        self.window.resize(1380, 850)
        central = QWidget()
        layout = QVBoxLayout(central)
        self.window.setCentralWidget(central)
        self.cards = {}
        top = QGridLayout()
        names = ["Event Stream", "Stage", "Motion Class", "Incipient Slip", "Gross Slip", "Rotation Risk", "Vibration Risk", "Stability Score", "Decision"]
        for i, name in enumerate(names):
            title = QLabel(name)
            value = QLabel("starting")
            value.setMinimumHeight(58)
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet(self.style("neutral"))
            top.addWidget(title, 0, i)
            top.addWidget(value, 1, i)
            self.cards[name] = value
        layout.addLayout(top)
        body = QGridLayout()
        self.heat = QLabel("DAVIS debug heatmap disabled\nSet DAVIS_DEBUG_HEATMAP=1 to show it.")
        self.timeline = QLabel("Timeline rendering disabled in live mode\nNumeric event metrics stay active.")
        self.info = QLabel("motion interpretation")
        for w in (self.heat, self.timeline):
            w.setMinimumSize(430, 360)
            w.setAlignment(Qt.AlignCenter)
            w.setStyleSheet("background:#0b1020;color:white;border:1px solid #cbd5e1;border-radius:6px;")
        self.info.setMinimumSize(390, 360)
        self.info.setWordWrap(True)
        self.info.setStyleSheet("background:white;color:#0f172a;border:1px solid #cbd5e1;border-radius:6px;padding:12px;font-size:15px;")
        body.addWidget(self.heat, 0, 0)
        body.addWidget(self.timeline, 0, 1)
        body.addWidget(self.info, 0, 2)
        layout.addLayout(body)
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(LIVE_TIMER_MS)

    def style(self, state):
        colors = {"good":("#ecfdf5","#16a34a","#14532d"),"warn":("#fff7ed","#f97316","#7c2d12"),"bad":("#fef2f2","#dc2626","#7f1d1d"),"neutral":("#f8fafc","#94a3b8","#0f172a"),"info":("#eff6ff","#2563eb","#1e3a8a")}
        bg, border, fg = colors[state]
        return f"background:{bg};border:1px solid {border};border-radius:6px;color:{fg};font-weight:700;font-size:14px;padding:6px;"

    def set_card(self, name, text, state="neutral"):
        self.cards[name].setText(str(text))
        self.cards[name].setStyleSheet(self.style(state))

    def pixmap(self, rgb, size=(460, 380)):
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        img = self.QImage(rgb.data, w, h, ch * w, self.QImage.Format_RGB888).copy()
        return self.QPixmap.fromImage(img).scaled(size[0], size[1], self.Qt.KeepAspectRatio, self.Qt.FastTransformation)

    def update_heatmap_image(self):
        width, height = DAVIS_FRAME_SIZE
        self.heat_accum *= LIVE_HEATMAP_DECAY
        arr = self.analyzer.latest_event_array
        if getattr(arr, "ndim", 0) == 2 and arr.shape[0]:
            if arr.shape[0] > LIVE_HEATMAP_MAX_EVENTS:
                arr = arr[-LIVE_HEATMAP_MAX_EVENTS:]
            xs = np.clip(arr[:, 1].astype(np.int32), 0, width - 1)
            ys = np.clip(arr[:, 2].astype(np.int32), 0, height - 1)
            np.add.at(self.heat_accum, (ys, xs), 1.0)
        t = np.log1p(self.heat_accum)
        t = t / max(float(np.max(t)), 1e-6)
        self.heat_rgb.fill(0)
        self.heat_rgb[..., 0] = np.clip(255 * np.minimum(1, 1.9 * t), 0, 255).astype(np.uint8)
        self.heat_rgb[..., 1] = np.clip(255 * np.maximum(0, 1.8 * t - 0.35), 0, 255).astype(np.uint8)
        self.heat_rgb[..., 2] = np.clip(255 * np.maximum(0, 2.8 * t - 1.5), 0, 255).astype(np.uint8)
        x, y, w, h = _clip_roi(self.analyzer.roi_xywh)
        self.heat_rgb[y:y + h, x:x + 2] = (0, 255, 255)
        self.heat_rgb[y:y + h, x + w - 2:x + w] = (0, 255, 255)
        self.heat_rgb[y:y + 2, x:x + w] = (0, 255, 255)
        self.heat_rgb[y + h - 2:y + h, x:x + w] = (0, 255, 255)
        report = self.analyzer.report or {}
        current = report.get("current_metrics") or {}
        debug_lines = [
            f"rate {float(current.get('event_rate', 0.0) or 0.0):.0f} ev/s",
            f"heatmap ~{int(round(LIVE_HEATMAP_WINDOW_SEC * 1000.0))} ms | scoring {int(round(STABILITY_WINDOW_SEC * 1000.0))} ms",
            f"rolling {report.get('rolling_buffer_age_sec', 0.0)}s",
            f"drops {report.get('dropped_event_windows', 0)}/{report.get('dropped_events_due_tick_cap', 0)}/{report.get('dropped_events_due_rolling_cap', 0)}",
        ]
        for idx, text in enumerate(debug_lines):
            yy = 18 + idx * 18
            cv2.putText(self.heat_rgb, text, (6, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(self.heat_rgb, text, (6, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return self.heat_rgb

    def timeline_image(self):
        width, height = 430, 260
        img = np.full((height, width, 3), 248, dtype=np.uint8)
        hist = list(self.analyzer.history)[-160:]
        if len(hist) < 2:
            return img
        arr = np.asarray(hist, dtype=float)
        xs = (arr[:, 0] - arr[0, 0]) / max(arr[-1, 0] - arr[0, 0], 1e-6)
        colors = [(37,99,235),(220,38,38),(127,29,29),(147,51,234),(249,115,22),(22,163,74)]
        for col, c in zip(range(1, 7), colors):
            ys = arr[:, col]
            if col == 1:
                ys = np.clip(ys / max(np.max(ys), 1.0) * 100.0, 0, 100)
            pts = np.column_stack([xs * (width - 20) + 10, (1 - ys / 100.0) * (height - 20) + 10]).astype(int)
            for a, b in zip(pts[:-1], pts[1:]):
                rr = np.linspace(a[1], b[1], 40).astype(int)
                cc = np.linspace(a[0], b[0], 40).astype(int)
                img[np.clip(rr,0,height-1), np.clip(cc,0,width-1)] = c
        return img

    def tick(self):
        tick_start = time.perf_counter()
        now = time.monotonic()
        analysis_period = 1.0 / max(DAVIS_EVENT_ANALYSIS_FPS, 1.0)
        if (now - self.last_analysis_time >= analysis_period) or not self.last_report:
            analysis_start = time.perf_counter()
            r = self.analyzer.update()
            analysis_ms = (time.perf_counter() - analysis_start) * 1000.0
            self.analysis_times.append(now)
            self.analysis_times = self.analysis_times[-120:]
            if len(self.analysis_times) >= 2:
                r["davis_event_analysis_fps"] = round((len(self.analysis_times) - 1) / max(self.analysis_times[-1] - self.analysis_times[0], 1e-6), 2)
            else:
                r["davis_event_analysis_fps"] = 0.0
            r["davis_analysis_time_ms"] = round(analysis_ms, 2)
            r["davis_capture_fps"] = r.get("davis_update_fps")
            r["davis_event_frame_age_ms"] = (
                r.get("event_timestamp_lag_ms")
                if r.get("event_stream_status") == "active"
                and r.get("event_host_timestamp") is not None
                else None
            )
            self.last_report = r
            self.last_analysis_time = now
        else:
            r = self.last_report
        self.display_times.append(now)
        self.display_times = self.display_times[-120:]
        if len(self.display_times) >= 2:
            r["davis_display_fps"] = round((len(self.display_times) - 1) / max(self.display_times[-1] - self.display_times[0], 1e-6), 2)
        else:
            r["davis_display_fps"] = 0.0
        event_rate = r.get("current_metrics", {}).get("event_rate", 0.0)
        if not self.analyzer.monitor.active:
            self.set_card("Event Stream", "UNAVAILABLE", "bad")
        elif event_rate <= 1:
            self.set_card("Event Stream", "active\nno motion", "neutral")
        else:
            self.set_card("Event Stream", f"{event_rate:.0f}/s", "info")
        self.set_card("Stage", r.get("current_stage"), "info")
        motion = r.get("motion_class")
        self.set_card("Motion Class", motion, "good" if motion in {"STABLE","IMPACT_ONLY"} else "bad" if motion in {"GROSS_SLIP","SPIN_OR_ROTATION"} else "warn")
        self.set_card("Incipient Slip", f"{r.get('incipient_slip_score',0):.0f}", "warn" if r.get("incipient_slip_score",0) >= 45 else "good")
        self.set_card("Gross Slip", f"{r.get('gross_slip_score',0):.0f}", "bad" if r.get("gross_slip_score",0) >= 55 else "good")
        self.set_card("Rotation Risk", f"{r.get('rotation_risk_score',0):.0f}", "bad" if r.get("rotation_risk_score",0) >= 55 else "good")
        self.set_card("Vibration Risk", f"{r.get('vibration_risk_score',0):.0f}", "warn" if r.get("vibration_risk_score",0) >= 45 else "good")
        self.set_card("Stability Score", f"{r.get('stability_score',0):.0f}", "good" if r.get("stability_score",0) >= 70 else "warn" if r.get("stability_score",0) >= 40 else "bad")
        self.set_card("Decision", r.get("decision"), "good" if r.get("decision") == "READY_TO_LIFT" else "bad" if r.get("decision") in {"REGRIP","DO_NOT_LIFT"} else "warn")
        if DAVIS_DEBUG_HEATMAP and now - self.last_heatmap_time >= LIVE_HEATMAP_INTERVAL_SEC:
            self.heat.setPixmap(self.pixmap(self.update_heatmap_image()))
            self.last_heatmap_time = now
        if DAVIS_DEBUG_HEATMAP and now - self.last_timeline_time >= LIVE_TIMELINE_INTERVAL_SEC:
            self.timeline.setPixmap(self.pixmap(self.timeline_image()))
            self.last_timeline_time = now
        baseline = "Calibrated" if r.get("baseline_calibrated") else "Calibrating"
        self.info.setText(
            f"Motion Interpretation\n\nBaseline: {baseline}\n"
            f"Baseline rejected: {r.get('baseline_rejected_count',0)} noisy windows\n"
            f"Baseline event rate: {r.get('baseline_event_rate',0):.0f}/s\n"
            f"Current event rate delta: {r.get('event_rate_delta',0):.0f}/s\n"
            f"Vertical slip evidence: bias={r.get('vertical_motion_abs',0):.2f}, dy={r.get('vertical_centroid_abs',0):.1f}px, windows={r.get('downward_persistence_windows',0)}\n"
            f"ROI source: {r.get('roi_source')} ({'Full-frame event analysis.' if r.get('roi_source') == 'full_frame' else 'Configured ROI.'})\n"
            f"Disturbance type: {r.get('disturbance_type')}\n"
            f"Safety response: {r.get('visual_safety_response')}\n\n"
            f"{r.get('response_reason')}\n\n"
            f"{r.get('stage_decision_reason')}\n\n"
            "No metric depth, no metric grip width, no force or pressure sensing."
        )
        if now - self.last_save_time >= LIVE_REPORT_INTERVAL_SEC:
            r["davis_gui_draw_time_ms"] = round((time.perf_counter() - tick_start) * 1000.0, 2)
            r["davis_bottleneck"] = "analysis" if float(r.get("davis_analysis_time_ms") or 0.0) > (1000.0 / max(DAVIS_TARGET_DISPLAY_FPS, 1.0)) else "display_or_idle"
            self.analyzer.report = r
            self.analyzer.save_outputs(include_images=bool(DAVIS_DEBUG_HEATMAP and DAVIS_SAVE_DEBUG_EVERY_FRAME))
            _write_json(OUTPUT_DIR / "davis_performance_profile.json", {
                "davis_display_fps": r.get("davis_display_fps"),
                "davis_capture_fps": r.get("davis_capture_fps"),
                "davis_event_analysis_fps": r.get("davis_event_analysis_fps"),
                "davis_event_rate": (r.get("current_metrics") or {}).get("event_rate"),
                "davis_aps_frame_age_ms": r.get("davis_aps_frame_age_ms"),
                "davis_event_frame_age_ms": r.get("davis_event_frame_age_ms"),
                "davis_gui_draw_time_ms": r.get("davis_gui_draw_time_ms"),
                "davis_analysis_time_ms": r.get("davis_analysis_time_ms"),
                "davis_bottleneck": r.get("davis_bottleneck"),
            })
            self.last_save_time = now

    def run(self):
        self.window.show()
        code = self.app.exec_()
        self.analyzer.save_outputs(include_images=bool(DAVIS_DEBUG_HEATMAP))
        self.analyzer.monitor.stop()
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description="DAVIS event-only motion safety GUI.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--stage", choices=["PRE_CLOSE", "CLOSE_IMPACT", "POST_CLOSE_STABILITY", "LIFT_CHECK"], default="PRE_CLOSE")
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument("--sensitivity", choices=["conservative", "balanced", "sensitive"], default="sensitive")
    args = parser.parse_args(argv)
    if os.environ.get("DAVIS_ALLOW_STANDALONE_GUI", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        print("Standalone DAVIS desktop GUI is deprecated for normal operation.")
        print("Use the main web dashboard instead: ./scripts/run_davis_motion_demo.sh")
        print("Set DAVIS_ALLOW_STANDALONE_GUI=1 only for legacy/debug use.")
        return 2
    return DavisMotionWindow(roi_xywh=args.roi, stage=args.stage, sensitivity=args.sensitivity).run()


if __name__ == "__main__":
    raise SystemExit(main())

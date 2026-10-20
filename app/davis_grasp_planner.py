import argparse
import json
import os
import time
from pathlib import Path

QT_PLATFORM_PATH = Path("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
if QT_PLATFORM_PATH.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLATFORM_PATH)

import cv2
import numpy as np

if QT_PLATFORM_PATH.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLATFORM_PATH)

from event_postgrasp_monitor import (
    DAVIS_FRAME_SIZE,
    DAVIS_STABILITY_WINDOW_SEC,
    EventPostGraspMonitor,
    _backend_import_status,
    _clip_roi,
    _detect_davis_usb_devices,
    _json_safe,
    _write_json,
    run_simulation as run_postgrasp_simulation,
)
from davis_bottle_detector import (
    DAVIS_YOLO_MODEL_PATH,
    USE_DAVIS_YOLO_DETECTOR,
    detect_bottle_aps,
    yolo_status,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "davis_grasp_demo_report.json"
HISTORY_PATH = OUTPUT_DIR / "davis_grasp_stage_history.json"
DASHBOARD_PATH = OUTPUT_DIR / "davis_grasp_dashboard.png"
FRAME_DEBUG_PATH = OUTPUT_DIR / "davis_frame_debug.json"
PERFORMANCE_PROFILE_PATH = OUTPUT_DIR / "davis_performance_profile.json"
EVENT_FEATURE_REPORT_PATH = OUTPUT_DIR / "davis_event_feature_report.json"
DETECTION_AUDIT_PATH = OUTPUT_DIR / "davis_detection_audit.txt"
SAM2_CHECKPOINT = ROOT / "checkpoints" / "sam2_hiera_tiny.pt"
SAM2_CONFIG = os.environ.get("DAVIS_SAM2_CONFIG", "sam2_hiera_t.yaml")
DAVIS_BAYER_PATTERN = os.environ.get("DAVIS_BAYER_PATTERN", "auto").strip().upper()
DAVIS_BAYER_PATTERN_CHOICES = {"AUTO", "RG", "BG", "GR", "GB"}
if DAVIS_BAYER_PATTERN not in DAVIS_BAYER_PATTERN_CHOICES:
    DAVIS_BAYER_PATTERN = "AUTO"
DAVIS_APS_DISPLAY_MODE = os.environ.get("DAVIS_APS_DISPLAY_MODE", "auto").strip().lower()
DAVIS_APS_DISPLAY_MODES = {
    "auto",
    "raw",
    "gray",
    "rgb",
    "bgr",
    "debayer_rg",
    "debayer_bg",
    "debayer_gr",
    "debayer_gb",
}
if DAVIS_APS_DISPLAY_MODE not in DAVIS_APS_DISPLAY_MODES:
    DAVIS_APS_DISPLAY_MODE = "auto"
BAYER_CONVERSIONS = {
    "RG": cv2.COLOR_BAYER_RG2BGR,
    "BG": cv2.COLOR_BAYER_BG2BGR,
    "GR": cv2.COLOR_BAYER_GR2BGR,
    "GB": cv2.COLOR_BAYER_GB2BGR,
}

STATE_SEARCH = "DAVIS_SEARCH_BOTTLE"
STATE_ACQUIRE = "DAVIS_TARGET_ACQUISITION"
STATE_TRACK = "DAVIS_TRACKING"
STATE_VERIFY = "DAVIS_CLOSE_UP_VERIFY"
STATE_READY = "DAVIS_READY_TO_CLOSE_APPROXIMATE"
STATE_POST = "DAVIS_POST_GRASP_EVALUATION"
STATE_UNAVAILABLE = "DAVIS_SENSOR_UNAVAILABLE"
DETECTION_NOT_DETECTED = "NOT_DETECTED"
DETECTION_LOW_CONFIDENCE = "LOW_CONFIDENCE"
DETECTION_DETECTED = "DETECTED"
DETECTION_MANUAL_ROI = "MANUAL_ROI"
DETECTION_CONFIDENCE_DETECTED = 70.0
DETECTION_CONFIDENCE_LOW = 40.0
APS_GLARE_BRIGHT_THRESHOLD_U8 = 240.0
APS_GLARE_MAX_BRIGHT_FRACTION = 0.15
APS_GLARE_MAX_INTERIOR_EDGE_DENSITY = 0.045
DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK = 5
DAVIS_MAX_LOST_FRAMES = 5
DAVIS_CLAMP_BODY_FRACTION = 0.52
DAVIS_CLAMP_PATCH_HEIGHT_PX = 40
DAVIS_CLAMP_PATCH_WIDTH_PX = 12
DAVIS_EVENT_STATUS_BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
DAVIS_EVENT_STATUS_STREAM_NOT_STARTED = "STREAM_NOT_STARTED"
DAVIS_EVENT_STATUS_ACTIVE_NO_MOTION = "STREAM_ACTIVE_NO_MOTION"
DAVIS_EVENT_STATUS_ACTIVE_EVENTS = "STREAM_ACTIVE_EVENTS"
DAVIS_EVENT_STATUS_ERROR = "ERROR"
DAVIS_FAST_DISPLAY_FPS = 20.0
DAVIS_ANALYSIS_FPS = 3.0
DAVIS_EVENT_UPDATE_FPS = 10.0
DAVIS_FULL_DETECT_INTERVAL_FRAMES = 10
DAVIS_TRACKING_ROI_EXPAND_RATIO = 0.25
DAVIS_SAVE_DEBUG_EVERY_FRAME = False
DAVIS_REPORT_INTERVAL_SEC = 2.0
DAVIS_SAVE_DEBUG_ON_STATE_CHANGE = True
DAVIS_EVENT_SHORT_WINDOW_SEC = 0.20
DAVIS_EVENT_LONG_WINDOW_SEC = 1.00
DAVIS_EVENT_MIN_EVENTS_FOR_MOTION_BBOX = 12
DAVIS_EVENT_APS_OVERLAP_GOOD = 0.12
DAVIS_EVENT_APS_OVERLAP_WEAK = 0.04
DAVIS_EVENT_CONFIDENCE_RATE_SCALE = 900.0
DAVIS_PRE_CLOSE_MOTION_RATE_HOLD = 700.0
DAVIS_BACKGROUND_MOTION_PENALTY = 12.0
DAVIS_APS_EVENT_AGREEMENT_BOOST = 8.0


LIMITATIONS = [
    "DAVIS-only mode has no metric depth.",
    "DAVIS-only mode cannot directly measure grip width in mm.",
    "DAVIS-only mode cannot measure true contact force or pressure.",
    "Event-only mode cannot reliably detect a stationary bottle.",
    "Clamp/contact patches are image-space estimates only.",
]


def _empty_aps_info():
    return {
        "aps_available": False,
        "raw_aps_frame_type": None,
        "aps_frame_shape": None,
        "aps_frame_dtype": None,
        "aps_min_value": None,
        "aps_max_value": None,
        "aps_ndim": None,
        "aps_channel_count": 0,
        "aps_mode": "unavailable",
        "aps_display_mode": "unavailable",
        "aps_display_mode_requested": DAVIS_APS_DISPLAY_MODE,
        "aps_color_confirmed": False,
        "aps_format_assessment": "unavailable",
        "aps_true_color_reason": "No APS frame was available.",
        "aps_channel_behavior": None,
        "aps_bayer_pattern_config": DAVIS_BAYER_PATTERN,
        "aps_bayer_pattern_selected": None,
        "aps_debayer_scores": None,
        "aps_display_note": "APS frame unavailable from current backend.",
    }


def _to_uint8_display(image):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.dtype == np.uint8:
        out = arr.copy()
    else:
        out = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _auto_contrast_u8(image, gamma=0.85):
    arr = _to_uint8_display(image)
    if arr is None:
        return None
    if arr.ndim == 2:
        channels = [arr]
    elif arr.ndim == 3:
        channels = cv2.split(arr)
    else:
        return arr
    stretched = []
    for channel in channels:
        low, high = np.percentile(channel, [1.0, 99.5])
        if high <= low:
            norm = channel.copy()
        else:
            norm = np.clip((channel.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        if gamma and gamma != 1.0:
            lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
            norm = cv2.LUT(norm, lut)
        stretched.append(norm)
    if arr.ndim == 2:
        return stretched[0]
    return cv2.merge(stretched)


def _colorfulness_score_bgr(bgr):
    if bgr is None:
        return 0.0
    img = _to_uint8_display(bgr)
    if img is None or img.ndim != 3 or img.shape[-1] < 3:
        return 0.0
    b, g, r = cv2.split(img[:, :, :3].astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(std_root + 0.3 * mean_root)


def _gray_from_any_aps(arr):
    if arr is None:
        return None
    src = _to_uint8_display(arr)
    if src is None:
        return None
    if src.ndim == 2:
        return src
    if src.ndim == 3 and src.shape[-1] == 1:
        return src[:, :, 0]
    if src.ndim == 3 and src.shape[-1] >= 3:
        return cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2GRAY)
    return None


def _candidate_with_label(image_bgr, label, size=(220, 170)):
    if image_bgr is None:
        tile = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(tile, "unavailable", (10, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        bgr = _to_uint8_display(image_bgr)
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(bgr, size)
    header = np.full((28, size[0], 3), 22, dtype=np.uint8)
    cv2.putText(header, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)
    return cv2.vconcat([header, tile])


def _make_contact_sheet(candidates):
    tiles = [_candidate_with_label(img, label) for label, img in candidates]
    if not tiles:
        return None
    while len(tiles) % 3:
        tiles.append(_candidate_with_label(None, ""))
    rows = []
    for idx in range(0, len(tiles), 3):
        rows.append(cv2.hconcat(tiles[idx:idx + 3]))
    return cv2.vconcat(rows)


def aps_display_candidates(frame):
    arr = np.asarray(frame) if frame is not None else None
    if arr is None or arr.size == 0:
        return {}, {}
    raw_u8 = _to_uint8_display(arr)
    gray = _gray_from_any_aps(raw_u8)
    candidates = {}
    debug = {
        "debayer_source": None,
        "candidate_colorfulness_scores": {},
    }
    if raw_u8.ndim == 2:
        raw_display = cv2.cvtColor(_auto_contrast_u8(raw_u8), cv2.COLOR_GRAY2BGR)
    elif raw_u8.ndim == 3 and raw_u8.shape[-1] == 1:
        raw_display = cv2.cvtColor(_auto_contrast_u8(raw_u8[:, :, 0]), cv2.COLOR_GRAY2BGR)
    elif raw_u8.ndim == 3 and raw_u8.shape[-1] >= 3:
        raw_display = _auto_contrast_u8(raw_u8[:, :, :3])
    else:
        raw_display = None
    candidates["raw"] = raw_display
    if gray is not None:
        gray_display = cv2.cvtColor(_auto_contrast_u8(gray), cv2.COLOR_GRAY2BGR)
        candidates["gray"] = gray_display
    if raw_u8.ndim == 3 and raw_u8.shape[-1] >= 3:
        rgb_assumed = cv2.cvtColor(_auto_contrast_u8(raw_u8[:, :, :3]), cv2.COLOR_RGB2BGR)
        bgr_assumed = _auto_contrast_u8(raw_u8[:, :, :3])
        candidates["rgb"] = rgb_assumed
        candidates["bgr"] = bgr_assumed
    elif gray is not None:
        gray_bgr = cv2.cvtColor(_auto_contrast_u8(gray), cv2.COLOR_GRAY2BGR)
        candidates["rgb"] = gray_bgr
        candidates["bgr"] = gray_bgr
    if raw_u8.ndim == 2:
        bayer_source = raw_u8
        debug["debayer_source"] = "single_channel_raw"
    elif raw_u8.ndim == 3 and raw_u8.shape[-1] >= 1:
        bayer_source = raw_u8[:, :, 0]
        debug["debayer_source"] = "channel_0_from_multichannel_frame"
    else:
        bayer_source = gray
        debug["debayer_source"] = "grayscale_conversion"
    if bayer_source is not None:
        bayer_u8 = _to_uint8_display(bayer_source)
        for pattern, code in BAYER_CONVERSIONS.items():
            key = f"debayer_{pattern.lower()}"
            try:
                debayered = cv2.cvtColor(bayer_u8, code)
                debayered = _auto_contrast_u8(debayered)
                candidates[key] = debayered
            except cv2.error:
                candidates[key] = None
    for key, img in candidates.items():
        debug["candidate_colorfulness_scores"][key] = round(_colorfulness_score_bgr(img), 3) if img is not None else None
    ordered = [
        ("raw display", candidates.get("raw")),
        ("gray", candidates.get("gray")),
        ("RGB assumed", candidates.get("rgb")),
        ("BGR assumed", candidates.get("bgr")),
        ("debayer RG", candidates.get("debayer_rg")),
        ("debayer BG", candidates.get("debayer_bg")),
        ("debayer GR", candidates.get("debayer_gr")),
        ("debayer GB", candidates.get("debayer_gb")),
    ]
    candidates["contact_sheet"] = _make_contact_sheet(ordered)
    return candidates, debug


def _channel_behavior(arr):
    if arr.ndim != 3 or arr.shape[-1] < 2:
        return None
    sample = arr[:, :, :min(arr.shape[-1], 4)].astype(np.float32)
    channel_means = [float(np.mean(sample[:, :, idx])) for idx in range(sample.shape[-1])]
    channel_stds = [float(np.std(sample[:, :, idx])) for idx in range(sample.shape[-1])]
    pairwise = []
    for a in range(sample.shape[-1]):
        for b in range(a + 1, sample.shape[-1]):
            pairwise.append(float(np.mean(np.abs(sample[:, :, a] - sample[:, :, b]))))
    max_abs_diff = max(pairwise) if pairwise else 0.0
    data_range = max(float(np.max(sample) - np.min(sample)), 1.0)
    normalized_max_abs_diff = max_abs_diff / data_range
    return {
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "pairwise_mean_abs_diff": pairwise,
        "max_pairwise_mean_abs_diff": float(max_abs_diff),
        "normalized_max_pairwise_mean_abs_diff": float(normalized_max_abs_diff),
        "channels_nearly_identical": bool(max_abs_diff < 2.0 or normalized_max_abs_diff < 0.01),
    }


def _debayer_candidates(gray):
    gray_u8 = _to_uint8_display(gray)
    if gray_u8 is None or gray_u8.ndim != 2:
        return None, None, []
    candidates = []
    for pattern, code in BAYER_CONVERSIONS.items():
        try:
            bgr = cv2.cvtColor(gray_u8, code)
        except cv2.error:
            continue
        enhanced = _auto_contrast_u8(bgr)
        candidates.append({
            "pattern": pattern,
            "score": _colorfulness_score_bgr(enhanced),
            "bgr": enhanced,
        })
    if not candidates:
        return None, None, []
    if DAVIS_BAYER_PATTERN != "AUTO":
        selected = next((item for item in candidates if item["pattern"] == DAVIS_BAYER_PATTERN), candidates[0])
    else:
        selected = max(candidates, key=lambda item: item["score"])
    score_report = [{"pattern": item["pattern"], "colorfulness_score": round(item["score"], 3)} for item in candidates]
    return selected["pattern"], selected["bgr"], score_report


def _debayer_comparison_image(candidates):
    if not candidates:
        return None
    tiles = []
    for item in candidates:
        pattern = item.get("pattern")
        bgr = item.get("bgr")
        if bgr is None:
            continue
        tile = bgr.copy()
        cv2.putText(tile, f"Bayer {pattern}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return None
    min_h = min(tile.shape[0] for tile in tiles)
    resized = [cv2.resize(tile, (int(tile.shape[1] * min_h / tile.shape[0]), min_h)) for tile in tiles]
    return cv2.hconcat(resized)


def aps_channel_info(frame):
    if frame is None:
        return _empty_aps_info()
    arr = np.asarray(frame)
    if arr.size == 0:
        return _empty_aps_info()
    if arr.ndim == 2:
        channels = 1
        mode = "grayscale"
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        channels = 1
        mode = "grayscale"
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        channels = 3
        behavior = _channel_behavior(arr)
        mode = "grayscale_3ch" if behavior and behavior["channels_nearly_identical"] else "color_candidate"
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        channels = 4
        mode = "color_alpha_or_unknown"
    else:
        channels = int(arr.shape[-1]) if arr.ndim >= 3 else 1
        mode = "unknown"
    behavior = _channel_behavior(arr) if arr.ndim == 3 else None
    return {
        "aps_available": True,
        "raw_aps_frame_type": str(type(frame)),
        "aps_frame_shape": list(arr.shape),
        "aps_frame_dtype": str(arr.dtype),
        "aps_min_value": float(np.min(arr)),
        "aps_max_value": float(np.max(arr)),
        "aps_ndim": int(arr.ndim),
        "aps_channel_count": channels,
        "aps_mode": mode,
        "aps_display_mode": mode,
        "aps_display_mode_requested": DAVIS_APS_DISPLAY_MODE,
        "aps_color_confirmed": False,
        "aps_format_assessment": mode,
        "aps_true_color_reason": "True color is not verified from frame shape alone.",
        "aps_channel_behavior": behavior,
        "aps_bayer_pattern_config": DAVIS_BAYER_PATTERN,
        "aps_bayer_pattern_selected": None,
        "aps_debayer_scores": None,
        "aps_display_note": "3-channel APS differs, but RGB/BGR order is not independently verified." if mode == "color_candidate" else None,
    }


def analyze_aps_frame(frame, display_mode=None):
    requested_display_mode = (display_mode or DAVIS_APS_DISPLAY_MODE or "auto").strip().lower()
    if requested_display_mode not in DAVIS_APS_DISPLAY_MODES:
        requested_display_mode = "auto"
    if frame is None:
        info = _empty_aps_info()
        info["aps_display_mode_requested"] = requested_display_mode
        return None, None, info, {}
    arr = np.asarray(frame)
    if arr.size == 0:
        info = _empty_aps_info()
        info["aps_display_mode_requested"] = requested_display_mode
        return None, None, info, {}
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in {1, 3}:
        arr = arr[-1]
    info = aps_channel_info(arr)
    info["aps_display_mode_requested"] = requested_display_mode
    candidates, candidate_debug = aps_display_candidates(arr)
    debug_images = {"candidates": candidates, "candidate_debug": candidate_debug}
    if arr.ndim == 3 and arr.shape[-1] == 1:
        gray_raw = arr[:, :, 0]
        gray = _auto_contrast_u8(gray_raw)
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        info["aps_mode"] = "grayscale"
        info["aps_display_mode"] = "grayscale"
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        behavior = _channel_behavior(arr)
        if behavior and behavior["channels_nearly_identical"]:
            gray_raw = arr[:, :, 0]
            gray = _auto_contrast_u8(gray_raw)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            info["aps_mode"] = "grayscale_3ch"
            info["aps_display_mode"] = "grayscale"
            info["aps_display_note"] = "Three channels are nearly identical; displaying as grayscale, not color."
            info["aps_format_assessment"] = "grayscale_3ch"
            info["aps_true_color_reason"] = "Three APS channels are nearly identical."
        else:
            # Use the 3-channel frame for display, but label it as a color
            # candidate because the backend does not prove RGB/BGR semantics.
            bgr_candidate = cv2.cvtColor(_to_uint8_display(arr), cv2.COLOR_RGB2BGR)
            display = _auto_contrast_u8(bgr_candidate)
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            info["aps_mode"] = "color_candidate"
            info["aps_display_mode"] = "color_candidate_display"
            info["aps_display_note"] = "3-channel APS frame has different channels; displayed as color candidate, not confirmed true RGB."
            info["aps_format_assessment"] = "color_candidate_not_verified"
            info["aps_true_color_reason"] = "Channels differ, but backend channel order/color filter interpretation is not verified."
            debug_images["color_bgr"] = display
        info["aps_channel_behavior"] = behavior
        info["aps_colorfulness_score"] = _colorfulness_score_bgr(display)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        rgb = _to_uint8_display(arr[:, :, :3])
        display = _auto_contrast_u8(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        info["aps_mode"] = "color_alpha_or_unknown"
        info["aps_display_mode"] = "color_alpha_or_unknown"
        info["aps_display_note"] = "4-channel APS frame; first three channels displayed, color not confirmed."
        info["aps_format_assessment"] = "color_alpha_or_unknown"
        info["aps_true_color_reason"] = "4-channel APS frame needs backend-specific interpretation."
        debug_images["color_bgr"] = display
    elif arr.ndim == 2:
        gray_raw = arr
        gray = _auto_contrast_u8(gray_raw)
        pattern, debayered, scores = _debayer_candidates(gray_raw)
        info["aps_debayer_scores"] = scores
        info["aps_bayer_pattern_selected"] = pattern
        if debayered is not None and (DAVIS_BAYER_PATTERN != "AUTO" or _colorfulness_score_bgr(debayered) >= 12.0):
            display = debayered
            info["aps_mode"] = "raw_bayer"
            info["aps_display_mode"] = "debayered_color"
            info["aps_display_note"] = f"Single-channel APS displayed after Bayer {pattern} demosaicing."
            info["aps_format_assessment"] = "raw_bayer_candidate"
            info["aps_true_color_reason"] = "Single-channel APS can be debayered for inspection, but true Bayer pattern must be selected visually or calibrated."
            debug_images["color_bgr"] = debayered
        else:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            info["aps_mode"] = "grayscale"
            info["aps_display_mode"] = "grayscale"
            info["aps_display_note"] = "Single-channel APS displayed as normalized grayscale."
            info["aps_format_assessment"] = "grayscale"
            info["aps_true_color_reason"] = "Single-channel APS is treated as grayscale unless a Bayer pattern is explicitly selected."
        debayer_previews = []
        for item in scores or []:
            p = item["pattern"]
            try:
                bgr = _auto_contrast_u8(cv2.cvtColor(_to_uint8_display(gray_raw), BAYER_CONVERSIONS[p]))
                debayer_previews.append({"pattern": p, "bgr": bgr})
            except cv2.error:
                pass
        comparison = _debayer_comparison_image(debayer_previews)
        if comparison is not None:
            debug_images["debayer_comparison_bgr"] = comparison
    else:
        info["aps_mode"] = "unknown"
        info["aps_display_mode"] = "unknown"
        info["aps_display_note"] = "Unsupported APS frame layout."
        return None, None, info, debug_images
    gray = _to_uint8_display(gray)
    if requested_display_mode != "auto":
        selected = candidates.get(requested_display_mode)
        if selected is not None:
            display = selected
            info["aps_display_mode"] = requested_display_mode
            if requested_display_mode == "gray":
                info["aps_display_note"] = "Manual display override: normalized grayscale."
            elif requested_display_mode in {"rgb", "bgr"}:
                info["aps_display_note"] = f"Manual display override: {requested_display_mode.upper()} interpretation. Color is not automatically verified."
                info["aps_true_color_reason"] = f"Manual {requested_display_mode.upper()} interpretation selected; color still requires visual/calibrated verification."
            elif requested_display_mode.startswith("debayer_"):
                info["aps_display_note"] = f"Manual display override: {requested_display_mode.replace('_', ' ').upper()} from {candidate_debug.get('debayer_source')}."
                info["aps_mode"] = "raw_bayer"
                info["aps_bayer_pattern_selected"] = requested_display_mode.split("_", 1)[1].upper()
                info["aps_format_assessment"] = "manual_raw_bayer_candidate"
                info["aps_true_color_reason"] = "Manual debayer selected; true color depends on visually choosing the correct Bayer pattern."
            elif requested_display_mode == "raw":
                info["aps_display_note"] = "Manual display override: raw channel display."
        else:
            info["aps_display_note"] = f"Requested display mode {requested_display_mode} was unavailable; using auto display."
    info["aps_display_candidate_scores"] = candidate_debug.get("candidate_colorfulness_scores")
    info["aps_debayer_source"] = candidate_debug.get("debayer_source")
    display = _to_uint8_display(display)
    debug_images["display_bgr"] = display
    if gray is not None:
        debug_images["display_gray"] = gray
    return gray, display, info, debug_images


def normalize_aps_frame(frame):
    gray, display, _info, _debug_images = analyze_aps_frame(frame)
    return gray, display


def fast_aps_display_frame(frame, display_mode=None):
    if frame is None:
        return None, None
    arr = np.asarray(frame)
    if arr.size == 0:
        return None, None
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in {1, 3, 4}:
        arr = arr[-1]
    arr_u8 = _to_uint8_display(arr)
    if arr_u8 is None:
        return None, None
    requested = (display_mode or DAVIS_APS_DISPLAY_MODE or "auto").strip().lower()
    if arr_u8.ndim == 2:
        gray = _auto_contrast_u8(arr_u8, gamma=0.85)
        return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if arr_u8.ndim == 3 and arr_u8.shape[-1] == 1:
        gray = _auto_contrast_u8(arr_u8[:, :, 0], gamma=0.85)
        return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if arr_u8.ndim == 3 and arr_u8.shape[-1] >= 3:
        src = arr_u8[:, :, :3]
        if requested == "rgb":
            bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        else:
            bgr = src.copy()
        bgr = _auto_contrast_u8(bgr, gamma=0.92)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray, bgr
    gray = _gray_from_any_aps(arr_u8)
    if gray is None:
        return None, None
    gray = _auto_contrast_u8(gray, gamma=0.85)
    return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def aps_mode_label(report):
    mode = report.get("aps_mode", "unknown")
    pattern = report.get("aps_bayer_pattern_selected")
    if mode == "grayscale":
        return "APS Mode: Grayscale"
    if mode == "grayscale_3ch":
        return "APS Mode: Grayscale 3-channel"
    if mode == "raw_bayer":
        suffix = f", debayered as {pattern}" if pattern else ""
        return f"APS Mode: Raw Bayer{suffix}"
    if mode == "color_candidate":
        return "APS Mode: Color candidate (not verified RGB)"
    if mode == "true_color":
        return "APS Mode: True Color"
    if mode == "color_alpha_or_unknown":
        return "APS Mode: 4-channel color/alpha or unknown"
    if mode == "unavailable":
        return "APS Mode: Unavailable"
    return "APS Mode: Unknown"


def raw_aps_png(raw_aps):
    arr = np.asarray(raw_aps)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return _auto_contrast_u8(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return _auto_contrast_u8(arr[:, :, 0])
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return _auto_contrast_u8(arr[:, :, :3])
    return _auto_contrast_u8(arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim > 2 else arr)


def save_aps_visual_debug(raw_frame, display_bgr, debug_images):
    OUTPUT_DIR.mkdir(exist_ok=True)
    if raw_frame is not None:
        np.save(OUTPUT_DIR / "davis_latest_aps_raw.npy", np.asarray(raw_frame))
    candidates = (debug_images or {}).get("candidates", {})
    file_map = {
        "raw": "davis_aps_raw_display.png",
        "gray": "davis_aps_gray.png",
        "rgb": "davis_aps_rgb_assumed.png",
        "bgr": "davis_aps_bgr_assumed.png",
        "debayer_rg": "davis_aps_debayer_RG.png",
        "debayer_bg": "davis_aps_debayer_BG.png",
        "debayer_gr": "davis_aps_debayer_GR.png",
        "debayer_gb": "davis_aps_debayer_GB.png",
        "contact_sheet": "davis_aps_contact_sheet.png",
    }
    for key, filename in file_map.items():
        image = candidates.get(key)
        if image is not None:
            cv2.imwrite(str(OUTPUT_DIR / filename), image)
    if raw_frame is not None:
        raw_png = raw_aps_png(raw_frame)
        if raw_png is not None:
            cv2.imwrite(str(OUTPUT_DIR / "davis_latest_aps_raw.png"), raw_png)
    if display_bgr is not None:
        cv2.imwrite(str(OUTPUT_DIR / "davis_latest_aps.png"), display_bgr)
        cv2.imwrite(str(OUTPUT_DIR / "davis_latest_aps_display.png"), display_bgr)
    if candidates.get("contact_sheet") is not None:
        cv2.imwrite(str(OUTPUT_DIR / "davis_debayer_comparison.png"), candidates["contact_sheet"])
    if debug_images and debug_images.get("color_bgr") is not None:
        cv2.imwrite(str(OUTPUT_DIR / "davis_latest_aps_color.png"), debug_images["color_bgr"])


def save_detection_debug(debug_images):
    if not debug_images:
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    file_map = {
        "detection_input": "davis_aps_detection_input.png",
        "detection_input_gray": "davis_aps_detection_gray.png",
        "enhanced": "davis_aps_enhanced.png",
        "edges": "davis_aps_edges.png",
        "threshold": "davis_aps_threshold.png",
        "dark_body": "davis_aps_dark_body.png",
        "contours_debug": "davis_aps_contours_debug.png",
        "candidates_debug": "davis_detection_candidates.png",
    }
    for key, filename in file_map.items():
        image = debug_images.get(key)
        if image is not None:
            cv2.imwrite(str(OUTPUT_DIR / filename), image)
    if debug_images.get("sam2_mask") is not None:
        cv2.imwrite(str(OUTPUT_DIR / "davis_sam2_mask.png"), debug_images["sam2_mask"])


def event_heatmap(events, roi_xywh=None, frame_size=DAVIS_FRAME_SIZE, centroid=None):
    width, height = frame_size
    heat = np.zeros((height, width), dtype=np.float32)
    if events is not None:
        arr = np.asarray(events)
        if arr.ndim == 2 and arr.shape[1] >= 4:
            xs = arr[:, 1].astype(np.int32)
            ys = arr[:, 2].astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            np.add.at(heat, (ys[valid], xs[valid]), 1.0)
    heat_u8 = cv2.normalize(np.log1p(heat), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_MAGMA)
    x, y, w, h = _clip_roi(roi_xywh)
    cv2.rectangle(color, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 1)
    cv2.line(color, (x + w // 2, y), (x + w // 2, y + h - 1), (0, 180, 255), 1)
    cv2.line(color, (x, y + h // 2), (x + w - 1, y + h // 2), (0, 180, 255), 1)
    if centroid and centroid[0] is not None and centroid[1] is not None:
        cx = int(round(float(centroid[0])))
        cy = int(round(float(centroid[1])))
        if 0 <= cx < width and 0 <= cy < height:
            cv2.drawMarker(color, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
    return color


def _point_in_bbox(point, bbox):
    if point is None or bbox is None:
        return False
    px, py = [float(v) for v in point]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return bool(x1 <= px <= x2 and y1 <= py <= y2)


def _event_time_filtered(events, window_sec):
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    latest_t = float(np.max(arr[:, 0]))
    cutoff = latest_t - max(float(window_sec), 0.001)
    return arr[arr[:, 0] >= cutoff]


def _event_bbox_and_centroid(events, frame_size=DAVIS_FRAME_SIZE, min_events=DAVIS_EVENT_MIN_EVENTS_FOR_MOTION_BBOX):
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] < int(min_events):
        return None, None
    width, height = frame_size
    xs = np.clip(arr[:, 1].astype(np.float32), 0, width - 1)
    ys = np.clip(arr[:, 2].astype(np.float32), 0, height - 1)
    x1, x2 = np.percentile(xs, [5, 95])
    y1, y2 = np.percentile(ys, [5, 95])
    bbox = [
        int(np.clip(round(x1), 0, width - 1)),
        int(np.clip(round(y1), 0, height - 1)),
        int(np.clip(round(x2), 1, width)),
        int(np.clip(round(y2), 1, height)),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        bbox = None
    centroid = [round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1)]
    return bbox, centroid


def compute_event_motion_features(events, aps_bbox=None, roi_xywh=None, frame_size=DAVIS_FRAME_SIZE):
    width, height = frame_size
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
        return {
            "event_count": 0,
            "event_rate": 0.0,
            "event_heatmap": "davis_event_heatmap.png",
            "event_roi": _clip_roi(roi_xywh),
            "event_centroid": None,
            "event_motion_bbox": None,
            "event_edge_activity": 0.0,
            "event_vertical_bias": 0.0,
            "event_horizontal_bias": 0.0,
            "event_left_right_asymmetry": 0.0,
            "event_top_bottom_asymmetry": 0.0,
            "event_stability_score": 100.0,
            "event_motion_confidence": 0.0,
            "event_motion_class": "STABLE",
            "event_activity_score": 0.0,
            "event_roi_overlap_with_aps_bbox": 0.0,
            "event_centroid_inside_aps_bbox": False,
            "event_motion_near_target": False,
            "event_status_interpretation": "STREAM_ACTIVE_NO_MOTION",
            "motion_stability_score": 100.0,
            "event_heatmap_stats": {"nonzero_pixels": 0, "max_count": 0.0, "mean_count": 0.0},
        }

    long_events = _event_time_filtered(arr, DAVIS_EVENT_LONG_WINDOW_SEC)
    short_events = _event_time_filtered(arr, DAVIS_EVENT_SHORT_WINDOW_SEC)
    duration = max(DAVIS_EVENT_LONG_WINDOW_SEC, 1e-6)
    event_count = int(long_events.shape[0])
    event_rate = float(event_count / duration)
    bbox, centroid = _event_bbox_and_centroid(short_events if short_events.shape[0] else long_events, frame_size=frame_size)
    xs = long_events[:, 1].astype(np.float32) if event_count else np.array([], dtype=np.float32)
    ys = long_events[:, 2].astype(np.float32) if event_count else np.array([], dtype=np.float32)
    if event_count:
        roi_x, roi_y, roi_w, roi_h = _clip_roi(roi_xywh)
        left_count = int(np.count_nonzero(xs < roi_x + roi_w / 2.0))
        right_count = int(np.count_nonzero(xs >= roi_x + roi_w / 2.0))
        upper_count = int(np.count_nonzero(ys < roi_y + roi_h / 2.0))
        lower_count = int(np.count_nonzero(ys >= roi_y + roi_h / 2.0))
        denom = max(event_count, 1)
        lr_asym = abs(left_count - right_count) / denom
        tb_asym = abs(upper_count - lower_count) / denom
        vertical_bias = (lower_count - upper_count) / denom
        horizontal_bias = (right_count - left_count) / denom
        spread_x = float(np.std(xs) / max(width, 1))
        spread_y = float(np.std(ys) / max(height, 1))
        heat = np.zeros((height, width), dtype=np.float32)
        xi = np.clip(xs.astype(np.int32), 0, width - 1)
        yi = np.clip(ys.astype(np.int32), 0, height - 1)
        np.add.at(heat, (yi, xi), 1.0)
        nonzero = int(np.count_nonzero(heat))
        heat_stats = {
            "nonzero_pixels": nonzero,
            "max_count": round(float(np.max(heat)), 2) if heat.size else 0.0,
            "mean_count": round(float(np.mean(heat[heat > 0])), 2) if nonzero else 0.0,
        }
    else:
        lr_asym = tb_asym = vertical_bias = horizontal_bias = spread_x = spread_y = 0.0
        heat_stats = {"nonzero_pixels": 0, "max_count": 0.0, "mean_count": 0.0}
    event_density = event_count / max(float(width * height), 1.0)
    event_edge_activity = float(np.clip(100.0 * event_density / 0.010, 0.0, 100.0))
    event_motion_confidence = float(np.clip((event_rate / DAVIS_EVENT_CONFIDENCE_RATE_SCALE) * 100.0, 0.0, 100.0))
    overlap = _bbox_iou(bbox, aps_bbox) if bbox and aps_bbox else 0.0
    centroid_inside = _point_in_bbox(centroid, aps_bbox)
    near_target = bool(overlap >= DAVIS_EVENT_APS_OVERLAP_WEAK or centroid_inside)
    if event_rate < 30.0:
        motion_class = "STABLE"
    elif abs(vertical_bias) > 0.28:
        motion_class = "VERTICAL_MOTION"
    elif lr_asym > 0.34 or abs(horizontal_bias) > 0.28:
        motion_class = "SIDE_ASYMMETRIC_MOTION"
    elif event_edge_activity > 35.0 and (spread_x + spread_y) > 0.20:
        motion_class = "BROAD_MOTION"
    else:
        motion_class = "LOCAL_MOTION"
    motion_stability = float(np.clip(100.0 - max(event_motion_confidence, event_edge_activity * 0.65), 0.0, 100.0))
    return {
        "event_count": event_count,
        "event_rate": round(event_rate, 1),
        "event_heatmap": "davis_event_heatmap.png",
        "event_roi": _clip_roi(roi_xywh),
        "event_centroid": centroid,
        "event_motion_bbox": bbox,
        "event_edge_activity": round(event_edge_activity, 1),
        "event_vertical_bias": round(float(vertical_bias), 3),
        "event_horizontal_bias": round(float(horizontal_bias), 3),
        "event_left_right_asymmetry": round(float(lr_asym), 3),
        "event_top_bottom_asymmetry": round(float(tb_asym), 3),
        "event_stability_score": round(motion_stability, 1),
        "event_motion_confidence": round(event_motion_confidence, 1),
        "event_motion_class": motion_class,
        "event_activity_score": round(float(np.clip(event_motion_confidence * 0.7 + event_edge_activity * 0.3, 0.0, 100.0)), 1),
        "event_roi_overlap_with_aps_bbox": round(float(overlap), 3),
        "event_centroid_inside_aps_bbox": bool(centroid_inside),
        "event_motion_near_target": bool(near_target),
        "event_status_interpretation": "STREAM_ACTIVE_EVENTS" if event_rate > 1.0 else "STREAM_ACTIVE_NO_MOTION",
        "motion_stability_score": round(motion_stability, 1),
        "event_spread_score": round(float(np.clip((spread_x + spread_y) * 120.0, 0.0, 100.0)), 1),
        "event_heatmap_stats": heat_stats,
    }


def fuse_davis_aps_events(detection, clamp, metrics, event_features):
    aps_conf = float(detection.get("davis_detection_confidence", 0.0) or 0.0)
    event_conf = float(event_features.get("event_motion_confidence", 0.0) or 0.0)
    event_rate = float(event_features.get("event_rate", 0.0) or 0.0)
    overlap = float(event_features.get("event_roi_overlap_with_aps_bbox", 0.0) or 0.0)
    centroid_inside = bool(event_features.get("event_centroid_inside_aps_bbox"))
    detected = detection.get("davis_detection_status") in {DETECTION_DETECTED, DETECTION_MANUAL_ROI}
    source = detection.get("davis_detection_source", "none")
    agreement = "no_motion_no_event_penalty"
    explanation = "Stationary scene or no DVS motion; APS confidence is used without event penalty."
    combined = aps_conf
    tracking = aps_conf
    pre_close_motion_instability = False
    if event_rate > 1.0:
        if detected and (overlap >= DAVIS_EVENT_APS_OVERLAP_GOOD or centroid_inside):
            agreement = "aps_plus_events_agree"
            source = "aps_plus_events"
            boost = min(DAVIS_APS_EVENT_AGREEMENT_BOOST, event_conf * 0.12)
            combined = min(100.0, aps_conf + boost)
            tracking = min(100.0, 0.72 * aps_conf + 0.28 * event_conf + boost)
            explanation = "DVS motion is near the APS bottle target, supporting tracking confidence."
        elif detected:
            agreement = "event_motion_outside_aps_target"
            combined = max(0.0, aps_conf - DAVIS_BACKGROUND_MOTION_PENALTY)
            tracking = max(0.0, aps_conf - max(DAVIS_BACKGROUND_MOTION_PENALTY, event_conf * 0.18))
            explanation = "DVS motion does not overlap the APS bottle target; treating it as background/camera disturbance."
            pre_close_motion_instability = True
        elif event_conf >= 35.0:
            agreement = "event_motion_without_aps_confirmation"
            source = "events_motion_roi"
            combined = min(69.0, max(aps_conf, event_conf * 0.65))
            tracking = combined
            explanation = "Events show a moving object, but APS does not confirm the known bottle."
            pre_close_motion_instability = True
    if event_rate >= DAVIS_PRE_CLOSE_MOTION_RATE_HOLD:
        pre_close_motion_instability = True
    clamp_ready = bool(clamp.get("davis_clamp_estimate_ready"))
    event_support = 0.0
    if clamp_ready and event_rate > 1.0 and (overlap >= DAVIS_EVENT_APS_OVERLAP_WEAK or centroid_inside):
        event_support = float(np.clip(30.0 + 70.0 * max(overlap, 0.0), 0.0, 100.0))
    elif event_rate <= 1.0:
        event_support = None
    return {
        "aps_confidence": round(aps_conf, 1),
        "event_confidence": round(event_conf, 1),
        "aps_event_overlap_score": round(overlap, 3),
        "davis_combined_confidence": round(float(np.clip(combined, 0.0, 100.0)), 1),
        "davis_tracking_confidence": round(float(np.clip(tracking, 0.0, 100.0)), 1),
        "davis_detection_source": source,
        "aps_event_agreement": agreement,
        "event_assisted_tracking": {
            "previous_bbox": detection.get("davis_bbox_xyxy"),
            "current_aps_bbox": detection.get("davis_bbox_xyxy"),
            "current_event_motion_bbox": event_features.get("event_motion_bbox"),
            "overlap_score": round(overlap, 3),
            "centroid_motion_score": round(100.0 if centroid_inside else min(100.0, overlap * 100.0 / max(DAVIS_EVENT_APS_OVERLAP_GOOD, 1e-6)), 1),
            "tracking_confidence": round(float(np.clip(tracking, 0.0, 100.0)), 1),
            "explanation": explanation,
        },
        "event_support_score": None if event_support is None else round(event_support, 1),
        "left_patch_event_support_score": None if event_support is None else round(event_support, 1),
        "right_patch_event_support_score": None if event_support is None else round(event_support, 1),
        "event_motion_near_patch": bool(event_support and event_support > 0.0),
        "event_instability_near_patch": bool(pre_close_motion_instability and event_features.get("event_motion_near_target")),
        "pre_close_motion_instability": bool(pre_close_motion_instability),
        "post_close_slip_risk": metrics.get("slip_risk_score"),
        "post_close_rotation_risk": metrics.get("rotation_risk_score"),
        "post_close_vibration_risk": metrics.get("vibration_risk_score"),
        "fusion_explanation": explanation,
    }


def build_davis_perception_result(aps_info, detection, clamp, metrics, event_features, fusion, decision):
    aps_bbox = detection.get("davis_bbox_xyxy")
    return {
        "aps_available": bool(aps_info.get("aps_available")),
        "aps_mode": aps_info.get("aps_mode"),
        "aps_frame_shape": aps_info.get("aps_frame_shape"),
        "aps_display_mode": aps_info.get("aps_display_mode"),
        "aps_detection_status": detection.get("davis_detection_status"),
        "aps_detection_confidence": detection.get("davis_detection_confidence"),
        "aps_bbox_xyxy": aps_bbox,
        "aps_contour_available": bool(detection.get("davis_contour_available")),
        "aps_centerline": detection.get("davis_centerline_px"),
        "aps_candidate_scores": detection.get("best_candidate_scores"),
        "event_stream_status": event_features.get("event_status_interpretation"),
        "event_count": event_features.get("event_count"),
        "event_rate": event_features.get("event_rate"),
        "event_heatmap": event_features.get("event_heatmap"),
        "event_roi": event_features.get("event_roi"),
        "event_centroid": event_features.get("event_centroid"),
        "event_motion_bbox": event_features.get("event_motion_bbox"),
        "event_edge_activity": event_features.get("event_edge_activity"),
        "event_vertical_bias": event_features.get("event_vertical_bias"),
        "event_horizontal_bias": event_features.get("event_horizontal_bias"),
        "event_left_right_asymmetry": event_features.get("event_left_right_asymmetry"),
        "event_top_bottom_asymmetry": event_features.get("event_top_bottom_asymmetry"),
        "event_stability_score": event_features.get("event_stability_score"),
        "event_motion_confidence": event_features.get("event_motion_confidence"),
        "davis_bottle_detected": bool(detection.get("davis_bottle_detected")),
        "davis_detection_source": fusion.get("davis_detection_source", detection.get("davis_detection_source")),
        "davis_combined_confidence": fusion.get("davis_combined_confidence"),
        "davis_tracking_confidence": fusion.get("davis_tracking_confidence"),
        "davis_motion_class": metrics.get("motion_class"),
        "davis_decision": decision,
        "davis_limitations": list(LIMITATIONS),
    }


def draw_event_feature_overlay(image, event_features):
    if image is None or not event_features:
        return image
    bbox = event_features.get("event_motion_bbox")
    if bbox:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(image, "event motion bbox", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    centroid = event_features.get("event_centroid")
    if centroid:
        cx, cy = [int(round(v)) for v in centroid]
        cv2.drawMarker(image, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
    return image


def event_status_visual(report, frame_size=DAVIS_FRAME_SIZE):
    width, height = frame_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (16, 24, 39)
    status = report.get("event_status", DAVIS_EVENT_STATUS_STREAM_NOT_STARTED)
    if status == DAVIS_EVENT_STATUS_ACTIVE_NO_MOTION:
        title = "No motion events currently"
        detail = "Stream is active. Move the object to produce DVS events."
        color = (0, 180, 255)
    elif status == DAVIS_EVENT_STATUS_STREAM_NOT_STARTED:
        title = "Event stream not started"
        detail = "DAVIS backend opened no active event stream yet."
        color = (148, 163, 184)
    elif status == DAVIS_EVENT_STATUS_BACKEND_UNAVAILABLE:
        title = "DAVIS backend unavailable"
        detail = str(report.get("davis_unavailable_reason") or "Check USB permissions/backend.")
        color = (60, 80, 220)
    else:
        title = str(status)
        detail = str(report.get("event_stream_status") or "")
        color = (148, 163, 184)
    cv2.putText(image, title, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    words = detail.split()
    lines = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) > 36:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    for idx, text in enumerate(lines[:4]):
        cv2.putText(image, text, (18, 118 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1)
    rate = float(report.get("event_rate", 0.0) or 0.0)
    cv2.putText(image, f"{rate:.0f} events/s", (18, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (203, 213, 225), 1)
    return image


def status_placeholder_image(title, detail="", frame_size=DAVIS_FRAME_SIZE):
    width, height = frame_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (248, 250, 252)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (148, 163, 184), 1)
    cv2.putText(image, str(title)[:34], (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 41, 59), 2)
    words = str(detail or "").split()
    line = ""
    lines = []
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) > 40:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    for idx, text in enumerate(lines[:5]):
        cv2.putText(image, text, (16, 120 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (71, 85, 105), 1)
    return image


def detection_status_label(report):
    status = report.get("davis_detection_status", DETECTION_NOT_DETECTED)
    if status == DETECTION_DETECTED:
        return "DETECTED"
    if status == DETECTION_LOW_CONFIDENCE:
        return "LOW_CONFIDENCE"
    if status == DETECTION_MANUAL_ROI:
        return "MANUAL_ROI"
    return "NOT_DETECTED"


def event_status_label(report):
    status = report.get("event_status")
    if status == DAVIS_EVENT_STATUS_ACTIVE_EVENTS:
        return "STREAM_ACTIVE_EVENTS"
    if status == DAVIS_EVENT_STATUS_ACTIVE_NO_MOTION:
        return "STREAM_ACTIVE_NO_MOTION"
    if status == DAVIS_EVENT_STATUS_STREAM_NOT_STARTED:
        return "STREAM_NOT_STARTED"
    if status == DAVIS_EVENT_STATUS_BACKEND_UNAVAILABLE:
        return "BACKEND_UNAVAILABLE"
    if status == DAVIS_EVENT_STATUS_ERROR:
        return "ERROR"
    return str(report.get("event_stream_status", "UNKNOWN"))


class DavisSAM2Segmenter:
    def __init__(self):
        self.available = False
        self.reason = None
        self.predictor = None
        self.load_attempted = False

    def ensure_loaded(self):
        if self.load_attempted:
            return self.available
        self.load_attempted = True
        if not SAM2_CHECKPOINT.exists():
            self.reason = f"SAM2 checkpoint missing: {SAM2_CHECKPOINT}"
            return False
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            self.reason = f"SAM2 Python package unavailable: {exc}"
            return False
        try:
            model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cpu")
            self.predictor = SAM2ImagePredictor(model)
            self.available = True
            self.reason = None
            return True
        except Exception as exc:
            self.reason = f"SAM2 failed to load: {exc}"
            return False

    def segment(self, image_bgr, prompt_point=None, prompt_box=None):
        if image_bgr is None:
            return None, "No APS image available for SAM2."
        if not self.ensure_loaded():
            return None, self.reason
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            self.predictor.set_image(image_rgb)
            kwargs = {"multimask_output": True}
            if prompt_point is not None:
                kwargs["point_coords"] = np.array([prompt_point], dtype=np.float32)
                kwargs["point_labels"] = np.array([1], dtype=np.int32)
            if prompt_box is not None:
                kwargs["box"] = np.array(prompt_box, dtype=np.float32)
            masks, scores, _logits = self.predictor.predict(**kwargs)
        except Exception as exc:
            return None, f"SAM2 prediction failed: {exc}"
        if masks is None or len(masks) == 0:
            return None, "SAM2 returned no masks."
        best_idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
        mask = masks[best_idx].astype(np.uint8)
        return {
            "mask": mask,
            "score": float(scores[best_idx]) if scores is not None and len(scores) else None,
            "prompt_point": [float(prompt_point[0]), float(prompt_point[1])] if prompt_point is not None else None,
            "prompt_box": [float(v) for v in prompt_box] if prompt_box is not None else None,
        }, None


def sam2_status():
    try:
        __import__("sam2")
    except Exception as exc:
        return False, f"SAM2 Python package unavailable: {exc}"
    if not SAM2_CHECKPOINT.exists():
        return False, f"SAM2 checkpoint missing: {SAM2_CHECKPOINT}"
    return True, None


def detection_from_mask(mask, source="sam2_prompted", confidence=75.0, reason="SAM2 prompted segmentation."):
    if mask is None or not np.any(mask):
        return {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_bottle_center_px": None,
            "davis_detection_confidence": 0.0,
            "aps_detection_source": source,
            "davis_detection_source": source,
            "davis_detection_backend": source,
            "aps_detection_input_mode": source,
            "davis_candidate_count": 0,
            "best_candidate_scores": None,
            "rejection_reasons": ["empty_mask"],
            "final_mask_source": source,
            "detection_reason": "SAM2/manual mask was empty.",
        }
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_bottle_center_px": None,
            "davis_detection_confidence": 0.0,
            "aps_detection_source": source,
            "davis_detection_source": source,
            "davis_detection_backend": source,
            "aps_detection_input_mode": source,
            "davis_candidate_count": 0,
            "best_candidate_scores": None,
            "rejection_reasons": ["mask_no_contour"],
            "final_mask_source": source,
            "detection_reason": "SAM2/manual mask had no contour.",
        }
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area = float(np.count_nonzero(mask_u8))
    frame_area = float(mask_u8.shape[0] * mask_u8.shape[1])
    aspect = h / max(float(w), 1.0)
    status = DETECTION_DETECTED if confidence >= DETECTION_CONFIDENCE_DETECTED else DETECTION_LOW_CONFIDENCE
    return {
        "davis_bottle_detected": status == DETECTION_DETECTED,
        "davis_detection_status": status,
        "davis_bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
        "davis_bottle_center_px": [round(float(x + w / 2.0), 1), round(float(y + h / 2.0), 1)],
        "davis_detection_confidence": round(float(confidence), 1),
        "aps_detection_source": source,
        "davis_detection_source": source,
        "davis_detection_backend": source,
        "aps_detection_input_mode": source,
        "davis_candidate_count": 1,
        "best_candidate_scores": {
            "mask_area_ratio": round(area / max(frame_area, 1.0), 4),
            "mask_aspect_ratio_h_over_w": round(float(aspect), 3),
            "sam2_confidence": round(float(confidence), 1),
        },
        "davis_detection_candidates": [],
        "davis_candidate_bboxes": [[int(x), int(y), int(x + w), int(y + h)]],
        "rejection_reasons": [] if status == DETECTION_DETECTED else ["sam2_confidence_below_detected_threshold"],
        "final_mask_source": source,
        "detection_reason": reason,
    }


def _score_range(value, low, ideal_low, ideal_high, high):
    value = float(value)
    if value < low or value > high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return float(np.clip(100.0 * (value - low) / max(ideal_low - low, 1e-6), 0.0, 100.0))
    return float(np.clip(100.0 * (high - value) / max(high - ideal_high, 1e-6), 0.0, 100.0))


def _bbox_iou(a, b):
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _bbox_from_component(stats, label):
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    return [x, y, x + w, y + h]


def _add_bbox_candidate(candidates, bbox, source, contour=None, min_w=6, min_h=12):
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    w = max(x2 - x1, 0)
    h = max(y2 - y1, 0)
    if w < min_w or h < min_h:
        return
    clipped = [x1, y1, x2, y2]
    for existing in candidates:
        if _bbox_iou(existing["bbox_xyxy"], clipped) > 0.82:
            if len(source) < len(existing.get("source", "")):
                existing["source"] = source
            return
    candidates.append({"bbox_xyxy": clipped, "source": source, "contour": contour})


def _projection_candidate(edges, image_shape):
    h_img, w_img = image_shape[:2]
    if edges is None or not np.any(edges):
        return None
    col_strength = edges.sum(axis=0).astype(np.float32)
    row_strength = edges.sum(axis=1).astype(np.float32)
    if np.max(col_strength) <= 0 or np.max(row_strength) <= 0:
        return None
    col_threshold = max(float(np.percentile(col_strength, 72)), float(np.max(col_strength) * 0.18))
    row_threshold = max(float(np.percentile(row_strength, 60)), float(np.max(row_strength) * 0.10))
    cols = np.where(col_strength >= col_threshold)[0]
    rows = np.where(row_strength >= row_threshold)[0]
    if cols.size < 6 or rows.size < 18:
        return None
    x1, x2 = int(cols.min()), int(cols.max())
    y1, y2 = int(rows.min()), int(rows.max())
    pad_x = max(4, int((x2 - x1 + 1) * 0.12))
    pad_y = max(4, int((y2 - y1 + 1) * 0.08))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w_img, x2 + pad_x + 1),
        min(h_img, y2 + pad_y + 1),
    ]


def _preprocess_aps_for_detection(gray):
    if gray is None:
        return None, None, None, None, None, None
    detection_input = _auto_contrast_u8(gray, gamma=0.72)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(detection_input)
    enhanced = _auto_contrast_u8(enhanced, gamma=0.85)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(denoised, 35, 110)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)
    threshold_options = [adaptive, adaptive_inv, otsu, otsu_inv]
    threshold = min(
        threshold_options,
        key=lambda img: abs((float(np.count_nonzero(img)) / max(float(img.size), 1.0)) - 0.28),
    )
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    dark_cut = float(np.percentile(denoised, 36))
    dark_body = (denoised <= dark_cut).astype(np.uint8) * 255
    h_img, w_img = denoised.shape[:2]
    plausible_band = np.zeros_like(dark_body)
    plausible_band[int(h_img * 0.16):int(h_img * 0.96), int(w_img * 0.08):int(w_img * 0.92)] = 255
    dark_body = cv2.bitwise_and(dark_body, plausible_band)
    dark_body = cv2.morphologyEx(dark_body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dark_body = cv2.morphologyEx(dark_body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contour_mask = cv2.bitwise_or(edges, threshold)
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return detection_input, enhanced, edges, threshold, contour_mask, dark_body


def _generate_aps_candidates(edges, threshold, contour_mask, dark_body, image_shape):
    candidates = []
    masks = [
        ("dark_body_component", dark_body),
        ("edge_contour", contour_mask),
        ("threshold_component", threshold),
        ("threshold_inverse_component", cv2.bitwise_not(threshold) if threshold is not None else None),
    ]
    edge_closed = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) if edges is not None else None
    if edge_closed is not None:
        masks.append(("edge_component", edge_closed))
    for source, mask in masks:
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            _add_bbox_candidate(candidates, [x, y, x + w, y + h], source, contour=contour)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 80:
                continue
            _add_bbox_candidate(candidates, _bbox_from_component(stats, label), source)
    projection = _projection_candidate(edges, image_shape)
    _add_bbox_candidate(candidates, projection, "edge_projection")
    h_img, w_img = image_shape[:2]
    _add_bbox_candidate(
        candidates,
        [int(w_img * 0.38), int(h_img * 0.20), int(w_img * 0.58), int(h_img * 0.83)],
        "center_body_prior",
        min_w=10,
        min_h=40,
    )
    for item in list(candidates):
        x1, y1, x2, y2 = item["bbox_xyxy"]
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        aspect = h / max(float(w), 1.0)
        if aspect >= 2.8 and w / max(float(w_img), 1.0) < 0.10:
            pad = int(max(w * 1.25, 10))
            _add_bbox_candidate(
                candidates,
                [max(0, x1 - pad), y1, min(w_img, x2 + pad), y2],
                "expanded_body_from_edge",
            )
    return candidates


def _oriented_candidate_geometry(candidate):
    """Describe a contour along its long axis, independent of image rotation."""
    contour = candidate.get("contour")
    if contour is not None and len(contour) >= 3 and cv2.contourArea(contour) > 1.0:
        (cx, cy), (rect_w, rect_h), angle_deg = cv2.minAreaRect(contour)
        length = float(max(rect_w, rect_h))
        width = float(min(rect_w, rect_h))
        axis_angle_deg = float(angle_deg if rect_w >= rect_h else angle_deg + 90.0)
        box = cv2.boxPoints(((cx, cy), (rect_w, rect_h), angle_deg))
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(0.75, 0.006 * perimeter), True).reshape(-1, 2)
        if approx.shape[0] > 128:
            approx = approx[:: int(np.ceil(approx.shape[0] / 128.0))]
        contour_points = [[round(float(px), 2), round(float(py), 2)] for px, py in approx]
    else:
        x1, y1, x2, y2 = [float(value) for value in candidate["bbox_xyxy"]]
        rect_w = max(x2 - x1, 1.0)
        rect_h = max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length = float(max(rect_w, rect_h))
        width = float(min(rect_w, rect_h))
        axis_angle_deg = 0.0 if rect_w >= rect_h else 90.0
        box = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        contour_points = None
    radians = np.deg2rad(axis_angle_deg)
    axis = np.asarray([np.cos(radians), np.sin(radians)], dtype=float)
    if (abs(axis[0]) >= abs(axis[1]) and axis[0] < 0.0) or (abs(axis[1]) > abs(axis[0]) and axis[1] < 0.0):
        axis = -axis
    return {
        "center_px": [float(cx), float(cy)],
        "length_px": length,
        "width_px": width,
        "aspect_long_over_short": length / max(width, 1.0),
        "axis_vector_px": [float(axis[0]), float(axis[1])],
        "axis_angle_degrees": float(np.degrees(np.arctan2(axis[1], axis[0]))),
        "box_points": [[round(float(px), 2), round(float(py), 2)] for px, py in box],
        "contour_points": contour_points,
    }


def _raw_aps_to_u8(raw_gray, detection_config=None):
    """Scale raw APS codes using configured or inferred ADC range, not dtype width."""
    raw = np.asarray(raw_gray)
    if raw.ndim == 3:
        raw = np.mean(raw[..., :3], axis=2)
    config = detection_config or {}
    finite = raw[np.isfinite(raw)] if np.issubdtype(raw.dtype, np.floating) else raw.reshape(-1)
    observed_max = float(np.max(finite)) if finite.size else 0.0
    normalization_source = "uint8_native"
    normalization_bit_depth = 8
    code_max = 255.0
    if raw.dtype == np.uint8:
        raw_u8 = raw.copy()
    elif np.issubdtype(raw.dtype, np.integer):
        configured_bits = config.get("raw_adc_bit_depth", "auto")
        try:
            configured_bits = int(configured_bits)
        except (TypeError, ValueError):
            configured_bits = None
        dtype_bits = int(np.iinfo(raw.dtype).bits)
        if configured_bits is not None and 8 <= configured_bits <= dtype_bits:
            normalization_bit_depth = configured_bits
            normalization_source = "configured_adc_bit_depth"
        else:
            required_bits = max(1, int(np.ceil(np.log2(max(observed_max, 1.0) + 1.0))))
            common_depths = [depth for depth in (8, 10, 12, 14, 16, dtype_bits) if depth <= dtype_bits]
            # uint16 DAVIS frames commonly carry 10/12-bit ADC codes. Requiring
            # at least 10 bits avoids treating an underexposed frame's maximum
            # as a saturated white reference.
            minimum_bits = 10 if dtype_bits > 8 else 8
            normalization_bit_depth = next(
                (depth for depth in sorted(set(common_depths)) if depth >= max(required_bits, minimum_bits)),
                dtype_bits,
            )
            normalization_source = "inferred_common_adc_range"
        code_max = float((1 << normalization_bit_depth) - 1)
        raw_u8 = np.clip(raw.astype(np.float64) * 255.0 / max(code_max, 1.0), 0.0, 255.0).astype(np.uint8)
    else:
        raw_float = np.nan_to_num(raw.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
        if raw_float.size and float(np.max(raw_float)) <= 1.0 and float(np.min(raw_float)) >= 0.0:
            raw_float *= 255.0
            normalization_source = "unit_float_range"
        else:
            normalization_source = "float_0_255_clipped"
        normalization_bit_depth = None
        code_max = 1.0 if normalization_source == "unit_float_range" else 255.0
        raw_u8 = np.clip(raw_float, 0.0, 255.0).astype(np.uint8)
    return raw_u8, {
        "raw_normalization_source": normalization_source,
        "raw_normalization_bit_depth": normalization_bit_depth,
        "raw_normalization_code_max": code_max,
        "raw_observed_max": observed_max,
    }


def _raw_aps_candidate_evidence(raw_gray, candidate, bbox, detection_config=None):
    """Measure saturation and texture on unenhanced APS pixels."""
    raw_u8, normalization = _raw_aps_to_u8(raw_gray, detection_config)
    x1, y1, x2, y2 = bbox
    crop = raw_u8[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            "raw_bright_fraction": 0.0,
            "raw_bbox_bright_fraction": 0.0,
            "raw_interior_edge_density": 0.0,
            "raw_intensity_median": 0.0,
            "glare_like_low_texture": False,
            **normalization,
        }
    mask = np.full(crop.shape, 255, dtype=np.uint8)
    contour = candidate.get("contour")
    if contour is not None and len(contour) >= 3:
        mask.fill(0)
        shifted = np.asarray(contour, dtype=np.int32).copy()
        shifted[:, 0, 0] -= x1
        shifted[:, 0, 1] -= y1
        cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    kernel_size = max(3, int(round(min(crop.shape[:2]) * 0.08)) | 1)
    kernel_size = min(kernel_size, 11)
    interior = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1) > 0
    if not np.any(interior):
        interior = mask > 0
    values = crop[interior]
    raw_edges = cv2.Canny(crop, 30, 90)
    config = detection_config or {}
    bright_threshold = float(config.get("glare_bright_threshold_u8", APS_GLARE_BRIGHT_THRESHOLD_U8))
    max_bright_fraction = float(config.get("glare_max_bright_fraction", APS_GLARE_MAX_BRIGHT_FRACTION))
    max_edge_density = float(config.get("glare_max_interior_edge_density", APS_GLARE_MAX_INTERIOR_EDGE_DENSITY))
    bright_fraction = float(np.mean(values >= bright_threshold)) if values.size else 0.0
    bbox_bright_fraction = float(np.mean(crop >= bright_threshold))
    interior_edge_density = float(np.mean(raw_edges[interior] > 0)) if values.size else 0.0
    median = float(np.median(values)) if values.size else 0.0
    glare_like = bool(
        max(bright_fraction, bbox_bright_fraction) >= max_bright_fraction
        and interior_edge_density <= max_edge_density
    )
    return {
        "raw_bright_fraction": bright_fraction,
        "raw_bbox_bright_fraction": bbox_bright_fraction,
        "raw_interior_edge_density": interior_edge_density,
        "raw_intensity_median": median,
        "glare_like_low_texture": glare_like,
        **normalization,
    }


def _score_aps_candidate(
    candidate,
    detection_input,
    enhanced,
    edges,
    raw_gray,
    roi=None,
    previous_bbox=None,
    detection_config=None,
    raw_evidence_source="raw_aps_frame",
):
    h_img, w_img = detection_input.shape[:2]
    frame_area = float(w_img * h_img)
    x1, y1, x2, y2 = candidate["bbox_xyxy"]
    x1 = int(np.clip(x1, 0, w_img - 1))
    y1 = int(np.clip(y1, 0, h_img - 1))
    x2 = int(np.clip(x2, x1 + 1, w_img))
    y2 = int(np.clip(y2, y1 + 1, h_img))
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    bbox_area = float(w * h)
    contour_area = float(cv2.contourArea(candidate["contour"])) if candidate.get("contour") is not None else bbox_area * 0.55
    area_ratio = contour_area / max(frame_area, 1.0)
    bbox_area_ratio = bbox_area / max(frame_area, 1.0)
    aspect = h / max(float(w), 1.0)
    height_ratio = h / max(float(h_img), 1.0)
    width_ratio = w / max(float(w_img), 1.0)
    oriented = _oriented_candidate_geometry(candidate)
    oriented_aspect = float(oriented["aspect_long_over_short"])
    oriented_length_ratio = float(oriented["length_px"]) / max(float(max(w_img, h_img)), 1.0)
    oriented_width_ratio = float(oriented["width_px"]) / max(float(min(w_img, h_img)), 1.0)
    cx, cy = oriented["center_px"]
    x_norm = cx / max(float(w_img), 1.0)
    y_norm = cy / max(float(h_img), 1.0)
    center_distance = np.sqrt(((cx - w_img / 2.0) / max(w_img / 2.0, 1.0)) ** 2 + ((cy - h_img / 2.0) / max(h_img / 2.0, 1.0)) ** 2)
    center_score = float(np.clip(100.0 * (1.0 - center_distance / 1.25), 0.0, 100.0))
    centerline_score = float(np.clip(100.0 * (1.0 - abs(x_norm - 0.5) / 0.42), 0.0, 100.0))
    lower_center_score = _score_range(y_norm, 0.18, 0.34, 0.78, 0.98)
    edge_crop = edges[y1:y2, x1:x2]
    enhanced_crop = enhanced[y1:y2, x1:x2]
    edge_density = float(np.count_nonzero(edge_crop)) / max(float(edge_crop.size), 1.0)
    contrast_std = float(np.std(enhanced_crop)) if enhanced_crop.size else 0.0
    local_range = float(np.percentile(enhanced_crop, 95) - np.percentile(enhanced_crop, 5)) if enhanced_crop.size else 0.0
    edge_strength_score = float(np.clip((edge_density / 0.055) * 62.0 + (contrast_std / 42.0) * 24.0 + (local_range / 135.0) * 14.0, 0.0, 100.0))
    contrast_score = float(np.clip((contrast_std / 42.0) * 70.0 + (local_range / 135.0) * 30.0, 0.0, 100.0))
    area_score = _score_range(bbox_area_ratio, 0.0025, 0.018, 0.45, 0.82)
    contour_area_score = _score_range(area_ratio, 0.0012, 0.010, 0.35, 0.70)
    aspect_ratio_score = _score_range(oriented_aspect, 1.0, 1.35, 6.5, 12.0)
    oriented_length_score = _score_range(oriented_length_ratio, 0.10, 0.22, 0.92, 1.25)
    oriented_width_score = _score_range(oriented_width_ratio, 0.025, 0.055, 0.55, 0.95)
    height_width_score = float(np.clip(0.68 * oriented_length_score + 0.32 * oriented_width_score, 0.0, 100.0))
    body_width_score = oriented_width_score
    bbox_height_score = oriented_length_score
    bbox_width_score = oriented_width_score
    extent = contour_area / max(bbox_area, 1.0)
    extent_score = _score_range(extent, 0.035, 0.12, 0.86, 0.99)
    solidity_score = 55.0
    if candidate.get("contour") is not None:
        hull = cv2.convexHull(candidate["contour"])
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity_score = float(np.clip(100.0 * contour_area / hull_area, 0.0, 100.0))
    temporal_consistency_score = 0.0
    if previous_bbox is not None:
        temporal_consistency_score = float(np.clip(100.0 * _bbox_iou([x1, y1, x2, y2], previous_bbox) / 0.55, 0.0, 100.0))
    roi_score = 100.0
    if roi is not None:
        rx, ry, rw, rh = roi
        ix1 = max(x1, rx)
        iy1 = max(y1, ry)
        ix2 = min(x2, rx + rw)
        iy2 = min(y2, ry + rh)
        intersection = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        union = bbox_area + float(rw * rh) - intersection
        roi_score = float(np.clip(100.0 * intersection / max(union, 1.0), 0.0, 100.0))
    contour_available = candidate.get("contour") is not None and cv2.contourArea(candidate["contour"]) > 40
    source = candidate.get("source", "unknown")
    raw_evidence = _raw_aps_candidate_evidence(
        raw_gray,
        candidate,
        [x1, y1, x2, y2],
        detection_config=detection_config,
    )
    bottle_body_shape_score = float(np.clip(
        0.32 * aspect_ratio_score
        + 0.24 * oriented_length_score
        + 0.16 * oriented_width_score
        + 0.14 * extent_score
        + 0.14 * solidity_score,
        0.0,
        100.0,
    ))
    background_rejection_score = 100.0
    if x_norm < 0.12 or x_norm > 0.88:
        background_rejection_score -= 18.0
    if oriented_width_ratio < 0.065 and source in {"edge_component", "edge_contour"}:
        background_rejection_score -= 35.0
    if oriented_width_ratio < 0.035:
        background_rejection_score -= 28.0
    if oriented_length_ratio < 0.12:
        background_rejection_score -= 34.0
    if bbox_area_ratio > 0.74:
        background_rejection_score -= 35.0
    if not contour_available and source in {"edge_component", "edge_contour"}:
        background_rejection_score -= 18.0
    if center_score < 30.0 and oriented_width_ratio < 0.10:
        background_rejection_score -= 12.0
    if raw_evidence["glare_like_low_texture"]:
        background_rejection_score -= 60.0
    background_rejection_score = float(np.clip(background_rejection_score, 0.0, 100.0))
    total = float(np.clip(
        0.08 * area_score
        + 0.06 * contour_area_score
        + 0.15 * aspect_ratio_score
        + 0.08 * oriented_length_score
        + 0.10 * height_width_score
        + 0.09 * edge_strength_score
        + 0.08 * contrast_score
        + 0.08 * centerline_score
        + 0.05 * lower_center_score
        + 0.08 * body_width_score
        + 0.12 * bottle_body_shape_score
        + 0.06 * background_rejection_score
        + 0.02 * temporal_consistency_score,
        0.0,
        100.0,
    ))
    if candidate.get("source") in {"edge_component", "edge_contour"} and oriented_width_ratio < 0.07 and oriented_aspect > 3.0:
        total = max(0.0, total - 16.0)
    if raw_evidence["glare_like_low_texture"]:
        total = min(total, DETECTION_CONFIDENCE_LOW - 1.0)
    if background_rejection_score < 45.0:
        total = min(total, 55.0)
    elif background_rejection_score < 65.0:
        total = min(total, 68.0)
    if center_score < 30.0:
        total = max(0.0, total - 6.0)
    if source == "expanded_body_from_edge" and oriented_length_ratio < 0.20:
        total = min(total, 66.0)
    if not contour_available and source in {"edge_component", "edge_contour", "expanded_body_from_edge"}:
        total = min(total, 68.0)
    if source in {"edge_component", "edge_contour"} and not contour_available:
        total = min(total, 76.0)
    if not contour_available:
        # Projection/center priors are proposal generators, not confirmed APS
        # object boundaries. They may remain visible diagnostics but cannot by
        # themselves authorize an object-specific ROI.
        total = min(total, DETECTION_CONFIDENCE_DETECTED - 2.0)
    if roi is not None:
        total = float(np.clip(0.88 * total + 0.12 * roi_score, 0.0, 100.0))
    cue_scores = {
        "area_score": round(area_score, 1),
        "contour_area_score": round(contour_area_score, 1),
        "aspect_ratio_score": round(aspect_ratio_score, 1),
        "oriented_aspect_score": round(aspect_ratio_score, 1),
        "oriented_length_score": round(oriented_length_score, 1),
        "oriented_width_score": round(oriented_width_score, 1),
        "bbox_height_score": round(bbox_height_score, 1),
        "bbox_width_score": round(bbox_width_score, 1),
        "height_width_score": round(height_width_score, 1),
        "body_width_score": round(body_width_score, 1),
        "edge_strength_score": round(edge_strength_score, 1),
        "contrast_score": round(contrast_score, 1),
        "center_score": round(center_score, 1),
        "centerline_score": round(centerline_score, 1),
        "lower_center_score": round(lower_center_score, 1),
        "bottle_body_shape_score": round(bottle_body_shape_score, 1),
        "background_rejection_score": round(background_rejection_score, 1),
        "extent_score": round(extent_score, 1),
        "solidity_score": round(solidity_score, 1),
        "temporal_consistency_score": round(temporal_consistency_score, 1),
        "manual_roi_overlap_score": round(roi_score, 1) if roi is not None else None,
        "total_score": round(total, 1),
    }
    weakest = [name for name, score in cue_scores.items() if score is not None and score < 45.0]
    hard_rejections = []
    if background_rejection_score < 35.0:
        hard_rejections.append("background_or_side_candidate")
    if oriented_length_ratio < 0.10:
        hard_rejections.append("candidate_too_short")
    if oriented_width_ratio < 0.025:
        hard_rejections.append("candidate_too_narrow")
    if raw_evidence["glare_like_low_texture"]:
        hard_rejections.append("glare_like_low_texture")
    if bbox_area_ratio < 0.0025:
        hard_rejections.append("candidate_too_small")
    if hard_rejections:
        weakest = hard_rejections + weakest
    return {
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "center_px": [round(float(cx), 1), round(float(cy), 1)],
        "confidence": round(total, 1),
        "source": source,
        "area": round(contour_area, 1),
        "area_ratio": round(area_ratio, 4),
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "aspect_ratio_h_over_w": round(float(aspect), 3),
        "oriented_aspect_ratio_long_over_short": round(oriented_aspect, 3),
        "oriented_length_px": round(float(oriented["length_px"]), 2),
        "oriented_width_px": round(float(oriented["width_px"]), 2),
        "axis_vector_px": [round(float(value), 6) for value in oriented["axis_vector_px"]],
        "axis_angle_degrees": round(float(oriented["axis_angle_degrees"]), 3),
        "oriented_box_points": oriented["box_points"],
        "contour_points": oriented["contour_points"],
        "height_ratio": round(float(height_ratio), 3),
        "width_ratio": round(float(width_ratio), 3),
        "extent": round(float(extent), 3),
        "edge_density": round(edge_density, 4),
        "contrast_std": round(contrast_std, 2),
        "raw_bright_fraction": round(float(raw_evidence["raw_bright_fraction"]), 4),
        "raw_bbox_bright_fraction": round(float(raw_evidence["raw_bbox_bright_fraction"]), 4),
        "raw_interior_edge_density": round(float(raw_evidence["raw_interior_edge_density"]), 4),
        "raw_intensity_median": round(float(raw_evidence["raw_intensity_median"]), 2),
        "glare_like_low_texture": bool(raw_evidence["glare_like_low_texture"]),
        "raw_aps_evidence_source": str(raw_evidence_source),
        "raw_normalization_source": raw_evidence["raw_normalization_source"],
        "raw_normalization_bit_depth": raw_evidence["raw_normalization_bit_depth"],
        "raw_normalization_code_max": raw_evidence["raw_normalization_code_max"],
        "raw_observed_max": raw_evidence["raw_observed_max"],
        "scores": cue_scores,
        "weakest_cues": weakest[:5],
        "rejection_reasons": weakest[:5],
        "contour_available": bool(contour_available),
    }


def _centerline_from_bbox(bbox):
    if not bbox:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    return [[round(cx, 1), round(y1, 1)], [round(cx, 1), round(y2, 1)]]


def detect_bottle_from_aps(
    gray,
    roi_xywh=None,
    aps_detection_input_mode="unknown",
    previous_bbox=None,
    detection_config=None,
    raw_gray=None,
):
    debug_images = {}
    if gray is None:
        return {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_bottle_center_px": None,
            "davis_detection_confidence": 0.0,
            "aps_detection_source": "none",
            "davis_detection_source": "none",
            "davis_detection_backend": "contour",
            "aps_detection_input_mode": aps_detection_input_mode,
            "detection_reason": "No APS frame available.",
            "davis_candidate_count": 0,
            "best_candidate_scores": None,
            "rejection_reasons": ["no_aps_frame"],
        }, debug_images
    detection_input, enhanced, edges, threshold, contour_mask, dark_body = _preprocess_aps_for_detection(gray)
    debug_images = {
        "detection_input": cv2.cvtColor(detection_input, cv2.COLOR_GRAY2BGR),
        "detection_input_gray": detection_input,
        "enhanced": cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        "edges": edges,
        "threshold": threshold,
        "dark_body": dark_body,
    }
    h_img, w_img = gray.shape[:2]
    raw_evidence_gray = None if raw_gray is None else np.asarray(raw_gray)
    if raw_evidence_gray is not None and raw_evidence_gray.shape[:2] == gray.shape[:2]:
        raw_evidence_source = "raw_aps_frame"
    else:
        raw_evidence_gray = gray
        raw_evidence_source = "detection_input_fallback_not_verified_raw"
    roi = _clip_roi(roi_xywh) if roi_xywh is not None else None
    raw_candidates = _generate_aps_candidates(edges, threshold, contour_mask, dark_body, gray.shape)
    candidates = [
        _score_aps_candidate(
            candidate,
            detection_input,
            enhanced,
            edges,
            raw_evidence_gray,
            roi=roi,
            previous_bbox=previous_bbox,
            detection_config=detection_config,
            raw_evidence_source=raw_evidence_source,
        )
        for candidate in raw_candidates
    ]
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    top_candidates = candidates[:8]
    best = top_candidates[0] if top_candidates else None
    contours_debug = cv2.cvtColor(detection_input, cv2.COLOR_GRAY2BGR)
    for candidate in top_candidates:
        x1, y1, x2, y2 = candidate["bbox_xyxy"]
        color = (120, 120, 120)
        if candidate is best:
            color = (0, 180, 255) if candidate["confidence"] < DETECTION_CONFIDENCE_DETECTED else (0, 255, 0)
        cv2.rectangle(contours_debug, (x1, y1), (x2, y2), color, 1)
        cv2.putText(contours_debug, f"{candidate['confidence']:.0f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    debug_images["contours_debug"] = contours_debug
    debug_images["candidates_debug"] = contours_debug
    if best is None:
        return {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_bottle_center_px": None,
            "davis_detection_confidence": 0.0,
            "aps_detection_source": "aps_contour_shape_score",
            "davis_detection_source": "opencv_aps",
            "davis_detection_backend": "opencv_aps",
            "aps_detection_input_mode": aps_detection_input_mode,
            "davis_candidate_count": 0,
            "best_candidate_scores": None,
            "davis_detection_candidates": [],
            "davis_candidate_bboxes": [],
            "rejection_reasons": ["no_candidate_contours"],
            "detection_reason": "No APS contour candidate passed minimum size filters.",
            "yolo_optional_available": False,
            "davis_contour_available": False,
            "davis_outline_source": "none",
            "davis_centerline_px": None,
        }, debug_images
    confidence = float(best["confidence"])
    weakest_cues = list(best.get("weakest_cues", []))
    if confidence >= DETECTION_CONFIDENCE_DETECTED:
        status = DETECTION_DETECTED
        detected = True
        reason = (
            f"Automatic DAVIS bottle cue detected: confidence={confidence:.1f}, "
            f"source={best.get('source')}, "
            f"oriented_aspect={best['oriented_aspect_ratio_long_over_short']:.2f}, "
            f"weakest_cues={weakest_cues or ['none']}."
        )
    elif confidence >= DETECTION_CONFIDENCE_LOW:
        status = DETECTION_LOW_CONFIDENCE
        detected = False
        reason = (
            f"Bottle-like cue exists but confidence is low: confidence={confidence:.1f}, "
            f"source={best.get('source')}, weakest_cues={weakest_cues or ['score_below_detected_threshold']}."
        )
    else:
        status = DETECTION_NOT_DETECTED
        detected = False
        reason = (
            f"Best APS contour is not reliable enough: confidence={confidence:.1f}, "
            f"source={best.get('source')}, weakest_cues={weakest_cues or ['score_below_low_confidence_threshold']}."
        )
    outline_source = "contour" if best.get("contour_available") else "bbox_approximation"
    return {
        "davis_bottle_detected": detected,
        "davis_detection_status": status,
        "davis_bbox_xyxy": best["bbox_xyxy"],
        "davis_bottle_center_px": best["center_px"],
        "davis_detection_confidence": round(confidence, 1),
        "aps_detection_source": "aps_contour_shape_score",
        "davis_detection_source": "opencv_aps",
        "davis_detection_backend": "opencv_aps",
        "aps_detection_input_mode": aps_detection_input_mode,
        "davis_candidate_count": len(candidates),
        "best_candidate_scores": best["scores"],
        "selected_candidate": best,
        "davis_detection_candidates": top_candidates,
        "davis_candidate_bboxes": [candidate["bbox_xyxy"] for candidate in top_candidates],
        "rejection_reasons": weakest_cues,
        "detection_reason": reason,
        "yolo_optional_available": False,
        "davis_contour_available": bool(best.get("contour_available")),
        "davis_contour_points": best.get("contour_points"),
        "davis_outline_source": outline_source,
        "davis_centerline_px": [
            [
                round(float(best["center_px"][0] - best["axis_vector_px"][0] * best["oriented_length_px"] * 0.5), 1),
                round(float(best["center_px"][1] - best["axis_vector_px"][1] * best["oriented_length_px"] * 0.5), 1),
            ],
            [
                round(float(best["center_px"][0] + best["axis_vector_px"][0] * best["oriented_length_px"] * 0.5), 1),
                round(float(best["center_px"][1] + best["axis_vector_px"][1] * best["oriented_length_px"] * 0.5), 1),
            ],
        ],
        "davis_axis_vector_px": best.get("axis_vector_px"),
        "davis_axis_angle_degrees": best.get("axis_angle_degrees"),
        "davis_oriented_length_px": best.get("oriented_length_px"),
        "davis_oriented_width_px": best.get("oriented_width_px"),
        "aps_raw_evidence_source": best.get("raw_aps_evidence_source"),
        "aps_raw_normalization_source": best.get("raw_normalization_source"),
        "aps_raw_normalization_bit_depth": best.get("raw_normalization_bit_depth"),
        "aps_raw_normalization_code_max": best.get("raw_normalization_code_max"),
        "aps_raw_observed_max": best.get("raw_observed_max"),
    }, debug_images


def planner_detection_from_stack_result(stack_result, opencv_result=None):
    opencv_result = opencv_result or stack_result.get("opencv_result") or {}
    bbox = stack_result.get("bbox_xyxy")
    confidence = float(stack_result.get("confidence", 0.0) or 0.0)
    status = stack_result.get("status", DETECTION_NOT_DETECTED)
    source = stack_result.get("source", "failed")
    center = None
    if bbox:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        center = [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)]
    candidate = {
        "bbox_xyxy": bbox,
        "confidence": round(confidence, 1),
        "source": source,
        "contour_available": stack_result.get("contour") is not None or stack_result.get("mask") is not None,
        "scores": opencv_result.get("best_candidate_scores"),
    }
    return {
        "davis_bottle_detected": bool(status == DETECTION_DETECTED and bbox),
        "davis_detection_status": status,
        "davis_bbox_xyxy": bbox,
        "davis_bottle_center_px": center,
        "davis_detection_confidence": round(confidence, 1),
        "aps_detection_source": source,
        "davis_detection_source": source,
        "davis_detection_backend": source,
        "aps_detection_input_mode": source,
        "davis_candidate_count": opencv_result.get("davis_candidate_count", 1 if bbox else 0),
        "best_candidate_scores": opencv_result.get("best_candidate_scores"),
        "selected_candidate": candidate,
        "davis_detection_candidates": opencv_result.get("davis_detection_candidates", [candidate] if bbox else []),
        "davis_candidate_bboxes": opencv_result.get("davis_candidate_bboxes", [bbox] if bbox else []),
        "rejection_reasons": opencv_result.get("rejection_reasons", [] if status == DETECTION_DETECTED else ["detector_low_confidence"]),
        "detection_reason": stack_result.get("reason"),
        "davis_contour_available": bool(candidate["contour_available"]),
        "davis_outline_source": "mask" if stack_result.get("mask") is not None else ("contour" if stack_result.get("contour") is not None else "bbox_approximation"),
        "davis_centerline_px": _centerline_from_bbox(bbox),
        "detector_source": source,
        "yolo_available": bool(stack_result.get("yolo_available")),
        "yolo_model_path": stack_result.get("yolo_model_path", str(DAVIS_YOLO_MODEL_PATH)),
        "sam2_available": bool(stack_result.get("sam2_available")),
        "opencv_confidence": stack_result.get("opencv_confidence"),
        "final_detection_status": status,
        "final_confidence": round(confidence, 1),
        "detector_stack_reason": stack_result.get("reason"),
        "detector_limitations": stack_result.get("limitations", []),
        "detector_fallback_reason": stack_result.get("detector_fallback_reason"),
    }


def manual_roi_detection(roi_xywh):
    x, y, w, h = _clip_roi(roi_xywh)
    return {
        "davis_bottle_detected": True,
        "davis_detection_status": DETECTION_MANUAL_ROI,
        "davis_bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
        "davis_bottle_center_px": [float(x + w / 2.0), float(y + h / 2.0)],
        "davis_detection_confidence": 100.0,
        "aps_detection_source": "manual_roi",
        "davis_detection_source": "manual_roi",
        "davis_detection_backend": "manual_roi",
        "aps_detection_input_mode": "manual_roi",
        "detection_reason": "Manual ROI, not automatic bottle detection.",
        "davis_candidate_count": 1,
        "best_candidate_scores": {"manual_roi_overlap_score": 100.0},
        "davis_detection_candidates": [],
        "davis_candidate_bboxes": [[int(x), int(y), int(x + w), int(y + h)]],
        "rejection_reasons": [],
        "manual_roi_active": True,
    }


def bbox_xyxy_to_roi_xywh(bbox):
    if not bbox:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)]


def event_motion_roi_detection(events, roi_xywh=None):
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] < 30:
        return {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_bottle_center_px": None,
            "davis_detection_confidence": 0.0,
            "aps_detection_source": "events",
            "davis_detection_source": "none",
            "davis_detection_backend": "events_motion_roi",
            "aps_detection_input_mode": "events_only",
            "detection_reason": "No APS frame and insufficient event activity for a motion ROI.",
            "moving_object_roi_xyxy": None,
            "event_stream_active": False,
        }
    x, y, w, h = _clip_roi(roi_xywh)
    in_roi = (arr[:, 1] >= x) & (arr[:, 1] < x + w) & (arr[:, 2] >= y) & (arr[:, 2] < y + h)
    roi_events = arr[in_roi]
    if roi_events.shape[0] < 30:
        return {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_bottle_center_px": None,
            "davis_detection_confidence": 0.0,
            "aps_detection_source": "events",
            "davis_detection_source": "none",
            "davis_detection_backend": "events_motion_roi",
            "aps_detection_input_mode": "events_only",
            "detection_reason": "Event activity is outside the selected ROI.",
            "moving_object_roi_xyxy": None,
            "event_stream_active": True,
        }
    xs = roi_events[:, 1]
    ys = roi_events[:, 2]
    x1, x2 = np.percentile(xs, [5, 95])
    y1, y2 = np.percentile(ys, [5, 95])
    confidence = float(np.clip(roi_events.shape[0] / 20.0, 0.0, 100.0))
    return {
        "davis_bottle_detected": False,
        "davis_detection_status": DETECTION_LOW_CONFIDENCE,
        "davis_bbox_xyxy": None,
        "davis_bottle_center_px": [float(np.mean(xs)), float(np.mean(ys))],
        "davis_detection_confidence": round(confidence, 1),
        "aps_detection_source": "events_motion_roi",
        "davis_detection_source": "motion_roi",
        "davis_detection_backend": "events_motion_roi",
        "aps_detection_input_mode": "events_only",
        "detection_reason": "Event-based moving object ROI only; not confirmed bottle detection.",
        "moving_object_roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "event_stream_active": True,
    }


def estimate_clamp_patches(detection, patch_fraction=DAVIS_CLAMP_BODY_FRACTION):
    bbox = detection.get("davis_bbox_xyxy")
    if detection.get("davis_detection_status") not in {DETECTION_DETECTED, DETECTION_MANUAL_ROI} or not bbox:
        return {
            "davis_clamp_patch_left": None,
            "davis_clamp_patch_right": None,
            "left_clamp_patch_px": None,
            "right_clamp_patch_px": None,
            "grasp_line_px": None,
            "grip_width_px": None,
            "clamp_patch_estimate": None,
            "davis_alignment_score": 0.0,
            "davis_contact_confidence": 0.0,
            "davis_clamp_estimate_confidence": 0.0,
            "davis_clamp_estimate_ready": False,
            "davis_ready_to_close_approximate": False,
            "davis_visual_contact_estimate": "unavailable",
        }
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cy = y1 + patch_fraction * height
    patch_h = max(12.0, min(float(DAVIS_CLAMP_PATCH_HEIGHT_PX), height * 0.30))
    patch_w = max(3.0, min(float(DAVIS_CLAMP_PATCH_WIDTH_PX), width * 0.24, width * 0.34))
    inset = max(1.0, min(width * 0.08, width * 0.12))
    left = [x1 + inset, cy - patch_h / 2.0, x1 + inset + patch_w, cy + patch_h / 2.0]
    right = [x2 - inset - patch_w, cy - patch_h / 2.0, x2 - inset, cy + patch_h / 2.0]
    if left[2] >= right[0]:
        patch_w = max(2.0, width * 0.16)
        left_center_x = x1 + width * 0.22
        right_center_x = x2 - width * 0.22
        left = [left_center_x - patch_w / 2.0, cy - patch_h / 2.0, left_center_x + patch_w / 2.0, cy + patch_h / 2.0]
        right = [right_center_x - patch_w / 2.0, cy - patch_h / 2.0, right_center_x + patch_w / 2.0, cy + patch_h / 2.0]
    frame_w, frame_h = DAVIS_FRAME_SIZE
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dx = abs(center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
    dy = abs(center_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0)
    alignment = float(np.clip(100.0 * (1.0 - 0.65 * dx - 0.35 * dy), 0.0, 100.0))
    confidence = float(np.clip(0.65 * detection.get("davis_detection_confidence", 0.0) + 0.35 * alignment, 0.0, 100.0))
    return {
        "davis_clamp_patch_left": [round(v, 1) for v in left],
        "davis_clamp_patch_right": [round(v, 1) for v in right],
        "left_clamp_patch_px": [round(v, 1) for v in left],
        "right_clamp_patch_px": [round(v, 1) for v in right],
        "grasp_line_px": [[round(left[2], 1), round(cy, 1)], [round(right[0], 1), round(cy, 1)]],
        "grip_width_px": round(max(right[0] - left[2], 0.0), 1),
        "clamp_patch_estimate": {
            "left_patch_px": [round(v, 1) for v in left],
            "right_patch_px": [round(v, 1) for v in right],
            "patch_height_px": round(patch_h, 1),
            "patch_width_px": round(patch_w, 1),
            "body_fraction": patch_fraction,
            "estimate_type": "image_space_only",
        },
        "davis_alignment_score": round(alignment, 1),
        "davis_contact_confidence": round(confidence, 1),
        "davis_clamp_estimate_confidence": round(confidence, 1),
        "davis_clamp_estimate_ready": bool(confidence >= 70.0),
        "davis_ready_to_close_approximate": bool(confidence >= 70.0 and alignment >= 70.0),
        "davis_visual_contact_estimate": "image_space_approximation",
    }


class DavisGraspPlanner:
    def __init__(self, roi_xywh=None, sensitivity="balanced", display_mode=None):
        self.roi_xywh = _clip_roi(roi_xywh)
        self.manual_roi_provided = roi_xywh is not None
        self.sensitivity = sensitivity
        self.display_mode = (display_mode or DAVIS_APS_DISPLAY_MODE or "auto").strip().lower()
        if self.display_mode not in DAVIS_APS_DISPLAY_MODES:
            self.display_mode = "auto"
        self.monitor = EventPostGraspMonitor(enabled=True, roi_xywh=self.roi_xywh, sensitivity=sensitivity)
        self.events = np.empty((0, 4), dtype=np.float64)
        self.stage_history = []
        self.rate_history = []
        self.last_aps_frame = None
        self.last_aps_info = aps_channel_info(None)
        self.last_aps_debug_images = {}
        self.last_detection_debug_images = {}
        self.sam2_segmenter = DavisSAM2Segmenter()
        self.sam2_prompt_point = None
        self.sam2_requested = False
        self.sam2_last_detection = None
        self.sam2_mask = None
        self.sam2_used = False
        self.sam2_available, self.sam2_unavailable_reason = sam2_status()
        self.opencv_low_confidence_frames = 0
        self.previous_detection_bbox = None
        self.previous_detection_confidence = 0.0
        self.previous_detection_center = None
        self.stable_detection_frames = 0
        self.lost_detection_frames = 0
        self.frame_id = 0
        self.analysis_frame_id = 0
        self.last_analysis_time = 0.0
        self.last_event_update_time = 0.0
        self.last_report_write_time = 0.0
        self.last_debug_save_state = None
        self.latest_gray = None
        self.latest_aps_display = None
        self.latest_heatmap = event_status_visual({"event_status": DAVIS_EVENT_STATUS_STREAM_NOT_STARTED})
        self.cached_detection = None
        self.cached_clamp = estimate_clamp_patches({})
        self.cached_metrics = None
        self.cached_event_features = compute_event_motion_features(self.events, roi_xywh=self.roi_xywh)
        self.cached_fusion = {}
        self.cached_state = STATE_SEARCH
        self.cached_decision = "SEARCH"
        self.cached_limitation = "DAVIS-only grasp planning has no metric depth or grip width."
        self.cached_aps_info = self.last_aps_info
        self.performance_profile = {
            "config": {
                "davis_fast_display_fps": DAVIS_FAST_DISPLAY_FPS,
                "davis_analysis_fps": DAVIS_ANALYSIS_FPS,
                "davis_event_update_fps": DAVIS_EVENT_UPDATE_FPS,
                "davis_full_detect_interval_frames": DAVIS_FULL_DETECT_INTERVAL_FRAMES,
                "davis_tracking_roi_expand_ratio": DAVIS_TRACKING_ROI_EXPAND_RATIO,
                "davis_report_interval_sec": DAVIS_REPORT_INTERVAL_SEC,
            },
            "last_ms": {},
            "avg_ms": {},
            "slow_stages": [],
            "display_fps": 0.0,
            "analysis_fps": 0.0,
            "event_update_fps": 0.0,
        }
        self._perf_counts = {}
        self._perf_totals = {}
        self._display_tick_times = []
        self._analysis_tick_times = []
        self._event_tick_times = []
        self.last_report = self.base_report()

    def base_report(self):
        return {
            "sensor_mode": "DAVIS346",
            "backend": self.monitor.backend_name,
            "current_state": STATE_SEARCH,
            "davis_decision": "SEARCH",
            "main_limitation": "DAVIS-only grasp planning has no metric depth or grip width.",
            "davis_limitations": list(LIMITATIONS),
            "recommended_use_cases": [
                "fast motion detection",
                "slip/spin/vibration detection",
                "impact detection",
                "post-close stability monitoring",
            ],
            "not_recommended_use_cases": [
                "metric distance",
                "grip width in mm",
                "depth-based contact model",
                "true contact force",
                "full 3D geometry",
            ],
            "event_only_stationary_limitation": "DAVIS event-only mode is motion-sensitive and cannot reliably detect a stationary bottle without APS frames.",
        }

    def start(self):
        report = self.monitor.start()
        if not self.monitor.active:
            self.last_report = {
                **self.base_report(),
                "current_state": STATE_UNAVAILABLE,
                "davis_decision": "UNAVAILABLE",
                "davis_status": report.get("davis_status"),
                "backend": report.get("davis_backend"),
                "davis_unavailable_reason": report.get("davis_unavailable_reason"),
                "aps_available": False,
                "aps_mode": "unavailable",
                "aps_display_mode": "unavailable",
                "aps_frame_shape": None,
                "aps_channel_count": 0,
                "aps_color_confirmed": False,
                "davis_bottle_detected": False,
                "davis_detection_status": DETECTION_NOT_DETECTED,
                "davis_detection_confidence": 0.0,
                "davis_detection_source": "none",
                "davis_detection_backend": "none",
                "davis_bbox_xyxy": None,
                "davis_candidate_count": 0,
                "davis_detection_candidates": [],
                "davis_candidate_bboxes": [],
                "rejection_reasons": ["davis_backend_unavailable"],
                "detection_reason": report.get("davis_unavailable_reason"),
                "event_backend_available": False,
                "event_stream_started": False,
                "event_packets_received": False,
                "event_packet_count_recent": 0,
                "event_rate": 0.0,
                "no_motion": False,
                "event_status": DAVIS_EVENT_STATUS_BACKEND_UNAVAILABLE,
                "event_stream_active": False,
                "event_stream_status": "DAVIS backend unavailable",
            }
            return self.last_report
        self.stage_history.append({"timestamp": time.time(), "state": STATE_SEARCH, "reason": "DAVIS monitor started"})
        return self.last_report

    def stop(self):
        self.monitor.stop()

    def request_sam2_segmentation(self, point=None):
        if point is not None:
            self.sam2_prompt_point = [float(point[0]), float(point[1])]
        self.sam2_requested = True

    def run_sam2_fallback(self, aps_display, opencv_detection):
        prompt_box = opencv_detection.get("davis_bbox_xyxy") if opencv_detection else None
        prompt_point = self.sam2_prompt_point
        if prompt_point is None and prompt_box:
            x1, y1, x2, y2 = prompt_box
            prompt_point = [(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0]
        result, error = self.sam2_segmenter.segment(aps_display, prompt_point=prompt_point, prompt_box=prompt_box)
        self.sam2_requested = False
        self.sam2_available = self.sam2_segmenter.available
        self.sam2_unavailable_reason = error or self.sam2_segmenter.reason
        if error:
            return None
        score = result.get("score")
        confidence = float(np.clip((score if score is not None else 0.75) * 100.0, 0.0, 100.0))
        detection = detection_from_mask(
            result.get("mask"),
            source="sam2_prompted",
            confidence=confidence,
            reason=f"SAM2 prompted segmentation from point={result.get('prompt_point')} box={result.get('prompt_box')}.",
        )
        detection["sam2_prompt_point"] = result.get("prompt_point")
        detection["sam2_prompt_box"] = result.get("prompt_box")
        detection["sam2_score"] = score
        self.sam2_last_detection = detection
        self.sam2_mask = result.get("mask")
        self.sam2_used = True
        return detection

    def apply_temporal_detection(self, detection):
        status = detection.get("davis_detection_status")
        bbox = detection.get("davis_bbox_xyxy")
        confidence = float(detection.get("davis_detection_confidence", 0.0) or 0.0)
        if status in {DETECTION_DETECTED, DETECTION_MANUAL_ROI} and bbox:
            if self.previous_detection_bbox is not None and _bbox_iou(bbox, self.previous_detection_bbox) > 0.25:
                self.stable_detection_frames += 1
            else:
                self.stable_detection_frames = 1
            self.lost_detection_frames = 0
            self.previous_detection_bbox = list(bbox)
            self.previous_detection_confidence = confidence
            self.previous_detection_center = detection.get("davis_bottle_center_px")
            detection["davis_outline_source"] = detection.get("davis_outline_source", "contour")
            detection["davis_centerline_px"] = detection.get("davis_centerline_px", _centerline_from_bbox(bbox))
            return detection
        can_temporal_track = (
            self.previous_detection_bbox is not None
            and self.stable_detection_frames >= DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK
            and self.previous_detection_confidence >= DETECTION_CONFIDENCE_DETECTED
            and self.lost_detection_frames < DAVIS_MAX_LOST_FRAMES
        )
        if can_temporal_track:
            self.lost_detection_frames += 1
            decayed_confidence = max(DETECTION_CONFIDENCE_LOW, self.previous_detection_confidence - 6.0 * self.lost_detection_frames)
            x1, y1, x2, y2 = self.previous_detection_bbox
            temporal = {
                **detection,
                "davis_bottle_detected": True,
                "davis_detection_status": DETECTION_DETECTED,
                "davis_bbox_xyxy": list(self.previous_detection_bbox),
                "davis_bottle_center_px": self.previous_detection_center or [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
                "davis_detection_confidence": round(float(decayed_confidence), 1),
                "aps_detection_source": "temporal_track",
                "davis_detection_source": "temporal_track",
                "davis_detection_backend": "temporal_track",
                "davis_outline_source": "temporal_track",
                "davis_centerline_px": _centerline_from_bbox(self.previous_detection_bbox),
                "detection_reason": f"Tracking previous automatic bottle bbox; current OpenCV confidence was {confidence:.1f}.",
                "rejection_reasons": detection.get("rejection_reasons", []),
            }
            return temporal
        self.lost_detection_frames += 1
        if self.lost_detection_frames > DAVIS_MAX_LOST_FRAMES:
            self.stable_detection_frames = 0
            self.previous_detection_bbox = None
            self.previous_detection_center = None
            self.previous_detection_confidence = 0.0
        return detection

    def record_perf(self, stage, elapsed_sec):
        ms = float(elapsed_sec * 1000.0)
        self.performance_profile["last_ms"][stage] = round(ms, 2)
        self._perf_counts[stage] = self._perf_counts.get(stage, 0) + 1
        self._perf_totals[stage] = self._perf_totals.get(stage, 0.0) + ms
        self.performance_profile["avg_ms"][stage] = round(self._perf_totals[stage] / max(self._perf_counts[stage], 1), 2)
        if ms > 100.0:
            warning = {"timestamp": time.time(), "stage": stage, "elapsed_ms": round(ms, 2)}
            self.performance_profile["slow_stages"].append(warning)
            self.performance_profile["slow_stages"] = self.performance_profile["slow_stages"][-20:]
            print(f"DAVIS slow stage: {stage} took {ms:.1f} ms", flush=True)

    def update_rate_counter(self, name, store):
        now = time.monotonic()
        store.append(now)
        cutoff = now - 2.0
        while store and store[0] < cutoff:
            store.pop(0)
        self.performance_profile[name] = round(len(store) / 2.0, 1)

    def lightweight_track_detection(self, gray):
        if gray is None or self.previous_detection_bbox is None:
            return None
        h_img, w_img = gray.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in self.previous_detection_bbox]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        pad_x = bw * DAVIS_TRACKING_ROI_EXPAND_RATIO
        pad_y = bh * DAVIS_TRACKING_ROI_EXPAND_RATIO
        rx1 = int(np.clip(x1 - pad_x, 0, w_img - 1))
        ry1 = int(np.clip(y1 - pad_y, 0, h_img - 1))
        rx2 = int(np.clip(x2 + pad_x, rx1 + 1, w_img))
        ry2 = int(np.clip(y2 + pad_y, ry1 + 1, h_img))
        crop = gray[ry1:ry2, rx1:rx2]
        tracked_bbox = [x1, y1, x2, y2]
        confidence = max(DETECTION_CONFIDENCE_DETECTED, float(self.previous_detection_confidence or 0.0) - 2.0)
        reason = "Reused stable detection bbox in lightweight ROI tracking."
        if crop.size:
            fast = _auto_contrast_u8(crop, gamma=0.85)
            blur = cv2.GaussianBlur(fast, (3, 3), 0)
            edges = cv2.Canny(blur, 45, 120)
            col_strength = edges.sum(axis=0).astype(np.float32)
            row_strength = edges.sum(axis=1).astype(np.float32)
            if np.max(col_strength) > 0 and np.max(row_strength) > 0:
                cols = np.where(col_strength >= max(np.percentile(col_strength, 72), np.max(col_strength) * 0.20))[0]
                rows = np.where(row_strength >= max(np.percentile(row_strength, 60), np.max(row_strength) * 0.12))[0]
                if cols.size >= 8 and rows.size >= 18:
                    nx1, nx2 = rx1 + int(cols.min()), rx1 + int(cols.max()) + 1
                    ny1, ny2 = ry1 + int(rows.min()), ry1 + int(rows.max()) + 1
                    candidate = [float(nx1), float(ny1), float(nx2), float(ny2)]
                    if _bbox_iou(candidate, self.previous_detection_bbox) > 0.25:
                        alpha = 0.25
                        tracked_bbox = [
                            (1.0 - alpha) * x1 + alpha * candidate[0],
                            (1.0 - alpha) * y1 + alpha * candidate[1],
                            (1.0 - alpha) * x2 + alpha * candidate[2],
                            (1.0 - alpha) * y2 + alpha * candidate[3],
                        ]
                        confidence = min(100.0, confidence + 1.0)
                        reason = "Lightweight local ROI edge tracking updated the stable bottle bbox."
        bbox = [int(round(v)) for v in tracked_bbox]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        detection = {
            "davis_bottle_detected": True,
            "davis_detection_status": DETECTION_DETECTED,
            "davis_bbox_xyxy": bbox,
            "davis_bottle_center_px": [round(cx, 1), round(cy, 1)],
            "davis_detection_confidence": round(confidence, 1),
            "aps_detection_source": "lightweight_roi_track",
            "davis_detection_source": "lightweight_roi_track",
            "davis_detection_backend": "lightweight_roi_track",
            "aps_detection_input_mode": "fast_tracking_roi",
            "davis_candidate_count": 1,
            "best_candidate_scores": {"tracking_confidence": round(confidence, 1)},
            "selected_candidate": {"bbox_xyxy": bbox, "source": "lightweight_roi_track", "confidence": round(confidence, 1)},
            "davis_detection_candidates": [],
            "davis_candidate_bboxes": [bbox],
            "rejection_reasons": [],
            "detection_reason": reason,
            "davis_contour_available": False,
            "davis_outline_source": "temporal_track",
            "davis_centerline_px": _centerline_from_bbox(bbox),
        }
        self.previous_detection_bbox = list(bbox)
        self.previous_detection_confidence = confidence
        self.previous_detection_center = detection["davis_bottle_center_px"]
        self.stable_detection_frames += 1
        self.lost_detection_frames = 0
        return detection

    def event_status_from_metrics(self, metrics, event_features=None):
        event_features = event_features or {}
        event_rate = float(event_features.get("event_rate", metrics.get("event_rate", 0.0)) or 0.0)
        if not self.monitor.active:
            return DAVIS_EVENT_STATUS_BACKEND_UNAVAILABLE, "DAVIS backend unavailable"
        if event_rate > 1.0:
            return DAVIS_EVENT_STATUS_ACTIVE_EVENTS, "Event stream active with motion events"
        return DAVIS_EVENT_STATUS_ACTIVE_NO_MOTION, "Stream active; no motion events currently"

    def maybe_write_live_status(self, report, force=False):
        now = time.monotonic()
        state_key = (report.get("current_state"), report.get("davis_decision"), report.get("davis_detection_status"), report.get("event_status"))
        state_changed = state_key != self.last_debug_save_state
        if force or now - self.last_report_write_time >= DAVIS_REPORT_INTERVAL_SEC or (DAVIS_SAVE_DEBUG_ON_STATE_CHANGE and state_changed):
            OUTPUT_DIR.mkdir(exist_ok=True)
            _write_json(REPORT_PATH, report)
            _write_json(EVENT_FEATURE_REPORT_PATH, report.get("event_features", {}))
            _write_json(PERFORMANCE_PROFILE_PATH, self.performance_profile)
            self.last_report_write_time = now
            self.last_debug_save_state = state_key

    def step(self):
        step_start = time.perf_counter()
        self.frame_id += 1
        now = time.monotonic()
        self.update_rate_counter("display_fps", self._display_tick_times)
        poll_start = time.perf_counter()
        new_events, aps_frames = self.monitor.poll_event_and_frame_arrays()
        self.record_perf("event_and_frame_poll", time.perf_counter() - poll_start)
        if new_events.size:
            self.events = np.vstack([self.events, new_events]) if self.events.size else new_events
            cutoff = time.monotonic() - 2.0
            self.events = self.events[self.events[:, 0] >= cutoff]
        latest_aps = None
        aps_array = np.asarray(aps_frames) if aps_frames is not None else np.empty((0,), dtype=np.uint8)
        if aps_array.size:
            latest_aps = aps_array[-1] if aps_array.ndim >= 3 else aps_array
            self.last_aps_frame = latest_aps
        elif self.last_aps_frame is not None:
            latest_aps = self.last_aps_frame

        display_start = time.perf_counter()
        gray_fast, aps_display = fast_aps_display_frame(latest_aps, display_mode=self.display_mode)
        self.record_perf("fast_aps_display", time.perf_counter() - display_start)
        if latest_aps is not None:
            self.latest_gray = gray_fast
            if self.cached_aps_info.get("aps_available") is not True:
                self.cached_aps_info = aps_channel_info(latest_aps)
            self.last_aps_frame = latest_aps
        aps_info = dict(self.cached_aps_info)

        event_due = now - self.last_event_update_time >= 1.0 / max(DAVIS_EVENT_UPDATE_FPS, 1.0)
        if event_due or self.cached_metrics is None:
            event_start = time.perf_counter()
            metrics = self.compute_event_metrics()
            self.cached_metrics = metrics
            self.last_event_update_time = now
            self.update_rate_counter("event_update_fps", self._event_tick_times)
            heat_start = time.perf_counter()
            self.latest_heatmap = event_heatmap(self.events, self.roi_xywh, centroid=(metrics.get("mean_x"), metrics.get("mean_y")))
            self.record_perf("event_heatmap_generation", time.perf_counter() - heat_start)
            self.record_perf("event_metrics", time.perf_counter() - event_start)
        else:
            metrics = dict(self.cached_metrics)

        analysis_due = now - self.last_analysis_time >= 1.0 / max(DAVIS_ANALYSIS_FPS, 1.0)
        detection = self.cached_detection
        clamp = self.cached_clamp
        if analysis_due or detection is None:
            analysis_start = time.perf_counter()
            self.analysis_frame_id += 1
            self.update_rate_counter("analysis_fps", self._analysis_tick_times)
            detection_debug_images = {}
            gray = self.latest_gray
            full_detect_due = (
                self.analysis_frame_id % max(DAVIS_FULL_DETECT_INTERVAL_FRAMES, 1) == 1
                or detection is None
                or detection.get("davis_detection_status") != DETECTION_DETECTED
                or self.sam2_requested
            )
            if self.manual_roi_provided:
                detection = manual_roi_detection(self.roi_xywh)
                self.stable_detection_frames = max(self.stable_detection_frames, DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK)
            elif aps_info.get("aps_available") and gray is not None:
                if full_detect_due:
                    aps_analysis_start = time.perf_counter()
                    gray_analysis, _analysis_display, aps_info, aps_debug_images = analyze_aps_frame(latest_aps, display_mode=self.display_mode)
                    self.record_perf("aps_format_analysis", time.perf_counter() - aps_analysis_start)
                    if gray_analysis is not None:
                        gray = gray_analysis
                        self.latest_gray = gray_analysis
                    self.cached_aps_info = dict(aps_info)
                    self.last_aps_info = dict(aps_info)
                    if DAVIS_SAVE_DEBUG_EVERY_FRAME:
                        self.last_aps_debug_images = aps_debug_images
                    opencv_cache = {"result": None}

                    def opencv_fallback():
                        detect_start = time.perf_counter()
                        opencv_result, debug_images = detect_bottle_from_aps(
                            gray,
                            roi_xywh=None,
                            aps_detection_input_mode=aps_info.get("aps_display_mode", "unknown"),
                            previous_bbox=self.previous_detection_bbox,
                        )
                        self.record_perf("opencv_detection", time.perf_counter() - detect_start)
                        opencv_cache["result"] = opencv_result
                        opencv_cache["debug_images"] = debug_images
                        return opencv_result

                    stack_start = time.perf_counter()
                    prompt_box = self.cached_detection.get("davis_bbox_xyxy") if self.cached_detection else None
                    stack_result = detect_bottle_aps(
                        aps_display if aps_display is not None else _analysis_display,
                        opencv_fallback=opencv_fallback,
                        sam2_segmenter=self.sam2_segmenter,
                        prompt_box=prompt_box,
                        prompt_point=self.sam2_prompt_point,
                        allow_sam2=bool(self.sam2_requested or (self.sam2_available and self.opencv_low_confidence_frames >= 3)),
                    )
                    self.record_perf("detector_stack", time.perf_counter() - stack_start)
                    opencv_detection = opencv_cache.get("result") or {}
                    detection_debug_images = opencv_cache.get("debug_images") or {}
                    self.opencv_low_confidence_frames = self.opencv_low_confidence_frames + 1 if opencv_detection.get("davis_detection_status") != DETECTION_DETECTED else 0
                    detection = planner_detection_from_stack_result(stack_result, opencv_detection)
                    self.sam2_requested = False
                    self.sam2_available = bool(detection.get("sam2_available") or self.sam2_available)
                else:
                    track_start = time.perf_counter()
                    detection = self.lightweight_track_detection(gray) or detection
                    self.record_perf("lightweight_roi_tracking", time.perf_counter() - track_start)
            else:
                detection = {
                    "davis_bottle_detected": False,
                    "davis_detection_status": DETECTION_NOT_DETECTED,
                    "davis_bbox_xyxy": None,
                    "davis_bottle_center_px": None,
                    "davis_detection_confidence": 0.0,
                    "aps_detection_source": "event_only",
                    "davis_detection_source": "events",
                    "davis_detection_backend": "events_motion_roi",
                    "aps_detection_input_mode": "events_only",
                    "detection_reason": "APS unavailable; event-only mode cannot reliably detect a stationary bottle.",
                    "davis_candidate_count": 0,
                    "best_candidate_scores": None,
                    "rejection_reasons": ["aps_unavailable"],
                }
            if not aps_info.get("aps_available") and not detection.get("davis_bottle_detected") and not self.manual_roi_provided:
                detection = event_motion_roi_detection(self.events, self.roi_xywh)
            if not self.manual_roi_provided and detection.get("davis_detection_source") not in {"lightweight_roi_track", "temporal_track"}:
                detection = self.apply_temporal_detection(detection)
            if DAVIS_SAVE_DEBUG_EVERY_FRAME:
                self.last_detection_debug_images = detection_debug_images
            elif detection_debug_images and (self.cached_detection is None or detection.get("davis_detection_status") != self.cached_detection.get("davis_detection_status")):
                self.last_detection_debug_images = detection_debug_images
            if detection.get("davis_detection_status") in {DETECTION_DETECTED, DETECTION_MANUAL_ROI}:
                roi_from_detection = bbox_xyxy_to_roi_xywh(detection.get("davis_bbox_xyxy"))
                if roi_from_detection is not None and detection.get("davis_detection_source") in {"sam2_prompted", "manual_roi", "opencv_aps"}:
                    self.roi_xywh = _clip_roi(roi_from_detection)
                    self.monitor.roi_xywh = self.roi_xywh
            clamp_start = time.perf_counter()
            clamp = estimate_clamp_patches(detection)
            self.record_perf("clamp_patch_generation", time.perf_counter() - clamp_start)
            self.cached_detection = detection
            self.cached_clamp = clamp
            self.last_analysis_time = now
            self.record_perf("analysis_total", time.perf_counter() - analysis_start)

        detection = detection or {
            "davis_bottle_detected": False,
            "davis_detection_status": DETECTION_NOT_DETECTED,
            "davis_bbox_xyxy": None,
            "davis_detection_confidence": 0.0,
            "davis_detection_source": "none",
            "davis_detection_backend": "none",
            "detection_reason": "Waiting for first DAVIS analysis update.",
        }
        clamp = clamp or estimate_clamp_patches({})
        event_feature_start = time.perf_counter()
        event_features = compute_event_motion_features(self.events, aps_bbox=detection.get("davis_bbox_xyxy"), roi_xywh=self.roi_xywh)
        fusion = fuse_davis_aps_events(detection, clamp, metrics, event_features)
        self.cached_event_features = event_features
        self.cached_fusion = fusion
        self.record_perf("event_feature_fusion", time.perf_counter() - event_feature_start)
        state, decision, limitation = self.decide(aps_info, detection, clamp, metrics, event_features, fusion)
        self.cached_state = state
        self.cached_decision = decision
        self.cached_limitation = limitation
        self.rate_history.append({"t": time.time(), "event_rate": metrics.get("event_rate", 0.0)})
        self.rate_history = self.rate_history[-300:]
        event_rate = float(event_features.get("event_rate", metrics.get("event_rate", 0.0)) or 0.0)
        event_packets_received = bool(self.events.size)
        event_status, event_stream_status = self.event_status_from_metrics(metrics, event_features)
        aps_info = dict(self.cached_aps_info)
        perception_result = build_davis_perception_result(aps_info, detection, clamp, metrics, event_features, fusion, decision)
        report = {
            **self.base_report(),
            **aps_info,
            **detection,
            **clamp,
            **metrics,
            **event_features,
            **fusion,
            "backend": self.monitor.backend_name,
            "davis_status": "streaming" if self.monitor.active else "unavailable",
            "current_state": state,
            "davis_decision": decision,
            "main_limitation": limitation,
            "roi_xywh": self.roi_xywh,
            "timestamp": time.time(),
            "manual_roi_active": bool(self.manual_roi_provided and detection.get("davis_detection_source") == "manual_roi"),
            "bottle_detected_by_davis": detection.get("davis_bottle_detected"),
            "detection_confidence": detection.get("davis_detection_confidence"),
            "detection_status": detection.get("davis_detection_status"),
            "candidate_scores": detection.get("best_candidate_scores"),
            "bbox_xyxy": detection.get("davis_bbox_xyxy"),
            "contour_source": detection.get("davis_outline_source"),
            "event_backend_available": bool(self.monitor.active),
            "event_stream_started": bool(self.monitor.active),
            "event_packets_received": event_packets_received,
            "event_packet_count_recent": int(self.events.shape[0]) if self.events.ndim == 2 else 0,
            "event_rate": event_rate,
            "no_motion": bool(self.monitor.active and event_rate <= 1.0),
            "event_status": event_status,
            "event_stream_active": bool(event_status == DAVIS_EVENT_STATUS_ACTIVE_EVENTS),
            "event_stream_status": event_stream_status,
            "detection_backend_used": detection.get("davis_detection_backend"),
            "detector_source": detection.get("detector_source", detection.get("davis_detection_source")),
            "yolo_available": bool(detection.get("yolo_available", yolo_status()[0])),
            "yolo_model_path": str(DAVIS_YOLO_MODEL_PATH),
            "use_davis_yolo_detector": bool(USE_DAVIS_YOLO_DETECTOR),
            "opencv_confidence": detection.get("opencv_confidence", detection.get("davis_detection_confidence") if detection.get("davis_detection_backend") in {"opencv_aps", "lightweight_roi_track", "temporal_track"} else None),
            "sam2_available": bool(detection.get("sam2_available", self.sam2_available)),
            "sam2_used": bool(detection.get("davis_detection_source") == "sam2_prompted"),
            "sam2_requested": bool(self.sam2_requested),
            "sam2_unavailable_reason": self.sam2_unavailable_reason,
            "manual_roi_used": bool(detection.get("davis_detection_source") == "manual_roi"),
            "final_detection_status": detection.get("davis_detection_status"),
            "final_confidence": detection.get("davis_detection_confidence"),
            "final_bbox": detection.get("davis_bbox_xyxy"),
            "final_mask_source": detection.get("final_mask_source", detection.get("davis_detection_source")),
            "bbox_xyxy": detection.get("davis_bbox_xyxy"),
            "clamp_patches": {
                "left": clamp.get("left_clamp_patch_px"),
                "right": clamp.get("right_clamp_patch_px"),
                "grasp_line": clamp.get("grasp_line_px"),
            },
            "decision": decision,
            "reason": detection.get("detection_reason"),
            "previous_detection_bbox": self.previous_detection_bbox,
            "previous_detection_confidence": self.previous_detection_confidence,
            "stable_detection_frames": self.stable_detection_frames,
            "lost_detection_frames": self.lost_detection_frames,
            "davis_detection_stable_frames_to_lock": DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK,
            "davis_max_lost_frames": DAVIS_MAX_LOST_FRAMES,
            "event_only_stationary_limitation": "DAVIS event-only mode is motion-sensitive and cannot reliably detect a stationary bottle without APS frames.",
            "davis_perception_result": perception_result,
            "aps_detection": {
                "status": detection.get("davis_detection_status"),
                "confidence": detection.get("davis_detection_confidence"),
                "bbox_xyxy": detection.get("davis_bbox_xyxy"),
                "source": detection.get("davis_detection_source"),
                "candidate_scores": detection.get("best_candidate_scores"),
            },
            "event_features": event_features,
            "aps_event_overlap": fusion.get("aps_event_overlap_score"),
            "combined_confidence": fusion.get("davis_combined_confidence"),
            "event_assisted_tracking": fusion.get("event_assisted_tracking"),
            "explanation": fusion.get("fusion_explanation"),
            "frame_id": self.frame_id,
            "analysis_frame_id": self.analysis_frame_id,
            "davis_fast_display_fps": DAVIS_FAST_DISPLAY_FPS,
            "davis_analysis_fps_target": DAVIS_ANALYSIS_FPS,
            "davis_event_update_fps_target": DAVIS_EVENT_UPDATE_FPS,
            "display_fps": self.performance_profile.get("display_fps", 0.0),
            "analysis_fps": self.performance_profile.get("analysis_fps", 0.0),
            "event_update_fps": self.performance_profile.get("event_update_fps", 0.0),
            "last_analysis_age_ms": round((time.monotonic() - self.last_analysis_time) * 1000.0, 1) if self.last_analysis_time else None,
            "performance_profile": self.performance_profile,
        }
        if self.latest_heatmap is not None:
            self.latest_heatmap = draw_event_feature_overlay(self.latest_heatmap.copy(), event_features)
        self.last_report = report
        self.stage_history.append({"timestamp": report["timestamp"], "state": state, "decision": decision, "reason": limitation})
        if aps_display is not None:
            overlay_start = time.perf_counter()
            self.draw_aps_overlay(aps_display, report)
            self.record_perf("overlay_draw", time.perf_counter() - overlay_start)
        self.latest_aps_display = aps_display
        self.maybe_write_live_status(report)
        self.record_perf("step_total", time.perf_counter() - step_start)
        return report

    def compute_event_metrics(self):
        x, y, w, h = self.roi_xywh
        events = self.events
        if events.size:
            in_roi = (events[:, 1] >= x) & (events[:, 1] < x + w) & (events[:, 2] >= y) & (events[:, 2] < y + h)
            roi_events = events[in_roi]
        else:
            roi_events = np.empty((0, 4), dtype=np.float64)
        metrics = self.monitor._metrics(roi_events, 1.2, w, h)
        metrics["motion_class"] = self.live_motion_class(metrics)
        metrics["stability_score"] = float(np.clip(100.0 - metrics["motion_spike_score"], 0.0, 100.0))
        metrics["slip_risk_score"] = float(np.clip(max(metrics["vertical_event_bias"], 0.0) * metrics["motion_spike_score"], 0.0, 100.0))
        metrics["rotation_risk_score"] = float(np.clip(metrics["left_right_event_asymmetry"] * metrics["motion_spike_score"], 0.0, 100.0))
        metrics["vibration_risk_score"] = float(np.clip(metrics["event_density"] * 800.0, 0.0, 100.0))
        metrics["confidence_score"] = float(np.clip(metrics["event_count"] / 5.0, 0.0, 100.0))
        return metrics

    def live_motion_class(self, metrics):
        if metrics["event_rate"] < 120.0:
            return "STABLE"
        if metrics["vertical_event_bias"] > 0.28:
            return "SLIP_DOWN"
        if metrics["left_right_event_asymmetry"] > 0.35:
            return "SPIN_OR_ROTATION"
        if metrics["event_density"] > 0.01:
            return "SHAKE_OR_VIBRATION"
        return "UNKNOWN_MOTION"

    def decide(self, aps_info, detection, clamp, metrics, event_features=None, fusion=None):
        event_features = event_features or {}
        fusion = fusion or {}
        event_rate = float(event_features.get("event_rate", metrics.get("event_rate", 0.0)) or 0.0)
        aps_event_agreement = fusion.get("aps_event_agreement", "no_motion_no_event_penalty")
        combined_confidence = float(fusion.get("davis_combined_confidence", detection.get("davis_detection_confidence", 0.0)) or 0.0)
        if not self.monitor.active:
            return STATE_UNAVAILABLE, "UNAVAILABLE", "DAVIS backend or device unavailable."
        if not aps_info["aps_available"]:
            if event_features.get("event_count", metrics.get("event_count", 0)) > 30:
                return STATE_TRACK, "EVENT_ONLY_TRACKING", "APS unavailable; tracking moving activity only, not stationary bottle geometry."
            return STATE_SEARCH, "SEARCH", "No APS frame; event-only mode needs motion and cannot confirm a stationary bottle."
        detection_status = detection.get("davis_detection_status", DETECTION_NOT_DETECTED)
        if detection_status not in {DETECTION_DETECTED, DETECTION_MANUAL_ROI} and event_features.get("event_motion_confidence", 0.0) >= 35.0:
            return STATE_ACQUIRE, "TRACK_OBJECT_LOW_CONFIDENCE", "DVS events show motion, but APS does not yet confirm the known bottle."
        if detection_status == DETECTION_LOW_CONFIDENCE:
            return STATE_SEARCH, "IMPROVE_APS_VIEW", "Bottle cue exists, but confidence is low. Improve lighting/position or inspect candidate debug output."
        if detection_status not in {DETECTION_DETECTED, DETECTION_MANUAL_ROI}:
            return STATE_SEARCH, "SEARCH_BOTTLE", detection.get("detection_reason", "Bottle not detected in APS.")
        if self.stable_detection_frames < DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK and detection_status != DETECTION_MANUAL_ROI:
            return STATE_ACQUIRE, "TRACK_TARGET", f"Automatic bottle candidate found; waiting for stable frames {self.stable_detection_frames}/{DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK}."
        if clamp["davis_contact_confidence"] < 45.0:
            return STATE_ACQUIRE, "ADJUST_VIEW", "APS bottle cue exists, but clamp estimate confidence is low."
        if aps_event_agreement == "event_motion_outside_aps_target":
            return STATE_VERIFY, "WAIT_FOR_STABILITY", "Motion events are outside the APS bottle target; wait or adjust view before closing."
        if event_rate > 2200.0:
            return STATE_VERIFY, "HOLD_STILL", "High DAVIS event activity; wait for motion to settle before approximate close."
        if event_rate >= DAVIS_PRE_CLOSE_MOTION_RATE_HOLD and aps_event_agreement == "aps_plus_events_agree":
            return STATE_TRACK, "TRACKING_WAIT_FOR_STABILITY", "APS and events agree on the target, but the target/camera is still moving."
        if clamp["davis_alignment_score"] >= 70.0 and event_rate < DAVIS_PRE_CLOSE_MOTION_RATE_HOLD and combined_confidence >= DETECTION_CONFIDENCE_DETECTED:
            return STATE_READY, "READY_TO_CLOSE_APPROXIMATE", "APS target is stable enough and event activity is quiet; depth/grip width still unknown."
        return STATE_TRACK, "CENTER_OR_STABILIZE", "DAVIS can guide image alignment, but RealSense is needed for metric geometry."

    def draw_aps_overlay(self, image, report):
        x, y, w, h = _clip_roi(self.roi_xywh)
        if self.sam2_mask is not None:
            mask = self.sam2_mask.astype(bool)
            if mask.shape[:2] == image.shape[:2]:
                overlay = image.copy()
                overlay[mask] = (0, 120, 255)
                cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 1)
        cv2.line(image, (x + w // 2, y), (x + w // 2, y + h - 1), (0, 180, 255), 1)
        cv2.line(image, (x, y + h // 2), (x + w - 1, y + h // 2), (0, 180, 255), 1)
        for candidate in report.get("davis_detection_candidates", [])[1:8]:
            bbox_candidate = candidate.get("bbox_xyxy")
            if not bbox_candidate:
                continue
            cx1, cy1, cx2, cy2 = [int(round(v)) for v in bbox_candidate]
            cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (120, 120, 120), 1)
        bbox = report.get("davis_bbox_xyxy")
        if bbox:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            status = report.get("davis_detection_status", DETECTION_NOT_DETECTED)
            if status == DETECTION_MANUAL_ROI:
                color = (0, 255, 255)
            elif status == DETECTION_DETECTED:
                color = (0, 255, 0)
            elif status == DETECTION_LOW_CONFIDENCE:
                color = (0, 165, 255)
            else:
                color = (120, 120, 120)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, f"{status} {report.get('davis_detection_confidence', 0):.0f}",
                        (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            centerline = report.get("davis_centerline_px")
            if centerline:
                (cx1, cy1), (cx2, cy2) = centerline
                cv2.line(image, (int(round(cx1)), int(round(cy1))), (int(round(cx2)), int(round(cy2))), color, 1)
            if report.get("davis_outline_source") in {"bbox_approximation", "temporal_track"}:
                cx = int(round((x1 + x2) / 2.0))
                cv2.ellipse(image, (cx, int(round((y1 + y2) / 2.0))), (max(3, int((x2 - x1) / 2)), max(5, int((y2 - y1) / 2))), 0, 0, 360, color, 1)
        motion_roi = report.get("moving_object_roi_xyxy")
        if motion_roi:
            x1, y1, x2, y2 = [int(round(v)) for v in motion_roi]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(image, "motion_roi only", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        event_bbox = report.get("event_motion_bbox")
        if event_bbox:
            x1, y1, x2, y2 = [int(round(v)) for v in event_bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(image, "events", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        event_centroid = report.get("event_centroid")
        if event_centroid:
            cx, cy = [int(round(v)) for v in event_centroid]
            cv2.drawMarker(image, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        for key, color in [("davis_clamp_patch_left", (255, 180, 0)), ("davis_clamp_patch_right", (255, 180, 0))]:
            patch = report.get(key)
            if patch:
                x1, y1, x2, y2 = [int(round(v)) for v in patch]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        grasp_line = report.get("grasp_line_px")
        if grasp_line:
            p1, p2 = grasp_line
            cv2.line(image, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), (255, 220, 0), 1)
        cv2.putText(image, f"{report.get('aps_mode', 'unknown')} | {report.get('davis_detection_status', DETECTION_NOT_DETECTED)}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(image, f"source={report.get('detector_source', report.get('davis_detection_source', 'none'))}",
                    (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    def save_outputs(self, report=None):
        report = report or self.last_report
        OUTPUT_DIR.mkdir(exist_ok=True)
        _write_json(REPORT_PATH, report)
        _write_json(HISTORY_PATH, self.stage_history[-200:])
        _write_json(PERFORMANCE_PROFILE_PATH, self.performance_profile)
        _write_json(EVENT_FEATURE_REPORT_PATH, report.get("event_features", {}))
        heat = getattr(self, "latest_heatmap", event_heatmap(self.events, self.roi_xywh))
        event_panel = heat if report.get("event_stream_active") else event_status_visual(report)
        cv2.imwrite(str(OUTPUT_DIR / "davis_event_activity.png"), event_panel)
        cv2.imwrite(str(OUTPUT_DIR / "davis_event_heatmap.png"), event_panel)
        aps = getattr(self, "latest_aps_display", None)
        raw_aps = self.last_aps_frame
        debug_images = getattr(self, "last_aps_debug_images", {})
        save_aps_visual_debug(raw_aps, aps, debug_images)
        save_detection_debug(getattr(self, "last_detection_debug_images", {}))
        if aps is not None:
            cv2.imwrite(str(OUTPUT_DIR / "davis_detection_overlay.png"), aps)
        else:
            placeholder = status_placeholder_image("APS unavailable", report.get("davis_unavailable_reason") or report.get("main_limitation") or "No APS frame was captured.")
            cv2.imwrite(str(OUTPUT_DIR / "davis_detection_overlay.png"), placeholder)
        if not (OUTPUT_DIR / "davis_detection_candidates.png").exists():
            placeholder = status_placeholder_image("No APS candidates", report.get("detection_reason") or "Candidate scoring did not run because no APS frame was available.")
            cv2.imwrite(str(OUTPUT_DIR / "davis_detection_candidates.png"), placeholder)
        cv2.imwrite(str(OUTPUT_DIR / "davis_latest_event_heatmap.png"), event_panel)
        _write_json(OUTPUT_DIR / "davis_detection_debug.json", {
            "selected_bbox": report.get("davis_bbox_xyxy"),
            "selected_candidate": report.get("selected_candidate"),
            "candidate_scores": report.get("candidate_scores"),
            "candidate_count": report.get("davis_candidate_count"),
            "candidate_bboxes": report.get("davis_candidate_bboxes"),
            "candidate_details": report.get("davis_detection_candidates"),
            "rejection_reasons": report.get("rejection_reasons"),
            "final_selected_reason": report.get("detection_reason"),
            "detection_status": report.get("davis_detection_status"),
            "detection_confidence": report.get("davis_detection_confidence"),
            "detection_source": report.get("davis_detection_source"),
            "detector_source": report.get("detector_source"),
            "yolo_available": report.get("yolo_available"),
            "yolo_model_path": report.get("yolo_model_path"),
            "sam2_available": report.get("sam2_available"),
            "opencv_confidence": report.get("opencv_confidence"),
            "outline_source": report.get("davis_outline_source"),
        })
        _write_json(OUTPUT_DIR / "davis_event_status_debug.json", {
            "event_backend_available": report.get("event_backend_available"),
            "event_stream_started": report.get("event_stream_started"),
            "event_packets_received": report.get("event_packets_received"),
            "event_packet_count_recent": report.get("event_packet_count_recent"),
            "event_rate": report.get("event_rate"),
            "event_status": report.get("event_status"),
            "event_stream_status": report.get("event_stream_status"),
            "no_motion": report.get("no_motion"),
            "backend": report.get("backend"),
            "event_motion_bbox": report.get("event_motion_bbox"),
            "event_centroid": report.get("event_centroid"),
            "event_motion_confidence": report.get("event_motion_confidence"),
            "aps_event_overlap_score": report.get("aps_event_overlap_score"),
            "aps_event_agreement": report.get("aps_event_agreement"),
        })
        _write_json(FRAME_DEBUG_PATH, {
            "aps_available": report.get("aps_available"),
            "raw_aps_frame_type": report.get("raw_aps_frame_type"),
            "aps_shape": report.get("aps_frame_shape"),
            "aps_dtype": report.get("aps_frame_dtype"),
            "aps_min_value": report.get("aps_min_value"),
            "aps_max_value": report.get("aps_max_value"),
            "aps_ndim": report.get("aps_ndim"),
            "aps_channel_count": report.get("aps_channel_count"),
            "aps_mode": report.get("aps_mode"),
            "aps_display_mode": report.get("aps_display_mode"),
            "aps_display_mode_requested": report.get("aps_display_mode_requested"),
            "aps_color_confirmed": report.get("aps_color_confirmed"),
            "aps_format_assessment": report.get("aps_format_assessment"),
            "aps_true_color_reason": report.get("aps_true_color_reason"),
            "aps_channel_behavior": report.get("aps_channel_behavior"),
            "aps_colorfulness_score": report.get("aps_colorfulness_score"),
            "aps_display_candidate_scores": report.get("aps_display_candidate_scores"),
            "aps_bayer_pattern_config": report.get("aps_bayer_pattern_config"),
            "aps_bayer_pattern_selected": report.get("aps_bayer_pattern_selected"),
            "aps_debayer_scores": report.get("aps_debayer_scores"),
            "aps_debayer_source": report.get("aps_debayer_source"),
            "aps_display_note": report.get("aps_display_note"),
            "event_count": report.get("event_count"),
            "event_rate": report.get("event_rate"),
            "backend": report.get("backend"),
            "davis_detection_source": report.get("davis_detection_source"),
            "davis_bottle_detected": report.get("davis_bottle_detected"),
            "davis_detection_status": report.get("davis_detection_status"),
            "davis_detection_confidence": report.get("davis_detection_confidence"),
            "davis_candidate_count": report.get("davis_candidate_count"),
            "aps_detection_input_mode": report.get("aps_detection_input_mode"),
            "event_stream_active": report.get("event_stream_active"),
            "event_stream_status": report.get("event_stream_status"),
            "event_status": report.get("event_status"),
        })
        self.write_dashboard(report, aps, heat)
        self.write_event_timeline(report)

    def raw_aps_png(self, raw_aps):
        return raw_aps_png(raw_aps)

    def write_dashboard(self, report, aps, heat):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
        if aps is None:
            axes[0].text(0.5, 0.5, "APS unavailable", ha="center", va="center")
            axes[0].set_axis_off()
        else:
            axes[0].imshow(cv2.cvtColor(aps, cv2.COLOR_BGR2RGB))
            axes[0].set_axis_off()
        axes[0].set_title(f"APS: {report.get('aps_mode', 'unknown')}")
        axes[1].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Events: {report.get('event_rate', 0):.0f} ev/s")
        axes[1].set_axis_off()
        fig.suptitle(f"{report.get('current_state')} | {report.get('davis_decision')}")
        fig.tight_layout()
        fig.savefig(DASHBOARD_PATH)
        fig.savefig(OUTPUT_DIR / "davis_professional_dashboard.png")
        plt.close(fig)

    def write_event_timeline(self, report):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        history = self.rate_history or [{"t": time.time(), "event_rate": report.get("event_rate", 0.0)}]
        t0 = history[0]["t"]
        xs = [item["t"] - t0 for item in history]
        ys = [item["event_rate"] for item in history]
        fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=120)
        ax.plot(xs, ys, color="#2563eb", linewidth=1.6)
        ax.set_title("DAVIS Event Rate Timeline")
        ax.set_xlabel("seconds")
        ax.set_ylabel("events / second")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "davis_event_timeline.png")
        plt.close(fig)


def diagnose(duration=1.0, roi_xywh=None, sensitivity="balanced", save_debug=False, display_mode=None):
    backends = _backend_import_status()
    devices, usb_error = _detect_davis_usb_devices()
    monitor = EventPostGraspMonitor(enabled=True, roi_xywh=roi_xywh, sensitivity=sensitivity)
    report = monitor.start()
    aps_info = aps_channel_info(None)
    aps_debug_images = {}
    event_available = False
    total_events = 0
    latest_frame = None
    latest_heat = None
    imu_available = None
    if monitor.active:
        deadline = time.monotonic() + max(float(duration), 0.1)
        while time.monotonic() < deadline:
            events, frames = monitor.poll_event_and_frame_arrays()
            if events.size:
                event_available = True
                total_events += int(events.shape[0])
            arr = np.asarray(frames) if frames is not None else np.empty((0,), dtype=np.uint8)
            if arr.size:
                frame = arr[-1] if arr.ndim >= 3 else arr
                latest_frame = frame
                _gray, _display, aps_info, aps_debug_images = analyze_aps_frame(frame, display_mode=display_mode)
            time.sleep(0.02)
        latest_heat = event_heatmap(monitor.poll_event_array(), roi_xywh)
    monitor.stop()
    event_rate = total_events / max(float(duration), 1e-6)
    if aps_info["aps_available"]:
        print(
            f"DAVIS APS frame: shape={aps_info['aps_frame_shape']} dtype={aps_info['aps_frame_dtype']} "
            f"min={aps_info['aps_min_value']} max={aps_info['aps_max_value']} "
            f"ndim={aps_info['aps_ndim']} channel_count={aps_info['aps_channel_count']} "
            f"mode=\"{aps_info['aps_mode']}\" display=\"{aps_info.get('aps_display_mode')}\" "
            f"color_confirmed={aps_info.get('aps_color_confirmed', False)}",
            flush=True,
        )
        if aps_info.get("aps_channel_behavior"):
            print(f"Channel behavior: {json.dumps(_json_safe(aps_info['aps_channel_behavior']))}", flush=True)
        if aps_info.get("aps_display_note"):
            print(f"APS note: {aps_info['aps_display_note']}", flush=True)
        if aps_info.get("aps_display_candidate_scores"):
            print(f"Display candidate colorfulness: {json.dumps(_json_safe(aps_info['aps_display_candidate_scores']))}", flush=True)
        print(f"Selected APS display mode: {aps_info.get('aps_display_mode')}", flush=True)
    else:
        print("DAVIS APS frame: unavailable", flush=True)
    diagnosis = {
        "davis_usb_device_found": bool(devices),
        "usb_devices": devices,
        "usb_error": usb_error,
        "permission_status": "ok" if any(d.get("permission_ok") for d in devices) else "not_ok_or_unknown",
        "backend_availability": backends,
        "selected_backend": report.get("davis_backend"),
        "backend_opened": bool(report.get("davis_backend_available")),
        "event_stream_available": event_available,
        "event_count_1s": total_events,
        "event_rate_hz": event_rate,
        "imu_available": imu_available,
        "camera_color_assessment": aps_info["aps_mode"],
        **aps_info,
        "open_error": report.get("davis_unavailable_reason"),
    }
    if save_debug:
        OUTPUT_DIR.mkdir(exist_ok=True)
        if latest_frame is not None:
            _gray, display, _info, aps_debug_images = analyze_aps_frame(latest_frame, display_mode=display_mode)
            save_aps_visual_debug(latest_frame, display, aps_debug_images)
        if latest_heat is not None:
            cv2.imwrite(str(OUTPUT_DIR / "davis_latest_event_heatmap.png"), latest_heat)
        _write_json(FRAME_DEBUG_PATH, {
            **aps_info,
            "aps_shape": aps_info.get("aps_frame_shape"),
            "aps_dtype": aps_info.get("aps_frame_dtype"),
            "event_count": total_events,
            "event_rate": event_rate,
            "backend": report.get("davis_backend"),
            "event_stream_available": event_available,
        })
    print(json.dumps(_json_safe(diagnosis), indent=4))
    return diagnosis


def run_live(duration=5.0, roi_xywh=None, live_print=False, save_report=False, sensitivity="balanced", display_mode=None):
    planner = DavisGraspPlanner(roi_xywh=roi_xywh, sensitivity=sensitivity, display_mode=display_mode)
    report = planner.start()
    deadline = time.monotonic() + max(float(duration), 0.0)
    next_print = 0.0
    try:
        while planner.monitor.active and time.monotonic() < deadline:
            report = planner.step()
            now = time.monotonic()
            if live_print and now >= next_print:
                print(
                    "state={current_state} decision={davis_decision} aps={aps_mode} "
                    "detected={davis_bottle_detected} rate={event_rate:.0f}/s motion={motion_class} "
                    "stable={stability_score:.0f} limit={main_limitation}".format(**report),
                    flush=True,
                )
                next_print = now + 0.25
            time.sleep(0.02)
    finally:
        planner.stop()
    planner.save_outputs(report)
    if save_report:
        print(json.dumps(_json_safe(report), indent=4))
    return report


class DavisGraspPlannerWindow:
    def __init__(self, duration=0.0, roi_xywh=None, sensitivity="balanced", display_mode=None):
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

        self.Qt = Qt
        self.QImage = QImage
        self.QPixmap = QPixmap
        self.app = QApplication.instance() or QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("DAVIS346 Grasp Planner Demo")
        self.window.resize(1360, 860)
        self.window.setStyleSheet("""
            QMainWindow, QWidget { background: #eef2f7; color: #0f172a; font-family: Inter, Arial, sans-serif; }
            QLabel { color: #0f172a; }
            QPushButton { padding: 8px 12px; border-radius: 5px; background: #2563eb; color: white; font-weight: 600; }
            QPushButton:hover { background: #1d4ed8; }
        """)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        self.window.setCentralWidget(central)
        self.top = QGridLayout()
        self.top.setSpacing(10)
        self.labels = {}
        for idx, title in enumerate(["Detection Status", "APS Confidence", "Event Status", "Event Confidence", "Combined Confidence", "Decision"]):
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #475569; font-size: 11px; font-weight: 700;")
            value = QLabel("starting")
            value.setWordWrap(True)
            value.setMinimumHeight(66)
            value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value.setStyleSheet(self.card_style("neutral"))
            self.top.addWidget(title_label, 0, idx)
            self.top.addWidget(value, 1, idx)
            self.labels[title] = value
        layout.addLayout(self.top)
        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        self.aps_label = QLabel("DAVIS APS visual estimate")
        self.aps_label.mousePressEvent = self.aps_clicked
        self.event_label = QLabel("Event heatmap")
        self.metrics_label = QLabel("Decision and interpretation")
        self.limit_label = QLabel(self.limitation_text({}))
        self.aps_label.setMinimumSize(540, 395)
        self.event_label.setMinimumSize(420, 395)
        self.metrics_label.setMinimumSize(330, 395)
        self.limit_label.setMinimumSize(1120, 130)
        for widget in [self.aps_label, self.event_label]:
            widget.setAlignment(Qt.AlignCenter)
            widget.setWordWrap(True)
            widget.setStyleSheet("background: #0b1020; color: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px;")
        for widget in [self.metrics_label, self.limit_label]:
            widget.setWordWrap(True)
            widget.setStyleSheet("background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px;")
        self.grid.addWidget(self.aps_label, 0, 0)
        self.grid.addWidget(self.event_label, 0, 1)
        self.grid.addWidget(self.metrics_label, 0, 2)
        self.grid.addWidget(self.limit_label, 1, 0, 1, 3)
        layout.addLayout(self.grid)
        self.sam2_button = QPushButton("Segment with SAM2")
        self.sam2_button.clicked.connect(self.segment_with_sam2)
        layout.addWidget(self.sam2_button)
        self.planner = DavisGraspPlanner(roi_xywh=roi_xywh, sensitivity=sensitivity, display_mode=display_mode)
        if not self.planner.sam2_available:
            self.sam2_button.setText("SAM2 unavailable")
            self.sam2_button.setEnabled(False)
        self.report = self.planner.start()
        self.started = time.monotonic()
        self.duration = float(duration or 0.0)
        self.pending_prompt_point = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(max(1, int(1000.0 / max(DAVIS_FAST_DISPLAY_FPS, 1.0))))

    def card_style(self, state="neutral"):
        palette = {
            "good": ("#ecfdf5", "#16a34a", "#14532d"),
            "warn": ("#fff7ed", "#f97316", "#7c2d12"),
            "bad": ("#fef2f2", "#dc2626", "#7f1d1d"),
            "neutral": ("#f8fafc", "#94a3b8", "#0f172a"),
            "info": ("#eff6ff", "#2563eb", "#1e3a8a"),
        }
        bg, border, fg = palette.get(state, palette["neutral"])
        return (
            f"padding: 10px; border: 1px solid {border}; border-radius: 6px; "
            f"background: {bg}; color: {fg}; font-size: 15px; font-weight: 700;"
        )

    def set_card(self, title, text, state="neutral"):
        label = self.labels.get(title)
        if label is not None:
            label.setText(str(text))
            label.setStyleSheet(self.card_style(state))

    def limitation_text(self, report):
        debug = []
        if report:
            debug = [
                f"Detection source: {report.get('davis_detection_source', 'none')}",
                f"Selected bbox: {report.get('davis_bbox_xyxy')}",
                f"Event/APS agreement: {report.get('aps_event_agreement', 'N/A')}",
                f"Candidates: {report.get('davis_candidate_count', 0)}",
                f"Stable frames: {report.get('stable_detection_frames', 0)}/{DAVIS_DETECTION_STABLE_FRAMES_TO_LOCK}",
                f"Reason: {report.get('detection_reason', 'N/A')}",
            ]
        limits = [
            "No metric depth",
            "No metric grip width",
            "No force or pressure sensing",
            "Events require motion",
            "APS detection depends on lighting and contrast",
            "Clamp patches are image-space estimates only",
        ]
        return "Limitations\n- " + "\n- ".join(limits) + ("\n\nDebug summary\n- " + "\n- ".join(debug) if debug else "")

    def pixmap(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        image = self.QImage(rgb.data, w, h, ch * w, self.QImage.Format_RGB888).copy()
        return self.QPixmap.fromImage(image).scaled(560, 410, self.Qt.KeepAspectRatio, self.Qt.SmoothTransformation)

    def aps_clicked(self, event):
        shape = self.report.get("aps_frame_shape") or [DAVIS_FRAME_SIZE[1], DAVIS_FRAME_SIZE[0]]
        if len(shape) < 2:
            return
        frame_h, frame_w = float(shape[0]), float(shape[1])
        label_w = max(float(self.aps_label.width()), 1.0)
        label_h = max(float(self.aps_label.height()), 1.0)
        x = np.clip(event.pos().x() * frame_w / label_w, 0, frame_w - 1)
        y = np.clip(event.pos().y() * frame_h / label_h, 0, frame_h - 1)
        self.pending_prompt_point = [float(x), float(y)]
        self.limit_label.setText(f"SAM2 prompt point selected: ({x:.0f}, {y:.0f})\nClick 'Segment with SAM2' to run prompted segmentation once.")

    def segment_with_sam2(self):
        self.planner.request_sam2_segmentation(self.pending_prompt_point)
        if self.pending_prompt_point is None:
            self.limit_label.setText("SAM2 requested with approximate OpenCV bbox/center prompt. Click APS image first for a point prompt.")
        else:
            self.limit_label.setText(f"SAM2 requested at point ({self.pending_prompt_point[0]:.0f}, {self.pending_prompt_point[1]:.0f}).")

    def tick(self):
        if self.planner.monitor.active:
            self.report = self.planner.step()
        report = self.report
        detection_status = detection_status_label(report)
        event_status = event_status_label(report)
        confidence = float(report.get("aps_confidence", report.get("davis_detection_confidence", 0.0)) or 0.0)
        event_confidence = float(report.get("event_confidence", report.get("event_motion_confidence", 0.0)) or 0.0)
        combined_confidence = float(report.get("davis_combined_confidence", confidence) or 0.0)
        self.set_card("Detection Status", detection_status, "good" if detection_status == "DETECTED" else "warn" if detection_status == "LOW_CONFIDENCE" else "bad")
        self.set_card("APS Confidence", f"{confidence:.0f}/100", "good" if confidence >= 70 else "warn" if confidence >= 40 else "bad")
        self.set_card("Event Status", f"{event_status}\n{float(report.get('event_rate', 0.0) or 0.0):.0f} ev/s", "good" if report.get("event_stream_active") else "neutral" if report.get("event_stream_started") else "bad")
        self.set_card("Event Confidence", f"{event_confidence:.0f}/100", "good" if event_confidence >= 60 else "info" if event_confidence > 0 else "neutral")
        self.set_card("Combined Confidence", f"{combined_confidence:.0f}/100", "good" if combined_confidence >= 70 else "warn" if combined_confidence >= 40 else "bad")
        decision = str(report.get("davis_decision", "N/A"))
        self.set_card("Decision", decision, "good" if "READY" in decision else "warn" if decision in {"IMPROVE_APS_VIEW", "CENTER_OR_STABILIZE", "TRACK_TARGET"} else "bad" if decision in {"SEARCH_BOTTLE", "UNAVAILABLE"} else "neutral")
        aps = getattr(self.planner, "latest_aps_display", None)
        heat = getattr(self.planner, "latest_heatmap", event_heatmap(self.planner.events, self.planner.roi_xywh))
        if aps is not None:
            self.aps_label.clear()
            self.aps_label.setPixmap(self.pixmap(aps))
        else:
            self.aps_label.clear()
            self.aps_label.setText("APS unavailable\nEvent-only mode cannot reliably detect a stationary bottle.")
        self.event_label.clear()
        if report.get("event_stream_active"):
            self.event_label.setPixmap(self.pixmap(heat))
        else:
            self.event_label.setPixmap(self.pixmap(event_status_visual(report)))
        clamp_ready = "Ready" if report.get("davis_clamp_estimate_ready") or report.get("davis_ready_to_close_approximate") else "Unavailable"
        if detection_status == "LOW_CONFIDENCE":
            interpretation = "Bottle cue exists, but automatic detection is not reliable enough for confident clamp patches."
        elif detection_status == "DETECTED":
            interpretation = "Bottle body candidate is automatic; clamp patches are image-space estimates only."
        elif detection_status == "MANUAL_ROI":
            interpretation = "Manual ROI debug mode is active; this is not automatic bottle detection."
        else:
            interpretation = "Search the APS frame for a stronger vertical bottle-body cue."
        self.metrics_label.setText(
            "Decision and Interpretation\n\n"
            f"Backend: {report.get('backend', 'N/A')}\n"
            f"APS mode: {aps_mode_label(report).replace('APS Mode: ', '')}\n"
            f"Detector stack: {report.get('detector_source', 'none')}\n"
            f"State: {report.get('current_state', 'N/A')}\n"
            f"Decision: {report.get('davis_decision', 'N/A')}\n"
            f"Detection source: {report.get('davis_detection_source', 'none')}\n"
            f"APS confidence: {confidence:.0f}/100\n"
            f"Event confidence: {event_confidence:.0f}/100\n"
            f"Combined confidence: {combined_confidence:.0f}/100\n"
            f"Event/APS agreement: {report.get('aps_event_agreement', 'N/A')}\n"
            f"Clamp estimate: {clamp_ready}\n"
            f"Grip width: {report.get('grip_width_px', 'N/A')} px only\n"
            f"Motion class: {report.get('motion_class', 'N/A')}\n"
            f"Event status: {report.get('event_stream_status', 'N/A')}\n\n"
            "Events support motion/tracking. A stationary bottle may produce zero events.\n\n"
            f"Display FPS: {report.get('display_fps', 0):.1f}\n"
            f"Analysis FPS: {report.get('analysis_fps', 0):.1f} / target {DAVIS_ANALYSIS_FPS:.0f}\n"
            f"Event Update FPS: {report.get('event_update_fps', 0):.1f} / target {DAVIS_EVENT_UPDATE_FPS:.0f}\n"
            f"Last detection update: {report.get('last_analysis_age_ms', 'N/A')} ms ago\n\n"
            f"{interpretation}\n\n"
            f"{report.get('main_limitation', '')}"
        )
        self.limit_label.setText(self.limitation_text(report))
        if self.duration > 0 and time.monotonic() - self.started >= self.duration:
            self.close()

    def close(self):
        self.planner.save_outputs(self.report)
        self.planner.stop()
        self.window.close()
        self.app.quit()

    def run(self):
        self.window.show()
        code = self.app.exec_()
        self.planner.save_outputs(self.report)
        self.planner.stop()
        return code


def run_gui(duration=0.0, roi_xywh=None, sensitivity="balanced", display_mode=None):
    if os.environ.get("DAVIS_ALLOW_STANDALONE_GUI", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        print("The standalone DAVIS-only GUI is deprecated for normal operation.")
        print("Run: ./scripts/run_davis_motion_demo.sh")
        print("Then open http://127.0.0.1:8080/?tab=davis")
        return 2
    return DavisGraspPlannerWindow(duration=duration, roi_xywh=roi_xywh, sensitivity=sensitivity, display_mode=display_mode).run()


def run_simulation(pattern, sensitivity="balanced", roi_xywh=None):
    event_report = run_postgrasp_simulation(pattern, sensitivity=sensitivity, roi_xywh=roi_xywh)
    report = {
        "sensor_mode": "DAVIS346",
        "backend": "simulation",
        "current_state": STATE_POST,
        "davis_decision": event_report.get("post_grasp_decision"),
        "motion_class": event_report.get("motion_class"),
        "slip_risk_score": event_report.get("slip_risk_score"),
        "rotation_risk_score": event_report.get("rotation_risk_score"),
        "vibration_risk_score": event_report.get("vibration_score"),
        "stability_score": event_report.get("post_grasp_stability_score"),
        "confidence_score": event_report.get("confidence_score"),
        "davis_limitations": list(LIMITATIONS),
        "post_grasp_event_report": event_report,
    }
    _write_json(REPORT_PATH, report)
    _write_json(HISTORY_PATH, [{"timestamp": time.time(), "state": STATE_POST, "simulation": pattern}])
    print(json.dumps(_json_safe(report), indent=4))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone DAVIS346-only grasp planning demo.")
    parser.add_argument("--diagnose", action="store_true", help="Check DAVIS USB/backend/event/APS availability.")
    parser.add_argument("--diagnose-frame", action="store_true", help="Capture DAVIS APS/event diagnostics and save latest frame debug outputs.")
    parser.add_argument("--save-frame-debug", action="store_true", help="Save raw APS, display APS, event heatmap, and frame-debug JSON during diagnostics.")
    parser.add_argument("--live", action="store_true", help="Run DAVIS-only live grasp planner.")
    parser.add_argument("--gui", action="store_true", help="Show DAVIS-only GUI during --live.")
    parser.add_argument("--duration", type=float, default=None, help="Live run duration in seconds. GUI stays open unless this is provided.")
    parser.add_argument("--live-print", action="store_true", help="Print live DAVIS-only planner status.")
    parser.add_argument("--save-report", action="store_true", help="Print and save final report.")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"), help="Manual DAVIS ROI x y w h.")
    parser.add_argument("--sensitivity", choices=["conservative", "balanced", "sensitive"], default="balanced")
    parser.add_argument("--display-mode", choices=sorted(DAVIS_APS_DISPLAY_MODES), default=DAVIS_APS_DISPLAY_MODE, help="DAVIS APS display interpretation for live/diagnostic views.")
    parser.add_argument("--simulate", choices=["stable", "impact_only", "slip_down", "spin", "vibration"], help="Run post-grasp motion simulation.")
    args = parser.parse_args(argv)

    if args.diagnose or args.diagnose_frame:
        diagnose(roi_xywh=args.roi, sensitivity=args.sensitivity, save_debug=args.diagnose_frame or args.save_frame_debug, display_mode=args.display_mode)
        return 0
    if args.simulate:
        run_simulation(args.simulate, sensitivity=args.sensitivity, roi_xywh=args.roi)
        return 0
    if (args.live or args.live_print or args.gui) and os.environ.get("DAVIS_ALLOW_STANDALONE_BACKEND", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        print("DAVIS-only live planner is deprecated for normal operation and would open a competing camera owner.")
        print("Run: ./scripts/run_davis_motion_demo.sh")
        print("Then open http://127.0.0.1:8080/?tab=davis")
        print("For explicit legacy debugging only, set DAVIS_ALLOW_STANDALONE_BACKEND=1.")
        return 2
    if args.live and args.gui:
        return run_gui(duration=0.0 if args.duration is None else args.duration, roi_xywh=args.roi, sensitivity=args.sensitivity, display_mode=args.display_mode)
    if args.live or args.live_print or args.save_report:
        report = run_live(duration=5.0 if args.duration is None else args.duration, roi_xywh=args.roi, live_print=args.live_print, save_report=args.save_report, sensitivity=args.sensitivity, display_mode=args.display_mode)
        return 0 if report.get("current_state") != STATE_UNAVAILABLE else 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

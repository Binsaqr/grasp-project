"""DAVIS APS frame format detection, demosaicing, and lightweight features.

The frame-based DAVIS output is called APS. Backend metadata can verify the
array's channel encoding (for example, pyaer's RGB contract); this does not
claim calibrated colour accuracy or white balance.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output" / "davis"
APS_STATUS_PATH = OUTPUT_DIR / "davis_aps_status.json"
APS_FEATURES_PATH = OUTPUT_DIR / "davis_aps_features.json"
APS_LATEST_IMAGE_PATH = OUTPUT_DIR / "latest_aps.jpg"
BAYER_PATTERNS = {
    # OpenCV's Bayer conversion aliases describe the resulting first/last
    # channel ordering rather than the conventional top-left mosaic name.
    "rggb": cv2.COLOR_BayerBG2RGB,
    "bggr": cv2.COLOR_BayerRG2RGB,
    "grbg": cv2.COLOR_BayerGB2RGB,
    "gbrg": cv2.COLOR_BayerGR2RGB,
}
KNOWN_CHANNEL_ORDERS = {"rgb", "bgr"}


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _write_image_atomic(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(path)


@dataclass
class DavisApsFrame:
    """Normalized APS frame record for dashboards, sync, and diagnostics."""

    frame_id: int | None
    device_timestamp: float | None
    host_timestamp: float
    image: Any = field(repr=False, default=None)
    shape: list[int] = field(default_factory=list)
    dtype: str = "unknown"
    bit_depth: int | None = None
    channel_count: int = 0
    raw_format: str = "unknown"
    aps_mode: str = "DAVIS APS Unknown"
    aps_color_verified: bool = False
    aps_bayer_pattern: str | None = None
    aps_channel_order: str | None = None
    aps_channel_order_source: str | None = None
    device_timestamp_unit: str | None = None
    aps_frame_valid: bool = False
    aps_validation_reason: str = "not_processed"
    exposure: float | None = None
    frame_rate: float | None = None
    processing_latency_ms: float | None = None

    def to_status(self) -> dict:
        payload = asdict(self)
        payload.pop("image", None)
        return payload


def _bit_depth(dtype: np.dtype) -> int | None:
    try:
        return int(np.iinfo(dtype).bits)
    except Exception:
        try:
            return int(np.finfo(dtype).bits)
        except Exception:
            return None


def _channel_count(arr: np.ndarray) -> int:
    if arr.ndim == 2:
        return 1
    if arr.ndim == 3:
        return int(arr.shape[2])
    return 0


def latest_aps_array(frames: Any) -> Any:
    """Return the newest APS image from a backend frame or frame stack."""
    if frames is None:
        return None
    if isinstance(frames, (list, tuple)):
        return None if not frames else frames[-1]
    arr = np.asarray(frames)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # A color image is HWC; pyaer monochrome stacks are NHW.
        return arr if arr.shape[-1] in {1, 3, 4} else arr[-1]
    if arr.ndim >= 4:
        return arr[-1]
    return None


def aps_backend_metadata(backend: str | None, frame: Any) -> dict:
    """Describe verified array semantics supplied by the active backend.

    pyaer 0.2.7 returns DAVIS color frames through
    ``get_rgb_frame_event_346`` as HWC RGB arrays. This verifies encoding
    order only; it does not validate colour accuracy or white balance.
    """
    name = str(backend or "").strip().lower()
    arr = None if frame is None else np.asarray(frame)
    metadata = {"backend": name or None}
    if name == "pyaer" and arr is not None and arr.ndim == 3 and arr.shape[-1] == 3:
        metadata.update({
            "color_sensor": True,
            "channel_order": "rgb",
            "channel_order_source": "pyaer_rgb_array_contract",
            "raw_format": "pyaer_rgb_array",
        })
    return metadata


def _configured_choice(value: Any, allowed: set[str]) -> str | None:
    choice = str(value or "").strip().lower()
    return choice if choice in allowed else None


def channel_independence_score(arr: np.ndarray) -> dict:
    """Return conservative channel statistics for possible color APS frames."""
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {
            "channels_differ": False,
            "mean_abs_channel_delta": 0.0,
            "max_correlation": None,
            "chroma_variation": 0.0,
        }
    data = arr[:, :, :3].astype(np.float32)
    means = [float(np.mean(data[:, :, i])) for i in range(3)]
    stds = [float(np.std(data[:, :, i])) for i in range(3)]
    deltas = [
        abs(means[0] - means[1]),
        abs(means[1] - means[2]),
        abs(means[0] - means[2]),
    ]
    flat = [data[:, :, i].reshape(-1) for i in range(3)]
    correlations = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        if stds[i] < 1e-6 or stds[j] < 1e-6:
            correlations.append(1.0)
        else:
            correlations.append(float(np.corrcoef(flat[i], flat[j])[0, 1]))
    chroma = float(np.mean(np.std(data, axis=2)))
    return {
        "channels_differ": bool(max(deltas) >= 2.0 or chroma >= 2.0),
        "channel_means": means,
        "channel_stds": stds,
        "mean_abs_channel_delta": float(max(deltas)),
        "max_correlation": float(max(abs(v) for v in correlations)),
        "chroma_variation": chroma,
    }


def classify_aps_array(frame: Any, metadata: dict | None = None, config: dict | None = None) -> dict:
    """Classify an APS frame without pretending unverified data is RGB."""
    metadata = metadata or {}
    config = config or {}
    if frame is None:
        return {
            "aps_mode": "DAVIS APS Unavailable",
            "aps_color_verified": False,
            "aps_bayer_pattern": None,
            "aps_channel_order": None,
            "aps_channel_order_source": None,
            "aps_frame_valid": False,
            "aps_validation_reason": "no_aps_frame",
            "raw_format": "unavailable",
            "channel_count": 0,
            "shape": [],
            "dtype": "none",
            "bit_depth": None,
        }
    arr = np.asarray(frame)
    channel_count = _channel_count(arr)
    raw_format = str(metadata.get("raw_format") or metadata.get("format") or "array")
    configured_pattern = (
        _configured_choice(config.get("bayer_pattern"), set(BAYER_PATTERNS))
        or _configured_choice(metadata.get("bayer_pattern"), set(BAYER_PATTERNS))
    )
    channel_order = (
        _configured_choice(config.get("channel_order"), KNOWN_CHANNEL_ORDERS)
        or _configured_choice(metadata.get("channel_order"), KNOWN_CHANNEL_ORDERS)
    )
    channel_order_source = None
    if channel_order:
        channel_order_source = (
            "configured"
            if _configured_choice(config.get("channel_order"), KNOWN_CHANNEL_ORDERS)
            else str(metadata.get("channel_order_source") or "backend_metadata")
        )
    color_sensor = bool(metadata.get("color_sensor") or metadata.get("is_color") or metadata.get("color_filter"))
    reason = "format_inspected"
    mode = "DAVIS APS Unknown"
    color_verified = False
    bayer_pattern = None
    stats = {}
    if arr.ndim == 2 or (arr.ndim == 3 and channel_count == 1):
        if color_sensor or configured_pattern in BAYER_PATTERNS:
            mode = "DAVIS APS Raw Bayer"
            bayer_pattern = configured_pattern
            reason = "single_channel_color_sensor_or_bayer_config"
        else:
            mode = "DAVIS APS Grayscale"
            reason = "single_channel_without_color_metadata"
    elif arr.ndim == 3 and channel_count == 3:
        stats = channel_independence_score(arr)
        if channel_order and (color_sensor or stats["channels_differ"]):
            mode = "DAVIS APS Color"
            color_verified = True
            reason = "three_channels_with_verified_backend_or_config_encoding"
        elif stats["channels_differ"]:
            mode = "DAVIS APS Color"
            channel_order = "unverified_color_order"
            channel_order_source = "not_available"
            reason = "three_channels_with_independent_color_information_but_unknown_order"
        else:
            mode = "DAVIS APS Unknown"
            reason = "three_channels_but_channels_not_independent"
    elif arr.ndim == 3 and channel_count == 4:
        mode = "DAVIS APS Unknown"
        reason = "four_channel_or_packed_format_requires_manual_validation"
    else:
        mode = "DAVIS APS Unknown"
        reason = f"unsupported_shape_{arr.shape}"
    return {
        "aps_mode": mode,
        "aps_color_verified": color_verified,
        "aps_bayer_pattern": bayer_pattern,
        "aps_channel_order": channel_order,
        "aps_channel_order_source": channel_order_source,
        "aps_frame_valid": bool(arr.size > 0 and mode != "DAVIS APS Unavailable"),
        "aps_validation_reason": reason,
        "raw_format": raw_format,
        "channel_count": channel_count,
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "bit_depth": _bit_depth(arr.dtype),
        "channel_statistics": stats,
    }


def normalize_aps_frame(frame: Any, frame_id=None, device_timestamp=None, metadata=None, config=None) -> DavisApsFrame:
    started = time.perf_counter()
    status = classify_aps_array(frame, metadata=metadata, config=config)
    arr = None if frame is None else np.asarray(frame)
    return DavisApsFrame(
        frame_id=frame_id,
        device_timestamp=device_timestamp,
        host_timestamp=time.time(),
        image=arr,
        processing_latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        **{key: status[key] for key in (
            "shape",
            "dtype",
            "bit_depth",
            "channel_count",
            "raw_format",
            "aps_mode",
            "aps_color_verified",
            "aps_bayer_pattern",
            "aps_channel_order",
            "aps_channel_order_source",
            "aps_frame_valid",
            "aps_validation_reason",
        )},
        device_timestamp_unit=(metadata or {}).get("device_timestamp_unit") if device_timestamp is not None else None,
    )


def normalize_backend_aps_frame(
    frame: Any,
    *,
    backend: str | None,
    frame_id: int | None = None,
    device_timestamp_us: int | float | None = None,
    config: dict | None = None,
) -> DavisApsFrame:
    """Normalize one backend frame with its verified encoding/timestamp metadata."""
    metadata = aps_backend_metadata(backend, frame)
    if device_timestamp_us is not None:
        metadata["device_timestamp_unit"] = "microseconds"
    return normalize_aps_frame(
        frame,
        frame_id=frame_id,
        device_timestamp=device_timestamp_us,
        metadata=metadata,
        config=config,
    )


def demosaic_bayer(frame: Any, pattern: str) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError("Bayer demosaicing requires a single-channel APS frame")
    pattern = str(pattern).lower()
    if pattern not in BAYER_PATTERNS:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    return cv2.cvtColor(arr, BAYER_PATTERNS[pattern])


def aps_display_image(
    frame: Any,
    mode: str | None = None,
    bayer_pattern: str | None = None,
    channel_order: str | None = None,
) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if mode == "DAVIS APS Raw Bayer" and bayer_pattern in BAYER_PATTERNS:
        return demosaic_bayer(arr, bayer_pattern)
    if arr.ndim == 2:
        return cv2.cvtColor(_to_uint8(arr), cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(_to_uint8(arr[:, :, 0]), cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        image = _to_uint8(arr[:, :, :3])
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if channel_order == "bgr" else image
    return None


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.copy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


APS_VISIBILITY_DEFAULTS = {
    "display_mode": "auto_normalize",
    "gamma": 0.7,
    "clahe_enabled": False,
    "low_light_warn_p95_threshold": 40.0,
    "low_contrast_warn_std_threshold": 12.0,
}
APS_DISPLAY_MODES = ("raw", "auto_normalize", "gamma", "clahe", "false_color")


def compute_aps_stats(frame: Any) -> dict:
    """Raw APS intensity statistics. Never altered by display enhancement."""
    if frame is None:
        return {"aps_stats_valid": False, "aps_stats_reason": "no_aps_frame"}
    arr = np.asarray(frame)
    if arr.size == 0:
        return {"aps_stats_valid": False, "aps_stats_reason": "empty_aps_frame"}
    data = arr.astype(np.float64)
    flat = data.reshape(-1) if arr.ndim == 2 else data.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else data.reshape(-1)
    gray = data if arr.ndim == 2 else (np.mean(data, axis=-1) if arr.ndim == 3 else data)
    g = gray.reshape(-1)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {"aps_stats_valid": False, "aps_stats_reason": "no_finite_values"}
    p1, p50, p95, p99 = (float(v) for v in np.percentile(g, [1, 50, 95, 99]))
    stats = {
        "aps_stats_valid": True,
        "aps_min": float(np.min(g)),
        "aps_p1": round(p1, 2),
        "aps_p50": round(p50, 2),
        "aps_p95": round(p95, 2),
        "aps_p99": round(p99, 2),
        "aps_max": float(np.max(g)),
        "aps_mean": round(float(np.mean(g)), 2),
        "aps_std": round(float(np.std(g)), 2),
        "aps_dtype": str(arr.dtype),
        "aps_shape": [int(v) for v in arr.shape],
        "aps_channel_count": _channel_count(arr),
        "aps_dtype_max": float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else float(np.nanmax(g)) or 1.0,
    }
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        ch = data[..., :3].reshape(-1, 3)
        stats["aps_channel_means"] = [round(float(v), 2) for v in np.mean(ch, axis=0)]
        stats["aps_channel_stds"] = [round(float(v), 2) for v in np.std(ch, axis=0)]
    else:
        stats["aps_channel_means"] = [stats["aps_mean"]]
        stats["aps_channel_stds"] = [stats["aps_std"]]
    return stats


def classify_aps_visibility(stats: dict, config: dict | None = None) -> dict:
    """Honest visibility state from RAW stats (display enhancement excluded)."""
    cfg = {**APS_VISIBILITY_DEFAULTS, **(config or {})}
    if not (stats or {}).get("aps_stats_valid"):
        return {
            "aps_visibility_status": "UNKNOWN",
            "aps_visibility_score": None,
            "aps_underexposed": None,
            "aps_overexposed": None,
            "aps_low_contrast": None,
            "aps_display_warning": "APS statistics unavailable.",
        }
    dtype_max = max(float(stats.get("aps_dtype_max") or 255.0), 1.0)
    scale = 255.0 / dtype_max
    p1 = float(stats["aps_p1"]) * scale
    p95 = float(stats["aps_p95"]) * scale
    p99 = float(stats["aps_p99"]) * scale
    std = float(stats["aps_std"]) * scale
    low_light = p95 < float(cfg["low_light_warn_p95_threshold"])
    overexposed = p1 > 215.0
    low_contrast = (not low_light) and (not overexposed) and std < float(cfg["low_contrast_warn_std_threshold"])
    # score: 60% dynamic-range utilization + 40% contrast
    range_util = float(np.clip((p99 - p1) / 255.0, 0.0, 1.0))
    contrast_util = float(np.clip(std / 50.0, 0.0, 1.0))
    score = round(100.0 * (0.6 * range_util + 0.4 * contrast_util), 1)
    if low_light and score < 10.0:
        status = "UNUSABLE"
        warning = "DAVIS APS is severely underexposed and unusable under current lighting."
    elif low_light:
        status = "LOW_LIGHT"
        warning = ("DAVIS APS is underexposed. Use RealSense RGB for object detection; "
                   "DAVIS APS is context only under current lighting.")
    elif overexposed:
        status = "OVEREXPOSED"
        warning = "DAVIS APS is saturated; reduce exposure/gain or lighting."
    elif low_contrast:
        status = "LOW_CONTRAST"
        warning = "DAVIS APS has very low contrast; details may not be visible."
    elif p95 < 2.0 * float(cfg["low_light_warn_p95_threshold"]):
        status = "DISPLAY_NORMALIZED"
        warning = ("DAVIS APS raw frame is dim (p95 in the borderline zone); the preview is "
                   "normalized for visibility. Raw statistics are unchanged.")
    else:
        status = "GOOD"
        warning = None
    return {
        "aps_visibility_status": status,
        "aps_visibility_score": score,
        "aps_underexposed": bool(low_light),
        "aps_overexposed": bool(overexposed),
        "aps_low_contrast": bool(low_contrast),
        "aps_display_warning": warning,
    }


def enhance_aps_display(image_rgb_u8: np.ndarray | None, mode: str = "auto_normalize", gamma: float = 0.7, clahe_enabled: bool = False) -> tuple[np.ndarray | None, dict]:
    """Display-only enhancement. Raw APS metrics are computed elsewhere on the
    raw frame and are never affected by this function."""
    meta = {"aps_display_mode": str(mode or "raw"), "aps_display_enhanced": False, "aps_display_label": None}
    if image_rgb_u8 is None:
        return None, meta
    img = np.asarray(image_rgb_u8)
    if img.dtype != np.uint8:
        img = _to_uint8(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    out = img
    mode = str(mode or "raw").lower()
    if mode in {"auto_normalize", "gamma", "clahe", "false_color"}:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        if mode == "auto_normalize":
            lo, hi = np.percentile(gray, [1, 99])
            if hi > lo + 1e-3:
                stretched = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
            else:
                stretched = img.astype(np.float32)
            if gamma and abs(float(gamma) - 1.0) > 1e-3:
                stretched = 255.0 * np.power(np.clip(stretched / 255.0, 0.0, 1.0), float(gamma))
            out = stretched.astype(np.uint8)
            meta["aps_display_enhanced"] = True
        elif mode == "gamma":
            out = (255.0 * np.power(np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0), float(gamma or 0.7))).astype(np.uint8)
            meta["aps_display_enhanced"] = True
        elif mode == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            eq = clahe.apply(gray)
            out = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            meta["aps_display_enhanced"] = True
        elif mode == "false_color":
            out = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            meta["aps_display_enhanced"] = True
        if clahe_enabled and mode == "auto_normalize":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lum = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            out = cv2.cvtColor(clahe.apply(lum), cv2.COLOR_GRAY2RGB)
    if meta["aps_display_enhanced"]:
        meta["aps_display_label"] = "Display enhanced for visibility."
    return out, meta


def save_aps_visibility_diagnostics(
    frame: Any,
    output_dir: Path = OUTPUT_DIR,
    config: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Save raw + enhanced previews and a visibility report for live validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_aps_stats(frame)
    visibility = classify_aps_visibility(stats, config)
    status = classify_aps_array(frame, metadata=metadata, config=config)
    base = aps_display_image(
        frame,
        mode=status.get("aps_mode"),
        bayer_pattern=status.get("aps_bayer_pattern"),
        channel_order=status.get("aps_channel_order"),
    )
    written = {}
    if base is not None:
        raw_u8 = np.asarray(frame)
        raw_view = raw_u8 if raw_u8.dtype == np.uint8 and raw_u8.ndim == 2 else cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(str(output_dir / "aps_raw.png"), raw_view)
        written["aps_raw"] = str(output_dir / "aps_raw.png")
        tiles = []
        for mode, name in (("auto_normalize", "aps_normalized"), ("gamma", "aps_gamma"), ("clahe", "aps_clahe")):
            enhanced, _meta = enhance_aps_display(base, mode=mode, gamma=(config or {}).get("gamma", 0.7))
            cv2.imwrite(str(output_dir / f"{name}.png"), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
            written[name] = str(output_dir / f"{name}.png")
            tiles.append((name, enhanced))
        sheet_tiles = [cv2.cvtColor(raw_view, cv2.COLOR_GRAY2RGB)] + [t[1] for t in tiles]
        h = min(t.shape[0] for t in sheet_tiles)
        sheet = np.hstack([t[:h, :, :3] for t in sheet_tiles])
        cv2.imwrite(str(output_dir / "aps_display_contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        written["aps_display_contact_sheet"] = str(output_dir / "aps_display_contact_sheet.png")
    report = {"generated_at": time.time(), **status, **stats, **visibility, "files": written}
    _write_json_atomic(output_dir / "aps_visibility_report.json", report)
    return report


def aps_features(frame: Any, previous_frame: Any = None) -> dict:
    if frame is None:
        return {
            "aps_available": False,
            "aps_feature_valid": False,
            "aps_feature_invalid_reason": "no_aps_frame",
        }
    image = aps_display_image(frame)
    if image is None:
        return {
            "aps_available": True,
            "aps_feature_valid": False,
            "aps_feature_invalid_reason": "unsupported_aps_format",
        }
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]))
    motion = 0.0
    if previous_frame is not None:
        prev = aps_display_image(previous_frame)
        if prev is not None and prev.shape[:2] == image.shape[:2]:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            motion = float(np.mean(cv2.absdiff(gray, prev_gray)))
    return {
        "aps_available": True,
        "aps_feature_valid": True,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "saturation": saturation,
        "frame_to_frame_motion": motion,
        "target_displacement": None,
        "target_angle_change": None,
        "appearance_tracking_confidence": None,
        "global_lighting_change_score": motion,
    }


def save_aps_diagnostics(frame: Any, output_dir: Path = OUTPUT_DIR, metadata=None, config=None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = classify_aps_array(frame, metadata=metadata, config=config)
    if frame is None:
        report = {"status": status, "saved_files": []}
        (output_dir / "aps_format_report.json").write_text(json.dumps(report, indent=2))
        return report
    arr = np.asarray(frame)
    saved = []
    np.save(output_dir / "aps_raw.npy", arr)
    saved.append("aps_raw.npy")
    gray = _to_uint8(arr[:, :, 0] if arr.ndim == 3 else arr)
    cv2.imwrite(str(output_dir / "aps_raw.png"), gray)
    cv2.imwrite(str(output_dir / "aps_gray.png"), gray)
    saved.extend(["aps_raw.png", "aps_gray.png"])
    panels = [cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)]
    labels = ["raw/gray"]
    for pattern in ("rggb", "bggr", "grbg", "gbrg"):
        try:
            img = demosaic_bayer(arr, pattern)
            cv2.imwrite(str(output_dir / f"aps_debayer_{pattern}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved.append(f"aps_debayer_{pattern}.png")
            panels.append(img)
            labels.append(pattern)
        except Exception:
            pass
    if panels:
        h = min(panel.shape[0] for panel in panels)
        resized = []
        for panel, label in zip(panels, labels):
            scale = h / max(panel.shape[0], 1)
            out = cv2.resize(panel, (max(1, int(panel.shape[1] * scale)), h))
            cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            resized.append(out)
        sheet = np.concatenate(resized, axis=1)
        cv2.imwrite(str(output_dir / "aps_color_contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        saved.append("aps_color_contact_sheet.png")
    report = {
        "status": status,
        "features": aps_features(arr),
        "saved_files": saved,
    }
    (output_dir / "aps_format_report.json").write_text(json.dumps(report, indent=2))
    return report


def write_aps_outputs(frame: DavisApsFrame, previous_frame: Any = None, output_dir: Path = OUTPUT_DIR) -> dict:
    """Write standalone DAVIS APS status/image/features files.

    These files are diagnostic and benchmark-facing only. They do not affect the
    RealSense planner or production grasp decisions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image = None if frame is None else frame.image
    status = {"aps_mode": "DAVIS APS Unavailable", "aps_frame_valid": False} if frame is None else frame.to_status()
    features = aps_features(image, previous_frame=previous_frame)
    if image is not None:
        display = aps_display_image(
            image,
            mode=status.get("aps_mode"),
            bayer_pattern=status.get("aps_bayer_pattern"),
            channel_order=status.get("aps_channel_order"),
        )
        if display is not None:
            _write_image_atomic(output_dir / APS_LATEST_IMAGE_PATH.name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            status["latest_aps_image_path"] = str(output_dir / APS_LATEST_IMAGE_PATH.name)
    _write_json_atomic(output_dir / APS_STATUS_PATH.name, status)
    _write_json_atomic(output_dir / APS_FEATURES_PATH.name, features)
    return {"status": status, "features": features}


def _diagnose_from_live(duration: float) -> Any:
    from grasp_project.davis.camera import DavisCamera

    camera = DavisCamera(enable_aps=True, enable_events=True)
    try:
        camera.start()
        deadline = time.time() + float(duration)
        last = None
        while time.time() < deadline:
            camera.poll_once()
            last = camera.latest_aps_frame
            if last and last.image is not None:
                return last.image
            time.sleep(0.02)
        return None
    finally:
        camera.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose DAVIS APS frame format and color/Bayer candidates.")
    parser.add_argument("--diagnose-color", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--save-contact-sheet", action="store_true")
    args = parser.parse_args(argv)
    frame = _diagnose_from_live(args.duration) if args.diagnose_color else None
    report = save_aps_diagnostics(frame)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

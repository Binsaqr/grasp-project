"""Common image encoding helpers."""

import time
from pathlib import Path

import cv2
import numpy as np

from grasp_project.common.file_io import replace_path_retry


def rgb_to_bgr(rgb):
    """Convert an RGB image array to BGR."""
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr):
    """Convert a BGR image array to RGB."""
    return cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)


def to_uint8_display(image):
    """Convert an image-like array to uint8 display range."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo + 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def encode_file_to_jpeg(path, quality=78):
    """Read an image file and encode it as JPEG bytes with timing metadata."""
    t0 = time.perf_counter()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    read_ms = (time.perf_counter() - t0) * 1000.0
    if image is None:
        return None, read_ms, 0.0
    t1 = time.perf_counter()
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    encode_ms = (time.perf_counter() - t1) * 1000.0
    if not ok:
        return None, read_ms, encode_ms
    return encoded.tobytes(), read_ms, encode_ms


def write_image_atomic(path, image, params=None):
    """Write an image through a temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp{suffix}")
    ok = cv2.imwrite(str(tmp_path), image, params or [])
    if not ok:
        return False
    try:
        replace_path_retry(tmp_path, path)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return False
    return True


def encode_rgb_to_jpeg(rgb, quality=75):
    """Encode an RGB array as JPEG bytes with conversion/encode timings."""
    t0 = time.perf_counter()
    bgr = rgb_to_bgr(rgb)
    draw_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    encode_ms = (time.perf_counter() - t1) * 1000.0
    if not ok:
        return None, draw_ms, encode_ms
    return encoded.tobytes(), draw_ms, encode_ms


def fit_image_to_canvas(image, width, height, bg=(12, 18, 32)):
    """Resize an image to fit inside a fixed canvas without cropping."""
    canvas = np.full((int(height), int(width), 3), bg, dtype=np.uint8)
    if image is None:
        return canvas
    h, w = image.shape[:2]
    if h <= 0 or w <= 0 or width <= 0 or height <= 0:
        return canvas
    scale = min(float(width) / float(w), float(height) / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    x = (int(width) - nw) // 2
    y = (int(height) - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


__all__ = [
    "bgr_to_rgb",
    "encode_file_to_jpeg",
    "encode_rgb_to_jpeg",
    "fit_image_to_canvas",
    "rgb_to_bgr",
    "to_uint8_display",
    "write_image_atomic",
]

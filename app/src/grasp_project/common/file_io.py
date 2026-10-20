"""Common JSON file helpers.

These helpers are shared by dashboard/status code and are intentionally small:
they do not import camera, GUI, or dashboard modules.
"""

import json
import os
import time
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in this project
    np = None

DEFAULT_JSON_EXCLUDED_KEYS = {
    "mask",
    "raw_mask",
    "cleaned_mask",
    "body_mask",
    "contour",
    "body_contour",
    "_overlay_rgb",
    "_quality_rgb_snapshot",
    "_quality_depth_snapshot",
    "_quality_intrinsics_snapshot",
}


def json_safe(value, excluded_keys=DEFAULT_JSON_EXCLUDED_KEYS):
    """Convert common numpy/container values into JSON-serializable objects."""
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    if isinstance(value, dict):
        return {k: json_safe(v, excluded_keys=excluded_keys) for k, v in value.items() if k not in excluded_keys}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, excluded_keys=excluded_keys) for v in value]
    return value


def read_json(path, default=None):
    """Read a JSON file, returning ``default`` if it is missing or invalid."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def read_text(path, default=""):
    """Read a text file, returning ``default`` if it is missing or invalid."""
    try:
        return Path(path).read_text()
    except Exception:
        return default


def replace_path_retry(tmp_path, target_path, *, attempts=4, initial_delay_sec=0.01):
    """Replace a file with bounded retries for transient bind-mount locks."""

    tmp_path = Path(tmp_path)
    target_path = Path(target_path)
    last_error = None
    for attempt in range(max(1, int(attempts))):
        try:
            tmp_path.replace(target_path)
            return True
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(max(0.0, float(initial_delay_sec)) * (2 ** attempt))
        except OSError:
            raise
    if last_error is not None:
        raise last_error
    return False


def write_json_atomic(path, value, indent=2):
    """Write JSON via a same-directory temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp_path, "w") as f:
        if indent is None:
            json.dump(json_safe(value), f, separators=(",", ":"))
        else:
            json.dump(json_safe(value), f, indent=indent)
    try:
        replace_path_retry(tmp_path, path)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def write_text_atomic(path, text):
    """Write text via a same-directory temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(str(text))
    try:
        replace_path_retry(tmp_path, path)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def append_jsonl(path, value):
    """Append one JSON-safe record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(json_safe(value)) + "\n")


def append_jsonl_bounded(path, value, *, max_bytes, backup_suffix=".previous"):
    """Best-effort JSONL append with one bounded rotated backup.

    Runtime timelines are diagnostics, not control outputs. A Windows bind
    mount may temporarily keep either file open, so rotation failure preserves
    the readable active file and never propagates into a camera/status loop.
    """

    path = Path(path)
    backup_path = path.with_name(path.name + str(backup_suffix))
    try:
        limit = int(max_bytes)
        line = json.dumps(json_safe(value), separators=(",", ":")) + "\n"
        line_size = len(line.encode("utf-8"))
        if limit <= 0 or line_size > limit:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            current_size = 0
        except OSError:
            return False
        if current_size + line_size > limit:
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    return False
            if current_size > limit:
                # A deployment may inherit a legacy unbounded timeline. Keep
                # only its newest complete records before starting the bounded
                # active file; never preserve a multi-gigabyte backup.
                try:
                    with path.open("rb") as source:
                        start = max(0, current_size - limit)
                        source.seek(start)
                        tail = source.read(limit)
                    if start > 0 and b"\n" in tail:
                        tail = tail.split(b"\n", 1)[1]
                    backup_path.write_bytes(tail[-limit:])
                    path.write_text("")
                    current_size = 0
                except OSError:
                    return False
            if current_size == 0:
                with open(path, "a") as f:
                    f.write(line)
                return True
            rotated = False
            for attempt in range(4):
                try:
                    path.replace(backup_path)
                    rotated = True
                    break
                except PermissionError:
                    if attempt < 3:
                        time.sleep(0.01 * (2 ** attempt))
                except FileNotFoundError:
                    rotated = True
                    break
                except OSError:
                    break
            if not rotated:
                return False
        with open(path, "a") as f:
            f.write(line)
        return True
    except (OSError, TypeError, ValueError):
        return False


__all__ = [
    "append_jsonl",
    "append_jsonl_bounded",
    "json_safe",
    "read_json",
    "read_text",
    "replace_path_retry",
    "write_json_atomic",
    "write_text_atomic",
]

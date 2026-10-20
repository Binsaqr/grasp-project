"""Configuration path and lightweight config-loading helpers.

The functions in this module are intentionally dependency-light and safe to
import without cameras, Flask, ROS, or GUI libraries.
"""

import json
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _coerce_scalar(value):
    text = str(value).strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_bool(value, default=False):
    """Parse common truthy/falsy config strings."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def get_env_bool(name, default=False):
    """Read a boolean environment variable."""
    return parse_bool(os.environ.get(name), default=default)


def get_env_int(name, default=0):
    """Read an integer environment variable."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def get_env_float(name, default=0.0):
    """Read a float environment variable."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _parse_flat_yaml(text):
    """Parse the simple flat YAML used by repo config placeholders.

    This fallback is deliberately small. If PyYAML is installed, ``load_config``
    uses it first; otherwise this handles ``key: value`` config files.
    """
    result = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            result[key] = _coerce_scalar(value)
    return result


def config_path(name):
    """Return a path inside the repository ``config/`` directory."""
    return CONFIG_DIR / name


def output_path(name):
    """Return a path inside the repository ``output/`` directory."""
    return OUTPUT_DIR / name


def ensure_output_dir():
    """Create and return the repository output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_config_file(path, default=None):
    """Load JSON/YAML config data, returning ``default`` on missing/invalid input."""
    path = Path(path)
    fallback = {} if default is None else default
    if not path.exists():
        return fallback
    try:
        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                return json.load(f)
        if path.suffix.lower() in {".yaml", ".yml"}:
            text = path.read_text()
            try:
                import yaml  # type: ignore

                loaded = yaml.safe_load(text)
                return fallback if loaded is None else loaded
            except Exception:
                parsed = _parse_flat_yaml(text)
                return parsed if parsed else fallback
    except Exception:
        return fallback
    return fallback


def load_named_config(name, default=None):
    """Load a config file by name from ``config/``."""
    return load_config_file(config_path(name), default=default)


__all__ = [
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "config_path",
    "ensure_output_dir",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "load_config_file",
    "load_named_config",
    "output_path",
    "parse_bool",
]

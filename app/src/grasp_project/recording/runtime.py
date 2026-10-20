"""Explicit process-local recording sessions; importing this opens no resources."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .session import RecordingWriter, _name

_ENV = ("GRASP_RECORDING_ROOT", "GRASP_RECORDING_TRIAL_ID",
        "GRASP_RECORDING_SESSION_ID", "GRASP_RECORDING_OBJECT_ID")


def runtime_recording_requested(environ=None):
    environ = os.environ if environ is None else environ
    return any(environ.get(key) for key in _ENV)


def _revision(root):
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                    text=True, timeout=5, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    status = git("status", "--short")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "working_tree_dirty": None if status is None else bool(status),
            "status_short": None if status is None else status.splitlines()}


def create_runtime_recording(role, configuration, *, environ=None, project_root=None):
    """Return None unless all explicit session environment values are supplied.

    Each process owns a separate new ``ROOT/role`` archive. Related process parts
    share trial/session/object identity, never an in-memory writer or manifest.
    The caller closes only after successful worker teardown, and aborts on errors.
    """
    environ = os.environ if environ is None else environ
    supplied = [environ.get(key, "") for key in _ENV]
    if not any(supplied):
        return None
    if not all(isinstance(value, str) and value.strip() for value in supplied):
        raise ValueError("runtime recording requires all of " + ", ".join(_ENV))
    _name(role, "runtime recording role")
    trial = _name(supplied[1], "trial_id")
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    destination = Path(supplied[0]).resolve() / role
    if any((parent / "manifest.json").exists() for parent in destination.parents):
        raise ValueError("runtime recording must not be nested inside another archive")
    snapshots = {}
    for path in sorted((root / "config").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
            data = path.read_bytes()
            snapshots[path.relative_to(root).as_posix()] = {
                "contents_utf8": data.decode("utf-8"), "sha256": hashlib.sha256(data).hexdigest(),
            }
    return RecordingWriter(
        destination, recording_id=f"{role[:60]}-{uuid.uuid4().hex}",
        config={**configuration, "runtime_role": role, "effective_settings": configuration,
                "config_files": snapshots, "python": sys.version},
        revision=_revision(root),
        metadata={"trial_id": trial, "session_id": supplied[2], "object_id": supplied[3],
                  "data_origin": "recorded", "ground_truth_status": "not_labelled",
                  "process_role": role, "archive_part": True},
    )

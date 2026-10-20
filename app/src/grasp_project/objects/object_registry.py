"""File-backed object profile registry.

The registry stores reusable profile JSON under ``config/object_profiles`` and
keeps generated scan artifacts in ``output/object_profiles``. It is safe to
import without RealSense or DAVIS hardware.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from grasp_project.common.config import CONFIG_DIR, OUTPUT_DIR, PROJECT_ROOT
from grasp_project.common.file_io import read_json, write_json_atomic
from .object_profile import (
    GENERIC_OBJECT_ID,
    LEGACY_BLUE_BOTTLE_ID,
    artifact_paths_ok,
    generic_object_profile,
    legacy_blue_bottle_profile,
    profile_to_legacy_grasp_profile,
    slugify,
    utc_iso,
)

PROFILE_DIR = CONFIG_DIR / "object_profiles"
PROFILE_INDEX_PATH = PROFILE_DIR / "index.json"
PROFILE_OUTPUT_DIR = OUTPUT_DIR / "object_profiles"
LEGACY_PROFILE_PATH = PROJECT_ROOT / "known_bottle_grasp_profile.json"


def ensure_profile_dirs(profile_dir: Path = PROFILE_DIR, output_dir: Path = PROFILE_OUTPUT_DIR) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def profile_path(object_id: str, profile_dir: Path = PROFILE_DIR) -> Path:
    return profile_dir / f"{slugify(object_id)}.json"


def _display_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _default_index() -> dict[str, Any]:
    return {
        "profile_index_version": 1,
        "active_object_id": LEGACY_BLUE_BOTTLE_ID,
        "updated_at": utc_iso(),
        "profiles": [],
        "generic_mode_available": True,
    }


def _profile_summary(profile: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    return {
        "object_id": profile.get("object_id"),
        "display_name": profile.get("display_name"),
        "object_type": profile.get("object_type"),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "last_used_at": profile.get("last_used_at"),
        "profile_confidence": profile.get("profile_confidence"),
        "active_target_allowed": profile.get("active_target_allowed", True),
        "compatible_embodiments": profile.get("compatible_embodiments") or [],
        "thumbnail_path": profile.get("representative_rgb_snapshot_path"),
        "profile_path": _display_path(path) if path else profile.get("profile_path"),
        "archived": bool(profile.get("archived")),
    }


def ensure_default_profiles(profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    ensure_profile_dirs(profile_dir)
    legacy = read_json(LEGACY_PROFILE_PATH, {})
    blue_path = profile_path(LEGACY_BLUE_BOTTLE_ID, profile_dir)
    if not blue_path.exists():
        profile = legacy_blue_bottle_profile(legacy)
        profile["profile_path"] = _display_path(blue_path)
        write_json_atomic(blue_path, profile)
    index = read_json(index_path, _default_index()) or _default_index()
    summaries = []
    for path in sorted(profile_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        profile = read_json(path, {})
        if profile:
            profile["profile_path"] = _display_path(path)
            summaries.append(_profile_summary(profile, path))
    if not any(item.get("object_id") == LEGACY_BLUE_BOTTLE_ID for item in summaries):
        profile = legacy_blue_bottle_profile(legacy)
        profile["profile_path"] = _display_path(blue_path)
        write_json_atomic(blue_path, profile)
        summaries.append(_profile_summary(profile, blue_path))
    active = index.get("active_object_id") or LEGACY_BLUE_BOTTLE_ID
    if active != GENERIC_OBJECT_ID and not any(item.get("object_id") == active and not item.get("archived") for item in summaries):
        active = LEGACY_BLUE_BOTTLE_ID
    current_values = {
        "active_object_id": active,
        "profiles": summaries,
        "generic_mode_available": True,
    }
    if not index_path.exists() or any(index.get(key) != value for key, value in current_values.items()):
        index.update(current_values)
        index["updated_at"] = utc_iso()
        write_json_atomic(index_path, index)
    return index


def load_index(profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    return ensure_default_profiles(profile_dir, index_path)


def load_profile(object_id: str, profile_dir: Path = PROFILE_DIR, index_path: Path | None = None) -> dict[str, Any]:
    object_id = slugify(object_id)
    if object_id == GENERIC_OBJECT_ID:
        return generic_object_profile()
    ensure_default_profiles(profile_dir, index_path or profile_dir / "index.json")
    profile = read_json(profile_path(object_id, profile_dir), {})
    if profile:
        profile["profile_path"] = _display_path(profile_path(object_id, profile_dir))
    return profile


def list_profiles(profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH, include_archived: bool = False) -> list[dict[str, Any]]:
    index = load_index(profile_dir, index_path)
    profiles = list(index.get("profiles") or [])
    if not include_archived:
        profiles = [p for p in profiles if not p.get("archived")]
    return profiles


def save_profile(profile: dict[str, Any], profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    ensure_profile_dirs(profile_dir)
    object_id = slugify(profile.get("object_id") or profile.get("display_name") or f"object_{int(time.time())}")
    profile = dict(profile)
    profile["object_id"] = object_id
    profile.setdefault("created_at", utc_iso())
    profile["updated_at"] = utc_iso()
    path = profile_path(object_id, profile_dir)
    profile["profile_path"] = _display_path(path)
    write_json_atomic(path, profile)
    load_index(profile_dir, index_path)
    return profile


def active_profile(profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    index = load_index(profile_dir, index_path)
    active_id = index.get("active_object_id") or LEGACY_BLUE_BOTTLE_ID
    profile = load_profile(active_id, profile_dir, index_path)
    if not profile:
        profile = load_profile(LEGACY_BLUE_BOTTLE_ID, profile_dir, index_path)
    profile["active"] = True
    profile["active_selection_source"] = "object_profile_registry"
    profile.update(artifact_paths_ok(profile, root=PROJECT_ROOT))
    return profile


def active_grasp_profile(profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    return profile_to_legacy_grasp_profile(active_profile(profile_dir, index_path))


def set_active_profile(object_id: str, profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    object_id = slugify(object_id)
    if object_id != GENERIC_OBJECT_ID:
        profile = load_profile(object_id, profile_dir, index_path)
        if not profile:
            raise ValueError(f"Object profile not found: {object_id}")
        if profile.get("active_target_allowed") is False:
            raise ValueError(f"Object profile is not allowed as an active target: {object_id}")
        profile["last_used_at"] = utc_iso()
        save_profile(profile, profile_dir, index_path)
    index = load_index(profile_dir, index_path)
    index["active_object_id"] = object_id
    index["updated_at"] = utc_iso()
    write_json_atomic(index_path, index)
    return active_profile(profile_dir, index_path)


def rename_profile(object_id: str, display_name: str, profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    profile = load_profile(object_id, profile_dir, index_path)
    if not profile:
        raise ValueError(f"Object profile not found: {object_id}")
    profile["display_name"] = str(display_name or profile.get("display_name") or object_id).strip()
    return save_profile(profile, profile_dir, index_path)


def archive_profile(object_id: str, profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    object_id = slugify(object_id)
    if object_id in {LEGACY_BLUE_BOTTLE_ID, GENERIC_OBJECT_ID}:
        raise ValueError("The blue bottle fallback and generic mode cannot be archived.")
    profile = load_profile(object_id, profile_dir, index_path)
    if not profile:
        raise ValueError(f"Object profile not found: {object_id}")
    profile["archived"] = True
    profile["active_target_allowed"] = False
    profile["updated_at"] = utc_iso()
    save_profile(profile, profile_dir, index_path)
    index = load_index(profile_dir, index_path)
    if index.get("active_object_id") == object_id:
        index["active_object_id"] = LEGACY_BLUE_BOTTLE_ID
        write_json_atomic(index_path, index)
    return profile


def export_profile(object_id: str, destination: str | Path | None = None, profile_dir: Path = PROFILE_DIR, index_path: Path | None = None) -> Path:
    profile = load_profile(object_id, profile_dir, index_path)
    if not profile:
        raise ValueError(f"Object profile not found: {object_id}")
    destination = Path(destination) if destination else PROFILE_OUTPUT_DIR / f"{slugify(object_id)}_profile_export.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, profile)
    return destination


def import_profile(path: str | Path, profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not payload.get("object_id"):
        payload["object_id"] = slugify(payload.get("display_name") or Path(path).stem)
    return save_profile(payload, profile_dir, index_path)


def registry_status(profile_dir: Path = PROFILE_DIR, index_path: Path = PROFILE_INDEX_PATH) -> dict[str, Any]:
    index = load_index(profile_dir, index_path)
    active = active_profile(profile_dir, index_path)
    return {
        "active_profile": active,
        "active_object_id": active.get("object_id"),
        "profiles": list_profiles(profile_dir, index_path),
        "profile_count": len(list_profiles(profile_dir, index_path)),
        "index_path": str(index_path),
        "profile_dir": str(profile_dir),
        "generic_mode": generic_object_profile(),
        "object_profile_status": "ready",
        "updated_at": index.get("updated_at"),
        "production_safety_unchanged": True,
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Object profile registry")
    parser.add_argument("--list", action="store_true", help="List saved object profiles")
    parser.add_argument("--set-active", metavar="OBJECT_ID", help="Set active object profile")
    parser.add_argument("--export", metavar="OBJECT_ID", help="Export a profile JSON")
    parser.add_argument("--import", dest="import_path", metavar="PROFILE_JSON", help="Import a profile JSON")
    parser.add_argument("--archive", metavar="OBJECT_ID", help="Archive a profile without deleting it")
    parser.add_argument("--destination", metavar="PATH", help="Export destination path")
    args = parser.parse_args(argv)
    try:
        if args.list:
            print(json.dumps(registry_status(), indent=2))
        elif args.set_active:
            print(json.dumps(set_active_profile(args.set_active), indent=2))
        elif args.export:
            print(str(export_profile(args.export, args.destination)))
        elif args.import_path:
            print(json.dumps(import_profile(args.import_path), indent=2))
        elif args.archive:
            print(json.dumps(archive_profile(args.archive), indent=2))
        else:
            parser.print_help()
    except Exception as exc:
        print(f"object_registry error: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

"""Camera-free tests for object enrollment profiles and registry behavior."""

from pathlib import Path

import numpy as np
import grasp_project.objects.object_registry as object_registry

from grasp_project.objects.object_profile import (
    GENERIC_OBJECT_ID,
    LEGACY_BLUE_BOTTLE_ID,
    artifact_paths_ok,
    create_object_profile,
    generic_object_profile,
    match_profiles,
    observation_descriptor,
    profile_to_legacy_grasp_profile,
)
from grasp_project.objects.object_registry import (
    active_profile,
    archive_profile,
    ensure_default_profiles,
    list_profiles,
    load_profile,
    save_profile,
    set_active_profile,
)
from grasp_project.objects.object_scanner import scan_from_arrays


def synthetic_rgb_depth_mask():
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[:, :, :] = [30, 36, 42]
    rgb[14:70, 46:76, :] = [36, 92, 190]
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[14:70, 46:76] = 1
    depth = np.zeros((80, 120), dtype=np.uint16)
    depth[mask > 0] = 600
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 60.0, "cy": 40.0, "width": 120, "height": 80}
    return rgb, depth, mask, intrinsics


def temp_registry(tmp_path):
    profile_dir = tmp_path / "profiles"
    index_path = profile_dir / "index.json"
    ensure_default_profiles(profile_dir=profile_dir, index_path=index_path)
    return profile_dir, index_path


def test_scan_from_arrays_creates_reviewable_profile_draft():
    rgb, depth, mask, intrinsics = synthetic_rgb_depth_mask()

    result = scan_from_arrays(
        display_name="Demo Bottle",
        rgb=rgb,
        depth=depth,
        mask=mask,
        intrinsics=intrinsics,
        save_artifacts=False,
    )

    assert result["ok"] is True
    draft = result["profile_draft"]
    assert draft["object_id"] == "demo_bottle"
    assert draft["active_target_allowed"] is True
    assert draft["expected_contact_count"] == 2
    assert draft["expected_grip_width_mm"] is not None
    assert result["scan_report"]["object_scan_status"] == "REVIEW_PROFILE"


def test_registry_preserves_blue_bottle_default_and_generic_mode(tmp_path):
    profile_dir, index_path = temp_registry(tmp_path)

    status = ensure_default_profiles(profile_dir=profile_dir, index_path=index_path)
    profiles = list_profiles(profile_dir=profile_dir, index_path=index_path)
    generic = load_profile(GENERIC_OBJECT_ID, profile_dir=profile_dir, index_path=index_path)

    assert status["active_object_id"] == LEGACY_BLUE_BOTTLE_ID
    assert any(p["object_id"] == LEGACY_BLUE_BOTTLE_ID for p in profiles)
    assert generic["object_id"] == GENERIC_OBJECT_ID
    assert generic["source"] == "generic_mode"


def test_registry_read_does_not_rewrite_unchanged_index(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles"
    index_path = profile_dir / "index.json"
    current_time = ["2026-08-14T10:00:00Z"]
    monkeypatch.setattr(object_registry, "utc_iso", lambda: current_time[0])

    first = ensure_default_profiles(profile_dir=profile_dir, index_path=index_path)
    original_bytes = index_path.read_bytes()
    current_time[0] = "2026-08-14T11:00:00Z"
    second = ensure_default_profiles(profile_dir=profile_dir, index_path=index_path)

    assert second["updated_at"] == first["updated_at"]
    assert index_path.read_bytes() == original_bytes


def test_registry_save_set_active_and_archive_without_deleting(tmp_path):
    profile_dir, index_path = temp_registry(tmp_path)
    rgb, depth, mask, intrinsics = synthetic_rgb_depth_mask()
    profile = create_object_profile(
        object_id="demo_bottle",
        display_name="Demo Bottle",
        rgb=rgb,
        mask=mask,
        depth=depth,
        intrinsics=intrinsics,
        source="unit_test",
    )

    saved = save_profile(profile, profile_dir=profile_dir, index_path=index_path)
    active = set_active_profile(saved["object_id"], profile_dir=profile_dir, index_path=index_path)
    archived = archive_profile(saved["object_id"], profile_dir=profile_dir, index_path=index_path)
    fallback = active_profile(profile_dir=profile_dir, index_path=index_path)

    assert active["object_id"] == "demo_bottle"
    assert archived["archived"] is True
    assert (profile_dir / "demo_bottle.json").exists()
    assert fallback["object_id"] == LEGACY_BLUE_BOTTLE_ID


def test_profile_matching_flags_identical_profiles_as_ambiguous():
    rgb, depth, mask, intrinsics = synthetic_rgb_depth_mask()
    profile_a = create_object_profile(
        object_id="object_a",
        display_name="Object A",
        rgb=rgb,
        mask=mask,
        depth=depth,
        intrinsics=intrinsics,
        source="unit_test",
    )
    profile_b = {**profile_a, "object_id": "object_b", "display_name": "Object B"}
    observation = observation_descriptor(rgb=rgb, mask=mask, depth=depth, intrinsics=intrinsics)

    result = match_profiles(observation, [profile_a, profile_b])

    assert result["profile_match_status"] == "AMBIGUOUS"
    assert result["matched_object_id"] is None


def test_profile_to_legacy_planner_shape_and_artifact_validation(tmp_path):
    profile = generic_object_profile()
    profile["object_id"] = "custom_object"
    profile["expected_grip_width_mm"] = {"min": 35, "max": 95}
    profile["representative_rgb_snapshot_path"] = "missing/object.jpg"

    legacy = profile_to_legacy_grasp_profile(profile)
    artifacts = artifact_paths_ok(profile, root=tmp_path)

    assert legacy["object_name"] == "custom_object"
    assert legacy["expected_diameter_mm"] == {"min": 35, "max": 95}
    assert legacy["profile_type"] == "object_profile"
    assert artifacts["artifact_paths_valid"] is False
    assert artifacts["missing_artifacts"][0]["field"] == "representative_rgb_snapshot_path"

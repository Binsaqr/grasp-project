"""Embodiment registry and config loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from grasp_project.common.config import config_path, load_config_file
from grasp_project.embodiment.base import (
    EMBODIMENT_IDS,
    LEGACY_EMBODIMENT_ALIASES,
    EmbodimentDefinition,
    canonical_embodiment_id,
)


def default_embodiment_definitions() -> dict[str, EmbodimentDefinition]:
    """Return built-in embodiment definitions without importing hardware code."""

    return {
        "parallel_jaw_two_finger": EmbodimentDefinition(
            "parallel_jaw_two_finger",
            "Parallel Two-Finger Gripper",
            "robot_gripper",
            2,
            2,
            ["patch", "fingertip", "finger_link"],
            ["rgb_bottle_detection", "depth_contact_consistency", "left_right_contact_patches"],
            ["pointcloud_contact_validation", "davis_slip", "davis_rotation", "davis_vibration"],
            ["two_contact_epsilon", "grip_width", "opposition", "contact_coverage", "3d_surface_support"],
            "two_contact_epsilon",
            ["grip_width", "opposition", "visual_contact", "depth_consistency"],
            ["suction_seal", "wrap_coverage", "support_triangle"],
            ["two_valid_opposing_contacts", "grip_width_within_limits", "depth_supported_patches"],
            ["no_one_sided_detected_contact", "no_force_or_pressure_claims"],
            ["Assumes two opposing side contacts and no force sensors."],
            "PRODUCTION_READY",
            "stable",
            "READY_TO_CLOSE",
            ["two_point_parallel_gripper"],
        ),
        "articulated_two_finger": EmbodimentDefinition(
            "articulated_two_finger",
            "Articulated Two-Finger Gripper",
            "robot_gripper",
            2,
            2,
            ["fingertip", "finger_link", "patch"],
            ["finger_segment_geometry", "object_surface"],
            ["pointcloud_contact_validation", "davis_slip"],
            ["opposition", "contact_coverage", "3d_surface_support", "stability"],
            "opposition",
            ["contact_angle_compatibility", "contact_spread"],
            ["suction_seal", "wrap_coverage"],
            ["two_contacts_visible", "contact_angles_compatible"],
            ["asymmetric_contacts_warn_not_fail"],
            ["Fingers may not be parallel; grip width is not sufficient by itself."],
            "CONFIG_ONLY",
            "experimental",
            "READY_TO_ENGAGE",
        ),
        "three_finger_gripper": EmbodimentDefinition(
            "three_finger_gripper",
            "Three-Finger Gripper",
            "robot_gripper",
            3,
            3,
            ["patch", "fingertip"],
            ["three_contact_regions", "object_center"],
            ["pointcloud_contact_validation", "davis_slip"],
            ["support_triangle", "contact_coverage", "contact_count_validity", "3d_surface_support"],
            "support_triangle",
            ["angular_coverage", "contact_spread"],
            ["two_contact_epsilon", "grip_width", "suction_seal"],
            ["three_valid_contacts", "object_center_inside_support_region"],
            ["missing_finger_blocks_ready"],
            ["Do not feed three contacts into the current two-contact epsilon function."],
            "READY_FOR_BENCHMARKING",
            "experimental",
            "READY_TO_ENGAGE",
            ["three_finger_grip"],
        ),
        "multi_finger_robotic_hand": EmbodimentDefinition(
            "multi_finger_robotic_hand",
            "Multi-Finger Robotic Hand",
            "robot_hand",
            2,
            None,
            ["fingertip", "finger_link", "palm_region", "patch"],
            ["multi_contact_regions"],
            ["pointcloud_contact_validation", "davis_slip", "davis_rotation"],
            ["support_polygon", "contact_coverage", "3d_surface_support", "stability"],
            "support_polygon",
            ["object_enclosure", "contact_distribution"],
            ["two_contact_epsilon", "grip_width", "suction_seal"],
            ["sufficient_contact_distribution"],
            ["no_force_or_pressure_claims"],
            ["Visual evidence cannot measure human/robot finger force."],
            "CONFIG_ONLY",
            "experimental",
            "READY_TO_ENGAGE",
        ),
        "human_hand": EmbodimentDefinition(
            "human_hand",
            "Human Hand",
            "human_observation",
            1,
            None,
            ["fingertip", "finger_link", "palm_region", "patch"],
            ["hand_like_geometry"],
            ["davis_slip", "davis_rotation", "pointcloud_contact_validation"],
            ["support_polygon", "contact_coverage", "stability"],
            "support_polygon",
            ["object_motion", "contact_distribution"],
            ["two_contact_epsilon", "grip_width", "suction_seal"],
            ["research_observation_only"],
            ["must_not_control_robot_actions"],
            ["Research/observation mode only unless explicitly configured."],
            "CONFIG_ONLY",
            "experimental",
            "HOLD",
        ),
        "vacuum_suction": EmbodimentDefinition(
            "vacuum_suction",
            "Vacuum / Suction Gripper",
            "robot_gripper",
            1,
            1,
            ["suction_area", "patch"],
            ["surface_patch", "depth_surface_normal"],
            ["pointcloud_contact_validation", "davis_slip", "davis_vibration"],
            ["suction_seal", "surface_flatness", "normal_alignment", "3d_surface_support"],
            "suction_seal",
            ["surface_flatness", "normal_alignment", "edge_leakage_risk"],
            ["two_contact_epsilon", "grip_width", "opposition"],
            ["valid_planar_seal_area", "normal_alignment", "not_near_edge"],
            ["no_suction_pressure_claims"],
            ["No suction pressure sensor is modeled; seal quality is visual/geometric only."],
            "CONFIG_ONLY",
            "experimental",
            "READY_TO_APPLY_SUCTION",
            ["vacuum_grip"],
        ),
        "soft_gripper": EmbodimentDefinition(
            "soft_gripper",
            "Soft Gripper",
            "robot_gripper",
            1,
            None,
            ["distributed_patch", "wrap_region", "patch"],
            ["distributed_contact_regions"],
            ["pointcloud_contact_validation", "davis_slip", "davis_vibration"],
            ["wrap_coverage", "contact_coverage", "stability"],
            "wrap_coverage",
            ["distributed_contact_quality", "enclosure_score", "symmetry"],
            ["suction_seal", "support_triangle"],
            ["sufficient_wrap_coverage", "continuous_contact_regions"],
            ["do_not_reduce_to_two_precise_contact_pixels"],
            ["Point-contact epsilon is only approximate if a validated reduction exists."],
            "READY_FOR_BENCHMARKING",
            "experimental",
            "READY_TO_WRAP",
        ),
        "custom": EmbodimentDefinition(
            "custom",
            "Custom Gripper",
            "custom",
            0,
            None,
            ["point", "line", "patch", "distributed_patch", "suction_area", "wrap_region"],
            ["configured_contact_model"],
            [],
            ["contact_count_validity"],
            "contact_count_validity",
            [],
            [],
            ["custom_plugin_must_validate_contacts"],
            ["experimental_cannot_control_robot_by_default"],
            ["Custom plugins must provide tests and release-gate evidence."],
            "CONFIG_ONLY",
            "experimental",
            "READY_TO_ENGAGE",
        ),
        "unknown": EmbodimentDefinition(
            "unknown",
            "Unknown / Unsupported Gripper",
            "unknown",
            0,
            None,
            [],
            [],
            [],
            [],
            "not_configured",
            [],
            ["two_contact_epsilon", "grip_width", "suction_seal", "wrap_coverage"],
            ["select_or_configure_embodiment"],
            ["do_not_engage_unknown_embodiment"],
            ["Select an embodiment before interpreting quality metrics."],
            "NOT_IMPLEMENTED",
            "unsupported",
            "DO_NOT_ENGAGE",
        ),
    }


class EmbodimentRegistry:
    """Registry for built-in and config-backed embodiment definitions."""

    def __init__(self, definitions: dict[str, EmbodimentDefinition] | None = None):
        self.definitions = definitions or default_embodiment_definitions()

    def get(self, embodiment_id: str | None) -> EmbodimentDefinition:
        return self.definitions.get(canonical_embodiment_id(embodiment_id), self.definitions["unknown"])

    def list(self) -> list[EmbodimentDefinition]:
        return [self.definitions[key] for key in EMBODIMENT_IDS if key in self.definitions]

    def to_dict(self) -> dict[str, Any]:
        return {key: value.to_dict() for key, value in self.definitions.items()}

    def register(self, definition: EmbodimentDefinition) -> None:
        self.definitions[canonical_embodiment_id(definition.embodiment_id)] = definition

    @classmethod
    def from_config_dir(cls, path: Path | None = None) -> "EmbodimentRegistry":
        registry = cls()
        folder = path or config_path("embodiments")
        if not Path(folder).exists():
            return registry
        for item in sorted(Path(folder).glob("*.yaml")):
            data = load_config_file(item, default={}) or {}
            embodiment_id = canonical_embodiment_id(data.get("embodiment_id") or item.stem)
            base = registry.get(embodiment_id)
            payload = base.to_dict()
            payload.update({k: v for k, v in data.items() if k in payload})
            payload["embodiment_id"] = embodiment_id
            registry.register(EmbodimentDefinition(**payload))
        return registry


@lru_cache(maxsize=1)
def load_registry() -> EmbodimentRegistry:
    return EmbodimentRegistry.from_config_dir()


def clear_registry_cache() -> None:
    """Clear cached config-backed embodiment definitions after config edits."""

    load_registry.cache_clear()


def active_embodiment_id(config: dict[str, Any] | None = None) -> str:
    config = config or load_config_file(config_path("embodiment.yaml"), default={}) or {}
    return canonical_embodiment_id(
        config.get("configured_embodiment")
        or config.get("active_embodiment")
        or config.get("embodiment_id")
        or "parallel_jaw_two_finger"
    )


def legacy_embodiment_id(embodiment_id: str | None) -> str:
    canonical = canonical_embodiment_id(embodiment_id)
    for legacy, mapped in LEGACY_EMBODIMENT_ALIASES.items():
        if mapped == canonical:
            return legacy
    return canonical


def registry_json() -> str:
    return json.dumps(load_registry().to_dict(), indent=2, sort_keys=True)

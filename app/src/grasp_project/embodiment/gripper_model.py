"""Configurable gripper embodiment models for contact interpretation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from grasp_project.common.config import OUTPUT_DIR, config_path, load_config_file
from grasp_project.common.file_io import write_json_atomic, write_text_atomic
from grasp_project.embodiment.base import canonical_embodiment_id
from grasp_project.embodiment.metric_applicability import metric_applicability
from grasp_project.embodiment.registry import load_registry, legacy_embodiment_id

SUPPORTED_EMBODIMENTS = {
    "two_point_parallel_gripper",
    "parallel_jaw_two_finger",
    "articulated_two_finger",
    "human_hand",
    "multi_finger_robotic_hand",
    "three_finger_grip",
    "three_finger_gripper",
    "vacuum_grip",
    "vacuum_suction",
    "soft_gripper",
    "custom",
    "unknown",
}


@dataclass
class GripperModel:
    embodiment_type: str
    expected_contact_count: int | str
    contact_model_type: str
    contact_patch_shape: str
    contact_quality_rules: list[str]
    allowed_quality_metrics: list[str]
    limitations: list[str]


DEFAULT_MODELS: dict[str, GripperModel] = {
    "two_point_parallel_gripper": GripperModel(
        "two_point_parallel_gripper",
        2,
        "opposing_side_contacts",
        "left_right_rectangular_patches",
        ["grip_width", "opposing_contact_balance", "epsilon", "patch_validity"],
        ["epsilon_force_closure", "grip_width", "visual_contact", "depth_consistency"],
        ["Assumes two opposing contacts and no force sensors."],
    ),
    "human_hand": GripperModel(
        "human_hand",
        "multiple",
        "irregular_multi_contact_regions",
        "freeform_regions",
        ["contact_coverage", "support_polygon", "stability_estimate"],
        ["contact_coverage", "support_polygon", "slip_risk"],
        ["Visual-only contacts are not true tactile/force measurements.", "Two-contact epsilon is not primary."],
    ),
    "three_finger_grip": GripperModel(
        "three_finger_grip",
        3,
        "three_patch_support",
        "three_discrete_patches",
        ["triangular_support", "contact_spread", "approximate_closure"],
        ["contact_spread", "support_polygon", "approximate_closure"],
        ["Three-contact closure is not implemented in the current epsilon worker."],
    ),
    "vacuum_grip": GripperModel(
        "vacuum_grip",
        1,
        "suction_patch",
        "area_patch",
        ["surface_normal", "patch_area", "flatness", "seal_risk"],
        ["surface_normal", "patch_area", "flatness", "seal_risk"],
        ["Two-contact force closure is not applicable for suction-only grasping."],
    ),
    "soft_gripper": GripperModel(
        "soft_gripper",
        "distributed",
        "deformable_contact_patch",
        "distributed_wrap_patch",
        ["wrap_coverage", "symmetry", "slip_risk"],
        ["coverage", "symmetry", "slip_risk"],
        ["Point-contact epsilon should be treated as secondary for deformable contact."],
    ),
    "unknown": GripperModel(
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        [],
        [],
        ["Select an embodiment before interpreting quality metrics."],
    ),
}


DISPLAY_METRICS_BY_EMBODIMENT: dict[str, list[dict[str, str]]] = {
    "parallel_jaw_two_finger": [
        {"metric_id": "grip_width", "label": "Grip Width"},
        {"metric_id": "opposition", "label": "Opposing Contacts"},
        {"metric_id": "two_contact_epsilon", "label": "Epsilon"},
    ],
    "articulated_two_finger": [
        {"metric_id": "opposition", "label": "Opposition"},
        {"metric_id": "contact_coverage", "label": "Contact Coverage"},
        {"metric_id": "contact_count_validity", "label": "Two Contacts"},
    ],
    "three_finger_gripper": [
        {"metric_id": "support_triangle", "label": "Support Triangle"},
        {"metric_id": "contact_coverage", "label": "Contact Spread"},
    ],
    "vacuum_suction": [
        {"metric_id": "suction_seal", "label": "Seal Area"},
        {"metric_id": "surface_flatness", "label": "Flatness"},
        {"metric_id": "normal_alignment", "label": "Normal Alignment"},
    ],
    "soft_gripper": [
        {"metric_id": "wrap_coverage", "label": "Wrap Coverage"},
        {"metric_id": "contact_coverage", "label": "Distributed Contact"},
    ],
    "multi_finger_robotic_hand": [
        {"metric_id": "support_polygon", "label": "Support Polygon"},
        {"metric_id": "contact_coverage", "label": "Contact Spread"},
    ],
    "human_hand": [
        {"metric_id": "support_polygon", "label": "Support Polygon"},
        {"metric_id": "contact_coverage", "label": "Contact Spread"},
    ],
    "custom": [
        {"metric_id": "contact_count_validity", "label": "Contact Count"},
    ],
    "unknown": [
        {"metric_id": "contact_count_validity", "label": "Contact Count"},
    ],
}

PRODUCTION_REFERENCE_EMBODIMENT = "parallel_jaw_two_finger"
BENCHMARK_ONLY_STATES = {"READY_FOR_BENCHMARKING", "READY_WITH_LIMITATIONS"}
CONFIG_ONLY_STATES = {"CONFIG_ONLY", "NOT_IMPLEMENTED"}


def embodiment_display_metrics(embodiment_id: str | None) -> list[dict[str, Any]]:
    """Return dashboard-visible metric rows for the selected embodiment."""

    canonical = canonical_embodiment_id(embodiment_id)
    applicability = metric_applicability(canonical)
    rows = []
    for item in DISPLAY_METRICS_BY_EMBODIMENT.get(canonical, DISPLAY_METRICS_BY_EMBODIMENT["unknown"]):
        metric_id = item["metric_id"]
        metric_state = applicability["metric_applicability"].get(metric_id, {})
        rows.append({
            "metric_id": metric_id,
            "label": item["label"],
            "status": metric_state.get("status"),
            "reason": metric_state.get("reason"),
        })
    return rows


def release_policy(embodiment_id: str | None) -> dict[str, Any]:
    """Describe whether an embodiment is production, benchmark-only, or config-only."""

    canonical = canonical_embodiment_id(embodiment_id)
    definition = load_registry().get(canonical)
    release_status = definition.release_status
    production_ready = canonical == PRODUCTION_REFERENCE_EMBODIMENT and release_status == "PRODUCTION_READY"
    benchmark_only = (canonical != PRODUCTION_REFERENCE_EMBODIMENT) and release_status in BENCHMARK_ONLY_STATES
    config_only = release_status in CONFIG_ONLY_STATES or (canonical != PRODUCTION_REFERENCE_EMBODIMENT and not benchmark_only)
    if production_ready:
        mode = "production_reference"
        note = "Production decisions use the validated parallel-jaw reference path."
    elif benchmark_only:
        mode = "display_only_benchmark_only"
        note = "Display-only and benchmark-only; not allowed to drive production close decisions."
    else:
        mode = "config_only"
        note = "Configuration/display support only until embodiment-specific validation passes."
    return {
        "release_status": release_status,
        "release_mode": mode,
        "production_ready": production_ready,
        "production_reference": production_ready,
        "benchmark_enabled": bool(benchmark_only or production_ready),
        "benchmark_only": benchmark_only,
        "display_only": not production_ready,
        "config_only": config_only,
        "production_decisions_allowed": production_ready,
        "release_note": note,
    }


def embodiment_selector_options() -> list[dict[str, Any]]:
    """Return dashboard selector choices without importing camera code."""

    options = []
    for definition in load_registry().list():
        if definition.embodiment_id == "unknown":
            continue
        policy = release_policy(definition.embodiment_id)
        options.append({
            "embodiment_id": definition.embodiment_id,
            "display_name": definition.display_name,
            "release_status": definition.release_status,
            "release_mode": policy["release_mode"],
            "benchmark_enabled": policy["benchmark_enabled"],
            "production_decisions_allowed": policy["production_decisions_allowed"],
        })
    return options


def save_active_embodiment(embodiment_id: str, path=None) -> dict[str, Any]:
    """Persist the configured active embodiment for future frames."""

    canonical = canonical_embodiment_id(embodiment_id)
    if canonical == "unknown":
        raise ValueError(f"Unsupported embodiment: {embodiment_id}")
    config_file = path or config_path("embodiment.yaml")
    config = load_config_file(config_file, default={}) or {}
    config["active_embodiment"] = legacy_embodiment_id(canonical)
    config["configured_embodiment"] = canonical
    config["selection_mode"] = config.get("selection_mode") or "configured"
    config["embodiment_source"] = "dashboard_selector"
    config["lock_embodiment_during_grasp"] = True
    _write_yaml_like(config_file, config)
    return write_embodiment_status(config_file)


def _write_yaml_like(path, data: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(data, sort_keys=False)
    except Exception:
        text = json.dumps(data, indent=2)
    write_text_atomic(path, text)


def load_active_embodiment(path=None) -> GripperModel:
    config = load_config_file(path or config_path("embodiment.yaml"), default={}) or {}
    raw_active = str(config.get("active_embodiment") or config.get("configured_embodiment") or "two_point_parallel_gripper")
    canonical = canonical_embodiment_id(raw_active)
    active = legacy_embodiment_id(canonical)
    if active not in DEFAULT_MODELS:
        active = canonical
    model = DEFAULT_MODELS.get(active, DEFAULT_MODELS["unknown"])
    definition = load_registry().get(canonical)
    if active not in DEFAULT_MODELS and canonical != "unknown":
        model = GripperModel(
            embodiment_type=active,
            expected_contact_count=definition.expected_contact_count_min if definition.expected_contact_count_min == definition.expected_contact_count_max else "multiple",
            contact_model_type=",".join(definition.contact_geometry_types) or "custom",
            contact_patch_shape=definition.contact_geometry_types[0] if definition.contact_geometry_types else "custom",
            contact_quality_rules=definition.readiness_rules,
            allowed_quality_metrics=definition.supported_quality_metrics,
            limitations=definition.limitations,
        )
    return GripperModel(
        embodiment_type=active,
        expected_contact_count=config.get("expected_contact_count", definition.expected_contact_count_min if definition.expected_contact_count_min == definition.expected_contact_count_max else model.expected_contact_count),
        contact_model_type=config.get("contact_geometry", model.contact_model_type),
        contact_patch_shape=config.get("contact_patch_shape", model.contact_patch_shape),
        contact_quality_rules=list(config.get("contact_quality_rules", model.contact_quality_rules) or definition.readiness_rules or []),
        allowed_quality_metrics=list(config.get("allowed_quality_metrics", model.allowed_quality_metrics) or definition.supported_quality_metrics or []),
        limitations=list(config.get("limitations", model.limitations) or definition.limitations or []),
    )


def embodiment_status(path=None) -> dict[str, Any]:
    model = load_active_embodiment(path)
    canonical = canonical_embodiment_id(model.embodiment_type)
    definition = load_registry().get(canonical)
    applicability = metric_applicability(canonical)
    policy = release_policy(canonical)
    display_metrics = embodiment_display_metrics(canonical)
    displayed_metric_ids = {item["metric_id"] for item in display_metrics}
    payload = asdict(model)
    payload.update({
        "active_embodiment": model.embodiment_type,
        "canonical_embodiment_id": canonical,
        "display_name": definition.display_name,
        "category": definition.category,
        **policy,
        "stable_status": definition.stable_status,
        "engagement_action": definition.engagement_action,
        "detected_contact_count": None,
        "expected_contacts": model.expected_contact_count,
        "contact_model_valid": model.embodiment_type != "unknown",
        "contact_model_limitations": model.limitations,
        "primary_metric": applicability["primary_metric"],
        "primary_quality_metric": applicability["primary_metric"],
        "secondary_quality_metrics": applicability["secondary_metrics"],
        "hidden_metrics": applicability["hidden_metrics"],
        "metric_applicability": applicability["metric_applicability"],
        "metric_applicability_reasons": applicability["metric_applicability_reasons"],
        "dashboard_metrics": display_metrics,
        "dashboard_metric_ids": sorted(displayed_metric_ids),
        "invalid_metrics_hidden": all(metric not in displayed_metric_ids for metric in applicability["hidden_metrics"]),
        "selector_options": embodiment_selector_options(),
    })
    return payload


def write_embodiment_status(path=None) -> dict[str, Any]:
    status = embodiment_status(path)
    write_json_atomic(OUTPUT_DIR / "embodiment_status.json", status)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Embodiment contact-model diagnostics.")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(write_embodiment_status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

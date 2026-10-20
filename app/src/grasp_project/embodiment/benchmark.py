"""Benchmark-facing embodiment policy helpers.

Benchmark Lab consumes these summaries to decide which metrics can be shown for
an embodiment. Production close/lift actions remain restricted to the validated
parallel-jaw reference path.
"""

from __future__ import annotations

from typing import Any

from grasp_project.embodiment.base import EmbodimentQualityResult, canonical_embodiment_id
from grasp_project.embodiment.gripper_model import release_policy
from grasp_project.embodiment.metric_applicability import metric_applicability
from grasp_project.embodiment.registry import load_registry


def production_action_allowed(embodiment_id: str | None) -> bool:
    """Return whether an embodiment may control production grasp actions."""

    return bool(release_policy(embodiment_id)["production_decisions_allowed"])


def benchmark_profile(embodiment_id: str | None) -> dict[str, Any]:
    """Return the release/applicability payload used by Benchmark Lab."""

    canonical = canonical_embodiment_id(embodiment_id)
    definition = load_registry().get(canonical)
    policy = release_policy(canonical)
    applicability = metric_applicability(canonical)
    applicable_metrics = [
        metric
        for metric, data in applicability["metric_applicability"].items()
        if data.get("status") in {"APPLICABLE", "PARTIALLY_APPLICABLE", "EXPERIMENTAL"}
    ]
    return {
        "embodiment_id": canonical,
        "display_name": definition.display_name,
        "release_status": definition.release_status,
        "release_mode": policy["release_mode"],
        "production_actions_allowed": policy["production_decisions_allowed"],
        "benchmark_enabled": bool(policy["benchmark_only"] or policy["production_ready"]),
        "benchmark_only": policy["benchmark_only"],
        "display_only": policy["display_only"],
        "config_only": policy["config_only"],
        "primary_metric": applicability["primary_metric"],
        "supported_metrics": applicable_metrics,
        "hidden_metrics": applicability["hidden_metrics"],
        "metric_applicability": applicability["metric_applicability"],
        "release_note": policy["release_note"],
    }


def benchmark_result_from_quality(result: EmbodimentQualityResult) -> dict[str, Any]:
    """Wrap an embodiment-specific quality result with release-policy fields."""

    profile = benchmark_profile(result.embodiment_id)
    production_allowed = bool(profile["production_actions_allowed"])
    decision = result.decision
    if not production_allowed and decision in {"READY_TO_CLOSE", "READY_TO_LIFT"}:
        decision = "DISPLAY_ONLY"
    return {
        "embodiment_id": result.embodiment_id,
        "benchmark_enabled": profile["benchmark_enabled"],
        "benchmark_only": profile["benchmark_only"],
        "production_actions_allowed": production_allowed,
        "quality_result": result.to_dict(),
        "benchmark_decision": decision,
        "production_decision_blocked": not production_allowed,
        "release_status": profile["release_status"],
        "metric_applicability": profile["metric_applicability"],
    }


__all__ = [
    "benchmark_profile",
    "benchmark_result_from_quality",
    "production_action_allowed",
]

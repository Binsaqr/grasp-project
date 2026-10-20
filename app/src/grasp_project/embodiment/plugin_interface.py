"""Embodiment plugin interface and safe plugin loading.

Plugins are contact-model adapters. They may estimate/validate a ContactSet and
compute embodiment-specific display or benchmark quality. They must not bypass
release policy: only the production reference embodiment can control production
actions.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Type

from grasp_project.common.config import config_path, load_config_file
from grasp_project.embodiment.base import ContactModelPlugin, EmbodimentQualityResult, QualityMetricPlugin
from grasp_project.embodiment.contact_primitives import ContactSet
from grasp_project.embodiment.gripper_model import release_policy
from grasp_project.embodiment.registry import load_registry


class EmbodimentPlugin(ContactModelPlugin, QualityMetricPlugin, Protocol):
    """Minimal embodiment plugin contract."""

    embodiment_id: str

    def estimate_contacts(self, evidence: dict[str, Any]) -> ContactSet:
        ...

    def validate_contact_set(self, contact_set: ContactSet) -> ContactSet:
        ...

    def compute_quality(self, contact_set: ContactSet, evidence: dict[str, Any] | None = None) -> EmbodimentQualityResult:
        ...


@dataclass
class PluginLoadResult:
    ok: bool
    embodiment_id: str
    plugin: EmbodimentPlugin | None = None
    module: str | None = None
    class_name: str | None = None
    source: str = "unknown"
    error: str | None = None
    production_actions_allowed: bool = False
    benchmark_enabled: bool = False
    release_status: str = "CONFIG_ONLY"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plugin"] = None if self.plugin is None else self.plugin.__class__.__name__
        return payload


def builtin_plugin_classes() -> dict[str, Type[EmbodimentPlugin]]:
    """Return built-in plugins without importing camera or dashboard code."""

    from grasp_project.embodiment.plugins.articulated_two_finger import ArticulatedTwoFingerPlugin
    from grasp_project.embodiment.plugins.custom import CustomPlugin
    from grasp_project.embodiment.plugins.human_hand import HumanHandPlugin
    from grasp_project.embodiment.plugins.multi_finger_hand import MultiFingerHandPlugin
    from grasp_project.embodiment.plugins.parallel_jaw import ParallelJawPlugin
    from grasp_project.embodiment.plugins.soft_gripper import SoftGripperPlugin
    from grasp_project.embodiment.plugins.three_finger import ThreeFingerPlugin
    from grasp_project.embodiment.plugins.vacuum_suction import VacuumSuctionPlugin

    return {
        "parallel_jaw_two_finger": ParallelJawPlugin,
        "articulated_two_finger": ArticulatedTwoFingerPlugin,
        "three_finger_gripper": ThreeFingerPlugin,
        "multi_finger_robotic_hand": MultiFingerHandPlugin,
        "human_hand": HumanHandPlugin,
        "vacuum_suction": VacuumSuctionPlugin,
        "soft_gripper": SoftGripperPlugin,
        "custom": CustomPlugin,
    }


def load_builtin_plugin(embodiment_id: str | None) -> PluginLoadResult:
    """Instantiate a built-in embodiment plugin."""

    definition = load_registry().get(embodiment_id)
    cls = builtin_plugin_classes().get(definition.embodiment_id)
    policy = release_policy(definition.embodiment_id)
    if cls is None:
        return PluginLoadResult(
            ok=False,
            embodiment_id=definition.embodiment_id,
            source="builtin",
            error="builtin_plugin_not_available",
            release_status=definition.release_status,
        )
    return PluginLoadResult(
        ok=True,
        embodiment_id=definition.embodiment_id,
        plugin=cls(),
        module=cls.__module__,
        class_name=cls.__name__,
        source="builtin",
        production_actions_allowed=bool(policy["production_decisions_allowed"]),
        benchmark_enabled=bool(policy["benchmark_only"] or policy["production_ready"]),
        release_status=definition.release_status,
    )


def load_custom_plugin(config: dict[str, Any] | None = None) -> PluginLoadResult:
    """Load a custom plugin from config, failing safely by default.

    Expected config keys:
    - embodiment_id: custom
    - plugin_module: importable module path
    - plugin_class: class implementing the plugin interface

    Custom plugins are never production-action capable unless their release
    policy has separately been validated and changed by configuration/review.
    """

    config = config or load_config_file(config_path("custom_gripper_plugin.yaml"), default={}) or {}
    embodiment_id = str(config.get("embodiment_id") or "custom")
    module_name = str(config.get("plugin_module") or "")
    class_name = str(config.get("plugin_class") or "")
    policy = release_policy(embodiment_id)
    if not module_name or not class_name:
        result = load_builtin_plugin("custom")
        result.source = "custom_config_fallback"
        result.error = "custom_plugin_module_or_class_not_configured"
        return result
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        plugin = cls()
        validate_plugin_contract(plugin)
    except Exception as exc:
        return PluginLoadResult(
            ok=False,
            embodiment_id=embodiment_id,
            module=module_name,
            class_name=class_name,
            source="custom_config",
            error=str(exc),
            release_status=policy["release_status"],
        )
    return PluginLoadResult(
        ok=True,
        embodiment_id=embodiment_id,
        plugin=plugin,
        module=module_name,
        class_name=class_name,
        source="custom_config",
        production_actions_allowed=bool(policy["production_decisions_allowed"]),
        benchmark_enabled=bool(policy["benchmark_only"] or policy["production_ready"]),
        release_status=policy["release_status"],
    )


def validate_plugin_contract(plugin: Any) -> None:
    """Raise ValueError if a plugin misses required methods."""

    for name in ("estimate_contacts", "validate_contact_set", "compute_quality"):
        if not callable(getattr(plugin, name, None)):
            raise ValueError(f"plugin_missing_required_method:{name}")
    if not getattr(plugin, "embodiment_id", None):
        raise ValueError("plugin_missing_embodiment_id")


def compute_quality_for_evidence(embodiment_id: str | None, evidence: dict[str, Any] | None = None) -> EmbodimentQualityResult:
    """Convenience path for Benchmark Lab and tests."""

    loaded = load_builtin_plugin(embodiment_id)
    if not loaded.ok or loaded.plugin is None:
        loaded = load_builtin_plugin("custom")
    contact_set = loaded.plugin.estimate_contacts(evidence or {})
    result = loaded.plugin.compute_quality(contact_set, evidence=evidence or {})
    policy = release_policy(loaded.embodiment_id)
    if not policy["production_decisions_allowed"]:
        result.warnings.append("display_only_or_benchmark_only")
        if str(result.decision or "").upper().startswith("READY_"):
            result.decision = "READY_FOR_BENCHMARKING" if policy["benchmark_only"] else "DISPLAY_ONLY"
    return result


__all__ = [
    "EmbodimentPlugin",
    "PluginLoadResult",
    "builtin_plugin_classes",
    "compute_quality_for_evidence",
    "load_builtin_plugin",
    "load_custom_plugin",
    "validate_plugin_contract",
]

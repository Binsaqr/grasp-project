"""Embodiment-aware contact model metadata and universal contact primitives."""

from grasp_project.embodiment.base import (
    EmbodimentDefinition,
    EmbodimentQualityResult,
    EmbodimentSelectionResult,
    canonical_embodiment_id,
)
from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet
from grasp_project.embodiment.benchmark import benchmark_profile, production_action_allowed
from grasp_project.embodiment.registry import EmbodimentRegistry, load_registry

__all__ = [
    "benchmark_profile",
    "ContactObservation",
    "ContactSet",
    "EmbodimentDefinition",
    "EmbodimentQualityResult",
    "EmbodimentRegistry",
    "EmbodimentSelectionResult",
    "canonical_embodiment_id",
    "load_registry",
    "production_action_allowed",
]

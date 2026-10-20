"""Display-only benchmarking helpers for the main web dashboard.

The Benchmark Lab compares feature masks against existing synchronized status
outputs. It never opens cameras and never changes production grasp decisions.
"""

from .benchmark_runner import BenchmarkRunner, load_benchmark_settings
from .bottle_only import generate_group_combinations
from .bottle_only import write_expanded_report_outputs
from .feature_mask import build_benchmark_lab_status, validate_feature_selection
from .feature_registry import (
    FEATURE_CATEGORIES,
    PRESET_DEFINITIONS,
    default_comparison_configurations,
    default_run_configurations,
    feature_registry_payload,
    preset_feature_ids,
)
from .masked_feature_view import FeaturePacket, MaskedFeatureView, build_feature_packet
from .evaluators import run_poison_value_tests
from .trial_recorder import BenchmarkTrialRecorder

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTrialRecorder",
    "FeaturePacket",
    "FEATURE_CATEGORIES",
    "MaskedFeatureView",
    "PRESET_DEFINITIONS",
    "build_benchmark_lab_status",
    "build_feature_packet",
    "default_comparison_configurations",
    "default_run_configurations",
    "feature_registry_payload",
    "generate_group_combinations",
    "load_benchmark_settings",
    "preset_feature_ids",
    "run_poison_value_tests",
    "validate_feature_selection",
    "write_expanded_report_outputs",
]

"""Finalization checks for embodiment-aware gripper support."""

from grasp_project.embodiment.benchmark import benchmark_profile, benchmark_result_from_quality, production_action_allowed
from grasp_project.embodiment.metric_applicability import NOT_APPLICABLE, metric_applicability
from grasp_project.embodiment.plugin_interface import compute_quality_for_evidence, load_builtin_plugin, load_custom_plugin, validate_plugin_contract
from grasp_project.embodiment.plugins.custom import CustomPlugin
from grasp_project.embodiment.plugins.three_finger import ThreeFingerPlugin
from grasp_project.embodiment.plugins.vacuum_suction import VacuumSuctionPlugin
from grasp_project.embodiment.pointcloud_validation import validate_pointcloud_contact_set
from gripper.embodiment_adapter import contact_set_from_gripper_evidence, quality_from_gripper_evidence
from visual_contact_model import contact_set_from_visual_contact_model


def _patch(center, confidence=90.0):
    x, y = center
    return {
        "center": [x, y],
        "polygon": [[x - 2, y - 5], [x + 2, y - 5], [x + 2, y + 5], [x - 2, y + 5]],
        "quality_score": confidence,
        "valid_depth_ratio": 0.9,
    }


def _contact(center, confidence=90.0):
    return {"image_center_px": list(center), "confidence": confidence, "valid": True}


def test_builtin_plugin_registry_loads_all_supported_models():
    for embodiment_id in (
        "parallel_jaw_two_finger",
        "articulated_two_finger",
        "three_finger_gripper",
        "multi_finger_robotic_hand",
        "human_hand",
        "vacuum_suction",
        "soft_gripper",
        "custom",
    ):
        loaded = load_builtin_plugin(embodiment_id)
        assert loaded.ok, loaded.to_dict()
        assert loaded.plugin is not None
        validate_plugin_contract(loaded.plugin)


def test_custom_plugin_config_fallback_is_display_only():
    loaded = load_custom_plugin({})
    assert loaded.ok
    assert loaded.embodiment_id == "custom"
    assert loaded.source == "custom_config_fallback"
    assert loaded.production_actions_allowed is False

    result = compute_quality_for_evidence("custom", {"contacts": [_contact([10, 10])]})
    assert result.decision == "DISPLAY_ONLY"
    assert "custom_plugin_display_only" in result.warnings


def test_parallel_regression_keeps_reference_two_contact_quality():
    result = compute_quality_for_evidence(
        "parallel_jaw_two_finger",
        {"left_clamp_patch": _patch([10, 20]), "right_clamp_patch": _patch([40, 20])},
    )
    assert result.primary_metric_name == "two_contact_epsilon"
    assert result.decision == "READY_TO_CLOSE"
    assert production_action_allowed("parallel_jaw_two_finger") is True


def test_non_parallel_ready_decisions_are_sanitized_in_shared_quality_path():
    three = compute_quality_for_evidence(
        "three_finger_gripper",
        {"contacts": [_contact([0, 0]), _contact([20, 0]), _contact([10, 20])]},
    )
    soft = compute_quality_for_evidence(
        "soft_gripper",
        {"distributed_contacts": [{"center": [10, 10], "area_px": 500, "confidence": 80}]},
    )
    vacuum = compute_quality_for_evidence(
        "vacuum_suction",
        {"suction_area": {"center": [20, 20], "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "area_px": 100, "confidence": 85, "depth_valid_ratio": 0.9}},
    )

    assert three.decision == "READY_FOR_BENCHMARKING"
    assert soft.decision == "READY_FOR_BENCHMARKING"
    assert vacuum.decision == "DISPLAY_ONLY"


def test_three_finger_missing_contact_blocks_without_two_contact_epsilon():
    plugin = ThreeFingerPlugin()
    contact_set = plugin.estimate_contacts({"contacts": [_contact([0, 0]), _contact([20, 0])]})
    result = plugin.compute_quality(contact_set)
    applicability = metric_applicability("three_finger_gripper")

    assert contact_set.contact_set_valid is False
    assert "missing_three_finger_contact" in contact_set.blockers
    assert result.primary_metric_name == "support_triangle"
    assert applicability["metric_applicability"]["two_contact_epsilon"]["status"] == NOT_APPLICABLE


def test_vacuum_uses_single_suction_area_not_two_contacts():
    plugin = VacuumSuctionPlugin()
    contact_set = plugin.estimate_contacts(
        {"suction_area": {"center": [20, 20], "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "area_px": 100, "confidence": 85, "depth_valid_ratio": 0.9}}
    )
    result = plugin.compute_quality(contact_set)

    assert contact_set.detected_contact_count == 1
    assert contact_set.contact_set_valid is True
    assert result.primary_metric_name == "suction_seal"
    assert metric_applicability("vacuum_suction")["metric_applicability"]["two_contact_epsilon"]["status"] == NOT_APPLICABLE


def test_pointcloud_contact_validation_uses_embodiment_counts_and_freshness():
    three_contacts = ThreeFingerPlugin().estimate_contacts(
        {"contacts": [_contact([0, 0]), _contact([20, 0]), _contact([10, 20])]}
    )
    three_valid = validate_pointcloud_contact_set(
        three_contacts,
        {"freshness": "CURRENT", "contact_3d_valid": True, "cloud_valid_for_contact_validation": True, "contact_points_on_bottle_cloud": True},
    )
    assert three_valid["required_contact_count"] == 3
    assert three_valid["pointcloud_contact_validation_status"] == "CURRENT_VALID"

    two_of_three = ThreeFingerPlugin().estimate_contacts({"contacts": [_contact([0, 0]), _contact([20, 0])]})
    three_invalid = validate_pointcloud_contact_set(
        two_of_three,
        {"freshness": "CURRENT", "contact_3d_valid": True, "cloud_valid_for_contact_validation": True},
    )
    assert three_invalid["pointcloud_contact_validation_status"] == "CURRENT_INVALID"
    assert "missing_required_pointcloud_contacts" in three_invalid["pointcloud_contact_reason"]

    stale = validate_pointcloud_contact_set(
        three_contacts,
        {"freshness": "STALE", "contact_3d_valid": False},
    )
    assert stale["pointcloud_contact_validation_status"] == "STALE_EXCLUDED"


def test_benchmark_profile_and_result_never_promote_experimental_to_production():
    profile = benchmark_profile("three_finger_gripper")
    assert profile["benchmark_enabled"] is True
    assert profile["production_actions_allowed"] is False

    result = compute_quality_for_evidence(
        "three_finger_gripper",
        {"contacts": [_contact([0, 0]), _contact([20, 0]), _contact([10, 20])]},
    )
    payload = benchmark_result_from_quality(result)
    assert payload["benchmark_only"] is True
    assert payload["production_decision_blocked"] is True
    assert payload["benchmark_decision"] == "READY_FOR_BENCHMARKING"


def test_gripper_compatibility_adapter_returns_contact_set_and_sanitized_quality():
    contact_set = contact_set_from_gripper_evidence(
        "parallel_jaw_two_finger",
        {"left_clamp_patch": _patch([10, 20]), "right_clamp_patch": _patch([40, 20])},
    )
    quality = quality_from_gripper_evidence(
        "vacuum_suction",
        {"suction_area": {"center": [20, 20], "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "area_px": 100, "confidence": 85, "depth_valid_ratio": 0.9}},
    )
    assert contact_set.contact_set_valid is True
    assert quality.decision == "DISPLAY_ONLY"


def test_visual_contact_model_adapter_preserves_legacy_parallel_patches():
    result = {
        "left_clamp_patch": _patch([10, 20]),
        "right_clamp_patch": _patch([40, 20]),
        "visual_contact_ready": True,
        "visual_contact_pair_quality_score": 88.0,
    }
    contact_set = contact_set_from_visual_contact_model(result)
    assert contact_set is not None
    assert contact_set.expected_contact_count == 2
    assert contact_set.detected_contact_count == 2
    assert contact_set.contact_set_valid is True
    assert contact_set.left_contact.contact_id == "left_contact"

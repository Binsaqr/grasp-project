"""Camera-free tests for the universal embodiment framework."""

from grasp_project.embodiment.base import EMBODIMENT_IDS, canonical_embodiment_id
from grasp_project.embodiment.contact_primitives import ContactObservation, ContactSet, contact_from_legacy_patch
from grasp_project.embodiment.detector import EmbodimentLockState, select_embodiment
from grasp_project.embodiment.gripper_model import embodiment_display_metrics, release_policy
from grasp_project.embodiment.metric_applicability import APPLICABLE, NOT_APPLICABLE, metric_applicability
from grasp_project.embodiment.plugins.parallel_jaw import ParallelJawPlugin
from grasp_project.embodiment.plugins.soft_gripper import SoftGripperPlugin
from grasp_project.embodiment.plugins.three_finger import ThreeFingerPlugin
from grasp_project.embodiment.plugins.vacuum_suction import VacuumSuctionPlugin
from grasp_project.embodiment.registry import load_registry


def _patch(center, confidence=90.0):
    x, y = center
    return {
        "center": [x, y],
        "polygon": [[x - 2, y - 5], [x + 2, y - 5], [x + 2, y + 5], [x - 2, y + 5]],
        "quality_score": confidence,
        "valid_depth_ratio": 0.9,
    }


def test_registry_loads_all_canonical_embodiments():
    registry = load_registry()
    loaded = {definition.embodiment_id for definition in registry.list()}
    assert set(EMBODIMENT_IDS).issubset(loaded)
    assert registry.get("two_point_parallel_gripper").embodiment_id == "parallel_jaw_two_finger"
    assert canonical_embodiment_id("vacuum_grip") == "vacuum_suction"


def test_unknown_embodiment_fails_safely():
    definition = load_registry().get("definitely_not_real")
    assert definition.embodiment_id == "unknown"
    applicability = metric_applicability("unknown")
    assert applicability["primary_metric"] == "contact_count_validity"
    assert "two_contact_epsilon" in applicability["hidden_metrics"]


def test_embodiment_lock_prevents_active_switch():
    lock = EmbodimentLockState()
    selection = select_embodiment(config={"selection_mode": "configured", "configured_embodiment": "parallel_jaw_two_finger"})
    lock.lock(selection)
    later = select_embodiment(
        config={"selection_mode": "configured", "configured_embodiment": "vacuum_suction", "lock_embodiment_during_grasp": True},
        lock_state=lock,
    )
    assert later.embodiment_id == "parallel_jaw_two_finger"
    assert later.embodiment_locked is True


def test_parallel_adapter_preserves_left_right_fields():
    left = contact_from_legacy_patch("left_contact", "parallel_jaw_two_finger", _patch([10, 20]), associated_finger_or_element="left_inner_jaw")
    right = contact_from_legacy_patch("right_contact", "parallel_jaw_two_finger", _patch([40, 20]), associated_finger_or_element="right_inner_jaw")
    contact_set = ContactSet("parallel_jaw_two_finger", 2, [left, right])
    assert contact_set.left_contact.contact_id == "left_contact"
    assert contact_set.right_contact.contact_id == "right_contact"
    assert contact_set.to_dict()["detected_contact_count"] == 2


def test_parallel_plugin_valid_pair_and_missing_contact():
    plugin = ParallelJawPlugin()
    valid = plugin.estimate_contacts({"left_clamp_patch": _patch([10, 20]), "right_clamp_patch": _patch([40, 20])})
    result = plugin.compute_quality(valid)
    assert valid.contact_set_valid is True
    assert result.primary_metric_name == "two_contact_epsilon"
    assert result.decision == "READY_TO_CLOSE"
    missing = plugin.estimate_contacts({"left_clamp_patch": _patch([10, 20])})
    assert missing.contact_set_valid is False
    assert "missing_left_or_right_contact" in missing.blockers


def test_parallel_jaw_remains_only_production_reference():
    parallel = release_policy("parallel_jaw_two_finger")
    assert parallel["release_status"] == "PRODUCTION_READY"
    assert parallel["production_decisions_allowed"] is True
    for embodiment_id in ("three_finger_gripper", "vacuum_suction", "soft_gripper"):
        policy = release_policy(embodiment_id)
        assert policy["production_decisions_allowed"] is False
        assert policy["display_only"] is True


def test_three_finger_does_not_apply_two_contact_epsilon():
    plugin = ThreeFingerPlugin()
    contacts = [
        {"image_center_px": [0, 0], "confidence": 90},
        {"image_center_px": [20, 0], "confidence": 90},
        {"image_center_px": [10, 20], "confidence": 90},
    ]
    contact_set = plugin.estimate_contacts({"contacts": contacts})
    result = plugin.compute_quality(contact_set)
    assert contact_set.contact_set_valid is True
    assert result.primary_metric_name == "support_triangle"
    assert metric_applicability("three_finger_gripper")["metric_applicability"]["two_contact_epsilon"]["status"] == NOT_APPLICABLE
    assert "two_contact_epsilon" not in {item["metric_id"] for item in embodiment_display_metrics("three_finger_gripper")}


def test_vacuum_hides_grip_width_and_epsilon():
    plugin = VacuumSuctionPlugin()
    contact_set = plugin.estimate_contacts({"suction_area": {"center": [20, 20], "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "area_px": 100, "confidence": 85, "depth_valid_ratio": 0.9}})
    result = plugin.compute_quality(contact_set)
    applicability = metric_applicability("vacuum_suction")
    assert contact_set.contact_set_valid is True
    assert result.primary_metric_name == "suction_seal"
    assert "grip_width" in applicability["hidden_metrics"]
    assert applicability["metric_applicability"]["two_contact_epsilon"]["status"] == NOT_APPLICABLE
    assert "two_contact_epsilon" not in {item["metric_id"] for item in embodiment_display_metrics("vacuum_suction")}


def test_soft_gripper_keeps_distributed_contact():
    plugin = SoftGripperPlugin()
    contact_set = plugin.estimate_contacts({"distributed_contacts": [{"center": [10, 10], "area_px": 500, "confidence": 80}]})
    result = plugin.compute_quality(contact_set)
    applicability = metric_applicability("soft_gripper")
    assert contact_set.contact_set_valid is True
    assert result.primary_metric_name == "wrap_coverage"
    assert result.decision == "READY_TO_WRAP"
    assert applicability["metric_applicability"]["two_contact_epsilon"]["status"] == NOT_APPLICABLE
    assert "two_contact_epsilon" not in {item["metric_id"] for item in embodiment_display_metrics("soft_gripper")}


def test_detector_requires_persistence_and_avoids_one_frame_gripper_claim():
    one_frame = select_embodiment(
        evidence={"visible_finger_like_elements": 2, "parallel_jaw_symmetry": True, "temporal_persistence_frames": 1},
        config={"selection_mode": "automatic", "allow_automatic_detection": True},
    )
    assert one_frame.state == "UNKNOWN"
    persistent = select_embodiment(
        evidence={"visible_finger_like_elements": 2, "parallel_jaw_symmetry": True, "temporal_persistence_frames": 5},
        config={"selection_mode": "automatic", "allow_automatic_detection": True, "minimum_detection_confidence": 0.75},
    )
    assert persistent.embodiment_id == "parallel_jaw_two_finger"

"""Closing requires explicit visibility, contact readiness, and pose evidence."""

import pytest

from pregrasp_safety import evaluate_pregrasp_safety


def _healthy_inputs():
    pose = {
        "detected": True,
        "center_px": [160, 120],
        "detection_confidence": 1.0,
        "depth_confidence": 1.0,
        "axis_angle_degrees": 90.0,
    }
    plan = {
        "full_bottle_visible": True,
        "contact_model_ready": True,
        "body_fraction": 0.5,
        "target_contact_quality": 100.0,
        "raw_grip_width_score": 100.0,
        "positioning_guidance_score": 100.0,
        "left_right_balance_score": 100.0,
        "left_contact_px": [130, 120],
        "right_contact_px": [190, 120],
        "contacts_on_body": True,
        "grip_width_in_range": True,
        "left_reachable": True,
        "right_reachable": True,
    }
    return pose, plan, {"force_closure": True, "epsilon": 0.2}


def _evaluate(pose, plan, epsilon):
    return evaluate_pregrasp_safety(pose, plan, epsilon, (240, 320, 3))


def test_explicit_healthy_evidence_preserves_ready_decision():
    result = _evaluate(*_healthy_inputs())

    assert result["decision"] == "READY TO CLOSE"
    assert result["final_score"] == 100.0
    assert result["hard_gates"]["passed"] is True


@pytest.mark.parametrize(
    "section, field, gate, reason",
    [
        (
            "plan",
            "full_bottle_visible",
            "full_bottle_visible",
            "visibility evidence is missing",
        ),
        (
            "plan",
            "contact_model_ready",
            "contact_model_ready",
            "readiness evidence is missing",
        ),
        ("pose", "axis_angle_degrees", "axis_pose", "axis/pose evidence is missing"),
    ],
)
@pytest.mark.parametrize("missing", [True, False], ids=["absent", "null"])
def test_missing_required_evidence_blocks_closing(
    section, field, gate, reason, missing
):
    pose, plan, epsilon = _healthy_inputs()
    source = pose if section == "pose" else plan
    if missing:
        del source[field]
    else:
        source[field] = None

    result = _evaluate(pose, plan, epsilon)

    assert result["decision"] == "DO NOT CLOSE"
    assert result["hard_gates"]["passed"] is False
    assert result["hard_gates"]["status"][gate] is False
    assert result["final_score"] <= 49.0
    assert any(reason in message for message in result["failure_reasons"])


@pytest.mark.parametrize("field", ["full_bottle_visible", "contact_model_ready"])
def test_explicit_failed_readiness_still_blocks_closing(field):
    pose, plan, epsilon = _healthy_inputs()
    plan[field] = False

    result = _evaluate(pose, plan, epsilon)

    assert result["decision"] == "DO NOT CLOSE"
    assert result["hard_gates"]["status"][field] is False


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), "invalid"])
def test_invalid_axis_evidence_blocks_closing_without_inventing_an_angle(angle):
    pose, plan, epsilon = _healthy_inputs()
    pose["axis_angle_degrees"] = angle

    result = _evaluate(pose, plan, epsilon)

    assert result["decision"] == "DO NOT CLOSE"
    assert result["axis_angle_degrees"] is None
    assert result["axis_vertical_error_degrees"] is None
    assert result["hard_gates"]["status"]["axis_pose"] is False

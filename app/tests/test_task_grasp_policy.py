from grasp_project.planning.task_grasp_policy import plan_task_grasp
from grasp_project.realsense.object_state import ObjectState


def _state(orientation, *, available=True, occluded=False, reasons=None, tilt=0.0):
    return ObjectState(
        orientation=orientation,
        tilt_angle_deg=tilt,
        standing_confidence=1.0 if orientation == "STANDING" else 0.0,
        occluded=occluded,
        available_for_pickup=available,
        reasons=reasons or [],
        evidence={},
    )


def test_standing_uses_existing_side_candidate_and_epsilon_evidence():
    candidate = {
        "candidate_id": "existing-1",
        "left_contact_px": [10, 20],
        "right_contact_px": [30, 20],
        "contacts_on_body": True,
        "epsilon_value": 0.2,
        "epsilon_score": 80.0,
        "force_closure": True,
    }
    plan = plan_task_grasp(_state("STANDING"), pregrasp_plan={"selected_candidate": candidate}, target_pose={"name": "place"})
    assert plan.strategy == "SIDE_ANTIPODAL"
    assert plan.approach == "SIDE"
    assert plan.selected_candidate["candidate_id"] == "existing-1"
    assert plan.epsilon_evidence["recomputed"] is False
    assert plan.epsilon_evidence["status"] == "ESTIMATED_PLANNING_ONLY_FROM_VISUAL_CANDIDATE"
    assert plan.epsilon_evidence["applicable_to_live_contact_quality"] is False
    assert plan.selected_candidate["epsilon_physically_validated"] is False
    assert not plan.physically_validated


def test_lying_uses_top_grasp_and_allows_controlled_rotation():
    plan = plan_task_grasp(_state("LYING", tilt=90.0))
    assert plan.strategy == "TOP_GRASP_ROTATE_UPRIGHT"
    assert plan.approach == "TOP"
    assert plan.allow_rotation
    assert plan.rotation_target_deg == 0.0
    assert plan.force_profile["name"] == "MINIMAL_ROTATION_REGULATED"


def test_occluded_object_is_do_not_attempt():
    plan = plan_task_grasp(_state("STANDING", available=False, occluded=True, reasons=["clear_space_blocked"]))
    assert plan.strategy == "DO_NOT_ATTEMPT"
    assert "clear_space_blocked" in plan.stop_conditions


def test_tilted_object_is_blocked_until_supported():
    plan = plan_task_grasp(_state("TILTED", tilt=45.0))
    assert plan.strategy == "DO_NOT_ATTEMPT"
    assert "unsupported_orientation_tilted" in plan.stop_conditions


def test_epsilon_is_not_applicable_without_contact_geometry():
    plan = plan_task_grasp(_state("STANDING"), epsilon_result={"epsilon_score": 99.0, "force_closure": True})
    assert plan.epsilon_evidence["status"] == "NOT_APPLICABLE"


def test_live_quality_not_applicable_overrides_planned_candidate_epsilon():
    candidate = {
        "candidate_id": "planned-only",
        "left_contact_px": [10, 20],
        "right_contact_px": [30, 20],
        "contacts_on_body": True,
        "epsilon_value": 0.2,
        "epsilon_score": 80.0,
        "force_closure": True,
    }
    plan = plan_task_grasp(
        _state("STANDING"),
        pregrasp_plan={"selected_candidate": candidate},
        epsilon_result={
            "quality_applicability": "NOT_TESTED_MISSING_VALID_CONTACT_GEOMETRY",
            "quality_not_applicable_reason": "valid_current_contact_geometry_unavailable",
        },
    )
    assert plan.epsilon_evidence["status"] == "NOT_APPLICABLE"
    assert plan.epsilon_evidence["reason"] == "valid_current_contact_geometry_unavailable"
    assert plan.epsilon_evidence["planning_candidate_available"] is True

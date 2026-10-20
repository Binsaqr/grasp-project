"""Camera-free regressions for authoritative grasp-contact provenance."""

import copy

import numpy as np

from epsilon_quality import compute_epsilon_quality
from grasp_project.realsense.realtime_runner import (
    FRICTION_COEFFICIENT,
    attach_visual_contact_model,
)


def _pose_and_depth():
    mask = np.zeros((120, 100), dtype=np.uint8)
    mask[10:111, 20:81] = 1
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "detected": True,
        "cleaned_mask": mask,
        "body_mask": mask,
        # Deliberately below the local grasp-pair midpoint. Visual diagnostics
        # therefore produce non-antipodal global-centroid normals.
        "center_px": [50.0, 80.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 500.0,
    }
    return pose, depth


def _canonical_plan(include_contacts=True):
    contacts = [
        {"point": [30.0, 50.0], "inward_normal": [1.0, 0.0]},
        {"point": [70.0, 50.0], "inward_normal": [-1.0, 0.0]},
    ]
    plan = {
        "left_contact_px": [30.0, 50.0],
        "right_contact_px": [70.0, 50.0],
        "grasp_center_px": [50.0, 50.0],
        "grasp_direction": [1.0, 0.0],
        "grip_width_px": 40.0,
        "grip_width_mm": 40.0,
        "grip_width_in_range": True,
        "body_fraction": 0.5,
        "selected_contact_pair": {"candidate_id": "synthetic_off_center"},
    }
    if include_contacts:
        plan["epsilon_contacts"] = contacts
    return plan


def _epsilon(contacts):
    return compute_epsilon_quality(
        contacts,
        object_centroid=[50.0, 80.0],
        object_radius=20.0,
        mu=FRICTION_COEFFICIENT,
        samples_per_contact=8,
    )


def test_visual_contact_attachment_preserves_authoritative_planner_contacts():
    pose, depth = _pose_and_depth()
    plan = _canonical_plan()
    authoritative = copy.deepcopy(plan["epsilon_contacts"])

    visual = attach_visual_contact_model(
        plan,
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0},
        (*depth.shape, 3),
    )

    assert plan["epsilon_contacts"] == authoritative
    assert plan["visual_epsilon_contacts"] == visual["epsilon_contacts"]
    assert plan["visual_epsilon_contacts"] != authoritative
    assert plan["epsilon_contacts_source"] == "planner_or_locked_projection_authoritative"
    assert plan["visual_epsilon_contacts_role"] == "DIAGNOSTIC_NOT_DECISION_INPUT"


def test_final_epsilon_matches_selected_candidate_not_visual_diagnostic_normals():
    pose, depth = _pose_and_depth()
    plan = _canonical_plan()
    selected_result = _epsilon(plan["epsilon_contacts"])
    assert selected_result["force_closure"] is True

    visual = attach_visual_contact_model(
        plan,
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0},
        (*depth.shape, 3),
    )
    visual_result = _epsilon(visual["epsilon_contacts"])
    final_result = _epsilon(plan["epsilon_contacts"])

    assert visual_result["force_closure"] is False
    assert final_result["force_closure"] is True
    assert final_result["epsilon"] == selected_result["epsilon"]


def test_visual_contacts_are_only_a_fallback_when_plan_has_no_contacts():
    pose, depth = _pose_and_depth()
    plan = _canonical_plan(include_contacts=False)

    visual = attach_visual_contact_model(
        plan,
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0},
        (*depth.shape, 3),
    )

    assert plan["epsilon_contacts"] == visual["epsilon_contacts"]
    assert plan["epsilon_contacts_source"] == "visual_contact_model_fallback_no_plan_contacts"
    assert plan["visual_epsilon_contacts_role"] == "FALLBACK_DECISION_INPUT_NO_PLANNER_CONTACTS"


def test_false_authoritative_epsilon_still_blocks_visual_contact_readiness():
    pose, depth = _pose_and_depth()
    plan = _canonical_plan()
    false_result = _epsilon(
        [
            {"point": [30.0, 50.0], "inward_normal": [0.0, 1.0]},
            {"point": [70.0, 50.0], "inward_normal": [0.0, 1.0]},
        ]
    )
    assert false_result["force_closure"] is False

    attach_visual_contact_model(
        plan,
        pose,
        depth,
        {"fx": 500.0, "fy": 500.0},
        (*depth.shape, 3),
        epsilon_result=false_result,
    )

    assert plan["contact_model_ready"] is False
    assert "epsilon_not_ready" in plan["contact_model_ready_reasons"]


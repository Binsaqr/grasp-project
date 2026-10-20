"""Camera-free regressions for bottle-centered grasp-band selection."""

import numpy as np

import pregrasp_planner
from grasp_project.planning.task_grasp_policy import evaluate_grasp_band_clearance


INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0}


def _planner_scene():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[30:220, 120:200] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = 500
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [159.5, 124.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 189.0,
    }
    return pose, depth


def _prefer_off_center_score(monkeypatch):
    original = pregrasp_planner.evaluate_candidate

    def evaluate(*args, **kwargs):
        result, error = original(*args, **kwargs)
        if result is not None:
            fraction = float(kwargs["fraction"] if "fraction" in kwargs else args[6])
            result["candidate"]["candidate_score"] = 99.0 if abs(fraction - 0.60) < 1e-6 else 40.0
        return result, error

    monkeypatch.setattr(pregrasp_planner, "evaluate_candidate", evaluate)


def test_planner_prefers_profile_band_nearest_bottle_center_over_raw_score(monkeypatch):
    pose, depth = _planner_scene()
    _prefer_off_center_score(monkeypatch)

    plan, error = pregrasp_planner.plan_pregrasp(pose, depth, INTRINSICS)

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.52
    assert plan["selected_grasp_preference_tier"] == "PREFERRED"
    assert plan["candidate_ranking_table"][0]["requested_body_fraction"] == 0.52


def test_planner_uses_closest_safe_band_when_center_band_is_not_allowed(monkeypatch):
    pose, depth = _planner_scene()
    _prefer_off_center_score(monkeypatch)

    plan, error = pregrasp_planner.plan_pregrasp(
        pose,
        depth,
        INTRINSICS,
        allowed_body_fractions=[0.45, 0.60, 0.32],
    )

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.45
    assert plan["grasp_body_fraction_constraint"]["candidate_safety_gates_bypassed"] is False


def test_hand_reach_anchor_selects_nearest_safe_lower_band(monkeypatch):
    pose, depth = _planner_scene()
    _prefer_off_center_score(monkeypatch)

    plan, error = pregrasp_planner.plan_pregrasp(
        pose,
        depth,
        INTRINSICS,
        preferred_contact_point_px=[160.0, 235.0],
    )

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.74
    assert plan["grasp_band_selection_mode"] == "HAND_REACH_ALIGNED_BOTTLE_LOCAL_BAND"
    assert plan["hand_reach_alignment_applied"] is True
    assert plan["candidate_ranking_table"][0]["requested_body_fraction"] == 0.74
    assert plan["selected_hand_reach_distance_px"] < next(
        row["hand_reach_distance_px"]
        for row in plan["candidate_ranking_table"]
        if row["requested_body_fraction"] == 0.52
    )


def test_hand_reach_anchor_never_bypasses_clearance_gate():
    pose, depth = _planner_scene()

    plan, error = pregrasp_planner.plan_pregrasp(
        pose,
        depth,
        INTRINSICS,
        allowed_body_fractions=[0.45, 0.52, 0.60],
        preferred_contact_point_px=[160.0, 235.0],
    )

    assert error is None
    assert plan["selected_requested_body_fraction"] == 0.60
    assert plan["selected_contact_pair"]["clearance_allowed"] is True
    assert plan["grasp_body_fraction_constraint"]["candidate_safety_gates_bypassed"] is False


def test_clearance_prefers_clear_band_nearest_bottle_center():
    pose = {
        "bbox_xywh": [100, 40, 42, 126],
        "depth_median_mm": 500.0,
        "preferred_grasp_body_fractions": [0.45, 0.52, 0.60],
        "fallback_grasp_body_fractions": [0.32, 0.74],
    }

    clearance = evaluate_grasp_band_clearance(
        0,
        [{"target_id": "bottle_01", "pose": pose}],
        image_shape=(240, 360, 3),
        intrinsics=INTRINSICS,
    )

    assert clearance["grasp_clearance_status"] == "ESTIMATED_CLEAR"
    assert clearance["selected_grasp_body_fraction"] == 0.52
    assert clearance["preferred_grasp_body_fraction"] == 0.52
    assert clearance["grasp_band_fallback_used"] is False


def test_clearance_uses_closest_safe_fallback_when_center_is_blocked():
    target = {
        "bbox_xywh": [100, 40, 42, 126],
        "depth_median_mm": 500.0,
        "preferred_grasp_body_fractions": [0.50],
        "fallback_grasp_body_fractions": [0.42, 0.65, 0.74],
    }
    center_blocker = {
        "bbox_xywh": [142, 100, 42, 6],
        "depth_median_mm": 500.0,
    }

    clearance = evaluate_grasp_band_clearance(
        0,
        [
            {"target_id": "bottle_01", "pose": target},
            {"target_id": "bottle_02", "pose": center_blocker},
        ],
        image_shape=(240, 360, 3),
        intrinsics=INTRINSICS,
        patch_height_mm=8.0,
    )

    assert clearance["grasp_clearance_status"] == "ESTIMATED_CLEAR"
    assert clearance["selected_grasp_body_fraction"] == 0.42
    assert clearance["preferred_grasp_body_fraction"] == 0.50
    assert clearance["grasp_band_fallback_used"] is True
    assert 0.50 not in clearance["clear_grasp_body_fractions"]
    assert 0.42 in clearance["clear_grasp_body_fractions"]

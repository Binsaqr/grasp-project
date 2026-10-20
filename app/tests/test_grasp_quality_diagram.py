"""Camera-free contracts for the live wrench-space dashboard diagram."""

import pytest

from grasp_project.dashboard import web_dashboard


def test_full_hull_artifact_replaces_identity_matched_compact_summary():
    matrix = {
        "quality_compute_id": "qm-1",
        "grasp_hulls": {"status": "updated", "wrench_hull_source": "summary"},
    }
    full = {
        "quality_compute_id": "qm-1",
        "wrench_space_hull": {"wrench_points": [[1, 0, 0]]},
    }
    merged = web_dashboard.merge_full_grasp_hulls(matrix, full)
    assert merged["grasp_hulls"]["status"] == "updated"
    assert merged["grasp_hulls"]["wrench_space_hull"]["wrench_points"] == [[1, 0, 0]]


def test_full_hull_artifact_cannot_cross_quality_compute_identity():
    matrix = {"quality_compute_id": "qm-current", "grasp_hulls": {"status": "updated"}}
    full = {
        "quality_compute_id": "qm-old",
        "wrench_space_hull": {"wrench_points": [[1, 0, 0]]},
    }
    merged = web_dashboard.merge_full_grasp_hulls(matrix, full)
    assert "wrench_space_hull" not in merged["grasp_hulls"]


def analytic_matrix(**overrides):
    matrix = {
        "quality_matrix_status": "updated",
        "quality_compute_id": "qm-1",
        "quality_label": "GOOD",
        "normalized_quality_score": 80.0,
        "epsilon": 0.2,
        "force_closure": True,
        "contact_source": "planned_contact",
        "grasp_hulls": {
            "status": "updated",
            "wrench_space_hull": {
                "source": "epsilon_quality.compute_epsilon_quality",
                "coordinate_frame": "planar_wrench_space_fx_fy_tau",
                "wrench_points": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, -1.0, -1.0],
                ],
                "hull_simplices": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                "weakest_direction": [-0.4, 0.8, -0.2],
            },
        },
    }
    matrix.update(overrides)
    return matrix


def live_quality(**overrides):
    payload = {
        "analytic_quality_score": 80.0,
        "live_quality_score": 60.0,
        "live_quality_label": "WEAK",
        "trend": "falling",
        "primary_risk": "rotation",
        "decision": "ADJUST",
        "davis_motion_used_for_live_quality": True,
        "davis_closeup_used_for_live_quality": True,
        "davis_closeup_confidence": 0.8,
        "deviations": {
            "incipient_slip": 3.0,
            "gross_slip": 4.0,
            "rotation": 9.0,
            "vibration": 2.0,
            "uncertainty": 0.5,
        },
        "formula": "live_score = analytic_score - gated penalties; epsilon is not recomputed",
    }
    payload.update(overrides)
    return payload


def valid_davis(**overrides):
    payload = {
        "current_stage": "POST_CLOSE_STABILITY",
        "object_motion_valid": True,
        "raw_motion_class": "ROTATION",
        "disturbance_type": "OBJECT_MOTION",
        "motion_confidence": 0.8,
        "displacement_px": 3.0,
        "rotation_delta_deg": 4.0,
        "vibration_amplitude_px_rms": 1.5,
        "roll_valid": False,
    }
    payload.update(overrides)
    return payload


def test_diagram_payload_uses_real_analytic_wrench_geometry():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        analytic_matrix(), live_quality(), valid_davis()
    )

    assert payload["status"] == "CURRENT"
    assert payload["coordinate_frame"] == "planar_wrench_space_fx_fy_tau"
    assert payload["analytic"]["wrench_points"][0] == [1.0, 0.0, 0.0]
    assert payload["analytic"]["hull_faces"] == [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]
    assert payload["analytic"]["weak_direction"] == [-0.4, 0.8, -0.2]
    assert payload["geometry_source"] == "epsilon_quality.compute_epsilon_quality"


def test_live_region_scales_from_score_without_recomputing_epsilon():
    matrix = analytic_matrix()
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        matrix, live_quality(), valid_davis()
    )

    assert matrix["epsilon"] == 0.2
    assert payload["analytic"]["epsilon"] == 0.2
    assert payload["live"]["operational_radius_ratio"] == pytest.approx(0.75)
    assert payload["live"]["operational_radius"] == pytest.approx(0.15)
    assert payload["live"]["epsilon_recomputed"] is False
    assert "not analytic epsilon" in payload["live"]["radius_semantics"]


def test_invalid_or_global_davis_motion_has_no_displayed_motion_penalty():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        analytic_matrix(),
        live_quality(
            davis_motion_used_for_live_quality=False,
            davis_closeup_used_for_live_quality=False,
            davis_closeup_exclusion_reason="lighting_change",
        ),
        valid_davis(
            object_motion_valid=False,
            disturbance_type="LIGHTING_CHANGE",
            davis_motion_excluded_reason="lighting_change",
        ),
    )

    assert payload["davis"]["used_for_live_quality"] is False
    assert payload["davis"]["exclusion_reason"] == "lighting_change"
    assert payload["live"]["penalties"]["slip"] == 0.0
    assert payload["live"]["penalties"]["rotation"] == 0.0
    assert payload["live"]["penalties"]["vibration"] == 0.0
    assert payload["davis"]["wrench_direction_available"] is False


def test_pre_close_diagram_reports_armed_instead_of_lighting_disturbance():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        analytic_matrix(),
        live_quality(
            davis_motion_used_for_live_quality=False,
            davis_closeup_used_for_live_quality=False,
        ),
        valid_davis(
            current_stage="PRE_CLOSE",
            object_motion_valid=False,
            disturbance_type="LIGHTING_CHANGE",
            davis_motion_excluded_reason="lighting_change",
        ),
    )

    assert payload["davis"]["current_stage"] == "PRE_CLOSE"
    assert payload["davis"]["used_for_live_quality"] is False
    assert payload["davis"]["exclusion_reason"] == "pre_close_monitoring_only"
    assert payload["live"]["penalties"]["slip"] == 0.0


def test_stale_baseline_keeps_geometry_but_disables_live_region():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        analytic_matrix(
            quality_matrix_status="STALE_NEEDS_RECOMPUTE",
            quality_label_stale=True,
            quality_matrix_message="LOCK BASELINE INVALID - RESET / NEW LOCK",
        ),
        live_quality(),
        valid_davis(),
    )

    assert payload["status"] == "STALE"
    assert payload["analytic"]["available"] is True
    assert payload["analytic"]["stale"] is True
    assert payload["live"]["available"] is False
    assert payload["live"]["operational_radius"] is None
    assert payload["action"] == (
        "Reset and acquire a new lock; the immutable baseline is never recomputed in place."
    )


def test_missing_hull_returns_honest_empty_state_without_fake_geometry():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        {"quality_matrix_status": "idle", "quality_matrix_message": "waiting for contacts"},
        {},
        {},
    )

    assert payload["status"] == "UNAVAILABLE"
    assert payload["analytic"]["available"] is False
    assert payload["analytic"]["wrench_points"] == []
    assert payload["analytic"]["hull_faces"] == []
    assert payload["live"]["available"] is False
    assert "waiting for contacts" in payload["reason"]


def test_post_grasp_stray_matrix_cannot_render_under_frozen_baseline():
    snapshot = {
        "writer_session_id": "writer-frozen",
        "attempt_id": "attempt-frozen",
        "lock_frame_id": 72,
        "quality_compute_id": "quality-frozen",
        "quality_epsilon": 0.126,
        "quality_score": 84.0,
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
    }
    stray = analytic_matrix(
        writer_session_id="writer-stray",
        lock_frame_id=10,
        lock_attempt_id="attempt-stray",
        quality_compute_id="quality-stray",
        normalized_quality_score=1.0,
        epsilon=0.9,
        quality_matrix_status="failed",
        matrix_stale_due_to_motion=True,
        quality_label_stale=True,
        quality_matrix_error="stray matrix failed",
    )
    current_live = live_quality(
        analytic_quality_status="CURRENT",
        analytic_quality_score=84.0,
        analytic_quality_label="GOOD",
        baseline={
            "writer_session_id": "writer-frozen",
            "lock_frame_id": 72,
            "quality_compute_id": "quality-frozen",
        },
    )

    payload = web_dashboard.build_grasp_quality_diagram_payload(
        stray,
        current_live,
        valid_davis(),
        snapshot,
    )

    assert payload["status"] == "BASELINE_CURRENT_GEOMETRY_UNAVAILABLE"
    assert payload["analytic"]["baseline_current"] is True
    assert payload["analytic"]["geometry_matches_frozen_baseline"] is False
    assert payload["analytic"]["quality_compute_id"] == "quality-frozen"
    assert payload["analytic"]["quality_score"] == 84.0
    assert payload["analytic"]["epsilon"] == 0.126
    assert payload["analytic"]["wrench_points"] == []
    assert payload["analytic"]["hull_faces"] == []
    assert payload["analytic"]["weak_direction"] is None
    assert payload["analytic"]["force_closure"] is None
    assert payload["coordinate_frame"] == "unavailable_no_matching_frozen_hull"
    assert payload["geometry_source"] is None
    assert payload["live"]["available"] is False
    assert "another lock's geometry" in payload["action"]

    combined = web_dashboard.summarize_combined_quality(
        stray,
        valid_davis(
            raw_motion_class="STABLE",
            motion_class="STABLE",
            davis_motion_usable_for_grasp=True,
        ),
        post_grasp_active=True,
        live_quality={
            **current_live,
            "davis_motion_used_for_live_quality": True,
            "davis_closeup_motion_class": "STABLE",
        },
        post_grasp_baseline=snapshot,
    )
    assert combined["combined_quality_status"] == "STABLE_AFTER_GRASP"
    assert combined["initial_epsilon_score"] == 84.0
    assert combined["analytic_quality_status"] == "CURRENT"
    assert combined["quality_compute_id"] == "quality-frozen"
    assert combined["baseline_source"] == "IMMUTABLE_POST_GRASP_SNAPSHOT"
    assert "STALE" not in combined["combined_quality_status"]
    assert "FAILED" not in combined["combined_quality_status"]


def test_no_current_lock_suppresses_stale_prior_geometry_and_reset_instruction():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        analytic_matrix(
            contact_quality_gate_status="NO_CURRENT_LOCK",
            quality_bundle_status="STALE",
            quality_label_stale=True,
        ),
        live_quality(),
        valid_davis(),
    )

    assert payload["status"] == "UNAVAILABLE"
    assert payload["analytic"]["available"] is False
    assert payload["analytic"]["stale"] is False
    assert payload["analytic"]["epsilon"] is None
    assert payload["analytic"]["wrench_points"] == []
    assert "automatically after lock" in payload["action"]
    assert "Reset" not in payload["action"]


def test_operator_ui_contains_live_wrench_diagram_and_honest_semantics():
    html = web_dashboard.INDEX_HTML
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        analytic_matrix(), live_quality(), valid_davis()
    )

    assert 'id="handDemoWrenchCanvas"' in html
    assert "Analytic hull" in html
    assert "Analytic ε region" in html
    assert "Live operational region" in html
    assert "Analytic weak direction" in html
    assert "No calibrated event-motion-to-physical-wrench mapping" in payload["davis"]["wrench_direction_reason"]
    assert "not analytic epsilon" in payload["live"]["radius_semantics"]
    assert "Wrench-space quality unavailable" in html
    assert "qualityWrenchViewer" in html

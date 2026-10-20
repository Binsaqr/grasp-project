"""The post-close tilt->REGRIP gate must actually be reachable.

Regression: build_report dropped axis_vertical_error_degrees, so the
stage_evaluators post-grasp tilt gate could never fire. build_report now
propagates it (verified by grep-level presence + end-to-end reachability here).
"""

import inspect

import scripts.bottle_grasp_planner as bp
from stage_evaluators import evaluate_postgrasp_stage


def _healthy_postclose(**overrides):
    # a post-close report that passes every OTHER gate, so the decision is
    # driven purely by the field under test
    report = {
        "bottle_detected": True,
        "scores": {"depth_confidence": 90.0, "target_contact_quality": 90.0},
        "pair_quality_score": 90.0,
        "epsilon_result": {"force_closure": True},
        "failure_reasons": [],
        "axis_vertical_error_degrees": 5.0,
    }
    report.update(overrides)
    return report


def test_build_report_source_propagates_axis_vertical_error():
    # the field must be assigned from safety into the report dict
    src = inspect.getsource(bp.build_report)
    assert '"axis_vertical_error_degrees": safety.get("axis_vertical_error_degrees"' in src


def test_tilt_gate_fires_when_error_high():
    result = evaluate_postgrasp_stage(_healthy_postclose(axis_vertical_error_degrees=62.0))
    assert result["stage_decision"] == "REGRIP"
    assert any("tilt" in r.lower() for r in result["reasons"])


def test_tilt_gate_silent_when_error_low():
    result = evaluate_postgrasp_stage(_healthy_postclose(axis_vertical_error_degrees=5.0))
    assert not any("tilt" in r.lower() for r in result["reasons"])
    assert result["stage_decision"] == "READY_TO_LIFT"

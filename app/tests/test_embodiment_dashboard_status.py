"""Camera-free checks for dashboard embodiment status wiring."""

from grasp_project.dashboard.web_dashboard import build_embodiment_dashboard_status, embodiment_change_locked
from grasp_project.embodiment.gripper_model import embodiment_status


def test_dashboard_embodiment_status_publishes_release_and_contacts():
    status = build_embodiment_dashboard_status(
        embodiment_status(),
        realtime={"target_status": "NO_TARGET"},
        matrix={"detected_contact_count": 2, "contact_source": "planned_contact"},
    )

    assert status["canonical_embodiment_id"] == "parallel_jaw_two_finger"
    assert status["release_status"] == "PRODUCTION_READY"
    assert status["benchmark_enabled"] is True
    assert status["expected_contacts"] == 2
    assert status["detected_contacts"] == 2
    assert status["production_decisions_allowed"] is True
    assert status["invalid_metrics_hidden"] is True


def test_embodiment_selector_locks_during_active_grasp():
    assert embodiment_change_locked({"target_status": "TRACKING", "lock_frame_id": 12}) is True
    assert embodiment_change_locked({"target_status": "NO_TARGET", "state": "SEARCHING"}) is False


def test_non_parallel_dashboard_status_is_display_only():
    base = {"active_embodiment": "vacuum_suction", "canonical_embodiment_id": "vacuum_suction"}
    status = build_embodiment_dashboard_status(base, realtime={}, matrix={})

    assert status["production_decisions_allowed"] is False
    assert status["benchmark_enabled"] is False
    assert status["display_only"] is True
    assert "two_contact_epsilon" in status["hidden_metrics"]
    assert "two_contact_epsilon" not in status["dashboard_metric_ids"]

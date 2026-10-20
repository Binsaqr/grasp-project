"""Camera-free tests for the operational live quality monitor."""

from grasp_project.quality.live_quality_monitor import BaselineGraspQuality
from grasp_project.quality.live_quality_monitor import LiveGraspObservation
from grasp_project.quality.live_quality_monitor import LiveGraspQualityMonitor
from grasp_project.quality.live_quality_monitor import baseline_from_post_grasp_snapshot
from grasp_project.quality.live_quality_monitor import baseline_from_quality_matrix
from grasp_project.quality.live_quality_monitor import observation_from_status
from grasp_project.quality.live_quality_monitor import quality_applicability_from_status


def baseline(score=82.0):
    return BaselineGraspQuality(
        valid=True,
        analytic_score=score,
        analytic_label="GOOD",
        quality_compute_id="qc-1",
        writer_session_id="writer-1",
        lock_frame_id=42,
        expected_contacts=2,
        required_contacts=2,
        locked_depth_mm=300.0,
        locked_grip_width_mm=62.0,
    )


def observation(**overrides):
    values = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "bottle_target_id": "bottle-1",
        "target_status": "TRACKING",
        "ready_to_close": True,
        "detected_contacts": 2,
        "visual_contact_ready": True,
        "current_depth_mm": 300.0,
        "current_grip_width_mm": 62.0,
        "pointcloud_freshness": "CURRENT",
        "pointcloud_contact_valid": True,
        "sync_quality": "good",
        "sync_delta_sec": 0.01,
        "davis_motion_class": "STABLE",
        "event_state": "STREAM_ACTIVE_NO_MOTION",
        "davis_event_freshness": "CURRENT",
        "davis_events_valid_for_live_quality": True,
        "davis_motion_usable_for_grasp": True,
        "davis_object_motion_valid": True,
        "davis_motion_confidence": 0.9,
        "davis_target_lock_provenance_valid": True,
        "davis_target_lock_source_writer_session_id": "writer-1",
        "davis_target_lock_source_lock_frame_id": 42,
        "davis_target_lock_source_attempt_id": "attempt-1",
        "davis_target_lock_source_bottle_target_id": "bottle-1",
        "grasp_stage": "POST_GRASP_MONITORING",
        "post_grasp_active": True,
    }
    values.update(overrides)
    return LiveGraspObservation(**values)


def test_baseline_analytic_quality_is_immutable_when_live_evidence_changes():
    monitor = LiveGraspQualityMonitor()
    base = baseline(score=88.0)
    first = monitor.update(base, observation(post_grasp_active=False, grasp_stage="PRE_CLOSE"))
    second = monitor.update(
        base,
        observation(
            angle_error_deg=16.0,
            depth_delta_mm=30.0,
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert first.analytic_quality_score == 88.0
    assert second.analytic_quality_score == 88.0
    assert second.live_quality_score < first.live_quality_score
    assert second.baseline.quality_compute_id == "qc-1"


def test_post_grasp_snapshot_remains_authoritative_when_matrix_is_missing_or_stray():
    realtime = {
        "post_grasp_active": True,
        "writer_session_id": "writer-frozen",
        "target_lock": {
            "attempt_id": "attempt-frozen",
            "lock_frame_id": 72,
            "locked_target": {
                "attempt_id": "attempt-frozen",
                "lock_frame_id": 72,
            },
        },
        "post_grasp_baseline": {
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
        },
    }

    frozen = baseline_from_post_grasp_snapshot(realtime)

    assert frozen is not None
    assert frozen.valid is True
    assert frozen.writer_session_id == "writer-frozen"
    assert frozen.lock_frame_id == 72
    assert frozen.quality_compute_id == "quality-frozen"
    assert frozen.raw_epsilon == 0.126
    assert frozen.analytic_score == 84.0
    assert frozen.analytic_label == "GOOD"

    for transient_matrix in (
        {},
        {
            "writer_session_id": "writer-stray",
            "lock_frame_id": 10,
            "quality_compute_id": "quality-stray",
            "quality_matrix_status": "updated",
            "raw_epsilon": 0.0,
            "normalized_quality_score": 0.0,
            "quality_label": "WEAK",
        },
    ):
        selected = baseline_from_post_grasp_snapshot(realtime) or baseline_from_quality_matrix(
            transient_matrix,
            realtime=realtime,
        )
        assert selected.identity_key == frozen.identity_key
        assert selected.raw_epsilon == frozen.raw_epsilon
        assert selected.analytic_score == frozen.analytic_score

    stale_matrix = {
        "writer_session_id": "writer-stray",
        "lock_frame_id": 10,
        "quality_compute_id": "quality-stray",
        "quality_matrix_status": "STALE_NEEDS_RECOMPUTE",
        "quality_label_stale": True,
        "matrix_stale_due_to_motion": True,
        "quality_stale_reason": "stray_matrix",
        "raw_epsilon": 0.0,
        "normalized_quality_score": 0.0,
    }
    stale_matrix_observation = observation_from_status(
        {
            **realtime,
            "lock_frame_id": 72,
            "lock_attempt_id": "attempt-frozen",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING_UNCERTAIN",
        },
        matrix=stale_matrix,
        davis={
            "stream": "active",
            "report_heartbeat_current": True,
            "grasp_zone_motion_valid": True,
            "davis_motion_usable_for_grasp": True,
            "motion_estimation_confidence": 0.9,
            "raw_motion_class": "INCIPIENT_SLIP",
            "incipient_slip_score": 80.0,
        },
        sync={
            "davis_event_freshness": "CURRENT",
            "davis_events_valid_for_live_quality": True,
        },
        baseline=frozen,
    )
    assert stale_matrix_observation.notes["immutable_post_grasp_baseline_current"] is True
    assert stale_matrix_observation.notes["analytic_quality_stale"] is False
    assert stale_matrix_observation.notes["matrix_stale_due_to_motion"] is False
    stale_matrix_result = LiveGraspQualityMonitor().update(frozen, stale_matrix_observation)
    assert stale_matrix_result.davis_motion_used_for_live_quality is True
    assert stale_matrix_result.live_quality_score < frozen.analytic_score
    assert stale_matrix_result.to_dict()["analytic_quality_status"] == "CURRENT"

    result = LiveGraspQualityMonitor().update(
        frozen,
        observation(
            writer_session_id="writer-frozen",
            lock_frame_id=72,
            lock_attempt_id="attempt-frozen",
            davis_target_lock_source_writer_session_id="writer-frozen",
            davis_target_lock_source_lock_frame_id=72,
            davis_target_lock_source_attempt_id="attempt-frozen",
            davis_motion_class="INCIPIENT_SLIP",
            incipient_slip_score=80.0,
        ),
    )
    assert result.analytic_quality_score == frozen.analytic_score
    assert result.baseline.quality_compute_id == frozen.quality_compute_id
    result_payload = result.to_dict()
    assert round(result.analytic_quality_score + result_payload["davis_adjustment_points"], 6) == round(
        result.live_quality_score,
        6,
    )


def test_post_grasp_snapshot_rejects_mismatched_current_lineage():
    realtime = {
        "post_grasp_active": True,
        "writer_session_id": "writer-current",
        "target_lock": {"attempt_id": "attempt-current", "lock_frame_id": 73},
        "post_grasp_baseline": {
            "writer_session_id": "writer-old",
            "attempt_id": "attempt-old",
            "lock_frame_id": 72,
            "quality_compute_id": "quality-old",
            "quality_epsilon": 0.126,
            "quality_score": 84.0,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    }

    assert baseline_from_post_grasp_snapshot(realtime) is None


def test_one_point_quality_score_remains_one_point_not_one_hundred():
    realtime = {
        "post_grasp_active": True,
        "writer_session_id": "writer-weak",
        "lock_frame_id": 9,
        "lock_attempt_id": "attempt-weak",
        "post_grasp_baseline": {
            "writer_session_id": "writer-weak",
            "attempt_id": "attempt-weak",
            "lock_frame_id": 9,
            "quality_compute_id": "quality-weak",
            "quality_epsilon": 0.0005,
            "quality_score": 1.0,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    }

    frozen = baseline_from_post_grasp_snapshot(realtime)
    matrix = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "updated",
            "normalized_quality_score": 1.0,
        }
    )

    assert frozen is not None
    assert frozen.analytic_score == 1.0
    assert frozen.analytic_label == "RISK"
    assert matrix.analytic_score == 1.0


def test_post_grasp_fixed_davis_zone_can_adjust_live_quality_without_object_claim():
    result = LiveGraspQualityMonitor().update(
        baseline(score=84.0),
        observation(
            davis_object_motion_valid=False,
            davis_grasp_zone_motion_valid=True,
            davis_motion_usable_for_grasp=True,
            gross_slip_score=70.0,
            davis_motion_class="GROSS_SLIP",
        ),
    )
    assert result.davis_motion_used_for_live_quality is True
    assert result.davis_motion_live_quality_reason == "post_grasp_fixed_grasp_zone_valid"
    assert result.live_quality_score < result.analytic_quality_score


def test_post_grasp_quiet_hold_freezes_rgbd_approach_penalties_at_immutable_baseline():
    result = LiveGraspQualityMonitor().update(
        baseline(score=84.0),
        observation(
            target_status="TRACKING_UNCERTAIN",
            distance_measurement_valid=False,
            band_visibility=0.10,
            band_drift_state="STALE",
            detected_contacts=0,
            visual_contact_ready=False,
            pointcloud_contact_valid=False,
            davis_motion_class="STABLE",
            davis_motion_usable_for_grasp=False,
            davis_object_motion_valid=False,
            davis_grasp_zone_motion_valid=False,
        ),
    )
    payload = result.to_dict()

    assert result.live_quality_score == 84.0
    assert result.deviations.total_penalty == 0.0
    assert result.hard_gates == []
    assert result.decision == "MONITOR"
    assert payload["davis_adjustment_points"] == 0.0


def test_post_grasp_displayed_davis_adjustment_exactly_reconciles_smoothed_live_score():
    monitor = LiveGraspQualityMonitor()
    base = baseline(score=84.0)
    monitor.update(base, observation())
    result = monitor.update(
        base,
        observation(
            davis_motion_class="INCIPIENT_SLIP",
            incipient_slip_score=55.0,
            davis_displacement_px=5.0,
        ),
    )
    payload = result.to_dict()

    assert result.davis_motion_used_for_live_quality is True
    assert payload["davis_adjustment_points"] < 0.0
    assert result.live_quality_score == round(
        result.analytic_quality_score + payload["davis_adjustment_points"],
        3,
    )
    assert payload["davis_raw_penalty_points"] >= payload["live_quality_davis_penalty"]


def test_post_grasp_global_disturbance_has_zero_adjustment_despite_rgbd_occlusion():
    result = LiveGraspQualityMonitor().update(
        baseline(score=84.0),
        observation(
            target_status="LOST",
            band_relock_required=True,
            band_visibility=0.0,
            distance_measurement_valid=False,
            davis_motion_class="LIGHTING_CHANGE",
            davis_disturbance_type="LIGHTING_CHANGE",
            davis_motion_usable_for_grasp=False,
            davis_object_motion_valid=False,
            gross_slip_score=100.0,
        ),
    )
    payload = result.to_dict()

    assert result.live_quality_score == 84.0
    assert "target_lost" not in result.hard_gates
    assert "band_relock_required" not in result.hard_gates
    assert payload["davis_adjustment_points"] == 0.0


def test_post_grasp_quiet_active_stream_does_not_flicker_when_recent_event_window_is_empty():
    result = LiveGraspQualityMonitor().update(
        baseline(score=84.0),
        observation(
            event_state="STREAM_ACTIVE_NO_MOTION",
            davis_report_heartbeat_current=True,
            davis_event_freshness="STALE",
            davis_events_valid_for_live_quality=False,
            davis_motion_usable_for_grasp=False,
            davis_object_motion_valid=False,
        ),
    )

    assert result.live_quality_score == 84.0
    assert result.decision == "MONITOR"
    assert result.davis_motion_used_for_live_quality is False


def test_post_grasp_stopped_davis_stream_fails_closed_without_faking_penalty():
    result = LiveGraspQualityMonitor().update(
        baseline(score=84.0),
        observation(
            event_state="NOT_RUNNING",
            davis_report_heartbeat_current=False,
            davis_event_freshness="UNAVAILABLE",
            davis_events_valid_for_live_quality=False,
            davis_motion_usable_for_grasp=False,
            davis_object_motion_valid=False,
        ),
    )

    assert result.live_quality_score == 84.0
    assert result.decision == "WAIT_FOR_DAVIS"
    assert result.to_dict()["davis_adjustment_points"] == 0.0


def test_post_grasp_stale_heartbeat_fails_closed_even_if_last_report_says_active():
    result = LiveGraspQualityMonitor().update(
        baseline(score=84.0),
        observation(
            event_state="STREAM_ACTIVE_NO_MOTION",
            davis_report_heartbeat_current=False,
            davis_report_heartbeat_age_sec=30.0,
            davis_event_freshness="STALE",
            davis_events_valid_for_live_quality=False,
            davis_motion_usable_for_grasp=False,
            davis_object_motion_valid=False,
        ),
    )

    assert result.live_quality_score == 84.0
    assert result.decision == "WAIT_FOR_DAVIS"
    assert result.to_dict()["davis_adjustment_points"] == 0.0


def test_angle_and_depth_deviations_affect_only_live_quality():
    monitor = LiveGraspQualityMonitor()
    base = baseline(score=80.0)
    result = monitor.update(
        base,
        observation(
            angle_error_deg=18.0,
            depth_delta_mm=35.0,
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert result.analytic_quality_score == 80.0
    assert result.live_quality_score < 80.0
    assert result.deviations.angle > 0
    assert result.deviations.depth > 0


def test_locked_band_drift_and_visibility_lower_live_quality_without_changing_analytic_quality():
    result = LiveGraspQualityMonitor().update(
        baseline(score=86.0),
        observation(
            band_drift_px=18.0,
            band_visibility=0.20,
            band_drift_state="LOW_CONFIDENCE",
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert result.analytic_quality_score == 86.0
    assert result.live_quality_score < 86.0
    assert result.deviations.band_drift > 0.0
    assert result.deviations.band_visibility > 0.0


def test_invalid_low_confidence_distance_lowers_live_quality_and_holds_guidance():
    result = LiveGraspQualityMonitor().update(
        baseline(score=86.0),
        observation(
            distance_confidence=0.20,
            depth_valid_ratio=0.25,
            distance_measurement_valid=False,
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert result.analytic_quality_score == 86.0
    assert result.live_quality_score < 86.0
    assert result.deviations.depth > 0.0
    assert result.decision == "HOLD_AND_RECHECK"
    depth = next(item for item in result.sensor_contributions if item.sensor == "depth")
    assert depth.status.startswith("UNRELIABLE")


def test_relock_required_is_a_fail_closed_live_quality_gate():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        observation(
            band_relock_required=True,
            band_drift_state="RELOCK_REQUIRED",
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert "band_relock_required" in result.hard_gates
    assert result.decision == "HOLD_AND_RELOCK"
    assert result.live_quality_score <= 15.0


def test_observation_reads_compact_locked_band_fields():
    obs = observation_from_status(
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING",
            "grasp_band_drift_px": 7.5,
            "grasp_band_visibility": 0.82,
            "band_drift_state": "OK",
            "grasp_band_relock_required": False,
            "distance_confidence": 0.72,
            "depth_valid_ratio": 0.65,
            "distance_measurement_valid": True,
        },
        baseline=baseline(),
    )

    assert obs.band_drift_px == 7.5
    assert obs.band_visibility == 0.82
    assert obs.band_drift_state == "OK"
    assert obs.band_relock_required is False
    assert obs.distance_confidence == 0.72
    assert obs.depth_valid_ratio == 0.65
    assert obs.distance_measurement_valid is True


def test_planned_contact_baseline_does_not_require_finger_detection_for_davis_monitoring():
    base = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "updated",
            "quality_label": "GOOD",
            "normalized_quality_score": 84.0,
            "quality_compute_id": "planned-q",
            "lock_frame_id": 12,
            "contact_source": "planned_contact",
        },
        realtime={"writer_session_id": "writer-1"},
    )
    obs = observation_from_status(
        {
            "target_status": "TRACKING",
            "post_grasp_active": True,
            "writer_session_id": "writer-1",
            "lock_frame_id": 12,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "detected_contact_count": 0,
            "visual_contact_ready": False,
        },
        matrix={"contact_source": "planned_contact", "detected_contact_count": 0},
        davis={
            "stream": "active",
            "raw_motion_class": "INCIPIENT_SLIP",
            "incipient_slip_score": 45.0,
            "motion_estimation_confidence": 0.9,
            "object_motion_valid": True,
            "davis_motion_usable_for_grasp": True,
            "davis_target_lock_provenance_valid": True,
            "davis_target_lock_source_writer_session_id": "writer-1",
            "davis_target_lock_source_lock_frame_id": 12,
            "davis_target_lock_source_attempt_id": "attempt-1",
            "davis_target_lock_source_bottle_target_id": "bottle-1",
        },
        sync={
            "davis_event_freshness": "CURRENT",
            "davis_realsense_valid_for_live_quality": True,
            "davis_realsense_sync_quality": "good",
            "davis_realsense_offset_ms": 10.0,
        },
        baseline=base,
    )
    result = LiveGraspQualityMonitor().update(base, obs)

    assert obs.detected_contacts is None
    assert obs.visual_contact_ready is None
    assert obs.notes["contact_evidence_mode"] == "planned_contact_estimate_no_finger_detection_required"
    assert "missing_required_contacts" not in result.hard_gates
    assert result.davis_motion_used_for_live_quality is True
    assert result.deviations.incipient_slip > 0.0
    assert result.live_quality_score < result.analytic_quality_score


def test_detected_gripper_contact_source_still_enforces_missing_contacts():
    base = baseline()
    obs = observation_from_status(
        {"target_status": "TRACKING", "detected_contact_count": 0},
        matrix={"contact_source": "detected_gripper_contact", "detected_contact_count": 0},
        baseline=base,
    )
    result = LiveGraspQualityMonitor().update(base, obs)

    assert obs.detected_contacts == 0
    assert "missing_required_contacts" in result.hard_gates


def test_live_quality_output_has_requested_analytic_and_live_aliases():
    result = LiveGraspQualityMonitor().update(baseline(), observation())
    payload = result.to_dict()

    assert payload["live_quality_trend"] == result.trend
    assert payload["live_quality_primary_risk"] == result.primary_risk
    assert payload["live_quality_action"] == result.decision
    assert payload["live_quality_explanation"] == result.next_action
    assert payload["analytic_quality_status"] == "CURRENT"
    assert payload["analytic_quality_last_label"] == "GOOD"


def test_incipient_slip_warns_without_claiming_epsilon_recompute():
    monitor = LiveGraspQualityMonitor()
    result = monitor.update(baseline(), observation(incipient_slip_score=75.0, davis_motion_class="INCIPIENT_SLIP"))

    assert result.deviations.incipient_slip > 0
    assert result.decision == "HOLD_AND_RECHECK"
    assert "epsilon is not recomputed" in result.formula


def test_pre_close_davis_motion_is_diagnostic_only_with_zero_penalty():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        observation(
            grasp_stage="PRE_CLOSE",
            post_grasp_active=False,
            davis_motion_class="GROSS_SLIP",
            gross_slip_score=100.0,
            rotation_score=100.0,
            vibration_score=100.0,
        ),
    )

    payload = result.to_dict()
    assert result.davis_motion_used_for_live_quality is False
    assert result.davis_closeup_exclusion_reason == "pre_close_monitoring_only"
    assert result.deviations.incipient_slip == 0.0
    assert result.deviations.gross_slip == 0.0
    assert result.deviations.rotation == 0.0
    assert result.deviations.vibration == 0.0
    assert payload["davis_adjustment_points"] == 0.0


def test_pre_close_current_davis_motion_has_same_score_as_rgbd_only():
    base = baseline(score=84.0)
    rgbd_only = LiveGraspQualityMonitor().update(
        base,
        observation(
            grasp_stage="PRE_CLOSE",
            post_grasp_active=False,
            davis_events_valid_for_live_quality=False,
            davis_event_freshness="UNAVAILABLE",
            davis_motion_usable_for_grasp=False,
            davis_object_motion_valid=False,
        ),
    )
    diagnostic_motion = LiveGraspQualityMonitor().update(
        base,
        observation(
            grasp_stage="PRE_CLOSE",
            post_grasp_active=False,
            davis_motion_class="GROSS_SLIP",
            gross_slip_score=100.0,
            rotation_score=100.0,
            vibration_score=100.0,
        ),
    )

    assert diagnostic_motion.davis_closeup_exclusion_reason == "pre_close_monitoring_only"
    assert diagnostic_motion.deviations.uncertainty == rgbd_only.deviations.uncertainty
    assert diagnostic_motion.live_quality_score == rgbd_only.live_quality_score


def test_native_davis_motion_does_not_require_cross_camera_lock_provenance():
    base = baseline(score=82.0)
    realtime = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "locked_bottle_target_id": "bottle-1",
        "target_status": "TRACKING",
        "post_grasp_active": True,
    }
    common_davis = {
        "stream": "active",
        "raw_motion_class": "INCIPIENT_SLIP",
        "incipient_slip_score": 100.0,
        "motion_estimation_confidence": 0.9,
        "object_motion_valid": True,
        "davis_motion_usable_for_grasp": True,
        "davis_target_lock_source_writer_session_id": "writer-1",
        "davis_target_lock_source_lock_frame_id": 42,
        "davis_target_lock_source_attempt_id": "attempt-1",
        "davis_target_lock_source_bottle_target_id": "bottle-1",
    }
    sync = {
        "davis_event_freshness": "CURRENT",
        "davis_realsense_valid_for_live_quality": True,
        "davis_realsense_sync_quality": "good",
        "davis_realsense_offset_ms": 0.0,
        "pointcloud_valid_for_live_quality": True,
    }
    pointcloud = {
        "freshness": "CURRENT",
        "contact_3d_valid": True,
    }

    cases = (
        {**common_davis, "davis_target_lock_provenance_valid": False},
        {
            **common_davis,
            "davis_target_lock_provenance_valid": True,
            "davis_target_lock_source_attempt_id": "different-attempt",
        },
    )
    for davis in cases:
        obs = observation_from_status(
            realtime,
            davis=davis,
            pointcloud=pointcloud,
            sync=sync,
            baseline=base,
        )
        result = LiveGraspQualityMonitor().update(base, obs)

        assert result.davis_motion_used_for_live_quality is True
        assert result.deviations.incipient_slip > 0.0
        assert result.deviations.uncertainty == 0.0
        assert result.live_quality_score < base.analytic_score
        assert result.to_dict()["davis_adjustment_points"] < 0.0
        assert result.davis_closeup_exclusion_reason is None


def test_matching_davis_lock_provenance_allows_live_quality_penalty():
    base = baseline(score=82.0)
    obs = observation_from_status(
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING",
            "post_grasp_active": True,
        },
        davis={
            "stream": "active",
            "raw_motion_class": "INCIPIENT_SLIP",
            "incipient_slip_score": 80.0,
            "motion_estimation_confidence": 0.9,
            "object_motion_valid": True,
            "davis_motion_usable_for_grasp": True,
            "davis_target_lock_provenance_valid": True,
            "davis_target_lock_source_writer_session_id": "writer-1",
            "davis_target_lock_source_lock_frame_id": 42,
            "davis_target_lock_source_attempt_id": "attempt-1",
            "davis_target_lock_source_bottle_target_id": "bottle-1",
        },
        pointcloud={"freshness": "CURRENT", "contact_3d_valid": True},
        sync={
            "davis_event_freshness": "CURRENT",
            "davis_realsense_valid_for_live_quality": True,
            "davis_realsense_sync_quality": "good",
            "davis_realsense_offset_ms": 0.0,
            "pointcloud_valid_for_live_quality": True,
        },
        baseline=base,
    )
    result = LiveGraspQualityMonitor().update(base, obs)

    assert obs.notes["davis_target_lock_identity_valid"] is True
    assert result.davis_motion_used_for_live_quality is True
    assert result.deviations.incipient_slip > 0.0
    assert result.live_quality_score < base.analytic_score


def test_manual_release_status_returns_live_quality_to_pre_close():
    obs = observation_from_status(
        {"target_status": "TRACKING", "post_grasp_active": False},
        davis={
            "stream": "active",
            "object_motion_valid": True,
            "davis_motion_usable_for_grasp": True,
            "motion_estimation_confidence": 0.9,
        },
        sync={
            "davis_event_freshness": "CURRENT",
            "davis_events_valid_for_live_quality": True,
            "davis_realsense_sync_quality": "good",
        },
        baseline=baseline(),
    )
    result = LiveGraspQualityMonitor().update(baseline(), obs)

    assert obs.grasp_stage == "PRE_CLOSE"
    assert obs.post_grasp_active is False
    assert result.davis_motion_used_for_live_quality is False
    assert result.davis_closeup_exclusion_reason == "pre_close_monitoring_only"


def test_davis_confidence_gate_uses_monitor_configuration():
    monitor = LiveGraspQualityMonitor(
        {"thresholds": {"davis_motion_confidence": 0.95}}
    )
    result = monitor.update(
        baseline(),
        observation(
            davis_motion_confidence=0.90,
            davis_motion_class="INCIPIENT_SLIP",
            incipient_slip_score=100.0,
        ),
    )

    assert result.davis_motion_used_for_live_quality is False
    assert result.davis_closeup_exclusion_reason == "motion_confidence_below_threshold"
    assert result.deviations.incipient_slip == 0.0


def test_measured_davis_slip_pixels_reduce_live_quality_only():
    monitor = LiveGraspQualityMonitor()
    base = baseline(score=86.0)
    result = monitor.update(
        base,
        observation(
            davis_motion_class="INCIPIENT_SLIP",
            davis_displacement_px=5.0,
            davis_cumulative_slip_px=7.0,
            davis_downward_slip_px=5.0,
            davis_motion_confidence=0.8,
            davis_disturbance_type="OBJECT_MOTION",
        ),
    )

    assert result.analytic_quality_score == 86.0
    assert result.live_quality_score < 86.0
    assert result.deviations.incipient_slip > 0
    assert "epsilon is not recomputed" in result.formula
    payload = result.to_dict()
    assert payload["davis_adjustment_status"] == "INCLUDED"
    assert payload["davis_adjustment_points"] < 0.0
    assert payload["live_quality_davis_penalty"] == -payload["davis_adjustment_points"]


def test_excluded_davis_disturbance_reports_zero_matrix_adjustment():
    result = LiveGraspQualityMonitor().update(
        baseline(score=86.0),
        observation(
            davis_motion_class="LIGHTING_CHANGE",
            davis_disturbance_type="LIGHTING_CHANGE",
            davis_motion_usable_for_grasp=False,
            davis_motion_excluded_reason="lighting_change",
            gross_slip_score=100.0,
            rotation_score=100.0,
            vibration_score=100.0,
        ),
    )

    payload = result.to_dict()
    assert payload["davis_adjustment_status"] == "EXCLUDED"
    assert payload["davis_adjustment_points"] == 0.0
    assert payload["live_quality_davis_penalty"] == 0.0
    assert "lighting_change" in payload["davis_adjustment_reason"]


def test_valid_davis_motion_cannot_claim_matrix_adjustment_without_baseline():
    invalid_baseline = BaselineGraspQuality(
        valid=False,
        analytic_score=None,
        analytic_label="INVALID",
        failure_reason="no_active_lock_for_quality",
    )

    result = LiveGraspQualityMonitor().update(
        invalid_baseline,
        observation(davis_motion_class="GROSS_SLIP", gross_slip_score=100.0),
    )
    payload = result.to_dict()

    assert result.davis_motion_used_for_live_quality is False
    assert result.davis_closeup_exclusion_reason == "analytic_baseline_unavailable"
    assert payload["davis_adjustment_status"] == "EXCLUDED"
    assert payload["davis_adjustment_points"] == 0.0


def test_boundary_precursor_reduces_live_quality_and_is_traceable():
    davis = {
        "stream": "active",
        "motion_class": "MICRO SLIP",
        "davis_motion_usable_for_grasp": True,
        "object_motion_valid": True,
        "motion_estimation_confidence": 0.9,
        "davis_target_lock_provenance_valid": True,
        "davis_target_lock_source_writer_session_id": "writer-1",
        "davis_target_lock_source_lock_frame_id": 42,
        "davis_target_lock_source_attempt_id": "attempt-1",
        "davis_target_lock_source_bottle_target_id": "bottle-1",
        "boundary_event_field": {
            "object_motion_valid": True,
            "measurement_status": "ESTIMATED_OBJECT_LOCAL_UNCALIBRATED",
            "alignment_status": "APPROXIMATE_UNCALIBRATED",
            "event_count": 36,
            "active_band_count": 3,
            "incipient_slip_precursor_score": 35.0,
            "slip_cue": 5.0,
            "rotation_cue": 0.0,
            "vibration_cue": 0.0,
        },
    }
    sync = {"davis_event_freshness": "CURRENT", "davis_events_valid_for_live_quality": True, "sync_quality": "good"}
    obs = observation_from_status(
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING",
            "post_grasp_active": True,
            "ready_to_close": True,
            "visual_contact_ready": True,
        },
        davis=davis,
        sync=sync,
        baseline=baseline(),
    )
    result = LiveGraspQualityMonitor().update(baseline(), obs)

    assert obs.boundary_incipient_slip_score == 35.0
    assert obs.notes["boundary_evidence_used"] is True
    assert result.deviations.incipient_slip > 0.0
    assert result.live_quality_score < result.analytic_quality_score
    event_contribution = next(item for item in result.sensor_contributions if item.sensor == "DAVIS events")
    assert "precursor 35.0" in event_contribution.details


def test_raw_davis_class_and_pairwise_offset_take_precedence():
    obs = observation_from_status(
        {"target_status": "TRACKING"},
        davis={
            "stream": "active",
            "motion_class": "MICRO SLIP",
            "raw_motion_class": "INCIPIENT_SLIP",
            "davis_motion_usable_for_grasp": True,
        },
        sync={
            "davis_event_freshness": "CURRENT",
            "davis_events_valid_for_live_quality": True,
            "davis_realsense_offset_ms": 20.0,
            "live_quality_max_sensor_offset_ms": 1000.0,
            "max_sensor_time_delta_sec": 1.0,
        },
        baseline=baseline(),
    )

    assert obs.davis_motion_class == "INCIPIENT_SLIP"
    assert obs.sync_delta_sec == 0.02


def test_pair_quality_is_diagnostic_for_asynchronous_post_grasp_monitoring():
    obs = observation_from_status(
        {"target_status": "TRACKING"},
        davis={
            "stream": "active",
            "motion_class": "STABLE",
            "davis_motion_usable_for_grasp": True,
        },
        sync={
            "davis_realsense_sync_quality": "good",
            "davis_realsense_valid_for_live_quality": True,
            "davis_event_freshness": "CURRENT",
            "live_quality_max_sensor_offset_ms": 800.0,
        },
        baseline=baseline(),
    )

    result = LiveGraspQualityMonitor().update(baseline(), obs)

    assert obs.sync_delta_sec == 0.8
    assert "unacceptable_synchronization" not in result.hard_gates
    sync_contribution = next(item for item in result.sensor_contributions if item.sensor == "synchronization")
    assert sync_contribution.available is False
    assert sync_contribution.penalty == 0.0


def test_stale_optional_pointcloud_cannot_veto_current_realsense_davis_pair():
    obs = observation_from_status(
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING",
            "post_grasp_active": True,
            "visual_contact_ready": True,
        },
        davis={
            "stream": "active",
            "motion_class": "STABLE",
            "davis_motion_usable_for_grasp": True,
            "object_motion_valid": True,
            "motion_estimation_confidence": 0.9,
            "davis_target_lock_provenance_valid": True,
            "davis_target_lock_source_writer_session_id": "writer-1",
            "davis_target_lock_source_lock_frame_id": 42,
            "davis_target_lock_source_attempt_id": "attempt-1",
            "davis_target_lock_source_bottle_target_id": "bottle-1",
        },
        pointcloud={"freshness": "STALE"},
        sync={
            "sync_quality": "offset_too_large",
            "max_sensor_time_offset_ms": 1000.0,
            "live_quality_max_sensor_offset_ms": 1000.0,
            "davis_realsense_offset_ms": 20.0,
            "davis_realsense_sync_quality": "good",
            "davis_realsense_valid_for_live_quality": True,
            "davis_event_freshness": "CURRENT",
            "pointcloud_freshness": "STALE",
            "pointcloud_valid_for_live_quality": False,
        },
        baseline=baseline(),
    )

    result = LiveGraspQualityMonitor().update(baseline(), obs)

    assert obs.sync_quality == "offset_too_large"
    assert obs.davis_realsense_sync_quality == "good"
    assert obs.sync_delta_sec == 0.02
    assert "unacceptable_synchronization" not in result.hard_gates
    assert result.davis_motion_used_for_live_quality is True
    sync_contribution = next(item for item in result.sensor_contributions if item.sensor == "synchronization")
    assert sync_contribution.status == "good"
    assert "global bundle quality offset_too_large" in sync_contribution.details


def test_suppressed_boundary_evidence_cannot_reduce_live_quality():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        observation(
            davis_motion_usable_for_grasp=False,
            davis_motion_excluded_reason="activity_not_local_to_roi",
            boundary_incipient_slip_score=100.0,
            boundary_slip_score=100.0,
            boundary_rotation_score=100.0,
            boundary_vibration_score=100.0,
        ),
    )

    assert result.deviations.incipient_slip == 0.0
    assert result.deviations.gross_slip == 0.0
    assert result.deviations.rotation == 0.0
    assert result.deviations.vibration == 0.0


def test_gross_slip_hard_gate_blocks_immediately():
    monitor = LiveGraspQualityMonitor()
    monitor.update(baseline(), observation())
    result = monitor.update(baseline(), observation(gross_slip_score=90.0, davis_motion_class="GROSS_SLIP"))

    assert "gross_slip" in result.hard_gates
    assert result.decision == "REGRIP"
    assert result.live_quality_score <= 12.0
    assert result.smoothing_bypassed_reason == "hard_gate"


def test_lighting_or_camera_shake_does_not_automatically_become_slip():
    monitor = LiveGraspQualityMonitor()
    result = monitor.update(
        baseline(),
        observation(davis_motion_class="CAMERA_SHAKE", incipient_slip_score=0.0, gross_slip_score=0.0),
    )

    assert "gross_slip" not in result.hard_gates
    assert result.deviations.gross_slip == 0.0
    assert result.decision in {"READY_TO_CLOSE", "MONITOR"}


def test_missing_optional_pointcloud_is_advisory_and_does_not_change_score():
    monitor = LiveGraspQualityMonitor()
    result = monitor.update(
        baseline(),
        observation(pointcloud_freshness="UNAVAILABLE", pointcloud_contact_valid=None),
    )

    assert "clearly_off_surface_3d_contacts" not in result.hard_gates
    assert result.deviations.pointcloud == 0
    assert result.live_quality_score > 50.0


def test_stale_pointcloud_missing_optional_marker_has_zero_quality_penalty():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        observation(
            pointcloud_freshness="STALE",
            pointcloud_contact_valid=None,
            missing_optional_sensors=["point_cloud_stale"],
        ),
    )

    assert result.deviations.pointcloud == 0
    pointcloud = next(item for item in result.sensor_contributions if item.sensor == "point cloud")
    assert pointcloud.available is False
    assert pointcloud.penalty == 0.0


def test_stale_pointcloud_off_surface_is_excluded_from_hard_gate():
    monitor = LiveGraspQualityMonitor()
    result = monitor.update(
        baseline(),
        observation(pointcloud_freshness="STALE", pointcloud_contact_valid=False, pointcloud_off_surface=True),
    )

    assert "clearly_off_surface_3d_contacts" not in result.hard_gates
    assert result.decision in {"READY_TO_CLOSE", "MONITOR"}


def test_stale_davis_events_are_excluded_from_live_quality_penalties():
    realtime = {
        "target_status": "TRACKING",
        "ready_to_close": True,
        "visual_contact_ready": True,
        "bottle_distance_mm": 300,
    }
    davis = {
        "event_state": "STREAM_ACTIVE_EVENTS",
        "motion_class": "GROSS_SLIP",
        "gross_slip_score": 100.0,
        "cumulative_slip_px": 100.0,
        "motion_estimation_confidence": 1.0,
    }
    sync = {
        "sync_quality": "stale_modalities",
        "max_sensor_time_offset_ms": 20.0,
        "davis_event_freshness": "STALE",
        "davis_event_age_ms": 600.0,
        "davis_events_valid_for_live_quality": False,
    }
    obs = observation_from_status(realtime, davis=davis, sync=sync, pointcloud={"freshness": "CURRENT", "contact_3d_valid": True}, baseline=baseline())
    result = LiveGraspQualityMonitor().update(baseline(), obs)

    assert "gross_slip" not in result.hard_gates
    assert result.deviations.gross_slip == 0.0
    assert "davis_events_stale" in result.observation.missing_optional_sensors
    assert result.observation.notes["davis_not_used_reason"] == "DAVIS not used because event data is stale"


def test_post_grasp_object_motion_with_complete_identity_still_requires_current_events():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        observation(
            davis_motion_class="GROSS_SLIP",
            gross_slip_score=100.0,
            davis_cumulative_slip_px=100.0,
            davis_event_freshness="STALE",
            davis_events_valid_for_live_quality=False,
        ),
    )

    assert result.davis_motion_used_for_live_quality is False
    assert result.davis_closeup_exclusion_reason == "stale_events"
    assert result.deviations.gross_slip == 0.0
    assert "gross_slip" not in result.hard_gates


def test_stale_analytic_baseline_is_not_adjusted_by_davis_motion():
    base = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "STALE_NEEDS_RECOMPUTE",
            "quality_label": "GOOD",
            "normalized_quality_score": 82.0,
            "quality_compute_id": "old-quality",
            "contact_source": "planned_contact",
        },
        realtime={"writer_session_id": "writer-1"},
    )
    obs = observation_from_status(
        {"target_status": "TRACKING"},
        matrix={
            "quality_matrix_status": "STALE_NEEDS_RECOMPUTE",
            "quality_label_stale": True,
            "contact_source": "planned_contact",
        },
        davis={
            "stream": "active",
            "raw_motion_class": "INCIPIENT_SLIP",
            "incipient_slip_score": 80.0,
            "davis_motion_usable_for_grasp": True,
        },
        sync={
            "davis_event_freshness": "CURRENT",
            "davis_realsense_valid_for_live_quality": True,
            "davis_realsense_sync_quality": "good",
            "davis_realsense_offset_ms": 5.0,
        },
        baseline=base,
    )
    result = LiveGraspQualityMonitor().update(base, obs)

    assert base.valid is True
    assert obs.notes["analytic_quality_stale"] is True
    assert result.davis_motion_used_for_live_quality is False
    assert result.davis_closeup_exclusion_reason == "analytic_baseline_stale"
    assert result.deviations.incipient_slip == 0.0


def test_unspecified_davis_freshness_and_object_validity_cannot_penalize_quality():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        LiveGraspObservation(
            target_status="TRACKING",
            detected_contacts=2,
            visual_contact_ready=True,
            event_state="STREAM_ACTIVE_EVENTS",
            davis_motion_class="GROSS_SLIP",
            gross_slip_score=100.0,
            davis_cumulative_slip_px=100.0,
        ),
    )

    assert result.deviations.gross_slip == 0.0
    assert "gross_slip" not in result.hard_gates
    assert result.davis_motion_used_for_live_quality is False


def test_target_loss_bypasses_smoothing():
    monitor = LiveGraspQualityMonitor()
    monitor.update(baseline(), observation(post_grasp_active=False, grasp_stage="PRE_CLOSE"))
    result = monitor.update(
        baseline(),
        observation(target_status="LOST", post_grasp_active=False, grasp_stage="PRE_CLOSE"),
    )

    assert "target_lost" in result.hard_gates
    assert result.decision == "DO_NOT_CLOSE"
    assert result.live_quality_score <= 5.0
    assert result.smoothing_bypassed_reason == "hard_gate"


def test_result_dict_dashboard_fields_agree_with_top_level_values():
    result = LiveGraspQualityMonitor().update(baseline(score=73.0), observation())
    payload = result.to_dict()

    assert payload["display"]["analytic_quality"] == "GOOD 73.0"
    assert payload["display"]["live_quality"] == f"{result.live_quality_label} {result.live_quality_score:.1f}"
    assert payload["display"]["decision"] == result.decision


def test_baseline_builder_marks_missing_matrix_invalid():
    base = baseline_from_quality_matrix({}, realtime={})

    assert not base.valid
    assert base.analytic_score is None
    assert base.failure_reason


def test_live_quality_uses_embodiment_contact_count_not_parallel_default():
    base = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "ok",
            "quality_label": "GOOD",
            "normalized_quality_score": 82.0,
        },
        realtime={},
        embodiment={"canonical_embodiment_id": "three_finger_gripper"},
    )
    obs = observation(
        embodiment_id="three_finger_gripper",
        detected_contacts=2,
        expected_contacts=3,
        visual_contact_ready=True,
        post_grasp_active=False,
        grasp_stage="PRE_CLOSE",
    )
    result = LiveGraspQualityMonitor().update(base, obs)

    assert base.embodiment_id == "three_finger_gripper"
    assert base.required_contacts == 3
    assert "missing_required_contacts" in result.hard_gates


def test_vacuum_live_quality_requires_one_contact_when_configured():
    base = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "ok",
            "quality_label": "OK",
            "normalized_quality_score": 70.0,
        },
        realtime={},
        embodiment={"canonical_embodiment_id": "vacuum_suction"},
    )
    result = LiveGraspQualityMonitor().update(
        base,
        observation(
            embodiment_id="vacuum_suction",
            detected_contacts=1,
            expected_contacts=1,
            visual_contact_ready=True,
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert base.required_contacts == 1
    assert "missing_required_contacts" not in result.hard_gates


def test_custom_zero_contact_requirement_does_not_fall_back_to_parallel_default():
    base = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "ok",
            "quality_label": "OK",
            "normalized_quality_score": 70.0,
        },
        realtime={},
        embodiment={"canonical_embodiment_id": "custom"},
    )
    result = LiveGraspQualityMonitor().update(
        base,
        observation(
            embodiment_id="custom",
            detected_contacts=0,
            expected_contacts=0,
            visual_contact_ready=True,
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert base.required_contacts == 0
    assert result.deviations.contact == 0.0
    assert "missing_required_contacts" not in result.hard_gates


def test_parallel_two_contact_requirement_still_blocks_one_contact():
    result = LiveGraspQualityMonitor().update(
        baseline(),
        observation(
            detected_contacts=1,
            expected_contacts=2,
            post_grasp_active=False,
            grasp_stage="PRE_CLOSE",
        ),
    )

    assert result.deviations.contact > 0.0
    assert "missing_required_contacts" in result.hard_gates


def test_stale_matrix_keeps_analytic_baseline_valid():
    base = baseline_from_quality_matrix(
        {
            "quality_matrix_status": "STALE_NEEDS_RECOMPUTE",
            "quality_label": "GOOD",
            "quality_label_stale": True,
            "normalized_quality_score": 78.0,
            "quality_compute_id": "old-lock",
        },
        realtime={},
    )

    assert base.valid
    assert base.analytic_score == 78.0
    assert base.analytic_label == "GOOD"


def test_quality_is_not_applicable_for_top_down_pose_without_contacts():
    result = quality_applicability_from_status(
        {
            "object_state": {
                "pose_class": "TOP_DOWN_BODY",
                "pose_valid_for_grasp": False,
            },
            "multi_bottle_target_selection": {
                "candidate_count": 1,
                "selected_target_id": "bottle_01",
                "candidates": [
                    {
                        "target_id": "bottle_01",
                        "grasp_clearance_status": "ESTIMATED_CLEAR",
                    }
                ],
            },
        },
        {"quality_matrix_status": "idle"},
    )

    assert result["quality_applicability"] == "NOT_APPLICABLE_UNSUPPORTED_OR_UNVERIFIED_POSE"
    assert result["quality_pose_valid"] is False
    assert result["quality_not_applicable_reason"].startswith("pose_not_valid")


def test_clutter_blocked_selection_cannot_show_applicable_quality():
    result = quality_applicability_from_status(
        {
            "object_state": {
                "pose_class": "UPRIGHT_FRONT",
                "pose_valid_for_grasp": True,
            },
            "multi_bottle_target_selection": {
                "candidate_count": 2,
                "selected_target_id": None,
                "candidates": [
                    {
                        "target_id": "bottle_01",
                        "grasp_clearance_status": "BLOCKED_BY_NEIGHBOR_OR_IMAGE_EDGE",
                        "reasons": [
                            "graspable_region_clear_in_current_rgbd_estimate",
                            "no_collision_free_side_grasp_band",
                        ],
                    }
                ],
            },
        },
        {
            "quality_matrix_status": "updated",
            "contact_3d_inputs_valid": True,
        },
    )

    assert result["quality_applicability"] == "BLOCKED_BY_CURRENT_GRASP_FEASIBILITY"
    assert result["quality_clutter_valid"] is False
    assert result["cannot_grasp_reason"] == "no_collision_free_side_grasp_band"


def test_less_optimal_clear_band_is_explicitly_estimated_not_confirmed():
    result = quality_applicability_from_status(
        {
            "object_state": {
                "pose_class": "UPRIGHT_FRONT",
                "pose_valid_for_grasp": True,
            },
            "multi_bottle_target_selection": {
                "candidate_count": 1,
                "selected_target_id": "bottle_01",
                "candidates": [
                    {
                        "target_id": "bottle_01",
                        "grasp_clearance_status": "ESTIMATED_CLEAR",
                        "grasp_band_fallback_used": True,
                    }
                ],
            },
        },
        {
            "quality_matrix_status": "updated",
            "contact_3d_inputs_valid": True,
            "contact_source": "planned_contact",
        },
    )

    assert result["quality_applicability"] == "APPLICABLE_ESTIMATED_ANALYTIC"
    assert result["less_optimal_grasp_reason"]
    assert result["quality_estimated_vs_confirmed"] == "ESTIMATED_ANALYTIC_NOT_PHYSICALLY_CONFIRMED"


def test_monitor_configuration_mutation_cannot_leak_into_other_arms_or_defaults():
    from copy import deepcopy

    from grasp_project.quality.live_quality_monitor import DEFAULT_CONFIG, LiveGraspQualityMonitor

    before = deepcopy(DEFAULT_CONFIG)
    first, second = LiveGraspQualityMonitor(), LiveGraspQualityMonitor()
    first.config["weights"]["incipient_slip"] += 12
    assert DEFAULT_CONFIG == before
    assert second.config == before
    override = {"weights": {"incipient_slip": 5}, "custom_list": ["frozen"]}
    explicit = LiveGraspQualityMonitor(override)
    override["weights"]["incipient_slip"] = 99
    override["custom_list"].append("caller-change")
    assert explicit.config["weights"]["incipient_slip"] == 5
    assert explicit.config["custom_list"] == ["frozen"]

"""Camera-free regressions for the canonical close-up DAVIS monitoring path."""

import json

import numpy as np


def test_davis_runtime_timeline_writer_is_bounded(monkeypatch, tmp_path):
    from grasp_project.davis import motion_gui

    path = tmp_path / "motion_timeline.jsonl"
    monkeypatch.setattr(motion_gui, "DAVIS_TIMELINE_MAX_BYTES", 1024)
    for index in range(100):
        assert motion_gui._append_jsonl(path, {"index": index, "motion_class": "STABLE"})

    backup = tmp_path / "motion_timeline.jsonl.previous"
    assert path.stat().st_size <= 1024
    assert backup.stat().st_size <= 1024


def test_davis_runtime_output_replace_failure_is_nonfatal(monkeypatch, tmp_path):
    from grasp_project.davis import motion_gui

    def blocked_replace(*_args, **_kwargs):
        raise PermissionError("destination remains open")

    monkeypatch.setattr(motion_gui, "replace_path_retry", blocked_replace)
    path = tmp_path / "latest_motion_report.json"
    assert motion_gui._write_json(path, {"state": "CURRENT"}) is False
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_closeup_window_names_use_action_relative_boundaries():
    from grasp_project.davis.motion_gui import closeup_window_for_elapsed

    assert closeup_window_for_elapsed(None) == "not_started"
    assert closeup_window_for_elapsed(0.299) == "impact"
    assert closeup_window_for_elapsed(0.300) == "settle"
    assert closeup_window_for_elapsed(0.699) == "settle"
    assert closeup_window_for_elapsed(0.700) == "stability"
    assert closeup_window_for_elapsed(1.199) == "stability"
    assert closeup_window_for_elapsed(1.200) == "post_window_rolling"


def test_davis_roi_priority_preserves_calibrated_lock_and_aps_before_configured_fallback():
    from grasp_project.dashboard.web_dashboard import use_davis_native_grasp_zone

    roi = [48, 0, 250, 260]
    assert use_davis_native_grasp_zone(
        roi,
        "PRE_CLOSE",
        {"davis_target_context_mode": "live_realsense_mapping"},
    ) is False
    assert use_davis_native_grasp_zone(
        roi,
        "PRE_CLOSE",
        {"davis_target_context_mode": "pre_close_held_target_diagnostic"},
    ) is True
    assert use_davis_native_grasp_zone(
        roi,
        "POST_CLOSE_STABILITY",
        {
            "davis_target_context_mode": "post_grasp_locked_mapping",
            "davis_target_context_valid": True,
            "davis_target_roi_priority_selected": "calibrated_realsense_contact_roi",
            "davis_target_alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
        },
    ) is False
    assert use_davis_native_grasp_zone(
        roi,
        "POST_CLOSE_STABILITY",
        {
            "davis_target_context_mode": "davis_aps_target",
            "davis_target_context_valid": True,
            "davis_target_roi_priority_selected": "davis_aps_target_roi",
        },
    ) is False
    assert use_davis_native_grasp_zone(
        roi,
        "POST_CLOSE_STABILITY",
        {
            "davis_target_context_mode": "post_grasp_locked_mapping",
            "davis_target_context_valid": True,
            "davis_target_roi_priority_selected": "uncalibrated_realsense_contact_motion_diagnostic_roi",
            "davis_target_alignment_status": "APPROXIMATE_UNCALIBRATED",
        },
    ) is True
    assert use_davis_native_grasp_zone(
        None,
        "POST_CLOSE_STABILITY",
        {"davis_target_context_mode": "post_grasp_locked_mapping"},
    ) is False


def test_post_grasp_payload_uses_requested_aliases_and_fail_closed_recommendation():
    from grasp_project.davis.motion_gui import post_grasp_monitoring_payload, post_grasp_recommendation

    payload = post_grasp_monitoring_payload(
        {
            "roi_source": "configured_roi",
            "roi_xywh": [48, 0, 250, 260],
            "roi_object_specific": False,
            "roi_confidence": 0.0,
            "roi_projection_warning": "Configured ROI is diagnostic-only.",
            "event_health_valid": True,
            "object_motion_valid": False,
            "vibration_amplitude_px_rms": 2.5,
            "davis_closeup_used_for_live_quality": False,
            "davis_closeup_exclusion_reason": "no_object_roi",
        },
        action_id="action-1",
        started_at=123.0,
    )

    assert payload["davis_event_band_active"] is True
    assert payload["davis_event_band_source"] == "configured_roi"
    assert payload["davis_event_band_roi_xywh"] == [48, 0, 250, 260]
    assert payload["davis_event_band_object_specific"] is False
    assert payload["davis_event_band_confidence"] == 0.0
    assert payload["event_health"] is True
    assert payload["vibration_rms_px"] == 2.5
    assert payload["roll_valid"] is False
    assert payload["roll_delta_deg"] is None
    assert payload["roll_status"] == "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION"
    assert payload["davis_used_for_live_quality"] is False
    assert payload["davis_exclusion_reason"] == "no_object_roi"
    assert payload["recommendation"] == "NOT_OBJECT_SPECIFIC"
    assert post_grasp_recommendation({
        "roi_object_specific": True,
        "object_motion_valid": False,
        "event_health_valid": True,
    }) == "HOLD"


def test_post_grasp_stage_writes_dedicated_latest_report_and_timeline(monkeypatch):
    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **kwargs):
            self.roi_xywh = kwargs.get("roi_xywh")
            self.configured_roi = kwargs.get("roi_xywh")

        def start(self):
            return {}

    writes = []
    appends = []
    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    monkeypatch.setattr(motion_gui, "_write_json", lambda path, payload: writes.append((path, payload)))
    monkeypatch.setattr(motion_gui, "_append_jsonl", lambda path, payload: appends.append((path, payload)))
    analyzer = motion_gui.DavisMotionAnalyzer()
    analyzer.set_stage("POST_CLOSE_STABILITY")
    analyzer.report = {
        "roi_source": "full_frame",
        "roi_xywh": [0, 0, 346, 260],
        "event_health_valid": True,
        "object_motion_valid": False,
        "davis_closeup_used_for_live_quality": False,
        "davis_closeup_exclusion_reason": "full_frame_roi_only",
    }

    analyzer.save_outputs(include_images=False)

    latest = [payload for path, payload in writes if path == motion_gui.DAVIS_POST_GRASP_LATEST_REPORT_PATH]
    timeline = [payload for path, payload in appends if path == motion_gui.DAVIS_POST_GRASP_TIMELINE_PATH]
    assert len(latest) == 1
    assert len(timeline) == 1
    assert latest[0]["post_grasp_monitoring_action_id"] == analyzer.close_action_id
    assert latest[0]["recommendation"] == "NOT_OBJECT_SPECIFIC"
    assert timeline[0]["davis_exclusion_reason"] == "full_frame_roi_only"
    assert timeline[0]["roll_valid"] is False
    assert timeline[0]["roll_status"] == "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION"


def test_closeup_action_windows_are_non_overlapping():
    from grasp_project.davis.motion_gui import closeup_action_window_metrics

    origin = 10.0
    events = np.asarray(
        [
            [origin + 0.000, 50, 50, 1],
            [origin + 0.299, 51, 50, 0],
            [origin + 0.300, 52, 50, 1],
            [origin + 0.699, 53, 50, 0],
            [origin + 0.700, 54, 50, 1],
            [origin + 1.200, 55, 50, 0],
        ],
        dtype=float,
    )

    metrics = closeup_action_window_metrics(events, origin, [0, 0, 100, 100])

    assert [metrics[name]["event_count"] for name in ("impact", "settle", "stability")] == [2, 2, 2]
    assert sum(metrics[name]["event_count"] for name in metrics) == len(events)


def test_action_accumulator_preserves_counts_before_rolling_caps():
    from grasp_project.davis.motion_gui import (
        _empty_closeup_accumulators,
        accumulate_closeup_action_events,
        closeup_metrics_from_accumulators,
    )

    origin = 20.0
    count = 12000
    events = np.column_stack(
        [
            np.linspace(origin, origin + 0.299, count),
            np.full(count, 50.0),
            np.full(count, 50.0),
            np.arange(count) % 2,
        ]
    )
    accumulators = _empty_closeup_accumulators()
    accumulate_closeup_action_events(accumulators, events[:6000], origin, [0, 0, 100, 100])
    accumulate_closeup_action_events(accumulators, events[6000:], origin, [0, 0, 100, 100])

    metrics = closeup_metrics_from_accumulators(accumulators, [0, 0, 100, 100])
    assert metrics["impact"]["event_count"] == count
    assert metrics["impact"]["event_rate"] == count / 0.3


def test_uncalibrated_projection_is_diagnostic_only():
    from grasp_project.davis.motion_gui import object_motion_validity, roi_projection_diagnostics

    gates = object_motion_validity(
        {
            "motion_hull_area_fraction": 0.05,
            "motion_hull_border_touch_ratio": 0.0,
        },
        "realsense_bbox_mapped",
        200,
        3.0,
        "STABLE",
        0.9,
    )
    projection = roi_projection_diagnostics("realsense_bbox_mapped")

    assert gates["object_motion_valid"] is False
    assert gates["roi_object_specific"] is False
    assert projection["roi_confidence"] == 0.0
    assert "diagnostic-only" in projection["roi_projection_warning"]


def test_calibrated_roi_without_reprojection_evidence_fails_closed():
    from grasp_project.davis.motion_gui import roi_projection_diagnostics

    projection = roi_projection_diagnostics("realsense_contact_projected")

    assert projection["roi_confidence"] == 0.0
    assert "suppressed" in projection["roi_projection_warning"]


def test_raw_neighborhood_pattern_is_named_without_bypassing_object_gate():
    from grasp_project.davis.motion_gui import diagnostic_movement_pattern

    result = diagnostic_movement_pattern({
        "raw_window_event_count": 6000,
        "assigned_event_count": 296,
        "raw_rotation_cue": 26.5,
        "raw_slip_cue": 0.0,
        "raw_vibration_cue": 100.0,
        "raw_incipient_slip_precursor_score": 0.0,
        "object_motion_valid": False,
        "suppression_reason": "lighting_change",
    })

    assert result["diagnostic_movement_pattern"] == "VIBRATION_LIKE_ACTIVITY"
    assert result["diagnostic_movement_pattern_score"] == 100.0
    assert result["diagnostic_movement_pattern_status"] == "DIAGNOSTIC_ONLY_NOT_OBJECT_ATTRIBUTED"
    assert result["diagnostic_movement_pattern_object_attributed"] is False
    assert result["diagnostic_movement_pattern_physically_validated"] is False
    assert "does not bypass" in result["diagnostic_movement_pattern_reason"]


def test_raw_neighborhood_pattern_reports_missing_support_honestly():
    from grasp_project.davis.motion_gui import diagnostic_movement_pattern

    result = diagnostic_movement_pattern({
        "raw_window_event_count": 20,
        "assigned_event_count": 0,
        "raw_rotation_cue": 90.0,
    })

    assert result["diagnostic_movement_pattern"] == "NO_LOCAL_NEIGHBORHOOD_SUPPORT"
    assert result["diagnostic_movement_pattern_status"] == "NOT_MEASURED_NO_ASSIGNED_BOUNDARY_EVENTS"


def test_raw_neighborhood_pattern_distinguishes_drift_rotation_and_local_burst():
    from grasp_project.davis.motion_gui import diagnostic_movement_pattern

    cases = (
        ("raw_slip_cue", "VERTICAL_DRIFT_LIKE_ACTIVITY"),
        ("raw_rotation_cue", "ROTATION_LIKE_ACTIVITY"),
        ("raw_incipient_slip_precursor_score", "LOCALIZED_BURST_LIKE_ACTIVITY"),
    )
    for cue, expected in cases:
        field = {
            "raw_window_event_count": 200,
            "assigned_event_count": 80,
            "object_motion_valid": False,
        }
        field[cue] = 75.0
        result = diagnostic_movement_pattern(field)
        assert result["diagnostic_movement_pattern"] == expected
        assert result["diagnostic_movement_pattern_score"] == 75.0


def test_published_disturbance_matches_the_gated_global_class():
    from grasp_project.davis.motion_gui import canonical_disturbance_type

    assert canonical_disturbance_type("LIGHTING_CHANGE", "object_motion") == "lighting_change"
    assert canonical_disturbance_type("CAMERA_SHAKE", "none") == "camera_shake"
    assert canonical_disturbance_type("BACKGROUND_MOTION", "object_motion") == "background_motion"
    assert canonical_disturbance_type("STABLE", "object_motion") == "none"
    assert canonical_disturbance_type("UNKNOWN_MOTION", "unknown_global_motion") == "unknown_global_motion"


def test_calibrated_contact_roi_has_priority_over_bottle_roi(monkeypatch, tmp_path):
    from grasp_project.dashboard import web_dashboard as dashboard

    path = tmp_path / "davis_to_realsense_homography.json"
    path.write_text(
        json.dumps(
            {
                "homography": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                "reprojection_error_px": 0.5,
                "transform_direction": "davis_to_realsense",
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )
    monkeypatch.setattr(dashboard, "DAVIS_HOMOGRAPHY_PATH", path)
    realtime = {
        "writer_session_id": "writer-a",
        "frame_id": 3,
        "target_status": "TRACKING",
        "tracking_good": True,
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "bottle_geometry_tracking_ok": True,
        "live_rgb_shape": [480, 640, 3],
        "bottle_geometry": {
            "bbox_xywh": [160, 40, 240, 400],
            "oriented_bbox": [[160, 40], [400, 40], [400, 440], [160, 440]],
        },
        "target_clamp": {
            "clamp_band_polygon": [[190, 190], [370, 190], [370, 250], [190, 250]],
        },
    }

    context, reason = dashboard.build_davis_target_context(realtime, 0.1, now=1.0)

    assert reason == "fresh_current_tracked_target"
    assert context["roi_source"] == "realsense_contact_projected"
    assert context["roi_priority_selected"] == "calibrated_realsense_contact_roi"
    assert context["projection_metadata"]["alignment_status"] == "CALIBRATED_2D_REFERENCE_PLANE"


def test_boundary_sets_bottle_axis_without_camera(monkeypatch):
    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **kwargs):
            self.roi_xywh = kwargs.get("roi_xywh")
            self.configured_roi = kwargs.get("roi_xywh")

        def start(self):
            return {}

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    analyzer = motion_gui.DavisMotionAnalyzer(roi_xywh=[20, 20, 100, 200])
    analyzer.set_boundary([[40, 20], [60, 20], [60, 220], [40, 220]], source="realsense_projected_boundary")

    assert analyzer.bottle_axis_vector_px is not None
    assert abs(analyzer.bottle_axis_vector_px[1]) > 0.99
    assert analyzer.bottle_axis_source == "realsense_projected_boundary"


def test_live_quality_publishes_closeup_aliases_and_stale_reason():
    from grasp_project.quality.live_quality_monitor import (
        BaselineGraspQuality,
        LiveGraspObservation,
        LiveGraspQualityMonitor,
    )

    baseline = BaselineGraspQuality(valid=True, analytic_score=80.0, analytic_label="GOOD")
    observation = LiveGraspObservation(
        target_status="TRACKING",
        detected_contacts=2,
        grasp_stage="POST_GRASP_MONITORING",
        post_grasp_active=True,
        davis_object_motion_valid=True,
        davis_event_freshness="STALE",
        davis_events_valid_for_live_quality=False,
        davis_motion_usable_for_grasp=True,
        davis_motion_confidence=0.8,
        davis_motion_class="GROSS_SLIP",
        sync_quality="GOOD",
        davis_closeup_window="stability",
        gross_slip_score=100.0,
    )

    result = LiveGraspQualityMonitor().update(baseline, observation)
    payload = result.to_dict()

    assert payload["davis_closeup_used_for_live_quality"] is False
    assert payload["davis_closeup_exclusion_reason"] == "stale_events"
    assert payload["davis_closeup_confidence"] == 0.8
    assert payload["davis_closeup_motion_class"] == "GROSS_SLIP"
    assert payload["davis_closeup_window"] == "stability"
    assert payload["deviations"]["gross_slip"] == 0.0


def test_benchmark_keeps_global_activity_out_of_closeup_motion_metrics():
    from grasp_project.benchmarking.benchmark_runner import _sample_from_card
    from grasp_project.benchmarking.evaluators import evaluate_configuration_card
    from grasp_project.benchmarking.masked_feature_view import build_feature_packet

    davis = {
        "event_state": "STREAM_ACTIVE_EVENTS",
        "event_rate": 20000.0,
        "raw_motion_class": "LIGHTING_CHANGE",
        "object_motion_valid": False,
        "roi_source": "full_frame",
        "displacement_px": 20.0,
        "rotation_delta_deg": 12.0,
        "vibration_amplitude_px_rms": 8.0,
        "davis_closeup_used_for_live_quality": False,
    }
    packet = build_feature_packet(
        realsense={},
        davis=davis,
        pointcloud={},
        sync={"davis_event_freshness": "CURRENT"},
        quality={},
        live_quality={},
        embodiment={},
    )
    config = {
        "configuration_id": "events",
        "display_name": "DAVIS Events",
        "enabled_groups": ["davis_events"],
        "feature_ids": ["event_rate", "event_motion_activity", "event_stability"],
    }
    card = evaluate_configuration_card(config, packet)
    sample = _sample_from_card(card, {"source_status": {}}, config)

    assert sample["event_health_valid"] is True
    assert sample["closeup_object_motion_valid"] is False
    assert sample["closeup_roi_source"] == "full_frame"
    assert sample["closeup_displacement_px"] is None
    assert sample["closeup_rotation_deg"] is None
    assert sample["closeup_used_for_live_quality"] is False
    assert sample["davis_displacement_px"] is None


def test_dashboard_exposes_four_closeup_diagnostic_groups():
    from grasp_project.dashboard import web_dashboard

    response = web_dashboard.app.test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    for element_id in (
        "davisRawActivityCards",
        "davisObjectMotionCards",
        "davisCloseupCards",
        "davisDisturbanceCards",
    ):
        assert element_id in html

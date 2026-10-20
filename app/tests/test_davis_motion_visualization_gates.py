"""Camera-free tests for the DAVIS object-motion visualization gates.

A convex hull of recent events may only be presented as OBJECT motion when it
is local, object-specific, and not a global disturbance (lighting change,
camera shake, background motion). Global activity must stay diagnostic.
"""

import numpy as np
import pytest

from grasp_project.davis.motion_gui import (
    GLOBAL_DISTURBANCE_CLASSES,
    MOTION_HULL_MIN_EVENT_COUNT,
    apply_object_motion_presentation_gate,
    local_to_global_event_density_ratio,
    motion_hull,
    object_motion_validity,
)

FRAME = (346, 260)


def make_events(n, x_range, y_range, t0=100.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*x_range, size=n)
    ys = rng.uniform(*y_range, size=n)
    ts = np.linspace(t0, t0 + 0.09, n)
    ps = rng.integers(0, 2, size=n)
    return np.column_stack([ts, xs, ys, ps]).astype(np.float64)


def full_frame_events(n=400):
    return make_events(n, (0, FRAME[0] - 1), (0, FRAME[1] - 1))


def local_cluster_events(n=200):
    return make_events(n, (150, 205), (100, 150))


ROI_LOCAL = [140, 90, 80, 75]


def test_full_frame_burst_is_global_activity_not_object_motion():
    events = full_frame_events()
    hull = motion_hull(events, [0, 0, *FRAME], frame_size=FRAME)
    ratio = local_to_global_event_density_ratio(events, [0, 0, *FRAME], frame_size=FRAME)
    gates = object_motion_validity(hull, "full_frame", hull["motion_hull_event_count"], ratio, "LIGHTING_CHANGE", 0.9)
    assert gates["motion_hull_valid"] is False
    assert gates["object_motion_valid"] is False
    assert gates["motion_activity_label"] == "GLOBAL_ACTIVITY"
    assert "hull_area_too_large" in gates["motion_hull_reason"] or "disturbance_lighting_change" in gates["motion_hull_reason"]


def test_roi_local_cluster_is_valid_object_motion():
    events = local_cluster_events()
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    ratio = local_to_global_event_density_ratio(events, ROI_LOCAL, frame_size=FRAME)
    gates = object_motion_validity(hull, "aps_target", hull["motion_hull_event_count"], ratio, "INCIPIENT_SLIP", 0.8)
    assert gates["motion_hull_valid"] is True
    assert gates["object_motion_valid"] is True
    assert gates["motion_activity_label"] == "OBJECT_MOTION"


def test_fixed_grasp_zone_is_usable_only_when_explicitly_post_grasp():
    events = local_cluster_events()
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    preclose = object_motion_validity(
        hull, "configured_roi", 200, 3.0, "INCIPIENT_SLIP", 0.8,
        baseline_compatible=True,
    )
    postgrasp = object_motion_validity(
        hull, "configured_roi", 200, 3.0, "INCIPIENT_SLIP", 0.8,
        baseline_compatible=True,
        allow_post_grasp_zone=True,
    )
    assert preclose["object_motion_valid"] is False
    assert preclose["grasp_zone_motion_valid"] is False
    assert postgrasp["object_motion_valid"] is False
    assert postgrasp["grasp_zone_motion_valid"] is True
    assert postgrasp["motion_attribution"] == "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED"


def test_fixed_grasp_zone_rejects_global_disturbance():
    events = local_cluster_events()
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    gates = object_motion_validity(
        hull, "configured_roi", 200, 3.0, "LIGHTING_CHANGE", 0.9,
        baseline_compatible=True,
        allow_post_grasp_zone=True,
    )
    assert gates["grasp_zone_motion_valid"] is False


def test_huge_hull_rejected_for_area():
    events = full_frame_events()
    hull = motion_hull(events, [0, 0, *FRAME], frame_size=FRAME)
    assert hull["motion_hull_area_fraction"] > 0.35
    gates = object_motion_validity(hull, "aps_target", 400, 3.0, "GROSS_SLIP", 0.9)
    assert gates["object_motion_valid"] is False
    assert "hull_area_too_large" in gates["object_motion_reason"]


def test_border_touching_hull_rejected():
    # tight band hugging the left border: small area, most vertices at border
    events = make_events(120, (0, 2), (0, FRAME[1] - 1))
    hull = motion_hull(events, [0, 0, 40, FRAME[1]], frame_size=FRAME)
    assert hull["motion_hull_border_touch_ratio"] > 0.30
    gates = object_motion_validity(hull, "aps_target", 120, 2.0, "INCIPIENT_SLIP", 0.9)
    assert gates["object_motion_valid"] is False
    assert "border_touch_too_high" in gates["object_motion_reason"]


def test_full_frame_roi_is_not_object_specific():
    events = local_cluster_events()
    hull = motion_hull(events, [0, 0, *FRAME], frame_size=FRAME)
    gates = object_motion_validity(hull, "full_frame", hull["motion_hull_event_count"], None, "STABLE", 0.9)
    assert gates["roi_object_specific"] is False
    assert gates["object_motion_valid"] is False
    assert "full_frame_roi_only" in gates["object_motion_reason"]
    assert gates["object_specific_motion_confidence"] == "low"


def test_too_few_events_rejected():
    events = local_cluster_events(n=MOTION_HULL_MIN_EVENT_COUNT - 5)
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    gates = object_motion_validity(hull, "aps_target", hull["motion_hull_event_count"], 3.0, "STABLE", 0.9)
    assert gates["motion_hull_valid"] is False
    assert "too_few_events" in gates["motion_hull_reason"]


def test_low_confidence_and_nonlocal_activity_rejected():
    events = local_cluster_events()
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    low_conf = object_motion_validity(hull, "aps_target", 200, 3.0, "STABLE", 0.3)
    assert low_conf["object_motion_valid"] is False
    assert "low_motion_confidence" in low_conf["object_motion_reason"]
    nonlocal_gates = object_motion_validity(hull, "aps_target", 200, 1.0, "STABLE", 0.9)
    assert nonlocal_gates["object_motion_valid"] is False
    assert "activity_not_local_to_roi" in nonlocal_gates["object_motion_reason"]


def test_low_roi_projection_confidence_and_new_baseline_suppress_object_motion():
    events = local_cluster_events()
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    low_projection = object_motion_validity(
        hull,
        "aps_target",
        200,
        3.0,
        "STABLE",
        0.9,
        roi_confidence=0.3,
    )
    assert low_projection["object_motion_valid"] is False
    assert "low_roi_projection_confidence" in low_projection["object_motion_reason"]

    baseline_not_ready = object_motion_validity(
        hull,
        "realsense_contact_projected",
        200,
        3.0,
        "STABLE",
        0.9,
        roi_confidence=0.9,
        baseline_compatible=False,
    )
    assert baseline_not_ready["object_motion_valid"] is False
    assert "baseline_roi_context_not_ready" in baseline_not_ready["object_motion_reason"]


@pytest.mark.parametrize("klass", sorted(GLOBAL_DISTURBANCE_CLASSES))
def test_disturbance_classes_never_object_motion(klass):
    events = local_cluster_events()
    hull = motion_hull(events, ROI_LOCAL, frame_size=FRAME)
    gates = object_motion_validity(hull, "aps_target", 200, 3.0, klass, 0.95)
    assert gates["motion_hull_valid"] is False
    assert gates["object_motion_valid"] is False
    assert gates["motion_activity_label"] == "GLOBAL_ACTIVITY"


def test_local_to_global_ratio_math():
    # all events inside a small ROI -> very high density ratio
    events = local_cluster_events()
    ratio = local_to_global_event_density_ratio(events, ROI_LOCAL, frame_size=FRAME)
    assert ratio is None or ratio > 10.0
    # uniform events -> ratio near 1
    uniform_ratio = local_to_global_event_density_ratio(full_frame_events(2000), ROI_LOCAL, frame_size=FRAME)
    assert uniform_ratio is not None and 0.5 < uniform_ratio < 1.6
    # full-frame ROI -> None (no outside region)
    assert local_to_global_event_density_ratio(events, [0, 0, *FRAME], frame_size=FRAME) is None


def test_lighting_change_excluded_from_live_quality_penalties_and_gates():
    from grasp_project.quality.live_quality_monitor import (
        BaselineGraspQuality,
        LiveGraspObservation,
        LiveGraspQualityMonitor,
    )

    monitor = LiveGraspQualityMonitor()
    baseline = BaselineGraspQuality(
        valid=True,
        analytic_score=85.0,
        analytic_label="GOOD",
        quality_compute_id="test-bottle",
        writer_session_id="writer-1",
        lock_frame_id=42,
        required_contacts=2,
    )

    def make_obs(motion_class, usable=None):
        return LiveGraspObservation(
            timestamp=1000.0,
            writer_session_id="writer-1",
            lock_frame_id=42,
            lock_attempt_id="attempt-1",
            bottle_target_id="bottle-1",
            target_status="TRACKING",
            detected_contacts=2,
            grasp_stage="POST_GRASP_MONITORING",
            post_grasp_active=True,
            incipient_slip_score=82.0,
            gross_slip_score=90.0,
            rotation_score=10.0,
            vibration_score=5.0,
            davis_cumulative_slip_px=25.0,
            davis_motion_confidence=0.9,
            davis_motion_class=motion_class,
            davis_event_freshness="CURRENT",
            davis_events_valid_for_live_quality=True,
            davis_object_motion_valid=True,
            davis_target_lock_provenance_valid=True,
            davis_target_lock_source_writer_session_id="writer-1",
            davis_target_lock_source_lock_frame_id=42,
            davis_target_lock_source_attempt_id="attempt-1",
            davis_target_lock_source_bottle_target_id="bottle-1",
            sync_quality="GOOD",
            davis_motion_usable_for_grasp=usable,
        )

    excluded = monitor.update(baseline, make_obs("LIGHTING CHANGE", usable=False))
    assert excluded.davis_motion_used_for_live_quality is False
    assert "excluded" in (excluded.davis_motion_live_quality_reason or "")
    assert excluded.deviations.gross_slip == 0.0
    assert excluded.deviations.incipient_slip == 0.0
    assert "gross_slip" not in excluded.hard_gates

    monitor2 = LiveGraspQualityMonitor()
    slipping = monitor2.update(baseline, make_obs("GROSS_SLIP", usable=True))
    assert slipping.davis_motion_used_for_live_quality is True
    assert slipping.deviations.gross_slip > 0.0
    assert "gross_slip" in slipping.hard_gates


def test_pre_close_object_motion_stays_diagnostic_not_slip_quality():
    scores = {
        "raw_motion_class": "GROSS_SLIP",
        "motion_class": "GROSS_SLIP",
        "incipient_slip_score": 80.0,
        "gross_slip_score": 95.0,
        "rotation_risk_score": 20.0,
        "vibration_risk_score": 10.0,
    }
    gated = apply_object_motion_presentation_gate(
        scores,
        {
            "object_motion_valid": True,
            "davis_motion_stage": "PRE_CLOSE",
            "davis_motion_usable_for_grasp": False,
        },
    )

    assert gated["motion_class"] == "EVENT ACTIVITY"
    assert gated["object_motion_class"] == "PRE-CLOSE MOTION"
    assert gated["gross_slip_score"] == 0.0
    assert gated["raw_event_activity_scores"]["gross_slip_score"] == 95.0
    assert "excluded from grasp-slip quality" in gated["response_reason"]


def test_camera_shake_excluded_like_lighting():
    from grasp_project.quality.live_quality_monitor import (
        BaselineGraspQuality,
        LiveGraspObservation,
        LiveGraspQualityMonitor,
    )

    monitor = LiveGraspQualityMonitor()
    baseline = BaselineGraspQuality(valid=True, analytic_score=80.0, analytic_label="GOOD", quality_compute_id="k", required_contacts=2)
    obs = LiveGraspObservation(
        timestamp=1.0,
        target_status="TRACKING",
        detected_contacts=2,
        gross_slip_score=95.0,
        davis_motion_class="CAMERA SHAKE",
        davis_event_freshness="CURRENT",
        davis_events_valid_for_live_quality=True,
    )
    result = monitor.update(baseline, obs)
    assert result.davis_motion_used_for_live_quality is False
    assert result.deviations.gross_slip == 0.0
    assert "gross_slip" not in result.hard_gates


def test_benchmark_packet_separates_event_health_from_object_motion():
    from grasp_project.benchmarking.masked_feature_view import build_feature_packet

    packet = build_feature_packet(
        realsense={},
        davis={
            "event_state": "STREAM_ACTIVE_EVENTS",
            "event_rate": 60000.0,
            "raw_motion_class": "LIGHTING_CHANGE",
            "motion_class": "LIGHTING CHANGE",
            "object_motion_valid": False,
            "object_motion_reason": "disturbance_lighting_change",
            "davis_motion_excluded_reason": "lighting_change",
        },
        pointcloud={}, sync={}, quality={}, live_quality={}, embodiment={},
    )
    values = packet.values if isinstance(packet.values, dict) else packet.values
    activity = values["event_motion_activity"]
    assert activity["event_health_valid"] is True
    assert activity["object_motion_valid"] is False
    assert activity["disturbance_detected"] is True
    assert activity["davis_motion_validity"] == "disturbance"
    assert activity["davis_motion_exclusion_reason"] == "lighting_change"


def test_overlay_draws_no_hull_for_invalid_object_motion():
    cv2 = pytest.importorskip("cv2")
    from grasp_project.dashboard.web_dashboard import draw_davis_motion_hull

    hull_points = [[10, 10], [340, 10], [340, 250], [10, 250]]
    base_report = {
        "roi_xywh": [0, 0, 346, 260],
        "roi_source": "full_frame",
        "motion_hull_points": hull_points,
        "raw_motion_class": "LIGHTING_CHANGE",
        "davis_safety_class": "LIGHTING CHANGE",
        "motion_hull_valid": False,
        "object_motion_valid": False,
        "object_motion_reason": "disturbance_lighting_change;full_frame_roi_only",
        "motion_activity_label": "GLOBAL_ACTIVITY",
        "lightweight_event_metrics": {"event_rate": 60000.0, "centroid_x": 170.0, "centroid_y": 130.0},
    }
    img = draw_davis_motion_hull(base_report)
    fill = np.all(img == np.array([20, 80, 45], dtype=np.uint8), axis=-1)
    assert not bool(fill.any()), "green object-motion fill must not appear for global disturbance"

    valid_report = dict(base_report)
    valid_report.update({
        "roi_source": "aps_target",
        "raw_motion_class": "INCIPIENT_SLIP",
        "davis_safety_class": "MICRO SLIP",
        "motion_hull_valid": True,
        "object_motion_valid": True,
        "roi_object_specific": True,
        "motion_activity_label": "OBJECT_MOTION",
        "motion_hull_points": [[150, 100], [205, 100], [205, 150], [150, 150]],
        "dx_px": 2.4, "dy_px": -1.1, "displacement_px": 2.6,
        "motion_estimation_confidence": 0.71,
    })
    img_valid = draw_davis_motion_hull(valid_report)
    fill_valid = np.all(img_valid == np.array([20, 80, 45], dtype=np.uint8), axis=-1)
    assert bool(fill_valid.any()), "valid object motion should render the compact hull"


def test_operator_davis_footer_hides_event_rate_until_post_grasp(monkeypatch):
    from grasp_project.dashboard import web_dashboard

    texts = []
    original_put_text = web_dashboard.cv2.putText

    def record_text(image, text, *args, **kwargs):
        texts.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(web_dashboard.cv2, "putText", record_text)
    report = {
        "current_stage": "PRE_CLOSE",
        "lightweight_event_metrics": {"event_rate": 1344.0},
    }
    web_dashboard.draw_davis_motion_hull(report)
    assert "DAVIS  |  ARMED" in texts
    assert "POST-GRASP ONLY" in texts
    assert not any("events/s" in text for text in texts)

    texts.clear()
    web_dashboard.draw_davis_motion_hull({
        **report,
        "current_stage": "POST_GRASP",
    })
    assert "1344 events/s" in texts
    assert "POST-GRASP ONLY" not in texts


def test_event_heatmap_uses_aps_background_and_hides_isolated_display_noise():
    from grasp_project.davis.motion_gui import heatmap_rgb

    background = np.full((260, 346, 3), [70, 85, 100], dtype=np.uint8)
    events = np.asarray(
        [
            [1.000, 60, 60, 1],
            [1.002, 61, 60, 1],
            [1.004, 60, 61, 0],
            [1.004, 150, 150, 1],  # isolated display noise
        ],
        dtype=float,
    )
    original = events.copy()

    image = heatmap_rgb(
        events,
        [20, 20, 100, 80],
        background_rgb=background,
        neighborhood_filter=True,
    )

    np.testing.assert_array_equal(events, original)
    np.testing.assert_array_equal(image[150, 150], background[150, 150])
    assert not np.array_equal(image[60, 60], background[60, 60])
    np.testing.assert_array_equal(image[200, 200], background[200, 200])


def test_benchmark_sample_separates_event_health_from_object_motion():
    from grasp_project.benchmarking.masked_feature_view import build_feature_packet
    from grasp_project.benchmarking.evaluators import evaluate_configuration_card
    from grasp_project.benchmarking.benchmark_runner import _sample_from_card

    davis = {
        "event_state": "STREAM_ACTIVE_EVENTS", "event_rate": 20000.0,
        "raw_motion_class": "LIGHTING_CHANGE", "motion_class": "LIGHTING CHANGE",
        "object_motion_valid": False, "motion_activity_label": "GLOBAL_ACTIVITY",
        "davis_motion_excluded_reason": "lighting_change",
        "motion_estimate": {"disturbance_type": "LIGHTING_CHANGE"},
    }
    packet = build_feature_packet(realsense={}, davis=davis, pointcloud={}, sync={}, quality={}, live_quality={}, embodiment={})
    config = {"configuration_id": "expanded_davis_events", "display_name": "DAVIS Events",
              "enabled_groups": ["davis_events"], "feature_ids": ["event_rate", "event_motion_activity", "event_stability"]}
    sample = _sample_from_card(evaluate_configuration_card(config, packet), {"source_status": {}}, config)
    # event health is valid, but object motion is a disturbance and must not
    # register as UNKNOWN (the old bug)
    assert sample.get("event_health_valid") is True
    assert sample.get("object_motion_valid") is False
    assert sample.get("disturbance_detected") is True
    assert sample.get("davis_motion_validity") == "disturbance"
    assert sample.get("davis_motion_exclusion_reason") == "lighting_change"
    # a stationary bottle is never "detected" by DAVIS events
    assert sample.get("bottle_detected") is False

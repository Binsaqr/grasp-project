"""Camera-free quantitative DAVIS event-motion tests."""

import math
import time

import numpy as np

from grasp_project.benchmarking.feature_mask import build_benchmark_lab_status
from grasp_project.davis.motion_estimation import DavisMotionEstimator


ROI = [40, 40, 160, 160]


def _cluster(center, t0, t1, *, count=120, angle_deg=0.0, spread=16.0, seed=1):
    rng = np.random.default_rng(seed)
    ts = np.linspace(t0, t1, count, endpoint=False)
    base = np.column_stack([
        np.linspace(-spread, spread, count),
        rng.normal(0.0, 1.0, count),
    ])
    theta = math.radians(angle_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    pts = base @ rot.T + np.asarray(center, dtype=float)
    polarity = np.arange(count) % 2
    return np.column_stack([ts, pts[:, 0], pts[:, 1], polarity])


def _uniform_events(t0, t1, *, count=1600, seed=2, polarity=1):
    rng = np.random.default_rng(seed)
    ts = np.linspace(t0, t1, count, endpoint=False)
    xs = rng.uniform(0, 345, count)
    ys = rng.uniform(0, 259, count)
    pol = np.full(count, polarity)
    return np.column_stack([ts, xs, ys, pol])


def _estimate(events, **kwargs):
    estimator = kwargs.pop("estimator", DavisMotionEstimator())
    return estimator.estimate(events, now=kwargs.pop("now", 0.2), roi_xywh=kwargs.pop("roi", ROI), roi_source=kwargs.pop("roi_source", "configured_roi"), **kwargs)


def _translation_pair(previous_center, current_center, end, *, seed=40):
    return np.vstack([
        _cluster(previous_center, end - 0.2, end - 0.101, seed=seed),
        _cluster(current_center, end - 0.1, end - 0.001, seed=seed),
    ])


def test_no_events_reports_stable_without_fake_precision():
    est = _estimate(np.empty((0, 4)))
    assert est.motion_class == "STABLE"
    assert est.estimate_valid is False
    assert est.invalid_reason == "no_events"
    assert est.dx_px is None


def test_pure_translation_reports_pixel_dx():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.099),
        _cluster((105, 100), 0.1, 0.199),
    ])
    est = _estimate(events)
    assert est.estimate_valid is True
    assert est.dx_px == pytest_approx(5.0, abs=0.8)
    assert abs(est.dy_px or 0.0) < 1.0
    assert est.translation_method in {"event_image_phase_correlation", "event_centroid"}


def test_explicit_analysis_end_and_selected_200ms_window_use_same_event_interval():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.199, count=180, seed=51),
        _cluster((106, 100), 0.2, 0.399, count=180, seed=51),
    ])

    est = DavisMotionEstimator().estimate(
        events,
        now=0.55,
        analysis_end=0.4,
        motion_window_ms=200.0,
        roi_xywh=ROI,
        roi_source="configured_roi",
    )

    assert est.event_window_end == pytest_approx(0.4)
    assert est.event_window_ms == pytest_approx(200.0)
    assert est.estimate_valid is True
    assert est.dx_px == pytest_approx(6.0, abs=1.0)


def test_analyzer_lagged_fallback_and_estimator_share_host_end_and_800ms_window():
    from grasp_project.davis import motion_gui

    analysis_end = 1.0
    events = np.vstack([
        _cluster((100, 100), 0.61, 0.79, count=120, seed=52),
        _cluster((106, 100), 0.81, 0.89, count=120, seed=52),
    ])

    selected, window_sec, mode, selected_end = motion_gui.select_event_analysis_window(
        events,
        analysis_end,
        ROI,
    )
    estimate = DavisMotionEstimator().estimate(
        events,
        analysis_end=selected_end,
        motion_window_ms=window_sec * 1000.0,
        roi_xywh=ROI,
        roi_source="configured_roi",
    )

    assert mode == "fallback_800ms"
    assert selected_end == pytest_approx(analysis_end)
    assert estimate.event_window_end == pytest_approx(analysis_end)
    assert estimate.event_window_ms == pytest_approx(800.0)
    assert estimate.event_count == selected.shape[0]


def test_downward_motion_reports_slip_pixels():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.099),
        _cluster((100, 106), 0.1, 0.199),
    ])
    est = _estimate(events)
    assert est.downward_slip_px == pytest_approx(6.0, abs=1.0)
    assert est.displacement_px >= 5.0


def test_small_repeated_downward_motion_becomes_incipient_slip():
    estimator = DavisMotionEstimator()
    first = np.vstack([_cluster((100, 100), 0.0, 0.099), _cluster((100, 103), 0.1, 0.199)])
    second = np.vstack([_cluster((100, 103), 0.2, 0.299), _cluster((100, 106), 0.3, 0.399)])
    estimator.estimate(first, now=0.2, roi_xywh=ROI, roi_source="configured_roi")
    est = estimator.estimate(second, now=0.4, roi_xywh=ROI, roi_source="configured_roi")
    assert est.motion_class in {"INCIPIENT_SLIP", "UNKNOWN_MOTION"}
    assert est.cumulative_downward_slip_px >= 4.0
    assert est.incipient_slip_score > 0


def test_large_downward_motion_becomes_gross_slip():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.099),
        _cluster((100, 114), 0.1, 0.199),
    ])
    est = _estimate(events)
    assert est.motion_class == "GROSS_SLIP"
    assert est.gross_slip_score >= 55


def test_pure_rotation_reports_degrees_when_confident():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.099, angle_deg=0.0),
        _cluster((100, 100), 0.1, 0.199, angle_deg=5.0),
    ])
    est = _estimate(events)
    assert est.rotation_valid is True
    assert est.rotation_delta_deg == pytest_approx(5.0, abs=1.0)
    assert est.rotation_method in {"feature_affine_ransac", "event_cloud_pca_orientation"}


def test_oscillating_motion_reports_vibration_not_gross_slip():
    events = np.vstack([
        _cluster((100, 100), 0.00, 0.049, count=80),
        _cluster((100, 106), 0.05, 0.099, count=80),
        _cluster((100, 100), 0.10, 0.149, count=80),
        _cluster((100, 106), 0.15, 0.199, count=80),
    ])
    est = _estimate(events)
    assert est.vibration_amplitude_px_rms is not None
    assert est.vibration_amplitude_px_rms > 2.5
    assert est.motion_class == "SHAKE_OR_VIBRATION"
    assert est.motion_class != "GROSS_SLIP"


def test_uniform_full_frame_flash_is_lighting_change_not_slip():
    events = np.vstack([
        _uniform_events(0.0, 0.099, polarity=1),
        _uniform_events(0.1, 0.199, polarity=1, seed=3),
    ])
    est = _estimate(events, roi=None, roi_source="full_frame")
    assert est.motion_class == "LIGHTING_CHANGE"
    assert est.gross_slip_score == 0
    assert est.disturbance_rejection_applied is True


def test_dense_local_motion_is_not_misclassified_as_uniform_lighting():
    xs, ys = np.meshgrid(np.arange(80.0, 110.0), np.arange(80.0, 110.0))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    count = points.shape[0]
    previous = np.column_stack([
        np.linspace(0.0, 0.099, count, endpoint=False),
        points,
        np.arange(count) % 2,
    ])
    current = np.column_stack([
        np.linspace(0.1, 0.199, count, endpoint=False),
        points + np.asarray([2.0, 0.0]),
        np.arange(count) % 2,
    ])

    est = _estimate(np.vstack([previous, current]))

    assert est.estimate_valid is True
    assert est.dx_px == pytest_approx(2.0, abs=0.8)
    assert est.full_frame_uniformity < 0.2
    assert est.disturbance_type != "LIGHTING_CHANGE"


def test_global_camera_shake_is_not_object_slip():
    events = np.vstack([
        _uniform_events(0.0, 0.099, polarity=0),
        _uniform_events(0.1, 0.199, polarity=0, seed=4),
    ])
    est = _estimate(events, roi=ROI)
    assert est.motion_class in {"CAMERA_SHAKE", "BACKGROUND_MOTION", "LIGHTING_CHANGE"}
    assert est.disturbance_rejection_applied is True


def test_background_motion_outside_roi_is_separated():
    events = np.vstack([
        _cluster((300, 220), 0.0, 0.099, count=120),
        _cluster((310, 220), 0.1, 0.199, count=120),
    ])
    est = _estimate(events, roi=ROI)
    assert est.motion_class == "BACKGROUND_MOTION"
    assert est.estimate_valid is False
    assert est.background_motion_score > 0


def test_alternating_translation_accumulates_path_not_false_net_slip():
    estimator = DavisMotionEstimator({"phase_correlation_min_response": 1.1})
    estimate = None
    for index, dy in enumerate((3.0, -3.0, 3.0, -3.0, 3.0, -3.0)):
        end = 0.2 + index * 0.1
        estimate = estimator.estimate(
            _translation_pair((100.0, 100.0), (100.0, 100.0 + dy), end),
            now=end,
            roi_xywh=ROI,
            roi_source="configured_roi",
        )

    assert estimate is not None
    assert estimate.cumulative_slip_px < 1.0
    assert estimate.cumulative_motion_path_px >= 17.0
    assert estimate.motion_class != "GROSS_SLIP"


def test_one_way_translation_reaches_persistent_net_slip():
    estimator = DavisMotionEstimator({"phase_correlation_min_response": 1.1})
    estimate = None
    for index in range(7):
        end = 0.2 + index * 0.1
        estimate = estimator.estimate(
            _translation_pair((100.0, 100.0), (100.0, 103.0), end),
            now=end,
            roi_xywh=ROI,
            roi_source="configured_roi",
        )

    assert estimate is not None
    assert estimate.cumulative_slip_px >= 20.0
    assert estimate.motion_class == "GROSS_SLIP"


def test_cumulative_motion_is_analysis_cadence_equivalent():
    def run(step):
        estimator = DavisMotionEstimator({"phase_correlation_min_response": 1.1})
        end = 0.2
        estimate = None
        while end <= 0.700001:
            estimate = estimator.estimate(
                _translation_pair((100.0, 100.0), (103.0, 100.0), end),
                now=end,
                roi_xywh=ROI,
                roi_source="configured_roi",
            )
            end += step
        return estimate

    ten_hz = run(0.1)
    twenty_hz = run(0.05)

    assert ten_hz is not None and twenty_hz is not None
    assert twenty_hz.cumulative_slip_px == pytest_approx(ten_hz.cumulative_slip_px, abs=0.2)
    assert twenty_hz.cumulative_motion_path_px == pytest_approx(ten_hz.cumulative_motion_path_px, abs=0.2)


def test_insufficient_events_are_invalid():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.099, count=10),
        _cluster((105, 100), 0.1, 0.199, count=10),
    ])
    est = _estimate(events)
    assert est.estimate_valid is False
    assert est.invalid_reason == "insufficient_events"


def test_baseline_jitter_suppresses_tiny_false_slip():
    estimator = DavisMotionEstimator()
    baseline = np.vstack([
        _cluster((100, 100), 0.00, 0.049, count=70, seed=10),
        _cluster((100, 101), 0.05, 0.099, count=70, seed=11),
        _cluster((100, 99), 0.10, 0.149, count=70, seed=12),
        _cluster((100, 101), 0.15, 0.199, count=70, seed=13),
    ])
    estimator.calibrate_baseline(baseline, now=0.2, roi_xywh=ROI)
    events = np.vstack([
        _cluster((100, 100), 0.2, 0.299, count=80),
        _cluster((100, 101), 0.3, 0.399, count=80),
    ])
    est = estimator.estimate(events, now=0.4, roi_xywh=ROI, roi_source="configured_roi")
    assert est.displacement_above_baseline_px <= 1.0
    assert est.motion_class in {"STABLE", "UNKNOWN_MOTION"}


def test_roi_local_motion_increases_object_confidence():
    local = np.vstack([
        _cluster((100, 100), 0.0, 0.099),
        _cluster((105, 100), 0.1, 0.199),
    ])
    est = _estimate(local, roi=ROI)
    assert est.local_to_global_event_ratio == 1.0
    assert est.object_motion_score > 0
    assert est.roi_used_for_motion is True


def test_rotation_confidence_low_with_too_few_features():
    events = np.vstack([
        _cluster((100, 100), 0.0, 0.099, count=30, angle_deg=0.0),
        _cluster((100, 100), 0.1, 0.199, count=30, angle_deg=5.0),
    ])
    est = _estimate(events)
    assert est.estimate_valid is False
    assert est.rotation_valid is False


def test_benchmark_feature_mask_exposes_quantitative_davis_fields():
    status = build_benchmark_lab_status(
        davis={
            "event_rate": 500,
            "motion_class": "INCIPIENT_SLIP",
            "displacement_px": 4.2,
            "rotation_delta_deg": 5.0,
            "cumulative_slip_px": 6.0,
            "vibration_amplitude_px_rms": 1.5,
            "motion_estimation_confidence": 0.8,
            "disturbance_type": "OBJECT_MOTION",
            "stability": 82,
        },
        configurations=[{
            "configuration_id": "davis_quant",
            "display_name": "DAVIS Quant",
            "feature_ids": ["event_rate", "event_motion_activity", "event_incipient_slip", "event_rotation", "event_vibration"],
        }],
    )
    card = status["comparison_cards"][0]
    assert card["davis_displacement_px"] == 4.2
    assert card["davis_rotation_delta_deg"] == 5.0
    assert any("measured 4.2 px" in item["message"] for item in card["contributions"])


def test_event_stream_monitor_report_includes_quantitative_motion_fields():
    from grasp_project.davis.event_stream import EventPostGraspMonitor

    now = time.monotonic() + 0.04
    events = np.vstack([
        _cluster((100, 100), now - 0.20, now - 0.101),
        _cluster((106, 100), now - 0.10, now - 0.001),
    ])
    monitor = EventPostGraspMonitor(enabled=False, roi_xywh=ROI)
    monitor.active = True
    monitor.start_time = now - 0.30

    event_dicts = [
        {"t": float(row[0]), "x": float(row[1]), "y": float(row[2]), "polarity": int(row[3])}
        for row in events
    ]
    report = monitor.update(events=event_dicts, write_report=False, analysis_origin=now - 0.30)

    assert report["motion_estimate_valid"] is True
    assert report["dx_px"] == pytest_approx(6.0, abs=1.0)
    assert abs(report["dy_px"] or 0.0) < 1.0
    assert report["motion_confidence"] == report["motion_estimation_confidence"]
    assert report["motion_estimate"]["coordinate_frame"] == "davis_pixel"


def test_event_stream_long_session_prunes_history_and_uses_rolling_window():
    from grasp_project.davis.event_stream import DAVIS_LIVE_HISTORY_SEC, EventPostGraspMonitor

    now = time.monotonic()
    monitor = EventPostGraspMonitor(enabled=False, roi_xywh=ROI)
    monitor.active = True
    monitor.start_time = now - 10.0
    monitor.events = [
        (now - DAVIS_LIVE_HISTORY_SEC - 1.0, 100.0, 100.0, 1),
        (now - 0.1, 101.0, 100.0, 1),
    ]

    report = monitor.update(events=[], write_report=False, analysis_origin=monitor.start_time)

    assert len(monitor.events) == 1
    assert report["event_count"] == 1
    assert report["analysis_window_mode"] == "rolling_after_initial_post_close"
    assert report["analysis_window_sec"] == 1.2


def test_motion_analyzer_histories_have_fixed_capacity(monkeypatch):
    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            return {}

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    analyzer = motion_gui.DavisMotionAnalyzer()

    assert analyzer.history.maxlen == 500
    assert analyzer.motion_class_history.maxlen == 500
    assert analyzer.decision_history.maxlen == 500


def pytest_approx(value, **kwargs):
    import pytest

    return pytest.approx(value, **kwargs)

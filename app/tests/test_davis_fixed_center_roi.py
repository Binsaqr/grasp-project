"""Camera-free regressions for the fixed DAVIS demo analysis region."""

import numpy as np

from grasp_project.common.config import load_named_config
from grasp_project.dashboard.web_dashboard import draw_davis_motion_hull
from grasp_project.davis import motion_gui
from grasp_project.davis.boundary_event_search import search_boundary_events


class _MonitorStub:
    def __init__(self, **kwargs):
        self.roi_xywh = kwargs.get("roi_xywh")
        self.configured_roi = kwargs.get("roi_xywh")
        self.active = True
        self.backend_name = "synthetic"
        self.owner_info = {}
        self.event_timestamp_basis = "synthetic_monotonic"
        self.event_timestamp_reanchor_count = 0

    def poll_event_and_frame_arrays(self):
        return np.empty((0, 4), dtype=float), []

    def start(self):
        return {}


def _fixed_analyzer(monkeypatch):
    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", _MonitorStub)
    return motion_gui.DavisMotionAnalyzer(
        roi_xywh=[1, 2, 3, 4],
        analysis_roi_mode=motion_gui.FIXED_CENTER_THIRD_MODE,
    )


def _bound_aps_context(**overrides):
    context = {
        "roi_source": "aps_target",
        "roi_xywh": [100, 70, 80, 100],
        "boundary_points": [[112, 78], [160, 78], [160, 160], [112, 160]],
        "boundary_source": "davis_aps_target_contour",
        "contact_band_points": [[120, 105], [150, 105], [150, 125], [120, 125]],
        "target_confidence": 0.92,
        "projection_metadata": {"target_confidence": 0.92},
        "captured_monotonic": 10.0,
        "aps_target_context_max_age_sec": 1.5,
        "aps_target_confirmed": True,
        "target_lock_provenance_valid": True,
        "source_writer_session_id": "writer-1",
        "source_lock_frame_id": 42,
        "source_attempt_id": "attempt-1",
        "source_bottle_target_id": "bottle-1",
        "mapping_physically_validated": False,
    }
    context.update(overrides)
    return context


def test_center_third_geometry_uses_both_axes_and_keeps_remainder_centered():
    assert motion_gui.configured_center_third_roi((346, 260)) == [115, 86, 116, 88]
    assert motion_gui.configured_center_third_roi((9, 9)) == [3, 3, 3, 3]


def test_release_config_uses_fixed_davis_native_grasp_zone():
    events = load_named_config("davis.yaml", default={})["events"]

    assert events["analysis_roi_mode"] == "fixed_center_third"
    assert events["diagnostic_roi_xywh"] == [115, 86, 116, 88]
    assert events["baseline_max_quiet_rate"] == 3000


def test_fixed_mode_ignores_moving_aps_or_realsense_roi(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)
    expected = [115, 86, 116, 88]

    analyzer.set_roi(
        [10, 12, 300, 220],
        source="aps_target",
        projection_metadata={"target_confidence": 0.99},
    )
    analyzer.set_boundary(
        [[10, 10], [300, 10], [300, 230], [10, 230]],
        source="davis_aps_target_contour",
    )
    analyzer.set_contact_band([[20, 80], [100, 80], [100, 100], [20, 100]])

    assert analyzer.roi_xywh == expected
    assert analyzer.roi_source == "configured_roi"
    assert analyzer.boundary_source == motion_gui.FIXED_CENTER_THIRD_BOUNDARY_SOURCE
    assert analyzer.contact_band_polygon is None
    assert analyzer.roi_projection["roi_projection_status"] == (
        motion_gui.FIXED_CENTER_THIRD_BOUNDARY_SOURCE
    )


def test_fixed_center_roi_remains_non_object_specific_and_boundary_cues_suppressed(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)
    gates = motion_gui.object_motion_validity(
        {
            "motion_hull_area_fraction": 0.02,
            "motion_hull_border_touch_ratio": 0.0,
        },
        analyzer.roi_source,
        event_count=100,
        local_global_ratio=10.0,
        raw_motion_class="LOCAL_MOTION",
        motion_confidence=0.95,
        baseline_compatible=True,
    )
    field = search_boundary_events(
        analyzer.boundary_points,
        np.asarray([[0.10, 115.0, 90.0, 1.0], [0.11, 116.0, 91.0, -1.0]]),
        0.1,
        now=0.15,
        boundary_source=analyzer.boundary_source,
        alignment_status=analyzer.roi_projection["roi_projection_status"],
        object_motion_valid=gates["object_motion_valid"],
        suppression_reason=gates["object_motion_reason"],
    ).to_dict()

    assert gates["roi_object_specific"] is False
    assert gates["roi_projection_uncalibrated"] is True
    assert gates["object_motion_valid"] is False
    assert "not_object_specific" in gates["object_motion_reason"]
    assert field["boundary_source"] == "CONFIGURED_CENTER_THIRD_UNCALIBRATED"
    assert field["measurement_status"] == "SUPPRESSED_BY_OBJECT_MOTION_VALIDITY"
    assert field["rotation_cue"] == field["slip_cue"] == field["vibration_cue"] == 0.0


def test_confirmed_aps_contour_is_separate_object_evidence_inside_fixed_view(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)

    assert analyzer.set_object_evidence_context(_bound_aps_context(), now=10.2) is True
    analysis = analyzer._active_analysis_context()

    assert analyzer.roi_xywh == [115, 86, 116, 88]
    assert analyzer.roi_source == "configured_roi"
    assert analysis["roi_source"] == "aps_target"
    assert analysis["roi_xywh"] == [115, 86, 65, 84]
    assert np.asarray(analysis["boundary_points"]).shape[0] >= 3
    assert analyzer.object_evidence_context["mapping_physically_validated"] is False
    assert analyzer.object_evidence_context["aps_object_evidence_status"] == (
        "ESTIMATED_OBJECT_LOCAL_UNCALIBRATED"
    )

    gates = motion_gui.object_motion_validity(
        {"motion_hull_area_fraction": 0.02, "motion_hull_border_touch_ratio": 0.0},
        analysis["roi_source"],
        event_count=100,
        local_global_ratio=3.0,
        raw_motion_class="LOCAL_MOTION",
        motion_confidence=0.9,
        roi_confidence=analysis["roi_projection"]["roi_confidence"],
        baseline_compatible=True,
    )
    assert gates["object_motion_valid"] is True
    assert gates["roi_object_specific"] is True
    assert gates["roi_projection_uncalibrated"] is True


def test_fixed_view_accepts_confirmed_native_aps_evidence_without_cross_camera_provenance(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)
    context = _bound_aps_context(target_lock_provenance_valid=False)

    assert analyzer.set_object_evidence_context(context, now=10.2) is True
    assert analyzer._active_analysis_context()["roi_source"] == "aps_target"
    assert analyzer.object_evidence_context["mapping_physically_validated"] is False


def test_fixed_view_rejects_aps_bbox_fallback_and_outside_contour(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)
    bbox_fallback = _bound_aps_context(boundary_source="davis_aps_target_bbox_fallback")
    assert analyzer.set_object_evidence_context(bbox_fallback, now=10.2) is False
    assert "requires_confirmed_contour" in analyzer.object_evidence_invalid_reason

    outside = _bound_aps_context(
        roi_xywh=[0, 0, 40, 40],
        boundary_points=[[5, 5], [30, 5], [30, 30], [5, 30]],
    )
    assert analyzer.set_object_evidence_context(outside, now=10.2) is False
    assert "outside_acquisition" in analyzer.object_evidence_invalid_reason


def test_analyzer_report_keeps_fixed_view_while_using_separate_evidence_roi(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)
    now = motion_gui.time.monotonic()
    assert analyzer.set_object_evidence_context(
        _bound_aps_context(captured_monotonic=now),
        now=now,
    )

    report = analyzer.update()

    assert report["davis_roi_xywh"] == [115, 86, 116, 88]
    assert report["acquisition_roi_xywh"] == [115, 86, 116, 88]
    assert report["roi_source"] == "aps_target"
    assert report["roi_xywh"] == [115, 86, 65, 84]
    assert report["object_evidence_context_current"] is True
    assert report["object_evidence_physically_validated"] is False
    assert report["aps_target_used_as_analysis_roi"] is False
    assert report["aps_target_used_as_object_evidence_roi"] is True
    assert report["object_motion_valid"] is False
    assert "too_few_events" in report["object_motion_reason"]


def test_stale_object_evidence_expires_before_analysis(monkeypatch):
    analyzer = _fixed_analyzer(monkeypatch)
    assert analyzer.set_object_evidence_context(_bound_aps_context(), now=10.2)

    assert analyzer._expire_object_evidence_if_stale(now=12.0) is True
    assert analyzer.object_evidence_context is None
    assert analyzer._active_analysis_context()["roi_source"] == "configured_roi"
    assert analyzer.object_evidence_invalid_reason == "aps_object_evidence_stale"


def test_presentation_heatmap_does_not_draw_aps_target_bbox():
    report = {
        "analysis_roi_fixed": True,
        "davis_roi_xywh": [115, 86, 116, 88],
        "object_motion_valid": False,
        "motion_hull_valid": False,
        "motion_class": "EVENT ACTIVITY",
    }
    without_target = draw_davis_motion_hull(report)
    with_unused_aps_target = draw_davis_motion_hull({
        **report,
        "davis_bbox_xyxy": [10, 20, 320, 240],
        "aps_target_bbox": [20, 30, 300, 230],
    })

    assert np.array_equal(without_target, with_unused_aps_target)

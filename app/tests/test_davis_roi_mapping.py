"""Tests for RealSense->DAVIS ROI mapping and its object-specific gating.

The mapper must accept the bbox_xyxy the realtime status actually publishes and
resolve a frame shape even when no shape field is present, so DAVIS can display
the approximate target neighborhood. The mapped ROI is uncalibrated, so it is
diagnostic-only and cannot authorize object-motion quality penalties.
"""

import json

import numpy as np
import pytest

from grasp_project.davis.motion_gui import (
    OBJECT_SPECIFIC_ROI_SOURCES,
    UNCALIBRATED_ROI_SOURCES,
    aps_target_context_from_detection,
    event_analysis_status_for_roi,
    motion_hull,
    object_motion_validity,
)

dashboard = pytest.importorskip("grasp_project.dashboard.web_dashboard")


def test_aps_detection_produces_clipped_object_specific_roi_only_when_detected():
    context = aps_target_context_from_detection(
        {
            "davis_bottle_detected": True,
            "davis_detection_status": "DETECTED",
            "davis_bbox_xyxy": [-3, 12, 340, 258],
            "davis_detection_confidence": 82.0,
            "aps_detection_source": "aps_contour_shape_score",
            "davis_detection_backend": "opencv_aps",
        },
        frame_id=14,
        captured_monotonic=10.0,
        host_timestamp=20.0,
    )

    assert context["roi_source"] == "aps_target"
    assert context["roi_priority_selected"] == "davis_aps_target_roi"
    assert context["roi_xywh"] == [0, 4, 346, 256]
    assert context["target_confidence"] == pytest.approx(0.82)
    assert context["projection_metadata"]["alignment_status"] == "DAVIS_NATIVE_APS"
    assert aps_target_context_from_detection(
        {
            "davis_bottle_detected": False,
            "davis_detection_status": "LOW_CONFIDENCE",
            "davis_bbox_xyxy": [50, 20, 120, 220],
            "davis_detection_confidence": 55.0,
        },
        frame_id=15,
        captured_monotonic=10.1,
    ) is None


def test_dashboard_rejects_stale_aps_target_context_and_preserves_calibrated_priority():
    class Analyzer:
        latest_aps_target_context = aps_target_context_from_detection(
            {
                "davis_bottle_detected": True,
                "davis_detection_status": "DETECTED",
                "davis_bbox_xyxy": [40, 20, 130, 240],
                "davis_detection_confidence": 90.0,
            },
            frame_id=3,
            captured_monotonic=5.0,
        )

    context, status = dashboard.current_davis_aps_target_context(Analyzer(), now=5.2)
    assert context["roi_source"] == "aps_target"
    assert context["aps_target_confirmed"] is True
    assert status["davis_target_context_mode"] == "davis_aps_target"
    assert status["davis_target_roi_priority_selected"] == "davis_aps_target_roi"
    stale, reason = dashboard.current_davis_aps_target_context(Analyzer(), now=6.0)
    assert stale is None
    assert reason.startswith("aps_target_stale_age_")

    calibrated_context = {
        "roi_priority_selected": "calibrated_realsense_contact_roi",
        "alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
    }
    assert dashboard.calibrated_realsense_target_context(
        calibrated_context,
        {"davis_target_context_valid": True},
    ) is True
    assert dashboard.calibrated_realsense_target_context(
        {"roi_priority_selected": "uncalibrated_realsense_contact_motion_diagnostic_roi"},
        {"davis_target_context_valid": True},
    ) is False


def test_dashboard_uses_bounded_analyzer_cadence_age_for_aps_context():
    context = aps_target_context_from_detection(
        {
            "davis_bottle_detected": True,
            "davis_detection_status": "DETECTED",
            "davis_bbox_xyxy": [40, 20, 130, 240],
            "davis_detection_confidence": 90.0,
        },
        frame_id=3,
        captured_monotonic=5.0,
    )
    context["aps_target_context_max_age_sec"] = 1.25
    analyzer = type("Analyzer", (), {"latest_aps_target_context": context})()

    current, _status = dashboard.current_davis_aps_target_context(analyzer, now=6.0)
    stale, reason = dashboard.current_davis_aps_target_context(analyzer, now=6.26)

    assert current["roi_source"] == "aps_target"
    assert stale is None
    assert reason.startswith("aps_target_stale_age_")


def test_aps_priority_preserves_complete_current_lock_provenance_for_export():
    aps_context = aps_target_context_from_detection(
        {
            "davis_bottle_detected": True,
            "davis_detection_status": "DETECTED",
            "davis_bbox_xyxy": [40, 20, 130, 240],
            "davis_detection_confidence": 90.0,
        },
        frame_id=9,
        captured_monotonic=5.0,
    )
    _, aps_status = dashboard.current_davis_aps_target_context(
        type("Analyzer", (), {"latest_aps_target_context": aps_context})(),
        now=5.1,
    )
    bound, status = dashboard.bind_aps_target_context_to_realsense_lock(
        aps_context,
        aps_status,
        {
            "source_frame_id": 77,
            "source_writer_session_id": "writer-1",
            "source_lock_frame_id": 42,
            "source_attempt_id": "attempt-1",
            "source_bottle_target_id": "bottle-1",
        },
        {
            "davis_target_context_valid": True,
            "davis_target_context_mode": "live_realsense_mapping",
            "davis_target_lock_age_ms": None,
        },
    )

    assert bound["roi_source"] == "aps_target"
    assert bound["alignment_status"] == "DAVIS_NATIVE_APS"
    assert bound["source_frame_id"] == 9
    assert bound["source_lock_provenance_frame_id"] == 77
    assert bound["aps_target_confirmed"] is True
    assert bound["target_lock_provenance_valid"] is True
    assert status["davis_target_lock_provenance_valid"] is True
    assert status["davis_target_lock_source_writer_session_id"] == "writer-1"
    assert status["davis_target_lock_source_lock_frame_id"] == 42
    assert status["davis_target_lock_source_attempt_id"] == "attempt-1"
    assert status["davis_target_lock_source_bottle_target_id"] == "bottle-1"
    assert dashboard._demo_identity_validation(
        status,
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "attempt_id": "attempt-1",
            "bottle_target_id": "bottle-1",
        },
        kind="davis",
    )["valid"] is True


def test_mapper_accepts_status_bbox_xyxy_without_shape_field():
    # exactly the shape of the live realtime status: bbox_xyxy, no live_rgb_shape
    realtime = {
        "bottle_geometry": {
            "bbox_xyxy": [301.0, 116.0, 427.0, 480.0],
            "bottle_extent_bbox_xyxy": [301.0, 116.0, 427.0, 480.0],
        },
        "target_status": "TRACKING",
    }
    roi, reason = dashboard.map_realsense_bbox_to_davis_roi(realtime)
    assert roi is not None, reason
    assert reason.startswith("approximate (uncalibrated)")
    x, y, w, h = roi
    assert 0 <= x < dashboard.DAVIS_ROI_FRAME_WIDTH
    assert 0 <= y < dashboard.DAVIS_ROI_FRAME_HEIGHT
    assert w >= 1 and h >= 1
    # box is right-of-center and tall -> ROI centroid should be right-of-center in DAVIS frame
    assert (x + w / 2.0) > dashboard.DAVIS_ROI_FRAME_WIDTH * 0.4


def test_mapper_still_bails_when_no_bbox():
    roi, reason = dashboard.map_realsense_bbox_to_davis_roi({"target_status": "TRACKING"})
    assert roi is None
    assert reason == "no_realsense_bbox"


def test_mapping_context_requires_fresh_current_target():
    realtime = {
        "target_status": "TRACKING",
        "tracking_good": True,
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "bottle_geometry_tracking_ok": True,
        "bottle_geometry": {"bbox_xyxy": [200, 100, 300, 400]},
    }

    assert dashboard.davis_realsense_mapping_context(realtime, 0.1) == (
        True,
        "fresh_current_tracked_target",
    )
    current, reason = dashboard.davis_realsense_mapping_context(realtime, 1.01)
    assert current is False
    assert reason == "realsense_status_stale"
    current, reason = dashboard.davis_realsense_mapping_context(
        {**realtime, "target_status": "LOST"},
        0.1,
    )
    assert current is False
    assert reason.startswith("target_not_current:")


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"grasp_band_relock_required": True}, "locked_grasp_band_relock_required"),
        ({"grasp_band_stale": True}, "locked_grasp_band_stale"),
        (
            {"grasp_band_locked": True, "grasp_band_current": False},
            "locked_grasp_band_not_current",
        ),
        (
            {"grasp_band_locked": True, "grasp_band_projected": False},
            "locked_grasp_band_not_projected",
        ),
    ],
)
def test_mapping_context_rejects_noncurrent_locked_band(updates, reason):
    realtime = {
        "target_status": "TRACKING",
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "bottle_geometry_tracking_ok": True,
        "bottle_geometry": {"bbox_xyxy": [200, 100, 300, 400]},
        **updates,
    }

    assert dashboard.davis_realsense_mapping_context(realtime, 0.1) == (False, reason)


def test_mapping_context_uses_current_identity_not_contact_readiness():
    realtime = {
        "target_status": "TRACKING",
        "tracking_good": False,
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "bottle_geometry_tracking_ok": True,
        "contact_patch_tracking_ok": False,
        "bottle_geometry": {"bbox_xyxy": [200, 100, 300, 400]},
    }

    assert dashboard.davis_realsense_mapping_context(realtime, 0.1) == (
        True,
        "fresh_current_tracked_target",
    )

    current, reason = dashboard.davis_realsense_mapping_context(
        {**realtime, "identity_tracking_ok": False},
        0.1,
    )
    assert current is False
    assert reason == "target_identity_tracking_invalid"


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("identity_measurement_current", False, "target_identity_measurement_not_current"),
        ("identity_measurement_current", None, "target_identity_measurement_not_current"),
        ("bottle_geometry_tracking_ok", False, "target_bottle_geometry_tracking_invalid"),
        ("bottle_geometry_tracking_ok", None, "target_bottle_geometry_tracking_invalid"),
    ],
)
def test_mapping_context_requires_current_identity_measurement_and_geometry(
    field,
    value,
    expected_reason,
):
    realtime = {
        "target_status": "TRACKING",
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "bottle_geometry_tracking_ok": True,
        "bottle_geometry": {"bbox_xyxy": [200, 100, 300, 400]},
    }
    realtime[field] = value

    current, reason = dashboard.davis_realsense_mapping_context(realtime, 0.1)

    assert current is False
    assert reason == expected_reason


def test_mapping_loss_clears_roi_boundary_contact_and_recent_motion(monkeypatch):
    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **kwargs):
            self.roi_xywh = kwargs.get("roi_xywh")
            self.configured_roi = kwargs.get("roi_xywh")

        def start(self):
            return {}

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    analyzer = motion_gui.DavisMotionAnalyzer(roi_xywh=[40, 30, 100, 160])
    analyzer.set_boundary([[40, 30], [140, 30], [140, 190], [40, 190]])
    analyzer.set_contact_band([[50, 90], [130, 90], [130, 120], [50, 120]])
    analyzer.events_arr = np.asarray([[1.0, 50.0, 100.0, 1.0]])
    analyzer.motion_estimator.cumulative_dx_px = 8.0

    assert analyzer.clear_object_context() is True

    assert analyzer.roi_source == "full_frame"
    assert analyzer.roi_xywh == [0, 0, 346, 260]
    assert analyzer.boundary_source == "full_frame_boundary"
    assert analyzer.contact_band_polygon is None
    assert analyzer.events_arr.shape == (0, 4)
    assert analyzer.motion_estimator.cumulative_dx_px == 0.0


def test_object_roi_reacquisition_discards_full_frame_baseline_once(monkeypatch):
    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **kwargs):
            self.roi_xywh = kwargs.get("roi_xywh")
            self.configured_roi = kwargs.get("roi_xywh")

        def start(self):
            return {}

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    analyzer = motion_gui.DavisMotionAnalyzer()
    analyzer.baseline_metrics = {"event_rate": 2000.0}
    analyzer.events_arr = np.asarray([[1.0, 50.0, 100.0, 1.0]])
    analyzer.motion_estimator.cumulative_dx_px = 8.0

    analyzer.set_roi([40, 30, 100, 160], source="realsense_bbox_mapped")

    assert analyzer.baseline_metrics is None
    assert analyzer.events_arr.shape == (0, 4)
    assert analyzer.motion_estimator.cumulative_dx_px == 0.0

    analyzer.events_arr = np.asarray([[2.0, 60.0, 100.0, 1.0]])
    analyzer.motion_estimator.cumulative_dx_px = 2.0
    analyzer.set_roi([41, 30, 100, 160], source="realsense_bbox_mapped")
    assert analyzer.events_arr.shape == (1, 4)
    assert analyzer.motion_estimator.cumulative_dx_px == 2.0


def test_roi_source_or_material_geometry_change_invalidates_baseline(monkeypatch):
    from grasp_project.davis import motion_gui

    class MonitorStub:
        def __init__(self, **kwargs):
            self.roi_xywh = kwargs.get("roi_xywh")
            self.configured_roi = kwargs.get("roi_xywh")

        def start(self):
            return {}

    monkeypatch.setattr(motion_gui, "EventPostGraspMonitor", MonitorStub)
    analyzer = motion_gui.DavisMotionAnalyzer(roi_xywh=[40, 30, 100, 160])
    analyzer.baseline_metrics = {"event_rate": 10.0}
    analyzer.baseline_roi_context_revision = analyzer.roi_context_revision
    analyzer.events_arr = np.asarray([[1.0, 50.0, 100.0, 1.0]])

    analyzer.set_roi(
        [40, 30, 100, 160],
        source="realsense_contact_projected",
        projection_metadata={
            "alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
            "reprojection_error_px": 0.5,
        },
    )

    assert analyzer.baseline_metrics is None
    assert analyzer.baseline_roi_context_revision is None
    assert analyzer.events_arr.shape == (0, 4)

    analyzer.baseline_metrics = {"event_rate": 10.0}
    analyzer.baseline_roi_context_revision = analyzer.roi_context_revision
    analyzer.events_arr = np.asarray([[2.0, 60.0, 100.0, 1.0]])
    analyzer.set_roi(
        [41, 30, 100, 160],
        source="realsense_contact_projected",
        projection_metadata={
            "alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
            "reprojection_error_px": 0.5,
        },
    )
    assert analyzer.baseline_metrics is not None
    assert analyzer.events_arr.shape == (1, 4)

    analyzer.set_roi(
        [90, 30, 100, 160],
        source="realsense_contact_projected",
        projection_metadata={
            "alignment_status": "CALIBRATED_2D_REFERENCE_PLANE",
            "reprojection_error_px": 0.5,
        },
    )
    assert analyzer.baseline_metrics is None
    assert analyzer.events_arr.shape == (0, 4)


@pytest.mark.parametrize(
    ("state", "extra", "expected"),
    [
        ("APPROACH_TRACKING", {}, "PRE_CLOSE"),
        ("READY_TO_CLOSE", {}, "PRE_CLOSE"),
        ("POST_GRASP_EVALUATION", {}, "POST_CLOSE_STABILITY"),
        ("APPROACH_TRACKING", {"gripper_closed_confirmed": True}, "POST_CLOSE_STABILITY"),
        ("READY_TO_LIFT", {}, "LIFT_CHECK"),
    ],
)
def test_davis_stage_is_derived_from_realsense_state(state, extra, expected):
    assert dashboard.davis_stage_from_realsense({"state": state, **extra}) == expected


def test_oriented_bottle_boundary_uses_same_uncalibrated_mapping():
    realtime = {
        "live_rgb_shape": [480, 640, 3],
        "bottle_geometry": {"oriented_box_points": [[200, 100], [300, 100], [300, 400], [200, 400]]},
    }
    boundary, reason = dashboard.map_realsense_boundary_to_davis(realtime)
    assert boundary.shape == (4, 2)
    assert np.all(boundary[:, 0] >= 0) and np.all(boundary[:, 0] < dashboard.DAVIS_ROI_FRAME_WIDTH)
    assert np.all(boundary[:, 1] >= 0) and np.all(boundary[:, 1] < dashboard.DAVIS_ROI_FRAME_HEIGHT)
    assert "uncalibrated" in reason


def test_current_grasp_band_uses_same_uncalibrated_mapping():
    realtime = {
        "live_rgb_shape": [480, 640, 3],
        "target_clamp": {"clamp_band_polygon": [[220, 210], [420, 210], [420, 270], [220, 270]]},
    }
    band, reason = dashboard.map_realsense_grasp_band_to_davis(realtime)

    assert band.shape == (4, 2)
    assert np.all(band[:, 0] >= 0) and np.all(band[:, 0] < dashboard.DAVIS_ROI_FRAME_WIDTH)
    assert np.all(band[:, 1] >= 0) and np.all(band[:, 1] < dashboard.DAVIS_ROI_FRAME_HEIGHT)
    assert "uncalibrated" in reason


def test_compact_status_preserves_current_clamp_polygon_for_davis_mapping():
    from grasp_project.realsense.realtime_runner import compact_live_status

    polygon = [[220, 210], [420, 210], [420, 270], [220, 270]]
    compact = compact_live_status({
        "state": "APPROACH_TRACKING",
        "target_status": "TRACKING",
        "tracking_good": True,
        "bottle_geometry": {"bbox_xyxy": [200, 100, 440, 420]},
        "target_clamp": {
            "clamp_source": "locked_target",
            "clamp_band_polygon": polygon,
        },
    })

    assert compact["target_clamp"]["clamp_band_polygon"] == polygon
    band, reason = dashboard.map_realsense_grasp_band_to_davis(compact)
    assert band is not None
    assert "uncalibrated" in reason


def test_compact_status_publishes_rgbd_host_acquisition_timestamps(monkeypatch):
    from grasp_project.realsense import realtime_runner

    monkeypatch.setattr(realtime_runner, "FAST_LIVE_STATUS_COMPACT", True)

    compact = realtime_runner.compact_live_status({
        "timestamp": "2026-08-12T12:00:01Z",
        "realtime_pipeline_audit": {
            "rgb_host_timestamp": 1000.125,
            "depth_host_timestamp": 1000.120,
        },
    })

    assert compact["rgb_timestamp"] == 1000.125
    assert compact["depth_timestamp"] == 1000.120
    assert compact["realtime_pipeline_audit"]["rgb_host_timestamp"] == 1000.125


def test_first_bbox_prefers_tracking_box_and_converts_xyxy():
    realtime = {"bottle_geometry": {"contact_tracking_bbox_xyxy": [10.0, 20.0, 60.0, 120.0]}}
    xywh = dashboard._first_bbox_xywh(realtime)
    assert xywh == [10.0, 20.0, 50.0, 100.0]


def test_frame_shape_from_profile_env(monkeypatch):
    monkeypatch.setenv("REALSENSE_COLOR_PROFILE", "848x480x30")
    assert dashboard._realsense_frame_shape_hw() == (480.0, 848.0)
    monkeypatch.delenv("REALSENSE_COLOR_PROFILE", raising=False)
    monkeypatch.delenv("REALSENSE_DEPTH_PROFILE", raising=False)
    assert dashboard._realsense_frame_shape_hw() == (480.0, 640.0)


def test_mapped_roi_source_is_diagnostic_only_when_uncalibrated():
    assert "realsense_bbox_mapped" not in OBJECT_SPECIFIC_ROI_SOURCES
    assert "realsense_bbox_mapped" in UNCALIBRATED_ROI_SOURCES
    assert "realsense_contour_mapped" not in OBJECT_SPECIFIC_ROI_SOURCES
    assert "realsense_contour_mapped" in UNCALIBRATED_ROI_SOURCES
    assert "realsense_contact_band_mapped" not in OBJECT_SPECIFIC_ROI_SOURCES
    assert "realsense_contact_band_mapped" in UNCALIBRATED_ROI_SOURCES
    assert "held_preclose_motion_diagnostic" not in OBJECT_SPECIFIC_ROI_SOURCES
    assert "held_preclose_motion_diagnostic" in UNCALIBRATED_ROI_SOURCES
    # A clean local cluster can still be displayed in the approximate mapped
    # ROI, but it cannot authorize object-slip quality without calibration.
    rng = np.random.default_rng(0)
    xs = rng.uniform(150, 205, size=200)
    ys = rng.uniform(100, 150, size=200)
    ts = np.linspace(100.0, 100.09, 200)
    events = np.column_stack([ts, xs, ys, rng.integers(0, 2, 200)]).astype(np.float64)
    hull = motion_hull(events, [140, 90, 80, 75])
    gates = object_motion_validity(hull, "realsense_bbox_mapped", 200, 3.0, "INCIPIENT_SLIP", 0.8)
    assert gates["object_motion_valid"] is False
    assert gates["roi_object_specific"] is False
    assert gates["roi_projection_uncalibrated"] is True
    assert gates["object_specific_motion_confidence"] == "low"
    assert "not_object_specific" in gates["object_motion_reason"]


def test_event_status_distinguishes_diagnostic_roi_activity_from_global_activity():
    assert event_analysis_status_for_roi(20, 30, 30, "realsense_contact_band_mapped") == (
        "DIAGNOSTIC_ROI_EVENTS_AVAILABLE_NOT_OBJECT_ATTRIBUTED"
    )
    assert event_analysis_status_for_roi(20, 30, 30, "held_preclose_motion_diagnostic") == (
        "DIAGNOSTIC_ROI_EVENTS_AVAILABLE_NOT_OBJECT_ATTRIBUTED"
    )
    assert event_analysis_status_for_roi(20, 30, 30, "realsense_contact_projected") == (
        "OBJECT_ROI_EVENTS_AVAILABLE"
    )
    assert event_analysis_status_for_roi(20, 30, 30, "full_frame") == (
        "GLOBAL_EVENTS_AVAILABLE_NO_OBJECT_ROI"
    )


def test_mapped_roi_still_rejected_under_lighting_disturbance():
    # the plumbing fix must NOT let a global disturbance through as object motion
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, 345, size=400)
    ys = rng.uniform(0, 259, size=400)
    ts = np.linspace(100.0, 100.09, 400)
    events = np.column_stack([ts, xs, ys, rng.integers(0, 2, 400)]).astype(np.float64)
    hull = motion_hull(events, [0, 0, 346, 260])
    gates = object_motion_validity(hull, "realsense_bbox_mapped", 400, 3.0, "LIGHTING_CHANGE", 0.95)
    assert gates["object_motion_valid"] is False
    assert gates["motion_activity_label"] == "GLOBAL_ACTIVITY"


def _fresh_mapping_status():
    return {
        "writer_session_id": "writer-a",
        "frame_id": 42,
        "target_status": "TRACKING",
        "tracking_good": True,
        "identity_tracking_ok": True,
        "identity_measurement_current": True,
        "bottle_geometry_tracking_ok": True,
        "live_rgb_shape": [480, 640, 3],
        "bottle_geometry": {
            "bbox_xywh": [200, 80, 120, 320],
            "oriented_bbox": [[210, 80], [310, 80], [320, 400], [200, 400]],
        },
        "target_lock": {
            "lock_frame_id": 17,
            "target_status": "TRACKING",
            "attempt_id": "attempt-a",
            "locked_target": {
                "bottle_target_id": "bottle-02",
                "attempt_id": "attempt-a",
                "lock_frame_id": 17,
            },
        },
        "target_clamp": {
            "clamp_band_polygon": [[210, 210], [310, 210], [310, 260], [210, 260]],
        },
    }


def test_uncalibrated_grasp_band_focuses_motion_roi_without_upgrading_validity(monkeypatch):
    monkeypatch.setattr(dashboard, "DAVIS_HOMOGRAPHY_PATH", dashboard.ROOT / "config" / "missing-test-homography.json")
    realtime = _fresh_mapping_status()

    context, reason = dashboard.build_davis_target_context(realtime, 0.1, now=10.0)

    assert reason == "fresh_current_tracked_target"
    assert context["roi_source"] == "realsense_contact_band_mapped"
    assert context["roi_priority_selected"] == "uncalibrated_realsense_contact_motion_diagnostic_roi"
    assert context["contact_band_points"] is not None
    assert context["roi_xywh"][3] < 150
    gates = object_motion_validity(
        {"motion_hull_area_fraction": 0.05, "motion_hull_border_touch_ratio": 0.0},
        context["roi_source"],
        200,
        3.0,
        "STABLE",
        0.9,
    )
    assert gates["object_motion_valid"] is False
    assert gates["roi_object_specific"] is False


def test_post_grasp_handoff_requires_a_pre_close_capture():
    handoff = dashboard.DavisTargetHandoff(max_capture_age_sec=1.0)

    context, status = handoff.update(
        {**_fresh_mapping_status(), "post_grasp_active": True},
        0.1,
        "POST_CLOSE_STABILITY",
        now=10.0,
    )

    assert context is None
    assert status["davis_target_context_valid"] is False
    assert status["davis_target_lock_active"] is False


def test_post_grasp_handoff_invalidates_when_locked_band_requires_relock():
    handoff = dashboard.DavisTargetHandoff(max_capture_age_sec=1.0)
    handoff.update(_fresh_mapping_status(), 0.1, "PRE_CLOSE", now=10.0)
    invalid = _fresh_mapping_status()
    invalid["grasp_band_relock_required"] = True

    context, status = handoff.update(
        invalid,
        0.1,
        "POST_CLOSE_STABILITY",
        now=10.2,
    )

    assert context is None
    assert status["davis_target_context_valid"] is False
    assert "locked_grasp_band_relock_required" in status["davis_target_context_reason"]


def test_pre_close_loss_holds_last_motion_roi_briefly_without_object_validity():
    handoff = dashboard.DavisTargetHandoff(
        max_capture_age_sec=1.0,
        preclose_diagnostic_hold_sec=5.0,
    )
    live, _ = handoff.update(_fresh_mapping_status(), 0.1, "PRE_CLOSE", now=10.0)
    lost = {
        "writer_session_id": "writer-a",
        "target_status": "LOST",
        "target_lock": {
            "lock_frame_id": 17,
            "attempt_id": "attempt-a",
            "bottle_target_id": "bottle-02",
        },
    }

    held, status = handoff.update(lost, 0.1, "PRE_CLOSE", now=12.0)

    assert held["roi_xywh"] == live["roi_xywh"]
    assert held["roi_source"] == "held_preclose_motion_diagnostic"
    assert status["davis_target_context_mode"] == "pre_close_held_target_diagnostic"
    assert status["davis_target_lock_active"] is False
    gates = object_motion_validity(
        {"motion_hull_area_fraction": 0.05, "motion_hull_border_touch_ratio": 0.0},
        held["roi_source"],
        200,
        3.0,
        "STABLE",
        0.9,
    )
    assert gates["object_motion_valid"] is False

    expired, expired_status = handoff.update(lost, 0.1, "PRE_CLOSE", now=15.1)
    assert expired is None
    assert expired_status["davis_target_context_mode"] == "none"


def test_pre_close_search_clears_held_motion_roi():
    handoff = dashboard.DavisTargetHandoff(preclose_diagnostic_hold_sec=5.0)
    handoff.update(_fresh_mapping_status(), 0.1, "PRE_CLOSE", now=10.0)

    context, status = handoff.update(
        {"writer_session_id": "writer-a", "target_status": "SEARCHING"},
        0.1,
        "PRE_CLOSE",
        now=10.1,
    )

    assert context is None
    assert handoff.last_live_context is None
    assert status["davis_target_context_mode"] == "none"


def test_post_grasp_handoff_freezes_pre_close_roi_boundary_and_grasp_band():
    handoff = dashboard.DavisTargetHandoff(max_capture_age_sec=1.0)
    live, live_status = handoff.update(_fresh_mapping_status(), 0.1, "PRE_CLOSE", now=10.0)
    lost = {
        "writer_session_id": "writer-a",
        "frame_id": 43,
        "state": "DO_NOT_LIFT",
        "post_grasp_active": True,
        "target_status": "LOST",
        "tracking_good": False,
        "target_lock": {
            "lock_frame_id": 17,
            "target_status": "LOST",
            "attempt_id": "attempt-a",
            "bottle_target_id": "bottle-02",
        },
    }

    locked, locked_status = handoff.update(lost, 0.1, "POST_CLOSE_STABILITY", now=10.2)

    assert live_status["davis_target_context_mode"] == "live_realsense_mapping"
    assert locked["roi_xywh"] == live["roi_xywh"]
    assert locked["boundary_points"] == live["boundary_points"]
    assert locked["contact_band_points"] == live["contact_band_points"]
    assert locked_status["davis_target_context_mode"] == "post_grasp_locked_mapping"
    assert locked_status["davis_target_lock_active"] is True
    assert locked_status["davis_target_lock_source_attempt_id"] == "attempt-a"
    assert locked_status["davis_target_lock_source_bottle_target_id"] == "bottle-02"
    assert "frozen lock-time ROI" in locked_status["davis_target_context_reason"]
    assert locked_status["davis_target_mapping_physically_validated"] is False


@pytest.mark.parametrize(
    "post_status,age,reason_fragment",
    [
        (
            {
                "writer_session_id": "writer-a",
                "target_lock": {
                    "lock_frame_id": 17,
                    "attempt_id": "attempt-a",
                    "bottle_target_id": "bottle-02",
                },
                "post_grasp_active": True,
            },
            1.01,
            "invalidated",
        ),
        (
            {
                "writer_session_id": "writer-b",
                "target_lock": {
                    "lock_frame_id": 17,
                    "attempt_id": "attempt-a",
                    "bottle_target_id": "bottle-02",
                },
                "post_grasp_active": True,
            },
            0.1,
            "identity_mismatch",
        ),
        (
            {
                "writer_session_id": "writer-a",
                "target_lock": {
                    "lock_frame_id": 18,
                    "attempt_id": "attempt-a",
                    "bottle_target_id": "bottle-02",
                },
                "post_grasp_active": True,
            },
            0.1,
            "identity_mismatch",
        ),
        (
            {
                "writer_session_id": "writer-a",
                "target_lock": {
                    "lock_frame_id": 17,
                    "attempt_id": "attempt-b",
                    "bottle_target_id": "bottle-02",
                },
                "post_grasp_active": True,
            },
            0.1,
            "source_attempt_id",
        ),
        (
            {
                "writer_session_id": "writer-a",
                "target_lock": {
                    "lock_frame_id": 17,
                    "attempt_id": "attempt-a",
                    "bottle_target_id": "bottle-03",
                },
                "post_grasp_active": True,
            },
            0.1,
            "source_bottle_target_id",
        ),
    ],
)
def test_post_grasp_handoff_fails_closed_for_stale_or_changed_identity(post_status, age, reason_fragment):
    handoff = dashboard.DavisTargetHandoff(max_capture_age_sec=1.0)
    handoff.update(_fresh_mapping_status(), 0.1, "PRE_CLOSE", now=10.0)

    context, status = handoff.update(post_status, age, "POST_CLOSE_STABILITY", now=10.2)

    assert context is None
    assert status["davis_target_context_valid"] is False
    assert reason_fragment in status["davis_target_context_reason"]


def test_post_grasp_handoff_requires_complete_pre_close_target_identity():
    handoff = dashboard.DavisTargetHandoff(max_capture_age_sec=1.0)
    incomplete = _fresh_mapping_status()
    incomplete["target_lock"].pop("attempt_id")
    incomplete["target_lock"]["locked_target"].pop("attempt_id")

    live, _status = handoff.update(incomplete, 0.1, "PRE_CLOSE", now=10.0)
    context, status = handoff.update(
        incomplete,
        0.1,
        "POST_CLOSE_STABILITY",
        now=10.2,
    )

    assert live is not None
    assert context is None
    assert status["davis_target_context_valid"] is False
    assert "source_attempt_id" in status["davis_target_context_reason"]


def test_release_clears_handoff_and_new_grasp_requires_fresh_pre_close_mapping():
    handoff = dashboard.DavisTargetHandoff(max_capture_age_sec=1.0)
    handoff.update(_fresh_mapping_status(), 0.1, "PRE_CLOSE", now=10.0)
    handoff.update(
        {
            "writer_session_id": "writer-a",
            "target_lock": {
                "lock_frame_id": 17,
                "attempt_id": "attempt-a",
                "bottle_target_id": "bottle-02",
            },
        },
        0.1,
        "POST_CLOSE_STABILITY",
        now=10.1,
    )

    released, released_status = handoff.update(
        {"writer_session_id": "writer-a", "target_status": "LOST"},
        0.1,
        "PRE_CLOSE",
        now=10.2,
    )
    repeated, repeated_status = handoff.update(
        {"writer_session_id": "writer-a", "target_lock": {"lock_frame_id": 17}},
        0.1,
        "POST_CLOSE_STABILITY",
        now=10.3,
    )

    assert released is None
    assert released_status["davis_target_lock_active"] is False
    assert repeated is None
    assert repeated_status["davis_target_lock_active"] is False


def test_mapper_uses_inverse_validated_homography(monkeypatch, tmp_path):
    path = tmp_path / "davis_to_realsense_homography.json"
    path.write_text(
        json.dumps(
            {
                "homography": [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]],
                "reprojection_error_px": 0.25,
                "transform_direction": "davis_to_realsense",
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )
    monkeypatch.setattr(dashboard, "DAVIS_HOMOGRAPHY_PATH", path)
    monkeypatch.setattr(dashboard, "DAVIS_ROI_PADDING", 0.0)
    realtime = {
        "live_rgb_shape": [480, 640, 3],
        "bottle_geometry": {
            "bbox_xywh": [30.0, 40.0, 20.0, 20.0],
            "oriented_bbox": [[30.0, 40.0], [50.0, 40.0], [50.0, 60.0], [30.0, 60.0]],
        },
    }

    roi, reason = dashboard.map_realsense_bbox_to_davis_roi(realtime)
    boundary, boundary_reason = dashboard.map_realsense_boundary_to_davis(realtime)
    alignment = dashboard._davis_alignment_status(realtime)

    assert roi == [10, 10, 10, 10]
    assert "inverse of measured" in reason
    assert np.allclose(boundary, [[10, 10], [20, 10], [20, 20], [10, 20]])
    assert "CALIBRATED_2D_REFERENCE_PLANE" in boundary_reason
    assert alignment["davis_realsense_alignment_status"] == "calibrated_2d_reference_plane"


def test_calibrated_davis_roi_is_conditionally_projected_onto_realsense_overlay(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "davis_to_realsense_homography.json"
    path.write_text(
        json.dumps(
            {
                "homography": [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]],
                "reprojection_error_px": 0.25,
                "transform_direction": "davis_to_realsense",
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )
    monkeypatch.setattr(dashboard, "DAVIS_HOMOGRAPHY_PATH", path)
    overlay = dashboard.davis_roi_on_realsense_overlay(
        {
            "roi_source": "aps_target",
            "roi_xywh": [10, 10, 10, 10],
            "davis_target_context_valid": True,
        },
        {"live_rgb_shape": [480, 640, 3]},
    )

    assert overlay["valid"] is True
    assert overlay["polygon_px"] == [
        [30.0, 40.0],
        [50.0, 40.0],
        [50.0, 60.0],
        [30.0, 60.0],
    ]
    assert overlay["realsense_frame_size"] == [640, 480]
    assert overlay["physically_validated"] is False

    hidden = dashboard.davis_roi_on_realsense_overlay(
        {"roi_source": "full_frame", "roi_xywh": [0, 0, 346, 260]},
        {"live_rgb_shape": [480, 640, 3]},
    )
    assert hidden["valid"] is False
    assert hidden["reason"] == "davis_roi_not_object_specific"


def test_mapper_rejects_homography_without_error_metric(monkeypatch, tmp_path):
    path = tmp_path / "davis_to_realsense_homography.json"
    path.write_text(json.dumps({"homography": np.eye(3).tolist()}))
    monkeypatch.setattr(dashboard, "DAVIS_HOMOGRAPHY_PATH", path)

    realtime = _fresh_mapping_status()
    _roi, reason = dashboard.map_realsense_bbox_to_davis_roi(realtime)
    alignment = dashboard._davis_alignment_status(realtime)

    assert reason.startswith("approximate (uncalibrated)")
    assert "missing_reprojection_error" in reason
    assert alignment["davis_realsense_alignment_status"] == "invalid_calibration"


@pytest.mark.parametrize(
    "bbox",
    [
        [0.0, 0.0, 0.0, 20.0],
        [0.0, 0.0, float("nan"), 20.0],
        [0.0, 0.0, 20.0, float("inf")],
    ],
)
def test_mapper_rejects_degenerate_or_nonfinite_bbox(bbox):
    roi, reason = dashboard.map_realsense_bbox_to_davis_roi(
        {"bottle_geometry": {"bbox_xywh": bbox}}
    )

    assert roi is None
    assert "nonfinite_or_degenerate" in reason


def test_compact_status_preserves_native_bottle_geometry_for_davis_handoff():
    from grasp_project.realsense.realtime_runner import compact_live_status

    compact = compact_live_status(_fresh_mapping_status())

    assert compact["live_rgb_shape"] == [480, 640, 3]
    assert compact["bottle_geometry"]["bbox_xywh"] == [200, 80, 120, 320]
    assert compact["bottle_geometry"]["oriented_bbox"] == [
        [210, 80],
        [310, 80],
        [320, 400],
        [200, 400],
    ]
    assert compact["locked_bottle_target_id"] == "bottle-02"
    assert compact["lock_attempt_id"] == "attempt-a"
    assert compact["target_lock"]["bottle_target_id"] == "bottle-02"
    assert compact["target_lock"]["attempt_id"] == "attempt-a"

    context, reason = dashboard.build_davis_target_context(compact, 0.1, now=10.0)
    assert context is not None, reason
    assert context["source_bottle_target_id"] == "bottle-02"
    assert context["source_attempt_id"] == "attempt-a"


def test_dashboard_summary_preserves_structured_target_handoff_honesty():
    summary = dashboard.summarize_davis(
        {
            "roi_source": "realsense_bbox_mapped",
            "roi_projection_uncalibrated": True,
            "object_specific_motion_confidence": "approximate_uncalibrated",
            "davis_target_context_mode": "post_grasp_locked_mapping",
            "davis_target_context_valid": True,
            "davis_target_lock_active": True,
            "davis_target_lock_age_ms": 250.0,
            "davis_target_lock_source_attempt_id": "attempt-a",
            "davis_target_lock_source_bottle_target_id": "bottle-02",
            "davis_target_lock_provenance_valid": True,
            "davis_target_mapping_mode": "fixed_linear_side_by_side",
            "davis_target_alignment_status": "APPROXIMATE_UNCALIBRATED",
            "davis_target_mapping_physically_validated": False,
            "raw_window_event_count": 321,
            "object_roi_window_event_count": 42,
            "event_analysis_status": "OBJECT_ROI_EVENTS_AVAILABLE",
            "event_timestamp_status": "CURRENT",
            "event_timestamp_lag_ms": 7.5,
        }
    )

    assert summary["davis_target_context_mode"] == "post_grasp_locked_mapping"
    assert summary["davis_target_lock_active"] is True
    assert summary["davis_target_lock_source_attempt_id"] == "attempt-a"
    assert summary["davis_target_lock_source_bottle_target_id"] == "bottle-02"
    assert summary["davis_target_lock_provenance_valid"] is True
    assert summary["roi_projection_uncalibrated"] is True
    assert summary["object_specific_motion_confidence"] == "approximate_uncalibrated"
    assert summary["davis_target_mapping_physically_validated"] is False
    assert summary["raw_window_event_count"] == 321
    assert summary["object_roi_window_event_count"] == 42
    assert summary["event_analysis_status"] == "OBJECT_ROI_EVENTS_AVAILABLE"
    assert summary["event_timestamp_status"] == "CURRENT"

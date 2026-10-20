import numpy as np

from grasp_project.davis.boundary_event_search import search_boundary_events
from grasp_project.davis.motion_gui import fuse_boundary_motion_evidence, object_motion_validity


BOUNDARY = np.asarray([[40, 30], [80, 30], [80, 90], [40, 90]], dtype=float)


def _events_for_tracks(tracks, *, start=1.0, window=0.12, per_bin=10):
    rows = []
    for bin_index, points in enumerate(tracks):
        t0 = start + bin_index * window / len(tracks)
        for point_index, (x, y) in enumerate(points):
            for repeat in range(per_bin):
                rows.append([t0 + (repeat + 1) * window / len(tracks) / (per_bin + 1), x, y, (point_index + repeat) % 2])
    return np.asarray(rows, dtype=float)


def test_downward_bottom_drift_is_slip_only():
    tracks = [[(50, 88 + step), (70, 88 + step)] for step in (0, 1, 2, 4)]
    field = search_boundary_events(BOUNDARY, _events_for_tracks(tracks), 0.12, now=1.12, object_motion_valid=True)
    assert field.slip_cue >= 50
    assert field.rotation_cue < 20
    assert field.vibration_cue < 20


def test_tangential_circulation_is_rotation_only():
    # Clockwise tracks follow each local boundary tangent with steady speed.
    tracks = []
    for step in (0, 1, 2, 3):
        tracks.append([(46 + step, 30), (80, 46 + step), (74 - step, 90), (40, 74 - step)])
    field = search_boundary_events(BOUNDARY, _events_for_tracks(tracks), 0.12, now=1.12, object_motion_valid=True)
    assert field.rotation_cue >= 20
    assert field.slip_cue < 20
    assert field.vibration_cue < 20


def test_oscillation_is_vibration_only():
    tracks = [[(40 + dx, 55), (80 - dx, 65)] for dx in (0, 3, -3, 3)]
    field = search_boundary_events(BOUNDARY, _events_for_tracks(tracks), 0.12, now=1.12, object_motion_valid=True)
    assert field.vibration_cue >= 50
    assert field.slip_cue < 20
    assert field.rotation_cue < 20


def test_grasp_band_reports_only_contact_neighborhood_events():
    contact_band = np.asarray([[35, 48], [85, 48], [85, 72], [35, 72]], dtype=float)
    tracks = [[(40 + dx, 55), (80 - dx, 65)] for dx in (0, 2, -2, 2)]
    field = search_boundary_events(
        BOUNDARY,
        _events_for_tracks(tracks),
        0.12,
        now=1.12,
        object_motion_valid=True,
        contact_band_polygon=contact_band,
    )

    assert 0 < field.contact_band_count < len(field.bands)
    assert field.contact_band_active_count > 0
    assert field.contact_band_event_count > 0
    assert field.contact_band_measurement_status == "ESTIMATED_APPROXIMATE_UNCALIBRATED"


def test_uniform_full_frame_activity_is_suppressed_by_existing_gate():
    rng = np.random.default_rng(5)
    events = np.column_stack([
        np.linspace(1.0, 1.12, 400),
        rng.uniform(0, 346, 400),
        rng.uniform(0, 260, 400),
        rng.integers(0, 2, 400),
    ])
    hull = {
        "motion_hull_valid": True,
        "motion_hull_reason": "ok",
        "motion_hull_event_count": 400,
        "motion_hull_area_fraction": 0.8,
        "motion_hull_border_touch_ratio": 0.8,
    }
    gates = object_motion_validity(hull, "realsense_bbox_mapped", 400, 1.0, "LIGHTING_CHANGE", 0.95)
    field = search_boundary_events(
        BOUNDARY,
        events,
        0.12,
        now=1.12,
        object_motion_valid=gates["object_motion_valid"],
        suppression_reason=gates["object_motion_reason"],
    )
    assert field.event_count == field.assigned_event_count
    assert field.assigned_event_count == sum(band.event_count for band in field.bands)
    assert field.raw_window_event_count == len(events)
    assert field.assigned_event_count < field.raw_window_event_count
    assert not field.object_motion_valid
    assert field.rotation_cue == field.slip_cue == field.vibration_cue == 0
    assert field.incipient_slip_precursor_score == 0
    assert field.contact_band_measurement_status == "SUPPRESSED_BY_OBJECT_MOTION_VALIDITY"


def test_localized_growing_burst_is_an_incipient_precursor():
    rows = []
    counts = (2, 5, 10, 18)
    for bin_index, count in enumerate(counts):
        t0 = 1.0 + bin_index * 0.03
        for event_index in range(count):
            rows.append([t0 + (event_index + 1) * 0.03 / (count + 1), 40.0, 55.0 + 0.03 * event_index, event_index % 2])
    field = search_boundary_events(BOUNDARY, np.asarray(rows), 0.12, now=1.12, object_motion_valid=True)
    assert field.incipient_slip_precursor_score >= 50
    assert field.slip_cue < 20


def test_valid_boundary_cues_add_to_existing_motion_scores_only_after_gate():
    base = {"raw_motion_class": "STABLE", "motion_class": "STABLE", "stability_score": 90.0}
    fused = fuse_boundary_motion_evidence(base, {"object_motion_valid": True, "rotation_cue": 70.0})
    assert fused["rotation_risk_score"] == 70.0
    assert fused["raw_motion_class"] == "SPIN_OR_ROTATION"
    suppressed = fuse_boundary_motion_evidence(base, {"object_motion_valid": False, "rotation_cue": 100.0})
    assert "rotation_risk_score" not in suppressed

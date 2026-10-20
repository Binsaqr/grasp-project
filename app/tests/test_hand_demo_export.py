"""Camera-free checks for the bounded hand-operated demo export bundle."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from grasp_project.dashboard import web_dashboard


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _identity(*, davis=False, attempt="attempt-1", target="bottle-1"):
    if davis:
        return {
            "davis_target_lock_source_writer_session_id": "writer-1",
            "davis_target_lock_source_lock_frame_id": 42,
            "davis_target_lock_source_attempt_id": attempt,
            "davis_target_lock_source_bottle_target_id": target,
        }
    return {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "attempt_id": attempt,
        "bottle_target_id": target,
    }


def _write_image_provenance(path, **identity):
    _write_json(
        path.with_name(path.name + ".json"),
        {
            "artifact_name": path.name,
            "artifact_mtime_ns": path.stat().st_mtime_ns,
            **(_identity() | identity),
        },
    )


def test_hand_demo_export_writes_exact_bundle_from_current_evidence(monkeypatch, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    monkeypatch.setattr(
        web_dashboard,
        "davis_analyzer",
        SimpleNamespace(close_action_id="action-1", close_action_started_at=1234.5),
    )

    (output / "latest_overlay_live.jpg").write_bytes(b"locked-band-jpeg")
    (output / "locked_band_snapshot.jpg").write_bytes(b"lock-frame-jpeg")
    _write_image_provenance(output / "locked_band_snapshot.jpg")
    _write_json(
        output / "latest_realtime_status.json",
        {
            "writer_session_id": "writer-1",
            "frame_id": 99,
            "overlay_frame_id": 99,
            "overlay_current_path": "output/latest_overlay_live.jpg",
            "overlay_stale": False,
            "dashboard_overlay_cache_buster": (
                output / "latest_overlay_live.jpg"
            ).stat().st_mtime,
            "bottle_detected": True,
            "target_status": "TRACKING",
            "locked_bottle_target_id": "bottle-1",
            "lock_attempt_id": "attempt-1",
            "locked_body_fraction": 0.52,
            "distance_to_bottle_mm": 360.0,
            "distance_measurement_valid": True,
            "grasp_band_lock_confidence": 91.0,
            "grasp_band_projected": True,
            "grasp_band_current": True,
            "grasp_band_stale": False,
            "grasp_band_relock_required": False,
            "decision": "CONTINUE_APPROACH",
            "target_lock": {
                "target_status": "TRACKING",
                "lock_frame_id": 42,
                "grasp_band_projected": True,
                "grasp_band_stale": False,
                "grasp_band_relock_required": False,
            },
        },
    )
    _write_json(
        output / "locked_target.json",
        {
            **_identity(),
            "target_status": "TRACKING",
            "locked_target": {
                "bottle_target_id": "bottle-1",
                "lock_frame_id": 42,
                "locked_body_fraction": 0.52,
            },
        },
    )
    _write_json(
        output / "realtime_wrench_matrix_status.json",
        {
            **_identity(),
            "quality_matrix_status": "updated",
            "quality_compute_id": "quality-42",
            "quality_label": "GOOD",
            "epsilon": 0.2,
        },
    )
    _write_json(
        output / "live_quality" / "latest_status.json",
        {
            "baseline": _identity(),
            "analytic_quality_score": 77.0,
            "analytic_quality_label": "GOOD",
            "live_quality_score": 70.0,
            "live_quality_label": "OK",
            "trend": "stable",
            "primary_risk": "none",
            "decision": "HOLD",
            "davis_motion_used_for_live_quality": True,
            "live_quality_davis_penalty": 7.0,
            "davis_adjustment_points": -7.0,
            "davis_adjustment_reason": "object_motion_valid",
        },
    )
    _write_json(
        output / "davis" / "latest_motion_report.json",
        {
            **_identity(davis=True),
            "post_grasp_monitoring_action_id": "action-1",
            "post_grasp_monitoring_started_at": 1234.5,
            "event_health_valid": True,
            "event_count": 500,
            "event_rate": 2500.0,
            "object_motion_valid": True,
            "motion_class": "GROSS_SLIP",
            "downward_slip_px": 2.0,
            "rotation_delta_deg": 3.5,
            "vibration_amplitude_px_rms": 1.25,
            "disturbance_type": "OBJECT_MOTION",
        },
    )
    _write_json(
        output / "event_postgrasp_report.json",
        {
            "post_grasp_decision": "REGRIP",
            "post_grasp_monitoring_action_id": "action-1",
            "post_grasp_monitoring_started_at": 1234.5,
            **_identity(davis=True),
        },
    )

    target_trace = output / "target_lock_trace.jsonl"
    target_trace.write_text(
        "".join(
            json.dumps(
                {
                    **_identity(),
                    "frame_id": frame,
                    "distance_to_bottle_mm": 500.0 - frame,
                    "distance_measurement_valid": True,
                    "grasp_band_drift_px": frame / 10.0,
                    "grasp_band_visibility_ratio": 0.9 - frame / 1000.0,
                    "guidance_valid": frame % 2 == 0,
                }
            )
            + "\n"
            for frame in range(150)
        )
    )
    davis_trace = output / "davis" / "motion_timeline.jsonl"
    davis_trace.parent.mkdir(parents=True, exist_ok=True)
    davis_trace.write_text(
        "".join(
            json.dumps(
                {
                    "sample": sample,
                    "post_grasp_monitoring_action_id": "action-1",
                    "post_grasp_monitoring_started_at": 1234.5,
                    **_identity(davis=True),
                }
            )
            + "\n"
            for sample in range(130)
        )
    )
    stage_dir = output / "hand_demo_stage_screenshots"
    stage_dir.mkdir()
    (stage_dir / "operator_start.png").write_bytes(b"operator-png")
    (stage_dir / "post_grasp.png").write_bytes(b"post-grasp-png")
    _write_image_provenance(stage_dir / "operator_start.png")
    _write_image_provenance(stage_dir / "post_grasp.png")

    manifest = web_dashboard.record_demo_snapshot()
    run_dir = Path(manifest["run_dir"])

    for name in (
        "demo_manifest.json",
        "manifest.json",
        "locked_band_snapshot.jpg",
        "projected_band_snapshot.jpg",
        "approach_timeline.jsonl",
        "davis_event_timeline.jsonl",
        "post_grasp_report.json",
        "final_summary.json",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "locked_band_snapshot.jpg").read_bytes() == b"lock-frame-jpeg"
    assert (run_dir / "projected_band_snapshot.jpg").read_bytes() == b"locked-band-jpeg"
    assert len((run_dir / "approach_timeline.jsonl").read_text().splitlines()) == 120
    assert json.loads((run_dir / "approach_timeline.jsonl").read_text().splitlines()[0])["frame_id"] == 30
    assert len((run_dir / "davis_event_timeline.jsonl").read_text().splitlines()) == 120
    assert json.loads((run_dir / "post_grasp_report.json").read_text())[
        "post_grasp_decision"
    ] == "REGRIP"
    assert (run_dir / "screenshots" / "operator_start.png").read_bytes() == b"operator-png"
    assert (run_dir / "screenshots" / "post_grasp.png").read_bytes() == b"post-grasp-png"
    assert not (run_dir / "screenshots" / "locked_band.png").exists()

    summary = json.loads((run_dir / "final_summary.json").read_text())
    assert summary["bottle_detected"] is True
    assert summary["band_locked"] is True
    assert summary["band_lock_confidence"] == 91.0
    assert summary["locked_band"]["body_fraction"] == 0.52
    assert summary["analytic_quality"]["quality_compute_id"] == "quality-42"
    assert summary["approach_distance_range_mm"] == {
        "min": 351.0,
        "max": 470.0,
        "sample_count": 121,
    }
    assert summary["band_drift_max_px"] == 14.9
    assert summary["max_band_drift_px"] == 14.9
    assert summary["min_band_visibility_ratio"] == 0.751
    assert summary["distance_start_mm"] == 470.0
    assert summary["distance_end_mm"] == 360.0
    assert summary["guidance_valid_rate"] == 0.5
    assert summary["initial_analytic_quality"] == summary["analytic_quality"]
    assert summary["object_motion_valid"] is True
    assert summary["davis_object_motion_valid"] is True
    assert summary["davis_motion_class"] == "GROSS_SLIP"
    assert summary["slip_like_motion"]["detected"] is True
    assert summary["rotation"] == {
        "valid": True,
        "delta_deg": 3.5,
        "motion_attribution": "OBJECT_SPECIFIC",
        "object_specific": True,
    }
    assert summary["roll"] == {
        "valid": False,
        "delta_deg": None,
        "status": "NOT_ESTIMATED_NO_3D_OBJECT_ORIENTATION",
        "exclusion_reason": "No calibrated 3D object orientation source is available.",
    }
    assert summary["vibration"] == {
        "valid": True,
        "rms_px": 1.25,
        "motion_attribution": "OBJECT_SPECIFIC",
        "object_specific": True,
    }
    assert summary["final_live_quality"]["score"] == 70.0
    assert summary["final_live_quality"]["davis_adjustment_points"] == -7.0
    assert summary["davis_live_quality_adjustment"] == {
        "status": "INCLUDED",
        "points": -7.0,
        "reason": "object_motion_valid",
        "epsilon_recomputed": False,
    }
    assert summary["live_quality_final"] == summary["final_live_quality"]
    assert summary["final_decision"] == "HOLD"

    assert manifest["schema_version"] == "hand_demo_bundle_v1"
    assert manifest["approach_timeline"]["captured_line_count"] == 120
    assert manifest["davis_event_timeline"]["captured_line_count"] == 120
    assert manifest["approach_timeline"]["identity_validation"]["valid"] is True
    assert manifest["davis_event_timeline"]["identity_validation"]["valid"] is True
    assert manifest["projected_band_snapshot"]["available"] is True
    assert [item["name"] for item in manifest["stage_screenshots"]] == [
        "operator_start.png",
        "post_grasp.png",
    ]
    assert manifest["missing_stage_screenshots"] == [
        "locked_band.png",
        "approach.png",
        "davis_closeup.png",
    ]
    assert json.loads((run_dir / "demo_manifest.json").read_text()) == manifest
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest


def test_hand_demo_export_lists_missing_images_and_never_fabricates_them(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)

    manifest = web_dashboard.record_demo_snapshot()
    run_dir = Path(manifest["run_dir"])

    assert manifest["locked_band_snapshot"]["available"] is False
    assert not (run_dir / "locked_band_snapshot.jpg").exists()
    assert manifest["stage_screenshots"] == []
    assert manifest["missing_stage_screenshots"] == list(
        web_dashboard.DEMO_STAGE_SCREENSHOTS
    )
    assert list((run_dir / "screenshots").iterdir()) == []
    assert (run_dir / "approach_timeline.jsonl").read_bytes() == b""
    assert (run_dir / "davis_event_timeline.jsonl").read_bytes() == b""
    unavailable = json.loads((run_dir / "post_grasp_report.json").read_text())
    assert unavailable["status"] == "UNAVAILABLE"
    assert "current lock identity" in unavailable["reason"]
    summary = json.loads((run_dir / "final_summary.json").read_text())
    assert summary["bottle_detected"] is None
    assert summary["object_motion_valid"] is None
    assert summary["slip_like_motion"]["detected"] is None
    assert any("No locked-band image" in item for item in summary["limitations"])
    assert any("Stage screenshots missing" in item for item in summary["limitations"])


def test_hand_demo_export_excludes_mismatched_structured_evidence_and_rows(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    _write_json(
        output / "latest_realtime_status.json",
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING",
            "decision": "CONTINUE_APPROACH",
        },
    )
    _write_json(
        output / "realtime_wrench_matrix_status.json",
        {
            **_identity(),
            "writer_session_id": "old-writer",
            "quality_compute_id": "stale-quality",
            "quality_label": "GOOD",
            "epsilon": 0.9,
        },
    )
    _write_json(output / "realtime_grasp_hulls.json", {"status": "old-unscoped"})
    _write_json(
        output / "live_quality" / "latest_status.json",
        {
            "baseline": {**_identity(), "lock_frame_id": 41},
            "live_quality_score": 99.0,
            "decision": "READY_TO_LIFT",
        },
    )
    _write_json(
        output / "davis" / "latest_motion_report.json",
        {
            **_identity(davis=True),
            "davis_target_lock_source_attempt_id": "old-attempt",
            "object_motion_valid": True,
            "motion_class": "GROSS_SLIP",
        },
    )
    (output / "target_lock_trace.jsonl").write_text(
        json.dumps({**_identity(), "frame_id": 1, "distance_to_bottle_mm": 400, "distance_measurement_valid": True})
        + "\n"
        + json.dumps(
            {
                **_identity(),
                "writer_session_id": "old-writer",
                "frame_id": 2,
                "distance_to_bottle_mm": 100,
                "distance_measurement_valid": True,
            }
        )
        + "\n"
    )
    davis_timeline = output / "davis" / "motion_timeline.jsonl"
    davis_timeline.parent.mkdir(parents=True, exist_ok=True)
    davis_timeline.write_text(
        json.dumps({**_identity(davis=True), "sample": "current"})
        + "\n"
        + json.dumps(
            {
                **_identity(davis=True),
                "davis_target_lock_source_lock_frame_id": 41,
                "sample": "old",
            }
        )
        + "\n"
    )

    manifest = web_dashboard.record_demo_snapshot()
    run_dir = Path(manifest["run_dir"])
    summary = json.loads((run_dir / "final_summary.json").read_text())

    assert summary["analytic_quality"]["quality_compute_id"] is None
    assert summary["final_live_quality"]["score"] is None
    assert summary["object_motion_valid"] is None
    assert summary["final_decision"] == "CONTINUE_APPROACH"
    assert len((run_dir / "approach_timeline.jsonl").read_text().splitlines()) == 1
    assert len((run_dir / "davis_event_timeline.jsonl").read_text().splitlines()) == 1
    assert manifest["approach_timeline"]["rejected_line_count"] == 1
    assert manifest["davis_event_timeline"]["rejected_line_count"] == 1
    assert manifest["evidence_validation"]["analytic_quality"]["valid"] is False
    assert manifest["evidence_validation"]["live_quality"]["valid"] is False
    assert manifest["evidence_validation"]["davis_motion_report"]["valid"] is False
    assert not (run_dir / "realtime_grasp_hulls.json").exists()
    hull_validation = manifest["evidence_validation"][
        "bundle_artifact:realtime_grasp_hulls.json"
    ]
    assert hull_validation["valid"] is False
    assert "evidence_writer_session_id_missing" in hull_validation["reasons"]
    assert "evidence_lock_frame_id_missing" in hull_validation["reasons"]


def test_hand_demo_export_requires_current_identity_provenance_for_screenshots(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    _write_json(
        output / "latest_realtime_status.json",
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
        },
    )
    stage_dir = output / "hand_demo_stage_screenshots"
    stage_dir.mkdir()
    unprovenanced = stage_dir / "operator_start.png"
    mismatched = stage_dir / "locked_band.png"
    current = stage_dir / "approach.png"
    replaced = stage_dir / "davis_closeup.png"
    unprovenanced.write_bytes(b"old")
    mismatched.write_bytes(b"wrong-lock")
    current.write_bytes(b"current")
    replaced.write_bytes(b"original")
    _write_image_provenance(mismatched, lock_frame_id=41)
    _write_image_provenance(current)
    _write_image_provenance(replaced)
    replaced.write_bytes(b"replacement-with-old-sidecar")

    manifest = web_dashboard.record_demo_snapshot()
    run_dir = Path(manifest["run_dir"])

    assert not (run_dir / "screenshots" / "operator_start.png").exists()
    assert not (run_dir / "screenshots" / "locked_band.png").exists()
    assert (run_dir / "screenshots" / "approach.png").read_bytes() == b"current"
    assert not (run_dir / "screenshots" / "davis_closeup.png").exists()
    assert manifest["stage_screenshot_validation"]["operator_start.png"][
        "reasons"
    ] == ["image_identity_provenance_missing"]
    assert any(
        "lock_frame_id_mismatch" in reason
        for reason in manifest["stage_screenshot_validation"]["locked_band.png"][
            "reasons"
        ]
    )
    assert any(
        "artifact_mtime_ns_mismatch" in reason
        for reason in manifest["stage_screenshot_validation"]["davis_closeup.png"][
            "reasons"
        ]
    )


def test_hand_demo_export_rejects_prior_post_grasp_action_under_same_lock(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    monkeypatch.setattr(
        web_dashboard,
        "davis_analyzer",
        SimpleNamespace(close_action_id="new-action", close_action_started_at=2000.0),
    )
    _write_json(
        output / "latest_realtime_status.json",
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "decision": "HOLD",
        },
    )
    _write_json(
        output / "davis" / "post_grasp_monitoring" / "latest_report.json",
        {
            **_identity(davis=True),
            "post_grasp_monitoring_action_id": "old-action",
            "post_grasp_monitoring_started_at": 1000.0,
            "object_motion_valid": True,
            "motion_class": "GROSS_SLIP",
        },
    )

    manifest = web_dashboard.record_demo_snapshot()
    run_dir = Path(manifest["run_dir"])
    report = json.loads((run_dir / "post_grasp_report.json").read_text())
    summary = json.loads((run_dir / "final_summary.json").read_text())

    assert report["status"] == "UNAVAILABLE"
    assert summary["object_motion_valid"] is None
    validation = manifest["evidence_validation"]["post_grasp_report"]
    assert validation["valid"] is False
    assert any("post_grasp_action_id_mismatch" in reason for reason in validation["reasons"])


def test_post_grasp_davis_timeline_row_keeps_lock_source_identity():
    from grasp_project.davis.motion_gui import post_grasp_timeline_entry

    row = post_grasp_timeline_entry(
        {
            **_identity(davis=True),
            "object_motion_valid": True,
        },
        action_id="action-1",
        started_at=1234.5,
    )

    assert row["post_grasp_monitoring_action_id"] == "action-1"
    assert row["post_grasp_monitoring_started_at"] == 1234.5
    assert row["davis_target_lock_source_writer_session_id"] == "writer-1"
    assert row["davis_target_lock_source_lock_frame_id"] == 42
    assert row["davis_target_lock_source_attempt_id"] == "attempt-1"
    assert row["davis_target_lock_source_bottle_target_id"] == "bottle-1"


def test_demo_summary_excludes_numeric_depth_when_measurement_is_invalid():
    summary = web_dashboard._demo_final_summary(
        {
            "target_status": "TRACKING",
            "distance_to_bottle_mm": 95.0,
            "distance_measurement_valid": False,
        },
        {},
        {},
        {},
        {},
        [
            {
                "distance_to_bottle_mm": 120.0,
                "distance_measurement_valid": False,
            }
        ],
    )

    assert summary["approach_distance_range_mm"] is None
    assert any(
        "No current valid depth measurements" in limitation
        for limitation in summary["limitations"]
    )


def test_post_grasp_export_rejects_stray_quality_compute_and_keeps_frozen_values():
    realtime = {
        "post_grasp_active": True,
        "writer_session_id": "writer-1",
        "target_status": "TRACKING",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "target_lock": {
            "lock_frame_id": 42,
            "attempt_id": "attempt-1",
        },
        "post_grasp_baseline": {
            "writer_session_id": "writer-1",
            "attempt_id": "attempt-1",
            "lock_frame_id": 42,
            "quality_compute_id": "quality-frozen",
            "quality_epsilon": 0.126,
            "quality_score": 84.0,
            "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
            "quality_input_source": "planned_contact",
            "quality_contact_source_forced": True,
            "baseline_frozen": True,
        },
    }
    authority = web_dashboard._demo_current_lock_identity(realtime)
    stray_quality = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-stray",
        "quality_matrix_status": "updated",
        "raw_epsilon": 0.9,
        "normalized_quality_score": 1.0,
    }
    validation = web_dashboard._demo_identity_validation(
        stray_quality,
        authority,
        kind="quality",
    )
    assert validation["valid"] is False
    assert any("quality_compute_id_mismatch" in reason for reason in validation["reasons"])

    live_quality = {
        "analytic_quality_status": "CURRENT",
        "analytic_quality_score": 84.0,
        "analytic_quality_label": "GOOD",
        "baseline": {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "quality_compute_id": "quality-frozen",
        },
    }
    summary = web_dashboard._demo_final_summary(
        realtime,
        {},
        {},
        live_quality,
        {},
        [],
        authoritative_identity=authority,
        evidence_validation={"analytic_quality": validation},
    )

    assert authority["quality_compute_id"] == "quality-frozen"
    assert summary["analytic_quality"]["status"] == "CURRENT"
    assert summary["analytic_quality"]["quality_compute_id"] == "quality-frozen"
    assert summary["analytic_quality"]["score"] == 84.0
    assert summary["analytic_quality"]["epsilon"] == 0.126
    assert summary["analytic_quality"]["baseline_source"] == "IMMUTABLE_POST_GRASP_SNAPSHOT"


def test_native_stage_capture_links_prelock_operator_frame_to_next_current_lock(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    ok, encoded = cv2.imencode(".png", np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert ok is True
    png = encoded.tobytes()

    start = web_dashboard.capture_hand_demo_stage_screenshot(
        "operator_start",
        png,
        realtime={"writer_session_id": "writer-1", "frame_id": 10},
    )
    assert start["ok"] is True
    stage_dir = output / "hand_demo_stage_screenshots"
    operator = stage_dir / "operator_start.png"
    operator_sidecar = operator.with_name("operator_start.png.json")
    before_link = json.loads(operator_sidecar.read_text())
    assert before_link["pending_lock_link"] is True
    assert before_link["lock_frame_id"] is None

    realtime = {
        **_identity(),
        "frame_id": 50,
        "target_status": "TRACKING",
    }
    locked = web_dashboard.capture_hand_demo_stage_screenshot(
        "locked_band",
        png,
        realtime=realtime,
    )
    assert locked["ok"] is True
    assert locked["operator_start_linked_to_current_lock"] is True
    after_link = json.loads(operator_sidecar.read_text())
    assert after_link["pending_lock_link"] is False
    assert after_link["lock_frame_id"] == 42
    assert after_link["capture_identity_at_time"]["lock_frame_id"] is None
    assert after_link["lock_link_method"] == "next_current_lock_same_writer_session"
    authority = web_dashboard._demo_current_lock_identity(realtime)
    assert web_dashboard._demo_image_validation(
        operator,
        authority,
        realtime,
        require_sidecar=True,
    )["valid"] is True
    assert web_dashboard._demo_image_validation(
        stage_dir / "locked_band.png",
        authority,
        realtime,
        require_sidecar=True,
    )["valid"] is True

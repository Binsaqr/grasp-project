"""Camera-free contracts for the video-ready Hand Grasp Demo surface."""

import json
from pathlib import Path

from grasp_project.dashboard import web_dashboard


HTML = web_dashboard.INDEX_HTML
OPERATOR = HTML.split('id="operatorSection"', 1)[1].split(
    'id="hardwareStatusSection"', 1
)[0]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_capture(path, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\ncurrent-browser-capture")
    _write_json(
        path.with_name(path.name + ".json"),
        {
            "artifact_name": path.name,
            "artifact_mtime_ns": path.stat().st_mtime_ns,
            **identity,
        },
    )


def test_hand_grasp_demo_route_and_three_panel_presentation_render():
    response = web_dashboard.app.test_client().get("/")

    assert response.status_code == 200
    assert b"Hand Grasp Demo" in response.data
    assert '<body class="presentation">' in HTML
    assert 'id="handGraspDemoView"' in OPERATOR
    assert 'id="handDemoRealSensePanel"' in OPERATOR
    assert 'id="handDemoDavisPanel"' in OPERATOR
    assert 'id="handDemoQualityPanel"' in OPERATOR
    assert "grid-template-columns:minmax(0, 1fr) minmax(0, 1fr) minmax(280px, .82fr)" in HTML
    assert "@media (max-width: 1180px)" in HTML
    assert ".hand-demo-grid { grid-template-columns:1fr; }" in HTML


def test_quality_baseline_is_automatic_and_manual_recompute_controls_are_not_exposed():
    assert "Quality baseline is captured automatically once per lock" in HTML
    assert "Baseline captured automatically at lock" in HTML
    assert ">Compute Quality</button>" not in HTML
    assert ">Recompute Quality</button>" not in HTML


def test_operator_target_state_separates_search_waiting_and_active_reacquisition(monkeypatch):
    initial_search = web_dashboard.realsense_operator_target_state(
        {
            "target_status": "SEARCHING",
            "automatic_reacquisition_enabled": True,
            "target_loss_reason": "historical reason must not make search look lost",
        }
    )
    waiting = web_dashboard.realsense_operator_target_state(
        {
            "target_status": "LOST",
            "automatic_reacquisition_enabled": True,
            "lost_target_reacquisition_status": (
                "SINGLE_BOTTLE_AUTO_REACQUIRE_REQUIRES_EXACTLY_ONE_CANDIDATE"
            ),
        }
    )
    active = web_dashboard.realsense_operator_target_state(
        {
            "target_status": "LOST",
            "automatic_reacquisition_enabled": True,
            "lost_target_reacquisition_status": (
                "SINGLE_BOTTLE_AUTO_REACQUIRE_STABILITY_PENDING"
            ),
        }
    )

    assert initial_search["operator_target_phase"] == "SEARCHING"
    assert initial_search["operator_reacquisition_active"] is False
    assert initial_search["operator_reacquisition_waiting"] is False
    assert waiting["operator_target_phase"] == "WAITING_FOR_ONE_STABLE_CANDIDATE"
    assert waiting["operator_reacquisition_active"] is False
    assert waiting["operator_reacquisition_waiting"] is True
    assert active["operator_target_phase"] == "AUTO_REACQUIRING"
    assert active["operator_reacquisition_active"] is True
    assert active["operator_reacquisition_waiting"] is False
    monkeypatch.setattr(web_dashboard, "read_json", lambda *_args, **_kwargs: {})
    summarized = web_dashboard.summarize_realsense(
        {
            "target_status": "LOST",
            "automatic_reacquisition_enabled": True,
            "lost_target_reacquisition_status": (
                "SINGLE_BOTTLE_AUTO_REACQUIRE_STABILITY_PENDING"
            ),
        }
    )
    assert summarized["operator_target_phase"] == "AUTO_REACQUIRING"
    assert summarized["operator_reacquisition_active"] is True
    assert "rs.operator_reacquisition_active === true" in HTML
    assert "WAITING FOR ONE STABLE CANDIDATE" in HTML


def test_operator_bottle_state_javascript_statement_has_balanced_delimiters():
    assert HTML.count("const bottleState =") == 1
    statement = HTML.split("const bottleState =", 1)[1].split(
        "const bandLocked =", 1
    )[0]
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    quote = None
    escaped = False
    for character in statement:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}":
            assert stack and stack.pop() == pairs[character]
    assert quote is None
    assert stack == []


def test_fixed_center_third_roi_is_presented_as_diagnostic_not_object_specific():
    summary = web_dashboard.summarize_davis(
        {
            "roi_source": "configured_roi",
            "roi_projection_method": "fixed_center_third",
            "roi_object_specific": False,
        }
    )

    assert summary["fixed_center_third_diagnostic"] is True
    assert summary["operator_roi_role"] == "RAW_EVENT_DIAGNOSTIC_ONLY"
    assert summary["operator_roi_label"] == (
        "Fixed center-third diagnostic ROI - not object-specific"
    )
    assert "dm.operator_roi_label || 'Fixed center grasp zone · armed'" in HTML


def test_davis_summary_preserves_post_grasp_zone_attribution():
    summary = web_dashboard.summarize_davis(
        {
            "grasp_zone_motion_valid": True,
            "grasp_zone_motion_reason": "ok",
            "grasp_zone_motion_class": "MICRO SLIP",
            "motion_attribution": "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED",
        }
    )
    assert summary["grasp_zone_motion_valid"] is True
    assert summary["grasp_zone_motion_class"] == "MICRO SLIP"
    assert summary["motion_attribution"] == "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED"


def test_fixed_center_third_view_can_report_separate_uncalibrated_aps_evidence():
    summary = web_dashboard.summarize_davis(
        {
            "roi_source": "aps_target",
            "roi_xywh": [120, 90, 50, 70],
            "davis_roi_xywh": [115, 86, 116, 88],
            "analysis_roi_fixed": True,
            "acquisition_roi_fixed": True,
            "object_evidence_context_current": True,
            "object_evidence_roi_source": "aps_target",
            "object_evidence_physically_validated": False,
            "roi_object_specific": True,
            "roi_projection_method": "davis_aps_object_evidence_within_fixed_center_third",
        }
    )

    assert summary["roi_xywh"] == [115, 86, 116, 88]
    assert summary["fixed_center_third_acquisition"] is True
    assert summary["fixed_center_third_diagnostic"] is False
    assert summary["operator_roi_role"] == (
        "ESTIMATED_OBJECT_LOCAL_NOT_PHYSICALLY_VALIDATED"
    )
    assert "APS-local object evidence" in summary["operator_roi_label"]


def test_stale_at_lock_analytic_quality_cannot_look_current_in_presentation():
    payload = web_dashboard.build_grasp_quality_diagram_payload(
        {
            "quality_matrix_status": "updated",
            "quality_bundle_status": "STALE",
            "matrix_valid_for_current_pose": False,
            "quality_label": "GOOD",
            "raw_epsilon": 0.04,
            "normalized_quality_score": 82.0,
            "wrench_space_hull": {
                "wrench_points": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, -1.0, -1.0],
                ],
                "hull_simplices": [
                    [0, 1, 2],
                    [0, 1, 3],
                    [0, 2, 3],
                    [1, 2, 3],
                ],
            },
        },
        {"analytic_quality_score": 82.0, "live_quality_score": 80.0},
        {},
    )

    assert payload["status"] == "STALE"
    assert payload["analytic"]["available"] is True
    assert payload["analytic"]["stale"] is True
    assert payload["live"]["available"] is False
    assert "BASELINE INVALID - RESET / NEW LOCK" in HTML
    assert "!bandLocked ? 'WAITING FOR LOCK'" in HTML
    assert "INVALID · RESET / NEW LOCK · last ${baselineDisplay}" in HTML
    assert "if (!bandLocked) demoDecision = 'WAITING FOR LOCK'" in HTML
    assert "else if (!analyticReady || qualityStale) demoDecision = 'CANNOT CONFIRM'" in HTML
    assert "const operatorLiveQuality = !bandLocked" in HTML
    assert "Place one complete bottle in view and hold it steady for automatic lock confirmation." in HTML
    assert "Hold steady — automatic baseline refresh pending" not in HTML


def test_old_quality_artifact_cannot_present_stale_or_refresh_state_before_lock():
    assert "const analyticReady = bandLocked && !analyticInvalid" in HTML
    assert "const qualityStale = bandLocked && qualityArtifactStale;" in HTML
    assert "const analyticState = !bandLocked ? 'WAITING FOR LOCK'" in HTML
    assert "Baseline will be computed once after lock" in HTML
    assert "!bandLocked ? 'WAITING FOR LOCK'" in HTML
    assert "automatic baseline refresh pending" not in HTML


def test_fresh_lock_failures_have_stable_machine_codes():
    assert web_dashboard.fresh_lock_failure_code(
        "Bottle is not detected.", stage="target"
    ) == "no_bottle"
    assert web_dashboard.fresh_lock_failure_code(
        "Missing aligned depth.", stage="lock"
    ) == "missing_depth"
    assert web_dashboard.fresh_lock_failure_code(
        "Quality worker is busy.", stage="quality", worker_busy=True
    ) == "quality_worker_busy"
    assert web_dashboard.fresh_lock_failure_code(
        "Lock command accepted, but no locked target became available.", stage="lock"
    ) == "invalid_band"
    assert ".evidence-list {\n      display:grid;\n      gap:7px;\n    }\n    .evidence-row" in HTML


def test_fresh_lock_requires_matching_current_lock_transaction():
    payload = {
        "writer_session_id": "writer-1",
        "target_lock": {
            "target_status": "TRACKING",
            "lock_state": "LOCKED",
            "grasp_band_locked": True,
            "lock_frame_id": 42,
            "lock_command_id": "lock-command-2",
            "locked_target": {
                "valid": True,
                "lock_frame_id": 42,
                "lock_command_id": "lock-command-2",
                "attempt_id": "attempt-42",
                "bottle_target_id": "bottle-1",
            },
        },
    }

    evidence = web_dashboard.lock_transaction_evidence(
        payload,
        expected_command_id="lock-command-2",
        expected_writer_session_id="writer-1",
        previous_lock_frame=41,
    )

    assert evidence["valid"] is True
    assert evidence["lock_attempt_id"] == "attempt-42"
    assert evidence["locked_target_id"] == "bottle-1"


def test_fresh_lock_rejects_stale_lost_or_wrong_command_lock():
    stale = {
        "writer_session_id": "writer-1",
        "target_status": "LOST",
        "lock_state": "LOST",
        "grasp_band_locked": False,
        "lock_frame_id": 42,
        "lock_command_id": "old-command",
        "locked_target": {
            "valid": False,
            "lock_frame_id": 42,
            "lock_command_id": "old-command",
        },
    }

    evidence = web_dashboard.lock_transaction_evidence(
        stale,
        expected_command_id="new-command",
        expected_writer_session_id="writer-1",
        previous_lock_frame=41,
    )

    assert evidence["valid"] is False
    assert "lock_command_mismatch" in evidence["reasons"]
    assert "lock_state_not_locked" in evidence["reasons"]
    assert "locked_target_not_valid" in evidence["reasons"]


def test_fresh_quality_requires_current_result_from_matching_lock():
    matrix = {
        "quality_matrix_status": "updated",
        "quality_compute_id": "quality-42",
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-42",
        "quality_input_source": "planned_contact",
        "epsilon": 0.04,
        "epsilon_score": 80.0,
        "epsilon_force_closure": True,
    }

    evidence = web_dashboard.quality_result_evidence(
        matrix,
        expected_lock_frame=42,
        expected_lock_attempt_id="attempt-42",
        expected_writer_session_id="writer-1",
    )

    assert evidence["valid"] is True
    assert evidence["quality_input_source"] == "planned_contact"


def test_fresh_quality_rejects_stale_or_wrong_lock_result():
    matrix = {
        "quality_matrix_status": "stale_needs_recompute",
        "quality_compute_id": "quality-old",
        "writer_session_id": "writer-old",
        "lock_frame_id": 41,
        "lock_attempt_id": "attempt-old",
        "quality_input_source": "planned_contact",
        "epsilon": 0.04,
        "epsilon_score": 80.0,
        "epsilon_force_closure": True,
    }

    evidence = web_dashboard.quality_result_evidence(
        matrix,
        expected_lock_frame=42,
        expected_lock_attempt_id="attempt-42",
        expected_writer_session_id="writer-1",
    )

    assert evidence["valid"] is False
    assert "analytic_quality_stale" in evidence["reasons"]
    assert "quality_lock_frame_mismatch" in evidence["reasons"]
    assert "quality_lock_attempt_mismatch" in evidence["reasons"]
    assert "quality_writer_session_mismatch" in evidence["reasons"]


def test_realsense_demo_panel_has_live_overlay_and_operator_status_contract():
    panel = OPERATOR.split('id="handDemoRealSensePanel"', 1)[1].split(
        "</article>", 1
    )[0]

    assert "RealSense · Grasp Guidance" in panel
    assert 'id="imgHandDemoRealSense"' in panel
    for label in ("Bottle", "Grasp Band", "Distance", "Guidance", "Baseline Quality"):
        assert f"['{label}'" in HTML
    assert "RealSense overlay unavailable" in panel
    assert "Start RealSense" in panel


def test_davis_demo_panel_keeps_raw_activity_separate_from_object_motion():
    panel = OPERATOR.split('id="handDemoDavisPanel"', 1)[1].split(
        "</article>", 1
    )[0]

    assert "Sensor Activity" in panel
    assert "Grasp-Zone Estimate" in panel
    assert 'id="imgHandDemoDavis"' in panel
    assert "['Raw Events'" in HTML
    assert "['Motion Signal'" in HTML
    assert "['Slip-like Motion'" in HTML
    assert "['Disturbance'" in HTML
    assert "Full-frame event activity only" in HTML
    assert "object-specific ROI not calibrated" in HTML
    assert "objectMotionValid ? 'OBJECT-LOCAL'" in HTML
    assert "['Used for Live Quality', davisUsed ? 'YES' : 'NO']" in HTML
    assert "if (stage === 'PRE_CLOSE') return 'ARMED';" in HTML
    assert "presentationPreClose ? 'ARMED AFTER GRASP'" in HTML
    assert "presentationPreClose ? 'POST-GRASP ONLY'" in HTML
    assert "presentationPreClose ? 'NOT MONITORED YET'" in HTML


def test_offline_realsense_never_presents_cached_lock_as_live():
    assert "const realsenseLive = Boolean(data.running?.realsense && rs.overlay_stale !== true);" in HTML
    assert "const operatorBottleState = realsenseLive ? bottleState : 'OFFLINE';" in HTML
    assert "const operatorBandState = realsenseLive ? bandState : 'UNAVAILABLE';" in HTML
    assert "const operatorDistanceState = realsenseLive ? fmtOperatorDistance(rs) : 'N/A · REALSENSE OFFLINE';" in HTML
    assert "const operatorAnalyticState = realsenseLive ? analyticState : 'UNAVAILABLE';" in HTML


def test_presentation_keeps_held_3d_honest_without_mode_flicker():
    assert "dashboardMode === 'engineering' ? 'LOCKED CURRENT 3D' : 'LOCKED · 3D GUIDANCE'" in HTML
    assert "? 'LOCKED · HELD 3D'" in HTML
    assert ": 'LOCKED · HELD GUIDANCE';" in HTML


def test_pre_close_davis_cannot_drive_presentation_disturbance_decision():
    assert "['DAVIS Gate', presentationPreClose ? 'ARMED'" in HTML
    assert "['DAVIS Contribution', presentationPreClose ? 'Scoring starts after grasp'" in HTML
    assert "else if (!presentationPreClose && !objectMotionValid" in HTML
    assert "DAVIS is armed; grasp the bottle before slip, rotation, and disturbance scoring begins." in HTML
    assert "? '0.0 points · ARMED'" in HTML
    assert "DAVIS armed; scoring starts after grasp" in HTML


def test_demo_quality_panel_states_analytic_and_live_semantics_honestly():
    panel = OPERATOR.split('id="handDemoQualityPanel"', 1)[1].split(
        "</article>", 1
    )[0]

    assert "Initial Analytic Quality" in panel
    assert "Live Operational Quality" in panel
    assert "Live Grasp Quality Matrix" in panel
    assert "without recomputing epsilon" in panel
    assert 'id="handDemoMatrixBaseline"' in panel
    assert 'id="handDemoMatrixDavis"' in panel
    assert 'id="handDemoMatrixLive"' in panel
    assert "height:105px" in HTML
    assert "Hand-operated RealSense + DAVIS grasp-quality monitoring demo" in HTML
    assert "['Last Analytic Quality'" in HTML
    assert "['Status', analyticState]" in HTML
    assert "['Epsilon', epsilonText]" in HTML
    assert "['Quality Source', contactSourceLabel" in HTML
    assert "DAVIS unavailable / excluded from live quality" in HTML
    assert "DAVIS not used: global/lighting/camera disturbance" in HTML
    assert "['DAVIS Adjustment', davisAdjustmentText]" in HTML
    assert "SLIP-LIKE MOTION" in HTML


def test_demo_main_action_bar_is_nonfloating_and_contains_only_operator_actions():
    toolbar = OPERATOR.split('aria-label="Operator actions"', 1)[1].split(
        "</div>", 1
    )[0]

    for label in (
        "Start Approach Mode",
        "Start Close-Up DAVIS Monitor",
        "Start Post-Grasp Monitor",
        "Export Demo Summary",
    ):
        assert label in toolbar
    assert 'class="primary engineering-only" id="freshLockQualityBtn"' in toolbar
    assert 'class="danger engineering-only" id="resetHandPrototypeBtn"' in toolbar
    assert "normal single-bottle locking and baseline creation are automatic" in toolbar
    assert "normal target loss uses bounded automatic reacquisition" in toolbar
    assert "Waiting for Baseline" in HTML
    assert "rs.post_grasp_baseline_ready === true" in HTML
    assert "graspStateButton.disabled = !manualPostGraspActive" in HTML
    assert "Hold exactly one bottle steady while automatic lock and analytic baseline complete." in OPERATOR
    assert "Baseline will be computed once after lock" in HTML
    assert "Fresh Lock + Quality is required" not in HTML
    assert "Complete Fresh Lock + Quality" not in HTML
    for label in ("Restart DAVIS", "Calibrate Event Baseline", "Clear Motion History"):
        assert label not in toolbar
    action_css = HTML.split(".operator-action-bar {", 1)[1].split("}", 1)[0]
    assert "position:fixed" not in action_css
    assert "grid-template-columns:repeat(4, minmax(0, 1fr))" in HTML
    assert ".presentation .engineering-only { display:none !important; }" in HTML
    assert ".presentation .release-nav { display:none !important; }" in HTML
    assert 'class="hand-offset-tuning engineering-only"' in OPERATOR


def test_demo_export_writes_exact_summary_timeline_and_valid_browser_captures(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    identity = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "attempt_id": "attempt-1",
        "bottle_target_id": "bottle-1",
    }
    _write_json(
        output / "latest_realtime_status.json",
        {
            **identity,
            "lock_attempt_id": "attempt-1",
            "locked_bottle_target_id": "bottle-1",
            "target_status": "TRACKING",
            "grasp_band_locked": True,
        },
    )
    live_timeline = output / "live_quality" / "timeline.jsonl"
    live_timeline.parent.mkdir(parents=True)
    live_timeline.write_text(
        json.dumps({**identity, "live_quality_score": 73.0, "decision": "HOLD"})
        + "\n"
    )
    capture_dir = output / "hand_demo_stage_screenshots"
    for name in web_dashboard.DEMO_EXPORT_SCREENSHOTS:
        _write_capture(capture_dir / name, identity)

    manifest = web_dashboard.record_demo_snapshot()
    run_dir = Path(manifest["run_dir"])

    for name in (
        "demo_manifest.json",
        "demo_summary.json",
        "post_grasp_report.json",
        "davis_event_timeline.jsonl",
        "live_quality_timeline.jsonl",
        *web_dashboard.DEMO_EXPORT_SCREENSHOTS,
    ):
        assert (run_dir / name).is_file(), name
    assert len((run_dir / "live_quality_timeline.jsonl").read_text().splitlines()) == 1
    assert manifest["live_quality_timeline"]["captured_line_count"] == 1
    assert manifest["missing_export_screenshots"] == []
    assert [item["name"] for item in manifest["export_screenshots"]] == list(
        web_dashboard.DEMO_EXPORT_SCREENSHOTS
    )


def test_demo_export_keeps_uncalibrated_fixed_zone_motion_as_diagnostic_evidence():
    identity = {
        "writer_session_id": "writer-1",
        "attempt_id": "attempt-1",
        "lock_frame_id": 42,
    }
    summary = web_dashboard._demo_final_summary(
        {
            **identity,
            "target_status": "TRACKING",
            "target_lock": {**identity, "target_status": "TRACKING"},
        },
        {},
        {"quality_matrix_status": "updated", "raw_epsilon": 0.12},
        {"analytic_quality_score": 80.0, "live_quality_score": 62.0},
        {
            "object_motion_valid": False,
            "grasp_zone_motion_valid": True,
            "grasp_zone_motion_class": "MICRO SLIP",
            "motion_attribution": "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED",
            "downward_slip_px": 6.0,
            "rotation_delta_deg": 12.0,
            "vibration_amplitude_px_rms": 2.5,
        },
        [],
        authoritative_identity=identity,
        evidence_validation={},
    )

    assert summary["object_motion_valid"] is False
    assert summary["grasp_zone_motion_valid"] is True
    assert summary["motion_attribution"] == "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED"
    assert summary["slip_like_motion"]["status"] == "VALID_FIXED_GRASP_ZONE_UNCALIBRATED"
    assert summary["slip_like_motion"]["downward_slip_px"] == 6.0
    assert summary["rotation"] == {
        "valid": True,
        "delta_deg": 12.0,
        "motion_attribution": "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED",
        "object_specific": False,
    }
    assert summary["vibration"]["valid"] is True
    assert summary["vibration"]["object_specific"] is False

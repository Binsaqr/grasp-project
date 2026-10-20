"""Camera-free contracts for the hand-operated prototype dashboard surface."""

import inspect

from grasp_project.dashboard import web_dashboard


HTML = web_dashboard.INDEX_HTML
OPERATOR = HTML.split('id="operatorSection"', 1)[1].split(
    'id="hardwareStatusSection"', 1
)[0]


def test_hand_prototype_mode_banner_and_exact_five_stage_workflow_are_visible():
    assert "Hand Prototype Demo" in OPERATOR
    assert "Hand-operated RealSense + DAVIS grasp-quality monitoring demo" in OPERATOR
    assert "Manual hand movement only" in OPERATOR
    assert 'aria-label="Five-stage hand prototype workflow"' in OPERATOR
    assert (
        "const steps = ['Detect','Lock Band','Approach','Close Event Monitor','Pickup Monitor'];"
        in HTML
    )
    assert "grid-template-columns:repeat(5, minmax(0, 1fr))" in HTML
    for label in (
        "['Distance',",
        "['Move X',",
        "['Move Y',",
        "['Move Forward / Back',",
        "['Guidance confidence',",
    ):
        assert label in HTML
    assert "Analytic direction arrows are omitted here because they are not detected bottle motion." in HTML
    assert "The red arrow is the analytic weakest grasp-resistance direction, not detected bottle motion." in HTML
    assert 'alt="Cropped bottle quality overlay with contact patches and quality radius"' in HTML
    assert "LOCKED · WAITING FOR BOTTLE · NOT READY" in HTML


def test_previous_lock_quality_bundle_is_stale_and_hidden_from_operator_view():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "quality_compute_id": "old-quality",
            "writer_session_id": "old-writer",
            "lock_frame_id": 13348,
            "dashboard_artifact_fresh": True,
        },
        {
            "writer_session_id": "new-writer",
            "target_status": "SEARCHING",
            "target_lock": {"target_status": "SEARCHING", "lock_frame_id": None},
        },
        status_age_sec=0.1,
    )

    assert bundle["quality_bundle_status"] == "STALE"
    assert bundle["dashboard_artifact_fresh"] is False
    assert bundle["matrix_valid_for_current_pose"] is False
    assert bundle["stale_reasons"] == ["writer_session_mismatch", "no_active_lock_for_quality"]
    operator = web_dashboard.quality_image_info(
        {"bottle_quality_closeup": {"exists": True}},
        "bottle_quality_closeup",
        "/image/bottle_quality_closeup.jpg",
        {},
        bundle,
    )
    engineering = web_dashboard.quality_image_info(
        {"grasp_quality_weak_direction": {"exists": True}},
        "grasp_quality_weak_direction",
        "/image/grasp_quality_bottle_mask_weak_direction.png",
        {},
        bundle,
        advanced=True,
    )
    assert operator["exists"] is False
    assert engineering["exists"] is False
    assert engineering["stale"] is False


def test_matching_live_lock_keeps_current_quality_bundle_visible():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "dashboard_artifact_fresh": True,
        },
        {
            "writer_session_id": "writer-1",
            "lock_attempt_id": "attempt-1",
            "target_status": "TRACKING",
            "target_lock": {
                "target_status": "TRACKING",
                "lock_frame_id": 42,
                "attempt_id": "attempt-1",
            },
        },
        status_age_sec=0.1,
    )

    assert bundle["quality_bundle_status"] == "CURRENT"
    assert bundle["dashboard_artifact_fresh"] is True


def test_approach_motion_is_live_observation_not_analytic_staleness(monkeypatch):
    monkeypatch.setattr(
        web_dashboard,
        "read_json",
        lambda *_args, **_kwargs: {
            "locked_target": {
                "locked_target": {
                    "lock_frame_id": 42,
                    "depth_mm_at_lock": 500.0,
                    "locked_bottle_geometry": {
                        "bbox_xyxy": [100, 100, 200, 300],
                        "center_px": [150, 200],
                        "depth_median_mm": 500.0,
                    },
                }
            }
        },
    )
    status = {
        "writer_session_id": "writer-1",
        "frame_id": 80,
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-1",
        "target_status": "TRACKING",
        "bottle_geometry": {
            "bbox_xyxy": [250, 80, 550, 470],
            "center_px": [400, 275],
            "depth_median_mm": 180.0,
        },
        "target_lock": {"lock_frame_id": 42, "attempt_id": "attempt-1"},
    }
    matrix = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-1",
        "quality_matrix_status": "updated",
        "quality_label_stale": False,
        "matrix_stale_due_to_motion": False,
    }

    result = web_dashboard.live_matrix_staleness(status, matrix, {})

    assert result["matrix_stale_due_to_motion"] is False
    assert result["matrix_stale_reasons"] == []
    assert result["live_observation_degraded"] is True
    assert set(result["live_observation_reasons"]) == {
        "center_shift_live_observation",
        "bbox_scale_or_viewpoint_live_observation",
        "depth_change_live_observation",
    }


def test_immutable_baseline_survives_approach_and_post_grasp_slip_wins(monkeypatch):
    monkeypatch.setattr(
        web_dashboard,
        "read_json",
        lambda *_args, **_kwargs: {
            "locked_target": {
                "locked_target": {
                    "lock_frame_id": 42,
                    "depth_mm_at_lock": 500.0,
                    "locked_bottle_geometry": {
                        "bbox_xyxy": [100, 100, 200, 300],
                        "center_px": [150, 200],
                        "depth_median_mm": 500.0,
                    },
                }
            }
        },
    )
    status = {
        "writer_session_id": "writer-1",
        "frame_id": 90,
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-frozen-1",
        "target_status": "TRACKING",
        "bottle_geometry": {
            "bbox_xyxy": [250, 80, 550, 470],
            "center_px": [400, 275],
            "depth_median_mm": 180.0,
        },
        "target_lock": {"lock_frame_id": 42, "attempt_id": "attempt-1"},
    }
    matrix = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-frozen-1",
        "quality_matrix_status": "updated",
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "raw_epsilon": 0.126,
        "epsilon_score": 78.0,
    }

    observation = web_dashboard.live_matrix_staleness(status, matrix, {})

    assert observation["matrix_stale_due_to_motion"] is False
    assert observation["live_observation_degraded"] is True
    assert matrix["quality_compute_id"] == "quality-frozen-1"
    assert matrix["raw_epsilon"] == 0.126

    combined = web_dashboard.summarize_combined_quality(
        matrix,
        {
            "davis_motion_usable_for_grasp": True,
            "motion_class": "GROSS_SLIP",
            "gross_slip": 90.0,
        },
        post_grasp_active=True,
    )

    assert combined["combined_quality_status"] == "SLIP_RISK"
    assert combined["initial_epsilon_score"] == 78.0
    assert matrix["quality_compute_id"] == "quality-frozen-1"
    assert matrix["raw_epsilon"] == 0.126


def test_relock_or_quality_compute_identity_change_stales_analytic_quality(monkeypatch):
    monkeypatch.setattr(web_dashboard, "read_json", lambda *_args, **_kwargs: {})
    status = {
        "writer_session_id": "writer-1",
        "frame_id": 80,
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-new",
        "target_status": "TRACKING",
        "grasp_band_relock_required": True,
        "target_lock": {"lock_frame_id": 42, "attempt_id": "attempt-1"},
    }
    matrix = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-old",
    }

    result = web_dashboard.live_matrix_staleness(status, matrix, {})

    assert result["matrix_stale_due_to_motion"] is True
    assert "quality_compute_id_mismatch" in result["matrix_stale_reasons"]
    assert "grasp_band_relock_required" in result["matrix_stale_reasons"]


def test_same_lock_quality_refresh_keeps_previous_complete_baseline_current(monkeypatch):
    monkeypatch.setattr(web_dashboard, "read_json", lambda *_args, **_kwargs: {})
    status = {
        "writer_session_id": "writer-1",
        "frame_id": 80,
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-new",
        "target_status": "TRACKING",
        "target_lock": {"lock_frame_id": 42, "attempt_id": "attempt-1"},
    }
    matrix = {
        "writer_session_id": "writer-1",
        "lock_frame_id": 42,
        "lock_attempt_id": "attempt-1",
        "quality_compute_id": "quality-old",
    }

    result = web_dashboard.live_matrix_staleness(status, matrix, {})

    assert result["matrix_stale_due_to_motion"] is False
    assert result["matrix_stale_reasons"] == []
    assert result["same_lock_quality_refresh_in_progress"] is True
    assert "same_lock_quality_refresh_in_progress" in result["live_observation_reasons"]


def test_same_lock_quality_bundle_refresh_does_not_flash_stale():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "quality_compute_id": "quality-old",
        },
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "quality_compute_id": "quality-new",
            "target_status": "TRACKING",
            "target_lock": {
                "target_status": "TRACKING",
                "lock_frame_id": 42,
                "attempt_id": "attempt-1",
            },
        },
        status_age_sec=0.1,
    )

    assert bundle["quality_bundle_status"] == "CURRENT"
    assert bundle["same_lock_quality_refresh_in_progress"] is True
    assert bundle.get("stale_reasons") in (None, [])


def test_low_confidence_held_projection_keeps_same_locked_baseline_current():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "quality_compute_id": "quality-1",
        },
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "quality_compute_id": "quality-1",
            "target_status": "LOW_CONFIDENCE",
            "grasp_band_locked": True,
            "grasp_band_projected": True,
            "grasp_band_held": True,
            "grasp_band_relock_required": False,
            "target_lock": {
                "target_status": "LOW_CONFIDENCE",
                "lock_frame_id": 42,
                "attempt_id": "attempt-1",
                "grasp_band_locked": True,
                "grasp_band_projected": True,
                "grasp_band_held": True,
            },
        },
        status_age_sec=0.1,
    )

    assert bundle["quality_bundle_status"] == "CURRENT"
    assert bundle.get("stale_reasons") in (None, [])


def test_pre_close_combined_quality_never_uses_davis_penalty():
    result = web_dashboard.summarize_combined_quality(
        {"quality_matrix_status": "updated", "epsilon_score": 80.0},
        {
            "davis_motion_usable_for_grasp": True,
            "motion_class": "GROSS_SLIP",
            "gross_slip": 100.0,
        },
        post_grasp_active=False,
    )

    assert result["davis_motion_used_for_quality"] is False
    assert result["davis_motion_penalty"] == 0.0
    assert result["davis_motion_excluded_reason"] == "pre_close_monitoring_only"


def test_same_frame_different_attempt_does_not_reuse_quality():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-old",
        },
        {
            "writer_session_id": "writer-1",
            "lock_attempt_id": "attempt-new",
            "target_status": "TRACKING",
            "target_lock": {
                "target_status": "TRACKING",
                "lock_frame_id": 42,
                "attempt_id": "attempt-new",
            },
        },
        status_age_sec=0.1,
    )

    assert bundle["quality_bundle_status"] == "STALE"
    assert "lock_attempt_mismatch" in bundle["stale_reasons"]


def test_current_bundle_without_active_lock_or_lock_provenance_is_stale():
    no_active_lock = web_dashboard.validate_quality_bundle_for_live_status(
        {"quality_bundle_status": "CURRENT", "writer_session_id": "writer-1"},
        {"writer_session_id": "writer-1", "target_status": "SEARCHING"},
        status_age_sec=0.1,
    )
    missing_provenance = web_dashboard.validate_quality_bundle_for_live_status(
        {"quality_bundle_status": "CURRENT", "writer_session_id": "writer-1"},
        {
            "writer_session_id": "writer-1",
            "lock_attempt_id": "attempt-1",
            "target_status": "TRACKING",
            "target_lock": {"target_status": "TRACKING", "lock_frame_id": 42},
        },
        status_age_sec=0.1,
    )

    assert "no_active_lock_for_quality" in no_active_lock["stale_reasons"]
    assert "lock_frame_provenance_missing" in missing_provenance["stale_reasons"]
    assert "lock_attempt_provenance_missing" in missing_provenance["stale_reasons"]


def test_post_grasp_occlusion_keeps_same_lock_analytic_bundle_current():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "quality_compute_id": "quality-1",
        },
        {
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
            "target_status": "LOW_CONFIDENCE",
            "post_grasp_active": True,
            "post_grasp_baseline": {
                "writer_session_id": "writer-1",
                "lock_frame_id": 42,
                "attempt_id": "attempt-1",
                "quality_compute_id": "quality-1",
            },
            "target_lock": {
                "target_status": "LOW_CONFIDENCE",
                "lock_frame_id": 42,
                "attempt_id": "attempt-1",
            },
        },
        status_age_sec=0.1,
    )

    assert bundle["quality_bundle_status"] == "CURRENT"
    assert bundle["post_grasp_frozen_baseline_active"] is True
    assert bundle.get("stale_reasons") in (None, [])


def test_failed_quality_bundle_remains_failed_when_live_status_is_stale():
    bundle = web_dashboard.validate_quality_bundle_for_live_status(
        {"quality_bundle_status": "FAILED", "dashboard_artifact_fresh": False},
        {},
        status_age_sec=10.0,
    )

    assert bundle["quality_bundle_status"] == "FAILED"
    assert "realtime_status_stale" in bundle["stale_reasons"]


def test_current_quality_requires_symmetric_writer_and_attempt_provenance():
    missing_writer = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
        },
        {
            "writer_session_id": "writer-1",
            "target_status": "TRACKING",
            "target_lock": {"target_status": "TRACKING", "lock_frame_id": 42, "attempt_id": "attempt-1"},
        },
        status_age_sec=0.1,
    )
    missing_live_attempt = web_dashboard.validate_quality_bundle_for_live_status(
        {
            "quality_bundle_status": "CURRENT",
            "writer_session_id": "writer-1",
            "lock_frame_id": 42,
            "lock_attempt_id": "attempt-1",
        },
        {
            "writer_session_id": "writer-1",
            "target_status": "TRACKING",
            "target_lock": {"target_status": "TRACKING", "lock_frame_id": 42},
        },
        status_age_sec=0.1,
    )

    assert "writer_session_provenance_missing" in missing_writer["stale_reasons"]
    assert "live_lock_attempt_missing" in missing_live_attempt["stale_reasons"]


def test_partial_bundle_hides_artifact_not_generated_for_current_compute():
    image = web_dashboard.quality_image_info(
        {"bottle_quality_closeup": {"exists": True}},
        "bottle_quality_closeup",
        "/image/bottle_quality_closeup.jpg",
        {},
        {
            "quality_bundle_status": "PARTIAL",
            "quality_compute_id": "new-quality",
            "artifact_paths": {
                "bottle_quality_closeup": {"generated_for_current_compute": False},
            },
        },
    )

    assert image["exists"] is False


def test_quality_images_fail_closed_without_affirmative_bundle_artifact_provenance():
    files = {"bottle_quality_closeup": {"exists": True}}
    without_bundle = web_dashboard.quality_image_info(
        files,
        "bottle_quality_closeup",
        "/image/bottle_quality_closeup.jpg",
        {},
        {},
    )
    without_artifact_provenance = web_dashboard.quality_image_info(
        files,
        "bottle_quality_closeup",
        "/image/bottle_quality_closeup.jpg",
        {},
        {"quality_bundle_status": "CURRENT", "quality_compute_id": "quality-1"},
    )
    current = web_dashboard.quality_image_info(
        files,
        "bottle_quality_closeup",
        "/image/bottle_quality_closeup.jpg",
        {},
        {
            "quality_bundle_status": "CURRENT",
            "quality_compute_id": "quality-1",
            "artifact_paths": {
                "bottle_quality_closeup": {"generated_for_current_compute": True},
            },
        },
    )

    assert without_bundle["exists"] is False
    assert without_artifact_provenance["exists"] is False
    assert current["exists"] is True


def test_operator_sees_locked_band_drift_and_relock_without_opening_diagnostics():
    for label, element_id in (
        ("Locked Band", "lockedBandStatus"),
        ("Band Drift", "bandDriftStatus"),
        ("Relock", "bandRelockStatus"),
    ):
        assert label in OPERATOR
        assert f'id="{element_id}"' in OPERATOR
    assert 'id="handBandStatus"' in OPERATOR
    assert 'class="hand-band-status"' in OPERATOR
    assert '<div class="visual-tabs engineering-only"' in OPERATOR
    assert ".presentation .engineering-only { display:none !important; }" in HTML
    assert "RELOCKED · generation" in HTML
    assert "last_relock_event" in HTML
    assert "card.classList.remove('ok', 'warn', 'bad', 'info')" in HTML
    assert "card.classList.add('hand-band-item', cls)" in HTML
    assert 'id="davisRoiOnOverlay"' in OPERATOR
    assert 'id="davisRoiOnOverlayPolygon"' in OPERATOR
    assert "data.davis_roi_on_realsense_overlay" in HTML
    assert "davisRoi.valid === true" in HTML


def test_main_actions_use_safe_existing_endpoints_and_honest_approach_ack():
    expected_main_buttons = (
        ("Fresh Lock + Quality", 'onclick="freshLockQuality()"'),
        ("Start Approach Mode", 'onclick="markApproachPhase()"'),
        ("Start Close-Up DAVIS Monitor", 'onclick="startCloseUpDavisMonitor()"'),
        ("Start Post-Grasp Monitor", 'onclick="toggleGraspDeclaration()"'),
        ("Reset Lock", 'onclick="resetHandPrototype()"'),
    )
    main_toolbar = OPERATOR.split('aria-label="Operator actions"', 1)[1].split(
        "</div>", 1
    )[0]
    for label, handler in expected_main_buttons:
        assert label in main_toolbar
        assert handler in main_toolbar
    assert 'class="primary engineering-only" id="freshLockQualityBtn"' in main_toolbar
    assert 'class="danger engineering-only" id="resetHandPrototypeBtn"' in main_toolbar
    assert "Export Demo Summary" in main_toolbar
    assert 'onclick="recordSnapshot()"' in main_toolbar
    assert "Capture Selected Stage" not in main_toolbar

    expected_advanced_buttons = (
        ("Capture Selected Stage", 'onclick="captureSelectedHandDemoStage()"'),
    )
    for label, handler in expected_advanced_buttons:
        assert label in OPERATOR
        assert handler in OPERATOR
    assert '<details class="advanced-controls engineering-only">' in OPERATOR

    approach = HTML.split("async function markApproachPhase()", 1)[1].split(
        "async function startCloseUpDavisMonitor()", 1
    )[0]
    assert "bandLocked" in approach
    assert "Automatic lock and analytic baseline are pending" in approach
    assert "Fresh Lock + Quality is required" not in approach
    assert "realsenseCommand('start_approach')" in approach
    assert "Movement remains manual" in approach
    for automatic_capture in (
        "captureHandDemoStage('locked_band')",
        "captureHandDemoStage('approach')",
        "captureHandDemoStage('davis_closeup')",
        "captureHandDemoStage('post_grasp')",
        "captureHandDemoStage('operator_start')",
    ):
        assert automatic_capture not in HTML
    assert "async function captureSelectedHandDemoStage()" in HTML
    assert "return captureHandDemoStage(stage)" in HTML
    assert "Screen capture starts only when you press Capture Selected Stage" in OPERATOR
    for stage in ("operator_start", "locked_band", "approach", "davis_closeup", "post_grasp"):
        assert f'<option value="{stage}">' in OPERATOR
    assert "navigator.mediaDevices.getDisplayMedia" in HTML
    assert "captureHandDemoStage('hand_grasp_demo_screen')" in HTML
    assert "captureHandDemoStage('operator_snapshot', 'handDemoRealSensePanel')" in HTML
    assert "captureHandDemoStage('davis_snapshot', 'handDemoDavisPanel')" in HTML
    assert "captureHandDemoStage('quality_snapshot', 'handDemoQualityPanel')" in HTML
    assert "return davisCommand('start')" in HTML
    assert "postGraspActive ? 'mark_gripper_open' : 'mark_gripper_closed'" in HTML
    assert "return realsenseCommand('reset_lock_clear_matrix')" in HTML
    assert "return post('/api/record-snapshot')" in HTML
    assert '"start_approach": "Manual approach phase acknowledged' in inspect.getsource(
        web_dashboard.api_realsense_command
    )


def test_hand_config_and_live_band_status_are_passed_through_to_operator_payload():
    status = {
        "hand_mounted_camera_mode": True,
        "hand_mounted_camera": {
            "enabled": True,
            "camera_frame": "test_camera_frame",
            "use_camera_frame_guidance": True,
        },
        "hand_mounted_stage": "APPROACH_TRACKING",
        "approach_mode_active": True,
        "approach_mode_requested": True,
        "locked_band_visible": True,
        "locked_band_current_rgbd_supported": False,
        "operator_band_state": "CURRENT_BOTTLE_RELATIVE_GUIDE",
        "operator_locked_fraction_visual_guide_current": True,
        "operator_band_is_safety_measurement": False,
        "target_lock": {
            "grasp_band_locked": True,
            "grasp_band_projected": True,
            "grasp_band_held": False,
            "grasp_band_projection_unavailable": True,
            "grasp_band_state": "LOCKED_PROJECTION_UNAVAILABLE",
            "sticky_single_bottle_lock_enabled": True,
            "sticky_lock_waiting_reacquisition": True,
            "sticky_lock_dropout_frames": 18,
            "sticky_lock_preserved_attempt_id": "attempt-12",
            "grasp_band_drift_px": 2.5,
            "grasp_band_scale_change": 1.1,
            "grasp_band_relock_required": False,
            "band_drift_state": "OK",
            "locked_grasp_band": {"locked_body_fraction_center": 0.52},
        },
    }

    hand = web_dashboard.hand_prototype_dashboard_status(status)

    assert hand["hand_mounted_camera_mode"] is True
    assert hand["hand_mounted_mode_label"] == (
        "Hand-operated RealSense + DAVIS grasp-quality monitoring demo"
    )
    assert hand["hand_mounted_camera"]["camera_frame"] == "test_camera_frame"
    assert hand["hand_mounted_stage"] == "APPROACH_TRACKING"
    assert hand["approach_mode_requested"] is True
    assert hand["guidance_frame"] == "camera_frame"
    assert hand["robot_base_coordinates_available"] is False
    assert hand["grasp_band_locked"] is True
    assert hand["grasp_band_projected"] is True
    assert hand["grasp_band_projection_unavailable"] is True
    assert hand["locked_band_visible"] is True
    assert hand["locked_band_current_rgbd_supported"] is False
    assert hand["operator_band_state"] == "CURRENT_BOTTLE_RELATIVE_GUIDE"
    assert hand["operator_locked_fraction_visual_guide_current"] is True
    assert hand["operator_band_is_safety_measurement"] is False
    assert hand["grasp_band_state"] == "LOCKED_PROJECTION_UNAVAILABLE"
    assert hand["sticky_single_bottle_lock_enabled"] is True
    assert hand["sticky_lock_waiting_reacquisition"] is True
    assert hand["sticky_lock_dropout_frames"] == 18
    assert hand["sticky_lock_preserved_attempt_id"] == "attempt-12"
    assert hand["grasp_band_drift_px"] == 2.5
    assert hand["band_drift_state"] == "OK"
    assert hand["locked_body_fraction_center"] == 0.52
    assert '"hand_prototype": hand_prototype' in inspect.getsource(
        web_dashboard.api_status
    )


def test_dashboard_labels_current_visual_fallback_without_claiming_depth_or_ready():
    assert "operatorBandState === 'CURRENT_BOTTLE_RELATIVE_GUIDE'" in HTML
    assert "CURRENT BOTTLE GUIDE (DISPLAY)" in HTML
    assert "if (projectionUnavailable) lockedParts.push('WAITING FOR DEPTH')" in HTML
    assert "currentBottleGuide ? 'info'" in HTML


def test_config_fallback_keeps_camera_frame_claim_bounded_when_backend_is_idle():
    hand = web_dashboard.hand_prototype_dashboard_status({})

    assert hand["hand_mounted_camera_mode"] is True
    assert hand["hand_mounted_camera"]["mount_location"] == "forearm_behind_wrist"
    assert hand["hand_mounted_camera"]["camera_frame"] == (
        "realsense_color_optical_frame"
    )
    assert hand["guidance_frame"] == "camera_frame"
    assert hand["robot_base_calibrated"] is False
    assert hand["robot_base_coordinates_available"] is False

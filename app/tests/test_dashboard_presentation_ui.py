"""Camera-free checks for the final presentation dashboard layout."""

import re
import threading
import time
import inspect
from pathlib import Path

import numpy as np

from grasp_project.dashboard import web_dashboard


HTML = web_dashboard.INDEX_HTML


def css_block(selector):
    return HTML.split(selector, 1)[1].split("}", 1)[0]


def test_presentation_mode_is_default_and_engineering_reveals_debug():
    assert "|| 'presentation'" in HTML
    assert "presentationModeBtn" in HTML
    assert ".presentation .engineering-only { display:none !important; }" in HTML
    assert ".engineering .presentation-only { display:none !important; }" in HTML


def test_final_top_tabs_match_operator_workflow():
    nav = HTML.split('<nav class="release-nav"', 1)[1].split("</nav>", 1)[0]
    for label in (
        "Hand Grasp Demo",
        "DAVIS Motion",
        "Grasp Quality",
        "Point Cloud",
        "Multi-View Scan",
        "Benchmark Lab",
        "Embodiment",
        "Diagnostics",
    ):
        assert label in nav
    assert "Object Profiles" not in nav
    for label in ("Multi-View Scan", "Benchmark Lab", "Embodiment", "Diagnostics"):
        link = nav.split(f">{label}</a>", 1)[0].rsplit("<a ", 1)[1]
        assert 'class="engineering-only"' in link


def test_default_operator_view_is_the_hand_grasp_demo_with_engineering_realsense_context():
    nav = HTML.split('<nav class="release-nav"', 1)[1].split("</nav>", 1)[0]
    operator = HTML.split('id="operatorSection"', 1)[1].split('id="hardwareStatusSection"', 1)[0]
    assert "Hand Grasp Demo" in nav
    assert "Operator Workspace — RealSense Overview" in operator
    assert "Live RealSense RGB-D grasp overview" in operator
    assert '<span class="presentation-only">Hand Grasp Demo</span>' in operator


def test_realsense_start_is_single_flight(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def slow_start(name):
        calls.append(name)
        entered.set()
        assert release.wait(timeout=2.0)
        return True, "started"

    monkeypatch.setattr(web_dashboard, "_start_process_once", slow_start)
    first_result = []
    worker = threading.Thread(
        target=lambda: first_result.append(web_dashboard.start_process("realsense"))
    )
    worker.start()
    assert entered.wait(timeout=2.0)
    second = web_dashboard.start_process("realsense")
    release.set()
    worker.join(timeout=2.0)

    assert calls == ["realsense"]
    assert first_result == [(True, "started")]
    assert second[0] is True
    assert "already in progress" in second[1]


def test_davis_start_is_single_flight(monkeypatch):
    original_thread = web_dashboard.davis_thread
    created = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(web_dashboard.threading, "Thread", FakeThread)
    web_dashboard.davis_thread = None
    try:
        first = web_dashboard.start_davis_motion()
        second = web_dashboard.start_davis_motion()
    finally:
        web_dashboard.davis_thread = original_thread

    assert first[0] is True
    assert second[0] is True
    assert len(created) == 1
    assert "already running" in second[1]


def test_runtime_output_failure_is_best_effort_and_does_not_escape():
    def blocked_writer(*_args, **_kwargs):
        raise PermissionError("destination remains open")

    assert web_dashboard.write_runtime_output_best_effort(
        blocked_writer,
        "ignored-path",
        {"state": "CURRENT"},
    ) is False


def test_dashboard_diagnostic_image_retries_transient_bind_mount_lock(monkeypatch, tmp_path):
    output = tmp_path / "diagnostic.jpg"
    original_replace = Path.replace
    calls = {"count": 0}

    def transient_replace(self, target):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("destination temporarily open")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", transient_replace)
    assert web_dashboard._write_image(output, np.zeros((8, 8, 3), dtype=np.uint8)) is True
    assert output.exists()
    assert calls["count"] == 3


def test_dashboard_diagnostic_image_permanent_lock_is_nonfatal_and_cleans_temp(monkeypatch, tmp_path):
    output = tmp_path / "diagnostic.jpg"

    def blocked_replace(_self, _target):
        raise PermissionError("destination remains open")

    monkeypatch.setattr(Path, "replace", blocked_replace)
    assert web_dashboard._write_image(output, np.zeros((8, 8, 3), dtype=np.uint8)) is False
    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp.jpg")) == []


def test_davis_summary_distinguishes_fresh_quiet_report_from_stale_active_claim():
    fresh = web_dashboard.summarize_davis({
        "report_timestamp": time.time(),
        "event_stream_status": "active",
    })
    stale = web_dashboard.summarize_davis({
        "report_timestamp": time.time() - 30.0,
        "event_stream_status": "active",
    })

    assert fresh["report_heartbeat_current"] is True
    assert fresh["report_heartbeat_age_sec"] < 1.0
    assert stale["report_heartbeat_current"] is False
    assert stale["report_heartbeat_age_sec"] >= 30.0
    assert "dm.report_heartbeat_current !== false" in HTML


def test_operator_section_has_no_raw_json_or_writer_debug_labels():
    operator = HTML.split('id="operatorSection"', 1)[1].split('id="hardwareStatusSection"', 1)[0]
    forbidden = ("<pre", "writer_session_id", "raw json", "latest_realtime_status.json")
    assert not any(token in operator.lower() for token in forbidden)


def test_operator_presentation_hides_debug_panels_and_visual_switcher():
    operator = HTML.split('id="operatorSection"', 1)[1].split('id="hardwareStatusSection"', 1)[0]
    assert '<div class="visual-tabs engineering-only"' in operator
    assert '<div class="quality-card engineering-only">\n            <h3>Sensor Support</h3>' in operator
    assert '<div class="quality-card engineering-only">\n            <h3>Why This Score?</h3>' in operator
    assert '<div class="quality-card engineering-only">\n            <h3>Essential Status</h3>' in operator
    assert '<div class="release-summary-strip engineering-only">' in operator
    assert '<div class="hand-band-item engineering-only" id="handStageCard">' in operator
    assert '<details class="advanced-controls engineering-only">' in operator
    assert '<span class="chip engineering-only" id="pointcloudChip">' in HTML
    assert '<span class="chip engineering-only" id="syncChip">' in HTML
    assert '<div class="clock engineering-only" id="clock">' in HTML
    assert "el.classList.add('chip', stateClass(value))" in HTML
    assert ": (fresh.fresh_lock_quality_error" in HTML
    assert "Fresh Lock + Quality: ${freshStatus} - ${result.message}" not in HTML


def test_hidden_heavy_panels_are_guarded_by_active_section():
    assert "function heavyPanelVisible" in HTML
    assert "isSectionActive('pointcloudSection') || operatorVisualActive('pointcloud')" in HTML
    assert "isSectionActive('multiviewSection') || operatorVisualActive('multiview')" in HTML
    assert "isSectionActive('benchmarkSection')" in HTML
    assert "if (pointcloudVisible) refreshPointcloudViewer(pc);" in HTML
    assert "if (benchmarkVisible) void refreshBenchmarkStatus(false);" in HTML
    assert "DIAGNOSTICS_REFRESH_DIVIDER" in HTML
    assert "diagnosticsRefreshDue" in HTML


def test_presentation_hides_traces_and_pointcloud_diagnostics():
    assert ".presentation .benchmark-trace" in HTML
    assert ".presentation .pointcloud-diagnostics" in HTML
    assert "Point Cloud Diagnostics" in HTML
    assert "Multi-View Diagnostics" in HTML


def test_important_images_have_modal_and_download_handlers():
    assert "function openModalFromImg" in HTML
    assert "modalDownload" in HTML
    assert "document.querySelectorAll('.plot-card')" in HTML
    assert "document.querySelectorAll('.visual-panel img')" in HTML
    assert "cursor:zoom-in" in HTML


def test_dashboard_breakpoints_cover_1920_and_smaller_browsers():
    assert "@media (min-width: 1500px) and (min-height: 900px)" in HTML
    assert "width:min(100%, 900px)" in HTML
    assert "@media (max-width: 1180px)" in HTML
    assert "@media (max-width: 720px)" in HTML


def test_operator_video_and_controls_do_not_force_black_empty_space():
    visual_stage = css_block(".visual-stage {")
    visual_img = css_block(".visual-panel img {")
    action_bar = css_block(".operator-action-bar {")
    operator = HTML.split('id="operatorSection"', 1)[1].split('id="hardwareStatusSection"', 1)[0]
    assert "min-height:0" in visual_stage
    assert "width:min(100%, 900px)" in visual_stage
    assert "min-height:0" in visual_img
    assert "max-height:none" in visual_img
    assert "height:auto" in visual_img
    assert "position:fixed" not in action_bar
    assert '<div class="operator-action-bar hand-demo-actions">' in operator
    assert 'class="bottom-action-dock"' not in HTML
    assert '<details class="advanced-controls engineering-only">' in operator
    assert '<details class="advanced-controls engineering-only" open>' not in operator
    assert "I Have Grasped — Monitor Motion" in operator
    assert 'id="graspStateBtn"' in operator
    assert "toggleGraspDeclaration()" in operator
    assert "Release / End Motion Test" in HTML
    assert "'mark_gripper_open'" in HTML
    assert "'mark_gripper_closed'" in HTML
    assert "Single-bottle lock, analytic baseline, and approach tracking start automatically" in operator
    assert "Post-grasp/release controls only mark the physical phase" in operator
    assert "do not confirm physical contact or grasp success" in operator


def test_diagnostics_expose_fail_closed_original_target_occlusion_hold():
    assert "Original Target Hold" in HTML
    assert "PRESERVED / READY BLOCKED" in HTML
    assert "Locked Bottle / Band" in HTML
    assert "Post-Grasp Identity" in HTML


def test_operator_decision_rail_shows_only_actionable_guidance_in_presentation():
    primary = HTML.split("const primary = [", 1)[1].split("];", 1)[0]
    for label in (
        "['Distance'",
        "['Move X'",
        "['Move Y'",
        "['Move Forward / Back'",
        "['Guidance confidence'",
    ):
        line = next(row for row in primary.splitlines() if label in row)
        assert line.rstrip().endswith(", true],")
    for label in (
        "['Ready'",
        "['Target'",
        "['Object Pose'",
        "['Close-Range ROI'",
        "['Grasp Feasibility'",
        "['Grip Width'",
        "['Contact'",
        "['Fingers'",
        "['Analytic Quality'",
        "['Quality Applicability'",
        "['Live Quality'",
        "['Point Cloud'",
        "['DAVIS'",
        "['Clutter'",
        "['Main Blocker'",
    ):
        line = next(row for row in primary.splitlines() if label in row)
        assert line.rstrip().endswith(", false],")
    for technical_label in ("['Delta'", "['Trend'", "['Risk'", "Opposing Contacts", "writer_session"):
        assert technical_label not in primary
    assert "showInPresentation ? '' : ' engineering-only'" in HTML


def test_not_ready_overall_decision_does_not_promote_feasibility_label():
    decision_logic = HTML.split("const decisionValue =", 1)[1].split(
        "const decisionCard", 1
    )[0]

    assert "(!rs.ready_to_close && feasibilityLabel)" not in decision_logic
    assert "rs.ready_to_close" in decision_logic
    assert "rs.ui_primary_status || rs.decision || rs.state" in decision_logic


def test_realsense_close_range_guidance_and_depth_projection_reach_summary_and_ui():
    summary = web_dashboard.summarize_realsense(
        {
            "bottle_distance_mm": 92.0,
            "distance_status": "READY",
            "distance_ready": True,
            "distance_measurement_valid": True,
            "actionable_distance_mm": 92.0,
            "in_close_range": True,
            "desired_close_distance_mm": 90.0,
            "min_close_distance_mm": 80.0,
            "max_close_distance_mm": 100.0,
            "close_distance_range_source": "USER_CONFIGURED_TASK_TARGET",
            "close_distance_range_validation_status": "NOT_PHYSICALLY_VALIDATED",
            "close_distance_range_physically_validated": False,
            "roi_depth_projection_scale": 2.25,
            "roi_projected_bbox_unclipped": [-20, 30, 680, 450],
            "roi_projected_visible_fraction": 0.82,
            "roi_projected_extent_partial": True,
        }
    )

    assert summary["distance_status"] == "READY"
    assert summary["distance_ready"] is True
    assert summary["distance_measurement_valid"] is True
    assert summary["actionable_distance_mm"] == 92.0
    assert summary["in_close_range"] is True
    assert summary["desired_close_distance_mm"] == 90.0
    assert (summary["min_close_distance_mm"], summary["max_close_distance_mm"]) == (80.0, 100.0)
    assert summary["close_distance_range_validation_status"] == "NOT_PHYSICALLY_VALIDATED"
    assert summary["close_distance_range_physically_validated"] is False
    assert summary["roi_depth_projection_scale"] == 2.25
    assert summary["roi_projected_bbox_unclipped"] == [-20, 30, 680, 450]
    assert summary["roi_projected_visible_fraction"] == 0.82
    assert summary["roi_projected_extent_partial"] is True
    assert "function fmtCloseDistance(rs)" in HTML
    assert "function fmtOperatorDistance(rs)" in HTML
    assert "function fmtDepthRoiProjection(rs)" in HTML
    assert "['Distance', operatorDistance, true]" in HTML
    assert "['Close-Range ROI', depthRoiProjection, false]" in HTML
    assert "Estimated RGB-D" in HTML


def test_realsense_summary_preserves_explicit_false_visual_contact_gate():
    summary = web_dashboard.summarize_realsense(
        {"visual_contact_ready": False, "contact_model_ready": True}
    )

    assert summary["visual_contact_ready"] is False


def test_davis_tab_labels_force_policy_as_simulated_and_unvalidated():
    assert "Simulated Force Policy" in HTML
    assert "Simulated Grip Delta" in HTML
    assert "Simulated Torque Delta" in HTML
    assert "Force policy diagnostic only" in HTML
    assert "NOT_PHYSICALLY_VALIDATED" in HTML


def test_stale_quality_copy_is_actionable_not_conflicting():
    assert "BASELINE INVALID - RESET / NEW LOCK" in HTML
    assert "RESET / NEW LOCK REQUIRED" in HTML
    assert "STALE - RECOMPUTE" not in HTML
    assert "RECOMPUTE REQUIRED" not in HTML
    assert "['Last Quality', rawQualityLabel || 'N/A']" in HTML
    assert "['Status', qualityStatusText]" in HTML
    assert "GOOD (STALE)" not in HTML
    assert "`${raw} (STALE)`" not in HTML


def test_pointcloud_presentation_uses_summary_cards_not_raw_count_table():
    pointcloud = HTML.split('id="pointcloudSection"', 1)[1].split('id="multiviewSection"', 1)[0]
    visible_rows = HTML.split("cards(document.getElementById('pointcloudStatusRows')", 1)[1].split("]);", 1)[0]
    assert 'class="grid status-grid pointcloud-summary-grid" id="pointcloudStatusRows"' in pointcloud
    for label in ("['Freshness'", "['ROI Source'", "['Scene Points'", "['Bottle Points'", "['Bottle Dimensions'", "['3D Contact Validity'", "['Point Cloud FPS'"):
        assert label in visible_rows
    for raw_label in ("Raw Scene Points", "Filtered Scene Points", "Displayed Scene Points", "Raw Bottle Points", "Displayed Bottle Points"):
        assert raw_label not in visible_rows


def test_pointcloud_views_are_distinguished_by_frame_and_registration_semantics():
    pointcloud = HTML.split('id="pointcloudSection"', 1)[1].split('id="multiviewSection"', 1)[0]
    multiview = HTML.split('id="multiviewSection"', 1)[1].split('id="objectProfilesSection"', 1)[0]
    assert "Current RealSense Point Cloud" in pointcloud
    assert "Interactive Current-Frame RGB-D Point Cloud" in pointcloud
    assert "Rolling = unregistered temporal diagnostic only" in pointcloud
    assert "not geometry/contact evidence" in pointcloud
    assert "Registration-Gated Multi-View Bottle Reconstruction" in multiview
    assert "Current Frame + Accepted Scan" in multiview
    assert "Current Frame Only" in multiview


def test_pointcloud_refresh_preserves_operator_camera_after_initial_fit():
    assert "viewFitted: false" in HTML
    assert "if (!pointcloudViewer.viewFitted)" in HTML
    fit_block = HTML.split("if (!pointcloudViewer.viewFitted)", 1)[1].split("}", 1)[0]
    assert "pointcloudViewer.distance" in fit_block


def test_pointcloud_loads_packaged_three_js_before_cdn_fallbacks():
    loader_urls = HTML.split("const THREE_JS_CDN_URLS = [", 1)[1].split("];", 1)[0]
    assert '<script src="/vendor/three.min.js"></script>' in HTML
    assert loader_urls.index("'/vendor/three.min.js'") < loader_urls.index("'https://unpkg.com/")


def test_quality_secondary_visuals_and_contacts_are_collapsed_or_engineering_only():
    quality = HTML.split('id="qualitySection"', 1)[1].split('id="davisMotionSection"', 1)[0]
    assert '<div class="quality-card engineering-only">\n          <h3>Contact Inputs</h3>' in quality
    assert '<summary>Contact Hull Overlay</summary>' in quality
    assert '<summary>Advanced Wrench-Space Details</summary>' in quality
    assert '<summary>Quality Diagnostics</summary>' in quality


def test_empty_visual_panels_show_explanatory_empty_states():
    assert "plot-empty-state" in HTML
    assert "Quality overlay unavailable" in HTML
    assert "Point cloud viewer loading" in HTML
    assert "Multi-view unavailable" in HTML


def test_benchmark_presentation_cards_are_limited_and_traces_engineering_only():
    benchmark = HTML.split('id="benchmarkSection"', 1)[1].split('id="embodimentSection"', 1)[0]
    assert '<details class="benchmark-panel benchmark-picker">' in benchmark
    assert '<details class="benchmark-panel benchmark-trial">' in benchmark
    assert "(benchmark.comparison_cards || []).slice(0, 4)" in HTML
    assert '<details class="benchmark-trace engineering-only">' in HTML


def test_operator_display_policy_is_documented_in_code():
    assert "const OPERATOR_UI_DISPLAY_POLICY" in HTML
    for field in (
        "object_pose",
        "move_guidance",
        "grasp_feasibility",
        "clutter_warning",
        "quality_applicability",
    ):
        assert field in HTML
    assert "raw_backend_fields" in HTML
    assert "evaluator_trace" in HTML


def test_operator_state_coloring_does_not_mark_negative_substrings_as_safe():
    state_class = HTML.split("function stateClass(v)", 1)[1].split("function cards", 1)[0]
    assert state_class.index("['lost'") < state_class.index("['true'")
    for negative in ("cannot", "blocked", "not ready", "not detected", "unstable"):
        assert negative in state_class


def test_dashboard_template_has_unique_ids():
    ids = re.findall(r'id="([^"]+)"', HTML)
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    assert duplicates == []


def test_dashboard_root_route_still_serves_final_tabs():
    client = web_dashboard.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "RGB-D + DAVIS Grasp Quality Demo" in body
    assert "Operator Workspace" in body
    assert "DAVIS Motion" in body


def test_realtime_status_snapshot_survives_windows_replace_read_and_stat_races(
    monkeypatch, tmp_path
):
    status_path = tmp_path / "latest_realtime_status.json"
    status_path.write_text(
        '{"writer_session_id":"writer-live","writer_mode":"backend_only","frame_id":42}',
        encoding="utf-8",
    )
    with web_dashboard.realtime_status_cache_lock:
        web_dashboard.realtime_status_cache.update(
            {"payload": None, "read_monotonic": 0.0, "source_age_sec": None}
        )

    payload, age_sec, cached = web_dashboard.read_realtime_status_snapshot(status_path)
    assert payload["writer_session_id"] == "writer-live"
    assert age_sec is not None
    assert cached is False

    original_read_json = web_dashboard.read_json
    original_stat = Path.stat

    def transient_read(path, default=None):
        if Path(path) == status_path:
            return {} if default is None else default
        return original_read_json(path, default)

    def transient_stat(path, *args, **kwargs):
        if Path(path) == status_path:
            raise FileNotFoundError(str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(web_dashboard, "read_json", transient_read)
    monkeypatch.setattr(Path, "stat", transient_stat)

    payload, age_sec, cached = web_dashboard.read_realtime_status_snapshot(status_path)
    assert payload["writer_session_id"] == "writer-live"
    assert payload["frame_id"] == 42
    assert age_sec is not None
    assert cached is True


def test_dashboard_file_metadata_is_fail_closed_during_replace_race(monkeypatch, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    image_path = output / "latest_pointcloud_preview.jpg"
    image_path.write_bytes(b"jpeg")
    original_stat = Path.stat

    def transient_stat(path, *args, **kwargs):
        if Path(path) == image_path:
            raise FileNotFoundError(str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(web_dashboard, "OUTPUT_DIR", output)
    monkeypatch.setattr(Path, "stat", transient_stat)

    assert web_dashboard.file_info(image_path.name) == {
        "exists": False,
        "url": None,
        "mtime": None,
    }
    assert web_dashboard.final_file_status(image_path) == {
        "exists": False,
        "state": "MISSING",
        "age_sec": None,
        "mtime": None,
        "size": None,
    }


def test_packaged_three_route_serves_local_script(monkeypatch, tmp_path):
    script_path = tmp_path / "three.min.js"
    script_path.write_text("window.THREE = {};", encoding="utf-8")
    monkeypatch.setattr(web_dashboard, "THREE_JS_PATH", script_path)
    response = web_dashboard.app.test_client().get("/vendor/three.min.js")
    assert response.status_code == 200
    assert response.mimetype == "application/javascript"
    assert response.get_data(as_text=True) == "window.THREE = {};"


def test_packaged_three_route_fails_cleanly_when_asset_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(web_dashboard, "THREE_JS_PATH", tmp_path / "missing.js")
    response = web_dashboard.app.test_client().get("/vendor/three.min.js")
    assert response.status_code == 404


def test_dashboard_manual_post_grasp_transition_is_explicitly_not_sensor_confirmed(monkeypatch):
    monkeypatch.setattr(
        web_dashboard,
        "send_backend_command",
        lambda command, source: f"{source}:{command}",
    )
    monkeypatch.setattr(
        web_dashboard,
        "wait_backend_command",
        lambda command_id: (
            True,
            {
                "ok": True,
                "command_id": command_id,
                "message": "Manual operator declaration accepted; post-grasp estimation started. This is not a sensor-confirmed contact.",
            },
        ),
    )
    response = web_dashboard.app.test_client().post(
        "/api/realsense-command/mark_gripper_closed"
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["command_id"] == "web_dashboard:mark_gripper_closed"
    assert payload["backend_acknowledged"] is True
    assert "Manual operator declaration accepted" in payload["message"]
    assert "sensor-confirmed contact" in payload["message"]


def test_dashboard_manual_release_transition_is_explicitly_not_sensor_confirmed(monkeypatch):
    monkeypatch.setattr(
        web_dashboard,
        "send_backend_command",
        lambda command, source: f"{source}:{command}",
    )
    monkeypatch.setattr(
        web_dashboard,
        "wait_backend_command",
        lambda command_id: (
            True,
            {
                "ok": True,
                "command_id": command_id,
                "message": "Manual release declaration accepted; returned to pre-close monitoring without claiming sensor-confirmed release.",
            },
        ),
    )
    response = web_dashboard.app.test_client().post(
        "/api/realsense-command/mark_gripper_open"
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["command_id"] == "web_dashboard:mark_gripper_open"
    assert payload["backend_acknowledged"] is True
    assert "Manual release declaration accepted" in payload["message"]
    assert "sensor-confirmed release" in payload["message"]


def test_dashboard_manual_post_grasp_rejection_is_not_reported_as_accepted(monkeypatch):
    monkeypatch.setattr(
        web_dashboard,
        "send_backend_command",
        lambda command, source: f"{source}:{command}",
    )
    monkeypatch.setattr(
        web_dashboard,
        "wait_backend_command",
        lambda command_id: (
            False,
            {
                "ok": False,
                "command_id": command_id,
                "message": "Manual post-grasp transition rejected: a current locked target is required.",
            },
        ),
    )

    response = web_dashboard.app.test_client().post(
        "/api/realsense-command/mark_gripper_closed"
    )
    payload = response.get_json()

    assert response.status_code == 409
    assert payload["ok"] is False
    assert payload["backend_acknowledged"] is True
    assert "rejected" in payload["message"]


def test_davis_tab_explains_uncalibrated_post_grasp_native_roi():
    davis = HTML.split('id="davisMotionSection"', 1)[1].split(
        'id="benchmarkSection"', 1
    )[0]

    assert "used only for post-grasp movement monitoring" in davis
    assert "confirmed DAVIS APS bottle contour" in davis
    assert "No RealSense-to-DAVIS calibration" in davis
    assert "not physically calibrated" in davis
    assert "Diagnostic Pattern" in HTML
    assert "Pattern Attribution" in HTML
    assert "Raw Poll Events" in HTML
    assert "Global / Object Window" in HTML
    assert "Event Timestamp" in HTML


def test_compact_status_get_is_read_only_cached_and_under_operator_budget(monkeypatch):
    compact = web_dashboard.compact_operator_status_payload({
        "time": 1.0,
        "running": {"realsense": True, "davis_motion": True},
        "realsense": {
            "writer_session_id": "writer-1",
            "target_status": "TRACKING",
            "grasp_band_locked": True,
            "quality_matrix": {"quality_matrix_status": "updated", "raw_epsilon": 0.12},
        },
        "hand_prototype": {"grasp_band_locked": True},
        "davis_motion": {"current_stage": "PRE_CLOSE", "motion_class": "LIGHTING CHANGE"},
        "live_quality": {"decision": "HOLD", "timeline_recent": list(range(20))},
        "images": {"overlay": {"exists": True}, "davis_heatmap": {"exists": True}},
        "quality_bundle": {"quality_bundle_status": "OK"},
    })
    monkeypatch.setitem(web_dashboard.operator_status_snapshot_cache, "payload", compact)
    monkeypatch.setitem(web_dashboard.operator_status_snapshot_cache, "timestamp", time.time())
    monkeypatch.setitem(web_dashboard.derived_status_health, "last_success_time", time.time())
    for name in (
        "run_file_sync_once",
        "ensure_davis_confirmation_outputs",
        "write_live_quality_outputs",
        "common_write_json_atomic",
    ):
        monkeypatch.setattr(
            web_dashboard,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"compact GET called writer {_name}")
            ),
        )

    response = web_dashboard.app.test_client().get("/api/status?compact=1")

    assert response.status_code == 200
    assert len(response.data) < 25_000
    assert response.get_json()["realsense"]["writer_session_id"] == "writer-1"
    assert response.get_json()["live_quality"]["timeline_recent"] == list(range(17, 20))


def test_compact_contract_preserves_fixed_zone_post_grasp_semantics():
    compact = web_dashboard.compact_operator_status_payload({
        "running": {"realsense": True, "davis_motion": True},
        "realsense": {"quality_matrix": {}},
        "davis_motion": {
            "current_stage": "POST_GRASP",
            "grasp_zone_motion_valid": True,
            "grasp_zone_motion_class": "MICRO SLIP",
            "motion_attribution": "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED",
            "fixed_center_third_acquisition": True,
            "roi_projection_uncalibrated": True,
            "davis_target_context_mode": "fixed_center_third_with_aps_object_evidence",
            "davis_target_roi_priority_selected": "davis_aps_object_evidence_within_configured_center_third",
            "davis_motion_excluded_reason": None,
        },
        "live_quality": {"live_quality_explanation": "Post-grasp slip reduced the operational score."},
    })

    motion = compact["davis_motion"]
    assert motion["grasp_zone_motion_valid"] is True
    assert motion["grasp_zone_motion_class"] == "MICRO SLIP"
    assert motion["motion_attribution"] == "POST_GRASP_FIXED_GRASP_ZONE_UNCALIBRATED"
    assert motion["fixed_center_third_acquisition"] is True
    assert compact["live_quality"]["live_quality_explanation"].startswith("Post-grasp")
    assert "dashboardMode !== 'engineering'" in HTML
    assert "setMainVisualImages(data);" in HTML


def test_stalled_operator_snapshot_fails_closed_instead_of_preserving_green_state(monkeypatch):
    stale_payload = {
        "running": {"realsense": True, "davis_motion": True},
        "realsense": {
            "overlay_stale": False,
            "overlay_health": "OK",
            "ready_to_close": True,
            "distance_measurement_valid": True,
            "grasp_band_current": True,
            "grasp_band_projected": True,
            "operator_locked_fraction_visual_guide_current": True,
        },
        "images": {"overlay": {"stale": False, "health": "OK"}},
    }
    monkeypatch.setitem(web_dashboard.operator_status_snapshot_cache, "payload", stale_payload)
    monkeypatch.setitem(
        web_dashboard.operator_status_snapshot_cache,
        "timestamp",
        time.time() - web_dashboard.REALTIME_STATUS_CACHE_GRACE_SEC - 1.0,
    )
    monkeypatch.setitem(web_dashboard.derived_status_health, "last_success_time", 0.0)
    monkeypatch.setattr(
        web_dashboard,
        "build_compact_operator_status_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("producer frozen")),
    )

    payload = web_dashboard.cached_compact_operator_status_snapshot()

    assert payload["operator_snapshot_stale"] is True
    assert payload["running"] == {"realsense": False, "davis_motion": False}
    assert payload["realsense"]["overlay_stale"] is True
    assert payload["realsense"]["ready_to_close"] is False
    assert payload["realsense"]["distance_measurement_valid"] is False
    assert payload["realsense"]["grasp_band_current"] is False
    assert payload["images"]["overlay"]["stale"] is True


def test_readable_old_artifacts_cannot_refresh_a_dead_derived_producer(monkeypatch):
    old = {
        "running": {"realsense": True, "davis_motion": True},
        "realsense": {"target_status": "TRACKING", "overlay_stale": False},
        "live_quality": {
            "live_quality_status": "OK",
            "live_quality_label": "GOOD",
            "decision": "LIFT",
            "davis_motion_used_for_live_quality": True,
        },
        "combined_quality": {"combined_quality_status": "GOOD"},
        "sensor_sync": {"sync_quality": "good"},
        "davis_motion": {"davis_closeup_used_for_live_quality": True},
    }
    monkeypatch.setitem(web_dashboard.operator_status_snapshot_cache, "payload", old)
    monkeypatch.setitem(web_dashboard.operator_status_snapshot_cache, "timestamp", time.time() - 10.0)
    monkeypatch.setitem(web_dashboard.derived_status_health, "last_success_time", time.time() - 10.0)
    monkeypatch.setattr(web_dashboard, "build_compact_operator_status_snapshot", lambda: old)

    first = web_dashboard.cached_compact_operator_status_snapshot()
    second = web_dashboard.cached_compact_operator_status_snapshot()

    for payload in (first, second):
        assert payload["operator_snapshot_stale"] is True
        assert payload["running"]["realsense"] is True
        assert payload["live_quality"]["live_quality_status"] == "UNAVAILABLE"
        assert payload["live_quality"]["davis_motion_used_for_live_quality"] is False
        assert payload["combined_quality"]["combined_quality_status"] == "UNAVAILABLE"
        assert payload["sensor_sync"]["sync_quality"] == "stale"
        assert payload["davis_motion"]["davis_closeup_used_for_live_quality"] is False


def test_all_status_get_routes_are_read_only_and_producer_owns_derived_writes():
    route_source = inspect.getsource(web_dashboard.api_status)
    assert "run_file_sync_once(" not in route_source
    assert "ensure_davis_confirmation_outputs(" not in route_source
    assert "write_live_quality_outputs(" not in route_source
    producer_source = inspect.getsource(web_dashboard.refresh_derived_status_outputs_once)
    assert "run_file_sync_once(" in producer_source
    assert "ensure_davis_confirmation_outputs(" in producer_source
    assert "write_live_quality_outputs(" in producer_source
    assert producer_source.index("baseline_from_post_grasp_snapshot(") < producer_source.index(
        "baseline_from_quality_matrix("
    )

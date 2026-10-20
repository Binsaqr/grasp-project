"""Camera-free tests for projected hand-reach guidance."""

import json
import inspect

import numpy as np
import pregrasp_planner

from grasp_project.common.config import load_named_config
from grasp_project.dashboard import web_dashboard
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.hand_mount_guidance import (
    DEFAULT_CAMERA_TO_HAND_OFFSET_MM,
    DEFAULT_DESIRED_CONTACT_GAP_MM,
    hand_to_locked_band_error,
    project_hand_reach_point,
    validate_hand_offset_update,
)


INTRINSICS = {"fx": 600.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
OFFSET = {"x": -70.0, "y": 80.0, "z": 170.0}


def test_starting_offset_and_uncertainty_are_configured_honestly():
    config = load_named_config("hand_mounted_camera.yaml")
    assert config["camera_to_hand_offset_mm"] == OFFSET
    assert config["camera_to_hand_offset_uncertainty_mm"] == {"x": 15.0, "y": 10.0, "z": 10.0}
    assert config["hand_offset_physically_validated"] is False
    assert "APPROXIMATE" in config["hand_offset_calibration_status"]
    assert "NOT_PHYSICALLY" in config["hand_offset_calibration_status"]
    assert DEFAULT_CAMERA_TO_HAND_OFFSET_MM == OFFSET
    assert DEFAULT_DESIRED_CONTACT_GAP_MM == 10.0
    assert web_dashboard.HAND_OFFSET_RESET_DEFAULTS["desired_close_distance_mm"] == 180.0
    assert web_dashboard.HAND_OFFSET_RESET_DEFAULTS["min_close_distance_mm"] == 160.0
    assert web_dashboard.HAND_OFFSET_RESET_DEFAULTS["max_close_distance_mm"] == 220.0


def test_projection_uses_fixed_hand_contact_depth_not_bottle_depth():
    projected = project_hand_reach_point(INTRINSICS, 500.0, OFFSET, image_shape=(480, 640))
    assert projected["valid"] is True
    expected_x = 320.0 + 600.0 * -70.0 / 170.0
    expected_y = 240.0 + 500.0 * 80.0 / 170.0
    assert np.allclose(
        projected["offset_px"],
        [600.0 * -70.0 / 170.0, 500.0 * 80.0 / 170.0],
    )
    assert np.allclose(projected["point_px"], [expected_x, expected_y])
    assert projected["projection_depth_mm"] == 170.0
    assert projected["target_depth_mm"] == 500.0
    assert projected["visible"] is True

    closer = project_hand_reach_point(INTRINSICS, 300.0, OFFSET, image_shape=(480, 640))
    closest = project_hand_reach_point(INTRINSICS, 200.0, OFFSET, image_shape=(480, 640))
    assert np.allclose(closer["point_px"], projected["point_px"])
    assert np.allclose(closest["point_px"], projected["point_px"])
    assert closer["target_depth_mm"] == 300.0
    assert closest["target_depth_mm"] == 200.0

    farther_hand_point = project_hand_reach_point(
        INTRINSICS,
        500.0,
        {"x": -70.0, "y": 80.0, "z": 200.0},
    )
    assert farther_hand_point["point_px"][1] < projected["point_px"][1]


def test_nonzero_lateral_hand_offset_requires_positive_contact_depth():
    invalid = project_hand_reach_point(
        INTRINSICS,
        500.0,
        {"x": 0.0, "y": 100.0, "z": 0.0},
    )
    assert invalid["valid"] is False
    assert invalid["point_px"] is None
    assert invalid["reason"] == "invalid_hand_contact_point_depth"


def test_prelock_band_ranking_uses_same_fixed_reach_pixel_at_every_bottle_depth():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[40:440, 260:380] = 1
    pose = {
        "cleaned_mask": mask,
        "body_mask": mask,
        "center_px": [319.5, 239.5],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": 399.0,
    }
    selected = []
    projected_points = []
    for target_depth in (500.0, 300.0, 200.0):
        depth = np.zeros_like(mask, dtype=np.uint16)
        depth[mask > 0] = int(target_depth)
        projection = project_hand_reach_point(
            INTRINSICS,
            target_depth,
            OFFSET,
            image_shape=mask.shape,
        )
        plan, error = pregrasp_planner.plan_pregrasp(
            pose,
            depth,
            INTRINSICS,
            preferred_contact_point_px=projection["point_px"],
        )
        assert error is None
        assert projection["valid"] is True
        assert projection["visible"] is True
        projected_points.append(projection["point_px"])
        selected.append(plan["selected_requested_body_fraction"])

    assert np.allclose(projected_points, [projected_points[0]] * 3)
    assert selected == [selected[0]] * 3
    assert selected[0] == 0.74


def test_zero_or_disabled_offset_falls_back_to_camera_center():
    zero = project_hand_reach_point(INTRINSICS, 600.0, {"x": 0, "y": 0, "z": 0})
    disabled = project_hand_reach_point(INTRINSICS, 600.0, OFFSET, enabled=False)
    assert zero["point_px"] == [320.0, 240.0]
    assert disabled["point_px"] == [320.0, 240.0]


def test_alignment_error_is_band_minus_hand_reach_with_camera_frame_signs():
    error = hand_to_locked_band_error(
        [350.0, 320.0],
        [320.0, 340.0],
        600.0,
        INTRINSICS,
        OFFSET,
        20.0,
    )
    assert error["valid"] is True
    assert error["error_px"] == [30.0, -20.0]
    assert error["error_mm"] == [30.0, -24.0]
    assert error["hand_to_bottle_z_error_mm"] == 410.0
    assert error["recommended_move_z_mm"] == -410.0


def test_invalid_depth_never_emits_precise_hand_guidance():
    projected = project_hand_reach_point(INTRINSICS, 0.0, OFFSET)
    error = hand_to_locked_band_error([350.0, 320.0], None, None, INTRINSICS, OFFSET, 20.0)
    assert projected["valid"] is False
    assert projected["point_px"] is None
    assert error["valid"] is False
    assert error["error_mm"] is None
    assert error.get("recommended_move_z_mm") is None


def test_offset_validation_rejects_nonfinite_and_out_of_range_values():
    assert validate_hand_offset_update(OFFSET, 20.0)[0] is True
    assert validate_hand_offset_update({"x": "nan", "y": 120, "z": 240}, 20.0)[0] is False
    assert validate_hand_offset_update({"x": 0, "y": 120, "z": 700}, 20.0)[0] is False
    assert validate_hand_offset_update(OFFSET, 31.0)[0] is False


def test_overlay_draws_reach_marker_and_arrow_to_band_without_moving_band(monkeypatch):
    arrows = []
    markers = []
    monkeypatch.setattr(realtime_runner.cv2, "arrowedLine", lambda *args, **kwargs: arrows.append(args))
    monkeypatch.setattr(realtime_runner.cv2, "drawMarker", lambda *args, **kwargs: markers.append(args))
    monkeypatch.setattr(realtime_runner, "SHOW_MAIN_MOVEMENT_ARROW", True)
    monkeypatch.setattr(realtime_runner, "HAND_MOUNTED_CAMERA_MODE", True)
    band = [[90, 90], [110, 90], [110, 110], [90, 110]]
    report = {
        "target_clamp": {"clamp_center_px": [100.0, 100.0], "clamp_band_polygon": band},
        "overlay_diagnostics_enabled": True,
        "target_lock": {"target_status": "TRACKING", "approach_mode_requested": True},
        "approach_mode_requested": True,
        "distance_measurement_valid": True,
        "approach_action": "CENTER_TARGET",
        "recommended_move_x_mm": 20.0,
        "recommended_move_y_mm": 0.0,
        "hand_reach_projection_px": [60.0, 100.0],
        "hand_reach_projection_valid": True,
    }

    realtime_runner.draw_fast_overlay(np.zeros((200, 200, 3), dtype=np.uint8), None, None, report)

    assert report["target_clamp"]["clamp_band_polygon"] == band
    assert arrows[0][1] == (60, 100)
    assert arrows[0][2] == (100, 100)
    assert report["hand_reach_marker_visible"] is True
    assert report["hand_alignment_arrow_visible"] is True


def test_yaml_persistence_and_calibration_artifact_are_atomic_and_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(web_dashboard, "HAND_MOUNTED_CAMERA_CONFIG", dict(web_dashboard.HAND_MOUNTED_CAMERA_CONFIG))
    config_path = tmp_path / "hand_mounted_camera.yaml"
    artifact_path = tmp_path / "hand_mount_offset.json"
    config_path.write_text(
        "camera_to_hand_offset_enabled: true\n"
        "camera_to_hand_offset_mm:\n  x: 0.0\n  y: 100.0\n  z: 240.0\n"
        "desired_contact_gap_mm: 20.0\n"
        "hand_offset_calibration_status: APPROXIMATE_NOT_PHYSICALLY_CALIBRATED\n"
        "hand_offset_physically_validated: false\n"
        "hand_offset_source: initial\n"
    )
    artifact = web_dashboard.persist_hand_camera_offset(
        {"x": 12.0, "y": 98.0, "z": 255.0},
        15.0,
        path=config_path,
        calibration_path=artifact_path,
    )
    saved = config_path.read_text()
    assert "  x: 12.0" in saved
    assert "  y: 98.0" in saved
    assert "  z: 255.0" in saved
    assert "desired_contact_gap_mm: 15.0" in saved
    assert artifact["physically_validated"] is False
    assert json.loads(artifact_path.read_text())["camera_to_hand_offset_mm"]["z"] == 255.0


def test_hand_projection_does_not_modify_analytic_epsilon_inputs():
    helper_source = inspect.getsource(project_hand_reach_point) + inspect.getsource(hand_to_locked_band_error)
    assert "epsilon" not in helper_source.lower()


def test_realtime_guidance_uses_hand_projection_and_keeps_locked_band(monkeypatch):
    config = dict(realtime_runner.HAND_MOUNTED_CAMERA_CONFIG)
    config.update({
        "camera_to_hand_offset_enabled": True,
        "camera_to_hand_offset_mm": OFFSET,
        "desired_contact_gap_mm": 20.0,
    })
    monkeypatch.setattr(realtime_runner, "HAND_MOUNTED_CAMERA_CONFIG", config)
    depth = np.full((300, 300), 1200, dtype=np.uint16)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[25:276, 75:126] = 1
    depth[mask > 0] = 600
    clamp = {
        "left_clamp_patch": {"center_px": [80, 100], "polygon": [[74, 90], [86, 90], [86, 110], [74, 110]]},
        "right_clamp_patch": {"center_px": [120, 100], "polygon": [[114, 90], [126, 90], [126, 110], [114, 110]]},
        "clamp_center_px": [100, 100],
        "clamp_band_polygon": [[74, 88], [126, 88], [126, 112], [74, 112]],
        "grip_width_px": 40,
    }
    epsilon = {"force_closure": True, "epsilon": 0.25}
    epsilon_before = dict(epsilon)
    guidance = realtime_runner.compute_approach_guidance(
        {"center_px": [100, 100], "depth_median_mm": 600, "body_mask": mask, "cleaned_mask": mask},
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 100.0, "cy": 100.0},
        {},
        {
            "target_clamp": clamp,
            "locked_patch_visible": True,
            "bottle_geometry_tracking_ok": True,
            "identity_measurement_current": True,
            "current_grip_width_px": 40,
        },
        depth.shape,
        epsilon_result=epsilon,
    )
    expected_reach_x = 100.0 + 500.0 * -70.0 / 170.0
    expected_reach_y = 100.0 + 500.0 * 80.0 / 170.0
    expected_error_x_px = 100.0 - expected_reach_x
    expected_error_y_px = 100.0 - expected_reach_y
    assert np.allclose(guidance["hand_reach_projection_px"], [expected_reach_x, expected_reach_y])
    assert np.allclose(guidance["hand_to_grasp_error_px"], [expected_error_x_px, expected_error_y_px])
    assert np.allclose(
        guidance["hand_to_grasp_error_mm"],
        [expected_error_x_px * 600.0 / 500.0, expected_error_y_px * 600.0 / 500.0],
    )
    assert guidance["hand_projection_depth_mm"] == 170.0
    assert guidance["hand_projection_target_depth_mm"] == 600.0
    assert guidance["guidance_basis"] == "hand_projection_to_locked_band"
    assert guidance["target_x_px"] == clamp["clamp_center_px"][0]
    assert guidance["target_y_px"] == clamp["clamp_center_px"][1]
    assert epsilon == epsilon_before


def test_hand_offset_api_persists_and_sends_live_backend_command(monkeypatch):
    saved = {}
    sent = {}

    def fake_persist(offset, gap, **kwargs):
        saved.update({"offset": offset, "gap": gap, **kwargs})
        return {"camera_to_hand_offset_mm": offset, "desired_contact_gap_mm": gap, "physically_validated": False}

    def fake_send(command, source="web_dashboard", **fields):
        sent.update({"command": command, "source": source, **fields})
        return "offset-command"

    monkeypatch.setattr(web_dashboard, "persist_hand_camera_offset", fake_persist)
    monkeypatch.setattr(web_dashboard, "send_backend_command", fake_send)
    monkeypatch.setattr(web_dashboard, "wait_backend_command", lambda command_id: (True, {"command_id": command_id, "ok": True}))
    response = web_dashboard.app.test_client().post("/api/hand-camera-offset", json={
        "camera_to_hand_offset_mm": {"x": 3, "y": 118, "z": 245},
        "desired_contact_gap_mm": 12,
    })
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert saved["offset"] == {"x": 3.0, "y": 118.0, "z": 245.0}
    assert sent["command"] == "set_hand_camera_offset"
    assert sent["camera_to_hand_offset_mm"] == saved["offset"]
    effective = response.get_json()["effective_guidance"]
    assert effective["desired_close_distance_mm"] == 257.0
    assert effective["min_close_distance_mm"] == 237.0
    assert effective["max_close_distance_mm"] == 297.0


def test_dashboard_exposes_live_xyz_tuning_and_hand_guidance_fields():
    html = web_dashboard.INDEX_HTML
    for element_id in ("handOffsetX", "handOffsetY", "handOffsetZ", "handContactGap"):
        assert f'id="{element_id}"' in html
    assert "/api/hand-camera-offset" in html
    summary = web_dashboard.summarize_realsense({
        "camera_to_hand_offset_mm": OFFSET,
        "hand_reach_projection_px": [320.0, 340.0],
        "hand_reach_projection_valid": True,
        "hand_alignment_error_mm": [10.0, -20.0],
        "hand_to_bottle_z_error_mm": 40.0,
    })
    assert summary["camera_to_hand_offset_mm"] == OFFSET
    assert summary["hand_reach_projection_valid"] is True
    assert summary["hand_alignment_error_mm"] == [10.0, -20.0]


def test_compact_current_view_calibration_uses_fixed_hand_depth_only():
    fixed_evidence = {
        "hand_projection_depth_mm": 170.0,
        "hand_projection_target_depth_mm": 500.0,
        "camera_optical_center_px": [320.0, 240.0],
        "fx": 600.0,
        "fy": 500.0,
        "left_finger_detected": True,
        "right_finger_detected": True,
        "left_inner_edge": [300.0, 390.0],
        "right_inner_edge": [340.0, 390.0],
    }
    first = web_dashboard.compact_operator_status_payload({
        "realsense": {**fixed_evidence, "bottle_distance_mm": 500.0},
    })["realsense"]
    second = web_dashboard.compact_operator_status_payload({
        "realsense": {
            **fixed_evidence,
            "hand_projection_target_depth_mm": 200.0,
            "bottle_distance_mm": 200.0,
        },
    })["realsense"]

    def estimated_xy(status):
        midpoint = np.mean(
            [status["left_inner_edge"], status["right_inner_edge"]], axis=0
        )
        center = np.asarray(status["camera_optical_center_px"], dtype=float)
        return (
            (midpoint - center)
            * float(status["hand_projection_depth_mm"])
            / np.asarray([status["fx"], status["fy"]], dtype=float)
        )

    assert np.allclose(estimated_xy(first), estimated_xy(second))
    assert first["hand_projection_depth_mm"] == second["hand_projection_depth_mm"] == 170.0
    assert first["hand_projection_target_depth_mm"] == 500.0
    assert second["hand_projection_target_depth_mm"] == 200.0

    calibration_source = web_dashboard.INDEX_HTML.split(
        "async function setHandOffsetFromCurrentView()", 1
    )[1].split("let handDemoCaptureStream", 1)[0]
    assert "Number(rs.hand_projection_depth_mm)" in calibration_source
    assert "distance_to_locked_band_mm" not in calibration_source
    assert "bottle_distance_mm" not in calibration_source
    assert "fixed hand-contact projection depth" in calibration_source


def test_davis_mapping_remains_based_on_locked_bottle_band_not_hand_reach():
    source = inspect.getsource(web_dashboard.map_realsense_bbox_to_davis_roi)
    assert "hand_reach_projection" not in source

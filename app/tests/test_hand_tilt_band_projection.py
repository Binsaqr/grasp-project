"""Camera-free pitch tests for bottle-local 3D band reprojection."""

import copy

import cv2
import numpy as np
import pytest

from grasp_project.realsense.hand_mount_guidance import project_hand_reach_point
from grasp_project.realsense.realtime_runner import (
    create_locked_grasp_band_3d,
    project_locked_grasp_band_to_current_frame,
)


INTRINSICS = {"fx": 240.0, "fy": 240.0, "cx": 80.0, "cy": 80.0, "source": "synthetic"}


def _pitch_rgbd(pitch_deg, center_depth_mm=480.0):
    """Render a planar bottle face rotated about its physical 3D centre."""
    theta = np.radians(float(pitch_deg))
    cosine, sine = float(np.cos(theta)), float(np.sin(theta))

    def camera_point(x_mm, y_mm):
        return np.asarray([x_mm, cosine * y_mm, center_depth_mm + sine * y_mm], dtype=float)

    def pixel(point):
        return [
            INTRINSICS["fx"] * point[0] / point[2] + INTRINSICS["cx"],
            INTRINSICS["fy"] * point[1] / point[2] + INTRINSICS["cy"],
        ]

    corners = np.asarray([
        pixel(camera_point(-40.0, -100.0)),
        pixel(camera_point(40.0, -100.0)),
        pixel(camera_point(40.0, 100.0)),
        pixel(camera_point(-40.0, 100.0)),
    ])
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 255)
    yy, _xx = np.indices(mask.shape)
    denominator = cosine - sine * (yy - INTRINSICS["cy"]) / INTRINSICS["fy"]
    z = cosine * center_depth_mm / np.maximum(denominator, 1e-6)
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = np.round(z[mask > 0]).astype(np.uint16)
    top = np.asarray(pixel(camera_point(0.0, -100.0)))
    bottom = np.asarray(pixel(camera_point(0.0, 100.0)))
    axis = bottom - top
    axis /= np.linalg.norm(axis)
    center_px = pixel(camera_point(0.0, 0.0))
    x, y, width, height = cv2.boundingRect(np.round(corners).astype(np.int32))
    pose = {
        "detected": True,
        "target_id": "pitch-bottle",
        "center_px": center_px,
        "axis_vector": axis.tolist(),
        "axis_length_px": float(np.linalg.norm(bottom - top)),
        "axis_angle_degrees": float(np.degrees(np.arctan2(axis[1], axis[0]))),
        "bbox_xyxy": [x, y, x + width, y + height],
        "bbox_xywh": [x, y, width, height],
        "oriented_bbox": corners.tolist(),
        "cleaned_mask": mask,
        "body_mask": mask,
        "depth_median_mm": float(np.median(depth[depth > 0])),
        "detection_confidence": 0.98,
    }
    return pose, depth, camera_point, pixel


def _locked_pitch_band():
    pose, depth, camera_point, pixel = _pitch_rgbd(0.0)
    left = pixel(camera_point(-34.0, 0.0))
    right = pixel(camera_point(34.0, 0.0))
    band = [
        pixel(camera_point(-40.0, -20.0)),
        pixel(camera_point(40.0, -20.0)),
        pixel(camera_point(40.0, 20.0)),
        pixel(camera_point(-40.0, 20.0)),
    ]
    left_patch = [
        pixel(camera_point(-40.0, -20.0)),
        pixel(camera_point(-26.0, -20.0)),
        pixel(camera_point(-26.0, 20.0)),
        pixel(camera_point(-40.0, 20.0)),
    ]
    right_patch = [
        pixel(camera_point(26.0, -20.0)),
        pixel(camera_point(40.0, -20.0)),
        pixel(camera_point(40.0, 20.0)),
        pixel(camera_point(26.0, 20.0)),
    ]
    plan = {
        "grasp_center_px": [80.0, 80.0],
        "body_fraction": 0.5,
        "left_contact_px": left,
        "right_contact_px": right,
        "planned_depth_mm": 480.0,
        "grip_width_px": float(right[0] - left[0]),
        "grip_width_mm": 68.0,
    }
    target = {
        "valid": True,
        "attempt_id": "pitch-lock",
        "selected_candidate_id": "middle",
        "bottle_target_id": "pitch-bottle",
        "lock_frame_id": 1,
        "lock_timestamp": "2026-08-12T00:00:00Z",
        "locked_body_fraction": 0.5,
        "locked_grip_width_mm": 68.0,
        "fixed_clamp_zone": {
            "clamp_center_px": [80.0, 80.0],
            "clamp_band_width_px": float(right[0] - left[0]),
            "clamp_band_height_px": 20.0,
            "left_clamp_patch": {"center_px": left, "polygon": left_patch},
            "right_clamp_patch": {"center_px": right, "polygon": right_patch},
            "clamp_band_polygon": band,
        },
        "bottle_contact_model": {
            "body_fraction": 0.5,
            "left_contact_patch": {"center_px": left, "polygon": left_patch},
            "right_contact_patch": {"center_px": right, "polygon": right_patch},
            "contact_band_polygon": band,
            "grip_width_px": float(right[0] - left[0]),
            "grip_width_mm": 68.0,
            "patch_height_px": 20.0,
        },
    }
    locked = create_locked_grasp_band_3d(target, pose, plan, depth, INTRINSICS)
    assert locked.lock_valid is True
    return locked, pose, depth


@pytest.mark.parametrize("pitch_deg", [18.0, -18.0])
def test_pitch_projection_preserves_same_bottle_local_band(pitch_deg):
    locked, _lock_pose, _lock_depth = _locked_pitch_band()
    pose, depth, _camera_point, _pixel = _pitch_rgbd(pitch_deg)

    projected = project_locked_grasp_band_to_current_frame(locked, pose, depth, INTRINSICS)

    assert projected["projection_valid"] is True
    assert projected["projection_source"] == "BOTTLE_LOCAL_3D_REPROJECTION"
    assert projected["projected_body_fraction"] == pytest.approx(0.5)
    assert projected["band_reselected_this_frame"] is False
    current_frame = projected["current_bottle_frame_3d"]
    projected_center_xyz = np.asarray(projected["projected_band_centroid_xyz_mm"])
    origin = np.asarray(current_frame["origin_xyz_mm"])
    axis = np.asarray(current_frame["axis_unit"])
    current_axis_coordinate = float(np.dot(projected_center_xyz - origin, axis))
    assert current_axis_coordinate == pytest.approx(
        locked.locked_band_center_local_xyz[1],
        abs=3.0,
    )


def test_pitch_up_projection_keeps_same_body_fraction():
    locked, _lock_pose, _lock_depth = _locked_pitch_band()
    pose, depth, _camera_point, _pixel = _pitch_rgbd(20.0)
    projected = project_locked_grasp_band_to_current_frame(locked, pose, depth, INTRINSICS)
    assert projected["projection_valid"] is True
    assert projected["projected_body_fraction"] == pytest.approx(locked.locked_body_fraction)


def test_pitch_down_projection_keeps_same_body_fraction():
    locked, _lock_pose, _lock_depth = _locked_pitch_band()
    pose, depth, _camera_point, _pixel = _pitch_rgbd(-20.0)
    projected = project_locked_grasp_band_to_current_frame(locked, pose, depth, INTRINSICS)
    assert projected["projection_valid"] is True
    assert projected["projected_body_fraction"] == pytest.approx(locked.locked_body_fraction)


def test_hand_offset_changes_guidance_not_band():
    locked, pose, depth = _locked_pitch_band()
    band_before = project_locked_grasp_band_to_current_frame(locked, pose, depth, INTRINSICS)
    reach_zero = project_hand_reach_point(INTRINSICS, 480.0, {"x": 0, "y": 0, "z": 0})
    reach_offset = project_hand_reach_point(INTRINSICS, 480.0, {"x": 0, "y": 120, "z": 240})
    band_after = project_locked_grasp_band_to_current_frame(locked, copy.deepcopy(pose), depth, INTRINSICS)

    assert reach_zero["point_px"] != reach_offset["point_px"]
    assert band_after["projected_band_polygon_px"] == band_before["projected_band_polygon_px"]
    assert band_after["projected_band_center_px"] == band_before["projected_band_center_px"]


def test_zero_hand_offset_falls_back_camera_center():
    reach = project_hand_reach_point(INTRINSICS, 480.0, {"x": 0, "y": 0, "z": 0})
    assert reach["valid"] is True
    assert reach["point_px"] == [INTRINSICS["cx"], INTRINSICS["cy"]]


import numpy as np
import pytest

from grasp_project.realsense.grasp_band_3d import (
    camera_to_local,
    deproject_mask_mm,
    fit_bottle_local_frame,
    local_to_camera,
    project_camera_xyz_mm,
)


INTRINSICS = {"fx": 200.0, "fy": 200.0, "cx": 80.0, "cy": 80.0}


def test_mask_deprojection_returns_every_valid_selected_pixel():
    depth = np.zeros((160, 160), dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[20:140, 55:105] = 255
    depth[mask > 0] = 500
    depth[40, 60] = 0

    points, pixels, metadata = deproject_mask_mm(depth, INTRINSICS, mask)

    assert metadata["mask_pixel_count"] == 6000
    assert metadata["valid_point_count"] == 5999
    assert len(points) == len(pixels) == 5999
    assert np.allclose(points[:, 2], 500.0)


def test_fitted_frame_round_trips_camera_and_local_points():
    depth = np.zeros((160, 160), dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[20:140, 55:105] = 255
    depth[mask > 0] = 500
    points, _, metadata = deproject_mask_mm(depth, INTRINSICS, mask)
    frame = fit_bottle_local_frame(
        points,
        INTRINSICS,
        axis_px=[0.0, 1.0],
        depth_valid_ratio=metadata["depth_valid_ratio"],
    )

    assert frame["valid"] is True
    assert frame["point_count"] == 6000
    local = camera_to_local(points[::100], frame)
    restored = local_to_camera(local, frame)
    assert np.allclose(restored, points[::100], atol=1e-6)


def test_metric_geometry_projects_larger_at_smaller_depth():
    points_far = np.asarray([[-30.0, 0.0, 600.0], [30.0, 0.0, 600.0]])
    points_near = np.asarray([[-30.0, 0.0, 300.0], [30.0, 0.0, 300.0]])

    far_px = project_camera_xyz_mm(points_far, INTRINSICS)
    near_px = project_camera_xyz_mm(points_near, INTRINSICS)

    far_width = far_px[1, 0] - far_px[0, 0]
    near_width = near_px[1, 0] - near_px[0, 0]
    assert near_width == pytest.approx(2.0 * far_width)


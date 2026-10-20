"""Focused fail-closed tests for DAVIS-to-RealSense calibration files."""

import json

import numpy as np
import pytest

from grasp_project.sync.calibrate_2d import estimate_homography, main as calibrate_2d_main
from grasp_project.sync.calibration_status import calibration_status


def _points():
    davis = [[10, 10], [100, 10], [200, 20], [20, 100], [120, 120], [220, 140], [40, 220], [180, 230]]
    realsense = [[1.5 * x + 20, 0.8 * y + 10] for x, y in davis]
    return davis, realsense


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([[0, 0], [1, 1], [2, 2], [3, 3]], "non-collinear"),
        ([[0, 0], [1, 0], [1, 1], [float("nan"), 1]], "finite"),
        ([[-1, 0], [1, 0], [1, 1], [0, 1]], "non-negative"),
        ([[0, 0], [1, 0], [1, 1]], "at least 4"),
    ],
)
def test_estimator_rejects_degenerate_point_arrays(points, message):
    with pytest.raises(ValueError, match=message):
        estimate_homography(points, points)


def test_estimator_rejects_points_outside_recorded_frame():
    davis, realsense = _points()
    davis[0] = [346, 10]

    with pytest.raises(ValueError, match="outside"):
        estimate_homography(
            davis,
            realsense,
            davis_frame_size=[346, 260],
            realsense_frame_size=[640, 480],
        )


def test_opencv_ransac_rejects_outlier_and_reports_held_out_error():
    pytest.importorskip("cv2")
    davis, realsense = _points()
    realsense[-1] = [500, 20]

    result = estimate_homography(
        davis,
        realsense,
        davis_frame_size=[346, 260],
        realsense_frame_size=[640, 480],
        held_out_davis_points=[[75, 55], [160, 180]],
        held_out_realsense_points=[[132.5, 54], [260, 154]],
    )

    assert result["estimation_method"] == "opencv_ransac"
    assert result["outlier_count"] >= 1
    assert result["held_out_point_count"] == 2
    assert result["held_out_reprojection_error_px"] < 0.01


def test_cli_records_held_out_error_and_leaves_no_temporary_file(tmp_path):
    davis, realsense = _points()
    input_path = tmp_path / "points.json"
    output_path = tmp_path / "homography.json"
    input_path.write_text(
        json.dumps(
            {
                "davis_points": davis,
                "realsense_points": realsense,
                "held_out_davis_points": [[75, 55]],
                "held_out_realsense_points": [[132.5, 54]],
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )

    assert calibrate_2d_main(["--input", str(input_path), "--output", str(output_path)]) == 0
    payload = json.loads(output_path.read_text())
    assert payload["held_out_point_count"] == 1
    assert payload["held_out_reprojection_error_px"] < 0.01
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_cli_does_not_write_when_a_point_is_out_of_bounds(tmp_path):
    davis, realsense = _points()
    davis[0] = [346, 10]
    input_path = tmp_path / "points.json"
    output_path = tmp_path / "homography.json"
    input_path.write_text(
        json.dumps(
            {
                "davis_points": davis,
                "realsense_points": realsense,
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )

    assert calibrate_2d_main(["--input", str(input_path), "--output", str(output_path)]) == 2
    assert not output_path.exists()


@pytest.mark.parametrize(
    "homography",
    [None, [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, np.nan, 0], [0, 0, 1]]],
)
def test_status_rejects_missing_nonfinite_or_singular_homography(tmp_path, homography):
    path = tmp_path / "homography.json"
    path.write_text(
        json.dumps(
            {
                "homography": homography,
                "reprojection_error_px": 0.5,
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )

    status = calibration_status(path, tmp_path / "missing_extrinsics.json")

    assert status["homography_file_exists"] is True
    assert status["homography_available"] is False
    assert status["usable_for_overlay"] is False
    assert status["alignment_status"] == "approximate_side_by_side_only"
    assert status["homography_validation_status"] == "missing_or_invalid_homography"


def test_status_rejects_failing_held_out_error(tmp_path):
    path = tmp_path / "homography.json"
    path.write_text(
        json.dumps(
            {
                "homography": np.eye(3).tolist(),
                "reprojection_error_px": 0.5,
                "held_out_point_count": 2,
                "held_out_reprojection_error_px": 12.0,
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )

    status = calibration_status(path, tmp_path / "missing_extrinsics.json")

    assert status["homography_available"] is True
    assert status["homography_valid"] is False
    assert status["usable_for_overlay"] is False
    assert status["homography_validation_status"] == "missing_or_invalid_held_out_reprojection_error"


def test_status_rejects_held_out_error_without_a_point_count(tmp_path):
    path = tmp_path / "homography.json"
    path.write_text(
        json.dumps(
            {
                "homography": np.eye(3).tolist(),
                "reprojection_error_px": 0.5,
                "held_out_reprojection_error_px": 0.5,
                "davis_frame_size": [346, 260],
                "realsense_frame_size": [640, 480],
            }
        )
    )

    status = calibration_status(path, tmp_path / "missing_extrinsics.json")

    assert status["homography_valid"] is False
    assert status["homography_validation_status"] == "missing_or_invalid_held_out_reprojection_error"


def test_status_rejects_empty_extrinsics_file(tmp_path):
    extrinsics = tmp_path / "extrinsics.json"
    extrinsics.write_text("{}")

    status = calibration_status(tmp_path / "missing_homography.json", extrinsics)

    assert status["extrinsics_file_exists"] is True
    assert status["extrinsics_available"] is False
    assert status["calibration_exists"] is False
    assert status["alignment_status"] == "approximate_side_by_side_only"
    assert status["extrinsics_validation_status"] == "missing_or_invalid_extrinsics"


def test_status_accepts_structurally_valid_extrinsics(tmp_path):
    extrinsics = tmp_path / "extrinsics.json"
    extrinsics.write_text(
        json.dumps(
            {
                "transform_direction": "davis_to_realsense",
                "transform": np.eye(4).tolist(),
            }
        )
    )

    status = calibration_status(tmp_path / "missing_homography.json", extrinsics)

    assert status["extrinsics_available"] is True
    assert status["calibration_exists"] is True
    assert status["alignment_status"] == "calibrated_3d"

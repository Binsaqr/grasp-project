"""2D homography calibration utility for DAVIS APS to RealSense RGB alignment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from grasp_project.common.config import config_path
from grasp_project.common.file_io import write_text_atomic


RANSAC_REPROJECTION_THRESHOLD_PX = 3.0


def validate_frame_size(value, name="frame_size"):
    """Return a finite positive ``(width, height)`` pair."""
    try:
        size = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be [width, height].") from exc
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or np.any(size != np.floor(size))
    ):
        raise ValueError(f"{name} must be [width, height] with finite positive integer values.")
    return float(size[0]), float(size[1])


def validate_homography_matrix(value):
    """Return a normalized finite, invertible 3x3 homography matrix."""
    try:
        homography = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Homography must be a numeric 3x3 matrix.") from exc
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Homography must be a finite numeric 3x3 matrix.")
    norm = float(np.linalg.norm(homography))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("Homography is degenerate.")
    homography = homography / norm
    try:
        singular_values = np.linalg.svd(homography, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Homography is numerically degenerate.") from exc
    if (
        singular_values.shape != (3,)
        or not np.isfinite(singular_values).all()
        or singular_values[-1] <= singular_values[0] * 1e-12
    ):
        raise ValueError("Homography is singular or numerically degenerate.")
    scale = homography[2, 2]
    if abs(scale) > np.finfo(float).eps:
        homography = homography / scale
    return homography


def _validate_points(points, name, *, minimum_count, frame_size=None, require_2d_span=False):
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an Nx2 numeric array.") from exc
    if array.ndim != 2 or array.shape[1:] != (2,) or array.shape[0] < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} Nx2 points.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    if require_2d_span:
        if np.unique(array, axis=0).shape[0] < 4 or np.linalg.matrix_rank(array - np.mean(array, axis=0)) < 2:
            raise ValueError(f"{name} must contain at least four unique, non-collinear points.")
    if np.any(array < 0):
        raise ValueError(f"{name} contains coordinates outside non-negative pixel bounds.")
    if frame_size is not None:
        width, height = validate_frame_size(frame_size, f"{name}_frame_size")
        if (
            np.any(array[:, 0] >= width)
            or np.any(array[:, 1] >= height)
        ):
            raise ValueError(f"{name} contains coordinates outside its [width, height] frame bounds.")
    return array


def _project_points(points, homography):
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=float))) @ homography.T
    denominators = homogeneous[:, 2]
    if not np.isfinite(homogeneous).all() or np.any(np.abs(denominators) <= 1e-12):
        raise ValueError("Homography produces undefined point projections.")
    projected = homogeneous[:, :2] / denominators[:, None]
    if not np.isfinite(projected).all():
        raise ValueError("Homography produces non-finite point projections.")
    return projected


def _estimate_homography_dlt(davis, realsense):
    rows = []
    for (x, y), (u, v) in zip(davis, realsense):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    try:
        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Unable to estimate a homography from the supplied points.") from exc
    return vh[-1].reshape(3, 3)


def estimate_homography(
    davis_points,
    realsense_points,
    *,
    davis_frame_size=None,
    realsense_frame_size=None,
    held_out_davis_points=None,
    held_out_realsense_points=None,
):
    """Estimate a DAVIS-to-RealSense homography and optional held-out error."""
    davis = _validate_points(
        davis_points,
        "davis_points",
        minimum_count=4,
        frame_size=davis_frame_size,
        require_2d_span=True,
    )
    realsense = _validate_points(
        realsense_points,
        "realsense_points",
        minimum_count=4,
        frame_size=realsense_frame_size,
        require_2d_span=True,
    )
    if davis.shape != realsense.shape:
        raise ValueError("Need at least four matching DAVIS and RealSense 2D points.")
    if (held_out_davis_points is None) != (held_out_realsense_points is None):
        raise ValueError("Provide both held_out_davis_points and held_out_realsense_points.")
    held_out_davis = None
    held_out_realsense = None
    if held_out_davis_points is not None:
        held_out_davis = _validate_points(
            held_out_davis_points,
            "held_out_davis_points",
            minimum_count=1,
            frame_size=davis_frame_size,
        )
        held_out_realsense = _validate_points(
            held_out_realsense_points,
            "held_out_realsense_points",
            minimum_count=1,
            frame_size=realsense_frame_size,
        )
        if held_out_davis.shape != held_out_realsense.shape:
            raise ValueError("Held-out DAVIS and RealSense point arrays must have matching shapes.")

    try:
        import cv2
    except ImportError:
        h = _estimate_homography_dlt(davis, realsense)
        mask = np.ones((len(davis), 1), dtype=np.uint8)
        estimation_method = "numpy_dlt"
    else:
        try:
            h, mask = cv2.findHomography(
                davis.astype(np.float32),
                realsense.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD_PX,
            )
        except Exception as exc:
            raise ValueError("OpenCV could not estimate a homography from the supplied points.") from exc
        if h is None:
            raise ValueError("cv2.findHomography returned no matrix.")
        estimation_method = "opencv_ransac"

    h = validate_homography_matrix(h)
    projected = _project_points(davis, h)
    errors = np.linalg.norm(projected - realsense, axis=1)
    inlier_mask = np.ones(len(davis), dtype=bool) if mask is None else np.asarray(mask).reshape(-1).astype(bool)
    if inlier_mask.shape != (len(davis),) or int(np.count_nonzero(inlier_mask)) < 4:
        raise ValueError("Homography estimation produced fewer than four inliers.")
    inlier_errors = errors[inlier_mask]
    result = {
        "homography": h.tolist(),
        "estimation_method": estimation_method,
        "point_count": int(len(davis)),
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "outlier_count": int(len(davis) - np.count_nonzero(inlier_mask)),
        "reprojection_error_px": float(np.mean(inlier_errors)),
        "max_reprojection_error_px": float(np.max(inlier_errors)),
        "all_point_reprojection_error_px": float(np.mean(errors)),
        "held_out_point_count": 0,
        "held_out_reprojection_error_px": None,
        "held_out_max_reprojection_error_px": None,
    }
    if estimation_method == "opencv_ransac":
        result["ransac_reprojection_threshold_px"] = RANSAC_REPROJECTION_THRESHOLD_PX
    if held_out_davis is not None:
        held_out_errors = np.linalg.norm(_project_points(held_out_davis, h) - held_out_realsense, axis=1)
        result.update(
            {
                "held_out_point_count": int(len(held_out_davis)),
                "held_out_reprojection_error_px": float(np.mean(held_out_errors)),
                "held_out_max_reprojection_error_px": float(np.max(held_out_errors)),
            }
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate DAVIS-to-RealSense 2D homography from point correspondences.")
    parser.add_argument("--input", help="JSON file with davis_points and realsense_points arrays.")
    parser.add_argument("--output", default=str(config_path("davis_to_realsense_homography.json")))
    args = parser.parse_args(argv)
    if not args.input:
        print("Provide --input JSON with davis_points and realsense_points. No calibration was written.")
        return 2
    try:
        data = json.loads(Path(args.input).read_text())
        if not isinstance(data, dict):
            raise ValueError("Calibration input must be a JSON object.")
        davis_frame_size = validate_frame_size(data.get("davis_frame_size"), "davis_frame_size")
        realsense_frame_size = validate_frame_size(data.get("realsense_frame_size"), "realsense_frame_size")
        held_out_davis_points = data.get("held_out_davis_points", data.get("validation_davis_points"))
        held_out_realsense_points = data.get("held_out_realsense_points", data.get("validation_realsense_points"))
        result = estimate_homography(
            data.get("davis_points"),
            data.get("realsense_points"),
            davis_frame_size=davis_frame_size,
            realsense_frame_size=realsense_frame_size,
            held_out_davis_points=held_out_davis_points,
            held_out_realsense_points=held_out_realsense_points,
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"Invalid calibration input: {exc} No calibration was written.")
        return 2
    payload = {
        **result,
        "calibration_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.input,
        "transform_direction": "davis_to_realsense",
        "davis_frame_size": [int(value) if value.is_integer() else value for value in davis_frame_size],
        "realsense_frame_size": [int(value) if value.is_integer() else value for value in realsense_frame_size],
    }
    out = Path(args.output)
    write_text_atomic(out, json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

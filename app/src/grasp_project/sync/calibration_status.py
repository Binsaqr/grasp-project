"""Calibration status helpers for DAVIS-to-RealSense alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from grasp_project.common.config import config_path, load_config_file
from grasp_project.sync.calibrate_2d import validate_frame_size, validate_homography_matrix


MAX_REPROJECTION_ERROR_PX = 8.0


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _finite_error(value, *, maximum=MAX_REPROJECTION_ERROR_PX) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and 0.0 <= number <= maximum)


def _valid_rotation(value) -> bool:
    try:
        rotation = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    )


def _validate_extrinsics_payload(payload: dict[str, Any] | None) -> tuple[bool, str]:
    if not payload:
        return False, "missing_or_invalid_extrinsics"
    if str(payload.get("transform_direction") or "").lower() != "davis_to_realsense":
        return False, "missing_or_invalid_transform_direction"
    transform = next(
        (
            payload.get(key)
            for key in (
                "transform",
                "transform_matrix",
                "extrinsics_matrix",
                "davis_to_realsense_transform",
            )
            if payload.get(key) is not None
        ),
        None,
    )
    if transform is not None:
        try:
            matrix = np.asarray(transform, dtype=float)
        except (TypeError, ValueError):
            return False, "missing_or_invalid_extrinsics"
        valid = bool(
            matrix.shape == (4, 4)
            and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
            and _valid_rotation(matrix[:3, :3])
        )
        return valid, "usable_extrinsics" if valid else "missing_or_invalid_extrinsics"
    rotation = next(
        (payload.get(key) for key in ("rotation", "rotation_matrix") if payload.get(key) is not None),
        None,
    )
    translation = next(
        (payload.get(key) for key in ("translation", "translation_vector") if payload.get(key) is not None),
        None,
    )
    try:
        translation_array = np.asarray(translation, dtype=float)
    except (TypeError, ValueError):
        translation_array = np.asarray([], dtype=float)
    valid = bool(
        _valid_rotation(rotation)
        and translation_array.shape == (3,)
        and np.isfinite(translation_array).all()
    )
    return valid, "usable_extrinsics" if valid else "missing_or_invalid_extrinsics"


def calibration_status(
    homography_path: str | Path | None = None,
    extrinsics_path: str | Path | None = None,
) -> dict[str, Any]:
    homography_path = Path(homography_path) if homography_path else config_path("davis_to_realsense_homography.json")
    extrinsics_path = Path(extrinsics_path) if extrinsics_path else config_path("davis_to_realsense_extrinsics.json")
    mount = load_config_file(config_path("camera_mount.yaml"), default={}) or {}
    homography_file_exists = homography_path.exists()
    extrinsics_file_exists = extrinsics_path.exists()
    reprojection_error = None
    max_reprojection_error = None
    held_out_reprojection_error = None
    held_out_max_reprojection_error = None
    held_out_point_count = 0
    calibration_date = None
    davis_frame_size = None
    realsense_frame_size = None
    data = _read_json_object(homography_path) if homography_file_exists else None
    if data is not None:
        try:
            reprojection_error = data.get("reprojection_error_px")
            max_reprojection_error = data.get("max_reprojection_error_px")
            held_out_reprojection_error = data.get("held_out_reprojection_error_px")
            held_out_max_reprojection_error = data.get("held_out_max_reprojection_error_px")
            held_out_point_count = int(data.get("held_out_point_count") or 0)
            calibration_date = data.get("calibration_date")
            davis_frame_size = data.get("davis_frame_size")
            realsense_frame_size = data.get("realsense_frame_size")
        except Exception:
            pass
    try:
        validate_homography_matrix(None if data is None else data.get("homography"))
        homography_matrix_valid = True
    except ValueError:
        homography_matrix_valid = False
    homography_direction_valid = bool(
        data is not None
        and str(data.get("transform_direction") or "davis_to_realsense").lower() == "davis_to_realsense"
    )
    homography_error_valid = _finite_error(reprojection_error)
    homography_max_error_valid = max_reprojection_error is None or _finite_error(max_reprojection_error)
    try:
        validate_frame_size(davis_frame_size, "davis_frame_size")
        validate_frame_size(realsense_frame_size, "realsense_frame_size")
        frame_size_metadata_valid = True
    except ValueError:
        frame_size_metadata_valid = False
    held_out_metrics_present = bool(
        held_out_point_count != 0
        or held_out_reprojection_error is not None
        or held_out_max_reprojection_error is not None
    )
    held_out_error_valid = bool(
        not held_out_metrics_present
        or (
            held_out_point_count > 0
            and _finite_error(held_out_reprojection_error)
            and (
                held_out_max_reprojection_error is None
                or _finite_error(held_out_max_reprojection_error)
            )
        )
    )
    usable = bool(
        homography_matrix_valid
        and homography_direction_valid
        and homography_error_valid
        and homography_max_error_valid
        and frame_size_metadata_valid
        and held_out_error_valid
    )
    extrinsics_payload = _read_json_object(extrinsics_path) if extrinsics_file_exists else None
    extrinsics_valid, extrinsics_validation_status = _validate_extrinsics_payload(extrinsics_payload)
    homography_available = homography_matrix_valid
    extrinsics_available = extrinsics_valid
    if extrinsics_valid:
        calibration_type = "extrinsics_3d"
    elif usable:
        calibration_type = "homography_2d"
    else:
        calibration_type = "none"
    if not homography_file_exists:
        homography_validation_status = "not_configured"
    elif not homography_matrix_valid:
        homography_validation_status = "missing_or_invalid_homography"
    elif not homography_direction_valid:
        homography_validation_status = "unsupported_transform_direction"
    elif not homography_error_valid or not homography_max_error_valid:
        homography_validation_status = "missing_or_invalid_reprojection_error"
    elif not frame_size_metadata_valid:
        homography_validation_status = "missing_frame_size_metadata"
    elif not held_out_error_valid:
        homography_validation_status = "missing_or_invalid_held_out_reprojection_error"
    else:
        homography_validation_status = "usable_reprojection_error_and_frame_sizes_passed"
    return {
        "calibration_exists": bool(usable or extrinsics_valid),
        "calibration_file_exists": bool(homography_file_exists or extrinsics_file_exists),
        "calibration_type": calibration_type,
        "homography_available": homography_available,
        "homography_file_exists": homography_file_exists,
        "homography_valid": usable,
        "extrinsics_available": extrinsics_available,
        "extrinsics_file_exists": extrinsics_file_exists,
        "extrinsics_valid": extrinsics_valid,
        "homography_path": str(homography_path),
        "extrinsics_path": str(extrinsics_path),
        "reprojection_error_px": reprojection_error,
        "max_reprojection_error_px": max_reprojection_error,
        "held_out_point_count": held_out_point_count,
        "held_out_reprojection_error_px": held_out_reprojection_error,
        "held_out_max_reprojection_error_px": held_out_max_reprojection_error,
        "davis_frame_size": davis_frame_size,
        "realsense_frame_size": realsense_frame_size,
        "homography_validation_status": homography_validation_status,
        "extrinsics_validation_status": (
            "not_configured" if not extrinsics_file_exists else extrinsics_validation_status
        ),
        "calibration_date": calibration_date,
        "usable_for_overlay": usable,
        "mount_layout": mount.get("mount_layout", "unknown"),
        "alignment_status": "calibrated_3d" if extrinsics_valid else "calibrated_2d" if usable else "approximate_side_by_side_only",
        "approximate_alignment_only": not usable and not extrinsics_valid,
    }


if __name__ == "__main__":
    print(json.dumps(calibration_status(), indent=2, sort_keys=True))

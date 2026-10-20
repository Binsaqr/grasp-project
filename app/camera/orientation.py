import json
import os
from pathlib import Path

CAMERA_ROTATION = "upright"
CAMERA_ROTATION_DEGREES = 0
APPLY_CAMERA_ROTATION = False
SOFTWARE_ROTATION_LABEL = "none"
ORIENTATION_LOG = "Camera orientation: upright, no rotation applied."


def rotation_enabled(value=None):
    if value is None:
        return APPLY_CAMERA_ROTATION
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def rotate_image_if_needed(image, apply_rotation=True):
    return image


def rotate_intrinsics_if_needed(intrinsics, apply_rotation=True):
    if not intrinsics:
        return intrinsics

    unrotated = dict(intrinsics)
    unrotated["camera_rotation_correction"] = {
        "physical_rotation": CAMERA_ROTATION,
        "physical_rotation_degrees": CAMERA_ROTATION_DEGREES,
        "software_rotation": SOFTWARE_ROTATION_LABEL,
        "applied": False,
    }
    return unrotated


def orientation_metadata_path(output_dir):
    return Path(output_dir) / "camera_orientation.json"


def write_orientation_metadata(output_dir, applied):
    metadata = {
        "apply_camera_rotation": False,
        "physical_rotation": CAMERA_ROTATION,
        "physical_rotation_degrees": CAMERA_ROTATION_DEGREES,
        "software_rotation": SOFTWARE_ROTATION_LABEL,
        "message": ORIENTATION_LOG,
    }
    path = orientation_metadata_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


def orientation_already_applied(output_dir):
    return False


def correct_saved_rgbd(output_dir="output", apply_rotation=True, force=False):
    output_dir = Path(output_dir)
    intrinsics_path = output_dir / "camera_intrinsics.json"

    if intrinsics_path.exists():
        with open(intrinsics_path, "r") as f:
            intrinsics = json.load(f)
        intrinsics = rotate_intrinsics_if_needed(intrinsics, apply_rotation=False)
        with open(intrinsics_path, "w") as f:
            json.dump(intrinsics, f, indent=4)

    write_orientation_metadata(output_dir, applied=False)
    print(ORIENTATION_LOG)
    return False

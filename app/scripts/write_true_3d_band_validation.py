#!/usr/bin/env python3
"""Write camera-free evidence for the bottle-local 3D grasp-band pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from grasp_project.realsense.realtime_runner import (
    create_locked_grasp_band_3d,
    project_locked_grasp_band_to_current_frame,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "final_validation" / "true_3d_band_lock_report.json"


def synthetic_scene(depth_mm: int, x1: int, x2: int, y1: int, y2: int):
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[mask > 0] = depth_mm
    center_x = 0.5 * (x1 + x2 - 1)
    center_y = 0.5 * (y1 + y2 - 1)
    pose = {
        "detected": True,
        "target_id": "synthetic-bottle",
        "center_px": [center_x, center_y],
        "axis_vector": [0.0, 1.0],
        "axis_length_px": float(y2 - y1),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "cleaned_mask": mask,
        "body_mask": mask,
        "depth_median_mm": float(depth_mm),
        "detection_confidence": 0.95,
    }
    return pose, depth


def main():
    intrinsics = {"fx": 240.0, "fy": 240.0, "cx": 80.0, "cy": 80.0, "source": "synthetic_aligned_depth"}
    pose, depth = synthetic_scene(600, 60, 100, 25, 135)
    left, right = [62.0, 80.0], [97.0, 80.0]
    left_patch = [[60, 70], [69, 70], [69, 90], [60, 90]]
    right_patch = [[90, 70], [99, 70], [99, 90], [90, 90]]
    band_polygon = [[60, 70], [99, 70], [99, 90], [60, 90]]
    plan = {
        "grasp_center_px": [79.5, 80.0],
        "body_fraction": 0.5,
        "left_contact_px": left,
        "right_contact_px": right,
        "planned_depth_mm": 600.0,
        "grip_width_px": 35.0,
        "grip_width_mm": 87.5,
    }
    target = {
        "valid": True,
        "attempt_id": "synthetic-validation",
        "selected_candidate_id": "body_fraction_0.500",
        "bottle_target_id": "synthetic-bottle",
        "lock_frame_id": 1,
        "lock_timestamp": "synthetic",
        "locked_body_fraction": 0.5,
        "fixed_clamp_zone": {
            "clamp_center_px": [79.5, 80.0],
            "clamp_band_width_px": 35.0,
            "clamp_band_height_px": 20.0,
            "left_clamp_patch": {"center_px": left, "polygon": left_patch},
            "right_clamp_patch": {"center_px": right, "polygon": right_patch},
            "clamp_band_polygon": band_polygon,
        },
        "bottle_contact_model": {
            "body_fraction": 0.5,
            "left_contact_patch": {"center_px": left, "polygon": left_patch},
            "right_contact_patch": {"center_px": right, "polygon": right_patch},
            "contact_band_polygon": band_polygon,
        },
    }
    locked = create_locked_grasp_band_3d(target, pose, plan, depth, intrinsics).to_dict()
    current_pose, current_depth = synthetic_scene(400, 50, 110, 0, 160)
    projected = project_locked_grasp_band_to_current_frame(locked, current_pose, current_depth, intrinsics)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SYNTHETICALLY_VALIDATED_PHYSICAL_APPROACH_NOT_TESTED",
        "three_d_deprojection_happened": bool(locked.get("true_3d_lock")),
        "lock_bottle_point_count": len(locked.get("lock_bottle_points_xyz") or []),
        "lock_band_point_count": len(locked.get("lock_band_points_xyz") or []),
        "bottle_local_frame": {
            "origin_xyz_mm": locked.get("bottle_local_origin_xyz"),
            "axis_unit": locked.get("bottle_local_axis_unit"),
            "side_unit": locked.get("bottle_local_side_unit"),
            "depth_unit": locked.get("bottle_local_depth_unit"),
            "length_mm": locked.get("bottle_length_mm"),
            "width_mm": locked.get("bottle_width_mm"),
            "confidence": locked.get("bottle_frame_confidence"),
        },
        "locked_local_contacts_xyz_mm": {
            "left": locked.get("locked_left_contact_local_xyz"),
            "right": locked.get("locked_right_contact_local_xyz"),
        },
        "current_projected_contacts_px": {
            "left": projected.get("projected_left_contact_px"),
            "right": projected.get("projected_right_contact_px"),
        },
        "projection_valid": projected.get("projection_valid"),
        "projection_confidence": projected.get("projection_confidence"),
        "projection_reason": projected.get("projection_reason"),
        "projection_source": projected.get("projection_source"),
        "two_d_fallback_used": projected.get("two_d_fallback_used"),
        "band_reselected_this_frame": projected.get("band_reselected_this_frame"),
        "measured_vs_estimated": {
            "lock_depth_and_points": "synthetic_test_input",
            "local_frame": "estimated_from_synthetic_point_cloud_by_pca",
            "projection": "computed",
            "physical_live_approach": "not_tested",
        },
        "limitations": [
            "This report is camera-free synthetic validation, not a physical live approach result.",
            "Bottle frame PCA assumes enough current RGB-D bottle surface remains visible.",
            "No RealSense-to-DAVIS calibration and no physical gripper validation exist.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()


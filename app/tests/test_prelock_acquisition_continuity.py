import cv2
import numpy as np

from bottle_pose import detect_bottle_poses
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.realtime_runner import (
    TargetLockManager,
    build_lock_progress,
    detect_bottle_poses_scaled,
    prelock_full_resolution_confirmation_due,
)


INTRINSICS = {"fx": 420.0, "fy": 420.0, "cx": 240.0, "cy": 135.0}


def _thin_known_blue_bottle_scene():
    """A full bottle whose half-scale mask is below the old fixed 800 px gate."""
    rgb = np.full((270, 480, 3), [105, 92, 82], dtype=np.uint8)
    depth = np.full((270, 480), 820, dtype=np.uint16)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.ellipse(mask, (240, 135), (14, 65), 0, 0, 360, 1, thickness=-1)
    rgb[mask > 0] = [28, 48, 115]
    depth[mask > 0] = 540
    return rgb, depth


def _stability_pose(resolution="full_resolution", *, center=(240.0, 135.0)):
    x, y = center
    pose = {
        "target_id": "bottle_01",
        "center_px": [x, y],
        "bbox_xywh": [int(x - 20), int(y - 70), 40, 140],
        "bbox_xyxy": [x - 20.0, y - 70.0, x + 20.0, y + 70.0],
        "axis_vector": [0.0, 1.0],
        "depth_median_mm": 540.0,
        "prelock_detection_resolution": resolution,
    }
    if resolution == "scaled":
        pose["candidate_score_breakdown"] = {"scaled_to_full_frame": True}
    return pose


def _mixed_boundary_override_pose(*, temporally_confirmed):
    return {
        "prelock_full_resolution_temporally_confirmed": temporally_confirmed,
        "detection_confidence": 0.92,
        "depth_confidence": 0.98,
        "bottle_profile_agreement_score": 0.88,
        "bottle_color_fraction": 0.94,
        "bottle_profile_valid": True,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb",
        "depth_geometry_evidence": {
            "depth_foreground_separated": False,
            "depth_mixed_boundary_foreground_supported": True,
            "depth_surface_coherence_score": 0.97,
            "depth_surface_interdecile_range_mm": 8.0,
            "depth_far_background_support_fraction": 0.107,
            "depth_far_background_support_pixel_count": 96,
            "depth_far_background_min_required_pixel_count": 32,
            "depth_far_background_separation_mm": 101.0,
            "depth_far_background_distributed": True,
            "depth_far_background_supported_sector_count": 2,
            "depth_far_background_opposite_sector_pair_count": 1,
        },
    }


def test_known_blue_detector_is_equivalent_at_search_and_confirmation_resolution():
    rgb, depth = _thin_known_blue_bottle_scene()

    full, full_error = detect_bottle_poses(
        rgb,
        depth,
        intrinsics=INTRINSICS,
    )
    scaled, scaled_error = detect_bottle_poses_scaled(
        rgb,
        depth,
        0.5,
        intrinsics=INTRINSICS,
    )

    assert full_error is None
    assert scaled_error is None
    assert len(full) == len(scaled) == 1
    assert np.linalg.norm(
        np.asarray(full[0]["center_px"]) - np.asarray(scaled[0]["center_px"])
    ) <= 2.0
    assert abs(full[0]["depth_median_mm"] - scaled[0]["depth_median_mm"]) <= 1.0
    assert full[0]["bottle_color_fraction"] >= 0.95
    assert scaled[0]["bottle_color_fraction"] >= 0.95
    assert full[0]["approximate_bbox_dimensions_mm"][
        "broad_bottle_dimensions_valid"
    ] is True
    assert scaled[0]["approximate_bbox_dimensions_mm"][
        "broad_bottle_dimensions_valid"
    ] is True


def test_scaled_candidate_arms_but_never_counts_as_a_lock_frame():
    manager = TargetLockManager(enabled=True)

    manager.update_stability(True, _stability_pose("scaled"))

    assert manager.prelock_full_confirmation_active is True
    assert manager.prelock_full_confirmation_state == "FULL_CONFIRMATION_ARMED"
    assert manager.stable_full_bottle_frames == 0
    assert manager.ready_to_lock() is False
    assert prelock_full_resolution_confirmation_due(manager) is True
    assert manager.prelock_full_confirmation_last_rejection_reasons == [
        "scaled_candidate_has_no_lock_authority"
    ]


def test_mixed_boundary_override_requires_temporally_confirmed_full_resolution():
    pose = _mixed_boundary_override_pose(temporally_confirmed=False)

    eligible = realtime_runner.annotate_demo_depth_foreground_override(pose)

    assert eligible is False
    assert pose["demo_depth_foreground_override_eligible"] is False
    assert pose["demo_depth_foreground_override_reason"] == (
        "FULL_RESOLUTION_TEMPORAL_CONFIRMATION_REQUIRED"
    )


def test_mixed_boundary_override_accepts_complete_spatial_and_temporal_contract():
    pose = _mixed_boundary_override_pose(temporally_confirmed=True)

    eligible = realtime_runner.annotate_demo_depth_foreground_override(pose)

    assert eligible is True
    assert pose["demo_depth_foreground_override_eligible"] is True


def test_latched_full_confirmation_survives_bounded_miss_and_requires_two_new_full_frames():
    manager = TargetLockManager(enabled=True)
    manager.update_stability(True, _stability_pose("scaled"))
    manager.update_stability(True, _stability_pose())
    assert manager.stable_full_bottle_frames == 1

    manager.update_stability(False, None)
    assert manager.prelock_full_confirmation_active is True
    assert manager.prelock_full_confirmation_state == (
        "FULL_CONFIRMATION_RETRY_AFTER_MISS"
    )
    assert manager.stable_full_bottle_frames == 0

    manager.update_stability(True, _stability_pose())
    assert manager.stable_full_bottle_frames == 1
    assert manager.ready_to_lock() is False
    manager.update_stability(True, _stability_pose())
    assert manager.ready_to_lock() is True
    assert manager.prelock_full_confirmation_state == "FULL_CONFIRMATION_COMPLETE"


def test_full_signature_change_restarts_at_one_and_reports_exact_rejection():
    manager = TargetLockManager(enabled=True)
    manager.update_stability(True, _stability_pose("scaled"))
    manager.update_stability(True, _stability_pose(center=(180.0, 135.0)))
    manager.update_stability(True, _stability_pose(center=(260.0, 135.0)))

    assert manager.stable_full_bottle_frames == 1
    assert manager.ready_to_lock() is False
    assert "center_jump_exceeded" in (
        manager.prelock_full_confirmation_last_rejection_reasons
    )
    progress = build_lock_progress(
        manager,
        pose=_stability_pose(center=(260.0, 135.0)),
        visibility={"full_bottle_visible": True},
        yolo_report={"rgbd_search_resolution": "full_resolution_lock_confirmation"},
    )
    assert progress["prelock_full_confirmation_state"] == (
        "FULL_CONFIRMATION_IN_PROGRESS"
    )
    assert progress["prelock_full_confirmation_last_rejection_reasons"] == [
        "center_jump_exceeded",
        "bbox_iou_below_full_confirmation_minimum",
    ]
    metrics = progress["prelock_full_confirmation_last_rejection_metrics"]
    assert metrics["center_jump_px"] == 80.0
    assert metrics["center_jump_limit_px"] == 10.0
    assert metrics["bbox_iou"] == 0.0
    assert metrics["bbox_iou_minimum"] == 0.65


def test_full_confirmation_accepts_motion_inside_every_strict_gate():
    manager = TargetLockManager(enabled=True)
    manager.update_stability(True, _stability_pose("scaled"))
    first = _stability_pose()
    first["cleaned_mask"] = np.ones((10, 10), dtype=np.uint8)
    manager.update_stability(True, first)

    second = _stability_pose(center=(250.0, 135.0))
    # Keep the measured extent fixed so each continuity signal is exercised
    # independently instead of allowing a translated test box to fail IoU.
    second["bbox_xywh"] = list(first["bbox_xywh"])
    second["bbox_xyxy"] = list(first["bbox_xyxy"])
    second["depth_median_mm"] = 583.0
    angle = np.deg2rad(14.9)
    second["axis_vector"] = [float(np.sin(angle)), float(np.cos(angle))]
    second["cleaned_mask"] = np.ones((10, 12), dtype=np.uint8)
    manager.update_stability(True, second)

    assert manager.ready_to_lock() is True
    assert manager.prelock_full_confirmation_last_rejection_reasons == []
    metrics = manager.prelock_full_confirmation_last_rejection_metrics
    assert metrics["center_jump_px"] == 10.0
    assert metrics["depth_delta_mm"] == 43.0
    assert metrics["axis_delta_deg"] < 15.0
    assert metrics["area_ratio"] == 1.2


def test_full_confirmation_reports_each_strict_geometry_rejection():
    cases = []

    depth_jump = _stability_pose()
    depth_jump["depth_median_mm"] = 600.0
    cases.append((depth_jump, "depth_changed"))

    axis_jump = _stability_pose()
    angle = np.deg2rad(16.0)
    axis_jump["axis_vector"] = [float(np.sin(angle)), float(np.cos(angle))]
    cases.append((axis_jump, "axis_changed"))

    low_iou = _stability_pose(center=(250.0, 135.0))
    cases.append((low_iou, "bbox_iou_below_full_confirmation_minimum"))

    low_area = _stability_pose()
    low_area["cleaned_mask"] = np.ones((5, 10), dtype=np.uint8)
    cases.append((low_area, "mask_area_ratio_changed"))

    for candidate, expected_reason in cases:
        manager = TargetLockManager(enabled=True)
        manager.update_stability(True, _stability_pose("scaled"))
        first = _stability_pose()
        if expected_reason == "mask_area_ratio_changed":
            first["cleaned_mask"] = np.ones((10, 10), dtype=np.uint8)
        manager.update_stability(True, first)
        manager.update_stability(True, candidate)

        assert manager.stable_full_bottle_frames == 1
        assert expected_reason in (
            manager.prelock_full_confirmation_last_rejection_reasons
        )


def test_full_confirmation_latch_does_not_ping_pong_back_to_scaled(monkeypatch):
    calls = []
    manager = TargetLockManager(enabled=True)
    manager.update_stability(True, _stability_pose("scaled"))
    monkeypatch.setattr(realtime_runner, "SINGLE_BOTTLE_DEMO_MODE", True)
    monkeypatch.setattr(realtime_runner, "USE_YOLO_BOTTLE_TRIGGER", False)
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses_scaled",
        lambda *_args, **_kwargs: calls.append("scaled") or ([], None),
    )
    monkeypatch.setattr(
        realtime_runner,
        "detect_bottle_poses",
        lambda *_args, **_kwargs: calls.append("full") or ([], None),
    )
    rgb = np.zeros((270, 480, 3), dtype=np.uint8)
    depth = np.full((270, 480), 700, dtype=np.uint16)

    for _ in range(3):
        poses, _error, report = (
            realtime_runner.detect_bottle_poses_with_yolo_trigger(
                rgb,
                depth,
                lock_manager=manager,
            )
        )
        assert poses == []
        assert report["rgbd_search_resolution"] == (
            "full_resolution_lock_confirmation"
        )
        manager.update_stability(False, None)

    assert calls == ["full", "full", "full"]
    assert manager.prelock_full_confirmation_active is True


def test_black_or_skin_fixture_cannot_arm_full_confirmation():
    manager = TargetLockManager(enabled=True)
    rgb = np.full((270, 480, 3), [105, 92, 82], dtype=np.uint8)
    depth = np.full((270, 480), 820, dtype=np.uint16)
    cv2.rectangle(rgb, (205, 70), (245, 220), [12, 12, 12], thickness=-1)
    cv2.rectangle(depth, (205, 70), (245, 220), 430, thickness=-1)
    cv2.rectangle(rgb, (280, 90), (320, 235), [170, 105, 75], thickness=-1)
    cv2.rectangle(depth, (280, 90), (320, 235), 450, thickness=-1)

    poses, _error = detect_bottle_poses(rgb, depth, intrinsics=INTRINSICS)
    strict = []
    for pose in poses:
        realtime_runner.stamp_current_bottle_pose_provenance(pose, 1)
        eligible, _reasons = realtime_runner.demo_bottle_identity_eligibility(pose)
        if eligible:
            strict.append(pose)

    assert strict == []
    assert manager.prelock_full_confirmation_active is False

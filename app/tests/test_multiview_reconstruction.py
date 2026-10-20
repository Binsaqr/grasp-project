"""Camera-free tests for multi-view RealSense bottle reconstruction."""

import numpy as np
import pytest

from grasp_project.realsense.multiview_reconstruction import (
    CameraPoseEstimator,
    MultiViewBottleReconstructor,
    PoseEstimate,
    get_global_reconstructor,
    make_translation_transform,
    multiview_input_gate,
    transform_points,
)


@pytest.fixture(autouse=True)
def reset_global_reconstruction_status():
    yield
    get_global_reconstructor().reset()


def synthetic_cloud(offset=(0.0, 0.0, 0.0), count=12):
    xs = np.linspace(-0.03, 0.03, count)
    ys = np.linspace(-0.08, 0.08, count)
    zz = np.full(count, 0.55)
    pts = np.stack([xs, ys, zz], axis=1).astype(np.float32)
    pts += np.asarray(offset, dtype=np.float32)
    colors = np.tile(np.array([[0.1, 0.4, 0.9]], dtype=np.float32), (count, 1))
    scene = np.concatenate([
        pts,
        np.stack([xs * 2.0, ys * 0.5, np.full(count, 0.75)], axis=1).astype(np.float32) + np.asarray(offset, dtype=np.float32),
    ])
    scene_colors = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (len(scene), 1))
    return scene, scene_colors, pts, colors


def valid_report(frame_id=1, roi_source="cleaned_bottle_mask", timestamp=1.0, contact=True):
    report = {
        "pointcloud_freshness": "CURRENT",
        "pointcloud_timestamp": timestamp,
        "pointcloud_age_sec": 0.0,
        "pointcloud_snapshot_consistent": True,
        "scene_cloud_valid": True,
        "bottle_crop_valid": True,
        "bottle_geometry_valid": True,
        "cloud_valid_for_multiview_input": True,
        "pointcloud_source_rgb_frame_id": frame_id,
        "pointcloud_source_depth_frame_id": frame_id,
        "pointcloud_intrinsics_available": True,
        "target_status": "TRACKING",
        "locked_target_valid": True,
        "pointcloud_target_id": "locked-bottle-01",
        "pointcloud_target_id_scope": "LOCK_PERSISTENT",
        "locked_target": {"valid": True, "bottle_target_id": "locked-bottle-01"},
        "lock_frame_id": 1,
        "bottle_roi_source": roi_source,
        "bottle_pointcloud_valid_depth_ratio": 0.95,
    }
    if contact:
        report.update({
            "left_contact_xyz_m": [-0.025, 0.0, 0.55],
            "right_contact_xyz_m": [0.025, 0.0, 0.55],
            "contact_width_3d_mm": 50.0,
        })
    return report


def test_transform_points_known_translation():
    pts = np.array([[0.0, 0.0, 0.5], [0.1, -0.1, 0.6]], dtype=np.float32)
    transform = make_translation_transform([0.02, 0.0, -0.01])

    out = transform_points(pts, transform)

    assert np.allclose(out, pts + np.array([0.02, 0.0, -0.01], dtype=np.float32))


def test_numpy_centroid_pose_estimator_recovers_translation():
    ref, _ref_c, _bottle, _bottle_c = synthetic_cloud(offset=(0.0, 0.0, 0.0), count=80)
    cur, _cur_c, _bottle2, _bottle2_c = synthetic_cloud(offset=(0.02, 0.0, 0.0), count=80)
    estimator = CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False)

    estimate = estimator.estimate(reference_scene_points=ref, current_scene_points=cur)

    assert estimate.valid is True
    assert estimate.source == "rgbd_odometry_unavailable_numpy_centroid"
    assert np.allclose(estimate.transform_to_reference[:3, 3], [-0.02, 0.0, 0.0], atol=1e-5)


def test_transform_several_views_into_reference_frame():
    estimator = CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False)
    recon = MultiViewBottleReconstructor(pose_estimator=estimator, voxel_size_m=0.002)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=120)

    first = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1), frame_id=1)
    scene2, scene_colors2, bottle2, bottle_colors2 = synthetic_cloud(offset=(0.025, 0.0, 0.0), count=120)
    second = recon.process_keyframe(scene_points=scene2, scene_colors=scene_colors2, bottle_points=bottle2, bottle_colors=bottle_colors2, report=valid_report(2, timestamp=2.0), frame_id=2)

    assert first.accepted is True
    assert second.accepted is True
    assert recon.status.accepted_keyframe_count == 2
    assert recon.status.pose_source in {"reference_initial", "rgbd_odometry_unavailable_numpy_centroid"}


def test_duplicate_frames_do_not_multiply_points_excessively():
    recon = MultiViewBottleReconstructor(pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=100)

    first = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1, timestamp=1.0), frame_id=1, timestamp=1.0)
    duplicate = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(2, timestamp=1.1), frame_id=2, timestamp=1.1)

    assert first.accepted is True
    assert duplicate.accepted is False
    assert duplicate.rejection_reason == "duplicate_view"
    assert recon.status.accepted_keyframe_count == 1


def test_static_view_after_long_interval_stays_duplicate():
    recon = MultiViewBottleReconstructor(pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=100)

    first = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1, timestamp=1.0), frame_id=1, timestamp=1.0)
    duplicate = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(2, timestamp=5.0), frame_id=2, timestamp=5.0)

    assert first.accepted is True
    assert duplicate.accepted is False
    assert duplicate.rejection_reason == "duplicate_view"
    assert recon.status.accepted_keyframe_count == 1
    assert recon.status.rejected_keyframe_count == 1


def test_single_static_view_does_not_claim_complete_coverage():
    recon = MultiViewBottleReconstructor(pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=120)

    result = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1, timestamp=1.0), frame_id=1, timestamp=1.0)

    assert result.accepted is True
    assert recon.status.covered_view_angle_deg <= 30.0
    assert recon.status.estimated_surface_coverage_percent < 20.0
    assert "rear" in recon.status.missing_view_directions


class FailingEstimator(CameraPoseEstimator):
    def estimate(self, **kwargs):
        return PoseEstimate(np.eye(4), "registration_unavailable", 0.0, False, rejection_reason="synthetic_registration_fail")


def test_failed_registration_does_not_contaminate_model():
    recon = MultiViewBottleReconstructor(pose_estimator=FailingEstimator(), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=100)

    result = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1), frame_id=1)

    assert result.accepted is False
    assert recon.status.accepted_keyframe_count == 0
    assert recon.fused_points()[0].shape[0] == 0


def test_reset_clears_model():
    recon = MultiViewBottleReconstructor(pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=100)
    recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1), frame_id=1)

    recon.reset()

    assert recon.status.scan_status == "IDLE"
    assert recon.status.accepted_keyframe_count == 0
    assert recon.fused_points()[0].shape[0] == 0


class BottleMovedEstimator(CameraPoseEstimator):
    def estimate(self, **kwargs):
        return PoseEstimate(np.eye(4), "test_identity_pose", 1.0, True, fitness=1.0, inlier_rmse_m=0.0, overlap_ratio=1.0)


def test_bottle_moved_pauses_reconstruction():
    recon = MultiViewBottleReconstructor(pose_estimator=BottleMovedEstimator(), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=100)
    moved_scene, moved_scene_colors, moved_bottle, moved_bottle_colors = synthetic_cloud(offset=(0.08, 0.0, 0.0), count=100)

    recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1, timestamp=1.0), frame_id=1, timestamp=1.0)
    result = recon.process_keyframe(scene_points=moved_scene, scene_colors=moved_scene_colors, bottle_points=moved_bottle, bottle_colors=moved_bottle_colors, report=valid_report(2, timestamp=3.0), frame_id=2, timestamp=3.0)

    assert result.accepted is False
    assert result.rejection_reason == "bottle_moved_requires_reset"
    assert recon.status.scan_status == "BOTTLE_MOVED"
    assert recon.status.reconstruction_requires_reset is True


def test_keyframe_thresholds_and_voxel_accumulation():
    recon = MultiViewBottleReconstructor(pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False), voxel_size_m=0.004)
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=120)
    scene2, scene_colors2, bottle2, bottle_colors2 = synthetic_cloud(offset=(0.015, 0.0, 0.0), count=120)

    recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1, timestamp=1.0), frame_id=1, timestamp=1.0)
    result = recon.process_keyframe(scene_points=scene2, scene_colors=scene_colors2, bottle_points=bottle2, bottle_colors=bottle_colors2, report=valid_report(2, timestamp=1.2), frame_id=2, timestamp=1.2)
    pts, cols = recon.fused_points()

    assert result.accepted is True
    assert recon.status.fused_voxel_count == len(pts)
    assert len(pts) > 0
    assert cols.shape == pts.shape


def test_multiview_input_gate_rejects_bbox_by_default():
    scene, _scene_colors, bottle, _bottle_colors = synthetic_cloud(count=120)
    gate = multiview_input_gate(valid_report(1, roi_source="bbox_fallback"), scene_points=scene, bottle_points=bottle, timestamp=1.0)

    assert gate["multiview_input_valid"] is False
    assert gate["multiview_input_reject_reason"] == "bbox_roi_not_allowed_for_multiview"


def test_multiview_input_gate_rejects_structural_but_invalid_lock():
    scene, _scene_colors, bottle, _bottle_colors = synthetic_cloud(count=120)
    report = valid_report()
    report.pop("locked_target_valid")
    report.pop("pointcloud_target_id")
    report.pop("pointcloud_target_id_scope")
    report["target_status"] = "SEARCHING"
    report["target_lock"] = {"lock_frame_id": None}
    report["locked_target"] = {}

    gate = multiview_input_gate(report, scene_points=scene, bottle_points=bottle, timestamp=1.0)

    assert gate["multiview_input_valid"] is False
    assert "locked_or_stably_tracked_target_required" in gate["multiview_input_reject_reasons"]
    assert "explicit_valid_locked_target_required" in gate["multiview_input_reject_reasons"]
    assert "persistent_target_identity_required" in gate["multiview_input_reject_reasons"]


def test_multiview_input_gate_rejects_missing_cloud_timestamp():
    scene, _scene_colors, bottle, _bottle_colors = synthetic_cloud(count=120)
    report = valid_report()
    report.pop("pointcloud_timestamp")

    gate = multiview_input_gate(report, scene_points=scene, bottle_points=bottle, timestamp=1.0)

    assert gate["multiview_input_valid"] is False
    assert "pointcloud_stale_or_missing_timestamp" in gate["multiview_input_reject_reasons"]


def test_multiview_session_rejects_target_identity_change():
    recon = MultiViewBottleReconstructor(
        pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False),
        voxel_size_m=0.004,
    )
    recon.start_scan(target_id="locked-bottle-01", target_id_scope="LOCK_PERSISTENT")
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=120)
    changed = valid_report()
    changed["pointcloud_target_id"] = "locked-bottle-02"
    changed["locked_target"]["bottle_target_id"] = "locked-bottle-02"

    result = recon.process_keyframe(
        scene_points=scene,
        scene_colors=scene_colors,
        bottle_points=bottle,
        bottle_colors=bottle_colors,
        report=changed,
        frame_id=1,
        timestamp=1.0,
    )

    assert result.accepted is False
    assert result.rejection_reason == "target_identity_changed_requires_reset"
    assert recon.status.reconstruction_requires_reset is True
    assert recon.status.reconstruction_target_id == "locked-bottle-01"


def test_session_outputs_and_contact_support_are_recorded(tmp_path):
    recon = MultiViewBottleReconstructor(
        output_dir=tmp_path / "multiview",
        pose_estimator=CameraPoseEstimator(allow_numpy_centroid_registration=True, use_open3d=False),
        voxel_size_m=0.004,
    )
    recon.start_scan()
    scene, scene_colors, bottle, bottle_colors = synthetic_cloud(count=120)

    result = recon.process_keyframe(scene_points=scene, scene_colors=scene_colors, bottle_points=bottle, bottle_colors=bottle_colors, report=valid_report(1), frame_id=1, timestamp=1.0)

    assert result.accepted is True
    assert recon.status.scan_status == "SCANNING"
    assert recon.status.contact_support_status in {"supported", "weak_or_outside_cloud"}
    assert recon.status.output_session_dir

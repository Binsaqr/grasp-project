"""Deterministic camera-free replay of the final bottle-grasp demo.

This deliberately crosses the component boundaries that previously had only
separate unit coverage.  A passing replay means that one strict known-blue
identity can acquire one lock, preserve one bottle-relative band while the
hand-mounted camera approaches, freeze one analytic baseline, enter the manual
post-grasp phase, and let current object-local DAVIS motion change only the
live quality result.
"""

import copy

import numpy as np
import pytest

from grasp_project.quality.live_quality_monitor import (
    LiveGraspQualityMonitor,
    baseline_from_post_grasp_snapshot,
    observation_from_status,
)
from grasp_project.realsense import realtime_runner
from grasp_project.realsense.realtime_runner import (
    EvaluationWorker,
    TargetLockManager,
    demo_bottle_identity_eligibility,
    select_active_locked_bottle_recovery_candidate,
)


IMAGE_SHAPE = (360, 480, 3)
INTRINSICS = {
    "fx": 200.0,
    "fy": 200.0,
    "cx": 240.0,
    "cy": 180.0,
    "source": "synthetic_demo_replay",
}
BOTTLE_HEIGHT_MM = 140.0
BOTTLE_WIDTH_MM = 50.0
BODY_FRACTION = 0.50


def _projected_bottle(depth_mm, *, center=(240.0, 180.0), frame_id=1, occlude_bottom=False):
    """Create one metric-consistent known-blue RGB-D observation."""
    height_px = BOTTLE_HEIGHT_MM * INTRINSICS["fy"] / float(depth_mm)
    width_px = BOTTLE_WIDTH_MM * INTRINSICS["fx"] / float(depth_mm)
    cx, cy = center
    x0 = int(round(cx - width_px / 2.0))
    x1 = int(round(cx + width_px / 2.0))
    y0 = int(round(cy - height_px / 2.0))
    y1 = int(round(cy + height_px / 2.0))
    mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
    visible_y1 = int(round(cy + height_px * 0.30)) if occlude_bottom else y1
    mask[max(0, y0):min(mask.shape[0], visible_y1), max(0, x0):min(mask.shape[1], x1)] = 255
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask > 0] = int(depth_mm)

    # The black hand fixture is present at one fixed image location and is not
    # part of the bottle mask.  Its depth intentionally gets close to the
    # bottle during the approach, reproducing the physical demo geometry.
    fixture = np.zeros_like(mask)
    fixture[285:350, 42:126] = 255
    depth[fixture > 0] = max(80, int(depth_mm) - 12)

    bbox = [float(x0), float(y0), float(x1), float(y1)]
    pose = {
        "detected": True,
        "target_id": "known-blue-bottle",
        "target_id_scope": "CURRENT_FRAME",
        "frame_id": frame_id,
        "measurement_status": "CURRENT_MEASURED_RGBD",
        "bottle_detection_status": "CANDIDATE",
        "detection_confidence": 0.96,
        "depth_confidence": 0.98,
        "bottle_profile_valid": True,
        "bottle_profile_agreement_score": 0.94,
        "segmentation_rejection_reasons": [],
        "segmentation_source": "rgb_known_blue_component",
        "bottle_color_fraction": 0.91,
        "known_blue_support_fraction": 0.91,
        "known_blue_identity_supported": True,
        "center_px": [cx, cy],
        "axis_vector": [0.0, 1.0],
        "axis_angle_degrees": 90.0,
        "axis_length_px": height_px,
        "bbox_xyxy": bbox,
        "bottle_extent_bbox_xyxy": bbox,
        "bbox_xywh": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "oriented_bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "cleaned_mask": mask,
        "body_mask": mask,
        "mask": mask,
        "depth_median_mm": float(depth_mm),
        "partial_extent": bool(occlude_bottom),
        "projected_visible_fraction": 0.80 if occlude_bottom else 1.0,
    }
    return pose, depth, fixture


def _lock_plan(pose, depth_mm):
    bbox = pose["bbox_xyxy"]
    cx, cy = pose["center_px"]
    left = [bbox[0] + 1.0, cy]
    right = [bbox[2] - 1.0, cy]
    patch_half_height = 10.0
    patch_width = max(3.0, (bbox[2] - bbox[0]) * 0.20)
    left_polygon = [
        [bbox[0], cy - patch_half_height],
        [bbox[0] + patch_width, cy - patch_half_height],
        [bbox[0] + patch_width, cy + patch_half_height],
        [bbox[0], cy + patch_half_height],
    ]
    right_polygon = [
        [bbox[2] - patch_width, cy - patch_half_height],
        [bbox[2], cy - patch_half_height],
        [bbox[2], cy + patch_half_height],
        [bbox[2] - patch_width, cy + patch_half_height],
    ]
    return {
        "grasp_center_px": [cx, cy],
        "grasp_direction": [1.0, 0.0],
        "body_fraction": BODY_FRACTION,
        "selected_requested_body_fraction": BODY_FRACTION,
        "left_contact_px": left,
        "right_contact_px": right,
        "left_contact_3d_m": [
            (left[0] - INTRINSICS["cx"]) * depth_mm / INTRINSICS["fx"] / 1000.0,
            0.0,
            depth_mm / 1000.0,
        ],
        "right_contact_3d_m": [
            (right[0] - INTRINSICS["cx"]) * depth_mm / INTRINSICS["fx"] / 1000.0,
            0.0,
            depth_mm / 1000.0,
        ],
        "planned_depth_mm": float(depth_mm),
        "grip_width_px": float(right[0] - left[0]),
        "grip_width_mm": BOTTLE_WIDTH_MM - 4.0,
        "target_contact_quality": 92.0,
        "raw_grip_width_score": 90.0,
        "grip_width_score": 90.0,
        "contact_model_ready": True,
        "left_contact_patch": {"center_px": left, "polygon": left_polygon},
        "right_contact_patch": {"center_px": right, "polygon": right_polygon},
        "selected_candidate_id": "body_fraction_0.500",
        "selected_contact_pair": {
            "candidate_id": "body_fraction_0.500",
            "body_fraction": BODY_FRACTION,
            "requested_body_fraction": BODY_FRACTION,
        },
    }


def _clamp_center(tracking):
    clamp = tracking.get("target_clamp") or tracking.get("current_projected_grasp_band") or {}
    center = clamp.get("clamp_center_px") or clamp.get("center_px")
    if center is None:
        polygon = clamp.get("clamp_band_polygon") or clamp.get("band_polygon_px")
        if polygon:
            center = np.mean(np.asarray(polygon, dtype=float), axis=0).tolist()
    return center


def _frozen_quality_result(worker, compute_id="quality-initial"):
    target = worker.lock_manager.locked_target
    return {
        "quality_matrix_status": "updated",
        "writer_session_id": "writer-replay",
        "lock_attempt_id": target["attempt_id"],
        "attempt_id": target["attempt_id"],
        "lock_frame_id": target["lock_frame_id"],
        "quality_baseline_role": "IMMUTABLE_LOCK_BASELINE",
        "quality_input_source": "planned_contact",
        "quality_contact_source_forced": True,
        "baseline_frozen": True,
        "quality_compute_id": compute_id,
        "quality_label": "GOOD",
        "raw_epsilon": 0.126,
        "normalized_quality_score": 86.0,
        "epsilon_force_closure": True,
    }


def _davis_report(motion_class, **scores):
    return {
        "stream": "active",
        "report_heartbeat_current": True,
        "event_state": "STREAM_ACTIVE_EVENTS",
        "raw_motion_class": motion_class,
        "motion_class": motion_class,
        "davis_motion_usable_for_grasp": True,
        "object_motion_valid": True,
        "grasp_zone_motion_valid": True,
        "motion_estimation_confidence": 0.94,
        "davis_target_lock_provenance_valid": True,
        "davis_target_lock_source_writer_session_id": "writer-replay",
        "davis_target_lock_source_lock_frame_id": 3,
        "davis_target_lock_source_attempt_id": scores.pop("attempt_id"),
        "davis_target_lock_source_bottle_target_id": "known-blue-bottle",
        **scores,
    }


def test_final_demo_sequence_is_one_continuous_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_PATH", tmp_path / "locked.json")
    monkeypatch.setattr(realtime_runner, "LOCKED_GRASP_BAND_DEBUG_PATH", tmp_path / "locked_debug.json")
    monkeypatch.setattr(realtime_runner, "TARGET_RELOCK_EVENT_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(realtime_runner, "log_grasp_attempt", lambda *_args, **_kwargs: None)

    manager = TargetLockManager(enabled=True)
    prelock_poses = []
    for frame_id, center in enumerate(((240.0, 180.0), (241.0, 180.0), (240.0, 181.0)), start=1):
        pose, _depth, _fixture = _projected_bottle(400, center=center, frame_id=frame_id)
        eligible, reasons = demo_bottle_identity_eligibility(pose)
        assert eligible, reasons
        manager.update_stability(True, pose)
        prelock_poses.append(pose)

    assert manager.ready_to_lock() is True
    lock_pose, lock_depth, fixture = _projected_bottle(400, frame_id=3)
    plan = _lock_plan(lock_pose, 400)
    target = manager.lock_from_plan(
        plan,
        lock_pose,
        {"epsilon": 0.126, "force_closure": True},
        {"final_score": 92.0, "decision": "READY TO CLOSE"},
        frame_id=3,
        depth=lock_depth,
        intrinsics=INTRINSICS,
        image_shape=IMAGE_SHAPE,
    )

    assert target is not None, manager.last_lock_failure_reason
    assert manager.status() == "LOCKED"
    attempt_id = target["attempt_id"]
    locked_fraction = target["locked_body_fraction"]
    assert locked_fraction == pytest.approx(BODY_FRACTION)
    assert target["locked_grasp_band_3d"]["lock_valid"] is True
    assert np.count_nonzero(fixture) > 0

    # Approach through the physical demo range.  The fixture stays fixed in
    # the image while the bottle grows with perspective.  The band may scale
    # in pixels, but its normalized axial position may not climb the bottle.
    normalized_band_positions = []
    for frame_id, depth_mm in enumerate((340, 280, 220, 180, 160), start=4):
        pose, depth, current_fixture = _projected_bottle(
            depth_mm,
            frame_id=frame_id,
            occlude_bottom=(depth_mm == 160),
        )
        assert np.array_equal(current_fixture, fixture)
        recovered = select_active_locked_bottle_recovery_candidate(
            [pose], manager, IMAGE_SHAPE
        )
        assert recovered is pose
        _plan, tracking = manager.track(
            recovered,
            depth,
            INTRINSICS,
            {"full_bottle_visible": not pose["partial_extent"], "visibility_score": 90.0},
            IMAGE_SHAPE,
            frame_id=frame_id,
        )
        assert manager.status() == "LOCKED"
        assert manager.locked_target["attempt_id"] == attempt_id
        assert manager.locked_target["locked_body_fraction"] == pytest.approx(locked_fraction)
        assert tracking.get("band_reselected_this_frame") is not True
        assert tracking.get("locked_projection_mode") in {"TRUE_3D", "HELD_LAST_VALID_3D"}
        center = _clamp_center(tracking)
        assert center is not None, tracking
        bbox = pose["bottle_extent_bbox_xyxy"]
        normalized = (float(center[1]) - bbox[1]) / (bbox[3] - bbox[1])
        normalized_band_positions.append(normalized)

    assert max(abs(value - locked_fraction) for value in normalized_band_positions) <= 0.04
    assert max(normalized_band_positions) - min(normalized_band_positions) <= 0.04

    # Freeze the baseline exactly once.  A later same-lineage result cannot
    # replace the initial analytic hypothesis.
    worker = EvaluationWorker(lock_target=True)
    worker.lock_manager = manager
    worker.latest_report = {
        "writer_session_id": "writer-replay",
        "target_lock": {
            "target_status": "TRACKING",
            "attempt_id": attempt_id,
            "lock_frame_id": 3,
            "locked_target": copy.deepcopy(target),
            "locked_grasp_band": copy.deepcopy(target["locked_grasp_band"]),
        },
    }
    worker.attach_locked_quality_snapshot(_frozen_quality_result(worker))
    first_band = copy.deepcopy(manager.locked_target["locked_grasp_band"])
    worker.attach_locked_quality_snapshot(_frozen_quality_result(worker, "quality-duplicate"))
    assert manager.locked_target["locked_grasp_band"] == first_band
    assert first_band["locked_quality_compute_id"] == "quality-initial"
    assert first_band["locked_quality_label"] == "GOOD"
    assert first_band["locked_epsilon"] > 0.0
    assert first_band["locked_quality_force_closure"] is True

    worker.latest_report["target_lock"]["locked_target"] = copy.deepcopy(manager.locked_target)
    worker.latest_report["target_lock"]["locked_grasp_band"] = copy.deepcopy(first_band)

    class MonitorStub:
        baseline = None
        starts = 0

        def start(self, baseline):
            self.starts += 1
            self.baseline = copy.deepcopy(baseline)
            return {"stream": "active"}

    event_monitor = MonitorStub()
    worker.event_monitor = event_monitor
    transitions = []
    worker.post_grasp_transition_result.connect(transitions.append)
    worker.mark_gripper_closed_for_command("manual-close-replay")

    assert transitions[-1]["ok"] is True
    assert worker.post_grasp_active is True
    assert event_monitor.starts == 1
    assert event_monitor.baseline["quality_compute_id"] == "quality-initial"
    assert event_monitor.baseline["attempt_id"] == attempt_id

    # The immutable analytic value stays fixed while object-local DAVIS slip
    # and rotation produce visible, directionally correct graph changes.
    realtime = {
        **worker.latest_report,
        "post_grasp_active": True,
        "post_grasp_baseline": copy.deepcopy(worker.post_grasp_baseline),
        "lock_frame_id": 3,
        "lock_attempt_id": attempt_id,
        "locked_bottle_target_id": "known-blue-bottle",
        "target_status": "TRACKING",
        "ready_to_close": True,
        "visual_contact_ready": True,
    }
    baseline = baseline_from_post_grasp_snapshot(realtime)
    assert baseline is not None and baseline.valid is True
    sync = {
        "davis_event_freshness": "CURRENT",
        "davis_events_valid_for_live_quality": True,
        "davis_realsense_valid_for_live_quality": True,
        "davis_realsense_sync_quality": "good",
        "davis_realsense_offset_ms": 8.0,
    }
    pre_close = LiveGraspQualityMonitor().update(
        baseline,
        observation_from_status(
            {
                **realtime,
                "post_grasp_active": False,
                "grasp_stage": "PRE_CLOSE",
            },
            davis=_davis_report(
                "GROSS_SLIP",
                attempt_id=attempt_id,
                gross_slip_score=100.0,
            ),
            sync=sync,
            baseline=baseline,
        ),
    )
    assert pre_close.davis_motion_used_for_live_quality is False
    assert pre_close.deviations.gross_slip == 0.0

    monitor = LiveGraspQualityMonitor()
    stable = monitor.update(
        baseline,
        observation_from_status(
            realtime,
            davis=_davis_report("STABLE", attempt_id=attempt_id),
            sync=sync,
            baseline=baseline,
        ),
    )
    slip = monitor.update(
        baseline,
        observation_from_status(
            realtime,
            davis=_davis_report(
                "INCIPIENT_SLIP",
                attempt_id=attempt_id,
                incipient_slip_score=72.0,
            ),
            sync=sync,
            baseline=baseline,
        ),
    )
    rotation = monitor.update(
        baseline,
        observation_from_status(
            realtime,
            davis=_davis_report(
                "ROTATION",
                attempt_id=attempt_id,
                rotation_score=78.0,
            ),
            sync=sync,
            baseline=baseline,
        ),
    )

    assert stable.analytic_quality_score == slip.analytic_quality_score == rotation.analytic_quality_score
    assert slip.davis_motion_used_for_live_quality is True
    assert rotation.davis_motion_used_for_live_quality is True
    assert slip.live_quality_score < stable.live_quality_score
    assert rotation.live_quality_score < stable.live_quality_score
    assert slip.to_dict()["display"]["live_quality"] != stable.to_dict()["display"]["live_quality"]
    assert rotation.to_dict()["display"]["live_quality"] != stable.to_dict()["display"]["live_quality"]

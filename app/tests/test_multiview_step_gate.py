"""Multi-view registration must gate the INCREMENTAL step between keyframes,
not the cumulative pose relative to the first keyframe.

Regression: the cumulative-transform gating made an orbit impossible -- once
the camera moved past 0.10 m / 20 deg from the start, every keyframe was
rejected for 'translation/rotation_step_above_threshold'.
"""

import numpy as np

from grasp_project.realsense.multiview_reconstruction import (
    PoseEstimate,
    CameraPoseEstimator,
    make_translation_transform,
)


def _rot_z(deg):
    r = np.radians(deg)
    t = np.eye(4)
    t[0, 0] = np.cos(r); t[0, 1] = -np.sin(r)
    t[1, 0] = np.sin(r); t[1, 1] = np.cos(r)
    return t


def _good_estimate(transform):
    # a registration that passes fitness/rmse/overlap, so only the step gate matters
    return PoseEstimate(transform, "test", 1.0, True, fitness=1.0, inlier_rmse_m=0.0, overlap_ratio=1.0)


def test_small_step_far_from_reference_is_accepted():
    est = CameraPoseEstimator(use_open3d=False)
    # cumulative pose is 0.5 m and 90 deg from the first keyframe (deep into an
    # orbit) but the step from the previous keyframe is small
    prev = make_translation_transform([0.45, 0.0, 0.0]) @ _rot_z(84.0)
    cur = make_translation_transform([0.50, 0.0, 0.0]) @ _rot_z(90.0)  # +5 cm, +6 deg step
    gated = est._gate(_good_estimate(cur), previous_pose_to_reference=prev)
    assert gated.valid is True, gated.rejection_reason


def test_same_cumulative_pose_rejected_when_measured_cumulatively():
    est = CameraPoseEstimator(use_open3d=False)
    cur = make_translation_transform([0.50, 0.0, 0.0]) @ _rot_z(90.0)
    # no previous pose -> the step IS the cumulative transform -> rejected
    gated = est._gate(_good_estimate(cur), previous_pose_to_reference=None)
    assert gated.valid is False
    assert gated.rejection_reason in {"translation_step_above_threshold", "rotation_step_above_threshold"}


def test_large_step_still_rejected():
    est = CameraPoseEstimator(use_open3d=False)
    prev = make_translation_transform([0.10, 0.0, 0.0])
    cur = make_translation_transform([0.40, 0.0, 0.0])  # 30 cm step
    gated = est._gate(_good_estimate(cur), previous_pose_to_reference=prev)
    assert gated.valid is False
    assert gated.rejection_reason == "translation_step_above_threshold"


def test_full_orbit_of_small_steps_all_accepted():
    est = CameraPoseEstimator(use_open3d=False)
    # simulate a 360 deg orbit in 10 deg steps at fixed radius; every step is
    # within threshold, so all keyframes must be accepted
    radius = 0.30
    prev = None
    accepted = 0
    for k in range(0, 361, 10):
        r = np.radians(k)
        cur = make_translation_transform([radius * np.cos(r), radius * np.sin(r), 0.0]) @ _rot_z(k)
        gated = est._gate(_good_estimate(cur), previous_pose_to_reference=prev)
        if k == 0:
            # first pose relative to reference origin: 0.30 m > 0.10 m threshold,
            # which is fine -- the first keyframe defines the reference and is
            # not step-gated in process_keyframe (accepted_keyframe_count == 0)
            prev = cur
            continue
        assert gated.valid is True, f"orbit step at {k} deg rejected: {gated.rejection_reason}"
        accepted += 1
        prev = cur
    assert accepted == 36

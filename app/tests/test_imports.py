"""Camera-free import checks for the collaboration package facade."""

import importlib


def test_core_package_imports():
    modules = [
        "grasp_project",
        "grasp_project.realsense.realtime_runner",
        "grasp_project.realsense.bottle_detection",
        "grasp_project.realsense.contact_planning",
        "grasp_project.realsense.depth_processing",
        "grasp_project.realsense.epsilon_wrench",
        "grasp_project.realsense.quality_matrix",
        "grasp_project.realsense.pointcloud",
        "grasp_project.realsense.perception_result",
        "grasp_project.realsense.camera",
        "grasp_project.sync.sensor_sync",
        "grasp_project.sync.calibration_status",
        "grasp_project.sync.calibrate_2d",
        "grasp_project.embodiment.gripper_model",
        "grasp_project.embodiment.contact_model",
        "grasp_project.embodiment.base",
        "grasp_project.embodiment.contact_primitives",
        "grasp_project.embodiment.registry",
        "grasp_project.embodiment.detector",
        "grasp_project.embodiment.metric_applicability",
        "grasp_project.embodiment.plugins.parallel_jaw",
        "grasp_project.embodiment.plugins.articulated_two_finger",
        "grasp_project.embodiment.plugins.three_finger",
        "grasp_project.embodiment.plugins.multi_finger_hand",
        "grasp_project.embodiment.plugins.human_hand",
        "grasp_project.embodiment.plugins.vacuum_suction",
        "grasp_project.embodiment.plugins.soft_gripper",
        "grasp_project.davis.camera",
        "grasp_project.davis.aps_stream",
        "grasp_project.davis.event_stream",
        "grasp_project.davis.motion_features",
        "grasp_project.davis.motion_classifier",
        "grasp_project.davis.motion_gui",
        "grasp_project.dashboard.routes",
        "grasp_project.dashboard.status_models",
        "grasp_project.dashboard.web_dashboard",
        "grasp_project.common.config",
        "grasp_project.common.file_io",
        "grasp_project.common.geometry",
        "grasp_project.common.image_utils",
        "grasp_project.common.logging_utils",
        "grasp_project.common.timing",
        "grasp_project.hardware.device_discovery",
        "grasp_project.demos.run_realsense_demo",
        "grasp_project.demos.run_davis_motion_demo",
        "grasp_project.demos.run_davis_only_demo",
        "grasp_project.demos.run_sensor_sync_demo",
    ]
    for name in modules:
        assert importlib.import_module(name)


def test_legacy_entrypoints_import_without_camera():
    for name in ["epsilon_quality", "event_postgrasp_monitor", "davis_motion_gui", "web_dashboard"]:
        assert importlib.import_module(name)


def test_realsense_support_helpers_import_without_camera():
    modules = [
        "bottle_pose",
        "depth_utils",
        "camera.depth_utils",
        "visual_contact_model",
        "pregrasp_planner",
        "pregrasp_safety",
        "pregrasp_action",
        "state_machine",
        "stage_evaluators",
        "viewpoint_evaluator",
        "wrench_safety",
    ]
    for name in modules:
        assert importlib.import_module(name)

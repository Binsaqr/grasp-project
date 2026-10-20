"""Camera-free tests for RealSense stream-stall supervision."""

from grasp_project.realsense.realtime_runner import camera_stream_stall_reason


def test_watchdog_allows_startup_grace_and_recent_rgbd_frames():
    assert camera_stream_stall_reason(
        19.9,
        0.0,
        None,
        None,
        stall_timeout_sec=10.0,
        startup_grace_sec=20.0,
    ) is None
    assert camera_stream_stall_reason(
        100.0,
        0.0,
        90.1,
        95.0,
        stall_timeout_sec=10.0,
        startup_grace_sec=20.0,
    ) is None


def test_watchdog_fails_closed_after_missing_or_stalled_rgbd_frames():
    assert camera_stream_stall_reason(
        20.1,
        0.0,
        None,
        None,
        stall_timeout_sec=10.0,
        startup_grace_sec=20.0,
    ) == "no_rgbd_frames_after_startup_grace"
    assert camera_stream_stall_reason(
        100.1,
        0.0,
        90.0,
        90.0,
        stall_timeout_sec=10.0,
        startup_grace_sec=20.0,
    ) == "rgbd_frame_stream_stalled"


def test_watchdog_detects_color_or_depth_freezing_independently():
    assert camera_stream_stall_reason(
        100.1,
        0.0,
        90.0,
        99.0,
        stall_timeout_sec=10.0,
        startup_grace_sec=20.0,
    ) == "color_frame_stream_stalled"
    assert camera_stream_stall_reason(
        100.1,
        0.0,
        99.0,
        90.0,
        stall_timeout_sec=10.0,
        startup_grace_sec=20.0,
    ) == "aligned_depth_frame_stream_stalled"

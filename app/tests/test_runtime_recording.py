"""Process recording lifecycle with synthetic inputs and no sensor startup."""

import threading

import pytest

from grasp_project.quality.live_quality_monitor import (
    BaselineGraspQuality,
    LiveGraspObservation,
    LiveGraspQualityMonitor,
)
from grasp_project.recording import runtime
from grasp_project.recording.session import read_recording


def settings(path):
    return {"GRASP_RECORDING_ROOT": str(path), "GRASP_RECORDING_TRIAL_ID": "trial",
            "GRASP_RECORDING_SESSION_ID": "session", "GRASP_RECORDING_OBJECT_ID": "object"}


@pytest.fixture
def revision(monkeypatch):
    monkeypatch.setattr(runtime, "_revision", lambda root: {"commit": "synthetic-fixture"})


def test_defaults_off_without_filesystem_or_revision_work(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_revision", lambda root: pytest.fail("disabled recording read revision"))
    assert runtime.create_runtime_recording("dashboard", {}, environ={}, project_root=tmp_path) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", list(settings("unused")))
def test_partial_environment_rejected_before_directory_creation(tmp_path, missing):
    environment = settings(tmp_path / "session")
    del environment[missing]
    with pytest.raises(ValueError, match="requires all"):
        runtime.create_runtime_recording("dashboard", {}, environ=environment, project_root=tmp_path)
    assert not (tmp_path / "session").exists()


def test_process_parts_preserve_shared_identity_and_exact_recursive_config(tmp_path, revision):
    nested = tmp_path / "config" / "object_profiles" / "bottle.yaml"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"threshold: 3\r\n")
    environment = settings(tmp_path / "session")
    writers = [runtime.create_runtime_recording(role, {"davis_motion": {"value": 3}},
               environ=environment, project_root=tmp_path) for role in ("dashboard", "rgbd")]
    for writer in writers:
        writer.close()
    parts = [read_recording(writer.root).manifest for writer in writers]
    assert len({part["recording_id"] for part in parts}) == 2
    for part in parts:
        assert part["trial_id"] == "trial"
        assert part["metadata"]["session_id"] == "session"
        assert part["metadata"]["object_id"] == "object"
        assert part["config"]["davis_motion"] == {"value": 3}
        assert part["config"]["config_files"]["config/object_profiles/bottle.yaml"]["contents_utf8"] == "threshold: 3\r\n"
    with pytest.raises(FileExistsError):
        runtime.create_runtime_recording("dashboard", {}, environ=environment, project_root=tmp_path)


def test_runtime_writer_refuses_nesting_inside_archive(tmp_path, revision):
    environment = settings(tmp_path / "session")
    writer = runtime.create_runtime_recording("dashboard", {}, environ=environment, project_root=tmp_path)
    nested = settings(writer.root / "nested")
    with pytest.raises(ValueError, match="nested"):
        runtime.create_runtime_recording("rgbd", {}, environ=nested, project_root=tmp_path)


def test_abort_permanently_blocks_finalization_but_preserves_forensic_integrity(tmp_path, revision):
    writer = runtime.create_runtime_recording("dashboard", {}, environ=settings(tmp_path / "session"), project_root=tmp_path)
    writer.abort("synthetic_runtime_failure")
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(writer.root)
    manifest = read_recording(writer.root, require_complete=False).manifest
    assert manifest["failure_reason"] == "synthetic_runtime_failure"
    with pytest.raises(ValueError, match="previous write failed"):
        writer.close()


def test_failed_close_cannot_be_made_complete_by_subsequent_abort(tmp_path, revision, monkeypatch):
    writer = runtime.create_runtime_recording("dashboard", {}, environ=settings(tmp_path / "session"), project_root=tmp_path)
    persist = writer._persist
    monkeypatch.setattr(writer, "_persist", lambda: (_ for _ in ()).throw(OSError("fixture failure")))
    with pytest.raises(OSError, match="fixture failure"):
        writer.close()
    monkeypatch.setattr(writer, "_persist", persist)
    writer.abort("close_failed")
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(writer.root)


@pytest.fixture
def dashboard(tmp_path, monkeypatch, revision):
    from grasp_project.dashboard import web_dashboard

    for key, value in settings(tmp_path / "session").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(web_dashboard, "runtime_recording", None)
    monkeypatch.setattr(web_dashboard, "live_quality_monitor", LiveGraspQualityMonitor())
    monkeypatch.setattr(web_dashboard, "derived_status_stop", threading.Event())
    monkeypatch.setattr(web_dashboard, "derived_status_thread", None)
    monkeypatch.setattr(web_dashboard, "reconnect_stop", threading.Event())
    monkeypatch.setattr(web_dashboard, "reconnect_thread", None)
    monkeypatch.setattr(web_dashboard, "desired_processes", {"realsense": True, "davis_motion": True})
    monkeypatch.setattr(web_dashboard, "stop_davis_motion", lambda: (True, "synthetic stopped"))
    return web_dashboard


def test_dashboard_main_attaches_before_first_inference_and_finalizes_after_stop(dashboard, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["dashboard"])
    for name in ("write_debug_stream_state", "start_file_pollers", "start_reconnect_monitor", "start_derived_status_producer"):
        monkeypatch.setattr(dashboard, name, lambda: None)

    def serve(**kwargs):
        assert dashboard.live_quality_monitor.recording_sink is dashboard.runtime_recording
        dashboard.live_quality_monitor.update(BaselineGraspQuality(valid=True, analytic_score=84, analytic_label="GOOD"),
                                              LiveGraspObservation(timestamp=10))

    monkeypatch.setattr(dashboard.app, "run", serve)
    dashboard.main()
    archive = read_recording(dashboard.runtime_recording.root)
    assert [row["kind"] for row in archive.records] == ["live_quality_inputs"]
    assert archive.records[0]["metadata"]["initial_state"] == "constructor_reset"
    assert dashboard.live_quality_monitor.recording_sink is None


def test_dashboard_main_error_leaves_incomplete(dashboard, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["dashboard"])
    for name in ("write_debug_stream_state", "start_file_pollers", "start_reconnect_monitor", "start_derived_status_producer"):
        monkeypatch.setattr(dashboard, name, lambda: None)
    monkeypatch.setattr(dashboard.app, "run", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fixture")))
    with pytest.raises(RuntimeError, match="fixture"):
        dashboard.main()
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(dashboard.runtime_recording.root)


def test_dashboard_requested_sigterm_uses_verified_teardown_and_restores_handler(dashboard, monkeypatch):
    import signal
    import sys

    monkeypatch.setattr(sys, "argv", ["dashboard"])
    previous = signal.getsignal(signal.SIGTERM)
    for name in ("write_debug_stream_state", "start_file_pollers", "start_reconnect_monitor", "start_derived_status_producer"):
        monkeypatch.setattr(dashboard, name, lambda: None)

    def serve(**kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(dashboard.app, "run", serve)
    dashboard.main()
    assert signal.getsignal(signal.SIGTERM) == previous
    assert read_recording(dashboard.runtime_recording.root).manifest["status"] == "complete"


def test_dashboard_cannot_attach_after_unrecorded_warm_updates(dashboard):
    dashboard.live_quality_monitor.update(BaselineGraspQuality(False, None, "UNAVAILABLE"), LiveGraspObservation(timestamp=10))
    with pytest.raises(ValueError, match="before the first"):
        dashboard.start_dashboard_recording()
    assert dashboard.runtime_recording is None


def test_dashboard_failed_teardown_cannot_finalize(dashboard, monkeypatch):
    writer = dashboard.start_dashboard_recording()
    monkeypatch.setattr(dashboard, "stop_davis_motion", lambda: (False, "fixture blocked"))
    with pytest.raises(RuntimeError, match="workers did not stop"):
        dashboard.finish_dashboard_recording(True)
    assert read_recording(writer.root, require_complete=False).manifest["failure_reason"] == "dashboard_workers_did_not_stop"


def test_dashboard_revokes_and_joins_reconnect_before_davis_teardown(dashboard, monkeypatch):
    writer = dashboard.start_dashboard_recording()
    events = []

    class ReconnectThread:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert dashboard.reconnect_stop.is_set()
            assert not any(dashboard.desired_processes.values())
            events.append("reconnect_joined")
            self.alive = False

    monkeypatch.setattr(dashboard, "reconnect_thread", ReconnectThread())
    monkeypatch.setattr(dashboard, "stop_davis_motion", lambda: (events.append("davis_stopped") or True, ""))
    dashboard.finish_dashboard_recording(True)
    assert events == ["reconnect_joined", "davis_stopped"]
    assert read_recording(writer.root).manifest["status"] == "complete"

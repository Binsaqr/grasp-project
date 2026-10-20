import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_grasp_trial.py"
SPEC = importlib.util.spec_from_file_location("capture_grasp_trial", SCRIPT)
capture_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capture_module)


class CaptureGraspTrialTests(unittest.TestCase):
    def test_records_artifact_age_hash_and_unlabelled_truth(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            project = tmp_path / "project"
            output = project / "output"
            config = project / "config"
            output.mkdir(parents=True)
            config.mkdir()
            status = {"frame_id": 42, "target_status": "LOCKED"}
            (output / "latest_realtime_status.json").write_text(json.dumps(status))
            davis = output / "davis"
            davis.mkdir()
            (output / "latest_overlay_live.jpg").write_bytes(b"jpeg")
            (davis / "latest_status.json").write_text("{}")
            (davis / "latest_motion_report.json").write_text("{}")
            (config / "realsense.yaml").write_text("camera: test\n")

            trial_dir = capture_module.capture_trial(
                project,
                tmp_path / "trials",
                label="upright_bright",
                stage="distant",
                captured_at=2_000_000_000.0,
            )
            manifest = json.loads((trial_dir / "manifest.json").read_text())

            self.assertEqual(manifest["capture_mode"], "bounded_existing_files_no_sensor_open")
            self.assertEqual(manifest["schema_version"], "grasp_trial_snapshot_v3")
            self.assertEqual(manifest["ground_truth_status"], "not_labelled")
            self.assertEqual(manifest["capture_readiness"], "INCOMPLETE_CURRENT_EVIDENCE")
            self.assertTrue(manifest["synchronization_claim"].startswith("not_synchronized"))
            self.assertIn("no_raw_streams_or_timestamp_alignment", manifest["snapshot_scope"])
            self.assertTrue(manifest["capture_readiness_scope"].startswith("file_presence_and_mtime_only"))
            self.assertEqual(manifest["artifacts"][0]["source"], "output/latest_realtime_status.json")
            copied = trial_dir / manifest["artifacts"][0]["captured_as"]
            self.assertEqual(
                manifest["artifacts"][0]["sha256"],
                hashlib.sha256(copied.read_bytes()).hexdigest(),
            )
            self.assertEqual(json.loads((trial_dir / "prediction.json").read_text()), status)
            self.assertEqual(
                manifest["prediction"],
                {
                    "source": "output/latest_realtime_status.json",
                    "source_json_valid": True,
                    "captured_as": "prediction.json",
                    "size_bytes": (trial_dir / "prediction.json").stat().st_size,
                    "sha256": hashlib.sha256((trial_dir / "prediction.json").read_bytes()).hexdigest(),
                },
            )
            self.assertEqual(manifest["artifacts"][0]["captured_as"], "artifacts/latest_realtime_status.json")
            self.assertEqual(
                json.loads((trial_dir / "ground_truth.json").read_text()),
                {
                    "schema_version": "grasp_trial_truth_v1",
                    "labelled": False,
                    "ground_truth": {},
                },
            )
            self.assertEqual(manifest["config_hashes"][0]["path"], "config/realsense.yaml")

    def test_captures_bounded_complete_trace_tail(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            project = tmp_path / "project"
            output = project / "output"
            (output / "davis").mkdir(parents=True)
            (project / "config").mkdir()
            trace = output / "target_lock_trace.jsonl"
            trace.write_text("".join(json.dumps({"frame": frame}) + "\n" for frame in range(20)))

            trial_dir = capture_module.capture_trial(
                project,
                tmp_path / "trials",
                label="approach_trace",
                stage="approach",
                captured_at=2_000_000_000.0,
                trace_lines=3,
                trace_max_bytes=1024,
            )
            manifest = json.loads((trial_dir / "manifest.json").read_text())
            entry = manifest["trace_tails"][0]
            captured = trial_dir / entry["captured_as"]

            self.assertEqual(entry["captured_line_count"], 3)
            self.assertEqual(entry["captured_as"], "trace_tails/target_lock_trace.jsonl")
            self.assertEqual(
                [json.loads(line)["frame"] for line in captured.read_text().splitlines()],
                [17, 18, 19],
            )

    def test_marks_current_evidence_ready_only_when_required_files_are_fresh(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            project = tmp_path / "project"
            (project / "config").mkdir(parents=True)
            for relative in capture_module.REQUIRED_CURRENT_ARTIFACTS:
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            captured_at = max(
                (project / relative).stat().st_mtime
                for relative in capture_module.REQUIRED_CURRENT_ARTIFACTS
            ) + 2.0

            trial_dir = capture_module.capture_trial(
                project,
                tmp_path / "trials",
                label="fresh",
                stage="distant",
                captured_at=captured_at,
                max_current_age_sec=5.0,
            )
            manifest = json.loads((trial_dir / "manifest.json").read_text())

            self.assertEqual(manifest["capture_readiness"], "CURRENT_EVIDENCE_AVAILABLE")
            self.assertEqual(manifest["missing_required_current_artifacts"], [])
            self.assertEqual(manifest["stale_required_current_artifacts"], [])

    def test_prediction_and_age_use_archived_status_when_source_changes_after_copy(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            project = tmp_path / "project"
            status_path = project / "output/latest_realtime_status.json"
            status_path.parent.mkdir(parents=True)
            original_status = {"frame_id": 42, "target_status": "LOCKED"}
            status_path.write_text(json.dumps(original_status), encoding="utf-8")
            captured_at = 2_000_000_000.0
            os.utime(status_path, (captured_at - 20, captured_at - 20))
            copy2 = capture_module.shutil.copy2

            def copy_then_update(source, destination):
                result = copy2(source, destination)
                if source == status_path:
                    source.write_text(json.dumps({"frame_id": 43}), encoding="utf-8")
                    os.utime(source, (captured_at - 1, captured_at - 1))
                return result

            with patch.object(capture_module.shutil, "copy2", side_effect=copy_then_update):
                trial_dir = capture_module.capture_trial(
                    project, tmp_path / "trials", label="concurrent", stage="distant",
                    captured_at=captured_at,
                )
            manifest = json.loads((trial_dir / "manifest.json").read_text())
            entry = manifest["artifacts"][0]
            archived_status = trial_dir / entry["captured_as"]

            self.assertEqual(json.loads((trial_dir / "prediction.json").read_text()), original_status)
            self.assertEqual(json.loads(archived_status.read_text()), original_status)
            self.assertEqual(entry["source_mtime"], captured_at - 20)
            self.assertEqual(entry["source_age_sec"], 20)
            self.assertFalse(entry["fresh_at_capture"])
            self.assertEqual(entry["size_bytes"], archived_status.stat().st_size)
            self.assertNotEqual(entry["size_bytes"], status_path.stat().st_size)

    def test_archived_configs_survive_source_changes(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            project = tmp_path / "project"
            config_path = project / "config/nested/realsense.yaml"
            config_path.parent.mkdir(parents=True)
            original_bytes = b"camera: test\nwindow_ms: 100\n"
            config_path.write_bytes(original_bytes)

            trial_dir = capture_module.capture_trial(
                project, tmp_path / "trials", label="config", stage="distant",
            )
            config_path.write_text("camera: changed\n", encoding="utf-8")
            manifest = json.loads((trial_dir / "manifest.json").read_text())
            entry = manifest["config_hashes"][0]

            self.assertEqual(entry["path"], "config/nested/realsense.yaml")
            self.assertEqual(entry["captured_as"], "config/nested/realsense.yaml")
            self.assertEqual((trial_dir / entry["captured_as"]).read_bytes(), original_bytes)
            self.assertEqual(entry["sha256"], hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual(entry["size_bytes"], len(original_bytes))

    def test_invalid_status_is_preserved_but_never_becomes_prediction(self):
        invalid_statuses = (b"{", b"[]", b"null", b'{"score": NaN}', b'{"score": Infinity}', b'{"score": 1e999}', b"\xff", b"[" * 5000 + b"0" + b"]" * 5000, b'{"nested":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}")
        for content in invalid_statuses:
            with self.subTest(content=content), TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                project = tmp_path / "project"
                status_path = project / "output/latest_realtime_status.json"
                status_path.parent.mkdir(parents=True)
                status_path.write_bytes(content)

                trial_dir = capture_module.capture_trial(
                    project, tmp_path / "trials", label="invalid", stage="distant",
                )
                manifest = json.loads((trial_dir / "manifest.json").read_text())

                self.assertIsNone(json.loads((trial_dir / "prediction.json").read_text()))
                self.assertFalse(manifest["prediction"]["source_json_valid"])
                archived_status = trial_dir / manifest["artifacts"][0]["captured_as"]
                self.assertEqual(archived_status.read_bytes(), content)

    def test_missing_git_records_unknown_revision_and_dirty_state(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            with patch.object(capture_module.subprocess, "run", side_effect=FileNotFoundError):
                trial_dir = capture_module.capture_trial(
                    tmp_path, tmp_path / "trials", label="no_git", stage="distant",
                )
            manifest = json.loads((trial_dir / "manifest.json").read_text())

            self.assertEqual(manifest["git"], {"commit": None, "working_tree_dirty": None, "status_short": None})
            self.assertFalse(manifest["prediction"]["source_json_valid"])

    def test_rejects_nonfinite_or_nonpositive_capture_times_and_age_limits(self):
        for argument in ("captured_at", "max_current_age_sec"):
            for value in (float("nan"), float("inf"), float("-inf"), 0, -1):
                with self.subTest(argument=argument, value=value), TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        capture_module.capture_trial(
                            tmp_path, tmp_path / "trials", label="invalid", stage="distant",
                            **{argument: value},
                        )
                    self.assertFalse((tmp_path / "trials").exists())

    def test_rejects_path_unsafe_label(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            with self.assertRaisesRegex(ValueError, "path-safe"):
                capture_module.capture_trial(
                    tmp_path,
                    tmp_path / "trials",
                    label="../escape",
                    stage="distant",
                )


if __name__ == "__main__":
    unittest.main()

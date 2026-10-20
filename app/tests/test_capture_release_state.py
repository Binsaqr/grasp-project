import hashlib
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_release_state.py"
SPEC = importlib.util.spec_from_file_location("capture_release_state", SCRIPT)
capture_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capture_module)


class FakeRunner:
    def __init__(self, *, discovery_returncode=0):
        self.calls = []
        self.discovery_returncode = discovery_returncode

    def __call__(self, command, **_kwargs):
        command = tuple(command)
        self.calls.append(command)
        if command == ("git", "rev-parse", "HEAD"):
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        if command == ("git", "status", "--short"):
            return subprocess.CompletedProcess(command, 0, " M config/hardware.yaml\n", "")
        if command[:3] == ("podman", "inspect", "grasp-dashboard"):
            payload = {
                "Id": "container-123",
                "Name": "grasp-dashboard",
                "Created": "2026-08-06T08:00:00Z",
                "Image": "image-456",
                "ImageDigest": "sha256:digest",
                "ImageName": "localhost/grasp-project:release",
                "State": {
                    "Running": True,
                    "Status": "running",
                    "StartedAt": "2026-08-06T08:00:02Z",
                },
                "Config": {
                    "Env": ["ROS_DOMAIN_ID=67", "PORT=8080"],
                    "Cmd": ["bash", "-lc", "./run_final_grasp_demo.sh --with-davis"],
                },
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[:4] == ("podman", "image", "inspect", "localhost/grasp-project:release"):
            payload = {
                "Id": "image-456",
                "Digest": "sha256:digest",
                "Created": "2026-08-06T07:59:00Z",
                "RepoTags": ["localhost/grasp-project:release"],
                "RepoDigests": ["localhost/grasp-project@sha256:digest"],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[:4] == ("podman", "exec", "grasp-dashboard", "pgrep"):
            return subprocess.CompletedProcess(command, 0, "1\n", "")
        if "device_discovery" in command[-1]:
            payload = {
                "realsense": {
                    "status": "DEVICE_DISCOVERED",
                    "selection_mode": "auto",
                    "selected": {
                        "serial_number": "RS-FALLBACK",
                        "vendor_id": "8086",
                        "product_id": "0b5b",
                        "model_name": "RealSense D405",
                    },
                },
                "davis": {
                    "status": "DEVICE_DISCOVERED",
                    "selection_mode": "auto",
                    "selected": {
                        "serial_number": None,
                        "vendor_id": "152a",
                        "product_id": "841a",
                        "model_name": "DAVIS 346",
                    },
                },
            }
            return subprocess.CompletedProcess(command, self.discovery_returncode, json.dumps(payload), "warning")
        if "check_demo_system.sh" in command[-1]:
            stdout = "\n".join(
                [
                    "[PASS] RealSense camera_info topic is available",
                    "Summary: PASS=12 WARN=0 FAIL=0",
                ]
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        raise AssertionError(f"Unexpected command: {command}")


def api_reader(url, *, timeout):
    assert timeout == 10.0
    if url.endswith("/api/davis/status"):
        return {
            "running": True,
            "davis_device_open_count": 1,
            "davis_owner_status": "owned",
            "davis_writer_pid": 42,
            "davis_writer_session_id": "session-1",
            "duplicate_davis_owner_detected": False,
        }
    return {
        "time": 123.0,
        "running": {"realsense": True, "davis_motion": True},
        "hardware": {
            "realsense": {
                "status": "DEVICE_DISCOVERED",
                "selection_mode": "auto",
                "selected": {
                    "serial_number": "TEST-CAMERA-0001",
                    "vendor_id": "8086",
                    "product_id": "0b5b",
                    "model_name": "RealSense D405",
                    "usb_speed": "3.2",
                    "backend_name": "pyrealsense2",
                },
            },
            "davis": {
                "status": "DEVICE_DISCOVERED",
                "selection_mode": "auto",
                "selected": {
                    "serial_number": None,
                    "vendor_id": "152a",
                    "product_id": "841a",
                    "model_name": "DAVIS 346",
                    "device_node": "/dev/bus/usb/001/052",
                },
            },
        },
    }


class CaptureReleaseStateTests(unittest.TestCase):
    def test_captures_required_release_identity_and_green_gate_atomically(self):
        with TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            config = project / "config"
            config.mkdir(parents=True)
            (config / "hardware.yaml").write_text("camera: test\n")
            nested = config / "objects"
            nested.mkdir()
            (nested / "bottle.json").write_text('{"name": "bottle"}\n')
            output_root = project / "output" / "release_freezes"
            runner = FakeRunner()

            path = capture_module.capture_release_state(
                project,
                output_root,
                environment={"ROS_DOMAIN_ID": "99"},
                runner=runner,
                url_reader=api_reader,
                captured_at=1_786_000_000.0,
            )
            manifest = json.loads(path.read_text())

            self.assertEqual(path.parent, output_root.resolve())
            self.assertEqual(list(output_root.glob(".*.tmp")), [])
            self.assertEqual(manifest["capture_mode"], "read_only_inspection_no_restart_no_sensor_mutation")
            self.assertEqual(
                manifest["canonical_launch_command"],
                f"cd {project.resolve()} && ./run_docker_demo.sh",
            )
            self.assertEqual(manifest["git"]["head"], "abc123")
            self.assertTrue(manifest["git"]["working_tree_dirty"])
            self.assertEqual(manifest["ros_domain_id"]["effective"], "67")
            self.assertEqual(manifest["ros_domain_id"]["container"], "67")
            self.assertEqual(manifest["ros_domain_id"]["capture_environment"], "99")
            self.assertEqual(manifest["configuration_file_count"], 2)
            hashes = {item["path"]: item["sha256"] for item in manifest["configuration_sha256"]}
            self.assertEqual(
                hashes["config/hardware.yaml"],
                hashlib.sha256((config / "hardware.yaml").read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["podman"]["container"]["container_id"], "container-123")
            self.assertEqual(manifest["podman"]["container"]["container_created_at"], "2026-08-06T08:00:00Z")
            self.assertEqual(manifest["podman"]["image"]["image_id"], "image-456")
            self.assertEqual(manifest["podman"]["image"]["image_created_at"], "2026-08-06T07:59:00Z")
            self.assertEqual(manifest["devices"]["realsense"]["stable_device_id"], "serial:TEST-CAMERA-0001")
            self.assertEqual(
                manifest["devices"]["davis"]["stable_device_id"],
                "vid_pid_model:152a:841a:DAVIS 346",
            )
            self.assertEqual(manifest["process_ownership"]["realsense_backend_writer_count"], 1)
            self.assertEqual(manifest["process_ownership"]["headless_wrapper_count"], 1)
            self.assertEqual(manifest["process_ownership"]["davis_device_open_count"], 1)
            self.assertEqual(manifest["system_check"]["summary"]["pass"], 12)
            self.assertTrue(manifest["system_check"]["summary"]["exact_12_pass"])
            self.assertIn("source /opt/ros/humble/setup.bash", manifest["system_check"]["command"][-1])
            self.assertTrue(manifest["release_gate"]["passed"])
            self.assertFalse(manifest["device_discovery_fallback"]["attempted"])

            podman_calls = [call for call in runner.calls if call[0] == "podman"]
            self.assertFalse(any(call[1] in {"run", "start", "stop", "restart", "rm"} for call in podman_calls))

    def test_uses_discovery_json_even_when_discovery_reports_nonzero(self):
        runner = FakeRunner(discovery_returncode=2)

        def missing_hardware_reader(url, *, timeout):
            if url.endswith("/api/davis/status"):
                return {
                    "davis_device_open_count": 1,
                    "davis_owner_status": "owned",
                    "duplicate_davis_owner_detected": False,
                }
            return {"running": {"realsense": True, "davis_motion": True}, "hardware": {}}

        with TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            (project / "config").mkdir(parents=True)
            path = capture_module.capture_release_state(
                project,
                project / "output" / "release_freezes",
                runner=runner,
                url_reader=missing_hardware_reader,
            )
            manifest = json.loads(path.read_text())

        self.assertTrue(manifest["device_discovery_fallback"]["attempted"])
        self.assertEqual(manifest["device_discovery_fallback"]["returncode"], 2)
        self.assertEqual(manifest["devices"]["realsense"]["stable_device_id"], "serial:RS-FALLBACK")
        self.assertEqual(
            manifest["devices"]["davis"]["stable_device_id"],
            "vid_pid_model:152a:841a:DAVIS 346",
        )

    def test_system_check_requires_exactly_twelve_without_warnings(self):
        summary = capture_module.parse_system_check_summary(
            "[PASS] RealSense camera_info topic is available\nSummary: PASS=11 WARN=1 FAIL=0",
            0,
        )
        self.assertEqual(summary["pass"], 11)
        self.assertEqual(summary["warn"], 1)
        self.assertFalse(summary["exact_12_pass"])
        self.assertTrue(summary["camera_info_current"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_dashboard_ui.py"
SPEC = importlib.util.spec_from_file_location("capture_dashboard_ui", SCRIPT)
capture_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capture_module)


class CaptureDashboardUiTests(unittest.TestCase):
    def test_presentation_audit_only_visits_operator_facing_sections(self):
        self.assertEqual(
            capture_module.PRESENTATION_SECTIONS,
            (
                "operatorSection",
                "davisMotionSection",
                "qualitySection",
                "pointcloudSection",
            ),
        )
        self.assertEqual(
            capture_module.EXPECTED_SCREENSHOT_COUNT,
            len(capture_module.PRESENTATION_SECTIONS)
            + len(capture_module.SECTIONS)
            + 2,
        )

    def test_pointcloud_probe_reads_page_global_through_firefox_xray_wrapper(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("window.wrappedJSObject || window", source)
        self.assertIn("three_js_available: !!pageWindow.THREE", source)

    def test_url_is_limited_to_local_dashboard(self):
        self.assertEqual(
            capture_module.validate_local_url("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080",
        )
        with self.assertRaisesRegex(ValueError, "localhost"):
            capture_module.validate_local_url("https://example.com")

    def test_console_filter_keeps_only_local_errors(self):
        messages = [
            {"message": "boom", "source_name": "http://127.0.0.1:8080/", "flags": 0},
            {
                "message": "uncaught exception",
                "source_name": "http://127.0.0.1:8080/app.js",
                "flags": 2,
            },
            {"message": "warning", "source_name": "http://127.0.0.1:8080/", "flags": 1},
            {"message": "other", "source_name": "resource://browser", "flags": 0},
        ]
        self.assertEqual(
            capture_module.console_error_messages(messages, "http://127.0.0.1:8080"),
            [messages[0], messages[1]],
        )

    def test_external_network_filter_excludes_local_and_keeps_cdn_error(self):
        messages = [
            {"message": "local", "source_name": "http://127.0.0.1:8080/", "flags": 0},
            {"message": "cdn", "source_name": "https://cdn.example/three.js", "flags": 0},
            {
                "message": "Loading https://unpkg.example/three.js failed",
                "source_name": "http://127.0.0.1:8080/",
                "flags": 0,
            },
            {"message": "warning", "source_name": "https://cdn.example/three.js", "flags": 1},
        ]
        self.assertEqual(
            capture_module.external_network_error_messages(
                messages, "http://127.0.0.1:8080"
            ),
            [messages[1], messages[2]],
        )

    def test_manual_close_failure_path_always_attempts_release(self):
        close_ok = {
            "ok": True,
            "status": 200,
            "payload": {"backend_acknowledged": True},
        }
        release_ok = {
            "ok": True,
            "status": 200,
            "payload": {"backend_acknowledged": True},
        }
        with mock.patch.object(
            capture_module,
            "_http_json",
            side_effect=(close_ok, RuntimeError("open failed"), release_ok),
        ) as request, mock.patch.object(
            capture_module,
            "_wait_post_grasp_state",
            side_effect=(
                {"confirmed": True, "post_grasp_active": True},
                {"confirmed": True, "post_grasp_active": False},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                capture_module._exercise_manual_declaration("http://127.0.0.1:8080")

        self.assertEqual(request.call_count, 3)
        self.assertTrue(
            request.call_args_list[-1].args[0].endswith("/mark_gripper_open")
        )

    def test_manual_close_timeout_always_attempts_release_and_confirms_open(self):
        release_ok = {
            "ok": True,
            "status": 200,
            "payload": {"backend_acknowledged": True},
        }
        with mock.patch.object(
            capture_module,
            "_http_json",
            side_effect=(RuntimeError("close response timed out"), release_ok),
        ) as request, mock.patch.object(
            capture_module,
            "_wait_post_grasp_state",
            return_value={"confirmed": True, "post_grasp_active": False},
        ) as wait_state:
            with self.assertRaisesRegex(RuntimeError, "close response timed out"):
                capture_module._exercise_manual_declaration("http://127.0.0.1:8080")

        self.assertEqual(request.call_count, 2)
        self.assertTrue(request.call_args_list[-1].args[0].endswith("/mark_gripper_open"))
        wait_state.assert_called_once_with("http://127.0.0.1:8080", False)

    def test_manual_close_rejection_still_attempts_safe_release(self):
        close_rejected = {
            "ok": False,
            "status": 409,
            "payload": {"backend_acknowledged": True},
        }
        release_ok = {
            "ok": True,
            "status": 200,
            "payload": {"backend_acknowledged": True},
        }
        with mock.patch.object(
            capture_module, "_http_json", side_effect=(close_rejected, release_ok)
        ) as request, mock.patch.object(
            capture_module,
            "_wait_post_grasp_state",
            return_value={"confirmed": True, "post_grasp_active": False},
        ):
            results = capture_module._exercise_manual_declaration(
                "http://127.0.0.1:8080"
            )

        self.assertEqual(request.call_count, 2)
        self.assertEqual(results[-1]["cycle"], "cleanup")
        self.assertTrue(results[-1]["state_confirmed"])

    def test_manual_pass_requires_acknowledged_ordered_transitions(self):
        transitions = []
        for cycle, state, active in (
            (1, "closed", True),
            (1, "open", False),
            (2, "closed", True),
            (2, "open", False),
        ):
            transitions.append(
                {
                    "cycle": cycle,
                    "declared_state": state,
                    "ok": True,
                    "backend_acknowledged": True,
                    "state_confirmed": True,
                    "post_grasp_active": active,
                }
            )
        self.assertEqual(
            capture_module._manual_declaration_result(transitions, True),
            (True, True),
        )
        transitions[0]["backend_acknowledged"] = False
        self.assertEqual(
            capture_module._manual_declaration_result(transitions, True),
            (False, True),
        )

    def test_omitted_manual_exercise_is_not_full_pass(self):
        self.assertEqual(
            capture_module._manual_declaration_result([], False),
            (None, None),
        )
        checks = capture_module._audit_pass_checks(
            screenshot_count=capture_module.EXPECTED_SCREENSHOT_COUNT,
            all_sections_exist=True,
            all_sections_active=True,
            viewport_ok=True,
            visible_broken_images=[],
            dashboard_console_errors=[],
            external_network_errors=[],
            resource_failures=[],
            endpoint_checks={"status": {"ok": True}},
            manual_ok=None,
        )
        self.assertFalse(checks["manual_declaration"])
        self.assertEqual(
            capture_module._audit_status(checks, manual_exercised=False),
            "VISUAL_PASS_MANUAL_NOT_TESTED",
        )

    def test_viewport_adjusts_outer_height_to_exact_content_size(self):
        client = mock.Mock()
        client.command.side_effect = (
            {"width": 1920, "height": 1080},
            {"width": 1920, "height": 1166},
        )
        client.execute.side_effect = (
            {"width": 1920, "height": 994},
            {"width": 1920, "height": 1080},
        )

        rect = capture_module._set_content_viewport(client)

        self.assertEqual(rect, {"width": 1920, "height": 1166})
        self.assertEqual(client.command.call_args_list[-1].args[1]["height"], 1166)

    def test_pass_gate_rejects_visible_broken_images(self):
        checks = capture_module._audit_pass_checks(
            screenshot_count=capture_module.EXPECTED_SCREENSHOT_COUNT,
            all_sections_exist=True,
            all_sections_active=True,
            viewport_ok=True,
            visible_broken_images=[{"src": "broken.jpg"}],
            dashboard_console_errors=[],
            external_network_errors=[],
            resource_failures=[],
            endpoint_checks={"status": {"ok": True}},
            manual_ok=True,
        )
        self.assertFalse(checks["no_visible_broken_images"])
        self.assertFalse(all(checks.values()))


if __name__ == "__main__":
    unittest.main()

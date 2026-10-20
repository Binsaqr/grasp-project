#!/usr/bin/env python3
"""Capture a live 1920x1080 dashboard audit with Firefox and no Python browser dependency."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SECTIONS = (
    "operatorSection",
    "davisMotionSection",
    "qualitySection",
    "pointcloudSection",
    "multiviewSection",
    "benchmarkSection",
    "embodimentSection",
)
PRESENTATION_SECTIONS = SECTIONS[:4]
DIAGNOSTICS_SECTION = "diagnosticsDetails"
OBJECT_PROFILES_SECTION = "objectProfilesSection"
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
EXPECTED_SCREENSHOT_COUNT = len(PRESENTATION_SECTIONS) + len(SECTIONS) + 2
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+")


def _console_message_is_error(message: dict) -> bool:
    if str(message.get("level") or "").lower() == "error":
        return True
    flags = message.get("flags")
    if flags is None:
        return True
    try:
        value = int(flags)
    except (TypeError, ValueError):
        return False
    return value == 0 or bool(value & 2)  # nsIScriptError.exceptionFlag


def validate_local_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("dashboard URL must use http(s) on localhost")
    return value.rstrip("/")


def console_error_messages(messages: list[dict], dashboard_url: str) -> list[dict]:
    host = urllib.parse.urlsplit(dashboard_url).netloc
    errors = []
    for message in messages:
        source = str(message.get("source_name") or "")
        text = str(message.get("message") or "")
        if host not in source and host not in text:
            continue
        if _console_message_is_error(message):
            errors.append(message)
    return errors


def external_network_error_messages(messages: list[dict], dashboard_url: str) -> list[dict]:
    host = urllib.parse.urlsplit(dashboard_url).netloc
    errors = []
    for message in messages:
        source = str(message.get("source_name") or "")
        text = str(message.get("message") or "")
        urls = URL_PATTERN.findall(f"{source} {text}")
        mentions_external_url = any(
            urllib.parse.urlsplit(url.rstrip(".,);]}")).netloc != host for url in urls
        )
        if _console_message_is_error(message) and mentions_external_url:
            errors.append(message)
    return errors


def _set_content_viewport(
    client: "MarionetteClient", width: int = VIEWPORT_WIDTH, height: int = VIEWPORT_HEIGHT
) -> dict:
    """Adjust the outer window until Firefox's content viewport is exact."""

    outer_width, outer_height = width, height
    rect = {}
    for _ in range(3):
        rect = client.command(
            "WebDriver:SetWindowRect",
            {"x": 0, "y": 0, "width": outer_width, "height": outer_height},
        )
        inner = client.execute(
            "return {width: window.innerWidth, height: window.innerHeight};"
        )
        inner_width = int(inner.get("width") or 0)
        inner_height = int(inner.get("height") or 0)
        if inner_width == width and inner_height == height:
            return rect
        outer_width += width - inner_width
        outer_height += height - inner_height
    raise RuntimeError(
        f"Could not establish {width}x{height} content viewport; last inner size was "
        f"{inner_width}x{inner_height}"
    )


def _manual_declaration_result(
    transitions: list[dict], exercised: bool
) -> tuple[bool | None, bool | None]:
    if not exercised:
        return None, None
    ended_open = bool(
        transitions
        and transitions[-1].get("declared_state") == "open"
        and transitions[-1].get("state_confirmed") is True
        and transitions[-1].get("post_grasp_active") is False
    )
    expected = [(1, "closed"), (1, "open"), (2, "closed"), (2, "open")]
    exercise_ok = (
        [(item.get("cycle"), item.get("declared_state")) for item in transitions]
        == expected
        and all(
            item.get("ok") is True
            and item.get("backend_acknowledged") is True
            and item.get("state_confirmed") is True
            for item in transitions
        )
        and ended_open
    )
    return exercise_ok, ended_open


def _audit_pass_checks(
    *,
    screenshot_count: int,
    all_sections_exist: bool,
    all_sections_active: bool,
    viewport_ok: bool,
    visible_broken_images: list[dict],
    dashboard_console_errors: list[dict],
    external_network_errors: list[dict],
    resource_failures: list[dict],
    endpoint_checks: dict,
    manual_ok: bool | None,
) -> dict[str, bool]:
    return {
        "screenshot_count": screenshot_count == EXPECTED_SCREENSHOT_COUNT,
        "all_sections_exist": all_sections_exist,
        "all_sections_active": all_sections_active,
        "viewport_1920x1080": viewport_ok,
        "no_visible_broken_images": not visible_broken_images,
        "no_dashboard_console_errors": not dashboard_console_errors,
        "no_external_network_errors": not external_network_errors,
        "no_resource_failures": not resource_failures,
        "endpoints_healthy": all(item.get("ok") for item in endpoint_checks.values()),
        "manual_declaration": manual_ok is True,
    }


def _audit_status(pass_checks: dict[str, bool], manual_exercised: bool) -> str:
    if all(pass_checks.values()):
        return "PASS"
    visual_checks = {
        name: passed for name, passed in pass_checks.items() if name != "manual_declaration"
    }
    if not manual_exercised and all(visual_checks.values()):
        return "VISUAL_PASS_MANUAL_NOT_TESTED"
    return "FAIL"


class MarionetteClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = b""
        self.request_id = 0
        greeting = self._receive()
        if greeting.get("marionetteProtocol") != 3:
            raise RuntimeError(f"Unsupported Marionette greeting: {greeting}")

    def _receive(self):
        while b":" not in self.buffer:
            chunk = self.connection.recv(65536)
            if not chunk:
                raise RuntimeError("Firefox closed the Marionette connection")
            self.buffer += chunk
        length_text, self.buffer = self.buffer.split(b":", 1)
        length = int(length_text)
        while len(self.buffer) < length:
            chunk = self.connection.recv(65536)
            if not chunk:
                raise RuntimeError("Firefox closed the Marionette response")
            self.buffer += chunk
        payload, self.buffer = self.buffer[:length], self.buffer[length:]
        return json.loads(payload)

    def command(self, name: str, arguments: dict | None = None):
        self.request_id += 1
        payload = json.dumps(
            [0, self.request_id, name, arguments or {}], separators=(",", ":")
        ).encode()
        self.connection.sendall(str(len(payload)).encode() + b":" + payload)
        response = self._receive()
        if not isinstance(response, list) or len(response) != 4:
            raise RuntimeError(f"Malformed Marionette response: {response}")
        _, response_id, error, result = response
        if response_id != self.request_id:
            raise RuntimeError("Marionette response ID mismatch")
        if error:
            raise RuntimeError(f"{name}: {error.get('message') or error}")
        return result

    def value(self, name: str, arguments: dict | None = None):
        result = self.command(name, arguments)
        return result.get("value") if isinstance(result, dict) and "value" in result else result

    def execute(self, script: str):
        return self.value(
            "WebDriver:ExecuteScript",
            {
                "script": script,
                "args": [],
                "newSandbox": False,
                "sandbox": "default",
                "line": 0,
                "filename": "grasp-dashboard-ui-audit",
            },
        )


def _available_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _firefox_binary() -> str:
    for candidate in (
        "/usr/lib64/firefox/firefox",
        "/usr/lib/firefox/firefox",
        shutil.which("firefox"),
    ):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("Firefox is required for the live dashboard UI audit")


def _connect_marionette(port: int, process: subprocess.Popen, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Firefox exited before Marionette connected: {process.returncode}")
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            connection.settimeout(30.0)
            return connection
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"Timed out connecting to Firefox Marionette: {last_error}")


def _http_json(url: str, *, method: str = "GET") -> dict:
    request = urllib.request.Request(url, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            payload = json.loads(response.read().decode())
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                "payload": payload,
            }
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode())
        except (OSError, json.JSONDecodeError):
            payload = None
        return {
            "ok": False,
            "status": error.code,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            "error": str(error),
            "payload": payload,
        }
    except (OSError, json.JSONDecodeError) as error:
        return {
            "ok": False,
            "status": getattr(error, "code", None),
            "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            "error": str(error),
        }


def _wait_for_live_dashboard(client: MarionetteClient, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = client.execute(
            """
            return {
              ready_state: document.readyState,
              title: document.title,
              realsense: document.getElementById('realsenseChip')?.textContent || '',
              davis: document.getElementById('davisChip')?.textContent || '',
              pointcloud: document.getElementById('pointcloudChip')?.textContent || '',
              sync: document.getElementById('syncChip')?.textContent || '',
              grasp_state: document.getElementById('finalDemoStatus')?.textContent || '',
              quality: document.getElementById('finalQualityStatus')?.textContent || ''
            };
            """
        )
        if (
            last.get("ready_state") == "complete"
            and "waiting" not in str(last.get("realsense") or "").lower()
            and str(last.get("grasp_state") or "").lower() != "checking outputs"
        ):
            return last
        time.sleep(0.25)
    raise RuntimeError(f"Dashboard did not populate live status in {timeout}s: {last}")


def _wait_post_grasp_state(base_url: str, expected: bool, timeout: float = 6.0) -> dict:
    deadline = time.monotonic() + timeout
    last_value = None
    while time.monotonic() < deadline:
        status = _http_json(f"{base_url}/api/status")
        realsense = (status.get("payload") or {}).get("realsense") or {}
        last_value = realsense.get("post_grasp_active")
        if last_value is expected:
            return {"confirmed": True, "post_grasp_active": last_value}
        time.sleep(0.2)
    return {"confirmed": False, "post_grasp_active": last_value}


def _exercise_manual_declaration(base_url: str) -> list[dict]:
    results = []
    may_be_closed = False
    try:
        for cycle in range(1, 3):
            for declared_state, command in (
                ("closed", "mark_gripper_closed"),
                ("open", "mark_gripper_open"),
            ):
                if declared_state == "closed":
                    # Once a close request is attempted, its outcome is uncertain
                    # until a later open state is explicitly confirmed.
                    may_be_closed = True
                result = _http_json(
                    f"{base_url}/api/realsense-command/{command}", method="POST"
                )
                backend_acknowledged = (result.get("payload") or {}).get(
                    "backend_acknowledged"
                )
                accepted = result.get("ok") is True and backend_acknowledged is True
                observed = (
                    _wait_post_grasp_state(base_url, declared_state == "closed")
                    if accepted
                    else {"confirmed": False, "post_grasp_active": None}
                )
                results.append(
                    {
                        "cycle": cycle,
                        "declared_state": declared_state,
                        "ok": result.get("ok"),
                        "status": result.get("status"),
                        "message": (result.get("payload") or {}).get("message"),
                        "backend_acknowledged": backend_acknowledged,
                        "state_confirmed": observed.get("confirmed"),
                        "post_grasp_active": observed.get("post_grasp_active"),
                    }
                )
                if declared_state == "open" and observed.get("confirmed"):
                    may_be_closed = False
                if not accepted or not observed.get("confirmed"):
                    return results
                time.sleep(0.4)
    finally:
        if may_be_closed:
            release = {}
            release_error = None
            try:
                release = _http_json(
                    f"{base_url}/api/realsense-command/mark_gripper_open",
                    method="POST",
                )
            except Exception as exc:
                release_error = str(exc)
            try:
                observed = _wait_post_grasp_state(base_url, False)
            except Exception as exc:
                observed = {"confirmed": False, "post_grasp_active": None}
                release_error = release_error or str(exc)
            results.append(
                {
                    "cycle": "cleanup",
                    "declared_state": "open",
                    "ok": release.get("ok") is True,
                    "status": release.get("status"),
                    "message": (release.get("payload") or {}).get("message"),
                    "backend_acknowledged": (release.get("payload") or {}).get(
                        "backend_acknowledged"
                    ),
                    "state_confirmed": observed.get("confirmed"),
                    "post_grasp_active": observed.get("post_grasp_active"),
                    "error": release_error,
                }
            )
    return results


def capture_dashboard(
    base_url: str,
    output_dir: Path,
    *,
    wait_timeout: float = 20.0,
    exercise_manual_state: bool = False,
) -> dict:
    base_url = validate_local_url(base_url)
    output_dir.mkdir(parents=True, exist_ok=False)
    port = _available_port()
    firefox_log = output_dir / "firefox.log"
    screenshots = []
    section_checks = []
    visible_broken_images = []
    manual_transitions = []
    with tempfile.TemporaryDirectory(prefix="grasp-firefox-ui-") as profile_name:
        profile = Path(profile_name)
        (profile / "user.js").write_text(
            f'user_pref("marionette.enabled", true);\nuser_pref("marionette.port", {port});\n'
            'user_pref("marionette.log.level", "Info");\n'
        )
        environment = os.environ.copy()
        environment["MOZ_MARIONETTE"] = "1"
        with firefox_log.open("w") as log_handle:
            process = subprocess.Popen(
                (
                    _firefox_binary(),
                    "--headless",
                    "--profile",
                    str(profile),
                    "--marionette",
                    "-remote-allow-system-access",
                    "about:blank",
                ),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            client = None
            try:
                connection = _connect_marionette(port, process)
                client = MarionetteClient(connection)
                client.command(
                    "WebDriver:NewSession", {"capabilities": {"alwaysMatch": {}}}
                )
                viewport = _set_content_viewport(client)
                client.command("WebDriver:Navigate", {"url": f"{base_url}/"})
                live_state = _wait_for_live_dashboard(client, wait_timeout)
                if exercise_manual_state:
                    manual_transitions = _exercise_manual_declaration(base_url)
                    live_state = _wait_for_live_dashboard(client, wait_timeout)

                section_existence = client.execute(
                    "return Object.fromEntries(arguments[0].map(id => [id, !!document.getElementById(id)]));"
                    .replace(
                        "arguments[0]",
                        json.dumps(
                            [*SECTIONS, DIAGNOSTICS_SECTION, OBJECT_PROFILES_SECTION]
                        ),
                    )
                )
                for mode in ("presentation", "engineering"):
                    visible_sections = (
                        PRESENTATION_SECTIONS if mode == "presentation" else SECTIONS
                    )
                    mode_sections = [
                        (section_id, section_id) for section_id in visible_sections
                    ]
                    if mode == "engineering":
                        mode_sections.extend(
                            (
                                (DIAGNOSTICS_SECTION, DIAGNOSTICS_SECTION),
                                (OBJECT_PROFILES_SECTION, "embodimentSection"),
                            )
                        )
                    client.command(
                        "WebDriver:Navigate",
                        {"url": f"{base_url}/?mode={mode}#{mode_sections[0][1]}"},
                    )
                    live_state = _wait_for_live_dashboard(client, wait_timeout)
                    for index, (section_id, active_section_id) in enumerate(mode_sections):
                        if index:
                            client.execute(
                                f"window.location.hash = {json.dumps(active_section_id)}; return window.location.hash;"
                            )
                        deadline = time.monotonic() + 5.0
                        section_state = {}
                        while time.monotonic() < deadline:
                            section_state = client.execute(
                                f"""
                                const id = {json.dumps(section_id)};
                                const activeId = {json.dumps(active_section_id)};
                                const element = document.getElementById(id);
                                const active = document.querySelector(`.release-nav a[href="#${{activeId}}"]`);
                                const primary = {json.dumps([*SECTIONS, DIAGNOSTICS_SECTION])};
                                return {{
                                  hash: window.location.hash,
                                  body_class: document.body.className,
                                  exists: !!element,
                                  visible: !!element && getComputedStyle(element).display !== 'none',
                                  active_nav: !!active && active.classList.contains('active'),
                                  hidden_other_primary: primary
                                    .filter(other => other !== activeId)
                                    .every(other => {{
                                      const node = document.getElementById(other);
                                      return !node || getComputedStyle(node).display === 'none';
                                    }}),
                                  section_top: element ? element.getBoundingClientRect().top : null
                                }};
                                """
                            )
                            if (
                                section_state.get("hash") == f"#{active_section_id}"
                                and section_state.get("visible") is True
                                and section_state.get("active_nav") is True
                                and section_state.get("hidden_other_primary") is True
                                and mode in str(section_state.get("body_class") or "").split()
                            ):
                                break
                            time.sleep(0.1)
                        client.execute(
                            f"""
                            const element = document.getElementById({json.dumps(section_id)});
                            if (!element) return false;
                            if (element.tagName === 'DETAILS') element.open = true;
                            element.scrollIntoView({{block:'start'}});
                            window.scrollBy(0, -96);
                            return true;
                            """
                        )
                        time.sleep(0.25)
                        broken_for_section = client.execute(
                            """
                            return [...document.images]
                              .filter(image => image.offsetParent !== null && image.src && image.complete && image.naturalWidth === 0)
                              .map(image => ({id:image.id || null, src:image.src}));
                            """
                        )
                        resource_failures = client.execute(
                            """
                            return performance.getEntriesByType('resource')
                              .filter(entry => Number(entry.responseStatus || 0) >= 400)
                              .map(entry => ({
                                url: entry.name,
                                initiator_type: entry.initiatorType,
                                response_status: entry.responseStatus
                              }));
                            """
                        )
                        pointcloud_viewer = (
                            client.execute(
                                """
                                const canvas = document.getElementById('pointcloudThreeCanvas');
                                const status = document.getElementById('pointcloudThreeStatus')?.textContent || '';
                                // Firefox Marionette exposes page-defined globals through an
                                // Xray wrapper; wrappedJSObject is the actual content window.
                                const pageWindow = window.wrappedJSObject || window;
                                return {
                                  three_js_available: !!pageWindow.THREE,
                                  canvas_available: !!canvas,
                                  canvas_width: canvas ? canvas.width : 0,
                                  canvas_height: canvas ? canvas.height : 0,
                                  status: status,
                                  interactive_ok: !!pageWindow.THREE && !!canvas && canvas.width > 0 && canvas.height > 0
                                    && !/unavailable|could not load|could not initialize/i.test(status)
                                };
                                """
                            )
                            if section_id == "pointcloudSection"
                            else None
                        )
                        for broken in broken_for_section:
                            item = {"mode": mode, "section_id": section_id, **broken}
                            if item not in visible_broken_images:
                                visible_broken_images.append(item)
                        encoded = client.value(
                            "WebDriver:TakeScreenshot",
                            {
                                "id": None,
                                "full": False,
                                "scroll": False,
                                "highlights": [],
                            },
                        )
                        filename = f"{mode}_{section_id}.png"
                        (output_dir / filename).write_bytes(base64.b64decode(encoded))
                        screenshots.append(filename)
                        section_checks.append(
                            {
                                "mode": mode,
                                "section_id": section_id,
                                "active_section_id": active_section_id,
                                "details_open": (
                                    True
                                    if section_id != DIAGNOSTICS_SECTION
                                    else client.execute(
                                        "return document.getElementById('diagnosticsDetails')?.open === true;"
                                    )
                                ),
                                "resource_failures": resource_failures,
                                "pointcloud_viewer": pointcloud_viewer,
                                **section_state,
                            }
                        )

                client.command("Marionette:SetContext", {"value": "chrome"})
                console_messages = client.execute(
                    """
                    return Services.console.getMessageArray().map(message => ({
                      message: String(message.message || message.errorMessage || ''),
                      source_name: String(message.sourceName || ''),
                      category: String(message.category || ''),
                      flags: Number.isFinite(message.flags) ? message.flags : null,
                      level: String(message.logLevel || '')
                    }));
                    """
                )
                dashboard_console_errors = console_error_messages(
                    console_messages, base_url
                )
                external_network_errors = external_network_error_messages(
                    console_messages, base_url
                )
                client.command("Marionette:SetContext", {"value": "content"})
                final_dom = client.execute(
                    """
                    return {
                      inner_width: window.innerWidth,
                      inner_height: window.innerHeight,
                      body_class: document.body.className,
                      title: document.title
                    };
                    """
                )
            finally:
                if client is not None:
                    try:
                        client.command("WebDriver:DeleteSession", {})
                    except Exception:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)

    endpoint_checks = {
        "status": {
            key: value
            for key, value in _http_json(f"{base_url}/api/status").items()
            if key != "payload"
        },
        "benchmark": {
            key: value
            for key, value in _http_json(f"{base_url}/api/benchmark/status").items()
            if key != "payload"
        },
    }
    all_sections_exist = all(section_existence.values())
    manual_ok, manual_ended_open = _manual_declaration_result(
        manual_transitions, exercise_manual_state
    )
    all_sections_active = len(section_checks) == EXPECTED_SCREENSHOT_COUNT and all(
        item.get("visible") is True
        and item.get("active_nav") is True
        and item.get("hidden_other_primary") is True
        and item.get("mode") in str(item.get("body_class") or "").split()
        and item.get("hash") == f"#{item.get('active_section_id')}"
        and item.get("details_open") is True
        and (
            item.get("section_id") != "pointcloudSection"
            or (item.get("pointcloud_viewer") or {}).get("interactive_ok") is True
        )
        for item in section_checks
    )
    resource_failures = []
    for section in section_checks:
        for failure in section.get("resource_failures") or []:
            if failure not in resource_failures:
                resource_failures.append(failure)
    viewport_ok = (
        final_dom.get("inner_width") == VIEWPORT_WIDTH
        and final_dom.get("inner_height") == VIEWPORT_HEIGHT
    )
    pass_checks = _audit_pass_checks(
        screenshot_count=len(screenshots),
        all_sections_exist=all_sections_exist,
        all_sections_active=all_sections_active,
        viewport_ok=viewport_ok,
        visible_broken_images=visible_broken_images,
        dashboard_console_errors=dashboard_console_errors,
        external_network_errors=external_network_errors,
        resource_failures=resource_failures,
        endpoint_checks=endpoint_checks,
        manual_ok=manual_ok,
    )
    report = {
        "schema_version": "dashboard_ui_audit_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "viewport_request": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        "window_rect": viewport,
        "final_dom": final_dom,
        "viewport_ok": viewport_ok,
        "live_state": live_state,
        "section_existence": section_existence,
        "screenshots": screenshots,
        "screenshot_count": len(screenshots),
        "section_checks": section_checks,
        "all_sections_active": all_sections_active,
        "visible_broken_images": visible_broken_images,
        "dashboard_console_errors": dashboard_console_errors,
        "dashboard_console_error_count": len(dashboard_console_errors),
        "external_network_errors": external_network_errors,
        "external_network_error_count": len(external_network_errors),
        "resource_failures": resource_failures,
        "resource_failure_count": len(resource_failures),
        "endpoint_checks": endpoint_checks,
        "manual_declaration_transitions": manual_transitions,
        "manual_declaration_exercised": exercise_manual_state,
        "manual_declaration_ended_open": manual_ended_open,
        "manual_declaration_exercise_passed": manual_ok,
        "pass_checks": pass_checks,
        "status": _audit_status(pass_checks, exercise_manual_state),
        "limitations": [
            "Screenshots validate rendering and current dashboard state; they do not validate physical grasp success.",
            "Manual grasp/release exercise is a reversible operator declaration and does not actuate a gripper.",
        ],
    }
    (output_dir / "ui_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output-root", type=Path, default=Path("output/ui_validation"))
    parser.add_argument("--wait-timeout", type=float, default=20.0)
    parser.add_argument("--exercise-manual-state", action="store_true")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = args.output_root / stamp
    report = capture_dashboard(
        args.url,
        output_dir,
        wait_timeout=args.wait_timeout,
        exercise_manual_state=args.exercise_manual_state,
    )
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir)}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

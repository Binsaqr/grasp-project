#!/usr/bin/env python3
"""Capture an atomic, read-only release-state snapshot for the running demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence


DEFAULT_CONTAINER = "grasp-dashboard"
DEFAULT_IMAGE = "localhost/grasp-project:release"
DEFAULT_STATUS_URL = "http://127.0.0.1:8080/api/status?compact=1"
SYSTEM_CHECK_COMMAND = (
    "source /opt/ros/humble/setup.bash && cd /app && "
    "./scripts/check_demo_system.sh"
)
DISCOVERY_COMMAND = (
    "source /opt/ros/humble/setup.bash && cd /app && "
    "python3 -m grasp_project.hardware.device_discovery --summary --json"
)
PROCESS_PATTERNS = {
    "realsense_backend_writer_count": "real_time_live.py --backend-only",
    "headless_wrapper_count": "run_bottle_realtime.sh --headless-web",
    "dashboard_process_count": "web_dashboard.py --host",
}
SUMMARY_PATTERN = re.compile(r"Summary:\s*PASS=(\d+)\s+WARN=(\d+)\s+FAIL=(\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    runner: Callable = subprocess.run,
    timeout: float = 60.0,
) -> dict:
    try:
        completed = runner(
            tuple(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "ok": completed.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "ok": False,
        }


def _json_from_command(result: dict) -> dict | None:
    try:
        payload = json.loads(result.get("stdout") or "null")
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    return payload if isinstance(payload, dict) else None


def _read_json_url(url: str, *, timeout: float = 10.0) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return payload


def _fetch_json(url: str, *, reader: Callable = _read_json_url) -> dict:
    try:
        return {"ok": True, "url": url, "payload": reader(url, timeout=10.0), "error": None}
    except Exception as exc:
        return {"ok": False, "url": url, "payload": None, "error": str(exc)}


def _related_davis_url(status_url: str) -> str:
    return status_url.split("/api/", 1)[0].rstrip("/") + "/api/davis/status"


def _environment_map(values: object) -> dict[str, str]:
    result = {}
    for value in values if isinstance(values, list) else []:
        if isinstance(value, str) and "=" in value:
            key, item = value.split("=", 1)
            result[key] = item
    return result


def _container_record(payload: dict | None) -> dict:
    payload = payload or {}
    state = payload.get("State") or {}
    config = payload.get("Config") or {}
    env = _environment_map(config.get("Env"))
    return {
        "container_id": payload.get("Id"),
        "container_name": payload.get("Name"),
        "container_created_at": payload.get("Created"),
        "container_started_at": state.get("StartedAt"),
        "container_status": state.get("Status"),
        "container_running": state.get("Running"),
        "image_id": payload.get("Image"),
        "image_digest": payload.get("ImageDigest"),
        "image_name": payload.get("ImageName") or config.get("Image"),
        "ros_domain_id": env.get("ROS_DOMAIN_ID"),
        "command": config.get("Cmd") or payload.get("Args"),
    }


def _image_record(payload: dict | None) -> dict:
    payload = payload or {}
    return {
        "image_id": payload.get("Id"),
        "image_digest": payload.get("Digest"),
        "image_created_at": payload.get("Created"),
        "repo_tags": payload.get("RepoTags") or [],
        "repo_digests": payload.get("RepoDigests") or [],
    }


def _selected_device(discovery: dict | None, sensor: str, source: str) -> dict:
    result = (discovery or {}).get(sensor) or {}
    selected = result.get("selected") if isinstance(result, dict) else None
    selected = selected if isinstance(selected, dict) else {}
    serial = selected.get("serial_number")
    vendor = selected.get("vendor_id")
    product = selected.get("product_id")
    model = selected.get("model_name")
    stable_id = f"serial:{serial}" if serial else None
    if stable_id is None and any((vendor, product, model)):
        stable_id = "vid_pid_model:" + ":".join(str(value or "unknown") for value in (vendor, product, model))
    return {
        "source": source,
        "selection_status": result.get("status") if isinstance(result, dict) else None,
        "selection_mode": result.get("selection_mode") if isinstance(result, dict) else None,
        "configured_serial": result.get("configured_serial") if isinstance(result, dict) else None,
        "stable_device_id": stable_id,
        "serial_number": serial,
        "vendor_id": vendor,
        "product_id": product,
        "model_name": model,
        "usb_speed": selected.get("usb_speed"),
        "backend_name": selected.get("backend_name"),
        "permission_status": selected.get("permission_status"),
        "transient_device_node": selected.get("device_node"),
    }


def _device_records(api_payload: dict | None, fallback_payload: dict | None) -> dict:
    api_hardware = (api_payload or {}).get("hardware") or {}
    records = {}
    for sensor in ("realsense", "davis"):
        api_result = api_hardware.get(sensor) if isinstance(api_hardware, dict) else None
        source_payload = api_hardware if isinstance(api_result, dict) and api_result.get("selected") else fallback_payload
        source = "/api/status" if source_payload is api_hardware else "device_discovery"
        records[sensor] = _selected_device(source_payload, sensor, source)
    records["selected_device_ids"] = {
        sensor: records[sensor].get("stable_device_id") for sensor in ("realsense", "davis")
    }
    records["identity_policy"] = {
        "persistent": "serial, or model plus VID/PID when serial is unavailable",
        "transient_usb_node": "diagnostic_only",
    }
    return records


def parse_system_check_summary(output: str, returncode: int | None) -> dict:
    match = SUMMARY_PATTERN.search(output or "")
    counts = {
        "pass": int(match.group(1)) if match else None,
        "warn": int(match.group(2)) if match else None,
        "fail": int(match.group(3)) if match else None,
    }
    counts["exact_12_pass"] = bool(
        returncode == 0 and counts["pass"] == 12 and counts["warn"] == 0 and counts["fail"] == 0
    )
    counts["camera_info_current"] = bool(
        re.search(r"^\[PASS\].*camera_info", output or "", flags=re.IGNORECASE | re.MULTILINE)
    )
    return counts


def _config_hashes(project_root: Path) -> list[dict]:
    config_root = project_root / "config"
    return [
        {"path": str(path.relative_to(project_root)), "sha256": _sha256(path)}
        for path in sorted(config_root.rglob("*"))
        if path.is_file()
    ] if config_root.is_dir() else []


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capture_release_state(
    project_root: Path,
    output_root: Path,
    *,
    container: str = DEFAULT_CONTAINER,
    image: str = DEFAULT_IMAGE,
    status_url: str = DEFAULT_STATUS_URL,
    davis_status_url: str | None = None,
    environment: Mapping[str, str] | None = None,
    runner: Callable = subprocess.run,
    url_reader: Callable = _read_json_url,
    captured_at: float | None = None,
) -> Path:
    """Inspect the running release and write one immutable JSON snapshot."""

    project_root = project_root.resolve()
    output_root = output_root.resolve()
    environment = os.environ if environment is None else environment
    captured_at = time.time() if captured_at is None else float(captured_at)

    git_head = _run(("git", "rev-parse", "HEAD"), cwd=project_root, runner=runner, timeout=10.0)
    git_status = _run(("git", "status", "--short"), cwd=project_root, runner=runner, timeout=10.0)
    container_inspect = _run(
        ("podman", "inspect", container, "--format", "{{json .}}"),
        cwd=project_root,
        runner=runner,
    )
    image_inspect = _run(
        ("podman", "image", "inspect", image, "--format", "{{json .}}"),
        cwd=project_root,
        runner=runner,
    )
    container_record = _container_record(_json_from_command(container_inspect))
    image_record = _image_record(_json_from_command(image_inspect))
    ros_domain_id = container_record.get("ros_domain_id") or environment.get("ROS_DOMAIN_ID", "67")

    status_response = _fetch_json(status_url, reader=url_reader)
    davis_response = _fetch_json(davis_status_url or _related_davis_url(status_url), reader=url_reader)
    api_payload = status_response.get("payload")
    api_hardware = (api_payload or {}).get("hardware") or {}
    needs_discovery = any(
        not isinstance(api_hardware.get(sensor), dict) or not api_hardware[sensor].get("selected")
        for sensor in ("realsense", "davis")
    )
    discovery_result = None
    discovery_payload = None
    if needs_discovery:
        discovery_result = _run(
            ("podman", "exec", container, "bash", "-lc", DISCOVERY_COMMAND),
            cwd=project_root,
            runner=runner,
            timeout=30.0,
        )
        discovery_payload = _json_from_command(discovery_result)

    process_counts = {}
    process_evidence = {}
    for name, pattern in PROCESS_PATTERNS.items():
        result = _run(
            ("podman", "exec", container, "pgrep", "-fc", pattern),
            cwd=project_root,
            runner=runner,
            timeout=10.0,
        )
        try:
            count = int(result.get("stdout") or "")
        except ValueError:
            count = None
        process_counts[name] = count
        process_evidence[name] = {
            "pattern": pattern,
            "returncode": result.get("returncode"),
            "error": result.get("stderr") or None,
        }

    davis_payload = davis_response.get("payload") or {}
    process_counts["davis_device_open_count"] = davis_payload.get("davis_device_open_count")
    process_counts["davis_owner_status"] = davis_payload.get("davis_owner_status")
    process_counts["davis_writer_pid"] = davis_payload.get("davis_writer_pid")
    process_counts["davis_writer_session_id"] = davis_payload.get("davis_writer_session_id")
    process_counts["duplicate_davis_owner_detected"] = davis_payload.get("duplicate_davis_owner_detected")

    system_check_result = _run(
        (
            "podman",
            "exec",
            "-e",
            f"ROS_DOMAIN_ID={ros_domain_id}",
            container,
            "bash",
            "-lc",
            SYSTEM_CHECK_COMMAND,
        ),
        cwd=project_root,
        runner=runner,
        timeout=90.0,
    )
    system_check_summary = parse_system_check_summary(
        system_check_result.get("stdout") or "",
        system_check_result.get("returncode"),
    )

    running = (api_payload or {}).get("running") or {}
    devices = _device_records(api_payload, discovery_payload)
    gate_checks = {
        "container_running": container_record.get("container_running") is True,
        "container_uses_inspected_image": bool(
            container_record.get("image_id")
            and container_record.get("image_id") == image_record.get("image_id")
        ),
        "dashboard_responding": status_response.get("ok") is True,
        "realsense_running": running.get("realsense") is True,
        "davis_running": running.get("davis_motion") is True,
        "realsense_selected": bool(devices["realsense"].get("stable_device_id")),
        "davis_selected": bool(devices["davis"].get("stable_device_id")),
        "one_realsense_writer": process_counts.get("realsense_backend_writer_count") == 1,
        "one_headless_wrapper": process_counts.get("headless_wrapper_count") == 1,
        "one_dashboard_process": process_counts.get("dashboard_process_count") == 1,
        "one_davis_owner": process_counts.get("davis_device_open_count") == 1,
        "no_duplicate_davis_owner": process_counts.get("duplicate_davis_owner_detected") is False,
        "current_camera_info": system_check_summary.get("camera_info_current") is True,
        "system_check_12_0_0": system_check_summary.get("exact_12_pass") is True,
    }
    config_hashes = _config_hashes(project_root)
    status_lines = (git_status.get("stdout") or "").splitlines()
    stamp = datetime.fromtimestamp(captured_at, timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest_path = output_root / f"release_state_{stamp}_{time.time_ns()}.json"
    manifest = {
        "schema_version": "grasp_release_state_v1",
        "created_at_epoch_sec": captured_at,
        "created_at_utc": datetime.fromtimestamp(captured_at, timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_mode": "read_only_inspection_no_restart_no_sensor_mutation",
        "canonical_launch_command": f"cd {shlex.quote(str(project_root))} && ./run_docker_demo.sh",
        "ros_domain_id": {
            "effective": str(ros_domain_id),
            "container": container_record.get("ros_domain_id"),
            "capture_environment": environment.get("ROS_DOMAIN_ID"),
        },
        "git": {
            "head": git_head.get("stdout") or None,
            "head_query_error": git_head.get("stderr") or None,
            "working_tree_dirty": bool(status_lines),
            "status_short": status_lines,
            "status_query_error": git_status.get("stderr") or None,
        },
        "configuration_file_count": len(config_hashes),
        "configuration_sha256": config_hashes,
        "podman": {
            "container": container_record,
            "image": image_record,
            "container_inspect_error": container_inspect.get("stderr") or None,
            "image_inspect_error": image_inspect.get("stderr") or None,
        },
        "dashboard_api": {
            "status_url": status_response.get("url"),
            "status_ok": status_response.get("ok"),
            "status_error": status_response.get("error"),
            "reported_time": (api_payload or {}).get("time"),
            "running": running,
            "davis_status_url": davis_response.get("url"),
            "davis_status_ok": davis_response.get("ok"),
            "davis_status_error": davis_response.get("error"),
        },
        "devices": devices,
        "device_discovery_fallback": {
            "attempted": needs_discovery,
            "returncode": discovery_result.get("returncode") if discovery_result else None,
            "error": (discovery_result.get("stderr") or None) if discovery_result else None,
        },
        "process_ownership": {
            **process_counts,
            "process_evidence": process_evidence,
        },
        "system_check": {
            "command": system_check_result.get("command"),
            "returncode": system_check_result.get("returncode"),
            "summary": system_check_summary,
            "stdout": system_check_result.get("stdout"),
            "stderr": system_check_result.get("stderr"),
        },
        "release_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
        },
        "limitations": [
            "The Git state and image identity are recorded independently; a dirty worktree is not claimed to match the image.",
            "USB bus/device nodes are transient diagnostics and are not used as persistent camera identity.",
            "This capture does not restart the container or open a second streaming pipeline.",
        ],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--davis-status-url")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    path = capture_release_state(
        project_root,
        args.output_root or project_root / "output" / "release_freezes",
        container=args.container,
        image=args.image,
        status_url=args.status_url,
        davis_status_url=args.davis_status_url,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = manifest["system_check"]["summary"]
    print(f"Saved release state: {path}")
    print(
        f"Release gate: {'PASS' if manifest['release_gate']['passed'] else 'FAIL'}; "
        f"system check PASS={summary['pass']} WARN={summary['warn']} FAIL={summary['fail']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

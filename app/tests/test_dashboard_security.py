"""Camera-free regressions for browser controls and untrusted display metadata."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from grasp_project.dashboard import web_dashboard


@pytest.mark.parametrize(
    "host", ["rebind.example:8080", "localhost.attacker.example:8080", "192.0.2.1:8080"]
)
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_nonlocal_hosts_cannot_read_or_control_dashboard_even_with_matching_origin(
    monkeypatch, host, method
):
    mutations = []
    monkeypatch.setattr(
        web_dashboard, "set_desired_process", lambda *args: mutations.append(args)
    )
    monkeypatch.setattr(
        web_dashboard, "stop_process", lambda name: (True, f"Stopped {name}")
    )
    base_url = f"http://{host}"
    path = "/" if method == "GET" else "/api/stop/realsense"

    response = web_dashboard.app.test_client().open(
        path, method=method, base_url=base_url, headers={"Origin": base_url}
    )

    assert response.status_code == 403
    assert response.get_json()["ok"] is False
    assert mutations == []


@pytest.mark.parametrize(
    "base_url", ["http://localhost:8081", "http://127.0.0.1:8080", "http://[::1]:8080"]
)
def test_loopback_hosts_can_read_dashboard_with_custom_ports(base_url):
    response = web_dashboard.app.test_client().get("/", base_url=base_url)
    assert response.status_code == 200
    assert response.mimetype == "text/html"


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://example.com"},
        {"Origin": "http://127.0.0.1:8081"},
        {"Origin": "https://127.0.0.1:8080"},
        {"Origin": "http://127.0.0.1.evil.example:8080"},
        {"Origin": "http://127.0.0.1:invalid"},
        {"Origin": "null"},
        {"Origin": ""},
        {"Referer": "https://example.com/"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_cross_origin_browser_cannot_stop_a_camera(monkeypatch, headers):
    mutations = []
    monkeypatch.setattr(
        web_dashboard, "set_desired_process", lambda *args: mutations.append(args)
    )
    monkeypatch.setattr(
        web_dashboard, "stop_process", lambda *args: mutations.append(args)
    )

    response = web_dashboard.app.test_client().post(
        "/api/stop/realsense", base_url="http://127.0.0.1:8080", headers=headers
    )

    assert response.status_code == 403
    assert response.get_json()["ok"] is False
    assert mutations == []


@pytest.mark.parametrize(
    ("base_url", "headers"),
    [
        ("http://127.0.0.1:8080", {}),  # Native launchers do not send browser headers.
        (
            "http://127.0.0.1:8080",
            {"Origin": "http://127.0.0.1:8080", "Sec-Fetch-Site": "same-origin"},
        ),
        ("http://127.0.0.1:8080", {"Referer": "http://127.0.0.1:8080/?tab=davis"}),
        ("http://localhost", {"Origin": "http://localhost:80"}),
        ("https://localhost", {"Origin": "https://localhost:443"}),
        ("http://[::1]:8080", {"Origin": "http://[::1]:8080"}),
    ],
)
def test_same_origin_browser_and_native_launchers_keep_camera_controls(
    monkeypatch, base_url, headers
):
    mutations = []
    monkeypatch.setattr(
        web_dashboard, "set_desired_process", lambda *args: mutations.append(args)
    )
    monkeypatch.setattr(
        web_dashboard, "stop_process", lambda name: (True, f"Stopped {name}")
    )

    response = web_dashboard.app.test_client().post(
        "/api/stop/realsense", base_url=base_url, headers=headers
    )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert mutations == [("realsense", False)]


def test_container_dashboard_bindings_default_to_loopback():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    assert (
        compose["services"]["grasp-dashboard"]["environment"]["HOST"]
        == "${HOST:-127.0.0.1}"
    )
    windows = (root / "run_docker_demo_windows.ps1").read_text()
    assert '-p "127.0.0.1:8080:8080"' in windows


def test_dashboard_renderers_display_profile_and_benchmark_names_as_text():
    # Use the real renderer bodies. Node is optional for Python-only development;
    # CI runners with Node also exercise the browser JavaScript regression.
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the JavaScript rendering regression")
    names = ("esc", "cards", "statusRows", "qualityRows")
    functions = []
    for name in names:
        match = re.search(
            rf"    function {name}\([^\n]*\n.*?^    }}",
            web_dashboard.INDEX_HTML,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None
        functions.append(match.group(0))
    payload = "<img src=x onerror=\"globalThis.compromised=true\"> & 'profile'"
    script = (
        "\n".join(functions)
        + "\n"
        + """
      const stateClass = () => 'info';
      const payload = PAYLOAD;
      for (const render of [cards, statusRows, qualityRows]) {
        const el = {dataset: {}, innerHTML: ''};
        render(el, [[payload, payload], ['Missing', null], ['Zero', 0]]);
        if (el.innerHTML.includes('<img') || !el.innerHTML.includes('&lt;img')) {
          throw new Error('Untrusted metadata became HTML');
        }
        if (!el.innerHTML.includes('N/A') || !el.innerHTML.includes('>0<')) {
          throw new Error('Missing or zero display semantics changed');
        }
      }
    """.replace("PAYLOAD", json.dumps(payload))
    )
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)

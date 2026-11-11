import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import runpod_startup as boot
from run_gguf_test import pod_config


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(boot.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(boot.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def api_with_events(*lines):
    events = MagicMock()
    events.__enter__.return_value = events
    events.ok = True
    events.status_code = 200
    events.iter_lines.return_value = [b"data: " + json.dumps(dict(source="container", line=l)).encode() for l in lines]
    pod = Mock(ok=True, status_code=200)
    pod.json.return_value = {"desiredStatus": "RUNNING", "machine": {"gpuTypeId": "fixtureGPU"}}
    api = Mock()
    api.get.side_effect = lambda url, **kwargs: events if url.startswith("https://api.runpod.io/v2/") else pod
    return api


def test_timeout_even_when_pod_desired_status_stays_running(monkeypatch, clock):
    state = {}
    monkeypatch.setattr(boot.requests, "get", Mock(return_value=Mock(ok=False, status_code=502)))
    with pytest.raises(TimeoutError, match="startup deadline"):
        boot.wait_ready(api_with_events("CHUCKLES_BOOT download"), "pod", "https://proxy", {}, 17, state)
    assert clock[0] == 17 and state["stage"] == "download" and state["proxy"] == 502


def test_boot_failure_survives_teardown_without_leaking_raw_logs(monkeypatch, clock):
    api = api_with_events("docker run /bin/sh -c 'stage cuda; secret-value'",
                          "CHUCKLES_BOOT cuda", "CHUCKLES_BOOT download",
                          "CHUCKLES_DOWNLOAD http=403 bytes=0", "CHUCKLES_BOOT failed download 22")
    proxy = Mock()
    monkeypatch.setattr(boot.requests, "get", proxy)
    state = {}
    with pytest.raises(RuntimeError, match=r"download failed \(exit 22\)") as error:
        boot.wait_ready(api, "pod", "https://proxy", {}, 720, state)
    assert "secret-value" not in str(error.value)
    assert state["transfer"] == "HTTP 403, 0 bytes" and clock[0] == 0
    proxy.assert_not_called()


def test_transient_api_and_proxy_failures_then_exact_model(monkeypatch, clock):
    api = api_with_events()
    normal_get = api.get.side_effect
    def get(url, **kwargs):
        if clock[0] == 0:
            raise requests.ConnectionError("offline")
        return normal_get(url, **kwargs)
    api.get.side_effect = get
    wrong = Mock(ok=True, status_code=200)
    wrong.json.return_value = {"data": [{"id": "different-model"}]}
    right = Mock(ok=True, status_code=200)
    right.json.return_value = {"data": [{"id": boot.MODEL}]}
    monkeypatch.setattr(boot.requests, "get", Mock(side_effect=[requests.Timeout(), wrong, right]))
    state = {}
    info = boot.wait_ready(api, "pod", "https://proxy", {}, 25, state)
    assert info["machine"]["gpuTypeId"] == "fixtureGPU" and state["stage"] == "ready"


def test_shell_reports_a_failed_download_before_checksum(tmp_path):
    command = pod_config("n", "fixture", "g", "SECURE", 100, 1024)["dockerStartCmd"][0]
    command = command.replace("/app/llama-server", "fake_server").replace("/tmp/devices", str(tmp_path / "devices"))
    stub = "fake_server() { echo CUDA0:; }; curl() { return 22; }; "
    r = subprocess.run(["sh", "-c", stub + command], capture_output=True, text=True)
    assert r.returncode == 22
    assert "CHUCKLES_BOOT failed download 22" in r.stdout and "CHUCKLES_BOOT checksum" not in r.stdout

import json
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import MagicMock, Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import runpod_startup as boot
from run_gguf_test import allocate_pod, pod_config


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(boot.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(boot.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def unavailable():
    return Mock(ok=False, status_code=500, **{"json.return_value": {
        "error": "create pod: There are no instances currently available\nThere are no instances currently available",
        "status": 500,
    }})


def test_capacity_rejections_retry_same_rental_and_exclude_wait_from_rental_time(clock):
    config = pod_config("batch", "fixture", "NVIDIA RTX A6000,NVIDIA A40", "SECURE", 1536, 16384)
    api = Mock()
    api.post.side_effect = [unavailable(), unavailable(), Mock(ok=True, **{"json.return_value": {"id": "ready"}})]
    api.get.return_value.json.return_value = [{"name": "another-batch", "id": "other"}]
    stats = {}
    pod, rental_start = allocate_pod(api, config, 600, stats)
    assert pod == {"id": "ready"} and rental_start == 45
    assert stats == {"attempts": 3, "capacity_rejections": 2}
    assert all(call.kwargs["json"] == config for call in api.post.call_args_list)
    assert api.get.call_count == 2


def test_capacity_reconciliation_picks_up_own_pod_without_reposting(clock):
    api = Mock()
    api.post.return_value = unavailable()
    existing = {"name": "batch", "id": "already-rented"}
    api.get.return_value.json.return_value = [existing]
    assert allocate_pod(api, {"name": "batch"}, 600, {}) == (existing, 0)
    api.post.assert_called_once()


def test_allocation_wait_is_bounded_and_other_errors_are_not_retried(clock):
    api = Mock()
    api.post.return_value = unavailable()
    api.get.return_value.json.return_value = []
    with pytest.raises(TimeoutError, match="3 allocation attempts"):
        allocate_pod(api, {"name": "batch"}, 60, {})
    assert clock[0] == 45
    api.reset_mock()
    denied = Mock(ok=False, status_code=500, text="account balance too low")
    denied.json.return_value = {"error": denied.text}
    api.post.return_value = denied
    with pytest.raises(RuntimeError, match="balance too low"):
        allocate_pod(api, {"name": "batch", "env": {"LLAMA_API_KEY": "fixture"}}, 120, {})
    api.post.assert_called_once()
    api.get.assert_not_called()
    api.reset_mock()
    api.post.side_effect = requests.Timeout()
    with pytest.raises(requests.Timeout):
        allocate_pod(api, {"name": "batch"}, 120, {})
    api.post.assert_called_once()


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
                          "CHUCKLES_RANGE_ERROR 2 403 22", "CHUCKLES_BOOT failed download 22")
    proxy = Mock()
    monkeypatch.setattr(boot.requests, "get", proxy)
    state = {}
    with pytest.raises(RuntimeError, match=r"download failed \(exit 22\)") as error:
        boot.wait_ready(api, "pod", "https://proxy", {}, 720, state)
    assert "secret-value" not in str(error.value)
    assert state["range_error"] == "part 2, HTTP 403, exit 22" and clock[0] == 0
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
    command = command.replace("/tmp/chuckles-download.sh", str(tmp_path / "download.sh"))
    stub = "fake_server() { echo CUDA0:; }; curl() { return 22; }; "
    r = subprocess.run(["sh", "-c", stub + command], capture_output=True, text=True)
    assert r.returncode == 22
    assert "CHUCKLES_BOOT failed download 22" in r.stdout and "CHUCKLES_BOOT checksum" not in r.stdout


@pytest.mark.parametrize("honor_range", [True, False])
def test_pinned_ranges_reassemble_exact_bytes_or_reject_full_response(tmp_path, honor_range):
    model = bytes(range(251)) * 31
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            first, last = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            body = model[first:last+1] if honor_range else model
            self.send_response(206 if honor_range else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            cmd = (Path(boot.__file__).with_name("runpod_download.sh")).read_text()
            cmd += '\ndownload_weights "$1" "$2" "$3"'
            out = tmp_path / "model"
            url = f"http://127.0.0.1:{server.server_port}/weights?v=1"
            run = subprocess.run(["sh", "-c", cmd, "test", url, str(len(model)), str(out)],
                                 capture_output=True, text=True, timeout=15)
        finally:
            server.shutdown()
            thread.join()
    if honor_range:
        assert run.returncode == 0 and out.read_bytes() == model
    else:
        assert run.returncode and not out.exists()

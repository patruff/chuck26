"""Observe a RunPod cold start while the pod and its logs still exist."""

import json
import re
import time

import requests

from settings import MODEL

PODS = "https://rest.runpod.io/v1/pods"
LOGS = "https://api.runpod.io/v2/pods/{}/logs"


def startup_event(source, line):
    # Provider logs can contain credentials, signed URLs and the launch command.
    # Only report our own structured markers and fixed platform diagnoses.
    marker = re.search(r"(?:^|\s)CHUCKLES_BOOT (failed )?(cuda|download|checksum|load)(?: (\d{1,3}))?$", line)
    if marker and source == "container":
        failed, stage, code = marker.groups()
        if failed and code:
            return "error", f"{stage} failed (exit {code})"
        if not failed and not code:
            return "stage", stage
    transfer = re.search(r"(?:^|\s)CHUCKLES_DOWNLOAD http=(\d{3}) bytes=(\d+)$", line)
    if transfer and source == "container":
        return "transfer", f"HTTP {transfer[1]}, {transfer[2]} bytes"
    lower = line.lower()
    for needle, reason in (
        ("no space left on device", "host or container disk full"),
        ("unsatisfied condition: cuda", "host driver incompatible with container CUDA"),
        ("driver version is insufficient", "host driver incompatible with container CUDA"),
        ("cuda_error_out_of_memory", "GPU out of memory"),
        ("cudamalloc failed: out of memory", "GPU out of memory"),
    ):
        if needle in lower:
            return "error", reason
    if source == "system":
        if "pull" in lower or "fetching image" in lower:
            return "platform", "image pull"
        if "start container" in lower or "starting container" in lower:
            return "platform", "container start"
    return None


def read_startup_logs(api, pod_id, state, deadline):
    # This is an SSE stream even for a tail request. Limit the replay as well as
    # the idle read so a chatty pod cannot delay probes or cleanup.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    until = min(deadline, time.monotonic() + 5)
    try:
        with api.get(LOGS.format(pod_id), params={"tail": 100}, stream=True,
                     timeout=(min(3, remaining), min(2, remaining))) as response:
            state["logs_http"] = response.status_code
            if not response.ok:
                return
            for raw in response.iter_lines(chunk_size=1024):
                if time.monotonic() >= until:
                    break
                if not raw.startswith(b"data:"):
                    continue
                try:
                    event = json.loads(raw[5:])
                except ValueError:
                    continue
                if isinstance(event, dict) and isinstance(event.get("line"), str):
                    observation = startup_event(event.get("source"), event["line"].strip())
                    if observation:
                        key, value = observation
                        state[key] = value
    except requests.RequestException:
        # A connection with no new SSE events normally ends in a read timeout.
        pass


def wait_ready(api, pod_id, endpoint, headers, deadline, state):
    start = time.monotonic()
    next_logs = next_report = start
    previous = None
    info = {}
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = api.get(f"{PODS}/{pod_id}", params={"includeMachine": "true"}, timeout=min(5, remaining))
            state["pod_http"] = response.status_code
            if response.ok:
                info = response.json()
                state["desired_status"] = info.get("desiredStatus")
                state["gpu"] = (info.get("machine") or {}).get("gpuTypeId")
        except (requests.RequestException, ValueError):
            state["pod_http"] = "unreachable"
        now = time.monotonic()
        stopped = state.get("desired_status") in ("EXITED", "TERMINATED", "STOPPED")
        if now >= next_logs or stopped:
            read_startup_logs(api, pod_id, state, deadline)
            next_logs = now + 20
        if state.get("error") or stopped:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            r = requests.get(endpoint + "/v1/models", headers=headers, timeout=min(5, remaining))
            state["proxy"] = r.status_code
            if r.ok and any(m.get("id") == MODEL for m in r.json().get("data", [])):
                state["stage"] = "ready"
                print(f"Model ready on {pod_id} after {time.monotonic() - start:.0f}s", flush=True)
                return info
        except (requests.RequestException, ValueError):
            state["proxy"] = "unreachable"
        now = time.monotonic()
        if state != previous or now >= next_report:
            print(f"RunPod startup {pod_id} (+{now - start:.0f}s): {json.dumps(state)}", flush=True)
            previous = state.copy()
            next_report = now + 60
        time.sleep(max(0, min(8, deadline - time.monotonic())))
    detail = json.dumps(state)
    if state.get("error") or state.get("desired_status") in ("EXITED", "TERMINATED", "STOPPED"):
        raise RuntimeError(f"RunPod startup failed on {pod_id}: {detail}")
    raise TimeoutError(f"RunPod model startup deadline reached on {pod_id}: {detail}")

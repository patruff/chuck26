#!/usr/bin/env python3
"""Run the Qwen3.8 27B GGUF on one ephemeral RunPod GPU and save Chuckles results."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import requests
from march_source import BANK_URL, NOTEBOOK, preflight, snapshot_bank
from human_guidance import prepare_guidance

PIPELINE = Path(__file__).resolve().parent
MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF"
REVISION = "ab8d7bd81d03117501e4da60cccdff4c80f0eea0"
FILENAME = "RVN-Q4_K_M-multilingual.gguf"
SHA256 = "9ed40ccc8b8432f38b9a85d0ca67928167f5719e8c10bd299e56d34facaf6e61"
# llama.cpp b12182, CUDA 12.8, linux/amd64.
IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:859efa314dc34f1843a38ff28a4818aeac51a4cce125bdea4d2678aceba95469"
PODS = "https://rest.runpod.io/v1/pods"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pod_config(name, server_key, gpu, cloud, reasoning_budget, context):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{FILENAME}"
    boot = (
        "set -eu; /app/llama-server --list-devices > /tmp/devices 2>&1; cat /tmp/devices; "
        "grep -q 'CUDA0:' /tmp/devices; "
        f"curl -fL --retry 3 -o /tmp/model.gguf '{url}'; "
        f"echo '{SHA256}  /tmp/model.gguf' | sha256sum -c -; "
        f"exec /app/llama-server -m /tmp/model.gguf --alias {MODEL} "
        "--host 0.0.0.0 --port 8080 --jinja --reasoning-format deepseek "
        f"--device CUDA0 --reasoning-budget {reasoning_budget} -ngl 99 -c {context} -np 1"
    )
    return {
        "name": name, "imageName": IMAGE, "cloudType": cloud,
        "gpuTypeIds": [g.strip() for g in gpu.split(",")], "gpuTypePriority": "custom", "gpuCount": 1,
        # REST v1's CUDA enum stops at 13.0. Secure hosts can have newer drivers.
        **({"allowedCudaVersions": ["12.8", "12.9", "13.0"]} if cloud == "COMMUNITY" else {}),
        "containerDiskInGb": 32, "volumeInGb": 0, "ports": ["8080/http"],
        "dockerEntrypoint": ["/bin/sh", "-c"], "dockerStartCmd": [boot],
        "env": {"LLAMA_API_KEY": server_key},
    }


def terminate(api, pod_id):
    """A repeated delete or a 404 both mean teardown is complete."""
    url = f"{PODS}/{pod_id}"
    for attempt in range(5):
        try:
            res = api.delete(url, timeout=20)
            if res.status_code == 404:
                return
            res.raise_for_status()
            time.sleep(2)
            state = api.get(url, timeout=20)
            if state.status_code == 404 or (
                state.ok and state.json().get("desiredStatus") == "TERMINATED"
            ):
                return
        except requests.RequestException:
            pass
        time.sleep(2 + attempt)
    raise RuntimeError(f"Could not confirm termination of RunPod pod {pod_id}")


def wait_ready(api, pod_id, endpoint, headers, deadline):
    last_status = None
    while time.monotonic() < deadline:
        info = api.get(f"{PODS}/{pod_id}", params={"includeMachine": "true"}, timeout=20)
        if info.ok and info.json().get("desiredStatus") in ("EXITED", "TERMINATED", "STOPPED"):
            raise RuntimeError(f"Pod exited during startup: {info.json()['desiredStatus']}")
        try:
            r = requests.get(endpoint + "/v1/models", headers=headers, timeout=10)
            if r.ok and MODEL in [m["id"] for m in r.json()["data"]]:
                return info.json() if info.ok else {}
            status = r.status_code
        except requests.RequestException:
            status = "unreachable"
        if status != last_status:
            print(f"Waiting for model on {pod_id} (proxy {status})", flush=True)
            last_status = status
        time.sleep(10)
    raise TimeoutError("The GGUF server did not become ready in time")


def main(private_dir):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--titles", type=Path, help="Input CSV with a title column")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--output-dir", type=Path, help="New directory for this run")
    p.add_argument("--preflight-only", action="store_true", help="Verify the 2025 bank and tools on CPU and exit")
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--march-word-bank", type=Path, metavar="WORD_STRUCTURES.PY",
                         help="Run CodeAgent with the March word bank and custom pronunciations")
    sources.add_argument("--colab-notebook", type=Path, metavar="NOTEBOOK.IPYNB",
                         help="Run CodeAgent with the Colab funny_words list and 0.8 suggestion cutoff")
    sources.add_argument("--base-only", action="store_true", help="Use the original single-answer GGUF baseline")
    p.add_argument("--example-notebook", type=Path, metavar="NOTEBOOK.IPYNB",
                   help="With --march-word-bank, replay the saved Running Man cell and both finalists' word checks")
    p.add_argument("--humor-first", action="store_true",
                   help="Use the humor-first batch cell, record drafts before verification, and use a 0.60 floor")
    p.add_argument("--drafts-from", type=Path, help="Reuse initial pre-score drafts from an earlier humor-first run")
    p.add_argument("--human-guidance", type=Path, metavar="FEEDBACK.TXT",
                   help="Have Qwen revise its creative instructions from examples and human comments before drafting")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A6000,NVIDIA A40",
                   help="GPU type or comma-separated preference order (24 GB VRAM minimum)")
    p.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="SECURE")
    p.add_argument("--timeout-mins", type=int, default=80)
    args = p.parse_args()
    production = not (args.base_only or args.colab_notebook)
    plan = None
    if args.human_guidance:
        if not production or args.drafts_from:
            p.error("--human-guidance requires the 2025 tools and fresh drafts")
        if args.human_guidance.suffix.lower() != ".txt" or not args.human_guidance.is_file():
            p.error("--human-guidance requires an existing UTF-8 .txt file")
        from tokenizers import Tokenizer
        from export_humor_candidates import pinned_tokenizer_file
        raw_guidance = args.human_guidance.read_bytes()
        try:
            plan = prepare_guidance(raw_guidance, Tokenizer.from_file(pinned_tokenizer_file(MODEL, REVISION)))
        except ValueError as exc:
            p.error(str(exc))
        args.humor_first = True
    if not production and (args.example_notebook or args.humor_first or args.preflight_only):
        p.error("The humor-first notebook and preflight require the 2025 bank")
    if production:
        try:
            args.march_word_bank = snapshot_bank(args.march_word_bank, private_dir)
        except (ValueError, OSError, requests.RequestException, subprocess.TimeoutExpired) as exc:
            p.error(str(exc))
        if args.example_notebook is None:
            args.example_notebook = NOTEBOOK
            args.humor_first = True
        if not args.example_notebook.is_file():
            p.error("Example notebook file does not exist")
        controls = preflight(args.march_word_bank)
        print(json.dumps(controls, indent=2), flush=True)
        if args.preflight_only:
            return
    if args.output_dir is None:
        p.error("A GPU run requires --output-dir")
    if args.titles is None:
        args.titles = PIPELINE / ("humor_sample_titles.csv" if production else "movie_titles_1995_1999.csv")
    if args.limit < 1 or args.timeout_mins < 1 or not args.titles.is_file():
        p.error("Use an existing titles CSV, a positive limit and a positive timeout")
    if not os.getenv("RUNPOD_API_KEY"):
        p.error("RUNPOD_API_KEY is required")
    source = args.march_word_bank or args.colab_notebook
    if source and not source.is_file():
        p.error("The word bank or notebook must be a local file")
    if args.example_notebook and (not args.march_word_bank or not args.example_notebook.is_file()):
        p.error("--example-notebook requires a local notebook and --march-word-bank")
    if args.humor_first and not args.example_notebook:
        p.error("--humor-first requires --example-notebook")
    if args.drafts_from and (not args.humor_first or not args.drafts_from.is_file()):
        p.error("--drafts-from needs a local results JSONL and --humor-first")
    reasoning_budget, context = (1536, 16384) if source else (768, 4096)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if plan:
        (args.output_dir / "guidance.txt").write_bytes(raw_guidance)
        (args.output_dir / "editor-input.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    api = requests.Session()
    api.headers["Authorization"] = "Bearer " + os.environ["RUNPOD_API_KEY"]
    server_key = secrets.token_urlsafe(32)
    name = "chuckles-gguf-" + uuid.uuid4().hex[:12]
    meta = {
        "pod_name": name, "model": MODEL, "revision": REVISION, "file": FILENAME,
        "sha256": SHA256, "image": IMAGE, "gpu_requested": args.gpu, "cloud": args.cloud,
        "started_at": utc_now(), "titles_file": args.titles.name, "limit": args.limit,
        "temperature": 0.7 if source else 0.4, "seed": 42,
        "reasoning_budget": reasoning_budget, "context": context,
    }
    if source:
        meta.update(profile="colab-codeagent" if args.colab_notebook else "march-codeagent",
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    suggestion_cutoff=0.8 if args.colab_notebook else 0.6)
        if production:
            meta.update(word_bank_url=BANK_URL, preflight=controls)
    if args.example_notebook:
        meta.update(profile="running-man-codeagent",
                    notebook_sha256=hashlib.sha256(args.example_notebook.read_bytes()).hexdigest())
    if args.humor_first:
        meta.update(profile="humor-first-codeagent", reference_cell="GvGq8GcoGbCD", phonetic_floor=0.6,
                    changed_word_floor_measure="parody-suggestions", auxiliary_measure="word-phone")
    if args.drafts_from:
        meta["draft_source_sha256"] = hashlib.sha256(args.drafts_from.read_bytes()).hexdigest()
    if plan:
        meta.update(guidance_sha256=plan["feedback_sha256"], guidance_file=args.human_guidance.name)
    manifest = args.output_dir / "run.json"
    def save():
        manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Run interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    deadline = start + args.timeout_mins * 60
    pod_id = None
    save()
    try:
        config = pod_config(name, server_key, args.gpu, args.cloud, reasoning_budget, context)
        response = api.post(PODS, json=config, timeout=75)
        if not response.ok:
            # API errors can echo the request, which contains the temporary server credential.
            raise RuntimeError(f"RunPod creation returned {response.status_code}: "
                               + response.text.replace(server_key, "[redacted]")[:500])
        pod = response.json()
        pod_id = pod["id"]
        meta.update(pod_id=pod_id, cost_per_hour=pod.get("costPerHr"))
        save()
        print(f"Created pod {pod_id} at ${meta['cost_per_hour']}/hr", flush=True)
        endpoint = f"https://{pod_id}-8080.proxy.runpod.net"
        info = wait_ready(api, pod_id, endpoint, {"Authorization": "Bearer " + server_key}, deadline)
        meta.update(ready_at=utc_now(), gpu=(info.get("machine") or {}).get("gpuTypeId"))
        save()
        env = {**os.environ, "LLAMA_API_KEY": server_key}
        for name in ("RUNPOD_API_KEY", "CHUCKLES_2025_BANK_GZ", "CHUCK2025_READ_TOKEN"):
            env.pop(name, None)
        if source:
            option = "--colab-notebook" if args.colab_notebook else "--word-bank"
            generator = [str(PIPELINE / "generate_march.py"), option, str(source)]
            if args.example_notebook:
                generator.extend(["--example-notebook", str(args.example_notebook)])
            if args.humor_first:
                generator.append("--humor-first")
            if args.drafts_from:
                generator.extend(["--drafts-from", str(args.drafts_from)])
            if plan:
                generator.extend(["--guidance-plan", str(args.output_dir / "editor-input.json")])
        else:
            generator = [str(PIPELINE / "generate_with_reasoning.py"), "--no-adapter", "--max-new-tokens", "1280"]
        subprocess.run([
            sys.executable, *generator, "--llama-server", endpoint, "--base-model", MODEL,
            "--titles", str(args.titles), "--limit", str(args.limit),
            "--output", str(args.output_dir / "results.jsonl"),
        ], check=True, env=env, timeout=max(1, deadline - time.monotonic()))
        meta["generation_status"] = "complete"
    finally:
        # A lost creation response can still allocate a pod. Reconcile only this unique name.
        if not pod_id:
            listed = api.get(PODS, timeout=30)
            listed.raise_for_status()
            pod_id = next((d["id"] for d in listed.json() if d.get("name") == name), None)
        if pod_id:
            meta["pod_id"] = pod_id
            save()
            terminate(api, pod_id)
            meta["termination_confirmed"] = True
            print(f"Confirmed teardown of {pod_id}", flush=True)
        meta["ended_at"] = utc_now()
        meta["elapsed_seconds"] = round(time.monotonic() - start, 2)
        if meta.get("cost_per_hour") is not None:
            meta["estimated_compute_usd"] = round(float(meta["cost_per_hour"]) * meta["elapsed_seconds"] / 3600, 4)
        save()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="chuckles-2025-") as directory:
        main(Path(directory))

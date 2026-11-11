#!/usr/bin/env python3
"""Run the Qwen3.8 27B GGUF on one ephemeral RunPod GPU and save Chuckles results."""

import argparse
import base64
import csv
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
from common import read_titles
from runpod_startup import PODS, wait_ready

PIPELINE = Path(__file__).resolve().parent
from settings import MODEL, REVISION, FILENAME, SHA256, MODEL_CONFIG, live_revision
IMAGE = MODEL_CONFIG["runpod_image"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pod_config(name, server_key, gpu, cloud, reasoning_budget, context, slots=1):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{FILENAME}"
    # Keep the API's shell command one line even when the download helper has comments.
    download_script = base64.b64encode((PIPELINE / "runpod_download.sh").read_bytes()).decode("ascii")
    boot = (
        f"set -eu; echo '{download_script}' | base64 -d > /tmp/chuckles-download.sh; . /tmp/chuckles-download.sh; "
        'stage() { echo "CHUCKLES_BOOT $1"; }; '
        'failed() { echo "CHUCKLES_BOOT failed $1 $2"; exit "$2"; }; '
        'stage cuda; /app/llama-server --list-devices > /tmp/devices 2>&1 || failed cuda "$?"; '
        'cat /tmp/devices; grep -q "CUDA0:" /tmp/devices || failed cuda "$?"; '
        f'stage download; download_weights "{url}?download=true&boot=$(date +%s)" '
        f'{MODEL_CONFIG["size_bytes"]} /tmp/model.gguf || failed download "$?"; '
        f"stage checksum; echo '{SHA256}  /tmp/model.gguf' | sha256sum -c - || failed checksum \"$?\"; "
        f"stage load; /app/llama-server -m /tmp/model.gguf --alias {MODEL} "
        "--host 0.0.0.0 --port 8080 --jinja --reasoning-format deepseek "
        f'--device CUDA0 --reasoning-budget {reasoning_budget} -ngl 99 -c {context * slots} -np {slots} || failed load "$?"'
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


def find_named_pod(api, name, timeout=30):
    listed = api.get(PODS, timeout=timeout)
    listed.raise_for_status()
    return next((pod for pod in listed.json() if pod.get("name") == name), None)


def capacity_rejection(response):
    if response.status_code not in (409, 500, 503):
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    message = body.get("error") if isinstance(body, dict) else None
    if not isinstance(message, str):
        return False
    lines = [s.strip().lower().removeprefix("create pod: ").rstrip(".")
             for s in message.splitlines() if s.strip()]
    return bool(lines) and all(s == "there are no instances currently available" for s in lines)


def allocate_pod(api, config, deadline, allocation):
    """Only repeat an explicit capacity rejection, using the original pod name and GPU list."""
    delay = 15
    last_request = None
    allocation.update(attempts=0, capacity_rejections=0)
    while (remaining := deadline - time.monotonic()) > 0:
        if last_request is not None:
            existing = find_named_pod(api, config["name"], timeout=min(20, remaining))
            if existing:
                return existing, last_request
        last_request = time.monotonic()
        if last_request >= deadline:
            break
        allocation["attempts"] += 1
        response = api.post(PODS, json=config, timeout=min(75, deadline - last_request))
        if response.ok:
            return response.json(), last_request
        if not capacity_rejection(response):
            detail = response.text.replace(config["env"]["LLAMA_API_KEY"], "[redacted]")[:500]
            raise RuntimeError(f"RunPod creation returned {response.status_code}: {detail}")
        allocation["capacity_rejections"] += 1
        remaining = deadline - time.monotonic()
        if remaining <= delay:
            break
        print(f"RunPod has no available GPU (attempt {allocation['attempts']}). "
              f"Checking again in {delay}s.", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise TimeoutError(f"There are no instances currently available after {allocation['attempts']} allocation attempts")


def main(private_dir):
    p = argparse.ArgumentParser(description=__doc__)
    movies = p.add_mutually_exclusive_group()
    movies.add_argument("--titles", type=Path, help="Input CSV with a title column")
    movies.add_argument("--title", help="Run just this movie")
    p.add_argument("--limit", type=int, help="Optional cap. An explicit titles CSV defaults to every row.")
    p.add_argument("--output-dir", type=Path, help="New directory for this run")
    p.add_argument("--preflight-only", action="store_true", help="Verify the 2025 bank and tools on CPU and exit")
    p.add_argument("--march-word-bank", type=Path, metavar="WORD_STRUCTURES.PY",
                   help="Local copy of the pinned 2025 word bank and pronunciations")
    p.add_argument("--example-notebook", type=Path, metavar="NOTEBOOK.IPYNB",
                   help="With --march-word-bank, replay the saved Running Man cell and both finalists' word checks")
    p.add_argument("--humor-first", action="store_true",
                   help="Use the humor-first batch cell, record drafts before verification, and use a 0.60 floor")
    p.add_argument("--drafts-from", type=Path, help="Reuse initial pre-score drafts from an earlier humor-first run")
    p.add_argument("--human-guidance", type=Path, metavar="FEEDBACK.TXT",
                   help="Have Qwen revise its creative instructions from examples and human comments before drafting")
    p.add_argument("--pair-revision", type=Path, help="Run two independent samples per ranked film with a saved guidance revision")
    p.add_argument("--slots", type=int, default=1, help="Concurrent llama.cpp slots for a paired film batch")
    p.add_argument("--checkpoints", type=Path, help="Persistent directory for a paired batch's per-film samples")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A6000,NVIDIA A40",
                   help="GPU type or comma-separated preference order (24 GB VRAM minimum)")
    p.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="SECURE")
    p.add_argument("--timeout-mins", type=int, default=80)
    p.add_argument("--startup-timeout-mins", type=int, default=18,
                   help="Maximum minutes to bring up the model, within the overall run deadline")
    args = p.parse_args()
    plan = None
    if args.human_guidance:
        if args.drafts_from:
            p.error("--human-guidance requires the 2025 tools and fresh drafts")
        if args.human_guidance.suffix.lower() != ".txt" or not args.human_guidance.is_file():
            p.error("--human-guidance requires an existing UTF-8 .txt file")
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        raw_guidance = args.human_guidance.read_bytes()
        try:
            plan = prepare_guidance(raw_guidance, Tokenizer.from_file(hf_hub_download(MODEL, "GSQ-3bit/tokenizer.json", revision=REVISION)))
        except ValueError as exc:
            p.error(str(exc))
        args.humor_first = True
    if args.pair_revision:
        if not args.titles or args.slots < 1 or not args.checkpoints:
            p.error("--pair-revision needs --titles, --checkpoints and positive --slots")
        frozen = json.loads(args.pair_revision.read_text(encoding="utf-8"))
        if ((plan and frozen.get("feedback_sha256") != plan["feedback_sha256"]) or (not plan and frozen != live_revision()) or frozen.get("model") != MODEL
                or hashlib.sha256(frozen["applied_instructions"].encode()).hexdigest() != frozen.get("applied_instructions_sha256")):
            p.error("The frozen prompt revision does not match the feedback, model or instruction hash")
        with args.titles.open(newline="",encoding="utf-8") as f:
            films = list(csv.DictReader(f))
        if not films or any(not f.get("title") or not f.get("rank","").isdigit() for f in films):
            p.error("The paired film list needs rank and title columns")
        args.limit = len(films)
    elif args.slots != 1 or args.checkpoints:
        p.error("--slots and --checkpoints require a paired batch")
    try:
        args.march_word_bank = snapshot_bank(args.march_word_bank, private_dir)
    except (ValueError, OSError, requests.RequestException, subprocess.TimeoutExpired) as exc:
        p.error(str(exc))
    args.example_notebook = args.example_notebook or NOTEBOOK
    args.humor_first = True
    if not args.example_notebook.is_file():
        p.error("Example notebook file does not exist")
    controls = preflight(args.march_word_bank)
    print(json.dumps(controls, indent=2), flush=True)
    if args.preflight_only:
        return
    if args.output_dir is None:
        p.error("A GPU run requires --output-dir")
    if args.title is not None:
        if not args.title.strip():
            p.error("--title requires a nonempty movie title")
        args.titles = private_dir / "single-movie.csv"
        with args.titles.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["title"], [args.title.strip()]])
        args.limit = 1
    if args.titles is None:
        args.titles = PIPELINE / "humor_sample_titles.csv"
    if not args.titles.is_file():
        p.error("The titles CSV does not exist")
    try:
        count = len(read_titles(args.titles))
    except (OSError, csv.Error, ValueError) as exc:
        p.error(str(exc))
    if args.limit is None:
        args.limit = count
    if not count or min(args.limit, args.timeout_mins, args.startup_timeout_mins) < 1:
        p.error("Use an existing titles CSV, a positive limit and a positive timeout")
    if not os.getenv("RUNPOD_API_KEY"):
        p.error("RUNPOD_API_KEY is required")
    source = args.march_word_bank
    if args.drafts_from and not args.drafts_from.is_file():
        p.error("--drafts-from needs a local results JSONL")
    reasoning_budget, context = 1536, 16384
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if plan:
        (args.output_dir / "guidance.txt").write_bytes(raw_guidance)
        (args.output_dir / "editor-input.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.pair_revision:
        (args.output_dir / "prompt-revision.json").write_bytes(args.pair_revision.read_bytes())
    elif not plan:
        (args.output_dir / "prompt-revision.json").write_text(json.dumps(live_revision(),ensure_ascii=False,indent=2)+"\n")
    api = requests.Session()
    api.headers["Authorization"] = "Bearer " + os.environ["RUNPOD_API_KEY"]
    server_key = secrets.token_urlsafe(32)
    name = "chuckles-gguf-" + uuid.uuid4().hex[:12]
    meta = {
        "pod_name": name, "model": MODEL, "revision": REVISION, "file": FILENAME,
        "sha256": SHA256, "image": IMAGE, "gpu_requested": args.gpu, "cloud": args.cloud,
        "started_at": utc_now(), "titles_file": args.titles.name, "limit": args.limit,
        "temperature": 0.7, "seed": 42,
        "reasoning_budget": reasoning_budget, "context": context,
        "startup_timeout_mins": min(args.startup_timeout_mins, args.timeout_mins), "startup": {}, "allocation": {},
    }
    meta.update(profile="humor-first-codeagent", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                word_bank_url=BANK_URL, preflight=controls, suggestion_cutoff=0.6,
                reference_cell="GvGq8GcoGbCD", phonetic_floor=0.6,
                changed_word_floor_measure="parody-suggestions", auxiliary_measure="word-phone")
    if args.drafts_from:
        meta["draft_source_sha256"] = hashlib.sha256(args.drafts_from.read_bytes()).hexdigest()
    if args.title:
        meta["title"] = args.title.strip()
    if plan:
        meta.update(guidance_sha256=plan["feedback_sha256"], guidance_file=args.human_guidance.name)
    if args.pair_revision:
        meta.update(slots=args.slots, sampled_sides=2, ranks=[int(f["rank"]) for f in films],
                    applied_instructions_sha256=frozen["applied_instructions_sha256"])
    manifest = args.output_dir / "run.json"
    def save():
        manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Run interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    deadline = start + args.timeout_mins * 60
    pod_id = None
    rental_start = None
    save()
    try:
        config = pod_config(name, server_key, args.gpu, args.cloud, reasoning_budget, context, args.slots)
        pod, rental_start = allocate_pod(api, config, min(deadline, start + 600), meta["allocation"])
        pod_id = pod["id"]
        meta.update(pod_id=pod_id, cost_per_hour=pod.get("costPerHr"), allocated_at=utc_now())
        save()
        print(f"Created pod {pod_id} at ${meta['cost_per_hour']}/hr", flush=True)
        endpoint = f"https://{pod_id}-8080.proxy.runpod.net"
        startup_deadline = min(deadline, time.monotonic() + args.startup_timeout_mins * 60)
        info = wait_ready(api, pod_id, endpoint, {"Authorization": "Bearer " + server_key},
                          startup_deadline, meta["startup"])
        meta.update(ready_at=utc_now(), gpu=(info.get("machine") or {}).get("gpuTypeId"))
        save()
        env = {**os.environ, "LLAMA_API_KEY": server_key}
        for secret_name in ("RUNPOD_API_KEY", "CHUCKLES_2025_BANK_GZ", "CHUCK2025_READ_TOKEN"):
            env.pop(secret_name, None)
        generator = [str(PIPELINE / "generate_march.py"), "--word-bank", str(source)]
        if args.example_notebook:
            generator.extend(["--example-notebook", str(args.example_notebook)])
        if args.humor_first:
            generator.append("--humor-first")
        if args.drafts_from:
            generator.extend(["--drafts-from", str(args.drafts_from)])
        if plan:
            generator.extend(["--guidance-plan", str(args.output_dir / "editor-input.json")])
        else:
            generator.extend(["--applied-revision",str(args.output_dir / "prompt-revision.json")])
        if args.pair_revision:
            command = [sys.executable, str(PIPELINE / "generate_film_pairs.py"), "--server", endpoint,
                       "--model", MODEL, "--bank", str(source), "--notebook", str(args.example_notebook),
                       "--revision", str(args.output_dir / "prompt-revision.json"), "--titles", str(args.titles),
                       "--directory", str(args.checkpoints), "--slots", str(args.slots)]
        else:
            command = [sys.executable, *generator, "--llama-server", endpoint, "--base-model", MODEL,
                       "--titles", str(args.titles), "--limit", str(args.limit),
                       "--output", str(args.output_dir / "results.jsonl")]
        subprocess.run(command, check=True, env=env, timeout=max(1, deadline - time.monotonic()))
        meta["generation_status"] = "complete"
    finally:
        # A lost creation response can still allocate a pod. Reconcile only this unique name.
        if not pod_id:
            existing = find_named_pod(api, name)
            pod_id = existing["id"] if existing else None
        if pod_id:
            meta["pod_id"] = pod_id
            save()
            terminate(api, pod_id)
            meta["termination_confirmed"] = True
            print(f"Confirmed teardown of {pod_id}", flush=True)
        meta["ended_at"] = utc_now()
        meta["elapsed_seconds"] = round(time.monotonic() - start, 2)
        if rental_start is not None and meta.get("cost_per_hour") is not None:
            meta["rental_seconds"] = round(time.monotonic() - rental_start, 2)
            meta["estimated_compute_usd"] = round(float(meta["cost_per_hour"]) * meta["rental_seconds"] / 3600, 4)
        save()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="chuckles-2025-") as directory:
        main(Path(directory))

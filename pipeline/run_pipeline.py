#!/usr/bin/env python3
"""Orchestrate the reasoning SFT training run on a RunPod GPU pod.

No SSH required: the pod's docker command clones this (public) repo and runs
pipeline/pod_bootstrap.sh, which installs deps, trains, pushes the adapter to
the HF Hub, and uploads its run log. This script creates the pod, polls the
Hub until the adapter lands (or the pod dies / times out), prints the pod
log, and ALWAYS terminates the pod at the end.

Usage:
    export RUNPOD_API_KEY=... HF_TOKEN=hf_...
    python pipeline/run_pipeline.py --test          # cheap 0.6B validation
    python pipeline/run_pipeline.py                 # full Qwen3-8B run

    # emergency teardown of a leftover pod:
    python pipeline/run_pipeline.py --down <pod-id>

Requires: pip install runpod huggingface_hub
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

# Everything the pod installs before training/inference. smolagents +
# pronouncing are inference-only (tool-based scoring); training doesn't
# need them. Kept in sync with pod_bootstrap.sh.
PIP_DEPS = [
    "transformers>=4.51",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    # inference smoke test only:
    "smolagents",
    "pronouncing",
]

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# Tried in order until one schedules; all fit an 8B QLoRA run.
DEFAULT_GPU = (
    "NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA RTX A6000,"
    "NVIDIA A40,NVIDIA A100 80GB PCIe"
)
REPO_URL = "https://github.com/patruff/chuck26.git"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset-repo", default="patruff/chuckles-reasoning-sft")
    p.add_argument("--adapter-repo", default="patruff/chuckles-reasoning-adapter")
    p.add_argument("--epochs", default="3")
    p.add_argument("--git-ref", default="main", help="Branch/tag the pod checks out.")
    p.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help="RunPod GPU type id(s), comma-separated fallback order.",
    )
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--disk-gb", type=int, default=0, help="0 = auto (40 test / 100 full).")
    p.add_argument(
        "--volume-gb",
        type=int,
        default=0,
        help="Network volume size. Default 0 (none) -- outputs go to the HF "
        "Hub, and volume-less pods schedule far more reliably.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Cheap validation: Qwen3-0.6B, 20 steps, test adapter repo.",
    )
    p.add_argument("--no-inference", action="store_true")
    p.add_argument(
        "--infer-titles",
        default="pipeline/movie_titles_1995_1999.csv",
        help="Titles CSV (repo-relative) the pod generates parodies for.",
    )
    p.add_argument(
        "--infer-limit", default="", help="Max inference titles (default: 3 in test mode, all otherwise)."
    )
    p.add_argument("--timeout-mins", type=int, default=0, help="0 = auto (60 test / 240 full).")
    p.add_argument("--poll-secs", type=int, default=60)
    p.add_argument("--keep-pod", action="store_true", help="Skip termination (debugging).")
    p.add_argument("--down", metavar="POD_ID", default="", help="Just terminate this pod and exit.")
    return p.parse_args()


def hf_adapter_landed(api, repo_id: str, after: dt.datetime) -> bool:
    """True once the adapter repo holds adapter files newer than `after`."""
    try:
        info = api.model_info(repo_id, files_metadata=False)
    except Exception:
        return False
    files = {s.rfilename for s in (info.siblings or [])}
    if "adapter_config.json" not in files:
        return False
    if not any(f.startswith("adapter_model.") for f in files):
        return False
    last = info.last_modified
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    return last > after


def fetch_dataset_file(dataset_repo: str, path_in_repo: str) -> str | None:
    """Download a file the pod uploaded to the dataset repo, if present."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            dataset_repo, path_in_repo, repo_type="dataset", force_download=True
        )
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def print_inference_results(raw_jsonl: str) -> list[dict]:
    """Pretty-print the parodies the fine-tuned model produced."""
    import json

    rows = []
    for line in raw_jsonl.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return []
    print("\n===== fine-tuned model parodies =====")
    for r in rows:
        if "error" in r:
            print(f"  {r['input_title']!r}: ERROR {r['error']}")
        else:
            print(
                f"  {r['input_title']!r} -> {r.get('parody')!r} "
                f"(phonetic {r.get('avg_phonetic_score')})"
            )
    print("===== end parodies =====\n")
    return rows


def write_github_summary(
    success: bool, adapter_repo: str, elapsed_mins: float,
    cost_per_hr: float | None, results: list[dict],
) -> None:
    """Append a markdown summary when running inside GitHub Actions."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    est_cost = (
        f"${cost_per_hr * elapsed_mins / 60:.2f} (@ ${cost_per_hr:.2f}/hr)"
        if cost_per_hr
        else "unknown"
    )
    lines = [
        "## Reasoning SFT pipeline",
        "",
        f"- **Status**: {'success' if success else 'FAILED'}",
        f"- **Adapter**: [{adapter_repo}](https://huggingface.co/{adapter_repo})",
        f"- **Pod time**: {elapsed_mins:.0f} min",
        f"- **Estimated GPU cost**: {est_cost}",
        "",
    ]
    if results:
        lines += ["| Input title | Parody | Phonetic score |", "|---|---|---|"]
        for r in results:
            if "error" not in r:
                lines.append(
                    f"| {r['input_title']} | {r.get('parody', '')} "
                    f"| {r.get('avg_phonetic_score', '')} |"
                )
        lines.append("")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    import runpod
    from huggingface_hub import HfApi

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not runpod_key:
        raise SystemExit("RUNPOD_API_KEY env var is required")
    runpod.api_key = runpod_key

    if args.down:
        runpod.terminate_pod(args.down)
        print(f"Terminated pod {args.down}")
        return

    if not hf_token:
        raise SystemExit("HF_TOKEN env var is required")

    if args.test:
        args.adapter_repo = args.adapter_repo.rstrip("/") + "-test"
    timeout_mins = args.timeout_mins or (60 if args.test else 240)
    disk_gb = args.disk_gb or (40 if args.test else 100)

    api = HfApi(token=hf_token)
    start = dt.datetime.now(dt.timezone.utc)

    docker_cmd = (
        f"bash -c 'git clone --depth 1 --branch {args.git_ref} {REPO_URL} "
        f"/tmp/boot && bash /tmp/boot/pipeline/pod_bootstrap.sh'"
    )
    pod_name = f"chuckles-reasoning-{'test' if args.test else 'sft'}"
    pod_env = {
        "HF_TOKEN": hf_token,
        "REPO_URL": REPO_URL,
        "GIT_REF": args.git_ref,
        "BASE_MODEL": args.base_model,
        "DATASET_REPO": args.dataset_repo,
        "ADAPTER_REPO": args.adapter_repo,
        "EPOCHS": str(args.epochs),
        "TEST_MODE": "true" if args.test else "false",
        "RUN_INFERENCE": "false" if args.no_inference else "true",
        "INFER_TITLES": args.infer_titles,
        **({"INFER_LIMIT": str(args.infer_limit)} if args.infer_limit else {}),
    }
    pod_kwargs = dict(
        name=pod_name,
        image_name=args.image,
        gpu_count=args.gpu_count,
        cloud_type=args.cloud,
        container_disk_in_gb=disk_gb,
        support_public_ip=True,
        ports="22/tcp",
        docker_args=docker_cmd,
        env=pod_env,
    )
    if args.volume_gb > 0:
        pod_kwargs["volume_in_gb"] = args.volume_gb
        pod_kwargs["volume_mount_path"] = "/workspace"

    pod = None
    gpu_types = [g.strip() for g in args.gpu.split(",") if g.strip()]
    for gpu_type in gpu_types:
        print(f"Creating pod ({gpu_type}, {args.cloud}, disk {disk_gb}GB, "
              f"test={args.test}) ...")
        try:
            pod = runpod.create_pod(gpu_type_id=gpu_type, **pod_kwargs)
            break
        except Exception as e:
            print(f"  could not schedule on {gpu_type}: {e}")
    if pod is None:
        raise SystemExit(
            f"No pod could be scheduled on any of: {', '.join(gpu_types)}. "
            "Try again later, a different --gpu list, or --cloud SECURE."
        )
    pod_id = pod["id"]
    cost_per_hr = pod.get("costPerHr")
    if cost_per_hr:
        print(f"Pod {pod_id} created at ${float(cost_per_hr):.2f}/hr.")
    else:
        print(f"Pod {pod_id} created.")
    print(f"Polling for adapter at {args.adapter_repo} (timeout {timeout_mins} min) ...")

    success = False
    pod_gone = False
    deadline = time.time() + timeout_mins * 60
    try:
        while time.time() < deadline:
            time.sleep(args.poll_secs)

            if hf_adapter_landed(api, args.adapter_repo, start):
                success = True
                print(f"Adapter landed: https://huggingface.co/{args.adapter_repo}")
                break

            try:
                info = runpod.get_pod(pod_id) or {}
            except Exception as e:
                print(f"  pod status check failed: {e}")
                info = {}
            if info.get("costPerHr"):
                cost_per_hr = info["costPerHr"]
            status = info.get("desiredStatus", "UNKNOWN")
            elapsed = int((time.time() - (deadline - timeout_mins * 60)) / 60)
            print(f"  [{elapsed:3d}m] pod={status}, adapter not ready yet")
            if status in ("EXITED", "TERMINATED", "STOPPED"):
                # Container finished; give the final Hub push a grace period.
                pod_gone = status == "TERMINATED"
                time.sleep(30)
                success = hf_adapter_landed(api, args.adapter_repo, start)
                break
        else:
            print(f"TIMEOUT after {timeout_mins} minutes")
    finally:
        if args.keep_pod:
            print(f"--keep-pod set: pod {pod_id} left running. "
                  f"Tear down with: python pipeline/run_pipeline.py --down {pod_id}")
        elif not pod_gone:
            print(f"Terminating pod {pod_id} ...")
            try:
                runpod.terminate_pod(pod_id)
                print("Pod terminated.")
            except Exception as e:
                print(f"ERROR terminating pod {pod_id}: {e}\n"
                      f"TERMINATE IT MANUALLY in the RunPod dashboard to stop billing!")

    elapsed_mins = (dt.datetime.now(dt.timezone.utc) - start).total_seconds() / 60
    try:
        cost_per_hr = float(cost_per_hr) if cost_per_hr else None
    except (TypeError, ValueError):
        cost_per_hr = None
    if cost_per_hr:
        print(f"Pod ran ~{elapsed_mins:.0f} min at ${cost_per_hr:.2f}/hr "
              f"-> estimated GPU cost ${cost_per_hr * elapsed_mins / 60:.2f}")
    else:
        print(f"Pod ran ~{elapsed_mins:.0f} min (cost/hr unavailable from the API)")

    log = fetch_dataset_file(args.dataset_repo, "logs/pod-run-latest.log")
    if log:
        print("\n===== pod run log (tail) =====")
        print("\n".join(log.splitlines()[-60:]))
        print("===== end pod log =====\n")

    results: list[dict] = []
    if not args.no_inference:
        raw = fetch_dataset_file(args.dataset_repo, "results/inference-latest.jsonl")
        if raw:
            results = print_inference_results(raw)
        else:
            print("(no inference results found on the Hub)")

    write_github_summary(success, args.adapter_repo, elapsed_mins, cost_per_hr, results)

    if success:
        print(f"SUCCESS: adapter at https://huggingface.co/{args.adapter_repo}")
    else:
        print("FAILED: adapter never appeared on the Hub. See pod log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

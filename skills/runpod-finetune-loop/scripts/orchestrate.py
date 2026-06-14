#!/usr/bin/env python
"""
orchestrate.py — FALLBACK pod lifecycle via the runpod Python SDK.

Prefer driving pod create/terminate through the RunPod MCP (so Hermes keeps it in
the tool loop). Use this only if the MCP is unavailable. It spins up a GPU pod,
waits until it's RUNNING, prints SSH details, and (optionally) terminates it.

    pip install runpod
    export RUNPOD_API_KEY=...

    # create + wait + print connection info
    python orchestrate.py up --gpu "NVIDIA A100 80GB PCIe" --name ft-run

    # terminate when done (the step people forget)
    python orchestrate.py down --pod-id <id>
"""
import argparse
import os
import time

import runpod


def cmd_up(args):
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        gpu_count=args.gpu_count,
        volume_in_gb=args.volume_gb,
        volume_mount_path="/workspace",
        container_disk_in_gb=args.disk_gb,
        cloud_type=args.cloud,          # "COMMUNITY" (cheap) or "SECURE"
        support_public_ip=True,
        ports="22/tcp",
        env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    pod_id = pod["id"]
    print(f"created pod {pod_id}; waiting for RUNNING...")
    for _ in range(60):
        info = runpod.get_pod(pod_id)
        status = info.get("desiredStatus") or info.get("lastStatusChange")
        runtime = info.get("runtime")
        if runtime:  # runtime is populated once the container is live
            print(f"pod {pod_id} is up.")
            print("SSH / ports:", runtime.get("ports"))
            print(f"\nNext: stage train_lora.py + data, run training, push to HF,")
            print(f"then: python orchestrate.py down --pod-id {pod_id}")
            return
        time.sleep(10)
    print("timed out waiting for pod; check the RunPod dashboard.")


def cmd_down(args):
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    runpod.terminate_pod(args.pod_id)
    print(f"terminated pod {args.pod_id}. Confirm it's gone in the dashboard so billing stops.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up")
    up.add_argument("--name", default="ft-run")
    up.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                    help="Confirm the current PyTorch template tag from RunPod")
    up.add_argument("--gpu", default="NVIDIA A100 80GB PCIe", help="GPU type id (see RunPod GPU list)")
    up.add_argument("--gpu-count", type=int, default=1)
    up.add_argument("--volume-gb", type=int, default=100)
    up.add_argument("--disk-gb", type=int, default=50)
    up.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down")
    down.add_argument("--pod-id", required=True)
    down.set_defaults(func=cmd_down)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

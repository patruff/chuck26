---
name: runpod-finetune-loop
description: >-
  End-to-end loop for fine-tuning an LLM on a rented RunPod GPU and verifying it.
  Provisions a GPU pod, trains a LoRA/QLoRA adapter, pushes the adapter to the
  Hugging Face Hub, tears the pod down to stop billing, then pulls the adapter
  back and runs inference eval prompts to confirm training worked. Use this
  whenever the user wants to fine-tune, train, or LoRA a model on RunPod; rent a
  GPU for a training run; push or pull a fine-tuned adapter to/from Hugging Face;
  or test that a fine-tune "took" before shipping it. Trigger on mentions of
  RunPod, GPU pod, LoRA/QLoRA fine-tune, adapter, push_to_hub, or "test the
  fine-tuned model" — even if the user doesn't name every step.
compatibility: >-
  Hermes Agent (agentskills.io format). Needs the RunPod MCP (pod lifecycle) and,
  recommended, the Hugging Face MCP (verify the pushed adapter). Hermes native
  terminal/SSH runs the training scripts on the pod. Env: RUNPOD_API_KEY, HF_TOKEN.
---

# RunPod Fine-Tune Loop

Fine-tune a model on a rented GPU, ship the adapter to Hugging Face, then prove it
works — without leaving a pod billing while idle. A LoRA adapter is tiny (tens of
MB), so the expensive GPU is only needed for *training*; verification can happen on
a cheap/local GPU after the pod is gone.

## The loop (always run in this order)

1. **Provision** a GPU pod (RunPod MCP).
2. **Stage** the training script + dataset onto the pod (Hermes terminal/SSH or `runpodctl send`).
3. **Train** the adapter on the pod (`scripts/train_lora.py`) and push it to HF Hub.
4. **Verify the push landed** (HF MCP, or `huggingface-cli`).
5. **Terminate the pod immediately** (RunPod MCP) — this is the step people forget; an idle A100 bills ~$0.89–1.89/hr for nothing.
6. **Pull + test** the adapter anywhere (`scripts/test_inference.py`), run eval prompts, report pass/fail.
7. **Record** the run (HF repo id, base model, eval score) to memory so the next run can compare.

Steps 1, 5 use the RunPod MCP. Step 4 uses the HF MCP. Steps 2, 3, 6 are scripts Hermes runs over the terminal. Do not skip step 5 — confirm the pod is `TERMINATED` before moving on.

## Step 1 — Provision the pod

Use the RunPod MCP to create a pod. Sensible defaults for a 7–9B QLoRA run:

- **GPU**: one A100 80GB (fits 7–13B QLoRA comfortably) or RTX 4090 24GB (cheaper, for ≤8B QLoRA).
- **Cloud**: Community Cloud for cost; it can be interrupted, so checkpoint to a network volume.
- **Image**: a PyTorch/CUDA template (e.g. `runpod/pytorch`) — confirm the current tag from RunPod.
- **Volume**: attach a network volume mounted at `/workspace` for checkpoints, so an interruption doesn't lose progress.
- **Env**: pass `HF_TOKEN` so the pod can `push_to_hub`.

Ask the RunPod MCP for the pod's SSH connection details once it reports `RUNNING`. If the MCP path is unavailable, `scripts/orchestrate.py` does the same lifecycle via the `runpod` Python SDK as a fallback.

## Step 2 — Stage code + data

Copy `scripts/train_lora.py` and the dataset onto the pod. Options, in order of preference:

- Hermes terminal runs `scp` / `rsync` to the pod's SSH endpoint, or
- `runpodctl send <file>` from local + `runpodctl receive` on the pod, or
- clone from GitHub if the training code already lives in a repo.

Datasets should be JSONL with a `text` field (or `messages` for chat format). Keep the file on the network volume so it survives interruptions.

## Step 3 — Train

On the pod, install deps and run the trainer. The script does QLoRA SFT and pushes
**only the adapter** (not the merged base) to keep the upload tiny:

```bash
pip install -q "transformers>=4.44" peft trl datasets accelerate bitsandbytes
python train_lora.py \
  --base-model meta-llama/Llama-3.1-8B \
  --dataset /workspace/data/train.jsonl \
  --hf-repo <user>/<model>-adapter \
  --output /workspace/out \
  --epochs 3
```

See `scripts/train_lora.py` for all flags (rank, alpha, target modules, 4-bit
config, max steps). The script calls `trainer.push_to_hub(...)`, which uploads the
adapter weights + `adapter_config.json` and records the base model id in that
config — that's what lets `from_pretrained` reassemble the model later.

## Step 4 — Verify the push

Before killing the pod, confirm the adapter is actually on the Hub. Use the HF MCP
to look up the repo and check it contains `adapter_config.json` and an
`adapter_model.safetensors`. If the HF MCP isn't connected, run on the pod:

```bash
huggingface-cli repo files <user>/<model>-adapter
```

Only proceed once the adapter files are confirmed present.

## Step 5 — Terminate the pod

Call the RunPod MCP to **terminate** (not just stop) the pod and confirm the state
is `TERMINATED`. Stopping a pod still bills for the volume; terminating releases the
GPU. If you need the network volume retained for the next run, detach it but still
terminate the pod itself.

## Step 6 — Pull + test inference

The adapter is small, so verification runs anywhere — a cheap pod, a local GPU, or
even CPU for tiny bases. Pull the adapter from HF and run the eval prompts:

```bash
pip install -q transformers peft accelerate bitsandbytes
python test_inference.py \
  --hf-repo <user>/<model>-adapter \
  --evals scripts/eval_prompts.example.json
```

`scripts/test_inference.py` uses `AutoPeftModelForCausalLM.from_pretrained(<repo>)`,
which reads the base model id out of `adapter_config.json`, loads the base, attaches
the adapter, and generates. It runs each prompt, prints the completion, and applies
simple `contains` / `regex` checks from the eval file to give a pass rate. A healthy
fine-tune should clearly beat the base model on your task prompts; if it doesn't,
the training likely under-fit (bump epochs/rank) or the dataset was mis-formatted.

## Step 7 — Record the run

Save to memory: HF repo id, base model, GPU used, wall-clock + approx cost, and the
eval pass rate. On the next run, compare against this so the agent can tell whether a
change helped. If a GitHub MCP is connected, optionally open/append an issue with the
result for the team.

## Cost & safety notes

- Community Cloud is cheapest but interruptible — always checkpoint to the volume.
- Terminate, don't stop, when done (step 5). The single biggest waste is a forgotten idle pod.
- QLoRA inference must stay consistent with training: if you trained on a 4-bit base, either load 4-bit at inference or merge the adapter into a full-precision base deliberately — don't silently switch.
- To eliminate inference latency in production, `merge_and_unload()` folds the adapter into the base and you serve a single merged model (optionally via a RunPod Serverless endpoint).

## Hermes setup

Installing this skill into Hermes and wiring up the RunPod + Hugging Face MCPs is
covered in `references/hermes-setup.md`. Read it when setting up the agent or when
an MCP tool call fails with an auth error.

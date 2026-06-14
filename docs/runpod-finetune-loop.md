# runpod-finetune-loop (Hermes skill)

A Hermes Agent skill that runs an end-to-end LLM fine-tune loop on a rented RunPod
GPU and verifies it: **provision a pod → train a LoRA adapter → push it to the
Hugging Face Hub → terminate the pod → pull the adapter back → run eval prompts.**
A LoRA adapter is tiny, so the expensive GPU only exists during training;
verification happens after the pod is gone.

Skill lives in [`skills/runpod-finetune-loop/`](skills/runpod-finetune-loop/).

## Secrets you need

| Secret | Required | What for | Where to get it |
|---|---|---|---|
| `RUNPOD_API_KEY` | ✅ | Create + **terminate** pods | RunPod → Settings → API Keys (read/write) |
| `HF_TOKEN` | ✅ | Push the adapter; pull the base model. Also injected into the pod env | huggingface.co/settings/tokens (**write** scope) |
| SSH keypair | ✅ | Hermes scp/ssh into the pod to run training | `ssh-keygen`, add the **public** key to RunPod → Settings → SSH Public Keys |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | ⬜ optional | Log eval results via GitHub MCP / token push | GitHub fine-grained PAT scoped to this repo |
| `WANDB_API_KEY` | ⬜ optional | Experiment tracking (off by default) | wandb.ai |

Put the env secrets in `~/.hermes/.env`; the SSH **private** key stays on the
machine running Hermes and is never uploaded. Only the **public** key goes to
RunPod. Copy `.env.example` → `.env` (gitignored) to fill them in locally.

## Install into Hermes

```bash
cp -r skills/runpod-finetune-loop ~/.hermes/skills/
```

Then in a Hermes session: `/reload-skills`, confirm with `/skills`, load with
`/skill runpod-finetune-loop`. Full MCP wiring (RunPod + Hugging Face, with the
`config.yaml` block) is in
[`skills/runpod-finetune-loop/references/hermes-setup.md`](skills/runpod-finetune-loop/references/hermes-setup.md).

## First run — plumbing test

The goal of the first run is **to prove the pipes connect**, not to train a good
model. So we pick an ungated 7B base, a cheap 24 GB GPU, and cap training at a
handful of steps.

## Local smoke test — no pod, no push, no GPU spend

Before renting a GPU, run the same train/infer/evaluate shape against a tiny
model and toy dataset. This only proves that dataset loading, LoRA training,
adapter loading, and assertions are wired correctly. It does **not** prove model
quality.

```bash
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.44" peft trl datasets accelerate

mkdir -p data out/local-smoke
cat > data/local_smoke_train.jsonl <<'JSONL'
{"text":"### Instruction:\nSay the chuckles smoke keyword.\n\n### Response:\nCHUCKLES_SMOKE_OK\n"}
{"text":"### Instruction:\nReturn the training status as JSON.\n\n### Response:\n{\"status\":\"adapter_loaded\",\"suite\":\"chuckles_smoke\"}\n"}
JSONL

cat > data/local_smoke_evals.json <<'JSON'
[
  {
    "prompt": "### Instruction:\nSay the chuckles smoke keyword.\n\n### Response:\n",
    "expect_nonempty": true,
    "min_chars": 2
  },
  {
    "prompt": "### Instruction:\nReturn the training status as JSON.\n\n### Response:\n",
    "expect_nonempty": true,
    "min_chars": 2
  }
]
JSON

python skills/runpod-finetune-loop/scripts/train_lora.py \
  --base-model sshleifer/tiny-gpt2 \
  --dataset data/local_smoke_train.jsonl \
  --hf-repo local/smoke-adapter \
  --output out/local-smoke \
  --target-modules c_attn,c_proj \
  --max-steps 100 \
  --lr 0.01 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-len 128 \
  --lora-r 4 \
  --lora-alpha 8 \
  --no-4bit \
  --fp32 \
  --no-push \
  --no-packing

python skills/runpod-finetune-loop/scripts/test_inference.py \
  --hf-repo out/local-smoke \
  --evals data/local_smoke_evals.json \
  --max-new-tokens 16 \
  --fp32

python skills/runpod-finetune-loop/scripts/eval_adapter_loss.py \
  --base-model sshleifer/tiny-gpt2 \
  --adapter out/local-smoke \
  --dataset data/local_smoke_train.jsonl \
  --max-length 128 \
  --fp32 \
  --min-improvement 0.001
```

Expected result: training writes adapter files into `out/local-smoke`, inference
loads that adapter back from disk, and the eval script reports checked prompts
passing. The loss eval should show `adapter_loss` lower than `base_loss` by at
least `0.001`. If generated text is nonsense, that is okay for this tiny model;
this test is for plumbing and a basic learning signal, not product-quality
generation.

| Choice | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B` | Apache-2.0, **ungated** — no license-acceptance friction for `HF_TOKEN` |
| GPU | 1× RTX 4090 24 GB, Community Cloud | ~$0.34/hr; plenty for 7B QLoRA; the "small-ish GPU" |
| Method | QLoRA, 4-bit nf4, r=8 | Fits a 7B comfortably in 24 GB |
| Steps | `--max-steps 50` | We're testing plumbing, not convergence |
| Dataset | ~100 lines JSONL | Tiny on purpose |
| Target repo | `<hf-user>/qwen2.5-7b-plumbing-test-adapter` (private) | Throwaway |

Expected cost for the whole loop: well under **$1** (≈10–20 min on the 4090,
including setup), because the pod is terminated right after the push.

### What Hermes does, step by step

1. **Provision** — RunPod MCP `create-pod`: RTX 4090, Community Cloud, PyTorch image, 100 GB volume at `/workspace`, `HF_TOKEN` in env.
2. **Stage** — scp `train_lora.py` + the tiny dataset to the pod.
3. **Train** on the pod:
   ```bash
   pip install -q "transformers>=4.44" peft trl datasets accelerate bitsandbytes
   python train_lora.py \
     --base-model Qwen/Qwen2.5-7B \
     --dataset /workspace/data/train.jsonl \
     --hf-repo <hf-user>/qwen2.5-7b-plumbing-test-adapter \
     --output /workspace/out \
     --lora-r 8 --lora-alpha 16 \
     --max-steps 50 --private
   ```
4. **Verify the push** — HF MCP (or `huggingface-cli repo files <repo>`): confirm `adapter_config.json` + `adapter_model.safetensors` exist.
5. **Terminate** — RunPod MCP `delete-pod`; confirm the pod no longer appears in `list-pods` so billing stops.
6. **Pull + test** anywhere (cheap pod / local GPU):
   ```bash
   python test_inference.py \
     --hf-repo <hf-user>/qwen2.5-7b-plumbing-test-adapter \
     --evals scripts/eval_prompts.example.json --load-4bit
   ```
7. **Record** — save HF repo id, GPU, wall-clock, cost, eval pass rate to memory.

Success criterion for the plumbing test: the loop completes, the adapter appears on
the Hub, the pod terminates, and `test_inference.py` loads the adapter and generates
without error. Output quality is irrelevant at 50 steps — that's expected.

## Notes / gotchas

- **Terminate, don't stop.** A forgotten idle pod is the only way this gets expensive.
- **Ungated base on purpose.** If you later switch to Llama, the `HF_TOKEN` account must accept that model's license first, or the pod's base-model download 401s.
- **Keep QLoRA consistent.** Trained 4-bit ⇒ test with `--load-4bit` (the script flag), or deliberately merge to a full-precision base. Don't switch silently.
- The RunPod MCP package string in `hermes-setup.md` was verified against RunPod's current MCP docs: `npx -y @runpod/mcp-server@latest`.

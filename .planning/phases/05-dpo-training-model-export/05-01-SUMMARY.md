---
phase: 05-dpo-training-model-export
plan: 01
subsystem: training
tags: [dpo, qwen3, unsloth, trl, runpod, qlora, lora]

dependency-graph:
  requires: []
  provides:
    - "training/setup_runpod.sh -- one-shot RunPod environment setup"
    - "training/train_dpo.py -- standalone DPO training script for Qwen3-32B"
  affects:
    - "05-02: merge_and_push.py and validate_merge.py depend on adapter output from train_dpo.py"
    - "06-*: vLLM inference serving loads the adapter/merged model produced here"

tech-stack:
  added:
    - "unsloth (QLoRA model loading, LoRA attachment, merged saving)"
    - "trl (DPOTrainer, DPOConfig)"
    - "bitsandbytes (4-bit NF4 quantization backend)"
    - "peft (LoRA adapter management)"
  patterns:
    - "FastModel API for all Unsloth model loading (not legacy FastLanguageModel)"
    - "ref_model=None with PEFT for zero-VRAM reference model in DPO"
    - "processing_class=tokenizer (current TRL API)"
    - "enable_thinking=False for Qwen3 non-thinking mode"
    - "All outputs to /workspace/ (RunPod network volume persistence)"

key-files:
  created:
    - "training/setup_runpod.sh"
    - "training/train_dpo.py"
  modified: []

decisions:
  - id: "train-install-order"
    decision: "Install Unsloth before TRL to control dependency versions"
    rationale: "Unsloth monkey-patches TRL/Transformers internals; version mismatch causes crashes"
  - id: "train-save-lora-first"
    decision: "Save LoRA adapter to Hub immediately after training, before merge"
    rationale: "Merge step has known bugs (Unsloth #3146); adapter backup prevents total loss"
  - id: "train-no-thinking"
    decision: "Enforce enable_thinking=False in chat template verification"
    rationale: "DPO dataset has no <think> blocks; thinking mode creates train/inference mismatch"

metrics:
  tasks-completed: 2/2
  duration: "~3 minutes"
  completed: "2026-01-31"
---

# Phase 5 Plan 1: RunPod Setup & DPO Training Script Summary

**One-liner:** Standalone RunPod setup script and DPO training pipeline for Qwen3-32B using Unsloth QLoRA + TRL DPOTrainer with chat template verification

## What Was Built

Two standalone scripts that can be copied to a RunPod A6000 pod and executed without modification (except setting `HF_TOKEN`):

1. **`training/setup_runpod.sh`** -- One-shot environment setup that installs Unsloth (first, for version control), then TRL, optionally wandb, logs into HuggingFace Hub, and verifies all package versions including CUDA/GPU detection.

2. **`training/train_dpo.py`** -- Complete DPO training pipeline that loads Qwen3-32B in 4-bit QLoRA via `FastModel`, attaches LoRA adapters (rank 32, all linear layers), loads the `patruff/chuckles-dpo` dataset from Hub, verifies the Qwen3 chat template produces no `<think>` blocks, trains with `DPOTrainer` using `ref_model=None` (zero extra VRAM), saves checkpoints to `/workspace/dpo-output/`, and pushes the LoRA adapter to `patruff/chuckles-qwen3-32b-dpo-adapter` on Hub.

## Task Execution

### Task 1: Create RunPod setup script and training directory

| Attribute | Value |
|-----------|-------|
| Commit | `5fd546b` |
| Files | `training/setup_runpod.sh` |
| Status | Complete |

Created `training/` directory and `setup_runpod.sh` with:
- `set -e` fail-fast behavior
- HF_TOKEN validation with helpful error message
- Unsloth installed before TRL (critical for version resolution)
- Optional wandb installation with graceful failure
- HuggingFace CLI login
- Python-based version verification for 6 packages + CUDA/GPU info
- Next-step instructions

### Task 2: Create standalone DPO training script

| Attribute | Value |
|-----------|-------|
| Commit | `b8955b0` |
| Files | `training/train_dpo.py` |
| Status | Complete |

Created 314-line DPO training script with:
- Configuration section with all constants (model, dataset, output dir, Hub repo)
- `FastModel.from_pretrained()` for 4-bit model loading
- `FastModel.get_peft_model()` with LoRA on all 7 linear layer types
- Dataset loading from Hub with row count, column, and sample verification
- Chat template verification step (`enable_thinking=False`, assert no `<think>`)
- `DPOConfig` with all specified hyperparameters (3 epochs, lr=5e-6, beta=0.1, etc.)
- `DPOTrainer` with `ref_model=None` and `processing_class=tokenizer`
- Post-training adapter save to `/workspace/` and push to Hub
- Comprehensive docstring with hyperparameter tuning guidance

## Verification Results

All 8 plan verification checks passed:

1. `bash -n training/setup_runpod.sh` -- valid shell syntax
2. `python -c "import ast; ast.parse(...)"` -- valid Python syntax
3. Uses `FastModel` (no `FastLanguageModel` references)
4. Uses `processing_class=tokenizer` (not deprecated `tokenizer=`)
5. `ref_model=None` for PEFT-aware reference model
6. All output paths reference `/workspace/`
7. `enable_thinking=False` in template verification section
8. `HF_TOKEN` loaded from `os.environ`, never hardcoded

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Install Unsloth before TRL | Unsloth monkey-patches TRL internals; controls dependency versions |
| Save LoRA adapter to Hub before merge step | Merge has known bugs (#3146); adapter backup is a safety net |
| Verify chat template at training start | Catches Qwen3 thinking-mode mismatch before wasting GPU hours |
| Use `paged_adamw_8bit` optimizer | Prevents optimizer state OOM spikes on A6000 with 32B model |

## Requirements Coverage

| Requirement | Status | How |
|-------------|--------|-----|
| DPO-01 (Load Qwen3-32B) | Covered | `FastModel.from_pretrained()` with pre-quantized 4-bit |
| DPO-02 (QLoRA config) | Covered | LoRA rank 32, all 7 linear layers, gradient checkpointing |
| DPO-03 (Load DPO dataset) | Covered | `load_dataset("patruff/chuckles-dpo")` from Hub |
| DPO-04 (DPO training) | Covered | `DPOTrainer` with `DPOConfig`, ref_model=None |
| DPO-05 (Save checkpoints) | Covered | `/workspace/dpo-output/` with save_steps=100 |
| EXP-01 (Push adapter to Hub) | Covered | `push_to_hub_merged` to `patruff/chuckles-qwen3-32b-dpo-adapter` |

## Next Phase Readiness

Plan 05-02 (merge_and_push.py + validate_merge.py) can proceed. It depends on the adapter output at `/workspace/dpo-output/final-adapter/` which this script produces.

No blockers identified. The scripts are ready to be copied to RunPod once a pod with A6000 and 200GB+ network volume is provisioned.

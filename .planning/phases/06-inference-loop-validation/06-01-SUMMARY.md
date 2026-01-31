---
phase: 06-inference-loop-validation
plan: 01
subsystem: inference-serving
tags: [vllm, awq, quantization, inference, runpod]
dependency-graph:
  requires: [05-01, 05-02]
  provides: [vllm-inference-scripts, awq-quantization-pipeline]
  affects: [06-02, 07-01]
tech-stack:
  added: [vllm, autoawq]
  patterns: [openai-compatible-api, runtime-quantization, pre-quantization]
file-tracking:
  key-files:
    created: [training/setup_inference.sh, training/quantize_awq.py]
    modified: []
decisions:
  - id: "06-01-01"
    decision: "BitsAndBytes as default mode (simpler, no pre-quantization needed)"
    reason: "Lower barrier to entry -- user can serve immediately after merge"
  - id: "06-01-02"
    decision: "AutoAWQ over LLM Compressor for Qwen3-32B quantization"
    reason: "LLM Compressor has known quality issues with Qwen3-32B W4A16 (vllm-project/llm-compressor#1600)"
  - id: "06-01-03"
    decision: "max-model-len 8192 (not default 40960)"
    reason: "Default causes OOM on all consumer GPUs; parodies need < 8K tokens"
metrics:
  duration: "~2 minutes"
  completed: "2026-02-01"
---

# Phase 6 Plan 1: vLLM Inference Setup Summary

**One-liner:** vLLM serving scripts with BitsAndBytes/AWQ quantization modes, Qwen3 thinking-mode disabled, and OpenAI-compatible API on port 8000.

## What Was Done

### Task 1: Create vLLM inference launch script
**Commit:** `7791c81`
**File:** `training/setup_inference.sh`

Created a standalone bash script that launches vLLM on a RunPod GPU pod with two quantization modes:

- **--bnb (default):** BitsAndBytes NF4 runtime quantization. Simpler, no pre-quantization step needed. ~168 tok/s.
- **--awq:** Serves pre-quantized AWQ model via Marlin kernel. ~712 tok/s. Requires running quantize_awq.py first.

All six critical vLLM flags included:
1. `--served-model-name` -- matches settings.json model_name (prevents 404 errors)
2. `--max-model-len 8192` -- prevents OOM (default 40960 is too large for consumer GPUs)
3. `--default-chat-template-kwargs '{"enable_thinking": false}'` -- disables Qwen3 thinking mode (prevents `<think>` blocks that break smolagents parser)
4. `--gpu-memory-utilization 0.90` -- uses 90% of VRAM for model + KV cache
5. `--host 0.0.0.0` -- binds all interfaces (required for RunPod port forwarding)
6. `--dtype half` -- explicit FP16 (matches merged model dtype)

Pre-flight checks for GPU availability, system RAM (BitsAndBytes needs 65GB+), and HF_TOKEN. Post-launch instructions for settings.json configuration.

### Task 2: Create AWQ quantization script
**Commit:** `7ef87d3`
**File:** `training/quantize_awq.py`

Created a standalone Python script that quantizes the FP16 merged model to AWQ 4-bit:

- Loads FP16 model from Hub (`patruff/chuckles-qwen3-32b-dpo`)
- Quantizes with AutoAWQ (w_bit=4, GEMM version, group_size=128)
- Validates quantized output by loading back and generating a test response
- Saves locally to `/workspace/chuckles-qwen3-32b-dpo-awq/`
- Pushes to Hub as `patruff/chuckles-qwen3-32b-dpo-awq` via `HfApi.upload_folder`

Error handling includes fallback instructions if Hub push fails, and validation warnings that don't block the push.

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 06-01-01 | BitsAndBytes as default mode | Lower barrier -- user can serve immediately after merge, no pre-quantization step |
| 06-01-02 | AutoAWQ over LLM Compressor | LLM Compressor has known Qwen3-32B quality issues (llm-compressor#1600) |
| 06-01-03 | max-model-len 8192 | Default 40960 causes OOM; parodies need < 8K tokens |

## Verification Results

| Check | Result |
|-------|--------|
| setup_inference.sh bash syntax | PASS |
| quantize_awq.py Python syntax | PASS |
| served-model-name flag present | PASS (2 occurrences) |
| enable_thinking flag present | PASS |
| max-model-len 8192 present | PASS |
| AutoAWQForCausalLM present | PASS (3 occurrences) |
| upload_folder present | PASS |

## Next Phase Readiness

**For 06-02 (Inference Loop Validation):**
- setup_inference.sh is ready to launch on RunPod
- Once vLLM is running, the existing CLI can connect by updating settings.json api_base_url
- AWQ path available for faster inference if BitsAndBytes throughput is insufficient

**Blockers:** None. Scripts are standalone and copy-pasteable to RunPod.

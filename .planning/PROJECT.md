# chucklesPRIME

## What This Is

A Python pipeline that generates phonetically sound parody titles ("chuckles"), captures reasoning traces, converts them into RLVR datasets, and fine-tunes language models to produce better parodies. The system forms a complete improvement loop: generate data → train model → run inference → generate better data.

## Core Value

Generate quality reasoning data about what makes a good phonetic parody, in formats ready for GRPO and DPO fine-tuning — then close the loop by actually training and deploying the fine-tuned model.

## Current Milestone: v1.1 Fine-tuning & Inference Loop

**Goal:** Train Qwen3-32B on the generated datasets using RunPod, push the fine-tuned model to HuggingFace Hub, and run inference through the existing CLI — completing the full improvement loop.

**Target features:**
- DPO training script for RunPod (Unsloth + 4-bit QLoRA on Qwen3-32B)
- GRPO training script for RunPod (TRL GRPOTrainer with composite reward functions)
- Model merging and Hub push (LoRA adapters → merged model → HF Hub)
- Inference via vLLM/Ollama on cheaper GPU
- Existing `chuckles generate` CLI works with fine-tuned model endpoint
- Full loop: generate → train → inference → generate better data

## Requirements

### Validated

- ✓ **CFG-01–04**: External config loading (funny words, preferences, human examples, settings indirection) — v1.0
- ✓ **LLM-01–02**: Custom OpenAI-compatible model adapter with JSON backend config — v1.0
- ✓ **GEN-01–04**: CSV input, 2 candidates per title, full reasoning traces, HF Hub tools — v1.0
- ✓ **DATA-01–05**: GRPO/DPO datasets, Hub push, JSONL traces, composite rewards — v1.0
- ✓ **PROJ-01–02**: Clean package structure, CLI with generate/convert subcommands — v1.0

### Active

- [ ] DPO training on Qwen3-32B with 4-bit QLoRA via Unsloth on RunPod
- [ ] GRPO training with composite reward functions from v1 datasets
- [ ] LoRA adapter merging and model push to HuggingFace Hub
- [ ] Inference serving via vLLM on RunPod (cheaper GPU tier)
- [ ] Existing CLI (`chuckles generate`) compatible with vLLM/Ollama endpoints
- [ ] Full improvement loop: generate → train → inference → generate better data

### Out of Scope

- Fine-tuning loop automation — manual iteration for v1.1, automate later
- Web UI or API server — CLI/script tool only
- Google Drive integration — local files + HF Hub
- Real-time/interactive generation — batch processing via CSV
- Multi-model comparison — single model (Qwen3-32B) done well first
- Full-precision training — 4-bit QLoRA only for v1.1
- Custom CUDA kernels — rely on Unsloth optimizations

## Context

- v1.0 pipeline is complete: generates parody data, converts to GRPO/DPO datasets, pushes to Hub
- Existing DPO dataset has human parodies as "chosen" and worst model outputs as "rejected"
- Existing GRPO dataset has prompt-only format with phonetic_scores, tool_usage, structure_preservation metadata
- Three composite reward signals are continuous floats in [0.0, 1.0] — ready for GRPO training
- Target model: Qwen3-32B (latest generation Qwen)
- Training hardware: RunPod A6000 (48GB) or A100 (80GB) for training
- Inference hardware: RunPod RTX 3090/4090 for serving
- Training approach: Unsloth for efficient 4-bit QLoRA, TRL for DPO/GRPO trainers
- Inference approach: vLLM or Ollama as OpenAI-compatible server
- The existing `OpenAICompatibleModel` adapter should work with vLLM endpoints out of the box

## Constraints

- **Language**: Pure Python — no JS, no compiled extensions
- **Training framework**: Unsloth + TRL (HuggingFace ecosystem)
- **Training hardware**: RunPod GPU pods (A6000/A100 for training, 3090/4090 for inference)
- **Model**: Qwen3-32B with 4-bit quantization
- **Scripts**: Standalone training scripts (not integrated into chucklesPRIME CLI) — easy to copy to RunPod
- **Inference**: vLLM or Ollama serving as OpenAI-compatible endpoint
- **Existing tools**: Keep all v1.0 components as-is

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| JSON files for config (not env vars) | Funny words and preferences are structured data, not simple strings | ✓ Good |
| Keep phonetic tools as-is | They work, phonetic scoring is solid | ✓ Good |
| Custom LLM adapter (not LiteLLM) | Keep control over adapter layer, support any OpenAI-compatible API | ✓ Good |
| Dual dataset output (GRPO + DPO) | GRPO for verifiable reward training, DPO for preference learning from human examples | ✓ Good |
| User preferences opaque to app | App injects the preference text into prompts without parsing it | ✓ Good |
| Batch-only (CSV in/out) | Simplifies architecture, matches the data generation workflow | ✓ Good |
| Standalone training scripts | Easy to copy-paste to RunPod, no package install needed on training pod | — Pending |
| Unsloth for QLoRA | Efficient 4-bit training, 2x faster than standard PEFT | — Pending |
| vLLM for inference serving | High throughput, OpenAI-compatible API, works with existing adapter | — Pending |

---
*Last updated: 2026-01-31 after milestone v1.1 started*

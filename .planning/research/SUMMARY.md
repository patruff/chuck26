# Project Research Summary

**Project:** chucklesPRIME v1.1 — Fine-tuning & Inference
**Domain:** LLM fine-tuning (DPO/GRPO) and inference serving
**Researched:** 2026-01-31
**Confidence:** HIGH

## Executive Summary

chucklesPRIME v1.0 successfully generates phonetically-sound parody datasets. Version 1.1 closes the improvement loop by fine-tuning Qwen3-32B on this data and serving it for better parody generation. The recommended approach is to use Unsloth for memory-efficient 4-bit QLoRA training, TRL for DPO/GRPO trainers, and vLLM for inference serving. This stack allows training a 32B parameter model on a single A6000 (48GB) GPU at $0.35/hr on RunPod, with total iteration cost under $3.

The core technical challenge is adapting the existing reward functions (phonetic quality, structure preservation, tool usage) to TRL's GRPO interface while avoiding quantization degradation during model merging. Start with DPO training (simpler, well-documented) to validate the pipeline, then add GRPO with custom phonetic rewards for refinement. The biggest risks are: (1) QLoRA adapter merging into 4-bit weights producing degraded models — solved by merging to 16-bit; (2) chat template mismatches causing Qwen3 to ignore fine-tuning — solved by consistent `enable_thinking=False`; (3) GRPO reward functions returning identical scores — solved by continuous reward functions and group size G=8+.

The existing chucklesPRIME package requires zero code changes for inference. The OpenAICompatibleModel adapter already works with any OpenAI-compatible endpoint, so pointing settings.json at a vLLM URL completes the loop. Training scripts live in a standalone `training/` directory, importing reward functions from the installed package.

## Key Findings

### Recommended Stack

**Unsloth + TRL + vLLM** is the battle-tested stack for QLoRA fine-tuning and serving at this scale.

**Core technologies:**
- **Unsloth** (>= 2026.1.4): QLoRA training accelerator — 2-2.7x faster than standard PEFT, 70% less VRAM via hand-optimized Triton kernels. Only viable way to train 32B QLoRA on single A6000 (48GB) with headroom for GRPO's multi-completion generation.
- **TRL** (>= 0.27.1): DPOTrainer and GRPOTrainer — Official HuggingFace trainers with Unsloth integration. DPOTrainer supports preference pairs; GRPOTrainer supports custom reward functions with configurable weights and vLLM-accelerated generation.
- **vLLM** (>= 0.15.0): OpenAI-compatible inference server — De facto standard for LLM serving. PagedAttention for efficient KV cache, native Qwen3 support, AWQ/GPTQ quantization for fitting 32B on 24GB GPUs. Zero code changes needed — existing OpenAICompatibleModel adapter works as-is.
- **bitsandbytes** (>= 0.49.1): 4-bit NF4 quantization for QLoRA training — Required for fitting 32B models on single-GPU.
- **RunPod**: Cloud GPU provider — Pre-built vLLM templates, network volume storage, competitive pricing. A6000 (48GB) at $0.35/hr for training, RTX 4090 (24GB) at ~$0.44/hr for AWQ-quantized inference.

**Critical stack pattern:** Let `pip install unsloth` resolve torch/transformers/peft versions. Unsloth has specific version coupling. Install Unsloth FIRST, then TRL, then project dependencies.

### Expected Features

**Must have (table stakes):**
- **4-bit QLoRA model loading** — 32B model requires ~64GB at FP16; QLoRA brings to ~26-30GB on A6000.
- **TRL DPOTrainer integration** — Standard DPO implementation, fully compatible with Unsloth.
- **Chat template preservation** — Qwen3 uses `<|im_start|>/<|im_end|>` markers. Mismatch is #1 cause of "trains but doesn't work."
- **LoRA adapter save to Hub** — Version control for trained adapters (~100-300MB).
- **Merged 16-bit model save** — vLLM requires merged model for best performance.
- **vLLM OpenAI-compatible server** — Standard `/v1/chat/completions` endpoint.
- **Existing CLI compatibility** — OpenAICompatibleModel already supports configurable base_url.

**Should have (competitive advantage):**
- **Composite GRPO reward functions** — Adapt existing phonetic/structure/tool rewards to TRL interface. This is unique — no off-the-shelf reward for phonetic parody quality.
- **Full improvement loop automation** — Script the cycle: generate -> train -> serve -> generate better.
- **vLLM LoRA hot-swapping** — Serve multiple adapters on one base model for A/B comparison.
- **Phased training (DPO then GRPO)** — DPO establishes style baseline; GRPO refines phonetic discipline. Research shows DPO excels at "alignment/style" while GRPO excels at "reasoning/structured tasks."
- **Qwen3 thinking mode control** — Enable `/think` during GRPO training for reasoning; disable `/no_think` for fast inference.

**Defer (v2+):**
- **Online GRPO with vLLM generation** — Requires dedicated inference GPU, complex memory management.
- **Full fine-tuning (not LoRA)** — 10-30x more expensive, requires multi-GPU.
- **Multi-iteration automated pipeline** — Need manual quality validation first.
- **Production autoscaling** — RunPod serverless handles basic scaling; elaborate autoscaling is yak-shaving.

### Architecture Approach

Training scripts are standalone in `training/` directory, NOT part of the installable package. Heavy GPU dependencies (Unsloth, TRL) stay out of the main package. The scripts install `chuckles_prime` from GitHub on RunPod to import reward functions.

**Major components:**
1. **DPO training script** (`train_dpo.py`) — Loads dataset from Hub, trains QLoRA with Unsloth + TRL DPOTrainer, saves adapter. Zero custom code beyond config.
2. **GRPO training script** (`train_grpo.py`) — Loads dataset from Hub, wraps reward functions for TRL interface, trains with GRPOTrainer. Reward wrappers live in this script as glue code.
3. **Merge and push script** (`merge_and_push.py`) — Loads adapter, merges to 16-bit using Unsloth's `save_method="merged_16bit"`, pushes to Hub. Critical: must merge to 16-bit, not 4-bit.
4. **vLLM inference server** — RunPod pod or serverless worker serving merged model with OpenAI-compatible API.
5. **Reward function wrappers** — Bridge TRL's `(completions, **kwargs) -> list[float]` interface to existing reward functions. Extract text from conversational format, access `original_title` from dataset columns, compute scores.

**Data flow:** Local CLI pushes datasets to Hub -> RunPod training pod loads datasets -> trains adapters -> merges and pushes model to Hub -> vLLM pod serves model -> Local CLI points to vLLM endpoint -> generates better parodies -> loop closes.

**Integration point:** RunPod setup script runs `pip install "git+https://github.com/patruff/chucklesPRIME.git"` to make reward functions importable. Single source of truth for reward logic.

### Critical Pitfalls

1. **Merging QLoRA into 4-bit base produces degraded model** — Always use `save_method="merged_16bit"` which downloads FP16 base and merges LoRA into clean weights. Never use `merged_4bit`. Test merged model output vs adapter-loaded output before pushing to Hub.

2. **Unsloth/TRL/Transformers version mismatch causes silent failures** — Unsloth monkey-patches TRL/Transformers internals. Version skew causes TypeError or wrong loss computation. Always install Unsloth FIRST, let it resolve dependencies. Pin versions: `unsloth>=2026.1.4 trl>=0.27.1 transformers>=5.0.0`.

3. **GRPO reward functions with wrong signature crash training** — TRL passes completions as `list[list[dict]]` in conversational format, not strings. Must extract `completion[0]["content"]`. Must accept `**kwargs` for dataset columns. Must return `list[float]` with length matching batch. Higher value = better (GRPO maximizes).

4. **RunPod container storage loss on pod stop** — Checkpoints on container disk are lost on stop/reboot. Always attach network volume, save to `/workspace/`. Push to Hub immediately after training completes.

5. **Qwen3 chat template mismatch during fine-tuning** — If training data template doesn't match model's expected format, model ignores fine-tuning (1% style adherence). Use `tokenizer.apply_chat_template(enable_thinking=False)` consistently. Do NOT include `<think>` blocks unless explicitly training thinking mode.

6. **GRPO training stuck with zero reward std** — If all completions get same reward, `frac_reward_zero_std` approaches 1.0, no learning signal. Use continuous (not binary) reward functions, group size G=8+, and `scale_rewards="batch"`. Monitor `frac_reward_zero_std` — if > 0.5, fix reward functions.

## Implications for Roadmap

Based on research, suggested phase structure prioritizes validation before complexity:

### Phase 1: Environment & Foundation
**Rationale:** Must establish working training environment before any GPU work. RunPod setup, dependency pinning, and storage configuration prevent wasted GPU hours. GRPO reward wrappers must be tested offline before burning GPU credits on broken functions.

**Delivers:**
- RunPod setup script (`setup_runpod.sh`) with pinned versions
- Network volume configuration and validation
- GRPO reward function wrappers with unit tests
- Pre-flight validation script (single training step test)

**Addresses:**
- Version mismatch pitfall (pin Unsloth/TRL/Transformers)
- Container storage loss (network volume setup)
- GRPO reward signature errors (unit tests catch offline)

**Avoids:**
- Losing checkpoints to container disk
- Silent version compatibility bugs
- Discovering reward function bugs after hours of training

**Research flags:** Standard patterns — RunPod and Python environment setup are well-documented. No deep research needed.

### Phase 2: DPO Training Pipeline
**Rationale:** DPO is simpler than GRPO (no custom rewards, just preference pairs). Validates the full Unsloth + TRL + RunPod pipeline with fewer moving parts. Dataset format already correct from v1.0. Success here proves training, merging, and Hub push all work before adding GRPO complexity.

**Delivers:**
- DPO training script (`train_dpo.py`)
- Training run on small dataset subset (validation)
- Merged 16-bit model pushed to Hub
- Quality validation script (before/after comparison)

**Uses:**
- Unsloth FastModel with 4-bit QLoRA
- TRL DPOTrainer with existing Hub dataset
- `save_method="merged_16bit"` for merge

**Implements:**
- Training script component
- Merge and push component (for DPO first)

**Avoids:**
- QLoRA 4-bit merge degradation (use merged_16bit)
- Chat template mismatch (apply Qwen3 template with enable_thinking=False)

**Research flags:** Standard patterns — DPO with Unsloth is well-documented in TRL/Unsloth examples. Minimal research needed.

### Phase 3: vLLM Inference Serving
**Rationale:** With a merged model from Phase 2, validate serving and end-to-end integration before adding GRPO. Proves the existing CLI works with vLLM endpoint (zero code changes claim). Completes the loop for DPO-only pipeline first.

**Delivers:**
- vLLM server setup (RunPod pod or serverless)
- settings.json template for vLLM endpoint
- Inference validation (compare to base model)
- End-to-end test (generate -> train DPO -> serve -> generate)

**Uses:**
- vLLM serve with merged model from Phase 2
- AWQ/GPTQ quantization for 24GB GPUs (optional)
- OpenAI-compatible API

**Implements:**
- vLLM inference server component
- Integration with existing OpenAICompatibleModel

**Avoids:**
- Serving unmerged LoRA (merge first)
- max_model_len=40960 OOM (set to 8192 for parodies)

**Research flags:** Standard patterns — vLLM serving is well-documented. RunPod has vLLM worker templates.

### Phase 4: GRPO Training Pipeline
**Rationale:** With DPO pipeline validated and vLLM serving working, add GRPO for phonetic reward refinement. Reward wrappers from Phase 1 are pre-tested. This phase is higher risk due to custom rewards but builds on proven foundation.

**Delivers:**
- GRPO training script (`train_grpo.py`)
- Integrated reward function wrappers (phonetic, structure, format)
- Training run with reward monitoring
- Phased training option (load DPO adapter, continue with GRPO)

**Uses:**
- TRL GRPOTrainer with custom reward functions
- Reward function wrappers from Phase 1
- Existing GRPO dataset on Hub

**Implements:**
- GRPO training script component
- Composite reward function integration

**Avoids:**
- GRPO reward signature errors (pre-tested wrappers)
- Zero-std stuck training (continuous rewards, G=8, monitor frac_reward_zero_std)

**Research flags:** Needs research — Custom reward functions for phonetic similarity are novel. May need iteration on reward weighting and group size. Monitor training closely in first runs.

### Phase 5: Automation & Iteration
**Rationale:** With all components working independently, automate the full loop and add convenience features. This phase is optional for v1.1 but sets up productive iteration.

**Delivers:**
- End-to-end automation script (generate -> train -> serve)
- vLLM LoRA hot-swapping (A/B test adapters)
- Training metrics dashboard (W&B or TensorBoard)
- Qwen3 thinking mode toggling

**Uses:**
- All components from Phases 1-4
- vLLM LoRA serving features
- Monitoring tools

**Implements:**
- Full improvement loop automation
- Differentiator features from FEATURES.md

**Research flags:** Standard patterns — Orchestration and monitoring are well-understood. Optional for MVP.

### Phase Ordering Rationale

- **Environment first:** Prevents wasted GPU hours from avoidable setup errors. Fast feedback loop for validation.
- **DPO before GRPO:** Simpler training method validates pipeline before custom rewards. DPO dataset already correct.
- **Serving after DPO:** Can complete DPO loop (train -> serve -> generate) before GRPO complexity. Proves zero-code-change claim early.
- **GRPO after foundation:** Builds on validated components. Reward functions pre-tested offline. Phased training (DPO -> GRPO) becomes easy.
- **Automation last:** All components must work independently before automating. Optional for MVP.

**Dependency chain:** Phase 1 (setup) -> Phase 2 (DPO training) -> Phase 3 (serving) closes one loop. Phase 4 (GRPO) adds to loop. Phase 5 (automation) optimizes loop.

**Avoids compounding risk:** Each phase delivers value independently. Can stop after Phase 3 with working DPO pipeline. GRPO is additive, not blocking.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 4 (GRPO):** Custom reward function design for phonetic similarity is novel. Reward weighting (phonetic vs structure vs format) needs experimentation. Group size and loss type (GRPO vs GSPO) may need tuning.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Environment):** RunPod setup, Python dependencies, unit testing are well-documented.
- **Phase 2 (DPO):** DPO with Unsloth + TRL is extensively documented in official guides and examples.
- **Phase 3 (vLLM):** Inference serving with vLLM has official docs and RunPod templates.
- **Phase 5 (Automation):** Orchestration patterns are standard; implementation is straightforward scripting.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified via PyPI (Jan 2026). Unsloth/TRL/vLLM integration confirmed in official docs. RunPod pricing verified. |
| Features | MEDIUM-HIGH | DPO features are standard and well-documented (HIGH). GRPO custom reward patterns verified in TRL docs, but phonetic reward quality is unproven (MEDIUM). vLLM integration with existing code verified from source analysis (HIGH). |
| Architecture | HIGH | Integration pattern (pip install from GitHub for rewards) verified from existing pyproject.toml and is standard practice. Data flow and component boundaries are straightforward. Reward wrapper pattern confirmed from TRL docs. |
| Pitfalls | HIGH | All critical pitfalls verified from multiple GitHub issues and official docs. QLoRA merge degradation, version mismatch, GRPO reward signatures, RunPod storage loss, Qwen3 chat template — all have documented cases. |

**Overall confidence:** HIGH

The stack is mature and well-integrated (Unsloth + TRL is the recommended pairing). The architecture leverages existing project structure cleanly. The pitfalls are known and have documented solutions. The main uncertainty is GRPO reward function effectiveness for phonetic parody quality, which is domain-specific and requires experimentation.

### Gaps to Address

**GRPO reward function effectiveness:** The three reward functions (phonetic quality, structure preservation, tool usage) are proven for scoring existing parodies but untested as training signals. May need iteration on:
- Reward scaling and weighting
- Continuous vs binary rewards
- Group size (G) for diversity
- Whether to use GRPO, GSPO, or CISPO loss variant

**Mitigation:** Start with DPO (Phase 2) to establish baseline quality. Add GRPO (Phase 4) as refinement. Monitor `frac_reward_zero_std` and reward curves closely in first GRPO runs. Iterate on reward design if stuck.

**Qwen3 thinking mode strategy:** Research shows thinking mode can improve GRPO reasoning quality but adds latency and cost. Unclear if parody generation benefits from thinking.

**Mitigation:** Train with `enable_thinking=False` (standard mode) initially. Experiment with thinking mode in Phase 5 if non-thinking results are weak.

**vLLM quantization quality tradeoff:** AWQ 4-bit on RTX 4090 (24GB) is cheaper but quality loss is unknown for parody generation.

**Mitigation:** Serve FP8 or FP16 on A6000 (48GB) for quality; only quantize to AWQ if cost becomes prohibitive. Test before committing.

## Sources

### Primary (HIGH confidence)
- [TRL GRPOTrainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer) — Reward function interface, dataset format
- [TRL DPOTrainer Documentation](https://huggingface.co/docs/trl/en/dpo_trainer) — DPO training with PEFT/QLoRA
- [Unsloth Qwen3 Documentation](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — Model loading, VRAM requirements
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) — GRPO setup, save methods
- [vLLM Official Documentation](https://docs.vllm.ai/en/latest/) — OpenAI-compatible serving
- [RunPod Storage Documentation](https://docs.runpod.io/pods/storage/types) — Network volumes, persistence
- [PyPI verified versions](https://pypi.org/) — unsloth 2026.1.4, trl 0.27.1, vllm 0.15.0, transformers 5.0.0, peft 0.18.1

### Secondary (MEDIUM confidence)
- [HuggingFace blog: Unsloth + TRL](https://huggingface.co/blog/unsloth-trl) — Performance benchmarks
- [RunPod GPU pricing](https://www.runpod.io/pricing) — Verified 2026-01-31
- [DPO vs GRPO comparison](https://towardsai.net/p/artificial-intelligence/mastering-llm-fine-tuning-grpo-ppo-and-dpo-compared) — When to use each
- GitHub Issues: unslothai/unsloth #195, #2516, #1089 (merge degradation); #2916, #3750 (version mismatch); #2614 (GRPO loss=0)
- GitHub Issues: huggingface/trl #2644, #2771 (GRPO rewards); #2578 (DPO bugs)
- GitHub Issues: QwenLM/Qwen3 #1718 (chat template), #1286 (thinking mode)

### Tertiary (LOW confidence)
- VRAM estimates for Qwen3-32B QLoRA (~26-30GB) — Extrapolated from Unsloth's 32B table and community reports
- Training time estimates (2-4 hrs on A6000) — Based on smaller model benchmarks; actual time varies with dataset size
- AWQ quantization quality for parodies — No domain-specific research; general quantization tradeoffs apply

---
*Research completed: 2026-01-31*
*Ready for roadmap: yes*

# Requirements: chucklesPRIME

**Defined:** 2026-01-31
**Core Value:** Generate quality reasoning data about what makes a good phonetic parody, in formats ready for GRPO and DPO fine-tuning — then close the loop by training and deploying the model.

## v1.0 Requirements (Complete)

### Configuration

- [x] **CFG-01**: Load funny word lists from external `funny_words.json` at runtime (not in repo)
- [x] **CFG-02**: Load user style preferences from external `preferences.json` — opaque text injected into prompts
- [x] **CFG-03**: Load and clean human-generated parody examples from CSV (~1,234 rows, fix formatting issues)
- [x] **CFG-04**: All config files outside repo, referenced by path in a single settings file

### LLM Backend

- [x] **LLM-01**: Custom model adapter for OpenAI-compatible chat completion APIs
- [x] **LLM-02**: Backend config in JSON (model name, API base URL, API key env var name)

### Generation

- [x] **GEN-01**: Read input titles from CSV file
- [x] **GEN-02**: Generate 2 top parody candidates per input title via smolagents CodeAgent
- [x] **GEN-03**: Capture full reasoning traces per generation (agent thinking, tool calls, results)
- [x] **GEN-04**: Use existing phonetic tools (WordPhoneTool, ParodyWordSuggestionTool) from HF Hub

### Dataset Output

- [x] **DATA-01**: GRPO-compatible dataset — prompt-only with metadata columns (original_title, phonetic_scores) for verifiable reward functions during training
- [x] **DATA-02**: DPO-compatible dataset — human parodies as "chosen", model inferior outputs as "rejected"
- [x] **DATA-03**: Push both datasets to HuggingFace Hub
- [x] **DATA-04**: Archive full reasoning traces as JSONL
- [x] **DATA-05**: Composite verifiable reward signals (phonetic quality, tool usage, structure preservation — not binary threshold)

### Project Structure

- [x] **PROJ-01**: Clean Python package structure with `pyproject.toml`
- [x] **PROJ-02**: CLI entry point for batch generation and dataset conversion

## v1.1 Requirements (Current Milestone)

### DPO Training

- [ ] **DPO-01**: Load Qwen3-32B in 4-bit QLoRA via Unsloth with gradient checkpointing on RunPod A6000/A100
- [ ] **DPO-02**: Train with TRL DPOTrainer using existing DPO dataset from HuggingFace Hub
- [ ] **DPO-03**: Preserve Qwen3 chat template throughout training (match base model template exactly)
- [ ] **DPO-04**: Save training checkpoints every N steps to RunPod network volume
- [ ] **DPO-05**: Standalone training script in `training/` directory (copy-paste to RunPod)

### GRPO Training

- [ ] **GRPO-01**: Wrap existing reward functions (phonetic quality, tool usage, structure preservation) to TRL GRPOTrainer interface: `(completions, **kwargs) -> list[float]`
- [ ] **GRPO-02**: Train with TRL GRPOTrainer using existing GRPO dataset from HuggingFace Hub with multi-generation per prompt (8+ completions)
- [ ] **GRPO-03**: Support multiple concurrent reward functions with configurable weights
- [ ] **GRPO-04**: 4-bit QLoRA via Unsloth (same as DPO) with gradient checkpointing
- [ ] **GRPO-05**: Standalone training script in `training/` directory (copy-paste to RunPod)

### Model Export

- [ ] **EXP-01**: Save LoRA adapter to HuggingFace Hub (~100-300MB)
- [ ] **EXP-02**: Merge LoRA into FP16 base model via Unsloth `merged_16bit` (not into 4-bit base)
- [ ] **EXP-03**: Push merged model to HuggingFace Hub
- [ ] **EXP-04**: Validate merged model quality against adapter-loaded model (run test prompts, compare outputs)

### Inference Serving

- [ ] **INF-01**: Serve fine-tuned model via vLLM with OpenAI-compatible API on RunPod
- [ ] **INF-02**: Quantized inference (AWQ/INT8) to fit on RTX 3090/4090 or A6000
- [ ] **INF-03**: Existing `chuckles generate` CLI works with vLLM endpoint (zero code changes — settings.json config only)

### Improvement Loop

- [ ] **LOOP-01**: End-to-end validation: generate parodies with fine-tuned model, verify quality improvement over base model
- [ ] **LOOP-02**: Full loop documentation: generate → train → merge → serve → generate better data

## v2 Requirements

### Generation Enhancements

- **GEN-V2-01**: Checkpointed resume for interrupted batch runs
- **GEN-V2-02**: Multi-generation with best-of-N selection (GRPO-style group sampling)
- **GEN-V2-03**: Difficulty-aware data selection (30-70% pass@k sweet spot per GRPO++ research)

### Quality

- **QUAL-V2-01**: Diversity enforcement (transformation patterns, humor categories, Self-BLEU)
- **QUAL-V2-02**: Human example calibration (score known examples to set reward thresholds)
- **QUAL-V2-03**: Dynamic sampling — filter prompts where all completions are correct (zero gradient)

### Training Enhancements

- **TRAIN-V2-01**: DAPO/Dr.GRPO-compatible output (token-level loss normalization, no std dev in advantage)
- **TRAIN-V2-02**: Overlong reward shaping for truncated samples
- **TRAIN-V2-03**: Negative example generation (ablated runs without tools for DPO rejected pairs)
- **TRAIN-V2-04**: LoRA hot-swapping via vLLM for A/B testing
- **TRAIN-V2-05**: W&B experiment tracking dashboard
- **TRAIN-V2-06**: Automated loop orchestration script

### Infrastructure

- **INF-V2-01**: LiteLLM adapter for 100+ provider support
- **INF-V2-02**: Fallback/retry logic across providers

## Out of Scope

| Feature | Reason |
|---------|--------|
| Web UI or API server | CLI tool only |
| Google Drive integration | Simplify to local + HF Hub |
| Interactive labeling UI | Use auto-labeling with composite rewards |
| LLM-as-judge humor scoring | Subjective, not verifiable — use phonetic metrics |
| GSPO/GMPO/CISPO loss variants | Standard GRPO/DPO first |
| Curriculum learning | Data composition over curriculum |
| Full-precision training | 4-bit QLoRA only for v1.1 |
| Multi-GPU training | Single A6000/A100 sufficient for 32B QLoRA |
| Custom CUDA kernels | Rely on Unsloth optimizations |
| Automated retraining pipeline | Manual iteration for v1.1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 1 | Complete |
| CFG-02 | Phase 1 | Complete |
| CFG-03 | Phase 1 | Complete |
| CFG-04 | Phase 1 | Complete |
| LLM-01 | Phase 1 | Complete |
| LLM-02 | Phase 1 | Complete |
| GEN-01 | Phase 2 | Complete |
| GEN-02 | Phase 2 | Complete |
| GEN-03 | Phase 2 | Complete |
| GEN-04 | Phase 2 | Complete |
| DATA-01 | Phase 3 | Complete |
| DATA-02 | Phase 3 | Complete |
| DATA-03 | Phase 3 | Complete |
| DATA-04 | Phase 3 | Complete |
| DATA-05 | Phase 3 | Complete |
| PROJ-01 | Phase 1 | Complete |
| PROJ-02 | Phase 4 | Complete |
| DPO-01 | — | Pending |
| DPO-02 | — | Pending |
| DPO-03 | — | Pending |
| DPO-04 | — | Pending |
| DPO-05 | — | Pending |
| GRPO-01 | — | Pending |
| GRPO-02 | — | Pending |
| GRPO-03 | — | Pending |
| GRPO-04 | — | Pending |
| GRPO-05 | — | Pending |
| EXP-01 | — | Pending |
| EXP-02 | — | Pending |
| EXP-03 | — | Pending |
| EXP-04 | — | Pending |
| INF-01 | — | Pending |
| INF-02 | — | Pending |
| INF-03 | — | Pending |
| LOOP-01 | — | Pending |
| LOOP-02 | — | Pending |

**Coverage:**
- v1.0 requirements: 17 total (all complete)
- v1.1 requirements: 16 total (all pending)
- Mapped to phases: 17 (v1.0) + 0 (v1.1 — awaiting roadmap)
- Unmapped: 16

---
*Requirements defined: 2026-01-31*
*Last updated: 2026-01-31 after milestone v1.1 requirements definition*

# Requirements: chucklesPRIME

**Defined:** 2026-01-31
**Core Value:** Generate quality reasoning data about what makes a good phonetic parody, in formats ready for GRPO and DPO fine-tuning.

## v1 Requirements

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

## v2 Requirements

### Generation Enhancements

- **GEN-V2-01**: Checkpointed resume for interrupted batch runs
- **GEN-V2-02**: Multi-generation with best-of-N selection (GRPO-style group sampling)
- **GEN-V2-03**: Difficulty-aware data selection (30-70% pass@k sweet spot per GRPO++ research)

### Quality

- **QUAL-V2-01**: Diversity enforcement (transformation patterns, humor categories, Self-BLEU)
- **QUAL-V2-02**: Human example calibration (score known examples to set reward thresholds)
- **QUAL-V2-03**: Dynamic sampling — filter prompts where all completions are correct (zero gradient)

### Training Integration

- **TRAIN-V2-01**: DAPO/Dr.GRPO-compatible output (token-level loss normalization, no std dev in advantage)
- **TRAIN-V2-02**: Overlong reward shaping for truncated samples
- **TRAIN-V2-03**: Negative example generation (ablated runs without tools for DPO rejected pairs)

### Infrastructure

- **INF-V2-01**: LiteLLM adapter for 100+ provider support
- **INF-V2-02**: Fallback/retry logic across providers

## Out of Scope

| Feature | Reason |
|---------|--------|
| Fine-tuning loop | v1 is generate + convert + push only |
| Web UI or API server | CLI tool only |
| Google Drive integration | Simplify to local + HF Hub |
| Interactive labeling UI | Use auto-labeling with composite rewards |
| LLM-as-judge humor scoring | Subjective, not verifiable — use phonetic metrics |
| GSPO/GMPO/CISPO variants | Standard GRPO format first, advanced variants in v2+ |
| Curriculum learning | Data composition over curriculum for v1 |

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

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-01-31*
*Last updated: 2026-01-31 after Phase 4 completion*

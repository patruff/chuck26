# chucklesPRIME

## What This Is

A pure Python tool that generates phonetically sound parody titles ("chuckles") from input titles/words, captures the model's reasoning traces, and converts everything into HuggingFace RLVR datasets for fine-tuning. Funny words, user style preferences, and human-generated examples are loaded as external configs — the app never sees their contents, it just passes them through.

## Core Value

Generate quality reasoning data about what makes a good phonetic parody, in formats ready for GRPO and DPO fine-tuning.

## Requirements

### Validated

- ✓ Phonetic similarity scoring (WordPhoneTool) — existing, works well
- ✓ Parody word suggestion with detailed metrics (ParodyWordSuggestionTool) — existing, works well
- ✓ smolagents CodeAgent orchestration for multi-step parody generation — existing
- ✓ Known parody examples loaded from CSV (known100.csv) — existing
- ✓ Custom phonetic pronunciations for parody-specific words — existing

### Active

- [ ] External JSON config for funny words (loaded at runtime, out of repo)
- [ ] External JSON config for user preferences/style description (opaque to app, loaded at runtime)
- [ ] Load human-generated parody examples from CSV (thousands of input→output pairs)
- [ ] Swappable LLM backend (any HuggingFace model, Cerebras, or API-compatible endpoint)
- [ ] CSV input → 2 top parodies per title + reasoning traces output
- [ ] Convert output to GRPO-compatible dataset (prompt-only with metadata for verifiable reward functions)
- [ ] Convert output to DPO-compatible dataset (human parodies as chosen, model inferior as rejected)
- [ ] Push both datasets to HuggingFace Hub
- [ ] Clean project structure (not a Colab notebook dump)

### Out of Scope

- Fine-tuning loop — v1 is generate + convert + push only
- Web UI or API server — CLI/script tool only
- Google Drive integration — simplify to local files + HF Hub
- Real-time/interactive generation — batch processing via CSV
- Training evaluation — handled separately after dataset is on HF Hub

## Context

- Existing codebase in `parodies2026/` works but grew from a Colab notebook. Architecture is sound (phonetic tools, smolagents pipeline, output capture) but needs restructuring.
- Phonetic tools (`WordPhoneTool`, `ParodyWordSuggestionTool`) are deployed on HuggingFace Hub under `patruff/`. They work and should be kept as-is.
- Current output quality issue: parodies are too obvious/boring. The model plays it safe. Better human examples and user style preferences (injected externally) should improve this.
- RLVR with Qwen3 is the training target. The reasoning traces need to capture *why* a parody works phonetically and humoristically.
- Dual training approach: GRPO for reinforcement learning with verifiable rewards (phonetic scores as reward signals), DPO for preference learning (human parodies as gold standard).
- GRPO dataset is prompt-only — model generates during training, reward functions score outputs. Key insight from GRPO++ research: composite rewards, token-level loss, dynamic sampling.
- Team has ~1,234 human-generated parody examples (CSV: input, output, explanation) that serve as the gold standard and DPO "chosen" examples. Data has formatting issues that need cleaning.
- `known100.csv` has 100 curated examples with reasoning — subset of the larger human dataset.
- Custom model adapter preferred over LiteLLM — keep control over the adapter layer.

## Constraints

- **Language**: Pure Python — no JS, no compiled extensions
- **Framework**: smolagents (HuggingFace) for agent orchestration
- **Config isolation**: Funny words and user preferences must not be in the codebase — loaded from external JSON files at runtime
- **LLM flexibility**: Must support any model with a chat completion API (HuggingFace Inference, Cerebras, OpenAI-compatible)
- **Output format**: HuggingFace TRL-compatible GRPO and DPO dataset formats
- **Existing tools**: Keep WordPhoneTool and ParodyWordSuggestionTool as-is (loaded from HF Hub)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| JSON files for config (not env vars) | Funny words and preferences are structured data, not simple strings | — Pending |
| Keep phonetic tools as-is | They work, phonetic scoring is solid | — Pending |
| Custom LLM adapter (not LiteLLM) | Keep control over adapter layer, support any OpenAI-compatible API | — Pending |
| Dual dataset output (GRPO + DPO) | GRPO for verifiable reward training, DPO for preference learning from human examples | — Pending |
| User preferences opaque to app | App injects the preference text into prompts without parsing it — allows users to describe their humor style freely | — Pending |
| Batch-only (CSV in/out) | Simplifies architecture, matches the data generation workflow | — Pending |

---
*Last updated: 2026-01-31 after requirements definition*

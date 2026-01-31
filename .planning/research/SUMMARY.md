# Project Research Summary

**Project:** chucklesPRIME
**Domain:** RLVR training data generation pipeline for phonetic parody titles
**Researched:** 2026-01-31
**Confidence:** HIGH

## Executive Summary

chucklesPRIME is an RLVR (Reinforcement Learning from Verifiable Rewards) training data generation pipeline that produces phonetically similar parody titles for model training. The recommended approach uses HuggingFace's smolagents library with CodeAgent for multi-step reasoning, pronouncing library for phonetic verification, and generates data in TRL's GRPO-compatible format. This is a data generation pipeline, not a training pipeline, so it requires no GPU resources and operates entirely through remote LLM APIs.

The core architecture pattern is clean: external config files (funny words, style preferences, human examples) are loaded once at startup, an LLM adapter layer provides backend flexibility (Cerebras, OpenAI, etc.), smolagents CodeAgent orchestrates multi-step phonetic verification, and outputs are parsed into structured RLVR datasets pushed to HuggingFace Hub. The most critical insight from research is that RLVR trains models to find known-good solutions faster (search compression), not to expand capability, so data quality and verifiability are paramount.

The key risks are: (1) reward hacking via phonetic score gaming, mitigated by composite multi-dimensional rewards instead of single thresholds; (2) diversity collapse in training data, mitigated by difficulty-aware data selection and diversity monitoring; (3) smolagents code parsing failures with non-OpenAI models, mitigated by output sanitization and streaming disabled; (4) verifier imperfection creating false positives/negatives, mitigated by calibrating against human examples from known100.csv.

## Key Findings

### Recommended Stack

The stack is deliberately minimal because this is an API-only data generation tool, not a training pipeline. No torch, transformers, or GPU dependencies are needed. smolagents v1.24.0 is the official HuggingFace lightweight agent framework (successor to transformers.agents), providing first-class CodeAgent support and swappable model backends. The datasets library handles HuggingFace Hub upload with native Parquet serialization and Dataset Viewer support.

**Core technologies:**
- **smolagents >= 1.24.0**: Agent orchestration with CodeAgent for multi-step reasoning traces
- **datasets >= 4.5.0**: Dataset creation and push_to_hub for HuggingFace Hub upload
- **pronouncing >= 0.2.0**: CMU Pronouncing Dictionary interface for phonetic analysis
- **litellm >= 1.55.0**: Multi-provider LLM routing for 100+ backends including Cerebras
- **trl >= 0.27.0**: Format reference only (defines GRPOTrainer dataset contract)
- **Python >= 3.10**: For match/case and modern typing support

**Critical version note:** Pin smolagents to exact version (experimental library, breaks between versions). Use smolagents built-in model classes (InferenceClientModel, LiteLLMModel) instead of custom adapters. Cerebras is already a supported provider via InferenceClientModel with provider="cerebras" or LiteLLMModel with model_id="cerebras/llama-3.3-70b".

### RLVR Dataset Format

GRPOTrainer expects prompt-only datasets with conversational format. The dataset has one required column (prompt) and auxiliary columns passed to reward functions during training.

**Required schema:**
```python
{
    "prompt": [  # Conversational format (list of message dicts)
        {"role": "system", "content": "You create phonetic parodies..."},
        {"role": "user", "content": "Create a parody of 'The Matrix'"}
    ],
    # Auxiliary columns for reward functions:
    "original_title": "The Matrix",
    "parody_title": "The Grape Fatsby",  # Ground truth example
    "reasoning_trace": "[{...}]",  # JSON string of agent steps
    "phonetic_distance": 0.15,
    "generation_model": "Qwen/Qwen2.5-72B-Instruct"
}
```

**Critical insight:** GRPO training generates completions online; the dataset contains prompts only, not model responses. Reward functions receive auxiliary columns as kwargs. Must set `remove_unused_columns=False` in GRPOConfig to preserve metadata.

### Expected Features

**Must have (table stakes):**
- **Deterministic reproducibility**: Seed management, environment pinning, input versioning with hashes
- **VeRL-compatible output format**: Parquet schema with prompt, ability, reward_model, extra_info fields
- **Structured reasoning trace capture**: Full multi-step agent traces with tool calls, not just final output
- **Deduplication**: Input, output, and cross-batch deduplication with near-duplicate detection
- **Quality-gated auto-labeling**: Threshold-based filtering with quality labels and label sources
- **Batch processing with resume**: Checkpoint after each title, resume from checkpoint on crash
- **Logging and audit trail**: Run ID, config checksums, per-title scores, summary statistics

**Should have (competitive):**
- **Difficulty-aware data selection**: Track pass@k per title, prioritize medium-difficulty examples (30-70% success rate)
- **Composite verifiable reward function**: Decompose into phonetic_validity, phonetic_quality, structural_fidelity, tool_usage_completeness, reasoning_quality
- **Multi-generation with best-of-N selection**: Generate N candidates per title, select top-2, retain rejected for DPO
- **Provenance-rich records**: Full metadata for reproducibility (run_id, model, config_hashes, timestamps)
- **Negative example generation**: Ablated generation without tools, perturbations, low-threshold captures for DPO pairs
- **Diversity enforcement**: Track funny_words usage categories, measure output diversity with Self-BLEU

**Defer (v2+):**
- Interactive labeling (auto-labeling first, interactive for edge cases later)
- Real-time Hub pushing (explicit manual step instead)
- Custom phonetic embeddings (CMU dictionary + custom_phones sufficient)
- Multi-model comparison (focus on one model done well)

### Architecture Approach

The system has six components with clean boundaries: (1) Config Layer loads external files (funny_words.json, preferences.json, human_examples.csv) at startup; (2) LLM Adapter Layer uses smolagents built-in model classes (no custom adapter needed); (3) Agent Orchestration uses smolagents CodeAgent with word_phone_tool; (4) Prompt Builder assembles context from title, config, examples, suggestions; (5) Output Parser extracts structured data and computes quality signals; (6) Pipeline Orchestrator orchestrates CSV in -> process -> dataset out.

**Major components:**
1. **Config loading** (`config.py`) — Load external files, return frozen AppConfig dataclass
2. **LLM adapter** (`llm.py`) — Create smolagents model from backend string via factory function
3. **Prompt builder** (`prompt.py`) — Assemble prompts from title + config + examples + suggestions
4. **Agent execution** (`agent.py`) — Initialize CodeAgent, run, return raw output with traces
5. **Output parser** (`parser.py`) — Extract structured data from raw agent output (thinking, tool calls, attempts)
6. **Dataset converter** (`dataset.py`) — Convert to TRL formats (GRPO prompt-only, DPO preference, SFT), push to Hub
7. **Pipeline orchestrator** (`pipeline.py`) — Main loop: CSV -> agent -> parse -> dataset

**Data flow:** Input CSV + external configs -> load_config() -> create_model() -> for each title: pre-compute suggestions -> build_prompt() -> run_agent() -> parse_output() -> convert_to_RLVR_format() -> push_to_hub().

**Key architectural decisions:**
- Use smolagents built-in InferenceClientModel or LiteLLMModel (no custom CerebrasModel)
- Config objects via factory function (no DI framework, env vars only for secrets)
- JSONL as primary output format (supports nested structures, CSV only for viewing)
- Tools stay on HuggingFace Hub (load via load_tool(), keep local copies for testing)
- Human examples: random sampling for few-shot prompts, exclude target title from examples, include known_good_parody in dataset if available

### Critical Pitfalls

1. **Reward hacking via phonetic score gaming** — Model learns to exploit scoring function rather than produce genuinely funny parodies. Signs: same replacement words repeated, high phonetic scores but low humor. Prevention: composite rewards (phonetic + diversity penalty), track replacement word entropy, use RLVRR-style decomposed rewards. Address in: Architecture (reward function design).

2. **Diversity collapse in training data** — GRPO concentrates on narrow high-reward outputs, model loses creativity. Signs: Pass@1 improves but Pass@k degrades, entropy decreases. Prevention: forward-KL or JS-divergence instead of reverse-KL, difficulty-based curriculum, monitor Pass@k alongside Pass@1. Address in: Data Generation (diversity monitoring), Training (objective choice).

3. **smolagents code parsing failures** — CodeAgent expects specific markdown format, models like Qwen3 via Cerebras produce unparsable output causing syntax error loops. Signs: repeated "Error in code parsing", agent reaches max_steps without output. Prevention: set stream_outputs=False (smolagents issue #1872), output sanitization in model wrapper, max_steps=10-15 with graceful fallback. Address in: Architecture (model wrapper design).

4. **Verifier imperfection (false positives/negatives)** — Phonetic scoring systematically wrong in both directions. Signs: human reviewers disagree >20%, known-funny parodies score below threshold. Prevention: audit verifier against known100.csv examples, track false positive/negative rates, apply noise-corrected RLVR, expand custom_phones dictionary. Address in: Data Generation (verifier calibration).

5. **RLVR trains speed, not new capability** — RLVR performs search compression (teaches model to find known-good answers faster), doesn't expand capability frontier. Signs: trained model identical to base model's best-of-N, no improvement on consistent-fail titles. Prevention: establish Pass@k baseline before training, focus on sweet spot difficulty (30-70% success rate), use SFT first for new capabilities. Address in: Data Generation (difficulty filtering).

**Additional critical pitfalls:**
- **CMU dictionary OOV words** — Many creative words not in CMU dict (slang, proper nouns, neologisms). Prevention: implement g2p fallback, expand custom_phones (currently only 4 entries), pre-compute pronunciations for funny_words list.
- **Phonetic scoring inconsistency** — word_phone.py and parody_suggestions.py have different similarity algorithms. Prevention: extract shared phonetic logic into single module, write unit tests.
- **Binary reward granularity mismatch** — Binary 0/1 rewards destroy gradient information (0.61 and 0.99 both get reward=1). Prevention: use continuous scores, decompose into multiple components.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Foundation (Config + Parser + Prompts)
**Rationale:** These modules have zero external dependencies and can be tested with fixture data from existing parodies2026/ output. Parser is the most fragile component (regex-based) and benefits from early testing.

**Delivers:** AppConfig dataclass with load_config() factory, GenerationRecord dataclass with extraction functions, prompt templates ported from system_prompt.py

**Implements:** Config Layer, Output Parser (pure functions, no I/O), Prompt templates

**Avoids:** Schema drift (define schema as dataclass early), CSV output fragility (JSONL primary format)

**Research needed:** None (standard patterns)

### Phase 2: LLM + Agent Integration
**Rationale:** Once config and parsing work, wire up the LLM. Critical validation is that LiteLLMModel/InferenceClientModel works as drop-in replacement for CerebrasModel.

**Delivers:** create_model() factory for smolagents backends, PromptBuilder class with example selection, run_agent() function using CodeAgent

**Uses:** smolagents>=1.24.0 (pinned), litellm>=1.55.0, openai>=1.50.0

**Avoids:** smolagents code parsing failures (set stream_outputs=False, max_steps=10-15), tool import authorization errors (correct authorized_imports list), Hub tool loading as network dependency (vendor tools locally)

**Research needed:** Minimal (validate LiteLLMModel with Cerebras, test output sanitization patterns)

### Phase 3: Dataset Conversion + Hub Push
**Rationale:** Output end depends on having real generation records to convert. Build after generation pipeline works.

**Delivers:** Format converters (GRPO prompt-only, DPO preference, SFT), Hub push function using datasets library, format validation tests

**Implements:** Dataset converter component, VeRL parquet schema mapping

**Avoids:** GRPO dataset format mismatch (TRL-compatible from start), missing config_name in HF YAML (use push_to_hub())

**Research needed:** None (TRL format well-documented)

### Phase 4: Pipeline + Batch Processing
**Rationale:** Pure glue code, trivial once all components work independently. This is where batch processing, checkpointing, resume, and deduplication logic lives.

**Delivers:** pipeline.py orchestration loop, CLI with argparse, end-to-end test (CSV -> generation -> RLVR dataset -> Hub push), pyproject.toml packaging

**Implements:** Pipeline Orchestrator, Batch processing with resume

**Avoids:** No deduplication (implement unique IDs, dedup before training), rate limiting issues (token-aware limiting, backoff, caching), batch processing fragility (checkpoint after each title)

**Research needed:** None (orchestration patterns)

### Phase 5: Quality + Calibration
**Rationale:** Makes data actually useful for training. Composite rewards, human example calibration, difficulty tracking.

**Delivers:** Composite reward function implementation, human example scoring against phonetic metrics, calibrated auto-labeling thresholds, difficulty tracking (pass@k per title)

**Addresses:** Difficulty-aware data selection, composite verifiable rewards, provenance-rich records

**Avoids:** Reward hacking (composite rewards), verifier imperfection (calibrate against known100.csv), training data contamination (exclude target from examples)

**Research needed:** Deep (reward function tuning, threshold calibration against human judgments)

### Phase 6: Diversity + Optimization
**Rationale:** Makes data notably better. Multi-generation, best-of-N, diversity metrics, negative examples for DPO.

**Delivers:** Multi-generation with best-of-N selection, diversity metrics (Self-BLEU, entropy tracking), negative example generation (ablated, perturbation, low-threshold)

**Addresses:** Negative example generation, diversity enforcement

**Avoids:** Diversity collapse (forward-KL divergence, curriculum learning)

**Research needed:** Moderate (diversity metrics implementation, DPO pair construction strategies)

### Phase Ordering Rationale

- **Phase 1 first** because config, parser, and prompts have no network dependencies and can be tested immediately with existing output fixtures
- **Phase 2 next** to validate the LLM adapter replacement (biggest architectural risk: will LiteLLMModel work as claimed?)
- **Phase 3 after** because dataset conversion needs real generation records to test against
- **Phase 4 integration** once all components proven to work independently
- **Phase 5 quality tuning** once basic pipeline generates data (need data to calibrate against)
- **Phase 6 optimization** is polish on top of working system

This ordering front-loads the highest risks (smolagents integration, LLM adapter replacement) and defers optimization until the pipeline proves viable.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 5 (Quality + Calibration):** Complex reward function design, threshold calibration methodology needs experimentation
- **Phase 6 (Diversity + Optimization):** Diversity metrics selection, best-of-N selection strategies, DPO pair construction

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Config loading, dataclass design, regex parsing are well-understood
- **Phase 2 (LLM + Agent):** smolagents API is documented, model adapter pattern is standard
- **Phase 3 (Dataset + Hub):** TRL format is specified, datasets library push_to_hub is canonical
- **Phase 4 (Pipeline + Batch):** Orchestration, checkpointing, dedup are solved problems

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | smolagents v1.24.0 is official HF library, actively maintained. TRL format is documented. pronouncing is stable. All recommendations from official sources. |
| Features | **HIGH** | RLVR literature (DRIVE, DEPO, RLVRR papers) provides clear guidance on data quality requirements. VeRL parquet schema is de facto standard. |
| Architecture | **HIGH** | smolagents built-in model classes eliminate need for custom adapter. Config injection via factory function is proven pattern. Component boundaries are clean. |
| Pitfalls | **HIGH** | Pitfalls drawn from smolagents GitHub issues, RLVR research papers, analysis of existing parodies2026/ codebase. All have documented prevention strategies. |

**Overall confidence:** HIGH

### Gaps to Address

- **Phonetic scoring algorithm selection:** word_phone.py and parody_suggestions.py use different similarity algorithms. Need to reconcile and choose canonical version. Validate against all 100 known examples to establish empirical threshold.

- **smolagents streaming behavior:** Research indicates stream_outputs=False required for CodeAgent (issue #1872), but this needs validation with Cerebras backend specifically.

- **g2p fallback implementation:** CMU dictionary OOV handling is critical but no specific library recommendation emerged. Need to evaluate g2p_en vs other options during Phase 5.

- **VeRL vs TRL format compatibility:** Research found both VeRL parquet schema and TRL GRPO format. Need to validate these are compatible or choose one definitively during Phase 3.

- **Human example optimal count:** Research suggests 10-20 examples in few-shot prompt, but optimal number for this specific task needs empirical validation.

## Recommended Stack

| Library | Version | Role | Rationale |
|---------|---------|------|-----------|
| smolagents | >=1.24.0 | Agent orchestration | HuggingFace official lightweight agent framework, CodeAgent support, swappable backends |
| datasets | >=4.5.0 | Dataset creation & upload | Native push_to_hub, Parquet serialization, Dataset Viewer support |
| huggingface-hub | >=1.3.5 | Hub authentication & API | Handles login, token management, push/pull |
| trl | >=0.27.0 | RLVR format reference | Defines GRPOTrainer dataset contract (format spec only, not run) |
| pronouncing | >=0.2.0 | Phonetic analysis | CMU Pronouncing Dictionary interface, provides phones_for_word(), rhymes(), stresses() |
| litellm | >=1.55.0 | Multi-provider LLM routing | 100+ providers including Cerebras, OpenAI, Anthropic |
| openai | >=1.50.0 | OpenAI-compatible API client | For OpenAI or OpenAI-compatible endpoints (Cerebras) |
| python-dotenv | >=1.0.0 | Environment variable management | Load .env files for API keys |

**Python version:** >=3.10 (for match/case and modern typing)

**NOT needed:** torch, transformers, accelerate, vllm (this is data generation, not training)

## Architecture

### Component Structure

```
Config Layer (external files)
    funny_words.json, preferences.json, human_examples.csv
    -> load_config() -> AppConfig (frozen dataclass)

LLM Adapter Layer
    create_model(backend_string) -> smolagents Model
    (InferenceClientModel or LiteLLMModel, no custom adapter)

Agent Orchestration
    CodeAgent(model, tools=[word_phone_tool], max_steps=10-15)
    -> multi-step reasoning with tool calls

Prompt Builder
    PromptBuilder.build(title, suggestions, num_examples=15)
    -> fully-constructed prompt with examples + preferences + suggestions

Output Parser
    OutputParser.parse(raw_output) -> GenerationRecord (dataclass)
    -> extracts thinking, tool calls, attempts, final parody

Dataset Converter
    convert_to_grpo_format(records) -> List[dict]
    -> TRL-compatible prompt-only format + metadata

Pipeline Orchestrator
    CSV -> load_config -> for each title: suggest -> prompt -> agent -> parse -> dataset -> Hub
```

### Data Flow

```
input.csv (titles)
    + funny_words.json
    + preferences.json
    + human_examples.csv
    |
    v
load_config() -> AppConfig
    |
    v
create_model(config.llm_backend) -> smolagents Model
    |
    v
FOR EACH TITLE:
    |
    +-> pre_compute_suggestions(title, funny_words) -> {word: [suggestions]}
    |
    +-> PromptBuilder.build(title, suggestions, examples) -> prompt
    |
    +-> CodeAgent.run(prompt) -> raw_output
    |
    +-> OutputParser.parse(raw_output) -> GenerationRecord
    |
    v
List[GenerationRecord]
    |
    v
convert_to_grpo_format() -> List[dict]
    |
    v
Dataset.from_list() -> push_to_hub()
```

## Key Features for v1

### Table Stakes (non-negotiable)

1. **Deterministic reproducibility** — Seed management, environment pinning, input versioning
2. **VeRL-compatible output format** — Parquet schema with prompt, ability, reward_model, extra_info
3. **Structured reasoning trace capture** — Full multi-step agent traces with tool calls
4. **Deduplication** — Input, output, cross-batch with near-duplicate detection
5. **Quality-gated auto-labeling** — Threshold-based filtering with quality labels
6. **Batch processing with resume** — Checkpoint after each title, resume from checkpoint
7. **Logging and audit trail** — Run ID, config checksums, per-title scores, summary stats

### Differentiators (competitive advantage)

1. **Difficulty-aware data selection** — Track pass@k per title, prioritize medium-difficulty (30-70% success)
2. **Composite verifiable reward** — Decompose into phonetic_validity, phonetic_quality, structural_fidelity, tool_usage, reasoning_quality
3. **Multi-generation with best-of-N** — Generate N candidates, select top-2, retain rejected for DPO
4. **Provenance-rich records** — Full metadata for reproducibility (run_id, model, config_hashes, timestamps)
5. **Negative example generation** — Ablated, perturbation, low-threshold for DPO pairs
6. **Diversity enforcement** — Track usage categories, measure with Self-BLEU

## Critical Pitfalls

### Top 5 Things That Will Bite Us

1. **Reward hacking via phonetic gaming** — Model exploits scoring instead of being genuinely funny. Prevention: composite rewards, diversity penalties, entropy tracking.

2. **Diversity collapse in training data** — GRPO concentrates on narrow outputs, loses creativity. Prevention: forward-KL divergence, difficulty curriculum, Pass@k monitoring.

3. **smolagents code parsing failures** — Non-OpenAI models produce unparsable output. Prevention: stream_outputs=False, output sanitization, max_steps=10-15.

4. **Verifier false positives/negatives** — Phonetic scoring systematically wrong. Prevention: calibrate against known100.csv, track false rates, expand custom_phones.

5. **RLVR trains speed not capability** — Only compresses search, doesn't expand what model can do. Prevention: Pass@k baseline, sweet spot difficulty (30-70%), SFT first for new skills.

## Build Order

### Recommended Sequence

**Phase 1: Foundation** (no LLM calls, no network)
- config.py + AppConfig + load_config()
- parser.py + GenerationRecord + extraction functions
- prompts/system.py + prompts/templates.py
- Tests with fixture data from parodies2026/

**Phase 2: LLM + Agent** (requires API key)
- llm.py + create_model() factory
- prompt.py + PromptBuilder
- agent.py + run_agent()
- Smoke test: single title generation

**Phase 3: Dataset + Hub** (requires HF token)
- dataset.py + format converters (GRPO, DPO, SFT)
- Hub push using datasets library
- Integration test: push small test dataset

**Phase 4: Pipeline + CLI** (integration)
- pipeline.py + orchestration loop
- cli.py + argparse
- End-to-end test: CSV -> generation -> RLVR dataset -> Hub
- pyproject.toml + packaging

**Phase 5: Quality + Calibration**
- Composite reward function
- Human example scoring and threshold calibration
- Difficulty tracking (pass@k)

**Phase 6: Diversity + Optimization**
- Multi-generation with best-of-N
- Diversity metrics and enforcement
- Negative example generation for DPO

## Sources

### Primary (HIGH confidence)
- [smolagents Documentation](https://huggingface.co/docs/smolagents/index) — official HF docs, v1.24.0
- [smolagents Models Reference](https://huggingface.co/docs/smolagents/en/reference/models) — built-in model classes
- [smolagents Tools Tutorial](https://huggingface.co/docs/smolagents/en/tutorials/tools) — @tool decorator, Tool subclass
- [TRL Dataset Formats](https://huggingface.co/docs/trl/main/en/dataset_formats) — prompt-only, preference, SFT
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — RLVR via GRPO
- [datasets Upload Guide](https://huggingface.co/docs/datasets/en/upload_dataset) — push_to_hub patterns
- [pronouncing Library](https://pronouncing.readthedocs.io/en/latest/) — CMU dict interface

### Secondary (MEDIUM confidence)
- [RLVRR: From Verifiable Dot to Reward Chain](https://arxiv.org/abs/2601.18533) — decomposed rewards
- [DRIVE: Data Curation Best Practices for RLVR](https://arxiv.org/abs/2511.06307) — difficulty selection, dedup
- [DEPO: High Data Efficiency in RLVR](https://arxiv.org/abs/2509.01321) — medium-difficulty examples
- [One-Shot RLVR](https://arxiv.org/abs/2504.20571) — few-shot examples effectiveness
- [Precision over Diversity](https://arxiv.org/abs/2601.04954) — rule-based rewards outperform LLM-judge
- [Reward Hacking Mitigation](https://arxiv.org/abs/2509.15557) — composite rewards
- [RLVR Noisy Rewards](https://arxiv.org/abs/2510.00915) — verifier imperfection, 38% false negatives
- [DPH-RL: Diversity Collapse](https://arxiv.org/abs/2509.07430) — forward-KL vs reverse-KL

### Tertiary (LOW confidence, needs validation)
- [VeRL: Prepare Data](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) — VeRL parquet schema
- [smolagents Issue #1872](https://github.com/huggingface/smolagents/issues/1872) — streaming breaks CodeAgent
- [smolagents Issue #322](https://github.com/huggingface/smolagents/issues/322) — capturing full thinking
- [Promptfoo: RLVR Explained](https://www.promptfoo.dev/blog/rlvr-explained/) — "search compression" insight

---
*Research completed: 2026-01-31*
*Ready for roadmap: yes*

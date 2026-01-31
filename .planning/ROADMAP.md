# Roadmap: chucklesPRIME

## Overview

chucklesPRIME restructures the existing parodies2026/ codebase into a clean Python pipeline that generates phonetically sound parody titles with reasoning traces, then converts them into GRPO and DPO datasets for RLVR fine-tuning. The roadmap progresses from project skeleton and config loading, through the generation engine, to dataset conversion and Hub push, finishing with a unified CLI that orchestrates everything end-to-end.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Clean package structure, external config loading, and LLM adapter
- [x] **Phase 2: Generation Engine** - Parody generation with phonetic tools and reasoning trace capture
- [x] **Phase 3: Dataset Conversion** - GRPO/DPO format conversion, composite rewards, and Hub push
- [x] **Phase 4: Pipeline CLI** - End-to-end batch processing via single CLI entry point

## Phase Details

### Phase 1: Foundation
**Goal**: Users can install the package, load all external configuration, and connect to any OpenAI-compatible LLM backend
**Depends on**: Nothing (first phase)
**Requirements**: PROJ-01, CFG-01, CFG-02, CFG-03, CFG-04, LLM-01, LLM-02
**Success Criteria** (what must be TRUE):
  1. Running `pip install -e .` installs chucklesPRIME as a working Python package with all dependencies
  2. A settings file points to external funny_words.json, preferences.json, and human_examples.csv -- and load_config() returns a populated AppConfig with all data loaded
  3. Human examples CSV (~1,234 rows) loads with formatting issues cleaned (no broken fields, no encoding errors, consistent columns)
  4. create_model() connects to a Cerebras (or other OpenAI-compatible) endpoint and returns a working smolagents model that can complete a simple prompt
**Plans**: 2 plans in 2 waves

Plans:
- [x] 01-01-PLAN.md -- Package skeleton (pyproject.toml, src layout) + AppConfig + load_config() + CSV cleaner
- [x] 01-02-PLAN.md -- OpenAICompatibleModel adapter (smolagents.Model subclass) + create_model() factory

### Phase 2: Generation Engine
**Goal**: Users can feed a title and get back 2 phonetically sound parody candidates with full reasoning traces
**Depends on**: Phase 1
**Requirements**: GEN-01, GEN-02, GEN-03, GEN-04
**Success Criteria** (what must be TRUE):
  1. Given a CSV of input titles, the generation engine reads and processes each title
  2. For each title, the smolagents CodeAgent produces 2 top parody candidates using WordPhoneTool and ParodyWordSuggestionTool loaded from HF Hub
  3. Full reasoning traces (agent thinking steps, tool calls with arguments and results, intermediate attempts) are captured as structured data per generation
  4. Generation output is a list of structured GenerationRecord objects ready for downstream conversion
**Plans**: 2 plans in 2 waves

Plans:
- [x] 02-01-PLAN.md -- Data types (GenerationRecord, ParodyCandidate, AgentTrace) + HF Hub tool loading + prompt builder
- [x] 02-02-PLAN.md -- Core generation engine (CodeAgent orchestration, CSV reading, batch processing with error isolation)

### Phase 3: Dataset Conversion
**Goal**: Generation records are converted to training-ready GRPO and DPO datasets with composite reward signals and pushed to HuggingFace Hub
**Depends on**: Phase 2
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. GRPO dataset contains prompt-only records in TRL conversational format with metadata columns (original_title, phonetic_scores, generation_model) preserved for reward functions
  2. DPO dataset pairs human parodies (from cleaned examples) as "chosen" against model inferior outputs as "rejected", in TRL preference format
  3. Composite reward signals include phonetic quality, tool usage completeness, and structure preservation as continuous scores (not binary thresholds)
  4. Full reasoning traces are archived as JSONL with one record per generation
  5. Both GRPO and DPO datasets push successfully to HuggingFace Hub and appear in the Dataset Viewer
**Plans**: 2 plans in 2 waves

Plans:
- [x] 03-01-PLAN.md -- Composite reward functions (phonetic quality, tool usage, structure preservation) + JSONL trace archival + dependency update
- [x] 03-02-PLAN.md -- GRPO and DPO format converters with Hub push

### Phase 4: Pipeline CLI
**Goal**: Users run a single command to process a CSV of titles through generation, conversion, and Hub push
**Depends on**: Phase 3
**Requirements**: PROJ-02
**Success Criteria** (what must be TRUE):
  1. A CLI command (e.g., `chuckles generate input.csv`) reads titles, runs the generation engine, converts to both dataset formats, and pushes to Hub -- all in one invocation
  2. A CLI command for dataset-only conversion (e.g., `chuckles convert`) takes existing generation output and produces/pushes datasets without re-running generation
  3. CLI provides clear progress output showing which title is being processed and summary statistics on completion
**Plans**: 1 plan in 1 wave

Plans:
- [x] 04-01-PLAN.md -- Pipeline CLI entry point (argparse with generate/convert subcommands, rich progress, JSONL deserialization, summary table)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-01-31 |
| 2. Generation Engine | 2/2 | Complete | 2026-01-31 |
| 3. Dataset Conversion | 2/2 | Complete | 2026-01-31 |
| 4. Pipeline CLI | 1/1 | Complete | 2026-01-31 |

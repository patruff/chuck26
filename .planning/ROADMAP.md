# Roadmap: chucklesPRIME

## Milestones

- **v1.0 Data Generation Pipeline** - Phases 1-4 (shipped 2026-01-31)
- **v1.1 Fine-tuning & Inference Loop** - Phases 5-7 (in progress)

## Phases

<details>
<summary>v1.0 Data Generation Pipeline (Phases 1-4) - SHIPPED 2026-01-31</summary>

### Phase 1: Foundation
**Goal**: Users can install the package, load all external configuration, and connect to any OpenAI-compatible LLM backend
**Depends on**: Nothing (first phase)
**Requirements**: PROJ-01, CFG-01, CFG-02, CFG-03, CFG-04, LLM-01, LLM-02
**Success Criteria** (what must be TRUE):
  1. Running `pip install -e .` installs chucklesPRIME as a working Python package with all dependencies
  2. A settings file points to external funny_words.json, preferences.json, and human_examples.csv -- and load_config() returns a populated AppConfig with all data loaded
  3. Human examples CSV (~1,234 rows) loads with formatting issues cleaned (no broken fields, no encoding errors, consistent columns)
  4. create_model() connects to a Cerebras (or other OpenAI-compatible) endpoint and returns a working smolagents model that can complete a simple prompt
**Plans**: 2 plans

Plans:
- [x] 01-01: Package skeleton + AppConfig + load_config() + CSV cleaner
- [x] 01-02: OpenAICompatibleModel adapter + create_model() factory

### Phase 2: Generation Engine
**Goal**: Users can feed a title and get back 2 phonetically sound parody candidates with full reasoning traces
**Depends on**: Phase 1
**Requirements**: GEN-01, GEN-02, GEN-03, GEN-04
**Success Criteria** (what must be TRUE):
  1. Given a CSV of input titles, the generation engine reads and processes each title
  2. For each title, the smolagents CodeAgent produces 2 top parody candidates using WordPhoneTool and ParodyWordSuggestionTool loaded from HF Hub
  3. Full reasoning traces (agent thinking steps, tool calls with arguments and results, intermediate attempts) are captured as structured data per generation
  4. Generation output is a list of structured GenerationRecord objects ready for downstream conversion
**Plans**: 2 plans

Plans:
- [x] 02-01: Data types + HF Hub tool loading + prompt builder
- [x] 02-02: Core generation engine (CodeAgent orchestration, CSV reading, batch processing)

### Phase 3: Dataset Conversion
**Goal**: Generation records are converted to training-ready GRPO and DPO datasets with composite reward signals and pushed to HuggingFace Hub
**Depends on**: Phase 2
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. GRPO dataset contains prompt-only records in TRL conversational format with metadata columns preserved for reward functions
  2. DPO dataset pairs human parodies as "chosen" against model inferior outputs as "rejected", in TRL preference format
  3. Composite reward signals include phonetic quality, tool usage completeness, and structure preservation as continuous scores
  4. Full reasoning traces are archived as JSONL with one record per generation
  5. Both GRPO and DPO datasets push successfully to HuggingFace Hub
**Plans**: 2 plans

Plans:
- [x] 03-01: Composite reward functions + JSONL trace archival
- [x] 03-02: GRPO and DPO format converters with Hub push

### Phase 4: Pipeline CLI
**Goal**: Users run a single command to process a CSV of titles through generation, conversion, and Hub push
**Depends on**: Phase 3
**Requirements**: PROJ-02
**Success Criteria** (what must be TRUE):
  1. `chuckles generate input.csv` reads titles, runs generation, converts to both formats, and pushes to Hub
  2. `chuckles convert` takes existing generation output and produces/pushes datasets without re-running generation
  3. CLI provides clear progress output and summary statistics on completion
**Plans**: 1 plan

Plans:
- [x] 04-01: Pipeline CLI entry point (argparse with generate/convert subcommands)

</details>

### v1.1 Fine-tuning & Inference Loop (In Progress)

**Milestone Goal:** Train Qwen3-32B on generated datasets, serve the fine-tuned model, and close the improvement loop so the CLI generates better parodies with the trained model.

- [x] **Phase 5: DPO Training & Model Export** - Train DPO on Qwen3-32B, merge LoRA, push to Hub
- [ ] **Phase 6: Inference & Loop Validation** - Serve fine-tuned model, validate CLI compatibility, close the loop
- [ ] **Phase 7: GRPO Training Pipeline** - Train with custom phonetic reward functions for refinement

## Phase Details

### Phase 5: DPO Training & Model Export
**Goal**: A DPO-fine-tuned Qwen3-32B model merged and published on HuggingFace Hub, ready to serve
**Depends on**: Phase 4 (v1.0 complete -- datasets exist on Hub)
**Requirements**: DPO-01, DPO-02, DPO-03, DPO-04, DPO-05, EXP-01, EXP-02, EXP-03, EXP-04
**Success Criteria** (what must be TRUE):
  1. User can run a standalone DPO training script on RunPod that loads Qwen3-32B in 4-bit QLoRA and trains on the existing Hub dataset without errors
  2. Training checkpoints are saved to RunPod network volume and the LoRA adapter is pushed to HuggingFace Hub
  3. User can run a merge script that produces a 16-bit merged model and pushes it to HuggingFace Hub
  4. Merged model produces outputs comparable to adapter-loaded model on test prompts (no merge degradation)
  5. Qwen3 chat template is preserved -- model responds in correct conversational format after fine-tuning
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md -- RunPod setup script + DPO training script (setup_runpod.sh, train_dpo.py)
- [x] 05-02-PLAN.md -- LoRA merge + Hub push + merge validation (merge_and_push.py, validate_merge.py)

### Phase 6: Inference & Loop Validation
**Goal**: Fine-tuned model served via vLLM and accessible through existing CLI, completing the generate-train-serve loop
**Depends on**: Phase 5 (merged model on Hub)
**Requirements**: INF-01, INF-02, INF-03, LOOP-01, LOOP-02
**Success Criteria** (what must be TRUE):
  1. User can launch a vLLM server on RunPod that serves the fine-tuned model with an OpenAI-compatible API
  2. Existing `chuckles generate` CLI works with the vLLM endpoint by changing only settings.json (zero code changes)
  3. Parodies generated by fine-tuned model show measurable quality improvement over base Qwen3-32B on the same inputs
  4. User has a documented end-to-end workflow: generate data, train DPO, merge model, serve via vLLM, generate better data
**Plans**: 2 plans

Plans:
- [ ] 06-01-PLAN.md -- vLLM inference launch script + AWQ quantization (setup_inference.sh, quantize_awq.py)
- [ ] 06-02-PLAN.md -- Inference quality validation + end-to-end loop verification (validate_inference.py, human verify)

### Phase 7: GRPO Training Pipeline
**Goal**: GRPO training with custom phonetic reward functions refines the model beyond what DPO achieves
**Depends on**: Phase 5 (validated training environment), Phase 6 (inference serving works)
**Requirements**: GRPO-01, GRPO-02, GRPO-03, GRPO-04, GRPO-05
**Success Criteria** (what must be TRUE):
  1. User can run a standalone GRPO training script on RunPod that uses existing phonetic/structure/tool reward functions adapted to TRL's interface
  2. Multiple reward functions run concurrently with configurable weights during training
  3. GRPO-trained model produces parodies with higher phonetic reward scores than the DPO-only model on the same test inputs
  4. User can swap in the GRPO model on vLLM and generate parodies through the existing CLI
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 5 -> 6 -> 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 2/2 | Complete | 2026-01-31 |
| 2. Generation Engine | v1.0 | 2/2 | Complete | 2026-01-31 |
| 3. Dataset Conversion | v1.0 | 2/2 | Complete | 2026-01-31 |
| 4. Pipeline CLI | v1.0 | 1/1 | Complete | 2026-01-31 |
| 5. DPO Training & Model Export | v1.1 | 2/2 | Complete | 2026-01-31 |
| 6. Inference & Loop Validation | v1.1 | 0/2 | Planned | - |
| 7. GRPO Training Pipeline | v1.1 | 0/? | Not started | - |

---
*Roadmap created: 2026-01-31*
*Last updated: 2026-02-01*

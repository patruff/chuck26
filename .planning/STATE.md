# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Generate quality reasoning data about phonetic parodies, in formats ready for GRPO/DPO fine-tuning, then close the loop by training and deploying the model.
**Current focus:** Phase 5 - DPO Training & Model Export

## Current Position

Phase: 5 of 7 (DPO Training & Model Export)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-01-31 - Completed 05-01-PLAN.md (RunPod setup + DPO training script)

Progress: [========..] 80% (v1.0 complete, 05-01 done, 05-02 + phases 6-7 ahead)

## Performance Metrics

**Velocity:**
- Total plans completed: 8 (7 v1.0 + 1 v1.1)
- Average duration: ~3 min (05-01)
- Total execution time: -

**By Phase (v1.0 -- archived):**

| Phase | Plans | Status |
|-------|-------|--------|
| 1. Foundation | 2/2 | Complete |
| 2. Generation Engine | 2/2 | Complete |
| 3. Dataset Conversion | 2/2 | Complete |
| 4. Pipeline CLI | 1/1 | Complete |

**By Phase (v1.1 -- active):**

| Phase | Plans | Status |
|-------|-------|--------|
| 5. DPO Training & Model Export | 1/2 | In progress |
| 6. GRPO Training | 0/? | Not started |
| 7. vLLM Inference | 0/? | Not started |

**Recent Trend:**
- 05-01 executed cleanly, no deviations
- Trend: Stable

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1]: Standalone training scripts in `training/` (not part of CLI package)
- [v1.1]: Unsloth for QLoRA (2x faster, 70% less VRAM)
- [v1.1]: vLLM for inference serving (OpenAI-compatible, works with existing adapter)
- [v1.1]: DPO before GRPO (simpler method validates pipeline first)
- [v1.1]: Merge to 16-bit only (4-bit merge degrades quality)
- [05-01]: Install Unsloth before TRL (controls dependency versions via monkey-patching)
- [05-01]: Save LoRA adapter to Hub before merge step (safety backup against merge bugs)
- [05-01]: Verify chat template at training start (catch Qwen3 thinking-mode mismatch early)

### Pending Todos

None.

### Blockers/Concerns

- GRPO reward function effectiveness as training signals is unproven (mitigated by DPO-first approach)
- RunPod container storage is ephemeral -- addressed in setup_runpod.sh and train_dpo.py (all paths use /workspace/)
- Qwen3 chat template must match exactly -- addressed with enable_thinking=False verification in train_dpo.py

## Session Continuity

Last session: 2026-01-31
Stopped at: Completed 05-01-PLAN.md
Resume file: None

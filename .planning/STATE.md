# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Generate quality reasoning data about phonetic parodies, in formats ready for GRPO/DPO fine-tuning, then close the loop by training and deploying the model.
**Current focus:** Phase 6 - Inference & Loop Validation (Phase 5 verified complete)

## Current Position

Phase: 6 of 7 (Inference & Loop Validation)
Plan: 0 of ? in current phase
Status: Not started (Phase 5 verified complete)
Last activity: 2026-01-31 - Phase 5 verified (10/10 must-haves, 9/9 requirements)

Progress: [=========.] 90% (v1.0 complete, Phase 5 verified, phases 6-7 ahead)

## Performance Metrics

**Velocity:**
- Total plans completed: 9 (7 v1.0 + 2 v1.1)
- Average duration: ~4 min (05-02)
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
| 5. DPO Training & Model Export | 2/2 | Verified Complete |
| 6. Inference & Loop Validation | 0/? | Not started |
| 7. GRPO Training Pipeline | 0/? | Not started |

**Recent Trend:**
- 05-01 executed cleanly, no deviations
- 05-02 executed cleanly, no deviations
- Phase 5 verification: PASS (10/10 must-haves, 9/9 requirements)
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
- [05-02]: Save merged model locally before Hub push (safety net for known bug #3146)
- [05-02]: Include fallback HfApi.upload_folder when push_to_hub_merged fails
- [05-02]: Validate quality by non-empty parody content, not exact output match (quantization differences expected)

### Pending Todos

None.

### Blockers/Concerns

- GRPO reward function effectiveness as training signals is unproven (mitigated by DPO-first approach)
- RunPod container storage is ephemeral -- addressed in setup_runpod.sh and all training scripts (all paths use /workspace/)
- Qwen3 chat template must match exactly -- addressed with enable_thinking=False in all scripts
- Unsloth push_to_hub_merged bug #3146 -- mitigated with local save + fallback upload in merge_and_push.py

## Session Continuity

Last session: 2026-01-31
Stopped at: Phase 5 verified complete, ready for Phase 6
Resume file: None

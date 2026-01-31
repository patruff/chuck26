---
phase: 05-dpo-training-model-export
plan: 02
subsystem: training
tags: [lora-merge, fp16, huggingface-hub, model-export, validation, unsloth, qwen3]

dependency-graph:
  requires:
    - "05-01: train_dpo.py produces the LoRA adapter that merge_and_push.py consumes"
  provides:
    - "training/merge_and_push.py -- LoRA merge to FP16 and Hub push with safety checks"
    - "training/validate_merge.py -- Adapter vs merged model quality comparison"
  affects:
    - "06-*: vLLM inference serving loads the merged model from Hub (patruff/chuckles-qwen3-32b-dpo)"

tech-stack:
  added: []
  patterns:
    - "Local save before Hub push as safety net (known push_to_hub_merged bug #3146)"
    - "Fallback upload via HfApi.upload_folder when push_to_hub_merged fails"
    - "Merged 16-bit only (never 4-bit merge) -- compounding quantization avoidance"
    - "Adapter vs merged model side-by-side comparison on identical prompts"
    - "enable_thinking=False for all Qwen3 chat template applications"

key-files:
  created:
    - "training/merge_and_push.py"
    - "training/validate_merge.py"
  modified: []

decisions:
  - id: "merge-local-first"
    decision: "Save merged model locally before pushing to Hub"
    rationale: "Known Unsloth bug #3146 can cause push_to_hub_merged to only push README. Local save acts as safety net."
  - id: "merge-fallback-upload"
    decision: "Include fallback HfApi.upload_folder when push_to_hub_merged fails"
    rationale: "Ensures Hub push succeeds even if Unsloth's integrated push method has issues"
  - id: "validate-quality-not-identity"
    decision: "Validation checks for non-empty parody content, not exact output match"
    rationale: "Adapter model runs in 4-bit QLoRA, merged model runs in FP16 -- minor output differences are expected due to quantization"

metrics:
  duration: "~4 minutes"
  completed: "2026-01-31"
---

# Phase 5 Plan 2: LoRA Merge, Hub Push, and Merge Validation Summary

**One-liner:** LoRA-to-FP16 merge with Hub push (including fallback for bug #3146) and 5-prompt validation comparing adapter-loaded vs merged model quality.

## What Was Built

### training/merge_and_push.py (Task 1)
Standalone script that merges the trained LoRA adapter from DPO training into full FP16 base model weights and pushes to HuggingFace Hub.

Key features:
- **Prerequisite checks**: Verifies adapter path exists, HF_TOKEN is set, and warns if less than 150GB free disk space
- **Local save first**: Saves merged model to `/workspace/merged-model/` before attempting Hub push (safety net)
- **File verification**: Confirms safetensors files exist in output directory after merge
- **Hub push with fallback**: Attempts `push_to_hub_merged` first; if it fails (known bug #3146), falls back to `HfApi.upload_folder`
- **Clear documentation**: Module docstring covers prerequisites, disk requirements (~130GB), and expected runtime (30-60 min)

### training/validate_merge.py (Task 2)
Standalone script that validates the merged model produces comparable outputs to the adapter-loaded model on identical test prompts.

Key features:
- **5 test prompts** covering different movie titles (Shawshank Redemption, Pulp Fiction, The Godfather, Jurassic Park, Fight Club)
- **Two-phase generation**: First generates with adapter-loaded model, then with merged model, using identical parameters
- **Side-by-side comparison**: Prints both outputs for each prompt for visual inspection
- **Automated quality checks**: Verifies outputs are non-empty, not just echoed prompts, and contain actual parody content
- **PASS/FAIL summary**: Reports overall result based on merged model output quality
- **Flexible model source**: `USE_HUB_MERGED` flag switches between local and Hub merged model

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Save locally before Hub push | Known Unsloth bug #3146 can push only README. Local save is insurance. |
| Fallback to HfApi.upload_folder | Ensures Hub push succeeds regardless of push_to_hub_merged reliability |
| Quality check = non-empty + parody content | Adapter (4-bit) vs merged (FP16) outputs differ due to quantization; exact match is wrong metric |
| 5 diverse test prompts | Covers short/long titles, multi-word titles, single-word titles for varied testing |

## Deviations from Plan

None -- plan executed exactly as written.

## Commit Log

| Task | Commit | Message |
|------|--------|---------|
| 1 | 3d8beae | feat(05-02): create LoRA merge and Hub push script |
| 2 | 071d48f | feat(05-02): create merge validation script |

## Verification Results

All 9 verification checks passed:

1. merge_and_push.py syntax: PASS
2. validate_merge.py syntax: PASS
3. merged_16bit present, no merged_4bit: PASS
4. Local save before Hub push: PASS
5. Fallback upload logic present: PASS
6. 5+ test prompts: PASS
7. enable_thinking=False used: PASS
8. Both scripts use /workspace/ paths: PASS
9. No chuckles_prime imports: PASS

## Next Phase Readiness

Phase 5 is now complete with all 4 standalone scripts:
- `training/setup_runpod.sh` -- Environment setup (Plan 01)
- `training/train_dpo.py` -- DPO training (Plan 01)
- `training/merge_and_push.py` -- LoRA merge + Hub push (Plan 02)
- `training/validate_merge.py` -- Merge validation (Plan 02)

**Ready for Phase 6** (Inference & Loop Validation) once the user has:
1. Run train_dpo.py on RunPod to produce the adapter
2. Run merge_and_push.py to merge and push to Hub
3. Run validate_merge.py to confirm quality

Phase 6 will consume the merged model at `patruff/chuckles-qwen3-32b-dpo` for vLLM serving.

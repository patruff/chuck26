# Phase 5: DPO Training & Model Export - Research

**Researched:** 2026-01-31
**Domain:** QLoRA DPO fine-tuning, LoRA adapter management, model merging, HuggingFace Hub push
**Confidence:** HIGH

## Summary

Phase 5 produces a DPO-fine-tuned Qwen3-32B model published on HuggingFace Hub. The work involves three standalone scripts in `training/`: a DPO training script, a merge script, and a validation script. The core stack is Unsloth (QLoRA loading + memory optimization) + TRL DPOTrainer (training loop) running on a RunPod A6000 48GB GPU. Datasets already exist on Hub in the exact format TRL expects (conversational preference format with `prompt`, `chosen`, `rejected` columns).

The most critical technical decisions are: (1) using `FastModel` (not the older `FastLanguageModel`) for Qwen3-32B loading, (2) passing `ref_model=None` to DPOTrainer so it uses PEFT adapter toggling for reference logits (no extra VRAM), (3) merging to 16-bit only via `save_method="merged_16bit"` (never 4-bit merge), and (4) preserving the Qwen3 chat template by setting `enable_thinking=False` consistently. A known bug in Unsloth's `save_pretrained_merged` re-downloads FP16 weights into the output directory rather than using the HF cache; the workaround is to pre-populate the cache or accept the extra download time.

All checkpoints and outputs must go to `/workspace/` (RunPod network volume) to survive pod restarts. The LoRA adapter (~100-300MB) should be pushed to Hub immediately after training, before the merge step, as a safety backup.

**Primary recommendation:** Build three standalone scripts -- `train_dpo.py`, `merge_and_push.py`, `validate_merge.py` -- each idempotent and copy-pasteable to RunPod with zero local dependencies beyond the training stack.

## Standard Stack

The established libraries for this phase, verified from existing milestone research (STACK.md) and confirmed against current official docs.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| unsloth | >= 2026.1.4 | QLoRA model loading, LoRA attachment, merged model saving | 2x faster training, 70% less VRAM. Provides `FastModel.from_pretrained()` for 4-bit loading and `save_pretrained_merged(save_method="merged_16bit")` for vLLM-compatible export. Only viable way to QLoRA-train 32B on a single A6000. |
| trl | >= 0.27.1 | DPOTrainer, DPOConfig | Official HuggingFace DPO implementation. Supports conversational preference datasets, automatic chat template application, PEFT reference model optimization. |
| transformers | >= 5.0.0 | Base model architecture, tokenizer, chat templates | Required by TRL 0.27.x. Qwen3 natively supported. Provides `apply_chat_template()`. |
| bitsandbytes | >= 0.49.1 | 4-bit NF4 quantization backend | Required for QLoRA. NF4 + double quantization. |
| peft | >= 0.18.1 | LoRA adapter management | Required by Unsloth. Provides `merge_and_unload()` and adapter save/load. |
| datasets | >= 3.0.0 | Load DPO dataset from HuggingFace Hub | Already in project. `load_dataset("patruff/chuckles-dpo")` directly. |
| huggingface-hub | >= 0.20.0 | Push adapter and merged model to Hub | Already in project. Used for `push_to_hub_merged()` and `model.push_to_hub()`. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| accelerate | >= 1.12.0 | Training orchestration, device placement | Required by TRL. Use `accelerate launch` for the training script. |
| wandb | >= 0.19.0 | Training metrics logging | Optional but recommended. `report_to="wandb"` in DPOConfig. Track reward margins, loss curves. |
| safetensors | >= 0.5.0 | Safe model serialization | Dependency of transformers. Used automatically when saving merged model. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Unsloth `FastModel` | Standard PEFT + bitsandbytes | 2x slower, 70% more VRAM. 32B QLoRA would still fit A6000 but with no headroom. |
| `save_method="merged_16bit"` | `save_method="merged_4bit"` | 4-bit merge degrades quality. Never use for deployment. Decision locked. |
| `save_method="merged_16bit"` | `save_method="mxfp4"` | MXFP4 is 75% smaller on disk and merges 5-10x faster, but requires MXFP4-compatible inference engine. vLLM FP16 loading is the standard path. |
| Push merged model to Hub | Save LoRA adapter only | vLLM adapter loading is more complex and adds latency. Merge-then-push is the standard pattern. Keep adapter as backup. |

**Installation (RunPod):**
```bash
# CRITICAL: Install Unsloth FIRST -- it resolves torch/transformers/peft/bitsandbytes versions
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Then TRL (DPOTrainer)
pip install "trl>=0.27.1"

# Optional: experiment tracking
pip install wandb>=0.19.0

# Login to HuggingFace Hub
huggingface-cli login --token $HF_TOKEN
```

## Architecture Patterns

### Recommended File Structure for Phase 5

```
training/
  train_dpo.py          # Standalone DPO training script (DPO-05)
  merge_and_push.py     # Merge LoRA -> 16-bit, push to Hub (EXP-01, EXP-02, EXP-03)
  validate_merge.py     # Compare adapter vs merged model outputs (EXP-04)
  setup_runpod.sh       # One-shot environment setup
```

Each script is standalone and copy-pasteable to RunPod. No imports from the `chuckles_prime` package (DPO training does not need reward functions -- those are GRPO-only).

### Pattern 1: Unsloth QLoRA Model Loading

**What:** Load Qwen3-32B in 4-bit NF4 quantization with LoRA adapters attached.
**When:** Start of every training or merge script.
**Key detail:** Use `FastModel` (not the older `FastLanguageModel`). For dense Qwen3-32B (non-MOE), either works, but `FastModel` is the current API and required for MOE models.

```python
# Source: Unsloth Qwen3 docs + GitHub Issue #2329
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",  # Pre-quantized 4-bit
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    r=32,                    # LoRA rank -- 32 is good balance of quality/VRAM
    target_modules=[         # All linear layers for maximum quality
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,           # alpha == r is a common starting point
    lora_dropout=0,          # Unsloth recommends 0 dropout
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
)
```

### Pattern 2: DPO Training with TRL

**What:** Train the model using preference pairs from Hub dataset.
**When:** Main training loop in `train_dpo.py`.
**Key detail:** `ref_model=None` (default) triggers PEFT-aware reference model -- DPOTrainer disables the LoRA adapter to compute reference logits, using zero extra VRAM. The `processing_class=tokenizer` parameter is the current TRL API (replaces older `tokenizer=` kwarg).

```python
# Source: TRL DPOTrainer docs (https://huggingface.co/docs/trl/main/en/dpo_trainer)
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("patruff/chuckles-dpo", split="train")

training_args = DPOConfig(
    output_dir="/workspace/dpo-output",    # Network volume!
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,         # Effective batch size = 8
    learning_rate=5e-6,
    beta=0.1,                              # DPO temperature (default 0.1)
    bf16=True,                             # Required for Unsloth
    logging_steps=10,
    save_steps=100,                        # Checkpoint every 100 steps
    save_total_limit=3,                    # Keep only last 3 checkpoints
    max_length=2048,                       # Max total length (prompt + response)
    max_prompt_length=512,                 # Max prompt portion
    loss_type="sigmoid",                   # Standard DPO loss (default)
    report_to="wandb",                     # Optional: remove if no W&B
    warmup_ratio=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                        # PEFT-aware: disables adapter for ref logits
    args=training_args,
    processing_class=tokenizer,            # Current TRL API (not `tokenizer=`)
    train_dataset=dataset,
)
trainer.train()
```

### Pattern 3: LoRA Adapter Save to Hub

**What:** Save the trained LoRA adapter (~100-300MB) to Hub before merging.
**When:** Immediately after training completes.
**Why:** Safety backup. If the merge step fails (known Qwen3 merge bugs), the adapter is preserved.

```python
# Source: Unsloth wiki / save methods
# Save adapter locally first
model.save_pretrained_merged(
    "/workspace/dpo-adapter",
    tokenizer,
    save_method="lora",
)

# Push adapter to Hub
model.push_to_hub_merged(
    "patruff/chuckles-qwen3-32b-dpo-adapter",
    tokenizer,
    save_method="lora",
    token=os.environ["HF_TOKEN"],
)
```

### Pattern 4: Merge to 16-bit and Push

**What:** Merge LoRA adapter into FP16 base model weights, push merged model to Hub.
**When:** After adapter is saved and verified. Separate script (`merge_and_push.py`).
**Key detail:** `merged_16bit` downloads the original FP16 base model behind the scenes and merges LoRA into clean weights. Known bug: downloads into output dir's `.cache/` instead of HF cache (issue #3633). Workaround: ensure sufficient disk space (~65GB for FP16 base + ~65GB for merged output = ~130GB needed).

```python
# Source: Unsloth vLLM deployment guide + issue #3633
from unsloth import FastModel
import os

# Load the adapter (from local or Hub)
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# If loading from saved adapter:
# model.load_adapter("/workspace/dpo-adapter")

# Merge to 16-bit and push
# CRITICAL: save_method="merged_16bit" -- never "merged_4bit"
model.push_to_hub_merged(
    "patruff/chuckles-qwen3-32b-dpo",
    tokenizer,
    save_method="merged_16bit",
    token=os.environ["HF_TOKEN"],
)
```

### Pattern 5: Validate Merged Model Quality

**What:** Compare outputs from adapter-loaded model vs merged model on the same test prompts.
**When:** After merge completes, before declaring success.
**Key detail:** Load the merged model separately and run identical prompts. Outputs should be semantically similar (not identical due to quantization, but comparable quality).

```python
# Source: EXP-04 requirement
# Simple validation: generate with both and compare
test_prompts = [
    "Create a phonetically-sound parody of: 'The Shawshank Redemption'",
    "Create a phonetically-sound parody of: 'Pulp Fiction'",
    "Create a phonetically-sound parody of: 'The Godfather'",
]

# Generate with adapter-loaded model
adapter_outputs = []
for prompt in test_prompts:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    output = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7)
    adapter_outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))

# Then load merged model and generate same prompts
# Compare: outputs should be semantically similar
```

### Anti-Patterns to Avoid

- **Merging to 4-bit:** `save_method="merged_4bit"` introduces compounding quantization errors. The LoRA weights were learned to compensate for 4-bit representation; merging back into 4-bit double-quantizes. Always use `merged_16bit`.
- **Saving to container disk:** Anything outside `/workspace/` on RunPod is ephemeral. All `output_dir` paths must start with `/workspace/`.
- **Hardcoding HF tokens:** Use `os.environ["HF_TOKEN"]` or `huggingface-cli login`. Never put tokens in script files.
- **Skipping adapter save before merge:** The merge step has known bugs. Always save the LoRA adapter to Hub first as a backup.
- **Using `FastLanguageModel`:** While it works for dense Qwen3-32B, `FastModel` is the current API. TRL docs show `FastLanguageModel` but that is outdated documentation.

## Don't Hand-Roll

Problems with existing solutions that should NOT be rebuilt.

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DPO loss computation | Custom DPO loss function | `TRL DPOTrainer` | Supports 15+ loss variants (sigmoid, IPO, robust, etc.), automatic reference model handling with PEFT, gradient accumulation, mixed precision. Reimplementing would miss edge cases. |
| QLoRA weight management | Manual bitsandbytes + PEFT setup | `Unsloth FastModel` | Unsloth's Triton kernels optimize memory layout beyond what manual setup achieves. Monkey-patches TRL internals for 2x speedup. |
| Chat template application | Manual `<\|im_start\|>` formatting | `tokenizer.apply_chat_template()` | Qwen3 chat template includes thinking mode control, special token handling. Manual formatting risks mismatch. TRL DPOTrainer auto-applies it to preference datasets. |
| Merged model saving | Manual `merge_and_unload()` + `save_pretrained()` | `Unsloth save_pretrained_merged(save_method="merged_16bit")` | Unsloth handles FP16 base model download, clean merge without 4-bit artifacts, and safe serialization. Manual merge risks degradation. |
| Checkpoint management | Custom checkpoint save/resume logic | `DPOConfig(save_steps=N, resume_from_checkpoint=True)` | TRL Trainer handles optimizer state, scheduler state, RNG state, and gradient scaler. Custom checkpointing misses these. |
| Reference model in DPO | Loading two copies of the model | `ref_model=None` with PEFT | DPOTrainer automatically disables LoRA adapter for reference logits. Zero extra VRAM. Loading two models wastes 26GB+ VRAM. |

**Key insight:** DPO training is effectively a configuration exercise, not a coding exercise. The hard parts are in the libraries. The scripts are mostly configuration + dataset loading + save/push.

## Common Pitfalls

### Pitfall 1: QLoRA Merge to 4-bit Produces Degraded Model

**What goes wrong:** Merged model outputs are identical to base model, ignoring all fine-tuning.
**Why it happens:** Merging LoRA deltas into 4-bit quantized weights introduces compounding quantization errors. LoRA was trained to compensate for 4-bit representation; merging back into 4-bit creates double-quantization artifacts.
**How to avoid:** Always use `save_method="merged_16bit"`. This downloads FP16 base weights and merges LoRA into those clean weights. Test merged model output before pushing.
**Warning signs:** Merged model outputs identical to untrained base model. Higher perplexity than expected.
**Confidence:** HIGH -- verified across multiple GitHub issues (#195, #2516, #1089).

### Pitfall 2: Qwen3 Chat Template Mismatch

**What goes wrong:** Fine-tuned model ignores training data, behaves like base model (1% style adherence reported in QwenLM/Qwen3#1718).
**Why it happens:** Qwen3 has dual-mode architecture (thinking/non-thinking). If training data format does not match expected chat template, model ignores it. Default is thinking mode ON which adds `<think>` blocks.
**How to avoid:** For DPO training, the dataset from Phase 4 already uses conversational format (`prompt`/`chosen`/`rejected` with role+content dicts). TRL DPOTrainer auto-applies `tokenizer.apply_chat_template()`. For non-thinking parody generation, ensure `enable_thinking=False` is set. The Qwen3 non-thinking format wraps the assistant response with empty think tags: `<think>\n\n</think>`.
**Warning signs:** Model produces unexpected `<think>` blocks. Loss decreases but evaluation shows no improvement. Model enters infinite repetition loops.
**Confidence:** HIGH -- verified from Qwen3 model card and QwenLM/Qwen3#1718.

### Pitfall 3: Unsloth/TRL Version Mismatch

**What goes wrong:** Training crashes with `TypeError` on method signatures, or loss computation is silently wrong.
**Why it happens:** Unsloth monkey-patches TRL and Transformers internals. When versions are mismatched, patched methods have wrong signatures. RunPod templates may have pre-installed incompatible versions.
**How to avoid:** Install Unsloth FIRST (`pip install unsloth unsloth_zoo`), let it resolve dependencies. Then install TRL. Run a single training step to verify before committing to a full run.
**Warning signs:** `TypeError` about unexpected keyword arguments. Loss curves look qualitatively different from examples. Warnings about `num_items_in_batch`.
**Confidence:** HIGH -- verified from GitHub issues #2916, #3750, #3527.

### Pitfall 4: RunPod Container Storage Loss

**What goes wrong:** Checkpoints, trained adapters, merged models are permanently lost when pod stops.
**Why it happens:** Default filesystem on RunPod is container disk (ephemeral). Only `/workspace/` (network volume) persists.
**How to avoid:** Attach network volume when creating pod. Set ALL output paths to `/workspace/`. Verify persistence: `touch /workspace/test && pod stop && pod start && ls /workspace/test`. Push to Hub immediately after training as additional backup.
**Warning signs:** `ls /workspace/` shows empty directory (no network volume attached). Files gone after pod restart.
**Confidence:** HIGH -- verified from RunPod storage docs.

### Pitfall 5: Unsloth `save_pretrained_merged` Re-downloads FP16 Weights

**What goes wrong:** Merge step takes 30+ minutes because it re-downloads the ~65GB FP16 base model every time.
**Why it happens:** Bug in Unsloth: `merged_16bit` downloads FP16 weights into a `.cache/` folder inside the output directory, not the standard HF cache. If output directory is cleaned between runs, download repeats.
**How to avoid:** Either (a) accept the download time (one-time cost per merge), (b) pre-download FP16 base into HF cache: `from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B", torch_dtype="float16")`, or (c) do not delete the output directory between runs. Ensure ~130GB free disk space on network volume.
**Warning signs:** Merge step shows download progress bar for ~65GB. Disk space fills unexpectedly.
**Confidence:** HIGH -- verified from GitHub issue #3633. Fix reportedly in progress.

### Pitfall 6: DPO Dataset Column Name Mismatch

**What goes wrong:** DPOTrainer cannot find expected columns and crashes or uses wrong data.
**Why it happens:** TRL DPOTrainer expects `prompt`, `chosen`, `rejected` columns in conversational format (list of message dicts). If column names differ or format is non-conversational, training fails.
**How to avoid:** The existing `build_dpo_dataset()` in `dataset.py` already produces the correct format. Verify at training time: `dataset[0]` should show `prompt` (list of 2 dicts), `chosen` (list of 1 dict), `rejected` (list of 1 dict). All dicts have `role` and `content` keys.
**Warning signs:** `KeyError` on dataset columns. Tokenization errors. All rewards/accuracies at 0.5 (random).
**Confidence:** HIGH -- verified from TRL DPO docs and existing test_dataset.py tests.

## Code Examples

### Complete DPO Training Script Skeleton

```python
#!/usr/bin/env python3
"""DPO training script for chucklesPRIME on RunPod.

Usage:
    python train_dpo.py

Requires:
    - RunPod A6000 48GB (or A100 80GB)
    - Network volume mounted at /workspace/
    - HF_TOKEN environment variable set
    - pip install unsloth unsloth_zoo trl
"""
# Source: TRL DPOTrainer docs + Unsloth Qwen3 docs
import os
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastModel

# --- Configuration ---
MODEL_NAME = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
DATASET_NAME = "patruff/chuckles-dpo"
OUTPUT_DIR = "/workspace/dpo-output"
HUB_ADAPTER_REPO = "patruff/chuckles-qwen3-32b-dpo-adapter"
MAX_SEQ_LENGTH = 2048

# --- Model Loading ---
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# --- Dataset ---
dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset loaded: {dataset.num_rows} rows")
print(f"Columns: {dataset.column_names}")
print(f"Sample row[0]: {dataset[0]}")

# --- Training ---
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.1,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=512,
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",  # Prevent optimizer OOM spikes
    report_to="none",  # Set to "wandb" if W&B is configured
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # PEFT-aware: no extra VRAM
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train()

# --- Save Adapter ---
model.save_pretrained_merged(
    os.path.join(OUTPUT_DIR, "final-adapter"),
    tokenizer,
    save_method="lora",
)

# Push adapter to Hub as backup
model.push_to_hub_merged(
    HUB_ADAPTER_REPO,
    tokenizer,
    save_method="lora",
    token=os.environ["HF_TOKEN"],
)

print(f"Training complete. Adapter saved to {OUTPUT_DIR}/final-adapter")
print(f"Adapter pushed to Hub: {HUB_ADAPTER_REPO}")
```

### Complete Merge-and-Push Script Skeleton

```python
#!/usr/bin/env python3
"""Merge LoRA adapter into FP16 base model and push to Hub.

Usage:
    python merge_and_push.py

Requires:
    - ~130GB free disk space on /workspace/
    - HF_TOKEN environment variable set
"""
# Source: Unsloth save docs + issue #3633 workaround
import os
from unsloth import FastModel

# --- Configuration ---
BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
ADAPTER_PATH = "/workspace/dpo-output/final-adapter"
MERGED_OUTPUT_DIR = "/workspace/merged-model"
HUB_MERGED_REPO = "patruff/chuckles-qwen3-32b-dpo"
MAX_SEQ_LENGTH = 2048

# --- Load base model + adapter ---
model, tokenizer = FastModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Load the trained adapter
# Option A: from local path
model.load_adapter(ADAPTER_PATH)
# Option B: from Hub
# from peft import PeftModel
# model = PeftModel.from_pretrained(model, "patruff/chuckles-qwen3-32b-dpo-adapter")

# --- Merge to 16-bit and save locally ---
print("Merging to 16-bit (will download FP16 base model ~65GB)...")
model.save_pretrained_merged(
    MERGED_OUTPUT_DIR,
    tokenizer,
    save_method="merged_16bit",
)
print(f"Merged model saved to {MERGED_OUTPUT_DIR}")

# --- Push to Hub ---
print(f"Pushing merged model to Hub: {HUB_MERGED_REPO}")
model.push_to_hub_merged(
    HUB_MERGED_REPO,
    tokenizer,
    save_method="merged_16bit",
    token=os.environ["HF_TOKEN"],
)
print("Done!")
```

### Qwen3 Chat Template Verification

```python
# Source: Qwen3 model card + Unsloth Qwen3 docs
# Verify the chat template produces correct format BEFORE training

messages = [
    {"role": "system", "content": "You are a comedy writer who creates funny parody titles."},
    {"role": "user", "content": "Create a phonetically-sound parody of: 'The Matrix'"},
]

# Non-thinking mode (recommended for parody generation)
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Disable thinking mode
)
print("Formatted template:")
print(repr(formatted))
# Should contain <|im_start|>system, <|im_start|>user, <|im_start|>assistant
# Should NOT contain <think> blocks (unless enable_thinking=True)

# Verify DPO dataset format matches
from datasets import load_dataset
dataset = load_dataset("patruff/chuckles-dpo", split="train")
sample = dataset[0]
print(f"\nDataset prompt: {sample['prompt']}")
print(f"Dataset chosen: {sample['chosen']}")
print(f"Dataset rejected: {sample['rejected']}")
# prompt should be list of dicts with role+content
# chosen/rejected should be list with 1 assistant message dict
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `FastLanguageModel` | `FastModel` | Unsloth 2025 | `FastModel` is the unified API for text, vision, and MOE models. `FastLanguageModel` still works for dense text models but is legacy. |
| `tokenizer=` parameter in DPOTrainer | `processing_class=tokenizer` | TRL >= 0.12 | `processing_class` is the current API. Supports both tokenizers and processors (for VLMs). |
| Manual reference model loading | `ref_model=None` with PEFT | TRL DPOTrainer | Automatic adapter toggling for reference logits. Zero extra VRAM. The standard approach when using QLoRA/LoRA. |
| `merged_4bit` for deployment | `merged_16bit` then re-quantize | Unsloth 2025 | 4-bit merge degrades quality. 16-bit merge preserves quality, then vLLM handles runtime quantization. |
| Separate `save_pretrained` + manual Hub push | `push_to_hub_merged()` one-liner | Unsloth 2025 | Single method handles merge + upload. But has known bugs (#3146) -- always save locally first as backup. |

**Deprecated/outdated:**
- `FastLanguageModel`: Still works but legacy. Use `FastModel` for new code.
- `tokenizer=` parameter in DPOTrainer: Replaced by `processing_class=`. Old param may still work but is not documented.
- `merged_4bit` save method: Discouraged by Unsloth unless you know the implications. Degrades model quality.

## Open Questions

Things that could not be fully resolved and the planner should be aware of.

1. **Exact DPO hyperparameters for small parody datasets**
   - What we know: Standard DPO defaults (lr=5e-6, beta=0.1, epochs=3) work for large datasets (10K+ rows). The chucklesPRIME DPO dataset will likely have dozens to low hundreds of rows.
   - What is unclear: Whether such a small dataset needs different hyperparameters (fewer epochs to avoid overfitting, different learning rate).
   - Recommendation: Start with defaults. Monitor `rewards/accuracies` -- if it quickly reaches 1.0, reduce epochs. If it stays at 0.5, increase learning rate. Include hyperparameter notes in the training script as comments.

2. **Qwen3 `enable_thinking` behavior during DPO training**
   - What we know: TRL DPOTrainer auto-applies `tokenizer.apply_chat_template()`. The Qwen3 tokenizer supports `enable_thinking` parameter. Training with thinking ON means the model expects `<think>` blocks in chosen/rejected.
   - What is unclear: Whether TRL's auto-application passes `enable_thinking=False` by default, or whether we need to manually preprocess. The existing DPO dataset from Phase 4 does NOT include `<think>` blocks in chosen/rejected responses.
   - Recommendation: Add a verification step in the training script that prints the first tokenized sample to confirm no `<think>` blocks. If they appear, preprocess the dataset with manual `apply_chat_template(enable_thinking=False)` before passing to DPOTrainer.

3. **Unsloth `push_to_hub_merged` reliability for Qwen3**
   - What we know: GitHub issue #3146 reports that `push_to_hub_merged` sometimes pushes only the README, not model weights. Issue is not Qwen3-specific but has been reported broadly.
   - What is unclear: Whether this is fixed in Unsloth >= 2026.1.4.
   - Recommendation: Always `save_pretrained_merged` locally first, verify files exist, then push separately. Use `huggingface-hub` API as fallback if `push_to_hub_merged` fails.

4. **Disk space requirements for merge on RunPod**
   - What we know: `merged_16bit` downloads FP16 base (~65GB) + outputs merged model (~65GB). That is ~130GB minimum.
   - What is unclear: Whether RunPod network volume default size (usually 50-100GB) is sufficient. Container disk (default 20-50GB) is definitely not.
   - Recommendation: Create network volume with >= 200GB when setting up the RunPod pod. Document this in `setup_runpod.sh`.

## VRAM Budget (Phase 5 Specific)

| Operation | Component | VRAM | Notes |
|-----------|-----------|------|-------|
| DPO Training | Model weights (4-bit) | ~20 GB | 32.8B params at NF4 |
| DPO Training | LoRA adapter (rank 32) | ~1-2 GB | bf16 adapter weights |
| DPO Training | Optimizer states | ~2-3 GB | Paged AdamW 8-bit for LoRA params only |
| DPO Training | Activations/gradients | ~4-8 GB | With gradient checkpointing, batch_size=2, seq=2048 |
| DPO Training | **TOTAL** | **~27-33 GB** | **Fits A6000 (48GB) with 15-20GB headroom** |
| Merge | Model loading | ~20 GB | 4-bit base model |
| Merge | FP16 base download | (disk only) | ~65GB disk, not VRAM |
| Merge | Merge operation | ~2-4 GB extra | Temporary memory for merge |
| Validation | Same as training | ~27-33 GB | Load adapter model for comparison |

**A6000 (48GB) is sufficient for all Phase 5 operations.** No need for A100.

## DPO Dataset Format Verification

The existing `build_dpo_dataset()` in `src/chuckles_prime/dataset.py` produces exactly the format TRL DPOTrainer expects. Verified from both the source code and `test_dataset.py`:

```python
# What the dataset looks like (from test_dataset.py line 134-140):
{
    "prompt": [
        {"role": "system", "content": "You are a comedy writer..."},
        {"role": "user", "content": "Create a phonetically-sound parody of: 'The Matrix'"},
    ],
    "chosen": [
        {"role": "assistant", "content": "The Mattress"},
    ],
    "rejected": [
        {"role": "assistant", "content": "The Maitricks"},
    ],
}
```

This is the **conversational preference format** that TRL DPOTrainer expects. The trainer auto-applies `tokenizer.apply_chat_template()` to convert this to token IDs. No manual preprocessing needed.

**Confidence:** HIGH -- verified from both TRL docs (conversational format section) and existing test suite.

## Sources

### Primary (HIGH confidence)
- [TRL DPOTrainer documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) -- Dataset format, `processing_class`, reference model handling, loss functions, Unsloth integration
- [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) -- `FastModel`, model names, chat template format, `enable_thinking`
- [Unsloth GitHub wiki](https://github.com/unslothai/unsloth/wiki) -- Save methods (`merged_16bit`, `lora`), push_to_hub_merged
- Existing codebase: `dataset.py`, `rewards.py`, `types.py`, `test_dataset.py` -- DPO dataset format verified from source

### Secondary (MEDIUM confidence)
- [Unsloth GitHub issue #3633](https://github.com/unslothai/unsloth/issues/3633) -- `save_pretrained_merged` re-download bug, workaround
- [Unsloth GitHub issue #3146](https://github.com/unslothai/unsloth/issues/3146) -- `push_to_hub_merged` reliability issues
- [Unsloth GitHub issue #2329](https://github.com/unslothai/unsloth/issues/2329) -- `FastModel` vs `FastLanguageModel` clarification
- [QwenLM/Qwen3 issue #1718](https://github.com/QwenLM/Qwen3/issues/1718) -- Fine-tuning style adherence failure with chat template mismatch
- Milestone research: STACK.md, ARCHITECTURE.md, PITFALLS.md -- Version-verified stack, VRAM budgets, pitfall catalog

### Tertiary (LOW confidence)
- DPO hyperparameter recommendations for small datasets -- Extrapolated from general DPO literature; no domain-specific validation
- Exact VRAM numbers for Qwen3-32B QLoRA DPO -- Estimated from Unsloth's 32B table (26GB min); actual DPO overhead (reference model logits) may vary
- `enable_thinking` behavior in TRL auto-template application -- Not explicitly documented in TRL; needs runtime verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All versions verified via PyPI and official docs. Unsloth + TRL DPOTrainer integration confirmed.
- Architecture: HIGH -- Training script patterns verified from TRL docs and Unsloth examples. Dataset format confirmed from existing codebase.
- Pitfalls: HIGH -- All critical pitfalls verified from multiple GitHub issues. Merge degradation, template mismatch, storage loss all documented.
- Code examples: MEDIUM -- Based on official docs but not tested on actual Qwen3-32B (would require GPU). Patterns are standard but hyperparameters may need tuning.

**Research date:** 2026-01-31
**Valid until:** 2026-02-28 (Unsloth and TRL release frequently; check for breaking changes before executing)

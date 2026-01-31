# Architecture Research: Fine-tuning & Inference Integration

**Domain:** DPO/GRPO fine-tuning and vLLM inference serving for chucklesPRIME
**Researched:** 2026-01-31
**Confidence:** HIGH (core patterns verified with official TRL/Unsloth docs)

---

## System Overview

```
LOCAL MACHINE                       HUGGINGFACE HUB                    RUNPOD GPU POD
==================                  ==================                 ==================

chuckles generate
  |
  v
[DPO dataset] ----push---->  patruff/chuckles-dpo  <----load----  [train_dpo.py]
[GRPO dataset] ---push---->  patruff/chuckles-grpo <----load----  [train_grpo.py]
                                                                       |
                                                                       v
                                                                  [Unsloth + TRL]
                                                                  [4-bit QLoRA]
                                                                       |
                                                                       v
                                                                  [LoRA adapter]
                                                                       |
                                                                  merge_and_push.py
                                                                       |
                                                                       v
                              patruff/chuckles-qwen3-32b <---push--    |
                                     |
                                     |
                               +-----+------+
                               |            |
                               v            v
                          [vLLM serve]   [Ollama]     <-- RUNPOD INFERENCE POD
                          OpenAI API     OpenAI API       (cheaper GPU)
                               |            |
                               +-----+------+
                                     |
          chuckles generate  <-------+
          --api-base-url vllm_endpoint
          |
          v
        [Better parodies]
        [New datasets]
        [Push to Hub]
        [Train again...]
```

### Component Responsibilities

| Component | Responsibility | Location | New/Existing |
|-----------|---------------|----------|-------------|
| `chuckles generate` | Generate parodies, build datasets, push to Hub | Local | **Existing** (v1.0) |
| `chuckles convert` | Convert JSONL traces to datasets | Local | **Existing** (v1.0) |
| `chuckles label` | Human labeling web UI | Local | **Existing** (v1.0) |
| `chuckles export-labels` | Export labels to DPO dataset | Local | **Existing** (v1.0) |
| `train_dpo.py` | DPO training script | RunPod | **NEW** |
| `train_grpo.py` | GRPO training script with reward functions | RunPod | **NEW** |
| `merge_and_push.py` | Merge LoRA adapters, push merged model to Hub | RunPod | **NEW** |
| `rewards.py` | Phonetic quality, tool usage, structure preservation | Package (`src/`) | **Existing** (v1.0) |
| `vLLM server` | Serve fine-tuned model as OpenAI-compatible API | RunPod | **NEW** (config only) |
| `OpenAICompatibleModel` | Connect to any OpenAI-compatible endpoint | Package (`src/`) | **Existing** (no changes) |

---

## Recommended Project Structure

```
chucklesPRIME/
|-- pyproject.toml                    # Existing -- add [project.optional-dependencies] training
|-- src/
|   |-- chuckles_prime/               # Existing package -- NO CHANGES to source code
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- model.py                  # OpenAICompatibleModel (works with vLLM as-is)
|   |   |-- generator.py
|   |   |-- dataset.py
|   |   |-- rewards.py                # <-- CRITICAL: reward functions imported by train_grpo.py
|   |   |-- types.py
|   |   |-- tools.py
|   |   |-- prompts.py
|   |   |-- traces.py
|   |   |-- labeler.py
|   |   |-- csv_cleaner.py
|   |   |-- cli.py
|
|-- training/                         # NEW -- standalone training scripts
|   |-- train_dpo.py                  # DPO training (Unsloth + TRL DPOTrainer)
|   |-- train_grpo.py                 # GRPO training (Unsloth + TRL GRPOTrainer)
|   |-- merge_and_push.py            # Merge LoRA adapters -> push to Hub
|   |-- configs/
|   |   |-- dpo_config.yaml           # DPO hyperparameters
|   |   |-- grpo_config.yaml          # GRPO hyperparameters
|   |-- setup_runpod.sh              # RunPod environment setup script
|   |-- README.md                     # Training instructions
|
|-- tests/                            # Existing + new
|   |-- test_rewards_grpo_compat.py  # NEW: verify rewards work as GRPOTrainer callables
|   |-- ... (existing tests)
```

### Structure Rationale

- **`training/` is a sibling directory, not inside `src/chuckles_prime/`** -- Training scripts are standalone, meant to be run independently on RunPod. They are NOT part of the installable package. They DO import from the installed `chuckles_prime` package for reward functions.

- **No changes to existing `src/` code** -- The v1.0 package is complete and working. The training scripts consume its outputs (datasets on Hub) and import its reward functions. The `OpenAICompatibleModel` adapter already works with any OpenAI-compatible endpoint including vLLM.

- **`training/configs/` for hyperparameters** -- Keep training hyperparameters in YAML files rather than hardcoded in scripts. Easy to iterate on RunPod without editing Python.

- **`setup_runpod.sh` for reproducible environment** -- One script that installs everything needed on a fresh RunPod pod: Unsloth, TRL, the chucklesPRIME package from GitHub, etc.

---

## Critical Integration: Reward Functions on RunPod

This is the most important architectural decision for v1.1.

### The Problem

GRPO training needs custom reward functions. The chucklesPRIME reward functions live in `src/chuckles_prime/rewards.py`. Training runs on RunPod, not locally. How do the reward functions get to RunPod?

### Recommended Approach: pip install from GitHub

**Install the `chuckles_prime` package on RunPod from the GitHub repo.** The reward functions then become importable as `from chuckles_prime.rewards import ...`.

```bash
# In setup_runpod.sh
pip install "git+https://github.com/patruff/chucklesPRIME.git"
```

**Why this approach:**

1. **Single source of truth** -- Reward functions are defined once in `rewards.py`. No copy-paste, no drift.
2. **Version-locked** -- Can pin to a specific commit: `pip install "git+https://github.com/patruff/chucklesPRIME.git@v1.0"`.
3. **Existing package structure supports it** -- `pyproject.toml` already defines `chuckles_prime` as an installable package with `setuptools`.
4. **Lightweight** -- Only installs the package and its dependencies, not training-specific dependencies.
5. **Standard pattern** -- The popular GRPO training gist by willccbb uses `pip install git+https://github.com/...` for dependencies on RunPod.

**Confidence: HIGH** -- Verified by examining `pyproject.toml` (setuptools with `src/` layout), and confirmed this is the standard RunPod training pattern from multiple sources.

### Alternative Considered: Copy reward functions into training scripts

Copy-paste the three reward functions directly into `train_grpo.py`. Simpler initial setup, but creates divergence risk if reward functions are updated. **Rejected** because maintaining two copies of reward logic across local and RunPod is a maintenance hazard.

### Alternative Considered: Minimal rewards-only package

Extract rewards into a separate lightweight package. Adds complexity for minimal benefit when the full package installs cleanly. **Rejected** as over-engineering.

### Reward Function Adaptation for GRPOTrainer

**Critical finding from TRL docs (HIGH confidence):** GRPOTrainer reward functions receive:
- `prompts` -- list of prompts (conversational format: list of message dicts)
- `completions` -- list of completions (conversational format: list of message dicts)
- `completion_ids` -- list of tokenized completions
- `trainer_state` -- TrainerState object
- All additional dataset column names as keyword arguments
- Must accept `**kwargs` for forward compatibility

The existing reward functions in `rewards.py` have signatures like:
```python
def compute_phonetic_quality(candidate: ParodyCandidate) -> float
def compute_tool_usage_completeness(trace: AgentTrace, input_title: str) -> float
def compute_structure_preservation(input_title: str, parody_text: str) -> float
```

These are **not directly compatible** with the GRPOTrainer callable interface. The training script needs **wrapper functions** that bridge from GRPOTrainer's `(completions, **kwargs)` signature to the underlying reward computation.

### Wrapper Pattern for GRPO Reward Functions

```python
# In train_grpo.py (NOT in the main package)

from chuckles_prime.rewards import (
    compute_phonetic_quality,
    compute_structure_preservation,
)
from chuckles_prime.types import ParodyCandidate

def phonetic_reward(completions, original_title, **kwargs) -> list[float]:
    """GRPO-compatible wrapper for compute_phonetic_quality.

    Extracts parody text from model completions, constructs a minimal
    ParodyCandidate, and delegates to the core reward function.
    """
    rewards = []
    for completion, title in zip(completions, original_title):
        # completions in conversational format: [{"role": "assistant", "content": "..."}]
        text = completion[0]["content"] if isinstance(completion, list) else completion
        # Create minimal ParodyCandidate -- phonetic_scores must be computed
        # from the text vs original_title using pronouncing library
        candidate = ParodyCandidate(text=text, phonetic_scores={})
        # For GRPO, we recompute phonetic similarity at training time
        score = _compute_live_phonetic_score(text, title)
        rewards.append(score)
    return rewards

def structure_reward(completions, original_title, **kwargs) -> list[float]:
    """GRPO-compatible wrapper for compute_structure_preservation."""
    rewards = []
    for completion, title in zip(completions, original_title):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(compute_structure_preservation(title, text))
    return rewards
```

**Key insight:** The `original_title` column from the GRPO dataset on Hub gets automatically passed as a kwarg to each reward function. This is how dataset metadata flows into reward computation during training.

**What the wrappers need to handle:**
1. Extract text content from conversational-format completions
2. Access `original_title` from dataset columns (passed as kwarg)
3. Compute the actual reward score using the core functions (may need live phonetic computation)
4. Return `list[float]` matching the batch size

The wrappers live in `train_grpo.py`, NOT in the main package. The main package's reward functions remain clean domain logic; the wrappers are training-infrastructure glue.

---

## Data Flow: End-to-End Improvement Loop

### Flow 1: Generate Data (Existing, No Changes)

```
LOCAL:
  chuckles generate titles.csv \
    --settings settings.json \
    --grpo-repo patruff/chuckles-grpo \
    --dpo-repo patruff/chuckles-dpo

  Result:
    -> output/traces.jsonl (local archive)
    -> patruff/chuckles-grpo on HuggingFace Hub
    -> patruff/chuckles-dpo on HuggingFace Hub
```

### Flow 2: DPO Training (New)

```
RUNPOD (A6000 48GB or A100 80GB):

  1. setup_runpod.sh installs environment
  2. python train_dpo.py \
       --model unsloth/Qwen3-32B-unsloth-bnb-4bit \
       --dataset patruff/chuckles-dpo \
       --output ./dpo-output

  Data flow:
    HF Hub (patruff/chuckles-dpo) --(load_dataset)--> train_dpo.py
    Unsloth loads Qwen3-32B in 4-bit
    TRL DPOTrainer trains LoRA adapter
    -> ./dpo-output/ (LoRA adapter checkpoint)
```

### Flow 3: GRPO Training (New)

```
RUNPOD (A6000 48GB or A100 80GB):

  1. (same environment from setup_runpod.sh)
  2. python train_grpo.py \
       --model unsloth/Qwen3-32B-unsloth-bnb-4bit \
       --dataset patruff/chuckles-grpo \
       --output ./grpo-output

  Data flow:
    HF Hub (patruff/chuckles-grpo) --(load_dataset)--> train_grpo.py
    Dataset columns: prompt, original_title, phonetic_scores, avg_*, etc.
    Reward functions receive completions + original_title as kwargs
    -> ./grpo-output/ (LoRA adapter checkpoint)
```

### Flow 4: Merge and Push (New)

```
RUNPOD (same pod as training):

  python merge_and_push.py \
    --adapter ./grpo-output \
    --base unsloth/Qwen3-32B \
    --repo patruff/chuckles-qwen3-32b-v1 \
    --save-method merged_16bit

  Data flow:
    LoRA adapter + base model -> merged model -> HF Hub
    -> patruff/chuckles-qwen3-32b-v1 on HuggingFace Hub
```

### Flow 5: Inference Serving (New)

```
RUNPOD (RTX 4090 24GB or A6000 48GB -- cheaper inference tier):

  vllm serve patruff/chuckles-qwen3-32b-v1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

  Or using RunPod Serverless vLLM worker:
    Deploy with model: patruff/chuckles-qwen3-32b-v1
    OpenAI-compatible endpoint: https://api.runpod.ai/v2/<ID>/openai/v1

  The endpoint is OpenAI-compatible:
    POST /v1/chat/completions
    GET /v1/models
```

### Flow 6: Generate with Fine-tuned Model (Existing CLI, New Config)

```
LOCAL:
  # Update settings.json to point to vLLM endpoint:
  {
    "model_name": "patruff/chuckles-qwen3-32b-v1",
    "api_base_url": "https://api.runpod.ai/v2/<ID>/openai/v1",
    "api_key_env_var": "RUNPOD_API_KEY",
    ...
  }

  chuckles generate titles.csv --settings settings-finetuned.json

  Result:
    -> Better parodies from fine-tuned model
    -> New traces, new datasets
    -> Push to Hub again
    -> Train again (improvement loop complete)
```

---

## DPO Training Architecture

### DPO Dataset Format (Existing on Hub)

The existing DPO dataset from v1.0 already matches TRL's expected format:

```python
{
    "prompt": [
        {"role": "system", "content": "You are a comedy writer..."},
        {"role": "user", "content": "Create a phonetically-sound parody of: 'The Matrix'"}
    ],
    "chosen": [
        {"role": "assistant", "content": "The Mattress"}   # human example
    ],
    "rejected": [
        {"role": "assistant", "content": "The Hatrix"}     # worst model candidate
    ]
}
```

**Confidence: HIGH** -- Verified against TRL DPOTrainer docs. This is exactly the "conversational preference" format that DPOTrainer expects. The trainer auto-applies the chat template.

### DPO Training Script Structure

```python
# training/train_dpo.py

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastModel

# 1. Load model with Unsloth (4-bit QLoRA)
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

# 2. Attach LoRA adapters
model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
)

# 3. Load dataset from Hub
dataset = load_dataset("patruff/chuckles-dpo", split="train")

# 4. Configure and train
training_args = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    max_length=2048,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()

# 5. Save adapter
model.save_pretrained("./dpo-output/final")
tokenizer.save_pretrained("./dpo-output/final")
```

### Reference Model Handling

With Unsloth + QLoRA for DPO, the reference model is handled implicitly. When `ref_model=None` (default) and PEFT is detected, DPOTrainer disables the adapter to get reference model logits. This is memory-efficient and is the approach Unsloth recommends.

**Confidence: HIGH** -- Verified from TRL DPO docs: "Merge the adapter into the base model, create another adapter on top, then leave the ref_model param null, in which case DPOTrainer will unload the adapter for reference inference."

---

## GRPO Training Architecture

### GRPO Dataset Format (Existing on Hub)

The existing GRPO dataset from v1.0:

```python
{
    "prompt": [
        {"role": "system", "content": "You are a comedy writer..."},
        {"role": "user", "content": "Create a phonetically-sound parody of: 'The Matrix'"}
    ],
    "original_title": "The Matrix",
    "phonetic_scores": "{...}",            # JSON string of per-word scores
    "generation_model": "qwen-3-32b",
    "avg_phonetic_score": 0.78,
    "avg_tool_usage": 0.85,
    "avg_structure_preservation": 0.90
}
```

**Key: The `original_title` column is passed as a kwarg to reward functions.** The `avg_*` columns are pre-computed metadata for analysis but are NOT used as rewards during training. The reward functions recompute scores from the model's live completions.

### GRPO Training Script Structure

```python
# training/train_grpo.py

import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastModel
from chuckles_prime.rewards import compute_structure_preservation

# --- Reward Functions (GRPO-compatible wrappers) ---

def phonetic_reward(completions, original_title, **kwargs) -> list[float]:
    """Reward based on phonetic similarity of parody to original title."""
    import pronouncing
    rewards = []
    for completion, title in zip(completions, original_title):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        orig_words = title.split()
        parody_words = text.split()
        if not orig_words:
            rewards.append(0.0)
            continue
        # Compute phonetic similarity per word pair
        scores = []
        for ow, pw in zip(orig_words, parody_words):
            # Use CMU pronouncing dictionary for phonetic comparison
            op = pronouncing.phones_for_word(ow.lower())
            pp = pronouncing.phones_for_word(pw.lower())
            if op and pp:
                # Simple phoneme overlap ratio
                o_set = set(op[0].split())
                p_set = set(pp[0].split())
                if o_set or p_set:
                    scores.append(len(o_set & p_set) / len(o_set | p_set))
        rewards.append(sum(scores) / len(scores) if scores else 0.0)
    return rewards

def structure_reward(completions, original_title, **kwargs) -> list[float]:
    """Reward based on word count preservation."""
    rewards = []
    for completion, title in zip(completions, original_title):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(compute_structure_preservation(title, text))
    return rewards

def format_reward(completions, **kwargs) -> list[float]:
    """Reward for clean, single-line parody output (no explanations)."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        # Good: short, single-line parody title
        # Bad: long explanation, multiple lines, JSON output
        lines = text.strip().split("\n")
        if len(lines) == 1 and len(text) < 200:
            rewards.append(1.0)
        elif len(lines) <= 2 and len(text) < 300:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

# --- Model Setup ---

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
)

# --- Dataset ---

dataset = load_dataset("patruff/chuckles-grpo", split="train")

# --- Training ---

training_args = GRPOConfig(
    output_dir="./grpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    max_completion_length=256,
    num_generations=4,               # G completions per prompt
    # reward_weights=[0.5, 0.3, 0.2],  # Optional: weight the three rewards
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=[phonetic_reward, structure_reward, format_reward],
    train_dataset=dataset,
)
trainer.train()
```

**Confidence: HIGH** on the GRPOTrainer interface and reward function signature. Verified from official TRL docs that dataset columns are passed as kwargs to reward functions, and multiple reward functions can be combined via a list.

---

## vLLM Inference Architecture

### Qwen3-32B Serving Configuration

**Option A: RunPod Pod with vLLM (Recommended for development/testing)**

```bash
# On a RunPod pod with 48GB+ VRAM (A6000)
pip install vllm

# Serve the fine-tuned model
vllm serve patruff/chuckles-qwen3-32b-v1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

**Option B: RunPod Serverless vLLM Worker (Recommended for production)**

Deploy using RunPod's vLLM worker template. Configure:
- Model: `patruff/chuckles-qwen3-32b-v1`
- Environment: `MAX_MODEL_LEN=8192`
- GPU: A6000 (48GB) minimum
- OpenAI endpoint: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`

**Option C: 4-bit Quantized for cheaper GPUs**

For RTX 4090 (24GB) deployment, quantize the merged model to AWQ:
```bash
# Serve AWQ-quantized model
vllm serve patruff/chuckles-qwen3-32b-v1-AWQ \
  --quantization awq \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

### VRAM Budget for Qwen3-32B

| Configuration | Model Weights | KV Cache (8K ctx) | Total VRAM | GPU |
|---|---|---|---|---|
| FP16 (unquantized) | ~65 GB | ~5 GB | ~70 GB | 2x A100-80GB |
| FP8 | ~32.5 GB | ~5 GB | ~37.5 GB | A6000 (48GB) |
| AWQ/GPTQ 4-bit | ~18 GB | ~5 GB | ~23 GB | RTX 4090 (24GB) |

**Recommendation:** Use FP8 on A6000 for best quality/cost balance. Use AWQ 4-bit on RTX 4090 only if cost is the primary constraint and quality loss is acceptable.

**Confidence: HIGH** on VRAM numbers -- verified from multiple sources including the RunPod Qwen3-32B deployment guide and vLLM docs.

### Integration with Existing OpenAICompatibleModel

The existing `OpenAICompatibleModel` in `model.py` wraps `openai.OpenAI(api_key=..., base_url=...)` and implements `smolagents.Model.generate()`. It works with ANY OpenAI-compatible endpoint.

**No code changes needed.** To use the fine-tuned model:

1. Point `settings.json` at the vLLM endpoint:
   ```json
   {
     "model_name": "patruff/chuckles-qwen3-32b-v1",
     "api_base_url": "https://api.runpod.ai/v2/<ID>/openai/v1",
     "api_key_env_var": "RUNPOD_API_KEY"
   }
   ```
2. Run `chuckles generate` as before.

**Confidence: HIGH** -- Verified from existing `model.py` source code. The adapter uses `openai.OpenAI(api_key=..., base_url=...)` which is the standard pattern for vLLM clients.

### RunPod vLLM Limitations

Per RunPod documentation, the vLLM worker has these limitations:
- Function/tool calling APIs are NOT supported
- Some OpenAI-specific features (moderation) unavailable

**Impact on chucklesPRIME:** The existing `OpenAICompatibleModel.generate()` explicitly sets `tools_to_call_from=None` and relies on `CodeAgent` text-based tool calls (not OpenAI function calling). This means the RunPod vLLM limitation on function calling does **not** affect us.

**Confidence: HIGH** -- Verified from `model.py` line 83: `tools_to_call_from=None`.

---

## Merge and Push Architecture

### LoRA Adapter Merging

After training, the LoRA adapter must be merged with the base model before deployment on vLLM. Unsloth provides built-in methods:

```python
# training/merge_and_push.py

from unsloth import FastModel

# Load base model + adapter
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-32B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load the trained adapter
model.load_adapter("./grpo-output/final")

# Merge and push (16-bit for vLLM serving)
model.push_to_hub_merged(
    "patruff/chuckles-qwen3-32b-v1",
    tokenizer,
    save_method="merged_16bit",
    token=HF_TOKEN,
)
```

### Known Issue: Qwen3 LoRA Merge Bugs

**MEDIUM confidence warning:** Multiple GitHub issues report problems with Unsloth's `push_to_hub_merged` and `save_pretrained_merged` for Qwen3 models specifically:
- RuntimeError from `assert_same_keys` during merge
- Performance degradation after merge (training inference works, merged model quality drops)

**Mitigation:**
1. Keep Unsloth updated (`pip install --upgrade unsloth unsloth_zoo`)
2. Test merged model quality before pushing to Hub (run a few parody generations)
3. Fallback: save LoRA adapter only, load adapter at inference time (supported by vLLM)

---

## RunPod Environment Setup

### setup_runpod.sh

```bash
#!/bin/bash
# Setup script for RunPod training pods
# Run once after pod creation

set -e

# 1. Install Unsloth (handles torch, transformers, etc.)
pip install unsloth

# 2. Install TRL (for DPOTrainer, GRPOTrainer)
pip install trl

# 3. Install chucklesPRIME package from GitHub (for reward functions)
pip install "git+https://github.com/patruff/chucklesPRIME.git"

# 4. Install additional training dependencies
pip install wandb  # optional: experiment tracking

# 5. Login to HuggingFace (for dataset loading and model pushing)
huggingface-cli login --token $HF_TOKEN

# 6. Verify installation
python -c "from chuckles_prime.rewards import compute_phonetic_quality; print('Rewards imported successfully')"
python -c "from trl import GRPOTrainer, DPOTrainer; print('TRL trainers available')"
python -c "from unsloth import FastModel; print('Unsloth available')"
```

### Environment Variables on RunPod

| Variable | Purpose | Where Set |
|----------|---------|-----------|
| `HF_TOKEN` | HuggingFace Hub access (read datasets, push models) | RunPod pod env |
| `WANDB_API_KEY` | Weights & Biases experiment tracking (optional) | RunPod pod env |
| `RUNPOD_API_KEY` | For vLLM serverless endpoint access | Local machine env |

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Embedding Training Logic in the Package

**What people do:** Add `train_dpo` and `train_grpo` as subcommands of the `chuckles` CLI.
**Why it's wrong:** Training requires heavy GPU dependencies (Unsloth, TRL, flash-attn) that should NOT be in the main package's dependency chain. Anyone doing `pip install chucklesPRIME` would need to install CUDA, torch, etc.
**Do this instead:** Keep training scripts standalone in `training/`. They install `chuckles_prime` as a lightweight dependency for reward functions only.

### Anti-Pattern 2: Copying Reward Functions into Training Scripts

**What people do:** Copy-paste the three reward functions from `rewards.py` into `train_grpo.py`.
**Why it's wrong:** Creates two sources of truth. When you improve a reward function, you must remember to update both copies. Divergence bugs are hard to catch.
**Do this instead:** `pip install` the package from GitHub on RunPod, import reward functions normally.

### Anti-Pattern 3: Training on Pre-computed Reward Scores

**What people do:** Use the `avg_phonetic_score`, `avg_tool_usage`, `avg_structure_preservation` columns from the GRPO dataset as rewards.
**Why it's wrong:** GRPO is an ONLINE algorithm. The model generates NEW completions during training. Rewards must be computed on these new completions, not on pre-computed scores from a different model's outputs.
**Do this instead:** Use the pre-computed scores for analysis only. Define reward functions that compute scores from the model's live completions during training.

### Anti-Pattern 4: Serving Unmerged LoRA with vLLM

**What people do:** Try to serve the LoRA adapter directly without merging.
**Why it's wrong:** While vLLM technically supports LoRA serving, it adds latency, complexity, and the adapter-switching API differs from standard OpenAI compatibility.
**Do this instead:** Merge the adapter into the base model before deploying on vLLM. One merged model, one vLLM endpoint, full OpenAI compatibility.

### Anti-Pattern 5: Using max_model_len=40960 on a 48GB GPU

**What people do:** Accept Qwen3-32B's default 40K context length when deploying on vLLM.
**Why it's wrong:** KV cache for 40K tokens does not fit in remaining VRAM after model weights. vLLM will refuse to start or crash.
**Do this instead:** Set `--max-model-len` to what you actually need. For parody generation, 8192 tokens is more than sufficient. This leaves ample VRAM for KV cache and batching.

---

## Scaling Considerations

| Concern | Current Scale (v1.1) | Future Scale |
|---------|---------------------|-------------|
| Dataset size | Hundreds of rows | Use data augmentation; more titles from IMDb/TMDb APIs |
| Training time | Minutes on A6000 (small dataset + QLoRA) | Multi-epoch, larger dataset; same hardware |
| Inference throughput | Single-user, sequential | vLLM handles batching; RunPod Serverless scales |
| Model iterations | Manual: train, evaluate, retrain | Automate with scripts; track with W&B |
| Reward function complexity | 3 simple heuristics | Add embedding-based humor similarity; LLM-as-judge |

For v1.1, all scaling concerns are irrelevant. The dataset is small, training is fast, and inference is single-user. The architecture supports scaling without restructuring.

---

## Build Order Implications

Based on dependencies between components:

1. **GRPO reward wrappers + compatibility test** -- Must work before training can begin. Test locally that reward wrappers produce sensible outputs.
2. **DPO training script** -- Simpler (no custom rewards). Good first training script to validate the Unsloth + TRL pipeline on RunPod.
3. **GRPO training script** -- Depends on reward wrappers being correct. Run after DPO works.
4. **Merge and push** -- Depends on having a trained adapter. Run after either training script completes.
5. **vLLM serving + settings config** -- Depends on having a merged model on Hub.
6. **End-to-end loop validation** -- Generate with fine-tuned model, verify quality improvement.

This ordering ensures each step can be validated independently before proceeding.

---

## Sources

- [TRL GRPOTrainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer) -- HIGH confidence: reward function signature, dataset format, vLLM integration
- [TRL DPOTrainer Documentation](https://huggingface.co/docs/trl/en/dpo_trainer) -- HIGH confidence: dataset format, Unsloth integration, reference model handling
- [TRL Reward Functions Reference](https://huggingface.co/docs/trl/en/rewards) -- HIGH confidence: built-in reward function signatures
- [Unsloth Qwen3 Documentation](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) -- MEDIUM confidence: model names, loading pattern (some Qwen3-32B specifics not fully documented)
- [Qwen Unsloth Training Guide](https://qwen.readthedocs.io/en/latest/training/unsloth.html) -- MEDIUM confidence: FastLanguageModel pattern, LoRA config
- [Guide to Deploying Qwen 3 with vLLM on RunPod](https://medium.com/@mshojaei77/guide-to-deploying-qwen-3-with-vllm-on-runpod-31b9da6642d0) -- MEDIUM confidence: VRAM budget, max_model_len configuration
- [RunPod vLLM OpenAI Compatibility](https://docs.runpod.io/serverless/vllm/openai-compatibility) -- HIGH confidence: endpoint format, limitations
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) -- HIGH confidence: serve command, configuration
- [GRPO Training Gist (willccbb)](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) -- MEDIUM confidence: RunPod training patterns, pip install from GitHub
- [Unsloth GitHub Issues #3403, #3146](https://github.com/unslothai/unsloth/issues/3403) -- MEDIUM confidence: Qwen3 merge bugs
- Existing codebase: `model.py`, `rewards.py`, `dataset.py`, `config.py`, `pyproject.toml` -- HIGH confidence: direct source code analysis

---

*Architecture research for: chucklesPRIME v1.1 Fine-tuning & Inference Integration*
*Researched: 2026-01-31*

#!/usr/bin/env python3
"""DPO training script for chucklesPRIME on RunPod.

Fine-tunes Qwen3-32B on phonetic parody preference data using DPO (Direct
Preference Optimization) with 4-bit QLoRA via Unsloth.

Usage:
    python train_dpo.py

Prerequisites:
    - RunPod A6000 48GB (or A100 80GB) with network volume at /workspace/
    - Environment setup via setup_runpod.sh (installs unsloth, trl, etc.)
    - HF_TOKEN environment variable set (for dataset download and adapter push)

What this script does:
    1. Loads Qwen3-32B in 4-bit quantization with LoRA adapters
    2. Loads the DPO preference dataset from HuggingFace Hub
    3. Verifies chat template format (no stray <think> blocks)
    4. Trains with TRL DPOTrainer for 3 epochs
    5. Saves LoRA adapter locally to /workspace/dpo-output/final-adapter
    6. Pushes LoRA adapter to HuggingFace Hub as backup

Output:
    - Training checkpoints: /workspace/dpo-output/checkpoint-*/
    - Final adapter: /workspace/dpo-output/final-adapter/
    - Hub adapter: patruff/chuckles-qwen3-32b-dpo-adapter

Hyperparameter tuning notes:
    - If rewards/accuracies quickly reaches 1.0, reduce num_train_epochs
    - If rewards/accuracies stuck at 0.5, increase learning_rate (try 1e-5)
    - If loss diverges (NaN or spike), reduce learning_rate (try 1e-6)
    - For very small datasets (<50 rows), consider reducing epochs to 1-2
"""

import os
import sys

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastModel

# =============================================================================
# Configuration
# =============================================================================

# Pre-quantized 4-bit model from Unsloth (avoids on-the-fly quantization)
MODEL_NAME = "unsloth/Qwen3-32B-unsloth-bnb-4bit"

# DPO preference dataset on HuggingFace Hub (created by Phase 3-4 pipeline)
DATASET_NAME = "patruff/chuckles-dpo"

# All output paths use /workspace/ -- RunPod network volume that persists
# across pod restarts. Anything outside /workspace/ is ephemeral.
OUTPUT_DIR = "/workspace/dpo-output"

# Hub destination for the trained LoRA adapter (safety backup before merge)
HUB_ADAPTER_REPO = "patruff/chuckles-qwen3-32b-dpo-adapter"

# Maximum sequence length for tokenization (prompt + response combined)
MAX_SEQ_LENGTH = 2048

# LoRA hyperparameters
LORA_R = 32            # LoRA rank -- 32 balances quality vs VRAM
LORA_ALPHA = 32        # alpha == r is a common starting point (scaling = 1.0)
LORA_DROPOUT = 0       # Unsloth recommends 0 dropout for QLoRA


# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    """Load Qwen3-32B in 4-bit with LoRA adapters attached.

    Uses FastModel (the current Unsloth API, not the older legacy API)
    which works for all model types (text, vision, MOE).
    """
    print("=" * 60)
    print(" Loading Model")
    print("=" * 60)

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,           # NF4 quantization via bitsandbytes
        full_finetuning=False,       # LoRA only, not full fine-tuning
    )

    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            # All linear layers for maximum fine-tuning quality
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        # Unsloth's optimized gradient checkpointing -- uses less VRAM than
        # standard HF gradient checkpointing by offloading activations smarter
        use_gradient_checkpointing="unsloth",
    )

    print(f"  Model: {MODEL_NAME}")
    print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print()

    return model, tokenizer


# =============================================================================
# Dataset Loading and Verification
# =============================================================================

def load_and_verify_dataset(tokenizer):
    """Load the DPO dataset from Hub and verify chat template format.

    The dataset must be in TRL conversational preference format:
      - prompt: list of message dicts (system + user)
      - chosen: list of message dicts (assistant)
      - rejected: list of message dicts (assistant)

    We also verify that applying the chat template does NOT produce
    <think> blocks, which would indicate Qwen3's thinking mode is active.
    Thinking mode adds reasoning tokens that our dataset doesn't include,
    causing a train/inference mismatch (Pitfall 2 from research).
    """
    print("=" * 60)
    print(" Loading Dataset")
    print("=" * 60)

    dataset = load_dataset(DATASET_NAME, split="train")

    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Rows: {dataset.num_rows}")
    print(f"  Columns: {dataset.column_names}")
    print()
    print(f"  Sample row[0]:")
    print(f"    prompt:   {dataset[0]['prompt']}")
    print(f"    chosen:   {dataset[0]['chosen']}")
    print(f"    rejected: {dataset[0]['rejected']}")
    print()

    # -------------------------------------------------------------------------
    # Chat template verification (Pitfall 2: Qwen3 thinking mode mismatch)
    # -------------------------------------------------------------------------
    # Qwen3 has dual-mode architecture: thinking (adds <think> blocks) and
    # non-thinking (clean responses). Our DPO dataset was built without
    # thinking tokens, so we MUST verify the template does not inject them.
    # If <think> appears, training data format won't match inference format.
    # -------------------------------------------------------------------------
    print("  Verifying chat template (enable_thinking=False)...")
    formatted = tokenizer.apply_chat_template(
        dataset[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(f"  Formatted template preview:")
    print(f"    {formatted[:200]}...")
    print()

    if "<think>" in formatted:
        print("  ERROR: Chat template contains <think> blocks!")
        print("  This means Qwen3 thinking mode is active despite")
        print("  enable_thinking=False. The DPO dataset does not")
        print("  include thinking tokens -- training will fail.")
        sys.exit(1)
    else:
        print("  PASS: No <think> blocks found in formatted template.")
        print()

    return dataset


# =============================================================================
# Training
# =============================================================================

def train(model, tokenizer, dataset):
    """Run DPO training with TRL DPOTrainer.

    Key decisions documented inline:
    - ref_model=None: PEFT-aware mode. DPOTrainer disables the LoRA adapter
      to compute reference model logits, using zero extra VRAM. This is the
      standard approach for QLoRA DPO training.
    - processing_class=tokenizer: Current TRL API (replaces the older
      `tokenizer=` kwarg which is deprecated).
    - loss_type="sigmoid": Standard DPO loss from the original paper.
    - optim="paged_adamw_8bit": Prevents optimizer state OOM spikes by
      paging optimizer states to CPU when VRAM is tight.
    """
    print("=" * 60)
    print(" Starting DPO Training")
    print("=" * 60)

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # Effective batch size = 2 * 4 = 8
        learning_rate=5e-6,
        beta=0.1,                        # DPO temperature (controls preference strength)
        bf16=True,                       # Required for Unsloth -- uses bfloat16 mixed precision
        logging_steps=10,                # Log metrics every 10 steps
        save_steps=100,                  # Checkpoint every 100 steps to /workspace/
        save_total_limit=3,              # Keep only last 3 checkpoints (disk space)
        max_length=MAX_SEQ_LENGTH,       # Max total length (prompt + response)
        max_prompt_length=512,           # Max prompt portion (rest is for response)
        warmup_ratio=0.1,               # 10% warmup for learning rate scheduler
        optim="paged_adamw_8bit",        # 8-bit optimizer to prevent OOM spikes
        report_to="none",               # Set to "wandb" if W&B is configured
        loss_type="sigmoid",             # Standard DPO loss from Rafailov et al.
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # PEFT-aware: disables adapter for reference logits
        args=training_args,
        processing_class=tokenizer,      # Current TRL API (not the deprecated `tokenizer=`)
        train_dataset=dataset,
    )

    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Beta (DPO temp): {training_args.beta}")
    print(f"  Loss type: {training_args.loss_type}")
    print()
    print("  Training started...")
    print()

    trainer.train()

    print()
    print("  Training complete!")
    print()

    return model, tokenizer


# =============================================================================
# Save and Push
# =============================================================================

def save_and_push(model, tokenizer):
    """Save LoRA adapter locally and push to HuggingFace Hub.

    The adapter is saved BEFORE the merge step (done in a separate script)
    as a safety backup. The merge step has known bugs (Unsloth issue #3146)
    and if it fails, the adapter is still safe on Hub.

    Uses save_method="lora" to save only the adapter weights (~100-300MB),
    not the full model. This is fast and small.
    """
    print("=" * 60)
    print(" Saving Adapter")
    print("=" * 60)

    # Save adapter locally to network volume
    adapter_path = os.path.join(OUTPUT_DIR, "final-adapter")
    print(f"  Saving adapter locally to: {adapter_path}")
    model.save_pretrained_merged(
        adapter_path,
        tokenizer,
        save_method="lora",
    )
    print(f"  Local save complete.")
    print()

    # Push adapter to Hub as backup
    print(f"  Pushing adapter to Hub: {HUB_ADAPTER_REPO}")
    model.push_to_hub_merged(
        HUB_ADAPTER_REPO,
        tokenizer,
        save_method="lora",
        token=os.environ["HF_TOKEN"],
    )
    print(f"  Hub push complete.")
    print()

    print("=" * 60)
    print(" All Done!")
    print("=" * 60)
    print()
    print(f"  Adapter saved to: {adapter_path}")
    print(f"  Adapter on Hub:   https://huggingface.co/{HUB_ADAPTER_REPO}")
    print()
    print("  Next steps:")
    print("    1. Run merge_and_push.py to merge adapter into FP16 model")
    print("    2. Run validate_merge.py to verify merged model quality")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Verify HF_TOKEN is available (needed for dataset download and Hub push)
    if "HF_TOKEN" not in os.environ:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("Set it with: export HF_TOKEN=\"hf_your_token_here\"")
        sys.exit(1)

    model, tokenizer = load_model()
    dataset = load_and_verify_dataset(tokenizer)
    model, tokenizer = train(model, tokenizer, dataset)
    save_and_push(model, tokenizer)

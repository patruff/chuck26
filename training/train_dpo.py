"""DPO training script for parody generation models.

Train a small language model on DPO preference data to generate better
phonetically-sound parody titles. Uses TRL's DPOTrainer with LoRA for
efficient fine-tuning on consumer GPUs.

Recommended cheap setups:
- RTX 3090 (24GB): Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-1B, Llama-3.2-3B
- RTX 4090 (24GB): Same as above, slightly faster
- A40 (48GB): Qwen2.5-7B, Llama-3.1-8B for better quality

Usage:
    python train_dpo.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset patruff/chuckles-dpo \
        --output ./parody-model \
        --epochs 3

Environment variables:
    HF_TOKEN - HuggingFace token for gated models and dataset access
    WANDB_API_KEY - (optional) Weights & Biases logging
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a parody model with DPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        default="patruff/chuckles-dpo",
        help="HuggingFace dataset with DPO preference pairs",
    )
    parser.add_argument(
        "--output",
        default="./parody-model",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (higher = more conservative)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (saves VRAM)",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Hub model ID (default: derived from output path)",
    )
    parser.add_argument(
        "--wandb-project",
        default="parody-dpo",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def format_dpo_example(example: dict) -> dict:
    """Convert dataset example to DPO trainer format.

    The dataset has:
    - prompt: list of message dicts
    - chosen: list of message dicts
    - rejected: list of message dicts

    DPO trainer expects string format, so we apply chat template.
    """
    # Build full conversations
    prompt_msgs = example.get("prompt", [])
    chosen_msgs = example.get("chosen", [])
    rejected_msgs = example.get("rejected", [])

    # Return in the format DPOTrainer expects
    return {
        "prompt": prompt_msgs,
        "chosen": prompt_msgs + chosen_msgs,
        "rejected": prompt_msgs + rejected_msgs,
    }


def main():
    args = parse_args()

    print(f"Training parody model with DPO")
    print(f"  Base model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For generation

    # Quantization config
    quantization_config = None
    if args.use_4bit:
        print("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # LoRA config
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    print(f"  {len(dataset)} examples")

    # Format for DPO
    print("Formatting dataset for DPO...")
    dataset = dataset.map(format_dpo_example, remove_columns=dataset.column_names)

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training config
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = "wandb" if not args.no_wandb and os.environ.get("WANDB_API_KEY") else "none"

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        beta=args.beta,
        loss_type="sigmoid",  # Standard DPO loss
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_ratio=0.1,
        report_to=report_to,
        run_name=f"parody-dpo-{Path(args.model).name}",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Trainer
    print("Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Push to hub
    if args.push_to_hub:
        hub_id = args.hub_model_id or f"parody-{Path(args.model).name}-dpo"
        print(f"Pushing to HuggingFace Hub: {hub_id}")
        trainer.push_to_hub()

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

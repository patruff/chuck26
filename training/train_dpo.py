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
import time
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

from training_report import TrainingReport, generate_comparison_examples


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
        default=32,
        help="LoRA rank (higher=more capacity for creative tasks, 32-64 recommended for humor)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (best practice: 2x rank for stable training)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (10%% recommended for <13B models, 5%% for larger)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="DPO beta (lower=more aggressive preference learning, 0.05 good for humor)",
    )
    parser.add_argument(
        "--target-all-layers",
        action="store_true",
        default=True,
        help="Target all attention+MLP layers (best for creative tasks)",
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
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comprehensive training report with comparisons",
    )
    parser.add_argument(
        "--report-examples",
        type=int,
        default=5,
        help="Number of comparison examples in report",
    )
    parser.add_argument(
        "--gpu-type",
        default="",
        help="GPU type for cost estimation (e.g., rtx3090, a100-40)",
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

    # Initialize report
    report = TrainingReport(
        base_model=args.model,
        output_model=args.output,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
    )

    # Start timing
    start_time = time.time()

    report.add_log("Training parody model with DPO")
    report.add_log(f"Base model: {args.model}")
    report.add_log(f"Dataset: {args.dataset}")
    report.add_log(f"Output: {args.output}")
    report.add_log(f"Epochs: {args.epochs}")

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
        gpu_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram:.1f} GB")
        report.gpu_type = args.gpu_type if args.gpu_type else gpu_name
        report.add_log(f"GPU: {gpu_name} ({vram:.1f} GB)")
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

    # LoRA config - optimized for creative/humor tasks
    # Research shows targeting all attention + MLP layers works best for creative output
    # Higher rank (32-64) gives more capacity for nuanced humor patterns
    # Alpha = 2 * rank is the recommended ratio for stable training
    print("Applying LoRA (optimized for humor/creativity)...")
    report.add_log(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # Target modules for different model architectures
    # Attention layers: capture context and relationships (important for wordplay)
    # MLP/FFN layers: capture creative patterns and vocabulary (critical for humor)
    target_modules = [
        # Attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP/FFN layers - critical for creative output
        "gate_proj", "up_proj", "down_proj",
    ]

    # For models that support it, also target embeddings for better wordplay
    if args.target_all_layers:
        # Some models use different names - include common variants
        target_modules.extend([
            "embed_tokens",  # Input embeddings (helps with vocabulary)
            "lm_head",       # Output head (helps with generation)
            # Alternative names used by some models
            "dense", "fc1", "fc2",
        ])

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        # modules_to_save helps with better creative output
        modules_to_save=["lm_head"] if args.target_all_layers else None,
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    report.add_log(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    print(f"  {len(dataset)} examples")
    report.dataset_size = len(dataset)
    report.add_log(f"Dataset loaded: {len(dataset)} examples")

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
    report.add_log("Starting training...")

    trainer.train()
    report.add_log("Training complete")

    # Record end time
    end_time = time.time()
    report.set_timing(start_time, end_time)
    report.calculate_cost()
    report.add_log(f"Duration: {report.training_duration_human}")
    report.add_log(f"Estimated cost: ${report.estimated_cost:.2f}")

    # Save
    print(f"\nSaving model to {output_dir}")
    report.add_log(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Push to hub
    hub_id = None
    if args.push_to_hub:
        hub_id = args.hub_model_id or f"parody-{Path(args.model).name}-dpo"
        print(f"Pushing to HuggingFace Hub: {hub_id}")
        report.add_log(f"Pushing to HuggingFace Hub: {hub_id}")
        trainer.push_to_hub()
        report.huggingface_url = f"https://huggingface.co/{hub_id}"
        report.output_model = hub_id

    # Generate training report
    if args.generate_report:
        print("\n" + "=" * 60)
        print("Generating training report...")
        print("=" * 60)
        report.add_log("Generating training report with comparisons")

        # Generate comparison examples (base vs fine-tuned)
        try:
            # Free up memory from training
            del trainer, model
            torch.cuda.empty_cache()

            report.comparison_examples = generate_comparison_examples(
                base_model_path=args.model,
                finetuned_model_path=str(output_dir),
                num_examples=args.report_examples,
            )
            report.add_log(f"Generated {len(report.comparison_examples)} comparison examples")
        except Exception as e:
            report.add_log(f"Warning: Could not generate comparisons: {e}")

        # Run evaluation for report
        try:
            from evaluate_parodies import evaluate_model, DEFAULT_TEST_TITLES

            print("\nRunning evaluation for report...")
            report.add_log("Running evaluation...")
            eval_results = evaluate_model(
                str(output_dir),
                titles=DEFAULT_TEST_TITLES[:15],
                use_4bit=args.use_4bit,
            )
            report.eval_total = eval_results.total
            report.eval_passed = eval_results.passed
            report.eval_pass_rate = eval_results.pass_rate
            report.eval_avg_score = eval_results.avg_phonetic_score
            report.add_log(f"Evaluation: {eval_results.passed}/{eval_results.total} passed ({eval_results.pass_rate:.1%})")
        except Exception as e:
            report.add_log(f"Warning: Could not run evaluation: {e}")

        # Save reports
        report_json = output_dir / "training-report.json"
        report_md = output_dir / "training-report.md"
        report.save(report_json)
        report.save_markdown(report_md)
        report.print_summary()

        # Upload report to hub
        if args.push_to_hub and hub_id:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=str(report_json),
                    path_in_repo="training-report.json",
                    repo_id=hub_id,
                )
                api.upload_file(
                    path_or_fileobj=str(report_md),
                    path_in_repo="training-report.md",
                    repo_id=hub_id,
                )
                report.add_log(f"Uploaded reports to {hub_id}")
            except Exception as e:
                report.add_log(f"Warning: Could not upload reports: {e}")

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    if hub_id:
        print(f"HuggingFace: https://huggingface.co/{hub_id}")


if __name__ == "__main__":
    main()

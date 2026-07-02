#!/usr/bin/env python3
"""QLoRA SFT trainer for the reasoning parody model. Runs ON the RunPod pod.

Loads the chat-format reasoning dataset built by build_reasoning_dataset.py
(from the HF Hub or a local JSONL), fine-tunes a Qwen3 base with QLoRA, and
pushes ONLY the LoRA adapter to the Hub.

Usage (on the pod):
    export HF_TOKEN=hf_...   # read dataset + write adapter
    python pipeline/train_reasoning_sft.py \
        --base-model Qwen/Qwen3-8B \
        --dataset-repo patruff/chuckles-reasoning-sft \
        --hf-repo patruff/chuckles-reasoning-adapter \
        --epochs 3

    # Cheap end-to-end validation (0.6B model, 20 steps):
    python pipeline/train_reasoning_sft.py --test \
        --dataset-repo patruff/chuckles-reasoning-sft \
        --hf-repo patruff/chuckles-reasoning-adapter-test

Requires: transformers peft trl datasets accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
import inspect
import os

TEST_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument(
        "--dataset-repo",
        default="patruff/chuckles-reasoning-sft",
        help="HF dataset repo with a 'messages' column (chat format).",
    )
    p.add_argument(
        "--dataset", default="", help="Local JSONL path (overrides --dataset-repo)."
    )
    p.add_argument(
        "--hf-repo",
        default="patruff/chuckles-reasoning-adapter",
        help="HF repo to push the LoRA adapter to.",
    )
    p.add_argument("--output", default="/workspace/reasoning-sft-out")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.add_argument(
        "--test",
        action="store_true",
        help=f"Cheap validation: {TEST_BASE_MODEL}, 20 steps, no packing.",
    )
    return p.parse_args()


def build_sft_config(**kwargs):
    """Build SFTConfig across TRL versions with small argument-name drift."""
    from trl import SFTConfig

    params = inspect.signature(SFTConfig).parameters
    if "max_seq_length" not in params and "max_length" in params and "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return SFTConfig(**filtered)


def main() -> None:
    args = parse_args()

    if args.test:
        args.base_model = TEST_BASE_MODEL
        if args.max_steps <= 0:
            args.max_steps = 20

    if not args.no_push and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN env var is required to push the adapter.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",")],
    )

    if args.dataset:
        ds = load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset_repo, split="train")

    def to_text(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return example

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])
    print(f"Training on {len(ds)} examples; sample:\n{ds[0]['text'][:500]}\n...")

    sft_config = build_sft_config(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=args.max_seq_len,
        packing=not args.test,
        dataset_text_field="text",
        report_to="none",
        push_to_hub=not args.no_push,
        hub_model_id=None if args.no_push else args.hf_repo,
        hub_private_repo=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    if args.no_push:
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Adapter saved locally to {args.output}")
        return

    # Explicit final push so the adapter + tokenizer land even if the last
    # save_strategy checkpoint didn't sync to the Hub.
    trainer.push_to_hub(commit_message="final reasoning SFT adapter")
    tokenizer.push_to_hub(args.hf_repo, private=True)
    print(f"Adapter pushed to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()

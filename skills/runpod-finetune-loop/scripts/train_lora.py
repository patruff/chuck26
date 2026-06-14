#!/usr/bin/env python
"""
train_lora.py — runs ON the RunPod GPU pod.

QLoRA SFT fine-tune that pushes ONLY the LoRA adapter to the Hugging Face Hub.
The adapter_config.json written by PEFT records the base model id, which is what
lets test_inference.py reassemble the model later with a single from_pretrained.

Usage:
    python train_lora.py \
        --base-model meta-llama/Llama-3.1-8B \
        --dataset /workspace/data/train.jsonl \
        --hf-repo youruser/yourmodel-adapter \
        --output /workspace/out \
        --epochs 3

Dataset format: JSONL, one object per line, with a "text" field (plain SFT) OR a
"messages" field (list of {role, content}) for chat-formatted training.
Requires env HF_TOKEN for the push.
"""
import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="HF id of the base model to fine-tune")
    p.add_argument("--dataset", required=True, help="Path to JSONL with a 'text' or 'messages' field")
    p.add_argument("--hf-repo", required=True, help="HF repo id to push the adapter to, e.g. user/model-adapter")
    p.add_argument("--output", default="/workspace/out", help="Local output dir (use the network volume)")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs with a fixed step count if > 0")
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
        help="Comma-separated module names to apply LoRA to",
    )
    p.add_argument("--private", action="store_true", help="Push the adapter repo as private")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN env var is required to push the adapter. Pass it into the pod env.")

    # 4-bit (QLoRA) base — big memory savings, fits 7-13B on a single A100/4090.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",")],
    )

    # Load data. If 'messages' present, render with the chat template; else use 'text'.
    ds = load_dataset("json", data_files=args.dataset, split="train")

    def to_text(example):
        if "messages" in example and example["messages"]:
            example["text"] = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        return example

    if "messages" in ds.column_names:
        ds = ds.map(to_text)

    sft_config = SFTConfig(
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
        packing=True,
        dataset_text_field="text",
        report_to="none",
        # Push the ADAPTER ONLY (PEFT model => push_to_hub uploads adapter weights).
        push_to_hub=True,
        hub_model_id=args.hf_repo,
        hub_private_repo=args.private,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # Final explicit push so the adapter + tokenizer land on the Hub even if
    # save_strategy didn't trigger a hub sync on the last step.
    trainer.push_to_hub(commit_message="final adapter")
    tokenizer.push_to_hub(args.hf_repo)
    print(f"\nAdapter pushed to https://huggingface.co/{args.hf_repo}")
    print("Verify it has adapter_config.json + adapter_model.safetensors, then terminate the pod.")


if __name__ == "__main__":
    main()

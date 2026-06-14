#!/usr/bin/env python
"""
eval_adapter_loss.py — compare base-model loss vs adapter-loaded loss.

This is useful for local smoke tests where a tiny model may generate nonsense
even when the LoRA adapter trained correctly. The pass condition is simple:
loss on the training/eval text should be lower with the adapter attached than
with the base model alone.
"""
import argparse
import math

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True, help="Local path or HF repo id for the LoRA adapter")
    p.add_argument("--dataset", required=True, help="JSONL file with a text column")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--fp32", action="store_true", help="Use float32 for local CPU smoke tests")
    p.add_argument("--min-improvement", type=float, default=0.0, help="Required base_loss - adapter_loss")
    return p.parse_args()


def mean_loss(model, tokenizer, texts, max_length):
    losses = []
    model.eval()
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            out = model(**inputs, labels=labels)
        losses.append(float(out.loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def load_base(model_id, dtype):
    kwargs = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model


def main():
    args = parse_args()
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    ds = load_dataset("json", data_files=args.dataset, split="train")
    if "text" not in ds.column_names:
        raise SystemExit("Dataset must contain a 'text' column.")
    texts = [row["text"] for row in ds]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = load_base(args.base_model, dtype)
    adapter_base = load_base(args.base_model, dtype)
    adapted = PeftModel.from_pretrained(adapter_base, args.adapter)

    base_loss = mean_loss(base, tokenizer, texts, args.max_length)
    adapter_loss = mean_loss(adapted, tokenizer, texts, args.max_length)
    improvement = base_loss - adapter_loss

    print(f"base_loss={base_loss:.6f}")
    print(f"adapter_loss={adapter_loss:.6f}")
    print(f"improvement={improvement:.6f}")
    print(f"base_ppl={math.exp(base_loss):.2f}")
    print(f"adapter_ppl={math.exp(adapter_loss):.2f}")

    if improvement <= args.min_improvement:
        raise SystemExit(
            f"adapter did not improve loss by more than {args.min_improvement}; "
            f"improvement={improvement:.6f}"
        )
    print("RESULT: PASS")


if __name__ == "__main__":
    main()

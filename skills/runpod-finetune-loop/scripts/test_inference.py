#!/usr/bin/env python
"""
test_inference.py — runs ANYWHERE after the pod is gone (cheap pod / local GPU / CPU).

Pulls the LoRA adapter back from the Hugging Face Hub and runs eval prompts to
confirm the fine-tune "took". AutoPeftModelForCausalLM reads the base model id from
the adapter_config.json on the Hub, loads the base, and attaches the adapter — so
you only pass the adapter repo id.

Usage:
    python test_inference.py \
        --hf-repo youruser/yourmodel-adapter \
        --evals eval_prompts.example.json

Eval file format (JSON):
    [
      {"prompt": "...", "expect_contains": ["foo"], "expect_regex": "bar\\d+"},
      {"prompt": "..."}                      # no checks => just prints output
    ]
Both checks are optional per item. Exit code is non-zero if any check fails.
"""
import argparse
import json
import re
import sys

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo", required=True, help="Adapter repo id on the Hub")
    p.add_argument("--evals", required=True, help="Path to JSON list of eval prompts")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-4bit", action="store_true",
                   help="Load the base in 4-bit (match QLoRA training for consistency)")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_model(repo, load_4bit):
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoPeftModelForCausalLM.from_pretrained(repo, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def check(text, item):
    failures = []
    for needle in item.get("expect_contains", []):
        if needle.lower() not in text.lower():
            failures.append(f"missing substring: {needle!r}")
    rx = item.get("expect_regex")
    if rx and not re.search(rx, text):
        failures.append(f"regex no match: {rx!r}")
    return failures


def main():
    args = parse_args()
    with open(args.evals) as f:
        evals = json.load(f)

    print(f"Loading adapter {args.hf_repo} (base resolved from adapter_config.json)...")
    model, tokenizer = load_model(args.hf_repo, args.load_4bit)

    passed = 0
    has_checks = 0
    for i, item in enumerate(evals, 1):
        out = generate(model, tokenizer, item["prompt"], args.max_new_tokens, args.temperature)
        print(f"\n--- eval {i} ---")
        print(f"PROMPT: {item['prompt']}")
        print(f"OUTPUT: {out}")
        if "expect_contains" in item or "expect_regex" in item:
            has_checks += 1
            fails = check(out, item)
            if fails:
                print("RESULT: FAIL -> " + "; ".join(fails))
            else:
                print("RESULT: PASS")
                passed += 1

    if has_checks:
        print(f"\n=== {passed}/{has_checks} checked prompts passed ===")
        sys.exit(0 if passed == has_checks else 1)
    else:
        print("\nNo assertions defined — review outputs manually.")


if __name__ == "__main__":
    main()

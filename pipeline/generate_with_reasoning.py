#!/usr/bin/env python3
"""Generate parody titles with the reasoning SFT model and score them with
the repo's HF smolagents tools.

For each title it:
  1. Pre-computes suggestions with patruff/parody-suggestions (same as
     generator.py) and injects them into the user prompt -- matching how the
     training data was built.
  2. Generates with the fine-tuned model (base + LoRA adapter), letting it
     reason inside <think> tags.
  3. Scores every swapped word pair with patruff/word-phone
     (phone_tool.forward(word=orig, compare_to=swap)) and reports the
     per-swap and average phonetic similarity.

Usage:
    export HF_TOKEN=hf_...
    python pipeline/generate_with_reasoning.py \
        --adapter patruff/chuckles-reasoning-adapter \
        --titles titles.csv --limit 5 --output results.jsonl

Requires: transformers peft accelerate bitsandbytes smolagents pronouncing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    MIN_PHONETIC_SCORE,
    REASONING_SYSTEM_PROMPT,
    build_user_prompt,
    compact_suggestions,
    score_parody,
    split_think_answer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter",
        default="patruff/chuckles-reasoning-adapter",
        help="HF LoRA adapter repo. Pass --no-adapter to run the bare base.",
    )
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument(
        "--base-model",
        default="",
        help="Base model id. Default: read from the adapter's config; "
        "required with --no-adapter.",
    )
    p.add_argument("--titles", default=str(REPO_ROOT / "titles.csv"))
    p.add_argument("--limit", type=int, default=0, help="Max titles (0 = all).")
    p.add_argument("--output", default="", help="Write results JSONL here.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature; 0 = greedy decoding.",
    )
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument(
        "--funny-words",
        default=str(Path(__file__).resolve().parent / "funny_words.json"),
    )
    return p.parse_args()


def read_titles(csv_path: str) -> list[str]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "title" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have a 'title' column")
        return [row["title"].strip() for row in reader if row.get("title", "").strip()]


def load_model(args):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if not torch.cuda.is_available():
        model_kwargs = {"torch_dtype": torch.float32}

    if args.no_adapter:
        if not args.base_model:
            raise SystemExit("--base-model is required with --no-adapter")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    else:
        # Reads the base model id out of adapter_config.json, loads the
        # base, and attaches the adapter.
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    model.eval()
    return model, tokenizer


def generate_one(
    model, tokenizer, title: str, suggestions, max_new_tokens: int, temperature: float
) -> str:
    import torch

    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(title, suggestions)},
    ]
    try:
        # Qwen3 supports enable_thinking; older templates don't take the kwarg
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    sampling = (
        {"do_sample": True, "temperature": temperature, "top_p": 0.9}
        if temperature > 0
        else {"do_sample": False}
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **sampling,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    from chuckles_prime.tools import load_parody_tools, pre_compute_suggestions

    with open(args.funny_words, encoding="utf-8") as f:
        funny_words = json.load(f)

    titles = read_titles(args.titles)
    if args.limit > 0:
        titles = titles[: args.limit]
    print(f"Generating parodies for {len(titles)} titles")

    print("Loading HF tools (patruff/parody-suggestions, patruff/word-phone) ...")
    parody_tool, phone_tool = load_parody_tools()

    print(f"Loading model ({'base only' if args.no_adapter else args.adapter}) ...")
    model, tokenizer = load_model(args)

    results: list[dict] = []
    for i, title in enumerate(titles):
        try:
            raw_sugg = pre_compute_suggestions(title, funny_words, parody_tool)
            suggestions = compact_suggestions(raw_sugg)
            generated = generate_one(
                model, tokenizer, title, suggestions,
                args.max_new_tokens, args.temperature,
            )
            reasoning, parody = split_think_answer(generated)
            # The answer should be a single title line
            parody = parody.strip().splitlines()[0].strip() if parody.strip() else ""
            swap_scores, avg = score_parody(phone_tool, title, parody)
            ok = avg >= MIN_PHONETIC_SCORE if swap_scores else False
            results.append(
                {
                    "input_title": title,
                    "parody": parody,
                    "swap_scores": swap_scores,
                    "avg_phonetic_score": round(avg, 3),
                    "passes_threshold": ok,
                    "reasoning": reasoning,
                }
            )
            print(
                f"[{i + 1}/{len(titles)}] {title!r} -> {parody!r} "
                f"(avg score {avg:.3f}{'' if ok else ' -- below threshold'})"
            )
        except Exception as e:
            results.append({"input_title": title, "error": str(e)})
            print(f"[{i + 1}/{len(titles)}] {title!r} ERROR: {e}")

    scored = [r for r in results if "avg_phonetic_score" in r]
    if scored:
        overall = sum(r["avg_phonetic_score"] for r in scored) / len(scored)
        passing = sum(1 for r in scored if r["passes_threshold"])
        print(
            f"\n{len(scored)}/{len(results)} generated | "
            f"mean phonetic score {overall:.3f} | "
            f"{passing} pass the {MIN_PHONETIC_SCORE} threshold"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

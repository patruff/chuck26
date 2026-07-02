#!/usr/bin/env python3
"""Build a reasoning SFT dataset from human-approved parody pairs.

Runs on CPU (a GitHub Actions runner is fine). For each (title, parody)
pair it:
  1. Pre-computes parody word suggestions with the patruff/parody-suggestions
     HF tool -- the same tool the generation agent uses.
  2. Verifies each actual word swap with the patruff/word-phone tool.
  3. Injects the suggestions into the user prompt (matching inference) and
     synthesizes a <think> reasoning trace ending in the known-good parody.

Output rows are chat-format {"messages": [...]}, compatible with
train_reasoning_sft.py and skills/runpod-finetune-loop/scripts/train_lora.py.

Usage:
    export HF_TOKEN=hf_...   # read access to source, write to --dataset-repo
    python pipeline/build_reasoning_dataset.py \
        --source patruff/chucklesClean720 \
        --dataset-repo patruff/chuckles-reasoning-sft \
        --output data/reasoning_sft.jsonl

Requires: pip install smolagents pronouncing datasets huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    REASONING_SYSTEM_PROMPT,
    align_swaps,
    build_reasoning_trace,
    build_user_prompt,
    compact_suggestions,
    parse_alpaca_text,
    phonetic_similarity,
    split_output_explanation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        default="patruff/chucklesClean720",
        help="HF dataset with (title, parody) pairs. Supports alpaca 'text' "
        "rows, 'input'/'output' columns, or DPO 'prompt'/'chosen' columns.",
    )
    p.add_argument(
        "--dataset-repo",
        default="patruff/chuckles-reasoning-sft",
        help="HF dataset repo to push the built SFT data to (private).",
    )
    p.add_argument("--output", default="", help="Also write a local JSONL here.")
    p.add_argument("--limit", type=int, default=0, help="Max pairs (0 = all).")
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop pairs whose average verified swap similarity is below this.",
    )
    p.add_argument(
        "--funny-words",
        default=str(Path(__file__).resolve().parent / "funny_words.json"),
        help="Path to funny words JSON (categories -> word lists).",
    )
    p.add_argument("--no-push", action="store_true", help="Skip the HF push.")
    return p.parse_args()


def extract_pairs(ds) -> list[tuple[str, str, str]]:
    """Extract (title, parody, humor_note) triples from any supported schema.

    humor_note is '' for sources without explanations.
    """
    cols = ds.column_names
    pairs: list[tuple[str, str, str]] = []
    if "input" in cols and "output" in cols:
        # e.g. patruff/chucklesClean2WordsALPACA: output holds
        # 'Parody Title: <humor explanation>'
        for row in ds:
            if row["input"] and row["output"]:
                parody, note = split_output_explanation(row["output"].strip())
                if parody:
                    pairs.append((row["input"].strip(), parody, note))
    elif "prompt" in cols and "chosen" in cols:
        for row in ds:
            prompt, chosen = row["prompt"], row["chosen"]
            if isinstance(prompt, list):  # chat format
                prompt = next(
                    (m["content"] for m in prompt if m["role"] == "user"), ""
                )
            if isinstance(chosen, list):
                chosen = next(
                    (m["content"] for m in chosen if m["role"] == "assistant"), ""
                )
            # Pull the title out of "... parody of: 'TITLE'" style prompts
            title = prompt
            for sep in (":'", ": '", ":\n"):
                if sep in prompt:
                    title = prompt.split(sep)[-1].strip().strip("'\"")
                    break
            if title and chosen:
                pairs.append((title.strip(), chosen.strip(), ""))
    elif "text" in cols:
        for row in ds:
            parsed = parse_alpaca_text(row["text"])
            if parsed:
                pairs.append((*parsed, ""))
    else:
        raise ValueError(f"Unsupported dataset schema: {cols}")
    return pairs


def main() -> None:
    args = parse_args()

    from datasets import Dataset, load_dataset

    from chuckles_prime.tools import load_parody_tools, pre_compute_suggestions

    with open(args.funny_words, encoding="utf-8") as f:
        funny_words = json.load(f)

    print(f"Loading source dataset {args.source} ...")
    ds = load_dataset(args.source, split="train")
    pairs = extract_pairs(ds)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"Extracted {len(pairs)} (title, parody) pairs")

    print("Loading HF tools (patruff/parody-suggestions, patruff/word-phone) ...")
    parody_tool, phone_tool = load_parody_tools()

    rows: list[dict] = []
    skipped = 0
    for i, (title, parody, humor_note) in enumerate(pairs):
        try:
            raw_sugg = pre_compute_suggestions(title, funny_words, parody_tool)
            suggestions = compact_suggestions(raw_sugg)

            swap_scores: dict[str, float] = {}
            for ow, pw in align_swaps(title, parody):
                sim = phonetic_similarity(phone_tool, ow, pw)
                if sim is not None:
                    swap_scores[f"{ow}->{pw}"] = sim

            avg = (
                sum(swap_scores.values()) / len(swap_scores)
                if swap_scores
                else 0.0
            )
            if args.min_score > 0 and avg < args.min_score:
                skipped += 1
                continue

            assistant = build_reasoning_trace(
                title, parody, suggestions, swap_scores, humor_note
            )
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(title, suggestions)},
                        {"role": "assistant", "content": assistant},
                    ],
                    "input_title": title,
                    "parody": parody,
                    "avg_swap_score": round(avg, 3),
                }
            )
        except Exception as e:
            skipped += 1
            print(f"  [skip] {title!r}: {e}")

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(pairs)}")

    print(f"Built {len(rows)} rows ({skipped} skipped)")
    if not rows:
        raise SystemExit("No rows built -- check the source dataset.")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}")

    if not args.no_push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN env var required to push (or use --no-push)")
        dataset = Dataset.from_list(rows)
        dataset.push_to_hub(args.dataset_repo, split="train", private=True, token=token)
        print(f"Pushed {len(rows)} rows to https://huggingface.co/datasets/{args.dataset_repo}")


if __name__ == "__main__":
    main()

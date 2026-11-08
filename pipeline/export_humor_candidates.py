"""Export reviewed or review-pending Chuckles turns as prompt/completion pairs."""
import argparse
import hashlib
import json
from pathlib import Path

from common import clean_word


def outside_nominations(record):
    """Identify pair checks made outside the slate active at that point."""
    original = record["input_title"].split()
    rounds = record["draft_rounds"]
    outside = []
    for i, c in enumerate(record["tool_calls"]):
        current = next((rd for rd in reversed(rounds) if rd["prior_phone_calls"] <= i), None)
        pairs = {(clean_word(o), clean_word(r)) for title in (current["finalists"] if current else [])
                 for o, r in zip(original, title.split())}
        if (c["word"], c["compare_to"]) not in pairs:
            outside.append(c)
    return outside


def candidate_turns(record, review_status="unreviewed", token_count=None, thinking_budget=None):
    """Keep complete, clean, phonetic-pass runs. Humor still needs a human label."""
    active = record.get("steps", [])
    original_creativity = record.get("prior_creative_steps", [])
    steps = original_creativity + active
    drafts = record.get("draft_rounds", [])
    if (record.get("state") != "success" or record.get("error") or not record.get("passes_threshold")
            or not drafts or drafts[0]["prior_phone_calls"] or not active or not active[-1]["final"]):
        return
    if any(s.get("error") or s.get("finish_reason") != "stop" or not s.get("input_messages")
           or not (s.get("reasoning") or "").strip() or not (s.get("code") or "").strip() for s in steps):
        return
    if not record.get("tool_calls") or any(not c["visible_in_observation"] for c in record["tool_calls"]):
        return
    if outside_nominations(record):
        return
    for index, step in enumerate(steps):
        n_tokens = token_count(step["reasoning"]) if token_count is not None else None
        # Template whitespace and the closing tag can account for a few tokens.
        # Keep potentially budget-cut thoughts in raw traces, away from positive targets.
        if n_tokens is not None and thinking_budget is not None and n_tokens >= thinking_budget - 4:
            continue
        # CodeAgent executes the first closed action. Discard text emitted after that action.
        text, closing, _ = step["model_output"].partition("</code>")
        reused = index < len(original_creativity)
        yield {
            "input_title": record["input_title"], "parody": record["parody"], "step": step["step"],
            "model": record["model"], "seed": record.get("prior_creative_seed", 42) if reused else record["seed"],
            "review_status": review_status, "trajectory": "original_creative_pass" if reused else "batch_pass",
            "reasoning_tokens": n_tokens, "thinking_budget_checked": n_tokens is not None and thinking_budget is not None,
            "prompt": step["input_messages"],
            "completion": [{"role": "assistant", "reasoning_content": step["reasoning"], "content": text + closing}],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviews", type=Path, help="Export only titles marked keep for this raw results file")
    parser.add_argument("--tokenizer-json", type=Path, help="Optional matching HF tokenizer for screening near-budget traces")
    parser.add_argument("--thinking-budget", type=int, default=1536)
    args = parser.parse_args()
    token_count = None
    if args.tokenizer_json:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        def token_count(text):
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
    kept = None
    if args.reviews:
        digest = hashlib.sha256(args.results.read_bytes()).hexdigest()
        labels = [json.loads(line) for line in args.reviews.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(label["run_sha256"] != digest for label in labels):
            parser.error("Review labels belong to a different results file")
        kept = {label["input_title"] for label in labels if label["review_status"] == "keep"}
    with args.results.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as out:
        for line in source:
            record = json.loads(line)
            if kept is not None and record["input_title"] not in kept:
                continue
            for turn in candidate_turns(record, "keep" if kept is not None else "unreviewed", token_count, args.thinking_budget):
                out.write(json.dumps(turn, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

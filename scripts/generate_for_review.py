"""Generate parodies and write a review CSV for human approval.

Runs the chuckles generation pipeline against an input CSV of titles,
then writes a review CSV with columns for each parody candidate, its
reasoning, phonetic scores, and provenance (model + adapter). The
reviewer edits the 'status' column to 'chosen' or 'rejected' for each
row, forming DPO preference pairs.

Usage:
    python scripts/generate_for_review.py titles.csv \
        --settings settings.json \
        --model qwen-3-32b \
        --adapter ""
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")


def generate_review_csv(
    input_csv: str,
    settings: str,
    output_dir: str,
    model_override: str | None = None,
    adapter: str = "",
) -> Path:
    """Run generation and write a review CSV.

    Args:
        input_csv: Path to CSV with a 'title' column.
        settings: Path to settings.json.
        output_dir: Directory to write the review CSV.
        model_override: Override the model name from settings.
        adapter: Adapter/LoRA name if any (for provenance tracking).

    Returns:
        Path to the written review CSV.
    """
    from chuckles_prime import config as _config
    from chuckles_prime import generator as _generator
    from chuckles_prime import model as _model
    from chuckles_prime import tools as _tools
    from chuckles_prime import traces as _traces

    # Load config
    config = _config.load_config(settings)

    # Read titles
    titles = _generator.read_input_titles(input_csv)
    print(f"Loaded {len(titles)} titles from {input_csv}")

    # Create model and tools
    model = _model.create_model(config)
    model_name = model_override or config.model_name
    parody_tool, phone_tool = _tools.load_parody_tools()
    agent = _generator.create_agent(model, parody_tool, phone_tool)

    # Generate
    records = _generator.generate_batch(titles, agent, parody_tool, config)

    # Also archive raw traces for later use
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    traces_path = out / f"traces-{ts}.jsonl"
    _traces.archive_traces(records, traces_path)
    print(f"Archived traces to {traces_path}")

    # Build review CSV rows -- one row per candidate
    csv_path = out / f"review-{ts}.csv"
    fieldnames = [
        "id",
        "input_title",
        "parody_text",
        "humor_note",
        "phonetic_scores",
        "avg_phonetic_score",
        "model_name",
        "adapter",
        "status",
    ]

    row_id = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            if rec.error:
                # Write error rows so reviewer can see failures
                row_id += 1
                writer.writerow({
                    "id": row_id,
                    "input_title": rec.input_title,
                    "parody_text": f"[ERROR: {rec.error}]",
                    "humor_note": "",
                    "phonetic_scores": "",
                    "avg_phonetic_score": "",
                    "model_name": model_name,
                    "adapter": adapter,
                    "status": "error",
                })
                continue

            for candidate in rec.candidates:
                row_id += 1
                scores = candidate.phonetic_scores
                avg_score = (
                    sum(scores.values()) / len(scores) if scores else 0.0
                )
                writer.writerow({
                    "id": row_id,
                    "input_title": rec.input_title,
                    "parody_text": candidate.text,
                    "humor_note": candidate.humor_note,
                    "phonetic_scores": json.dumps(scores),
                    "avg_phonetic_score": f"{avg_score:.3f}",
                    "model_name": model_name,
                    "adapter": adapter,
                    "status": "pending",
                })

    print(f"Wrote {row_id} rows to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate parodies and write a review CSV"
    )
    parser.add_argument("input", help="Path to input CSV with 'title' column")
    parser.add_argument(
        "--settings",
        default="settings.json",
        help="Path to settings JSON (default: settings.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="reviews/pending",
        help="Directory for review CSV (default: reviews/pending/)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name for provenance tracking",
    )
    parser.add_argument(
        "--adapter",
        default="",
        help="Adapter/LoRA name for provenance tracking (default: none)",
    )
    args = parser.parse_args()

    csv_path = generate_review_csv(
        input_csv=args.input,
        settings=args.settings,
        output_dir=args.output_dir,
        model_override=args.model,
        adapter=args.adapter,
    )
    print(f"Review CSV ready: {csv_path}")


if __name__ == "__main__":
    main()

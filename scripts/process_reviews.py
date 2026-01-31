"""Process reviewed CSVs into DPO preference data and push to HuggingFace.

Reads review CSVs where a human has marked rows as 'chosen' or 'rejected'.
Builds DPO preference pairs from rows sharing the same input_title, then
pushes the resulting dataset to HuggingFace Hub. Moves processed CSVs
to reviews/processed/.

DPO pair formation:
- For each input_title, all 'chosen' rows are paired with all 'rejected'
  rows to form preference pairs.
- Rows left as 'pending' are ignored.
- Provenance (model_name, adapter) is preserved as metadata.

Usage:
    python scripts/process_reviews.py \
        --dpo-repo patruff/chuckles-dpo \
        --reviews-dir reviews/pending
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATASET_SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Replace words with phonetically similar but humorous alternatives. "
    "Use the phonetic analysis tool to verify similarity scores above 0.6."
)


def load_review_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load a review CSV and return rows as dicts."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_dpo_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Build DPO preference pairs from reviewed rows.

    Groups rows by input_title, then creates a pair for every
    (chosen, rejected) combination within each group.

    Args:
        rows: List of row dicts from review CSVs.

    Returns:
        List of DPO row dicts with prompt, chosen, rejected, and metadata.
    """
    # Group by input_title
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"chosen": [], "rejected": []}
    )

    for row in rows:
        status = row.get("status", "").strip().lower()
        if status in ("chosen", "rejected"):
            groups[row["input_title"]][status].append(row)

    # Form pairs
    dpo_rows: list[dict[str, Any]] = []
    for input_title, group in groups.items():
        for chosen in group["chosen"]:
            for rejected in group["rejected"]:
                dpo_rows.append({
                    "prompt": [
                        {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Create a phonetically-sound parody of: '{input_title}'",
                        },
                    ],
                    "chosen": [
                        {"role": "assistant", "content": chosen["parody_text"]},
                    ],
                    "rejected": [
                        {"role": "assistant", "content": rejected["parody_text"]},
                    ],
                    # Metadata for provenance
                    "chosen_model": chosen.get("model_name", ""),
                    "chosen_adapter": chosen.get("adapter", ""),
                    "chosen_phonetic_score": chosen.get("avg_phonetic_score", ""),
                    "chosen_humor_note": chosen.get("humor_note", ""),
                    "rejected_model": rejected.get("model_name", ""),
                    "rejected_adapter": rejected.get("adapter", ""),
                    "rejected_phonetic_score": rejected.get("avg_phonetic_score", ""),
                    "rejected_humor_note": rejected.get("humor_note", ""),
                })

    return dpo_rows


def process_reviews(
    reviews_dir: str,
    processed_dir: str,
    dpo_repo: str | None = None,
    no_push: bool = False,
    append: bool = True,
) -> dict[str, Any]:
    """Process all review CSVs in reviews_dir.

    Args:
        reviews_dir: Directory containing reviewed CSVs.
        processed_dir: Directory to move processed CSVs to.
        dpo_repo: HuggingFace Hub repo ID for DPO dataset.
        no_push: If True, build dataset but don't push.
        append: If True, pull existing dataset and append new rows.

    Returns:
        Summary dict with counts.
    """
    from datasets import Dataset, load_dataset
    from huggingface_hub import login

    reviews_path = Path(reviews_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Find all review CSVs
    csv_files = sorted(reviews_path.glob("review-*.csv"))
    if not csv_files:
        print("No review CSVs found in", reviews_dir)
        return {"files": 0, "pairs": 0, "pushed": False}

    # Collect all rows from all CSVs
    all_rows: list[dict[str, str]] = []
    for csv_file in csv_files:
        rows = load_review_csv(csv_file)
        # Only process files that have at least one chosen or rejected row
        statuses = {r.get("status", "").strip().lower() for r in rows}
        if "chosen" in statuses or "rejected" in statuses:
            all_rows.extend(rows)
            print(f"Loaded {len(rows)} rows from {csv_file.name}")
        else:
            print(f"Skipping {csv_file.name} (no chosen/rejected rows)")

    if not all_rows:
        print("No reviewed rows found")
        return {"files": len(csv_files), "pairs": 0, "pushed": False}

    # Build DPO pairs
    dpo_rows = build_dpo_rows(all_rows)
    print(f"Built {len(dpo_rows)} DPO preference pairs")

    # Count summary stats
    chosen_count = sum(
        1 for r in all_rows if r.get("status", "").strip().lower() == "chosen"
    )
    rejected_count = sum(
        1 for r in all_rows if r.get("status", "").strip().lower() == "rejected"
    )
    pending_count = sum(
        1 for r in all_rows if r.get("status", "").strip().lower() == "pending"
    )
    print(
        f"Review stats: {chosen_count} chosen, {rejected_count} rejected, "
        f"{pending_count} still pending"
    )

    pushed = False
    if dpo_rows and dpo_repo and not no_push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: HF_TOKEN not set, cannot push to Hub")
        else:
            login(token=token)

            new_ds = Dataset.from_list(dpo_rows)

            if append:
                # Try to load existing dataset and append
                try:
                    existing = load_dataset(dpo_repo, split="train")
                    # Merge: convert both to lists, combine, rebuild
                    existing_rows = existing.to_list()
                    combined = existing_rows + dpo_rows
                    final_ds = Dataset.from_list(combined)
                    print(
                        f"Appending {len(dpo_rows)} new pairs to "
                        f"{len(existing_rows)} existing ({len(combined)} total)"
                    )
                except Exception:
                    # Dataset doesn't exist yet, use new only
                    final_ds = new_ds
                    print(f"Creating new dataset with {len(dpo_rows)} pairs")
            else:
                final_ds = new_ds

            final_ds.push_to_hub(dpo_repo, split="train", private=True)
            pushed = True
            print(f"Pushed DPO dataset to {dpo_repo}")

    elif dpo_rows and no_push:
        print(f"Built {len(dpo_rows)} pairs (--no-push, skipping Hub push)")

    # Move processed CSVs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for csv_file in csv_files:
        rows = load_review_csv(csv_file)
        statuses = {r.get("status", "").strip().lower() for r in rows}
        if "chosen" in statuses or "rejected" in statuses:
            dest = processed_path / f"{csv_file.stem}-done-{ts}.csv"
            shutil.move(str(csv_file), str(dest))
            print(f"Moved {csv_file.name} -> {dest.name}")

    return {
        "files": len(csv_files),
        "pairs": len(dpo_rows),
        "chosen": chosen_count,
        "rejected": rejected_count,
        "pushed": pushed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process reviewed CSVs into DPO preference data"
    )
    parser.add_argument(
        "--reviews-dir",
        default="reviews/pending",
        help="Directory containing review CSVs (default: reviews/pending/)",
    )
    parser.add_argument(
        "--processed-dir",
        default="reviews/processed",
        help="Directory for processed CSVs (default: reviews/processed/)",
    )
    parser.add_argument(
        "--dpo-repo",
        default=None,
        help="HF Hub repo ID for DPO dataset (e.g. patruff/chuckles-dpo)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build dataset but skip pushing to Hub",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Replace existing dataset instead of appending",
    )
    args = parser.parse_args()

    result = process_reviews(
        reviews_dir=args.reviews_dir,
        processed_dir=args.processed_dir,
        dpo_repo=args.dpo_repo,
        no_push=args.no_push,
        append=not args.no_append,
    )
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()

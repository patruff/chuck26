"""JSONL trace archival for generation records.

Preserves full reasoning traces as one JSON line per GenerationRecord
for debugging, analysis, and future training data curation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from chuckles_prime.types import GenerationRecord


def archive_traces(records: list[GenerationRecord], output_path: Path) -> int:
    """Archive full reasoning traces as JSONL.

    One JSON line per GenerationRecord. Preserves all fields including
    the full AgentTrace with step-by-step reasoning.

    Args:
        records: List of generation records to archive.
        output_path: Path for the output JSONL file.

    Returns:
        Number of records written.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            record_dict = asdict(rec)
            f.write(json.dumps(record_dict, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count

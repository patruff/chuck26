"""GRPO and DPO dataset converters with HuggingFace Hub push.

Converts GenerationRecord objects into TRL-compatible HuggingFace datasets:
- GRPO: prompt-only with auxiliary metadata columns for reward functions
- DPO: preference pairs with human chosen vs model rejected
"""

from __future__ import annotations

import json
import os
from typing import Any

from datasets import Dataset
from huggingface_hub import login

from chuckles_prime.rewards import (
    compute_phonetic_quality,
    compute_structure_preservation,
    compute_tool_usage_completeness,
)
from chuckles_prime.types import GenerationRecord

DATASET_SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Replace words with phonetically similar but humorous alternatives. "
    "Use the phonetic analysis tool to verify similarity scores above 0.6."
)


def records_to_grpo_dataset(records: list[GenerationRecord]) -> Dataset:
    """Convert GenerationRecord list to TRL GRPO prompt-only dataset.

    Skips records with errors. Computes composite reward metadata
    as auxiliary columns for downstream reward functions.

    Args:
        records: List of generation records from the engine.

    Returns:
        HuggingFace Dataset with prompt and auxiliary columns.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.error is not None:
            continue

        # Compute aggregate scores
        phonetic_scores_list = [
            compute_phonetic_quality(c) for c in rec.candidates
        ]
        avg_phonetic = (
            sum(phonetic_scores_list) / len(phonetic_scores_list)
            if phonetic_scores_list
            else 0.0
        )

        avg_tool_usage = compute_tool_usage_completeness(rec.trace, rec.input_title)

        structure_scores = [
            compute_structure_preservation(rec.input_title, c.text)
            for c in rec.candidates
        ]
        avg_structure = (
            sum(structure_scores) / len(structure_scores)
            if structure_scores
            else 0.0
        )

        rows.append({
            "prompt": [
                {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a phonetically-sound parody of: '{rec.input_title}'"},
            ],
            "original_title": rec.input_title,
            "phonetic_scores": json.dumps(
                {c.text: c.phonetic_scores for c in rec.candidates}
            ),
            "generation_model": rec.model_name,
            "avg_phonetic_score": avg_phonetic,
            "avg_tool_usage": avg_tool_usage,
            "avg_structure_preservation": avg_structure,
        })

    return Dataset.from_list(rows)


def build_dpo_dataset(
    human_examples: list[tuple[str, str, str]],
    model_records: dict[str, GenerationRecord],
) -> Dataset:
    """Build DPO preference dataset pairing human chosen vs model rejected.

    Only includes pairs where both human example and model output exist
    for the same input title.

    Args:
        human_examples: List of (input_title, human_output, explanation) tuples.
        model_records: Dict mapping input_title to GenerationRecord.

    Returns:
        HuggingFace Dataset with prompt, chosen, and rejected columns.
    """
    rows: list[dict[str, Any]] = []
    for input_title, human_output, explanation in human_examples:
        record = model_records.get(input_title)
        if not record or not record.candidates:
            continue

        # Select worst candidate as rejected (use reward function for robust scoring)
        worst = min(
            record.candidates,
            key=lambda c: compute_phonetic_quality(c),
        )

        rows.append({
            "prompt": [
                {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a phonetically-sound parody of: '{input_title}'"},
            ],
            "chosen": [
                {"role": "assistant", "content": human_output},
            ],
            "rejected": [
                {"role": "assistant", "content": worst.text},
            ],
        })

    return Dataset.from_list(rows)


def push_dataset(
    dataset: Dataset,
    repo_id: str,
    split: str = "train",
    private: bool = True,
) -> None:
    """Push a dataset to HuggingFace Hub.

    Requires HF_TOKEN environment variable with write permission.

    Args:
        dataset: HuggingFace Dataset to push.
        repo_id: Hub repository ID (e.g., "username/dataset-name").
        split: Dataset split name.
        private: Whether the dataset should be private.

    Raises:
        ValueError: If HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Create a write token at https://huggingface.co/settings/tokens"
        )
    login(token=token)
    dataset.push_to_hub(repo_id, split=split, private=private)

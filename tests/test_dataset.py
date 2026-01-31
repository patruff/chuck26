"""Tests for GRPO/DPO dataset converters and Hub push."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from datasets import Dataset

from chuckles_prime.dataset import (
    DATASET_SYSTEM_PROMPT,
    build_dpo_dataset,
    push_dataset,
    records_to_grpo_dataset,
)
from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(title: str = "The Matrix", error: str | None = None) -> GenerationRecord:
    """Create a minimal but realistic GenerationRecord for testing."""
    return GenerationRecord(
        input_title=title,
        candidates=[
            ParodyCandidate(
                text="The Mattress",
                phonetic_scores={"Matrix": 0.78},
                humor_note="comfy",
            ),
            ParodyCandidate(
                text="The Maitricks",
                phonetic_scores={"Matrix": 0.65},
                humor_note="sneaky",
            ),
        ],
        trace=AgentTrace(
            steps=[{"tool_call": "word_phonetic_analyzer", "args": "Matrix"}],
            final_output="{}",
            token_usage=None,
            state="success",
        ),
        model_name="test-model",
        error=error,
    )


# ---------------------------------------------------------------------------
# GRPO converter tests
# ---------------------------------------------------------------------------


def test_grpo_single_valid_record():
    """Single valid record produces Dataset with 1 row and correct columns."""
    ds = records_to_grpo_dataset([_make_record()])
    assert isinstance(ds, Dataset)
    assert ds.num_rows == 1

    row = ds[0]
    assert isinstance(row["prompt"], list)
    assert len(row["prompt"]) == 2
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    assert row["original_title"] == "The Matrix"
    assert isinstance(row["phonetic_scores"], str)
    assert row["generation_model"] == "test-model"


def test_grpo_skips_errored_records():
    """Record with error is skipped (0 rows)."""
    ds = records_to_grpo_dataset([_make_record(error="timeout")])
    assert ds.num_rows == 0


def test_grpo_mixed_records():
    """Multiple records with one errored -> only valid records in dataset."""
    records = [
        _make_record("The Matrix"),
        _make_record("Gladiator", error="timeout"),
        _make_record("Inception"),
    ]
    ds = records_to_grpo_dataset(records)
    assert ds.num_rows == 2


def test_grpo_avg_phonetic_score():
    """Verify avg_phonetic_score is correct float."""
    ds = records_to_grpo_dataset([_make_record()])
    row = ds[0]
    # Candidates have scores 0.78 and 0.65, averages are 0.78 and 0.65
    # avg_phonetic_score = (0.78 + 0.65) / 2 = 0.715
    assert abs(row["avg_phonetic_score"] - 0.715) < 0.01


def test_grpo_avg_structure_preservation():
    """Verify avg_structure_preservation is correct float."""
    ds = records_to_grpo_dataset([_make_record()])
    row = ds[0]
    # "The Matrix" (2 words) vs "The Mattress" (2 words) = 1.0
    # "The Matrix" (2 words) vs "The Maitricks" (2 words) = 1.0
    # avg = 1.0
    assert row["avg_structure_preservation"] == 1.0


def test_grpo_phonetic_scores_is_json_string():
    """phonetic_scores column is a valid JSON string, not a nested dict."""
    ds = records_to_grpo_dataset([_make_record()])
    scores_str = ds[0]["phonetic_scores"]
    assert isinstance(scores_str, str)
    parsed = json.loads(scores_str)
    assert isinstance(parsed, dict)
    assert "The Mattress" in parsed


# ---------------------------------------------------------------------------
# DPO converter tests
# ---------------------------------------------------------------------------


def test_dpo_matching_pair():
    """Matching human example + model record -> 1 row with correct structure."""
    human_examples = [("The Matrix", "The Mattress", "comfy pun")]
    model_records = {"The Matrix": _make_record("The Matrix")}

    ds = build_dpo_dataset(human_examples, model_records)
    assert isinstance(ds, Dataset)
    assert ds.num_rows == 1

    row = ds[0]
    assert isinstance(row["prompt"], list)
    assert row["prompt"][0]["role"] == "system"
    assert isinstance(row["chosen"], list)
    assert row["chosen"][0]["role"] == "assistant"
    assert row["chosen"][0]["content"] == "The Mattress"
    assert isinstance(row["rejected"], list)
    assert row["rejected"][0]["role"] == "assistant"


def test_dpo_no_matching_record():
    """Human example without matching model record -> skipped (0 rows)."""
    human_examples = [("Gladiator", "Fladiator", "funny")]
    model_records = {"The Matrix": _make_record("The Matrix")}

    ds = build_dpo_dataset(human_examples, model_records)
    assert ds.num_rows == 0


def test_dpo_partial_matches():
    """Multiple human examples, partial matches -> only matching pairs."""
    human_examples = [
        ("The Matrix", "The Mattress", "comfy"),
        ("Gladiator", "Fladiator", "funny"),
        ("Inception", "Inseption", "lisp"),
    ]
    model_records = {
        "The Matrix": _make_record("The Matrix"),
        "Inception": _make_record("Inception"),
    }

    ds = build_dpo_dataset(human_examples, model_records)
    assert ds.num_rows == 2


def test_dpo_selects_worst_candidate():
    """Worst candidate (lowest avg phonetic score) selected as rejected."""
    human_examples = [("The Matrix", "Human Parody", "human")]
    model_records = {"The Matrix": _make_record("The Matrix")}

    ds = build_dpo_dataset(human_examples, model_records)
    row = ds[0]
    # "The Maitricks" has score 0.65, "The Mattress" has 0.78
    # Worst = "The Maitricks"
    assert row["rejected"][0]["content"] == "The Maitricks"


# ---------------------------------------------------------------------------
# Push function test
# ---------------------------------------------------------------------------


def test_push_missing_hf_token(monkeypatch):
    """Missing HF_TOKEN raises ValueError."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    ds = records_to_grpo_dataset([_make_record()])
    with pytest.raises(ValueError, match="HF_TOKEN"):
        push_dataset(ds, "test/repo")


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


def test_grpo_roundtrip_smoke():
    """Full round-trip: create multiple records, convert to GRPO Dataset.

    Verifies datasets.Dataset.from_list() accepts our data shapes
    without Arrow serialization errors.
    """
    records = [
        _make_record("The Matrix"),
        _make_record("Gladiator"),
        _make_record("Inception", error="timeout"),
    ]
    ds = records_to_grpo_dataset(records)

    assert isinstance(ds, Dataset)
    assert ds.num_rows == 2  # One errored record skipped

    expected_columns = {
        "prompt", "original_title", "phonetic_scores",
        "generation_model", "avg_phonetic_score",
        "avg_tool_usage", "avg_structure_preservation",
    }
    assert expected_columns.issubset(set(ds.column_names))

    for i in range(ds.num_rows):
        row = ds[i]
        # prompt is list of 2 message dicts
        assert isinstance(row["prompt"], list)
        assert len(row["prompt"]) == 2
        for msg in row["prompt"]:
            assert "role" in msg
            assert "content" in msg

        # phonetic_scores is valid JSON string
        parsed = json.loads(row["phonetic_scores"])
        assert isinstance(parsed, dict)


def test_dpo_roundtrip_smoke():
    """Full round-trip: create DPO Dataset from human examples + model records.

    Verifies datasets.Dataset.from_list() accepts our DPO data shapes.
    """
    human_examples = [
        ("The Matrix", "The Mattress", "comfy"),
        ("Gladiator", "Fladiator", "funny"),
    ]
    model_records = {
        "The Matrix": _make_record("The Matrix"),
        "Gladiator": _make_record("Gladiator"),
    }

    ds = build_dpo_dataset(human_examples, model_records)

    assert isinstance(ds, Dataset)
    assert ds.num_rows == 2

    expected_columns = {"prompt", "chosen", "rejected"}
    assert expected_columns.issubset(set(ds.column_names))

    for i in range(ds.num_rows):
        row = ds[i]
        # chosen and rejected are assistant-only messages
        assert len(row["chosen"]) == 1
        assert row["chosen"][0]["role"] == "assistant"
        assert len(row["rejected"]) == 1
        assert row["rejected"][0]["role"] == "assistant"

"""Tests for JSONL trace archival."""

from __future__ import annotations

import json
from pathlib import Path

from chuckles_prime.traces import archive_traces
from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate


def _make_record(title: str = "The Matrix") -> GenerationRecord:
    """Create a minimal but realistic GenerationRecord for testing."""
    return GenerationRecord(
        input_title=title,
        candidates=[
            ParodyCandidate(
                text="The Mattress",
                phonetic_scores={"Matrix": 0.78},
                humor_note="comfy",
            ),
        ],
        trace=AgentTrace(
            steps=[{"model_output": "thinking about Matrix..."}],
            final_output='{"parody1": "The Mattress"}',
            token_usage={"input_tokens": 100, "output_tokens": 50},
            state="success",
        ),
        model_name="test-model",
    )


def test_archive_empty_list(tmp_path: Path):
    """Empty records list creates file, returns 0."""
    output = tmp_path / "traces.jsonl"
    count = archive_traces([], output)
    assert count == 0
    assert output.exists()
    assert output.read_text() == ""


def test_archive_single_record(tmp_path: Path):
    """Single record produces 1 line of valid JSON containing input_title."""
    output = tmp_path / "traces.jsonl"
    count = archive_traces([_make_record()], output)
    assert count == 1

    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["input_title"] == "The Matrix"
    assert data["model_name"] == "test-model"
    assert len(data["candidates"]) == 1


def test_archive_multiple_records(tmp_path: Path):
    """Multiple records produce N lines, each valid JSON."""
    records = [_make_record("The Matrix"), _make_record("Gladiator"), _make_record("Inception")]
    output = tmp_path / "traces.jsonl"
    count = archive_traces(records, output)
    assert count == 3

    lines = output.read_text().strip().split("\n")
    assert len(lines) == 3

    titles = []
    for line in lines:
        data = json.loads(line)
        titles.append(data["input_title"])
    assert titles == ["The Matrix", "Gladiator", "Inception"]

"""Tests for the generation engine with mocked agent calls."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chuckles_prime.generator import (
    _extract_trace,
    _parse_agent_output,
    generate_batch,
    read_input_titles,
)
from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate


# ---------------------------------------------------------------------------
# Test read_input_titles
# ---------------------------------------------------------------------------


def test_read_titles_from_csv(tmp_path):
    """Read titles from a CSV with 'title' header, verify all 3 returned."""
    csv_path = tmp_path / "titles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title"])
        writer.writerow(["The Matrix"])
        writer.writerow(["Gladiator"])
        writer.writerow(["Inception"])

    titles = read_input_titles(csv_path)
    assert titles == ["The Matrix", "Gladiator", "Inception"]


def test_read_titles_skips_empty_rows(tmp_path):
    """CSV with empty title rows, verify they're skipped."""
    csv_path = tmp_path / "titles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title"])
        writer.writerow(["The Matrix"])
        writer.writerow([""])
        writer.writerow(["  "])
        writer.writerow(["Inception"])

    titles = read_input_titles(csv_path)
    assert titles == ["The Matrix", "Inception"]


def test_read_titles_missing_file():
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_input_titles("/nonexistent/path/titles.csv")


def test_read_titles_missing_column(tmp_path):
    """CSV without 'title' column raises ValueError."""
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "year"])
        writer.writerow(["The Matrix", "1999"])

    with pytest.raises(ValueError, match="title"):
        read_input_titles(csv_path)


# ---------------------------------------------------------------------------
# Test _parse_agent_output
# ---------------------------------------------------------------------------


def test_parse_valid_json_output():
    """Valid JSON with parody1, parody2, attempts -> returns 2 ParodyCandidate objects."""
    raw = json.dumps({
        "parody1": "The Mattress",
        "parody2": "The Hatrix",
        "attempts": [
            {"text": "The Mattress", "scores": {"Matrix": 0.75}, "humor_note": "Sleep pun"},
            {"text": "The Hatrix", "scores": {"Matrix": 0.80}, "humor_note": "Hat pun"},
        ],
    })

    candidates = _parse_agent_output(raw)
    assert len(candidates) == 2
    assert candidates[0].text == "The Mattress"
    assert candidates[0].phonetic_scores == {"Matrix": 0.75}
    assert candidates[0].humor_note == "Sleep pun"
    assert candidates[1].text == "The Hatrix"


def test_parse_json_with_wrapper_text():
    """JSON embedded in explanation text -> still parses correctly."""
    inner = json.dumps({"parody1": "Fartacus", "parody2": "Splatacus"})
    raw = f"Here are the results: {inner} Hope you like them!"

    candidates = _parse_agent_output(raw)
    assert len(candidates) == 2
    assert candidates[0].text == "Fartacus"
    assert candidates[1].text == "Splatacus"


def test_parse_invalid_output():
    """Completely non-JSON string -> returns empty list."""
    candidates = _parse_agent_output("This is not JSON at all")
    assert candidates == []


def test_parse_single_parody():
    """JSON with only parody1 (no parody2) -> returns 1 candidate."""
    raw = json.dumps({"parody1": "Fartacus"})

    candidates = _parse_agent_output(raw)
    assert len(candidates) == 1
    assert candidates[0].text == "Fartacus"


# ---------------------------------------------------------------------------
# Test _extract_trace
# ---------------------------------------------------------------------------


def test_extract_trace_success():
    """Mock RunResult with steps, output, state, token_usage -> correct AgentTrace."""
    mock_tu = SimpleNamespace(input_tokens=100, output_tokens=50)
    mock_result = SimpleNamespace(
        steps=[{"model_output": "thinking..."}],
        output="result text",
        state="success",
        token_usage=mock_tu,
    )

    trace = _extract_trace(mock_result)
    assert isinstance(trace, AgentTrace)
    assert trace.steps == [{"model_output": "thinking..."}]
    assert trace.final_output == "result text"
    assert trace.token_usage == {"input_tokens": 100, "output_tokens": 50}
    assert trace.state == "success"


def test_extract_trace_no_token_usage():
    """Mock RunResult with token_usage=None -> AgentTrace.token_usage is None."""
    mock_result = SimpleNamespace(
        steps=[],
        output="",
        state="success",
        token_usage=None,
    )

    trace = _extract_trace(mock_result)
    assert trace.token_usage is None


# ---------------------------------------------------------------------------
# Test generate_batch error isolation
# ---------------------------------------------------------------------------


def test_batch_continues_on_error():
    """Mock agent.run to succeed on first call and raise on second.

    Verify generate_batch returns 2 records: first with candidates,
    second with error set and empty candidates.
    """
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.model.model_id = "test-model"

    # First call succeeds, second raises
    success_result = SimpleNamespace(
        output=json.dumps({"parody1": "Fartacus", "parody2": "Splatacus"}),
        steps=[{"model_output": "step1"}],
        state="success",
        token_usage=None,
    )
    mock_agent.run.side_effect = [success_result, RuntimeError("LLM timeout")]

    # Create mock parody_tool
    mock_parody_tool = MagicMock()
    mock_parody_tool.forward.return_value = json.dumps({
        "target": "test",
        "suggestions": [],
    })

    # Create minimal config
    mock_config = SimpleNamespace(
        funny_words={"test": ["word"]},
        human_examples=[("A", "B", "C")],
        preferences_text="Be funny.",
    )

    records = generate_batch(
        titles=["Spartacus", "Gladiator"],
        agent=mock_agent,
        parody_tool=mock_parody_tool,
        config=mock_config,
    )

    assert len(records) == 2

    # First record should succeed
    assert isinstance(records[0], GenerationRecord)
    assert records[0].input_title == "Spartacus"
    assert records[0].error is None
    assert len(records[0].candidates) == 2

    # Second record should have error
    assert isinstance(records[1], GenerationRecord)
    assert records[1].input_title == "Gladiator"
    assert records[1].error is not None
    assert "LLM timeout" in records[1].error
    assert records[1].candidates == []
    assert records[1].trace.state == "error"

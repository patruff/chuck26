"""Tests for CLI entry point, argument parsing, and JSONL deserialization."""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from chuckles_prime.cli import (
    _build_parser,
    _print_summary,
    _record_from_dict,
    cmd_convert,
    cmd_generate,
    load_records,
)
from chuckles_prime.traces import archive_traces
from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate


def _make_record(
    title: str = "The Matrix", error: str | None = None
) -> GenerationRecord:
    """Create a minimal but realistic GenerationRecord for testing."""
    return GenerationRecord(
        input_title=title,
        candidates=[]
        if error
        else [
            ParodyCandidate(
                text="The Mattress",
                phonetic_scores={"Matrix": 0.78},
                humor_note="comfy",
            ),
        ],
        trace=AgentTrace(
            steps=[{"model_output": "thinking..."}],
            final_output='{"parody1": "The Mattress"}',
            token_usage={"input_tokens": 100, "output_tokens": 50},
            state="success" if not error else "error",
        ),
        model_name="test-model",
        error=error,
    )


# ---------------------------------------------------------------------------
# _record_from_dict tests
# ---------------------------------------------------------------------------


def test_record_from_dict_roundtrip():
    """Serialize a record via asdict+json, deserialize via _record_from_dict, compare."""
    original = _make_record()
    d = json.loads(json.dumps(dataclasses.asdict(original), default=str))
    result = _record_from_dict(d)

    assert result.input_title == "The Matrix"
    assert result.model_name == "test-model"
    assert len(result.candidates) == 1
    assert result.candidates[0].text == "The Mattress"
    assert result.candidates[0].phonetic_scores == {"Matrix": 0.78}
    assert result.trace.state == "success"
    assert result.trace.token_usage == {"input_tokens": 100, "output_tokens": 50}
    assert result.error is None


def test_record_from_dict_with_error():
    """Round-trip a record with error preserves the error field."""
    original = _make_record(error="test error")
    d = json.loads(json.dumps(dataclasses.asdict(original), default=str))
    result = _record_from_dict(d)

    assert result.error == "test error"


def test_record_from_dict_missing_optional_fields():
    """Minimal dict with only required fields uses sensible defaults."""
    d = {
        "input_title": "Minimal",
        "candidates": [],
        "trace": {"steps": [], "final_output": ""},
    }
    result = _record_from_dict(d)

    assert result.model_name == "unknown"
    assert result.error is None
    assert result.trace.state == "unknown"
    assert result.trace.token_usage is None


# ---------------------------------------------------------------------------
# load_records tests
# ---------------------------------------------------------------------------


def test_load_records_from_jsonl(tmp_path: Path):
    """Load records from JSONL written by archive_traces."""
    path = tmp_path / "traces.jsonl"
    archive_traces([_make_record("A"), _make_record("B")], path)

    records = load_records(path)
    assert len(records) == 2
    assert records[0].input_title == "A"
    assert records[1].input_title == "B"
    assert len(records[0].candidates) == 1
    assert len(records[1].candidates) == 1


def test_load_records_empty_file(tmp_path: Path):
    """Empty file returns empty list."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    records = load_records(path)
    assert records == []


def test_load_records_skips_blank_lines(tmp_path: Path):
    """Blank lines between records are skipped."""
    path = tmp_path / "traces.jsonl"
    rec = _make_record("A")
    line = json.dumps(dataclasses.asdict(rec), default=str)
    path.write_text(f"{line}\n\n{line}\n")

    records = load_records(path)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# _print_summary tests
# ---------------------------------------------------------------------------


def test_print_summary_output():
    """Summary table contains expected metrics and counts."""
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False)

    records = [_make_record(), _make_record(error="fail")]
    _print_summary(records, con)

    output = buf.getvalue()
    assert "Titles processed" in output
    assert "2" in output
    assert "Successful" in output
    assert "1" in output
    assert "Failed" in output
    assert "Total candidates" in output


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_parse_args_generate():
    """Parse generate subcommand with all flags."""
    args = _build_parser().parse_args(
        [
            "generate",
            "input.csv",
            "--settings",
            "s.json",
            "--output-dir",
            "out",
            "--grpo-repo",
            "user/grpo",
            "--no-push",
        ]
    )
    assert args.command == "generate"
    assert args.input == "input.csv"
    assert args.settings == "s.json"
    assert args.output_dir == "out"
    assert args.grpo_repo == "user/grpo"
    assert args.dpo_repo is None
    assert args.no_push is True


def test_parse_args_convert():
    """Parse convert subcommand with dpo-repo flag."""
    args = _build_parser().parse_args(
        ["convert", "traces.jsonl", "--dpo-repo", "user/dpo"]
    )
    assert args.command == "convert"
    assert args.traces == "traces.jsonl"
    assert args.dpo_repo == "user/dpo"
    assert args.no_push is False


def test_parse_args_no_command():
    """No subcommand sets args.command to None."""
    args = _build_parser().parse_args([])
    assert args.command is None


# ---------------------------------------------------------------------------
# cmd_generate smoke test
# ---------------------------------------------------------------------------


def test_cmd_generate_smoke(tmp_path, monkeypatch):
    """Integration smoke: cmd_generate calls pipeline modules in order."""
    # Mock config
    mock_config = MagicMock()
    mock_config.model_name = "test-model"
    mock_config.human_examples = []

    monkeypatch.setattr(
        "chuckles_prime.config.load_config", lambda *a, **kw: mock_config
    )
    monkeypatch.setattr(
        "chuckles_prime.generator.read_input_titles",
        lambda *a, **kw: ["Title A", "Title B"],
    )
    monkeypatch.setattr(
        "chuckles_prime.model.create_model", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "chuckles_prime.tools.load_parody_tools",
        lambda *a, **kw: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "chuckles_prime.generator.create_agent", lambda *a, **kw: MagicMock()
    )

    generate_single_mock = MagicMock(side_effect=lambda title, *a, **kw: _make_record(title))
    monkeypatch.setattr(
        "chuckles_prime.generator.generate_single", generate_single_mock
    )

    archive_mock = MagicMock(return_value=2)
    monkeypatch.setattr("chuckles_prime.traces.archive_traces", archive_mock)

    monkeypatch.setattr(
        "chuckles_prime.dataset.records_to_grpo_dataset",
        lambda *a, **kw: MagicMock(__len__=lambda self: 2),
    )
    monkeypatch.setattr(
        "chuckles_prime.dataset.build_dpo_dataset",
        lambda *a, **kw: MagicMock(__len__=lambda self: 0),
    )

    args = SimpleNamespace(
        input="test.csv",
        settings="settings.json",
        output_dir=str(tmp_path),
        grpo_repo=None,
        dpo_repo=None,
        no_push=True,
    )
    cmd_generate(args)

    assert generate_single_mock.call_count == 2
    assert archive_mock.call_count == 1


# ---------------------------------------------------------------------------
# cmd_convert error test
# ---------------------------------------------------------------------------


def test_cmd_convert_file_not_found():
    """cmd_convert raises FileNotFoundError for missing traces file."""
    args = SimpleNamespace(
        traces="/nonexistent/traces.jsonl",
        settings="settings.json",
        grpo_repo=None,
        dpo_repo=None,
        no_push=True,
    )
    with pytest.raises(FileNotFoundError, match="Traces file not found"):
        cmd_convert(args)

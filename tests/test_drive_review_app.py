"""Tests for scripts/drive_review_app.py: CSV helpers, DPO logic, CLI parsing."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure the scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from drive_review_app import (
    DATASET_SYSTEM_PROMPT,
    DRIVE_BASE_FOLDER,
    FINISHED_FOLDER,
    INPUT_FOLDER,
    REVIEW_FIELDS,
    TO_BE_CHECKED_FOLDER,
    _parse_csv_text,
    _read_titles_from_csv_text,
    _rows_to_csv_text,
    main,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_folder_constants():
    """Verify Drive folder names."""
    assert DRIVE_BASE_FOLDER == "chuck26"
    assert INPUT_FOLDER == "input"
    assert TO_BE_CHECKED_FOLDER == "to_be_checked"
    assert FINISHED_FOLDER == "finished_preference"


def test_review_fields():
    """Review CSV has expected columns."""
    assert "id" in REVIEW_FIELDS
    assert "input_title" in REVIEW_FIELDS
    assert "parody_text" in REVIEW_FIELDS
    assert "model_name" in REVIEW_FIELDS
    assert "adapter" in REVIEW_FIELDS
    assert "status" in REVIEW_FIELDS


# ---------------------------------------------------------------------------
# _parse_csv_text
# ---------------------------------------------------------------------------


def test_parse_csv_text_basic():
    """Parse a simple CSV string."""
    text = "title,score\nThe Matrix,0.82\nDie Hard,0.71\n"
    rows = _parse_csv_text(text)
    assert len(rows) == 2
    assert rows[0]["title"] == "The Matrix"
    assert rows[1]["score"] == "0.71"


def test_parse_csv_text_empty():
    """Empty CSV (header only) returns empty list."""
    text = "title,score\n"
    rows = _parse_csv_text(text)
    assert rows == []


def test_parse_csv_text_no_content():
    """Completely empty string returns empty list."""
    rows = _parse_csv_text("")
    assert rows == []


def test_parse_csv_text_quoted_fields():
    """Fields with commas and quotes parse correctly."""
    text = 'id,text,note\n1,"Hello, World","She said ""hi"""\n'
    rows = _parse_csv_text(text)
    assert len(rows) == 1
    assert rows[0]["text"] == "Hello, World"
    assert rows[0]["note"] == 'She said "hi"'


# ---------------------------------------------------------------------------
# _rows_to_csv_text
# ---------------------------------------------------------------------------


def test_rows_to_csv_roundtrip():
    """Write then parse produces the same data."""
    original = [
        {"id": "1", "title": "The Matrix", "score": "0.82"},
        {"id": "2", "title": "Die Hard", "score": "0.71"},
    ]
    text = _rows_to_csv_text(original, ["id", "title", "score"])
    parsed = _parse_csv_text(text)
    assert len(parsed) == 2
    assert parsed[0] == original[0]
    assert parsed[1] == original[1]


def test_rows_to_csv_empty():
    """Empty row list produces header-only CSV."""
    text = _rows_to_csv_text([], ["a", "b"])
    lines = text.strip().split("\n")
    assert len(lines) == 1
    assert "a,b" in lines[0]


def test_rows_to_csv_special_characters():
    """Fields with commas get properly quoted."""
    rows = [{"text": "Hello, World", "note": 'She said "hi"'}]
    text = _rows_to_csv_text(rows, ["text", "note"])
    parsed = _parse_csv_text(text)
    assert parsed[0]["text"] == "Hello, World"
    assert parsed[0]["note"] == 'She said "hi"'


# ---------------------------------------------------------------------------
# _read_titles_from_csv_text
# ---------------------------------------------------------------------------


def test_read_titles_with_title_column():
    """Reads from 'title' column."""
    text = "title\nThe Matrix\nDie Hard\n"
    titles = _read_titles_from_csv_text(text)
    assert titles == ["The Matrix", "Die Hard"]


def test_read_titles_with_original_column():
    """Falls back to 'original' column when 'title' is absent."""
    text = "original,parody\nThe Matrix,The Mattress\n"
    titles = _read_titles_from_csv_text(text)
    assert titles == ["The Matrix"]


def test_read_titles_strips_whitespace():
    """Titles are stripped of leading/trailing whitespace."""
    text = "title\n  The Matrix  \n  Die Hard\n"
    titles = _read_titles_from_csv_text(text)
    assert titles == ["The Matrix", "Die Hard"]


def test_read_titles_skips_empty():
    """Empty title rows are skipped."""
    text = "title\nThe Matrix\n\n  \nDie Hard\n"
    titles = _read_titles_from_csv_text(text)
    assert titles == ["The Matrix", "Die Hard"]


def test_read_titles_no_valid_column():
    """Raises ValueError when neither 'title' nor 'original' column exists."""
    text = "name,score\nThe Matrix,0.82\n"
    with pytest.raises(ValueError, match="title.*original"):
        _read_titles_from_csv_text(text)


def test_read_titles_prefers_title_over_original():
    """When both 'title' and 'original' exist, 'title' is used."""
    text = "title,original\nA,B\n"
    titles = _read_titles_from_csv_text(text)
    assert titles == ["A"]


# ---------------------------------------------------------------------------
# DPO pair building (same logic as process_reviews, tested via cmd_process)
# We test the DATASET_SYSTEM_PROMPT is coherent.
# ---------------------------------------------------------------------------


def test_dataset_system_prompt():
    """System prompt mentions comedy and phonetic."""
    assert "comedy" in DATASET_SYSTEM_PROMPT.lower()
    assert "phonetic" in DATASET_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Review CSV roundtrip with REVIEW_FIELDS
# ---------------------------------------------------------------------------


def test_review_csv_roundtrip():
    """A full review CSV can be written and parsed back."""
    rows = [
        {
            "id": "1",
            "input_title": "The Matrix",
            "parody_text": "The Mattress",
            "humor_note": "comfy",
            "phonetic_scores": '{"Matrix": 0.82}',
            "avg_phonetic_score": "0.820",
            "model_name": "qwen-3-32b",
            "adapter": "my-lora",
            "status": "chosen",
        },
        {
            "id": "2",
            "input_title": "The Matrix",
            "parody_text": "The Madness",
            "humor_note": "mental",
            "phonetic_scores": '{"Matrix": 0.71}',
            "avg_phonetic_score": "0.710",
            "model_name": "qwen-3-32b",
            "adapter": "",
            "status": "rejected",
        },
    ]
    text = _rows_to_csv_text(rows, REVIEW_FIELDS)
    parsed = _parse_csv_text(text)
    assert len(parsed) == 2
    assert parsed[0]["status"] == "chosen"
    assert parsed[1]["adapter"] == ""
    # Phonetic scores should round-trip as JSON string
    scores = json.loads(parsed[0]["phonetic_scores"])
    assert scores["Matrix"] == 0.82


# ---------------------------------------------------------------------------
# CLI parser (main function exits without command)
# ---------------------------------------------------------------------------


def test_main_no_args_exits(monkeypatch):
    """Running with no args prints help and exits."""
    monkeypatch.setattr("sys.argv", ["drive-review"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

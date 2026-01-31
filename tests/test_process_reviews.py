"""Tests for scripts/process_reviews.py: CSV loading, DPO pair building, file processing."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import pytest

# Import the module under test -- it manipulates sys.path on import,
# which is fine for tests run from the repo root.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from process_reviews import build_dpo_rows, load_review_csv, process_reviews


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REVIEW_FIELDS = [
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


def _write_review_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write a review CSV with standard field names."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _make_row(
    row_id: int = 1,
    title: str = "The Matrix",
    parody: str = "The Mattress",
    status: str = "pending",
    model: str = "qwen-3-32b",
    adapter: str = "",
    score: str = "0.820",
    humor: str = "comfy",
) -> dict[str, str]:
    return {
        "id": str(row_id),
        "input_title": title,
        "parody_text": parody,
        "humor_note": humor,
        "phonetic_scores": '{"Matrix": 0.82}',
        "avg_phonetic_score": score,
        "model_name": model,
        "adapter": adapter,
        "status": status,
    }


# ---------------------------------------------------------------------------
# load_review_csv
# ---------------------------------------------------------------------------


def test_load_review_csv_basic(tmp_path: Path):
    """Loads rows correctly from a standard review CSV."""
    csv_path = tmp_path / "review-test.csv"
    rows = [_make_row(1), _make_row(2, parody="The Madness", status="rejected")]
    _write_review_csv(csv_path, rows)

    loaded = load_review_csv(csv_path)
    assert len(loaded) == 2
    assert loaded[0]["input_title"] == "The Matrix"
    assert loaded[1]["parody_text"] == "The Madness"
    assert loaded[1]["status"] == "rejected"


def test_load_review_csv_empty(tmp_path: Path):
    """Empty CSV (header only) returns empty list."""
    csv_path = tmp_path / "review-empty.csv"
    _write_review_csv(csv_path, [])

    loaded = load_review_csv(csv_path)
    assert loaded == []


# ---------------------------------------------------------------------------
# build_dpo_rows
# ---------------------------------------------------------------------------


def test_build_dpo_single_pair():
    """One chosen + one rejected for same title -> 1 DPO pair."""
    rows = [
        _make_row(1, status="chosen", parody="The Mattress"),
        _make_row(2, status="rejected", parody="The Madness"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 1
    assert dpo[0]["chosen"][0]["content"] == "The Mattress"
    assert dpo[0]["rejected"][0]["content"] == "The Madness"
    assert "'The Matrix'" in dpo[0]["prompt"][1]["content"]


def test_build_dpo_multiple_chosen_rejected():
    """2 chosen x 2 rejected for same title -> 4 DPO pairs."""
    rows = [
        _make_row(1, status="chosen", parody="A"),
        _make_row(2, status="chosen", parody="B"),
        _make_row(3, status="rejected", parody="X"),
        _make_row(4, status="rejected", parody="Y"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 4
    chosen_texts = {r["chosen"][0]["content"] for r in dpo}
    rejected_texts = {r["rejected"][0]["content"] for r in dpo}
    assert chosen_texts == {"A", "B"}
    assert rejected_texts == {"X", "Y"}


def test_build_dpo_pending_rows_ignored():
    """Rows with status=pending are excluded from DPO pairs."""
    rows = [
        _make_row(1, status="chosen", parody="Good"),
        _make_row(2, status="pending", parody="Meh"),
        _make_row(3, status="rejected", parody="Bad"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 1
    assert dpo[0]["chosen"][0]["content"] == "Good"
    assert dpo[0]["rejected"][0]["content"] == "Bad"


def test_build_dpo_no_chosen():
    """Only rejected rows (no chosen) -> 0 pairs."""
    rows = [
        _make_row(1, status="rejected", parody="Bad1"),
        _make_row(2, status="rejected", parody="Bad2"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 0


def test_build_dpo_no_rejected():
    """Only chosen rows (no rejected) -> 0 pairs."""
    rows = [
        _make_row(1, status="chosen", parody="Good1"),
        _make_row(2, status="chosen", parody="Good2"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 0


def test_build_dpo_multiple_titles():
    """Multiple titles produce independent pairs."""
    rows = [
        _make_row(1, title="The Matrix", status="chosen", parody="The Mattress"),
        _make_row(2, title="The Matrix", status="rejected", parody="The Madness"),
        _make_row(3, title="Die Hard", status="chosen", parody="Dye Hard"),
        _make_row(4, title="Die Hard", status="rejected", parody="Pie Hard"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 2

    titles_in_prompts = {r["prompt"][1]["content"] for r in dpo}
    assert any("The Matrix" in t for t in titles_in_prompts)
    assert any("Die Hard" in t for t in titles_in_prompts)


def test_build_dpo_cross_title_not_paired():
    """Chosen from one title is NOT paired with rejected from another."""
    rows = [
        _make_row(1, title="The Matrix", status="chosen", parody="The Mattress"),
        _make_row(2, title="Die Hard", status="rejected", parody="Pie Hard"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 0


def test_build_dpo_provenance_metadata():
    """DPO rows include model/adapter provenance fields."""
    rows = [
        _make_row(1, status="chosen", model="model-a", adapter="lora-1", score="0.82"),
        _make_row(2, status="rejected", model="model-b", adapter="lora-2", score="0.65"),
    ]
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 1
    assert dpo[0]["chosen_model"] == "model-a"
    assert dpo[0]["chosen_adapter"] == "lora-1"
    assert dpo[0]["chosen_phonetic_score"] == "0.82"
    assert dpo[0]["rejected_model"] == "model-b"
    assert dpo[0]["rejected_adapter"] == "lora-2"
    assert dpo[0]["rejected_phonetic_score"] == "0.65"


def test_build_dpo_empty_rows():
    """Empty input returns empty output."""
    assert build_dpo_rows([]) == []


def test_build_dpo_case_insensitive_status():
    """Status matching is case-insensitive and whitespace-tolerant."""
    rows = [
        _make_row(1, status="Chosen"),
        _make_row(2, status=" rejected "),
    ]
    # Manually set status with whitespace for the test
    rows[1]["status"] = " Rejected "
    dpo = build_dpo_rows(rows)
    assert len(dpo) == 1


# ---------------------------------------------------------------------------
# process_reviews (integration with filesystem, mocked HF)
# ---------------------------------------------------------------------------


def test_process_reviews_no_csvs(tmp_path: Path):
    """No review CSVs returns zero counts."""
    reviews_dir = tmp_path / "pending"
    reviews_dir.mkdir()
    processed_dir = tmp_path / "processed"

    result = process_reviews(
        reviews_dir=str(reviews_dir),
        processed_dir=str(processed_dir),
        no_push=True,
    )
    assert result["files"] == 0
    assert result["pairs"] == 0


def test_process_reviews_unreviewed_csv(tmp_path: Path):
    """CSV with only pending rows is skipped."""
    reviews_dir = tmp_path / "pending"
    reviews_dir.mkdir()
    processed_dir = tmp_path / "processed"

    csv_path = reviews_dir / "review-test.csv"
    _write_review_csv(csv_path, [_make_row(1, status="pending")])

    result = process_reviews(
        reviews_dir=str(reviews_dir),
        processed_dir=str(processed_dir),
        no_push=True,
    )
    assert result["pairs"] == 0


def test_process_reviews_builds_and_moves(tmp_path: Path):
    """Reviewed CSV produces DPO pairs and gets moved to processed/."""
    reviews_dir = tmp_path / "pending"
    reviews_dir.mkdir()
    processed_dir = tmp_path / "processed"

    csv_path = reviews_dir / "review-2026-01-31.csv"
    _write_review_csv(csv_path, [
        _make_row(1, status="chosen", parody="The Mattress"),
        _make_row(2, status="rejected", parody="The Madness"),
    ])

    result = process_reviews(
        reviews_dir=str(reviews_dir),
        processed_dir=str(processed_dir),
        no_push=True,
    )
    assert result["pairs"] == 1
    assert result["chosen"] == 1
    assert result["rejected"] == 1

    # Original should be moved
    assert not csv_path.exists()
    moved = list(processed_dir.glob("review-*-done-*.csv"))
    assert len(moved) == 1


def test_process_reviews_multiple_files(tmp_path: Path):
    """Multiple reviewed CSVs are all processed."""
    reviews_dir = tmp_path / "pending"
    reviews_dir.mkdir()
    processed_dir = tmp_path / "processed"

    for i in range(3):
        csv_path = reviews_dir / f"review-batch{i}.csv"
        _write_review_csv(csv_path, [
            _make_row(i * 2 + 1, title=f"Title {i}", status="chosen", parody=f"Good {i}"),
            _make_row(i * 2 + 2, title=f"Title {i}", status="rejected", parody=f"Bad {i}"),
        ])

    result = process_reviews(
        reviews_dir=str(reviews_dir),
        processed_dir=str(processed_dir),
        no_push=True,
    )
    assert result["pairs"] == 3
    assert len(list(processed_dir.glob("*.csv"))) == 3

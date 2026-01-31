"""Tests for the human-in-the-loop labeler: label I/O, DPO conversion, Flask routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuckles_prime.labeler import (
    _prepare_items,
    build_dpo_from_labels,
    create_app,
    load_labels,
    save_labels,
)
from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate


def _make_record(
    title: str = "The Matrix",
    n_candidates: int = 2,
    error: str | None = None,
) -> GenerationRecord:
    """Create a GenerationRecord with configurable candidate count."""
    candidates = []
    if not error:
        texts = [("The Matt Rats", {"Matrix": 0.78}), ("Th' Muck Rats", {"Matrix": 0.62})]
        for i in range(min(n_candidates, len(texts))):
            candidates.append(
                ParodyCandidate(text=texts[i][0], phonetic_scores=texts[i][1])
            )
    return GenerationRecord(
        input_title=title,
        candidates=candidates,
        trace=AgentTrace(steps=[], final_output="", token_usage=None, state="success"),
        model_name="test-model",
        error=error,
    )


# ---------------------------------------------------------------------------
# load_labels / save_labels
# ---------------------------------------------------------------------------


def test_load_labels_missing_file(tmp_path: Path):
    """Missing file returns empty structure."""
    result = load_labels(tmp_path / "nope.json")
    assert result == {"version": 1, "labels": []}


def test_save_and_load_roundtrip(tmp_path: Path):
    """Save then load preserves data."""
    path = tmp_path / "labels.json"
    data = {
        "version": 1,
        "labels": [
            {
                "input_title": "The Matrix",
                "parody1": "The Matt Rats",
                "parody2": "Th' Muck Rats",
                "winner": "parody1",
                "timestamp": "2026-01-31T14:23:01+00:00",
            }
        ],
    }
    save_labels(path, data)
    loaded = load_labels(path)
    assert loaded == data


def test_save_creates_parent_dirs(tmp_path: Path):
    """save_labels creates parent directories if needed."""
    path = tmp_path / "sub" / "dir" / "labels.json"
    save_labels(path, {"version": 1, "labels": []})
    assert path.exists()


# ---------------------------------------------------------------------------
# _prepare_items
# ---------------------------------------------------------------------------


def test_prepare_items_filters_correctly():
    """Only records with 2+ candidates and no error pass through."""
    records = [
        _make_record("Good", n_candidates=2),
        _make_record("One Candidate", n_candidates=1),
        _make_record("Error", error="boom"),
    ]
    items = _prepare_items(records)
    assert len(items) == 1
    assert items[0]["input_title"] == "Good"
    assert items[0]["parody1"] == "The Matt Rats"
    assert items[0]["parody2"] == "Th' Muck Rats"


# ---------------------------------------------------------------------------
# build_dpo_from_labels
# ---------------------------------------------------------------------------


def test_build_dpo_parody1_winner(tmp_path: Path):
    """parody1 winner produces correct chosen/rejected."""
    path = tmp_path / "labels.json"
    save_labels(path, {
        "version": 1,
        "labels": [{
            "input_title": "The Matrix",
            "parody1": "The Matt Rats",
            "parody2": "Th' Muck Rats",
            "winner": "parody1",
            "timestamp": "2026-01-31T14:23:01+00:00",
        }],
    })
    rows = build_dpo_from_labels(path)
    assert len(rows) == 1
    assert rows[0]["chosen"][0]["content"] == "The Matt Rats"
    assert rows[0]["rejected"][0]["content"] == "Th' Muck Rats"
    assert rows[0]["prompt"][0]["role"] == "system"
    assert "The Matrix" in rows[0]["prompt"][1]["content"]


def test_build_dpo_parody2_winner(tmp_path: Path):
    """parody2 winner swaps chosen/rejected."""
    path = tmp_path / "labels.json"
    save_labels(path, {
        "version": 1,
        "labels": [{
            "input_title": "The Matrix",
            "parody1": "The Matt Rats",
            "parody2": "Th' Muck Rats",
            "winner": "parody2",
            "timestamp": "2026-01-31T14:23:01+00:00",
        }],
    })
    rows = build_dpo_from_labels(path)
    assert len(rows) == 1
    assert rows[0]["chosen"][0]["content"] == "Th' Muck Rats"
    assert rows[0]["rejected"][0]["content"] == "The Matt Rats"


def test_build_dpo_both_bad_excluded(tmp_path: Path):
    """both_bad labels are excluded from DPO output."""
    path = tmp_path / "labels.json"
    save_labels(path, {
        "version": 1,
        "labels": [
            {
                "input_title": "The Matrix",
                "parody1": "A",
                "parody2": "B",
                "winner": "both_bad",
                "timestamp": "2026-01-31T14:23:01+00:00",
            },
            {
                "input_title": "Inception",
                "parody1": "C",
                "parody2": "D",
                "winner": "parody1",
                "timestamp": "2026-01-31T14:23:02+00:00",
            },
        ],
    })
    rows = build_dpo_from_labels(path)
    assert len(rows) == 1
    assert rows[0]["chosen"][0]["content"] == "C"


def test_build_dpo_empty_labels(tmp_path: Path):
    """Empty labels file returns empty list."""
    path = tmp_path / "labels.json"
    save_labels(path, {"version": 1, "labels": []})
    rows = build_dpo_from_labels(path)
    assert rows == []


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    """Create a Flask test client with sample items."""
    items = [
        {
            "input_title": "The Matrix",
            "parody1": "The Matt Rats",
            "parody2": "Th' Muck Rats",
            "score1": {"Matrix": 0.78},
            "score2": {"Matrix": 0.62},
        },
        {
            "input_title": "Inception",
            "parody1": "Insheeption",
            "parody2": "Inncension",
            "score1": {"Inception": 0.71},
            "score2": {"Inception": 0.55},
        },
    ]
    labels_path = tmp_path / "labels.json"
    app = create_app(items, labels_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_returns_html(client):
    """GET / returns HTML page containing item data."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"The Matrix" in resp.data
    assert b"Chuckles Labeler" in resp.data


def test_label_post_creates_label(client, tmp_path: Path):
    """POST /label saves a label and returns progress."""
    resp = client.post(
        "/label",
        data=json.dumps({
            "input_title": "The Matrix",
            "parody1": "The Matt Rats",
            "parody2": "Th' Muck Rats",
            "winner": "parody1",
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["labeled"] == 1
    assert body["total"] == 2

    # Verify file was written
    labels_path = tmp_path / "labels.json"
    data = json.loads(labels_path.read_text())
    assert len(data["labels"]) == 1
    assert data["labels"][0]["winner"] == "parody1"


def test_label_post_upserts(client, tmp_path: Path):
    """Labeling the same title twice upserts rather than duplicating."""
    payload = {
        "input_title": "The Matrix",
        "parody1": "The Matt Rats",
        "parody2": "Th' Muck Rats",
        "winner": "parody1",
    }
    client.post("/label", data=json.dumps(payload), content_type="application/json")
    payload["winner"] = "parody2"
    client.post("/label", data=json.dumps(payload), content_type="application/json")

    labels_path = tmp_path / "labels.json"
    data = json.loads(labels_path.read_text())
    assert len(data["labels"]) == 1
    assert data["labels"][0]["winner"] == "parody2"


def test_label_post_invalid_winner(client):
    """POST /label with invalid winner returns 400."""
    resp = client.post(
        "/label",
        data=json.dumps({
            "input_title": "X",
            "parody1": "A",
            "parody2": "B",
            "winner": "invalid",
        }),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_stats_empty(client):
    """GET /stats with no labels shows zeros."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["labeled"] == 0
    assert body["total"] == 2


def test_stats_after_labeling(client):
    """GET /stats reflects labels after POST."""
    client.post(
        "/label",
        data=json.dumps({
            "input_title": "The Matrix",
            "parody1": "The Matt Rats",
            "parody2": "Th' Muck Rats",
            "winner": "parody1",
        }),
        content_type="application/json",
    )
    resp = client.get("/stats")
    body = resp.get_json()
    assert body["labeled"] == 1
    assert body["parody1"] == 1
    assert body["parody2"] == 0
    assert body["both_bad"] == 0


# ---------------------------------------------------------------------------
# CLI parser tests
# ---------------------------------------------------------------------------


def test_parse_args_label():
    """Parse label subcommand."""
    from chuckles_prime.cli import _build_parser

    args = _build_parser().parse_args(["label", "traces.jsonl", "--port", "8080"])
    assert args.command == "label"
    assert args.traces == "traces.jsonl"
    assert args.port == 8080
    assert args.labels is None


def test_parse_args_label_defaults():
    """Label subcommand defaults."""
    from chuckles_prime.cli import _build_parser

    args = _build_parser().parse_args(["label", "traces.jsonl"])
    assert args.port == 5117
    assert args.labels is None


def test_parse_args_export_labels():
    """Parse export-labels subcommand."""
    from chuckles_prime.cli import _build_parser

    args = _build_parser().parse_args(
        ["export-labels", "labels.json", "--dpo-repo", "user/dpo", "--no-push"]
    )
    assert args.command == "export-labels"
    assert args.labels == "labels.json"
    assert args.dpo_repo == "user/dpo"
    assert args.no_push is True

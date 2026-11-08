import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from review_films import ReviewStore, create_app, load_source, review_pair

RECENT = ROOT / "results/feedback-0c3cc72/results.jsonl.gz"


def test_recent_feedback_run_and_full_trace_render(tmp_path):
    pairs = list(load_source(RECENT))
    assert [p["id"] for p in load_source(RECENT.parent)] == [p["id"] for p in pairs]
    assert len(pairs) == 10 and pairs[0]["film"]["title"] == "The Running Man"
    assert {s["title"] for s in pairs[0]["sides"]} == {
        "The Cumming Klan",
        "The Nutting Klan",
    }
    app = create_app(pairs, tmp_path)
    text = app.test_client().get("/").get_data(as_text=True)
    assert (
        "The Cumming Klan" in text
        and "&lt;think&gt;" in text
        and "word_phone_tool" in text
    )
    assert (
        "Shared reasoning" in text
        and (tmp_path / "reviewed_dpo.jsonl").read_bytes() == b""
    )


def test_review_persistence_correction_clear_and_import(tmp_path):
    pairs = list(load_source(RECENT))
    p = pairs[0]
    directory = tmp_path / "first"
    client = create_app(pairs, directory).test_client()
    route = "/vote/" + p["id"]
    assert (
        client.post(
            route,
            data={"choice": "b", "feedback": "The second pitch lands.", "expected": ""},
        ).status_code
        == 303
    )
    labels = client.get("/download/preferences").data
    dpo = client.get("/download/dpo").data
    r = json.loads(dpo)
    assert r["chosen"][0]["content"] == p["sides"][1]["title"]
    assert (
        r["rejected"][0]["content"] == p["sides"][0]["title"]
        and r["feedback"] == "The second pitch lands."
    )
    assert r["chosen_trace"]["sha256"] == p["sides"][1]["sha256"]
    assert (
        client.post(
            route, data={"choice": "a", "feedback": "stale", "expected": ""}
        ).status_code
        == 409
    )
    updated = json.loads(labels)["updated"]
    assert (
        client.post(
            route,
            data={
                "choice": "neither",
                "feedback": "Neither landed.",
                "expected": updated,
            },
        ).status_code
        == 303
    )
    assert (
        client.get("/download/dpo").data == b""
        and json.loads(client.get("/download/preferences").data)["choice"] == "neither"
    )
    other = tmp_path / "second"
    other.mkdir()
    (other / "reviewed_dpo.jsonl").write_bytes(dpo)
    restored = create_app(pairs, other).test_client()
    assert json.loads(restored.get("/download/preferences").data)["choice"] == "b"
    import io

    assert (
        restored.post(
            "/import",
            data={"preferences": (io.BytesIO(labels), "film_preferences.jsonl")},
        ).status_code
        == 303
    )
    assert (
        restored.post(
            "/import", data={"preferences": (io.BytesIO(dpo), "reviewed_dpo.jsonl")}
        ).status_code
        == 303
    )
    expected = json.loads(restored.get("/download/preferences").data)["updated"]
    assert (
        restored.post(route, data={"choice": "clear", "expected": expected}).status_code
        == 303
    )
    assert restored.get("/download/dpo").data == b""
    assert (
        restored.post(
            "/import",
            data={
                "preferences": (
                    io.BytesIO(
                        labels.replace(b"The second pitch lands.", b"Import test.")
                    ),
                    "votes.jsonl",
                )
            },
        ).status_code
        == 303
    )
    assert json.loads(restored.get("/download/dpo").data)["feedback"] == "Import test."


def test_duplicate_independent_winners_use_a_distinct_observed_finalist(tmp_path):
    with gzip.open(RECENT, "rt") as f:
        row = json.loads(next(f))
    other = {**row, "seed": 101}
    p = review_pair(
        {"rank": 1, "title": row["input_title"]},
        (row, tmp_path / "a"),
        (other, tmp_path / "b"),
        "test",
    )
    assert (
        p and not p["shared_trace"] and p["sides"][0]["title"] != p["sides"][1]["title"]
    )
    assert {s["seed"] for s in p["sides"]} == {row["seed"], 101}
    assert all(s["pairs"] for s in p["sides"])


def test_replay_pending_projection_after_interruption(tmp_path):
    store = ReviewStore(tmp_path)
    pair = list(load_source(RECENT))[0]
    store.update([(pair, "a", "Initial preference", "")])
    store.pending.write_text("")
    recovered = ReviewStore(tmp_path)
    assert (
        recovered.read() == {}
        and recovered.export.read_text() == ""
        and not store.pending.exists()
    )


def test_changed_catalog_rejects_a_vote_from_an_old_tab(tmp_path):
    pairs = list(load_source(RECENT))
    client = create_app(pairs, tmp_path, refresh=lambda: pairs[1:]).test_client()
    response = client.post(
        "/vote/" + pairs[0]["id"],
        data={"choice": "a", "expected": "", "feedback": "stale"},
    )
    assert response.status_code == 409
    assert (tmp_path / "reviewed_dpo.jsonl").read_bytes() == b""

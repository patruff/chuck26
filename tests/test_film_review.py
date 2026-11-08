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


def test_blind_votes_gate_identity_and_keep_ties_out_of_dpo(tmp_path):
    pairs = list(load_source(RECENT))[:2]
    for pair in pairs:
        pair.update(blind=True, collection="blind-27b-15", shared_trace=False)
        pair["sides"][0]["model"] = "Hidden original identity"
        pair["sides"][1]["model"] = "Hidden adapter identity"
    pair = pairs[0]
    app = create_app(pairs, tmp_path)
    client = app.test_client()
    url = f"/?set=blind-27b-15&pair={pair['id']}"
    before = client.get(url).get_data(as_text=True)
    assert "Reasoning and tool calls" in before and "Equally funny" in before
    assert (
        "Hidden original identity" not in before
        and "Hidden adapter identity" not in before
    )
    assert client.get(url + "&reveal=1").status_code == 403
    page = client.post(
        "/vote/" + pair["id"],
        data={"choice": "tie", "feedback": "Same joke", "expected": ""},
        follow_redirects=True,
    ).get_data(as_text=True)
    assert (
        "Reveal models for this film" in page and "Hidden adapter identity" not in page
    )
    revealed = client.get(url + "&reveal=1").get_data(as_text=True)
    assert (
        "Hidden original identity" in revealed and "Hidden adapter identity" in revealed
    )
    other = client.get("/?set=blind-27b-15&pair=" + pairs[1]["id"] + "&reveal=1")
    assert other.status_code == 403
    assert (tmp_path / "reviewed_dpo.jsonl").read_bytes() == b""
    decision = json.loads((tmp_path / "film_preferences.jsonl").read_text())
    assert decision["choice"] == decision["first_choice"] == "tie"
    pair["sides"][1]["title"] = pair["sides"][0]["title"]
    client.post(
        "/vote/" + pair["id"],
        data={
            "choice": "a",
            "feedback": "Better pitch",
            "expected": decision["updated"],
        },
    )
    assert (tmp_path / "reviewed_dpo.jsonl").read_bytes() == b""
    decision = json.loads((tmp_path / "film_preferences.jsonl").read_text())
    assert decision["choice"] == "a" and decision["first_choice"] == "tie"


def test_blind_loader_preserves_equal_titles_and_independent_traces(tmp_path):
    directory = tmp_path / "blind-27b-15"
    directory.mkdir()
    film = {
        "rank": "42",
        "year": "2000",
        "movielens_id": "12",
        "ratings": "250",
        "title": "Moon",
    }
    check = {
        "word": "moon",
        "compare_to": "soon",
        "score": 0.7,
        "suggestion_score": 0.7,
        "word_phone_score": 1.0,
    }
    draft = {"first": "Soon", "second": "Moon", "idea": "A rushed lunar landing."}
    options = []
    for variant, scale, thought in [
        ("adapter", 1, "I prefer the rushed landing."),
        ("original", 0, "The rhyme is audible."),
    ]:
        row = dict(
            input_title="Moon",
            model=variant,
            seed=4200,
            word_count=1565,
            suggestion_cutoff=0.6,
            suggested_words={"moon": [{"word": "soon", "similarity": 0.7}]},
            suggestion_tool_results={
                "moon": {"suggestions": [{"word": "soon", "similarity": 0.7}]}
            },
            steps=[
                dict(
                    kind="Draft titles",
                    step=1,
                    input_messages=[],
                    reasoning=thought,
                    model_output=json.dumps(draft),
                ),
                dict(
                    kind="Choose after phonetic checks",
                    step=2,
                    input_messages=[],
                    reasoning="",
                    model_output='{"winner":"Soon"}',
                ),
            ],
            tool_calls=[check],
            finalist_checks={"Soon": [check]},
            parody="Soon",
            explanation="The character nearly misses the landing.",
            state="success",
            tool_revisions={"suggest": "hash"},
            applied_instructions_sha256="fixture",
        )
        options.append(
            dict(
                row=row,
                film=film,
                variant=variant,
                adapter_scale=scale,
                base_sha256="base",
                adapter_sha256="adapter",
                model_revision="rev",
                bank_sha256="bank",
            )
        )
    path = directory / "042.blind.json.gz"

    def save():
        with gzip.open(path, "wt") as f:
            json.dump({"options": options, "assignment_id": "fixture-id"}, f)

    save()
    (pair,) = load_source(directory)
    assert [s["title"] for s in pair["sides"]] == ["Soon", "Soon"]
    assert not pair["shared_trace"]
    trace = pair["sides"][0]["trace"]
    assert (
        trace.index("I prefer the rushed landing.")
        < trace.index("word_phone_tool(")
        < trace.index("Choose after phonetic checks")
    )
    assert pair["sides"][0]["thinking_entries"][0]["native"]
    options[1]["row"]["suggested_words"] = {}
    save()
    with pytest.raises(ValueError, match="suggested_words"):
        list(load_source(directory))

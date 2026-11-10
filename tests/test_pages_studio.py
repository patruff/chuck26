import json
from types import SimpleNamespace

import pytest

from scripts import pages_studio as studio


def test_training_replays_exact_saved_revision(tmp_path, monkeypatch):
    catalog = list(studio.load_source(studio.DEFAULT_SOURCES[1]))[:2]
    public, private, traces = studio.materialize(catalog, "test")
    assert "prompt" not in public[0] and "trace" not in public[0]["sides"][0]
    assert traces[0]["sides"][0] and private[0]["sides"][0]["source"]
    ledger, metadata = tmp_path / "votes", tmp_path / "pairs.gz"
    metadata.write_bytes(studio.bundle(private))
    votes = [
        dict(
            pair_id=p["id"],
            collection="test",
            choice=choice,
            updated="2026-01-01T12:00:00Z",
            feedback="fixture note",
            titles=["irrelevant browser data"] * 2,
        )
        for p, choice in zip(private, ("b", "neither"))
    ]
    ledger.write_text("\n".join(json.dumps(v) for v in votes))

    def download(path, revision):
        assert revision == "exact-commit"
        return ledger if path == studio.LEDGER else metadata

    monkeypatch.setattr(studio, "hub_file", download)
    monkeypatch.setattr(
        studio.api, "dataset_info", lambda *a, **kw: SimpleNamespace(sha="exact-commit")
    )
    work = tmp_path / "training"
    studio.training_request("main", work)
    original = (work / "request.json").read_bytes()
    req = json.loads(original)
    assert [r["pair_id"] for r in req["rows"]] == [catalog[0]["id"]]
    assert req["rows"][0]["chosen"][0]["content"] == catalog[0]["sides"][1]["title"]
    assert req["dataset_sha256"] == studio.checksum(
        work / "snapshot/reviewed_dpo.jsonl"
    )
    studio.training_request("exact-commit", work)
    assert (work / "request.json").read_bytes() == original
    ledger.write_text("")
    with pytest.raises(ValueError, match="No human A/B"):
        studio.training_request(None, work)


def test_prepared_csv_resolves_at_one_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        studio.api, "dataset_info", lambda *a, **kw: SimpleNamespace(sha="prepared")
    )
    settings, inputs = tmp_path / "settings", tmp_path / "uploaded"
    settings.write_text(json.dumps(dict(source="csv", model="base", rank="35")))
    inputs.write_bytes(b'\xef\xbb\xbftitle\r\n"One, Two"\r\nThree\r\n')

    def download(path, revision):
        assert revision == "prepared"
        return {"requests/inference.json": settings, "inputs/latest.csv": inputs}[path]

    monkeypatch.setattr(studio, "hub_file", download)
    studio.inference_settings(tmp_path, "page", 1, "latest-adapter", "prepared")
    assert studio.csv_titles((tmp_path / "input.csv").read_bytes()) == [
        "One, Two",
        "Three",
    ]
    assert (tmp_path / "rank.txt").read_text() == "35"
    for bad in (b"title\n", b"title\na,b", b'"unclosed', b'"multi\nline"'):
        with pytest.raises((ValueError, studio.csv.Error)):
            studio.csv_titles(bad)

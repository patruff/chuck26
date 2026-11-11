import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.film_lists import select_film
from pipeline.film_lists.merge_reviews import merge
from scripts import pages_studio


def test_range_reaches_runpod_generator_and_merges_in_input_order(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "pipeline"))
    import run_gguf_test
    from common import read_titles

    selection = tmp_path / "selection"
    gh_output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_output))
    monkeypatch.setattr(
        sys, "argv", ["select", "1-100", "--output-dir", str(selection)]
    )
    select_film.main()
    plan = dict(line.split("=", 1) for line in gh_output.read_text().splitlines())
    parts = json.loads(plan["parts"])
    assert plan["count"] == "100" and len(parts) == 4

    monkeypatch.setenv("RUNPOD_API_KEY", "test")
    api = Mock(headers={})
    api.post.return_value.json.return_value = {"id": "fixture"}
    monkeypatch.setattr(run_gguf_test.requests, "Session", lambda: api)
    monkeypatch.setattr(
        run_gguf_test, "snapshot_bank", lambda *a: selection / "selected.csv"
    )
    monkeypatch.setattr(run_gguf_test, "preflight", lambda *a: {})
    monkeypatch.setattr(run_gguf_test, "wait_ready", lambda *a: {})
    monkeypatch.setattr(run_gguf_test, "terminate", Mock())

    def generator(args, **kwargs):
        titles = read_titles(args[args.index("--titles") + 1])
        limit = int(args[args.index("--limit") + 1])
        assert limit == len(titles) == 25
        Path(args[args.index("--output") + 1]).write_text(
            "".join(
                json.dumps({"input_title": t, "state": "success", "steps": [t]}) + "\n"
                for t in titles[:limit]
            )
        )

    monkeypatch.setattr(run_gguf_test.subprocess, "run", generator)
    for part in reversed(parts):
        work = tmp_path / ("review-" + part)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run",
                "--titles",
                str(selection / (part + ".csv")),
                "--output-dir",
                str(work),
            ],
        )
        run_gguf_test.main(tmp_path)
    output = tmp_path / "combined/results.jsonl"
    merge(selection, tmp_path, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    expected = [r["title"] for r in select_film.catalog_films(range(1, 101))]
    assert [r["input_title"] for r in rows] == expected
    assert [r["steps"][0] for r in rows] == expected

    broken = tmp_path / ("review-" + parts[1]) / "results.jsonl"
    broken.write_text(broken.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        merge(selection, tmp_path, output)


def test_rank_order_boundaries_and_overlap():
    assert select_film.parse_ranks("200, 1 – 3, 2-5, 7, 200") == [200, 1, 2, 3, 4, 5, 7]
    assert select_film.film(35)["title"] == "Speed"
    for value in ("", "1-", "0", "201", "3-1", "1,,3", "2.5", "1-4-6", "1-999999"):
        with pytest.raises(ValueError):
            select_film.parse_ranks(value)


def test_uploaded_csv_has_same_order_and_ids_across_batches(tmp_path, monkeypatch):
    sha = "a" * 40
    uploaded = tmp_path / "upload.csv"
    uploaded.write_text(
        '\ufeffinput\r\n"One, Two"\r\nThree\r\nFour\r\nFive\r\nSix\r\n"Seven ""quoted"""\r\n'
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"source": "csv", "rank": "", "model": "base"}))
    monkeypatch.setattr(
        pages_studio.api, "dataset_info", lambda *a, **kw: SimpleNamespace(sha=sha)
    )

    def hub_file(path, revision):
        assert revision == sha
        return {"inputs/latest.csv": uploaded, "requests/inference.json": request}[path]

    monkeypatch.setattr(pages_studio, "hub_file", hub_file)
    work = tmp_path / "work"
    pages_studio.inference_settings(work, "page", "35", "base", sha, batches=True)
    with (work / "part-0001.csv").open() as f:
        assert list(csv.DictReader(f)) == [{"rank": "6", "title": 'Seven "quoted"'}]
    raw = select_film.csv_titles(uploaded.read_bytes())
    assert raw[0] == "One, Two" and len(raw) == 6
    assert select_film.csv_titles(b"One\nTwo\n") == ["One", "Two"]
    assert select_film.csv_titles(b"title\nOne\nTwo\n") == ["One", "Two"]

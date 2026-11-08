"""Offline regressions for the default bank, bundled notebook, and exact-spelling gate."""
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import march_source
import run_gguf_test
from generate_march import HUMOR_BATCH_CELL, WordPhoneTool, running_man_prompt
from humor_batch import clears_phonetic_floor


def test_reject_wrong_bank_before_any_pod_request(tmp_path, monkeypatch):
    wrong = tmp_path / "word_structures.py"
    wrong.write_text("funny_words = []\ncustom_phones = {}\n")
    monkeypatch.setattr(sys, "argv", ["run", "--march-word-bank", str(wrong), "--output-dir", str(tmp_path/"out")])
    post = Mock(side_effect=AssertionError("must fail before provisioning"))
    monkeypatch.setattr(run_gguf_test.requests.Session, "post", post)
    with pytest.raises(SystemExit):
        run_gguf_test.main(tmp_path)
    post.assert_not_called()
    assert not (tmp_path/"out").exists()


def test_actions_bank_is_verified_and_private_to_run(tmp_path, monkeypatch):
    raw = b"funny_words = ['pork']\ncustom_phones = {}\n"
    monkeypatch.setattr(march_source, "BANK_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setenv("CHUCKLES_2025_BANK_GZ", base64.b64encode(gzip.compress(raw)).decode())
    assert march_source.snapshot_bank(None, tmp_path).read_bytes() == raw
    monkeypatch.setenv("CHUCKLES_2025_BANK_GZ", "not gzip")
    with pytest.raises(ValueError, match="base64-encoded gzip"):
        march_source.snapshot_bank(None, tmp_path)


def test_bundled_prompt_uses_one_floor_and_asks_for_meaning():
    prompt = running_man_prompt(march_source.NOTEBOOK, "The Running Man", {"running": []}, HUMOR_BATCH_CELL)
    assert "The Running Man" in prompt and "0.8" not in prompt
    assert ">= 0.60" in prompt and "comic scene" in prompt
    assert "edgy, dark, sexual, violent, adult themes" in prompt
    assert all(not c.get("outputs") for c in json.loads(march_source.NOTEBOOK.read_text())["cells"])


def test_hunning_cannot_inherit_huning_score():
    suggest, verify = Mock(), Mock()
    suggest.forward.return_value = json.dumps({"suggestions":[{"word":"huning", "similarity":0.83}]})
    verify.forward.return_value = json.dumps({"similarity":{"similarity":1}})
    tool = WordPhoneTool(verify, {}, suggest)
    assert tool.forward("running", "huning")["score"] == 0.83
    wrong = tool.forward("running", "hunning")
    assert wrong["score"] is None
    assert not clears_phonetic_floor(["running"], ["hunning"], {("running","hunning"):wrong}, 0.60)


def test_default_launch_uses_bundled_colab_and_bank_before_renting(tmp_path, monkeypatch):
    bank = tmp_path / "verified.py"
    bank.write_text("private fixture")
    output = tmp_path / "run"
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture")
    monkeypatch.setenv("CHUCKLES_2025_BANK_GZ", "fixture")
    monkeypatch.setattr(sys, "argv", ["run", "--output-dir", str(output)])
    monkeypatch.setattr(run_gguf_test, "snapshot_bank", lambda explicit, directory: bank)
    calls = []
    def check(path):
        calls.append(("preflight", path))
        return {"passed": True}
    monkeypatch.setattr(run_gguf_test, "preflight", check)
    api = Mock(headers={})
    def create(*args, **kwargs):
        calls.append(("create", None))
        response = Mock(ok=True)
        response.json.return_value = {"id": "fixture", "costPerHr": 0.1}
        return response
    api.post.side_effect = create
    monkeypatch.setattr(run_gguf_test.requests, "Session", lambda: api)
    monkeypatch.setattr(run_gguf_test, "wait_ready", lambda *args: {})
    terminate = Mock()
    monkeypatch.setattr(run_gguf_test, "terminate", terminate)
    child = Mock()
    monkeypatch.setattr(run_gguf_test.subprocess, "run", child)
    run_gguf_test.main(tmp_path)
    command = child.call_args.args[0]
    assert calls == [("preflight", bank), ("create", None)]
    assert command[1].endswith("generate_march.py") and "--humor-first" in command
    assert command[command.index("--example-notebook")+1] == str(march_source.NOTEBOOK)
    assert command[command.index("--word-bank")+1] == str(bank)
    assert command[command.index("--titles")+1].endswith("humor_sample_titles.csv")
    assert "CHUCKLES_2025_BANK_GZ" not in child.call_args.kwargs["env"]
    assert json.loads((output/"run.json").read_text())["termination_confirmed"]
    terminate.assert_called_once_with(api, "fixture")

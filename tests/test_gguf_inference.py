"""The GGUF smoke run requires both a real trace and a completed answer."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from generate_with_reasoning import generate_gguf


@pytest.mark.parametrize(
    "message,finish,complete",
    [
        ({"content": "Jurassic Pork", "reasoning_content": "Check the vowel."}, "stop", True),
        ({"content": "<think>Check the vowel.</think>Jurassic Pork"}, "stop", True),
        ({"content": "Jurassic Pork"}, "stop", False),
        ({"content": "", "reasoning_content": "Check the vowel."}, "length", False),
    ],
)
def test_gguf_completion(monkeypatch, message, finish, complete):
    response = Mock()
    response.json.return_value = {
        "model": "test", "choices": [{"message": message, "finish_reason": finish}],
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    args = SimpleNamespace(llama_server="http://localhost", base_model="test",
                           temperature=0.4, max_new_tokens=1280)
    reasoning, title, record = generate_gguf(args, "Jurassic Park", {})
    assert ("error" not in record) == complete
    assert record["raw_message"] == message
    if complete:
        assert (reasoning, title) == ("Check the vowel.", "Jurassic Pork")
    assert post.call_args.kwargs["json"]["chat_template_kwargs"]["enable_thinking"]


def test_march_word_bank_reaches_both_tools(tmp_path):
    from generate_march import load_word_bank, suggestions_for, verify

    bank = tmp_path / "word_structures.py"
    bank.write_text('funny_words = ["pork"]\ncustom_phones = {"pork": {"primary_phones": "P AO R K"}}\n'
                    'raise RuntimeError("historic modules are data only")\n')
    words, phones = load_word_bank(bank)
    suggester = Mock()
    suggester.forward.return_value = '{"suggestions":[{"word":"pork","similarity":0.82}]}'
    suggestions, _ = suggestions_for("Park!", suggester, json.dumps(words), phones)
    assert suggestions["park"][0]["similarity"] == 0.82
    assert suggester.forward.call_args.kwargs["custom_phones"] == phones
    verifier = Mock()
    verifier.forward.return_value = '{"similarity":{"similarity":0.60}}'
    result = verify("Jurassic Park", "Jurassic Pork", verifier, phones)
    assert result["passes"] and result["swaps"][0]["score"] == 0.60
    verifier.forward.assert_called_once_with(word="park", compare_to="pork", custom_phones=phones)


@pytest.mark.parametrize("candidate", ["Jurassic Park", "Jurassic Pork Chop", "Jurassic Blorp"])
def test_march_requires_all_swaps_to_be_verifiable(candidate):
    from generate_march import verify

    analyzer = Mock()
    analyzer.forward.return_value = '{"found":false}'
    assert not verify("Jurassic Park", candidate, analyzer, {})["passes"]

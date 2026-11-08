"""The GGUF smoke run requires both a real trace and a completed answer."""

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

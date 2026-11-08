import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from blind_film_run import model_call, object_schema


def test_incomplete_response_retains_reasoning_and_retries_same_prompt(monkeypatch):
    inputs = []

    def fake_request(messages, seed, scale, schema, limit):
        inputs.append((messages, seed, scale, schema, limit))
        return {
            "model_output": '{"why":'
            if limit == 240
            else '{"why":"A double meaning."}',
            "reasoning": "Consider the original title.",
            "finish_reason": "length" if limit == 240 else "stop",
            "settings": {"max_tokens": limit},
        }

    monkeypatch.setattr("blind_film_run.request_step", fake_request)
    messages = [{"role": "user", "content": "Pick the funnier parody."}]
    schema = object_schema({"why": {"type": "string"}})
    result = model_call(messages, 42, 1, schema)
    assert [x[4] for x in inputs] == [240, 512]
    assert inputs[0][:3] == inputs[1][:3]
    assert "maxLength" not in schema["properties"]["why"]
    assert result["earlier_replies"][0]["reasoning"] == "Consider the original title."
    assert result["earlier_replies"][0]["model_output"] == '{"why":'
    assert result["finish_reason"] == "stop"

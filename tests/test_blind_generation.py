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


def test_native_stream_keeps_returned_thinking_when_connection_breaks(monkeypatch):
    import json
    import requests
    from blind_film_run import request_step

    class PartialResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self, chunk_size):
            yield (
                b"data: "
                + json.dumps(
                    {"choices": [{"delta": {"reasoning_content": "First thought."}}]}
                ).encode()
            )
            yield (
                b"data: "
                + json.dumps(
                    {"choices": [{"delta": {"content": '{"first":'}}]}
                ).encode()
            )
            raise requests.ConnectionError("stream stalled")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: PartialResponse())
    result = request_step([], 42, 0, None, 240)
    assert result["finish_reason"] == "transport_interrupted"
    assert (
        result["reasoning"] == "First thought."
        and result["model_output"] == '{"first":'
    )


def test_recovery_with_simpler_grammar_still_requires_an_eligible_title(monkeypatch):
    replies = iter(
        [
            ("", "transport_interrupted"),
            ('{"winner":"Other"}', "stop"),
            ('{"winner":"Chosen"}', "stop"),
        ]
    )

    def request(messages, seed, scale, schema, limit):
        text, finish = next(replies)
        if limit != 240:
            assert "enum" not in schema["properties"]["winner"]
        return dict(
            model_output=text,
            finish_reason=finish,
            reasoning="",
            settings={"max_tokens": limit},
        )

    monkeypatch.setattr("blind_film_run.request_step", request)
    schema = object_schema({"winner": {"type": "string", "enum": ["Chosen"]}})
    result = model_call([], 42, 1, schema)
    assert (
        result["settings"]["max_tokens"] == 768 and len(result["earlier_replies"]) == 2
    )

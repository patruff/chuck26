import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from blind_film_run import run_film
from settings import default_prompt, live_revision, system_prompt
from review_request import review_rows, make_request, encode_request


def test_action_override_is_the_applied_prompt(monkeypatch, tmp_path):
    assert (
        "One changed word" in default_prompt()
        and "Pretty bad and tame" in default_prompt()
    )
    path = tmp_path / "prompt.txt"
    path.write_text("A surprising but intelligible adult double meaning.")
    monkeypatch.setenv("CHUCKLES_PROMPT_FILE", str(path))
    assert (
        live_revision()["applied_instructions"] == system_prompt() == path.read_text()
    )


@pytest.mark.parametrize(
    "suggestions", [{}, {"park": [{"word": "pork", "similarity": 0.65}]}]
)
def test_native_model_can_pick_a_single_off_list_swap(
    monkeypatch, tmp_path, suggestions
):
    inputs = []

    def reply(messages, seed, scale, schema):
        inputs.append(messages)
        if "winner" in schema["properties"]:
            assert set(schema["properties"]["winner"]["enum"]) == {
                "Jurassic Pork",
                "Jurassic Lark",
            }
            answer = {
                "winner": "Jurassic Lark",
                "why": "A filthy practical joke at the park.",
            }
        else:
            answer = {
                "first": "Jurassic Pork",
                "second": "Jurassic Lark",
                "idea": "Two different premises.",
            }
        return dict(model_output=json.dumps(answer), reasoning="Compare the two jokes.")

    class Phone:
        def __init__(self, *args):
            self.calls = []

        def forward(self, a, b):
            score = 1 if a == b else 0.65
            observation = dict(
                word=a,
                compare_to=b,
                score=score,
                suggestion_score=score,
                word_phone_score=1,
            )
            self.calls.append(dict(**observation, returned=json.dumps(observation)))
            return observation

    monkeypatch.setattr("blind_film_run.model_call", reply)
    monkeypatch.setattr("blind_film_run.WordPhoneTool", Phone)
    s = {"jurassic": [], "park": [], **suggestions}
    run_film(
        {"rank": "7", "title": "Jurassic Park"},
        0,
        (s, {}),
        None,
        None,
        {},
        tmp_path,
        None,
        "Audience fixture",
    )
    obj = json.load(gzip.open(tmp_path / "007-0.json.gz"))["row"]
    assert obj["parody"] == "Jurassic Lark" and obj["runner_up"] == "Jurassic Pork"
    assert all(p[0]["content"] == "Audience fixture" for p in inputs)
    assert obj["audience_prompt"] == "Audience fixture" and obj["passes_threshold"]


def test_dpo_request_uses_choices_without_private_comments(tmp_path):
    p = tmp_path / "reviews.jsonl"
    row = dict(
        pair_id="one",
        film={"title": "Demo"},
        prompt=[
            {"role": "system", "content": "Audience"},
            {"role": "user", "content": "Movie: Demo"},
        ],
        chosen=[{"role": "assistant", "content": "A title"}],
        rejected=[{"role": "assistant", "content": "B title"}],
        feedback="Private commentary",
    )
    p.write_text(json.dumps(row) + "\n")
    request = make_request(p, "key")
    assert len(review_rows(p)) == 1 and "Private commentary" not in json.dumps(request)
    assert encode_request(request).startswith("gzip:")
    p.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="repeated"):
        review_rows(p)

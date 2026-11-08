import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from human_guidance import BASE_INSTRUCTIONS, prepare_guidance, revise_prompt, write_next_feedback


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        return SimpleNamespace(ids=list(map(ord, text)))

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_feedback_preserves_associations_and_excerpts_long_traces():
    source = ("previous result\r\nHUMAN_FEEDBACK: Go wilder.\r\nPREVIOUS_STYLE: first style\n"
              "MODEL_RUN: small example\nMODEL_RUN: beginning" + "x" * 20000 + "ending\n"
              "HUMAN_FEEDBACK:\nKeep the title intelligible.\nPREVIOUS_STYLE: newer style\nHUMAN_FEEDBACK:\n")
    plan = prepare_guidance(source.encode(), CharacterTokenizer())
    assert plan["feedback_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert plan["human_feedback"] == ["Go wilder.", "Keep the title intelligible."]
    context = plan["editor_context"]
    assert context.index("Go wilder.") < context.index("small example") < context.index("Keep the title intelligible.")
    assert "beginning" in context and "ending" in context and "middle of example omitted" in context
    assert "newer style" in context and "first style" not in context and plan["example_excerpts"] == 1
    assert len(context) < 11500 and plan["include_verbatim_in_task"]


def test_missing_or_over_context_comments_fail_before_inference():
    for text in ("MODEL_RUN: A historic trace\nHUMAN_FEEDBACK:\n", "HUMAN_FEEDBACK: " + "z" * 12000):
        with pytest.raises(ValueError):
            prepare_guidance(text.encode(), CharacterTokenizer())


def test_model_revision_and_editable_next_input_round_trip(tmp_path):
    source = "MODEL_RUN:\nPop Gun\nHUMAN_FEEDBACK:\nGive it higher stakes."
    (tmp_path / "guidance.txt").write_text(source)
    plan = prepare_guidance(source.encode(), CharacterTokenizer())
    model = Mock(model_id="Qwen")
    model.generate.return_value.content = '{"style_instructions":"Give each gag a dangerous consequence."}'
    revision = revise_prompt(model, plan)
    assert model.generate.call_args.args[0][1]["content"][0]["text"] == plan["editor_context"]
    assert revision["task_comments"] == ["Give it higher stakes."]
    assert BASE_INSTRUCTIONS in revision["applied_instructions"]
    assert revision["applied_instructions"].endswith("Give each gag a dangerous consequence.")
    row = {"input_title":"Top Gun", "qualifying_parody":"Pop Gun", "attempts":[
        {"title":"Pop Gun","why":"A dogfight.\nHUMAN_FEEDBACK: fake model remark"}]}
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n")
    write_next_feedback(tmp_path, revision)
    result = (tmp_path / "next-feedback.txt").read_text()
    following = prepare_guidance((result + "Keep the dogfight.").encode(), CharacterTokenizer())
    assert following["human_feedback"] == ["Give it higher stakes.", "Keep the dogfight."]
    assert revision["style_instructions"] in following["editor_context"] and "Qualifying parody: Pop Gun" in result


def test_large_human_comment_is_preserved_in_full_for_revision():
    note = "abcd " * 500
    plan = prepare_guidance(("HUMAN_FEEDBACK:\n" + note).encode(), CharacterTokenizer())
    assert not plan["include_verbatim_in_task"] and note.strip() in plan["editor_context"]


def test_editor_retries_malformed_reply_without_using_default_style():
    model = Mock(model_id="Qwen")
    model.generate.side_effect = [SimpleNamespace(content="{}"),
                                  SimpleNamespace(content='{"style_instructions":"Escalate the stakes."}')]
    plan = prepare_guidance(b"HUMAN_FEEDBACK: Make it wilder.", CharacterTokenizer())
    revision = revise_prompt(model, plan)
    assert revision["editor_attempt"] == 2 and revision["style_instructions"] == "Escalate the stakes."
    model.generate.side_effect = [SimpleNamespace(content="{}"), SimpleNamespace(content="{}")]
    with pytest.raises(ValueError, match="usable style_instructions"):
        revise_prompt(model, plan)

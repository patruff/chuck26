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
    from generate_march import WordPhoneTool, load_word_bank, suggestions_for, score_from_calls

    bank = tmp_path / "word_structures.py"
    bank.write_text('funny_words = ["pork"]\ncustom_phones = {"pork": {"primary_phones": "P AO R K"}}\n'
                    'raise RuntimeError("historic modules are data only")\n')
    words, phones = load_word_bank(bank)
    suggester = Mock()
    suggester.forward.return_value = '{"suggestions":[{"word":"pork","similarity":0.82}]}'
    suggestions, _ = suggestions_for("Park!", suggester, json.dumps(words), phones)
    assert suggestions["park"]["suggestions"][0]["similarity"] == 0.82
    assert suggester.forward.call_args.kwargs["custom_phones"] == phones
    verifier = Mock()
    verifier.forward.return_value = '{"similarity":{"similarity":0.60}}'
    tool = WordPhoneTool(verifier, phones)
    tool.forward("Park", "Pork")
    result = score_from_calls("Jurassic Park", "Jurassic Pork", {("park", "pork"): tool.calls[0]})
    assert result["passes"] and result["swaps"][0]["score"] == 0.60
    verifier.forward.assert_called_once_with(word="park", compare_to="pork", custom_phones=phones)


@pytest.mark.parametrize("candidate", ["Jurassic Park", "Jurassic Pork Chop", "Jurassic Blorp"])
def test_march_requires_all_swaps_to_be_verifiable(candidate):
    from generate_march import score_from_calls

    observed = {("park", "blorp"): {"score": None}}
    assert not score_from_calls("Jurassic Park", candidate, observed)["passes"]


def test_codeagent_retries_unobserved_final_and_uses_real_tool_feedback():
    from smolagents import Model
    from smolagents.models import ChatMessage, MessageRole
    from generate_march import run_agent

    titles = ["Jurassic Pork", "Jurassic Lark", "Jurassic Shark"]
    answer = {"attempts": [{"title": t, "why": "a pitch"} for t in titles],
              "winner": titles[0], "runner_up": titles[1], "explanation": "movie premise"}
    final = f"final_answer({answer!r})"

    class ScriptedModel(Model):
        def generate(self, messages, **kwargs):
            history = str([m.content for m in messages])
            if "Print the phonetic tool results" not in history:
                code = final
            elif '1.12' not in history:
                code = "\n".join(f'print(word_phone_tool("park", "{w}"))' for w in ("pork", "lark", "shark"))
            else:
                code = final
            content = "<code>\n" + code + "\n</code>"
            message = SimpleNamespace(content=content, reasoning_content="fixture reasoning")
            raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
            return ChatMessage(role=MessageRole.ASSISTANT, content=content, raw=raw)

    analyzer = Mock()
    analyzer.forward.return_value = json.dumps({"similarity": {"similarity": 1.12},
                                              "phones": ["P"], "comparison": {"phones": ["K"]}})
    row = {}
    run_agent(ScriptedModel(model_id="test"), "Jurassic Park", {}, {}, analyzer, row)
    assert row["state"] == "success" and row["parody"] == "Jurassic Pork"
    assert len(row["steps"]) == 3 and row["steps"][0]["error"]
    assert all(call["visible_in_observation"] and call["step"] == 2 for call in row["tool_calls"])
    assert all(s["reasoning"] == "fixture reasoning" for s in row["steps"])
    assert json.loads(row["tool_calls"][0]["returned"])["replacement_phones"] == ["K"]
    assert row["attempts"][0]["swaps"][0]["score"] == 1.12


def test_colab_word_list_is_data_only(tmp_path):
    from generate_march import notebook_words
    notebook = tmp_path / "source.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code",
                        "source": ["funny_words = ['one', 'one', 'two']\n", "raise Exception()"]}]}))
    assert notebook_words(notebook) == ["one", "one", "two"]


def test_running_man_prompt_renders_only_input_substitutions(tmp_path):
    from generate_march import RUNNING_MAN_CELL, running_man_prompt
    notebook = tmp_path / "May.ipynb"
    code = ('raise RuntimeError("never execute notebook code")\n'
            'def generate_parody(title):\n'
            '    prompt = f"Title {title}; suggestions {json.dumps(suggestions, indent=2)}"\n')
    notebook.write_text(json.dumps({"cells": [{"metadata": {"id": RUNNING_MAN_CELL}, "source": [code]}]}))
    suggestions = {"Man": {"suggestions": []}}
    assert running_man_prompt(notebook, "The Running Man", suggestions) == (
        "Title The Running Man; suggestions " + json.dumps(suggestions, indent=2))


def test_gguf_stop_tags_apply_to_visible_action_only():
    from generate_march import ReasoningGGUFModel
    model = ReasoningGGUFModel(model_id="fixture", api_base="http://localhost/v1", api_key="test")
    msg = SimpleNamespace(role="assistant", content="<code>final_answer(2)</code>extra", tool_calls=None,
                          reasoning_content="Plan: use </code> after the action.")
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10))
    endpoint = Mock(return_value=response)
    model.client.chat.completions.create = endpoint
    reply = model.generate([{"role": "user", "content": "test"}], stop_sequences=["</code>"])
    assert "stop" not in endpoint.call_args.kwargs
    assert reply.content == "<code>final_answer(2)"
    assert reply.raw.choices[0].message.reasoning_content == msg.reasoning_content


def test_exact_profile_requires_the_unchanged_pair_for_both_finalists():
    from smolagents import Model
    from smolagents.models import ChatMessage, MessageRole
    from generate_march import run_agent

    options = ["The Nutting Man", "The Rutted Man", "The Running Van"]
    answer = {"attempts": [{"title": t, "why": "pitch", "humor": 7} for t in options],
              "winner": options[0], "runner_up": options[2], "explanation": "scene"}
    final = f"final_answer({answer!r})"

    class Script(Model):
        def __init__(self):
            super().__init__(model_id="fixture")
            self.actions = iter([
                'for w, v in [("running","nutting"), ("man","man"), ("running","running"), ("man","van")]:\n'
                '    print(word_phone_tool(w, v)["score"])',
                final,
                'print(word_phone_tool("the", "the")["score"])',
                final,
            ])

        def generate(self, messages, **kwargs):
            return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\n" + next(self.actions) + "\n</code>")

    analyzer = Mock()
    analyzer.forward.return_value = json.dumps({"similarity":{"similarity":0.9}})
    row = {}
    run_agent(Script(), "The Running Man", {}, {}, analyzer, row, reference_prompt="Original prompt")
    assert row["state"] == "success" and "the" in row["steps"][1]["error"]
    assert all(len(checks) == 3 for checks in row["finalist_checks"].values())
    assert all(c["visible_in_observation"] for c in row["tool_calls"])
    assert all(c["returned"] in row["steps"][c["step"]-1]["observations"] for c in row["tool_calls"])


@pytest.mark.parametrize("guided", [False, True])
def test_humor_first_records_the_jokes_before_looking_at_scores(guided):
    from smolagents import Model
    from smolagents.models import ChatMessage, MessageRole
    from generate_march import run_agent

    drafts = [{"title": "Pork", "humor": 9, "why": "a pitch"},
              {"title": "Lark", "humor": 7, "why": "another pitch"},
              {"title": "Shark", "humor": 6, "why": "a third pitch"}]

    class Script(Model):
        custom_role_conversions = {}

        def __init__(self):
            super().__init__(model_id="fixture")
            self.actions = iter([
                'print(word_phone_tool("Park", "Pork"))',
                f'print(record_drafts({drafts!r}, ["Pork", "Lark"]))\n'
                'for w in ["Pork", "Lark"]:\n    print(word_phone_tool("Park", w))',
                'final_answer({"winner":"Lark","runner_up":"Pork","explanation":"I like the unlisted title"})',
                'final_answer({"winner":"Pork","runner_up":"Lark","explanation":"the stronger joke"})',
            ])

        def generate(self, messages, **kwargs):
            return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\n" + next(self.actions) + "\n</code>")

    analyzer = Mock()
    analyzer.forward.return_value = json.dumps({"similarity":{"similarity":1.0}})
    suggester = Mock()
    suggester.forward.side_effect = [json.dumps({"suggestions":[{"word":w, "similarity":v}]})
                                     for w, v in (("pork", 0.6), ("lark", 1.0))]
    row = {}
    revision = {"applied_instructions": "Give the joke higher stakes.", "task_comments": ["Go wilder."],
                "applied_instructions_sha256": "fixture-style", "feedback_sha256": "fixture-input"} if guided else None
    suggested = {"park": {"suggestions": [{"word": "pork", "similarity": 0.6}]}}
    run_agent(Script(), "Park", suggested, {}, analyzer, row, "scores > 0.8 and scores > 0.6", True, suggester,
              revision=revision)
    assert row["state"] == "success" and row["parody"] == "Pork" and row["passes_threshold"]
    assert row["steps"][0]["error"] and row["draft_rounds"][0]["prior_phone_calls"] == 0
    assert row["draft_rounds"][0]["attempts"] == drafts
    assert row["finalist_checks"]["Pork"][0]["score"] == 0.6
    assert row["finalist_checks"]["Lark"][0]["score"] == 1.0
    assert "suggested-word finalist" in row["steps"][-2]["error"]
    assert "0.8" not in row["task"] and row["steps"][-1]["input_messages"]
    if guided:
        assert "Go wilder." in row["task"]
        for step in row["steps"]:
            system = step["input_messages"][0]
            assert system["role"] == "system" and "Give the joke higher stakes." in str(system["content"])


def test_humor_floor_requires_observed_nonnul_scores():
    from humor_batch import clears_phonetic_floor
    assert clears_phonetic_floor(["park"], ["pork"], {("park", "pork"): {"score": 0.6}}, 0.6)
    for scores in [{}, {("park", "pork"): {"score": None}}, {("park", "pork"): {"score": 0.599}}]:
        assert not clears_phonetic_floor(["park"], ["pork"], scores, 0.6)


def test_candidate_export_uses_only_clean_turns_and_actual_input_messages():
    from export_humor_candidates import candidate_turns

    inputs = [{"role":"system","content":"agent instructions"}, {"role":"user","content":"movie prompt"}]
    step = {"step": 1, "final": True, "finish_reason":"stop", "reasoning":"A comic premise.",
            "code":'final_answer("title")', "model_output":'<code>final_answer("title")</code>extra',
            "input_messages":inputs}
    record = {"state":"success","passes_threshold":True,"input_title":"Movie","parody":"Parody",
              "model":"model","seed":42,"draft_rounds":[{"prior_phone_calls":0,"finalists":["Parody","Other"]}],
              "tool_calls":[{"word":"movie","compare_to":"parody","visible_in_observation":True}], "steps":[step]}
    turn, = candidate_turns(record)
    assert turn["prompt"] == inputs and turn["review_status"] == "unreviewed"
    assert turn["completion"] == [{"role":"assistant","reasoning_content":"A comic premise.",
                                   "content":'<code>final_answer("title")</code>'}]
    assert not list(candidate_turns({**record,"passes_threshold":False}))
    assert not list(candidate_turns({**record,"tool_calls":[{"word":"movie","compare_to":"unused","visible_in_observation":True}]}))
    assert not list(candidate_turns({**record,"steps":[{**step,"error":"invalid action"}]}))
    assert not list(candidate_turns({**record,"steps":[{**step,"finish_reason":"length"}]}))
    assert not list(candidate_turns(record, token_count=lambda _: 1535, thinking_budget=1536))
    below, = candidate_turns(record, token_count=lambda _: 1000, thinking_budget=1536)
    assert below["reasoning_tokens"] == 1000 and below["thinking_budget_checked"]


def test_humor_mode_reports_both_historical_scores_and_uses_suggestion_for_floor():
    from generate_march import WordPhoneTool
    from humor_batch import clears_phonetic_floor
    suggester, verifier = Mock(), Mock()
    suggester.forward.return_value = json.dumps({"suggestions":[{"word":"nutting","similarity":0.605}]})
    verifier.forward.return_value = json.dumps({"similarity":{"similarity":1.0}})
    phones = {"nutting":{"primary_phones":"N AH1 T IH0 NG"}}
    tool = WordPhoneTool(verifier, phones, suggester)
    receipt = tool.forward("Running", "Nutting")
    assert receipt["word_phone_score"] == 1.0 and receipt["suggestion_score"] == receipt["score"] == 0.605
    assert receipt["score_basis"] == "parody-suggestions"
    assert clears_phonetic_floor(["running"], ["nutting"], {("running","nutting"):receipt}, 0.6)
    suggester.forward.assert_called_once_with(target="running",word_list_str='["nutting"]',min_similarity="0.0",custom_phones=phones)
    suggester.forward.return_value = '{"suggestions":[]}'
    unscored = tool.forward("running", "madeupword")
    assert unscored["score"] is None and unscored["word_phone_score"] == 1.0
    identity = tool.forward("Man", "man")
    assert identity["score"] == 1.0 and identity["score_basis"] == "identical spelling"


def test_revised_slate_cannot_erase_a_verified_suggested_winner():
    from smolagents import Model
    from smolagents.models import ChatMessage, MessageRole
    from generate_march import run_agent

    def record(titles):
        drafts = [{"title": t, "humor": 9-i, "why": "a comic premise"} for i,t in enumerate(titles)]
        return (f"print(record_drafts({drafts!r}, {titles[:2]!r}))\n"
                f"for title in {titles[:2]!r}:\n"
                "    for a,b in zip('Toy Story'.split(),title.split()):\n"
                "        print(word_phone_tool(a,b))")

    class Script(Model):
        custom_role_conversions = {}
        def __init__(self):
            super().__init__(model_id="fixture")
            self.actions = iter([record(["Boy Toy","Hoe Story","Toy Gore"]),
                                 record(["Hoe Store","Boy Toy","Hoe Story"]),
                                 'final_answer({"winner":"Hoe Store","runner_up":"Boy Toy","explanation":"fallback"})',
                                 'final_answer({"winner":"Hoe Story","runner_up":"Boy Toy","explanation":"original winner"})'])
        def generate(self, messages, **kwargs):
            return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\n"+next(self.actions)+"\n</code>")

    raw = {"toy":{"suggestions":[{"word":"hoe","similarity":0.605},{"word":"boy","similarity":0.65}]}}
    pair_scores = {"boy":0.65,"toy":0.045,"hoe":0.605,"story":0.805,"store":0.385}
    def suggest(target, word_list_str, **kwargs):
        word = json.loads(word_list_str)[0]
        return json.dumps({"suggestions":[{"word":word,"similarity":pair_scores[word]}]})
    suggester, analyzer = Mock(), Mock()
    suggester.forward.side_effect = suggest
    analyzer.forward.return_value = '{"similarity":{"similarity":1.0}}'
    row = {}
    run_agent(Script(), "Toy Story", raw, {}, analyzer, row, "Creative phase", True, suggester)
    assert "Hoe Story" in row["steps"][2]["error"]
    assert row["state"] == "success" and row["qualifying_parody"] == "Hoe Story"
    assert row["selected_draft_round"] == 1 and len(row["draft_rounds"]) == 2
    assert row["finalist_checks"]["Hoe Story"][0]["suggestion_score"] == 0.605

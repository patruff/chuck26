import hashlib
import sys
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import generate_film_pairs as batch
from run_gguf_test import pod_config


def test_paired_slots_give_every_request_the_original_context():
    config = pod_config("test", "fixture", "NVIDIA L40S", "SECURE", 1536, 16384, 6)
    assert "-c 98304 -np 6" in config["dockerStartCmd"][0]


def test_resume_keeps_successful_samples_and_retries_only_the_missing_side(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "worker", {}, raising=False)
    generate = Mock(
        side_effect=[
            {"state": "success", "parody": "A"},
            {"error": "temporary"},
            {"error": "temporary"},
            {"state": "success", "parody": "B"},
        ]
    )
    monkeypatch.setattr(batch, "generate_title", generate)
    film = {"rank": "1", "title": "The Matrix", "year": "1999"}
    a = film, "a", tmp_path, "signature"
    b = film, "b", tmp_path, "signature"
    assert batch.generate_side(a) and not batch.generate_side(b)
    original = (tmp_path / "001-a.json.gz").read_bytes()
    assert batch.generate_side(a) and batch.generate_side(b)
    assert original == (tmp_path / "001-a.json.gz").read_bytes()
    assert len(batch.read_gzip(tmp_path / "001-b.json.gz")["failed_attempts"]) == 2
    assert [v.args[1] for v in generate.call_args_list] == [1042, 1542, 1543, 1544]


def test_qwen_recovery_can_choose_a_funnier_verified_off_list_swap():
    from types import SimpleNamespace
    from film_selection import checked_options, finish_sample

    def draft(title):
        return {"title": title, "why": "a different joke", "humor": 8}

    def receipt(word, replacement, score):
        return {
            "word": word,
            "compare_to": replacement,
            "score": score,
            "suggestion_score": score,
            "word_phone_score": 1.0,
            "score_basis": "parody-suggestions",
            "step": 2,
            "visible_in_observation": True,
        }

    row = {
        "input_title": "Park",
        "seed": 10042,
        "state": "max_steps_error",
        "error": "agent exhausted",
        "steps": [{"step": 1, "reasoning": "Three pitches."}],
        "suggestion_tool_results": {"park": {"suggestions": [{"word": "pork"}]}},
        "draft_rounds": [
            {
                "round": 1,
                "finalists": ["Pork", "Lark"],
                "attempts": list(map(draft, ["Pork", "Lark", "Dark"])),
            },
            {
                "round": 2,
                "finalists": ["Mark", "Arc"],
                "attempts": list(map(draft, ["Mark", "Arc", "Bark"])),
            },
        ],
        "tool_calls": [
            receipt("park", w, s)
            for w, s in [("pork", 0.60), ("lark", 0.82), ("mark", 0.3), ("arc", None)]
        ],
    }
    opts, _ = checked_options(row)
    assert [o["winner"] for o in opts] == ["Pork", "Lark"]
    assert [o["winner"] for o in checked_options(row, require_suggestion=True)[0]] == ["Pork"]
    model = Mock()
    model.generate.return_value = SimpleNamespace(
        content='{"choice":1,"explanation":"The stronger joke."}',
        raw=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        reasoning_content="Compare the checked pitches."
                    ),
                )
            ]
        ),
    )
    out = finish_sample(
        row, model, {"applied_instructions": "creative direction", "selection_policy":"humor-first-v2"}, 10044
    )
    assert out["qualifying_parody"] == "Lark" and out["runner_up"] == "Pork"
    assert (
        out["recovery"]["agent_state"] == "max_steps_error"
        and out["selected_draft_round"] == 1
    )
    assert (
        out["steps"][-1]["code"] is None
        and out["steps"][-1]["kind"] == "verified-slate decision"
    )
    assert row["state"] == "max_steps_error" and len(row["steps"]) == 1
    no_pass = {
        **row,
        "tool_calls": [
            {**c, "score": None, "suggestion_score": None} for c in row["tool_calls"]
        ],
    }
    assert (
        checked_options({**no_pass, "draft_rounds": no_pass["draft_rounds"][:1]})[0]
        == []
    )
    assert len(checked_options(no_pass)[0]) == 4


def test_transport_error_checkpoint_redacts_ephemeral_endpoint_and_key(
    tmp_path, monkeypatch
):
    key = "temporary-server-key-for-test"
    monkeypatch.setenv("LLAMA_API_KEY", key)
    path = tmp_path / "error.json.gz"
    batch.atomic_gzip(
        path, {"error": "source xyz-8080.proxy.runpod.net and " + key, "score": 0.665}
    )
    saved = batch.read_gzip(path)
    assert saved == {"error": "source [RunPod endpoint] and [redacted]", "score": 0.665}

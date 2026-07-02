"""Unit tests for pipeline/common.py pure helpers (no network, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from common import (
    align_swaps,
    build_reasoning_trace,
    build_user_prompt,
    compact_suggestions,
    parse_alpaca_text,
    split_output_explanation,
    split_think_answer,
)


class TestSplitOutputExplanation:
    def test_explanation_prefix(self):
        parody, note = split_output_explanation(
            "Mozfart: Explanation: Replacing 'art' with 'fart'. Category: juvenile."
        )
        assert parody == "Mozfart"
        assert note.startswith("Replacing 'art'")

    def test_plain_explanation(self):
        parody, note = split_output_explanation(
            "American Diaper: The title is altered to include a baby term."
        )
        assert parody == "American Diaper"
        assert note.startswith("The title is altered")

    def test_no_explanation(self):
        assert split_output_explanation("Jurassic Pork") == ("Jurassic Pork", "")


class TestAlignSwaps:
    def test_single_swap(self):
        assert align_swaps("Jurassic Park", "Jurassic Pork") == [("park", "pork")]

    def test_multiple_swaps(self):
        assert align_swaps("nasa artemis 1", "gasa fartemis 1") == [
            ("nasa", "gasa"),
            ("artemis", "fartemis"),
        ]

    def test_no_swap(self):
        assert align_swaps("The Matrix", "The Matrix") == []

    def test_case_and_punctuation_insensitive(self):
        assert align_swaps("Park!", "park") == []

    def test_length_mismatch_aligns_prefix(self):
        assert align_swaps("Die Hard", "Die Hard Again") == []


class TestParseAlpacaText:
    def test_chuckles_clean720_format(self):
        text = (
            "### Instruction:\nMake a chucklebot parody for the following input"
            "\n\n### Context:\nblueface\n\n### Response: pooface"
        )
        assert parse_alpaca_text(text) == ("blueface", "pooface")

    def test_garbage_returns_none(self):
        assert parse_alpaca_text("not an alpaca row") is None


class TestSplitThinkAnswer:
    def test_full_think_block(self):
        think, answer = split_think_answer("<think>reasoning here</think>\nThe Mattress")
        assert think == "reasoning here"
        assert answer == "The Mattress"

    def test_missing_open_tag(self):
        think, answer = split_think_answer("reasoning</think>Jurassic Pork")
        assert think == "reasoning"
        assert answer == "Jurassic Pork"

    def test_no_think_block(self):
        think, answer = split_think_answer("Jurassic Pork")
        assert think == ""
        assert answer == "Jurassic Pork"


class TestCompactSuggestions:
    def test_trims_to_top_n_and_two_keys(self):
        raw = {
            "Park": {
                "target": "park",
                "suggestions": [
                    {"word": "pork", "similarity": 0.85, "phones": "P AO1 R K"},
                    {"word": "bark", "similarity": 0.8, "rhyme_score": 1.0},
                    {"word": "fart", "similarity": 0.65},
                ],
            },
            "The": {"target": "the", "suggestions": [], "skipped": True},
        }
        compact = compact_suggestions(raw, top_n=2)
        assert compact["Park"] == [
            {"word": "pork", "similarity": 0.85},
            {"word": "bark", "similarity": 0.8},
        ]
        assert compact["The"] == []


class TestPromptAndTrace:
    def test_prompt_contains_title_and_suggestions(self):
        compact = {"Park": [{"word": "pork", "similarity": 0.85}]}
        prompt = build_user_prompt("Jurassic Park", compact)
        assert "Jurassic Park" in prompt
        assert "pork" in prompt

    def test_trace_ends_with_parody_after_think(self):
        compact = {"Park": [{"word": "pork", "similarity": 0.85}]}
        trace = build_reasoning_trace(
            "Jurassic Park", "Jurassic Pork", compact, {"park->pork": 0.85}
        )
        think, answer = split_think_answer(trace)
        assert answer == "Jurassic Pork"
        assert "0.85" in think

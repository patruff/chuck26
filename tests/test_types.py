"""Tests for Phase 2 data types, tools, and prompts modules."""

from __future__ import annotations

import pytest

from chuckles_prime.types import AgentTrace, GenerationRecord, ParodyCandidate
from chuckles_prime.prompts import PARODY_INSTRUCTIONS, build_generation_prompt


# ---------------------------------------------------------------------------
# ParodyCandidate
# ---------------------------------------------------------------------------


def test_parody_candidate_construction():
    """ParodyCandidate constructs with text + scores, humor_note defaults to ''."""
    pc = ParodyCandidate(text="Fartacus", phonetic_scores={"Spartacus": 0.85})
    assert pc.text == "Fartacus"
    assert pc.phonetic_scores == {"Spartacus": 0.85}
    assert pc.humor_note == ""


def test_parody_candidate_with_humor_note():
    """ParodyCandidate accepts a custom humor_note."""
    pc = ParodyCandidate(
        text="Fartacus",
        phonetic_scores={"Spartacus": 0.85},
        humor_note="Flatulence humor",
    )
    assert pc.humor_note == "Flatulence humor"


# ---------------------------------------------------------------------------
# AgentTrace
# ---------------------------------------------------------------------------


def test_agent_trace_construction():
    """AgentTrace constructs with steps=[], final_output='', token_usage=None, state='success'."""
    trace = AgentTrace(
        steps=[], final_output="", token_usage=None, state="success"
    )
    assert trace.steps == []
    assert trace.final_output == ""
    assert trace.token_usage is None
    assert trace.state == "success"


def test_agent_trace_with_data():
    """AgentTrace stores step data and token usage."""
    trace = AgentTrace(
        steps=[{"model_output": "thinking..."}],
        final_output='{"parody1": "test"}',
        token_usage={"input_tokens": 100, "output_tokens": 50},
        state="success",
    )
    assert len(trace.steps) == 1
    assert trace.token_usage["input_tokens"] == 100


# ---------------------------------------------------------------------------
# GenerationRecord
# ---------------------------------------------------------------------------


def test_generation_record_construction():
    """GenerationRecord constructs with all required fields, error defaults to None."""
    trace = AgentTrace(steps=[], final_output="", token_usage=None, state="success")
    record = GenerationRecord(
        input_title="Spartacus",
        candidates=[],
        trace=trace,
        model_name="qwen-3-32b",
    )
    assert record.input_title == "Spartacus"
    assert record.error is None


def test_generation_record_error_case():
    """GenerationRecord with error and empty candidates is valid."""
    trace = AgentTrace(steps=[], final_output="", token_usage=None, state="error")
    record = GenerationRecord(
        input_title="Spartacus",
        candidates=[],
        trace=trace,
        model_name="qwen-3-32b",
        error="Connection timeout",
    )
    assert record.error == "Connection timeout"
    assert record.candidates == []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_parody_instructions_content():
    """PARODY_INSTRUCTIONS is a non-empty string containing 'final_answer' and 'word_phonetic_analyzer'."""
    assert isinstance(PARODY_INSTRUCTIONS, str)
    assert len(PARODY_INSTRUCTIONS) > 0
    assert "final_answer" in PARODY_INSTRUCTIONS
    assert "word_phonetic_analyzer" in PARODY_INSTRUCTIONS


def test_build_generation_prompt_contains_title():
    """build_generation_prompt returns a string containing the title."""
    prompt = build_generation_prompt(
        title="The Matrix",
        suggestions={"The": {"skipped": True}, "Matrix": {"suggestions": []}},
        examples=[("Wolverine", "Pullverine", "Pull pun")],
        preferences_text="Make it absurd.",
    )
    assert isinstance(prompt, str)
    assert "The Matrix" in prompt
    assert "word_phonetic_analyzer" in prompt
    assert "Pullverine" in prompt
    assert "Make it absurd" in prompt


def test_build_generation_prompt_no_examples():
    """build_generation_prompt handles empty examples list."""
    prompt = build_generation_prompt(
        title="Gladiator",
        suggestions={"Gladiator": {"suggestions": ["Bladeiator"]}},
        examples=[],
        preferences_text="Punny style.",
    )
    assert "Gladiator" in prompt
    assert "no examples available" in prompt

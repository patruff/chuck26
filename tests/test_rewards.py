"""Tests for composite reward signal functions."""

from __future__ import annotations

from chuckles_prime.rewards import (
    compute_phonetic_quality,
    compute_structure_preservation,
    compute_tool_usage_completeness,
)
from chuckles_prime.types import AgentTrace, ParodyCandidate


# ---------------------------------------------------------------------------
# compute_phonetic_quality
# ---------------------------------------------------------------------------


def test_phonetic_quality_empty_scores():
    """Empty phonetic_scores dict returns 0.0."""
    candidate = ParodyCandidate(text="Fartacus", phonetic_scores={})
    assert compute_phonetic_quality(candidate) == 0.0


def test_phonetic_quality_single_score():
    """Single score returns that score."""
    candidate = ParodyCandidate(
        text="Fartacus", phonetic_scores={"Spartacus": 0.85}
    )
    assert compute_phonetic_quality(candidate) == 0.85


def test_phonetic_quality_multiple_scores():
    """Multiple scores returns the average."""
    candidate = ParodyCandidate(
        text="The Mattress",
        phonetic_scores={"The": 1.0, "Matrix": 0.78},
    )
    result = compute_phonetic_quality(candidate)
    assert abs(result - 0.89) < 0.01  # (1.0 + 0.78) / 2 = 0.89


# ---------------------------------------------------------------------------
# compute_tool_usage_completeness
# ---------------------------------------------------------------------------


def test_tool_usage_no_steps():
    """No steps means no words verified -> 0.0 for titles with words."""
    trace = AgentTrace(steps=[], final_output="", token_usage=None, state="success")
    assert compute_tool_usage_completeness(trace, "The Matrix") == 0.0


def test_tool_usage_short_words_skipped():
    """Words with len <= 2 are skipped. If all words are short, return 1.0."""
    trace = AgentTrace(steps=[], final_output="", token_usage=None, state="success")
    assert compute_tool_usage_completeness(trace, "It Is") == 1.0


def test_tool_usage_partial_match():
    """Only some title words appear in steps."""
    trace = AgentTrace(
        steps=[
            {"tool_call": "word_phonetic_analyzer", "args": "Matrix comparison"},
        ],
        final_output="",
        token_usage=None,
        state="success",
    )
    # "The" is len 3, "Matrix" is len 6. Both are significant.
    # Only "Matrix" appears in steps.
    result = compute_tool_usage_completeness(trace, "The Matrix")
    assert abs(result - 0.5) < 0.01  # 1 of 2 significant words


def test_tool_usage_full_match():
    """All significant words appear in steps."""
    trace = AgentTrace(
        steps=[
            {"tool_call": "analyzed The word"},
            {"tool_call": "analyzed Matrix word"},
        ],
        final_output="",
        token_usage=None,
        state="success",
    )
    result = compute_tool_usage_completeness(trace, "The Matrix")
    assert result == 1.0


# ---------------------------------------------------------------------------
# compute_structure_preservation
# ---------------------------------------------------------------------------


def test_structure_same_word_count():
    """Same word count returns 1.0."""
    assert compute_structure_preservation("The Matrix", "The Mattress") == 1.0


def test_structure_double_word_count():
    """Parody with double word count returns 0.5."""
    result = compute_structure_preservation("Matrix", "The Big Matrix")
    assert abs(result - (1 / 3)) < 0.01  # min(3,1)/max(3,1) = 1/3


def test_structure_empty_title():
    """Empty original title returns 0.0."""
    assert compute_structure_preservation("", "Something") == 0.0


def test_structure_half_words():
    """Parody with half word count."""
    result = compute_structure_preservation("The Matrix Reloaded", "The Matrix")
    assert abs(result - (2 / 3)) < 0.01  # min(2,3)/max(2,3) = 2/3

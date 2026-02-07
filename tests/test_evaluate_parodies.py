"""Tests for training/evaluate_parodies.py: phonetic scoring and parody evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add training directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from evaluate_parodies import (
    DEFAULT_TEST_TITLES,
    EvalReport,
    HAS_PRONOUNCING,
    ParodyResult,
    WordScore,
    _char_similarity,
    _levenshtein,
    _lcs_length,
    score_parody,
)

# Skip phonetic tests if pronouncing library is not installed
requires_pronouncing = pytest.mark.skipif(
    not HAS_PRONOUNCING,
    reason="pronouncing library not installed"
)


# ---------------------------------------------------------------------------
# Levenshtein distance tests
# ---------------------------------------------------------------------------


def test_levenshtein_identical():
    """Identical strings have distance 0."""
    assert _levenshtein("hello", "hello") == 0


def test_levenshtein_empty():
    """Distance to empty string is length of other."""
    assert _levenshtein("hello", "") == 5
    assert _levenshtein("", "world") == 5


def test_levenshtein_single_edit():
    """Single character difference is distance 1."""
    assert _levenshtein("cat", "bat") == 1
    assert _levenshtein("cat", "cats") == 1
    assert _levenshtein("cat", "at") == 1


def test_levenshtein_multiple_edits():
    """Multiple edits are counted correctly."""
    assert _levenshtein("kitten", "sitting") == 3


# ---------------------------------------------------------------------------
# Character similarity tests
# ---------------------------------------------------------------------------


def test_char_similarity_identical():
    """Identical strings have similarity 1.0."""
    assert _char_similarity("hello", "hello") == 1.0


def test_char_similarity_empty():
    """Empty string comparisons return 0.0."""
    assert _char_similarity("hello", "") == 0.0
    assert _char_similarity("", "world") == 0.0
    assert _char_similarity("", "") == 0.0


def test_char_similarity_one_edit():
    """One edit gives high similarity."""
    sim = _char_similarity("cat", "bat")
    assert 0.6 < sim < 1.0


def test_char_similarity_very_different():
    """Very different strings have low similarity."""
    sim = _char_similarity("abc", "xyz")
    assert sim == 0.0


# ---------------------------------------------------------------------------
# LCS tests
# ---------------------------------------------------------------------------


def test_lcs_identical():
    """Identical sequences have LCS = length."""
    seq = ["a", "b", "c"]
    assert _lcs_length(seq, seq) == 3


def test_lcs_empty():
    """Empty sequence has LCS 0."""
    assert _lcs_length([], ["a", "b"]) == 0
    assert _lcs_length(["a"], []) == 0


def test_lcs_partial():
    """Partial overlap."""
    assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert _lcs_length(["a", "b", "c"], ["b", "c", "d"]) == 2


def test_lcs_no_overlap():
    """No common elements."""
    assert _lcs_length(["a", "b"], ["c", "d"]) == 0


# ---------------------------------------------------------------------------
# score_parody tests
# ---------------------------------------------------------------------------


@requires_pronouncing
def test_score_parody_identical():
    """Identical title scores perfectly (requires pronouncing)."""
    result = score_parody("The Matrix", "The Matrix")
    assert result.avg_score == 1.0
    assert result.structure_score == 1.0
    assert result.passed is True


@requires_pronouncing
def test_score_parody_good_phonetic():
    """Good phonetic match passes (requires pronouncing)."""
    # "Matrix" -> "Mattress" is a known good match
    result = score_parody("The Matrix", "The Mattress")
    # Should have high avg score
    assert result.avg_score > 0.5
    assert len(result.word_scores) == 2


def test_score_parody_structure_preserved():
    """Same word count gives structure_score 1.0."""
    result = score_parody("Top Gun", "Top Bun")
    assert result.structure_score == 1.0


def test_score_parody_structure_different():
    """Different word count lowers structure_score."""
    result = score_parody("The Matrix", "Matrix")
    assert result.structure_score < 1.0


@requires_pronouncing
def test_score_parody_word_scores():
    """Word scores are computed for each position (requires pronouncing)."""
    result = score_parody("Die Hard", "Dye Hard")
    assert len(result.word_scores) == 2
    # "Die" -> "Dye" should score high
    assert result.word_scores[0].original == "die"
    assert result.word_scores[0].replacement == "dye"
    assert result.word_scores[0].score > 0.8


def test_score_parody_empty():
    """Empty parody fails."""
    result = score_parody("The Matrix", "")
    assert result.avg_score == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


def test_word_score_dataclass():
    """WordScore holds correct data."""
    ws = WordScore(original="cat", replacement="bat", score=0.85, passed=True)
    assert ws.original == "cat"
    assert ws.replacement == "bat"
    assert ws.score == 0.85
    assert ws.passed is True


def test_parody_result_dataclass():
    """ParodyResult initializes with defaults."""
    pr = ParodyResult(input_title="Test", generated_parody="Fest")
    assert pr.input_title == "Test"
    assert pr.generated_parody == "Fest"
    assert pr.word_scores == []
    assert pr.avg_score == 0.0
    assert pr.passed is False
    assert pr.error is None


def test_eval_report_dataclass():
    """EvalReport initializes with defaults."""
    report = EvalReport(model_name="test-model")
    assert report.model_name == "test-model"
    assert report.results == []
    assert report.total == 0
    assert report.passed == 0
    assert report.pass_rate == 0.0


# ---------------------------------------------------------------------------
# Default test titles
# ---------------------------------------------------------------------------


def test_default_titles_exist():
    """Default test titles are defined."""
    assert len(DEFAULT_TEST_TITLES) >= 10


def test_default_titles_are_movies():
    """Default titles are recognizable movies."""
    assert "The Matrix" in DEFAULT_TEST_TITLES
    assert "Die Hard" in DEFAULT_TEST_TITLES
    assert "Star Wars" in DEFAULT_TEST_TITLES


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_score_multiple_parodies():
    """Score a batch of parodies."""
    test_cases = [
        ("The Matrix", "The Mattress"),
        ("Die Hard", "Dye Hard"),
        ("Top Gun", "Top Bun"),
        ("Star Wars", "Star Whores"),
    ]

    for original, parody in test_cases:
        result = score_parody(original, parody)
        assert result.input_title == original
        assert result.generated_parody == parody
        assert 0.0 <= result.avg_score <= 1.0
        assert 0.0 <= result.structure_score <= 1.0


def test_score_case_insensitive():
    """Scoring is case-insensitive."""
    result1 = score_parody("The Matrix", "the mattress")
    result2 = score_parody("THE MATRIX", "THE MATTRESS")
    # Scores should be similar (not necessarily identical due to tokenization)
    assert abs(result1.avg_score - result2.avg_score) < 0.1

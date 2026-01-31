"""Composite reward signal computation for GRPO dataset metadata.

Three stateless functions that produce continuous float scores in [0.0, 1.0]
for phonetic quality, tool usage completeness, and structure preservation.
"""

from __future__ import annotations

from chuckles_prime.types import AgentTrace, ParodyCandidate


def compute_phonetic_quality(candidate: ParodyCandidate) -> float:
    """Average phonetic similarity score across all replaced words.

    Args:
        candidate: A parody candidate with phonetic_scores dict.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if no scores available.
    """
    import re

    numeric_scores: list[float] = []
    for v in candidate.phonetic_scores.values():
        try:
            numeric_scores.append(float(v))
        except (TypeError, ValueError):
            nums = re.findall(r"(\d+\.?\d*)", str(v))
            if nums:
                numeric_scores.append(float(nums[0]))
    if not numeric_scores:
        return 0.0
    return sum(numeric_scores) / len(numeric_scores)


def compute_tool_usage_completeness(trace: AgentTrace, input_title: str) -> float:
    """Fraction of title words that were phonetically verified in the trace.

    Args:
        trace: Agent reasoning trace with step dicts.
        input_title: Original input title string.

    Returns:
        Float in [0.0, 1.0]. Returns 1.0 if no significant words in title.
    """
    title_words = [w for w in input_title.split() if len(w) > 2]
    if not title_words:
        return 1.0

    verified_words: set[str] = set()
    for step in trace.steps:
        step_str = str(step).lower()
        for word in title_words:
            if word.lower() in step_str:
                verified_words.add(word.lower())

    return len(verified_words) / len(title_words)


def compute_structure_preservation(input_title: str, parody_text: str) -> float:
    """How well the parody preserves the word count of the original.

    Args:
        input_title: Original title string.
        parody_text: Parody title string.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if original is empty.
    """
    orig_words = input_title.split()
    parody_words = parody_text.split()

    if not orig_words:
        return 0.0

    return min(len(parody_words), len(orig_words)) / max(len(parody_words), len(orig_words))

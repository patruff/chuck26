"""HF Hub tool loading and pre-computation for parody generation.

Loads phonetic analysis tools from HuggingFace Hub and provides
pre-computation of parody suggestions outside the agent loop.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from smolagents import load_tool

if TYPE_CHECKING:
    from smolagents import Tool


def load_parody_tools() -> tuple[Tool, Tool]:
    """Load both phonetic tools from HuggingFace Hub.

    Returns:
        (parody_tool, phone_tool) -- parody_word_suggester and word_phonetic_analyzer
    """
    parody_tool = load_tool("patruff/parody-suggestions", trust_remote_code=True)
    phone_tool = load_tool("patruff/word-phone", trust_remote_code=True)
    return parody_tool, phone_tool


def pre_compute_suggestions(
    title: str,
    funny_words: dict[str, list[str]],
    parody_tool: Tool,
    min_similarity: str = "0.5",
) -> dict[str, Any]:
    """Pre-compute parody suggestions for each word in a title.

    Calls parody_word_suggester OUTSIDE the agent loop for efficiency.
    The agent only needs word_phonetic_analyzer for verification.

    Args:
        title: Input title string (e.g., "The Matrix").
        funny_words: Dict mapping categories to word lists from config.
        parody_tool: The loaded parody_word_suggester tool.
        min_similarity: Minimum phonetic similarity threshold as string.

    Returns:
        Dict mapping each word in title to its suggestion results.
    """
    word_list_str = json.dumps(
        [w for words in funny_words.values() for w in words]
    )
    suggestions: dict[str, Any] = {}
    for word in title.split():
        # Skip very short words (articles, prepositions)
        if len(word) <= 2:
            suggestions[word] = {"target": word, "suggestions": [], "skipped": True}
            continue
        try:
            result = parody_tool.forward(
                target=word,
                word_list_str=word_list_str,
                min_similarity=min_similarity,
            )
            suggestions[word] = json.loads(result)
        except Exception as e:
            suggestions[word] = {"target": word, "suggestions": [], "error": str(e)}
    return suggestions

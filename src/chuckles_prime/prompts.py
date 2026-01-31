"""System instructions and generation prompt builder for parody generation.

PARODY_INSTRUCTIONS is passed to CodeAgent(instructions=...) and gets
appended to smolagents' built-in system prompt via {{custom_instructions}}.
build_generation_prompt() constructs the per-title task prompt.
"""

from __future__ import annotations

import json
from typing import Any

PARODY_INSTRUCTIONS = """You are a parody title generator. Create phonetically similar but humorous parody versions of movie/show/song titles.

WORKFLOW:
1. Review the pre-computed suggestions provided in the task prompt
2. For each promising replacement, verify phonetic similarity with word_phonetic_analyzer(word=original, compare_to=replacement)
3. A score above 0.6 means acceptable phonetic similarity
4. Create at least 3 candidate parodies, then pick the 2 best
5. Call final_answer() with a JSON string containing your results

OUTPUT FORMAT (pass as JSON string to final_answer):
{
    "parody1": "First Parody Title",
    "parody2": "Second Parody Title",
    "attempts": [
        {"text": "Attempt Text", "scores": {"original_word": 0.85}, "humor_note": "Why it works"}
    ]
}

IMPORTANT: Always call final_answer() with a valid JSON string. Do not return plain text."""


def build_generation_prompt(
    title: str,
    suggestions: dict[str, Any],
    examples: list[tuple[str, str, str]],
    preferences_text: str,
) -> str:
    """Build a task prompt for the CodeAgent to generate parodies of a title.

    Args:
        title: The input title to parody.
        suggestions: Pre-computed suggestions from pre_compute_suggestions().
        examples: Human examples as (input, output, explanation) tuples.
        preferences_text: Style preferences text from config.

    Returns:
        Task prompt string to pass to agent.run(task=...).
    """
    # Format human examples (first 10)
    example_lines = []
    for inp, out, expl in examples[:10]:
        example_lines.append(f'  "{inp}" -> "{out}" ({expl})')
    examples_block = "\n".join(example_lines) if example_lines else "  (no examples available)"

    # Format suggestions as JSON
    suggestions_json = json.dumps(suggestions, indent=2)

    return f"""Create 2 phonetically-sound parody versions of this title:

TITLE: "{title}"

STYLE PREFERENCES:
{preferences_text}

KNOWN GOOD EXAMPLES (for style reference):
{examples_block}

PRE-COMPUTED SUGGESTIONS (parody word candidates for each word in the title):
{suggestions_json}

STEPS:
1. Review the pre-computed suggestions above for each word
2. For promising replacements, verify phonetic similarity using word_phonetic_analyzer(word=original_word, compare_to=replacement_word)
3. A phonetic similarity score above 0.6 is acceptable
4. Try at least 3 different parody combinations
5. Pick the 2 best and call final_answer() with a JSON string in the format specified in your instructions"""

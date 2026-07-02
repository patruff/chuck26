"""Shared helpers for the reasoning SFT pipeline.

The prompt built here is used by BOTH build_reasoning_dataset.py (training
data) and generate_with_reasoning.py (inference), so the model always sees
tool-computed suggestions in the same format it was trained on.

Pure-python helpers (word alignment, alpaca parsing, think-block parsing)
live here too so they can be unit-tested without smolagents or a GPU.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

REASONING_SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Replace words with phonetically similar but humorous alternatives. "
    "Think through the phonetic suggestions step by step inside <think> tags, "
    "then answer with ONLY the final parody title."
)

# Threshold used both in reasoning traces and inference-time scoring.
MIN_PHONETIC_SCORE = 0.6


def compact_suggestions(
    suggestions: dict[str, Any], top_n: int = 5
) -> dict[str, Any]:
    """Trim pre_compute_suggestions() output for prompt injection.

    Keeps only the top-N candidates per word as {word, similarity}, dropping
    phones/endings/sub-scores that would bloat the training context window.

    Args:
        suggestions: Raw output of pre_compute_suggestions().
        top_n: Max candidates to keep per title word.

    Returns:
        Dict mapping each title word to a list of {word, similarity} dicts.
    """
    compact: dict[str, Any] = {}
    for word, info in suggestions.items():
        if not isinstance(info, dict):
            compact[word] = []
            continue
        cands = info.get("suggestions") or []
        compact[word] = [
            {"word": c.get("word"), "similarity": c.get("similarity")}
            for c in cands[:top_n]
            if isinstance(c, dict)
        ]
    return compact


def build_user_prompt(title: str, suggestions: dict[str, Any]) -> str:
    """Build the user prompt with pre-computed tool suggestions injected.

    Args:
        title: The input title to parody.
        suggestions: Output of chuckles_prime.tools.pre_compute_suggestions().

    Returns:
        User prompt string shared by training data and inference.
    """
    suggestions_json = json.dumps(suggestions, indent=2)
    return f"""Create a phonetically-sound parody of: '{title}'

PHONETIC SUGGESTIONS (from the parody word suggester tool, per word):
{suggestions_json}

Pick replacements from the suggestions above (similarity >= {MIN_PHONETIC_SCORE} is acceptable),
keep the rest of the title intact, and answer with only the parody title."""


def clean_word(word: str) -> str:
    """Lowercase a word and strip surrounding punctuation."""
    return word.lower().strip(string.punctuation)


def align_swaps(original: str, parody: str) -> list[tuple[str, str]]:
    """Find (original_word, replacement_word) pairs between title and parody.

    Aligns word-by-word; positions where the cleaned words differ are swaps.
    If the word counts differ, aligns up to the shorter length.

    Args:
        original: Original title.
        parody: Parody title.

    Returns:
        List of (original_word, replacement_word) tuples, cleaned.
    """
    orig_words = original.split()
    par_words = parody.split()
    swaps: list[tuple[str, str]] = []
    for ow, pw in zip(orig_words, par_words):
        ow_c, pw_c = clean_word(ow), clean_word(pw)
        if ow_c and pw_c and ow_c != pw_c:
            swaps.append((ow_c, pw_c))
    return swaps


def parse_alpaca_text(text: str) -> tuple[str, str] | None:
    """Parse an alpaca-style row ('### Context:\\n<in>\\n\\n### Response: <out>').

    Used for the patruff/chucklesClean720 source dataset.

    Returns:
        (input_title, parody_output) or None if the row doesn't parse.
    """
    m = re.search(
        r"### Context:\s*\n(.+?)\n\s*\n### Response:\s*(.+)", text, re.DOTALL
    )
    if not m:
        return None
    inp = m.group(1).strip()
    out = m.group(2).strip()
    if not inp or not out:
        return None
    return inp, out


def phonetic_similarity(phone_tool: Any, original: str, replacement: str) -> float | None:
    """Score one word swap with the word_phonetic_analyzer HF tool.

    Args:
        phone_tool: Loaded patruff/word-phone tool.
        original: Original word.
        replacement: Replacement word.

    Returns:
        Similarity in [0, 1], or None if either word is out-of-dictionary
        or the tool errors.
    """
    try:
        raw = phone_tool.forward(word=original, compare_to=replacement)
        data = json.loads(raw)
    except Exception:
        return None
    sim = data.get("similarity")
    if isinstance(sim, dict):
        sim = sim.get("similarity")
    if isinstance(sim, (int, float)):
        return float(sim)
    return None


def score_parody(
    phone_tool: Any, original: str, parody: str
) -> tuple[dict[str, float], float]:
    """Score every swapped word pair in a parody with the phone tool.

    Args:
        phone_tool: Loaded patruff/word-phone tool.
        original: Original title.
        parody: Generated parody title.

    Returns:
        (per-swap scores keyed 'orig->repl', average score). Average is 0.0
        when no swap could be scored.
    """
    scores: dict[str, float] = {}
    for ow, pw in align_swaps(original, parody):
        sim = phonetic_similarity(phone_tool, ow, pw)
        if sim is not None:
            scores[f"{ow}->{pw}"] = sim
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return scores, avg


def split_think_answer(generated: str) -> tuple[str, str]:
    """Split generated text into (reasoning, final answer).

    Handles '<think>...</think>answer', a bare '</think>' (Qwen3 sometimes
    omits the opening tag), and plain text with no think block.
    """
    if "</think>" in generated:
        think_part, _, answer = generated.rpartition("</think>")
        think_part = think_part.replace("<think>", "").strip()
        return think_part, answer.strip()
    return "", generated.strip()


def build_reasoning_trace(
    title: str,
    parody: str,
    suggestions: dict[str, Any],
    swap_scores: dict[str, float],
) -> str:
    """Synthesize a deterministic <think> reasoning trace for an SFT target.

    Walks the tool suggestions and verified swap scores the way the agent
    would, ending with the known-good parody.

    Args:
        title: Original title.
        parody: Human-approved parody (the SFT answer).
        suggestions: compact_suggestions() output for the title.
        swap_scores: 'orig->repl' -> similarity for the actual swaps.

    Returns:
        Full assistant message: '<think>...</think>\\n<parody>'.
    """
    lines = [
        f'I need a parody of "{title}" that sounds nearly identical when '
        "spoken but means something absurd."
    ]

    swaps = align_swaps(title, parody)
    swapped_words = {ow for ow, _ in swaps}

    for word, cands in suggestions.items():
        word_c = clean_word(word)
        if len(word) <= 2:
            continue
        if cands:
            top = ", ".join(
                f"{c['word']} ({c['similarity']})" for c in cands[:4]
            )
            lines.append(f'Suggestions for "{word_c}": {top}.')
        elif word_c not in swapped_words:
            lines.append(
                f'No strong suggestions for "{word_c}" -- keeping it anchors '
                "the original sound."
            )

    for ow, pw in swaps:
        score = swap_scores.get(f"{ow}->{pw}")
        if score is not None:
            verdict = (
                "above the threshold, the swap holds up"
                if score >= MIN_PHONETIC_SCORE
                else "below the usual threshold, but the humor carries it"
            )
            lines.append(
                f'Verifying "{ow}" -> "{pw}" with the phonetic analyzer: '
                f"similarity {score} -- {verdict}."
            )
        elif ow[1:] == pw[1:]:
            lines.append(
                f'Swapping "{ow}" -> "{pw}" changes only the opening sound; '
                "the rhythm and vowels stay identical."
            )
        elif ow[:1] == pw[:1]:
            lines.append(
                f'Swapping "{ow}" -> "{pw}" keeps the opening sound and '
                "overall shape of the word."
            )
        else:
            lines.append(
                f'Swapping "{ow}" -> "{pw}" mirrors the original\'s '
                "syllable pattern, so it still scans the same way."
            )

    if not swaps:
        lines.append(
            "The parody keeps the title's phonetic skeleton intact, so it "
            "reads as the original at first glance."
        )

    lines.append(f'Best combination: "{parody}".')
    think = "\n".join(lines)
    return f"<think>\n{think}\n</think>\n{parody}"

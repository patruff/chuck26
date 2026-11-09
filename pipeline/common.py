"""Title alignment and the March phonetic score used by both inference paths."""
import csv
import string
from typing import Any

MIN_PHONETIC_SCORE = 0.6


def read_titles(csv_path: str) -> list[str]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "title" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have a 'title' column")
        return [row["title"].strip() for row in reader if row.get("title", "").strip()]


def compact_suggestions(
    suggestions: dict[str, Any], top_n: int = 5
) -> dict[str, Any]:
    """Keep each suggested word and its March similarity score for the model. How many to show is set by top_n."""
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


def clean_word(word: str) -> str:
    """Lowercase a word and strip surrounding punctuation."""
    return word.lower().strip(string.punctuation)


def suggested_pairs(results):
    """Use only exact replacements actually returned for the matching source word."""
    return {(clean_word(word), clean_word(item["word"]))
            for word, result in results.items() for item in result.get("suggestions", [])
            if clean_word(word) != clean_word(item["word"])}


def title_changes(original, parody):
    before, after = original.split(), parody.split()
    if len(before) != len(after):
        return []
    return [(clean_word(a), clean_word(b)) for a, b in zip(before, after) if clean_word(a) != clean_word(b)]


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

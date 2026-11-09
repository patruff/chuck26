"""Record CodeAgent's comic pitches and nominations before it checks their sound."""
import re

from common import clean_word, title_changes, suggested_pairs
from smolagents import Tool


class RecordDrafts(Tool):
    name = "record_drafts"
    description = ("Save three or four different parody titles with their humor ratings and comic premises. "
                   "Nominate the two funniest with the original word count for word-by-word checks. "
                   "Call again when revising after feedback.")
    inputs = {
        "attempts": {"type": "array", "description": "List of dicts with title, humor (0–10), and why."},
        "finalists": {"type": "array", "description": "Two distinct titles from attempts, funniest first."},
        "skip_reason": {"type": "string", "nullable": True, "description":
                        "Optional comment on why a suggestion fell flat."},
    }
    output_type = "object"

    def __init__(self, original, calls, listed=(), *, require_suggestion=False):
        super().__init__()
        self.original = original
        self.calls = calls
        self.listed = set(listed)
        self.require_suggestion = require_suggestion
        self.rounds = []

    def forward(self, attempts: list, finalists: list, skip_reason: str | None = None) -> dict:
        if len(attempts) not in (3, 4):
            raise ValueError("Give three or four different pitches")
        drafts = []
        for a in attempts:
            if not isinstance(a, dict) or not isinstance(a.get("title"), str) or not a["title"].strip():
                raise ValueError("Each draft needs a title")
            if type(a.get("humor")) not in (int, float) or not 0 <= a["humor"] <= 10:
                raise ValueError("Rate each draft with a number from 0 to 10")
            if not isinstance(a.get("why"), str) or not a["why"].strip():
                raise ValueError("Give each draft its comic premise")
            drafts.append({key: a[key] for key in ("title", "humor", "why")})
        titles = [a["title"] for a in drafts]
        forms = {tuple(map(clean_word, t.split())) for t in titles}
        if len(forms) != len(titles) or tuple(map(clean_word, self.original.split())) in forms:
            raise ValueError("Each draft must be a distinct change to the original title")
        if len(finalists) != 2 or finalists[0] == finalists[1] or any(t not in titles for t in finalists):
            raise ValueError("Nominate two different titles from the recorded drafts")
        if any(len(t.split()) != len(self.original.split()) for t in finalists):
            raise ValueError("Choose finalists with the same word count so each source word can be checked")
        ratings = {a["title"]: a["humor"] for a in drafts}
        aligned = [t for t in titles if len(t.split()) == len(self.original.split())]
        in_list = [t for t in aligned if self.listed.intersection(title_changes(self.original, t))]
        listed_finalists = [t for t in finalists if t in in_list]
        if self.require_suggestion and self.listed and not listed_finalists:
            cited = set(re.findall(r"[\w'-]+", (skip_reason or "").casefold()))
            if not cited.intersection(b for _, b in self.listed) or not skip_reason.strip():
                raise ValueError("Pitch and nominate a returned swap for its own original word. If none makes a "
                                 "joke, pass skip_reason naming a real suggestion and why it falls flat.")
        # Reserve one finalist slot for the funniest supplied-word pitch. Rank the other slot on humor.
        pool, open_slots = aligned, finalists
        if self.require_suggestion and listed_finalists:
            best = max(listed_finalists, key=ratings.get)
            if ratings[best] < max(ratings[t] for t in in_list):
                raise ValueError("Nominate your highest-rated pitch using a returned suggestion.")
            pool = [t for t in aligned if t != best]
            open_slots = [t for t in finalists if t != best]
        if any(ratings[t] > min(ratings[c] for c in open_slots) for t in pool if t not in finalists):
            raise ValueError("Nominate your two funniest word-aligned drafts by your own ratings. Ties are your choice.")
        result = {"round": len(self.rounds) + 1, "attempts": drafts, "finalists": list(finalists),
                  "prior_phone_calls": len(self.calls)}
        if self.listed:
            result.update(suggested_finalists=listed_finalists,
                          skip_reason=skip_reason if not listed_finalists else None)
        self.rounds.append(result)
        return {"round": result["round"], "finalists": result["finalists"]}


def clears_phonetic_floor(original_words, replacement_words, observed, floor):
    if not original_words or len(original_words) != len(replacement_words):
        return False
    for pair in zip(original_words, replacement_words):
        receipt = observed.get(pair)
        if receipt is None or receipt["score"] is None or receipt["score"] < floor:
            return False
    return True

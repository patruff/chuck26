"""Record CodeAgent's comic pitches and nominations before it checks their sound."""
from common import clean_word
from smolagents import Tool


class RecordDrafts(Tool):
    name = "record_drafts"
    description = ("Save three or four different parody titles with their humor ratings and comic premises. "
                   "Nominate the two funniest with the original word count for word-by-word checks. "
                   "Call again when revising after feedback.")
    inputs = {
        "attempts": {"type": "array", "description": "List of dicts with title, humor (0–10), and why."},
        "finalists": {"type": "array", "description": "Two distinct titles from attempts, funniest first."},
    }
    output_type = "object"

    def __init__(self, original, calls):
        super().__init__()
        self.original = original
        self.calls = calls
        self.rounds = []

    def forward(self, attempts: list, finalists: list) -> dict:
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
        other_ratings = [ratings[t] for t in titles if t not in finalists and len(t.split()) == len(self.original.split())]
        if other_ratings and max(other_ratings) > min(ratings[t] for t in finalists):
            raise ValueError("Nominate your two funniest word-aligned drafts by your own ratings. Ties are your choice.")
        result = {"round": len(self.rounds) + 1, "attempts": drafts, "finalists": list(finalists),
                  "prior_phone_calls": len(self.calls)}
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import align_swaps, compact_suggestions


class TestAlignSwaps:
    def test_single_swap(self):
        assert align_swaps("Jurassic Park", "Jurassic Pork") == [("park", "pork")]

    def test_multiple_swaps(self):
        assert align_swaps("nasa artemis 1", "gasa fartemis 1") == [
            ("nasa", "gasa"),
            ("artemis", "fartemis"),
        ]

    def test_no_swap(self):
        assert align_swaps("The Matrix", "The Matrix") == []

    def test_case_and_punctuation_insensitive(self):
        assert align_swaps("Park!", "park") == []

    def test_length_mismatch_aligns_prefix(self):
        assert align_swaps("Die Hard", "Die Hard Again") == []


class TestCompactSuggestions:
    def test_trims_to_top_n_and_two_keys(self):
        raw = {
            "Park": {
                "target": "park",
                "suggestions": [
                    {"word": "pork", "similarity": 0.85, "phones": "P AO1 R K"},
                    {"word": "bark", "similarity": 0.8, "rhyme_score": 1.0},
                    {"word": "fart", "similarity": 0.65},
                ],
            },
            "The": {"target": "the", "suggestions": [], "skipped": True},
        }
        compact = compact_suggestions(raw, top_n=2)
        assert compact["Park"] == [
            {"word": "pork", "similarity": 0.85},
            {"word": "bark", "similarity": 0.8},
        ]
        assert compact["The"] == []

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from humor_batch import RecordDrafts, suggested_pairs
from trace_review import bank_use, write_transcripts


def test_list_membership_is_per_original_word_and_selection_is_humor_first():
    raw = {"toy": {"suggestions": [{"word": "toy"}, {"word": "boy"}]}, "story": {"suggestions": []}}
    listed = suggested_pairs(raw)
    assert listed == {("toy", "boy")}
    slate = RecordDrafts("Toy Story", [], listed)
    drafts = [{"title": t, "why": "a different comic premise", "humor": score} for t, score in
              [("Boy Story", 7), ("Toy Gory", 10), ("Toy Boy", 9), ("Toy Gore", 6)]]
    with pytest.raises(ValueError, match="returned swap"):
        slate.forward(drafts, ["Toy Gory", "Toy Boy"])
    with pytest.raises(ValueError, match="returned swap"):
        slate.forward(drafts, ["Toy Gory", "Toy Boy"], "Nothing sounded good.")
    slate.forward(drafts, ["Boy Story", "Toy Gory"])
    assert slate.rounds[-1]["suggested_finalists"] == ["Boy Story"]
    slate.forward(drafts, ["Toy Gory", "Toy Boy"], "Boy merely renames the character; there is no gag.")
    assert slate.rounds[-1]["skip_reason"].startswith("Boy")
    assert bank_use({"input_title":"Toy Story","suggestion_tool_results":raw}, "Toy Boy") == [("story","boy",False)]


def test_transcript_exposes_complete_real_steps_and_lookup_before_drafts(tmp_path):
    reason = "Opening thought. " + "more " * 300 + "last part of the first thought"
    row = {"input_title": "Toy Story", "word_count":1565, "seed":42,
           "suggestion_tool_results":{"toy":{"suggestions":[{"word":"boy","similarity":0.65}]}},
           "qualifying_parody":"Boy Story", "parody":"Boy Story", "steps":[
               {"step":1,"reasoning":reason,"code":"print(record_drafts(...))","observations":"nominated"},
               {"step":2,"reasoning":"Now check it.","code":"print(word_phone_tool('toy', 'boy'))","observations":"heard"},
               {"step":3,"reasoning":"Choose the better joke.","code":"final_answer(...)"}],
           "tool_calls":[{"step":2, "word":"toy", "compare_to":"boy", "returned":'{"suggestion_score": 0.65}'}]}
    f=tmp_path/"results.jsonl"
    f.write_text(json.dumps(row)+"\n")
    write_transcripts(f,tmp_path)
    text=(tmp_path/"transcripts/01-toy-story.md").read_text()
    assert text.index("boy (0.65)") < text.index(reason) < text.index("print(record_drafts") < text.index("Now check it.")
    assert text.index("Executed `word_phone_tool") < text.index("Choose the better joke.")
    assert '{"suggestion_score": 0.65}' in text and "toy → boy (listed)" in text and "<think>" in text
    assert "1 / 1" in (tmp_path/"TRACE-INDEX.md").read_text()

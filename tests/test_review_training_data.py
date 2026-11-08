import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_dpo import format_preference


class Template:
    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, enable_thinking
    ):
        assert not tokenize and not enable_thinking
        prefix = "user: Movie title\nassistant: <think>\n\n</think>\n\n"
        return (
            prefix
            if add_generation_prompt
            else prefix + messages[-1]["content"] + "<|im_end|>"
        )


def test_empty_thinking_delimiter_belongs_to_shared_prompt():
    row = {
        "prompt": [{"role": "user", "content": "Movie title"}],
        "chosen": [{"role": "assistant", "content": "Title A"}],
        "rejected": [{"role": "assistant", "content": "Title B"}],
    }
    sample = format_preference(row, Template())
    assert sample["prompt"].endswith("</think>\n\n")
    assert (
        sample["chosen"] == "Title A<|im_end|>"
        and sample["rejected"] == "Title B<|im_end|>"
    )
    with pytest.raises(ValueError, match="identical"):
        format_preference({**row, "rejected": row["chosen"]}, Template())

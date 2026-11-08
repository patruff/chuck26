"""Check the archived Colab's tool source, saved inputs and scorer outputs without renting a GPU."""
import argparse
import ast
import json
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
from smolagents import load_tool

from generate_march import HUMOR_BATCH_CELL, RUNNING_MAN_TOOLS, WordPhoneTool, load_word_bank, suggestions_for


def output_text(output):
    text = "".join(output.get("text", output.get("data", {}).get("text/plain", [])))
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def suggestion_payload(text):
    # Rich adds a table border to each line of the notebook's saved task.
    text = "\n".join(line.strip().strip("│").strip() for line in text.splitlines())
    start = text.index("{", text.index("Here are some word suggestions"))
    return json.JSONDecoder().raw_decode(text[start:])[0]


def compare(notebook, bank, pasted=None):
    cell = next(c for c in json.loads(notebook.read_text())["cells"] if c["metadata"].get("id") == HUMOR_BATCH_CELL)
    outputs = [output_text(item) for item in cell["outputs"]]
    saved = suggestion_payload(next(t for t in outputs if "Here are some word suggestions" in t))
    source_text = next(t for t in outputs if "TOOL CODE:" in t)
    implementations = {}
    for segment in source_text.split("TOOL CODE:")[1:]:
        body = segment.split("Drive already mounted")[0].strip()
        tree = ast.parse(body)
        name = next(n.name for n in tree.body if isinstance(n, ast.ClassDef))
        repo = {"ParodyWordSuggestionTool": "patruff/parody-suggestions", "WordPhoneTool": "patruff/word-phone"}[name]
        remote = Path(hf_hub_download(repo, "tool.py", repo_type="space", revision=RUNNING_MAN_TOOLS[repo]))
        implementations[repo] = ast.dump(ast.parse(remote.read_text())) == ast.dump(tree)
    words, phones = load_word_bank(bank)
    tools = {k:load_tool(k, revision=v, trust_remote_code=True) for k,v in RUNNING_MAN_TOOLS.items()}
    suggest = tools["patruff/parody-suggestions"]
    actual, _ = suggestions_for("The Running Man", suggest, json.dumps(words), phones, 0.6)
    actual = {w: actual[w.lower()] for w in saved}
    dual = WordPhoneTool(tools["patruff/word-phone"], phones, suggest)
    return {
        "cell": HUMOR_BATCH_CELL, "revisions": RUNNING_MAN_TOOLS, "same_tool_code": implementations,
        "complete_saved_suggestions_match": actual == saved,
        "pasted_suggestions_match": suggestion_payload(pasted.read_text()) == saved if pasted else None,
        "word_count":len(words), "custom_pronunciations":len(phones),
        "paired_checks": [dual.forward(o, r) for o,r in (("running","cumming"),("running","nutting"),("man","klan"))],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example-notebook", type=Path, required=True)
    p.add_argument("--word-bank", type=Path, required=True)
    p.add_argument("--pasted-example", type=Path)
    args = p.parse_args()
    report = compare(args.example_notebook, args.word_bank, args.pasted_example)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if set(report["same_tool_code"]) != set(RUNNING_MAN_TOOLS) or not all(report["same_tool_code"].values()):
        raise SystemExit("Archived tool implementations differ")
    if not report["complete_saved_suggestions_match"] or report["pasted_suggestions_match"] is False:
        raise SystemExit("Suggestion payload mismatch")


if __name__ == "__main__":
    main()

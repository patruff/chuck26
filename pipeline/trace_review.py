"""Read Qwen's saved steps and tool receipts as a chronological movie-by-movie transcript."""
import argparse
import json
from pathlib import Path
import re

from common import suggested_pairs, title_changes


def bank_use(row, title):
    changes = title_changes(row["input_title"], title or "")
    listed = suggested_pairs(row.get("suggestion_tool_results", {}))
    return [(a, b, (a, b) in listed) for a, b in changes]


def recorded_choices(row):
    # Older humor-first traces stored only the selected title and its pass flag.
    winner = row.get("qualifying_parody", row.get("parody") if row.get("passes_threshold") else None)
    fallback = row.get("fallback", row.get("parody") if row.get("passes_threshold") is False else None)
    return winner, fallback


def block(text, language="text"):
    ticks = "`" * max([3] + [len(v) + 1 for v in re.findall(r"`+", text)])
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return f"{ticks}{language}\n{text}\n{ticks}"


def steps_text(steps, calls):
    lines = []
    for step in steps:
        number = step["step"]
        lines.extend([f"### Step {number}", "Qwen thinking"])
        if step.get("reasoning"):
            lines.append(block("<think>\n" + step["reasoning"] + "\n</think>"))
        else:
            lines.append("No reasoning text was returned for this step.")
        if step.get("code"):
            lines.extend(["Python action", block(step["code"], "python")])
        if step.get("kind") == "verified-slate decision":
            lines.extend(["Qwen decision using the previously observed scores", block(step["model_output"])])
        for c in calls:
            if c.get("step") == number:
                lines.append(f"Executed `word_phone_tool(word={c['word']!r}, compare_to={c['compare_to']!r})`")
                lines.append(block(c["returned"]))
        if step.get("observations"):
            lines.extend(["Observation passed back to the agent", block(step["observations"])])
        if step.get("error"):
            lines.extend(["Recorded error", block(step["error"])])
    return lines


def transcript(row):
    title = row["input_title"]
    winner, fallback = recorded_choices(row)
    lines = [f"# {title}", f"Qualifying parody: {winner or '(none)'}"]
    if fallback:
        lines.append("Fallback: " + fallback)
    lines += ["## Initial suggestion lookup", "The controller ran the archived parody-suggestions tool before "
              "Qwen drafted. These are the candidates returned for each original word."]
    for word, result in row.get("suggestion_tool_results", {}).items():
        lines.append(f"`parody_word_suggester.forward(target={word!r}, "
                     f"min_similarity={str(row.get('suggestion_cutoff', 0.6))!r})` "
                     f"using the {row.get('word_count', '?')}-word bank and its custom pronunciations.")
        found = [f"{s['word']} ({s['similarity']})" for s in result.get("suggestions", [])]
        lines.append(", ".join(found) or "No candidates returned.")
        if result.get("error"):
            lines.append("Tool error: " + str(result["error"]))
    for previous in row.get("prior_runs", []):
        lines.extend([f"## Earlier attempt (seed {previous.get('seed')})",
                      *steps_text(previous.get("steps", []), previous.get("tool_calls", []))])
        if previous.get("error"):
            lines.append("Earlier attempt ended with: " + previous["error"])
    if row.get("prior_creative_steps"):
        lines.extend(["## Reused creative steps", *steps_text(row["prior_creative_steps"], [])])
    if row.get("recovery"):
        repair = row["recovery"]
        lines.append(f"The agent's original state was {repair['agent_state']}. Qwen then made a separate "
                     f"decision from its verified slates using seed {repair['decision_seed']}.")
    lines.extend([f"## Agent run (seed {row.get('seed', '?')})",
                  *steps_text(row.get("steps", []), row.get("tool_calls", []))])
    lines.append("## Nominations and choice")
    rounds = row.get("draft_rounds", [])
    for slate in rounds:
        lines.append(f"Round {slate['round']} finalists: " + " / ".join(slate["finalists"]))
        if slate.get("skip_reason"):
            lines.append("Reason given for skipping the suggestions: " + slate["skip_reason"])
        for a in slate["attempts"]:
            lines.append(f"- {a['title']} ({a['humor']}/10) — {a['why']}")
    if not rounds:
        for a in row.get("attempts", []):
            lines.append(f"- {a['title']} — {a['why']}")
    if row.get("parody"):
        lines.append("Model choice: " + row["parody"])
        lines.append("Changed words: " + ", ".join(f"{a} → {b} ({'listed' if hit else 'off-list'})"
                                                  for a, b, hit in bank_use(row, row["parody"])))
    if row.get("explanation"):
        lines.append(row["explanation"])
    if row.get("error"):
        lines.append("Generation error: " + row["error"])
    return "\n\n".join(lines) + "\n"


def write_transcripts(results, directory):
    directory.mkdir(parents=True, exist_ok=True)
    traces = directory / "transcripts"
    traces.mkdir(exist_ok=True)
    index = ["# Qwen thinking and tool calls", "",
             "Open a movie to read its initial word list, Qwen’s recorded thinking, executed Python, "
             "tool returns, nominations and choice in order.", "",
             "| Movie transcript | Model choice | Listed / changed words | Verified pair calls |",
             "| --- | --- | ---: | ---: |"]
    for i, line in enumerate(results.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        title = row["input_title"]
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        path = traces / f"{i:02d}-{slug}.md"
        path.write_text(transcript(row), encoding="utf-8")
        changes = bank_use(row, row.get("parody"))
        winner, fallback = recorded_choices(row)
        choice = winner or (fallback or "No result") + " (no qualifying winner)"
        index.append(f"| [{title}](transcripts/{path.name}) | {choice} | "
                     f"{sum(hit for _, _, hit in changes)} / {len(changes)} | {len(row.get('tool_calls', []))} |")
    (directory / "TRACE-INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    write_transcripts(args.results, args.directory)

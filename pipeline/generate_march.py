"""Generate March-style parody attempts, check the actual swaps, and have Qwen choose a winner."""

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import time

import requests

from common import MIN_PHONETIC_SCORE, align_swaps, clean_word, compact_suggestions
from generate_with_reasoning import REPO_ROOT, read_titles
from chuckles_prime.tools import load_parody_tools

SYSTEM = (
    "You write movie-title parodies for an audience that likes dark humor, vulgar wordplay, "
    "sexual innuendo and slapstick. Put the laugh first. Keep the sound and cadence of the title "
    "so the audience recognizes it aloud. Look for a second meaning that paints a funny scene. "
    "A taboo word alone is not a punchline. Rate how funny each joke is to this audience. "
    "Use your thinking channel to brainstorm. Keep the requested JSON in the final answer."
)


def load_word_bank(path):
    """Read literal data from the historical module without importing its code."""
    wanted = {"funny_words", "custom_phones"}
    data = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                data[target.id] = ast.literal_eval(node.value)
    if not isinstance(data.get("funny_words"), list) or not isinstance(data.get("custom_phones"), dict):
        raise ValueError("Expected funny_words and custom_phones in the historical word bank")
    return data["funny_words"], data["custom_phones"]


def suggestions_for(title, suggester, words, phones):
    raw = {}
    for token in title.split():
        word = clean_word(token)
        if word in raw or not word:
            continue
        raw[word] = json.loads(suggester.forward(
            target=word, word_list_str=words, min_similarity=str(MIN_PHONETIC_SCORE), custom_phones=phones
        ))
    return compact_suggestions(raw, top_n=12), {w: r["error"] for w, r in raw.items() if "error" in r}


def examples_for(title):
    with (REPO_ROOT / "parodies2026/known100.csv").open(encoding="utf-8") as f:
        reference = [r for r in csv.DictReader(f) if r["original"] in
                     {"The Matrix", "Pulp Fiction", "Kill Bill", "The Godfather"}]
    return "\n".join(f'{r["original"]} → {r["parody"]}' for r in reference
                     if r["original"].casefold() not in title.casefold())


def completion(args, task):
    started = time.monotonic()
    payload = {
        "model": args.base_model, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": task},
        ],
        "temperature": 0.75, "top_p": 0.9, "seed": 42, "max_tokens": 3584,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True, "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": "Bearer " + os.environ["LLAMA_API_KEY"]} if os.getenv("LLAMA_API_KEY") else {}
    trace, content = [], []
    usage, finish_reason = {}, None
    with requests.post(args.llama_server.rstrip("/") + "/v1/chat/completions",
                       headers=headers, json=payload, stream=True, timeout=(15, 150)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            event = line[6:]
            if event == b"[DONE]":
                break
            chunk = json.loads(event)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                trace.append(delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    return {"reasoning": "".join(trace), "content": "".join(content), "finish_reason": finish_reason,
            "usage": usage, "elapsed_seconds": round(time.monotonic() - started, 2), "prompt": task}


def parse_response(message):
    if message["finish_reason"] != "stop" or not message["reasoning"].strip():
        raise ValueError("Incomplete model answer or missing thinking channel")
    content = message["content"].strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(content)


def request_json(args, row, stage, task):
    row[stage] = completion(args, task)
    try:
        return parse_response(row[stage])
    except (ValueError, TypeError) as exc:
        row[stage + "_first_response"] = row[stage]
        repair = ("\nThe last response could not be read: " + str(exc)
                  + ". Return a single complete JSON object, without markdown or extra text.")
        row[stage] = completion(args, task + repair)
        return parse_response(row[stage])


def verify(title, candidate, analyzer, phones):
    checks = []
    aligned = len(title.split()) == len(candidate.split())
    if aligned:
        for original, replacement in align_swaps(title, candidate):
            result = json.loads(analyzer.forward(word=original, compare_to=replacement, custom_phones=phones))
            checks.append({"original": original, "replacement": replacement,
                           "score": (result.get("similarity") or {}).get("similarity"), "tool_result": result})
    passes = aligned and 1 <= len(checks) <= 2 and all(
        isinstance(c["score"], (float, int)) and c["score"] >= MIN_PHONETIC_SCORE for c in checks)
    return {"aligned": aligned, "swaps": checks, "passes": passes}


def generate(args, row, suggester, analyzer, words, phones):
    title = row["input_title"]
    suggestions, warnings = suggestions_for(title, suggester, words, phones)
    row.update(suggested_words=suggestions, suggestion_warnings=warnings)
    task = f"""Movie: {title}
Examples of the house style:
{examples_for(title)}

suggested_words from parody-suggestions (score scale 0 to 1):
{json.dumps(suggestions, ensure_ascii=False)}

Write three or four genuinely distinct variations. Swap one or two words in place and keep the
original word count. You can pitch a better word of your own, including when the list is empty.
Give each title a humor rating out of ten and one sentence describing the joke or the movie
it now promises. Try different comic premises rather than slight rewrites of one punchline.
I will run word-phone on every swap before you select the winner. Leave sound scores to the tool.
Return only JSON in this shape, with a number for humor:
{{"attempts":[{{"title":"...","humor":8,"why":"..."}}]}}"""
    attempts = request_json(args, row, "draft", task)["attempts"]
    if not isinstance(attempts, list) or not 3 <= len(attempts) <= 4:
        raise ValueError("Expected three or four attempts")
    seen = set()
    row["attempts"] = []
    for a in attempts:
        text = a["title"].strip()
        if not text or text.casefold() in seen or not a["why"].strip() or not 0 <= a["humor"] <= 10:
            raise ValueError("Expected distinct, rated attempts with a humor note")
        seen.add(text.casefold())
        row["attempts"].append({**a, **verify(title, text, analyzer, phones)})
    checked = [
        {"id": i, **{k: a[k] for k in ("title", "humor", "why", "aligned", "passes")},
         "swaps": [{k: s[k] for k in ("original", "replacement", "score")} for s in a["swaps"]]}
        for i, a in enumerate(row["attempts"], 1)
    ]
    print(f"{title}: checked {len(checked)} proposals, asking the model to choose", flush=True)
    pick = request_json(args, row, "selection", f"""Movie: {title}
Below are your attempts with actual word-phone results. Null means unscorable. A pass requires
one or two replacements and at least {MIN_PHONETIC_SCORE} on each. Choose the funniest passing
attempt as winner. If none passes, choose the funniest attempt and explain the tradeoff.
Pick a different runner-up. Use the supplied IDs and explain why the winner lands better.
Verified attempts:
{json.dumps(checked, ensure_ascii=False)}
Return only JSON: {{"winner":1,"runner_up":2,"why":"one sentence"}}.""")
    winner, second = pick["winner"], pick["runner_up"]
    if (type(winner) is not int or type(second) is not int or winner == second or
            not (1 <= winner <= len(attempts) and 1 <= second <= len(attempts)) or not pick["why"].strip()):
        raise ValueError("Invalid selection IDs or missing explanation")
    row.update(selection_result=pick, parody=row["attempts"][winner-1]["title"],
               runner_up=row["attempts"][second-1]["title"], passes_threshold=row["attempts"][winner-1]["passes"])
    if not row["passes_threshold"] and any(a["passes"] for a in row["attempts"]):
        row["selection_warning"] = "The model selected a nonpassing candidate despite a passing alternative"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("llama-server", "base-model", "titles", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--word-bank", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args()
    titles = read_titles(args.titles)[:args.limit]
    if not titles:
        raise SystemExit("No titles to generate")
    words, phones = load_word_bank(args.word_bank)
    print(f"March profile: {len(words)} suggested words, {len(phones)} custom pronunciations", flush=True)
    words_json = json.dumps(words)
    suggester, analyzer = load_parody_tools()
    failed = False
    with Path(args.output).open("w", encoding="utf-8") as out:
        for title in titles:
            row = {"input_title": title, "profile": "march", "model": args.base_model}
            try:
                generate(args, row, suggester, analyzer, words_json, phones)
            except (ValueError, TypeError, KeyError, requests.RequestException, RuntimeError) as exc:
                row["error"] = str(exc)
                failed = True
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{title} → {row.get('parody', row.get('error'))}", flush=True)
    if failed:
        raise SystemExit("One or more March samples failed; inspect the per-title records")


if __name__ == "__main__":
    main()

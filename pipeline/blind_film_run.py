"""Paired, matched 27B inference for the blind film review collection."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import requests
from smolagents import load_tool

from common import clean_word, compact_suggestions
from generate_film_pairs import atomic_gzip
from generate_march import (
    RUNNING_MAN_TOOLS,
    WordPhoneTool,
    load_word_bank,
    suggestions_for,
)
from march_source import BANK_SHA256, snapshot_bank
from run_gguf_test import MODEL, REVISION

ROOT = Path(__file__).resolve().parents[1]
RANKS = [2, 5, 18, 23, 35, 36, 44, 51, 55, 61, 72, 76, 77, 80, 98]
ADAPTER_SHA = "ec6f81c028db8230f57b5617abd37d0da75a169579392a611a6cc022edf96806"
BASE_SHA = "9ed40ccc8b8432f38b9a85d0ca67928167f5719e8c10bd299e56d34facaf6e61"
SERVER = "http://127.0.0.1:8080"
SYSTEM = json.loads(
    (ROOT / "results/film-adult-premises/prompt-revision.json").read_text()
)["applied_instructions"] + (
    "\nUse intelligible wordplay. Aim the sting at consenting adults or powerful people behaving badly. "
    "Keep children, sexual coercion and slurs directed at protected groups out of the jokes and drafts."
)


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def model_call(messages, seed, scale, schema=None):
    settings = dict(
        temperature=0,
        seed=seed,
        max_tokens=240,
        chat_template_kwargs={"enable_thinking": True},
    )
    body = dict(
        messages=messages,
        model="film-review",
        lora=[{"id": 0, "scale": scale}],
        **settings,
    )
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema},
        }
    result = requests.post(SERVER + "/v1/chat/completions", json=body, timeout=6000)
    result.raise_for_status()
    completion = result.json()["choices"][0]
    reply = completion["message"]
    text = reply.get("content") or ""
    thinking = reply.get("reasoning_content") or reply.get("reasoning") or ""
    if "<think>" in text and "</think>" in text:
        before, text = text.split("</think>", 1)
        thinking = thinking or before.partition("<think>")[2]
    if completion["finish_reason"] != "stop" or not text.strip():
        raise ValueError("Model reply was incomplete")
    return dict(
        input_messages=messages,
        reasoning=thinking,
        model_output=text,
        settings=settings,
        finish_reason=completion["finish_reason"],
        response_schema=schema,
    )


def run_film(film, scale, prepared, suggester, analyzer, phones, directory):
    title, rank = film["title"], int(film["rank"])
    seed = 97031 + rank
    suggested, raw = prepared
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Movie: {title}\nSuggested replacements (March scores): {json.dumps(suggested, ensure_ascii=False)}\n"
            "Pitch two funny, different parody titles. Match the original word count and align each swap "
            "to its source word. Try at least one supplied change. In idea, briefly explain the competing "
            "jokes and why one might work better. Give the titles alone in first and second.",
        },
    ]
    phone = WordPhoneTool(analyzer, phones, suggester)
    steps, attempts, checks = [], [], {}
    draft_schema = object_schema(
        {k: {"type": "string"} for k in ("first", "second", "idea")}
    )
    originals = title.split()
    # Each fallback option contains a genuine non-identity suggestion at one word position.
    single_swaps = []
    for i, original in enumerate(originals):
        for v in suggested[clean_word(original)]:
            if clean_word(original) != clean_word(v["word"]):
                single_swaps.append(
                    " ".join(originals[:i] + [v["word"].title()] + originals[i + 1 :])
                )
    if not single_swaps:
        raise ValueError("Blind slate requires at least one suggested change")
    for round_number in (1, 2):
        if round_number == 2:
            messages = messages + [
                {
                    "role": "user",
                    "content": "The first slate had no verified qualifying title. "
                    "Pick at least one of these actual one-swap titles. Keep the other pitch original if you prefer.\n"
                    + json.dumps(single_swaps, ensure_ascii=False),
                }
            ]
            draft_schema = object_schema(
                {
                    "first": {"type": "string", "enum": single_swaps},
                    "second": {"type": "string"},
                    "idea": {"type": "string"},
                }
            )
        step = model_call(messages, seed, scale, draft_schema)
        draft = json.loads(step["model_output"])
        step.update(kind="Draft titles", step=len(steps) + 1)
        steps.append(step)
        if not step["reasoning"] and not any(
            v.get("kind") == "Creative deliberation" for v in steps
        ):
            reflection = model_call(
                messages
                + [
                    {"role": "assistant", "content": step["model_output"]},
                    {
                        "role": "user",
                        "content": "Briefly weigh these two ideas before phonetic verification. Consider their double meanings and any weak spot. Use one short sentence.",
                    },
                ],
                seed,
                scale,
            )
            reflection.update(kind="Creative deliberation", step=len(steps) + 1)
            steps.append(reflection)
        for key in ("first", "second"):
            candidate = draft[key].strip()
            if (
                candidate in checks
                or len(candidate.split()) != len(originals)
                or not any(
                    clean_word(a) != clean_word(b)
                    for a, b in zip(originals, candidate.split())
                )
            ):
                continue
            receipt = [
                phone.forward(a, b) for a, b in zip(originals, candidate.split())
            ]
            checks[candidate] = receipt
            attempts.append({"title": candidate, "why": draft["idea"]})
        valid = [
            k
            for k, v in checks.items()
            if all(p.get("score") is not None and p["score"] >= 0.6 for p in v)
        ]
        if valid:
            break
    if not valid:
        raise ValueError("No checked, above-floor candidate")
    listed = {(w, v["word"]) for w, group in suggested.items() for v in group}
    preferred = [
        k
        for k in valid
        if any(
            clean_word(a) != clean_word(b) and (clean_word(a), clean_word(b)) in listed
            for a, b in zip(originals, k.split())
        )
    ] or valid
    menu = {
        k: [
            {
                n: p.get(n)
                for n in (
                    "word",
                    "compare_to",
                    "score",
                    "suggestion_score",
                    "word_phone_score",
                )
            }
            for p in vals
        ]
        for k, vals in checks.items()
    }
    select_messages = [
        messages[0],
        {
            "role": "user",
            "content": f"Movie: {title}\nSuggested replacements: {json.dumps(suggested, ensure_ascii=False)}\n"
            f"Your pitches: {json.dumps(attempts, ensure_ascii=False)}\nActual phonetic tool returns: "
            + json.dumps(menu, ensure_ascii=False)
            + "\nChoose the funniest eligible title from "
            + json.dumps(preferred)
            + ". Give one concise sentence explaining the title’s joke and your choice.",
        },
    ]
    final = model_call(
        select_messages,
        seed,
        scale,
        object_schema(
            {"winner": {"type": "string", "enum": preferred}, "why": {"type": "string"}}
        ),
    )
    final.update(kind="Choose after phonetic checks", step=len(steps) + 1)
    steps.append(final)
    decision = json.loads(final["model_output"])
    winner = decision["winner"]
    assert winner in preferred
    row = dict(
        input_title=title,
        model=MODEL,
        seed=seed,
        word_count=1565,
        suggestion_cutoff=0.6,
        suggested_words=suggested,
        suggestion_tool_results=raw,
        steps=steps,
        tool_calls=phone.calls,
        finalist_checks=checks,
        attempts=attempts,
        parody=winner,
        explanation=decision["why"],
        qualifying_parody=winner,
        passes_threshold=True,
        state="success",
        tool_revisions=RUNNING_MAN_TOOLS,
        applied_instructions_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
    )
    result = dict(
        film=film,
        row=row,
        variant="adapter" if scale else "original",
        adapter_scale=scale,
        adapter_sha256=ADAPTER_SHA,
        base_sha256=BASE_SHA,
        model_revision=REVISION,
        bank_sha256=BANK_SHA256,
    )
    atomic_gzip(directory / f"{rank:03}-{scale}.json.gz", result)
    print(f"Completed a matched trajectory for film rank {rank}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=range(5), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    out = args.work / "samples"
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    assert (
        report["adapter_sha256"] == ADAPTER_SHA
        and report["base_weight_sha256"] == BASE_SHA
    )
    excluded = {p["film"]["title"].casefold() for p in report["preferences"]} | {
        p["film"].casefold() for p in report["generations"]
    }
    films = [
        f
        for f in csv.DictReader(
            (ROOT / "pipeline/film_lists/top200_1990_2015.csv").open()
        )
        if int(f["rank"]) in RANKS[args.shard :: 5]
    ]
    assert len(films) == 3 and not any(f["title"].casefold() in excluded for f in films)
    bank = snapshot_bank(None, args.work)
    words, phones = load_word_bank(bank)
    assert len(words) == len(phones) == 1565
    sg, az = [
        load_tool(repo, revision=rev, trust_remote_code=True)
        for repo, rev in RUNNING_MAN_TOOLS.items()
    ]
    prepared = {}
    for f in films:
        raw, errors = suggestions_for(f["title"], sg, json.dumps(words), phones, 0.6)
        if errors:
            raise ValueError(f"Suggestion lookup failed for rank {f['rank']}")
        prepared[f["rank"]] = compact_suggestions(raw, top_n=len(words)), raw
    for _ in range(180):
        try:
            if requests.get(SERVER + "/health", timeout=3).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Native server did not become ready")
    available = requests.get(SERVER + "/lora-adapters", timeout=12).json()
    assert len(available) == 1 and available[0]["id"] == 0
    for scale in (0, 1):
        # Set the global configuration too; every inference request also specifies its own scale.
        requests.post(
            SERVER + "/lora-adapters", json=[{"id": 0, "scale": scale}], timeout=30
        ).raise_for_status()
        got = requests.get(SERVER + "/lora-adapters", timeout=30).json()
        assert got[0]["scale"] == scale
        with ThreadPoolExecutor(max_workers=len(films)) as pool:
            tasks = [
                pool.submit(
                    run_film, f, scale, prepared[f["rank"]], sg, az, phones, out
                )
                for f in films
            ]
            for task in tasks:
                task.result()
    assert len(list(out.glob("*.json.gz"))) == 2 * len(films)


if __name__ == "__main__":
    main()

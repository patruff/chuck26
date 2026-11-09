"""Paired, matched 27B inference for the blind film review collection."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import copy
import gzip
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


def request_step(messages, seed, scale, schema, limit):
    settings = dict(
        temperature=0,
        seed=seed,
        max_tokens=limit,
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
    # The server can take a long time to decode on CPU. Apply a timeout to
    # inactivity and preserve the tokens it returned if the connection closes.
    text_parts, thinking_parts = [], []
    finish = "stream_closed"
    body["stream"] = True
    try:
        with requests.post(
            SERVER + "/v1/chat/completions", json=body, stream=True, timeout=(20, 420)
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                packet = json.loads(payload)
                if packet.get("error"):
                    raise RuntimeError("Native server returned a streaming error")
                for choice in packet.get("choices", []):
                    delta = choice.get("delta", {})
                    text_parts.append(delta.get("content") or "")
                    thinking_parts.append(
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    finish = choice.get("finish_reason") or finish
    except (
        requests.Timeout,
        requests.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ):
        finish = "transport_interrupted"
    text, thinking = "".join(text_parts), "".join(thinking_parts)
    if "<think>" in text and "</think>" in text:
        before, text = text.split("</think>", 1)
        thinking = thinking or before.partition("<think>")[2]
    return dict(
        input_messages=messages,
        reasoning=thinking,
        model_output=text,
        settings=settings,
        finish_reason=finish,
        transport="event-stream",
        response_schema=schema,
    )


def model_call(messages, seed, scale, schema=None):
    # Repeat the same request only when its actual response could not be used.
    earlier = []
    for limit in (240, 512, 768):
        applied = copy.deepcopy(schema)
        if applied and earlier:
            for name in ("idea", "why"):
                if name in applied["properties"]:
                    applied["properties"][name]["maxLength"] = 180
            if any(
                r["finish_reason"] in ("transport_interrupted", "stream_closed")
                for r in earlier
            ):
                # The exact permitted titles remain in the judge prompt and are
                # validated below, even when we retry with a simpler grammar.
                for prop in applied["properties"].values():
                    prop.pop("enum", None)
        step = request_step(messages, seed, scale, applied, limit)
        text = step["model_output"].strip()
        usable = bool(text) and step["finish_reason"] == "stop"
        if schema and text:
            try:
                parsed = json.loads(text)
                usable = isinstance(parsed, dict) and all(
                    isinstance(parsed.get(k), str)
                    and parsed[k].strip()
                    and (
                        "enum" not in schema["properties"][k]
                        or parsed[k] in schema["properties"][k]["enum"]
                    )
                    for k in schema["required"]
                )
            except (ValueError, TypeError):
                usable = False
        if usable:
            if earlier:
                step["earlier_replies"] = earlier
            return step
        earlier.append(step)
        print(
            f"Incomplete model reply at {limit} tokens; finish reason {step['finish_reason']}.",
            flush=True,
        )
    raise ValueError("Model did not return a complete reply after three attempts")


def run_film(film, scale, prepared, suggester, analyzer, phones, directory):
    title, rank = film["title"], int(film["rank"])
    seed = 97031 + rank
    checkpoint = directory / f"{rank:03}-{scale}.json.gz"
    if checkpoint.exists():
        with gzip.open(checkpoint, "rt") as handle:
            previous = json.load(handle)
        if (
            previous["film"] != film
            or previous["adapter_scale"] != scale
            or previous["adapter_sha256"] != ADAPTER_SHA
            or previous["base_sha256"] != BASE_SHA
            or previous["row"]["seed"] != seed
            or previous["row"]["suggested_words"] != prepared[0]
            or previous["row"]["applied_instructions_sha256"]
            != hashlib.sha256(SYSTEM.encode()).hexdigest()
            or previous["row"]["state"] != "success"
        ):
            raise ValueError(
                "Restored blind trajectory does not match this configuration"
            )
        return
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
    parser.add_argument(
        "--rank",
        type=int,
        choices=range(1, 201),
        help="Run one MovieLens film; additional held-out films use shard 0",
    )
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
    ranks = RANKS[args.shard :: 5]
    if args.rank is not None:
        if args.rank in RANKS and args.rank not in ranks:
            parser.error("The requested study rank is not in this shard")
        if args.rank not in RANKS and args.shard != 0:
            parser.error("Run an additional held-out film on shard 0")
        ranks = [args.rank]
    films = [
        f
        for f in csv.DictReader(
            (ROOT / "pipeline/film_lists/top200_1990_2015.csv").open()
        )
        if int(f["rank"]) in ranks
    ]
    assert len(films) == len(ranks) and not any(
        f["title"].casefold() in excluded for f in films
    )
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
    failures = []
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
            for film, task in zip(films, tasks):
                try:
                    task.result()
                except Exception as exc:
                    failures.append(
                        (film["rank"], scale, type(exc).__name__, str(exc)[:150])
                    )
    if failures:
        raise RuntimeError(f"Unfinished film trajectories: {failures}")

    assert len(list(out.glob("*.json.gz"))) == 2 * len(films)


if __name__ == "__main__":
    main()

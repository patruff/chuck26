"""Finish an interrupted sample using the same Qwen model and its already verified slates."""

import json

from common import MIN_PHONETIC_SCORE, clean_word, suggested_pairs, title_changes
from generate_march import parse_answer, store_checked_answer


def checked_options(row):
    title = row["input_title"]
    observed = {
        (c["word"], c["compare_to"]): c
        for c in row.get("tool_calls", [])
        if c.get("visible_in_observation")
    }
    checks = {}
    slates = row.get("draft_rounds", [])
    for slate in slates:
        for candidate in slate["finalists"]:
            keys = [
                (clean_word(a), clean_word(b))
                for a, b in zip(title.split(), candidate.split())
            ]
            if len(keys) == len(title.split()) == len(candidate.split()) and all(
                k in observed for k in keys
            ):
                checks[candidate] = [observed[k] for k in keys]
    passed = {
        t
        for t, cs in checks.items()
        if title_changes(title, t)
        and all(c["score"] is not None and c["score"] >= MIN_PHONETIC_SCORE for c in cs)
    }
    listed = suggested_pairs(row.get("suggestion_tool_results", {}))
    eligible = {
        t for t in passed if listed.intersection(title_changes(title, t))
    } or passed
    complete = [s for s in slates if all(t in checks for t in s["finalists"])]
    if not eligible and len({frozenset(s["finalists"]) for s in complete}) < 2:
        return [], observed
    options = []
    for slate in complete:
        for winner in slate["finalists"]:
            if eligible and winner not in eligible:
                continue
            runner = next(t for t in slate["finalists"] if t != winner)
            options.append(
                dict(
                    slate=slate,
                    winner=winner,
                    runner_up=runner,
                    passes=winner in passed,
                    checks={
                        t: [
                            {
                                k: c.get(k)
                                for k in (
                                    "word",
                                    "compare_to",
                                    "score",
                                    "suggestion_score",
                                    "word_phone_score",
                                )
                            }
                            for c in checks[t]
                        ]
                        for t in slate["finalists"]
                    },
                )
            )
    return options, observed


def finish_sample(row, model, revision, seed):
    """Make a new recorded decision call. Never assign a winner to an incomplete slate in Python."""
    options, observed = checked_options(row)
    if not options or not any(s.get("reasoning") for s in row.get("steps", [])):
        return None
    menu = [
        dict(
            choice=i,
            winner=o["winner"],
            runner_up=o["runner_up"],
            qualifies=o["passes"],
            pitches=[
                a
                for a in o["slate"]["attempts"]
                if a["title"] in o["slate"]["finalists"]
            ],
            checks=o["checks"],
        )
        for i, o in enumerate(options)
    ]
    task = (
        f"Finish your checked parody run for {row['input_title']}. These are your previously nominated "
        "pairs and actual observed verifier returns. Select the funniest allowed whole title. If all choices "
        "are fallbacks, explicitly say no choice clears 0.60. Do not create new titles, scores or tool calls. "
        'Return JSON {"choice": integer option index, "explanation": "why this choice wins"}.\n'
        + json.dumps(menu, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": revision["applied_instructions"]},
        {"role": "user", "content": task},
    ]
    api_messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in messages
    ]
    reply = model.generate(
        api_messages,
        seed=seed,
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    try:
        raw = reply.raw.choices[0]
        decision = json.loads(reply.content)
        idx = decision["choice"]
        if (
            raw.finish_reason != "stop"
            or type(idx) is not int
            or not 0 <= idx < len(options)
        ):
            return None
        o = options[idx]
        answer = parse_answer(
            dict(
                winner=o["winner"],
                runner_up=o["runner_up"],
                explanation=decision["explanation"],
                attempts=o["slate"]["attempts"],
                selected_draft_round=o["slate"]["round"],
            )
        )
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        return None
    result = dict(row)
    error = result.pop("error", None)
    result.update(
        state="success",
        recovery={
            "method": "Qwen decision from observed slates",
            "continued_seed": row["seed"],
            "decision_seed": seed,
            "agent_state": row.get("state"),
            "agent_error": error,
        },
    )
    number = max((s["step"] for s in row["steps"]), default=0) + 1
    result["steps"] = [
        *row["steps"],
        dict(
            step=number,
            kind="verified-slate decision",
            input_messages=messages,
            reasoning=getattr(raw.message, "reasoning_content", "") or "",
            model_output=reply.content,
            code=None,
            observations="",
            finish_reason=raw.finish_reason,
            error=None,
            final=True,
        ),
    ]
    store_checked_answer(result, row["input_title"], answer, observed, True, True, True)
    return result

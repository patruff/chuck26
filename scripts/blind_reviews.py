"""Read role-neutral comparisons and render their actual inference and verification traces."""

import gzip
import hashlib
import json
from pathlib import Path


COLLECTION = "blind-27b-15"
LABELS = {"original": "Original 27B", "adapter": "27B with DPO adapter"}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def blind_transcript(row):
    lines = [
        "Suggested-word tool · patruff/parody-suggestions",
        f"2025 bank: {row['word_count']} words and pronunciations · inclusive floor 0.60",
    ]
    for word, raw in row["suggestion_tool_results"].items():
        values = [
            f"{s['word']} ({s['similarity']})" for s in raw.get("suggestions", [])
        ]
        lines.append(f"{word} → " + (", ".join(values) or "no suggestions"))
    lines.append("")
    checks = row["finalist_checks"]
    seen = set()
    pending = []

    def append_receipts():
        for name in pending:
            if name in seen or name not in checks:
                continue
            seen.add(name)
            lines.append(f"Phonetic checks for {name}")
            for c in checks[name]:
                lines.append(
                    f"word_phone_tool(word={c['word']!r}, compare_to={c['compare_to']!r})"
                )
                lines.append(json.dumps(c, ensure_ascii=False))
        pending.clear()

    for step in row["steps"]:
        if step["kind"] != "Creative deliberation":
            append_receipts()
        lines.append(f"Step {step['step']} · {step['kind']}")
        for message in step["input_messages"]:
            lines.append(
                message["role"].capitalize() + " prompt\n" + message["content"]
            )
        if step.get("reasoning"):
            lines.append("Model thinking\n<think>\n" + step["reasoning"] + "\n</think>")
        lines.append("Model response\n" + step["model_output"])
        if step["kind"] == "Draft titles":
            draft = json.loads(step["model_output"])
            pending.extend([draft["first"].strip(), draft["second"].strip()])
        lines.append("")
    append_receipts()
    if seen != set(checks):
        raise ValueError(
            "The recorded checks cannot be attributed to their draft round"
        )
    return "\n".join(lines)


def load_blind_pairs(directory, prompt_system, location):
    for path in sorted(directory.glob("*.blind.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            entry = json.load(source)
        samples = entry["options"]
        if len(samples) != 2 or {x["variant"] for x in samples} != set(LABELS):
            raise ValueError(f"Missing original or adapted run in {path.name}")
        left, right = samples
        film = left["film"]
        if film != right["film"] or any(
            left[k] != right[k]
            for k in ("bank_sha256", "base_sha256", "adapter_sha256", "model_revision")
        ):
            raise ValueError(f"Mismatched pair in {path.name}")
        rows = [sample["row"] for sample in samples]
        for key in (
            "seed",
            "suggestion_tool_results",
            "suggested_words",
            "suggestion_cutoff",
            "tool_revisions",
            "applied_instructions_sha256",
        ):
            if rows[0][key] != rows[1][key]:
                raise ValueError(f"Mismatched {key} in {path.name}")
        film = {
            **film,
            **{k: int(film[k]) for k in ("rank", "year", "movielens_id", "ratings")},
        }
        sides = []
        for index, (sample, row) in enumerate(zip(samples, rows)):
            title = row["parody"]
            if (
                row["input_title"] != film["title"]
                or row["state"] != "success"
                or not row["tool_calls"]
            ):
                raise ValueError(f"Incomplete inference in {path.name}")
            if sample["adapter_scale"] != int(sample["variant"] == "adapter"):
                raise ValueError(f"Incorrect adapter scale in {path.name}")
            receipt = row["finalist_checks"][title]
            passed = bool(receipt) and all(
                p["score"] is not None and p["score"] >= 0.6 for p in receipt
            )
            sides.append(
                dict(
                    title=title,
                    why=row["explanation"],
                    passed=passed,
                    pairs=receipt,
                    listed={
                        k: {v["word"] for v in vs}
                        for k, vs in row["suggested_words"].items()
                    },
                    model_choice=True,
                    model=LABELS[sample["variant"]],
                    seed=row["seed"],
                    applied_instructions_sha256=row["applied_instructions_sha256"],
                    guidance_sha256=None,
                    decision_seed=None,
                    trace=blind_transcript(row),
                    sha256=fingerprint(row),
                    thinking_entries=[
                        {
                            "stage": step["kind"],
                            "text": step["reasoning"],
                            "native": True,
                        }
                        if step.get("reasoning")
                        else {
                            "stage": step["kind"],
                            "text": step["model_output"],
                            "native": False,
                        }
                        for step in row["steps"]
                        if step.get("reasoning")
                        or step["kind"] == "Creative deliberation"
                    ],
                    source=location(path) + "#" + ("AB"[index]),
                )
            )
        pair_id = fingerprint(
            [
                "blind-review-v1",
                film["movielens_id"],
                *[s["sha256"] for s in sides],
                entry["assignment_id"],
            ]
        )
        yield dict(
            id=pair_id,
            film=film,
            sides=sides,
            collection=COLLECTION,
            shared_trace=False,
            blind=True,
            prompt=[
                {"role": "system", "content": prompt_system},
                {
                    "role": "user",
                    "content": "Movie: "
                    + film["title"]
                    + "\nSuggested replacements (March scores): "
                    + json.dumps(rows[0]["suggested_words"], ensure_ascii=False),
                },
            ],
        )

"""Publish a review index only when all 200 ranked films have two usable traced options."""

import argparse
import csv
import json
from pathlib import Path
from review_films import ROOT, load_source, normalized


def escaped(text):
    return (
        text.replace("|", r"\|")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("\n", " ")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch", type=Path)
    p.add_argument(
        "--films", type=Path, default=ROOT / "pipeline/film_lists/top200_1990_2015.csv"
    )
    args = p.parse_args()
    with args.films.open(encoding="utf-8", newline="") as source:
        films = list(csv.DictReader(source))
    pairs = list(load_source(args.batch / "samples"))
    if len(pairs) != len(films) or [p["film"]["rank"] for p in pairs] != [
        int(f["rank"]) for f in films
    ]:
        missing = set(int(f["rank"]) for f in films) - {
            p["film"]["rank"] for p in pairs
        }
        raise SystemExit(
            f"Incomplete batch; missing review pairs at ranks {sorted(missing)}"
        )
    lines = [
        "# Movie parody pairs",
        "",
        "Open the local review app for pitches, scored word pairs, full Qwen thinking, "
        "executed Python, and verifier returns. A and B correspond to their stable positions in the app.",
        "",
        "| Rank | Film | Year | Option A | Option B |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    rows = []
    for pair, film in zip(pairs, films):
        a, b = pair["sides"]
        if (
            pair["film"]["title"] != film["title"]
            or normalized(a["title"]) == normalized(b["title"])
            or pair["shared_trace"]
        ):
            raise ValueError(f"Invalid paired samples for {film['title']}")
        for side in (a, b):
            if (
                not side["pairs"]
                or "<think>" not in side["trace"]
                or "word_phone_tool" not in side["trace"]
            ):
                raise ValueError(f"Missing trace or verification for {film['title']}")
        rows.append(
            {
                "pair_id": pair["id"],
                "film": pair["film"],
                "options": [
                    {
                        k: s[k]
                        for k in (
                            "title",
                            "why",
                            "passed",
                            "pairs",
                            "model_choice",
                            "model",
                            "seed",
                            "source",
                            "sha256",
                        )
                    }
                    for s in (a, b)
                ],
            }
        )

        source_a = "samples/" + Path(a["source"]).name
        source_b = "samples/" + Path(b["source"]).name
        lines.append(
            f"| {film['rank']} | {escaped(film['title'])} | {film['year']} | "
            f"[{escaped(a['title'])}]({source_a}) | [{escaped(b['title'])}]({source_b}) |"
        )
    (args.batch / "CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.batch / "pairs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    print(
        f"{len(rows)} distinct review pairs; both options clear the floor in "
        f"{sum(all(s['passed'] for s in r['options']) for r in rows)} films; "
        f"at least one clears it in {sum(any(s['passed'] for s in r['options']) for r in rows)}."
    )


if __name__ == "__main__":
    main()

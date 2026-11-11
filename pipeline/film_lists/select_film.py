"""Resolve MovieLens ranks or one-column CSV input into ordered inference batches."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re

CATALOG = Path(__file__).with_name("top200_1990_2015.csv")
HEADERS = {
    "title",
    "movie",
    "input",
    "parody",
    "input parody",
    "input parodies",
    "input_title",
}


def parse_ranks(value):
    ranks = {}
    for item in str(value).strip().split(","):
        match = re.fullmatch(r"\s*([0-9]{1,3})(?:\s*[-–]\s*([0-9]{1,3}))?\s*", item)
        if not match:
            raise ValueError("Enter movie ranks like 35, 1-100, or 1-5,12,35-40.")
        first = int(match[1])
        last = int(match[2] or match[1])
        if not 1 <= first <= last <= 200:
            raise ValueError(
                f"Invalid rank selection {item.strip()!r}. Use ascending ranks from 1 to 200."
            )
        ranks.update(dict.fromkeys(range(first, last + 1)))
    return list(ranks)


def catalog_films(ranks):
    with CATALOG.open(newline="", encoding="utf-8") as source:
        catalog = {int(r["rank"]): r for r in csv.DictReader(source)}
    return [catalog[i] for i in ranks]


def film(rank):
    return catalog_films([rank])[0]


def csv_titles(payload):
    rows = [
        r
        for r in csv.reader(
            io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True
        )
        if r and any(s.strip() for s in r)
    ]
    if any(len(r) != 1 for r in rows):
        raise ValueError("Use one CSV column and quote titles containing commas.")
    titles = [r[0].strip() for r in rows]
    if titles and titles[0].casefold() in HEADERS:
        titles.pop(0)
    if not titles or any(not t or "\n" in t or "\r" in t for t in titles):
        raise ValueError("Expected one nonempty title per row.")
    return titles


def title_rows(titles):
    return [{"rank": str(i), "title": title} for i, title in enumerate(titles, 1)]


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_batches(directory, rows, batch_size):
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / "selected.csv", rows)
    parts = []
    for start in range(0, len(rows), batch_size):
        name = f"part-{len(parts):04d}"
        write_rows(directory / (name + ".csv"), rows[start : start + batch_size])
        parts.append(name)
    return parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ranks", nargs="?", default="35", help="35, 1-100, or 1-5,12,35-40")
    alt = p.add_mutually_exclusive_group()
    alt.add_argument(
        "--csv",
        type=Path,
        help="One input per row, with an optional title or input header",
    )
    alt.add_argument("--movie", help="One literal title")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=25)
    args = p.parse_args()
    try:
        if args.csv:
            rows = title_rows(csv_titles(args.csv.read_bytes()))
        elif args.movie is not None:
            if not args.movie.strip() or len(args.movie.splitlines()) != 1:
                raise ValueError("Enter one nonempty title or use a CSV.")
            rows = title_rows([args.movie.strip()])
        else:
            rows = catalog_films(parse_ranks(args.ranks))
        if args.batch_size < 1:
            raise ValueError("The batch size must be positive.")
    except (OSError, csv.Error, ValueError) as exc:
        p.error(str(exc))
    if not args.output_dir:
        for row in rows:
            print(row["title"])
        return
    parts = write_batches(args.output_dir, rows, args.batch_size)
    publish_batches(parts, len(rows))


def publish_batches(parts, count):
    message = f"Selected {count} titles in {len(parts)} batch(es). See selected.csv in the inference-input artifact."
    print(message)
    if env := os.getenv("GITHUB_OUTPUT"):
        with open(env, "a") as out:
            print("parts=" + json.dumps(parts), file=out)
            print("part_count=" + str(len(parts)), file=out)
            print("count=" + str(count), file=out)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            print(message, file=f)


if __name__ == "__main__":
    main()

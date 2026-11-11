"""Combine completed inference artifacts in the order of the submitted inputs."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def merge(selection, artifacts, output):
    paths = sorted(selection.glob("part-*.csv"))
    if not paths:
        raise ValueError("The selection contains no batches")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "results.jsonl"
        with stage.open("w", encoding="utf-8") as out:
            for part in paths:
                with part.open(newline="", encoding="utf-8") as f:
                    titles = [r["title"] for r in csv.DictReader(f)]
                source = artifacts / ("review-" + part.stem) / "results.jsonl"
                with source.open(encoding="utf-8") as f:
                    for title, line in zip(titles, f, strict=True):
                        row = json.loads(line)
                        if (
                            row.get("input_title") != title
                            or row.get("state") != "success"
                        ):
                            raise ValueError(
                                f"{part.stem} has an incomplete or mismatched result for {title!r}"
                            )
                        out.write(line.rstrip("\n") + "\n")
        stage.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "artifacts", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    merge(args.selection, args.artifacts, args.output)

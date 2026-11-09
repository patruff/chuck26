"""Resolve one validated MovieLens rank to the exact film title."""

import argparse
import csv
from pathlib import Path


def film(rank):
    with Path(__file__).with_name("top200_1990_2015.csv").open(encoding="utf-8") as f:
        return next(row for row in csv.DictReader(f) if int(row["rank"]) == rank)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rank", type=int, choices=range(1, 201))
    print(film(p.parse_args().rank)["title"])

"""Rebuild the 1990–2015 film list from the fixed GroupLens MovieLens 32M release."""
import argparse
import csv
import hashlib
import io
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
MD5 = "d472be332d4daa821edc399621853b57"


def rank(archive, by):
    counts, halves = Counter(), Counter()
    with archive.open("ml-32m/ratings.csv") as f:
        next(f)
        for line in f:
            _, movie, score, _ = line.split(b",", 3)
            counts[movie] += 1
            halves[movie] += int(float(score)*2)
    films = []
    with archive.open("ml-32m/movies.csv") as f:
        for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
            key = row["movieId"].encode()
            match = re.fullmatch(r"(.+) \((\d{4})\)\s*", row["title"])
            if not match or not 1990 <= int(match[2]) <= 2015 or not counts[key]:
                continue
            name = re.sub(r"\s+\([^()]+\)", "", match[1]).strip()
            # MovieLens files sort English articles after the title.
            name = re.sub(r"^(.+), (The|An|A)$", r"\2 \1", name)
            name = " ".join("and" if token=="&" else token for token in name.split() if token != "-")
            item = dict(title=name,year=int(match[2]),movielens_id=int(key),ratings=counts[key],
                        mean_rating=halves[key]/(2*counts[key]),source_title=row["title"])
            if by == "votes" or counts[key] >= 10000:
                films.append(item)
    films.sort(key=lambda x: (-x["ratings"], x["movielens_id"]) if by == "votes" else
               (-x["mean_rating"], -x["ratings"], x["movielens_id"]))
    return films[:200]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--by", choices=["votes", "score"], default="votes")
    args = p.parse_args()
    with urllib.request.urlopen(URL, timeout=180) as response:
        compressed = response.read()
    if hashlib.md5(compressed).hexdigest()!=MD5:
        raise SystemExit("MovieLens archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
        films = rank(archive, args.by)
    with args.output.open("w", newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["rank",*films[0]], lineterminator="\n")
        writer.writeheader()
        for i, film in enumerate(films,1):
            writer.writerow(dict(rank=i,**{**film,"mean_rating":f"{film['mean_rating']:.4f}"}))

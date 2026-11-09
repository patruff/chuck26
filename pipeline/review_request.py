"""Validate human DPO exports and prepare private training requests."""

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def review_rows(path):
    return parse_reviews(path.read_bytes())


def parse_reviews(payload):
    rows = []
    seen = set()
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            pair_id = r["pair_id"]
            if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
                raise ValueError("missing or repeated pair_id")
            if [m["role"] for m in r["prompt"]] != ["system", "user"] or any(
                not isinstance(m["content"], str) or not m["content"].strip()
                for m in r["prompt"]
            ):
                raise ValueError("expected system and user messages")
            for k in ("chosen", "rejected"):
                if (
                    len(r[k]) != 1
                    or r[k][0]["role"] != "assistant"
                    or not isinstance(r[k][0]["content"], str)
                    or not r[k][0]["content"].strip()
                ):
                    raise ValueError("expected one assistant title per side")
            if (
                r["chosen"][0]["content"].strip().casefold()
                == r["rejected"][0]["content"].strip().casefold()
            ):
                raise ValueError("identical preferences")
            if (
                not isinstance(r["film"]["title"], str)
                or not r["film"]["title"].strip()
            ):
                raise ValueError("missing film title")
            seen.add(pair_id)
            rows.append(
                {k: r[k] for k in ("film", "pair_id", "prompt", "chosen", "rejected")}
            )
        except (KeyError, TypeError, IndexError, ValueError) as e:
            raise ValueError(f"Invalid preference on line {number}: {e}") from e
    if not rows:
        raise ValueError("No human A/B preferences in the export")
    return rows


def make_request(path, public_key):
    payload = path.read_bytes()
    rows = parse_reviews(payload)
    first = rows[0]
    probes = [
        {"film": first["film"]["title"], "prompt": first["prompt"], "reviewed": True}
    ]
    seen = {r["film"]["title"].casefold() for r in rows}
    for sample in sorted((ROOT / "results/movielens-200/samples").glob("*-a.json.gz")):
        with gzip.open(sample, "rt", encoding="utf-8") as f:
            data = json.load(f)
        title = data["film"]["title"]
        if title.casefold() in seen:
            continue
        prompt = [
            first["prompt"][0],
            {
                "role": "user",
                "content": "Movie: "
                + title
                + "\nSuggested replacements (March scores): "
                + json.dumps(data["row"]["suggested_words"], ensure_ascii=False),
            },
        ]
        probes.append({"film": title, "prompt": prompt, "reviewed": False})
        if len(probes) == 3:
            break
    return {
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": rows,
        "probes": probes,
        "public_key": public_key,
    }


def encode_request(request):
    payload = gzip.compress(json.dumps(request, ensure_ascii=False).encode(), mtime=0)
    encoded = "gzip:" + base64.b64encode(payload).decode()
    if len(encoded) > 48 * 1024:
        raise ValueError(
            "Request exceeds one GitHub secret. Use --tracked with a deliberately committed reviews/*.jsonl file."
        )
    return encoded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["unpack"])
    p.add_argument("output", type=Path)
    p.add_argument("--dataset", type=Path)
    args = p.parse_args()
    blob = os.environ.pop("PRIVATE_REQUEST", "")
    if not blob:
        p.error(
            "Missing private request. Use review_cycle.py train --prepare-only and supply its key ID."
        )
    data = (
        json.loads(gzip.decompress(base64.b64decode(blob[5:], validate=True)))
        if blob.startswith("gzip:")
        else json.loads(blob)
    )
    if args.dataset:
        data = make_request(args.dataset, data["public_key"])
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

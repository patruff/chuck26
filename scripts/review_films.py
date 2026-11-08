"""Review two Qwen parodies for one film and export human-labeled conversational DPO rows."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from trace_review import transcript

DEFAULT_SOURCES = [
    ROOT / "results" / "movielens-200" / "samples",
    ROOT / "results" / "feedback-0c3cc72" / "results.jsonl.gz",
]
PROMPT = (
    "Write one funny, recognizable parody of the movie title for an adult audience. Prefer a returned "
    "word when it makes the joke work. Return just the parody title."
)


def sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def normalized(title):
    return " ".join(re.findall(r"\w+", title.casefold()))


def location(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def choices(row):
    if row.get("error") or row.get("state") != "success":
        return []
    items = []
    for title in (row.get("parody"), row.get("runner_up")):
        if not title or title not in row.get("finalist_checks", {}):
            continue
        pairs = row["finalist_checks"][title]
        changed = [p for p in pairs if p["word"] != p["compare_to"]]
        passed = bool(changed) and all(
            p["score"] is not None and p["score"] >= 0.60 for p in pairs
        )
        pitch = next((a for a in row.get("attempts", []) if a["title"] == title), {})
        items.append(
            dict(
                title=title,
                why=pitch.get("why", ""),
                pairs=pairs,
                passed=passed,
                model_choice=title == row.get("parody"),
            )
        )
    return items


def review_pair(film, first, second, collection):
    # Each source is (record, path). Archived single-run files share both finalists' trace.
    film = dict(film)
    for key in ("rank", "year", "movielens_id", "ratings"):
        if key in film:
            film[key] = int(film[key])
    if "mean_rating" in film:
        film["mean_rating"] = float(film["mean_rating"])
    ra, pa = first
    rb, pb = second
    variants = [
        (a, b)
        for a in choices(ra)
        for b in choices(rb)
        if normalized(a["title"]) != normalized(b["title"])
    ]
    if not variants:
        return None
    # Preserve independent winners when distinct. If they coincide, use a recorded, checked alternative.
    a, b = max(
        variants,
        key=lambda v: (
            sum(c["model_choice"] for c in v),
            sum(c["passed"] for c in v),
            v[0]["model_choice"],
        ),
    )
    prompt = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": "Movie: "
            + film["title"]
            + "\nSuggested replacements (March scores): "
            + json.dumps(ra.get("suggested_words", {}), ensure_ascii=False),
        },
    ]
    if ra.get("suggested_words") != rb.get("suggested_words"):
        raise ValueError(
            "The two samples use different suggestion payloads: " + film["title"]
        )
    digest_a, digest_b = sha(ra), sha(rb)
    sides = []
    for c, row, file, digest in ((a, ra, pa, digest_a), (b, rb, pb, digest_b)):
        sides.append(
            {
                **c,
                "listed": {
                    word: {item["word"] for item in candidates}
                    for word, candidates in row.get("suggested_words", {}).items()
                },
                "trace": transcript(row),
                "source": location(file),
                "sha256": digest,
                "seed": row.get("seed"),
                "model": row.get("model"),
                "guidance_sha256": row.get("guidance_sha256"),
            }
        )
    pair_id = sha(
        [
            film["title"],
            film.get("movielens_id"),
            digest_a,
            digest_b,
            a["title"],
            b["title"],
        ]
    )
    if int(pair_id[:2], 16) % 2:
        sides.reverse()
    return dict(
        id=pair_id,
        film=film,
        sides=sides,
        prompt=prompt,
        collection=collection,
        shared_trace=digest_a == digest_b,
    )


def load_source(path):
    path = path.resolve()
    if path.is_dir() and (path / "samples").is_dir():
        path = path / "samples"
    if path.is_dir() and not list(path.glob("*-a.json.gz")):
        newest = next(
            (
                path / name
                for name in ("results.jsonl.gz", "results.jsonl")
                if (path / name).is_file()
            ),
            None,
        )
        if newest:
            yield from load_source(newest)
            return
    collection = (
        path.parent.name if path.name == "samples" or path.is_file() else path.name
    )
    if path.is_dir():
        for a in sorted(path.glob("*-a.json.gz")):
            b = a.with_name(a.name.replace("-a.json.gz", "-b.json.gz"))
            if not b.exists():
                continue
            with gzip.open(a, "rt", encoding="utf-8") as f:
                left = json.load(f)
            with gzip.open(b, "rt", encoding="utf-8") as f:
                right = json.load(f)
            if (
                left["film"] != right["film"]
                or left["signature"] != right["signature"]
                or (left["side"], right["side"]) != ("a", "b")
            ):
                raise ValueError(f"Checkpoint mismatch for {a.name}")
            pair = review_pair(
                left["film"], (left["row"], a), (right["row"], b), collection
            )
            if pair:
                samples = {location(a): left, location(b): right}
                for side in pair["sides"]:
                    sample = samples[side["source"]]
                    continued = sample["row"].get("recovery", {}).get("continued_seed")
                    earlier = [
                        r
                        for r in sample.get("failed_attempts", [])
                        if r.get("seed") != continued
                    ]
                    if earlier:
                        side["trace"] = transcript(
                            {**sample["row"], "prior_runs": earlier}
                        )
                yield pair
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                row = json.loads(line)
                pair = review_pair(
                    dict(rank=i, title=row["input_title"]),
                    (row, path),
                    (row, path),
                    collection,
                )
                if pair:
                    yield pair


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as f:
        tmp = Path(f.name)
        try:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


class ConflictingReview(ValueError):
    pass


class ReviewStore:
    def __init__(self, directory):
        self.directory = directory
        self.labels = directory / "film_preferences.jsonl"
        self.export = directory / "reviewed_dpo.jsonl"
        self.pending = directory / ".pending-film-preferences.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        with self.lock():
            if self.pending.exists():
                self.finish_update()
            else:
                if (
                    (not self.labels.exists() or not self.labels.stat().st_size)
                    and self.export.exists()
                    and self.export.stat().st_size
                ):
                    # A clone may contain just the trainer-ready export.
                    with self.export.open(encoding="utf-8") as f:
                        restored = [
                            self.restore_dpo(json.loads(line))
                            for line in f
                            if line.strip()
                        ]
                    write_lines(self.labels, restored)
                if not self.labels.exists():
                    write_lines(self.labels, [])
                self.project(self.read())

    @staticmethod
    def restore_dpo(r):
        choice = r["chosen_option"]
        chosen, rejected = r["chosen_trace"], r["rejected_trace"]
        if (
            choice not in ("a", "b")
            or chosen["title"] != r["chosen"][0]["content"]
            or rejected["title"] != r["rejected"][0]["content"]
        ):
            raise ValueError("The DPO row does not match its recorded choices.")
        return dict(
            pair_id=r["pair_id"],
            film=r["film"],
            prompt=r["prompt"],
            feedback=r["feedback"],
            updated=r["reviewed_at"],
            choice=choice,
            sides=[chosen, rejected] if choice == "a" else [rejected, chosen],
        )

    @contextmanager
    def lock(self):
        with (self.directory / ".review.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def read(self):
        with self.labels.open(encoding="utf-8") as f:
            return {
                r["pair_id"]: r
                for r in map(json.loads, (line for line in f if line.strip()))
            }

    def project(self, records):
        out = []
        for r in records.values():
            if r["choice"] == "neither":
                continue
            i = 0 if r["choice"] == "a" else 1
            chosen, rejected = r["sides"][i], r["sides"][1 - i]
            out.append(
                {
                    "prompt": r["prompt"],
                    "chosen": [{"role": "assistant", "content": chosen["title"]}],
                    "rejected": [{"role": "assistant", "content": rejected["title"]}],
                    "feedback": r["feedback"],
                    "pair_id": r["pair_id"],
                    "film": r["film"],
                    "reviewed_at": r["updated"],
                    "chosen_option": r["choice"],
                    "chosen_trace": chosen,
                    "rejected_trace": rejected,
                }
            )
        write_lines(self.export, out)

    def finish_update(self):
        with self.pending.open(encoding="utf-8") as f:
            records = {r["pair_id"]: r for r in map(json.loads, f)}
        write_lines(self.labels, records.values())
        self.project(records)
        self.pending.unlink()

    def update(self, updates):
        with self.lock():
            if self.pending.exists():
                self.finish_update()
            records = self.read()
            for pair, choice, feedback, expected in updates:
                current = records.get(pair["id"], {})
                if current.get("updated", "") != expected:
                    raise ConflictingReview(
                        "This choice has changed since you opened the page. Reload to see it."
                    )
                if choice == "clear":
                    records.pop(pair["id"], None)
                    continue
                if choice not in ("a", "b", "neither"):
                    raise ValueError("Select a preference.")
                sides = [
                    {
                        k: s[k]
                        for k in (
                            "title",
                            "source",
                            "sha256",
                            "model",
                            "seed",
                            "passed",
                            "guidance_sha256",
                        )
                    }
                    for s in pair["sides"]
                ]
                records[pair["id"]] = {
                    "pair_id": pair["id"],
                    "film": pair["film"],
                    "prompt": pair["prompt"],
                    "sides": sides,
                    "choice": choice,
                    "feedback": feedback,
                    "updated": datetime.now(timezone.utc).isoformat(),
                }
            write_lines(self.pending, records.values())
            self.finish_update()


def create_app(items, review_dir, refresh=None):
    app = Flask(__name__)
    by_id = {p["id"]: p for p in items}
    collections = {}
    for p in by_id.values():
        collections.setdefault(p["collection"], []).append(p)
    store = ReviewStore(review_dir)

    @app.before_request
    def refresh_sources():
        nonlocal by_id, collections
        if request.endpoint == "index" and refresh:
            latest = refresh()
            if latest is not None:
                new = {p["id"]: p for p in latest}
                groups = {}
                for p in new.values():
                    groups.setdefault(p["collection"], []).append(p)
                by_id, collections = new, groups

    @app.get("/")
    def index():
        records = store.read()
        collection = request.args.get("set") or next(iter(collections), "")
        group = collections.get(collection, [])
        if not group:
            abort(404, "No reviewable pairs in this collection.")
        wanted = request.args.get("pair")
        if wanted:
            pos = next((i for i, p in enumerate(group) if p["id"] == wanted), None)
            if pos is None:
                abort(404)
        else:
            pos = next((i for i, p in enumerate(group) if p["id"] not in records), 0)
        p = group[pos]
        return render_template(
            "film_review.html",
            pair=p,
            pos=pos,
            group=group,
            collection=collection,
            collections=collections,
            decision=records.get(p["id"], {}),
            reviewed=sum(q["id"] in records for q in group),
            notice=request.args.get("notice", ""),
        )

    @app.post("/vote/<pair_id>")
    def vote(pair_id):
        p = by_id.get(pair_id)
        if not p:
            abort(404)
        try:
            store.update(
                [
                    (
                        p,
                        request.form.get("choice"),
                        request.form.get("feedback", ""),
                        request.form.get("expected", ""),
                    )
                ]
            )
        except ConflictingReview as e:
            abort(409, str(e))
        except ValueError as e:
            abort(400, str(e))
        records = store.read()
        group = collections[p["collection"]]
        pos = group.index(p)
        following = next(
            (q for q in group[pos + 1 :] + group[:pos] if q["id"] not in records), p
        )
        return redirect(
            url_for("index", set=p["collection"], pair=following["id"], notice="Saved"),
            303,
        )

    @app.get("/download/<kind>")
    def download(kind):
        if kind not in ("preferences", "dpo"):
            abort(404)
        with store.lock():
            if kind == "dpo":
                store.project(store.read())
            return send_file(
                store.export if kind == "dpo" else store.labels,
                as_attachment=True,
                mimetype="application/x-ndjson",
            )

    @app.post("/import")
    def import_reviews():
        upload = request.files.get("preferences")
        if upload is None:
            abort(400, "Select a film_preferences.jsonl backup.")
        try:
            records = store.read()
            changes = []
            seen = set()
            for line in upload:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "chosen" in rec:
                    rec = store.restore_dpo(rec)
                key = rec["pair_id"]
                p = by_id[key]
                if key in seen:
                    raise ValueError("Duplicate pair in upload")
                seen.add(key)
                if [(s["sha256"], s["title"]) for s in rec["sides"]] != [
                    (s["sha256"], s["title"]) for s in p["sides"]
                ]:
                    raise ValueError("Trace mismatch")
                if rec["choice"] not in ("a", "b", "neither") or not isinstance(
                    rec["feedback"], str
                ):
                    raise ValueError("Invalid vote")
                prior = records.get(key)
                if prior:
                    if (prior["choice"], prior["feedback"]) != (
                        rec["choice"],
                        rec["feedback"],
                    ):
                        raise ConflictingReview("Existing vote conflicts with upload")
                else:
                    changes.append((p, rec["choice"], rec["feedback"], ""))
            store.update(changes)
        except ConflictingReview as e:
            abort(409, str(e))
        except (ValueError, KeyError, TypeError) as e:
            abort(400, f"Invalid or unavailable pair: {e}")
        return redirect(
            url_for("index", notice=f"Imported {len(changes)} reviews"), 303
        )

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        action="append",
        help="A batch directory, results.jsonl or results.jsonl.gz (repeatable)",
    )
    p.add_argument("--reviews", type=Path, default=ROOT / "reviews")
    p.add_argument("--port", type=int, default=5118)
    args = p.parse_args()
    sources = args.input if args.input else DEFAULT_SOURCES
    if args.input and any(not path.exists() for path in sources):
        p.error("An input path does not exist.")
    stamp = None

    def refresh():
        nonlocal stamp
        current = tuple(
            path.stat().st_mtime_ns if path.exists() else None for path in sources
        )
        if current == stamp:
            return None
        items = [
            item for path in sources if path.exists() for item in load_source(path)
        ]
        stamp = current
        return items

    items = refresh()
    if not items:
        p.error("No completed pairs found in the input.")
    print(f"Loaded {len(items)} pairs. Open http://127.0.0.1:{args.port}", flush=True)
    create_app(items, args.reviews, refresh=refresh).run(
        host="127.0.0.1", port=args.port, debug=False
    )


if __name__ == "__main__":
    main()

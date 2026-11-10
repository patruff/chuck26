"""Publish the static review catalog and train from revision-pinned Hub reviews."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.review_request import make_request
from scripts.review_films import DEFAULT_SOURCES, ReviewStore, load_source, write_lines

DATASET = os.getenv("CHUCKLES_REVIEW_DATASET") or "patruff/chuck26-reviews"
MODELS = os.getenv("CHUCKLES_ADAPTER_REPO") or "patruff/chuck26-adapters"
REPO = os.getenv("GITHUB_REPOSITORY") or "patruff/chuck26"
FILMS = ROOT / "pipeline/film_lists/top200_1990_2015.csv"
LEDGER = "reviews/film_preferences.jsonl"
SIDE_FIELDS = (
    "title",
    "source",
    "sha256",
    "model",
    "seed",
    "passed",
    "guidance_sha256",
    "applied_instructions_sha256",
    "decision_seed",
)
api = HfApi()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def bundle(rows):
    return gzip.compress(b"".join(encode(row) + b"\n" for row in rows), mtime=0)


def read_bundle(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hub_file(path, revision, repo=DATASET, kind="dataset"):
    return Path(hf_hub_download(repo, path, repo_type=kind, revision=revision))


def operation(path, content):
    return CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)


def bootstrap():
    for repo, kind in [(DATASET, "dataset"), (MODELS, "model")]:
        api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
    if not api.file_exists(DATASET, LEDGER, repo_type="dataset"):
        head = api.dataset_info(DATASET).sha
        api.create_commit(
            repo_id=DATASET,
            repo_type="dataset",
            parent_commit=head,
            operations=[operation(LEDGER, b"")],
            commit_message="Initialize review ledger",
        )


def materialize(pairs, collection):
    public, training, traces = [], [], []
    for pair in pairs:
        sides = pair["sides"]
        public.append(
            dict(
                id=pair["id"],
                film=pair["film"],
                blind=bool(pair.get("blind")),
                sides=[
                    {k: s.get(k) for k in ("title", "why", "pairs", "passed")}
                    for s in sides
                ],
            )
        )
        training.append(
            dict(
                id=pair["id"],
                film=pair["film"],
                prompt=pair["prompt"],
                collection=collection,
                blind=bool(pair.get("blind")),
                sides=[{k: s.get(k) for k in SIDE_FIELDS} for s in sides],
            )
        )
        traces.append(
            dict(
                id=pair["id"],
                sides=[s.get("trace", "") for s in sides],
                attribution=[s.get("model") for s in sides]
                if pair.get("blind")
                else [],
            )
        )
    return public, training, traces


def store_collection(collection, label, pairs, run=0, raw=()):
    prefix = f"collections/{collection}/"
    rows, training, traces = materialize(pairs, collection)
    if not rows:
        print(f"No complete pairs in {label}")
        return
    meta = dict(id=collection, label=label, run=int(run), count=len(rows))
    files = {
        "meta.json": encode(meta),
        "display.jsonl.gz": bundle(rows),
        "training.jsonl.gz": bundle(training),
        "traces.jsonl.gz": bundle(traces),
    }
    # Each collection is immutable. A retried importer can safely reuse its first publication.
    if api.file_exists(DATASET, prefix + "meta.json", repo_type="dataset"):
        return
    ops = [operation(prefix + k, v) for k, v in files.items()]
    ops += [operation(prefix + "source/" + name, str(path)) for name, path in raw]
    api.create_commit(
        repo_id=DATASET,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Archive {label} ({len(rows)} comparisons)",
    )


def archive_run(run, name, directory):
    if not re.fullmatch(r"review-ready|film-shard-\d+", name):
        raise ValueError("Unsupported review artifact")
    collection = f"inference-{int(run)}" + (
        "" if name == "review-ready" else "-" + name
    )
    if api.file_exists(
        DATASET, f"collections/{collection}/meta.json", repo_type="dataset"
    ):
        return
    folder = Path(directory)
    source = folder / "results.jsonl" if name == "review-ready" else folder / "samples"
    if not source.exists():
        return
    raw = (
        [(source.name, source)]
        if source.is_file()
        else [(p.name, p) for p in sorted(source.glob("*.json.gz"))]
    )
    pairs = list(load_source(source))
    for pair in pairs:
        for side in pair["sides"]:
            side["source"] = (
                f"hf://datasets/{DATASET}/collections/{collection}/source/"
                + Path(side["source"]).name
            )
    store_collection(
        collection,
        f"Inference #{run}" + ("" if name == "review-ready" else " · " + name),
        pairs,
        run,
        raw,
    )


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", "api", *args]))


def sync(run=None):
    bootstrap()
    for source in DEFAULT_SOURCES:
        if source.exists():
            name = (
                source.parent.name
                if source.is_file() or source.name == "samples"
                else source.name
            )
            if not api.file_exists(
                DATASET, f"collections/{name}/meta.json", repo_type="dataset"
            ):
                store_collection(
                    name,
                    {
                        "movielens-200": "MovieLens catalog",
                        "feedback-0c3cc72": "Earlier feedback",
                    }.get(name, name),
                    load_source(source),
                )
    # Seed the hosted page from recent completed generation runs, including runs made before Pages.
    if run:
        runs = [int(run)]
    else:
        runs = []
        for workflow in (
            "human-guidance.yml",
            "film-review-batch.yml",
            "studio-inference.yml",
        ):
            info = gh_json(
                f"repos/{REPO}/actions/workflows/{workflow}/runs?branch=main&status=success&per_page=12"
            )
            runs.extend(r["id"] for r in info["workflow_runs"])
    known = set(api.list_repo_files(DATASET, repo_type="dataset"))
    for rid in sorted(set(runs)):
        record = gh_json(f"repos/{REPO}/actions/runs/{rid}")
        if (
            record["head_branch"] != "main"
            or record["conclusion"] != "success"
            or record["head_repository"]["full_name"].lower() != REPO.lower()
            or Path(record["path"]).name
            not in {
                "human-guidance.yml",
                "film-review-batch.yml",
                "studio-inference.yml",
            }
        ):
            continue
        for artifact in gh_json(
            f"repos/{REPO}/actions/runs/{rid}/artifacts?per_page=100"
        )["artifacts"]:
            name = artifact["name"]
            if artifact["expired"] or not re.fullmatch(
                r"review-ready|film-shard-\d+", name
            ):
                continue
            key = f"inference-{rid}" + ("" if name == "review-ready" else "-" + name)
            if f"collections/{key}/meta.json" in known:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(
                    [
                        "gh",
                        "run",
                        "download",
                        str(rid),
                        "-R",
                        REPO,
                        "-n",
                        name,
                        "-D",
                        tmp,
                    ],
                    check=True,
                )
                archive_run(rid, name, tmp)


def build(output):
    output = Path(output)
    shutil.copytree(ROOT / "studio", output, dirs_exist_ok=True)
    data = output / "data"
    data.mkdir(exist_ok=True)
    revision = api.dataset_info(DATASET).sha
    collections = []
    for path in api.list_repo_files(DATASET, repo_type="dataset", revision=revision):
        if not re.fullmatch(r"collections/[\w-]+/meta.json", path):
            continue
        meta = json.loads(hub_file(path, revision).read_bytes())
        pairs = read_bundle(
            hub_file(path.replace("meta.json", "display.jsonl.gz"), revision)
        )
        for pair in pairs:
            for side in pair["sides"]:
                side["pairs"] = [
                    {
                        k: item[k]
                        for k in (
                            "compare_to",
                            "word",
                            "score",
                            "suggestion_score",
                            "word_phone_score",
                        )
                        if k in item
                    }
                    for item in side["pairs"]
                ]
        (data / (meta["id"] + ".json")).write_bytes(encode(pairs))
        collections.append(meta)
    collections.sort(
        key=lambda c: (c["run"], c["id"] == "movielens-200", c["id"]), reverse=True
    )
    if not collections:
        raise ValueError("No reviewable collections")
    with FILMS.open() as f:
        movies = [{"rank": r["rank"], "title": r["title"]} for r in csv.DictReader(f)]
    (data / "index.json").write_bytes(
        encode(
            dict(
                collections=collections,
                movies=movies,
                dataset=DATASET,
                models=MODELS,
                adapterAvailable=api.file_exists(
                    MODELS, "latest.json", repo_type="model"
                ),
                repository=REPO,
                revision=revision,
            )
        )
    )
    print(
        "Published",
        sum(c["count"] for c in collections),
        "pairs in",
        len(collections),
        "collections",
    )


def training_request(revision, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    revision = api.dataset_info(DATASET, revision=revision or "main").sha
    votes = [
        json.loads(s)
        for s in hub_file(LEDGER, revision).read_text().splitlines()
        if s.strip()
    ]
    library = {}
    for collection in {v["collection"] for v in votes}:
        if not re.fullmatch(r"[\w-]+", collection):
            raise ValueError("Invalid collection")
        for pair in read_bundle(
            hub_file(f"collections/{collection}/training.jsonl.gz", revision)
        ):
            library[collection, pair["id"]] = pair
    records = {}
    for vote in votes:
        key = vote["pair_id"]
        if key in records:
            raise ValueError("Repeated review ID")
        pair = library[vote["collection"], key]
        choice = vote["choice"]
        choices = (
            ("a", "b", "tie", "neither") if pair["blind"] else ("a", "b", "neither")
        )
        if (
            choice not in choices
            or not isinstance(vote["feedback"], str)
            or not vote["updated"]
        ):
            raise ValueError("Invalid saved review")
        record = dict(
            pair_id=key,
            film=pair["film"],
            prompt=pair["prompt"],
            sides=pair["sides"],
            choice=choice,
            updated=vote["updated"],
            feedback=vote["feedback"],
        )
        if pair["blind"]:
            first = vote.get("first_choice", choice)
            if first not in choices:
                raise ValueError("Invalid original blind vote")
            record.update(blind=True, first_choice=first)
        records[key] = record
    store = ReviewStore(work / "snapshot")
    write_lines(store.labels, records.values())
    store.project(records)
    # The trainer consumes the shared prompt and distinct human choices. No recipient key is needed.
    req = make_request(store.export, "")
    (work / "request.json").write_bytes(encode(req))
    (work / "dataset-revision.txt").write_text(revision)
    return revision


def checksum(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def publish_adapter(result, run, revision):
    result = Path(result)
    weights = result / "chuckles-27b-dpo.gguf"
    report = json.loads((result / "comparison.json").read_bytes())
    with weights.open("rb") as f:
        magic = f.read(4)
    if magic != b"GGUF" or checksum(weights) != report["adapter_sha256"]:
        raise ValueError("Native adapter checksum mismatch")
    prefix = f"runs/{int(run)}"
    meta = dict(
        run=int(run),
        path=prefix + "/chuckles-27b-dpo.gguf",
        sha256=report["adapter_sha256"],
        model=report["model"],
        base_revision=report["revision"],
        base_sha256=report["base_weight_sha256"],
        dataset_sha256=report["dataset_sha256"],
        dataset_revision=revision or None,
    )
    ops = [
        operation(meta["path"], str(weights)),
        operation(prefix + "/manifest.json", encode(meta)),
    ]
    if revision:
        export = result.parent / "snapshot/reviewed_dpo.jsonl"
        if checksum(export) != report["dataset_sha256"]:
            raise ValueError("Training report does not match saved reviews")
        api.create_commit(
            repo_id=DATASET,
            repo_type="dataset",
            operations=[
                operation(f"training/{int(run)}/reviewed_dpo.jsonl", str(export)),
                operation(
                    f"training/{int(run)}/comparison.json",
                    str(result / "comparison.json"),
                ),
            ],
            commit_message=f"Save training evidence for run {run}",
        )
    # Compare-and-swap keeps an older, slower training job from replacing a newer pointer.
    for attempt in range(4):
        head = api.model_info(MODELS).sha
        try:
            latest = json.loads(
                hub_file("latest.json", head, MODELS, "model").read_bytes()
            )["run"]
        except EntryNotFoundError:
            latest = 0
        changes = list(ops)
        if int(run) >= latest:
            changes.append(operation("latest.json", encode(meta)))
        try:
            api.create_commit(
                repo_id=MODELS,
                repo_type="model",
                parent_commit=head,
                operations=changes,
                commit_message=f"Publish verified adapter from Actions run {run}",
            )
            break
        except Exception:
            if attempt == 3 or api.model_info(MODELS).sha == head:
                raise
    print(f"Published {MODELS}/{meta['path']}")


def fetch_adapter(work, revision):
    work = Path(work)
    head = api.model_info(MODELS, revision=revision or "main").sha
    try:
        info = json.loads(hub_file("latest.json", head, MODELS, "model").read_bytes())
    except EntryNotFoundError as err:
        raise ValueError(
            "Train an adapter from saved reviews before selecting it for inference"
        ) from err
    base = json.loads((ROOT / "config/model.json").read_bytes())
    if (info["model"], info["base_revision"], info["base_sha256"]) != (
        base["repository"],
        base["revision"],
        base["sha256"],
    ):
        raise ValueError("Adapter belongs to another base model")
    path = hub_file(info["path"], head, MODELS, "model")
    if checksum(path) != info["sha256"]:
        raise ValueError("Adapter checksum mismatch")
    shutil.copyfile(path, work / "adapter.gguf")
    (work / "adapter-manifest.json").write_bytes(encode(info))


def csv_titles(payload):
    rows = list(
        csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    )
    rows = [r for r in rows if r and any(s.strip() for s in r)]
    if any(len(r) != 1 for r in rows):
        raise ValueError("Use a single column; quote titles containing commas")
    titles = [r[0].strip() for r in rows]
    if titles and titles[0].casefold() in {
        "title",
        "movie",
        "input",
        "parody",
        "input parody",
        "input parodies",
        "input_title",
    }:
        titles.pop(0)
    if not titles or any(not t or "\n" in t or "\r" in t for t in titles):
        raise ValueError("Expected one nonempty title per row")
    return titles


def prepare_csv(revision, output):
    data = hub_file("inputs/latest.csv", revision or "main").read_bytes()
    titles = csv_titles(data)
    with Path(output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title"])
        writer.writerows([s] for s in titles)


def inference_settings(work, source, rank, model, revision):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    revision = api.dataset_info(DATASET, revision=revision or "main").sha
    if source == "page":
        try:
            request = json.loads(
                hub_file("requests/inference.json", revision).read_bytes()
            )
        except EntryNotFoundError:
            request = {"source": "catalog", "rank": 35, "model": "base"}
        source, rank, model = request["source"], request["rank"], request["model"]
    if source not in {"csv", "catalog"} or model not in {"base", "latest-adapter"}:
        raise ValueError("Unsupported inference settings")
    if not 1 <= int(rank) <= 200:
        raise ValueError("Choose a catalog rank between 1 and 200")
    (work / "rank.txt").write_text(str(int(rank)))
    if source == "csv":
        prepare_csv(revision, work / "input.csv")
    if model == "latest-adapter":
        fetch_adapter(work, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="op", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--run", type=int)
    s = sub.add_parser("build")
    s.add_argument("output")
    s = sub.add_parser("train-request")
    s.add_argument("work")
    s.add_argument("--revision", default="")
    s = sub.add_parser("publish-adapter")
    s.add_argument("result")
    s.add_argument("run")
    s.add_argument("--revision", default="")
    s = sub.add_parser("inference-settings")
    s.add_argument("work")
    for field in ("source", "rank", "model", "revision"):
        s.add_argument("--" + field, required=True)
    args = p.parse_args()
    if args.op == "sync":
        sync(args.run)
    elif args.op == "build":
        build(args.output)
    elif args.op == "train-request":
        training_request(args.revision, args.work)
    elif args.op == "publish-adapter":
        publish_adapter(args.result, args.run, args.revision)
    elif args.op == "inference-settings":
        inference_settings(args.work, args.source, args.rank, args.model, args.revision)

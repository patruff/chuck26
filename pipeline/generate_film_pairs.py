"""Generate two independent, checkpointed Colab-style CodeAgent trajectories per film."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

from generate_march import generate_title, load_word_bank, make_model, RUNNING_MAN_TOOLS
from march_source import BANK_SHA256, NOTEBOOK
from smolagents import load_tool
from film_selection import checked_options, finish_sample


def atomic_gzip(path, data):
    payload = json.dumps(data, ensure_ascii=False)
    # HTTP exceptions can echo a temporary authenticated service endpoint.
    payload = re.sub(
        r"(?:https?://)?[\w.-]+\.proxy\.runpod\.net", "[RunPod endpoint]", payload
    )
    key = os.getenv("LLAMA_API_KEY", "")
    if len(key) >= 16:
        payload = payload.replace(key, "[redacted]")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            tmp.write(
                gzip.compress(
                    payload.encode(),
                    compresslevel=5,
                    mtime=0,
                )
            )
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def init_worker(model_id, server, bank, revision, notebook):
    global worker
    words, phones = load_word_bank(bank)
    suggest, analyze = [
        load_tool(repo, revision=sha, trust_remote_code=True)
        for repo, sha in RUNNING_MAN_TOOLS.items()
    ]
    model = make_model(model_id, server)
    model.client.timeout = 900
    worker = dict(
        model=model,
        words=words,
        phones=phones,
        cutoff=0.60,
        revisions=RUNNING_MAN_TOOLS,
        suggester=suggest,
        analyzer=analyze,
        notebook=notebook,
        humor_first=True,
        previous={},
        revision=revision,
    )


def generate_side(job):
    film, side, directory, signature = job
    rank = int(film["rank"])
    path = directory / f"{rank:03}-{side}.json.gz"
    history = []
    if path.exists():
        stored = read_gzip(path)
        if stored["signature"] != signature or stored["film"] != film:
            raise ValueError(f"Checkpoint mismatch at {path}")
        if stored["row"].get("state") == "success" and not stored["row"].get("error"):
            return True
        history = stored.get("failed_attempts", []) + [stored["row"]]
    # Separate seed ranges keep retries on one side independent of the other.
    seed = 42 + rank * 1000 + (500 if side == "b" else 0) + len(history)

    def finish_checked(records, decision_seed):
        viable = []
        for record in reversed(records):
            if not record.get("draft_rounds"):
                continue
            options, _ = checked_options(record)
            if options:
                viable.append((any(o["passes"] for o in options), record))
        for _, record in sorted(viable, key=lambda entry: entry[0], reverse=True):
            try:
                result = finish_sample(
                    record, worker["model"], worker["revision"], decision_seed
                )
            except Exception as exc:
                print(
                    f"Decision call for {rank:03}-{side} failed ({type(exc).__name__})",
                    flush=True,
                )
                continue
            if result:
                atomic_gzip(
                    path,
                    {
                        "film": film,
                        "side": side,
                        "signature": signature,
                        "row": result,
                        "failed_attempts": records,
                    },
                )
                print(
                    f"Finished {rank:03}-{side} from Qwen's verified slate: {result['parody']}",
                    flush=True,
                )
                return True
        return False

    if history and finish_checked(history, seed):
        return True
    for attempt in range(2):
        row = generate_title(
            film["title"],
            seed + attempt,
            **worker,
            extra_task="Keep sexual scenarios between adults. For family movies aim adult material "
            "at the adults. Make the joke work in the title itself.",
        )
        data = {
            "film": film,
            "side": side,
            "signature": signature,
            "row": row,
            "failed_attempts": history,
        }
        atomic_gzip(path, data)
        if row.get("state") == "success" and not row.get("error"):
            print(f"Saved {rank:03}-{side}: {row['parody']}", flush=True)
            return True
        history = [*history, row]
        if finish_checked(history, seed + attempt + 2):
            return True
    print(f"Unresolved {rank:03}-{side}: {row.get('error')}", flush=True)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--revision", type=Path, required=True)
    p.add_argument("--notebook", type=Path, default=NOTEBOOK)
    p.add_argument("--titles", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--slots", type=int, required=True)
    args = p.parse_args()
    revision = json.loads(args.revision.read_text())
    with args.titles.open(newline="", encoding="utf-8") as source:
        films = list(csv.DictReader(source))
    if not films or len({int(r["rank"]) for r in films}) != len(films):
        p.error("A nonempty film list with unique ranks is required")
    if hashlib.sha256(args.bank.read_bytes()).hexdigest() != BANK_SHA256:
        p.error("Incorrect word bank")
    args.directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        args.revision.read_bytes()
        + args.notebook.read_bytes()
        + args.model.encode()
        + BANK_SHA256.encode()
    ).hexdigest()
    jobs = [(r, side, args.directory, signature) for r in films for side in ("a", "b")]
    with ProcessPoolExecutor(
        args.slots,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(args.model, args.server, args.bank, revision, args.notebook),
    ) as pool:
        completed = list(pool.map(generate_side, jobs, chunksize=1))
    if not all(completed):
        raise SystemExit(f"{completed.count(False)} sample(s) need another attempt")


if __name__ == "__main__":
    main()

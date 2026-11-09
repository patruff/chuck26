"""Prepare DPO training, or import a completed Actions run into the local review app."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.review_request import encode_request, make_request, review_rows

REPO = "patruff/chuck26"


def gh(*args, data=None):
    return subprocess.check_output(
        ["gh", *map(str, args), "-R", REPO], input=data, text=True
    )


def keypair(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with os.fdopen(
            os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
        ) as f:
            f.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
    return serialization.load_pem_private_key(path.read_bytes(), None)


def train(args):
    from cryptography.hazmat.primitives import serialization

    key = keypair(args.key)
    pub = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode()
    )
    request = make_request(args.reviews, pub)
    fields = []
    if args.tracked:
        rel = args.reviews.resolve().relative_to(ROOT)
        if rel.parts[0] != "reviews" or rel.suffix != ".jsonl":
            raise ValueError("A tracked input must be in reviews/")
        subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", str(rel)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        remote = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{REPO}/contents/{rel}?ref=main",
                "-H",
                "Accept: application/vnd.github.raw+json",
            ]
        )
        if remote != args.reviews.read_bytes():
            raise ValueError("The tracked export on GitHub main differs from this file")
        fields = ["-f", f"dataset_file={rel}"]
        request = {"public_key": pub}
    payload = encode_request(request)
    suffix = secrets.token_hex(6).upper()
    gh("secret", "set", "CHUCKLES_DPO_KEY_" + suffix, data=args.key.read_text())
    gh("secret", "set", "CHUCKLES_DPO_" + suffix, data=payload)
    print("Key ID:", suffix)
    if args.prepare_only:
        print(
            "Run dpo-smoke.yml with test_id="
            + suffix
            + (" and dataset_file=" + str(rel) if args.tracked else "")
        )
    else:
        gh(
            "workflow",
            "run",
            "dpo-smoke.yml",
            "--ref",
            "main",
            "-f",
            "test_id=" + suffix,
            *fields,
        )
        print(
            "Submitted DPO training. The run number and this key ID identify the adapter."
        )


def import_artifact(run, stage, key_path, destination):
    from pipeline.blind_artifact import open_sealed
    from scripts.review_films import load_source

    if (stage / "results.jsonl").exists():
        if not list(load_source(stage / "results.jsonl")):
            raise ValueError("The artifact contains no reviewable pairs")
    else:
        from cryptography.hazmat.primitives import serialization

        if not key_path:
            raise ValueError("This artifact needs --key")
        key = serialization.load_pem_private_key(key_path.read_bytes(), None)
        with zipfile.ZipFile(io.BytesIO(open_sealed(stage / "films.bin", key))) as z:
            samples = {}
            for name in z.namelist():
                match = re.fullmatch(r"(\d{3})-([01])\.json\.gz", name)
                if not match:
                    raise ValueError("Unexpected filename in sealed review")
                rank, scale = map(int, match.groups())
                obj = json.loads(gzip.decompress(z.read(name)))
                if (
                    int(obj["film"]["rank"]) != rank
                    or obj["adapter_scale"] != scale
                    or (rank, scale) in samples
                ):
                    raise ValueError("Mismatched or repeated trajectory")
                samples[rank, scale] = obj
        for rank in sorted({n for n, _ in samples}):
            pair = [samples[rank, 0], samples[rank, 1]]
            if secrets.randbelow(2):
                pair.reverse()
            doc = {"options": pair, "assignment_id": f"actions-{run}-{rank}"}
            (stage / f"{rank:03}.blind.json.gz").write_bytes(
                gzip.compress(json.dumps(doc, ensure_ascii=False).encode(), mtime=0)
            )
        if not list(load_source(stage)):
            raise ValueError("No complete blind comparisons")
        (stage / "films.bin").unlink()
    # Publish a complete directory at once. Existing assignments and reviews are stable.
    stage.rename(destination)


def fetch(args):
    info = json.loads(gh("run", "view", args.run, "--json", "status,conclusion"))
    if info != {"status": "completed", "conclusion": "success"}:
        raise ValueError("Choose a successfully completed run")
    names = {
        a["name"]
        for a in json.loads(
            subprocess.check_output(
                ["gh", "api", f"repos/{REPO}/actions/runs/{args.run}/artifacts"]
            )
        )["artifacts"]
        if not a["expired"]
    }
    name = next(
        (n for n in ("review-ready", "sealed-review", "sealed-dpo-test") if n in names),
        None,
    )
    if not name:
        raise ValueError("No unexpired review or DPO artifact in this run")
    dest = args.output or (
        ROOT
        / "reviews"
        / (
            f"adapter-{args.run}"
            if name == "sealed-dpo-test"
            else f"inference-{args.run}"
        )
    )
    if dest.exists():
        raise ValueError(
            f"Already exists: {dest}. Existing blind assignments and labels are preserved."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "download"
        stage.mkdir()
        gh("run", "download", args.run, "-n", name, "--dir", stage)
        if name == "sealed-dpo-test":
            from cryptography.hazmat.primitives import serialization
            from pipeline.blind_artifact import open_sealed

            if not args.key:
                raise ValueError("The adapter needs --key")
            key = serialization.load_pem_private_key(args.key.read_bytes(), None)
            with zipfile.ZipFile(
                io.BytesIO(open_sealed(stage / "comparison.bin", key))
            ) as z:
                model = z.read("chuckles-27b-dpo.gguf")
                report = json.loads(z.read("comparison.json"))
            if (
                model[:4] != b"GGUF"
                or report["adapter_sha256"] != hashlib.sha256(model).hexdigest()
            ):
                raise ValueError("Adapter checksum mismatch")
            (stage / "chuckles-27b-dpo.gguf").write_bytes(model)
            (stage / "comparison.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )
            (stage / "comparison.bin").unlink()
            stage.rename(dest)
        else:
            import_artifact(args.run, stage, args.key, dest)
    print("Saved", dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument("reviews", type=Path)
    t = commands.add_parser("train")
    t.add_argument("reviews", type=Path)
    t.add_argument("--key", type=Path, required=True)
    t.add_argument("--prepare-only", action="store_true")
    t.add_argument("--tracked", action="store_true")
    f = commands.add_parser("fetch")
    f.add_argument("run", type=int)
    f.add_argument("--key", type=Path)
    f.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.command == "validate":
        rows = review_rows(args.reviews)
        print(
            f"{len(rows)} valid preferences; {len({r['film']['title'] for r in rows})} distinct movie titles"
        )
    elif args.command == "train":
        review_rows(args.reviews)
        train(args)
    else:
        fetch(args)


if __name__ == "__main__":
    main()

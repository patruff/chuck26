"""Use recipient-encrypted artifacts for the inputs and results of a blind comparison."""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
import zipfile

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def open_sealed(path, key):
    data = path.read_bytes()
    split = 2 + int.from_bytes(data[:2], "big")
    session = key.decrypt(
        data[2:split],
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    return AESGCM(session).decrypt(
        data[split : split + 12], data[split + 12 :], b"chuckles-dpo-v1"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "pack", "restore"])
    p.add_argument("work", type=Path)
    p.add_argument("bundle", type=Path)
    args = p.parse_args()
    key = serialization.load_pem_private_key(
        os.environ.pop("RECIPIENT_PRIVATE_KEY").encode(), None
    )
    if args.mode == "restore":
        target = args.work / "samples"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(open_sealed(args.bundle, key))) as z:
            for name in z.namelist():
                if not re.fullmatch(r"[0-9]{3}-[01]\.json\.gz", name):
                    raise ValueError("Unrecognized checkpoint filename")
                rank = os.getenv("BLIND_RANK")
                if rank and int(name[:3]) != int(rank):
                    continue
                (target / name).write_bytes(z.read(name))
        return
    if args.mode == "prepare":
        with zipfile.ZipFile(io.BytesIO(open_sealed(args.bundle, key))) as z:
            model = z.read("chuckles-27b-dpo.gguf")
            from pipeline.settings import MODEL, REVISION, SHA256

            report = json.loads(z.read("comparison.json"))
            if (
                report["model"],
                report["revision"],
                report["base_weight_sha256"],
                report["adapter_sha256"],
            ) != (MODEL, REVISION, SHA256, hashlib.sha256(model).hexdigest()):
                raise ValueError("Unexpected adapter")
            (args.work / "adapter.gguf").write_bytes(model)
            (args.work / "report.json").write_bytes(z.read("comparison.json"))
    else:
        from training.gguf_dpo.seal import seal

        files = sorted((args.work / "samples").glob("*.json.gz"))
        if files:
            pub = key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            seal(files, pub.decode(), args.bundle)


if __name__ == "__main__":
    main()

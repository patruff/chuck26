"""Seal the result archive for its requested recipient before uploading it from CI."""
import argparse
import io
import json
import os
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def seal(files, public_key, output):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    session_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    key = serialization.load_pem_public_key(public_key.encode())
    wrapped = key.encrypt(session_key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                    algorithm=hashes.SHA256(),label=None))
    body = AESGCM(session_key).encrypt(nonce, archive.getvalue(), b"chuckles-dpo-v1")
    output.write_bytes(len(wrapped).to_bytes(2,"big") + wrapped + nonce + body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    seal([args.results/"chuckles-27b-dpo.gguf", args.results/"comparison.json"],
         json.loads(args.request.read_text())["public_key"],args.output)

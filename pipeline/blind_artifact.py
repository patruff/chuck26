"""Use recipient-encrypted artifacts for the inputs and results of a blind comparison."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import zipfile

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def open_sealed(path, key):
    data = path.read_bytes()
    split = 2 + int.from_bytes(data[:2], "big")
    session = key.decrypt(data[2:split], padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                       algorithm=hashes.SHA256(), label=None))
    return AESGCM(session).decrypt(data[split:split+12], data[split+12:], b"chuckles-dpo-v1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["prepare","pack"])
    p.add_argument("work",type=Path)
    p.add_argument("bundle",type=Path)
    args=p.parse_args()
    key=serialization.load_pem_private_key(os.environ.pop("RECIPIENT_PRIVATE_KEY").encode(),None)
    if args.mode=="prepare":
        with zipfile.ZipFile(io.BytesIO(open_sealed(args.bundle,key))) as z:
            model=z.read("chuckles-27b-dpo.gguf")
            if hashlib.sha256(model).hexdigest()!="ec6f81c028db8230f57b5617abd37d0da75a169579392a611a6cc022edf96806":
                raise ValueError("Unexpected adapter")
            (args.work/"adapter.gguf").write_bytes(model)
            (args.work/"report.json").write_bytes(z.read("comparison.json"))
    else:
        from training.gguf_dpo.seal import seal
        files=sorted((args.work/"samples").glob("*.json.gz"))
        if files:
            pub=key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
            seal(files,pub.decode(),args.bundle)


if __name__=="__main__":
    main()

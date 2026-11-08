"""Resolve the private, pinned 2025 Chuckles word bank and check its historical scores."""
import base64
import binascii
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import requests

REPO = "patruff/chuck2025"
REVISION = "783c52912c0be25ccc27fbca768548c46c977a29"
FILENAME = "word_structures.py"
BANK_URL = f"https://github.com/{REPO}/blob/{REVISION}/{FILENAME}"
BANK_SHA256 = "c1ad29501bbc32c0fb8a2a0cefeefe82ca7bb441e1955d87e94fc630ffca9eff"
# Canonical JSON digest of the complete tool output printed in Colab cell GvGq8GcoGbCD.
CONTROL_SHA256 = "7280eb0ef3547a041c2dc0ee836cf0b43e7ee2fe12e4771d8660d65acc437f36"
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "Chuckles_RunPod.ipynb"


def snapshot_bank(local_path, directory):
    """Use the exact bytes for this whole run, in the caller's private temporary directory."""
    sibling = Path(__file__).resolve().parents[2] / "chuck2025" / FILENAME
    api_path = f"repos/{REPO}/contents/{FILENAME}?ref={REVISION}"
    if local_path is not None:
        data = local_path.read_bytes()
    elif encoded := os.getenv("CHUCKLES_2025_BANK_GZ"):
        try:
            data = gzip.decompress(base64.b64decode(encoded.strip(), validate=True))
        except (OSError, EOFError, binascii.Error) as exc:
            raise ValueError("CHUCKLES_2025_BANK_GZ must be base64-encoded gzip data") from exc
    elif sibling.is_file():
        data = sibling.read_bytes()
    elif token := os.getenv("CHUCK2025_READ_TOKEN"):
        response = requests.get("https://api.github.com/" + api_path, timeout=30,
                                headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github.raw+json"})
        response.raise_for_status()
        data = response.content
    elif shutil.which("gh"):
        result = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw+json", api_path],
                                capture_output=True, timeout=45)
        if result.returncode:
            raise ValueError(f"Cannot read {BANK_URL}. Use an authenticated gh login or --march-word-bank PATH")
        data = result.stdout
    else:
        raise ValueError(f"Supply --march-word-bank PATH or read access to {BANK_URL} via CHUCK2025_READ_TOKEN")
    if hashlib.sha256(data).hexdigest() != BANK_SHA256:
        raise ValueError(f"Word bank checksum mismatch. Use the pinned 2025 file at {BANK_URL}")
    path = directory / FILENAME
    path.write_bytes(data)
    return path


def preflight(bank):
    from generate_march import RUNNING_MAN_TOOLS, WordPhoneTool, load_tool, load_word_bank, suggestions_for

    words, phones = load_word_bank(bank)
    if len(words) != 1565 or len(phones) != 1565:
        raise ValueError("Expected 1,565 words and 1,565 custom pronunciations")
    suggest, verify = [load_tool(k, revision=v, trust_remote_code=True) for k, v in RUNNING_MAN_TOOLS.items()]
    title = "The Running Man"
    raw, _ = suggestions_for(title, suggest, json.dumps(words), phones, 0.6)
    payload = {word: raw[word.lower()] for word in title.split()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()
    if digest != CONTROL_SHA256:
        raise ValueError("The full Running Man suggestion payload differs from the saved 2025 Colab")
    checker = WordPhoneTool(verify, phones, suggest)
    checks = []
    for original, replacement, expected in [
        ("running", "cumming", (0.665, 1.0)), ("man", "klan", (0.638, 0.925)),
        ("running", "huning", (0.83, 1.0)), ("running", "hunning", (None, None)),
    ]:
        obs = checker.forward(original, replacement)
        if (obs["suggestion_score"], obs["word_phone_score"]) != expected:
            raise ValueError(f"Historical phonetic check failed for {original!r} → {replacement!r}")
        checks.append({k: obs[k] for k in ("word", "compare_to", "suggestion_score", "word_phone_score")})
    return {"bank_url": BANK_URL, "bank_sha256": BANK_SHA256, "word_count": len(words),
            "custom_pronunciations": len(phones), "suggestion_payload_sha256": digest,
            "tool_revisions": RUNNING_MAN_TOOLS, "checks": checks}

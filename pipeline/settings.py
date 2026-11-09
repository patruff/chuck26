"""Pinned model and the editable prompt displayed in GitHub's inference form."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
MODEL = MODEL_CONFIG["repository"]
REVISION = MODEL_CONFIG["revision"]
FILENAME = MODEL_CONFIG["file"]
SHA256 = MODEL_CONFIG["sha256"]
POLICY = "humor-first-v2"


def default_prompt():
    path = ROOT / ".github/workflows/human-guidance.yml"
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return workflow["on"]["workflow_dispatch"]["inputs"]["system_prompt"]["default"]


def system_prompt():
    path = os.environ.get("CHUCKLES_PROMPT_FILE")
    value = Path(path).read_text(encoding="utf-8") if path else default_prompt()
    if not value.strip():
        raise ValueError("The system prompt is empty")
    return value


def live_revision():
    text = system_prompt()
    return dict(
        model=MODEL,
        selection_policy=POLICY,
        applied_instructions=text,
        applied_instructions_sha256=hashlib.sha256(text.encode()).hexdigest(),
        feedback_sha256=None,
        task_comments=[],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["preview", "model-env"])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.command == "model-env":
        for key, value in {
            "MODEL_NAME": MODEL,
            "HF_REV": REVISION,
            "MODEL_FILE": FILENAME,
            "WEIGHT_SHA": SHA256,
            "LLAMA_REV": MODEL_CONFIG["llama_cpp"],
        }.items():
            print(f"{key}={value}")
        return
    prompt = os.getenv("CHUCKLES_PROMPT_OVERRIDE") or system_prompt()
    if not prompt.strip():
        p.error("Enter a nonempty system prompt")
    if args.output:
        args.output.write_bytes(prompt.encode())
    print(prompt)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## Inference prompt\n\nModel `{MODEL}` · `{FILENAME}`\n\n")
            f.write("\n".join("    " + line for line in prompt.splitlines()) + "\n")


if __name__ == "__main__":
    main()

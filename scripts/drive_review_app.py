"""Google Drive parody review app.

Manages a three-folder workflow in Google Drive for generating parodies,
reviewing them, and building DPO preference datasets:

    chuck26/input/           - Drop CSVs with a 'title' column here
    chuck26/to_be_checked/   - Generated parodies land here for review
    chuck26/finished_preference/ - Reviewed CSVs archived here after DPO export

Subcommands:
    generate  - Pull CSVs from input/, generate parodies, push to to_be_checked/
    process   - Pull reviewed CSVs from to_be_checked/, build DPO, push to HF
    status    - Show what's in each folder

Required environment variables:
    GOOGLE_DRIVE_CREDENTIALS - Service account JSON for Drive API
    CEREBRAS_API_KEY         - For generation (generate subcommand only)
    HF_TOKEN                 - For pushing DPO data (process subcommand only)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

DRIVE_BASE_FOLDER = "chuck26"
INPUT_FOLDER = "input"
TO_BE_CHECKED_FOLDER = "to_be_checked"
FINISHED_FOLDER = "finished_preference"
SCOPES = ["https://www.googleapis.com/auth/drive"]


def _get_drive_service():
    """Build an authenticated Google Drive API service."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
    if not creds_json:
        raise ValueError("GOOGLE_DRIVE_CREDENTIALS environment variable not set")

    creds_info = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(
        creds_info, scopes=SCOPES
    )
    return build("drive", "v3", credentials=credentials)


def _find_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    """Find a folder by name (creating it if missing) and return its ID."""
    query = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)")
        .execute()
    )
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    metadata: dict[str, Any] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = service.files().create(body=metadata, fields="id").execute()
    print(f"Created folder: {name}")
    return folder["id"]


def _setup_folders(service) -> dict[str, str]:
    """Ensure the full folder tree exists and return a name->id map."""
    base = _find_or_create_folder(service, DRIVE_BASE_FOLDER)
    return {
        "base": base,
        "input": _find_or_create_folder(service, INPUT_FOLDER, base),
        "to_be_checked": _find_or_create_folder(service, TO_BE_CHECKED_FOLDER, base),
        "finished": _find_or_create_folder(service, FINISHED_FOLDER, base),
    }


def _list_csvs(service, folder_id: str) -> list[dict[str, str]]:
    """List CSV files in a Drive folder."""
    q = f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
    results = (
        service.files()
        .list(q=q, spaces="drive", fields="files(id, name, modifiedTime)")
        .execute()
    )
    return results.get("files", [])


def _download_csv_text(service, file_id: str) -> str:
    """Download a CSV file's content as a UTF-8 string."""
    from googleapiclient.http import MediaIoBaseDownload

    buf = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8")


def _upload_csv(service, content: str, folder_id: str, filename: str) -> str:
    """Upload a CSV string to a Drive folder. Returns the new file ID."""
    from googleapiclient.http import MediaInMemoryUpload

    media = MediaInMemoryUpload(
        content.encode("utf-8"), mimetype="text/csv", resumable=False
    )
    metadata = {"name": filename, "parents": [folder_id]}
    f = (
        service.files()
        .create(body=metadata, media_body=media, fields="id, name")
        .execute()
    )
    print(f"  Uploaded: {filename}")
    return f["id"]


def _move_file(service, file_id: str, dest_folder_id: str) -> None:
    """Move a file to a different folder (remove old parents, add new)."""
    file = service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents", []))
    service.files().update(
        fileId=file_id,
        addParents=dest_folder_id,
        removeParents=previous_parents,
        fields="id, parents",
    ).execute()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

REVIEW_FIELDS = [
    "id",
    "input_title",
    "parody_text",
    "humor_note",
    "phonetic_scores",
    "avg_phonetic_score",
    "model_name",
    "adapter",
    "status",
]

DATASET_SYSTEM_PROMPT = (
    "You are a comedy writer who creates funny parody titles. "
    "Replace words with phonetically similar but humorous alternatives. "
    "Use the phonetic analysis tool to verify similarity scores above 0.6."
)


def _parse_csv_text(text: str) -> list[dict[str, str]]:
    """Parse CSV text into a list of row dicts."""
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def _rows_to_csv_text(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    """Serialize row dicts back to CSV text."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _read_titles_from_csv_text(text: str) -> list[str]:
    """Extract titles from a CSV string (column: 'title' or 'original')."""
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    col = "title" if "title" in fieldnames else "original" if "original" in fieldnames else None
    if col is None:
        raise ValueError(f"CSV must have a 'title' or 'original' column, found: {fieldnames}")
    return [row[col].strip() for row in reader if row.get(col, "").strip()]


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Download input CSVs, generate parodies, upload review CSVs."""
    from chuckles_prime import config as _config
    from chuckles_prime import generator as _generator
    from chuckles_prime import model as _model
    from chuckles_prime import tools as _tools
    from chuckles_prime.types import AgentTrace, GenerationRecord

    service = _get_drive_service()
    folders = _setup_folders(service)

    # List input CSVs
    input_files = _list_csvs(service, folders["input"])
    if not input_files:
        print("No input CSVs found in chuck26/input/")
        return

    print(f"Found {len(input_files)} input file(s)")

    # Load config and set up model/agent
    config = _config.load_config(args.settings)
    model = _model.create_model(config)
    model_name = args.model or config.model_name
    adapter = args.adapter or ""
    parody_tool, phone_tool = _tools.load_parody_tools()
    agent = _generator.create_agent(model, parody_tool, phone_tool)

    for file_info in input_files:
        fname = file_info["name"]
        print(f"\nProcessing: {fname}")

        csv_text = _download_csv_text(service, file_info["id"])
        titles = _read_titles_from_csv_text(csv_text)
        print(f"  {len(titles)} titles")

        if args.limit and args.limit > 0:
            titles = titles[: args.limit]
            print(f"  Limited to {len(titles)} titles")

        # Generate
        records: list[GenerationRecord] = []
        for i, title in enumerate(titles):
            try:
                record = _generator.generate_single(title, agent, parody_tool, config)
            except Exception as e:
                record = GenerationRecord(
                    input_title=title,
                    candidates=[],
                    trace=AgentTrace(
                        steps=[], final_output="", token_usage=None, state="error"
                    ),
                    model_name=model_name,
                    error=str(e),
                )
            status = "OK" if not record.error else f"ERROR: {record.error}"
            print(f"  [{i + 1}/{len(titles)}] {title} ... {status}")
            records.append(record)

        # Build review CSV
        review_rows: list[dict[str, str]] = []
        row_id = 0
        for rec in records:
            if rec.error:
                row_id += 1
                review_rows.append({
                    "id": str(row_id),
                    "input_title": rec.input_title,
                    "parody_text": f"[ERROR: {rec.error}]",
                    "humor_note": "",
                    "phonetic_scores": "",
                    "avg_phonetic_score": "",
                    "model_name": model_name,
                    "adapter": adapter,
                    "status": "error",
                })
                continue
            for cand in rec.candidates:
                row_id += 1
                scores = cand.phonetic_scores
                avg = sum(scores.values()) / len(scores) if scores else 0.0
                review_rows.append({
                    "id": str(row_id),
                    "input_title": rec.input_title,
                    "parody_text": cand.text,
                    "humor_note": cand.humor_note,
                    "phonetic_scores": json.dumps(scores),
                    "avg_phonetic_score": f"{avg:.3f}",
                    "model_name": model_name,
                    "adapter": adapter,
                    "status": "pending",
                })

        # Upload review CSV to to_be_checked/
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        stem = Path(fname).stem
        out_name = f"review-{stem}-{ts}.csv"
        csv_out = _rows_to_csv_text(review_rows, REVIEW_FIELDS)
        _upload_csv(service, csv_out, folders["to_be_checked"], out_name)
        print(f"  Uploaded {row_id} candidates to to_be_checked/{out_name}")

    print("\nGeneration complete. Review the CSVs in chuck26/to_be_checked/")
    print("Mark each row's status as 'chosen' or 'rejected', then run: process")


# ---------------------------------------------------------------------------
# Subcommand: process
# ---------------------------------------------------------------------------


def cmd_process(args: argparse.Namespace) -> None:
    """Download reviewed CSVs, build DPO pairs, push to HF, archive."""
    service = _get_drive_service()
    folders = _setup_folders(service)

    # List to_be_checked CSVs
    check_files = _list_csvs(service, folders["to_be_checked"])
    if not check_files:
        print("No CSVs found in chuck26/to_be_checked/")
        return

    print(f"Found {len(check_files)} file(s) in to_be_checked/")

    all_rows: list[dict[str, str]] = []
    reviewed_file_ids: list[tuple[str, str]] = []  # (file_id, file_name)

    for file_info in check_files:
        fname = file_info["name"]
        csv_text = _download_csv_text(service, file_info["id"])
        rows = _parse_csv_text(csv_text)

        statuses = {r.get("status", "").strip().lower() for r in rows}
        if "chosen" not in statuses and "rejected" not in statuses:
            print(f"  Skipping {fname} (no chosen/rejected rows yet)")
            continue

        chosen_count = sum(1 for r in rows if r.get("status", "").strip().lower() == "chosen")
        rejected_count = sum(1 for r in rows if r.get("status", "").strip().lower() == "rejected")
        print(f"  {fname}: {chosen_count} chosen, {rejected_count} rejected")

        all_rows.extend(rows)
        reviewed_file_ids.append((file_info["id"], fname))

    if not all_rows:
        print("No reviewed rows found. Mark status as 'chosen'/'rejected' first.")
        return

    # Build DPO pairs
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"chosen": [], "rejected": []}
    )
    for row in all_rows:
        status = row.get("status", "").strip().lower()
        if status in ("chosen", "rejected"):
            groups[row["input_title"]][status].append(row)

    dpo_rows: list[dict[str, Any]] = []
    for input_title, group in groups.items():
        for chosen in group["chosen"]:
            for rejected in group["rejected"]:
                dpo_rows.append({
                    "prompt": [
                        {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Create a phonetically-sound parody of: '{input_title}'",
                        },
                    ],
                    "chosen": [
                        {"role": "assistant", "content": chosen["parody_text"]},
                    ],
                    "rejected": [
                        {"role": "assistant", "content": rejected["parody_text"]},
                    ],
                    "chosen_model": chosen.get("model_name", ""),
                    "chosen_adapter": chosen.get("adapter", ""),
                    "chosen_phonetic_score": chosen.get("avg_phonetic_score", ""),
                    "chosen_humor_note": chosen.get("humor_note", ""),
                    "rejected_model": rejected.get("model_name", ""),
                    "rejected_adapter": rejected.get("adapter", ""),
                    "rejected_phonetic_score": rejected.get("avg_phonetic_score", ""),
                    "rejected_humor_note": rejected.get("humor_note", ""),
                })

    print(f"\nBuilt {len(dpo_rows)} DPO preference pairs")

    # Push to HuggingFace
    if dpo_rows and args.dpo_repo and not args.no_push:
        from datasets import Dataset, load_dataset
        from huggingface_hub import login

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: HF_TOKEN not set, cannot push to Hub")
        else:
            login(token=token)
            new_ds = Dataset.from_list(dpo_rows)

            # Append to existing dataset if it exists
            try:
                existing = load_dataset(args.dpo_repo, split="train")
                combined = existing.to_list() + dpo_rows
                final_ds = Dataset.from_list(combined)
                print(
                    f"Appending {len(dpo_rows)} new pairs to "
                    f"{len(existing)} existing ({len(combined)} total)"
                )
            except Exception:
                final_ds = new_ds
                print(f"Creating new dataset with {len(dpo_rows)} pairs")

            final_ds.push_to_hub(args.dpo_repo, split="train", private=True)
            print(f"Pushed to {args.dpo_repo}")
    elif dpo_rows and args.no_push:
        print(f"Built {len(dpo_rows)} pairs (--no-push, skipping Hub)")
    elif not dpo_rows:
        print("No DPO pairs to push (need both chosen and rejected for same title)")

    # Move reviewed files to finished_preference/
    for file_id, fname in reviewed_file_ids:
        _move_file(service, file_id, folders["finished"])
        print(f"  Moved {fname} -> finished_preference/")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> None:
    """Show what's in each Drive folder."""
    service = _get_drive_service()
    folders = _setup_folders(service)

    for label, key in [
        ("input", "input"),
        ("to_be_checked", "to_be_checked"),
        ("finished_preference", "finished"),
    ]:
        files = _list_csvs(service, folders[key])
        print(f"\nchuck26/{label}/ ({len(files)} CSV files)")
        if not files:
            print("  (empty)")
        for f in files:
            # Try to show review stats for to_be_checked files
            extra = ""
            if key == "to_be_checked":
                try:
                    text = _download_csv_text(service, f["id"])
                    rows = _parse_csv_text(text)
                    status_counts: dict[str, int] = defaultdict(int)
                    for r in rows:
                        status_counts[r.get("status", "pending").strip().lower()] += 1
                    parts = [f"{v} {k}" for k, v in sorted(status_counts.items())]
                    extra = f"  [{', '.join(parts)}]"
                except Exception:
                    pass
            print(f"  {f['name']}{extra}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="drive-review",
        description="Google Drive parody review app with DPO export",
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate --
    gen = sub.add_parser(
        "generate",
        help="Pull from input/, generate parodies, push to to_be_checked/",
    )
    gen.add_argument(
        "--settings",
        default="settings.json",
        help="Path to settings JSON (default: settings.json)",
    )
    gen.add_argument(
        "--model", default=None, help="Override model name for provenance"
    )
    gen.add_argument(
        "--adapter", default="", help="Adapter/LoRA name for provenance"
    )
    gen.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max titles per file (0 = all)",
    )
    gen.set_defaults(func=cmd_generate)

    # -- process --
    proc = sub.add_parser(
        "process",
        help="Pull reviewed CSVs, build DPO, push to HF, archive",
    )
    proc.add_argument(
        "--dpo-repo",
        default="patruff/chuckles-dpo",
        help="HF Hub repo ID (default: patruff/chuckles-dpo)",
    )
    proc.add_argument(
        "--no-push",
        action="store_true",
        help="Build DPO data but skip pushing to Hub",
    )
    proc.set_defaults(func=cmd_process)

    # -- status --
    st = sub.add_parser("status", help="Show folder contents")
    st.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

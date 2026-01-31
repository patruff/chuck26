#!/usr/bin/env python3
"""
Google Drive Batch Parody Processor

Manages input/output workflow with Google Drive:
- Input folder: parodiesdata/input/ - CSV files with titles to process
- Output folder: parodiesdata/output/ - Generated parodies with reasoning
- DPO folder: parodiesdata/dpo/ - Human-annotated chosen vs rejected parodies
- State tracking: Remembers what's been processed to avoid duplicates
"""

import os
import sys
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# Import the generate_parody function
from generate_parody import generate_parody

# Configuration
DRIVE_BASE_FOLDER = "parodiesdata"
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
DPO_FOLDER = "dpo"
STATE_FILE = Path.home() / ".parody_processor" / "processed_state.json"
SCOPES = ['https://www.googleapis.com/auth/drive']


def get_drive_service():
    """Initialize Google Drive API service."""
    creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS")

    if not creds_json:
        raise ValueError("GOOGLE_DRIVE_CREDENTIALS environment variable not set")

    try:
        creds_info = json.loads(creds_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in GOOGLE_DRIVE_CREDENTIALS: {e}")

    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=credentials)
    print("✅ Connected to Google Drive API")
    return service


def find_or_create_folder(service, folder_name: str, parent_id: Optional[str] = None) -> str:
    """Find or create a folder, optionally within a parent folder."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()

    files = results.get('files', [])

    if files:
        folder_id = files[0]['id']
        print(f"📁 Found folder: {folder_name} (ID: {folder_id})")
        return folder_id

    # Create folder
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        file_metadata['parents'] = [parent_id]

    folder = service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()

    folder_id = folder.get('id')
    print(f"📁 Created folder: {folder_name} (ID: {folder_id})")
    return folder_id


def setup_drive_folders(service) -> Dict[str, str]:
    """Set up the folder structure and return folder IDs."""
    print("\n📂 Setting up Google Drive folder structure...")

    # Create/find base folder
    base_id = find_or_create_folder(service, DRIVE_BASE_FOLDER)

    # Create/find subfolders
    input_id = find_or_create_folder(service, INPUT_FOLDER, base_id)
    output_id = find_or_create_folder(service, OUTPUT_FOLDER, base_id)
    dpo_id = find_or_create_folder(service, DPO_FOLDER, base_id)

    return {
        'base': base_id,
        'input': input_id,
        'output': output_id,
        'dpo': dpo_id
    }


def list_csv_files(service, folder_id: str) -> List[Dict]:
    """List all CSV files in a folder."""
    query = f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"

    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name, modifiedTime)',
        orderBy='modifiedTime desc'
    ).execute()

    return results.get('files', [])


def download_csv(service, file_id: str, local_path: Path) -> None:
    """Download a CSV file from Drive."""
    request = service.files().get_media(fileId=file_id)

    with io.FileIO(str(local_path), 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()


def upload_csv(service, local_path: Path, folder_id: str, filename: str) -> str:
    """Upload a CSV file to Drive."""
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }

    media = MediaFileUpload(
        str(local_path),
        mimetype='text/csv',
        resumable=True
    )

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, webViewLink'
    ).execute()

    print(f"  ✅ Uploaded: {filename} (ID: {file.get('id')})")
    return file.get('id')


def load_processed_state() -> Dict[str, Set[str]]:
    """Load state of processed files."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                # Convert lists back to sets
                return {k: set(v) for k, v in data.items()}
        except Exception as e:
            print(f"⚠️  Could not load state: {e}")
    return {}


def save_processed_state(state: Dict[str, Set[str]]) -> None:
    """Save state of processed files."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Convert sets to lists for JSON serialization
    data = {k: list(v) for k, v in state.items()}
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def extract_parody_from_result(result: str, original_title: str) -> Dict[str, str]:
    """Extract final parody and reasoning from generation result."""
    # Extract final chosen parody
    parody_pattern = r"### Final Chosen Parody:.*?\n\*\*\"?([^\*\"]+)\"?\*\*"
    parody_match = re.search(parody_pattern, result, re.DOTALL)

    if parody_match:
        final_parody = parody_match.group(1).strip()
        if '[' in final_parody or ']' in final_parody:
            final_parody = "Generation failed - no valid parody found"
    else:
        final_parody = "Generation failed - no parody found"

    # Extract reasoning
    reasoning_pattern = r"</think>(.*?)(?:--- observations ---|$)"
    reasoning_match = re.search(reasoning_pattern, result, re.DOTALL)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip().replace("</think>", "").strip()
    else:
        reasoning = "No reasoning extracted"

    return {
        'input': original_title,
        'parody_result': final_parody,
        'reasoning': reasoning,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def process_input_file(service, file_info: Dict, folders: Dict, api_key: str, model: str) -> int:
    """Process a single input CSV file."""
    file_id = file_info['id']
    file_name = file_info['name']

    print(f"\n{'='*80}")
    print(f"Processing: {file_name}")
    print(f"{'='*80}")

    # Download input file
    temp_dir = Path("/tmp/parody_processing")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / file_name
    download_csv(service, file_id, input_path)
    print(f"📥 Downloaded: {file_name}")

    # Read titles
    titles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'title' in row:
                titles.append(row['title'])

    print(f"📋 Found {len(titles)} titles to process")

    # Process each title
    results = []
    for i, title in enumerate(titles, 1):
        print(f"\n[{i}/{len(titles)}] Processing: {title}")

        try:
            result = generate_parody(
                title=title,
                model_name=model,
                api_key=api_key,
                output_dir=str(temp_dir / f"work_{i}")
            )

            extracted = extract_parody_from_result(result, title)
            results.append(extracted)
            print(f"✅ {title} → {extracted['parody_result']}")

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'input': title,
                'parody_result': f"ERROR: {str(e)}",
                'reasoning': "Generation failed",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    # Create output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{file_name.replace('.csv', '')}_{timestamp}.csv"
    output_path = temp_dir / output_filename

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'parody_result', 'reasoning', 'timestamp'])
        writer.writeheader()
        writer.writerows(results)

    # Upload to output folder
    upload_csv(service, output_path, folders['output'], output_filename)

    print(f"\n✅ Completed processing {file_name}")
    print(f"📤 Uploaded results to output folder")

    return len(results)


def main():
    """Main processing function."""
    print("🚀 Google Drive Batch Parody Processor")
    print("=" * 80)

    # Get configuration
    api_key = os.environ.get("CEREBRAS_API_KEY")
    model = os.environ.get("CEREBRAS_MODEL", "qwen-3-32b")

    if not api_key:
        print("❌ CEREBRAS_API_KEY not set")
        sys.exit(1)

    try:
        # Connect to Drive
        service = get_drive_service()

        # Set up folders
        folders = setup_drive_folders(service)

        # Load processed state
        state = load_processed_state()
        processed_files = state.get('processed_files', set())

        # List input files
        print(f"\n📋 Checking input folder for CSV files...")
        input_files = list_csv_files(service, folders['input'])

        if not input_files:
            print("ℹ️  No input files found")
            return

        print(f"Found {len(input_files)} input file(s)")

        # Filter out already processed files
        new_files = [f for f in input_files if f['id'] not in processed_files]

        if not new_files:
            print("✅ All input files have been processed")
            return

        print(f"\n🆕 Processing {len(new_files)} new file(s):")
        for f in new_files:
            print(f"  - {f['name']}")

        # Process each new file
        total_processed = 0
        for file_info in new_files:
            count = process_input_file(service, file_info, folders, api_key, model)
            total_processed += count

            # Mark as processed
            processed_files.add(file_info['id'])
            state['processed_files'] = processed_files
            save_processed_state(state)

        # Summary
        print("\n" + "=" * 80)
        print("📊 PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Files processed: {len(new_files)}")
        print(f"Total parodies generated: {total_processed}")
        print(f"Output folder: parodiesdata/{OUTPUT_FOLDER}")
        print(f"DPO folder: parodiesdata/{DPO_FOLDER} (for human annotations)")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

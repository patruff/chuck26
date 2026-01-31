#!/usr/bin/env python3
"""
Upload Parody Results to Google Drive

Automatically uploads generated parody files (CSV, raw output, debug logs)
to the "parodiesdata" Google Drive folder using service account credentials.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configuration
DRIVE_FOLDER_NAME = "parodiesdata"
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def get_drive_service():
    """Initialize Google Drive API service using service account credentials."""
    creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS")

    if not creds_json:
        raise ValueError(
            "GOOGLE_DRIVE_CREDENTIALS environment variable not set. "
            "This should contain your service account JSON credentials."
        )

    # Parse credentials from JSON string
    try:
        creds_info = json.loads(creds_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in GOOGLE_DRIVE_CREDENTIALS: {e}")

    # Create credentials from service account info
    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES
    )

    # Build Drive API service
    service = build('drive', 'v3', credentials=credentials)
    print("✅ Connected to Google Drive API")

    return service


def find_or_create_folder(service, folder_name: str) -> str:
    """Find a folder by name, create it if it doesn't exist, and return its ID."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()

    files = results.get('files', [])

    if files:
        folder_id = files[0]['id']
        print(f"📁 Found folder: {folder_name} (ID: {folder_id})")
        if len(files) > 1:
            print(f"⚠️  Warning: Multiple folders named '{folder_name}' found. Using the first one.")
        return folder_id

    # Create the folder if it doesn't exist
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }

    folder = service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()

    folder_id = folder.get('id')
    print(f"📁 Created folder: {folder_name} (ID: {folder_id})")

    return folder_id


def upload_file(service, file_path: Path, folder_id: str) -> Optional[str]:
    """Upload a file to Google Drive folder."""
    if not file_path.exists():
        print(f"  ⚠️  File not found: {file_path}")
        return None

    # Determine MIME type based on file extension
    mime_types = {
        '.csv': 'text/csv',
        '.txt': 'text/plain',
        '.log': 'text/plain',
        '.json': 'application/json',
        '.pdf': 'application/pdf',
    }

    mime_type = mime_types.get(file_path.suffix, 'application/octet-stream')

    file_metadata = {
        'name': file_path.name,
        'parents': [folder_id]
    }

    media = MediaFileUpload(
        str(file_path),
        mimetype=mime_type,
        resumable=True
    )

    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()

        file_id = file.get('id')
        file_name = file.get('name')
        web_link = file.get('webViewLink', 'N/A')

        print(f"  ✅ Uploaded: {file_name} (ID: {file_id})")
        return file_id

    except Exception as e:
        print(f"  ❌ Error uploading {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def upload_directory(service, directory: Path, folder_id: str, pattern: str = "*"):
    """Upload all files matching pattern from a directory to Google Drive."""
    if not directory.exists():
        print(f"⚠️  Directory not found: {directory}")
        return 0

    files = list(directory.glob(pattern))

    if not files:
        print(f"ℹ️  No files matching '{pattern}' found in {directory}")
        return 0

    print(f"\n📤 Uploading {len(files)} file(s) from {directory}...")

    success_count = 0
    for file_path in files:
        if file_path.is_file():
            if upload_file(service, file_path, folder_id):
                success_count += 1

    return success_count


def main():
    """Main function to upload parody results to Google Drive."""
    print("🚀 Starting upload to Google Drive...")
    print("=" * 80)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Upload parody results to Google Drive')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory containing results')
    parser.add_argument('--debug-dir', type=str, default='./parody_output',
                        help='Debug directory containing detailed logs')
    parser.add_argument('--debug-log', type=str, default='./debug.log',
                        help='Path to debug log file')

    args = parser.parse_args()

    try:
        # Initialize Google Drive service
        print("\n📡 Connecting to Google Drive...")
        service = get_drive_service()

        # Find or create the parodiesdata folder
        folder_id = find_or_create_folder(service, DRIVE_FOLDER_NAME)

        total_uploaded = 0

        # Upload CSV files from output directory
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            print(f"\n📊 Processing CSV files from {output_dir}...")
            total_uploaded += upload_directory(service, output_dir, folder_id, "*.csv")

        # Upload raw output files from output directory
        if output_dir.exists():
            print(f"\n📄 Processing raw output files from {output_dir}...")
            total_uploaded += upload_directory(service, output_dir, folder_id, "RAW_*.txt")

        # Upload debug files from parody_output directory
        debug_dir = Path(args.debug_dir)
        if debug_dir.exists():
            print(f"\n🔍 Processing debug files from {debug_dir}...")
            total_uploaded += upload_directory(service, debug_dir, folder_id, "*")

        # Upload debug log if it exists
        debug_log = Path(args.debug_log)
        if debug_log.exists():
            print(f"\n📝 Uploading debug log...")
            if upload_file(service, debug_log, folder_id):
                total_uploaded += 1

        # Summary
        print("\n" + "=" * 80)
        print("📊 UPLOAD SUMMARY")
        print("=" * 80)
        print(f"✅ Successfully uploaded: {total_uploaded} file(s)")
        print(f"📁 Google Drive Folder: {DRIVE_FOLDER_NAME}")
        print(f"🔗 Folder ID: {folder_id}")
        print("=" * 80)

        if total_uploaded > 0:
            print("\n🎉 Results are now available in Google Drive!")
        else:
            print("\n⚠️  No files were uploaded. Check your output directories.")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

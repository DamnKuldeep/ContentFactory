#!/usr/bin/env python3
"""
reset_pipeline.py — wipe the Drive folder + shared DB and local state for clean smoke tests.

Trashes (default) every child of DRIVE_PARENT_FOLDER_ID — all Batch_* video folders, the shared
manifest DB, and the lock file — and deletes the local manifest + work dirs. DESTRUCTIVE and
outward-facing, so it requires interactive confirmation or --yes.

    python ContentFactory/scripts/reset_pipeline.py            # confirm, trash, wipe local
    python ContentFactory/scripts/reset_pipeline.py --yes
    python ContentFactory/scripts/reset_pipeline.py --yes --permanent
    python ContentFactory/scripts/reset_pipeline.py --yes --keep-local
"""

import argparse
import os
import shutil
import sys

_CF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ContentFactory/
if _CF_ROOT not in sys.path:
    sys.path.insert(0, _CF_ROOT)

from shared import config
from shared.drive import get_drive_client


def _list_children(service, parent):
    files, page = [], None
    while True:
        resp = service.files().list(
            q=f"'{parent}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page, pageSize=1000,
        ).execute()
        files += resp.get("files", [])
        page = resp.get("nextPageToken")
        if not page:
            break
    return files


def main():
    ap = argparse.ArgumentParser(description="Reset the Drive folder + DB + local state.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    ap.add_argument("--permanent", action="store_true", help="Hard-delete instead of trashing.")
    ap.add_argument("--db-path", default=os.environ.get("DB_PATH", "./manifest.sqlite"),
                    help="Local SQLite manifest to delete.")
    ap.add_argument("--keep-local", action="store_true", help="Do not touch local files.")
    args = ap.parse_args()

    uploader = get_drive_client()
    if uploader is None:
        print("ERROR: no Drive client (check credentials.json/token.json + DRIVE_PARENT_FOLDER_ID).",
              file=sys.stderr)
        sys.exit(1)

    service, parent = uploader.service, uploader.parent_folder_id
    children = _list_children(service, parent)

    mode = "PERMANENTLY DELETE" if args.permanent else "trash"
    print(f"\nDrive folder {parent} — will {mode} {len(children)} item(s):")
    for f in children:
        kind = "dir " if f["mimeType"] == "application/vnd.google-apps.folder" else "file"
        print(f"  [{kind}] {f['name']}")

    local_targets = []
    if not args.keep_local:
        for p in (args.db_path, args.db_path + "-wal", args.db_path + "-shm"):
            if os.path.exists(p):
                local_targets.append(p)
        if os.path.isdir(config.LOCAL_OUTPUT_DIR):
            local_targets.append(config.LOCAL_OUTPUT_DIR + "/  (work dirs)")
    if local_targets:
        print("\nLocal items to remove:")
        for p in local_targets:
            print(f"  {p}")

    if not children and not local_targets:
        print("\nNothing to reset.")
        return

    if not args.yes:
        ans = input(f"\nProceed to {mode} the above? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # Drive
    for f in children:
        try:
            if args.permanent:
                service.files().delete(fileId=f["id"]).execute()
            else:
                service.files().update(fileId=f["id"], body={"trashed": True}).execute()
            print(f"  {mode}: {f['name']}")
        except Exception as e:
            print(f"  FAILED {f['name']}: {e}")

    # Local
    if not args.keep_local:
        for p in (args.db_path, args.db_path + "-wal", args.db_path + "-shm"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        if os.path.isdir(config.LOCAL_OUTPUT_DIR):
            shutil.rmtree(config.LOCAL_OUTPUT_DIR, ignore_errors=True)
        print("Local manifest + work dirs removed.")

    print("\nReset complete.")


if __name__ == "__main__":
    main()

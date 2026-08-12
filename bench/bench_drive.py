#!/usr/bin/env python3
"""
bench_drive.py — create / trash an isolated Drive folder for the A/B benchmark.

The benchmark must run with Drive ON (inter-stage artifacts transfer only via Drive), but we do NOT
want to pollute the user's real catalog. So we create a throwaway folder under the real parent and
point DRIVE_PARENT_FOLDER_ID at it for both arms; the real manifest.sqlite and videos table are
never touched. Trash the folder when done.

    python bench/bench_drive.py create          # prints + writes bench/bench_folder_id.txt
    python bench/bench_drive.py trash <id>      # trash a bench folder
"""

import os
import sys
import time

_CF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CF_ROOT not in sys.path:
    sys.path.insert(0, _CF_ROOT)

from shared.drive import get_drive_client


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("create", "trash"):
        print(__doc__)
        sys.exit(1)
    client = get_drive_client()           # uses the REAL parent from config/.env
    if client is None:
        print("ERROR: no Drive client (check credentials.json/token.json).", file=sys.stderr)
        sys.exit(2)

    if sys.argv[1] == "create":
        name = f"CF_bench_{time.strftime('%Y%m%d_%H%M%S')}"
        fid = client.create_folder(name)   # under config.DRIVE_PARENT_FOLDER_ID
        out = os.path.join(_CF_ROOT, "bench", "bench_folder_id.txt")
        with open(out, "w") as f:
            f.write(fid + "\n")
        print(fid)
    else:
        fid = sys.argv[2]
        client.service.files().update(fileId=fid, body={"trashed": True}).execute()
        print(f"trashed {fid}")


if __name__ == "__main__":
    main()

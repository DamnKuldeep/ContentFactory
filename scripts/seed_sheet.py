#!/usr/bin/env python3
"""
seed_sheet.py — one-time seed of the review Google Sheet from the pipeline's `videos` catalog.

For each finished video in a batch: make the Drive file anyone-with-link (so the web app can embed
it), then upsert a row {batch_id, story_num, title, drive_file_id, video_url(/preview), status=pending}.
Idempotent — re-running updates existing rows (won't clobber status/reviewer of already-reviewed rows;
it only fills identity columns).

    python scripts/seed_sheet.py --batch 20260628_153812
    python scripts/seed_sheet.py --batch 20260628_153812 --no-public   # skip make_public

Requires the OAuth token to include the Sheets scope (run scripts/drive_auth.py once after the scope
was added). Set SHEET_ID in .env, or omit it and this creates the sheet + prints the id.
"""

import argparse
import os
import sys

_CF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CF not in sys.path:
    sys.path.insert(0, _CF)

from shared.manifest import Manifest
from shared.drive import get_drive_client
from social import sheet as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--db-path", default=os.environ.get("DB_PATH", "./manifest.sqlite"))
    ap.add_argument("--no-public", action="store_true", help="Don't make the finals anyone-with-link.")
    args = ap.parse_args()

    vids = Manifest(args.db_path).list_videos(args.batch)
    if not vids:
        print(f"No videos for batch {args.batch} in {args.db_path}"); return
    uploader = None if args.no_public else get_drive_client()

    ws = S.open_ws(create_if_missing=True)
    existing = {(r["batch_id"], str(r["story_num"])) for r in S.all_rows(ws)}
    seeded = 0
    for v in vids:
        fid = v["drive_file_id"]
        if uploader and fid and not v.get("public"):
            try:
                uploader.make_public(fid)
            except Exception as e:
                print(f"  make_public failed for story {v['story_num']}: {e}")
        video_url = f"https://drive.google.com/file/d/{fid}/preview" if fid else (v.get("link") or "")
        key = (str(v["batch_id"]), str(v["story_num"]))
        if key in existing:
            # only refresh identity columns; preserve review/upload state
            S.upsert({"batch_id": v["batch_id"], "story_num": v["story_num"],
                      "title": v.get("title", ""), "drive_file_id": fid, "video_url": video_url}, ws)
        else:
            S.upsert({"batch_id": v["batch_id"], "story_num": v["story_num"],
                      "title": v.get("title", ""), "drive_file_id": fid, "video_url": video_url,
                      "status": "pending"}, ws)
            seeded += 1
    print(f"Seeded {seeded} new rows ({len(vids)} videos in batch {args.batch}). "
          f"Counts: {S.queue_counts(ws)}")


if __name__ == "__main__":
    main()

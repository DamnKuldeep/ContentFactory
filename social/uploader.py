#!/usr/bin/env python3
"""
Throttled uploader — drains the approved (in_queue) videos to Instagram + YouTube at most
ONE per platform per UPLOAD_INTERVAL_HOURS (default 4). Run on the LAPTOP (residential IP — safer
for instagrapi) when it's on; it's idempotent and resumes from the Sheet, so intermittent running
is fine for a slow trickle.

    # safe end-to-end rehearsal (no real posting, no LLM/download — exercises queue + throttle + Sheet):
    python social/uploader.py --once --dry-run
    # live (after creds): loop, checking every 30 min, posting when the per-platform gate is open:
    python social/uploader.py --loop --interval-hours 4

Per platform each cycle: if (now - last_post) >= interval and the queue has a not-yet-posted video,
take the oldest -> (generate description+hashtags if missing) -> download mp4 -> post -> mark Sheet.
status flips to 'uploaded' once both platforms are posted.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

_CF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CF not in sys.path:
    sys.path.insert(0, _CF)

from social import sheet as S

PLATFORMS = {"ig": "Instagram", "yt": "YouTube"}


def _parse_ts(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _last_posted(rows, platform):
    ts = [_parse_ts(r.get(f"{platform}_posted_at", "")) for r in rows]
    ts = [t for t in ts if t]
    return max(ts) if ts else None


def _gate_open(rows, platform, interval_h):
    last = _last_posted(rows, platform)
    if last is None:
        return True
    return datetime.now() - last >= timedelta(hours=interval_h)


def _ensure_metadata(row, manifest, uploader, dry_run):
    """Return (caption_text_for_ig, yt_title, yt_desc, tags). Generates + persists if missing."""
    desc, tags_str = (row.get("description") or "").strip(), (row.get("hashtags") or "").strip()
    if dry_run and not desc:
        # offline rehearsal: placeholder, no LLM
        desc = "[dry-run description]"
        tags = ["dryrun", "test"]
        return (desc + "\n\n#dryrun #test", row.get("title") or "Short", desc, tags)
    from social import metadata as M
    if desc and tags_str:
        tags = [t.lstrip("#") for t in tags_str.split()]
        ig = desc + "\n\n" + " ".join("#" + t for t in tags)
        return (ig, row.get("title") or "Short", desc, tags)
    meta = M.generate_for_video(row["batch_id"], row["story_num"], manifest, uploader)
    S.set_metadata(row["batch_id"], row["story_num"], meta.description, " ".join(meta.hashtags))
    return (M.ig_caption(meta), meta.title, M.yt_description(meta), meta.hashtags)


def _post(platform, video_path, ig_caption, yt_title, yt_desc, tags, dry_run):
    if dry_run:
        return f"dryrun://{platform}/{int(time.time())}"
    if platform == "ig":
        from social.ig_adapter import IGAdapter
        return IGAdapter().upload_reel(video_path, ig_caption)
    from social.yt_adapter import YTAdapter
    return YTAdapter().upload_short(video_path, yt_title, yt_desc, tags)


def process_one(platform, manifest, uploader, dry_run) -> bool:
    ws = S.open_ws()
    row = S.next_for_platform(platform, ws)
    if not row:
        return False
    b, n = row["batch_id"], row["story_num"]
    print(f"[uploader] {PLATFORMS[platform]}: posting {b}/story {n} — {row.get('title')}")
    ig_cap, yt_title, yt_desc, tags = _ensure_metadata(row, manifest, uploader, dry_run)
    video_path = None
    try:
        if not dry_run:
            video_path = f"/tmp/up_{b}_{n}.mp4"
            if not uploader.download_file(row["drive_file_id"], video_path):
                raise RuntimeError("video download failed")
        url = _post(platform, video_path, ig_cap, yt_title, yt_desc, tags, dry_run)
        S.mark_posted(b, n, platform, url, ws)
        print(f"[uploader] {PLATFORMS[platform]}: ✓ {url}")
        return True
    except Exception as e:
        S.mark_failed(b, n, platform, str(e), ws)
        print(f"[uploader] {PLATFORMS[platform]}: ✗ {e}")
        return False
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


def cycle(platforms, interval_h, manifest, uploader, dry_run):
    ws = S.open_ws()
    rows = S.all_rows(ws)
    for p in platforms:
        if not _gate_open(rows, p, interval_h):
            last = _last_posted(rows, p)
            print(f"[uploader] {PLATFORMS[p]}: throttled (last {last}, interval {interval_h}h)")
            continue
        process_one(p, manifest, uploader, dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="One cycle then exit (default if not --loop).")
    ap.add_argument("--loop", action="store_true", help="Run forever, checking every --poll-minutes.")
    ap.add_argument("--poll-minutes", type=int, default=30)
    ap.add_argument("--interval-hours", type=float, default=float(os.environ.get("UPLOAD_INTERVAL_HOURS", "4")))
    ap.add_argument("--platforms", default="ig,yt")
    ap.add_argument("--dry-run", action="store_true", help="No real posting/LLM/download — exercise queue+throttle+Sheet.")
    ap.add_argument("--db-path", default=os.environ.get("DB_PATH", "./manifest.sqlite"))
    args = ap.parse_args()

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip() in PLATFORMS]
    manifest = uploader = None
    if not args.dry_run:
        from shared.manifest import Manifest
        from shared.drive import get_drive_client
        manifest = Manifest(args.db_path)
        uploader = get_drive_client()

    if args.loop:
        print(f"[uploader] loop: platforms={platforms} interval={args.interval_hours}h poll={args.poll_minutes}m")
        while True:
            cycle(platforms, args.interval_hours, manifest, uploader, args.dry_run)
            time.sleep(args.poll_minutes * 60)
    else:
        cycle(platforms, args.interval_hours, manifest, uploader, args.dry_run)


if __name__ == "__main__":
    main()

"""
Content Factory — Multi-Stage CLI Entry Point

Single stage:
  python run.py --stage 1 --count 100 --workers 5
Multiple stages on one machine (sequential), with Drive-hosted shared DB:
  python run.py --stages 2 3 4 --batch 20260619_120000 --drive-db

Distributed topology:
  machine 1: python run.py --stage 1 --count N --drive-db       (prints batch id)
  machine 2: python run.py --stages 2 3 4 --batch <id> --drive-db
  machine 3: python run.py --stage 5 --batch <id> --drive-db
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time

from shared import config
from shared.config import DEFAULT_WORKERS
from shared.drive import get_drive_client
from shared.drive_db import DriveSyncManifest
from shared.manifest import Manifest
from shared.progress import monitor_progress

logger = logging.getLogger("contentfactory.run")

STAGE_NAMES = {
    1: "stage_01_story",
    2: "stage_02_narration",
    3: "stage_03_images",
    4: "stage_04_music",
    5: "stage_05_compose",
}


def _spawn(stage_int, db_path, batch_id, workers):
    if stage_int == 1:
        from stages.stage_01_story.run import spawn_stage_01_workers
        return spawn_stage_01_workers(db_path, batch_id, count=workers)
    if stage_int == 2:
        from stages.stage_02_narration.run import spawn_stage_02_workers
        return spawn_stage_02_workers(db_path, batch_id, count=workers)
    if stage_int == 3:
        from stages.stage_03_images.run import spawn_stage_03_workers
        return spawn_stage_03_workers(db_path, batch_id, count=workers)
    if stage_int == 4:
        from stages.stage_04_music.run import spawn_stage_04_workers
        return spawn_stage_04_workers(db_path, batch_id, count=workers)
    if stage_int == 5:
        from stages.stage_05_compose.run import spawn_stage_05_workers
        return spawn_stage_05_workers(db_path, batch_id, count=workers)
    return []


def _run_stage(stage_int, db_path, batch_id, args, manifest):
    stage_name = STAGE_NAMES[stage_int]
    print(f"\n--- Batch {batch_id} | Stage {stage_int} ({stage_name}) ---")

    if args.retry_failed:
        print(f"Reset {manifest.retry_failed(batch_id, stage_name)} FAILED jobs to PENDING.")

    if stage_int == 1:
        manifest.create_batch(batch_id, stage_name, list(range(1, args.count + 1)))

    processes = _spawn(stage_int, db_path, batch_id, args.workers)
    if not processes:
        if not manifest.has_pending_work(batch_id, stage_name):
            print(f"Stage {stage_name}: no pending/upstream work — skipping.")
            return
    print(f"Started {len(processes)} worker(s). Waiting...\n")

    try:
        total = manifest.get_stats(batch_id, stage_name).get("total", 0)
        monitor_progress(db_path, batch_id, stage_name, total)
    except KeyboardInterrupt:
        print("\nMonitor interrupted; waiting for workers to finish...")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nTerminating workers...")
        for p in processes:
            p.terminate()


def _videos_action(args):
    """Standalone: print the videos catalog + export CSV/JSON to Drive (+ optional --share-public)."""
    import csv
    import json as _json

    drive_db = args.drive_db or args.resume or config.DRIVE_DB
    uploader = get_drive_client() if (drive_db or args.share_public) else None
    manifest = (DriveSyncManifest(args.db_path, uploader)
                if (drive_db and uploader) else Manifest(args.db_path))
    if drive_db and uploader:
        manifest.sync_pull()

    if args.share_public:
        if not uploader:
            print("ERROR: --share-public needs a Drive client.", file=sys.stderr)
            sys.exit(1)
        for v in manifest.list_videos(args.batch):
            if v.get("drive_file_id") and not v.get("public"):
                link = uploader.make_public(v["drive_file_id"])
                manifest.record_video(v["batch_id"], v["story_num"], v.get("title"), link,
                                      v["drive_file_id"], v.get("folder_id"), public=1)
        if isinstance(manifest, DriveSyncManifest):
            manifest.sync_push()   # videos-only push

    videos = manifest.list_videos(args.batch)
    print(f"\nVideos catalog ({len(videos)} entr{'y' if len(videos)==1 else 'ies'}):")
    for v in videos:
        flag = "public " if v.get("public") else "private"
        print(f"  [{v['batch_id']} #{int(v['story_num']):>3}] {flag} {v.get('title','')}")
        print(f"        {v.get('link','')}")

    base = os.path.dirname(os.path.abspath(args.db_path)) or "."
    csv_path, json_path = os.path.join(base, "videos.csv"), os.path.join(base, "videos.json")
    fields = ["batch_id", "story_num", "title", "link", "drive_file_id", "folder_id", "public", "created_at"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in videos:
            w.writerow({k: v.get(k, "") for k in fields})
    with open(json_path, "w") as f:
        _json.dump(videos, f, indent=2)
    print(f"\nExported {csv_path} and {json_path}")
    if uploader:
        uploader.replace_or_upload(csv_path, uploader.parent_folder_id, "videos.csv")
        uploader.replace_or_upload(json_path, uploader.parent_folder_id, "videos.json")
        print(f"Uploaded videos.csv + videos.json to Drive folder {uploader.parent_folder_id}")


def main():
    parser = argparse.ArgumentParser(description="Content Factory Production Pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a single pipeline stage.")
    parser.add_argument("--stages", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
                        help="Run several stages on this machine (e.g. --stages 2 3 4); auto-ordered.")
    parser.add_argument("--count", type=int, default=1, help="Stories to generate (Stage 1).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers.")
    parser.add_argument("--batch", type=str, default=None, help="Batch ID (required unless stage 1).")
    parser.add_argument("--db-path", type=str, default=os.environ.get("DB_PATH", "./manifest.sqlite"),
                        help="Local SQLite path (working copy when --drive-db).")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reset FAILED jobs for each run stage back to PENDING.")
    parser.add_argument("--drive-db", action="store_true",
                        help="Use the Drive-hosted shared DB (else config.DRIVE_DB / local).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a batch from the Drive DB on this machine (implies --drive-db).")
    parser.add_argument("--list-videos", action="store_true",
                        help="Print the videos catalog (title+link) and export videos.csv/json to Drive.")
    parser.add_argument("--share-public", action="store_true",
                        help="With --list-videos: set each final video to anyone-with-link and update the catalog.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

    # Standalone catalog action — no stage run.
    if args.list_videos or args.share_public:
        _videos_action(args)
        return

    if args.resume:
        args.drive_db = True   # resume is inherently Drive-backed

    # Resolve ordered stage list (always ascending = dependency order).
    if args.stages:
        stages_to_run = sorted(set(args.stages))
    elif args.stage is not None:
        stages_to_run = [args.stage]
    else:
        print("ERROR: pass --stage N, --stages N [N ...], or --list-videos.", file=sys.stderr)
        sys.exit(1)

    # Batch id
    if 1 in stages_to_run and not args.resume:
        batch_id = args.batch or time.strftime("%Y%m%d_%H%M%S")
    else:
        if not args.batch:
            print("ERROR: --batch is required for --resume or any stage > 1.", file=sys.stderr)
            sys.exit(1)
        batch_id = args.batch

    drive_db = args.drive_db or config.DRIVE_DB
    uploader = get_drive_client() if drive_db else None
    if drive_db and uploader is None:
        if args.resume:
            print("ERROR: --resume needs a Drive client (credentials.json/token.json).", file=sys.stderr)
            sys.exit(1)
        print("WARNING: --drive-db requested but no Drive client — using LOCAL manifest only.")
        drive_db = False

    manifest = DriveSyncManifest(args.db_path, uploader) if (drive_db and uploader) else Manifest(args.db_path)

    mp.set_start_method("spawn", force=True)

    if drive_db:
        print(f"[drive-db] pulling shared DB from Drive folder {uploader.parent_folder_id} ...")
        manifest.sync_pull()

    print(f"Reset {manifest.reset_stale(timeout_minutes=60)} stale RUNNING jobs.")

    if args.resume:
        summary = ", ".join(
            f"{STAGE_NAMES[s]}: {manifest.get_stats(batch_id, STAGE_NAMES[s]).get('PENDING', 0)} pending"
            for s in stages_to_run)
        print(f"[resume] batch {batch_id} → {summary}")

    if 1 in stages_to_run and not args.batch and not args.resume:
        print(f"\n>>> New batch id: {batch_id}  (use --batch {batch_id} on the other machines)\n")

    for stage_int in stages_to_run:
        _run_stage(stage_int, args.db_path, batch_id, args, manifest)
        if drive_db:
            print(f"[drive-db] pushing stage {STAGE_NAMES[stage_int]} rows to Drive ...")
            manifest.sync_push({STAGE_NAMES[stage_int]})

    print("\nAll requested stages finished. Run complete!")


if __name__ == "__main__":
    main()

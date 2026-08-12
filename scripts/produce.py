#!/usr/bin/env python3
"""
produce.py — kick off a bulk run of N videos (run on the CPU/story box).

Seeds N videos for ALL stages in one shot (no per-stage backlog ceremony), then generates the
stories (stage 1). Afterwards run `worker.py <role>` on the appropriate machines to drain the rest.

    python scripts/produce.py --count 150            # Drive-synced (DRIVE_DB) bulk batch
    python scripts/produce.py --count 5 --workers 3  # local test

Then:
    GPU box : python scripts/worker.py narration ; worker.py images ; worker.py music
    CPU box : python scripts/worker.py compose
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

_CF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CF_ROOT not in sys.path:
    sys.path.insert(0, _CF_ROOT)

from shared import config
from shared.drive import get_drive_client
from shared.drive_db import DriveSyncManifest
from shared.manifest import Manifest
from shared.progress import monitor_progress


def main():
    ap = argparse.ArgumentParser(description="Seed + generate N stories for a bulk video run.")
    ap.add_argument("--count", type=int, required=True, help="Number of videos to produce.")
    ap.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS, help="Story workers.")
    ap.add_argument("--db-path", type=str, default=os.environ.get("DB_PATH", "./manifest.sqlite"))
    ap.add_argument("--no-drive", action="store_true", help="Local-only (skip Drive DB).")
    args = ap.parse_args()

    drive_db = (not args.no_drive) and config.DRIVE_DB
    uploader = get_drive_client() if drive_db else None
    if drive_db and uploader is None:
        print("WARNING: DRIVE_DB set but no Drive client — using local manifest only.")
        drive_db = False
    manifest = DriveSyncManifest(args.db_path, uploader) if (drive_db and uploader) else Manifest(args.db_path)

    mp.set_start_method("spawn", force=True)
    if drive_db:
        manifest.sync_pull()

    batch_id = time.strftime("%Y%m%d_%H%M%S")
    seeded = manifest.seed_all_stages(batch_id, list(range(1, args.count + 1)))
    print(f"Seeded {seeded} jobs for {args.count} videos across all stages (batch {batch_id}).")

    # Generate the stories (stage 1) on this box.
    from stages.stage_01_story.run import spawn_stage_01_workers
    procs = spawn_stage_01_workers(args.db_path, batch_id, count=args.workers)
    print(f"Started {len(procs)} story worker(s)...")
    try:
        total = manifest.get_stats(batch_id, "stage_01_story").get("total", 0)
        monitor_progress(args.db_path, batch_id, "stage_01_story", total)
    except KeyboardInterrupt:
        pass
    for p in procs:
        p.join()

    if drive_db:
        # Push the FULL seeded DAG: stage_01 COMPLETE *and* stages 2-5 PENDING. Pushing only
        # stage_01 would strip the seeded downstream rows on each worker's first sync_pull, so
        # `worker <role>` boxes would find no ready jobs and exit idle.
        from shared.manifest import STAGE_SEQUENCE
        manifest.sync_push(set(STAGE_SEQUENCE))

    done = len(manifest.get_completed_story_nums(batch_id, "stage_01_story"))
    print(f"\nStories ready: {done}/{args.count}  (batch {batch_id})")
    print("Next: run `python scripts/worker.py narration|images|music` on the GPU box, "
          "`worker.py compose` on the CPU box.")


if __name__ == "__main__":
    main()

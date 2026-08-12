#!/usr/bin/env python3
"""
worker.py — drain all ready videos for one role on this machine (bulk-friendly, no batch IDs).

A worker loads its model once and processes every video whose upstream stage is COMPLETE, then
exits (or keeps waiting with --watch). Run the role that matches the machine:

    GPU box : python scripts/worker.py narration
              python scripts/worker.py images
              python scripts/worker.py music
    CPU box : python scripts/worker.py compose      # (story is produced by produce.py)

Roles → stages: narration→2, images→3, music→4, compose→5, story→1.
Use one venv per GPU role (Fish/FLUX/ACE deps conflict) — see RUNME.md.
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback

_CF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CF_ROOT not in sys.path:
    sys.path.insert(0, _CF_ROOT)

from shared import config
from shared.drive import get_drive_client, DriveUploader
from shared.drive_db import DriveSyncManifest
from shared.manifest import Manifest
from shared.utils import unique_output_path

ROLE_STAGE = {
    "story": "stage_01_story", "narration": "stage_02_narration", "images": "stage_03_images",
    "music": "stage_04_music", "compose": "stage_05_compose",
}
PROCESS_FN = {
    "stage_01_story":     ("stages.stage_01_story.run", "process_stage_01"),
    "stage_02_narration": ("stages.stage_02_narration.run", "process_stage_02"),
    "stage_03_images":    ("stages.stage_03_images.run", "process_stage_03"),
    "stage_04_music":     ("stages.stage_04_music.run", "process_stage_04"),
    "stage_05_compose":   ("stages.stage_05_compose.run", "process_stage_05"),
}


def _handle_one(job, stage, manifest, uploader, out_dir, worker_id, process_fn):
    """Process one job + save/upload its stage JSON + mark COMPLETE (mirrors run_worker_loop)."""
    from shared.timing import step
    batch_id, story_num, job_id = job["batch_id"], job["story_num"], job["id"]
    with step(stage, "process_total", kind="total", story=story_num):
        final_output = process_fn(job, worker_id, uploader)
    prefix = f"ch_{batch_id}_{story_num}_{stage}"
    local_path = unique_output_path(out_dir, final_output, prefix=prefix)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    if uploader:
        folder = uploader.get_job_folder(batch_id, story_num, stage)   # → video_<n>/metadata/
        manifest.mark_upload_pending(job_id, local_path)
        file_id = uploader.upload_and_verify(local_path, folder)
        if file_id:
            manifest.upload_complete(job_id, folder, file_id)
            try:
                os.remove(local_path)
            except OSError:
                pass
        else:
            raise RuntimeError("stage JSON upload failed")
    else:
        manifest.complete_job(job_id, local_path=local_path, meta=final_output.get("meta"))


def main():
    import logging
    ap = argparse.ArgumentParser(description="Drain all ready videos for one role.")
    ap.add_argument("role", choices=list(ROLE_STAGE))
    ap.add_argument("--db-path", default=os.environ.get("DB_PATH", "./manifest.sqlite"))
    ap.add_argument("--no-drive", action="store_true", help="Local-only (skip Drive DB).")
    ap.add_argument("--watch", action="store_true", help="Wait for upstream instead of exiting when idle.")
    ap.add_argument("--poll", type=int, default=30, help="Seconds between retries when --watch.")
    ap.add_argument("--max-idle", type=int, default=5, help="Empty polls before giving up (--watch).")
    ap.add_argument("--max", type=int, default=None, help="Process at most N jobs then exit (e.g. 1 for a smoke test).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    stage = ROLE_STAGE[args.role]
    os.environ["DB_PATH"] = args.db_path
    out_dir = config.LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if stage == "stage_04_music":   # brief uses OpenRouter retries
        try:
            from shared.llm import init_client
            init_client(config.OPENROUTER_API_KEY)
        except Exception:
            pass

    drive_db = (not args.no_drive) and config.DRIVE_DB
    uploader: DriveUploader = get_drive_client() if drive_db else None
    if drive_db and uploader is None:
        print("WARNING: DRIVE_DB set but no Drive client — local manifest only.")
        drive_db = False
    manifest = DriveSyncManifest(args.db_path, uploader) if (drive_db and uploader) else Manifest(args.db_path)

    mod, fn = PROCESS_FN[stage]
    process_fn = getattr(importlib.import_module(mod), fn)
    print(f"[worker {args.role}] stage={stage} drive_db={drive_db} — draining ready videos...")

    done = idle = 0
    while True:
        if drive_db:
            manifest.sync_pull()
        manifest.reset_stale(timeout_minutes=60)
        job = manifest.claim_ready_job(stage)
        if job:
            idle = 0
            sn = job["story_num"]
            try:
                _handle_one(job, stage, manifest, uploader, out_dir, f"{args.role}-1", process_fn)
                done += 1
                print(f"[worker {args.role}] ✓ story {sn} ({job['batch_id']})  [{done} done]")
            except Exception as e:
                manifest.fail_job(job["id"], traceback.format_exc())
                print(f"[worker {args.role}] ✗ story {sn}: {e}")
            if drive_db:
                manifest.sync_push({stage})
            if args.max and done >= args.max:
                print(f"[worker {args.role}] reached --max {args.max} — exiting.")
                break
            continue
        # idle
        if not args.watch:
            break
        idle += 1
        if idle >= args.max_idle:
            print(f"[worker {args.role}] no work after {idle} polls — exiting.")
            break
        print(f"[worker {args.role}] idle; re-checking in {args.poll}s ({idle}/{args.max_idle})...")
        time.sleep(args.poll)

    print(f"[worker {args.role}] finished — {done} video(s) processed.")


if __name__ == "__main__":
    main()

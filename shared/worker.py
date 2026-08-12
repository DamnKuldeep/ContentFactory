"""
Content Factory — Shared worker logic for multi-process stage execution.
"""

import json
import logging
import traceback
import time
from typing import Callable, Optional

from shared.drive import DriveUploader
from shared.manifest import Manifest
from shared.utils import unique_output_path
from shared.timing import step

logger = logging.getLogger("contentfactory.worker")


def run_worker_loop(
    worker_id: str,
    manifest: Manifest,
    batch_id: str,
    stage: str,
    out_dir: str,
    process_fn: Callable[[dict, str, Optional[DriveUploader]], dict],
    uploader: Optional[DriveUploader] = None,
    poll_interval: int = 5
):
    """
    Run the worker loop for a specific stage.
    process_fn takes (job_dict, worker_id) and returns a meta dict (the final output) to be saved as JSON.
    """
    logger.info("Worker %s ready and looping for batch %s, stage %s...", worker_id, batch_id, stage)
    while True:
        job = manifest.claim_job(batch_id, stage)
        if not job:
            logger.info("Worker %s found no pending jobs for stage %s. Exiting.", worker_id, stage)
            break

        job_id = job["id"]
        story_num = job["story_num"]
        logger.info("Worker %s claimed job %d (Story %d, Stage %s)", worker_id, job_id, story_num, stage)

        try:
            # 1. Execute stage-specific logic (timed end-to-end incl. I/O when PROFILE_LOG set)
            with step(stage, "process_total", kind="total", story=story_num):
                final_output = process_fn(job, worker_id, uploader)

            # 2. Save locally
            prefix = f"ch_{batch_id}_{story_num}_{stage}"
            local_path = unique_output_path(out_dir, final_output, prefix=prefix)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            logger.info("Worker %s saved job %d to %s", worker_id, job_id, local_path)

            # 3. Upload to Drive (if configured)
            if uploader:
                target_folder_id = uploader.get_job_folder(batch_id, story_num, stage)
                manifest.mark_upload_pending(job_id, local_path)
                file_id = uploader.upload_and_verify(local_path, target_folder_id)
                if file_id:
                    manifest.upload_complete(job_id, target_folder_id, file_id)
                    # Production: Delete local copy after successful upload
                    try:
                        import os
                        os.remove(local_path)
                        logger.info("Worker %s deleted local artifact %s after upload.", worker_id, local_path)
                    except Exception as e:
                        logger.warning("Worker %s failed to delete %s: %s", worker_id, local_path, e)
                else:
                    logger.warning("Upload failed for job %d. Kept as UPLOAD_PENDING.", job_id)
            else:
                # If no uploader, just complete it
                manifest.complete_job(job_id, local_path=local_path, meta=final_output.get("meta"))

        except Exception as e:
            err = traceback.format_exc()
            logger.error("Worker %s failed job %d: %s", worker_id, job_id, err)
            manifest.fail_job(job_id, err)
            # small sleep on failure
            time.sleep(poll_interval)

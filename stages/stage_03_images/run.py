"""
Stage 3: Image generation entry point.
"""

import json
import logging
import multiprocessing
import os
from typing import Optional

from shared.config import DRIVE_PARENT_FOLDER_ID, LOCAL_OUTPUT_DIR
from shared.drive import get_drive_client, DriveUploader
from shared.manifest import Manifest
from shared.worker import run_worker_loop

from .generate import run_all

logger = logging.getLogger("contentfactory.stage_03")


ARTIFACT_CONTRACT = {
    "version": "1.0",
    "required": ["story.json"],
    "optional": [],
    "produces": ["stage_03.json"] # Images are uploaded incrementally
}

def verify_contract_inputs(work_dir: str):
    for req in ARTIFACT_CONTRACT["required"]:
        path = os.path.join(work_dir, req)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input artifact: {req}")

def process_stage_03(job: dict, worker_id: str, uploader: Optional[DriveUploader]) -> dict:
    """Execute Stage 3."""
    batch_id = job["batch_id"]
    story_num = job["story_num"]
    
    manifest = Manifest(os.environ.get("DB_PATH", "manifest.sqlite"))
    prev_job = manifest.get_completed_job(batch_id, "stage_02_narration", story_num)
    
    if not prev_job:
        raise ValueError(f"Could not find completed stage_02_narration job for story {story_num}")

    work_dir = os.path.join(LOCAL_OUTPUT_DIR, f"work_{batch_id}_{story_num}_s3")
    os.makedirs(work_dir, exist_ok=True)

    # We need story prompts. We can get them from stage_01 output.
    s1_job = manifest.get_completed_job(batch_id, "stage_01_story", story_num)
    if not s1_job:
        raise ValueError(f"Could not find completed stage_01_story job for story {story_num}")

    prev_json_path = os.path.join(work_dir, "story.json")
    file_id = s1_job.get("drive_file_id")
    if not file_id or not uploader:
        raise ValueError(f"No drive info available for stage 1 output of story {story_num}")
        
    logger.info("Worker %s downloading Stage 1 output (story.json) from Drive...", worker_id)
    if not uploader.download_file(file_id, prev_json_path):
        raise RuntimeError("Failed to download Stage 1 artifact")

    # Verify contracts
    verify_contract_inputs(work_dir)

    # Read Stage 1 output
    with open(prev_json_path, "r", encoding="utf-8") as f:
        story_data = json.load(f)

    # generate.run_all derives the per-video Drive folder from job_data's batch_id/story_num.
    # The stage_01 JSON carries neither, so inject them from the job — otherwise images land in
    # Batch_unknown/video_0000 instead of the correct per-video folder.
    story_data["batch_id"] = batch_id
    story_data["story_num"] = story_num

    # Run the Stage 3 pipeline (generates and uploads images)
    result = run_all(story_data, work_dir, uploader)
    
    # Clean up local input artifacts to enforce ephemeral local disk policy
    try:
        os.remove(prev_json_path)
    except Exception:
        pass

    return result


def populate_backlog(db_path: str, batch_id: str):
    """Find COMPLETE stage 2 jobs and insert them into stage 3 as PENDING."""
    manifest = Manifest(db_path)
    s2_completed = manifest.get_completed_story_nums(batch_id, "stage_02_narration")
    if not s2_completed:
        logger.warning("No complete stage 2 jobs found for batch %s.", batch_id)
        return
        
    created = manifest.create_batch(batch_id, "stage_03_images", s2_completed)
    if created > 0:
        logger.info("Populated %d new jobs for Stage 3.", created)


def run_stage_03_worker(
    worker_id: str,
    db_path: str,
    batch_id: str,
):
    """Entry point for a Stage 3 worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{worker_id}] [%(levelname)s] %(name)s: %(message)s"
    )

    os.environ["DB_PATH"] = db_path
    manifest = Manifest(db_path)
    uploader = get_drive_client()

    run_worker_loop(
        worker_id=worker_id,
        manifest=manifest,
        batch_id=batch_id,
        stage="stage_03_images",
        out_dir=LOCAL_OUTPUT_DIR,
        process_fn=process_stage_03,
        uploader=uploader
    )


def spawn_stage_03_workers(db_path: str, batch_id: str, count: int = 1):
    """Spawn multiple worker processes for Stage 3. Default to 1 because 5090 is single GPU."""
    populate_backlog(db_path, batch_id)
    processes = []
    for i in range(count):
        worker_id = f"stg3-{i+1}"
        p = multiprocessing.Process(
            target=run_stage_03_worker,
            args=(worker_id, db_path, batch_id)
        )
        p.start()
        processes.append(p)
    return processes

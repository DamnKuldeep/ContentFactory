"""
Stage 2: Narration entry point.
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

from .pipeline import run_all

logger = logging.getLogger("contentfactory.stage_02")


ARTIFACT_CONTRACT = {
    "version": "1.0",
    "required": ["stage_01.json"],
    "optional": [],
    "produces": ["narration.mp3", "stage_02.json"]
}

def verify_contract_inputs(work_dir: str):
    for req in ARTIFACT_CONTRACT["required"]:
        path = os.path.join(work_dir, req)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input artifact: {req}")

def process_stage_02(job: dict, worker_id: str, uploader: Optional[DriveUploader]) -> dict:
    """Execute Stage 2."""
    batch_id = job["batch_id"]
    story_num = job["story_num"]
    
    manifest = Manifest(os.environ.get("DB_PATH", "manifest.sqlite"))
    prev_job = manifest.get_completed_job(batch_id, "stage_01_story", story_num)
    
    if not prev_job:
        raise ValueError(f"Could not find completed stage_01_story job for story {story_num}")

    work_dir = os.path.join(LOCAL_OUTPUT_DIR, f"work_{batch_id}_{story_num}_s2")
    os.makedirs(work_dir, exist_ok=True)

    # Fetch previous stage artifact
    prev_json_path = os.path.join(work_dir, "stage_01.json")
    file_id = prev_job.get("drive_file_id")
    if not file_id or not uploader:
        raise ValueError(f"No drive info available for stage 1 output of story {story_num}")
        
    logger.info("Worker %s downloading Stage 1 output from Drive...", worker_id)
    if not uploader.download_file(file_id, prev_json_path):
        raise RuntimeError("Failed to download Stage 1 artifact")

    # Verify contracts
    verify_contract_inputs(work_dir)

    # Read Stage 1 output
    with open(prev_json_path, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    # Run the Stage 2 pipeline
    result = run_all(stage1_data, work_dir)
    
    audio_path = os.path.join(work_dir, "narration.mp3")
    
    if uploader and os.path.exists(audio_path):
        logger.info("Worker %s uploading generated audio %s", worker_id, audio_path)
        target_folder = uploader.get_video_subfolder(batch_id, story_num, "narration")
        audio_file_id = uploader.upload_and_verify(audio_path, target_folder)
        if audio_file_id:
            # Stages 4 & 5 read meta.stage_02.audio_drive_id — write it there.
            result["meta"].setdefault("stage_02", {})["audio_drive_id"] = audio_file_id
            result["meta"]["audio_drive_id"] = audio_file_id   # back-compat
            try:
                os.remove(audio_path)
            except Exception:
                pass
        else:
            logger.warning("Worker %s failed to upload audio %s", worker_id, audio_path)
            
    # Clean up local input artifacts to enforce ephemeral local disk policy
    try:
        os.remove(prev_json_path)
    except Exception:
        pass

    return result


def populate_backlog(db_path: str, batch_id: str):
    """Find COMPLETE stage 1 jobs and insert them into stage 2 as PENDING."""
    manifest = Manifest(db_path)
    s1_completed = manifest.get_completed_story_nums(batch_id, "stage_01_story")
    if not s1_completed:
        logger.warning("No complete stage 1 jobs found for batch %s.", batch_id)
        return
        
    created = manifest.create_batch(batch_id, "stage_02_narration", s1_completed)
    if created > 0:
        logger.info("Populated %d new jobs for Stage 2.", created)


def run_stage_02_worker(
    worker_id: str,
    db_path: str,
    batch_id: str,
):
    """Entry point for a Stage 2 worker process."""
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
        stage="stage_02_narration",
        out_dir=LOCAL_OUTPUT_DIR,
        process_fn=process_stage_02,
        uploader=uploader
    )


def spawn_stage_02_workers(db_path: str, batch_id: str, count: int = 2):
    """Spawn multiple worker processes for Stage 2."""
    populate_backlog(db_path, batch_id)
    processes = []
    for i in range(count):
        worker_id = f"stg2-{i+1}"
        p = multiprocessing.Process(
            target=run_stage_02_worker,
            args=(worker_id, db_path, batch_id)
        )
        p.start()
        processes.append(p)
    return processes

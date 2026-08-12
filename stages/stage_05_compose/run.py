"""
Stage 5: Composition entry point.
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
from shared.timing import step

from .compose import compose_video

_ST = "stage_05_compose"

logger = logging.getLogger("contentfactory.stage_05")


ARTIFACT_CONTRACT = {
    "version": "1.0",
    "required": ["narration.mp3", "music.mp3"], # stage_04 JSON already loaded into job_data; images are dynamic
    "optional": [],
    "produces": ["final.mp4", "stage_05.json"]
}

def verify_contract_inputs(work_dir: str):
    for req in ARTIFACT_CONTRACT["required"]:
        path = os.path.join(work_dir, req)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input artifact: {req}")

def process_stage_05(job: dict, worker_id: str, uploader: Optional[DriveUploader]) -> dict:
    """Execute Stage 5."""
    batch_id = job["batch_id"]
    story_num = job["story_num"]
    
    manifest = Manifest(os.environ.get("DB_PATH", "manifest.sqlite"))
    prev_job = manifest.get_completed_job(batch_id, "stage_04_music", story_num)
    
    if not prev_job:
        raise ValueError(f"Could not find completed stage_04_music job for story {story_num}")

    # Fetch previous stage artifact
    prev_json_path = prev_job.get("local_path")
    if not prev_json_path or not os.path.exists(prev_json_path):
        file_id = prev_job.get("drive_file_id")
        if not file_id or not uploader:
            raise ValueError(f"No local path and no drive info available for stage 4 output of story {story_num}")
            
        logger.info("Worker %s downloading Stage 4 output from Drive...", worker_id)
        download_path = os.path.join(LOCAL_OUTPUT_DIR, f"ch_{batch_id}_{story_num}_stage_04.json")
        if not uploader.download_file(file_id, download_path):
            raise RuntimeError("Failed to download Stage 4 artifact")
        prev_json_path = download_path

    with open(prev_json_path, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    # The 3→4→5 chain branches off stage_01, so narration data (alignment + audio_drive_id) that
    # stage_02 produced isn't in this JSON. Pull the stage_02 output and merge it in.
    if not job_data.get("alignment") or not job_data.get("meta", {}).get("stage_02", {}).get("audio_drive_id"):
        s2 = manifest.get_completed_job(batch_id, "stage_02_narration", story_num)
        if s2 and s2.get("drive_file_id") and uploader:
            s2_path = os.path.join(LOCAL_OUTPUT_DIR, f"ch_{batch_id}_{story_num}_stage_02.json")
            if uploader.download_file(s2["drive_file_id"], s2_path):
                with open(s2_path) as f:
                    s2_data = json.load(f)
                s2meta = s2_data.get("meta", {})
                job_data["alignment"] = s2_data.get("alignment", job_data.get("alignment"))
                dst = job_data.setdefault("meta", {}).setdefault("stage_02", {})
                dst.update(s2meta.get("stage_02", {}))
                dst["audio_drive_id"] = (dst.get("audio_drive_id")
                                         or s2meta.get("audio_drive_id"))   # back-compat location
                logger.info("Merged stage_02 narration metadata (alignment + audio_drive_id).")
                try:
                    os.remove(s2_path)
                except OSError:
                    pass

    # Required assets
    stage2 = job_data.get("meta", {}).get("stage_02", {})
    stage3 = job_data.get("meta", {}).get("stage_03", {})
    stage4 = job_data.get("meta", {}).get("stage_04", {})
    
    alignment = job_data.get("alignment", {})
    
    work_dir = os.path.join(LOCAL_OUTPUT_DIR, f"work_{batch_id}_{story_num}_s5")
    os.makedirs(work_dir, exist_ok=True)
    
    if not uploader:
        raise ValueError("DriveUploader required to download assets")

    images = stage3.get("images", [])
    with step(_ST, "assets_download", kind="io", images=len(images)):
        # Download Narration
        narration_drive_id = stage2.get("audio_drive_id")
        if not narration_drive_id:
            raise ValueError("No audio_drive_id found from stage 2")
        narration_path = os.path.join(work_dir, "narration.mp3")
        logger.info("Worker %s downloading narration...", worker_id)
        if not uploader.download_file(narration_drive_id, narration_path):
            raise RuntimeError("Failed to download narration from Drive")

        # Download Music
        music_drive_id = stage4.get("music_drive_id")
        if not music_drive_id:
            raise ValueError("No music_drive_id found from stage 4")
        music_path = os.path.join(work_dir, "music.mp3")
        logger.info("Worker %s downloading music...", worker_id)
        if not uploader.download_file(music_drive_id, music_path):
            raise RuntimeError("Failed to download music from Drive")

        # Download images SERIALLY. The Drive `service` (httplib2) is NOT thread-safe — sharing it
        # across a ThreadPoolExecutor (was max_workers=8) raced in OpenSSL and SIGSEGV'd the worker
        # mid-download on a flaky uplink (0-byte files, no traceback). Serial is crash-safe; the
        # download_file retries handle transient SSL/timeout. (To restore parallelism safely, build
        # a separate DriveUploader per thread.)
        if not images:
            raise ValueError("No images found from stage 3")
        ordered = sorted(images, key=lambda x: x.get("scene_id", 0))
        targets = [(img, os.path.join(work_dir, img["filename"])) for img in ordered]
        logger.info("Worker %s downloading %d images (serial)...", worker_id, len(targets))
        for img, p in targets:
            if not uploader.download_file(img["drive_file_id"], p):
                raise RuntimeError(f"Failed to download image {img['filename']} from Drive")
        image_paths = [p for _, p in targets]

    out_video_path = os.path.join(work_dir, "final.mp4")

    # Verify contracts
    verify_contract_inputs(work_dir)

    # 1. Compose
    with step(_ST, "compose_video", kind="cpu"):
        compose_video(job_data, narration_path, music_path, image_paths, out_video_path, work_dir)
    
    # Clean up local input artifacts to enforce ephemeral local disk policy
    try:
        os.remove(prev_json_path)
        os.remove(narration_path)
        os.remove(music_path)
        for p in image_paths:
            os.remove(p)
    except Exception:
        pass
    
    # 2. Upload
    drive_file_id = None
    if uploader and os.path.exists(out_video_path):
        logger.info("Worker %s uploading final video %s", worker_id, out_video_path)
        target_folder = uploader.get_video_subfolder(batch_id, story_num, "final")

        drive_file_id = uploader.upload_and_verify(out_video_path, target_folder)
        if drive_file_id:
            try:
                os.remove(out_video_path)
            except Exception:
                pass
        else:
            logger.warning("Worker %s failed to upload video %s", worker_id, out_video_path)

    # 3. Update data + record the finished video in the catalog (title + link)
    video_link = uploader.view_link(drive_file_id) if (uploader and drive_file_id) else None
    job_data["meta"]["stage_05"] = {
        "video_drive_id": drive_file_id,
        "video_link": video_link,
    }

    if drive_file_id:
        meta = job_data.get("meta", {})
        title = (
            (meta.get("creative_direction") or {}).get("title")
            or (meta.get("premise") or {}).get("logline")
            or f"video_{story_num:04d}"
        )
        folder_id = None
        try:
            folder_id = uploader.ensure_video_folders(batch_id, story_num)["_video"]
        except Exception:
            pass
        manifest.record_video(batch_id, story_num, title, video_link, drive_file_id, folder_id)

    return job_data


def populate_backlog(db_path: str, batch_id: str):
    """Find COMPLETE stage 4 jobs and insert them into stage 5 as PENDING."""
    manifest = Manifest(db_path)
    s4_completed = manifest.get_completed_story_nums(batch_id, "stage_04_music")
    if not s4_completed:
        logger.warning("No complete stage 4 jobs found for batch %s.", batch_id)
        return
        
    created = manifest.create_batch(batch_id, "stage_05_compose", s4_completed)
    if created > 0:
        logger.info("Populated %d new jobs for Stage 5.", created)


def run_stage_05_worker(
    worker_id: str,
    db_path: str,
    batch_id: str,
):
    """Entry point for a Stage 5 worker process."""
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
        stage="stage_05_compose",
        out_dir=LOCAL_OUTPUT_DIR,
        process_fn=process_stage_05,
        uploader=uploader
    )


def spawn_stage_05_workers(db_path: str, batch_id: str, count: int = 1):
    """Spawn multiple worker processes for Stage 5."""
    populate_backlog(db_path, batch_id)
    processes = []
    for i in range(count):
        worker_id = f"stg5-{i+1}"
        p = multiprocessing.Process(
            target=run_stage_05_worker,
            args=(worker_id, db_path, batch_id)
        )
        p.start()
        processes.append(p)
    return processes

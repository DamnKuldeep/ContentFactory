"""
Stage 4: Music generation entry point.
"""

import json
import logging
import multiprocessing
import os
from typing import Optional

from shared import config
from shared.config import DRIVE_PARENT_FOLDER_ID, LOCAL_OUTPUT_DIR
from shared.drive import get_drive_client, DriveUploader
from shared.manifest import Manifest
from shared.worker import run_worker_loop
from shared.narration_features import compute_energy_envelope
from shared.timing import step

from .brief import generate_music_brief, generate_refined_brief
from .ace import generate_music

_ST = "stage_04_music"

logger = logging.getLogger("contentfactory.stage_04")


ARTIFACT_CONTRACT = {
    "version": "1.0",
    "required": ["stage_03.json"],
    "optional": ["reference_narration.mp3"],
    "produces": ["music.mp3", "stage_04.json"]
}

def verify_contract_inputs(work_dir: str):
    for req in ARTIFACT_CONTRACT["required"]:
        path = os.path.join(work_dir, req)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input artifact: {req}")

def process_stage_04(job: dict, worker_id: str, uploader: Optional[DriveUploader]) -> dict:
    """Execute Stage 4."""
    batch_id = job["batch_id"]
    story_num = job["story_num"]
    
    manifest = Manifest(os.environ.get("DB_PATH", "manifest.sqlite"))
    prev_job = manifest.get_completed_job(batch_id, "stage_03_images", story_num)
    
    if not prev_job:
        raise ValueError(f"Could not find completed stage_03_images job for story {story_num}")

    work_dir = os.path.join(LOCAL_OUTPUT_DIR, f"work_{batch_id}_{story_num}_s4")
    os.makedirs(work_dir, exist_ok=True)

    # Fetch previous stage artifact
    prev_json_path = os.path.join(work_dir, "stage_03.json")
    file_id = prev_job.get("drive_file_id")
    if not file_id or not uploader:
        raise ValueError(f"No drive info available for stage 3 output of story {story_num}")
        
    logger.info("Worker %s downloading Stage 3 output from Drive...", worker_id)
    if not uploader.download_file(file_id, prev_json_path):
        raise RuntimeError("Failed to download Stage 3 artifact")

    with open(prev_json_path, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    story = job_data.get("story", "")
    script = job_data.get("script", "")

    if not story or not script:
        raise ValueError("Story or script missing from job data")

    # Stage 3 branches off stage_01, so merge stage_02's narration metadata (alignment for
    # duration + audio_drive_id for the energy-envelope reference).
    if not job_data.get("alignment") or not job_data.get("meta", {}).get("stage_02", {}).get("audio_drive_id"):
        s2 = manifest.get_completed_job(batch_id, "stage_02_narration", story_num)
        if s2 and s2.get("drive_file_id") and uploader:
            s2_path = os.path.join(work_dir, "stage_02.json")
            if uploader.download_file(s2["drive_file_id"], s2_path):
                with open(s2_path) as f:
                    s2_data = json.load(f)
                job_data["alignment"] = s2_data.get("alignment", job_data.get("alignment"))
                job_data.setdefault("meta", {}).setdefault("stage_02", {}).update(
                    s2_data.get("meta", {}).get("stage_02", {}))
                logger.info("Merged stage_02 narration metadata.")

    # Download Stage 2 audio for reference
    stage2 = job_data.get("meta", {}).get("stage_02", {})
    audio_drive_id = stage2.get("audio_drive_id")
    reference_mp3 = os.path.join(work_dir, "reference_narration.mp3")
    
    if audio_drive_id and uploader:
        logger.info("Worker %s downloading Stage 2 audio for reference...", worker_id)
        if not uploader.download_file(audio_drive_id, reference_mp3):
            logger.warning("Worker %s failed to download reference audio, will use text-only.", worker_id)
            reference_mp3 = None
    else:
        reference_mp3 = None
        
    # Verify contracts
    verify_contract_inputs(work_dir)
        
    # Calculate duration
    alignment = job_data.get("alignment", {})
    ends = alignment.get("character_end_times_seconds", [])
    duration = int(float(ends[-1])) + 2 if ends else 60

    # 0. Extract narration energy envelope (techC grounding)
    energy_envelope = None
    if reference_mp3 and os.path.exists(reference_mp3):
        try:
            with step(_ST, "energy_envelope", kind="cpu"):
                energy_envelope = compute_energy_envelope(reference_mp3)
            logger.info("Computed energy envelope: %d windows", len(energy_envelope))
        except Exception as e:
            logger.warning("Energy envelope extraction failed (%s); falling back to text-only brief.", e)

    # 1. Pass-1 brief (energy-grounded)
    brief = generate_music_brief(
        script=script,
        duration=duration,
        has_voiceover=True,
        reference_mp3=reference_mp3,
        energy_envelope=energy_envelope,
    )
    logger.info("Pass-1 music brief: %s", brief)

    # Free Qwen from VRAM before ACE-Step loads — they don't co-fit on a 24 GB GPU.
    if not config.BGM_TWO_PASS:
        from .brief import free_qwen
        free_qwen()

    # 2. Generate music (capture ACE-Step LM blueprint for refinement)
    audio_path = os.path.join(work_dir, "music.mp3")
    _, lm_blueprint = generate_music(brief, duration, audio_path, label="ace_generate_pass1")

    # 2b. Two-pass refinement (techC p2): verify vs LM blueprint → refined brief → regenerate
    refined_brief = None
    verification = None
    if config.BGM_TWO_PASS:
        try:
            verification, refined_brief = generate_refined_brief(
                script=script, duration=duration, brief1=brief, lm_blueprint=lm_blueprint,
                has_voiceover=True, energy_envelope=energy_envelope,
            )
            logger.info("Pass-2 refined brief: %s", refined_brief)
            generate_music(refined_brief, duration, audio_path, label="ace_generate_pass2")   # overwrite music.mp3
        except Exception as e:
            logger.warning("Two-pass refinement failed (%s); keeping pass-1 music.", e)
            refined_brief = None

    # 3. Upload
    drive_file_id = None
    if uploader and os.path.exists(audio_path):
        logger.info("Worker %s uploading generated music %s", worker_id, audio_path)
        target_folder = uploader.get_video_subfolder(batch_id, story_num, "music")

        drive_file_id = uploader.upload_and_verify(audio_path, target_folder)
        if drive_file_id:
            try:
                os.remove(audio_path)
            except Exception:
                pass
        else:
            logger.warning("Worker %s failed to upload music %s", worker_id, audio_path)

    # 4. Update data
    job_data["meta"]["stage_04"] = {
        "music_brief": brief,
        "refined_brief": refined_brief,
        "verification": verification,
        "lm_blueprint": lm_blueprint,
        "music_drive_id": drive_file_id,
    }

    # Clean up Stage 3 local artifact if we downloaded it
    if prev_json_path.startswith(LOCAL_OUTPUT_DIR) and "stage_03" in prev_json_path:
        try:
            os.remove(prev_json_path)
        except Exception:
            pass

    return job_data


def populate_backlog(db_path: str, batch_id: str):
    """Find COMPLETE stage 3 jobs and insert them into stage 4 as PENDING."""
    manifest = Manifest(db_path)
    s3_completed = manifest.get_completed_story_nums(batch_id, "stage_03_images")
    if not s3_completed:
        logger.warning("No complete stage 3 jobs found for batch %s.", batch_id)
        return
        
    created = manifest.create_batch(batch_id, "stage_04_music", s3_completed)
    if created > 0:
        logger.info("Populated %d new jobs for Stage 4.", created)


def run_stage_04_worker(
    worker_id: str,
    db_path: str,
    batch_id: str,
):
    """Entry point for a Stage 4 worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{worker_id}] [%(levelname)s] %(name)s: %(message)s"
    )

    from shared.llm import init_client
    from shared.config import OPENROUTER_API_KEY
    try:
        init_client(OPENROUTER_API_KEY)
    except Exception as e:
        logger.error("Worker %s failed to init LLM client: %s", worker_id, e)
        return

    os.environ["DB_PATH"] = db_path
    manifest = Manifest(db_path)
    uploader = get_drive_client()

    run_worker_loop(
        worker_id=worker_id,
        manifest=manifest,
        batch_id=batch_id,
        stage="stage_04_music",
        out_dir=LOCAL_OUTPUT_DIR,
        process_fn=process_stage_04,
        uploader=uploader
    )


def spawn_stage_04_workers(db_path: str, batch_id: str, count: int = 2):
    """Spawn multiple worker processes for Stage 4."""
    populate_backlog(db_path, batch_id)
    processes = []
    for i in range(count):
        worker_id = f"stg4-{i+1}"
        p = multiprocessing.Process(
            target=run_stage_04_worker,
            args=(worker_id, db_path, batch_id)
        )
        p.start()
        processes.append(p)
    return processes

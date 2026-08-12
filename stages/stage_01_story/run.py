"""
Stage 1: Story entry point.
"""

import logging
import multiprocessing
import os
import sys
from typing import Optional

# Stage-1 internal modules (pipeline → convergence → prompts/models) use bare sibling
# imports, so this directory must be importable.
_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from shared.config import OPENROUTER_API_KEY, DRIVE_SA_KEY_PATH, DRIVE_PARENT_FOLDER_ID, LOCAL_OUTPUT_DIR
from shared.drive import get_drive_client, DriveUploader
from shared.manifest import Manifest
from shared.worker import run_worker_loop

from shared.llm import init_client
from .pipeline import run_all

logger = logging.getLogger("contentfactory.stage_01")


ARTIFACT_CONTRACT = {
    "version": "1.0",
    "required": [],
    "optional": [],
    "produces": ["stage_01.json"]
}

def verify_contract_inputs(work_dir: str):
    for req in ARTIFACT_CONTRACT["required"]:
        path = os.path.join(work_dir, req)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input artifact: {req}")

def process_stage_01(job: dict, worker_id: str, uploader: Optional[DriveUploader]) -> dict:
    """Wrapper around run_all that meets the worker loop signature."""
    import os
    from shared.config import LOCAL_OUTPUT_DIR
    batch_id = job["batch_id"]
    story_num = job["story_num"]
    work_dir = os.path.join(LOCAL_OUTPUT_DIR, f"work_{batch_id}_{story_num}_s1")
    os.makedirs(work_dir, exist_ok=True)
    
    verify_contract_inputs(work_dir)
    
    aspect = "9:16"  # could be pulled from job or config
    # run_all returns the final dict that will be saved and uploaded
    final_output = run_all(aspect=aspect, workers=2)
    return final_output


def run_stage_01_worker(
    worker_id: str,
    db_path: str,
    batch_id: str,
):
    """Entry point for a Stage 1 worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{worker_id}] [%(levelname)s] %(name)s: %(message)s"
    )

    try:
        init_client(OPENROUTER_API_KEY)
    except Exception as e:
        logger.error("Worker %s failed to init LLM client: %s", worker_id, e)
        return

    manifest = Manifest(db_path)
    uploader = get_drive_client()

    run_worker_loop(
        worker_id=worker_id,
        manifest=manifest,
        batch_id=batch_id,
        stage="stage_01_story",
        out_dir=LOCAL_OUTPUT_DIR,
        process_fn=process_stage_01,
        uploader=uploader
    )


def spawn_stage_01_workers(db_path: str, batch_id: str, count: int = 2):
    """Spawn multiple worker processes for Stage 1."""
    processes = []
    for i in range(count):
        worker_id = f"stg1-{i+1}"
        p = multiprocessing.Process(
            target=run_stage_01_worker,
            args=(worker_id, db_path, batch_id)
        )
        p.start()
        processes.append(p)
    return processes

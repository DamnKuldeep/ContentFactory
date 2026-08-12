"""
Stage 3: Image generation (FLUX.2).
Iterates over scenes, generates an image for each using local diffusers, and uploads to Drive.
"""

import logging
import os
import time
from typing import Optional

from shared.drive import DriveUploader
from shared.timing import step

logger = logging.getLogger("contentfactory.stage_03")
_ST = "stage_03_images"

_pipe = None

# Pipeline Constants
BATCH_SIZE = 4
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
NUM_STEPS = 4
GUIDANCE_SCALE = 1.0
BASE_SEED = 1234
_DIMS = {"9:16": [752, 1328], "1:1": [1024, 1024], "16:9": [1328, 752], "4:5": [912, 1136]}

def _round16(x):
    return max(16, int(round(x / 16)) * 16)

def _get_pipeline():
    global _pipe
    if _pipe is None:
        logger.info("Loading FLUX.2 pipeline: %s", MODEL_ID)
        import torch
        # Fallback to FluxPipeline if Flux2KleinPipeline isn't available
        try:
            from diffusers import Flux2KleinPipeline as PipelineCls
        except ImportError:
            from diffusers import FluxPipeline as PipelineCls
            
        with step(_ST, "flux_load", kind="load", sync=True):
            _pipe = PipelineCls.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            _pipe = _pipe.to("cuda")
        try:
            _pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        from shared import config as _cfg
        if getattr(_cfg, "FLUX_COMPILE", False):   # opt-in; amortized over the batch
            # compile traces lazily on the first forward, so suppress_errors guarantees an eager
            # fallback (output-identical) if inductor fails mid-run rather than crashing the batch.
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
            except Exception:
                pass
            mode = getattr(_cfg, "FLUX_COMPILE_MODE", "default")   # "default" avoids CUDA-graph OOM on 24 GB
            try:
                _pipe.transformer = torch.compile(_pipe.transformer, mode=mode)
                logger.info("FLUX.2 transformer torch.compiled (mode=%s).", mode)
            except Exception as e:
                logger.warning("FLUX torch.compile skipped: %s", e)
        logger.info("FLUX.2 pipeline loaded successfully.")
    return _pipe

import torch

def _get_dynamic_batch_size() -> int:
    try:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if mem_gb > 23:
            return 4 # RTX 5090 / 3090 (24GB+)
        elif mem_gb > 15:
            return 2 # 16GB GPUs
        else:
            return 1 # 12GB or below
    except Exception:
        return 1

def run_all(job_data: dict, work_dir: str, uploader: Optional[DriveUploader] = None) -> dict:
    """
    Main orchestration for Stage 3.
    """
    scenes = job_data.get("scenes", [])
    if not scenes:
        raise ValueError("No scenes found in job data")
        
    aspect = job_data.get("meta", {}).get("flux2", {}).get("aspect_ratio", "9:16")
    batch_id = job_data.get("batch_id", "unknown")
    story_num = job_data.get("story_num", job_data.get("meta", {}).get("story_num", 0))
    
    # Target folder in Drive for this story's images (per-video images/ subfolder)
    story_folder_id = None
    if uploader:
        story_folder_id = uploader.get_video_subfolder(batch_id, story_num, "images")

    # Partial Resume Logic: Check which images already exist in Drive
    existing_files = {}
    if uploader and story_folder_id:
        try:
            query = f"'{story_folder_id}' in parents and trashed=false"
            results = uploader.service.files().list(q=query, fields="files(id, name)").execute()
            for f in results.get("files", []):
                existing_files[f["name"]] = f["id"]
        except Exception as e:
            logger.warning("Failed to query existing files in Drive for resume: %s", e)

    W, H = _DIMS.get(aspect, _DIMS["9:16"])
    W, H = _round16(W), _round16(H)

    image_results = []
    
    # Determine batch size dynamically
    batch_size = _get_dynamic_batch_size()
    logger.info("Dynamic batch size set to %d based on VRAM.", batch_size)

    # Lazily load pipeline
    pipe = None

    # Upload images in a background pool so the GPU keeps generating the next batch instead of
    # idling during ~3.5 s/image serial uploads (the dominant stage-3 wall cost).
    from concurrent.futures import ThreadPoolExecutor
    # max_workers=1: a single background uploader still overlaps with the NEXT generation batch
    # (lever B's benefit) but avoids CONCURRENT resumable uploads — under a flaky uplink those raced
    # in OpenSSL and SIGSEGV'd the whole process (rc=139). Serial uploads + retries are crash-safe.
    upload_pool = ThreadPoolExecutor(max_workers=1) if (uploader and story_folder_id) else None
    upload_futs = []   # (scene_id, filename, future)

    def _upload(path, filename):
        fid = uploader.upload_and_verify(path, story_folder_id)
        try:
            os.remove(path)
        except OSError:
            pass
        if not fid:
            logger.warning("Failed to upload image %s", filename)
        return fid

    for start in range(0, len(scenes), batch_size):
        chunk = scenes[start:start + batch_size]

        pending_chunk = []
        for c in chunk:
            scene_id = c.get("id", c.get("index", 1))
            filename = f"scene_{scene_id:03d}.png"
            if filename in existing_files:
                logger.info("Resuming: %s already in Drive, skipping.", filename)
                image_results.append({"scene_id": scene_id, "drive_file_id": existing_files[filename],
                                      "filename": filename})
            else:
                pending_chunk.append((c, scene_id, filename))

        if not pending_chunk:
            continue
        if pipe is None:
            pipe = _get_pipeline()

        prompts = [c[0].get("prompt", c[0].get("visual_brief", "")) for c in pending_chunk]
        gens = None
        if BASE_SEED is not None:
            gens = [torch.Generator(device="cuda").manual_seed(BASE_SEED + c[1]) for c in pending_chunk]

        logger.info("Generating batch of %d images (W:%d H:%d)...", len(pending_chunk), W, H)
        with step(_ST, "flux_generate", kind="infer", sync=True, n=len(pending_chunk), steps=NUM_STEPS):
            images = pipe(prompt=prompts, num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE,
                          height=H, width=W, generator=gens).images

        for (c, scene_id, filename), im in zip(pending_chunk, images):
            local_img_path = os.path.join(work_dir, filename)
            im.save(local_img_path)
            if upload_pool is not None:
                fut = upload_pool.submit(_upload, local_img_path, filename)
                upload_futs.append((scene_id, filename, fut))
            else:
                image_results.append({"scene_id": scene_id, "drive_file_id": None, "filename": filename})
        torch.cuda.empty_cache()

    # Gather background uploads (overlapped with generation above).
    if upload_pool is not None:
        with step(_ST, "image_uploads_wait", kind="io", n=len(upload_futs)):
            for scene_id, filename, fut in upload_futs:
                image_results.append({"scene_id": scene_id, "drive_file_id": fut.result(),
                                      "filename": filename})
        upload_pool.shutdown(wait=True)

    job_data["meta"]["stage_03"] = {
        "images": sorted(image_results, key=lambda x: x["scene_id"]),
        "drive_folder_id": story_folder_id
    }

    return job_data

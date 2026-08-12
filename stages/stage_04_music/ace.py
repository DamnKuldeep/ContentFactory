"""
Stage 4: ACE-Step music generation.
"""

import logging
import os
import sys

from shared import config
from shared.timing import step

logger = logging.getLogger("contentfactory.stage_04")
_ST = "stage_04_music"

_dit_handler = None
_llm_handler = None

def _get_ace():
    global _dit_handler, _llm_handler
    if _dit_handler is None or _llm_handler is None:
        logger.info("Initializing ACE-Step...")
        # Check environment
        REPO_DIR = config.ACE_REPO_DIR
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
            
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        
        try:
            import torchao
            quantization = "int8_weight_only"
            compile_model = True
        except ImportError:
            quantization = None
            compile_model = False
            
        with step(_ST, "ace_load", kind="load", sync=True):
            _dit_handler = AceStepHandler()
            msg, ok = _dit_handler.initialize_service(
                project_root=REPO_DIR,
                config_path="acestep-v15-turbo",
                device="cuda:0",
                offload_to_cpu=False,
                quantization=quantization,
                compile_model=compile_model
            )
            if not ok:
                raise RuntimeError(f"ACE-Step DiT init failed: {msg}")

            _llm_handler = LLMHandler()
            msg, ok = _llm_handler.initialize(
                checkpoint_dir=config.ACE_CHECKPOINTS,
                lm_model_path="acestep-5Hz-lm-0.6B",
                backend="pt",
                device="cuda:0",
                offload_to_cpu=False
            )
            if not ok:
                raise RuntimeError(f"ACE-Step LLM init failed: {msg}")
            
        logger.info("ACE-Step handlers loaded successfully.")
    return _dit_handler, _llm_handler


def generate_music(brief: dict, duration: int, out_audio_path: str, label: str = "ace_generate"):
    """
    ACE-Step music generation from a brief.

    Returns (success: bool, lm_blueprint: dict) where lm_blueprint is ACE-Step's LM
    interpretation (chosen BPM/key etc.), used by the two-pass refinement. Empty dict
    if the build doesn't expose it (two-pass then degrades to single-pass).
    """
    logger.info("Generating music for brief: %s...", brief.get("caption", "")[:50])
    
    dit_handler, llm_handler = _get_ace()
    from acestep.inference import GenerationParams, GenerationConfig, generate_music as ace_generate
    
    caption = brief.get("caption", "")
    tags = brief.get("tags", "")
    full_caption = caption if not tags else f"{caption}\n\nTags: {tags}"
    
    seed = brief.get("seed", 42)
    
    params = GenerationParams(
        task_type="text2music",
        caption=full_caption[:512],
        lyrics="[Instrumental]",
        instrumental=True,
        duration=float(duration),
        bpm=brief.get("bpm"),
        keyscale=brief.get("keyscale", ""),
        inference_steps=brief.get("inference_steps", 12),
        shift=3.0,
        infer_method="ode",
        seed=seed,
        thinking=True,
    )
    
    config = GenerationConfig(
        batch_size=brief.get("batch_size", 1),
        use_random_seed=(seed < 0),
        seeds=[seed] if seed >= 0 else None,
        audio_format="mp3"
    )
    
    work_dir = os.path.dirname(out_audio_path)
    with step(_ST, label, kind="infer", sync=True, steps=brief.get("inference_steps", 12), duration=duration):
        result = ace_generate(dit_handler, llm_handler, params, config, save_dir=work_dir)

    if not result.success:
        raise RuntimeError(f"Generation failed: {result.error}")

    # Capture the LM blueprint (BPM/key the planner actually chose) for two-pass refine.
    lm_blueprint = {}
    try:
        lm_blueprint = (getattr(result, "extra_outputs", None) or {}).get("lm_metadata", {}) or {}
    except Exception:
        lm_blueprint = {}

    # ACE-Step generates files like {seed}_...mp3 in the directory.
    # We move the first generated file to out_audio_path.
    import shutil
    if result.audios and len(result.audios) > 0:
        generated_path = result.audios[0]["path"]
        shutil.move(generated_path, out_audio_path)
        logger.info("Saved ACE-Step generated audio to %s", out_audio_path)
        return True, lm_blueprint
    else:
        raise RuntimeError("No audios returned by ACE-Step")

"""
Stage 2: Narration generation using ElevenLabs TTS and forced alignment.
"""

import json
import logging
import os
import requests
import time

from shared import config
from shared.config import ELEVENLABS_API_KEY

logger = logging.getLogger("contentfactory.stage_02")

# Default voice ID, configurable via job_data
DEFAULT_VOICE_ID = config.ELEVENLABS_VOICE_ID
ELEVENLABS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"


def generate_narration(script: str, out_audio_path: str, voice_id: str = DEFAULT_VOICE_ID) -> dict:
    """
    Call ElevenLabs TTS with timestamps.
    Returns a dict with alignment data.
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY is not set")

    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": script,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    url = ELEVENLABS_URL_TEMPLATE.format(voice_id=voice_id)
    logger.info("Calling ElevenLabs TTS with voice %s...", voice_id)
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API failed ({response.status_code}): {response.text}")

    resp_data = response.json()
    
    import base64
    audio_bytes = base64.b64decode(resp_data["audio_base64"])
    
    with open(out_audio_path, "wb") as f:
        f.write(audio_bytes)
        
    logger.info("Saved audio to %s", out_audio_path)
    
    alignment = resp_data.get("alignment", {})
    return {"alignment": alignment}


def run_all(job_data: dict, work_dir: str) -> dict:
    """
    Main orchestration for Stage 2.
    job_data is the full output of Stage 1 (downloaded from Drive or local).
    We save audio locally and return metadata that includes the alignment.

    The narration engine is selected by config.NARRATION_ENGINE:
      "fish"       → local Fish Speech S2 Pro + WhisperX (default, GPU)
      "elevenlabs" → ElevenLabs /with-timestamps API
    """
    script = job_data.get("script", "")
    if not script:
        raise ValueError("No script found in Stage 1 output")

    audio_path = os.path.join(work_dir, "narration.mp3")
    engine = config.NARRATION_ENGINE

    if engine == "fish":
        from . import fish
        logger.info("Narration engine: Fish S2 Pro + WhisperX")
        alignment = fish.generate_narration_fish(script, audio_path)
    else:
        voice_id = job_data.get("meta", {}).get("voice_id", DEFAULT_VOICE_ID)
        logger.info("Narration engine: ElevenLabs (voice %s)", voice_id)
        alignment = generate_narration(script, audio_path, voice_id)

    job_data["meta"]["stage_02"] = {
        "audio_file": "narration.mp3",
        "engine": engine,
    }
    job_data["alignment"] = alignment["alignment"]
    return job_data

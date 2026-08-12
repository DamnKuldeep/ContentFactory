"""
Stage 4: Generate ACE-Step music brief using Qwen2.5-Omni.
Reads the narrative and listens to the narration audio to match mood and pacing.
"""

import json
import logging
import os
import re
import textwrap

from shared import config
from shared.narration_features import energy_envelope_text
from shared.timing import step

logger = logging.getLogger("contentfactory.stage_04")
_ST = "stage_04_music"

QWEN_MODEL_ID = config.QWEN_MUSIC_MODEL
_qwen_model = None
_qwen_processor = None
_qwen_device = None

SYSTEM_PROMPT = textwrap.dedent("""
You are a professional film-music supervisor and prompt engineer for ACE-Step 1.5,
a state-of-the-art text-to-music diffusion model.

Your task: read the narration script (and optionally listen to the reference audio),
then produce a JSON music brief that will be passed DIRECTLY to the ACE-Step generator.

━━━ ACE-Step 1.5 parameter reference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

caption  (str ≤512 chars)
  Describe ONLY musical qualities — instruments, timbre, texture, mood, atmosphere,
  energy, tempo feel, genre. NEVER include BPM numbers, key names, duration, or
  story content. Be vivid and specific.
  ✓ "slow brooding cello ostinato under sparse tremolo strings, muted brass swells,
     dark neo-classical atmosphere, creeping investigative tension"
  ✗ "background music at 70 BPM in A minor for a 60-second documentary"

tags  (str)
  8–14 comma-separated genre/mood/instrument keywords, ALL lowercase, single string.
  Example: "dark ambient, cinematic, neo-classical, strings, piano, drone, instrumental"

bpm  (int | null)
  60–180. Use null to let ACE-Step's LM planner auto-select.
  Slow grief ≈ 55–70 | mystery ≈ 75–95 | thriller ≈ 100–120 | action ≈ 125–160

keyscale  (str)
  Common stable keys: "C Major" "G Major" "D Major" "F Major"
                      "A minor" "D minor" "E minor" "G minor"
  Pass "" to let the planner decide.

inference_steps  (int 8–20)
  8 = fast draft | 12 = good quality | 20 = maximum quality

seed  (int)
  42 = reproducible. -1 = random variation.

batch_size  (int 1–4)
  1 for a single output; 2–4 for variation candidates.

music_volume  (float 0.10–0.30)
  Voiceover bed (music under speech) → 0.12–0.16
  Featured / solo music              → 0.20–0.25

━━━ Rules ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Return ONLY a single valid JSON object — zero prose, zero markdown fences.
2. caption MUST be ≤ 512 characters.
3. tags MUST be a single comma-separated STRING, not a JSON array.
4. BPM and key MUST NOT appear inside the caption text.
5. If a reference audio was provided, match its energy, tempo feel, and mood.
6. If the script has voiceover, set music_volume ≤ 0.16.
7. If a narration ENERGY ENVELOPE is provided, the music's loudness and density
   MUST track it — sparse/quiet where the voice is quiet, swelling where it
   intensifies, and shaped to the described overall energy arc.

━━━ Output schema ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "caption":         "<≤512 char string>",
  "tags":            "<comma-separated string>",
  "bpm":             <int 60-180 or null>,
  "keyscale":        "<string or \"\">",
  "inference_steps": <int 8-20>,
  "seed":            <int>,
  "batch_size":      <int 1-4>,
  "music_volume":    <float 0.10-0.30>
}
""").strip()

USER_PROMPT_TEMPLATE = textwrap.dedent("""
Target duration : {duration}s
Has voiceover   : {has_voiceover}

Narration script:
\"\"\"
{script}
\"\"\"

{audio_note}
Return the ACE-Step 1.5 JSON brief. Pure JSON only — no explanation, no fences.
""").strip()

# Pass-2 verification/refinement (two-pass "techC p2"): Qwen is shown its pass-1 brief
# plus ACE-Step's actual LM interpretation (the BPM/key the planner chose), and must
# critique and refine.
VERIFY_SYSTEM_PROMPT = SYSTEM_PROMPT + textwrap.dedent("""

━━━ Verification & Refinement mode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You will be given a Pass 1 brief and ACE-Step's actual LM interpretation (the BPM
and key it chose). Critically evaluate whether the brief captured the narration's
energy arc and story theme, then generate a REFINED brief that fixes any weaknesses.

Return JSON with EXACTLY these two top-level keys (no other text, no fences):
{
  "verification": {
    "bpm_match":    <bool>,
    "key_match":    <bool>,
    "issues":       "<what was missing>",
    "improvements": "<what the refined brief addresses>"
  },
  "refined_brief": { <full ACE-Step brief, same schema as above> }
}""")


def _validate_brief(brief: dict) -> dict:
    assert isinstance(brief.get("caption"), str), "caption must be a string"
    if len(brief["caption"]) > 512:
        brief["caption"] = brief["caption"][:512]
    assert isinstance(brief.get("tags"), str), "tags must be a string"
    if brief.get("bpm") is not None:
        brief["bpm"] = max(60, min(180, int(brief["bpm"])))
    brief["inference_steps"] = max(8,  min(20,  int(brief.get("inference_steps", 12))))
    brief["batch_size"]      = max(1,  min(4,   int(brief.get("batch_size", 1))))
    brief["music_volume"]    = round(max(0.10, min(0.30, float(brief.get("music_volume", 0.15)))), 3)
    brief.setdefault("seed", 42)
    brief.setdefault("keyscale", "")
    return brief


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def free_qwen():
    """Release Qwen from VRAM so ACE-Step can fit on a single 24 GB GPU."""
    global _qwen_model, _qwen_processor, _qwen_device
    if _qwen_model is None:
        return
    try:
        import torch, gc
        del _qwen_model, _qwen_processor
        _qwen_model = _qwen_processor = _qwen_device = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed Qwen from VRAM.")
    except Exception as e:
        logger.warning("free_qwen failed: %s", e)


def _get_qwen():
    global _qwen_model, _qwen_processor, _qwen_device
    if _qwen_model is None:
        logger.info("Loading Qwen2.5-Omni-7B...")
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        # Calculate dynamic max_memory mapping
        max_memory = None
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            max_memory = {}
            for i in range(gpu_count):
                mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                # Keep ~2GB headroom
                safe_gb = max(1, int(mem_gb - 2))
                max_memory[i] = f"{safe_gb}GiB"
            logger.info("Dynamic VRAM mapping for Qwen: %s", max_memory)
            
        with step(_ST, "qwen_load", kind="load", sync=True):
            _qwen_processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
            _qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
        try:
            _qwen_model.disable_talker()
        except Exception:
            pass
        _qwen_model.eval()
        _qwen_device = next(_qwen_model.parameters()).device
        logger.info("Qwen loaded successfully.")
    return _qwen_model, _qwen_processor, _qwen_device


def _qwen_call(conversation) -> str:
    """Single Qwen multimodal generation → raw decoded text."""
    import torch
    from qwen_omni_utils import process_mm_info

    qwen_model, qwen_processor, first_device = _get_qwen()
    text_prompt = qwen_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = qwen_processor(
        text=text_prompt,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
    ).to(first_device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out_ids = qwen_model.generate(
            **inputs, max_new_tokens=640, do_sample=True, temperature=0.2,
            return_audio=False, pad_token_id=qwen_processor.tokenizer.eos_token_id,
        )
    return qwen_processor.tokenizer.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()


def _qwen_json(conversation, retries: int = 3) -> dict:
    """Run Qwen and parse JSON, with self-correction retries."""
    raw = ""
    for attempt in range(1, retries + 1):
        try:
            raw = _qwen_call(conversation)
            return _parse_json(raw)
        except Exception as e:
            logger.warning("Qwen JSON parse failed on attempt %d: %s", attempt, e, exc_info=True)
            conversation = conversation + [
                {"role": "assistant", "content": [{"type": "text", "text": raw}]},
                {"role": "user", "content": [{"type": "text",
                  "text": "Your response was not valid JSON. Return ONLY the JSON object, no prose, no fences."}]},
            ]
    raise RuntimeError("Qwen JSON generation failed after retries.")


def _build_user_text(script: str, duration: int, has_voiceover: bool,
                     energy_envelope=None, audio_note: str = "") -> str:
    """Assemble the user prompt, embedding the energy envelope when available (techC)."""
    user_text = USER_PROMPT_TEMPLATE.format(
        script=script.strip(),
        duration=duration,
        has_voiceover="yes" if has_voiceover else "no",
        audio_note=audio_note,
    )
    if energy_envelope:
        user_text += (
            "\n\n" + energy_envelope_text(energy_envelope) +
            "\nThe music loudness and density should track this energy envelope."
        )
    return user_text


def generate_music_brief(script: str, duration: int = 60, has_voiceover: bool = True,
                         reference_mp3: str = None, energy_envelope=None) -> dict:
    """Pass-1 brief. Grounds on the narration script + (optional) reference audio + energy envelope."""
    # Ground the brief on the script + energy envelope (techC). Note: Qwen2.5-Omni audio input via
    # qwen_omni_utils expects a path/URL STRING, not a (waveform, sr) tuple — the old tuple crashed
    # process_audio_info ('tuple' has no attribute 'startswith') so raw-audio grounding never worked.
    # The narration's energy/tempo character is already captured by `energy_envelope` (derived from
    # that same mp3), so text+envelope is both correct and OOM-safe (no ~90 s of audio into Qwen on
    # a 24 GB GPU). pass-2 (generate_refined_brief) is already text-only for the same reason.
    user_text = _build_user_text(script, duration, has_voiceover, energy_envelope, audio_note="")

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    logger.info("Generating pass-1 music brief with Qwen (text + energy-envelope)...")
    with step(_ST, "brief_pass1", kind="infer", sync=True):
        return _validate_brief(_qwen_json(conversation))


def generate_refined_brief(script: str, duration: int, brief1: dict, lm_blueprint: dict,
                           has_voiceover: bool = True, energy_envelope=None):
    """
    Pass-2 refinement: Qwen verifies the pass-1 brief against ACE-Step's LM blueprint
    (the BPM/key it actually chose) and returns (verification, refined_brief).
    Text-only (no audio) to keep context small.
    """
    base_user = _build_user_text(script, duration, has_voiceover, energy_envelope)
    bp_bpm = lm_blueprint.get("bpm", "unknown")
    bp_key = lm_blueprint.get("keyscale", lm_blueprint.get("key", "unknown"))
    pass2_user = (
        base_user
        + f"\n\n--- Pass 1 brief (sent to ACE-Step) ---\n{json.dumps(brief1, indent=2)}\n"
        + f"\n--- ACE-Step LM interpretation ---\n  BPM chosen : {bp_bpm}\n  Key chosen : {bp_key}\n\n"
        + "Verify whether Pass 1 captured the narration's energy arc and theme, then produce a "
          "REFINED brief. Return pure JSON with \"verification\" and \"refined_brief\" keys."
    )
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": VERIFY_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": pass2_user}]},
    ]
    logger.info("Generating pass-2 refined brief with Qwen...")
    with step(_ST, "brief_pass2", kind="infer", sync=True):
        result = _qwen_json(conversation)
    verification = result.get("verification", {})
    refined = _validate_brief(result.get("refined_brief", brief1))
    return verification, refined

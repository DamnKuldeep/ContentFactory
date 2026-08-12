"""
Content Factory — Shared configuration and environment setup.
"""

import os
from dotenv import load_dotenv

# Load .env file from the project root
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ContentFactory/
_env_path = os.path.join(_root, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# Repo-local HuggingFace cache, so the per-stage setup scripts and the runtime share one cache
# of downloaded weights. setdefault → an explicit HF_HOME (env/.env) still wins. Must run before
# any transformers/diffusers/huggingface_hub import (config is imported early by every stage).
HF_HOME = os.environ.setdefault("HF_HOME", os.path.join(_root, "models", "hf_cache"))

# Shared configuration values
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

DRIVE_SA_KEY_PATH = os.environ.get("DRIVE_SA_KEY_PATH", "./drive_sa_key.json")
# Target Drive folder. Defaults to the gdrive_tool.py FOLDER_ID so a fresh clone works out of box.
DRIVE_PARENT_FOLDER_ID = os.environ.get("DRIVE_PARENT_FOLDER_ID", "") or "11b1Se2OA5cm8qyA2rteNH4_L9fuKaIeV"

# OAuth installed-app login (refresh-only on workers). Copy these two files to every machine;
# authorization is done ONCE via scripts/drive_auth.py and never again. Empty → auto-resolve
# credentials.json/secrets.json and token.json from [project root, ContentFactory/, shared/].
DRIVE_CLIENT_SECRETS = os.environ.get("DRIVE_CLIENT_SECRETS", "")
DRIVE_TOKEN_PATH     = os.environ.get("DRIVE_TOKEN_PATH", "")

# Drive-hosted shared DB (opt-in: DRIVE_DB=1). Manifest lives in the Drive parent folder.
DRIVE_DB        = os.environ.get("DRIVE_DB", "").strip().lower() in ("1", "true", "yes")
DRIVE_DB_NAME   = os.environ.get("DRIVE_DB_NAME", "manifest.sqlite")
DRIVE_LOCK_NAME = os.environ.get("DRIVE_LOCK_NAME", "manifest.lock")

# Per-video Drive folder skeleton (media-typed subfolders, created up-front).
VIDEO_SUBFOLDERS = ["metadata", "narration", "images", "music", "subtitles", "final"]

# Directory where local outputs are staged before upload
LOCAL_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")

# Number of worker processes to spawn by default
DEFAULT_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))

# Model price table ($/1M tokens: input, output) — used by llm.py Meter for cost estimation.
# Empty dict means unknown models get cost=0.0 (safe fallback).
PRICES: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# Narration engine (stage_02)
# ─────────────────────────────────────────────────────────────────────────────
# "fish"       → local Fish Speech S2 Pro (NF4 4-bit) + WhisperX forced alignment (GPU)
# "elevenlabs" → ElevenLabs /with-timestamps API (needs ELEVENLABS_API_KEY)
NARRATION_ENGINE = os.environ.get("NARRATION_ENGINE", "fish").strip().lower()

_MODELS_DIR = os.path.join(_root, "models")

# Fish Speech S2 Pro (paths created by scripts/fish_narration.py first-run setup)
FISH_REPO_DIR  = os.environ.get("FISH_REPO_DIR",  os.path.join(_MODELS_DIR, "fish-speech-int4-patch"))
FISH_CKPT_DIR  = os.environ.get("FISH_CKPT_DIR",  os.path.join(_MODELS_DIR, "fish_s2_pro_ckpt"))
_FISH_NB_DIR   = os.path.join(_root, "Notebooks", "Fish_S2_Pro")
FISH_REF_AUDIO = os.environ.get("FISH_REF_AUDIO", os.path.join(_FISH_NB_DIR, "reference.mp3"))
FISH_REF_TEXT_FILE = os.environ.get("FISH_REF_TEXT_FILE", os.path.join(_FISH_NB_DIR, "Reference_text.txt"))
FISH_VOICE_CACHE   = os.environ.get("FISH_VOICE_CACHE", os.path.join(_FISH_NB_DIR, ".voice_cache"))
FISH_CHUNK_MAX_BYTES = int(os.environ.get("FISH_CHUNK_MAX_BYTES", "240"))
# Generation preset (balanced narration); max_new_tokens=0 → generate each chunk to EOS.
FISH_PRESET = {
    "temperature":        float(os.environ.get("FISH_TEMPERATURE", "0.65")),
    "top_p":              float(os.environ.get("FISH_TOP_P", "0.88")),
    "top_k":              int(os.environ.get("FISH_TOP_K", "30")),
    "repetition_penalty": float(os.environ.get("FISH_REP_PENALTY", "1.12")),
    "speed":              float(os.environ.get("FISH_SPEED", "1.0")),
    "seed":               int(os.environ.get("FISH_SEED", "42")),
}

# WhisperX forced-alignment model size
WHISPERX_MODEL = os.environ.get("WHISPERX_MODEL", "base")

# Optional torch.compile for steady-state inference speedup (amortized over a batch). Default OFF —
# enable per stage to trade a one-time compile cost for faster per-item inference. No output change.
FISH_COMPILE = os.environ.get("FISH_COMPILE", "").strip().lower() in ("1", "true", "yes")
FLUX_COMPILE = os.environ.get("FLUX_COMPILE", "").strip().lower() in ("1", "true", "yes")
# torch.compile mode for FLUX. "reduce-overhead" uses CUDA graphs (fastest, but reserves ~2.7 GB
# extra → OOMs FLUX.2 on a 24 GB GPU after a few batches). "default" gets the inductor speedup
# without CUDA-graph memory. Use "reduce-overhead" only on >24 GB cards.
FLUX_COMPILE_MODE = os.environ.get("FLUX_COMPILE_MODE", "default").strip()

# ElevenLabs default voice (used when NARRATION_ENGINE == "elevenlabs")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

# ─────────────────────────────────────────────────────────────────────────────
# BGM (stage_04 music + stage_05 mix)
# ─────────────────────────────────────────────────────────────────────────────
# Music loudness RELATIVE TO the narration (0.4 → BGM sits at 0.4× the voice's LUFS).
BGM_LOUDNESS_RATIO = float(os.environ.get("BGM_LOUDNESS_RATIO", "0.42"))
# Two-pass refinement: pass-1 brief → generate → Qwen verifies vs ACE-Step LM blueprint → refined brief → regenerate.
BGM_TWO_PASS  = os.environ.get("BGM_TWO_PASS", "1").strip().lower() in ("1", "true", "yes")
# Narration-grounding technique for the brief ("techC" = energy envelope).
BGM_TECHNIQUE = os.environ.get("BGM_TECHNIQUE", "techC")
# Local Qwen used to write/refine the ACE-Step music brief.
QWEN_MUSIC_MODEL = os.environ.get("QWEN_MUSIC_MODEL", "Qwen/Qwen2.5-Omni-7B")

# Prompt caching for OpenRouter LLM calls — marks the (verbatim) system prompt cacheable so the
# big, repeated stage-1 judge/critic prefixes bill as cached reads. Off by default (provider
# support varies; enable + measure). No change to prompt content or model routing.
LLM_PROMPT_CACHE = os.environ.get("LLM_PROMPT_CACHE", "").strip().lower() in ("1", "true", "yes")

# ACE-Step 1.5 (text-to-music). Provisioned out-of-band — clone the repo and place the
# checkpoints anywhere, then point these at them in .env. Defaults are repo-local so a fresh
# clone fails with a clear "not found" instead of silently reading someone else's paths;
# scripts/setup_bgm.py validates them and warns.
ACE_REPO_DIR    = os.environ.get("ACE_REPO_DIR",    os.environ.get("ACE_REPO", os.path.join(_MODELS_DIR, "ACE-Step-1.5")))
ACE_CHECKPOINTS = os.environ.get("ACE_CHECKPOINTS", os.path.join(_MODELS_DIR, "ace-checkpoints"))

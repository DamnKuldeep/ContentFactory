"""
Stage 2 narration engine — Fish Speech S2 Pro (NF4 4-bit) + WhisperX alignment.

Local GPU TTS that produces narration.mp3 + a character-level alignment dict with
the SAME shape ElevenLabs returns:
    {characters, character_start_times_seconds, character_end_times_seconds}

Long-form generation splits the script into small sentence chunks, generates each
INDEPENDENTLY against the fixed reference voice (no context accumulation — Fish's
4096 context cannot hold a full ~100s narration), then concatenates. Alignment uses
WhisperX forced alignment mapped onto the script via difflib (robust to ASR drift /
mispronunciations).

Models load once per worker process via cached singletons (mirrors stage_03/04).
"""

import difflib
import logging
import os
import re
import subprocess
import sys
import tempfile

from shared import config
from shared.audio import atempo_chain, audio_duration
from shared.timing import step

_ST = "stage_02_narration"

logger = logging.getLogger("contentfactory.stage_02.fish")

_FISH = None      # (model, decode_one_token, codec)
_WHISPERX = None  # whisperx asr model
_PROMPT_TOKENS = None

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NORM_RE = re.compile(r"[^a-z0-9]")


# ── Model loading (cached) ────────────────────────────────────────────────────

def _ensure_repo_on_path():
    if config.FISH_REPO_DIR not in sys.path:
        sys.path.insert(0, config.FISH_REPO_DIR)


def _get_fish():
    """Load Fish S2 Pro (NF4 4-bit) once per process. Returns (model, decode_one_token, codec)."""
    global _FISH
    if _FISH is not None:
        return _FISH
    if not os.path.exists(os.path.join(config.FISH_CKPT_DIR, "config.json")):
        raise FileNotFoundError(
            f"Fish checkpoints not found at {config.FISH_CKPT_DIR}. "
            "Run scripts/fish_narration.py once to download them, or set FISH_CKPT_DIR."
        )
    _ensure_repo_on_path()
    import torch
    from fish_speech.models.text2semantic.inference import init_model, load_codec_model

    logger.info("Loading Fish S2 Pro (NF4 4-bit) from %s ...", config.FISH_CKPT_DIR)
    from pathlib import Path
    # torch.compile compiles `decode_one_token` lazily on its FIRST call (deep in generate_long),
    # so a wrap-time try/except can't catch a Triton/inductor failure. suppress_errors makes any
    # compile failure fall back to eager execution mid-run (output-identical) instead of crashing
    # the whole narration batch — pure measurement safety, no output change.
    if config.FISH_COMPILE:
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
    with step(_ST, "fish_load", kind="load", sync=True):
        try:
            model, decode_one_token = init_model(
                checkpoint_path=config.FISH_CKPT_DIR,
                device="cuda",
                precision=torch.float16,
                compile=config.FISH_COMPILE,   # opt-in steady-state speedup (amortized over the batch)
                max_length=4096,
                bnb4=True,
            )
        except Exception as e:
            if config.FISH_COMPILE:
                logger.warning("Fish init_model(compile=True) failed (%s) — retrying eager.", e)
                model, decode_one_token = init_model(
                    checkpoint_path=config.FISH_CKPT_DIR,
                    device="cuda",
                    precision=torch.float16,
                    compile=False,
                    max_length=4096,
                    bnb4=True,
                )
            else:
                raise
        codec = load_codec_model(Path(config.FISH_CKPT_DIR) / "codec.pth",
                                 device="cuda", precision=torch.float16)
    _FISH = (model, decode_one_token, codec)
    logger.info("Fish S2 Pro loaded.")
    return _FISH


def _get_whisperx():
    global _WHISPERX
    if _WHISPERX is None:
        import whisperx
        logger.info("Loading WhisperX (%s) ...", config.WHISPERX_MODEL)
        with step(_ST, "whisperx_load", kind="load", sync=True):
            _WHISPERX = whisperx.load_model(config.WHISPERX_MODEL, "cuda", compute_type="float16")
    return _WHISPERX


def _get_prompt_tokens(codec):
    """Encode the reference voice once (with on-disk cache)."""
    global _PROMPT_TOKENS
    if _PROMPT_TOKENS is not None:
        return _PROMPT_TOKENS
    import torch
    from fish_speech.models.text2semantic.inference import encode_audio

    cache = os.path.join(config.FISH_VOICE_CACHE, "prompt_tokens.pt")
    if os.path.exists(cache):
        _PROMPT_TOKENS = torch.load(cache, map_location="cpu")
        return _PROMPT_TOKENS
    os.makedirs(config.FISH_VOICE_CACHE, exist_ok=True)
    logger.info("Encoding reference voice from %s ...", config.FISH_REF_AUDIO)
    with step(_ST, "voice_encode", kind="load", sync=True):
        _PROMPT_TOKENS = encode_audio(config.FISH_REF_AUDIO, codec, "cuda").cpu()
    torch.save(_PROMPT_TOKENS, cache)
    return _PROMPT_TOKENS


def _ref_text():
    if os.path.exists(config.FISH_REF_TEXT_FILE):
        return open(config.FISH_REF_TEXT_FILE).read().strip()
    return ""


# ── Chunking + generation ─────────────────────────────────────────────────────

def _split_script_chunks(script, max_bytes=None):
    """Sentence-group a script into chunks <= max_bytes UTF-8 bytes."""
    max_bytes = max_bytes or config.FISH_CHUNK_MAX_BYTES
    chunks = []
    for para in script.split("\n"):
        para = para.strip()
        if not para:
            continue
        cur = ""
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            cand = (cur + " " + sent).strip() if cur else sent
            if cur and len(cand.encode("utf-8")) > max_bytes:
                chunks.append(cur)
                cur = sent
            else:
                cur = cand
        if cur:
            chunks.append(cur)
    return chunks or [script.strip()]


def _generate_wav(script, out_wav):
    """Per-chunk independent generation against the fixed reference voice → WAV."""
    import torch
    import soundfile as sf

    # _get_fish() puts the Fish repo on sys.path, so load the model BEFORE importing fish_speech.
    model, decode_one_token, codec = _get_fish()
    from fish_speech.models.text2semantic.inference import generate_long, decode_to_audio
    prompt_tokens = _get_prompt_tokens(codec)
    prompt_text = _ref_text()
    preset = config.FISH_PRESET

    torch.manual_seed(preset["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(preset["seed"])

    chunks = _split_script_chunks(script)
    segments = []
    with step(_ST, "tts_generate", kind="infer", sync=True, chunks=len(chunks)):
        for chunk in chunks:
            gen = generate_long(
                model=model, device="cuda", decode_one_token=decode_one_token,
                text=chunk, num_samples=1, max_new_tokens=0,
                top_p=preset["top_p"], top_k=preset["top_k"],
                repetition_penalty=preset["repetition_penalty"],
                temperature=preset["temperature"], compile=False,
                iterative_prompt=False, chunk_length=0,
                prompt_text=[prompt_text], prompt_tokens=[prompt_tokens],
            )
            cur = [r.codes for r in gen if r.action == "sample"]
            if cur:
                segments.append(torch.cat(cur, dim=1))

    if not segments:
        raise RuntimeError("Fish S2 Pro generated no audio codes")

    codes = torch.cat(segments, dim=1)
    with step(_ST, "decode_to_audio", kind="infer", sync=True):
        audio = decode_to_audio(codes.to("cuda"), codec)
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    sf.write(out_wav, audio.cpu().float().numpy(), codec.sample_rate)

    speed = preset["speed"]
    if abs(speed - 1.0) > 0.01:
        adj = out_wav + ".adj.wav"
        subprocess.run(["ffmpeg", "-y", "-i", out_wav, "-filter:a", atempo_chain(speed), adj],
                       capture_output=True, check=True)
        os.replace(adj, out_wav)

    dur = audio_duration(out_wav)
    logger.info("Fish: %d chunks → %.1fs audio for %d chars", len(chunks), dur, len(script))
    if dur < 0.7 * len(script) / 18.0:
        logger.warning("Fish audio %.1fs looks short for %d chars — possible truncation", dur, len(script))
    return out_wav


# ── WhisperX alignment → char-level transcript ────────────────────────────────

def _norm(tok):
    return _NORM_RE.sub("", tok.lower())


def _script_words(script):
    return [{"text": m.group(0), "cstart": m.start(), "cend": m.end()}
            for m in re.finditer(r"\S+", script)]


def _align_to_chars(mp3_path, script):
    """WhisperX transcribe+align → difflib map onto script → char-level alignment dict."""
    import whisperx
    import torch

    wx = _get_whisperx()
    audio = whisperx.load_audio(mp3_path)
    with step(_ST, "whisperx_transcribe", kind="infer", sync=True):
        result = wx.transcribe(audio, batch_size=8)
    lang = result.get("language", "en")
    with step(_ST, "align_model_load", kind="load", sync=True):
        amodel, meta = whisperx.load_align_model(language_code=lang, device="cuda")
    with step(_ST, "whisperx_align", kind="infer", sync=True):
        result = whisperx.align(result["segments"], amodel, meta, audio, "cuda",
                                return_char_alignments=False)
    del amodel
    torch.cuda.empty_cache()
    with step(_ST, "char_map", kind="cpu"):
        return _words_to_char_alignment(script, result.get("word_segments", []))


def _words_to_char_alignment(script, word_segs):
    """difflib-anchored map of heard-word timings onto script chars (monotonic, full length)."""
    n = len(script)
    starts = [0.0] * n
    ends = [0.0] * n

    swords = _script_words(script)
    nsw = len(swords)
    if nsw == 0:
        return {"characters": list(script),
                "character_start_times_seconds": starts,
                "character_end_times_seconds": ends}

    heard = []
    for seg in word_segs:
        w = (seg.get("word") or "").strip()
        if not w:
            continue
        t0, t1 = seg.get("start"), seg.get("end")
        if t0 is None and t1 is None:
            continue
        t0 = float(t0) if t0 is not None else float(t1)
        t1 = float(t1) if t1 is not None else t0
        if t1 < t0:
            t1 = t0
        nw = _norm(w)
        if nw:
            heard.append({"norm": nw, "start": t0, "end": t1})

    audio_end = heard[-1]["end"] if heard else 0.0
    w_start = [None] * nsw
    w_end = [None] * nsw

    if heard:
        a = [_norm(sw["text"]) for sw in swords]
        b = [h["norm"] for h in heard]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    w_start[i1 + k] = heard[j1 + k]["start"]
                    w_end[i1 + k] = heard[j1 + k]["end"]
            elif tag == "replace":
                hs = heard[j1]["start"]
                he = heard[j2 - 1]["end"] if j2 > j1 else hs
                cnt = i2 - i1
                for k in range(cnt):
                    w_start[i1 + k] = hs + (he - hs) * k / cnt
                    w_end[i1 + k] = hs + (he - hs) * (k + 1) / cnt

    def _prev(idx):
        j = idx
        while j >= 0 and w_start[j] is None:
            j -= 1
        return j

    def _next(idx):
        j = idx
        while j < nsw and w_start[j] is None:
            j += 1
        return j

    i = 0
    while i < nsw:
        if w_start[i] is not None:
            i += 1
            continue
        p = _prev(i - 1)
        q = _next(i)
        lo = w_end[p] if p >= 0 else 0.0
        hi = w_start[q] if q < nsw else audio_end
        if hi < lo:
            hi = lo
        run = q - i if q < nsw else nsw - i
        for k in range(run):
            w_start[i + k] = lo + (hi - lo) * k / (run + 1)
            w_end[i + k] = lo + (hi - lo) * (k + 1) / (run + 1)
        i += run

    last = 0.0
    for k in range(nsw):
        if w_start[k] < last:
            w_start[k] = last
        if w_end[k] < w_start[k]:
            w_end[k] = w_start[k]
        last = w_end[k]

    prev_cend, prev_time = 0, 0.0
    for k, sw in enumerate(swords):
        cs, ce = sw["cstart"], sw["cend"]
        ws, we = w_start[k], w_end[k]
        if cs > prev_cend:
            gap_n = cs - prev_cend
            gap_dt = (ws - prev_time) / gap_n if gap_n else 0.0
            for g in range(gap_n):
                starts[prev_cend + g] = prev_time + gap_dt * g
                ends[prev_cend + g] = prev_time + gap_dt * (g + 1)
        nc = max(ce - cs, 1)
        for c in range(nc):
            starts[cs + c] = ws + (we - ws) * c / nc
            ends[cs + c] = ws + (we - ws) * (c + 1) / nc
        prev_cend, prev_time = ce, we

    for c in range(prev_cend, n):
        starts[c] = prev_time
        ends[c] = prev_time

    return {"characters": list(script),
            "character_start_times_seconds": starts,
            "character_end_times_seconds": ends}


# ── Public entrypoint (matches pipeline.generate_narration shape) ─────────────

def generate_narration_fish(script, out_audio_path):
    """Generate narration.mp3 + return {"alignment": {...}} (char-level, ElevenLabs-shaped)."""
    wav = out_audio_path + ".wav"
    _generate_wav(script, wav)
    with step(_ST, "mp3_encode", kind="cpu"):
        subprocess.run(["ffmpeg", "-y", "-i", wav, "-codec:a", "libmp3lame", "-qscale:a", "2",
                        out_audio_path], capture_output=True, check=True)
    try:
        os.remove(wav)
    except OSError:
        pass
    alignment = _align_to_chars(out_audio_path, script)
    return {"alignment": alignment}

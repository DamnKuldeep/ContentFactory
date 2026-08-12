#!/usr/bin/env python3
"""
fish_narration.py — Generate narration audio + character-level transcript
                    using Fish Speech S2 Pro (local GPU, NF4 4-bit).

Produces narration.mp3 + transcript.json for each story, in the same format
expected by narration_grounded_exp.py and video_composer.py.

First-run setup (automatic):
  - Clones groxaxo/fish-speech-int4-patch into ContentFactory/models/
  - Downloads S2 Pro checkpoints from HuggingFace (groxaxo/s2-pro)
  - Installs whisperx into the active Python env

Two-phase pipeline (sequential, VRAM-safe):
  Phase 1 — Fish S2 Pro: generate narration WAV for all stories
  Phase 2 — WhisperX:    align audio → char-level transcript.json

Usage:
  python fish_narration.py [--stories 1 2 3] [--preset balanced]
                           [--skip-existing] [--voice-ref /path/to/ref.mp3]
"""

import argparse
import difflib
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_CF_ROOT      = _SCRIPT_DIR.parent                       # ContentFactory/
_MODELS_DIR   = _CF_ROOT / 'models'
FISH_REPO_DIR = _MODELS_DIR / 'fish-speech-int4-patch'   # cloned repo
CKPT_DIR      = _MODELS_DIR / 'fish_s2_pro_ckpt'         # model.pth + codec.pth
NB_DIR        = _CF_ROOT / 'Notebooks' / 'Fish_S2_Pro'
REF_AUDIO     = NB_DIR / 'reference.mp3'
REF_TEXT_FILE = NB_DIR / 'Reference_text.txt'
VOICE_CACHE   = NB_DIR / '.voice_cache'

FISH_REPO_URL = 'https://github.com/groxaxo/fish-speech-int4-patch.git'
FISH_HF_REPO  = 'groxaxo/s2-pro'
OUT_ROOT      = os.path.expanduser('~/bt/outputs_narration_exp')
PY            = sys.executable

# ── GENERATION PRESETS ────────────────────────────────────────────────────────
PRESETS = {
    'balanced':   dict(temperature=0.65, top_p=0.88, top_k=30, repetition_penalty=1.12, speed=1.00),
    'deep':       dict(temperature=0.55, top_p=0.82, top_k=25, repetition_penalty=1.18, speed=0.92),
    'expressive': dict(temperature=0.75, top_p=0.92, top_k=35, repetition_penalty=1.08, speed=1.00),
    'fast':       dict(temperature=0.60, top_p=0.86, top_k=30, repetition_penalty=1.10, speed=1.12),
}

# ── FIRST-RUN SETUP ──────────────────────────────────────────────────────────

def _run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def ensure_setup():
    """Clone repo, download checkpoints, install whisperx — safe to call every run."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clone fish-speech-int4-patch
    if not (FISH_REPO_DIR / 'fish_speech').exists():
        print('Cloning fish-speech-int4-patch...')
        _run(['git', 'clone', '--depth', '1', FISH_REPO_URL, str(FISH_REPO_DIR)])
    else:
        print(f'[ok] fish-speech repo at {FISH_REPO_DIR}')

    # Add to path immediately so subsequent imports work
    if str(FISH_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(FISH_REPO_DIR))

    # 2. Download S2 Pro checkpoints (model.pth + codec.pth + config.json)
    if not (CKPT_DIR / 'config.json').exists():
        print(f'Downloading {FISH_HF_REPO} checkpoints (~4.7 GB)...')
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=FISH_HF_REPO,
            local_dir=str(CKPT_DIR),
            ignore_patterns=['*.gguf'],
        )
        print(f'Checkpoints saved to {CKPT_DIR}')
    else:
        print(f'[ok] checkpoints at {CKPT_DIR}')

    # 3. Install fish_speech package deps
    reqs = FISH_REPO_DIR / 'requirements.txt'
    if reqs.exists():
        _run([PY, '-m', 'pip', 'install', '-q', '-r', str(reqs)])

    # 4. Install whisperx for alignment
    try:
        import whisperx  # noqa
        print('[ok] whisperx installed')
    except ImportError:
        print('Installing whisperx...')
        _run([PY, '-m', 'pip', 'install', '-q', 'whisperx'])
        print('[ok] whisperx installed')


# ── FISH S2 PRO MODEL ─────────────────────────────────────────────────────────

def load_fish(device='cuda'):
    import torch
    if str(FISH_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(FISH_REPO_DIR))
    from fish_speech.models.text2semantic.inference import init_model, load_codec_model

    print(f'\nLoading Fish S2 Pro (NF4 4-bit)...')
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path=CKPT_DIR,
        device=device,
        precision=torch.float16,
        compile=False,
        max_length=4096,
        bnb4=True,
    )
    codec = load_codec_model(
        CKPT_DIR / 'codec.pth',
        device=device,
        precision=torch.float16,
    )
    print(f'Fish S2 Pro loaded in {time.time() - t0:.0f}s')
    return model, decode_one_token, codec


def unload_fish(model, codec):
    import torch
    try:
        model.cpu()
    except Exception:
        pass
    del model, codec
    torch.cuda.empty_cache()
    gc.collect()
    print('Fish S2 Pro unloaded from GPU')


# ── REFERENCE VOICE ───────────────────────────────────────────────────────────

def get_prompt_tokens(ref_audio: Path, codec, device: str) -> object:
    """Encode reference audio → prompt tokens, with .pt cache."""
    import torch
    from fish_speech.models.text2semantic.inference import encode_audio

    cache = VOICE_CACHE / 'prompt_tokens.pt'
    VOICE_CACHE.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        print(f'[ok] prompt_tokens cached at {cache}')
        return torch.load(str(cache), map_location='cpu')

    print(f'Encoding reference voice from {ref_audio}...')
    tokens = encode_audio(str(ref_audio), codec, device).cpu()
    torch.save(tokens, str(cache))
    print(f'Prompt tokens saved to {cache}')
    return tokens


# ── GENERATION ────────────────────────────────────────────────────────────────

# Long-form generation strategy:
#   Fish S2 Pro's context is 4096 tokens with 2048 reserved for generation, so a
#   prompt must stay <= 2048 tokens. generate_long's iterative_prompt mode
#   ACCUMULATES each generated chunk back into the conversation, so a long
#   (~111s) narration overflows context partway through. Instead we split the
#   script into small sentence-chunks and generate each INDEPENDENTLY against the
#   same fixed reference voice (no accumulation), then concatenate the audio.
#   Each chunk's prompt = reference (~775 tok) + short text → well under 2048.
CHUNK_MAX_BYTES = 240   # target per-chunk size in UTF-8 bytes

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _split_script_chunks(script: str, max_bytes: int = CHUNK_MAX_BYTES) -> list:
    """
    Split a script into sentence-grouped chunks, each <= max_bytes UTF-8 bytes.
    Whole sentences are kept intact; an over-long single sentence stands alone.
    """
    chunks = []
    for para in script.split('\n'):
        para = para.strip()
        if not para:
            continue
        cur = ''
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            cand = (cur + ' ' + sent).strip() if cur else sent
            if cur and len(cand.encode('utf-8')) > max_bytes:
                chunks.append(cur)
                cur = sent
            else:
                cur = cand
        if cur:
            chunks.append(cur)

    if not chunks:                      # fallback: whole script as one chunk
        chunks = [script.strip()]
    return chunks


def generate_wav(
    script: str,
    model, decode_one_token, codec,
    prompt_tokens, prompt_text: str,
    device='cuda',
    temperature=0.65, top_p=0.88, top_k=30,
    repetition_penalty=1.12, max_new_tokens=0,
    speed=1.0, seed=42,
    out_wav: Path = None,
) -> Path:
    """Run Fish S2 Pro inference → WAV file (full-length via per-chunk generation)."""
    import torch
    import soundfile as sf
    from fish_speech.models.text2semantic.inference import generate_long, decode_to_audio

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    chunks = _split_script_chunks(script)

    # Generate each chunk independently against the fixed reference voice.
    segments = []
    for ci, chunk in enumerate(chunks):
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=chunk,                      # plain text, no speaker tag → 1 batch
            num_samples=1,
            max_new_tokens=max_new_tokens,   # 0 = generate to EOS
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=False,          # NO context accumulation
            chunk_length=0,
            prompt_text=[prompt_text],
            prompt_tokens=[prompt_tokens],
        )
        cur = [resp.codes for resp in generator if resp.action == 'sample']
        if cur:
            segments.append(torch.cat(cur, dim=1))

    if not segments:
        raise RuntimeError('Fish S2 Pro generated no audio codes')

    codes = torch.cat(segments, dim=1)
    audio = decode_to_audio(codes.to(device), codec)

    if out_wav is None:
        out_wav = Path(tempfile.mktemp(suffix='.wav'))
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio.cpu().float().numpy(), codec.sample_rate)

    # Optional speed adjustment via FFmpeg atempo
    if abs(speed - 1.0) > 0.01:
        adj = out_wav.with_suffix('._adj.wav')
        _run(['ffmpeg', '-y', '-i', str(out_wav),
              '-filter:a', _atempo_chain(speed), str(adj)],
             capture_output=True)
        out_wav.unlink()
        adj.rename(out_wav)

    # Truncation guard: warn loudly if audio is implausibly short for the script.
    dur = _audio_seconds(out_wav)
    expected_min = 0.7 * len(script) / 18.0      # coarse lower bound (~18 chars/s)
    print(f'    [gen] {len(chunks)} chunks → {dur:.1f}s audio for {len(script)} chars')
    if dur < expected_min:
        print(f'    [WARN] audio {dur:.1f}s < expected min {expected_min:.1f}s '
              f'— narration may be truncated!')

    return out_wav


def _audio_seconds(path: Path) -> float:
    """Probe audio duration in seconds via ffprobe."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _atempo_chain(speed: float) -> str:
    """Build atempo chain clamped to [0.5, 2.0] per filter stage."""
    filters = []
    while speed > 2.0:
        filters.append('atempo=2.0')
        speed /= 2.0
    while speed < 0.5:
        filters.append('atempo=0.5')
        speed /= 0.5
    filters.append(f'atempo={speed:.4f}')
    return ','.join(filters)


def wav_to_mp3(wav_path: Path, mp3_path: Path):
    _run(['ffmpeg', '-y', '-i', str(wav_path),
          '-codec:a', 'libmp3lame', '-qscale:a', '2', str(mp3_path)],
         capture_output=True)


# ── WHISPERX ALIGNMENT ────────────────────────────────────────────────────────

def load_whisperx(device='cuda'):
    import whisperx
    print('\nLoading WhisperX (base)...')
    model = whisperx.load_model('base', device, compute_type='float16')
    return model


def unload_whisperx(model):
    import torch
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print('WhisperX unloaded from GPU')


def align_to_chars(mp3_path: str, script: str, wx_model, device='cuda') -> dict:
    """Transcribe + word-align → char-level transcript.json dict."""
    import whisperx

    audio  = whisperx.load_audio(mp3_path)
    result = wx_model.transcribe(audio, batch_size=8)
    lang   = result.get('language', 'en')

    align_model, metadata = whisperx.load_align_model(
        language_code=lang, device=device
    )
    result = whisperx.align(
        result['segments'], align_model, metadata,
        audio, device, return_char_alignments=False,
    )
    del align_model
    import torch; torch.cuda.empty_cache()

    word_segs = result.get('word_segments', [])
    return _words_to_char_alignment(script, word_segs)


_NORM_RE = re.compile(r'[^a-z0-9]')


def _norm(tok: str) -> str:
    """Normalize a token for fuzzy matching (lowercase, strip non-alphanumerics)."""
    return _NORM_RE.sub('', tok.lower())


def _script_words(script: str) -> list:
    """Tokenize script into words, each carrying its (cstart, cend) char span."""
    out = []
    for m in re.finditer(r'\S+', script):
        out.append({'text': m.group(0), 'cstart': m.start(), 'cend': m.end()})
    return out


def _words_to_char_alignment(script: str, word_segs: list) -> dict:
    """
    Map WhisperX *heard* word timings onto the *script* characters robustly.

    Strategy (handles ASR↔script divergence, mispronunciations, dropped words):
      1. Tokenize the script into words with char spans.
      2. difflib-align the normalized heard-word sequence to the normalized
         script-word sequence; matched script words inherit the heard word's
         [start, end]. Unmatched script-word runs are linearly interpolated
         between the nearest anchored neighbors.
      3. Force monotonic non-decreasing word times.
      4. Expand word [start, end] across each word's chars; fill inter-word
         whitespace by interpolation between neighbors.

    Output arrays have exactly len(script) entries (so the composer's
    aligned_ok check stays True) and are monotonic across the full audio.
    """
    n = len(script)
    starts = [0.0] * n
    ends   = [0.0] * n

    swords = _script_words(script)
    nsw    = len(swords)
    if nsw == 0:
        return {'characters': list(script),
                'character_start_times_seconds': starts,
                'character_end_times_seconds':   ends}

    # Heard words (skip empties / those without timing)
    heard = []
    for seg in word_segs:
        w = (seg.get('word') or '').strip()
        if not w:
            continue
        t0 = seg.get('start')
        t1 = seg.get('end')
        if t0 is None and t1 is None:
            continue
        t0 = float(t0) if t0 is not None else float(t1)
        t1 = float(t1) if t1 is not None else t0
        if t1 < t0:
            t1 = t0
        heard.append({'norm': _norm(w), 'start': t0, 'end': t1})
    heard = [h for h in heard if h['norm']]

    audio_end = heard[-1]['end'] if heard else 0.0

    w_start = [None] * nsw
    w_end   = [None] * nsw

    if heard:
        a = [_norm(sw['text']) for sw in swords]   # script tokens (normalized)
        b = [h['norm'] for h in heard]             # heard tokens (normalized)
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    w_start[i1 + k] = heard[j1 + k]['start']
                    w_end[i1 + k]   = heard[j1 + k]['end']
            elif tag == 'replace':
                # Spread the heard span for this region across the script words.
                hs = heard[j1]['start']
                he = heard[j2 - 1]['end'] if j2 > j1 else hs
                cnt = i2 - i1
                for k in range(cnt):
                    w_start[i1 + k] = hs + (he - hs) * k / cnt
                    w_end[i1 + k]   = hs + (he - hs) * (k + 1) / cnt
            # 'delete' (script words with no heard match) and 'insert'
            # (extra heard words) are left as gaps → interpolated below.

    # Interpolate unanchored script words between nearest anchored neighbors.
    def _prev_anchor(idx):
        j = idx
        while j >= 0 and w_start[j] is None:
            j -= 1
        return j
    def _next_anchor(idx):
        j = idx
        while j < nsw and w_start[j] is None:
            j += 1
        return j

    i = 0
    while i < nsw:
        if w_start[i] is not None:
            i += 1
            continue
        p = _prev_anchor(i - 1)
        q = _next_anchor(i)
        lo = w_end[p] if p >= 0 else 0.0
        hi = w_start[q] if q < nsw else audio_end
        if hi < lo:
            hi = lo
        run = q - i if q < nsw else nsw - i
        for k in range(run):
            w_start[i + k] = lo + (hi - lo) * k / (run + 1)
            w_end[i + k]   = lo + (hi - lo) * (k + 1) / (run + 1)
        i += run

    # Force monotonic non-decreasing.
    last = 0.0
    for k in range(nsw):
        if w_start[k] < last:
            w_start[k] = last
        if w_end[k] < w_start[k]:
            w_end[k] = w_start[k]
        last = w_end[k]

    # Expand word timings → per-char arrays.
    prev_cend = 0
    prev_time = 0.0
    for k, sw in enumerate(swords):
        cs, ce = sw['cstart'], sw['cend']
        ws, we = w_start[k], w_end[k]

        # Fill the gap chars (whitespace/punctuation) before this word.
        if cs > prev_cend:
            gap_n  = cs - prev_cend
            gap_dt = (ws - prev_time) / gap_n if gap_n else 0.0
            for g in range(gap_n):
                starts[prev_cend + g] = prev_time + gap_dt * g
                ends[prev_cend + g]   = prev_time + gap_dt * (g + 1)

        # Distribute the word's chars uniformly over [ws, we].
        nc = max(ce - cs, 1)
        for c in range(nc):
            starts[cs + c] = ws + (we - ws) * c / nc
            ends[cs + c]   = ws + (we - ws) * (c + 1) / nc

        prev_cend = ce
        prev_time = we

    # Trailing chars after the last word.
    for c in range(prev_cend, n):
        starts[c] = prev_time
        ends[c]   = prev_time

    return {
        'characters':                    list(script),
        'character_start_times_seconds': starts,
        'character_end_times_seconds':   ends,
    }


# ── STORY DISCOVERY ───────────────────────────────────────────────────────────

def discover_stories(out_root: str, story_filter=None) -> list:
    dirs = sorted(
        d for d in os.listdir(out_root)
        if os.path.isdir(os.path.join(out_root, d))
        and os.path.exists(os.path.join(out_root, d, 'source.json'))
    )
    if story_filter:
        dirs = [d for i, d in enumerate(dirs, 1) if i in story_filter]
    return [os.path.join(out_root, d) for d in dirs]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Generate narration + transcript using Fish Speech S2 Pro.')
    ap.add_argument('--out-root',      default=OUT_ROOT,
                    help='Root folder containing story subdirs')
    ap.add_argument('--stories',       nargs='+', type=int, metavar='N',
                    help='1-based story indices (default: all)')
    ap.add_argument('--preset',        choices=list(PRESETS), default='balanced',
                    help='Generation preset (default: balanced)')
    ap.add_argument('--voice-ref',     default=str(REF_AUDIO),
                    help='Path to reference voice audio')
    ap.add_argument('--voice-text',    default=None,
                    help='Reference transcript text (default: Reference_text.txt)')
    ap.add_argument('--max-tokens',    type=int, default=0,
                    help='Max new tokens PER chunk (0 = until EOS, recommended)')
    ap.add_argument('--seed',          type=int, default=42)
    ap.add_argument('--device',        default='cuda')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip story if narration.mp3 + transcript.json already exist')
    ap.add_argument('--no-setup',      action='store_true',
                    help='Skip first-run setup check')
    args = ap.parse_args()

    if not args.no_setup:
        ensure_setup()

    # Reference voice text
    ref_text = args.voice_text
    if ref_text is None:
        ref_text = (REF_TEXT_FILE.read_text().strip()
                    if REF_TEXT_FILE.exists() else '')

    preset       = PRESETS[args.preset]
    story_filter = set(args.stories) if args.stories else None
    story_dirs   = discover_stories(args.out_root, story_filter)

    print(f'\nFish Narration — {len(story_dirs)} stories | preset={args.preset}'
          f' | device={args.device}')

    # ── PHASE 1: Fish S2 Pro TTS ───────────────────────────────────────────
    print('\n══ Phase 1: Fish S2 Pro TTS ══════════════════════════════════════')
    model, decode_one_token, codec = load_fish(args.device)
    ref_audio    = Path(args.voice_ref)
    prompt_tokens = get_prompt_tokens(ref_audio, codec, args.device)

    wav_map = {}   # story_dir → temp WAV path (or None if skipped)

    for idx, story_dir in enumerate(story_dirs, 1):
        slug     = os.path.basename(story_dir)
        mp3_path = Path(story_dir) / 'narration.mp3'
        trs_path = Path(story_dir) / 'transcript.json'
        wav_path = Path(story_dir) / '_fish_narration.wav'

        if args.skip_existing and mp3_path.exists() and trs_path.exists():
            print(f'  [{idx}/{len(story_dirs)}] [skip] {slug}')
            wav_map[story_dir] = None
            continue

        source = json.load(open(os.path.join(story_dir, 'source.json')))
        script = source.get('script', '').strip()
        if not script:
            print(f'  [{idx}/{len(story_dirs)}] [WARN] empty script — skipping')
            wav_map[story_dir] = None
            continue

        print(f'  [{idx}/{len(story_dirs)}] {slug} ({len(script)} chars)')
        t0 = time.time()
        try:
            wav = generate_wav(
                script=script,
                model=model, decode_one_token=decode_one_token, codec=codec,
                prompt_tokens=prompt_tokens, prompt_text=ref_text,
                device=args.device,
                max_new_tokens=args.max_tokens,
                seed=args.seed,
                out_wav=wav_path,
                **preset,
            )
            wav_to_mp3(wav, mp3_path)
            size_mb = mp3_path.stat().st_size / 1e6
            print(f'    → narration.mp3  ({size_mb:.1f} MB)  [{time.time()-t0:.0f}s]')
            wav_map[story_dir] = wav
        except Exception as e:
            print(f'    ✗ FAILED: {e}')
            wav_map[story_dir] = None

    unload_fish(model, codec)

    # ── PHASE 2: WhisperX alignment ────────────────────────────────────────
    print('\n══ Phase 2: WhisperX alignment ═══════════════════════════════════')
    wx_model = load_whisperx(args.device)

    for idx, story_dir in enumerate(story_dirs, 1):
        slug     = os.path.basename(story_dir)
        mp3_path = Path(story_dir) / 'narration.mp3'
        trs_path = Path(story_dir) / 'transcript.json'

        if args.skip_existing and trs_path.exists():
            print(f'  [{idx}/{len(story_dirs)}] [skip] {slug}')
            continue
        if not mp3_path.exists():
            print(f'  [{idx}/{len(story_dirs)}] [skip] no mp3 — {slug}')
            continue

        source = json.load(open(os.path.join(story_dir, 'source.json')))
        script = source.get('script', '').strip()

        print(f'  [{idx}/{len(story_dirs)}] {slug}', end=' ', flush=True)
        t0 = time.time()
        try:
            alignment = align_to_chars(str(mp3_path), script, wx_model, args.device)
            json.dump(alignment, open(str(trs_path), 'w'), indent=2)
            n_chars = len(alignment['characters'])
            duration = alignment['character_end_times_seconds'][-1] if n_chars else 0
            print(f'→ {n_chars} chars | {duration:.1f}s  [{time.time()-t0:.0f}s]')
        except Exception as e:
            print(f'✗ FAILED: {e}')

        # Clean up temp WAV
        wav = wav_map.get(story_dir)
        if wav and Path(str(wav)).exists():
            Path(str(wav)).unlink()

    unload_whisperx(wx_model)
    print('\nDone. Run narration_grounded_exp.py --skip-existing to generate BGM briefs.')


if __name__ == '__main__':
    main()

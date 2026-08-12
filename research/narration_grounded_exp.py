"""
narration_grounded_exp.py — Narration-grounded BGM experiments.

For each story:
  1. Generate narration audio + transcript via ElevenLabs
  2. Extract pacing / energy features from the narration
  3. Exp 1: 4 prompting techniques → Qwen writes ACE-Step brief → generate music
  4. Exp 2: Same 4 techniques, 2 passes each (pass 2 refines using ACE-Step LM blueprint)

Usage:
    python narration_grounded_exp.py
    python narration_grounded_exp.py --stories 1 --no-install
    python narration_grounded_exp.py --dry-run        # no ElevenLabs / GPU calls
    python narration_grounded_exp.py --skip-existing  # resume aborted run
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile

# ── PATH / ENV BOOTSTRAP ──────────────────────────────────────────────────────
# Must happen before any shared imports that read API keys from config.py

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SCRIPT_DIR)           # ContentFactory/

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, '.env'))
except ImportError:
    pass  # assume env vars are already set

for _p in [
    _ROOT,                                                     # for `from shared.config import ...`
    os.path.join(_ROOT, 'shared'),                             # for `from llm import ...` etc.
    os.path.join(_ROOT, 'stages', 'stage_02_narration'),
    _SCRIPT_DIR,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── DEFAULTS ──────────────────────────────────────────────────────────────────

ZIP_PATH   = os.path.expanduser('~/bt/outputs.zip')
OUT_ROOT   = os.path.expanduser('~/bt/outputs_narration_exp')
REPO_DIR   = os.path.expanduser('~/bt/ACE-Step-1.5')
CKPT_DIR   = os.path.expanduser('~/bt/ace-checkpoints')
QWEN_LOCAL_MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'

DEFAULT_VOICE_ID = 'EXAVITQu4vr4xnSDxMaL'

TECHNIQUES = [
    'techA_script_only',
    'techB_pacing_wps',
    'techC_energy_envelope',
    'techD_scene_structured',
]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Narration-grounded BGM experiments')
    p.add_argument('--zip',           default=ZIP_PATH,
                   help='Path to outputs.zip')
    p.add_argument('--out-dir',       default=OUT_ROOT,
                   help='Output root directory')
    p.add_argument('--repo-dir',      default=REPO_DIR)
    p.add_argument('--ckpt-dir',      default=CKPT_DIR)
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--stories',       nargs='+', type=int, metavar='N',
                   help='1-based indices of stories to run (default: all)')
    p.add_argument('--no-install',    action='store_true',
                   help='Skip pip install step')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip outputs that already exist (resume mode)')
    p.add_argument('--dry-run',       action='store_true',
                   help='Check imports and feature extraction only — no API / GPU calls')
    return p.parse_args()

# ── ZIP EXTRACTION ────────────────────────────────────────────────────────────

def extract_fresh_copy(zip_path, out_root, skip_existing=False):
    # If skip_existing and stories are already there, reuse them
    if skip_existing and os.path.isdir(out_root):
        stories = sorted(
            e.path for e in os.scandir(out_root)
            if e.is_dir() and os.path.exists(os.path.join(e.path, 'source.json'))
        )
        if stories:
            print(f'[skip] {out_root} already has {len(stories)} story folders — reusing')
            return stories

    print(f'Extracting {os.path.basename(zip_path)} -> {out_root}')
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_root)
    # zip extracts to out_root/outputs/<slug>/ — flatten one level
    inner = os.path.join(out_root, 'outputs')
    if os.path.isdir(inner):
        for entry in os.scandir(inner):
            shutil.move(entry.path, out_root)
        os.rmdir(inner)
    stories = sorted(
        e.path for e in os.scandir(out_root)
        if e.is_dir() and os.path.exists(os.path.join(e.path, 'source.json'))
    )
    print(f'Found {len(stories)} story folders')
    return stories

# ── NARRATION GENERATION ──────────────────────────────────────────────────────

def gen_narration(story_dir, source_data, skip_existing=False):
    narration_path  = os.path.join(story_dir, 'narration.mp3')
    transcript_path = os.path.join(story_dir, 'transcript.json')

    if skip_existing and os.path.exists(narration_path) and os.path.exists(transcript_path):
        print('  [skip] narration.mp3 + transcript.json already exist')
        with open(transcript_path) as f:
            return json.load(f)

    from pipeline import generate_narration

    script   = source_data['script']
    voice_id = source_data.get('meta', {}).get('voice_id', DEFAULT_VOICE_ID)

    print(f'  Calling ElevenLabs TTS ({len(script.split())} words)...')
    result    = generate_narration(script, narration_path, voice_id)
    alignment = result.get('alignment', {})

    with open(transcript_path, 'w') as f:
        json.dump(alignment, f)

    print(f'  narration.mp3 saved ({os.path.getsize(narration_path) / 1e6:.1f} MB)')
    return alignment

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def compute_pacing_wps(alignment, window_sec=5):
    """Words-per-second per window from ElevenLabs char-level alignment."""
    chars  = alignment.get('characters', [])
    starts = alignment.get('character_start_times_seconds', [])
    ends   = alignment.get('character_end_times_seconds', [])
    if not chars or not starts or len(chars) != len(starts):
        return []

    # Reconstruct word timestamps from character-level data
    words        = []
    word_chars   = []
    word_start_t = None
    for i, ch in enumerate(chars):
        if ch in (' ', '\n', '\r', '\t'):
            if word_chars:
                words.append({'start': word_start_t, 'end': ends[i - 1]})
                word_chars   = []
                word_start_t = None
        else:
            if word_start_t is None:
                word_start_t = starts[i]
            word_chars.append(ch)
    if word_chars and word_start_t is not None:
        words.append({'start': word_start_t, 'end': ends[-1]})

    if not words:
        return []

    total_dur = ends[-1]
    result    = []
    i         = 0
    while i * window_sec < total_dur:
        w_start = i * window_sec
        w_end   = min((i + 1) * window_sec, total_dur)
        count   = sum(1 for w in words if w_start <= w['start'] < w_end)
        actual  = w_end - w_start
        wps     = count / actual if actual > 0 else 0.0
        label   = ('calm'     if wps < 1.5 else
                   'moderate' if wps < 2.5 else
                   'brisk'    if wps < 3.5 else 'rapid')
        result.append({
            'start': round(w_start, 1),
            'end':   round(w_end,   1),
            'wps':   round(wps,     2),
            'label': label,
        })
        i += 1
    return result


def _pacing_arc(wps_profile):
    if not wps_profile:
        return 'UNKNOWN'
    v = [w['wps'] for w in wps_profile]
    n = len(v)
    f = sum(v[:n // 3]) / max(1, n // 3)
    m = sum(v[n // 3: 2 * n // 3]) / max(1, n // 3)
    l = sum(v[2 * n // 3:]) / max(1, n - 2 * n // 3)
    if f < m > l:   return 'CALM → PEAK → RESOLVE'
    if f < m < l:   return 'STEADY BUILD'
    if f > m > l:   return 'OPENS INTENSE → QUIETS'
    if l > max(f, m): return 'BUILDS TO CLIMAX'
    return 'VARIED'


def compute_energy_envelope(narration_mp3, window_sec=2):
    """librosa RMS per window, normalised 0→1."""
    import librosa
    import numpy as np

    y, sr     = librosa.load(narration_mp3, sr=22050, mono=True)
    hop       = int(sr * window_sec)
    frame_len = min(int(sr * window_sec * 2), len(y))
    rms       = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    times     = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
    max_rms   = float(np.max(rms)) or 1.0
    rms_norm  = rms / max_rms
    q25, q50, q75 = (float(np.percentile(rms_norm, p)) for p in (25, 50, 75))

    result = []
    for t, r in zip(times, rms_norm):
        r     = float(r)
        label = ('whisper' if r < q25 else
                 'quiet'   if r < q50 else
                 'medium'  if r < q75 else 'loud')
        result.append({'time_sec': round(float(t), 1), 'rms_norm': round(r, 3), 'label': label})
    return result


_HIGH_KEYWORDS = frozenset({
    'fight', 'battle', 'death', 'died', 'dead', 'kill', 'murder', 'scream',
    'shock', 'crash', 'explode', 'explosion', 'betrayal', 'reveal', 'collapse',
    'desperate', 'rage', 'furious', 'terror', 'horror', 'flee', 'escape',
    'attack', 'strike', 'blood', 'wound', 'cry', 'sob', 'confess', 'discover',
    'sacrifice', 'condemn', 'execution', 'burning', 'fire', 'chaos', 'panic',
    'howl', 'shriek', 'gasp', 'shatter', 'breaks', 'truth', 'drown', 'falls',
})


def compute_scene_energy_tiers(scenes):
    """LOW / MED / HIGH / PEAK per scene based on word count + keyword density."""
    import numpy as np

    scores = []
    for scene in scenes:
        text  = scene.get('narration', '').lower()
        words = [w.strip('.,!?;:\'"()') for w in text.split()]
        hits  = sum(1 for w in words if w in _HIGH_KEYWORDS)
        scores.append(len(words) + 3 * hits)

    if not scores:
        return []

    q33, q67, q90 = (float(np.percentile(scores, p)) for p in (33, 67, 90))
    result = []
    for i, (scene, score) in enumerate(zip(scenes, scores)):
        tier = ('PEAK' if score >= q90 else
                'HIGH' if score >= q67 else
                'MED'  if score >= q33 else 'LOW')
        result.append({
            'scene_idx':        i + 1,
            'narration_snippet': scene.get('narration', '')[:90],
            'energy':            tier,
        })
    return result

# ── QWEN SYSTEM PROMPTS ───────────────────────────────────────────────────────

# Base system prompt (ACE-Step parameter reference from stage_04_music/brief.py)
# extended with narration-sync instructions
SYSTEM_PROMPT = """\
You are a professional film-music supervisor and prompt engineer for ACE-Step 1.5,
a state-of-the-art text-to-music diffusion model.

Your task: read the narration script and any pacing/energy data provided,
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
5. Since the music will play under a narration voiceover, set music_volume ≤ 0.16.

━━━ Narration-sync instructions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The music MUST mirror the narration's energy arc:
• Calm/slow narration   → sparse texture, long sustained notes, minimal percussion
• Intense/fast narration → dense texture, faster harmonic rhythm, active percussion
• The overall arc shape must match: if narration builds then resolves, music must too
• Use any pacing/energy data provided to set BPM and describe the dynamics arc
  explicitly in the caption (e.g. "starts sparse, builds to full orchestration at 60s,
  then gradually dissolves into stillness")
• Match the story's emotional theme: dark/tragedy → minor keys, triumphant → major keys

━━━ Output schema ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "caption":         "<≤512 char string>",
  "tags":            "<comma-separated string>",
  "bpm":             <int 60-180 or null>,
  "keyscale":        "<string or \\"\\">" ,
  "inference_steps": <int 8-20>,
  "seed":            <int>,
  "batch_size":      <int 1-4>,
  "music_volume":    <float 0.10-0.30>
}"""

VERIFY_SYSTEM_PROMPT = SYSTEM_PROMPT + """

━━━ Verification & Refinement mode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You will be given a Pass 1 brief and ACE-Step's actual LM interpretation (the BPM
and key it chose). Critically evaluate whether the brief captured the narration's
energy arc and story theme, then generate a REFINED brief that fixes any weaknesses.

Return JSON with EXACTLY these two top-level keys (no other text, no fences):
{
  "verification": {
    "bpm_match":    <bool — was the BPM choice appropriate for the narration pace?>,
    "key_match":    <bool — was the key appropriate for the story's emotional tone?>,
    "issues":       "<what was missing or misaligned in the Pass 1 brief>",
    "improvements": "<what the refined brief addresses differently>"
  },
  "refined_brief": {
    "caption":         "<≤512 char string>",
    "tags":            "<comma-separated string>",
    "bpm":             <int 60-180 or null>,
    "keyscale":        "<string or \\"\\">" ,
    "inference_steps": <int 8-20>,
    "seed":            <int>,
    "batch_size":      <int 1-4>,
    "music_volume":    <float 0.10-0.30>
  }
}"""

# ── BRIEF HELPERS ─────────────────────────────────────────────────────────────

def _parse_json(raw):
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$',          '', cleaned, flags=re.MULTILINE).strip()
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Strip JS-style // comments (Qwen sometimes annotates fields)
        repaired = re.sub(r'//[^\n]*', '', cleaned)
        # Remove trailing commas before } or ]
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        return json.loads(repaired)


def _validate_brief(brief):
    assert isinstance(brief.get('caption'), str), 'caption must be a string'
    brief['caption']         = brief['caption'][:512]
    assert isinstance(brief.get('tags'), str),    'tags must be a string'
    if brief.get('bpm') is not None:
        brief['bpm']         = max(60,  min(180, int(brief['bpm'])))
    brief['inference_steps'] = max(8,   min(20,  int(brief.get('inference_steps', 12))))
    brief['batch_size']      = max(1,   min(4,   int(brief.get('batch_size', 1))))
    brief['music_volume']    = round(max(0.10, min(0.30, float(brief.get('music_volume', 0.15)))), 3)
    brief.setdefault('seed',      42)
    brief.setdefault('keyscale',  '')
    return brief

# ── USER PROMPT BUILDERS (4 techniques) ──────────────────────────────────────

def _base_user(duration, script):
    return (
        f'Target duration : {duration}s\n'
        f'Has voiceover   : yes (music plays under narration)\n\n'
        f'Narration script:\n"""\n{script}\n"""\n'
    )


def build_user_techA(story_data, duration, **_):
    return (
        _base_user(duration, story_data['script'])
        + '\nGenerate a music brief where the theme and energy arc match the story.\n'
          'Return pure JSON only — no explanation, no fences.'
    )


def build_user_techB(story_data, duration, pacing_wps, **_):
    if pacing_wps:
        rows = ['Narration speaking pace (words/second per 5s window):']
        for w in pacing_wps:
            rows.append(f'  {w["start"]:>5.0f}s–{w["end"]:.0f}s:  {w["wps"]:.2f} w/s  [{w["label"]}]')
        arc = _pacing_arc(pacing_wps)
        rows.append(f'Overall arc: {arc}')
        pacing_text = '\n'.join(rows)
    else:
        pacing_text = '(pacing data unavailable — use story context only)'

    return (
        _base_user(duration, story_data['script'])
        + f'\n{pacing_text}\n\n'
          'Set BPM and describe the caption dynamics to mirror this pace curve.\n'
          'Return pure JSON only — no explanation, no fences.'
    )


def build_user_techC(story_data, duration, energy_envelope, **_):
    if energy_envelope:
        # Subsample: one reading every 4s
        step    = max(1, 4 // 2)
        sampled = energy_envelope[::step][:40]
        parts   = [f'{e["time_sec"]}s:{e["rms_norm"]:.2f}({e["label"][0].upper()})' for e in sampled]
        table   = '  ' + ' | '.join(parts)

        # Compute per-third averages to describe overall arc
        vals   = [e['rms_norm'] for e in energy_envelope]
        n      = len(vals)
        thirds = [vals[:n // 3], vals[n // 3: 2 * n // 3], vals[2 * n // 3:]]
        avgs   = [sum(t) / len(t) if t else 0.0 for t in thirds]
        if avgs[1] > avgs[0] and avgs[1] > avgs[2]:
            arc_desc = 'energy peaks in the middle section'
        elif avgs[2] > avgs[0] and avgs[2] > avgs[1]:
            arc_desc = 'energy builds toward the end'
        elif avgs[0] > avgs[1] and avgs[0] > avgs[2]:
            arc_desc = 'energy is highest at the opening'
        else:
            arc_desc = 'energy varies throughout'

        envelope_text = (
            'Narration audio energy (librosa RMS, normalised 0→1) every 4 seconds\n'
            f'  Labels: W=whisper Q=quiet M=medium L=loud\n{table}\n'
            f'Overall: {arc_desc}.'
        )
    else:
        envelope_text = '(energy data unavailable — use story context only)'

    return (
        _base_user(duration, story_data['script'])
        + f'\n{envelope_text}\n\n'
          'The music loudness and density should track this energy envelope.\n'
          'Return pure JSON only — no explanation, no fences.'
    )


def build_user_techD(story_data, duration, scene_tiers, **_):
    if scene_tiers:
        n   = len(scene_tiers)
        dur_per_scene = duration / max(n, 1)

        # Sample representative scenes: first, quarters, peak, last
        peak_idx = next(
            (t['scene_idx'] - 1 for t in scene_tiers if t['energy'] == 'PEAK'), None
        )
        idx_set  = sorted(set(
            [0, n // 4, n // 2, 3 * n // 4, n - 1]
            + ([peak_idx] if peak_idx is not None else [])
        ))
        sampled  = [scene_tiers[i] for i in idx_set if i < n]

        rows = [f'Scene energy tiers ({n} scenes, {duration}s total):']
        for t in sampled:
            rows.append(f'  Scene {t["scene_idx"]:>3}  [{t["energy"]:>4}]:  "{t["narration_snippet"][:70]}"')

        # Act-level summary
        a1, a2  = n // 3, 2 * n // 3
        t1_s, t1_e = scene_tiers[0]['energy'],  scene_tiers[a1 - 1]['energy']
        t2_s, t2_e = scene_tiers[a1]['energy'], scene_tiers[a2 - 1]['energy']
        t3_s, t3_e = scene_tiers[a2]['energy'], scene_tiers[-1]['energy']
        acts = (
            f'Act I   (scenes 1–{a1},   0–{round(a1*dur_per_scene)}s):  {t1_s}→{t1_e}  |  '
            f'Act II  (scenes {a1+1}–{a2},  {round(a1*dur_per_scene)}–{round(a2*dur_per_scene)}s):  {t2_s}→{t2_e}  |  '
            f'Act III (scenes {a2+1}–{n}, {round(a2*dur_per_scene)}–{duration}s):  {t3_s}→{t3_e}'
        )
        rows.append(acts)
        tier_text = '\n'.join(rows)
    else:
        tier_text = '(scene data unavailable — use story context only)'

    return (
        _base_user(duration, story_data['script'])
        + f'\n{tier_text}\n\n'
          'Map this energy sequence to the music\'s dynamic arc:\n'
          '  LOW=sparse/minimal | MED=moderate texture | HIGH=full instrumentation | PEAK=maximum energy\n'
          'Return pure JSON only — no explanation, no fences.'
    )


TECH_BUILDERS = {
    'techA_script_only':      build_user_techA,
    'techB_pacing_wps':       build_user_techB,
    'techC_energy_envelope':  build_user_techC,
    'techD_scene_structured': build_user_techD,
}

# ── LOCAL QWEN (Qwen2.5-Omni-7B) ──────────────────────────────────────────────

def load_qwen_local():
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    print(f'\nLoading {QWEN_LOCAL_MODEL_ID}...')
    t0 = time.time()
    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_LOCAL_MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        QWEN_LOCAL_MODEL_ID,
        torch_dtype='auto',
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    try:
        model.disable_talker()  # text-only — no speech output head needed
    except Exception:
        pass
    print(f'{QWEN_LOCAL_MODEL_ID} loaded in {time.time()-t0:.0f}s')
    return model, processor


def unload_qwen_local(model):
    import torch
    try:
        model.cpu()
    except Exception:
        pass
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f'{QWEN_LOCAL_MODEL_ID} unloaded from GPU')


def _qwen_infer(model, processor, system_prompt, user_prompt, max_new_tokens=700):
    import torch
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError:
        process_mm_info = None

    conversation = [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
        {'role': 'user',   'content': [{'type': 'text', 'text': user_prompt}]},
    ]
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    if process_mm_info is not None:
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    else:
        audios = images = videos = None

    device = next(model.parameters()).device
    inputs = processor(
        text=text_prompt,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors='pt',
        padding=True,
    ).to(device)

    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.2,
        )
    return processor.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)[0].strip()


# ── QWEN CALLS ────────────────────────────────────────────────────────────────

def call_qwen_brief(user_prompt, qwen_model, qwen_processor, retries=3):
    for attempt in range(retries):
        try:
            raw   = _qwen_infer(qwen_model, qwen_processor, SYSTEM_PROMPT, user_prompt, max_new_tokens=1500)
            print(f'      [raw first 300]: {repr(raw[:300])}')
            brief = _parse_json(raw)
            return _validate_brief(brief)
        except Exception as e:
            print(f'    [Qwen attempt {attempt + 1}/{retries}] {e}')
            if attempt == retries - 1:
                raise RuntimeError(f'Failed to get valid brief from Qwen after {retries} attempts') from e


def call_qwen_pass2(user_prompt, qwen_model, qwen_processor, retries=3):
    for attempt in range(retries):
        try:
            raw    = _qwen_infer(qwen_model, qwen_processor, VERIFY_SYSTEM_PROMPT, user_prompt, max_new_tokens=2000)
            print(f'      [pass2 raw 0:1000]: {repr(raw[:1000])}')
            result = _parse_json(raw)
            verif  = result.get('verification', {})
            raw_brief2 = (result.get('refined_brief')
                          or next((v for k, v in result.items()
                                   if k.startswith('refined') and isinstance(v, dict)), None)
                          or {})
            brief2 = _validate_brief(raw_brief2)
            return verif, brief2
        except Exception as e:
            print(f'    [Qwen pass2 attempt {attempt + 1}/{retries}] {e}')
            if attempt == retries - 1:
                raise RuntimeError(f'Pass 2 Qwen failed after {retries} attempts') from e


def build_pass2_user(base_user_prompt, brief1, lm_blueprint):
    bp_bpm = lm_blueprint.get('bpm', 'unknown')
    bp_key = lm_blueprint.get('keyscale', 'unknown')
    return (
        base_user_prompt
        + f'\n--- Pass 1 brief (sent to ACE-Step) ---\n{json.dumps(brief1, indent=2)}\n\n'
        + f'--- ACE-Step LM interpretation ---\n'
        + f'  BPM chosen : {bp_bpm}\n'
        + f'  Key chosen : {bp_key}\n\n'
        + 'Verify whether the Pass 1 brief captured the narration\'s energy arc and story theme.\n'
          'Then produce a REFINED brief that addresses any weaknesses.\n'
          'Return pure JSON with "verification" and "refined_brief" keys — no prose, no fences.'
    )

# ── MUSIC GENERATION ──────────────────────────────────────────────────────────

def run_music_from_brief(brief, duration, out_dir, dit, ace_llm, default_seed):
    """Call ACE-Step, move output to out_dir/music.mp3, return (result, lm_blueprint)."""
    from ace_story_grounded import make_music

    seed = brief.get('seed', default_seed)

    tmp = tempfile.mkdtemp(dir=out_dir, prefix='_tmp_ace_')
    try:
        result = make_music(
            dit, ace_llm, tmp,
            caption         = brief['caption'],
            tags            = brief.get('tags', ''),
            duration        = duration,
            bpm             = brief.get('bpm'),
            keyscale        = brief.get('keyscale', ''),
            inference_steps = brief.get('inference_steps', 12),
            seed            = seed,
        )
        if result is None or not result.audios:
            return None, {}

        src = result.audios[0].get('path', '')
        if src and os.path.exists(src):
            dst = os.path.join(out_dir, 'music.mp3')
            shutil.move(src, dst)
            result.audios[0]['path'] = dst

        lm_meta = result.extra_outputs.get('lm_metadata') or {}
        return result, lm_meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ── EXPERIMENT RUNNERS ────────────────────────────────────────────────────────

def gen_brief(tech, user_prompt, tech_dir, qwen_model, qwen_processor, skip_existing):
    """Phase 1 helper: generate and save brief.json for one technique."""
    brief_path = os.path.join(tech_dir, 'brief.json')
    if skip_existing and os.path.exists(brief_path):
        print(f'    [skip brief] {tech}')
        return json.load(open(brief_path))
    os.makedirs(tech_dir, exist_ok=True)
    print(f'    [{tech}] Qwen...', end=' ', flush=True)
    t0    = time.time()
    brief = call_qwen_brief(user_prompt, qwen_model, qwen_processor)
    with open(brief_path, 'w') as f:
        json.dump(brief, f, indent=2)
    print(f'{time.time()-t0:.1f}s | bpm={brief.get("bpm")} key="{brief.get("keyscale")}"')
    return brief


def gen_music(tech, brief, duration, tech_dir, dit, ace_llm, seed, skip_existing, label=''):
    """Phase 2/4 helper: generate music.mp3 from a saved brief."""
    music_path = os.path.join(tech_dir, 'music.mp3')
    if skip_existing and os.path.exists(music_path):
        print(f'    [skip music] {tech}{label}')
        return {}
    brief_path = os.path.join(tech_dir, 'brief.json')
    if not os.path.exists(brief_path):
        print(f'    [WARN] no brief.json for {tech}{label} — skipping')
        return {}
    brief = json.load(open(brief_path))
    print(f'    [{tech}{label}] ACE-Step generating...  caption: {brief["caption"][:80]}')
    _, lm_meta = run_music_from_brief(brief, duration, tech_dir, dit, ace_llm, seed)
    lm_meta = lm_meta or {}
    if lm_meta:
        with open(os.path.join(tech_dir, 'lm_blueprint.json'), 'w') as f:
            json.dump(lm_meta, f, indent=2)
        print(f'      LM chose: bpm={lm_meta.get("bpm")} key={lm_meta.get("keyscale")}')
    return lm_meta


# ── MAIN ──────────────────────────────────────────────────────────────────────

def _collect_story_context(story_dir, story_data, skip_existing):
    """Narration + feature extraction for one story. Returns feature dict."""
    scenes   = story_data['scenes']
    script   = story_data['script']
    duration = max(30, round(len(script.split()) / 150 * 60))

    alignment = gen_narration(story_dir, story_data, skip_existing=skip_existing)

    narration_mp3   = os.path.join(story_dir, 'narration.mp3')
    pacing_wps      = compute_pacing_wps(alignment)
    arc             = _pacing_arc(pacing_wps)
    energy_envelope = compute_energy_envelope(narration_mp3) if os.path.exists(narration_mp3) else []
    scene_tiers     = compute_scene_energy_tiers(scenes)
    tc = {t: sum(1 for s in scene_tiers if s['energy'] == t) for t in ['LOW', 'MED', 'HIGH', 'PEAK']}
    print(f'  pacing: {len(pacing_wps)} windows | arc: {arc}')
    print(f'  energy: {len(energy_envelope)} samples @ 2s intervals')
    print(f'  scene tiers: {tc}')

    features     = {'pacing_wps': pacing_wps, 'energy_envelope': energy_envelope, 'scene_tiers': scene_tiers}
    user_prompts = {tech: TECH_BUILDERS[tech](story_data, duration, **features) for tech in TECHNIQUES}
    return {'duration': duration, 'user_prompts': user_prompts}


def main():
    args = parse_args()
    skip = args.skip_existing

    print('=' * 70)
    print('Narration-Grounded BGM Experiments  (local Qwen + ACE-Step)')
    print('=' * 70)

    story_dirs = extract_fresh_copy(args.zip, args.out_dir, skip_existing=skip)
    if args.stories:
        story_dirs = [story_dirs[i - 1] for i in args.stories if i <= len(story_dirs)]

    print(f'\nProcessing {len(story_dirs)} stories:')
    for i, d in enumerate(story_dirs, 1):
        src = json.load(open(os.path.join(d, 'source.json')))
        title = src.get('meta', {}).get('creative_direction', {}).get('title', os.path.basename(d))
        print(f'  {i}. {title}')

    if args.dry_run:
        print('\n[DRY RUN] imports OK.')
        return

    # ── One-time ACE-Step env checks (no model load yet) ──────────────────
    from ace_story_grounded import (
        check_env, ensure_repo, install_deps, ensure_weights, setup_gpu, load_model,
    )
    check_env()
    ensure_repo(args.repo_dir)
    if not args.no_install:
        install_deps()
    ensure_weights(args.ckpt_dir)
    device = setup_gpu(args.repo_dir, args.ckpt_dir)

    total_t0    = time.time()
    story_ctxs  = {}   # story_dir → {duration, user_prompts}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Qwen: narration + all briefs (Exp1 + Exp2 pass1)
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('PHASE 1  Narration + Qwen brief generation (Exp1 + Exp2-pass1)')
    print('═' * 70)

    qwen_model, qwen_processor = load_qwen_local()

    for story_idx, story_dir in enumerate(story_dirs, 1):
        story_data = json.load(open(os.path.join(story_dir, 'source.json')))
        title = story_data.get('meta', {}).get('creative_direction', {}).get('title',
                                                                              os.path.basename(story_dir))
        print(f'\n── [{story_idx}/{len(story_dirs)}] {title}')
        ctx = _collect_story_context(story_dir, story_data, skip)
        story_ctxs[story_dir] = ctx
        user_prompts = ctx['user_prompts']

        # Exp1 briefs
        exp1_dir = os.path.join(story_dir, 'exp1_single_pass')
        for tech in TECHNIQUES:
            gen_brief(tech, user_prompts[tech],
                      os.path.join(exp1_dir, tech), qwen_model, qwen_processor, skip)

        # Exp2 pass1 briefs
        exp2_dir = os.path.join(story_dir, 'exp2_two_pass')
        for tech in TECHNIQUES:
            gen_brief(f'{tech} [exp2-p1]', user_prompts[tech],
                      os.path.join(exp2_dir, tech, 'pass1'), qwen_model, qwen_processor, skip)

    unload_qwen_local(qwen_model)
    del qwen_model, qwen_processor

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2 — ACE-Step: Exp1 music + Exp2 pass1 music (save lm_blueprints)
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('PHASE 2  ACE-Step — Exp1 + Exp2-pass1 music generation')
    print('═' * 70)

    dit, ace_llm = load_model(args.repo_dir, args.ckpt_dir, device)

    for story_idx, story_dir in enumerate(story_dirs, 1):
        ctx      = story_ctxs[story_dir]
        duration = ctx['duration']
        story_data = json.load(open(os.path.join(story_dir, 'source.json')))
        title = story_data.get('meta', {}).get('creative_direction', {}).get('title',
                                                                              os.path.basename(story_dir))
        print(f'\n── [{story_idx}/{len(story_dirs)}] {title}')

        exp1_dir = os.path.join(story_dir, 'exp1_single_pass')
        for tech in TECHNIQUES:
            gen_music(tech, None, duration, os.path.join(exp1_dir, tech),
                      dit, ace_llm, args.seed, skip)

        exp2_dir = os.path.join(story_dir, 'exp2_two_pass')
        for tech in TECHNIQUES:
            gen_music(tech, None, duration, os.path.join(exp2_dir, tech, 'pass1'),
                      dit, ace_llm, args.seed, skip, label=' [p1]')

    import torch, gc
    del dit, ace_llm
    torch.cuda.empty_cache(); gc.collect()
    print('\nACE-Step unloaded from GPU')

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3 — Qwen: Exp2 pass2 verify + refine
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('PHASE 3  Qwen — Exp2 pass2 verification + refinement')
    print('═' * 70)

    qwen_model, qwen_processor = load_qwen_local()

    for story_idx, story_dir in enumerate(story_dirs, 1):
        ctx          = story_ctxs[story_dir]
        user_prompts = ctx['user_prompts']
        story_data   = json.load(open(os.path.join(story_dir, 'source.json')))
        title = story_data.get('meta', {}).get('creative_direction', {}).get('title',
                                                                              os.path.basename(story_dir))
        print(f'\n── [{story_idx}/{len(story_dirs)}] {title}')
        exp2_dir = os.path.join(story_dir, 'exp2_two_pass')

        for tech in TECHNIQUES:
            pass1_dir   = os.path.join(exp2_dir, tech, 'pass1')
            pass2_dir   = os.path.join(exp2_dir, tech, 'pass2')
            brief2_path = os.path.join(pass2_dir, 'brief.json')
            if skip and os.path.exists(brief2_path):
                print(f'    [skip brief] {tech} [exp2-p2]')
                continue

            p1_brief_path = os.path.join(pass1_dir, 'brief.json')
            p1_bp_path    = os.path.join(pass1_dir, 'lm_blueprint.json')
            if not os.path.exists(p1_brief_path):
                print(f'    [WARN] no pass1 brief for {tech} — skip pass2')
                continue

            brief1      = json.load(open(p1_brief_path))
            lm_blueprint = json.load(open(p1_bp_path)) if os.path.exists(p1_bp_path) else {}

            print(f'    [{tech}] pass2 Qwen verify+refine...', end=' ', flush=True)
            t0 = time.time()
            p2_user = build_pass2_user(user_prompts[tech], brief1, lm_blueprint)
            verif, brief2 = call_qwen_pass2(p2_user, qwen_model, qwen_processor)
            print(f'{time.time()-t0:.1f}s | bpm_match={verif.get("bpm_match")} key_match={verif.get("key_match")}')

            os.makedirs(pass2_dir, exist_ok=True)
            json.dump(verif,  open(os.path.join(pass2_dir, 'verification.json'), 'w'), indent=2)
            json.dump(brief2, open(os.path.join(pass2_dir, 'brief.json'),        'w'), indent=2)

    unload_qwen_local(qwen_model)
    del qwen_model, qwen_processor

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4 — ACE-Step: Exp2 pass2 music
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('PHASE 4  ACE-Step — Exp2 pass2 music generation')
    print('═' * 70)

    dit, ace_llm = load_model(args.repo_dir, args.ckpt_dir, device)

    for story_idx, story_dir in enumerate(story_dirs, 1):
        ctx      = story_ctxs[story_dir]
        duration = ctx['duration']
        story_data = json.load(open(os.path.join(story_dir, 'source.json')))
        title = story_data.get('meta', {}).get('creative_direction', {}).get('title',
                                                                              os.path.basename(story_dir))
        print(f'\n── [{story_idx}/{len(story_dirs)}] {title}')
        exp2_dir = os.path.join(story_dir, 'exp2_two_pass')
        for tech in TECHNIQUES:
            gen_music(tech, None, duration, os.path.join(exp2_dir, tech, 'pass2'),
                      dit, ace_llm, args.seed, skip, label=' [p2]')

    # ── Final manifest ─────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print()
    print('=' * 70)
    print(f'ALL DONE  |  {total_elapsed / 60:.1f} min total')
    print('=' * 70)

    mp3s = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(args.out_dir)
        for f in files if f == 'music.mp3'
    )
    print(f'\n{len(mp3s)} music.mp3 files generated:')
    for p in mp3s:
        rel  = os.path.relpath(p, args.out_dir)
        size = os.path.getsize(p) / 1e6
        print(f'  {size:5.1f} MB  {rel}')

    manifest = [{'path': os.path.relpath(p, args.out_dir), 'mb': round(os.path.getsize(p) / 1e6, 2)}
                for p in mp3s]
    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    json.dump(manifest, open(manifest_path, 'w'), indent=2)
    print(f'\nManifest: {manifest_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
video_composer.py — Stitch narration + BGM experiments into vertical short-form videos.

For each story in outputs_narration_exp/, generates up to 12 MP4 files (one per BGM variant):
  exp1_techA.mp4  exp1_techB.mp4  exp1_techC.mp4  exp1_techD.mp4
  exp2_techA_p1.mp4  exp2_techA_p2.mp4  ...  exp2_techD_p2.mp4

Features:
  - 28 xfade transitions, energy-grounded per scene (LOW/MED/HIGH/PEAK)
  - Ken Burns panning on every frame (1.06× zoom, 8 pan directions)
  - ASS karaoke subtitles: per-word bounce+flash in story accent color
  - Sidechain-compressed audio (BGM ducks under narration)
  - h264_nvenc GPU encoding (fallback: libx264)
"""

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ── PATH BOOTSTRAP ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SCRIPT_DIR)   # ContentFactory/
_STAGE05    = os.path.join(_ROOT, 'stages', 'stage_05_compose')
for _p in [_ROOT, _STAGE05]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compose import (
    words_from_alignment,   # char-level ElevenLabs → word list
    cover_fit_png,          # PIL resize + center-crop to target WxH
    _fmt,                   # seconds → HH:MM:SS.cs for ASS
    _esc,                   # escape ASS special chars
    _ffmpeg_has_encoder,    # check NVENC availability
    _PANS,                  # 8 Ken Burns pan vectors
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
OUT_ROOT        = os.path.expanduser('~/bt/outputs_narration_exp')
W, H, FPS       = 1080, 1920, 30
OPEN_FADE       = 0.6   # fade-in at video start (s)
CLOSE_FADE      = 0.8   # fade-out at video end (s)
TAIL            = 0.6   # hold last frame after narration ends (s)
KEN_BURNS_ZOOM  = 1.06  # oversize factor for Ken Burns; crops back to W×H
FONT_NAME       = 'DejaVu Sans'   # resolved by libass/fontconfig
SUB_BASE_SIZE   = 66
SUB_POP_SCALE   = 115   # spoken word: 115% of base size (15% pop-out)
SUB_POP_RAMP    = 60    # ms for scale ramp-up animation
SUB_FALLBACK_COLOR = '#d97b5c'
SUB_WORD_COLOR     = '#FFD27F'   # warm gold — all context words
SUB_HIGHLIGHT_COLOR = '#FFFFFF'  # white pop — current spoken word

# ── VERIFIED XFADE TRANSITIONS (58 available in this ffmpeg build) ────────────
_ENERGY_TRANSITIONS = {
    'LOW':  ['fade', 'dissolve', 'fadegrays', 'fadeslow', 'distance'],
    'MED':  ['smoothleft', 'smoothright', 'smoothup', 'smoothdown',
             'horzopen', 'vertopen', 'coverleft', 'coverright'],
    'HIGH': ['wipeleft', 'wiperight', 'diagbr', 'diagtl', 'radial',
             'circleopen', 'coverleft', 'coverright', 'hlslice', 'vuslice'],
    'PEAK': ['pixelize', 'circleclose', 'zoomin', 'squeezev', 'hblur',
             'fadeblack', 'fadewhite', 'fadefast', 'rectcrop'],
}

_FULL_CYCLE = [
    'dissolve',   'fade',        'smoothleft',  'smoothright',
    'wipeleft',   'wiperight',   'diagbr',      'diagtl',
    'circleopen', 'radial',      'horzopen',    'vertopen',
    'slideleft',  'slideright',  'coverleft',   'coverright',
    'pixelize',   'zoomin',      'fadegrays',   'distance',
    'squeezev',   'squeezeh',    'hblur',       'fadeblack',
    'hlslice',    'vuslice',     'wipetr',      'wipetl',
]

# High-energy keywords for scene tier scoring (same set as narration_grounded_exp.py)
_HIGH_WORDS = {
    'fight', 'death', 'scream', 'shock', 'crash', 'betray', 'confess',
    'reveal', 'collapse', 'kill', 'murder', 'poison', 'fire', 'blood',
    'horror', 'escape', 'drown', 'fall', 'danger', 'attack',
}


# ── SCENE ENERGY SCORING ──────────────────────────────────────────────────────

def score_scenes(scenes):
    """Assign LOW/MED/HIGH/PEAK tier to each scene based on narration energy."""
    scores = []
    for s in scenes:
        words = re.sub(r'[^\w\s]', '', s.get('narration', '')).lower().split()
        hit   = sum(1 for w in words if any(k in w for k in _HIGH_WORDS))
        scores.append(len(words) + 3 * hit)

    n = len(scores)
    if n < 4:
        return ['MED'] * n
    srt = sorted(scores)
    q   = [srt[max(0, int(n * p) - 1)] for p in [0.25, 0.50, 0.75]]
    tiers = []
    for sc in scores:
        if   sc <= q[0]: tiers.append('LOW')
        elif sc <= q[1]: tiers.append('MED')
        elif sc <= q[2]: tiers.append('HIGH')
        else:            tiers.append('PEAK')
    return tiers


def pick_transitions(scenes):
    """Return one xfade transition name per scene boundary (length = n-1)."""
    tiers  = score_scenes(scenes)
    result = []
    for i, tier in enumerate(tiers[:-1]):
        pool = _ENERGY_TRANSITIONS[tier]
        result.append(pool[i % len(pool)])
    return result


# ── SCENE CHAR-POSITION BRIDGE ────────────────────────────────────────────────

def add_char_positions(scenes, script_text):
    """
    Add char_start / char_end to each scene dict by finding its narration
    snippet in the full script text.  Required by words_from_alignment mapping.
    """
    cursor = 0
    for scene in scenes:
        narr = scene.get('narration', '').strip()
        if not narr:
            scene['char_start'] = cursor
            scene['char_end']   = cursor
            continue
        pos = script_text.find(narr, cursor)
        if pos < 0 and len(narr) >= 20:
            pos = script_text.find(narr[:20], cursor)
        if pos < 0:
            pos = cursor
        scene['char_start'] = pos
        scene['char_end']   = pos + len(narr)
        cursor = pos + len(narr)


# ── ASS KARAOKE SUBTITLE BUILDER ──────────────────────────────────────────────

def _hex_to_ass(hex_color):
    """Convert #RRGGBB to ASS &H00BBGGRR (BGR byte order)."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return '&H0000D7FF'   # gold fallback
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'&H00{b:02X}{g:02X}{r:02X}'


SUB_WINDOW = 4   # 4 words per group

def build_ass_rolling(reel, path, accent_hex=SUB_FALLBACK_COLOR):
    """
    YouTube Shorts style subtitles: 3 words on screen, all in accent (orange).
    The currently-spoken word turns white and scales to 110%.

    Each window emits one Dialogue line per word (3 lines/window), precisely
    abutted — no gap frames, no \t() animation, no libass timing quirks.

    State layout for window [w0, w1, w2] while w1 is spoken:
      w0  {\\1c=white \\fscx110}w1  {\\1c=accent \\fscx100}w2
    Words before the highlight inherit PrimaryColour (accent) — no override needed.
    """
    ass_fill = _hex_to_ass(SUB_WORD_COLOR)       # warm gold for all context words
    WHITE    = _hex_to_ass(SUB_HIGHLIGHT_COLOR)  # white pop for current word
    head = (
        '[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n'
        'WrapStyle: 0\nScaledBorderAndShadow: yes\n\n'
        '[V4+ Styles]\n'
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, '
        'OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, '
        'ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
        'Alignment, MarginL, MarginR, MarginV, Encoding\n'
        f'Style: K,{FONT_NAME},{SUB_BASE_SIZE},'
        f'{ass_fill},{ass_fill},&H00000000,&H96000000,'
        f'-1,0,0,0,100,100,0,0,1,3.5,1.5,2,90,90,280,1\n\n'
        '[Events]\n'
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
    )

    all_words = reel['words']
    lines     = []

    for i in range(0, len(all_words), SUB_WINDOW):
        win     = all_words[i : i + SUB_WINDOW]
        win_end = (all_words[i + SUB_WINDOW]['start']
                   if i + SUB_WINDOW < len(all_words)
                   else win[-1]['end'] + 0.5)

        for k, w in enumerate(win):
            t0 = w['start']
            t1 = win[k + 1]['start'] if k + 1 < len(win) else win_end
            if t1 <= t0:
                t1 = t0 + 0.05   # safety minimum

            # Build 3-word text: word k = white+110%, words after k reset to accent+100%
            parts        = []
            reset_needed = False
            for j, ww in enumerate(win):
                if j == k:
                    parts.append(r'{\1c' + WHITE + r'\fscx140\fscy140}' + _esc(ww['text']))
                    reset_needed = True
                elif reset_needed:
                    parts.append(r'{\1c' + ass_fill + r'\fscx100\fscy100}' + _esc(ww['text']))
                    reset_needed = False
                else:
                    parts.append(_esc(ww['text']))   # inherits accent from style

            lines.append(
                f"Dialogue: 0,{_fmt(t0)},{_fmt(t1)},K,,0,0,0,,{' '.join(parts)}"
            )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(head + '\n'.join(lines) + '\n')
    return path


# ── VIDEO BUILD (ffmpeg filter_complex pipeline) ──────────────────────────────

def _measure_lufs(path):
    """Integrated loudness (LUFS) of an audio file via ffmpeg ebur128; None on failure."""
    try:
        out = subprocess.run(
            ['ffmpeg', '-hide_banner', '-nostats', '-i', path,
             '-filter:a', 'ebur128=metadata=1', '-f', 'null', '-'],
            capture_output=True, text=True).stderr
        vals = re.findall(r'I:\s*(-?\d+(?:\.\d+)?)\s*LUFS', out)
        if vals:
            v = float(vals[-1])
            return v if v > -70 else None     # -inf/silent guard
    except Exception:
        pass
    return None


def build_video_custom(reel, work_dir, out_path,
                       music_volume=0.12, trans_sec=0.5,
                       fps=FPS, gpu_enc=True):
    """
    Build one MP4 from a reel dict using ffmpeg.

    reel keys:
      scenes         — list of scene dicts with t_start, t_end, image, char_start, char_end
      words          — from words_from_alignment()
      total          — float, narration duration in seconds
      narration_path — path to narration.mp3
      music_path     — path to BGM music.mp3
      accent_hex     — '#RRGGBB' story accent color for subtitle highlight
    """
    scenes  = reel['scenes']
    n       = len(scenes)
    total   = reel['total']
    vid_len = total + TAIL

    # Per-scene display durations (min 0.30s)
    d = [max(0.30, s['t_end'] - s['t_start']) for s in scenes]

    # Transition duration — capped to 60% of shortest scene
    D = min(trans_sec, 0.6 * min(d)) if n > 1 else 0.06

    # Ken Burns oversize dimensions
    KBZ    = KEN_BURNS_ZOOM
    OW, OH = int(round(W * KBZ)), int(round(H * KBZ))
    px, py = OW - W, OH - H

    # Resize / center-crop all frames to Ken Burns size
    fdir = os.path.join(work_dir, 'frames')
    os.makedirs(fdir, exist_ok=True)
    frames = [
        cover_fit_png(s['image'], os.path.join(fdir, f's{i:03d}.png'), OW, OH)
        for i, s in enumerate(scenes)
    ]

    # ASS subtitle file with per-story accent color
    ass_path = os.path.join(work_dir, 'subs.ass')
    build_ass_rolling(reel, ass_path, reel.get('accent_hex', SUB_FALLBACK_COLOR))
    ass_esc = ass_path.replace(':', '\\:')   # ffmpeg filter path escaping (Linux)

    # ── ffmpeg input list ──────────────────────────────────────────────────
    durs, inputs = [], []
    for i in range(n):
        # Each frame plays for its scene duration + transition overlap on the right
        dur = d[i] + (D if i < n - 1 else 0.0)
        if i == n - 1:
            dur = d[i] + TAIL
        durs.append(dur)
        inputs += ['-loop', '1', '-t', f'{dur:.3f}', '-i', frames[i]]
    inputs += ['-i', reel['narration_path'], '-i', reel['music_path']]
    na_idx, mu_idx = n, n + 1

    # ── Video filter graph ─────────────────────────────────────────────────
    # Per-scene: fps normalise + Ken Burns crop (pan from (sx,sy) to (ex,ey))
    vp = ''
    for i in range(n):
        sx, sy, ex, ey = _PANS[i % len(_PANS)]
        DUR    = max(0.30, durs[i])
        xexpr  = f'clip({px:.1f}*({sx}+({ex}-{sx})*t/{DUR:.3f}),0,{px:.1f})'
        yexpr  = f'clip({py:.1f}*({sy}+({ey}-{sy})*t/{DUR:.3f}),0,{py:.1f})'
        vp += (f'[{i}:v]fps={fps},setpts=PTS-STARTPTS,'
               f'crop={W}:{H}:x=\'{xexpr}\':y=\'{yexpr}\','
               f'settb=AVTB,format=yuv420p,setsar=1[v{i}];')

    # xfade chain: energy-based transition per scene boundary
    transitions = pick_transitions(scenes)
    chain = ''
    prev  = 'v0'
    acc   = 0.0
    for k in range(1, n):
        acc += d[k - 1]
        t   = transitions[k - 1] if k - 1 < len(transitions) else 'dissolve'
        out = f'x{k}'
        chain += (f'[{prev}][v{k}]xfade=transition={t}:'
                  f'duration={D:.3f}:offset={acc:.3f}[{out}];')
        prev = out

    # Fade in/out + ASS subtitle burn-in
    fo     = max(0.0, vid_len - CLOSE_FADE)
    chain += (f'[{prev}]fade=t=in:st=0:d={OPEN_FADE},'
              f'fade=t=out:st={fo:.3f}:d={CLOSE_FADE},'
              f'ass=\'{ass_esc}\'[v];')

    # ── Audio filter graph ─────────────────────────────────────────────────
    # music_volume is the BGM level RELATIVE TO THE NARRATION's loudness, not a
    # raw scale on the BGM. We measure both tracks' integrated loudness (LUFS)
    # and set the BGM gain so its loudness = music_volume × narration loudness.
    # (Fish narration is quiet ~-25 LUFS, ACE-Step BGM is loud ~-15 LUFS, so a
    # plain volume=0.6 left the BGM on top — this makes 0.6 mean 0.6× the voice.)
    n_lufs = _measure_lufs(reel['narration_path'])
    b_lufs = _measure_lufs(reel['music_path'])
    if n_lufs is not None and b_lufs is not None and music_volume > 0:
        target_bgm_lufs = n_lufs + 20.0 * math.log10(music_volume)
        bgm_gain_db     = target_bgm_lufs - b_lufs
        bgm_vol         = f'volume={bgm_gain_db:.2f}dB'
        vol_desc        = (f'{music_volume}×narr → {bgm_gain_db:+.1f}dB '
                           f'(narr {n_lufs:.1f} / bgm {b_lufs:.1f} LUFS)')
    else:
        bgm_vol  = f'volume={music_volume}'      # linear fallback
        vol_desc = f'{music_volume} (linear fallback)'

    # narration splits into direct mix + sidechain for BGM ducking
    mfo    = max(0.0, vid_len - 3.0)
    chain += f'[{na_idx}:a]aresample=async=1:first_pts=0,asplit=2[na][nasc];'
    chain += f'[{mu_idx}:a]{bgm_vol},afade=t=out:st={mfo:.3f}:d=3[bg0];'
    chain += '[bg0][nasc]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[bg];'
    chain += '[na][bg]amix=inputs=2:duration=longest:normalize=0[a]'

    # ── Codec ──────────────────────────────────────────────────────────────
    NVENC  = gpu_enc and _ffmpeg_has_encoder('h264_nvenc')
    vcodec = (
        ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '21', '-b:v', '0']
        if NVENC else
        ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20']
    )

    cmd = [
        'ffmpeg', '-y', *inputs,
        '-filter_complex', vp + chain,
        '-map', '[v]', '-map', '[a]',
        '-r', str(fps),
        *vcodec,
        '-pix_fmt', 'yuv420p',
        '-filter_complex_threads', str(os.cpu_count() or 4),
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        '-t', f'{vid_len:.3f}',
        out_path,
    ]

    label = os.path.splitext(os.path.basename(out_path))[0]
    codec = 'NVENC' if NVENC else 'x264'
    print(f'    [{codec}] {label} | {n} scenes | {vid_len:.1f}s | vol={vol_desc}')

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(out_path):
        for line in res.stderr.splitlines()[-25:]:
            print(f'    [ffmpeg] {line}', flush=True)
        raise RuntimeError(f'ffmpeg failed → {out_path}')


# ── STORY / BGM DISCOVERY ─────────────────────────────────────────────────────

def discover_stories(out_root, story_filter=None):
    dirs = sorted(
        d for d in os.listdir(out_root)
        if os.path.isdir(os.path.join(out_root, d))
        and os.path.exists(os.path.join(out_root, d, 'source.json'))
        and os.path.exists(os.path.join(out_root, d, 'transcript.json'))
        and os.path.isdir(os.path.join(out_root, d, 'images'))
    )
    if story_filter:
        dirs = [d for i, d in enumerate(dirs, 1) if i in story_filter]
    return [os.path.join(out_root, d) for d in dirs]


def collect_bgm_variants(story_dir, exp_filter='all'):
    """Return list of (label, music_path, music_volume) for all available BGM variants."""
    variants = []

    def _vol(brief_path):
        try:
            return float(json.load(open(brief_path)).get('music_volume', 0.12))
        except Exception:
            return 0.12

    if exp_filter in ('exp1', 'all'):
        exp1 = os.path.join(story_dir, 'exp1_single_pass')
        for tech, lbl in [
            ('techA_script_only',     'exp1_techA'),
            ('techB_pacing_wps',      'exp1_techB'),
            ('techC_energy_envelope', 'exp1_techC'),
            ('techD_scene_structured','exp1_techD'),
        ]:
            mp3   = os.path.join(exp1, tech, 'music.mp3')
            brief = os.path.join(exp1, tech, 'brief.json')
            if os.path.exists(mp3):
                variants.append((lbl, mp3, _vol(brief)))

    if exp_filter in ('exp2', 'all'):
        exp2 = os.path.join(story_dir, 'exp2_two_pass')
        for tech, lbl in [
            ('techA_script_only',     'techA'),
            ('techB_pacing_wps',      'techB'),
            ('techC_energy_envelope', 'techC'),
            ('techD_scene_structured','techD'),
        ]:
            for pass_n, suffix in [('pass1', 'p1'), ('pass2', 'p2')]:
                mp3   = os.path.join(exp2, tech, pass_n, 'music.mp3')
                brief = os.path.join(exp2, tech, pass_n, 'brief.json')
                if os.path.exists(mp3):
                    variants.append((f'exp2_{lbl}_{suffix}', mp3, _vol(brief)))

    return variants


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Compose narration+BGM short-form videos for all experiment stories.')
    ap.add_argument('--out-root',        default=OUT_ROOT,
                    help='Root folder containing story subdirs')
    ap.add_argument('--stories',         nargs='+', type=int, metavar='N',
                    help='1-based story indices to process (default: all)')
    ap.add_argument('--exp',             choices=['exp1', 'exp2', 'all'], default='all',
                    help='Which experiment BGMs to include (default: all)')
    ap.add_argument('--fps',             type=int,   default=FPS)
    ap.add_argument('--transition-sec',  type=float, default=0.5,
                    help='Cross-fade transition duration in seconds (default: 0.5)')
    ap.add_argument('--skip-existing',   action='store_true',
                    help='Skip if output MP4 already exists')
    ap.add_argument('--no-gpu-enc',      action='store_true',
                    help='Use libx264 instead of h264_nvenc')
    ap.add_argument('--variants',        nargs='+', metavar='LABEL',
                    help='Only render specific variant labels (e.g. exp2_techC_p2)')
    args = ap.parse_args()

    story_filter = set(args.stories) if args.stories else None
    story_dirs   = discover_stories(args.out_root, story_filter)

    print(f'Video composer — {len(story_dirs)} stories, exp={args.exp}')

    total_ok = total_skip = total_fail = 0

    for idx, story_dir in enumerate(story_dirs, 1):
        slug = os.path.basename(story_dir)
        print(f'\n── [{idx}/{len(story_dirs)}] {slug}')

        # Load story assets
        source     = json.load(open(os.path.join(story_dir, 'source.json')))
        transcript = json.load(open(os.path.join(story_dir, 'transcript.json')))
        scenes     = copy.deepcopy(source.get('scenes', []))
        script     = source.get('script', '')
        palette    = source.get('meta', {}).get('style', {}).get('palette_hex', [])
        accent_hex = palette[1] if len(palette) > 1 else SUB_FALLBACK_COLOR

        # Compute scene char offsets from narration snippets
        add_char_positions(scenes, script)

        # Word-level timing
        chars  = transcript.get('characters', [])
        starts = transcript.get('character_start_times_seconds', [])
        ends   = transcript.get('character_end_times_seconds', [])
        words  = words_from_alignment(script, chars, starts, ends)
        total  = float(ends[-1]) if ends else 0.0

        # Scene timestamps (same logic as stage_05_compose/compose.py)
        aligned_ok = abs(len(chars) - len(script)) <= 2
        for s in scenes:
            cs, ce = s['char_start'], s['char_end']
            if aligned_ok and ce <= len(starts):
                s['t_start'] = float(starts[cs])
                s['t_end']   = float(ends[min(ce, len(ends)) - 1])
            else:
                L = max(1, len(script))
                s['t_start'] = total * cs / L
                s['t_end']   = total * ce / L
            s['t_start'] = max(0.0, s['t_start'])
            s['t_end']   = max(s['t_start'] + 0.2, s['t_end'])
        for a, b in zip(scenes, scenes[1:]):
            a['t_end'] = b['t_start']
        if scenes:
            scenes[-1]['t_end'] = total

        # Assign image paths
        img_dir = os.path.join(story_dir, 'images')
        for i, s in enumerate(scenes):
            s['image'] = os.path.join(img_dir, f'scene_{i + 1:03d}.png')

        os.makedirs(os.path.join(story_dir, 'videos'), exist_ok=True)

        variants = collect_bgm_variants(story_dir, args.exp)
        if args.variants:
            variants = [(l, p, v) for l, p, v in variants if l in args.variants]
        print(f'  {len(variants)} BGM variants | {len(scenes)} scenes | {total:.1f}s'
              f' | accent={accent_hex}')

        for label, bgm_path, music_vol in variants:
            out_path = os.path.join(story_dir, 'videos', f'{label}.mp4')
            if args.skip_existing and os.path.exists(out_path):
                print(f'  [skip] {label}')
                total_skip += 1
                continue

            work_dir = tempfile.mkdtemp(prefix='vc_')
            try:
                reel = {
                    'scenes':         copy.deepcopy(scenes),
                    'words':          words,
                    'total':          total,
                    'narration_path': os.path.join(story_dir, 'narration.mp3'),
                    'music_path':     bgm_path,
                    'accent_hex':     accent_hex,
                }
                build_video_custom(
                    reel, work_dir, out_path,
                    music_volume=0.4,
                    trans_sec=args.transition_sec,
                    fps=args.fps,
                    gpu_enc=not args.no_gpu_enc,
                )
                size_mb = os.path.getsize(out_path) / 1e6
                print(f'  ✓ {label}  ({size_mb:.1f} MB)')
                total_ok += 1
            except Exception as e:
                print(f'  ✗ {label}  FAILED: {e}')
                total_fail += 1
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

    print(f'\nDone.  ✓ {total_ok}  skipped {total_skip}  ✗ {total_fail}')


if __name__ == '__main__':
    main()

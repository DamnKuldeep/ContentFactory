"""
run_bgm_batch.py — Run all 5 BGM experiments for every story in a folder.

Outputs land inside each story's own stage_04_music/ subfolder so Stage 5
(compose.py) can find them without any additional reorganisation:

  <stories-dir>/
    <story-slug>/
      source.json          (existing)
      images/              (existing)
      stage_04_music/      ← created here
        exp1_baseline.mp3
        exp2_arc_caption.mp3
        exp3_lyrics_markers.mp3
        exp4_bpm_key_curve.mp3
        exp5_maximal.mp3
        music.mp3          ← copy of exp5_maximal (default pick for stitching)

Usage:
    python run_bgm_batch.py
    python run_bgm_batch.py --stories-dir /path/to/outputs
    python run_bgm_batch.py --exps 1 5 --seed 42
    python run_bgm_batch.py --no-install --stories 3 5  (pick specific stories by index)
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

# ── DEFAULTS ──────────────────────────────────────────────────────────────────

DEFAULT_STORIES_DIR = os.path.expanduser('~/bt/outputs_extracted/outputs')
DEFAULT_REPO_DIR    = os.path.expanduser('~/bt/ACE-Step-1.5')
DEFAULT_CKPT_DIR    = os.path.expanduser('~/bt/ace-checkpoints')

EXP_LABELS = {
    1: 'exp1_baseline',
    2: 'exp2_arc_caption',
    3: 'exp3_lyrics_markers',
    4: 'exp4_bpm_key_curve',
    5: 'exp5_maximal',
}
DEFAULT_EXP = 5  # which exp becomes music.mp3 (the Stage-5 default)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Batch BGM generation for all stories')
    p.add_argument('--stories-dir', default=DEFAULT_STORIES_DIR,
                   help='Root folder containing story subfolders (each with source.json)')
    p.add_argument('--exps', nargs='+', type=int, choices=[1,2,3,4,5],
                   default=[1,2,3,4,5], metavar='N',
                   help='Experiments to run per story (default: all 5)')
    p.add_argument('--stories', nargs='+', type=int, metavar='N',
                   help='Run only these 1-based story indices (default: all)')
    p.add_argument('--repo-dir',   default=DEFAULT_REPO_DIR)
    p.add_argument('--ckpt-dir',   default=DEFAULT_CKPT_DIR)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--no-install', action='store_true',
                   help='Skip pip install (use if deps already installed)')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip experiments whose output file already exists')
    return p.parse_args()

# ── DISCOVER STORIES ──────────────────────────────────────────────────────────

def find_stories(root):
    """Return list of (story_dir, source_json_path) sorted by folder name."""
    hits = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        src = os.path.join(entry.path, 'source.json')
        if os.path.exists(src):
            hits.append((entry.path, src))
    return hits

# ── RUN ONE EXPERIMENT (rename output to labeled name) ────────────────────────

def run_one_exp(exp_num, story_data, dit, llm, out_dir, seed, skip_existing):
    """Run a single experiment and rename the output to expN_label.mp3.
    Returns the final path on success, None on failure / skip.
    """
    from ace_story_grounded import derive_key, make_music

    label    = EXP_LABELS[exp_num]
    dst_path = os.path.join(out_dir, f'{label}.mp3')

    if skip_existing and os.path.exists(dst_path):
        print(f'  [skip] {label}.mp3 already exists')
        return dst_path

    s   = story_data
    bpm = s['bpm_est']
    key = derive_key(s['style'], s['first_scene'], s['last_scene'])
    style_short = s['style'][:40].rsplit(' ', 1)[0] if len(s['style']) > 40 else s['style']

    # Use a temp subdir so the snapshot trick works cleanly even if out_dir
    # already has other MP3s from prior runs.
    tmp = tempfile.mkdtemp(dir=out_dir, prefix=f'_tmp_exp{exp_num}_')
    try:
        result = None
        if exp_num == 1:
            result = make_music(
                dit, llm, tmp,
                caption         = 'ambient, orchestral, suspenseful, dramatic, cinematic',
                tags            = 'ambient, suspense, orchestral, cinematic',
                duration        = s['duration'],
                inference_steps = 12,
                seed            = seed,
            )
        elif exp_num == 2:
            arc_caption = (
                'Opens with sparse, uneasy atmosphere: {}. '
                'Builds in tension and density through the middle: {}. '
                'Reaches a dramatic peak at the turning point, then resolves: {}. '
                'Cinematic, orchestral, emotionally reactive score.'
            ).format(s['first_scene'][:85], s['mid_scene'][:85], s['last_scene'][:65])[:512]
            print(f'  caption: {arc_caption[:90]}...')
            result = make_music(
                dit, llm, tmp,
                caption         = arc_caption,
                tags            = '{}, cinematic, orchestral, dramatic'.format(s['style']),
                duration        = s['duration'],
                inference_steps = 12,
                seed            = seed,
            )
        elif exp_num == 3:
            structured_lyrics = (
                '[Intro]\n'   + s['first_scene'][:120]
                + '\n\n[Verse]\n'  + s['act1_narr'][:200]
                + '\n\n[Bridge]\n' + s['act2_narr'][:200]
                + '\n\n[Chorus]\n' + s['climax_narr'][:160]
                + '\n\n[Outro]\n'  + s['last_scene'][:100] + '\n'
            )
            result = make_music(
                dit, llm, tmp,
                caption         = '{} cinematic score in three acts, emotionally reactive, orchestral'.format(s['style']),
                tags            = '{}, orchestral, cinematic, three movements'.format(s['style']),
                duration        = s['duration'],
                lyrics          = structured_lyrics,
                inference_steps = 12,
                seed            = seed,
                instrumental    = False,
            )
        elif exp_num == 4:
            curve_caption = (
                'Begins quiet and sparse, builds through three emotional movements that mirror the story arc, '
                'peaks with full orchestration at the climax, then gradually dissolves into a contemplative close. '
                'Style: {}. Cinematic, orchestral, score for video narration.'
            ).format(s['style'])[:512]
            print(f'  BPM: {bpm}  Key: {key}')
            result = make_music(
                dit, llm, tmp,
                caption         = curve_caption,
                tags            = '{}, cinematic, orchestral, emotional arc'.format(s['style']),
                duration        = s['duration'],
                bpm             = bpm,
                keyscale        = key,
                inference_steps = 14,
                seed            = seed,
            )
        elif exp_num == 5:
            full_caption = (
                '{} cinematic score. Act I: {}. Act II — builds in tension: {}. Act III: {}. '
                'Emotionally reactive orchestral score for narrated video.'
            ).format(style_short, s['act1_narr'][:105], s['act2_narr'][:105], s['act3_narr'][:85])[:512]
            full_lyrics = (
                '[Intro]\n'   + s['first_scene'][:140]
                + '\n\n[Verse]\n'  + s['act1_narr'][:230]
                + '\n\n[Bridge]\n' + s['act2_narr'][:230]
                + '\n\n[Chorus]\n' + s['climax_narr'][:190]
                + '\n\n[Outro]\n'  + s['act3_narr'][:160] + '\n'
            )
            print(f'  BPM: {bpm}  Key: {key}')
            print(f'  caption: {full_caption[:90]}...')
            result = make_music(
                dit, llm, tmp,
                caption         = full_caption,
                tags            = '{}, cinematic, orchestral, three acts, emotionally reactive'.format(s['style']),
                duration        = s['duration'],
                bpm             = bpm,
                keyscale        = key,
                lyrics          = full_lyrics,
                inference_steps = 16,
                seed            = seed,
                instrumental    = False,
            )

        if result is None or not result.audios:
            return None

        # Move the generated file out of tmp into out_dir with the label name
        src = result.audios[0]['path']
        shutil.move(src, dst_path)
        return dst_path

    except Exception as e:
        print(f'  ERROR in exp {exp_num}: {e}')
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Add scripts dir to path so we can import ace_story_grounded
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from ace_story_grounded import (
        check_env, ensure_repo, install_deps, ensure_weights,
        setup_gpu, load_model, analyze_story,
    )

    stories = find_stories(args.stories_dir)
    if not stories:
        raise SystemExit(f'No story folders with source.json found in: {args.stories_dir}')

    # Optional subset filter
    if args.stories:
        indices = [i - 1 for i in args.stories]
        stories = [stories[i] for i in indices if i < len(stories)]

    print('=' * 70)
    print('ACE-Step Batch BGM Generation')
    print('=' * 70)
    print(f'Stories dir : {args.stories_dir}')
    print(f'Stories     : {len(stories)}')
    print(f'Experiments : {args.exps}')
    print(f'Seed        : {args.seed}')
    print()
    for i, (d, _) in enumerate(stories, 1):
        print(f'  {i}. {os.path.basename(d)}')
    print()

    # ── One-time setup ────────────────────────────────────────────────────────
    check_env()
    ensure_repo(args.repo_dir)
    if not args.no_install:
        install_deps()
    ensure_weights(args.ckpt_dir)
    device = setup_gpu(args.repo_dir, args.ckpt_dir)
    dit, llm = load_model(args.repo_dir, args.ckpt_dir, device)

    # ── Per-story loop ────────────────────────────────────────────────────────
    manifest = []
    batch_t0 = time.time()

    for story_idx, (story_dir, src_json) in enumerate(stories, 1):
        slug  = os.path.basename(story_dir)
        title = json.load(open(src_json)).get('meta', {}).get(
                    'creative_direction', {}).get('title', slug)

        print()
        print('─' * 70)
        print(f'STORY {story_idx}/{len(stories)}: {title}')
        print(f'  dir  : {slug}')

        stage01 = json.load(open(src_json))
        story   = analyze_story(stage01)
        from ace_story_grounded import derive_key
        key     = derive_key(story['style'], story['first_scene'], story['last_scene'])
        print(f'  data : {story["n_scenes"]} scenes | {story["duration"]}s | '
              f'BPM {story["bpm_est"]} | {key} | style: {story["style"][:60]}')

        out_dir = os.path.join(story_dir, 'stage_04_music')
        os.makedirs(out_dir, exist_ok=True)

        story_results = {}
        story_t0 = time.time()

        for exp_num in args.exps:
            label = EXP_LABELS[exp_num]
            print(f'\n  [Exp {exp_num}] {label}')
            path = run_one_exp(exp_num, story, dit, llm, out_dir, args.seed,
                               args.skip_existing)
            if path:
                size_mb = os.path.getsize(path) / 1e6
                print(f'  -> {os.path.basename(path)}  ({size_mb:.1f} MB)')
                story_results[exp_num] = path
                manifest.append({
                    'story': title,
                    'slug':  slug,
                    'exp':   exp_num,
                    'label': label,
                    'path':  path,
                    'mb':    round(size_mb, 2),
                })
            else:
                print(f'  -> FAILED')

        # Copy the default experiment as music.mp3 (for Stage-5 stitching)
        default_path = story_results.get(DEFAULT_EXP) or next(iter(story_results.values()), None)
        if default_path:
            music_mp3 = os.path.join(out_dir, 'music.mp3')
            shutil.copy2(default_path, music_mp3)
            print(f'\n  music.mp3 <- {os.path.basename(default_path)}')

        print(f'\n  Story done in {time.time()-story_t0:.0f}s')

    # ── Final manifest ────────────────────────────────────────────────────────
    total_time = time.time() - batch_t0
    print()
    print('=' * 70)
    print(f'BATCH COMPLETE  |  {len(manifest)} file(s)  |  {total_time/60:.1f} min')
    print('=' * 70)

    col_w = max((len(r['story']) for r in manifest), default=10)
    print(f'  {"Story":<{col_w}}  {"Exp":<4}  {"Label":<24}  {"MB":>5}')
    print(f'  {"-"*col_w}  {"-"*4}  {"-"*24}  {"-"*5}')
    for r in manifest:
        print(f'  {r["story"]:<{col_w}}  {r["exp"]:<4}  {r["label"]:<24}  {r["mb"]:>5.1f}')

    # Write manifest JSON alongside the stories
    manifest_path = os.path.join(args.stories_dir, 'bgm_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nManifest saved: {manifest_path}')


if __name__ == '__main__':
    main()

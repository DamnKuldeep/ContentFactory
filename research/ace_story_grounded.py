"""
ace_story_grounded.py — BGM prompting experiments, script version of ace-story-grounded.ipynb

Usage:
    python ace_story_grounded.py --story path/to/source.json
    python ace_story_grounded.py --story path/to/source.json --exps 1 3 5
    python ace_story_grounded.py --story path/to/source.json --output-dir /tmp/bgm

Arguments:
    --story       Path to stage_01.json or source.json / ch_reel__*.json  (required)
    --exps        Which experiments to run: 1 2 3 4 5  (default: all)
    --output-dir  Where to save the MP3s  (default: ~/bt/ace-outputs)
    --repo-dir    ACE-Step-1.5 clone path (default: ~/bt/ACE-Step-1.5)
    --ckpt-dir    Checkpoints path         (default: ~/bt/ace-checkpoints)
    --seed        Generation seed          (default: 42)
    --no-install  Skip pip install step
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='ACE-Step story-grounded BGM experiments')
    p.add_argument('--story',      required=True,
                   help='Path to stage_01.json, source.json, or ch_reel__*.json')
    p.add_argument('--exps',       nargs='+', type=int, choices=[1,2,3,4,5],
                   default=[1,2,3,4,5], metavar='N',
                   help='Experiments to run (default: 1 2 3 4 5)')
    p.add_argument('--output-dir', default=os.path.expanduser('~/bt/ace-outputs'))
    p.add_argument('--repo-dir',   default=os.path.expanduser('~/bt/ACE-Step-1.5'))
    p.add_argument('--ckpt-dir',   default=os.path.expanduser('~/bt/ace-checkpoints'))
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--no-install', action='store_true',
                   help='Skip pip install step (use if deps already installed)')
    return p.parse_args()

# ── ENV CHECK ─────────────────────────────────────────────────────────────────

def check_env():
    import torch
    print('Python :', sys.version.split()[0])
    print('Torch  :', torch.__version__)
    print('CUDA   :', torch.version.cuda, '| available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit('No GPU detected.')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}  |  {p.total_memory/1e9:.1f} GB')
    print()

# ── INSTALL DEPS ──────────────────────────────────────────────────────────────

DEPS = [
    'transformers>=4.51.0,<4.58.0',
    'diffusers>=0.37.0',
    'accelerate>=1.0.0',
    'vector-quantize-pytorch>=1.27.15',
    'pytorch-wavelets>=1.3.0',
    'pywavelets>=1.9.0',
    'einops>=0.8.1',
    'loguru>=0.7.3',
    'soundfile>=0.13.1',
    'librosa>=0.10.0',
    'diskcache',
    'toml',
    'numba>=0.59',
    'peft>=0.10.0',
    'lycoris-lora',
    'modelscope',
    'requests',
    'huggingface_hub',
]

def install_deps():
    print('Installing dependencies...')
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q'] + DEPS,
        check=True,
    )
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'torchao'],
        check=False,
    )
    print('Dependencies ready.\n')

# ── CLONE ACE-STEP ────────────────────────────────────────────────────────────

def ensure_repo(repo_dir):
    if not os.path.exists(repo_dir):
        print(f'Cloning ACE-Step-1.5 -> {repo_dir}')
        subprocess.run(
            ['git', 'clone', '--depth', '1',
             'https://github.com/ace-step/ACE-Step-1.5.git', repo_dir],
            check=True,
        )
    else:
        print('ACE-Step repo already present.')
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

# ── DOWNLOAD WEIGHTS ──────────────────────────────────────────────────────────

def ensure_weights(ckpt_dir):
    from huggingface_hub import snapshot_download
    os.makedirs(ckpt_dir, exist_ok=True)

    dit_ready = os.path.exists(f'{ckpt_dir}/acestep-v15-turbo/model.safetensors')
    lm_ready  = os.path.exists(f'{ckpt_dir}/acestep-5Hz-lm-0.6B/model.safetensors')

    if not dit_ready:
        print('Downloading ACE-Step weights (~8 GB)...')
        snapshot_download(repo_id='ACE-Step/Ace-Step1.5', local_dir=ckpt_dir,
                          local_dir_use_symlinks=False)
        print('DiT weights ready.')
    else:
        print('DiT weights already present — skipping.')

    if not lm_ready:
        print('Downloading 0.6B LM weights (~1.3 GB)...')
        snapshot_download(repo_id='ACE-Step/acestep-5Hz-lm-0.6B',
                          local_dir=f'{ckpt_dir}/acestep-5Hz-lm-0.6B',
                          local_dir_use_symlinks=False)
        print('LM weights ready.')
    else:
        print('LM weights already present — skipping.')

    # Remove the large 1.7B LM if it got downloaded alongside the bundle
    large_lm = f'{ckpt_dir}/acestep-5Hz-lm-1.7B'
    if os.path.exists(large_lm):
        shutil.rmtree(large_lm)
        print('Removed 1.7B planner (not needed).')

# ── GPU + DIFFUSERS SETUP ─────────────────────────────────────────────────────

def setup_gpu(repo_dir, ckpt_dir):
    import torch

    os.environ['CUDA_VISIBLE_DEVICES']  = '0'
    os.environ['TORCHDYNAMO_DISABLE']   = '1'
    os.environ['TORCHINDUCTOR_DISABLE'] = '1'
    os.environ['HF_HUB_OFFLINE']        = '1'  # weights already downloaded; skip HF network check on every run
    try:
        torch._dynamo.config.disable = True
    except Exception:
        pass

    # Symlink checkpoints into the path ACE-Step expects
    target = f'{repo_dir}/checkpoints'
    os.makedirs(repo_dir, exist_ok=True)
    if not os.path.exists(target):
        os.symlink(ckpt_dir, target)

    # Patch diffusers to avoid meta-tensor device issues
    from diffusers.models.modeling_utils import ModelMixin
    if not getattr(ModelMixin.from_pretrained, '_ace_patched', False):
        _orig = ModelMixin.from_pretrained.__func__
        def _patched(cls, *a, **kw):
            kw.setdefault('low_cpu_mem_usage', False)
            kw.setdefault('device_map', None)
            return _orig(cls, *a, **kw)
        _patched._ace_patched = True
        ModelMixin.from_pretrained = classmethod(_patched)
    try:
        from diffusers import AutoencoderOobleck
        if not getattr(AutoencoderOobleck.from_pretrained, '_ace_patched', False):
            _orig_oob = AutoencoderOobleck.from_pretrained.__func__
            def _patched_oob(cls, *a, **kw):
                kw.setdefault('low_cpu_mem_usage', False)
                kw.setdefault('device_map', None)
                return _orig_oob(cls, *a, **kw)
            _patched_oob._ace_patched = True
            AutoencoderOobleck.from_pretrained = classmethod(_patched_oob)
    except Exception:
        pass

    device = 'cuda:0'
    print(f'GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM')
    return device

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model(repo_dir, ckpt_dir, device):
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    try:
        import torchao
        quantization = 'int8_weight_only'
        compile_model = True
        print('torchao -> INT8 quantisation (DiT ~2.4 GB)')
    except ImportError:
        quantization = None
        compile_model = False
        print('torchao not found -> bf16 (~4.8 GB DiT)')

    print('\nLoading DiT...')
    t0 = time.time()
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root   = repo_dir,
        config_path    = 'acestep-v15-turbo',
        device         = device,
        offload_to_cpu = False,
        quantization   = quantization,
        compile_model  = compile_model,
    )
    print(f'DiT ({time.time()-t0:.0f}s): ok={ok}')
    assert ok, f'DiT init failed: {msg}'

    print('Loading LM...')
    t0 = time.time()
    llm = LLMHandler()
    msg, ok = llm.initialize(
        checkpoint_dir = ckpt_dir,
        lm_model_path  = 'acestep-5Hz-lm-0.6B',
        backend        = 'pt',
        device         = device,
        offload_to_cpu = False,
    )
    print(f'LM  ({time.time()-t0:.0f}s): ok={ok}')
    assert ok, f'LM init failed: {msg}'

    import torch
    alloc = torch.cuda.memory_allocated(0) / 1e9
    print(f'Model ready | {alloc:.1f} GB VRAM used\n')
    return dit, llm

# ── STORY ANALYSIS ────────────────────────────────────────────────────────────

def analyze_story(s1):
    scenes = s1['scenes']
    n      = len(scenes)
    words  = s1['script'].split()
    duration_est = max(30, round(len(words) / 150 * 60))
    bpm_est = max(60, min(120, round(len(words) / (duration_est / 60) / 1.5)))
    a1 = n // 3
    a2 = 2 * n // 3
    act1_narr  = ' '.join(s['narration'] for s in scenes[:a1])[:300]
    act2_narr  = ' '.join(s['narration'] for s in scenes[a1:a2])[:300]
    act3_narr  = ' '.join(s['narration'] for s in scenes[a2:])[:300]
    climax_narr = scenes[int(n * 0.70)]['narration'][:200]
    if 'style_anchor' in s1:
        style = s1['style_anchor']
    elif 'style_id' in s1:
        style = s1['style_id']
    elif 'meta' in s1 and isinstance(s1['meta'].get('style'), dict):
        st = s1['meta']['style']
        style = st.get('style_anchor', st.get('style_id', 'cinematic'))
    else:
        style = 'cinematic'
    return {
        'duration':    duration_est,
        'n_scenes':    n,
        'style':       style,
        'script':      s1['script'],
        'first_scene': scenes[0]['narration'][:180],
        'mid_scene':   scenes[n // 2]['narration'][:180],
        'last_scene':  scenes[-1]['narration'][:180],
        'act1_narr':   act1_narr,
        'act2_narr':   act2_narr,
        'act3_narr':   act3_narr,
        'climax_narr': climax_narr,
        'bpm_est':     bpm_est,
    }


def derive_key(style, first_scene, last_scene):
    text = (style + ' ' + first_scene + ' ' + last_scene).lower()
    DARK      = ('death','murder','horror','plague','doom','terror','monster','dread','darkness','curse','haunted')
    TRIUMPH   = ('triumph','victory','hope','discovery','freedom','glory','salvation','liberation','dawn','rise')
    MYSTERY   = ('ancient','ritual','mystery','occult','ruin','forgotten','legend','labyrinth','oracle','forbidden')
    MELANCHOLY= ('exile','farewell','memory','nostalgia','sorrow','regret','lonely','loss','grief','longing','faded')
    MARTIAL   = ('battle','war','siege','conflict','clash','fight','army','conquest','invasion')
    if any(w in text for w in DARK):       return 'A minor'
    if any(w in text for w in TRIUMPH):    return 'D major'
    if any(w in text for w in MYSTERY):    return 'E minor'
    if any(w in text for w in MELANCHOLY): return 'D minor'
    if any(w in text for w in MARTIAL):    return 'C minor'
    return 'F minor'

# ── GENERATION ────────────────────────────────────────────────────────────────

def make_music(dit, llm, output_dir, caption, tags='', duration=60,
               instrumental=True, bpm=None, keyscale='',
               lyrics='[Instrumental]', seed=42, inference_steps=8,
               batch_size=1, thinking=True):
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    full_caption = caption if not tags else f'{caption}\n\nTags: {tags}'
    params = GenerationParams(
        task_type       = 'text2music',
        caption         = full_caption[:512],
        lyrics          = lyrics if not instrumental else '[Instrumental]',
        instrumental    = instrumental,
        duration        = float(duration),
        bpm             = bpm,
        keyscale        = keyscale,
        inference_steps = inference_steps,
        shift           = 3.0,
        infer_method    = 'ode',
        seed            = seed,
        thinking        = thinking,
    )
    config = GenerationConfig(
        batch_size      = batch_size,
        use_random_seed = (seed < 0),
        seeds           = [seed] if seed >= 0 else None,
        audio_format    = 'wav',  # WAV uses soundfile fallback; MP3 requires torchcodec (not available on cu130)
    )
    t0     = time.time()
    result = generate_music(dit, llm, params, config, save_dir=output_dir)
    elapsed = time.time() - t0

    if not result.success:
        print(f'  Generation failed: {result.error}')
        return None

    tc = result.extra_outputs.get('time_costs', {})
    print(f'  {len(result.audios)} track(s) | {duration}s | {elapsed:.1f}s wall-clock')
    if tc:
        print(f'  LM: {tc.get("lm_phase1_time",0)+tc.get("lm_phase2_time",0):.1f}s  '
              f'DiT: {tc.get("dit_total_time_cost",0):.1f}s')
    lm_meta = result.extra_outputs.get('lm_metadata') or {}
    if lm_meta:
        print(f'  LM blueprint -> bpm={lm_meta.get("bpm")}  key={lm_meta.get("keyscale")}')

    # Convert each saved WAV → MP3 via ffmpeg (avoids the torchcodec dependency)
    import subprocess as _sp
    for audio in result.audios:
        wav_path = audio.get('path', '')
        if wav_path and wav_path.endswith('.wav') and os.path.exists(wav_path):
            mp3_path = wav_path[:-4] + '.mp3'
            ret = _sp.run(
                ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                 '-i', wav_path, '-codec:a', 'libmp3lame', '-b:a', '128k', mp3_path],
                capture_output=True,
            )
            if ret.returncode == 0:
                os.remove(wav_path)
                audio['path'] = mp3_path
        print(f'  Saved: {audio["path"]}')
    return result

# ── EXPERIMENTS ───────────────────────────────────────────────────────────────

def run_experiments(exps, story, dit, llm, output_dir, seed):
    s   = story
    bpm = s['bpm_est']
    key = derive_key(s['style'], s['first_scene'], s['last_scene'])
    style_short = s['style'][:40].rsplit(' ', 1)[0] if len(s['style']) > 40 else s['style']

    results = {}

    if 1 in exps:
        print('\n=== EXP 1: BASELINE (CONTROL) ===')
        results[1] = make_music(
            dit, llm, output_dir,
            caption         = 'ambient, orchestral, suspenseful, dramatic, cinematic',
            tags            = 'ambient, suspense, orchestral, cinematic',
            duration        = s['duration'],
            inference_steps = 12,
            seed            = seed,
        )

    if 2 in exps:
        arc_caption = (
            'Opens with sparse, uneasy atmosphere: {}. '
            'Builds in tension and density through the middle: {}. '
            'Reaches a dramatic peak at the turning point, then resolves: {}. '
            'Cinematic, orchestral, emotionally reactive score.'
        ).format(s['first_scene'][:85], s['mid_scene'][:85], s['last_scene'][:65])[:512]
        print('\n=== EXP 2: NARRATIVE ARC CAPTION ===')
        print(f'  Caption ({len(arc_caption)}c): {arc_caption[:100]}...')
        results[2] = make_music(
            dit, llm, output_dir,
            caption         = arc_caption,
            tags            = '{}, cinematic, orchestral, dramatic'.format(s['style']),
            duration        = s['duration'],
            inference_steps = 12,
            seed            = seed,
        )

    if 3 in exps:
        structured_lyrics = (
            '[Intro]\n' + s['first_scene'][:120]
            + '\n\n[Verse]\n' + s['act1_narr'][:200]
            + '\n\n[Bridge]\n' + s['act2_narr'][:200]
            + '\n\n[Chorus]\n' + s['climax_narr'][:160]
            + '\n\n[Outro]\n' + s['last_scene'][:100] + '\n'
        )
        print('\n=== EXP 3: TEMPORAL LYRICS MARKERS ===')
        results[3] = make_music(
            dit, llm, output_dir,
            caption         = '{} cinematic score in three acts, emotionally reactive, orchestral'.format(s['style']),
            tags            = '{}, orchestral, cinematic, three movements'.format(s['style']),
            duration        = s['duration'],
            lyrics          = structured_lyrics,
            inference_steps = 12,
            seed            = seed,
            instrumental    = False,
        )

    if 4 in exps:
        curve_caption = (
            'Begins quiet and sparse, builds through three emotional movements that mirror the story arc, '
            'peaks with full orchestration at the climax, then gradually dissolves into a contemplative close. '
            'Style: {}. Cinematic, orchestral, score for video narration.'
        ).format(s['style'])[:512]
        print('\n=== EXP 4: BPM + KEY + INTENSITY CURVE ===')
        print(f'  BPM: {bpm}  Key: {key}')
        results[4] = make_music(
            dit, llm, output_dir,
            caption         = curve_caption,
            tags            = '{}, cinematic, orchestral, emotional arc'.format(s['style']),
            duration        = s['duration'],
            bpm             = bpm,
            keyscale        = key,
            inference_steps = 14,
            seed            = seed,
        )

    if 5 in exps:
        full_caption = (
            '{} cinematic score. Act I: {}. Act II — builds in tension: {}. Act III: {}. '
            'Emotionally reactive orchestral score for narrated video.'
        ).format(style_short, s['act1_narr'][:105], s['act2_narr'][:105], s['act3_narr'][:85])[:512]
        full_lyrics = (
            '[Intro]\n' + s['first_scene'][:140]
            + '\n\n[Verse]\n' + s['act1_narr'][:230]
            + '\n\n[Bridge]\n' + s['act2_narr'][:230]
            + '\n\n[Chorus]\n' + s['climax_narr'][:190]
            + '\n\n[Outro]\n' + s['act3_narr'][:160] + '\n'
        )
        print('\n=== EXP 5: MAXIMAL (ALL CHANNELS) ===')
        print(f'  BPM: {bpm}  Key: {key}')
        print(f'  Caption ({len(full_caption)}c): {full_caption[:100]}...')
        results[5] = make_music(
            dit, llm, output_dir,
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

    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate story path
    if not os.path.exists(args.story):
        raise FileNotFoundError(f'Story JSON not found: {args.story}')
    with open(args.story) as f:
        stage01 = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('ACE-Step Story-Grounded BGM Experiments')
    print('=' * 60)
    print(f'Story      : {args.story}')
    print(f'Experiments: {args.exps}')
    print(f'Output dir : {args.output_dir}')
    print(f'Seed       : {args.seed}')
    print()

    check_env()

    ensure_repo(args.repo_dir)

    if not args.no_install:
        install_deps()

    ensure_weights(args.ckpt_dir)

    device = setup_gpu(args.repo_dir, args.ckpt_dir)

    dit, llm = load_model(args.repo_dir, args.ckpt_dir, device)

    story = analyze_story(stage01)
    print(f'Story analysed: {story["n_scenes"]} scenes | {story["duration"]}s | '
          f'BPM est {story["bpm_est"]} | key {derive_key(story["style"], story["first_scene"], story["last_scene"])}')
    print(f'Style: {story["style"][:80]}')

    run_experiments(args.exps, story, dit, llm, args.output_dir, args.seed)

    # Summary
    files = sorted(f for f in os.listdir(args.output_dir) if f.lower().endswith(('.mp3','.wav','.flac')))
    print(f'\n{len(files)} file(s) in {args.output_dir}:')
    for f in files:
        path = os.path.join(args.output_dir, f)
        print(f'  {os.path.getsize(path)/1e6:6.2f} MB  {f}')


if __name__ == '__main__':
    main()

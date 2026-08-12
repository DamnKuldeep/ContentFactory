"""
Stage 5: Final video composition using FFMPEG.

Per-word-state rolling karaoke subtitles, energy-grounded transitions, bounds-guarded
Ken Burns, and loudness-relative BGM mixing (BGM sits at config.BGM_LOUDNESS_RATIO ×
the narration's loudness). Reusable helpers live in shared/ and are re-exposed here so
the legacy scripts/video_composer.py imports keep working.
"""

import logging
import os
import subprocess

from shared import config
from shared.subtitles import (
    build_ass_rolling, words_from_alignment, _fmt, _esc, SUB_FALLBACK_COLOR,
)
from shared.audio import bgm_volume_filter
from shared.video_fx import (
    _PANS, cover_fit_png, ffmpeg_has_encoder as _ffmpeg_has_encoder,
    pick_transitions, kenburns_crop_expr,
)
from shared.timing import step

logger = logging.getLogger("contentfactory.stage_05")
_ST = "stage_05_compose"

# Geometry / timing
W, H, FPS = 1080, 1920, 30
OPEN, CLOSE, TAIL = 0.6, 0.8, 0.6          # open/close fades (s) and end hold
XF_MAX = 0.45                              # max transition length (s)
TRANSITIONS = True
KEN_BURNS = True
KEN_BURNS_ZOOM = 1.06
USE_GPU_ENCODE = True


def build_video(reel, work_dir, out_path, music_volume=None):
    scenes = reel["scenes"]
    n = len(scenes)
    total = reel["total"]
    vid_len = total + TAIL

    d = [max(0.30, s["t_end"] - s["t_start"]) for s in scenes]
    D = (min(XF_MAX, 0.6 * min(d)) if (TRANSITIONS and n > 1) else 0.06)

    use_kb = bool(KEN_BURNS) and float(KEN_BURNS_ZOOM) > 1.001
    OW = int(round(W * float(KEN_BURNS_ZOOM))) if use_kb else W
    OH = int(round(H * float(KEN_BURNS_ZOOM))) if use_kb else H
    px, py = OW - W, OH - H

    fdir = os.path.join(work_dir, "frames")
    os.makedirs(fdir, exist_ok=True)
    with step(_ST, "frames_prep", kind="cpu", n=len(scenes)):
        frames = [cover_fit_png(s["image"], os.path.join(fdir, f"s{i:03d}.png"), OW, OH)
                  for i, s in enumerate(scenes)]

    # Per-word-state rolling subtitles (gold context + white current word).
    ass_path = os.path.join(work_dir, "subs.ass")
    with step(_ST, "ass_build", kind="cpu"):
        ass = build_ass_rolling(reel, ass_path, reel.get("accent_hex"))
    ass = ass.replace("\\", "\\\\").replace(":", "\\:")

    durs, inputs = [], []
    for i in range(n):
        dur = d[i] + (D if (n > 1 and i < n - 1) else 0.0)
        if i == n - 1:
            dur = d[i] + TAIL
        durs.append(dur)
        inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-i", frames[i]]

    inputs += ["-i", reel["narration_path"], "-i", reel["music_path"]]
    na_idx, mu_idx = n, n + 1

    # Ken Burns pan/zoom with bounds-guarded crop.
    vp = ""
    for i in range(n):
        if use_kb and (px > 0 or py > 0):
            sx, sy, ex, ey = _PANS[i % len(_PANS)]
            DUR = max(0.30, durs[i])
            xexpr, yexpr = kenburns_crop_expr(px, py, sx, sy, ex, ey, DUR)
            vp += (f"[{i}:v]fps={FPS},setpts=PTS-STARTPTS,"
                   f"crop={W}:{H}:x='{xexpr}':y='{yexpr}',"
                   f"settb=AVTB,format=yuv420p,setsar=1[v{i}];")
        else:
            vp += f"[{i}:v]fps={FPS},setpts=PTS-STARTPTS,settb=AVTB,format=yuv420p,setsar=1[v{i}];"

    # Energy-grounded transitions, one per scene boundary.
    transitions = pick_transitions(scenes) if (TRANSITIONS and n > 1) else []
    chain = ""
    prev = "v0"
    acc = 0.0
    for k in range(1, n):
        acc += d[k - 1]
        t = transitions[k - 1] if k - 1 < len(transitions) else "fade"
        out = f"x{k}"
        chain += f"[{prev}][v{k}]xfade=transition={t}:duration={D:.3f}:offset={acc:.3f}[{out}];"
        prev = out

    fo = max(0.0, vid_len - CLOSE)
    chain += f"[{prev}]fade=t=in:st=0:d={OPEN},fade=t=out:st={fo:.3f}:d={CLOSE},ass='{ass}'[v];"

    # Loudness-relative BGM mix: music = music_volume × narration loudness, sidechain-ducked.
    mv = music_volume if music_volume is not None else config.BGM_LOUDNESS_RATIO
    with step(_ST, "lufs_measure", kind="cpu"):
        bgm_vol, vol_desc = bgm_volume_filter(reel["narration_path"], reel["music_path"], mv)
    logger.info("BGM mix: %s", vol_desc)
    mfo = max(0.0, vid_len - 3.0)
    chain += f"[{na_idx}:a]aresample=async=1:first_pts=0,asplit=2[na][nasc];"
    chain += f"[{mu_idx}:a]{bgm_vol},afade=t=out:st={mfo:.3f}:d=3[bg0];"
    chain += "[bg0][nasc]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[bg];"
    chain += "[na][bg]amix=inputs=2:duration=longest:normalize=0[a]"

    NVENC = bool(USE_GPU_ENCODE) and _ffmpeg_has_encoder("h264_nvenc")
    if NVENC:
        VCODEC = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    else:
        VCODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", vp + chain,
           "-map", "[v]", "-map", "[a]", "-r", str(FPS),
           *VCODEC, "-pix_fmt", "yuv420p",
           "-filter_complex_threads", str(os.cpu_count() or 4),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           "-t", f"{vid_len:.3f}", out_path]

    logger.info("Running FFMPEG for final composition...")
    with step(_ST, "ffmpeg_encode", kind="cpu", encoder=("nvenc" if NVENC else "libx264"), scenes=n):
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(out_path):
        logger.error("FFMPEG failed:\n%s", "\n".join(res.stderr.splitlines()[-40:]))
        raise RuntimeError(f"ffmpeg did not produce {out_path}")

    return out_path


def _add_char_positions(scenes, script_text):
    """
    Ensure every scene has char_start/char_end into the script (stage_01 doesn't set them).
    Locates each scene's `narration` snippet in the full script sequentially. Without this,
    all scenes map to t≈0 and the images flash by in the first seconds.
    """
    cursor = 0
    for scene in scenes:
        narr = (scene.get("narration") or "").strip()
        if not narr:
            scene["char_start"] = scene["char_end"] = cursor
            continue
        pos = script_text.find(narr, cursor)
        if pos < 0 and len(narr) >= 20:
            pos = script_text.find(narr[:20], cursor)   # tolerate minor punctuation drift
        if pos < 0:
            pos = cursor
        scene["char_start"] = pos
        scene["char_end"] = pos + len(narr)
        cursor = pos + len(narr)


def compose_video(job_data: dict, narration_path: str, music_path: str, image_paths: list,
                  out_path: str, work_dir: str):
    """Entry point from run.py to compose the final video."""
    full_text = job_data.get("script", "")
    align = job_data.get("alignment", {})
    chars = align.get("characters", [])
    starts = align.get("character_start_times_seconds", [])
    ends = align.get("character_end_times_seconds", [])

    words = words_from_alignment(full_text, chars, starts, ends)
    total = float(ends[-1]) if ends else 0.0

    scenes = job_data.get("scenes", [])
    aligned_ok = abs(len(chars) - len(full_text)) <= 2

    # stage_01 scenes don't carry char offsets — derive them so scene timing isn't all ≈0.
    if scenes and not any(s.get("char_start") for s in scenes):
        _add_char_positions(scenes, full_text)

    for idx, s in enumerate(scenes):
        char_start = s.get("char_start", 0)
        char_end = s.get("char_end", 0)
        if aligned_ok and char_end <= len(starts):
            s["t_start"] = float(starts[char_start])
            s["t_end"] = float(ends[min(char_end, len(ends)) - 1])
        else:
            L = max(1, len(full_text))
            s["t_start"] = total * char_start / L
            s["t_end"] = total * char_end / L
        s["t_start"] = max(0.0, s["t_start"])
        s["t_end"] = max(s["t_start"] + 0.2, s["t_end"])

        if idx < len(image_paths):
            s["image"] = image_paths[idx]
        else:
            s["image"] = image_paths[-1] if image_paths else ""

    for a, b in zip(scenes, scenes[1:]):
        a["t_end"] = b["t_start"]
    if scenes:
        scenes[-1]["t_end"] = total

    # Optional per-story accent for context words; None → the approved gold/white look.
    palette = (job_data.get("meta", {}).get("style", {}) or {}).get("palette_hex", []) or []
    accent_hex = None   # keep the finalized gold scheme; set to palette[0] to tint per-story

    reel = {
        "slug": f"story_{job_data.get('meta', {}).get('story_num', 0)}",
        "scenes": scenes,
        "words": words,
        "total": total,
        "narration_path": narration_path,
        "music_path": music_path,
        "accent_hex": accent_hex,
    }

    build_video(reel, work_dir, out_path, music_volume=config.BGM_LOUDNESS_RATIO)
    return True

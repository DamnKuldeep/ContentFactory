"""
Content Factory — shared video FX helpers for composition.

Ken Burns crop expressions (bounds-guarded), energy-grounded xfade transition
selection, image cover-fit, and ffmpeg encoder detection. Used by
stage_05_compose/compose.py (and re-exposed there for the legacy scripts).
"""

import subprocess

from PIL import Image

# ── Ken Burns pan vectors (start-x, start-y, end-x, end-y), normalised 0..1 ────
_PANS = [
    (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5),
    (0.5, 0.0, 0.5, 1.0), (0.5, 1.0, 0.5, 0.0), (0.5, 0.5, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0),
]

# ── Energy-grounded xfade pools, keyed by scene tier ──────────────────────────
_ENERGY_TRANSITIONS = {
    "LOW":  ["fade", "dissolve", "fadegrays", "fadeslow", "distance"],
    "MED":  ["smoothleft", "smoothright", "smoothup", "smoothdown",
             "horzopen", "vertopen", "coverleft", "coverright"],
    "HIGH": ["wipeleft", "wiperight", "diagbr", "diagtl", "radial",
             "circleopen", "coverleft", "coverright", "hlslice", "vuslice"],
    "PEAK": ["pixelize", "circleclose", "zoomin", "squeezev", "hblur",
             "fadeblack", "fadewhite", "fadefast", "rectcrop"],
}

# High-energy keywords for scene tier scoring
_HIGH_WORDS = {
    "fight", "death", "scream", "shock", "crash", "betray", "confess",
    "reveal", "collapse", "kill", "murder", "poison", "fire", "blood",
    "horror", "escape", "drown", "fall", "danger", "attack",
}


def ffmpeg_has_encoder(name):
    """True if this ffmpeg build lists the given encoder (e.g. h264_nvenc)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
        return name in out
    except Exception:
        return False


def cover_fit_png(path, dst, tw, th):
    """Resize + center-crop an image to exactly tw×th (cover fit), saved as PNG."""
    im = Image.open(path).convert("RGB")
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - tw) // 2, (nh - th) // 2
    im.crop((l, t, l + tw, t + th)).save(dst)
    return dst


def score_scenes(scenes):
    """Assign LOW/MED/HIGH/PEAK tier to each scene from word count + high-energy keyword hits."""
    import re
    scores = []
    for s in scenes:
        words = re.sub(r"[^\w\s]", "", s.get("narration", "")).lower().split()
        hit   = sum(1 for w in words if any(k in w for k in _HIGH_WORDS))
        scores.append(len(words) + 3 * hit)

    n = len(scores)
    if n < 4:
        return ["MED"] * n
    srt = sorted(scores)
    q   = [srt[max(0, int(n * p) - 1)] for p in (0.25, 0.50, 0.75)]
    tiers = []
    for sc in scores:
        if   sc <= q[0]: tiers.append("LOW")
        elif sc <= q[1]: tiers.append("MED")
        elif sc <= q[2]: tiers.append("HIGH")
        else:            tiers.append("PEAK")
    return tiers


def pick_transitions(scenes):
    """One xfade transition name per scene boundary (length = n-1), energy-grounded."""
    tiers  = score_scenes(scenes)
    result = []
    for i, tier in enumerate(tiers[:-1]):
        pool = _ENERGY_TRANSITIONS[tier]
        result.append(pool[i % len(pool)])
    return result


def kenburns_crop_expr(px, py, sx, sy, ex, ey, dur):
    """
    Bounds-guarded Ken Burns crop x/y expressions for ffmpeg `crop`.

    px/py = max pan offset (oversized image minus target). The clip() guard keeps
    the crop origin inside the image during xfade overlap (prevents edge bleed).
    """
    xexpr = f"clip({px:.1f}*({sx}+({ex}-{sx})*t/{dur:.3f}),0,{px:.1f})"
    yexpr = f"clip({py:.1f}*({sy}+({ey}-{sy})*t/{dur:.3f}),0,{py:.1f})"
    return xexpr, yexpr

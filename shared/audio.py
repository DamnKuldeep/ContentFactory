"""
Content Factory — shared audio helpers.

Loudness measurement + loudness-relative BGM mixing, FFmpeg atempo chains, and
audio duration probing. Used by stage_02 (Fish narration) and stage_05 (compose).
"""

import math
import re
import subprocess


def measure_lufs(path):
    """Integrated loudness (LUFS) of an audio file via ffmpeg ebur128; None on failure/silence."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
             "-filter:a", "ebur128=metadata=1", "-f", "null", "-"],
            capture_output=True, text=True).stderr
        vals = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", out)
        if vals:
            v = float(vals[-1])
            return v if v > -70 else None      # -inf / silent guard
    except Exception:
        pass
    return None


def loudness_relative_gain(narration_path, music_path, ratio):
    """
    Compute the BGM gain so the music sits at `ratio`× the NARRATION's loudness.

    `ratio` is a linear amplitude ratio relative to the voice (e.g. 0.4 → BGM is
    0.4× the narration, i.e. 20*log10(0.4) ≈ -8 dB below it). This is NOT a raw
    scale on the BGM — narration and BGM are measured independently (Fish voice is
    quiet ~-25 LUFS, ACE-Step BGM is loud ~-15 LUFS), so a plain volume=ratio would
    leave the music on top.

    Returns (gain_db, desc). gain_db is None if either track can't be measured
    (caller should fall back to a linear `volume={ratio}`).
    """
    n_lufs = measure_lufs(narration_path)
    b_lufs = measure_lufs(music_path)
    if n_lufs is None or b_lufs is None or ratio <= 0:
        return None, f"{ratio} (linear fallback — LUFS unavailable)"
    target_bgm_lufs = n_lufs + 20.0 * math.log10(ratio)
    gain_db = target_bgm_lufs - b_lufs
    desc = (f"{ratio}×narr → {gain_db:+.1f}dB "
            f"(narr {n_lufs:.1f} / bgm {b_lufs:.1f} LUFS)")
    return gain_db, desc


def bgm_volume_filter(narration_path, music_path, ratio):
    """Return (ffmpeg `volume=` argument, human-readable description) for the BGM track."""
    gain_db, desc = loudness_relative_gain(narration_path, music_path, ratio)
    if gain_db is None:
        return f"volume={ratio}", desc
    return f"volume={gain_db:.2f}dB", desc


def atempo_chain(speed):
    """Build an ffmpeg atempo filter chain for arbitrary speed (each stage clamped to [0.5, 2.0])."""
    speed = float(speed)
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.4f}")
    return ",".join(filters)


def audio_duration(path):
    """Audio/container duration in seconds via ffprobe; 0.0 on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0

"""
Content Factory — narration analysis features for BGM grounding.

Extracts an energy envelope (librosa RMS) and a pacing curve (words-per-second)
from the narration so the music brief can track the story's energy arc.
Used by stage_04_music. The energy envelope drives the "techC" brief technique.
"""


def compute_energy_envelope(narration_mp3, window_sec=2):
    """librosa RMS per window, normalised 0→1, with whisper/quiet/medium/loud labels."""
    import librosa
    import numpy as np

    y, sr     = librosa.load(narration_mp3, sr=22050, mono=True)
    hop       = int(sr * window_sec)
    frame_len = min(int(sr * window_sec * 2), len(y)) or 1
    rms       = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    times     = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
    max_rms   = float(np.max(rms)) or 1.0
    rms_norm  = rms / max_rms
    q25, q50, q75 = (float(np.percentile(rms_norm, p)) for p in (25, 50, 75))

    result = []
    for t, r in zip(times, rms_norm):
        r     = float(r)
        label = ("whisper" if r < q25 else
                 "quiet"   if r < q50 else
                 "medium"  if r < q75 else "loud")
        result.append({"time_sec": round(float(t), 1), "rms_norm": round(r, 3), "label": label})
    return result


def describe_energy_arc(energy_envelope):
    """Return a one-line description of the overall energy arc (thirds comparison)."""
    if not energy_envelope:
        return "energy data unavailable"
    vals   = [e["rms_norm"] for e in energy_envelope]
    n      = len(vals)
    thirds = [vals[:n // 3], vals[n // 3: 2 * n // 3], vals[2 * n // 3:]]
    avgs   = [sum(t) / len(t) if t else 0.0 for t in thirds]
    if avgs[1] > avgs[0] and avgs[1] > avgs[2]:
        return "energy peaks in the middle section"
    if avgs[2] > avgs[0] and avgs[2] > avgs[1]:
        return "energy builds toward the end"
    if avgs[0] > avgs[1] and avgs[0] > avgs[2]:
        return "energy is highest at the opening"
    return "energy varies throughout"


def energy_envelope_text(energy_envelope):
    """Render the envelope as the compact text block embedded in the techC music brief."""
    if not energy_envelope:
        return "(energy data unavailable — use story context only)"
    sampled = energy_envelope[::2][:40]   # one reading every ~4s, capped
    parts   = [f'{e["time_sec"]}s:{e["rms_norm"]:.2f}({e["label"][0].upper()})' for e in sampled]
    table   = "  " + " | ".join(parts)
    return (
        "Narration audio energy (librosa RMS, normalised 0→1) every 4 seconds\n"
        "  Labels: W=whisper Q=quiet M=medium L=loud\n"
        f"{table}\n"
        f"Overall: {describe_energy_arc(energy_envelope)}."
    )


def compute_pacing_wps(alignment, window_sec=5):
    """Words-per-second per window from char-level alignment, with calm/.../rapid labels."""
    chars  = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends   = alignment.get("character_end_times_seconds", [])
    if not chars or not starts or len(chars) != len(starts):
        return []

    words        = []
    word_chars   = []
    word_start_t = None
    for i, ch in enumerate(chars):
        if ch in (" ", "\n", "\r", "\t"):
            if word_chars:
                words.append({"start": word_start_t, "end": ends[i - 1]})
                word_chars, word_start_t = [], None
        else:
            if word_start_t is None:
                word_start_t = starts[i]
            word_chars.append(ch)
    if word_chars and word_start_t is not None:
        words.append({"start": word_start_t, "end": ends[-1]})
    if not words:
        return []

    total_dur = ends[-1]
    result, i = [], 0
    while i * window_sec < total_dur:
        w_start = i * window_sec
        w_end   = min((i + 1) * window_sec, total_dur)
        count   = sum(1 for w in words if w_start <= w["start"] < w_end)
        actual  = w_end - w_start
        wps     = count / actual if actual > 0 else 0.0
        label   = ("calm"     if wps < 1.5 else
                   "moderate" if wps < 2.5 else
                   "brisk"    if wps < 3.5 else "rapid")
        result.append({"start": round(w_start, 1), "end": round(w_end, 1),
                       "wps": round(wps, 2), "label": label})
        i += 1
    return result

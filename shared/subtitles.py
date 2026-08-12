"""
Content Factory — shared subtitle (ASS) helpers.

Canonical home for the per-word-state rolling karaoke subtitle builder plus the
char-alignment → word-list conversion and ASS formatting helpers. Imported by
stage_05_compose/compose.py (and re-exposed from there for the legacy scripts).
"""

# ── Subtitle style ────────────────────────────────────────────────────────────
FONT_NAME           = "DejaVu Sans"
SUB_BASE_SIZE       = 66
SUB_WINDOW          = 4          # words visible on screen at once
SUB_WORD_COLOR      = "#FFD27F"  # warm gold — all context words
SUB_HIGHLIGHT_COLOR = "#FFFFFF"  # white pop — current spoken word
SUB_FALLBACK_COLOR  = "#d97b5c"  # used only if a caller passes a bad accent
SUB_POP_SCALE       = 140        # current word scale-up (%)


def _hex_to_ass(hex_color):
    """Convert #RRGGBB to ASS &H00BBGGRR (BGR byte order)."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return "&H0000D7FF"      # gold fallback
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _fmt(sec):
    """Seconds → ASS timecode H:MM:SS.cs."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _esc(s):
    """Escape ASS special chars (braces start override blocks; newlines break lines)."""
    return s.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def words_from_alignment(full_text, chars, starts, ends):
    """Char-level alignment → word list with start/end times and char offsets."""
    n = min(len(chars), len(starts), len(ends))
    words, i = [], 0
    src = "".join(chars[:n]) if "".join(chars[:n]).strip() else full_text
    while i < n:
        if src[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not src[j].isspace():
            j += 1
        words.append({"text": src[i:j], "start": float(starts[i]), "end": float(ends[j - 1]),
                      "cstart": i, "cend": j})
        i = j
    return words


def build_ass_rolling(reel, path, accent_hex=None):
    """
    Per-word-state rolling karaoke subtitles (the finalized look).

    A sliding window of SUB_WINDOW words is shown; within it the currently-spoken
    word is white + enlarged (SUB_POP_SCALE%) and the rest are warm gold. One
    Dialogue line is emitted per word-state, precisely abutted (no flicker, no
    layout jump). `accent_hex`, if given, overrides the context-word colour;
    default None → the approved gold/white scheme.
    """
    ass_fill = _hex_to_ass(accent_hex) if accent_hex else _hex_to_ass(SUB_WORD_COLOR)
    white    = _hex_to_ass(SUB_HIGHLIGHT_COLOR)
    pop      = SUB_POP_SCALE

    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: K,{FONT_NAME},{SUB_BASE_SIZE},"
        f"{ass_fill},{ass_fill},&H00000000,&H96000000,"
        "-1,0,0,0,100,100,0,0,1,3.5,1.5,2,90,90,280,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    all_words = reel["words"]
    lines = []
    for i in range(0, len(all_words), SUB_WINDOW):
        win = all_words[i:i + SUB_WINDOW]
        win_end = (all_words[i + SUB_WINDOW]["start"]
                   if i + SUB_WINDOW < len(all_words)
                   else win[-1]["end"] + 0.5)

        for k, w in enumerate(win):
            t0 = w["start"]
            t1 = win[k + 1]["start"] if k + 1 < len(win) else win_end
            if t1 <= t0:
                t1 = t0 + 0.05

            parts = []
            reset_needed = False
            for j, ww in enumerate(win):
                if j == k:
                    parts.append(r"{\1c" + white + r"\fscx%d\fscy%d}" % (pop, pop) + _esc(ww["text"]))
                    reset_needed = True
                elif reset_needed:
                    parts.append(r"{\1c" + ass_fill + r"\fscx100\fscy100}" + _esc(ww["text"]))
                    reset_needed = False
                else:
                    parts.append(_esc(ww["text"]))

            lines.append(f"Dialogue: 0,{_fmt(t0)},{_fmt(t1)},K,,0,0,0,,{' '.join(parts)}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    return path

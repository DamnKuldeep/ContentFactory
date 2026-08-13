# Stage 5 — video

Pulls down every image, the narration and the music, and builds one big FFmpeg filter graph: Ken Burns panning, energy-graded crossfades, per-word karaoke subtitles, and a loudness-matched audio mix with the music ducking under the voice.

## What it needs

- **GPU:** optional. An Nvidia card gets you `h264_nvenc`; without one it falls back to `libx264` automatically.
- **RAM:** 8 GB
- **Disk:** 10 GB of headroom — it downloads all 34 images plus both audio tracks, writes the video, uploads it, then deletes the lot.
- **Network:** this is the bandwidth-heavy stage. Nothing can start until every image is down.

## Settings

Full `.env` in [RUNME.md](../RUNME.md#the-env-file).

- `BGM_LOUDNESS_RATIO` (default `0.42`) — where the music sits against the narration's measured loudness. Lower is quieter.
- The usual Drive variables

## Setup

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg    # system dependency, not pip
python scripts/setup_compose.py                           # ../.venv_compose
source ../.venv_compose/bin/activate
```

## Running

```bash
python scripts/worker.py compose                                # bulk
python run.py --stage 5 --batch <id> --drive-db --workers 1     # staged
```

## Speed

Per video, 34 scenes, ~90 seconds of output:

| | Time |
|---|---|
| Downloading images, narration and music | 59 s |
| Preparing frames | 20 s |
| ffmpeg with NVENC | 23 s |
| ffmpeg with libx264 instead | 1–2 min depending on cores |

Roughly half the stage is downloading. Downloads are deliberately serial — Google's Drive client isn't thread-safe, and a parallel pool produced 0-byte files and segfaults. If you want them parallel, build a separate client per thread.

## If it goes wrong

**ffmpeg not found** — it needs to be on `$PATH`, and it's a system package, not a pip one.

**ffmpeg crashes on the subtitle filter** — the ASS builder escapes braces and newlines, but something odd in the generated script can still get through. Look at the script text for that story.

**NVENC fails on an Nvidia card** — you need the proprietary drivers. Without them it detects the missing encoder and falls back to CPU on its own.

**Images flash past at the start** — that was a real bug, fixed. If you see it, the scenes are missing their character offsets and `_add_char_positions()` didn't manage to locate their narration in the script.

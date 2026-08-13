# Stage 2 — narration

Reads the script out loud and works out when every character of it was spoken. Default is Fish Speech S2 Pro on your own GPU with WhisperX doing the alignment. Set `NARRATION_ENGINE=elevenlabs` to use the API instead — both return the same timestamps, so nothing downstream notices.

## What it needs

- **GPU:** yes for Fish, about 10 GB of VRAM including WhisperX. None at all if you're using ElevenLabs.
- **RAM:** 16 GB for Fish, 4 GB for ElevenLabs. More if you turn on `FISH_COMPILE`, which is hungry for system RAM while it builds.
- **Disk:** ~10 GB for the Fish checkpoints, cached under `models/`
- **Network:** outbound to Drive, plus ElevenLabs if you're using it

## Settings

Full `.env` in [RUNME.md](../RUNME.md#the-env-file).

- `NARRATION_ENGINE` — `fish` (default) or `elevenlabs`
- `ELEVENLABS_API_KEY` — only if you picked elevenlabs
- `FISH_COMPILE=1` — roughly halves generation time, but only worth it if this process is doing lots of videos
- `HF_HOME`, plus the usual Drive variables

## Setup

```bash
python scripts/setup_narration.py    # ../.venv_narration, Fish + WhisperX, applies the Fish patches
source ../.venv_narration/bin/activate
```

The setup script patches two files in the Fish repo (`tokenizer_config.json` and `llama.py`). If stage 2 ever starts failing on a tokenizer error, just re-run setup.

## Running

```bash
python scripts/worker.py narration                              # bulk
python run.py --stage 2 --batch <id> --drive-db --workers 1     # staged
```

Use one worker. It's a single GPU.

## Speed

Measured on an RTX 5090 laptop with 88 seconds of narration:

| | Time |
|---|---|
| Loading Fish | 40 s, once per process |
| Generating audio | 128 s, or **60 s with `FISH_COMPILE=1`** once warm |
| WhisperX transcribe + align | under 3 s |

TTS is the single biggest GPU step in the whole pipeline, which is why the compile flag matters so much here.

With ElevenLabs instead it's 10–30 seconds a story and scales horizontally as far as your concurrency quota allows.

## If it goes wrong

**Tokenizer errors on import** — re-run `setup_narration.py`.

**Audio comes out suspiciously short** — the code warns about this. Fish occasionally truncates a chunk; re-running the job usually fixes it.

**ElevenLabs 401 or 429** — check the key and your concurrency quota. The worker sleeps and retries.

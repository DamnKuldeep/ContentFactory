# Stage 4 — music

Pulls an energy curve out of the finished narration, has Qwen2.5-Omni write a music brief that follows it, then hands that brief to ACE-Step 1.5 to generate the track.

## What it needs

- **GPU:** 24 GB. Note that Qwen and ACE-Step **do not fit at the same time** — the code explicitly evicts Qwen from VRAM before ACE-Step loads. On exactly 24 GB you also want `BGM_TWO_PASS=0`, because the two-pass flow needs both models available.
- **RAM:** 64 GB is comfortable. Audio tensors during generation are large.
- **Disk:** ~50 GB for Qwen plus the ACE-Step checkpoints
- **Network:** fast for the initial weight download

## Settings

Full `.env` in [RUNME.md](../RUNME.md#the-env-file).

- `ACE_REPO_DIR` and `ACE_CHECKPOINTS` — you provision ACE-Step yourself; point these at it
- `BGM_TWO_PASS` — `1` for the verify-and-refine second pass, `0` for single. **Use 0 on 24 GB.**
- `HF_HOME`, plus the usual Drive variables

## Setup

```bash
python scripts/setup_bgm.py    # ../.venv_bgm, downloads Qwen2.5-Omni (~21 GB), checks ACE-Step
source ../.venv_bgm/bin/activate
```

ACE-Step is fussy about torchaudio and torchao versions, which is exactly why it gets its own venv away from Fish and FLUX.

## Running

```bash
python scripts/worker.py music                                  # bulk
python run.py --stage 4 --batch <id> --drive-db --workers 1     # staged
```

## Speed

Per video, on an RTX 5090 laptop:

| | Time |
|---|---|
| Loading Qwen | 12 s |
| Qwen writing the brief | 36 s |
| Loading ACE-Step | 22 s |
| Generating the track | 10 s |

Generation itself is quick. Most of the stage is loading two large models, which is why the bulk worker path — load once, do everything — matters here.

## If it goes wrong

**`libsndfile` errors on Linux** — `sudo apt-get install libsndfile1`.

**OOM during ACE-Step decode** — set `BGM_TWO_PASS=0` first. If it still OOMs, uncomment `model.enable_cpu_offload()` in `ace.py`.

**"Qwen JSON generation failed after retries"** — Qwen returned something that isn't valid JSON three times running. The job goes back in the queue; it usually succeeds on a retry.

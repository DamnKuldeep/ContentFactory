# Stage 3 — images

One illustration per scene, 28–40 per video, from FLUX.2-klein running locally. It checks how much VRAM the card has and picks a batch size to match.

## What it needs

- **GPU:** 24 GB is comfortable. 16 GB works at batch size 2, 12 GB at batch size 1.
- **RAM:** 32 GB+
- **Disk:** ~50 GB for the FLUX weights. Generated images are uploaded and deleted immediately, so runtime disk use stays flat.
- **Network:** fast, and ideally stable. The first run downloads the model; every run after that uploads 30-odd PNGs per video.

## Settings

Full `.env` in [RUNME.md](../RUNME.md#the-env-file).

- `FLUX_COMPILE=1` — about 18% faster
- `FLUX_COMPILE_MODE=default` — **do not** use `reduce-overhead` on 24 GB. Its CUDA graphs reserve an extra 2.7 GB and OOM you after roughly 20 images.
- `HF_HOME`, plus the usual Drive variables

## Setup

```bash
python scripts/setup_images.py    # ../.venv_images, downloads FLUX.2-klein (~23 GB)
source ../.venv_images/bin/activate
```

## Running

```bash
python scripts/worker.py images                                 # bulk
python run.py --stage 3 --batch <id> --drive-db --workers 1     # staged
```

Interrupt it whenever. On restart it lists what's already in the Drive images folder for that story and skips those scenes, so a half-finished render carries on instead of starting over.

## Speed

Batches of 4 at 4 inference steps, on an RTX 5090 laptop:

| | Time |
|---|---|
| Loading FLUX | ~6 s |
| A batch of 4 images | 13.5 s, or **11.0 s with `FLUX_COMPILE=1`** |
| 34 images total | ~115 s of GPU |

Uploads used to add another 125 seconds with the GPU idle. They now happen on a background thread while the next batch renders, so that time mostly disappears.

## If it goes wrong

**CUDA OOM** — use `--workers 1`, check `FLUX_COMPILE_MODE=default`, or drop `BATCH_SIZE` in `generate.py`.

**The first run seems to hang** — it's downloading ~23 GB of weights. Let it finish.

**Lots of SSL retries or a worker dying with `rc=139`** — that's the network dropping TLS mid-upload. The upload pool is already serialised to avoid the OpenSSL race that used to crash it; just re-run and it resumes from what's already uploaded.

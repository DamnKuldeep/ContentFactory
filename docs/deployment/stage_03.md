# Stage 03: Image Generation Runbook

This node fetches the story script (`stage_01.json`) and runs FLUX.2 locally to generate all scene images. It actively calculates available VRAM to determine batch sizing (e.g. 1, 2, or 4 simultaneous images). 

## Hardware Requirements
- **GPU:** RTX 3090 / 4090 / 5090 (24GB+ VRAM strongly recommended)
- **RAM:** 32GB+
- **Disk:** 50GB+ (Required to hold the downloaded FLUX.2 diffusers model weights. Generated images are instantly uploaded and deleted, so runtime disk usage stays flat.)
- **Network:** High-speed internet required for initial model download.

## Environment Variables
See [../RUNME.md §3](../RUNME.md#3-configure-env). Auth is OAuth, refresh-only.
- `DRIVE_PARENT_FOLDER_ID`, `DRIVE_CLIENT_SECRETS`, `DRIVE_TOKEN_PATH`, `DRIVE_DB=1`.
- `HF_HOME`: repo-local model cache (`models/hf_cache`).

## Installation
```bash
# Creates ../.venv_images, installs deps, downloads FLUX.2-klein weights into the HF cache
python scripts/setup_images.py
source ../.venv_images/bin/activate
```

## Running the Node
```bash
python run.py --stage 3 --batch <id> --drive-db --workers 1
```
*(The node will claim pending Stage 3 jobs from the SQLite manifest. If you interrupt it via Ctrl+C, the next time it boots, it will query Google Drive to see which images were already successfully uploaded for that story and seamlessly resume without wasting compute.)*

## Expected Throughput
- **RTX 5090 (32GB VRAM):** The dynamic resizer will pick `batch_size = 4`. With 4 inference steps, expect ~1 second per image. A 50-scene story takes ~15 seconds of compute.
- **RTX 4080 (16GB VRAM):** The dynamic resizer will pick `batch_size = 2`. Expect ~3 seconds per image.
- **12GB GPUs:** Will fall back to `batch_size = 1`.

## Troubleshooting
- **OOM Errors:** If CUDA Out Of Memory occurs, manually override `batch_size = 1` in `generate.py` or rent a machine with more VRAM.
- **Model Download Failures:** The first execution will hang while HuggingFace downloads ~18GB of safetensors. Let it finish.

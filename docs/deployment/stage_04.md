# Stage 04: Music Generation Runbook

This node analyzes the script via `Qwen2.5-Omni` to generate a structural musical brief, and then feeds that brief alongside the narration MP3 into `ACE-Step` (a heavyweight audio generation model) to produce an emotional backing track.

## Hardware Requirements
- **GPU:** RTX 3090 / 4090 / 5090 (24GB+ VRAM required to hold both Qwen and ACE-Step simultaneously, or they will page to RAM).
- **RAM:** 64GB+ (Crucial, as large audio tensors during 5-minute waveform generation will consume system RAM).
- **Disk:** 50GB+ (Required to hold Qwen and ACE-Step weights).
- **Network:** High bandwidth for initial weight downloading.

## Environment Variables
See [../RUNME.md §3](../RUNME.md#3-configure-env). Auth is OAuth, refresh-only.
- `DRIVE_PARENT_FOLDER_ID`, `DRIVE_CLIENT_SECRETS`, `DRIVE_TOKEN_PATH`, `DRIVE_DB=1`, `HF_HOME`.
- `ACE_REPO_DIR` / `ACE_CHECKPOINTS`: ACE-Step repo + checkpoints (provisioned out-of-band).
- `BGM_TWO_PASS` (default `1`): two-pass techC refinement.

## Installation
```bash
# Creates ../.venv_bgm, installs deps, downloads Qwen2.5-Omni; validates ACE-Step
python scripts/setup_bgm.py
source ../.venv_bgm/bin/activate
```
*Note: ACE-Step deps are brittle on torchaudio/torchao versions; the venv isolates them from Fish/FLUX.*

## Running the Node
```bash
python run.py --stage 4 --batch <id> --drive-db --workers 1
```
*(Claims pending Stage 4 jobs. Stage 4 reads the narration energy envelope, writes an ACE-Step
brief with Qwen, generates music, and — if `BGM_TWO_PASS` — verifies/refines against the LM blueprint.)*

## Expected Throughput
- Generating a 3-minute backing track generally takes ~2-4 minutes of compute on an RTX 4090 depending on ACE-Step CFG scale and chunking. 

## Troubleshooting
- **Librosa / Soundfile Warnings:** If you get `libsndfile` errors on Linux, run `sudo apt-get install libsndfile1`.
- **OOM Errors:** If CUDA Out Of Memory occurs during the ACE-Step decode phase, it means your sequence length is too long for the VRAM. You must enable aggressive offloading in `ace.py` by uncommenting `model.enable_cpu_offload()`.

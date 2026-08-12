# Stage 02: Narration Generation Runbook

This node takes the completed story scripts from Stage 1 and synthesizes narration + char-level
forced-alignment timestamps. Default engine is **Fish S2 Pro (local GPU) + WhisperX**; set
`NARRATION_ENGINE=elevenlabs` to use the ElevenLabs API instead.

## Hardware Requirements
- **GPU:** Required for Fish (NF4, ~10GB VRAM incl. WhisperX). None if `NARRATION_ENGINE=elevenlabs`.
- **RAM:** 16GB+ (Fish) / 4GB+ (ElevenLabs)
- **Disk:** ~10GB for Fish checkpoints (cached under `models/`), else negligible.
- **Network:** Outbound internet (Google Drive; ElevenLabs if used).

## Environment Variables
See [../RUNME.md §3](../RUNME.md#3-configure-env). Auth is OAuth, refresh-only (`scripts/drive_auth.py`
once, then copy `credentials.json` + `token.json`).
- `NARRATION_ENGINE`: `fish` (default) or `elevenlabs`.
- `ELEVENLABS_API_KEY`: only if `elevenlabs`.
- `DRIVE_PARENT_FOLDER_ID`, `DRIVE_CLIENT_SECRETS`, `DRIVE_TOKEN_PATH`, `DRIVE_DB=1`, `HF_HOME`.

## Installation
```bash
# Creates ../.venv_narration, installs Fish+WhisperX deps, downloads checkpoints, applies patches
python scripts/setup_narration.py
source ../.venv_narration/bin/activate
```

## Running the Node
```bash
python run.py --stage 2 --batch <id> --drive-db --workers 1
```
*(Claims pending Stage 2 jobs from the manifest until none remain; `--workers 1` for the GPU.)*

## Expected Throughput
- ~10-30 seconds per story, strictly bottlenecked by ElevenLabs API generation speed.
- Easily scales horizontally. You can run 50 of these nodes simultaneously to chew through a backlog, provided you do not hit ElevenLabs concurrency limits.

## Troubleshooting
- **API Errors:** If ElevenLabs returns a 401/429, verify your API key and concurrency quota. The worker will naturally sleep and retry on transient failures.

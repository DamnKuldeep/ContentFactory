# Stage 05: Composition Runbook

This node takes all distributed artifacts (Images, Narration audio, Music audio, and the JSON state file with word alignment) and orchestrates FFMPEG to compile the final video. It applies dynamic Ken Burns panning, crossfades, precise Karaoke subtitle styling, and sidechain audio ducking.

## Hardware Requirements
- **GPU:** Optional but Highly Recommended (Nvidia GPU supports `h264_nvenc` encoding).
- **RAM:** 8GB+
- **Disk:** 10GB+ (Needs room to temporarily download all FLUX.2 images, the MP3s, and output the final MP4 before uploading).
- **Network:** Very high bandwidth. This node must download every single image from Google Drive for the story before it can run FFMPEG.

## Environment Variables
See [../RUNME.md §3](../RUNME.md#3-configure-env). Auth is OAuth, refresh-only.
- `DRIVE_PARENT_FOLDER_ID`, `DRIVE_CLIENT_SECRETS`, `DRIVE_TOKEN_PATH`, `DRIVE_DB=1`.
- `BGM_LOUDNESS_RATIO` (default `0.4`): music level relative to the narration's loudness.

## Installation
```bash
# System dependency required!
sudo apt-get update && sudo apt-get install -y ffmpeg

# Creates ../.venv_compose and installs deps
python scripts/setup_compose.py
source ../.venv_compose/bin/activate
```

## Running the Node
```bash
python run.py --stage 5 --batch <id> --drive-db --workers 1
```

## Expected Throughput
- If running on a GPU with NVENC (`USE_GPU_ENCODE = True` in `compose.py`), encoding a 3-minute 1080x1920 30FPS video takes roughly 20-30 seconds.
- If falling back to CPU `libx264`, expect encoding to take 1-2 minutes depending on your core count.
- The majority of wall-clock time for this stage is spent downloading the PNG sequence from Google Drive.

## Troubleshooting
- **FFMPEG Not Found:** Ensure `ffmpeg` is accessible in the system `$PATH`.
- **Subtitle Errors:** The `.ass` subtitle generator escapes most bracketed characters, but if FFMPEG crashes on the subtitle filter, it is likely due to a malformed special character in the LLM's generated script.
- **NVENC Fails:** If you are on an Nvidia GPU but FFMPEG fails, install the proprietary Nvidia drivers. The node will automatically fallback to CPU if the `h264_nvenc` encoder isn't detected.

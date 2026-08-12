# Stage 01: Story Generation Runbook

This node is responsible for taking base topics (from `Notebooks/` or custom generators) and turning them into full JSON scripts with narration dialogue, visual prompts, and metadata using the OpenRouter LLM APIs.

## Hardware Requirements
- **GPU:** None (Runs on CPU)
- **RAM:** 4GB+
- **Disk:** 5GB+ (negligible artifact footprint since JSON is small)
- **Network:** Requires reliable outbound internet (OpenRouter, Google Drive)

## Environment Variables
See [../RUNME.md §3](../RUNME.md#3-configure-env) for the full `.env`. Auth is OAuth, refresh-only:
authorize once via `scripts/drive_auth.py`, then copy `credentials.json` + `token.json` to each machine.
- `OPENROUTER_API_KEY`: Required.
- `DRIVE_PARENT_FOLDER_ID`: Master Drive folder.
- `DRIVE_CLIENT_SECRETS` / `DRIVE_TOKEN_PATH`: OAuth credentials / token.
- `DRIVE_DB=1`: shared manifest lives in Drive.

## Installation
```bash
# Creates ../.venv_story and installs deps (no model downloads on stage 1)
python scripts/setup_story.py
source ../.venv_story/bin/activate
```

## Running the Node
```bash
python run.py --stage 1 --count 100 --drive-db
```
*(`--count` = stories to seed into a new batch. The run prints the batch id for downstream machines.)*

## Expected Throughput
- ~1-3 minutes per story depending on LLaMA/Qwen API latency.
- Negligible disk IO.

## Troubleshooting
- **RateLimits/Timeouts:** The `shared.llm` module handles HTTP 429s automatically with exponential backoff.
- **Empty completions:** The retry loop will attempt up to 8 times for structured JSON. If it hard fails, verify your `OPENROUTER_API_KEY` has credits.

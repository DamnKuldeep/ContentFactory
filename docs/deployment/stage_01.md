# Stage 1 — stories

Takes nothing and produces a JSON file per video: the story, the spoken script, 28–40 scenes with an image prompt each, character and setting sheets, and a visual style. All of it through OpenRouter, so this box does no GPU work at all.

## What it needs

- **GPU:** none
- **RAM:** 4 GB is plenty
- **Disk:** basically nothing — the output is a JSON file
- **Network:** reliable outbound to OpenRouter and Drive. It makes a *lot* of API calls.

## Settings

Full `.env` in [RUNME.md](../RUNME.md#the-env-file). Authorise Drive once with `scripts/drive_auth.py`, then copy `credentials.json` and `token.json` here.

- `OPENROUTER_API_KEY` — required, and needs credit
- `DRIVE_PARENT_FOLDER_ID`, `DRIVE_CLIENT_SECRETS`, `DRIVE_TOKEN_PATH`
- `DRIVE_DB=1` if the job database lives in Drive

## Setup

```bash
python scripts/setup_story.py     # builds ../.venv_story, no model downloads
source ../.venv_story/bin/activate
```

## Running

```bash
python scripts/produce.py --count 150            # bulk: seeds all stages, then writes stories
python run.py --stage 1 --count 100 --drive-db   # staged: prints a batch id for the other machines
```

## Speed and cost

Measured at **~13.6 minutes and about $0.10 per story**. That's a lot of wall clock, but it's all waiting on API calls, so it's the easiest stage to scale — raise `--workers` until you start hitting rate limits.

## If it goes wrong

**429s and timeouts** are normal and handled with exponential backoff in `shared/llm.py`.

**Empty or malformed completions** get retried up to 8 times with a JSON repair pass. If it still gives up, check the key has credit.

**403 total spend limit** means the key is out of budget. Jobs get marked `FAILED` with that error; raise the limit and run with `--retry-failed`.

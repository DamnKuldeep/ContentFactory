# Runbook

Provision a node, authorize Drive once, configure, run, resume, reset.

> Design behind this: **[ARCHITECTURE.md](ARCHITECTURE.md)** · overview: **[README](../README.md)**

> All commands run **from the repo root**. Per-stage virtualenvs are created one level *above* the
> repo, so they activate as `../.venv_<stage>/bin/activate`.

## Contents
1. [Prerequisites](#1-prerequisites)
2. [One-time Google Drive authorization](#2-one-time-google-drive-authorization)
3. [Configure `.env`](#3-configure-env)
4. [Per-node setup](#4-per-node-setup-venv--deps--models)
5. [Run — bulk mode (recommended)](#5-run--bulk-mode-recommended)
6. [Run — staged mode](#6-run--staged-mode)
7. [Run — single machine / local](#7-run--single-machine--local)
8. [Toggles & options](#8-toggles--options)
9. [Resume, retry & verify](#9-resume-retry--verify)
10. [Reset](#10-reset)
11. [Publishing](#11-publishing)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

- **Python 3.10+** on every machine (the setup scripts build per-stage virtualenvs with it).
- **ffmpeg** on the compose node (`sudo apt-get install -y ffmpeg`).
- **NVIDIA GPU** for stages 2 (Fish), 3 (FLUX.2), 4 (ACE-Step) — 24 GB recommended. Stages 1 and 5 are
  CPU-friendly.
- **OpenRouter API key** with credit (stage 1 only; stages 2–5 use local models).
- **Google Drive**: an OAuth *Desktop* client `credentials.json` from the Google Cloud console, and a
  target folder id.
- **ACE-Step 1.5** repo + checkpoints provisioned out-of-band for stage 4 (`ACE_REPO_DIR`,
  `ACE_CHECKPOINTS`).

Clone the repo on each machine. You do **not** install every stage's deps everywhere — each node runs
only its own stage's setup script.

---

## 2. One-time Google Drive authorization

Authorization is interactive **exactly once**, on any machine with a browser:

```bash
python scripts/drive_auth.py
# opens a browser → approve → writes token.json next to credentials.json
```

Then copy **two files** to every other machine (they are gitignored — keep them out of the repo):

```
credentials.json   token.json
```

The pipeline is **refresh-only**: it loads `token.json`, silently refreshes the access token, and never
opens a browser. Worker machines therefore never prompt for login. (Desktop OAuth refresh tokens are
not machine-bound, so the same `token.json` works everywhere.)

The requested scopes are Drive **and** Sheets — the Sheets scope is what the review queue
([PUBLISHING.md](PUBLISHING.md)) uses. If your token predates that, re-run `drive_auth.py`.

---

## 3. Configure `.env`

Copy `.env.example` to `.env` and fill it in. Minimum for a distributed run:

```dotenv
OPENROUTER_API_KEY=sk-or-...

DRIVE_PARENT_FOLDER_ID=<your folder id>
DRIVE_CLIENT_SECRETS=/abs/path/credentials.json
DRIVE_TOKEN_PATH=/abs/path/token.json
DRIVE_DB=1                                   # manifest lives in Drive (distributed)

NARRATION_ENGINE=fish                        # fish (default) | elevenlabs
BGM_TWO_PASS=1                               # set 0 on a 24 GB GPU
BGM_LOUDNESS_RATIO=0.42
HF_HOME=/abs/path/ContentFactory/models/hf_cache
DB_PATH=manifest.sqlite

FISH_COMPILE=1                               # measured ~2.1x faster TTS in bulk mode
FLUX_COMPILE=1
FLUX_COMPILE_MODE=default                    # never "reduce-overhead" on 24 GB
```

`shared/config.py` loads this before any HuggingFace import, so `HF_HOME` reliably points every stage at
the same repo-local weight cache.

---

## 4. Per-node setup (venv + deps + models)

Each setup script **creates a dedicated venv**, installs that stage's packages, and **downloads its
models** into `models/hf_cache`. Per-stage venvs are intentional — Fish, FLUX.2 and ACE-Step have
conflicting dependency pins and must not share one environment.

| Node | Command | Creates | Downloads |
|------|---------|---------|-----------|
| Story (1) | `python scripts/setup_story.py` | `../.venv_story` | — (CPU, LLM API) |
| Narration (2) | `python scripts/setup_narration.py` | `../.venv_narration` | Fish S2 Pro + WhisperX (also patches the Fish repo) |
| Images (3) | `python scripts/setup_images.py` | `../.venv_images` | FLUX.2-klein (~23 GB) |
| Music (4) | `python scripts/setup_bgm.py` | `../.venv_bgm` | Qwen2.5-Omni (~21 GB), validates ACE-Step |
| Compose (5) | `python scripts/setup_compose.py` | `../.venv_compose` | — (needs system ffmpeg) |

Notes:

- Run a setup script with any `python3`; it builds the venv and installs/downloads using the venv's
  python. First-run downloads are tens of GB — let them finish.
- Setup is **idempotent** — re-running re-uses the venv and skips cached weights.
- `torch.compile` and model loading are CPU-RAM heavy. Close other large applications so the worker
  isn't OOM-killed.

---

## 5. Run — bulk mode (recommended)

Seed once, then run one draining worker per role. No batch IDs, no coordinator. Each worker loads its
model **once** and processes every video whose upstream stage is complete — which is what makes
`FISH_COMPILE`/`FLUX_COMPILE` worth enabling.

```bash
# ── CPU / story box: seed N videos across ALL stages and generate the stories
python scripts/produce.py --count 150

# ── GPU box: run each role in its own venv, sequentially (the stacks don't co-fit on one GPU)
source ../.venv_narration/bin/activate && python scripts/worker.py narration
source ../.venv_images/bin/activate    && python scripts/worker.py images
source ../.venv_bgm/bin/activate       && python scripts/worker.py music

# ── CPU box (or the GPU box, for NVENC): compose
source ../.venv_compose/bin/activate   && python scripts/worker.py compose
```

Each `worker.py <role>` loop: `sync_pull` the Drive DB → `claim_ready_job` (atomic; only jobs whose
predecessor is `COMPLETE`) → process → `sync_push` → repeat until nothing is ready, then exit.

| Flag | Effect |
|------|--------|
| `--watch` | Wait for upstream work instead of exiting when idle |
| `--poll N` / `--max-idle N` | Poll interval and how many empty polls before giving up |
| `--max N` | Process at most N jobs then exit (e.g. `--max 1` for a smoke test) |
| `--no-drive` | Local-only; skip the Drive DB |

It is crash-safe and resumable: re-run a role and it claims only what's left.

---

## 6. Run — staged mode

Explicit per-stage control with worker pools and a live progress dashboard.

```bash
# Machine 1 — story (prints the batch id; copy it)
source ../.venv_story/bin/activate
python run.py --stage 1 --count 5 --drive-db

# Machine 2 — media, in order, each in its own venv
source ../.venv_narration/bin/activate && python run.py --stage 2 --batch <id> --drive-db --workers 1
source ../.venv_images/bin/activate    && python run.py --stage 3 --batch <id> --drive-db --workers 1
source ../.venv_bgm/bin/activate       && python run.py --stage 4 --batch <id> --drive-db --workers 1

# Machine 3 — compose
source ../.venv_compose/bin/activate
python run.py --stage 5 --batch <id> --drive-db --workers 1
```

Each command does one `sync_pull` at start (so it sees upstream completions) and one `sync_push` at the
end (publishing its rows for the next machine). Use `--workers 1` for GPU stages.

### Resume on another machine

```bash
python run.py --resume --stages 4 5 --batch <id>
```

`--resume` implies `--drive-db`: it pulls the shared DB, resets stale jobs, prints a pending summary,
then runs only the still-`PENDING` work for the given stages, pulling artifacts from Drive. It requires
`--batch` and a Drive client. Stages always execute in dependency order regardless of the order you
list them (`--stages 5 4` → 4 then 5).

### Video catalog

When stage 5 finishes a video it records `{title, link}` in the manifest's `videos` table.

```bash
python run.py --list-videos --batch <id>                  # print + export videos.csv/json to Drive
python run.py --list-videos                                # all batches
python run.py --list-videos --batch <id> --share-public    # also set each final.mp4 to anyone-with-link
```

Links are **private by default** (`https://drive.google.com/file/d/<id>/view`) — you own the files
(Desktop OAuth), so you can share them any time.

---

## 7. Run — single machine / local

If one machine has every dependency in one environment, you can run multiple stages in one process
and/or skip Drive entirely:

```bash
# all stages in one env, Drive-synced
python run.py --stage 1 --count 3 --drive-db
python run.py --stages 2 3 4 5 --batch <id> --drive-db

# fully local (no Drive); manifest stays in ./manifest.sqlite
python run.py --stage 1 --count 1
python run.py --stages 2 3 4 5 --batch <id>
```

> `--stages 2 3 4` in **one** process requires one environment with all deps. The per-stage-venv layout
> is the distributed path; this is for local/all-in-one boxes.

---

## 8. Toggles & options

| Flag / env | Effect |
|------------|--------|
| `--stage N` / `--stages N N…` | Run one stage / several in dependency order |
| `--count N` | Stage 1 only: number of stories (the batch size) |
| `--batch <id>` | Required for stages 2–5; the id printed by stage 1 |
| `--workers N` | Parallel workers (use `1` for GPU stages) |
| `--drive-db` | Use the Drive-hosted shared DB (else local SQLite) |
| `--resume` | Resume a batch from the Drive DB (implies `--drive-db`; needs `--batch`) |
| `--retry-failed` | Reset `FAILED` jobs for the run's stage(s) back to `PENDING` |
| `--list-videos` / `--share-public` | Print/export the catalog; optionally make finals public |
| `--db-path P` | Local manifest path (default `manifest.sqlite`) |
| `NARRATION_ENGINE` | `fish` (default) / `elevenlabs` |
| `BGM_TWO_PASS` | `1` two-pass techC (default) / `0` single-pass — **use 0 on 24 GB** |
| `BGM_LOUDNESS_RATIO` | BGM level vs voice (default `0.42`; lower = quieter music) |
| `FISH_COMPILE` / `FLUX_COMPILE` | `torch.compile` opt-ins — see [PERFORMANCE.md](PERFORMANCE.md) |
| `FLUX_COMPILE_MODE` | `default` (safe) / `reduce-overhead` (>24 GB cards only) |
| `PROFILE_LOG=path.jsonl` | Record every step's timing for `scripts/profile_report.py` |

---

## 9. Resume, retry & verify

- **Resume** — just re-run the same command. Interrupted `RUNNING` jobs auto-reset after a timeout;
  stage 3 also skips images already uploaded to Drive.
- **Retry failures** — add `--retry-failed`.
- **Verify a run**:
  - Drive: `Batch_<id>/video_<n>/{metadata,narration,images,music,final}` populated, and
    `manifest.sqlite` present in the parent folder.
  - Local:
    ```bash
    python -c "import sys; sys.path.insert(0,'.'); from shared.manifest import Manifest; \
    print(Manifest('manifest.sqlite').get_stats('<batch_id>'))"
    ```

---

## 10. Reset

`scripts/reset_pipeline.py` clears the Drive folder (all `Batch_*`, the shared DB, the lock) and wipes
local state. **Destructive** — it confirms unless `--yes`.

```bash
python scripts/reset_pipeline.py                       # show plan, ask y/N, trash + wipe local
python scripts/reset_pipeline.py --yes                 # unattended (trash → recoverable from Drive Trash)
python scripts/reset_pipeline.py --yes --permanent     # hard delete (unrecoverable)
python scripts/reset_pipeline.py --yes --keep-local    # only touch Drive
```

---

## 11. Publishing

Seeding the review Sheet, deploying the Gradio review app, and running the throttled Instagram/YouTube
uploader are covered in **[PUBLISHING.md](PUBLISHING.md)**.

```bash
python scripts/seed_sheet.py --batch <id>          # seed the review queue
python social/uploader.py --once --dry-run         # safe rehearsal, posts nothing
python social/uploader.py --loop --interval-hours 4
```

---

## 12. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No Drive token … run scripts/drive_auth.py` | Missing/expired `token.json`. Authorize once (§2) and copy it over. |
| Drive writes go to the wrong folder | Set a real `DRIVE_PARENT_FOLDER_ID` in `.env` — the default is a fallback. |
| Stage 2 import/load error about the tokenizer | Re-run `setup_narration.py`; it re-applies the Fish `tokenizer_config.json` + `llama.py` patches. |
| Stage 4 "ACE repo/checkpoints missing" | Provision ACE-Step and set `ACE_REPO_DIR` / `ACE_CHECKPOINTS`. |
| CUDA OOM in stage 3 | Use `--workers 1`; ensure `FLUX_COMPILE_MODE=default`; reduce batch size in `generate.py`. |
| CUDA OOM in stage 4 | Set `BGM_TWO_PASS=0` — two-pass keeps Qwen and ACE-Step resident together. |
| Worker killed with no traceback (`rc=139`) | Almost always a network/TLS race. Uploads and downloads are serialized for this reason; re-run and it resumes. |
| Many `[SSL] record layer failure` / size-mismatch retries | The uplink is dropping TLS. Workers retry 5×, then a re-run resumes via partial-resume logic. Use a stable connection. |
| GPU workers pull the DB and find 0 jobs | The seeder must push the **full** DAG. Use `scripts/produce.py` (it pushes all stages), not a stage-1-only push. |
| Models re-download every run | Ensure `HF_HOME` is the same repo-local path in `.env` *and* was used at setup time. |
| 429s from OpenRouter | Handled automatically with backoff; check the key has credit. A `403` means the key hit its total spend limit. |
| A leftover `Batch_unknown/` folder in Drive | Residue from an old bug — safe to trash. |

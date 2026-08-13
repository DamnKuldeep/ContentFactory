# Running it

Setting up a machine, authorising Drive once, and every way to run the thing.

> For why it's built this way, see [ARCHITECTURE.md](ARCHITECTURE.md). For what it produced, [RESULTS.md](RESULTS.md).

Run everything from the repo root. The per-stage venvs get created one level *up*, so you'll activate them as `../.venv_<stage>/bin/activate`.

1. [What you need](#what-you-need)
2. [Authorising Drive, once](#authorising-drive-once)
3. [The .env file](#the-env-file)
4. [Setting up a machine](#setting-up-a-machine)
5. [Normal use: bulk workers](#normal-use-bulk-workers)
6. [When you want control: staged mode](#when-you-want-control-staged-mode)
7. [One machine, no Drive](#one-machine-no-drive)
8. [Every flag](#every-flag)
9. [Resuming and retrying](#resuming-and-retrying)
10. [Starting over](#starting-over)
11. [Posting](#posting)
12. [When things go wrong](#when-things-go-wrong)

---

## What you need

- **Python 3.10+** everywhere.
- **ffmpeg** on whichever machine does stage 5 (`sudo apt-get install -y ffmpeg`).
- **An NVIDIA GPU** for stages 2, 3 and 4. 24 GB is comfortable; less and you'll be fighting VRAM. Stages 1 and 5 are happy on CPU.
- **An OpenRouter key** with credit. Only stage 1 uses it.
- **A Google OAuth desktop client** (`credentials.json` from the Cloud console) and a Drive folder to write into.
- **ACE-Step 1.5** cloned with its checkpoints somewhere, for stage 4.

Clone the repo on each machine. You don't install everything everywhere — each box only runs the setup for the stages it's doing.

---

## Authorising Drive, once

This is the only interactive step, and you do it once on a machine with a browser:

```bash
python scripts/drive_auth.py
# browser opens → approve → writes token.json next to credentials.json
```

Then copy two files to every other machine:

```
credentials.json
token.json
```

Those are gitignored, so don't worry about them ending up in a commit. After this, no machine will ever ask you to log in again — the worker code only refreshes a token that's already there, and physically can't open a browser.

The scopes include Sheets as well as Drive, because the review queue needs it. If your token is older than that change, just run `drive_auth.py` again.

---

## The .env file

Copy `.env.example` to `.env` and fill it in. The minimum for a multi-machine run:

```dotenv
OPENROUTER_API_KEY=sk-or-...

DRIVE_PARENT_FOLDER_ID=<your folder id>
DRIVE_CLIENT_SECRETS=/abs/path/credentials.json
DRIVE_TOKEN_PATH=/abs/path/token.json
DRIVE_DB=1                                   # keep the job database in Drive

NARRATION_ENGINE=fish                        # or elevenlabs
BGM_TWO_PASS=1                               # set 0 on a 24 GB card
BGM_LOUDNESS_RATIO=0.42
HF_HOME=/abs/path/ContentFactory/models/hf_cache
DB_PATH=manifest.sqlite

FISH_COMPILE=1                               # roughly halves TTS time in bulk mode
FLUX_COMPILE=1
FLUX_COMPILE_MODE=default                    # never reduce-overhead on 24 GB
```

`shared/config.py` reads this before anything imports HuggingFace, which is how `HF_HOME` reliably points every stage at the same weights cache instead of downloading 40 GB twice.

---

## Setting up a machine

Each script builds a venv, installs that stage's packages, and downloads its models into `models/hf_cache`.

| Machine does | Run | Makes | Downloads |
|---|---|---|---|
| Stories | `python scripts/setup_story.py` | `../.venv_story` | nothing — it's all API |
| Narration | `python scripts/setup_narration.py` | `../.venv_narration` | Fish S2 Pro + WhisperX, and patches the Fish repo |
| Images | `python scripts/setup_images.py` | `../.venv_images` | FLUX.2-klein, ~23 GB |
| Music | `python scripts/setup_bgm.py` | `../.venv_bgm` | Qwen2.5-Omni, ~21 GB. Checks ACE-Step is where you said. |
| Compose | `python scripts/setup_compose.py` | `../.venv_compose` | nothing, but wants system ffmpeg |

Separate venvs isn't fussiness. Fish, FLUX.2 and ACE-Step pin conflicting versions of the same packages and simply cannot share an environment.

Run these with any `python3`; each one builds its venv and then installs using that venv's python. First run downloads tens of gigabytes, so give it time. Re-running is safe — it reuses the venv and skips weights it already has.

One thing to watch: `torch.compile` and model loading are hungry for *system* RAM, not just VRAM. Close your browser before a long run or the worker may get OOM-killed by the kernel.

---

## Normal use: bulk workers

Seed once, then run a worker per role. Each one loads its model a single time and works through everything ready.

```bash
# story box — seed 150 videos across all stages, then write the stories
python scripts/produce.py --count 150

# GPU box — one role at a time, each in its own venv
source ../.venv_narration/bin/activate && python scripts/worker.py narration
source ../.venv_images/bin/activate    && python scripts/worker.py images
source ../.venv_bgm/bin/activate       && python scripts/worker.py music

# compose box
source ../.venv_compose/bin/activate   && python scripts/worker.py compose
```

Each worker: pull the database from Drive, claim a job whose previous stage is done, do it, push back, repeat. When nothing's left it exits.

| Flag | Does |
|---|---|
| `--watch` | Wait around for upstream work instead of quitting when idle |
| `--poll N` / `--max-idle N` | How long between checks, and how many empty checks before giving up |
| `--max N` | Stop after N jobs. `--max 1` is a good smoke test. |
| `--no-drive` | Local only |

Kill it whenever. Re-run it and it picks up only what's left.

---

## When you want control: staged mode

Explicit stages, worker pools, a live progress view.

```bash
# machine 1 — stories. Prints a batch id; copy it.
source ../.venv_story/bin/activate
python run.py --stage 1 --count 5 --drive-db

# machine 2 — narration, images, music, in that order
source ../.venv_narration/bin/activate && python run.py --stage 2 --batch <id> --drive-db --workers 1
source ../.venv_images/bin/activate    && python run.py --stage 3 --batch <id> --drive-db --workers 1
source ../.venv_bgm/bin/activate       && python run.py --stage 4 --batch <id> --drive-db --workers 1

# machine 3 — compose
source ../.venv_compose/bin/activate
python run.py --stage 5 --batch <id> --drive-db --workers 1
```

Each command pulls the database once at the start (so it sees what upstream finished) and pushes once at the end (so the next machine can see its work). Use `--workers 1` on GPU stages.

### Picking up on a different machine

```bash
python run.py --resume --stages 4 5 --batch <id>
```

`--resume` implies `--drive-db`. It pulls the database, releases anything stuck, tells you what's still pending, and runs only that. Needs a batch id and Drive credentials. Stages always run in dependency order regardless of what order you type them — `--stages 5 4` still does 4 first.

### What came out

Stage 5 writes each finished video's title and link into the database.

```bash
python run.py --list-videos --batch <id>                  # print, and export videos.csv/json to Drive
python run.py --list-videos                                # everything, all batches
python run.py --list-videos --batch <id> --share-public    # make the files link-shareable too
```

Links are private by default. You own the files, so you can share them whenever.

---

## One machine, no Drive

If you've got everything installed in one environment, you can run stages together, and skip Drive entirely:

```bash
# everything in one env, still syncing to Drive
python run.py --stage 1 --count 3 --drive-db
python run.py --stages 2 3 4 5 --batch <id> --drive-db

# fully local — database stays in ./manifest.sqlite
python run.py --stage 1 --count 1
python run.py --stages 2 3 4 5 --batch <id>
```

`--stages 2 3 4` in one process needs one environment with all the deps, which mostly means you've already solved the version conflicts yourself.

---

## Every flag

| Flag / setting | Does |
|---|---|
| `--stage N` / `--stages N N…` | One stage, or several in order |
| `--count N` | Stage 1 only — how many stories |
| `--batch <id>` | Needed for stages 2–5 |
| `--workers N` | Parallel workers. Use 1 on GPU stages. |
| `--drive-db` | Keep the job database in Drive |
| `--resume` | Continue a batch here (implies `--drive-db`) |
| `--retry-failed` | Put failed jobs back in the queue |
| `--list-videos` / `--share-public` | The finished-video catalog |
| `--db-path P` | Where the local database lives |
| `NARRATION_ENGINE` | `fish` or `elevenlabs` |
| `BGM_TWO_PASS` | `1` for the two-pass brief, `0` for single. Use 0 on 24 GB. |
| `BGM_LOUDNESS_RATIO` | How loud the music sits against the voice. Lower is quieter. |
| `FISH_COMPILE` / `FLUX_COMPILE` | `torch.compile` — see [PERFORMANCE.md](PERFORMANCE.md) |
| `FLUX_COMPILE_MODE` | `default`, or `reduce-overhead` only if you have more than 24 GB |
| `PROFILE_LOG=path.jsonl` | Record timings for `scripts/profile_report.py` |

---

## Resuming and retrying

Re-run the same command. Jobs whose worker died get released after a timeout, and stage 3 skips images already sitting in Drive. For jobs that gave up entirely, add `--retry-failed`.

To check on a batch:

```bash
python -c "import sys; sys.path.insert(0,'.'); from shared.manifest import Manifest; \
print(Manifest('manifest.sqlite').get_stats('<batch_id>'))"
```

Or just look in Drive — `Batch_<id>/video_<n>/` should have `metadata`, `narration`, `images`, `music` and `final` filling up, with `manifest.sqlite` sitting in the parent folder.

---

## Starting over

`scripts/reset_pipeline.py` throws away the Drive folder's batches, the shared database and the lock, and wipes local state. It asks first unless you tell it not to.

```bash
python scripts/reset_pipeline.py                       # show the plan, ask y/N
python scripts/reset_pipeline.py --yes                 # unattended, goes to Drive Trash
python scripts/reset_pipeline.py --yes --permanent     # actually gone
python scripts/reset_pipeline.py --yes --keep-local    # only clean Drive
```

---

## Posting

Seeding the review sheet, hosting the review app, and running the uploader are all in [PUBLISHING.md](PUBLISHING.md).

```bash
python scripts/seed_sheet.py --batch <id>
python social/uploader.py --once --dry-run      # rehearsal, posts nothing
python social/uploader.py --loop --interval-hours 4
```

---

## When things go wrong

| What you see | What it is |
|---|---|
| `No Drive token … run scripts/drive_auth.py` | Missing or dead `token.json`. Authorise once and copy it over. |
| Files landing in the wrong Drive folder | `DRIVE_PARENT_FOLDER_ID` isn't set in `.env`. |
| Stage 2 blows up on the tokenizer | Re-run `setup_narration.py` — it re-applies the Fish patches. |
| Stage 4 says the ACE repo is missing | Set `ACE_REPO_DIR` and `ACE_CHECKPOINTS` to wherever you put them. |
| CUDA OOM in stage 3 | `--workers 1`, and make sure `FLUX_COMPILE_MODE=default`. |
| CUDA OOM in stage 4 | `BGM_TWO_PASS=0`. Two-pass keeps Qwen and ACE-Step loaded together. |
| Worker dies with `rc=139` and no traceback | Network race. Uploads and downloads are already serialised for this; just re-run, it resumes. |
| Endless `[SSL] record layer failure` retries | Your connection is dropping TLS mid-transfer. Workers retry five times, then a re-run picks up where it left off. Use a stable link for long batches. |
| GPU workers pull the database and find nothing | Seed with `scripts/produce.py`, which pushes the whole job list. Pushing only stage 1 wipes the rest. |
| Models re-downloading every run | `HF_HOME` needs to be the same path in `.env` *and* when you ran setup. |
| 429s from OpenRouter | Normal, handled with backoff. A `403` means the key hit its spending limit. |
| A `Batch_unknown/` folder in Drive | Leftover from an old bug. Delete it. |

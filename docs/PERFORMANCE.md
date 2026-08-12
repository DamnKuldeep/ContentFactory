# Performance — latency & GPU-cost

Measured from one full video end-to-end (batch `20260625_141924`, "The Forged Caravan", 34 scenes,
88.5 s narration) on an **RTX 5090 Laptop (24 GB)**.

> Back to the [README](../README.md) · design in [ARCHITECTURE.md](ARCHITECTURE.md)

Instrumentation is `shared/timing.py`: a `step()` context manager wrapped around every meaningful
operation, which appends one JSONL record per step (with optional CUDA sync so GPU time is attributed
correctly) whenever `PROFILE_LOG` is set. Nothing here is estimated.

```bash
PROFILE_LOG=run.jsonl python scripts/worker.py <role>
python scripts/profile_report.py run.jsonl -o latency.md
python scripts/profile_compare.py off.jsonl on.jsonl        # A/B two arms
```

Raw per-step data for the baseline below: [latency_report.md](latency_report.md), generated from
[profile_run.jsonl](profile_run.jsonl).

## Measured baseline (one video, N=1)

| stage | wall | GPU-s | dominant steps |
|------|-----:|------:|----------------|
| 1 story (API, no GPU) | ~819 s | 0 | LLM convergence (OpenRouter) |
| 2 narration (Fish) | 187 s | ~170 | `tts_generate` **128 s**, `fish_load` **40 s** |
| 3 images (FLUX.2) | 267 s | ~121 | 9× `flux_generate` **~115 s**; 34 serial uploads **~125 s I/O (GPU idle)** |
| 4 music (Qwen+ACE) | 83 s | ~81 | `brief`(Qwen) **36 s**, `ace_load` **22 s**, `qwen_load` **12 s**, `ace_gen` **10 s** |
| 5 compose (CPU/NVENC) | 119 s | ~0 | `assets_download` **59 s**, `frames_prep` 20 s, ffmpeg(nvenc) 23 s |
| **total** | **24.6 min** | **~375 GPU-s (6.3 GPU-min)** | load 84 + infer 292 |

**Cost number: ~375 GPU-seconds/video.** Model loads alone are 84 s (22%) at N=1.

## Constraint honored
No model swaps, no prompt edits, no chaining changes, no inference-step/quality changes. Every lever
below is execution / transport / memory / caching only.

## Levers (status)

| # | lever | mechanism | measured baseline | est. saving | status |
|---|-------|-----------|-------------------|-------------|--------|
| A | **Warm-model batching** | `worker <role>` loads the model once and drains all N videos | loads 84 s/video @N=1 | ~80 GPU-s/video amortized over N | **done** (Part 3) |
| B | **Overlap FLUX uploads w/ generation** | background ThreadPool uploads while GPU makes the next batch | 34 serial uploads ~125 s, GPU idle | ~100 s stage-3 wall, GPU stays busy | **done** (`stage_03/generate.py`) |
| C | **Parallel Drive downloads** | stage-5 downloads images/narration/music in a ThreadPool | `assets_download` 59 s serial | ~45 s stage-5 wall | **done** (`stage_05/run.py`) |
| D | **claim_job atomic guard** | `UPDATE … WHERE status='PENDING'` + rowcount | 2 workers redid 1 story (2× GPU) | prevents duplicate GPU work | **done** (`manifest.py`) |
| E | **Stage-4 memory mgmt** | free Qwen before ACE loads (no chaining change) | OOM on 24 GB | unblocks stage 4 | **done** |
| F | **torch.compile (Fish + FLUX)** | opt-in `FISH_COMPILE=1` / `FLUX_COMPILE=1` (FLUX `mode=default`) | tts 128 s; flux 13.5 s/batch | **MEASURED (3-video A/B): Fish TTS ≈2.1× faster (138→60 s/video); FLUX +18% (13.5→11.0 s warm). ≈90 GPU-s/video saved (~24%).** | **opt-in; recommend ON for bulk worker** |
| G | **Prompt caching** | `LLM_PROMPT_CACHE=1` marks the verbatim system prompt cacheable | stage-1 LLM $ | **MEASURED: no-op — control arm already 17% cached; OpenRouter providers auto-cache regardless of the flag.** | **leave OFF (redundant)** |
| — | two-pass BGM | `BGM_TWO_PASS=0` skips one ACE generation | ace_gen ~10 s ×2 | ~10 GPU-s | your config toggle |

**Projected:** A alone → per-video GPU **375 → ~295 s** (loads amortized across the batch). B+C →
wall **24.6 → ~15 min** (excl. the API stage 1) by removing serial Drive I/O. F adds ~10–25% off the
two big infer steps when enabled. G cuts the stage-1 API bill where the provider supports caching.

## How to enable the opt-in levers (measured recommendations — see [bench/AB_FINDINGS.md](../bench/AB_FINDINGS.md))
```bash
# .env  (recommended for the bulk worker path, where compile amortizes over the batch)
FISH_COMPILE=1          # ≈2.1× faster Fish TTS (the biggest GPU step) — MEASURED, strongly worth it
FLUX_COMPILE=1          # ~18% faster FLUX — MEASURED
FLUX_COMPILE_MODE=default   # REQUIRED on 24 GB: reduce-overhead's CUDA graphs OOM FLUX.2 after ~20 imgs
# LLM_PROMPT_CACHE=1    # leave OFF — MEASURED no-op (OpenRouter providers auto-cache regardless)
BGM_TWO_PASS=0          # single-pass BGM (skip the refine generation)
```
torch.compile pays a one-time cold build on the first call of each worker process (~50–110 s Fish,
~15–25 s FLUX), so it's a **batch** optimization — net win only when one worker drains many videos,
net loss at N=1. A full 3-video A/B was run with the `worker` path; re-profile with
`PROFILE_LOG=… scripts/worker.py <role>` then `scripts/profile_compare.py off.jsonl on.jsonl`.

> Caveat: the A/B's image→music→compose stages were blocked by a flaky-uplink Drive bug (concurrent
> resumable uploads SIGSEGV'd the FLUX worker, rc=139) — fixed by serializing the upload pool to
> `max_workers=1` in `stage_03_images/generate.py`. The compile/cache latency numbers above come from
> the GPU `infer` steps, which are measured before upload and unaffected.

## Bulk operating model (the simple path for 100–150 videos)
The batch/populate ceremony is gone — seed once, then run a looping worker per machine that drains
all ready videos of its type (this *is* lever A):
```bash
# CPU box — seed N videos + generate stories
python scripts/produce.py --count 150

# GPU box — one venv per role (Fish/FLUX/ACE deps conflict); each drains all videos
source ../.venv_narration/bin/activate && python scripts/worker.py narration
source ../.venv_images/bin/activate    && python scripts/worker.py images
source ../.venv_bgm/bin/activate       && python scripts/worker.py music

# CPU box — compose (libx264) drains all
source ../.venv_compose/bin/activate && python scripts/worker.py compose
```
No batch IDs, no populate_backlog: `worker` uses `manifest.claim_ready_job(stage)` (claims a PENDING
job whose predecessor is COMPLETE, atomically) and Drive checkpoint-syncs the DB. Idempotent and
resumable — re-running a worker claims only what's left.

## Bugs found & fixed during the profiling run (the chain had never run end-to-end)
1. `fish.py` imported `fish_speech` before the repo was on `sys.path` → fixed (load model first).
2. Stage 2 wrote `meta.audio_drive_id` but 4/5 read `meta.stage_02.audio_drive_id`, and the 3→4→5
   chain branches off stage_01 so it dropped stage_02's alignment/audio → fixed (key + a stage_02
   merge in stages 4 & 5).
3. Stage 5 contract required `stage_04.json` in work_dir (never placed there) → fixed.
4. Stage 4 OOM (Qwen+ACE on 24 GB) + two-pass thrash → fixed by freeing Qwen before ACE.
5. **Sync bug**: stage_01 scenes carry no `char_start/char_end`, so compose mapped every scene to
   t≈0 → images flashed by in the first seconds → fixed: `compose.py` derives char offsets from each
   scene's narration snippet; scenes now span the full narration.
6. `claim_job` had no atomic guard → 2 workers processed the same story → fixed (lever D).

## Bugs found & fixed during the caching+compile A/B (the bulk `worker` path had never run E2E)
7. **`produce.py` pushed only stage-1 to Drive** (`sync_push({"stage_01_story"})`) → downstream
   `worker <role>` boxes pulled a DB with no seeded stage-2..5 rows and sat idle (0 jobs). Fixed:
   push the full seeded DAG (`sync_push(set(STAGE_SEQUENCE))`). **Real distributed-pipeline bug.**
8. **`stage_03` uploaded images to `Batch_unknown/video_0000`** — `generate.run_all` reads
   batch_id/story_num from `job_data`, but the stage-1 JSON carries neither. Fixed: inject them from
   the job in `stage_03/run.py`. (Downstream finds images by `drive_file_id`, so it was cosmetic, but
   it broke the per-video folder layout.) **Real bug.**
9. **FLUX `torch.compile(mode="reduce-overhead")` OOMs on 24 GB** — CUDA graphs reserve ~2.7 GB →
   VRAM OOM after ~20 images. Fixed: `FLUX_COMPILE_MODE` (default `default`, no CUDA graphs). **Real bug.**
10. **Concurrent Drive uploads SIGSEGV the FLUX worker (rc=139)** under a flaky uplink (OpenSSL races
    across upload threads). Fixed: upload pool `max_workers` 4→1 (serial, still overlaps with the next
    generation batch). **Real bug.**
11. Bench-only: `HF_HOME` pointed at an empty repo-local cache → re-downloading 44 GB of FLUX/Qwen;
    and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` was tried as an OOM mitigation but was unrelated
    to (and not the cause of) the upload segfault — removed.

## Notes / not changed
FLUX stays 4 steps (klein minimum), ACE turbo config unchanged, Qwen-Omni brief kept, NVENC kept,
model routing/chaining and all prompts untouched. BGM is now 0.42× the voice loudness (was 0.40).

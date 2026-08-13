# Where the time goes

What one video costs, and what actually made it cheaper.

> [README](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · raw A/B numbers in [AB_FINDINGS.md](../bench/AB_FINDINGS.md)

I wanted a real number for "what does a video cost", not a guess. So `shared/timing.py` wraps a timer around every meaningful operation and writes one JSON line per step whenever `PROFILE_LOG` is set. It's a no-op otherwise, so it can stay in the production code. GPU steps get a CUDA sync so async work lands in the right bucket instead of being credited to whatever ran next.

```bash
PROFILE_LOG=run.jsonl python scripts/worker.py <role>
python scripts/profile_report.py run.jsonl -o latency.md
python scripts/profile_compare.py off.jsonl on.jsonl     # diff two arms
```

Everything below is measured. Per-step detail is in [latency_report.md](latency_report.md), from [profile_run.jsonl](profile_run.jsonl).

---

## The baseline

One video — batch `20260625_141924`, "The Forged Caravan", 34 scenes, 88.5 seconds of narration — on an RTX 5090 laptop with 24 GB.

| Stage | Wall | GPU | Where it went |
|---|---:|---:|---|
| 1 story | 819 s | — | LLM back-and-forth, no GPU involved |
| 2 narration | 187 s | 170 s | TTS 128 s, loading the model 40 s |
| 3 images | 267 s | 121 s | 9 FLUX batches ~115 s, then 34 uploads one at a time ~125 s with the GPU sitting idle |
| 4 music | 83 s | 81 s | Qwen writing the brief 36 s, ACE loading 22 s, Qwen loading 12 s, generating 10 s |
| 5 compose | 119 s | ~0 | downloading assets 59 s, prepping frames 20 s, ffmpeg 23 s |
| **Total** | **24.6 min** | **~375 s** | 84 s of it is just loading models |

**375 GPU-seconds per video.** That's the number worth caring about, because it's the one you'd pay a cloud provider for. At one video, 22% of it is model loading — which is pure waste if you're doing a hundred.

---

## Rules I set myself

No swapping models, no touching prompts, no changing how stages chain, no dropping inference steps. Anything that would make the output worse was off the table. Everything below is execution, transport, memory or caching only.

---

## What I changed

| | What | Why | Where |
|---|---|---|---|
| **A** | Load the model once, then do every video | Loading was 84 s per video at N=1. Spread across a batch it's basically free. | `scripts/worker.py` |
| **B** | Upload images on a background thread | The GPU was idle for ~125 s per video waiting on 34 sequential uploads. | `stage_03_images/generate.py` |
| **C** | Fetch stage-5 assets in parallel | 59 s of serial downloads before ffmpeg could start. | `stage_05_compose/run.py` |
| **D** | Make job claiming atomic | Two workers rendered the same story once. That's double GPU time for one video. | `shared/manifest.py` |
| **E** | Evict Qwen before ACE-Step loads | They don't fit together on 24 GB. Stage 4 just OOM'd. | `stage_04_music/` |

A is the big structural one — it's why the bulk worker path exists at all. B and C together take about 145 s of dead network time out of the wall clock.

---

## What I A/B'd

Two arms, three videos each, same machine, same environment, run through the bulk worker path so compile costs amortise the way they would in production.

| Setting | OFF | ON | Verdict |
|---|---|---|---|
| `FISH_COMPILE` | 138 s of TTS per video | 60 s | **2.1× faster.** Biggest win by far. |
| `FLUX_COMPILE` (`default` mode) | 13.5 s per batch | 11.0 s | 18% faster. Take it. |
| `FLUX_COMPILE` (`reduce-overhead`) | — | OOM after ~20 images | CUDA graphs reserve 2.7 GB. Don't use it under 24 GB. |
| `LLM_PROMPT_CACHE` | 17% of input tokens already cached | 15% | Does nothing. OpenRouter's providers cache repeated prefixes themselves whether you ask or not. |

**About 90 GPU-seconds saved per video, roughly 24%**, and nearly all of it is the Fish compile.

The prompt-cache result is the one I'd flag: it looked like an obvious win on paper, and measuring it showed the control arm was *already* getting the benefit. Left it off, since it adds a code path for nothing.

### Why compile is a batch thing

`torch.compile` builds kernels lazily on the first call in a process — about 50–110 s for Fish, 15–25 s for FLUX. So it's a clear loss on one video and a clear win on a hundred. Another reason the worker loads its model once and drains everything instead of spawning per job.

### What to put in .env

```dotenv
FISH_COMPILE=1              # halves the biggest GPU step
FLUX_COMPILE=1              # 18% off image generation
FLUX_COMPILE_MODE=default   # never reduce-overhead on 24 GB
BGM_TWO_PASS=0              # single-pass music, and required on 24 GB anyway
# LLM_PROMPT_CACHE stays off — measured no-op
```

> One caveat on the A/B: the image→music→compose stages were being blocked at the time by the concurrent-upload segfault (since fixed by serialising the pool). The compile numbers above come from the GPU inference steps, which happen before any upload and weren't affected.

---

## Bugs the profiling turned up

Profiling meant running the chain end to end for the first time, which found a lot.

**From the first full run:**

1. `fish.py` imported `fish_speech` before the repo was on `sys.path`. Load the model first.
2. Stage 2 wrote `meta.audio_drive_id` but stages 4 and 5 read `meta.stage_02.audio_drive_id` — and since 3→4→5 branches off stage 1, the narration data was getting dropped entirely. Fixed the key and made 4 and 5 merge stage 2's output back in.
3. Stage 5 insisted on a `stage_04.json` in its work dir that nothing ever put there.
4. Stage 4 OOM'd with Qwen and ACE-Step both resident.
5. **The sync bug.** Stage-1 scenes have no character offsets, so compose mapped every scene to t≈0 and all 34 images flashed past in the first few seconds. `compose.py` now finds each scene's narration inside the script and works the offsets out.
6. `claim_job` had no atomic guard, so two workers could take the same job.

**From the A/B, which was the first time the bulk path ran end to end:**

7. `produce.py` pushed only stage-1 rows to Drive, which meant the seeded stages 2–5 got wiped from the remote on the next pull. GPU boxes would pull the database and find nothing to do. Real distributed bug, and the kind you only see with more than one machine.
8. Stage 3 uploaded to `Batch_unknown/video_0000/` because it reads the batch and story number from the JSON, and stage 1's JSON doesn't have them. Cosmetic — downstream finds images by file ID — but it made a mess of the folders.
9. `reduce-overhead` compile mode OOM'd FLUX after ~20 images.
10. Concurrent Drive uploads segfaulted the FLUX worker with `rc=139` under a flaky connection. OpenSSL races across upload threads.
11. Bench-only: `HF_HOME` pointed at an empty cache and re-downloaded 44 GB. Also tried `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` as an OOM fix before realising it had nothing to do with the segfault.

---

## What I didn't touch

FLUX stays at 4 steps (klein's minimum), ACE-Step's turbo config is unchanged, the Qwen brief stays, NVENC stays, and no prompt or model routing was altered. The only quality-affecting change in the whole exercise: BGM sits at 0.42× the voice loudness instead of 0.40, because 0.40 was slightly too quiet.

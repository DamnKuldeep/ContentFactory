# A/B re-profile — caching + torch.compile (measured)

> Back to the [README](../README.md) · method and baseline in [docs/PERFORMANCE.md](../docs/PERFORMANCE.md)

Two matched arms, **3 videos each**, RTX 5090 Laptop (24 GB), same `.conda` env, run via the
`worker` path (model loads/compiles **once per process**, drains all 3 → compile cost amortizes).
Raw: `profile_off.jsonl` / `profile_on.jsonl`. OFF = all flags off; ON = `FISH_COMPILE=1
FLUX_COMPILE=1 LLM_PROMPT_CACHE=1` (FLUX `mode=default`).

## Headline
| lever | metric | OFF | ON | result |
|------|--------|----:|---:|--------|
| **FISH_COMPILE** | `tts_generate` per video (Fish TTS) | 138.5 / 113.7 / 138.3 s (~13.5 s/chunk) | 69.2 / 63.8 / 49.7 s (~6 s/chunk) | **≈2.1× faster TTS** (~53% less) ✅ |
| **FISH_COMPILE** | `fish_load` | 38.3 s | 39.1 s | unchanged (compile is lazy) |
| **FLUX_COMPILE=default** | `flux_generate` full batch (n=4) | **13.5 s** (median, 22) | **11.0 s** warm (median, 15) | **~18% faster** ✅ |
| **FLUX_COMPILE=default** | `flux_load` | ~6.0 s | ~6.0 s | unchanged (lazy) |
| **FLUX_COMPILE=reduce-overhead** | VRAM | — | **OOM @ ~20 imgs** | ❌ CUDA graphs +2.7 GB → unusable on 24 GB |
| **LLM_PROMPT_CACHE** | stage-1 cached input tokens | **111,864 / 657,583 (17%)** | 75,670 / 505,411 (15%) | **no-op** — providers auto-cache *regardless* of the flag |
| **LLM_PROMPT_CACHE** | stage-1 cost_usd | $0.0957 | $0.1069 | non-deterministic (different stories), not a caching effect |

## Reading it
- **FISH_COMPILE is the big win** — Fish TTS (the single largest GPU step, ~128 s baseline) roughly
  **halves** to ~55–65 s warm. Even video 1 (which pays the one-time compile build) already beats the
  eager arm, because Fish compiles `decode_one_token` with `mode=default` (no CUDA graphs), amortized
  across the chunks within the very first video.
- **FLUX_COMPILE helps ~18%** but **only with `mode=default`**. The original `reduce-overhead` uses
  CUDA graphs that reserve ~2.7 GB and **OOM FLUX.2 on a 24 GB card** after ~20 images — do not use it
  here (it's now `FLUX_COMPILE_MODE`, default `default`; reduce-overhead is for >24 GB cards).
- **LLM_PROMPT_CACHE is effectively a no-op.** The control arm (flag OFF) still shows 17% of input
  tokens cached — OpenRouter's providers auto-cache repeated prefixes server-side without our
  `cache_control` breakpoint. The flag neither increases caching nor reduces cost measurably; the
  cost delta is just run-to-run story non-determinism. Leave it off (no harm either way).

## The compile cold tax (why this is a *batch* optimization)
torch.compile builds kernels lazily on the **first** call of each worker process (no warmup):
- Fish: first chunk of video 1 pays ~50–110 s of inductor build.
- FLUX: first batch pays ~+15–25 s (the per-relaunch "cold spikes" 14.5–36.8 s in the ON data are
  exactly this — each crash-relaunch recompiled).
So compile **only pays off when one worker drains many videos** (the bulk `worker <role>` model). At
100–150 videos the build is ~free; at N=1 it's a net loss. Loads themselves don't change (lazy).

## Net effect on GPU cost (bulk worker, warm)
Per video the two compiles cut the two dominant infer steps:
- TTS  ~128 s → ~60 s   (save ~68 s)
- FLUX ~120 s → ~99 s   (save ~21 s, ~9 batches × 2.5 s)
≈ **90 GPU-s/video saved** off the ~375 GPU-s baseline → **~24% GPU-cost reduction**, almost all from
FISH_COMPILE. Prompt caching contributes ~0.

## Recommendations
- **FISH_COMPILE=1** for production (the worker drains the batch → compile amortizes). Big win.
- **FLUX_COMPILE=1** with **FLUX_COMPILE_MODE=default** on 24 GB (never `reduce-overhead`). Modest win.
- **LLM_PROMPT_CACHE** — leave **0**; it's redundant with provider auto-caching.

## Out of scope of this test (infra issues, not compile/caching) — separate follow-ups
- **Drive uploads SIGSEGV (rc=139)** under a flaky uplink: concurrent resumable uploads raced in
  OpenSSL and crashed the FLUX worker, blocking images→music→compose for both arms. Mitigated by
  serializing the upload pool to `max_workers=1` (`generate.py`); the underlying network was dropping
  TLS mid-transfer (SSL record-layer failures, HTTP 400 size-mismatch). Needs a clean-network re-run
  to confirm full image→music→compose drain.
- **stage_04 Qwen brief**: "Qwen JSON generation failed after retries" on the one ON video that got
  images — a separate stage-4 issue (no compile/cache involvement).
- Music (Qwen+ACE) and compose (ffmpeg/NVENC) have **no** compile/cache changes, so their latency is
  unchanged from baseline ([docs/latency_report.md](../docs/latency_report.md): music ~83 s, compose ~119 s).

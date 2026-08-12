# Deployment runbooks

Per-stage hardware requirements, expected throughput, and stage-specific troubleshooting. These
complement the general operations guide in [RUNME.md](../RUNME.md) — read that first for setup, auth,
and run commands.

| Stage | Runbook | Needs | Rough cost per video |
|---|---|---|---|
| 1 · story | [stage_01.md](stage_01.md) | CPU + OpenRouter API | ~14 min wall, ~$0.10, no GPU |
| 2 · narration | [stage_02.md](stage_02.md) | NVIDIA GPU (Fish S2 Pro NF4 + WhisperX) | ~170 GPU-s (~60 s with `FISH_COMPILE`) |
| 3 · images | [stage_03.md](stage_03.md) | NVIDIA GPU 24 GB+ (FLUX.2-klein) | ~120 GPU-s for 34 scenes |
| 4 · music | [stage_04.md](stage_04.md) | NVIDIA GPU 24 GB+ (Qwen2.5-Omni + ACE-Step) | ~81 GPU-s |
| 5 · compose | [stage_05.md](stage_05.md) | CPU + ffmpeg (NVENC optional) | ~119 s wall, ~0 GPU |

A typical deployment is three machines: one CPU box for stage 1, one GPU box running stages 2→3→4
sequentially (their model stacks don't co-fit on a single 24 GB card), and a CPU box for stage 5.
Stages 1 and 5 are the ones you'd scale horizontally first.

Measured numbers and the optimization levers behind them: [PERFORMANCE.md](../PERFORMANCE.md).

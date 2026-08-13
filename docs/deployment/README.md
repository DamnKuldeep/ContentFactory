# What each stage wants

Hardware, throughput and the stage-specific gotchas. Read [RUNME.md](../RUNME.md) first for setup and how to actually run things — these are just the per-stage notes.

| Stage | Notes | Needs | Roughly |
|---|---|---|---|
| 1 · story | [stage_01.md](stage_01.md) | CPU and an API key | 13.6 min, ~$0.10, no GPU |
| 2 · narration | [stage_02.md](stage_02.md) | GPU, ~10 GB VRAM | 170 GPU-s, or 60 s with the compile flag |
| 3 · images | [stage_03.md](stage_03.md) | GPU, 24 GB is comfortable | 120 GPU-s for 34 scenes |
| 4 · music | [stage_04.md](stage_04.md) | GPU, 24 GB | 81 GPU-s |
| 5 · video | [stage_05.md](stage_05.md) | CPU and ffmpeg, NVENC helps | 2 min, no GPU needed |

A typical setup is three machines: a CPU box writing stories, a GPU box running stages 2, 3 and 4 one after the other, and a CPU box cutting video. The GPU stages have to be sequential — their model stacks won't share a 24 GB card.

If you want to go faster, stage 1 is the one to parallelise. It's pure API work and the slowest by wall clock, and it doesn't compete for the GPU with anything.

Measured numbers and how they were arrived at: [PERFORMANCE.md](../PERFORMANCE.md).

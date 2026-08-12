# A/B latency comparison

- **OFF** (control): `bench/profile_off.jsonl` — 3 video(s)
- **ON** (treatment): `bench/profile_on.jsonl` — 3 video(s)

Per-video steps report **warm** (mean excluding the first run, which pays the one-time compile cost in the treatment arm). Single-occurrence steps (model loads) report the lone value.

## Compile-sensitive steps — cold (1st) vs warm (rest)

| stage | step | OFF cold | OFF warm | ON cold | ON warm | warm Δ | warm Δ% |
|-------|------|--------:|--------:|--------:|--------:|------:|-------:|
| s2_narration | fish_load | 38.3 | 38.3 | 39.1 | 39.1 | +0.8 | +2% |
| s2_narration | tts_generate | 138.5 | 126.0 | 69.2 | 56.7 | -69.3 | -55% |
| s3_images | flux_generate | 13.8 | 11.2 | 14.5 | 12.9 | +1.7 | +15% |
| s3_images | flux_load | 7.9 | 6.1 | 7.4 | 6.0 | -0.1 | -2% |

## All steps — warm (OFF vs ON)

| stage | step | kind | OFF | ON | Δ | Δ% | runs OFF/ON |
|-------|------|------|----:|----:|---:|---:|:----------:|
| s2_narration | tts_generate | infer | 126.0 | 56.7 | -69.3 | -55% | 3/3 |
| s2_narration | fish_load | load | 38.3 | 39.1 | +0.8 | +2% | 1/1 |
| s4_music | qwen_load | load | 0.0 | 14.2 | +14.2 | +0% | 0/1 |
| s3_images | flux_generate | infer | 11.2 | 12.9 | +1.7 | +15% | 36/39 |
| s3_images | flux_load | load | 6.1 | 6.0 | -0.1 | -2% | 16/16 |
| s3_images | image_uploads_wait | io | 0.0 | 3.4 | +3.4 | +0% | 0/1 |
| s2_narration | whisperx_load | load | 1.8 | 2.0 | +0.2 | +10% | 1/1 |
| s2_narration | decode_to_audio | infer | 0.8 | 0.8 | +0.0 | +5% | 3/3 |
| s2_narration | whisperx_align | infer | 0.5 | 0.4 | -0.0 | -9% | 3/3 |
| s2_narration | mp3_encode | cpu | 0.3 | 0.4 | +0.1 | +17% | 3/3 |
| s2_narration | align_model_load | load | 0.3 | 0.3 | -0.0 | -1% | 3/3 |
| s2_narration | whisperx_transcribe | infer | 0.2 | 0.2 | +0.0 | +6% | 3/3 |
| s4_music | energy_envelope | cpu | 0.0 | 0.0 | +0.0 | +0% | 0/5 |
| s2_narration | char_map | cpu | 0.0 | 0.0 | +0.0 | +0% | 3/3 |
| s4_music | brief_pass1 | infer | 0.0 | 0.0 | +0.0 | +0% | 0/5 |

## Stage wall per video (process_total: load+infer+I/O) — cold vs warm

| stage | OFF cold | OFF warm | ON cold | ON warm | warm Δ% |
|-------|--------:|--------:|--------:|--------:|-------:|
| s1_story | 977.3 | 1115.1 | 568.6 | 672.8 | -40% |
| s2_narration | 195.0 | 138.2 | 126.2 | 69.2 | -50% |
| s3_images | 0.0 | 0.0 | 27.3 | 27.3 | +0% |
| s4_music | 0.0 | 0.0 | 23.3 | 4.2 | +0% |

## Category roll-up (whole arm, all videos)

| category | OFF (s) | ON (s) | Δ% |
|----------|----:|----:|---:|
| load | 140.7 | 153.5 | +9% |
| infer | 802.4 | 708.1 | -12% |
| io | 0.0 | 3.4 | +0% |
| cpu | 1.1 | 2.7 | +156% |
| **GPU (load+infer)** | **943.1** | **861.6** | **-9%** |
| GPU per video (÷OFF=3, ÷ON=3) | 314.4 | 287.2 | -9% |

## Notes
- Negative Δ% = ON faster than OFF. The **warm** columns are the steady-state numbers; the ON **cold** column is inflated by the one-time torch.compile build and amortizes away over a real 100–150 video batch.
- Prompt-caching savings are not in this table (no latency effect) — see the stage-1 `cost_usd` / `cached_input_tokens` comparison in the run summary.


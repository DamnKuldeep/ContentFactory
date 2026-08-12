# Latency report

Source: `bench/profile_on.jsonl`

## stage_01_story —  12.7m total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| _(unmeasured: I/O + overhead)_ | io | 760.3 |  100% | download/upload/json |

## stage_02_narration —   62.2s total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| tts_generate | infer | 69.2 |  111% | chunks=10 |
| tts_generate | infer | 63.8 |  103% | chunks=9 |
| tts_generate | infer | 49.7 |   80% | chunks=9 |
| fish_load | load | 39.1 |   63% |  |
| whisperx_load | load | 2.0 |    3% |  |
| decode_to_audio | infer | 1.0 |    2% |  |
| decode_to_audio | infer | 0.8 |    1% |  |
| decode_to_audio | infer | 0.8 |    1% |  |
| whisperx_align | infer | 0.5 |    1% |  |
| align_model_load | load | 0.4 |    1% |  |
| whisperx_align | infer | 0.4 |    1% |  |
| whisperx_align | infer | 0.4 |    1% |  |
| mp3_encode | cpu | 0.4 |    1% |  |
| mp3_encode | cpu | 0.4 |    1% |  |
| mp3_encode | cpu | 0.4 |    1% |  |
| whisperx_transcribe | infer | 0.4 |    1% |  |
| align_model_load | load | 0.3 |    0% |  |
| align_model_load | load | 0.3 |    0% |  |
| whisperx_transcribe | infer | 0.2 |    0% |  |
| whisperx_transcribe | infer | 0.2 |    0% |  |
| char_map | cpu | 0.0 |    0% |  |
| char_map | cpu | 0.0 |    0% |  |
| char_map | cpu | 0.0 |    0% |  |

## stage_03_images —   27.3s total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| flux_generate | infer | 36.8 |  135% | n=4 steps=4 |
| flux_generate | infer | 29.0 |  106% | n=3 steps=4 |
| flux_generate | infer | 16.9 |   62% | n=4 steps=4 |
| flux_generate | infer | 16.8 |   61% | n=4 steps=4 |
| flux_generate | infer | 16.6 |   61% | n=4 steps=4 |
| flux_generate | infer | 16.6 |   61% | n=4 steps=4 |
| flux_generate | infer | 16.4 |   60% | n=4 steps=4 |
| flux_generate | infer | 16.4 |   60% | n=4 steps=4 |
| flux_generate | infer | 16.3 |   60% | n=4 steps=4 |
| flux_generate | infer | 14.7 |   54% | n=4 steps=4 |
| flux_generate | infer | 14.6 |   53% | n=4 steps=4 |
| flux_generate | infer | 14.5 |   53% | n=4 steps=4 |
| flux_generate | infer | 14.5 |   53% | n=4 steps=4 |
| flux_generate | infer | 12.2 |   45% | n=3 steps=4 |
| flux_generate | infer | 12.2 |   44% | n=3 steps=4 |
| flux_generate | infer | 12.1 |   44% | n=3 steps=4 |
| flux_generate | infer | 12.0 |   44% | n=3 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 11.0 |   40% | n=4 steps=4 |
| flux_generate | infer | 10.9 |   40% | n=4 steps=4 |
| flux_generate | infer | 10.9 |   40% | n=4 steps=4 |
| flux_generate | infer | 10.9 |   40% | n=4 steps=4 |
| flux_generate | infer | 10.9 |   40% | n=4 steps=4 |
| flux_generate | infer | 10.9 |   40% | n=4 steps=4 |
| flux_generate | infer | 9.4 |   35% | n=2 steps=4 |
| flux_generate | infer | 9.3 |   34% | n=2 steps=4 |
| flux_generate | infer | 9.0 |   33% | n=2 steps=4 |
| flux_load | load | 7.4 |   27% |  |
| flux_generate | infer | 6.3 |   23% | n=1 steps=4 |
| flux_load | load | 6.2 |   23% |  |
| flux_load | load | 6.1 |   22% |  |
| flux_load | load | 6.1 |   22% |  |
| flux_load | load | 6.1 |   22% |  |
| flux_load | load | 6.1 |   22% |  |
| flux_load | load | 6.0 |   22% |  |
| flux_load | load | 6.0 |   22% |  |
| flux_generate | infer | 6.0 |   22% | n=1 steps=4 |
| flux_load | load | 6.0 |   22% |  |
| flux_generate | infer | 6.0 |   22% | n=1 steps=4 |
| flux_load | load | 6.0 |   22% |  |
| flux_load | load | 5.9 |   22% |  |
| flux_load | load | 5.9 |   22% |  |
| flux_load | load | 5.9 |   22% |  |
| flux_load | load | 5.9 |   22% |  |
| flux_load | load | 5.9 |   22% |  |
| flux_generate | infer | 5.9 |   22% | n=1 steps=4 |
| flux_load | load | 5.9 |   22% |  |
| image_uploads_wait | io | 3.4 |   12% | n=1 |

## stage_04_music —    4.4s total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| brief_pass1 | infer | 16.0 |  364% |  |
| qwen_load | load | 14.2 |  322% |  |
| energy_envelope | cpu | 1.3 |   30% |  |
| energy_envelope | cpu | 0.1 |    1% |  |
| energy_envelope | cpu | 0.1 |    1% |  |
| energy_envelope | cpu | 0.0 |    1% |  |
| energy_envelope | cpu | 0.0 |    1% |  |
| brief_pass1 | infer | 0.0 |    0% |  |
| brief_pass1 | infer | 0.0 |    0% |  |
| brief_pass1 | infer | 0.0 |    0% |  |
| brief_pass1 | infer | 0.0 |    0% |  |

## Category roll-up

| category | seconds | note |
|----------|--------:|------|
| load | 153.5 | |
| infer | 708.1 | |
| io | 3.4 | |
| cpu | 2.7 | |
| **GPU (load+infer)** | **861.6** | the cost-relevant number |
| **grand total (wall)** | **854.2** | ≈ 14.2 min for one video |

## Slowest steps

| stage | step | kind | seconds |
|-------|------|------|--------:|
| stage_02_narration | tts_generate | infer | 69.2 |
| stage_02_narration | tts_generate | infer | 63.8 |
| stage_02_narration | tts_generate | infer | 49.7 |
| stage_02_narration | fish_load | load | 39.1 |
| stage_03_images | flux_generate | infer | 36.8 |
| stage_03_images | flux_generate | infer | 29.0 |
| stage_03_images | flux_generate | infer | 16.9 |
| stage_03_images | flux_generate | infer | 16.8 |
| stage_03_images | flux_generate | infer | 16.6 |
| stage_03_images | flux_generate | infer | 16.6 |
| stage_03_images | flux_generate | infer | 16.4 |
| stage_03_images | flux_generate | infer | 16.4 |


# Latency report

Source: `bench/profile_off.jsonl`

## stage_01_story —  19.0m total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| _(unmeasured: I/O + overhead)_ | io | 1142.2 |  100% | download/upload/json |

## stage_02_narration —  150.9s total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| tts_generate | infer | 138.5 |   92% | chunks=10 |
| tts_generate | infer | 138.3 |   92% | chunks=10 |
| tts_generate | infer | 113.7 |   75% | chunks=9 |
| fish_load | load | 38.3 |   25% |  |
| whisperx_load | load | 1.8 |    1% |  |
| decode_to_audio | infer | 1.1 |    1% |  |
| decode_to_audio | infer | 0.8 |    1% |  |
| decode_to_audio | infer | 0.7 |    0% |  |
| whisperx_align | infer | 0.6 |    0% |  |
| whisperx_align | infer | 0.5 |    0% |  |
| align_model_load | load | 0.4 |    0% |  |
| mp3_encode | cpu | 0.4 |    0% |  |
| mp3_encode | cpu | 0.4 |    0% |  |
| whisperx_align | infer | 0.4 |    0% |  |
| whisperx_transcribe | infer | 0.4 |    0% |  |
| mp3_encode | cpu | 0.3 |    0% |  |
| align_model_load | load | 0.3 |    0% |  |
| align_model_load | load | 0.2 |    0% |  |
| whisperx_transcribe | infer | 0.2 |    0% |  |
| whisperx_transcribe | infer | 0.2 |    0% |  |
| char_map | cpu | 0.0 |    0% |  |
| char_map | cpu | 0.0 |    0% |  |
| char_map | cpu | 0.0 |    0% |  |

## stage_03_images —  506.8s total

| step | kind | seconds | % stage | notes |
|------|------|--------:|--------:|-------|
| flux_generate | infer | 13.9 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.9 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.9 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.8 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.8 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.8 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.6 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.5 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.5 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.5 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.5 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.5 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.4 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.3 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.3 |    3% | n=4 steps=4 |
| flux_generate | infer | 13.3 |    3% | n=4 steps=4 |
| flux_generate | infer | 10.7 |    2% | n=3 steps=4 |
| flux_generate | infer | 10.7 |    2% | n=3 steps=4 |
| flux_generate | infer | 10.7 |    2% | n=3 steps=4 |
| flux_generate | infer | 10.1 |    2% | n=3 steps=4 |
| flux_generate | infer | 10.0 |    2% | n=3 steps=4 |
| flux_generate | infer | 10.0 |    2% | n=3 steps=4 |
| flux_load | load | 7.9 |    2% |  |
| flux_generate | infer | 7.3 |    1% | n=2 steps=4 |
| flux_generate | infer | 7.3 |    1% | n=2 steps=4 |
| flux_generate | infer | 7.3 |    1% | n=2 steps=4 |
| flux_generate | infer | 7.2 |    1% | n=2 steps=4 |
| flux_load | load | 7.0 |    1% |  |
| flux_generate | infer | 6.7 |    1% | n=2 steps=4 |
| flux_load | load | 6.3 |    1% |  |
| flux_load | load | 6.1 |    1% |  |
| flux_load | load | 6.1 |    1% |  |
| flux_load | load | 6.1 |    1% |  |
| flux_load | load | 6.1 |    1% |  |
| flux_load | load | 6.1 |    1% |  |
| flux_load | load | 6.0 |    1% |  |
| flux_load | load | 6.0 |    1% |  |
| flux_load | load | 6.0 |    1% |  |
| flux_load | load | 6.0 |    1% |  |
| flux_load | load | 6.0 |    1% |  |
| flux_load | load | 6.0 |    1% |  |
| flux_load | load | 5.9 |    1% |  |
| flux_load | load | 5.9 |    1% |  |
| flux_generate | infer | 3.9 |    1% | n=1 steps=4 |
| flux_generate | infer | 3.8 |    1% | n=1 steps=4 |
| flux_generate | infer | 3.8 |    1% | n=1 steps=4 |

## Category roll-up

| category | seconds | note |
|----------|--------:|------|
| load | 140.7 | |
| infer | 802.4 | |
| cpu | 1.1 | |
| **GPU (load+infer)** | **943.1** | the cost-relevant number |
| **grand total (wall)** | **1799.9** | ≈ 30.0 min for one video |

## Slowest steps

| stage | step | kind | seconds |
|-------|------|------|--------:|
| stage_02_narration | tts_generate | infer | 138.5 |
| stage_02_narration | tts_generate | infer | 138.3 |
| stage_02_narration | tts_generate | infer | 113.7 |
| stage_02_narration | fish_load | load | 38.3 |
| stage_03_images | flux_generate | infer | 13.9 |
| stage_03_images | flux_generate | infer | 13.9 |
| stage_03_images | flux_generate | infer | 13.9 |
| stage_03_images | flux_generate | infer | 13.8 |
| stage_03_images | flux_generate | infer | 13.8 |
| stage_03_images | flux_generate | infer | 13.8 |
| stage_03_images | flux_generate | infer | 13.6 |
| stage_03_images | flux_generate | infer | 13.5 |


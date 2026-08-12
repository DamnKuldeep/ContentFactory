# Results

What the pipeline actually produced, and how the production run went.

> Back to the [README](../README.md) · design in [ARCHITECTURE.md](ARCHITECTURE.md) · numbers in [PERFORMANCE.md](PERFORMANCE.md)

---

## Sample videos

Five finished videos, unedited pipeline output. Click to play on GitHub.

| | Preview | Length | Notes |
|---|---|---|---|
| **1** | [▶ sample_1.mp4](media/samples/sample_1.mp4) | 1:40 | Sepia ink-wash style; a scholar's study of manuscripts. Heavy Ken Burns motion, gold/white karaoke subtitles. |
| **2** | [▶ sample_2.mp4](media/samples/sample_2.mp4) | 1:35 | Cel-shaded; a weaving mill and a marionette. Cool blue-grey palette held across every scene. |
| **3** | [▶ sample_3.mp4](media/samples/sample_3.mp4) | 1:32 | Muted comic style; a boarding-house room. Note the style anchor keeps the wallpaper and furniture consistent between scenes. |
| **4** | [▶ sample_4.mp4](media/samples/sample_4.mp4) | 1:33 | Low-key noir; near-black frames with a single candle source. Demonstrates the palette lock at the dark end. |
| **5** | [▶ sample_5.mp4](media/samples/sample_5.mp4) | 1:37 | Soft line-art; a monastery interior with recurring characters rendered from the stage-1 character sheets. |

**Full-resolution 1080×1920 originals:**
[Google Drive folder](https://drive.google.com/drive/folders/1lbGwujEgAfGrJiuR9XZ7fJmmGF2hBxvV?usp=drive_link)

The copies in this repo are downscaled to 540×960 (CRF 30, ~6 MB each) purely to keep the clone small.
Everything else — timing, subtitles, mix, transitions — is exactly as the pipeline rendered it.

### What each sample demonstrates

| Feature | Where to look |
|---|---|
| **Word-accurate subtitles** | The white word is always the one being spoken; the surrounding gold words are context. This is driven by per-character alignment, not a per-line timer. |
| **Beat-accurate cuts** | Images change on the narrative beat because each scene carries `char_start`/`char_end` offsets into the script, resolved through the same alignment. |
| **Style consistency** | 28–40 images per video, all from independent FLUX calls, held together by a locked style anchor + hex palette + character/setting sheets. |
| **Score that tracks the voice** | The music thins out under quiet delivery and swells at the escalation — it was written against an RMS energy envelope of the finished narration. |
| **Stable voice/music balance** | BGM sits at a fixed ratio of the narration's *measured* LUFS, then ducks under it via sidechain compression. The balance is identical across all five despite different narration loudness. |
| **Style variety across stories** | Each story independently selects a visual style from a shuffled pool of six candidates, which is why no two samples look alike. |

---

## The 100-video production run

The pipeline was run as a real batch, not a demo: **batch `20260628_153812`, 100 videos**, split across
three machines handing off through Google Drive.

### Topology

| Machine | Role | Stages |
|---|---|---|
| CPU laptop | story node | 1 |
| GPU laptop (RTX 5090 Laptop, 24 GB) | media node | 2 → 3 → 4, one venv per role, run sequentially |
| CPU laptop | compose node | 5 (libx264) — or NVENC on the GPU box |

Stage 1 ran on the CPU box and pushed the full seeded DAG to the Drive-hosted manifest. The GPU box
then pulled that DB and drained each role in turn with `scripts/worker.py <role>` — no batch IDs, no
coordinator. Each role loads its model once and processes every video whose upstream stage is
`COMPLETE`, pushing its rows back to Drive after each job.

### Outcome

| | |
|---|---|
| Stories requested | 100 |
| Stories completed through stage 1 | **88** |
| Stories not completed | 12 (`story_num` 89–100) |
| Cause | OpenRouter returned **403 — total spend limit reached** on the API key |

The 12 failures are a **billing ceiling, not a code failure**. The manifest recorded them as `FAILED`
with the error text; raising the key's limit and running `--retry-failed` requeues exactly those 12 into
the same batch. Downstream stages correctly skipped them the whole time, because a stage-N job only
becomes claimable once the same story's stage-(N−1) job is `COMPLETE` — the 12 incomplete stories were
simply never offered to the narration/image/music workers.

### Cross-machine resume, verified

Before the bulk drain, the handoff itself was validated end-to-end on 2026-06-29:

1. One full video (story 1, *"The Dentist of Chacabuco"*) was taken through stages 2→5 on the 24 GB box:
   narration → images → music → compose → `final.mp4` (1080×1920, 92 s, h264+aac), uploaded to Drive and
   recorded in the video catalog, **with the DB pushed to the Drive manifest after every stage**.
2. A **completely fresh, empty local database** was then pointed at the same Drive folder. It
   `sync_pull`ed, correctly saw story 1 as done, and **claimed story 2** — resuming the batch on what was
   effectively a new machine.

No OOM, no segfault. That test is what established that the Drive-DB checkpoint model, the refresh-only
OAuth, and the per-video folder layout all actually work together.

### Configuration used

```dotenv
DRIVE_DB=1
BGM_TWO_PASS=0            # REQUIRED on 24 GB — two-pass keeps Qwen and ACE-Step resident together
FISH_COMPILE=1            # ~2.1x faster TTS; amortizes across the batch
FLUX_COMPILE=1
FLUX_COMPILE_MODE=default # reduce-overhead's CUDA graphs OOM FLUX.2 on 24 GB
```

### What the run taught us

Every one of these was found by running the thing for real, and every one is fixed in this repo:

| Failure seen in production | Root cause | Fix |
|---|---|---|
| Image worker died with `rc=139`, no traceback | Concurrent resumable Drive uploads raced in OpenSSL | Upload pool serialized to 1 worker (still overlaps with the next generation batch) |
| Stage 5 produced 0-byte images, then segfaulted | The Drive `service` object is httplib2-backed and not thread-safe | Downloads made serial, with retries |
| Two workers rendered the same story (2× GPU cost) | `claim_job` selected then updated with no guard | Atomic `UPDATE … WHERE status='PENDING'`, winner decided by `rowcount` |
| GPU boxes pulled the DB and sat idle with 0 jobs | `produce.py` pushed only stage-1 rows, stripping the seeded stages 2–5 on the next pull | Push the full seeded DAG |
| Images landed in `Batch_unknown/video_0000/` | `batch_id`/`story_num` weren't in the stage-1 JSON that stage 3 reads | Injected from the job record in `stage_03/run.py` |
| Images flashed past in the first few seconds | Stage-1 scenes carry no character offsets, so every scene mapped to t≈0 | `compose.py` derives `char_start`/`char_end` by locating each scene's narration in the script |
| Stage 4 crashed writing the brief | Qwen-Omni's audio input expects a path string; a `(waveform, sr)` tuple was passed | Brief is grounded on the energy envelope (text), which is also OOM-safe on 24 GB |
| CUDA OOM after ~20 images | `torch.compile(mode="reduce-overhead")` CUDA graphs reserve ~2.7 GB | `FLUX_COMPILE_MODE`, defaulting to `default` |

### Known quality limits

- **Fish occasionally mispronounces proper nouns.** This is TTS quality and is independent of the sync
  work — the subtitles still land on the correct word, because alignment is computed from what was
  actually said and then mapped back onto the script.
- **FLUX renders in-image text as gibberish** when a prompt asks for legible writing (visible on some
  ledger/label close-ups). Not corrected; it's a known limitation of the image model at this size.
- **Two machines running the *same* stage concurrently is unsupported** by the checkpoint-sync model.
  One machine per stage-group is the tested envelope.

---

## Reading the catalog

Every finished video is recorded in a `videos` table in the manifest (title, link, Drive file id).
From any machine with credentials:

```bash
python run.py --list-videos --batch 20260628_153812
python run.py --list-videos --batch 20260628_153812 --share-public   # flip finals to anyone-with-link
```

Both also export `videos.csv` and `videos.json` to the Drive parent folder.

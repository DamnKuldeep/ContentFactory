# Results

The samples, and how the big run actually went.

> [README](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [PERFORMANCE.md](PERFORMANCE.md)

---

## Samples

Five videos, straight out of the pipeline, nothing touched up. Click to play.

| | Length | What's in it |
|---|---|---|
| [Sample 1](media/samples/sample_1.mp4) | 1:40 | Sepia ink-wash. A scholar's room full of manuscripts. Lots of Ken Burns movement. |
| [Sample 2](media/samples/sample_2.mp4) | 1:35 | Cel-shaded, cold blue-grey. A weaving mill and a marionette. |
| [Sample 3](media/samples/sample_3.mp4) | 1:32 | Muted comic style. Watch the wallpaper and furniture stay the same between scenes — that's the style anchor doing its job. |
| [Sample 4](media/samples/sample_4.mp4) | 1:33 | Nearly black, one candle. Shows the palette holding together at the dark end. |
| [Sample 5](media/samples/sample_5.mp4) | 1:37 | Soft line-art. Recurring characters drawn from the character sheets stage 1 wrote. |

They're shrunk to 540×960 so cloning this repo doesn't take ten minutes. The originals are 1080×1920 at about 5 Mbps. Everything else — timing, subtitles, transitions, mix — is exactly what came out.

### What each one shows off

**Subtitles that follow the voice.** The white word is the one being said right now, the gold ones around it are context. Every character of the script has its own timestamp, so this survives the TTS mispronouncing things.

**Cuts that land on the beat.** Each scene knows which characters of the script it covers, and those resolve to real times through the same alignment. So images change when the sentence changes, not every N seconds.

**One consistent look across 34 images.** Every image is a separate FLUX call with no memory of the others. What holds them together is a style anchor, a fixed hex palette, and character/setting descriptions written in stage 1 and pasted into every prompt.

**Music that moves with the narration.** The score is written against an energy curve pulled out of the finished voice track, so it goes sparse when the delivery is quiet and swells at the turn.

**The same voice/music balance every time.** Music sits at a fixed fraction of the narration's *measured* loudness rather than a hardcoded volume, then ducks under it. All five sound balanced despite having different narration levels.

**Different looks per story.** Each story picks its style from a shuffled shortlist of six, which is why no two samples look alike.

---

## The 100-video run

Batch `20260628_153812`. Not a demo — 100 videos, three laptops, handing off through Drive.

| Machine | Job |
|---|---|
| CPU laptop | Stage 1, writing stories |
| GPU laptop (RTX 5090, 24 GB) | Stages 2, 3, 4 — one venv each, run one after the other |
| CPU laptop | Stage 5, cutting video |

Stage 1 ran on the CPU box and pushed the whole seeded job list up to Drive. The GPU box pulled that database and drained one role at a time with `scripts/worker.py <role>`. No batch IDs passed around, no coordinator. Each role loads its model once, works through every video whose previous stage is done, and pushes its rows back after each one.

### How it ended

**88 of 100 finished.** The last twelve (stories 89–100) failed in stage 1 with a `403 — total spend limit reached` from OpenRouter.

That's a billing ceiling, not a bug. The database recorded them as `FAILED` with the error text; raise the limit, run `--retry-failed`, and those twelve go back in the same batch. Everything downstream ignored them the entire time without being told to, because a stage-N job only becomes claimable once that story's stage-(N−1) job says `COMPLETE`. The narration, image and music workers simply never saw them.

### Proving the handoff worked first

Before committing to the bulk run I tested the thing I was least sure about — whether a different machine could really pick up mid-batch.

One video (story 1, *"The Dentist of Chacabuco"*) went through stages 2→5 on the GPU box: narration, images, music, compose, out came a 92-second 1080×1920 h264+aac file, uploaded and logged, with the database pushed to Drive after every single stage.

Then I pointed a **completely empty local database** at the same Drive folder. It pulled, saw story 1 was done, and claimed story 2. No OOM, no segfault.

That's what convinced me the Drive-hosted database, the refresh-only auth, and the folder layout actually worked together rather than just looking like they should.

### Settings used

```dotenv
DRIVE_DB=1
BGM_TWO_PASS=0            # two-pass keeps Qwen and ACE-Step in VRAM together; 24 GB won't take it
FISH_COMPILE=1            # roughly halves TTS time once the model's warm
FLUX_COMPILE=1
FLUX_COMPILE_MODE=default # reduce-overhead OOMs FLUX on this card
```

---

## Everything that broke

All of these came from running it for real, and all of them are fixed in this repo. Most would never have shown up in a five-video test.

**Image worker died with `rc=139` and no traceback.** Concurrent resumable uploads racing inside OpenSSL. Fixed by dropping the upload pool to one worker — it still overlaps with the next batch of images, so it cost nothing.

**Stage 5 wrote 0-byte images then segfaulted.** Google's Drive client is built on httplib2 and isn't thread-safe. Sharing one across a download pool corrupts things. Downloads are serial now, with retries.

**Two workers rendered the same story.** Double GPU time for one video. The claim was a SELECT followed by an UPDATE with a gap in between. Now it's a single conditional UPDATE and whoever gets `rowcount == 1` wins.

**GPU boxes pulled the database and found nothing to do.** `produce.py` was only pushing stage-1 rows, which wiped the seeded stages 2–5 off the remote on the next pull. Push the whole job list.

**Images landed in `Batch_unknown/video_0000/`.** Stage 3 reads the batch and story number out of the JSON it's handed, and stage 1's JSON doesn't have them. Now they're injected from the job record. Harmless — downstream finds images by file ID — but it made a mess of the folders.

**Images flashed past in the first few seconds.** Stage-1 scenes carry no character offsets, so every scene mapped to t≈0. `compose.py` now works out each scene's offsets by finding its narration inside the script.

**Stage 4 crashed writing the music brief.** Qwen-Omni wants a file path for audio input; it was being handed a `(waveform, sample_rate)` tuple. Grounding the brief on the energy curve instead is both correct and avoids pushing 90 seconds of audio into a model on a 24 GB card.

**CUDA OOM after about 20 images.** `torch.compile(mode="reduce-overhead")` builds CUDA graphs that reserve ~2.7 GB. Added `FLUX_COMPILE_MODE`, defaulting to `default`.

---

## Where it's still rough

**The voice mispronounces some proper nouns.** That's the TTS. It doesn't affect sync — the alignment is computed from what was actually said and then mapped back onto the script, so subtitles still land on the right word.

**FLUX writes gibberish when a prompt asks for legible text.** Visible on a couple of ledger and label close-ups. Known limit of the model at this size, not corrected.

**Two machines can't run the same stage at once.** The checkpoint sync assumes one machine owns a stage. One machine per stage-group is what's tested.

---

## Reading the catalog

Every finished video gets a row with its title and link. From any machine with credentials:

```bash
python run.py --list-videos --batch 20260628_153812
python run.py --list-videos --batch 20260628_153812 --share-public   # make the files link-shareable
```

Both also drop `videos.csv` and `videos.json` in the Drive folder.

# Results

The samples, and what came out of running it at scale.

> [README](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [PERFORMANCE.md](PERFORMANCE.md)

---

## Samples

Five videos, straight out of the pipeline, nothing touched up. Click to play.

| | Length | What's in it |
|---|---|---|
| [Sample 1](media/samples/sample_1.mp4) | 1:40 | Sepia ink-wash. A scholar's room full of manuscripts. Lots of Ken Burns movement. |
| [Sample 2](media/samples/sample_2.mp4) | 1:35 | Cel-shaded, cold blue-grey. A weaving mill and a marionette. |
| [Sample 3](media/samples/sample_3.mp4) | 1:32 | Muted comic style. Watch the wallpaper and furniture stay put between scenes — that's the style anchor doing its job. |
| [Sample 4](media/samples/sample_4.mp4) | 1:33 | Nearly black, one candle. Shows the palette holding together at the dark end. |
| [Sample 5](media/samples/sample_5.mp4) | 1:37 | Soft line-art. Recurring characters drawn from the character sheets stage 1 wrote. |

They're shrunk to 540×960 so cloning the repo stays quick. The originals are 1080×1920 at around 5 Mbps. Timing, subtitles, transitions and mix are exactly what came out.

### What each one shows off

**Subtitles that follow the voice.** The white word is the one being said right now, the gold ones around it are context. Every character of the script has its own timestamp, which is what makes this survive the TTS mispronouncing things.

**Cuts that land on the beat.** Each scene knows which characters of the script it covers, and those resolve to real times through the same alignment. Images change when the sentence changes, not every N seconds.

**One consistent look across 34 images.** Every image is a separate FLUX call with no memory of the others. What holds them together is a style anchor, a fixed hex palette, and character and setting descriptions written in stage 1 and pasted into every prompt.

**Music that moves with the narration.** The score is written against an energy curve pulled out of the finished voice track, so it thins out when the delivery is quiet and swells at the turn.

**The same voice-to-music balance every time.** Music sits at a fraction of the narration's *measured* loudness rather than a fixed volume, then ducks under it. All five sound balanced despite having different narration levels.

**A different look per story.** Each story picks its style from a shuffled shortlist of six, which is why no two samples look alike.

---

## Running it at scale

**88 videos, one batch, three laptops**, handing work to each other through Drive.

| Machine | Job |
|---|---|
| CPU laptop | Stage 1 — writing stories |
| GPU laptop (RTX 5090, 24 GB) | Stages 2, 3 and 4 — one venv each, run back to back |
| CPU laptop | Stage 5 — cutting video |

Stage 1 ran on the CPU box and pushed the whole seeded job list to Drive. The GPU box pulled that database and drained one role at a time with `scripts/worker.py <role>`. No batch ids passed between machines, no coordinator process. Each role loads its model once, works through every video whose previous stage is finished, and pushes its rows back after each one.

Roughly 25 minutes of wall clock and 375 GPU-seconds per video, dropping to about 285 GPU-seconds with the compile flags on. Full breakdown in [PERFORMANCE.md](PERFORMANCE.md).

### Proving the handoff first

Before committing to the full batch I tested the thing I was least sure about: whether a different machine could really pick up mid-run.

One video went through stages 2→5 on the GPU box — narration, images, music, compose — and out came a 92-second 1080×1920 h264+aac file, uploaded and logged, with the database pushed to Drive after every stage.

Then I pointed a **completely empty local database** at the same Drive folder. It pulled, saw the first story was done, and claimed the second one.

That's what convinced me the Drive-hosted database, the refresh-only auth and the folder layout actually worked together rather than just looking like they should.

### Settings used

```dotenv
DRIVE_DB=1
BGM_TWO_PASS=0            # two-pass keeps Qwen and ACE-Step in VRAM together; 24 GB won't take it
FISH_COMPILE=1            # roughly halves TTS time once the model's warm
FLUX_COMPILE=1
FLUX_COMPILE_MODE=default # reduce-overhead OOMs FLUX on this card
```

---

## What running it properly taught me

Every one of these came out of real use, and all of them are fixed in the repo. None would have shown up in a five-video test, which is the point.

**Concurrency bugs only appear at volume.** Two workers took the same job once — the claim was a SELECT followed by an UPDATE with a gap between them. It's a single conditional UPDATE now, and whoever gets `rowcount == 1` wins. Cheap fix, but it costs double GPU time every time it happens and you'd never see it with one worker.

**Google's client libraries aren't as thread-safe as you'd hope.** Parallel uploads race inside OpenSSL and kill the process with no traceback. A shared Drive client across a download pool writes 0-byte files. Both are serial now, with retries — and in the upload case, moving it to a single background thread kept the overlap benefit anyway.

**Distributed bugs need more than one machine to find.** The seeder was only pushing stage-1 rows to the shared database, which wiped the seeded downstream rows on the next pull. Everything looked perfect on a single box and the GPU machines found nothing to do.

**Sync problems are invisible until you watch the output.** Stage-1 scenes carry no character offsets, so compose mapped every scene to t≈0 and all 34 images flashed past in the first few seconds. Nothing errored. `compose.py` now finds each scene's narration inside the script and works the offsets out.

**VRAM limits shape the architecture.** Qwen and ACE-Step don't co-exist on 24 GB, so stage 4 evicts one before loading the other. `torch.compile`'s `reduce-overhead` mode reserves 2.7 GB of CUDA graphs and OOMs FLUX after about 20 images, which is why there's a `FLUX_COMPILE_MODE` setting at all.

---

## Where it's still rough

**The voice mispronounces some proper nouns.** That's the TTS. It doesn't affect sync — the alignment is computed from what was actually said and mapped back onto the script, so subtitles still land on the right word.

**FLUX writes gibberish when a prompt asks for legible text**, visible on a couple of ledger and label close-ups. A known limit of the model at this size.

**One machine per stage.** The checkpoint sync assumes it. Two machines on the same stage at once isn't supported.

---

## Reading the catalog

Every finished video gets a row with its title and link. From any machine with credentials:

```bash
python run.py --list-videos --batch <id>
python run.py --list-videos --batch <id> --share-public   # make the files link-shareable
```

Both also drop `videos.csv` and `videos.json` into the Drive folder.

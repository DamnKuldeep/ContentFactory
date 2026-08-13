# Review and posting

Getting a finished video from Drive onto Instagram and YouTube, with a person checking it first.

> [README](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Why there's a person in the loop

The pipeline can run unattended, but I didn't want it posting on its own. One bad video on a real account does more damage than ten good ones being late. So nothing gets posted until someone has actually watched it.

```
   finished          Google           review          approved         posted
    video    ──►     Sheet    ──►      app     ──►     queue    ──►    1 per platform
  (in Drive)                       approve /                          per 4 hours
                                    reject
```

**The Google Sheet is the database.** That sounds like a shortcut, and it isn't. Reviewers need no account, no login and no VPN. The state is a spreadsheet anyone can read, sort or fix by hand. And the same rows work from a laptop uploader and a hosted web app without me running a backend for either.

---

## The pieces

| File | Job |
|---|---|
| `scripts/seed_sheet.py` | Takes finished videos, makes them link-shareable, adds a row each. Safe to re-run — it only fills in identity columns, never overwrites someone's review decision. |
| `social/sheet.py` | All the Sheet reading and writing. Only needs `gspread` and `google-auth` so it can be dropped into the hosted app. Reads the OAuth token from a file *or* from an env var, for hosting secrets. |
| `review_app/app.py` | The Gradio app. Log in, watch a random pending video, approve or reject with a reason, skip. Shows live queue counts. |
| `social/metadata.py` | Writes the description, 12–20 hashtags and a YouTube title from the story's own logline and script. Runs locally inside the uploader, so the OpenRouter key never goes near the hosted app. |
| `social/uploader.py` | The throttled drain. Per platform: if the gate's open and something's waiting, take the oldest, get its metadata, download, post, update the Sheet. |
| `social/ig_adapter.py` | Instagram Reels via `instagrapi`, with a saved session. |
| `social/yt_adapter.py` | YouTube Shorts via the official Data API. |

---

## What's in the Sheet

One row per video, keyed on batch + story number.

| Columns | For |
|---|---|
| `batch_id`, `story_num` | Matching it back to the pipeline |
| `title`, `drive_file_id`, `video_url` | Where the video lives |
| `status` | `pending` → `in_queue` → `uploaded`, or `rejected` |
| `rejection_reason`, `reviewer`, `updated_at` | Who decided what, when, and why |
| `description`, `hashtags` | Generated once, then reused |
| `ig_status`, `ig_url`, `ig_posted_at` | `""` → `queued` → `posted` or `failed` |
| `yt_status`, `yt_url`, `yt_posted_at` | Same for YouTube |

A video sits at `pending` until someone looks at it. Approving flips it to `in_queue` and marks both platforms `queued`. The uploader moves each platform independently, and once both say `posted` the row becomes `uploaded`. Nothing can post twice because the uploader only picks rows whose platform status isn't already `posted`.

---

## Throttling

Both platforms hate bursts, so the uploader posts at most once per platform per `UPLOAD_INTERVAL_HOURS` (default 4).

The important bit: it works out whether the gate is open by reading the newest `*_posted_at` in the Sheet, not by remembering anything. Which means you can kill it whenever, restart it, only run it when your laptop happens to be on — and it still paces itself correctly. Two copies running by accident won't double-post either, since both read the same gate and the same "not yet posted" filter.

```bash
# Rehearsal — exercises the queue, the throttle and the Sheet writes. Posts nothing, calls no LLM.
python social/uploader.py --once --dry-run

# Live — check every 30 minutes, post when a platform's gate opens
python social/uploader.py --loop --interval-hours 4
```

---

## Setting it up

### Seed the Sheet

Your OAuth token needs the Sheets scope as well as Drive. `shared/drive.py` asks for both, so if your token predates that, run `scripts/drive_auth.py` again.

```bash
python scripts/seed_sheet.py --batch <batch_id>
#   creates the spreadsheet if SHEET_ID isn't set, and prints the id to put in .env
python scripts/seed_sheet.py --batch <batch_id> --no-public   # skip making files shareable
```

### Deploy the review app

It runs free on a Hugging Face Space. Make a Gradio Space and upload `review_app/app.py`, a copy of `social/sheet.py`, and `review_app/requirements.txt`. Then set these secrets:

| Secret | What |
|---|---|
| `SHEET_ID` | The id `seed_sheet.py` printed |
| `GOOGLE_TOKEN_JSON` | The whole contents of `token.json` |
| `REVIEW_USERS` | `alice:pw1,bob:pw2` — the login name gets recorded as the reviewer |
| `SHEET_TAB` | Optional, defaults to `videos` |

Or just run it locally:

```bash
SHEET_ID=... REVIEW_USERS=admin:admin python review_app/app.py
```

### Platform credentials

**YouTube** is the easy one — official API, works from anywhere, no ToS problem. It needs its own OAuth token with the `youtube.upload` scope for whoever owns the channel. This is *not* the Drive token. Point `YT_TOKEN_PATH` or `YT_TOKEN_JSON` at it. An upload costs about 1600 of your 10,000 daily quota units, so roughly six a day, which is well above one-per-four-hours.

**Instagram** is the awkward one. `instagrapi` is unofficial and against Instagram's terms — it logs in as a real account. If you use it: point it at a burner, keep the throttle on, and run it from a home connection. Datacenter logins get challenged and blocked fast. Credentials go in `IG_USERNAME` and `IG_PASSWORD`; the session is saved to `social/.ig_session.json` (gitignored) so it isn't logging in every run. It's imported lazily, so everything else works fine without `instagrapi` installed at all.

---

## When something fails

A failed post writes `failed` and a truncated error into the Sheet and moves on. Next cycle just tries again, because the row is still `in_queue` and still not `posted`.

The downloaded video gets deleted in a `finally` block, so a crash can't slowly fill your disk. And metadata is cached in the Sheet, so a retry never pays for a second LLM call.

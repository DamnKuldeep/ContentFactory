---
title: Video Review Queue
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Video Review Queue

A small Gradio app for checking generated videos before anything gets posted. Log in, watch a random pending video, approve it or reject it with a reason. Approved ones go into the upload queue, which a local uploader drains at one post per platform every few hours. The header shows how many are waiting.

All the state lives in a Google Sheet, so there's no database to run and reviewers don't need accounts on anything.

## Hosting it free on Hugging Face Spaces

1. Make a **Gradio** Space and upload `app.py`, a copy of `social/sheet.py`, and `requirements.txt`.
2. Under **Settings → Secrets**, add:
   - `SHEET_ID` — the spreadsheet id that `seed_sheet.py` printed
   - `GOOGLE_TOKEN_JSON` — the whole contents of `token.json` (needs Drive + Sheets scope)
   - `REVIEW_USERS` — `alice:pw1,bob:pw2`. Whichever name they log in with gets recorded as the reviewer.
   - `SHEET_TAB` — optional, defaults to `videos`
3. It boots, asks for a login, and starts serving the queue.

## Running it locally

```bash
SHEET_ID=... REVIEW_USERS=admin:admin python app.py
```

From inside `review_app/`, it finds `token.json` on its own.

Full write-up of the review and posting flow: [docs/PUBLISHING.md](../docs/PUBLISHING.md).

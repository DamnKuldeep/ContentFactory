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

Reviewers log in, get a random **pending** video, and **Approve** or **Reject (+reason)**. State
lives in a Google Sheet; approved videos enter the upload queue (drained by the local uploader at
1/platform/3-4 h). The header shows live queue counts.

## Deploy free on Hugging Face Spaces
1. Create a **Gradio** Space; upload `app.py`, `sheet.py` (copy of `social/sheet.py`), `requirements.txt`.
2. Space **Settings → Secrets**:
   - `SHEET_ID` — the spreadsheet id (from `seed_sheet.py`).
   - `GOOGLE_TOKEN_JSON` — full contents of `token.json` (OAuth token with Drive + Sheets scope).
   - `REVIEW_USERS` — `alice:pw1,bob:pw2` (each name is recorded as approver/rejecter).
   - optional `SHEET_TAB` (default `videos`).
3. The Space boots, prompts for login, and serves the queue.

## Run locally
```bash
SHEET_ID=... REVIEW_USERS=admin:admin ../.venv_review/bin/python app.py   # from review_app/, token.json auto-found
```

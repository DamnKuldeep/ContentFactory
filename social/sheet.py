"""
Google Sheet data-access for the review/publish system — the single source of truth.

Standalone (only needs `gspread` + `google-auth`) so it can be imported by the local seeder/uploader
AND bundled into the Gradio HF Space. Auth reuses the existing OAuth token (Drive + Sheets scope);
the token can be a file (DRIVE_TOKEN_PATH / ./token.json) or raw JSON in env (GOOGLE_TOKEN_JSON) for
HF Spaces secrets. The spreadsheet is addressed by its key in env SHEET_ID (tab SHEET_TAB, "videos").

Status lifecycle:  pending -> in_queue (approved) -> uploaded (both platforms posted)  |  rejected
Per-platform:      ig_status / yt_status in {"", queued, posted, failed}
"""

import json
import os
import random
import time

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

COLUMNS = [
    "batch_id", "story_num", "title", "drive_file_id", "video_url",
    "status", "rejection_reason", "reviewer", "updated_at",
    "description", "hashtags",
    "ig_status", "ig_url", "ig_posted_at",
    "yt_status", "yt_url", "yt_posted_at",
]

SHEET_TAB = os.environ.get("SHEET_TAB", "videos")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ── auth ──────────────────────────────────────────────────────────────────────

def _resolve_token_text():
    """Return the OAuth token JSON text from env (GOOGLE_TOKEN_JSON) or a file path."""
    raw = os.environ.get("GOOGLE_TOKEN_JSON", "").strip()
    if raw:
        return raw
    for p in (os.environ.get("DRIVE_TOKEN_PATH", "").strip(),
              "token.json",
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "token.json")):
        if p and os.path.exists(p):
            with open(p) as f:
                return f.read()
    raise RuntimeError("No OAuth token found (set GOOGLE_TOKEN_JSON or DRIVE_TOKEN_PATH / token.json). "
                       "Run scripts/drive_auth.py once with the Sheets scope added.")


def _client():
    import gspread
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    info = json.loads(_resolve_token_text())
    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return gspread.authorize(creds)


# ── worksheet open / create ───────────────────────────────────────────────────

def open_ws(create_if_missing=False):
    """Open the `videos` worksheet by SHEET_ID. If create_if_missing and SHEET_ID is unset,
    create a new spreadsheet, write the header, and print its id to put in .env."""
    gc = _client()
    sid = os.environ.get("SHEET_ID", "").strip()
    if sid:
        sh = gc.open_by_key(sid)
    else:
        if not create_if_missing:
            raise RuntimeError("SHEET_ID not set. Run seed_sheet.py once to create the sheet.")
        sh = gc.create(os.environ.get("SHEET_TITLE", "ContentFactory Review Queue"))
        print(f"[sheet] created spreadsheet — put this in .env:\n  SHEET_ID={sh.id}")
        share = os.environ.get("SHEET_SHARE_EMAIL", "").strip()
        if share:
            sh.share(share, perm_type="user", role="writer")
    try:
        ws = sh.worksheet(SHEET_TAB)
    except Exception:
        ws = sh.add_worksheet(title=SHEET_TAB, rows=1000, cols=len(COLUMNS))
    # ensure header
    head = ws.row_values(1)
    if head != COLUMNS:
        ws.update([COLUMNS], "A1", value_input_option="RAW")
    return ws


# ── reads ─────────────────────────────────────────────────────────────────────

def all_rows(ws=None):
    ws = ws or open_ws()
    return ws.get_all_records(expected_headers=COLUMNS)


def _find_row_index(ws, batch_id, story_num):
    """1-based sheet row for (batch_id, story_num), or None. Row 1 is the header."""
    col_b = ws.col_values(1)   # batch_id
    col_s = ws.col_values(2)   # story_num
    for i in range(1, len(col_b)):           # skip header
        if col_b[i] == str(batch_id) and str(col_s[i]) == str(story_num):
            return i + 1
    return None


def get_pending_random(ws=None):
    ws = ws or open_ws()
    pend = [r for r in all_rows(ws) if (r.get("status") or "pending") == "pending"]
    return random.choice(pend) if pend else None


def queue_counts(ws=None):
    ws = ws or open_ws()
    rows = all_rows(ws)
    def n(pred): return sum(1 for r in rows if pred(r))
    st = lambda r: (r.get("status") or "pending")
    return {
        "pending":   n(lambda r: st(r) == "pending"),
        "in_queue":  n(lambda r: st(r) == "in_queue"),
        "uploaded":  n(lambda r: st(r) == "uploaded"),
        "rejected":  n(lambda r: st(r) == "rejected"),
        "ig_waiting": n(lambda r: st(r) == "in_queue" and (r.get("ig_status") or "") != "posted"),
        "yt_waiting": n(lambda r: st(r) == "in_queue" and (r.get("yt_status") or "") != "posted"),
        "total":     len(rows),
    }


def next_for_platform(platform, ws=None):
    """Oldest in_queue row not yet posted to `platform` (by updated_at). platform in {ig,yt}."""
    ws = ws or open_ws()
    col = f"{platform}_status"
    cand = [r for r in all_rows(ws)
            if (r.get("status") or "") == "in_queue" and (r.get(col) or "") != "posted"]
    cand.sort(key=lambda r: r.get("updated_at") or "")
    return cand[0] if cand else None


# ── writes ────────────────────────────────────────────────────────────────────

def _update_fields(ws, batch_id, story_num, fields: dict):
    idx = _find_row_index(ws, batch_id, story_num)
    if idx is None:
        raise RuntimeError(f"row not found: {batch_id}/{story_num}")
    # write each changed column by its A1 cell
    updates = []
    for k, v in fields.items():
        c = COLUMNS.index(k) + 1
        updates.append({"range": _a1(idx, c), "values": [[v]]})
    ws.batch_update(updates, value_input_option="RAW")


def _a1(row, col):
    s = ""
    while col:
        col, r = divmod(col - 1, 26)
        s = chr(65 + r) + s
    return f"{s}{row}"


def upsert(row: dict, ws=None):
    """Insert or update a video row keyed by (batch_id, story_num). Only sets provided columns;
    new rows default status='pending'."""
    ws = ws or open_ws()
    idx = _find_row_index(ws, row["batch_id"], row["story_num"])
    if idx is None:
        row.setdefault("status", "pending")
        ws.append_row([row.get(c, "") for c in COLUMNS], value_input_option="RAW")
    else:
        _update_fields(ws, row["batch_id"], row["story_num"],
                       {k: v for k, v in row.items() if k in COLUMNS})


def set_decision(batch_id, story_num, status, reviewer, reason="", ws=None):
    """Approve (status='in_queue') or reject (status='rejected'). Sets queued platform states on approve."""
    ws = ws or open_ws()
    fields = {"status": status, "reviewer": reviewer, "updated_at": _now(), "rejection_reason": reason}
    if status == "in_queue":
        fields["ig_status"] = "queued"
        fields["yt_status"] = "queued"
    _update_fields(ws, batch_id, story_num, fields)


def set_metadata(batch_id, story_num, description, hashtags, ws=None):
    ws = ws or open_ws()
    _update_fields(ws, batch_id, story_num,
                   {"description": description, "hashtags": hashtags})


def mark_posted(batch_id, story_num, platform, url, ws=None):
    """Mark a platform posted; flip status->uploaded when both ig and yt are posted."""
    ws = ws or open_ws()
    _update_fields(ws, batch_id, story_num,
                   {f"{platform}_status": "posted", f"{platform}_url": url,
                    f"{platform}_posted_at": _now()})
    # re-read this row to see if both done
    for r in all_rows(ws):
        if r.get("batch_id") == str(batch_id) and str(r.get("story_num")) == str(story_num):
            if (r.get("ig_status") == "posted") and (r.get("yt_status") == "posted"):
                _update_fields(ws, batch_id, story_num, {"status": "uploaded", "updated_at": _now()})
            break


def mark_failed(batch_id, story_num, platform, err="", ws=None):
    ws = ws or open_ws()
    _update_fields(ws, batch_id, story_num,
                   {f"{platform}_status": "failed", f"{platform}_url": (err or "")[:200]})

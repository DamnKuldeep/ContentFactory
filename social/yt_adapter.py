"""
YouTube adapter — official Data API v3 resumable upload (Shorts = vertical <=60s; the #Shorts tag /
vertical aspect makes YT treat it as a Short). IP-agnostic, ToS-compliant.

Auth: a separate OAuth token with scope https://www.googleapis.com/auth/youtube.upload, for the
account that owns the target channel. Provide it via YT_TOKEN_JSON (raw json) or YT_TOKEN_PATH
(file). Create it once with a small consent script (see RUNME) — it is NOT the Drive token.

Quota: videos.insert ~1600 units; default 10000/day -> ~6 uploads/day (fine at 1 / 3-4 h).
google-api-python-client is already a pipeline dep; imported lazily here.
"""

import json
import os

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class YTAdapter:
    def __init__(self, token_json=None, token_path=None):
        self._token_json = token_json or os.environ.get("YT_TOKEN_JSON", "").strip()
        self._token_path = token_path or os.environ.get("YT_TOKEN_PATH", "").strip()
        self._svc = None

    def _service(self):
        if self._svc is not None:
            return self._svc
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        if self._token_json:
            info = json.loads(self._token_json)
        elif self._token_path and os.path.exists(self._token_path):
            info = json.load(open(self._token_path))
        else:
            raise RuntimeError("No YouTube token (set YT_TOKEN_JSON or YT_TOKEN_PATH).")
        creds = Credentials.from_authorized_user_info(info, YT_SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        self._svc = build("youtube", "v3", credentials=creds, cache_discovery=False)
        return self._svc

    def upload_short(self, video_path: str, title: str, description: str, tags=None) -> str:
        from googleapiclient.http import MediaFileUpload
        svc = self._service()
        body = {
            "snippet": {"title": title[:100], "description": description[:4900],
                        "tags": (tags or [])[:15], "categoryId": "24"},   # 24 = Entertainment
            "status": {"privacyStatus": os.environ.get("YT_PRIVACY", "public"),
                       "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024)
        req = svc.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        return f"https://youtube.com/shorts/{resp['id']}"

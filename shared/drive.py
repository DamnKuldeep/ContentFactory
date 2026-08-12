"""
Content Factory — Google Drive upload/download.

Auth: OAuth "installed-app" (Desktop) flow, **refresh-only** on the pipeline path — it loads a
copied token.json and refreshes silently, and NEVER opens a browser (so headless workers can't
hang). One-time interactive authorization is done separately by scripts/drive_auth.py
(authorize_interactive). A service-account key is used as a fallback if no OAuth secrets exist.

Folder layout (per video): <parent>/Batch_<batch_id>/video_<story_num:04d>/{metadata,narration,
images,music,subtitles,final}. Stage JSONs go to metadata/; media to their typed subfolders.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger("contentfactory.drive")

SCOPES = ["https://www.googleapis.com/auth/drive",
          "https://www.googleapis.com/auth/spreadsheets"]   # sheets: review-queue Google Sheet


def _lazy_imports():
    """Import the Google API client pieces only when needed."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    return build, MediaFileUpload, MediaIoBaseDownload


# ── OAuth credential resolution (refresh-only on the pipeline path) ───────────

def _search_dirs():
    here = os.path.dirname(os.path.abspath(__file__))   # .../ContentFactory/shared
    cf = os.path.dirname(here)                            # .../ContentFactory
    root = os.path.dirname(cf)                            # project root
    return [root, cf, here]


def _resolve_client_secrets(explicit: str = "") -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    for d in _search_dirs():
        for nm in ("credentials.json", "secrets.json"):
            p = os.path.join(d, nm)
            if os.path.exists(p):
                return p
    return None


def _resolve_token_path(explicit: str = "", client_secrets: str = None) -> str:
    if explicit:
        return explicit
    if client_secrets:
        return os.path.join(os.path.dirname(os.path.abspath(client_secrets)), "token.json")
    return os.path.join(_search_dirs()[0], "token.json")


def _oauth_service(client_secrets: str, token_path: str):
    """Build a Drive service from a copied token.json. REFRESH-ONLY — never opens a browser."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    build, _, _ = _lazy_imports()

    if not token_path or not os.path.exists(token_path):
        raise RuntimeError(
            f"No Drive token at '{token_path}'. Run `python scripts/drive_auth.py` on a machine "
            f"with a browser, then copy credentials.json + token.json to this machine."
        )
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w") as f:
                f.write(creds.to_json())
            logger.info("Refreshed Drive access token (%s)", token_path)
        else:
            raise RuntimeError(
                f"Drive token at '{token_path}' is invalid / not refreshable. Re-run "
                f"`python scripts/drive_auth.py` and copy the new token.json."
            )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def authorize_interactive(client_secrets: str = None, token_path: str = None) -> str:
    """One-time browser authorization → writes token.json. Used ONLY by scripts/drive_auth.py."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    cs = _resolve_client_secrets(client_secrets or "")
    if not cs:
        raise FileNotFoundError(
            "OAuth client secrets not found (credentials.json / secrets.json). "
            "Download the Desktop OAuth client JSON and place it at the project root."
        )
    tp = token_path or _resolve_token_path("", cs)
    flow = InstalledAppFlow.from_client_secrets_file(cs, SCOPES)
    creds = flow.run_local_server(port=0)
    with open(tp, "w") as f:
        f.write(creds.to_json())
    logger.info("Wrote Drive token to %s", tp)
    return tp


class DriveUploader:
    """Google Drive client (OAuth installed-app, refresh-only; SA key fallback)."""

    def __init__(self, parent_folder_id: str, client_secrets: str = None,
                 token_path: str = None, sa_key_path: str = None):
        self.parent_folder_id = parent_folder_id
        self._folder_cache = {}
        self._video_cache = {}

        if sa_key_path and os.path.isfile(sa_key_path) and not client_secrets:
            from google.oauth2 import service_account
            build, _, _ = _lazy_imports()
            creds = service_account.Credentials.from_service_account_file(sa_key_path, scopes=SCOPES)
            self.service = build("drive", "v3", credentials=creds, cache_discovery=False)
            logger.info("Drive client initialized via service account (folder: %s)", parent_folder_id)
        else:
            cs = _resolve_client_secrets(client_secrets or "")
            if not cs:
                raise FileNotFoundError("OAuth client secrets (credentials.json/secrets.json) not found.")
            tp = _resolve_token_path(token_path or "", cs)
            self.service = _oauth_service(cs, tp)
            logger.info("Drive client initialized via OAuth (folder: %s)", parent_folder_id)

    # ── folders ───────────────────────────────────────────────────────────────
    def create_folder(self, name: str, parent_id: str = None) -> str:
        pid = parent_id or self.parent_folder_id
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [pid]}
        folder = self.service.files().create(body=meta, fields="id").execute()
        fid = folder["id"]
        logger.info("Created Drive folder '%s' -> %s", name, fid)
        return fid

    def find_folder(self, name: str, parent_id: str = None) -> Optional[str]:
        pid = parent_id or self.parent_folder_id
        query = (f"'{pid}' in parents and name='{name}' "
                 f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def ensure_folder(self, name: str, parent_id: str = None) -> str:
        return self.find_folder(name, parent_id) or self.create_folder(name, parent_id)

    def ensure_video_folders(self, batch_id: str, story_num: int) -> dict:
        """
        Ensure Batch_<id>/video_<n:04d>/<each media subfolder> exists (skeleton created up-front).
        Returns {kind: folder_id, "_video": video_folder_id}. Cached per (batch, story).
        """
        from shared import config
        key = f"{batch_id}/{story_num}"
        if key in self._video_cache:
            return self._video_cache[key]
        batch_folder = self.ensure_folder(f"Batch_{batch_id}", self.parent_folder_id)
        video_folder = self.ensure_folder(f"video_{int(story_num):04d}", batch_folder)
        subs = {"_video": video_folder}
        for kind in config.VIDEO_SUBFOLDERS:
            subs[kind] = self.ensure_folder(kind, video_folder)
        self._video_cache[key] = subs
        return subs

    def get_video_subfolder(self, batch_id: str, story_num: int, kind: str) -> str:
        """Folder id for a media kind (metadata/narration/images/music/subtitles/final)."""
        return self.ensure_video_folders(batch_id, story_num)[kind]

    def get_job_folder(self, batch_id: str, story_num: int, stage: str) -> str:
        """Stage JSONs live in the video's metadata/ folder (filename encodes the stage)."""
        return self.get_video_subfolder(batch_id, story_num, "metadata")

    # ── files ───────────────────────────────────────────────────────────────
    def find_file(self, name: str, parent_id: str = None, max_retries: int = 5) -> Optional[str]:
        """Find a (non-folder) file by name inside parent_id. Returns file id or None.
        Retries on transient network errors (SSL/timeout) — this is on the Drive-DB sync hot path
        (sync_pull/sync_push) which previously crashed the worker on a flaky uplink."""
        import time
        pid = parent_id or self.parent_folder_id
        query = f"'{pid}' in parents and name='{name}' and trashed=false"
        for attempt in range(1, max_retries + 1):
            try:
                results = self.service.files().list(q=query, fields="files(id, name)").execute()
                files = results.get("files", [])
                return files[0]["id"] if files else None
            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning("find_file '%s' attempt %d/%d failed: %s", name, attempt, max_retries, e)
                time.sleep(min(2 ** attempt, 12))

    def replace_or_upload(self, local_path: str, parent_id: str = None, name: str = None) -> Optional[str]:
        """Overwrite an existing same-named file's content (update), else create. Returns file id."""
        _, MediaFileUpload, _ = _lazy_imports()
        pid = parent_id or self.parent_folder_id
        name = name or os.path.basename(local_path)
        media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)
        existing = self.find_file(name, pid)
        if existing:
            self.service.files().update(fileId=existing, media_body=media).execute()
            logger.info("Replaced Drive file '%s' (%s)", name, existing)
            return existing
        meta = {"name": name, "parents": [pid]}
        f = self.service.files().create(body=meta, media_body=media, fields="id").execute()
        logger.info("Created Drive file '%s' (%s)", name, f["id"])
        return f["id"]

    @staticmethod
    def view_link(file_id: str) -> str:
        """Standard Drive 'view' URL for a file id (works for the owner; public if shared)."""
        return f"https://drive.google.com/file/d/{file_id}/view"

    def folder_link(self, folder_id: str) -> str:
        """Drive URL for a folder id."""
        return f"https://drive.google.com/drive/folders/{folder_id}"

    def make_public(self, file_id: str) -> str:
        """Grant 'anyone with the link can view' and return the view URL. Idempotent."""
        try:
            self.service.permissions().create(
                fileId=file_id, body={"role": "reader", "type": "anyone"},
            ).execute()
        except Exception as e:
            logger.warning("make_public failed for %s: %s", file_id, e)
        return self.view_link(file_id)

    def upload_file(self, local_path: str, folder_id: str, max_retries: int = 5) -> Optional[str]:
        """Upload a file with resumable upload + size verification. Returns file ID or None."""
        _, MediaFileUpload, _ = _lazy_imports()
        filename = os.path.basename(local_path)
        file_size = os.path.getsize(local_path)
        meta = {"name": filename, "parents": [folder_id]}

        for attempt in range(1, max_retries + 1):
            try:
                media = MediaFileUpload(
                    local_path, mimetype="application/octet-stream",
                    resumable=True, chunksize=5 * 1024 * 1024,
                )
                request = self.service.files().create(body=meta, media_body=media, fields="id,size")
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        logger.debug("Upload %s: %d%%", filename, int(status.progress() * 100))

                file_id = response["id"]
                remote_size = int(response.get("size", 0))
                if remote_size != file_size:
                    logger.warning("Size mismatch for %s: local=%d remote=%d (attempt %d/%d)",
                                   filename, file_size, remote_size, attempt, max_retries)
                    try:
                        self.service.files().delete(fileId=file_id).execute()
                    except Exception:
                        pass
                    continue
                logger.info("Uploaded %s -> %s (%d bytes)", filename, file_id, file_size)
                return file_id
            except Exception as e:
                logger.warning("Upload attempt %d/%d failed for %s: %s", attempt, max_retries, filename, e)
                import time
                time.sleep(min(2 ** attempt, 12))   # capped backoff (resilient to transient SSL/timeout)

        logger.error("Upload failed after %d attempts: %s", max_retries, local_path)
        return None

    def upload_and_verify(self, local_path: str, folder_id: str) -> Optional[str]:
        return self.upload_file(local_path, folder_id)

    def download_file(self, file_id: str, local_path: str, max_retries: int = 3) -> bool:
        """Download a file from Drive. Returns True on success."""
        _, _, MediaIoBaseDownload = _lazy_imports()
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        for attempt in range(1, max_retries + 1):
            try:
                request = self.service.files().get_media(fileId=file_id)
                with io.FileIO(local_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            logger.debug("Download %s: %d%%", file_id, int(status.progress() * 100))
                logger.info("Downloaded %s -> %s", file_id, local_path)
                return True
            except Exception as e:
                logger.warning("Download attempt %d/%d failed for %s: %s", attempt, max_retries, file_id, e)
                import time
                time.sleep(2 ** attempt)
        logger.error("Download failed after %d attempts: %s", max_retries, file_id)
        return False


def get_drive_client() -> Optional[DriveUploader]:
    """Build a DriveUploader (OAuth refresh-only; SA fallback) from config, or None if unconfigured."""
    from shared import config
    folder_id = config.DRIVE_PARENT_FOLDER_ID
    if not folder_id:
        logger.warning("DRIVE_PARENT_FOLDER_ID not set — uploads disabled")
        return None

    client_secrets = _resolve_client_secrets(config.DRIVE_CLIENT_SECRETS)
    sa_key = config.DRIVE_SA_KEY_PATH
    try:
        if client_secrets:
            token_path = _resolve_token_path(config.DRIVE_TOKEN_PATH, client_secrets)
            return DriveUploader(folder_id, client_secrets=client_secrets, token_path=token_path)
        if sa_key and os.path.isfile(sa_key):
            return DriveUploader(folder_id, sa_key_path=sa_key)
        logger.warning("No OAuth secrets (credentials.json) and no SA key — uploads disabled")
        return None
    except Exception as e:
        logger.warning("Drive init failed: %s — uploads disabled", e)
        return None

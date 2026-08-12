"""
Instagram adapter (instagrapi, unofficial). Posts a vertical mp4 as a Reel.

WARNING: instagrapi logs in as a real account and violates Instagram's ToS — use a burner/
non-critical account, keep posting human-paced (the uploader throttles to 1 / 3-4 h), and run from a
residential IP (the laptop), NEVER a datacenter (datacenter logins get challenged/blocked).

Session is persisted to IG_SETTINGS (default social/.ig_session.json) so we don't re-login every run.
Creds via env: IG_USERNAME, IG_PASSWORD. instagrapi is imported lazily so the rest of the system
(seed/app/dry-run) works without it installed.
"""

import os

IG_SETTINGS = os.environ.get("IG_SETTINGS", os.path.join(os.path.dirname(__file__), ".ig_session.json"))


class IGAdapter:
    def __init__(self, username=None, password=None, settings_path=IG_SETTINGS):
        self.username = username or os.environ.get("IG_USERNAME", "")
        self.password = password or os.environ.get("IG_PASSWORD", "")
        self.settings_path = settings_path
        self._cl = None

    def _client(self):
        if self._cl is not None:
            return self._cl
        from instagrapi import Client      # lazy
        cl = Client()
        cl.delay_range = [2, 5]            # gentle pacing between requests
        if os.path.exists(self.settings_path):
            try:
                cl.load_settings(self.settings_path)
            except Exception:
                pass
        if not self.username or not self.password:
            raise RuntimeError("IG_USERNAME / IG_PASSWORD not set.")
        cl.login(self.username, self.password)   # reuses session if settings loaded
        cl.dump_settings(self.settings_path)
        self._cl = cl
        return cl

    def upload_reel(self, video_path: str, caption: str) -> str:
        """Upload the mp4 as a Reel; return the post URL."""
        cl = self._client()
        media = cl.clip_upload(video_path, caption)
        code = getattr(media, "code", None)
        return f"https://www.instagram.com/reel/{code}/" if code else "posted"

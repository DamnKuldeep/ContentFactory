"""
Content Factory — Drive-hosted shared manifest (opt-in via DRIVE_DB=1).

Coarse "checkpoint" sync: workers/stages keep using the LOCAL SQLite at full speed; the
orchestrator (run.py) calls `sync_pull()` once at start and `sync_push(owned_stages)` after each
stage. Cross-machine safety comes from a brief Drive lock around the push plus a disjoint-stage
merge (each machine owns distinct stages, so upserting only its own stage rows never clobbers
another machine's). A best-effort Drive mutex (`drive_lock`) guards the push.

This matches the deployment topology (one machine per stage-group, sequential handoff) and keeps
the hot path free of per-job Drive I/O.
"""

import logging
import os
import random
import sqlite3
import time
from datetime import datetime, timezone

from shared import config
from shared.drive import get_drive_client
from shared.manifest import Manifest

logger = logging.getLogger("contentfactory.drive_db")


# ── Best-effort Drive mutex ───────────────────────────────────────────────────

class drive_lock:
    """
    Best-effort distributed lock backed by a single Drive file in the parent folder.

    Acquire: if no lock file exists, create one and win only if ours is the oldest (resolves
    concurrent-create races by createdTime, then id). If a lock exists and is older than `ttl`
    (crashed holder), steal it. Otherwise back off with jitter until `timeout`.
    Drive has no atomic compare-and-swap, so this is best-effort — correct for a small fleet /
    handoff use, where it's held only for a quick pull→merge→push.
    """

    def __init__(self, uploader, name=None, ttl=120, timeout=180, poll=2.0):
        self.uploader = uploader
        self.service = uploader.service
        self.parent = uploader.parent_folder_id
        self.name = name or config.DRIVE_LOCK_NAME
        self.ttl = ttl
        self.timeout = timeout
        self.poll = poll
        self._my_id = None

    def _list(self):
        q = f"'{self.parent}' in parents and name='{self.name}' and trashed=false"
        return self.service.files().list(q=q, fields="files(id, createdTime)").execute().get("files", [])

    def _create(self):
        f = self.service.files().create(
            body={"name": self.name, "parents": [self.parent]}, fields="id, createdTime"
        ).execute()
        return f["id"]

    def _delete(self, fid):
        try:
            self.service.files().delete(fileId=fid).execute()
        except Exception:
            pass

    @staticmethod
    def _age(created):
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - dt).total_seconds()
        except Exception:
            return 0.0

    def acquire(self):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            locks = self._list()
            if not locks:
                self._my_id = self._create()
                time.sleep(0.5 + random.uniform(0, 0.5))   # let concurrent creates settle
                locks = self._list()
                if locks:
                    winner = min(locks, key=lambda f: (f.get("createdTime", ""), f["id"]))
                    if winner["id"] == self._my_id:
                        return self
                # lost the race (or our create vanished) — drop ours and retry
                if self._my_id:
                    self._delete(self._my_id)
                    self._my_id = None
                time.sleep(self.poll + random.uniform(0, self.poll))
                continue
            winner = min(locks, key=lambda f: (f.get("createdTime", ""), f["id"]))
            if self._age(winner.get("createdTime", "")) > self.ttl:
                self._delete(winner["id"])          # steal stale lock
                continue
            time.sleep(self.poll + random.uniform(0, self.poll))
        raise TimeoutError(f"Could not acquire Drive lock '{self.name}' within {self.timeout}s")

    def release(self):
        if self._my_id:
            self._delete(self._my_id)
            self._my_id = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


# ── Drive-synced manifest ─────────────────────────────────────────────────────

def _clear_sidecars(db_path):
    for ext in ("-wal", "-shm"):
        try:
            os.remove(db_path + ext)
        except OSError:
            pass


class DriveSyncManifest(Manifest):
    """Local-speed Manifest + coarse Drive sync (sync_pull at start, sync_push per stage)."""

    def __init__(self, db_path, uploader, db_name=None, lock_name=None):
        super().__init__(db_path)
        self.uploader = uploader
        self.db_name = db_name or config.DRIVE_DB_NAME
        self.lock_name = lock_name or config.DRIVE_LOCK_NAME
        self.parent = uploader.parent_folder_id

    def sync_pull(self):
        """Download the shared Drive DB into the local file (overwrite). No-op if none yet."""
        fid = self.uploader.find_file(self.db_name, self.parent)
        if not fid:
            logger.info("No remote DB '%s' yet — starting from fresh local schema.", self.db_name)
            return
        _clear_sidecars(self.db_path)
        if self.uploader.download_file(fid, self.db_path):
            self._init_db()   # CREATE TABLE IF NOT EXISTS — harmless if already present
            logger.info("Pulled shared DB from Drive (%s)", fid)

    def sync_push(self, owned_stages=None):
        """
        Merge this machine's rows into the remote DB, under a Drive lock:
        `jobs` rows for `owned_stages` (disjoint per machine) + the whole `videos` catalog
        (written only by the compose machine). Pass owned_stages=None to push only videos
        (e.g. after --share-public updates links).
        """
        owned = [s for s in (owned_stages or []) if s]
        # Fold WAL into the single .sqlite file (safe: workers have joined before this is called).
        try:
            with self._conn() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass

        tmp = self.db_path + ".remote.tmp"
        # Retry the whole lock→merge→upload block on transient network errors (flaky uplink) — the
        # merge is idempotent, so re-running is safe. Without this a single SSL/timeout crashed the
        # worker mid-batch (the resilient driver relaunched, but progress could stall).
        for attempt in range(1, 6):
            try:
                with drive_lock(self.uploader, self.lock_name):
                    fid = self.uploader.find_file(self.db_name, self.parent)
                    if fid:
                        _clear_sidecars(tmp)
                        self.uploader.download_file(fid, tmp)
                    else:
                        Manifest(tmp)   # initialize schema in a fresh tmp DB
                    if owned:
                        self._merge_owned_rows_into(tmp, owned)
                    self._merge_videos_into(tmp)
                    self.uploader.replace_or_upload(tmp, self.parent, self.db_name)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                logger.warning("sync_push attempt %d/5 failed: %s — retrying", attempt, e)
                time.sleep(min(2 ** attempt, 12))
        for p in (tmp, tmp + "-wal", tmp + "-shm"):
            try:
                os.remove(p)
            except OSError:
                pass
        logger.info("Pushed shared DB to Drive (stages=%s, +videos)", sorted(owned))

    def _merge_owned_rows_into(self, tmp_path, owned_stages):
        """Upsert local rows for owned stages into tmp_path by UNIQUE(batch_id, stage, story_num)."""
        with self._conn() as lc:
            cols = [d[0] for d in lc.execute("SELECT * FROM jobs LIMIT 0").description]
            qmarks = ",".join("?" * len(owned_stages))
            rows = lc.execute(
                f"SELECT * FROM jobs WHERE stage IN ({qmarks})", tuple(owned_stages)
            ).fetchall()

        cols_no_id = [c for c in cols if c != "id"]   # let tmp assign fresh autoincrement ids
        collist = ",".join(cols_no_id)
        placeholders = ",".join("?" * len(cols_no_id))

        tc = sqlite3.connect(tmp_path, timeout=30)
        try:
            tc.execute("PRAGMA journal_mode=DELETE")   # single-file for clean upload
            tc.execute("PRAGMA busy_timeout=10000")
            for r in rows:
                tc.execute(
                    "DELETE FROM jobs WHERE batch_id=? AND stage=? AND story_num=?",
                    (r["batch_id"], r["stage"], r["story_num"]),
                )
                tc.execute(
                    f"INSERT INTO jobs ({collist}) VALUES ({placeholders})",
                    tuple(r[c] for c in cols_no_id),
                )
            tc.commit()
        finally:
            tc.close()

    def _merge_videos_into(self, tmp_path):
        """Upsert the local `videos` catalog into tmp_path by UNIQUE(batch_id, story_num)."""
        from shared.manifest import _SCHEMA
        with self._conn() as lc:
            try:
                cols = [d[0] for d in lc.execute("SELECT * FROM videos LIMIT 0").description]
                rows = lc.execute("SELECT * FROM videos").fetchall()
            except Exception:
                return
        if not rows:
            return
        collist = ",".join(cols)
        placeholders = ",".join("?" * len(cols))
        tc = sqlite3.connect(tmp_path, timeout=30)
        try:
            tc.execute("PRAGMA journal_mode=DELETE")
            tc.execute("PRAGMA busy_timeout=10000")
            tc.executescript(_SCHEMA)   # ensure `videos` exists in older remote DBs
            for r in rows:
                tc.execute("DELETE FROM videos WHERE batch_id=? AND story_num=?",
                           (r["batch_id"], r["story_num"]))
                tc.execute(f"INSERT INTO videos ({collist}) VALUES ({placeholders})",
                           tuple(r[c] for c in cols))
            tc.commit()
        finally:
            tc.close()


def get_manifest(db_path, uploader=None):
    """Factory: DriveSyncManifest if DRIVE_DB and a Drive client is available, else local Manifest."""
    if config.DRIVE_DB:
        up = uploader or get_drive_client()
        if up:
            return DriveSyncManifest(db_path, up)
        logger.warning("DRIVE_DB set but no Drive client available — falling back to local manifest.")
    return Manifest(db_path)

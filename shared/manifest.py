"""
Content Factory — SQLite manifest for crash-safe state management.

WAL mode, atomic claim, full ACID guarantees.
Stage-aware tracking for multi-stage pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger("contentfactory.manifest")

# Linear stage order for readiness (a stage is ready when its predecessor is COMPLETE).
STAGE_SEQUENCE = ["stage_01_story", "stage_02_narration", "stage_03_images",
                  "stage_04_music", "stage_05_compose"]
_PRED = {STAGE_SEQUENCE[i]: STAGE_SEQUENCE[i - 1] for i in range(1, len(STAGE_SEQUENCE))}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,
    story_num       INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED','UPLOAD_PENDING')),
    attempts        INTEGER DEFAULT 0,
    max_attempts    INTEGER DEFAULT 5,
    started_at      TEXT,
    completed_at    TEXT,
    last_error      TEXT,
    local_path      TEXT,
    drive_folder_id TEXT,
    drive_file_id   TEXT,
    meta            TEXT,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(batch_id, stage, story_num)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_stage ON jobs(status, stage);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);

CREATE TABLE IF NOT EXISTS rate_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    worker_id   TEXT,
    model       TEXT,
    event_type  TEXT CHECK(event_type IN ('call','rate_limit','error')),
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_rate_ts ON rate_events(ts);

CREATE TABLE IF NOT EXISTS videos (
    batch_id      TEXT NOT NULL,
    story_num     INTEGER NOT NULL,
    title         TEXT,
    link          TEXT,
    drive_file_id TEXT,
    folder_id     TEXT,
    public        INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(batch_id, story_num)
);
CREATE INDEX IF NOT EXISTS idx_videos_batch ON videos(batch_id);
"""


class Manifest:
    """Thread-safe, multi-process-safe SQLite manifest."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Batch & Stage management ──────────────────────────────────────────────

    def create_batch(self, batch_id: str, stage: str, story_nums: list[int], max_attempts: int = 5) -> int:
        """Insert PENDING jobs for a batch and stage. Returns number created."""
        created = 0
        with self._conn() as conn:
            for num in story_nums:
                try:
                    conn.execute(
                        "INSERT INTO jobs (batch_id, stage, story_num, status, max_attempts) VALUES (?, ?, ?, 'PENDING', ?)",
                        (batch_id, stage, num, max_attempts),
                    )
                    created += 1
                except sqlite3.IntegrityError:
                    pass  # Already exists (UNIQUE constraint)
        if created > 0:
            logger.info("Created %d jobs for batch %s, stage %s", created, batch_id, stage)
        return created

    def seed_all_stages(self, batch_id: str, story_nums: list[int], max_attempts: int = 5) -> int:
        """Pre-seed PENDING jobs for ALL pipeline stages × stories (replaces per-stage backlog)."""
        total = 0
        for stage in STAGE_SEQUENCE:
            total += self.create_batch(batch_id, stage, story_nums, max_attempts)
        return total

    def get_completed_story_nums(self, batch_id: str, stage: str) -> list[int]:
        """Get list of story_nums that are COMPLETE for a given batch and stage."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT story_num FROM jobs WHERE batch_id = ? AND stage = ? AND status = 'COMPLETE'",
                (batch_id, stage),
            ).fetchall()
            return [r["story_num"] for r in rows]

    # ── Videos registry (final catalog: title + link) ─────────────────────────

    def record_video(self, batch_id: str, story_num: int, title: str, link: str,
                     drive_file_id: str, folder_id: str = None, public: int = 0):
        """Upsert a finished video's title + link into the catalog (one row per video)."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO videos "
                "(batch_id, story_num, title, link, drive_file_id, folder_id, public) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (batch_id, story_num, title, link, drive_file_id, folder_id, int(public)),
            )
        logger.info("Recorded video: %s | story %s | %s", title, story_num, link)

    def list_videos(self, batch_id: str = None) -> list[dict]:
        """Return the videos catalog (optionally filtered to a batch), ordered by story_num."""
        with self._conn() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT * FROM videos WHERE batch_id = ? ORDER BY story_num", (batch_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM videos ORDER BY batch_id, story_num").fetchall()
            return [dict(r) for r in rows]

    # ── Job claim (atomic) ────────────────────────────────────────────────────

    def _try_claim(self, select_sql, params) -> dict | None:
        """Atomically claim the row returned by select_sql. Retries on lost races (multi-process)."""
        now = datetime.now(timezone.utc).isoformat()
        for _ in range(200):
            with self._conn() as conn:          # fresh transaction each attempt → sees others' commits
                row = conn.execute(select_sql, params).fetchone()
                if row is None:
                    return None
                job_id = row["id"]
                cur = conn.execute(
                    "UPDATE jobs SET status='RUNNING', attempts=attempts+1, started_at=?, "
                    "last_error=NULL WHERE id=? AND status='PENDING'",
                    (now, job_id),
                )
                if cur.rowcount == 1:           # we won; anyone else gets rowcount 0 and retries
                    return dict(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        return None

    def claim_job(self, batch_id: str, stage: str) -> dict | None:
        """Atomically claim one PENDING job for a specific batch+stage."""
        return self._try_claim(
            "SELECT id FROM jobs WHERE batch_id=? AND stage=? AND status='PENDING' "
            "ORDER BY story_num LIMIT 1",
            (batch_id, stage),
        )

    def claim_ready_job(self, stage: str) -> dict | None:
        """
        Atomically claim one PENDING `stage` job (ANY batch) whose linear predecessor is COMPLETE
        for the same (batch, story). Powers the simple `worker <role>` loop — no batch IDs needed.
        """
        pred = _PRED.get(stage)
        if pred is None:   # stage 1 has no predecessor
            return self._try_claim(
                "SELECT id FROM jobs WHERE stage=? AND status='PENDING' "
                "ORDER BY batch_id, story_num LIMIT 1",
                (stage,),
            )
        return self._try_claim(
            "SELECT j.id FROM jobs j WHERE j.stage=? AND j.status='PENDING' AND EXISTS("
            "  SELECT 1 FROM jobs p WHERE p.batch_id=j.batch_id AND p.story_num=j.story_num "
            "  AND p.stage=? AND p.status='COMPLETE') ORDER BY j.batch_id, j.story_num LIMIT 1",
            (stage, pred),
        )

    # ── Job completion ────────────────────────────────────────────────────────

    def complete_job(self, job_id: int, local_path: str = None, meta: dict = None):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'COMPLETE', completed_at = ?, local_path = ?, meta = ? WHERE id = ?",
                (now, local_path, json.dumps(meta) if meta else None, job_id),
            )

    def fail_job(self, job_id: int, error: str):
        """Mark FAILED if max attempts reached, else back to PENDING for retry."""
        with self._conn() as conn:
            row = conn.execute("SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row and row["attempts"] >= row["max_attempts"]:
                conn.execute(
                    "UPDATE jobs SET status = 'FAILED', last_error = ? WHERE id = ?",
                    (error[:4000], job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = 'PENDING', last_error = ? WHERE id = ?",
                    (error[:4000], job_id),
                )

    def mark_upload_pending(self, job_id: int, local_path: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'UPLOAD_PENDING', local_path = ? WHERE id = ?",
                (local_path, job_id),
            )

    def upload_complete(self, job_id: int, drive_folder_id: str, drive_file_id: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'COMPLETE', completed_at = ?, drive_folder_id = ?, drive_file_id = ? WHERE id = ?",
                (now, drive_folder_id, drive_file_id, job_id),
            )

    # ── Stale job recovery ────────────────────────────────────────────────────

    def reset_stale(self, timeout_minutes: int = 30) -> int:
        """Reset RUNNING jobs older than timeout back to PENDING."""
        with self._conn() as conn:
            result = conn.execute(
                """UPDATE jobs SET status = 'PENDING', last_error = 'stale: reset after timeout'
                   WHERE status = 'RUNNING'
                   AND started_at IS NOT NULL
                   AND (julianday('now') - julianday(started_at)) * 24 * 60 > ?""",
                (timeout_minutes,),
            )
            count = result.rowcount
            if count > 0:
                logger.warning("Reset %d stale RUNNING jobs (>%d min)", count, timeout_minutes)
            return count

    def retry_failed(self, batch_id: str, stage: str = None) -> int:
        """Reset FAILED jobs back to PENDING with attempts reset to 0."""
        with self._conn() as conn:
            if stage:
                result = conn.execute(
                    "UPDATE jobs SET status = 'PENDING', attempts = 0, last_error = NULL WHERE batch_id = ? AND stage = ? AND status = 'FAILED'",
                    (batch_id, stage),
                )
            else:
                result = conn.execute(
                    "UPDATE jobs SET status = 'PENDING', attempts = 0, last_error = NULL WHERE batch_id = ? AND status = 'FAILED'",
                    (batch_id,),
                )
            count = result.rowcount
            if count > 0:
                logger.info("Reset %d FAILED jobs back to PENDING", count)
            return count

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_stats(self, batch_id: str, stage: str = None) -> dict:
        with self._conn() as conn:
            if stage:
                rows = conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM jobs WHERE batch_id = ? AND stage = ? GROUP BY status",
                    (batch_id, stage),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM jobs WHERE batch_id = ? GROUP BY status",
                    (batch_id,),
                ).fetchall()
                
            stats = {r["status"]: r["cnt"] for r in rows}
            stats["total"] = sum(stats.values())
            return stats

    def get_failed(self, batch_id: str, stage: str = None) -> list:
        with self._conn() as conn:
            if stage:
                return [dict(r) for r in conn.execute(
                    "SELECT id, story_num, stage, attempts, last_error FROM jobs WHERE batch_id = ? AND stage = ? AND status = 'FAILED'",
                    (batch_id, stage),
                ).fetchall()]
            else:
                return [dict(r) for r in conn.execute(
                    "SELECT id, story_num, stage, attempts, last_error FROM jobs WHERE batch_id = ? AND status = 'FAILED'",
                    (batch_id,),
                ).fetchall()]

    def get_upload_pending(self, batch_id: str, stage: str = None) -> list:
        with self._conn() as conn:
            if stage:
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM jobs WHERE batch_id = ? AND stage = ? AND status = 'UPLOAD_PENDING'",
                    (batch_id, stage),
                ).fetchall()]
            else:
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM jobs WHERE batch_id = ? AND status = 'UPLOAD_PENDING'",
                    (batch_id,),
                ).fetchall()]

    def get_all_complete(self, batch_id: str, stage: str = None) -> list:
        with self._conn() as conn:
            if stage:
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM jobs WHERE batch_id = ? AND stage = ? AND status = 'COMPLETE'",
                    (batch_id, stage),
                ).fetchall()]
            else:
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM jobs WHERE batch_id = ? AND status = 'COMPLETE'",
                    (batch_id,),
                ).fetchall()]

    def get_job_meta(self, batch_id: str, stage: str, story_num: int) -> dict | None:
        """Fetch the meta dict of a completed job."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT meta FROM jobs WHERE batch_id = ? AND stage = ? AND story_num = ? AND status = 'COMPLETE'",
                (batch_id, stage, story_num)
            ).fetchone()
            if row and row["meta"]:
                return json.loads(row["meta"])
            return None

    def get_completed_job(self, batch_id: str, stage: str, story_num: int) -> dict | None:
        """Fetch the full record of a completed job."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE batch_id = ? AND stage = ? AND story_num = ? AND status = 'COMPLETE'",
                (batch_id, stage, story_num)
            ).fetchone()
            return dict(row) if row else None

    def has_pending_work(self, batch_id: str, stage: str = None) -> bool:
        with self._conn() as conn:
            if stage:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE batch_id = ? AND stage = ? AND status IN ('PENDING', 'RUNNING', 'UPLOAD_PENDING')",
                    (batch_id, stage),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE batch_id = ? AND status IN ('PENDING', 'RUNNING', 'UPLOAD_PENDING')",
                    (batch_id,),
                ).fetchone()
            return row[0] > 0

    # ── Rate event logging ────────────────────────────────────────────────────

    def log_rate_event(self, worker_id: str, model: str, event_type: str, detail: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO rate_events (ts, worker_id, model, event_type, detail) VALUES (?, ?, ?, ?, ?)",
                (time.time(), worker_id, model, event_type, detail),
            )

    def recent_rate_limits(self, window_seconds: int = 60) -> int:
        with self._conn() as conn:
            cutoff = time.time() - window_seconds
            row = conn.execute(
                "SELECT COUNT(*) FROM rate_events WHERE event_type = 'rate_limit' AND ts > ?",
                (cutoff,),
            ).fetchone()
            return row[0]

"""
Content Factory — Cross-process rate limiting via SQLite.

Prevents all workers from simultaneously hammering the API if we hit a hard 429.
"""

import logging
import time

from .manifest import Manifest

logger = logging.getLogger("contentfactory.ratelimit")


class RateLimiter:
    def __init__(self, manifest: Manifest, max_429_per_minute: int = 5):
        self.manifest = manifest
        self.max_429 = max_429_per_minute

    def check_and_wait(self, worker_id: str):
        """Check if the global 429 threshold is exceeded. If so, sleep."""
        # Check how many 429 events all workers logged in the last 90 seconds
        recent_429s = self.manifest.recent_rate_limits(window_seconds=90)
        
        if recent_429s >= self.max_429:
            sleep_time = 90
            logger.warning(
                "Worker %s pausing for %ds — %d rate limits across all workers in last minute",
                worker_id, sleep_time, recent_429s
            )
            time.sleep(sleep_time)

    def log_429(self, worker_id: str, model: str, detail: str = ""):
        """Log a 429 event so other workers can see it."""
        self.manifest.log_rate_event(worker_id, model, "rate_limit", detail)
        logger.info("Worker %s logged 429 for %s", worker_id, model)

"""
Tests for the job state machine — the part of the system that must not be wrong.

These cover the scheduler's actual guarantees: a stage cannot start before its predecessor
finished, a job cannot be claimed twice, and failures are retried a bounded number of times.
Pure stdlib + SQLite, no GPU, no network — runnable anywhere.

    python -m pytest tests/ -q
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.manifest import Manifest, STAGE_SEQUENCE

S1, S2, S3 = STAGE_SEQUENCE[0], STAGE_SEQUENCE[1], STAGE_SEQUENCE[2]
BATCH = "20260101_000000"


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.m = Manifest(os.path.join(self._dir.name, "manifest.sqlite"))

    def tearDown(self):
        self._dir.cleanup()


class TestSeeding(ManifestTestCase):
    def test_seed_all_stages_creates_one_job_per_stage_per_story(self):
        created = self.m.seed_all_stages(BATCH, [1, 2, 3])
        self.assertEqual(created, 3 * len(STAGE_SEQUENCE))
        self.assertEqual(self.m.get_stats(BATCH)["PENDING"], 15)

    def test_seeding_is_idempotent(self):
        self.m.seed_all_stages(BATCH, [1, 2])
        again = self.m.seed_all_stages(BATCH, [1, 2])
        self.assertEqual(again, 0, "re-seeding must not duplicate jobs (UNIQUE constraint)")


class TestClaiming(ManifestTestCase):
    def test_a_job_is_never_claimed_twice(self):
        self.m.create_batch(BATCH, S1, [1])
        first = self.m.claim_job(BATCH, S1)
        second = self.m.claim_job(BATCH, S1)
        self.assertIsNotNone(first)
        self.assertIsNone(second, "the atomic claim must hand a job to exactly one worker")

    def test_claiming_marks_running_and_counts_the_attempt(self):
        self.m.create_batch(BATCH, S1, [1])
        job = self.m.claim_job(BATCH, S1)
        self.assertEqual(job["status"], "RUNNING")
        self.assertEqual(job["attempts"], 1)
        self.assertIsNotNone(job["started_at"])

    def test_workers_claim_distinct_stories_in_order(self):
        self.m.create_batch(BATCH, S1, [1, 2, 3])
        claimed = [self.m.claim_job(BATCH, S1)["story_num"] for _ in range(3)]
        self.assertEqual(claimed, [1, 2, 3])
        self.assertIsNone(self.m.claim_job(BATCH, S1))


class TestReadiness(ManifestTestCase):
    """claim_ready_job is the entire scheduler for the bulk `worker.py <role>` path."""

    def test_stage_2_is_not_claimable_until_stage_1_completes(self):
        self.m.seed_all_stages(BATCH, [1])
        self.assertIsNone(self.m.claim_ready_job(S2),
                          "stage 2 must not start before stage 1 is COMPLETE")

        job1 = self.m.claim_ready_job(S1)
        self.assertIsNotNone(job1, "stage 1 has no predecessor and is immediately ready")
        self.assertIsNone(self.m.claim_ready_job(S2), "RUNNING upstream is not COMPLETE")

        self.m.complete_job(job1["id"])
        job2 = self.m.claim_ready_job(S2)
        self.assertIsNotNone(job2)
        self.assertEqual(job2["story_num"], 1)

    def test_readiness_is_per_story_not_per_batch(self):
        self.m.seed_all_stages(BATCH, [1, 2])
        a = self.m.claim_ready_job(S1)
        self.m.complete_job(a["id"])                     # only story 1 is done

        ready = self.m.claim_ready_job(S2)
        self.assertEqual(ready["story_num"], a["story_num"])
        self.assertIsNone(self.m.claim_ready_job(S2),
                          "story 2's stage 2 must stay blocked while its stage 1 is unfinished")

    def test_a_failed_upstream_never_unblocks_downstream(self):
        """This is what let the 100-video run skip its 12 spend-limited stories cleanly."""
        self.m.seed_all_stages(BATCH, [1])
        job = self.m.claim_ready_job(S1)
        for _ in range(10):
            self.m.fail_job(job["id"], "403 spend limit")
            nxt = self.m.claim_ready_job(S1)
            if nxt is None:
                break
            job = nxt
        self.assertEqual(self.m.get_stats(BATCH, S1).get("FAILED"), 1)
        self.assertIsNone(self.m.claim_ready_job(S2))
        self.assertIsNone(self.m.claim_ready_job(S3))


class TestFailureHandling(ManifestTestCase):
    def test_failure_returns_to_pending_until_max_attempts(self):
        self.m.create_batch(BATCH, S1, [1], max_attempts=3)
        for expected_attempt in (1, 2):
            job = self.m.claim_job(BATCH, S1)
            self.assertEqual(job["attempts"], expected_attempt)
            self.m.fail_job(job["id"], "transient")
            self.assertEqual(self.m.get_stats(BATCH, S1).get("PENDING"), 1)

        job = self.m.claim_job(BATCH, S1)          # third and final attempt
        self.m.fail_job(job["id"], "permanent")
        self.assertEqual(self.m.get_stats(BATCH, S1).get("FAILED"), 1)
        self.assertIsNone(self.m.claim_job(BATCH, S1))

    def test_retry_failed_requeues_and_resets_attempts(self):
        self.m.create_batch(BATCH, S1, [1], max_attempts=1)
        self.m.fail_job(self.m.claim_job(BATCH, S1)["id"], "boom")
        self.assertEqual(self.m.retry_failed(BATCH, S1), 1)

        job = self.m.claim_job(BATCH, S1)
        self.assertIsNotNone(job)
        self.assertEqual(job["attempts"], 1, "attempts reset to 0, so this claim is attempt 1 again")

    def test_error_text_is_truncated_not_rejected(self):
        self.m.create_batch(BATCH, S1, [1], max_attempts=1)
        self.m.fail_job(self.m.claim_job(BATCH, S1)["id"], "x" * 10_000)
        failed = self.m.get_failed(BATCH, S1)
        self.assertEqual(len(failed), 1)
        self.assertLessEqual(len(failed[0]["last_error"]), 4000)


class TestStaleRecovery(ManifestTestCase):
    def test_reset_stale_requeues_jobs_from_a_dead_process(self):
        self.m.create_batch(BATCH, S1, [1])
        self.m.claim_job(BATCH, S1)                       # worker "dies" here, still RUNNING
        self.assertEqual(self.m.get_stats(BATCH, S1).get("RUNNING"), 1)

        self.assertEqual(self.m.reset_stale(timeout_minutes=60), 0, "a fresh job is not stale")
        self.assertEqual(self.m.reset_stale(timeout_minutes=0), 1)
        self.assertEqual(self.m.get_stats(BATCH, S1).get("PENDING"), 1)


class TestUploadHandoff(ManifestTestCase):
    def test_upload_pending_then_complete_records_drive_ids(self):
        self.m.create_batch(BATCH, S1, [1])
        job = self.m.claim_job(BATCH, S1)
        self.m.mark_upload_pending(job["id"], "/tmp/out.json")
        self.assertEqual(self.m.get_stats(BATCH, S1).get("UPLOAD_PENDING"), 1)
        self.assertTrue(self.m.has_pending_work(BATCH, S1),
                        "an un-uploaded job still counts as outstanding work")

        self.m.upload_complete(job["id"], "folder123", "file456")
        done = self.m.get_completed_job(BATCH, S1, 1)
        self.assertEqual(done["drive_file_id"], "file456")
        self.assertEqual(done["drive_folder_id"], "folder123")
        self.assertFalse(self.m.has_pending_work(BATCH, S1))


class TestVideoCatalog(ManifestTestCase):
    def test_record_video_upserts_by_batch_and_story(self):
        self.m.record_video(BATCH, 1, "First title", "http://a", "fid1")
        self.m.record_video(BATCH, 1, "Corrected title", "http://b", "fid1", public=1)
        rows = self.m.list_videos(BATCH)
        self.assertEqual(len(rows), 1, "the same video must not appear twice")
        self.assertEqual(rows[0]["title"], "Corrected title")
        self.assertEqual(rows[0]["public"], 1)

    def test_list_videos_orders_by_story_num(self):
        for n in (3, 1, 2):
            self.m.record_video(BATCH, n, f"t{n}", f"http://{n}", f"fid{n}")
        self.assertEqual([r["story_num"] for r in self.m.list_videos(BATCH)], [1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)

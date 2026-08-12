"""
Content Factory — lightweight, env-gated step profiler.

Enable by setting PROFILE_LOG to a JSONL path; otherwise every call is a no-op (zero behavior
change in production). Each `step(...)` records one line:
    {"ts", "stage", "step", "kind", "seconds", ...extra}
`kind` ∈ {load, infer, io, api, cpu} so the report can separate model-load vs GPU-inference vs
Drive-I/O vs API vs CPU. For GPU steps pass sync=True to torch.cuda.synchronize() around the block
so async CUDA work is captured in the measured time.

Usage:
    from shared.timing import step, stage_timer
    with step("stage_02_narration", "fish_load", kind="load"):
        ...
    with step("stage_02_narration", "tts_generate", kind="infer", sync=True, chunks=10):
        ...
"""

import json
import os
import time
from contextlib import contextmanager

_LOG = os.environ.get("PROFILE_LOG", "").strip()


def enabled() -> bool:
    return bool(_LOG)


def _emit(record: dict):
    if not _LOG:
        return
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _maybe_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def step(stage: str, name: str, kind: str = "infer", sync: bool = False, **extra):
    """Time a block and append a JSONL record (no-op if PROFILE_LOG unset)."""
    if not _LOG:
        yield
        return
    if sync:
        _maybe_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            _maybe_sync()
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
               "step": name, "kind": kind, "seconds": round(time.perf_counter() - t0, 3)}
        rec.update(extra)
        _emit(rec)


@contextmanager
def stage_timer(stage: str, **extra):
    """Record the whole-stage wall clock as step='stage_total' (kind='total')."""
    if not _LOG:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
               "step": "stage_total", "kind": "total",
               "seconds": round(time.perf_counter() - t0, 3)}
        rec.update(extra)
        _emit(rec)


def mark(stage: str, name: str, seconds: float, kind: str = "cpu", **extra):
    """Record a pre-measured duration (e.g. one-time setup/download markers)."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
           "step": name, "kind": kind, "seconds": round(float(seconds), 3)}
    rec.update(extra)
    _emit(rec)

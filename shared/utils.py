"""
Shared utilities — slugify, path guards, dependency checks.
"""

import os
import re


def slugify(text: str, maxlen: int = 60) -> str:
    """Slugify text for filenames. MUST match the renderer's _slugify."""
    out = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return out[:maxlen].strip("-") or "reel"


def story_dir(base: str, batch_id: str, stage: str, story_num: int) -> str:
    """Canonical local directory for a story's artifacts within a stage."""
    return os.path.join(base, batch_id, stage, f"story_{story_num:03d}")


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def require_upstream(path: str, stage_name: str, what: str = "artifact"):
    """Hard-refuse to proceed if an upstream artifact is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{stage_name}] Required upstream {what} not found: {path}\n"
            f"Run the previous stage first to produce this artifact."
        )
    return path


def require_env(name: str, default: str = "") -> str:
    """Get an environment variable or raise with a clear message."""
    val = os.environ.get(name, default)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set.\n"
            f"Set it in .env or export it before running."
        )
    return val


def unique_output_path(out_dir: str, data: dict = None, prefix: str = "output", ext: str = ".json") -> str:
    """
    Build a unique output file path under out_dir.

    Used by the worker loop to stage a stage's JSON result before upload. Derives an
    optional slug from a title-ish field in `data`, then appends a numeric suffix if a
    file already exists so concurrent/repeat runs don't clobber each other.
    """
    ensure_dir(out_dir)
    base = prefix
    if isinstance(data, dict):
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        premise = meta.get("premise", {}) if isinstance(meta.get("premise"), dict) else {}
        title = premise.get("logline") or data.get("title") or ""
        if title:
            base = f"{prefix}-{slugify(title, 40)}"

    path = os.path.join(out_dir, base + ext)
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        cand = os.path.join(out_dir, f"{base}-{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

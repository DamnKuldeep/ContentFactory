"""
Shared logging — structured per-stage, per-worker file + console logging.
"""

import logging
import os
import sys


def setup_logger(
    name: str,
    log_dir: str = "./logs",
    worker_id: str = "",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that writes to both a file and stderr.

    Parameters
    ----------
    name : str
        Logger name (usually stage name like 'stage_02_narration').
    log_dir : str
        Directory for log files.
    worker_id : str
        Optional worker ID prefix for multi-process runs.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    prefix = f"[{worker_id}] " if worker_id else ""
    fmt = f"[%(asctime)s] {prefix}[%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on re-init
    if not logger.handlers:
        # File handler — one file per stage per run
        fname = f"{name}.log" if not worker_id else f"{name}_{worker_id}.log"
        fh = logging.FileHandler(os.path.join(log_dir, fname), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler — stderr (stays visible but doesn't pollute stdout)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.WARNING)  # Only warnings+ to console
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

#!/usr/bin/env python3
"""
setup_compose.py — provision a Stage-5 (compose) node. Needs system ffmpeg; no model downloads.

Creates .venv_compose and installs the stage packages (Pillow + Drive). Install ffmpeg separately
(e.g. `sudo apt-get install -y ffmpeg`). Run once:

    python ContentFactory/scripts/setup_compose.py

Then:
    source .venv_compose/bin/activate
    python run.py --stage 5 --batch <id> --drive-db
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _setup_common import BASE_DEPS, done, log, make_venv, pip_install, stage_requirements


def main():
    vpy, venv_dir = make_venv(".venv_compose")
    pip_install(vpy, packages=BASE_DEPS + ["pillow"],
                requirements=stage_requirements("stage_05_compose"))
    if shutil.which("ffmpeg") is None:
        log("[warn] ffmpeg not found on PATH — install it (e.g. `sudo apt-get install -y ffmpeg`).")
    else:
        log(f"ffmpeg found: {shutil.which('ffmpeg')}")
    done("compose (stage 5)", venv_dir)


if __name__ == "__main__":
    main()

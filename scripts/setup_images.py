#!/usr/bin/env python3
"""
setup_images.py — provision a Stage-3 (image generation) node: FLUX.2-klein.

Creates .venv_images, installs the stage packages, and downloads the FLUX.2 weights into the
repo-local HF cache. Run once on the image machine:

    python ContentFactory/scripts/setup_images.py

Then:
    source .venv_images/bin/activate
    python run.py --stage 3 --batch <id> --drive-db
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _setup_common import (BASE_DEPS, done, log, make_venv, pip_install,
                           run_py, stage_requirements)


def main():
    vpy, venv_dir = make_venv(".venv_images")
    pip_install(vpy, packages=BASE_DEPS, requirements=stage_requirements("stage_03_images"))

    # Trigger the FLUX.2 weight download into the repo-local HF cache.
    log("downloading FLUX.2-klein weights (~tens of GB, first time only)...")
    run_py(vpy, (
        "import sys; sys.path.insert(0,'.')\n"
        "try:\n"
        "    from stages.stage_03_images import generate\n"
        "    generate._get_pipeline()\n"
        "    print('[setup] FLUX.2 pipeline loaded OK')\n"
        "except Exception as e:\n"
        "    print('[setup][warn] pipeline load skipped/failed (weights may still be cached):', e)\n"
    ))
    done("images (stage 3)", venv_dir)


if __name__ == "__main__":
    main()

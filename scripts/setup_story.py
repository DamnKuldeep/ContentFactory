#!/usr/bin/env python3
"""
setup_story.py — provision a Stage-1 (story) node. CPU-only; no model downloads.

Creates .venv_story and installs the stage packages (OpenRouter LLM client + Drive). Run once:

    python ContentFactory/scripts/setup_story.py

Then:
    source .venv_story/bin/activate
    python run.py --stage 1 --count N --drive-db    # needs OPENROUTER_API_KEY in .env
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _setup_common import BASE_DEPS, done, make_venv, pip_install, stage_requirements


def main():
    vpy, venv_dir = make_venv(".venv_story")
    pip_install(vpy, packages=BASE_DEPS, requirements=stage_requirements("stage_01_story"))
    done("story (stage 1)", venv_dir)


if __name__ == "__main__":
    main()

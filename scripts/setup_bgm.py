#!/usr/bin/env python3
"""
setup_bgm.py — provision a Stage-4 (BGM) node: Qwen2.5-Omni brief + ACE-Step 1.5.

Creates .venv_bgm, installs the stage packages, downloads the Qwen brief model into the repo-local
HF cache, and validates the ACE-Step handlers (ACE repo + checkpoints must be provisioned via
ACE_REPO_DIR / ACE_CHECKPOINTS — those are large and supplied out-of-band). Run once:

    python ContentFactory/scripts/setup_bgm.py

Then:
    source .venv_bgm/bin/activate
    python run.py --stage 4 --batch <id> --drive-db
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _setup_common import (BASE_DEPS, done, log, make_venv, pip_install,
                           run_py, stage_requirements)

BGM_EXTRAS = ["librosa", "qwen-omni-utils"]   # librosa for energy envelope; qwen-omni-utils for Qwen-Omni


def main():
    vpy, venv_dir = make_venv(".venv_bgm")
    pip_install(vpy, packages=BASE_DEPS, requirements=stage_requirements("stage_04_music"))
    pip_install(vpy, packages=BGM_EXTRAS)

    log("downloading Qwen2.5-Omni brief model + validating ACE-Step...")
    run_py(vpy, (
        "import sys; sys.path.insert(0,'.')\n"
        "from shared import config\n"
        "try:\n"
        "    from stages.stage_04_music import brief\n"
        "    brief._get_qwen(); print('[setup] Qwen2.5-Omni loaded OK')\n"
        "except Exception as e:\n"
        "    print('[setup][warn] Qwen load skipped/failed (cache may be partial):', e)\n"
        "import os\n"
        "if os.path.isdir(config.ACE_REPO_DIR) and os.path.isdir(config.ACE_CHECKPOINTS):\n"
        "    try:\n"
        "        from stages.stage_04_music import ace; ace._get_ace(); print('[setup] ACE-Step handlers OK')\n"
        "    except Exception as e:\n"
        "        print('[setup][warn] ACE-Step init failed:', e)\n"
        "else:\n"
        "    print(f'[setup][warn] ACE repo/checkpoints missing: {config.ACE_REPO_DIR} / {config.ACE_CHECKPOINTS} '\n"
        "          '— provision them, then set ACE_REPO_DIR/ACE_CHECKPOINTS in .env')\n"
    ))
    done("bgm (stage 4)", venv_dir)


if __name__ == "__main__":
    main()

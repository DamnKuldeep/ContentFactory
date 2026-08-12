#!/usr/bin/env python3
"""
setup_narration.py — provision a Stage-2 (narration) node: Fish Speech S2 Pro + WhisperX.

Creates .venv_narration, installs all packages into it, downloads the Fish S2 Pro checkpoints +
WhisperX models into the repo-local HF cache, and re-applies the two required Fish patches.
Run once on the narration machine (with any python3):

    python ContentFactory/scripts/setup_narration.py

Then run the stage with that venv:
    source .venv_narration/bin/activate
    python run.py --stage 2 --batch <id> --drive-db
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _setup_common import (BASE_DEPS, CF_ROOT, HF_CACHE, done, log, make_venv,
                           pip_install, run_py, stage_requirements)

FISH_EXTRAS = [
    "torch", "torchaudio", "bitsandbytes", "whisperx", "hydra-core",
    "descript-audiotools", "descript-audio-codec", "resampy", "soundfile",
    "gitpython", "protobuf>=4.25.8",
]


def _patch_fish_repo():
    """Idempotently re-apply the two Fish patches (needed after a fresh clone/download)."""
    # 1) checkpoint tokenizer_config.json: TokenizersBackend -> PreTrainedTokenizerFast
    tcfg = os.path.join(CF_ROOT, "models", "fish_s2_pro_ckpt", "tokenizer_config.json")
    if os.path.exists(tcfg):
        d = json.load(open(tcfg))
        if d.get("tokenizer_class") != "PreTrainedTokenizerFast":
            d["tokenizer_class"] = "PreTrainedTokenizerFast"
            json.dump(d, open(tcfg, "w"), indent=2)
            log("patched tokenizer_config.json -> PreTrainedTokenizerFast")
        else:
            log("tokenizer_config.json already patched")
    else:
        log(f"[warn] {tcfg} not found yet (download may have failed)")

    # 2) llama.py: ensure `tokenizer = None` before the try that loads FishTokenizer
    llama = os.path.join(CF_ROOT, "models", "fish-speech-int4-patch",
                         "fish_speech", "models", "text2semantic", "llama.py")
    if os.path.exists(llama):
        src = open(llama).read()
        if "tokenizer = None\n        try:\n            tokenizer = FishTokenizer.from_pretrained" not in src:
            patched = src.replace(
                "        try:\n            tokenizer = FishTokenizer.from_pretrained(path)",
                "        tokenizer = None\n        try:\n            tokenizer = FishTokenizer.from_pretrained(path)",
                1,
            )
            if patched != src:
                open(llama, "w").write(patched)
                log("patched llama.py (tokenizer = None guard)")
            else:
                log("[warn] llama.py guard anchor not found — verify manually")
        else:
            log("llama.py already patched")
    else:
        log(f"[warn] {llama} not found yet")


def main():
    vpy, venv_dir = make_venv(".venv_narration")
    pip_install(vpy, packages=BASE_DEPS, requirements=stage_requirements("stage_02_narration"))
    pip_install(vpy, packages=FISH_EXTRAS)

    # Clone Fish repo + download S2 Pro checkpoints into the repo (uses HF cache for any HF pulls).
    log("downloading Fish S2 Pro checkpoints + cloning repo (this can take a while)...")
    run_py(vpy, "import sys; sys.path.insert(0,'scripts'); "
                "import fish_narration; fish_narration.ensure_setup()")

    _patch_fish_repo()

    # Validate-load on GPU (warn, don't fail, if no GPU at setup time).
    log("validating model load (Fish + WhisperX)...")
    run_py(vpy, (
        "import sys; sys.path.insert(0,'.')\n"
        "try:\n"
        "    from stages.stage_02_narration import fish\n"
        "    m,d,codec = fish._get_fish(); fish._get_prompt_tokens(codec); fish._get_whisperx()\n"
        "    print('[setup] Fish + WhisperX loaded OK')\n"
        "except Exception as e:\n"
        "    print('[setup][warn] model load skipped/failed (downloads are cached):', e)\n"
    ))
    done("narration (stage 2)", venv_dir)


if __name__ == "__main__":
    main()

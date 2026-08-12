#!/usr/bin/env bash
# Smoke gate: full 1-video chain with ALL flags ON, isolated in the bench Drive folder.
# Validates (a) the worker.py + Drive handoff end-to-end, (b) torch.compile builds for Fish & FLUX,
# (c) prompt caching emits cached_input_tokens. ~25 min. Run BEFORE the full A/B.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.conda/bin/python
BENCH_FOLDER="$(cat bench/bench_folder_id.txt)"

export DRIVE_DB=1 DRIVE_PARENT_FOLDER_ID="$BENCH_FOLDER"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"     # populated cache (avoid 44 GB re-download)
export FISH_COMPILE=1 FLUX_COMPILE=1 LLM_PROMPT_CACHE=1
export DB_PATH=bench/smoke.sqlite OUTPUT_DIR=bench/out_smoke PROFILE_LOG=bench/profile_smoke.jsonl
export DRIVE_DB_NAME=manifest_smoke.sqlite DRIVE_LOCK_NAME=manifest_smoke.lock
rm -f "$PROFILE_LOG"; mkdir -p "$OUTPUT_DIR"

echo "[smoke] folder=$BENCH_FOLDER  flags: compile+cache ON"
$PY scripts/produce.py --count 1 --workers 1 2>&1 | tee bench/smoke_story.log
$PY scripts/worker.py narration 2>&1 | tee bench/smoke_narr.log
$PY scripts/worker.py images    2>&1 | tee bench/smoke_img.log
$PY scripts/worker.py music     2>&1 | tee bench/smoke_music.log
$PY scripts/worker.py compose   2>&1 | tee bench/smoke_comp.log
echo "[smoke] DONE"

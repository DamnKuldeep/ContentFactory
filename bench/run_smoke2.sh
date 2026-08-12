#!/usr/bin/env bash
# Smoke part 2: run the workers (flags ON) against the already-seeded batch to validate the
# worker.py path + torch.compile end-to-end without re-running stage 1.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.conda/bin/python
export DRIVE_DB=1 DRIVE_PARENT_FOLDER_ID="$(cat bench/bench_folder_id.txt)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"     # populated cache (avoid 44 GB re-download)
export FISH_COMPILE=1 FLUX_COMPILE=1 LLM_PROMPT_CACHE=1
export DB_PATH=bench/smoke.sqlite OUTPUT_DIR=bench/out_smoke PROFILE_LOG=bench/profile_smoke.jsonl
export DRIVE_DB_NAME=manifest_smoke.sqlite DRIVE_LOCK_NAME=manifest_smoke.lock
rm -f "$PROFILE_LOG"; mkdir -p "$OUTPUT_DIR"
for role in narration images music compose; do
  echo "########## worker $role ##########"
  $PY scripts/worker.py "$role" 2>&1 | tee "bench/smoke2_${role}.log"
done
echo "[smoke2] DONE"

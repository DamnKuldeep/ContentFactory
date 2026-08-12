#!/usr/bin/env bash
# Resume the A/B at the IMAGE stage for both arms, preserving the already-captured narration data
# in profile_off/on.jsonl. Used after removing the expandable_segments segfault. Resets stage_03..05
# to PENDING for each arm's batch, then drains images -> music -> compose (appending to the profile).
#
#   bench/run_ab_resume.sh <bench_folder_id>
set -u
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
PY=../.conda/bin/python
BENCH_FOLDER="${1:?need bench folder id}"
export DRIVE_DB=1 DRIVE_PARENT_FOLDER_ID="$BENCH_FOLDER"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"        # no expandable_segments (it SIGSEGVs FLUX)

ARM=""
drain () {
  local role="$1" max=8 i=0 rc=0
  while [ "$i" -lt "$max" ]; do
    i=$((i + 1))
    $PY scripts/worker.py "$role" 2>&1 | tee -a "bench/${ARM}_${role}.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && return 0
    echo "[drain ${role}/${ARM}] worker exited rc=$rc (attempt $i/$max) — relaunch in 15s"
    free -h | sed -n '2p'; sleep 15
  done
  echo "[drain ${role}/${ARM}] GAVE UP after $max attempts (rc=$rc)"; return "$rc"
}

resume_arm () {
  local arm="$1" fish="$2" flux="$3" cache="$4"
  ARM="$arm"
  echo "================= RESUME ARM $arm (FISH=$fish FLUX=$flux CACHE=$cache) ================="
  export FISH_COMPILE="$fish" FLUX_COMPILE="$flux" LLM_PROMPT_CACHE="$cache"
  export DB_PATH="bench/${arm}.sqlite"
  export OUTPUT_DIR="bench/out_${arm}"
  export PROFILE_LOG="bench/profile_${arm}.jsonl"          # APPEND (keep narration data)
  export DRIVE_DB_NAME="manifest_${arm}.sqlite"
  export DRIVE_LOCK_NAME="manifest_${arm}.lock"
  rm -f "bench/${arm}_images.log" "bench/${arm}_music.log" "bench/${arm}_compose.log"
  mkdir -p "$OUTPUT_DIR"
  # reset downstream stages to PENDING for this arm's batch
  $PY - <<PYEOF
import sys; sys.path.insert(0,'.')
from shared.drive import get_drive_client
from shared.drive_db import DriveSyncManifest
from shared.manifest import STAGE_SEQUENCE
import sqlite3
up=get_drive_client(); m=DriveSyncManifest("bench/${arm}.sqlite", up); m.sync_pull()
c=sqlite3.connect("bench/${arm}.sqlite")
c.execute("UPDATE jobs SET status='PENDING',attempts=0,started_at=NULL,last_error=NULL WHERE stage IN ('stage_03_images','stage_04_music','stage_05_compose')")
c.commit(); m.sync_push(set(STAGE_SEQUENCE))
print("reset:", [tuple(r) for r in c.execute("SELECT stage,status FROM jobs ORDER BY stage")])
PYEOF
  drain images
  drain music
  drain compose
  $PY scripts/profile_report.py "$PROFILE_LOG" -o "bench/latency_${arm}.md" >/dev/null 2>&1 || true
  echo "[resume arm $arm done]"
}

t0=$(date +%s)
resume_arm off 0 0 0
resume_arm on  1 1 1
$PY scripts/profile_compare.py bench/profile_off.jsonl bench/profile_on.jsonl \
    --labels OFF ON -o bench/latency_ab.md >/dev/null 2>&1 || true
echo "[A/B resume complete in $(( ($(date +%s) - t0) / 60 )) min] -> bench/latency_ab.md"

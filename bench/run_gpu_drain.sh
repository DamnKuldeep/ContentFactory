#!/usr/bin/env bash
# Drain the GPU stages for batch 20260628_153812 as four sequential role-drains:
#   narration x88 -> images x88 -> music x88 -> compose x88
# Each role loads its model ONCE and processes all ready videos (load amortized).
# Crash-resilient: relaunch a role's worker on any non-zero exit (OOM/segfault), but only as long
# as it keeps making progress; give up a role after 4 consecutive no-progress relaunches.
set -u
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
PY=../.conda/bin/python
BATCH=20260628_153812

export DRIVE_DB=1 HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export BGM_TWO_PASS=0          # REQUIRED on 24 GB (free Qwen before ACE)
export FISH_COMPILE=1          # ~2x TTS (amortizes over the batch)
export FLUX_COMPILE=0          # eager FLUX (OOM-safe; validated)
mkdir -p bench

stage_of () { case "$1" in
  narration) echo stage_02_narration;; images) echo stage_03_images;;
  music) echo stage_04_music;; compose) echo stage_05_compose;; esac; }

ccount () {  # COMPLETE count for a stage (from the local synced manifest)
  $PY - "$1" <<'PYEOF' 2>/dev/null || echo 0
import sqlite3,sys
print(sqlite3.connect("manifest.sqlite").execute(
  "SELECT COUNT(*) FROM jobs WHERE batch_id=? AND stage=? AND status='COMPLETE'",
  ("20260628_153812", sys.argv[1])).fetchone()[0])
PYEOF
}

drain () {
  local role="$1" st; st=$(stage_of "$1"); local noprog=0 rc=0 before after
  while [ "$noprog" -lt 6 ]; do
    before=$(ccount "$st")
    $PY scripts/worker.py "$role" 2>&1 | tee -a "bench/drain_${role}.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && { echo "[drain $role] clean exit ($(ccount "$st") complete)"; return 0; }
    after=$(ccount "$st")
    if [ "$after" -gt "$before" ]; then noprog=0; else noprog=$((noprog + 1)); fi
    echo "[drain $role] rc=$rc  progress ${before}->${after}  noprog=${noprog}/4 — relaunch in 20s"
    free -h | sed -n '2p'; sleep 20
  done
  echo "[drain $role] GAVE UP after no progress (rc=$rc, $(ccount "$st") complete)"; return 1
}

t0=$(date +%s)
for role in narration images music compose; do
  echo "================= DRAIN $role  $(date '+%F %T') ================="
  free -h | sed -n '1,2p'
  drain "$role" || { echo "[gpu-drain] $role did not fully drain — stopping (inspect, then resume)."; break; }
  echo "[gpu-drain] $role done at $(date '+%F %T')  [elapsed $(( ($(date +%s)-t0)/60 )) min]"
done
echo "[gpu-drain COMPLETE in $(( ($(date +%s)-t0)/60 )) min]"

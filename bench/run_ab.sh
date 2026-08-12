#!/usr/bin/env bash
# A/B re-profile: OFF arm (flags off) then ON arm (FISH_COMPILE+FLUX_COMPILE+LLM_PROMPT_CACHE on).
# Drive ON but isolated in a throwaway bench folder (DRIVE_PARENT_FOLDER_ID), per-arm DB names —
# the user's real manifest.sqlite / videos catalog are never touched.
#
#   bench/run_ab.sh <count> <bench_folder_id>
#
# Crash-resilient: if a worker is OOM-killed mid-job (hard SIGKILL, exit!=0), it is relaunched.
# The killed job is still PENDING in Drive (push happens only after completion), so the relaunched
# worker re-claims it and resumes (stage_03 skips already-uploaded scenes via its resume logic).
set -u
set -o pipefail                                # so PIPESTATUS/tee surfaces the worker's real exit code
cd "$(dirname "$0")/.." || exit 1             # ContentFactory/
PY=../.conda/bin/python
COUNT="${1:-3}"
BENCH_FOLDER="${2:?need bench folder id (from bench/bench_drive.py create)}"
mkdir -p bench

export DRIVE_DB=1
export DRIVE_PARENT_FOLDER_ID="$BENCH_FOLDER"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"              # populated cache (avoid 44 GB re-download)
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — it SIGSEGVs (rc=139) the FLUX
# image worker (multithreaded upload pool + CUDA). RAM headroom (free apps) handles OOM instead.

ARM=""
drain () {                                     # relaunch a role's worker until it exits cleanly
  local role="$1" max=8 i=0 rc=0
  while [ "$i" -lt "$max" ]; do
    i=$((i + 1))
    $PY scripts/worker.py "$role" 2>&1 | tee -a "bench/${ARM}_${role}.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && return 0               # clean drain (no ready jobs left)
    echo "[drain ${role}/${ARM}] worker exited rc=$rc (attempt $i/$max) — relaunch in 15s"
    free -h | sed -n '1,3p'
    sleep 15
  done
  echo "[drain ${role}/${ARM}] GAVE UP after $max attempts (rc=$rc)"; return "$rc"
}

run_arm () {
  local arm="$1" fish="$2" flux="$3" cache="$4"
  ARM="$arm"
  echo "=================================================================="
  echo "  ARM $arm  (FISH_COMPILE=$fish FLUX_COMPILE=$flux LLM_PROMPT_CACHE=$cache)  count=$COUNT"
  free -h | sed -n '1,3p'
  echo "=================================================================="
  export FISH_COMPILE="$fish" FLUX_COMPILE="$flux" LLM_PROMPT_CACHE="$cache"
  export DB_PATH="bench/${arm}.sqlite"
  export OUTPUT_DIR="bench/out_${arm}"
  export PROFILE_LOG="bench/profile_${arm}.jsonl"
  export DRIVE_DB_NAME="manifest_${arm}.sqlite"
  export DRIVE_LOCK_NAME="manifest_${arm}.lock"
  rm -f "$PROFILE_LOG" "bench/${arm}"_*.log    # fresh profile + per-role logs per arm
  mkdir -p "$OUTPUT_DIR"

  $PY scripts/produce.py --count "$COUNT" --workers "$COUNT" 2>&1 | tee "bench/${arm}_story.log"
  drain narration
  drain images
  drain music
  drain compose
  $PY scripts/profile_report.py "$PROFILE_LOG" -o "bench/latency_${arm}.md" >/dev/null 2>&1 || true
  echo "[arm $arm done]"
}

t0=$(date +%s)
run_arm off 0 0 0
run_arm on  1 1 1
$PY scripts/profile_compare.py bench/profile_off.jsonl bench/profile_on.jsonl \
    --labels OFF ON -o bench/latency_ab.md >/dev/null 2>&1 || true
echo "[A/B complete in $(( ($(date +%s) - t0) / 60 )) min]  -> bench/latency_ab.md"

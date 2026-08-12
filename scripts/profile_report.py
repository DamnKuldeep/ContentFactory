#!/usr/bin/env python3
"""
profile_report.py — aggregate a PROFILE_LOG JSONL into a latency report.

    python scripts/profile_report.py profile_<ts>.jsonl [-o latency_report.md]

Emits a per-stage step table, per-stage totals, a category roll-up (load / infer / io / api / cpu)
and the GPU-seconds (load+infer) total — the cost-relevant number — plus the slowest steps.
"""

import argparse
import json
import sys
from collections import defaultdict

STAGE_ORDER = ["stage_01_story", "stage_02_narration", "stage_03_images",
               "stage_04_music", "stage_05_compose"]
GPU_KINDS = {"load", "infer"}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def fmt(s):
    s = float(s)
    return f"{s:6.1f}s" if s < 600 else f"{s/60:5.1f}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="PROFILE_LOG jsonl path")
    ap.add_argument("-o", "--out", default="latency_report.md")
    args = ap.parse_args()

    rows = load(args.log)
    if not rows:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    # split process_total/stage_total from step rows
    steps = [r for r in rows if r.get("step") not in ("process_total", "stage_total")]
    totals = {r["stage"]: r["seconds"] for r in rows if r.get("step") == "process_total"}

    by_stage = defaultdict(list)
    for r in steps:
        by_stage[r["stage"]].append(r)

    cat_tot = defaultdict(float)        # kind -> seconds (step rows only)
    out = ["# Latency report", "", f"Source: `{args.log}`", ""]

    ordered = [s for s in STAGE_ORDER if s in by_stage or s in totals]
    ordered += [s for s in by_stage if s not in STAGE_ORDER]

    grand = 0.0
    for st in ordered:
        srows = sorted(by_stage.get(st, []), key=lambda r: -r["seconds"])
        stage_total = totals.get(st, sum(r["seconds"] for r in srows))
        grand += stage_total
        out.append(f"## {st} — {fmt(stage_total)} total")
        out.append("")
        out.append("| step | kind | seconds | % stage | notes |")
        out.append("|------|------|--------:|--------:|-------|")
        for r in srows:
            cat_tot[r["kind"]] += r["seconds"]
            pct = 100 * r["seconds"] / stage_total if stage_total else 0
            notes = " ".join(f"{k}={v}" for k, v in r.items()
                             if k not in ("ts", "stage", "step", "kind", "seconds"))
            out.append(f"| {r['step']} | {r['kind']} | {r['seconds']:.1f} | {pct:4.0f}% | {notes} |")
        measured = sum(r["seconds"] for r in srows)
        io_overhead = stage_total - measured
        if io_overhead > 0.5:
            out.append(f"| _(unmeasured: I/O + overhead)_ | io | {io_overhead:.1f} | "
                       f"{100*io_overhead/stage_total:4.0f}% | download/upload/json |")
        out.append("")

    gpu_secs = cat_tot.get("load", 0) + cat_tot.get("infer", 0)
    out.append("## Category roll-up")
    out.append("")
    out.append("| category | seconds | note |")
    out.append("|----------|--------:|------|")
    for k in ("load", "infer", "io", "api", "cpu"):
        if cat_tot.get(k):
            out.append(f"| {k} | {cat_tot[k]:.1f} | |")
    out.append(f"| **GPU (load+infer)** | **{gpu_secs:.1f}** | the cost-relevant number |")
    out.append(f"| **grand total (wall)** | **{grand:.1f}** | ≈ {grand/60:.1f} min for one video |")
    out.append("")

    out.append("## Slowest steps")
    out.append("")
    top = sorted(steps, key=lambda r: -r["seconds"])[:12]
    out.append("| stage | step | kind | seconds |")
    out.append("|-------|------|------|--------:|")
    for r in top:
        out.append(f"| {r['stage']} | {r['step']} | {r['kind']} | {r['seconds']:.1f} |")
    out.append("")

    report = "\n".join(out)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[written: {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()

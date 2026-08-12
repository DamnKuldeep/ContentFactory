#!/usr/bin/env python3
"""
profile_compare.py — A/B diff of two PROFILE_LOG JSONL files (control vs treatment).

Built for the caching + torch.compile re-profile: each arm drains N videos in one worker process,
so the model loads/compiles ONCE and the compile cost lands inside the *first* tts_generate /
flux_generate. This tool therefore splits each per-video step into:
    cold = first occurrence  (pays the one-time compile cost in the ON arm)
    warm = mean of the rest  (steady state — the number that matters at scale)

    python scripts/profile_compare.py bench/profile_off.jsonl bench/profile_on.jsonl \
        --labels OFF ON -o bench/latency_ab.md

Latency only. Prompt-caching savings are a $ effect (stage-1 Usage cost), extracted separately.
"""

import argparse
import json
import sys
from collections import defaultdict

# steps whose first occurrence carries the one-time torch.compile cost in the ON arm
COMPILE_STEPS = ("tts_generate", "flux_generate", "fish_load", "flux_load")
GPU_KINDS = ("load", "infer")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def group(rows):
    """(stage, step) -> [seconds...] in order of appearance (≈ video order)."""
    g = defaultdict(list)
    for r in rows:
        if "seconds" not in r or "step" not in r:
            continue
        g[(r.get("stage", "?"), r["step"])].append(float(r["seconds"]))
    return g


def cold_warm(secs):
    """cold = first run; warm = mean of subsequent runs (or the single value if only one)."""
    if not secs:
        return 0.0, 0.0, 0
    if len(secs) == 1:
        return secs[0], secs[0], 1
    warm = sum(secs[1:]) / len(secs[1:])
    return secs[0], warm, len(secs)


def n_videos(rows):
    g = group(rows)
    for key in (("stage_05_compose", "process_total"), ("stage_02_narration", "tts_generate"),
                ("stage_02_narration", "process_total")):
        if g.get(key):
            return len(g[key])
    return max((len(v) for v in g.values()), default=1)


def cat_totals(rows):
    c = defaultdict(float)
    for r in rows:
        if r.get("step") in ("process_total", "stage_total"):
            continue
        c[r.get("kind", "?")] += float(r.get("seconds", 0))
    return c


def pct(a, b):
    return 0.0 if not a else 100.0 * (b - a) / a


def main():
    ap = argparse.ArgumentParser(description="A/B diff of two PROFILE_LOG JSONL files.")
    ap.add_argument("off", help="control (flags OFF) jsonl")
    ap.add_argument("on", help="treatment (flags ON) jsonl")
    ap.add_argument("--labels", nargs=2, default=["OFF", "ON"])
    ap.add_argument("-o", "--out", default="latency_ab.md")
    args = ap.parse_args()

    A, B = args.labels
    ra, rb = load(args.off), load(args.on)
    if not ra or not rb:
        print("One of the logs is empty.", file=sys.stderr)
        sys.exit(1)
    ga, gb = group(ra), group(rb)
    na, nb = n_videos(ra), n_videos(rb)

    out = ["# A/B latency comparison", "",
           f"- **{A}** (control): `{args.off}` — {na} video(s)",
           f"- **{B}** (treatment): `{args.on}` — {nb} video(s)", "",
           "Per-video steps report **warm** (mean excluding the first run, which pays the one-time "
           "compile cost in the treatment arm). Single-occurrence steps (model loads) report the lone "
           "value.", ""]

    # ── 1. compile-sensitive steps: cold vs warm, both arms ──
    out += ["## Compile-sensitive steps — cold (1st) vs warm (rest)", "",
            f"| stage | step | {A} cold | {A} warm | {B} cold | {B} warm | warm Δ | warm Δ% |",
            "|-------|------|--------:|--------:|--------:|--------:|------:|-------:|"]
    keys = sorted(set(ga) | set(gb))
    for (st, step) in keys:
        if step not in COMPILE_STEPS:
            continue
        ca, wa, _ = cold_warm(ga.get((st, step), []))
        cb, wb, _ = cold_warm(gb.get((st, step), []))
        out.append(f"| {st.replace('stage_0','s')} | {step} | {ca:.1f} | {wa:.1f} | "
                   f"{cb:.1f} | {wb:.1f} | {wb-wa:+.1f} | {pct(wa,wb):+.0f}% |")
    out.append("")

    # ── 2. all steps, warm side-by-side ──
    out += ["## All steps — warm (OFF vs ON)", "",
            f"| stage | step | kind | {A} | {B} | Δ | Δ% | runs {A}/{B} |",
            "|-------|------|------|----:|----:|---:|---:|:----------:|"]
    # map step -> kind from whichever arm has it
    kind_of = {}
    for r in ra + rb:
        kind_of[(r.get("stage"), r.get("step"))] = r.get("kind", "?")
    rows_sorted = sorted(keys, key=lambda k: -cold_warm(gb.get(k, ga.get(k, [])))[1])
    for (st, step) in rows_sorted:
        if step in ("process_total", "stage_total"):
            continue
        _, wa, ka = cold_warm(ga.get((st, step), []))
        _, wb, kb = cold_warm(gb.get((st, step), []))
        out.append(f"| {st.replace('stage_0','s')} | {step} | {kind_of.get((st,step),'?')} | "
                   f"{wa:.1f} | {wb:.1f} | {wb-wa:+.1f} | {pct(wa,wb):+.0f}% | {ka}/{kb} |")
    out.append("")

    # ── 3. stage wall (process_total) cold vs warm ──
    out += ["## Stage wall per video (process_total: load+infer+I/O) — cold vs warm", "",
            f"| stage | {A} cold | {A} warm | {B} cold | {B} warm | warm Δ% |",
            "|-------|--------:|--------:|--------:|--------:|-------:|"]
    for (st, step) in keys:
        if step != "process_total":
            continue
        ca, wa, _ = cold_warm(ga.get((st, step), []))
        cb, wb, _ = cold_warm(gb.get((st, step), []))
        out.append(f"| {st.replace('stage_0','s')} | {ca:.1f} | {wa:.1f} | {cb:.1f} | {wb:.1f} | "
                   f"{pct(wa,wb):+.0f}% |")
    out.append("")

    # ── 4. category roll-up (whole arm) ──
    cta, ctb = cat_totals(ra), cat_totals(rb)
    out += ["## Category roll-up (whole arm, all videos)", "",
            f"| category | {A} (s) | {B} (s) | Δ% |", "|----------|----:|----:|---:|"]
    for k in ("load", "infer", "io", "api", "cpu"):
        if cta.get(k) or ctb.get(k):
            out.append(f"| {k} | {cta.get(k,0):.1f} | {ctb.get(k,0):.1f} | {pct(cta.get(k,0),ctb.get(k,0)):+.0f}% |")
    gpa = sum(cta.get(k, 0) for k in GPU_KINDS)
    gpb = sum(ctb.get(k, 0) for k in GPU_KINDS)
    out.append(f"| **GPU (load+infer)** | **{gpa:.1f}** | **{gpb:.1f}** | **{pct(gpa,gpb):+.0f}%** |")
    out.append(f"| GPU per video (÷{A}={na}, ÷{B}={nb}) | {gpa/max(na,1):.1f} | {gpb/max(nb,1):.1f} | "
               f"{pct(gpa/max(na,1), gpb/max(nb,1)):+.0f}% |")
    out.append("")
    out += ["## Notes",
            "- Negative Δ% = ON faster than OFF. The **warm** columns are the steady-state numbers; "
            "the ON **cold** column is inflated by the one-time torch.compile build and amortizes "
            "away over a real 100–150 video batch.",
            "- Prompt-caching savings are not in this table (no latency effect) — see the stage-1 "
            "`cost_usd` / `cached_input_tokens` comparison in the run summary.", ""]

    report = "\n".join(out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[written: {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()

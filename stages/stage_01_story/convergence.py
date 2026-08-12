"""
Content Factory — Convergence harness, critics, reconcile.

Extracted verbatim from the notebook. The design that converges instead of oscillating.
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import logging
from typing import Callable, List

from .config import SAMPLING, model_for
from shared.llm import call_struct, call_text
from .models import Critique, Issue, LoopResult, Reconciled

logger = logging.getLogger("contentfactory.convergence")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _wc(t: str) -> int:
    return len((t or "").split())


def _ok_or(prev, candidate, frac=0.55):
    """Use the new candidate only if it is substantial (non-empty and >= frac of the previous text's length);
    otherwise keep the previous good draft. Guards against empty/refused/truncated showrunner replies."""
    c = (candidate or "").strip()
    if c and _wc(c) >= max(1, int(frac * max(1, _wc(str(prev))))):
        return c
    logger.info("guard: showrunner output empty/short → kept best refined draft")
    return str(prev).strip()


# ── Logging helpers ───────────────────────────────────────────────────────────

def _log_line(msg: str):
    logger.info(msg)


def _log_kv(key: str, value: str):
    logger.info("%s: %s", key, value)


def _log_block(label: str, text: str):
    logger.info("[%s]\n%s", label, text)


def _log_issues(items, kind):
    for it in items:
        sid = getattr(it, "scene_id", getattr(it, "prompt_id", "?"))
        _log_line(f"      • {kind} {sid} — {it.problem}")
        if getattr(it, "suggestion", ""):
            _log_line(f"        ↳ {it.suggestion}")


# ── Critic machinery ─────────────────────────────────────────────────────────

from prompts import CALIBRATION, CRITIC_SYS, FEEDBACK_EDITOR_SYS

def _critic(role, lane, focus, payload) -> Critique:
    """Run one critic. Two safety nets so a single critic can never pin the loop:
    • if it can't produce valid output (even after retries), it ABSTAINS (passes);
    • if it returns a NOT-satisfied verdict with NO concrete issue, that verdict is useless (nothing to act on)
      and per our own rules invalid → we treat it as a PASS for this loop and log it."""
    try:
        c = call_struct(model_for(role), CRITIC_SYS.format(lane=lane, focus=focus), payload,
                        Critique, **SAMPLING["judge"], max_tokens=2000)
    except Exception as e:
        _log_kv("critic abstain", f"{lane} ({type(e).__name__}) — passing this loop")
        return Critique(score=8, satisfied=True, issues=[])
    if (not c.satisfied) and (not c.issues):
        _log_kv("critic abstain", f"{lane}: low score with no concrete issue → treated as pass")
        return Critique(score=max(c.score, 7), satisfied=True, issues=[])
    return c


def run_critics_parallel(critics, payload) -> dict:
    out = {}
    with _cf.ThreadPoolExecutor(max_workers=max(1, len(critics))) as ex:
        futs = {ex.submit(_critic, r, lane, focus, payload): lane for (r, lane, focus) in critics}
        for f in _cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out  # {lane: Critique}


def _length_note(text, lo, hi, tgt):
    wc = _wc(text)
    if wc > hi:
        return (f"LENGTH FACT: {wc} words — {wc - hi} OVER the {lo}–{hi} range (target ~{tgt}); "
                f"trimming toward ~{tgt} is a top-priority fix and your score stays capped until it fits.")
    if wc < lo:
        return f"LENGTH FACT: {wc} words — a bit short of {lo}–{hi}; make sure nothing essential is missing."
    return f"LENGTH FACT: {wc} words — within the {lo}–{hi} range; do NOT ask for cuts on length grounds."


def _aggregate_raw(verdicts: dict, passed: list) -> str:
    lines = []
    if passed:
        lines.append("LANES ALREADY PASSED (do NOT break these): " + ", ".join(passed) + "\n")
    lines.append("RAW CRITIC VERDICTS THIS LOOP:")
    for lane, v in verdicts.items():
        if v.satisfied:
            lines.append(f"\n[{lane}] PASSED — no changes."); continue
        lines.append(f"\n[{lane}] needs work (score {v.score}/10):")
        for i in v.issues:
            lines.append(f"  • Problem: {i.problem}")
            if i.quote: lines.append(f'    In the draft: "{i.quote}"')
            lines.append(f"    Direction: {i.fix}")
    return "\n".join(lines)


def reconcile(verdicts: dict, passed: list) -> Reconciled:
    if all(v.satisfied for v in verdicts.values()) and not any(v.issues for v in verdicts.values()):
        return Reconciled(summary="All lanes satisfied.", changes=[], keep=[])
    return call_struct(model_for("feedback_editor"), FEEDBACK_EDITOR_SYS,
                       "Reconcile these raw verdicts into one prioritized, non-contradictory set of changes.\n\n"
                       + _aggregate_raw(verdicts, passed), Reconciled, **SAMPLING["mechanical"], max_tokens=1600)


def _fmt_reconciled(rf: Reconciled) -> str:
    out = [f"OVERALL: {rf.summary}"]
    if rf.keep:
        out.append("\nKEEP (don't break these):"); out += [f"  ✓ {k}" for k in rf.keep]
    if rf.changes:
        out.append("\nCHANGES (in priority order):"); out += [f"  {i}. {c}" for i, c in enumerate(rf.changes, 1)]
    else:
        out.append("\n(No substantive changes needed.)")
    return "\n".join(out)


def _stats(verdicts: dict):
    return (sum(1 for v in verdicts.values() if v.satisfied), sum(v.score for v in verdicts.values()))


# ── Convergence loop ──────────────────────────────────────────────────────────

def converge(draft, critics, payload_fn, revise_fn, *, max_loops: int, label: str) -> LoopResult:
    """Score-gated convergence — the design that converges instead of oscillating:
    • critics score 1–10 + a binary `satisfied`; the loop runs until EVERY lane is satisfied or max_loops —
      it does NOT bail on 'no improvement', so we push toward a genuinely good piece.
    • a lane that passes is RETIRED, so each loop changes LESS and a fixed lane can't be re-torn-down.
    • the BEST draft = (most lanes satisfied, then highest total score, then latest), stored WITH its verdicts.
    """
    n = len(critics)
    active = list(critics)
    passed, last_by_lane = [], {}
    cur = draft
    best = {"draft": draft, "verdicts": {}, "rank": (-1, -1)}
    for loop in range(1, max_loops + 1):
        _log_line(f"   ┏━ {label} loop {loop} · active: {', '.join(l for _, l, _ in active)}")
        verdicts = run_critics_parallel(active, payload_fn(cur))
        last_by_lane.update(verdicts)
        full = dict(last_by_lane)                       # all lanes (passed lanes keep their last verdict)
        newly_passed = []
        for (r, lane, focus) in active:
            v = verdicts[lane]
            _log_line(f"   ┃ {'✅' if v.satisfied else '❌'} {lane} — {v.score}/10")
            for i in v.issues:
                _log_line(f"   ┃     • {i.problem}")
                if i.quote: _log_line(f'   ┃       ↳ "{i.quote}"')
                _log_line(f"   ┃       → {i.fix}")
            if v.satisfied:
                newly_passed.append(lane)
        rank = _stats(full)
        if rank >= best["rank"]:
            best = {"draft": cur, "verdicts": dict(full), "rank": rank}
        _log_line(f"   ┃ 📊 satisfied {rank[0]}/{n}  (scores: {', '.join(str(v.score) for v in verdicts.values())})")
        for lane in newly_passed:
            passed.append(lane); active = [c for c in active if c[1] != lane]
        if (len(full) >= n and all(v.satisfied for v in full.values())) or not active:
            _log_line(f"   ┗━ ✅ {label}: all lanes satisfied at loop {loop}"); break
        if loop == max_loops:
            _log_line(f"   ┗━ ⏹ {label}: max loops → hand over best ({best['rank'][0]}/{n} satisfied)"); break
        rf = reconcile(verdicts, passed)
        _log_block(f"{label} reconciled feedback (loop {loop})", _fmt_reconciled(rf))
        cur = revise_fn(cur, rf)
        _log_block(f"{label} revised draft (after loop {loop}, {_wc(cur)}w)", cur)
    return LoopResult(best["draft"], float(best["rank"][0]), best["verdicts"])


# ── Judge⇄fix loop (used for plan and prompt stages) ─────────────────────────

def _keep_best_loop(label, max_iters, judge_fn, update_fn, count_fn):
    """Shared judge⇄fix loop. Each round APPLIES the judge's fixes and keeps the latest result — the updater only
    edits the flagged items, so a later pass surfacing *new* issues is not a regression, and we must never revert
    to an earlier un-fixed version (that bug shipped the original batch with all its leaks). Stops when a pass is
    clean or the cap is hit. Names/blur have deterministic backstops after this loop regardless."""
    obj, prev = count_fn(), None
    for it in range(max_iters):
        verdict = judge_fn(obj, prev)
        items = verdict.issues if hasattr(verdict, "issues") else verdict.fixes
        if verdict.passed:
            _log_kv(f"{label} iter {it + 1}", "✓ passed"); return obj
        _log_kv(f"{label} iter {it + 1}", f"{len(items)} issue(s)"); _log_issues(items, label)
        obj = update_fn(obj, items); prev = items      # apply fixes every round (incl. the last); keep the latest
    _log_kv(label, "reached cap"); return obj

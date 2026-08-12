"""
Content Factory — Pipeline stages A, B, C + final assembly.

Every function, every prompt call, every parameter extracted verbatim from the notebook.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .config import (
    CONFIG, DIMS, LOOPS, SAMPLING, VARIETY_AXES,
    banned_block, good_script_block, good_story_block, model_for,
)
from .convergence import (
    _fmt_reconciled, _keep_best_loop, _log_issues, _ok_or, _wc,
    converge, reconcile,
)
from shared.llm import METER, call_struct, call_text, set_stage
from .models import (
    AmateurTurn, Blueprint, Character, ConceptOut, Critique,
    CutResult, ExtractResult, PlanVerdict, Premise, PremiseRanking,
    PremiseSet, PromptBatch, PromptFix, PromptItem, PromptVerdict,
    Questions, Reconciled, Scene, ScenePlan, Segment, SegmentResult,
    Setting, StyleResult, StructuredPrompt,
)
from .prompts import (
    AMATEUR_SYS, BRIEF_AMATEUR_SYS, BRIEF_DIRECTOR_SYS,
    CONCEPT_SYS, DRAFTER_SYS, EXPERT_SYS, EXTRACTOR_SYS,
    FLUX2_CARD, IDEATOR_SYS, LENGTH_FIX_SYS,
    PLAN_JUDGE_SYS, PLAN_UPDATER_SYS, PLANNER_SYS,
    PREMISE_JUDGE_SYS, PROMPT_REVIEWER_SYS, PROMPT_REWRITER_SYS,
    PROMPT_WRITER_SYS, REVISER_SYS, SCRIPT_CRITICS,
    SCRIPT_DRAFTER_SYS, SCRIPT_GOAL, SCRIPT_NAME_GUARD_SYS,
    SCRIPT_REVISER_SYS, SCRIPT_SHOWRUNNER_SYS,
    STORY_CRITICS, STORY_GOAL, STORY_SHOWRUNNER_SYS,
    STYLE_POOL, STYLE_SYS, VS_INSTRUCTIONS, SEGMENTER_SYS_TMPL,
)

logger = logging.getLogger("contentfactory.pipeline")


def _log_kv(k, v):
    logger.info("%s: %s", k, v)


def _log_block(label, text):
    logger.info("[%s]\n%s", label, text)


def _log_step(msg):
    logger.info("── %s", msg)


def _log_stage(msg):
    logger.info("\n" + "█" * 60 + f"\n█ {msg}\n" + "█" * 60)


# ============================================================================
# VARIETY ENGINE — Verbalized Sampling
# ============================================================================

def sample_axes(rng: random.Random) -> dict:
    """Draw ONE value from each variety axis — the combination IS the creative seed for this story."""
    return {
        "era": rng.choice(VARIETY_AXES["era"]),
        "region": rng.choice(VARIETY_AXES["region"]),
        "domain": rng.choice(VARIETY_AXES["domain"]),
        "milieu": rng.choice(VARIETY_AXES["milieu"]),
        "motif": rng.choice(VARIETY_AXES["motif"]),
        "flavor": rng.choice(VARIETY_AXES["flavor"]),
        "structure": rng.choice(VARIETY_AXES["structure"]),
        "telling_register": rng.choice(VARIETY_AXES["register"]),
    }


def axes_brief(axes: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in axes.items())


def compose_setting(rng: random.Random):
    """One LLM call that turns a random axis combination into a coherent creative direction."""
    axes = sample_axes(rng)
    user = ("These creative AXES were drawn at random. Build ONE coherent, SPECIFIC creative direction:\n"
            + axes_brief(axes))
    out = call_struct(model_for("concept_director"), CONCEPT_SYS, user, ConceptOut,
                      temperature=0.85, min_p=0.05, max_tokens=1200)
    return axes, out.direction


# ============================================================================
# STAGE A — STORY
# ============================================================================

def ideate(direction, axes, rng) -> PremiseSet:
    anchor = (f"- era: {direction.era}\n- place: {direction.place}\n- world: {direction.premise_space}\n"
              f"- domain: {axes['domain']}\n- milieu: {axes['milieu']}\n- recurring motif: {axes['motif']}\n"
              f"- flavor: {axes['flavor']}\n- shape: {axes['structure']}\n- telling register: {axes['telling_register']}")
    user = (good_story_block() + "\n\n" + banned_block() +
            "\n\nANCHOR HERE (a coherent creative direction — honour it):\n" + anchor +
            "\n\nProduce 7 DISTINCT premises with probabilities (include fresh, lower-probability ones).")
    return call_struct(model_for("ideator"), IDEATOR_SYS, user, PremiseSet,
                       **SAMPLING["ideate"], max_tokens=2400)


def select_premise(pset: PremiseSet, rng) -> Premise:
    rank = call_struct(model_for("premise_judge"), PREMISE_JUDGE_SYS,
                       "PREMISES:\n" + json.dumps([p.model_dump() for p in pset.premises], indent=2),
                       PremiseRanking, **SAMPLING["judge"], max_tokens=2000)
    by_id = {p.id: p for p in pset.premises}
    # weighted-sample among near-top (within 0.5 of best) for extra run-to-run variety (stateless via rng)
    scored = [(s.id, s.overall) for s in rank.ranked if s.id in by_id]
    if not scored:
        return by_id.get(rank.pick_id, pset.premises[0])
    top = max(s for _, s in scored)
    near = [(i, s) for i, s in scored if s >= top - 0.5]
    ids, weights = zip(*near)
    chosen = rng.choices(ids, weights=[w + 0.1 for w in weights], k=1)[0]
    return by_id[chosen]


def run_interview(premise: Premise) -> str:
    """Adaptive amateur↔expert exploration. The expert offers OPTIONS (not a locked plot); we return the whole
    conversation as raw material for the drafter. Variable questions per round (≤3 then 1–2), early-exit, logged."""
    seed = f"PREMISE\nHOOK: {premise.logline}\nSPINE: {premise.spine}"
    transcript = []
    lo, hi = LOOPS["min_interview"], LOOPS["max_interview"]
    for rnd in range(1, hi + 1):
        can_stop = rnd > lo
        if rnd == 1:
            ask = ("Begin the interview. Ask your up-to-3 opening questions to explore the BACKBONE: (1) who the "
                   "ordinary person is + their normal life, (2) the line they cross + what pushes them to it, "
                   "(3) how it escalates and what the turn/ending could be. Set thread_focus to 'building the story spine'.")
        else:
            stop = ("If you now have ENOUGH to write a really good story (you can picture the start, the line crossed, the "
                    "escalation, and one or two strong endings, plus a few concrete specifics), set done=true and leave. "
                    if can_stop else "You may NOT stop yet (minimum rounds not reached); set done=false. ")
            ask = ("Read the expert's latest answer. If something vivid is still MISSING, ask only 1–2 questions that fill "
                   "that gap or pin down a concrete specific (a year/place/name/number, a clearer cause, a stronger ending) — "
                   "do NOT just reopen settled ground and do NOT go past the ending (no aftermath/rebuilding/legacy). "
                   "Name the part you're deepening in thread_focus. " + stop)
        ctx = seed + ("\n\nINTERVIEW SO FAR:\n" + "\n\n".join(transcript) if transcript else "")
        turn = call_struct(model_for("interview_amateur"), AMATEUR_SYS, ctx + "\n\n" + ask,
                           AmateurTurn, temperature=0.7, max_tokens=1100)
        if turn.done and can_stop:
            _log_kv(f"interview round {rnd}", "amateur has enough material; ending interview.")
            break
        qs = turn.questions or ["What's the most important thing I haven't asked about yet?"]
        ans = call_text(model_for("interview_expert"), EXPERT_SYS,
                        seed + "\n\nQUESTIONS:\n" + "\n".join(f"Q{i}: {q}" for i, q in enumerate(qs, 1))
                        + ("\n\nEARLIER:\n" + "\n\n".join(transcript) if transcript else ""),
                        temperature=0.6, max_tokens=1700)
        _log_block(f"interview · round {rnd}",
                   f"focus: {turn.thread_focus}\n" + "\n".join(f"🧑 Q{i}: {q}" for i, q in enumerate(qs, 1))
                   + f"\n\n🎓 EXPERT (options):\n{ans}")
        transcript.append("Q: " + " | ".join(qs) + "\nEXPERT OPTIONS: " + ans)
    return "\n\n".join(transcript)


def draft_story(premise: Premise, direction, axes, transcript: str) -> str:
    sl = CONFIG["story_len"]
    user = (STORY_GOAL + "\n\n" + good_story_block()
            + "\n\nPREMISE\nHOOK: " + premise.logline + "\nSPINE: " + premise.spine
            + f"\n\nCREATIVE DIRECTION\n{direction.title} — {direction.era} · {direction.place}\n{direction.premise_space}"
            + "\n\nBRAINSTORM INTERVIEW (raw material + inspiration — use it freely: combine compatible pieces, leave parts out, or invent something better it sparked):\n"
            + (transcript or "(no interview captured)")
            + f"\n\nNow write ONE focused, coherent SOURCE story (~{sl['target']} words, within {sl['low']}–{sl['high']}). "
            "Build the strongest single through-line and keep it coherent; stop at the ending.")
    return call_text(model_for("story_drafter"), DRAFTER_SYS, user, **SAMPLING["create"], max_tokens=2200)


def revise_story(draft, rf: Reconciled):
    if not rf.changes:
        return draft
    return call_text(model_for("story_reviser"), REVISER_SYS,
                     STORY_GOAL + "\n\nCURRENT STORY:\n" + draft + "\n\nRECONCILED FEEDBACK:\n" + _fmt_reconciled(rf),
                     **SAMPLING["revise"], max_tokens=2200)


def showrunner_story(best_draft, verdicts):
    notes = ""
    if verdicts:
        notes = "\n\nPANEL NOTES:\n" + json.dumps(
            {lane: {"score": c.score, "issues": [i.model_dump() for i in c.issues]} for lane, c in verdicts.items()}, indent=2)
    return _ok_or(best_draft, call_text(model_for("story_showrunner"), STORY_SHOWRUNNER_SYS,
                     STORY_GOAL + "\n\nBEST DRAFT:\n" + best_draft + notes, **SAMPLING["finalize"], max_tokens=2200))


def run_story() -> dict:
    rng = random.Random()
    set_stage("stage_a")
    _log_stage("STAGE A — STORY")
    _log_step("compose setting + creative direction")
    axes, direction = compose_setting(rng)
    _log_block("variety axes", axes_brief(axes))
    _log_block("creative direction", f"{direction.title}\n{direction.era} · {direction.place}\n{direction.premise_space}")
    _log_step("ideation (Verbalized Sampling)")
    pset = ideate(direction, axes, rng)
    _log_block("premises", "\n".join(f"[{p.id}] p={p.probability:.2f}  {p.logline}" for p in pset.premises))
    chosen = select_premise(pset, rng)
    _log_kv("selected premise", f"[{chosen.id}] {chosen.logline}")
    _log_step("develop (adaptive interview)")
    transcript = run_interview(chosen)
    _log_step("draft (source story)")
    story = draft_story(chosen, direction, axes, transcript)
    _log_block(f"draft ({_wc(story)}w)", story)
    _log_step("critique ⇄ refine")
    sl = CONFIG["story_len"]
    from .convergence import _length_note
    payload = lambda d: STORY_GOAL + "\n\n" + _length_note(d, sl["low"], sl["high"], sl["target"]) + "\n\n---\nSTORY:\n" + d + "\n---"
    res = converge(story, STORY_CRITICS, payload, revise_story, max_loops=LOOPS["story_refine"], label="story")
    _log_step("showrunner")
    story_wc = _wc(res.best)
    all_satisfied = bool(res.verdicts) and all(v.satisfied for v in res.verdicts.values())
    if all_satisfied and sl["low"] <= story_wc <= sl["high"]:
        final = res.best
        _log_kv("showrunner", "draft satisfied every lane and is in-band — kept verbatim (no rewrite)")
    else:
        final = showrunner_story(res.best, res.verdicts)
    _log_block(f"FINAL STORY ({_wc(final)}w)", final)
    return {"story": final, "premise": chosen, "axes": axes, "direction": direction.model_dump(),
            "score": res.score}


# ============================================================================
# STAGE B — SCRIPT
# ============================================================================

def build_blueprint(story: str) -> Blueprint:
    transcript = []
    for r in range(LOOPS["brief_rounds"]):
        qs = call_struct(model_for("interview_amateur"), BRIEF_AMATEUR_SYS,
                         "STORY:\n" + story + ("\n\nPRIOR:\n" + "\n".join(transcript) if transcript else ""),
                         Questions, temperature=0.7, max_tokens=700)
        ans = call_text(model_for("interview_expert"), BRIEF_DIRECTOR_SYS,
                        "STORY:\n" + story + "\n\nQUESTIONS:\n" + "\n".join(qs.questions), temperature=0.5, max_tokens=1000)
        transcript.append("Q: " + " ".join(qs.questions) + "\nA: " + ans)
    return call_struct(model_for("interview_expert"),
                       "Summarize the decisions into the blueprint fields. Be concrete and spoken.",
                       "STORY:\n" + story + "\n\nDISCUSSION:\n" + "\n\n".join(transcript),
                       Blueprint, temperature=0.4, max_tokens=1500)


def draft_script(story: str, bp: Blueprint) -> str:
    sys = SCRIPT_DRAFTER_SYS.format(lo=CONFIG["script_len"]["low"], hi=CONFIG["script_len"]["high"])
    user = (SCRIPT_GOAL + "\n\n" + good_script_block() + "\n\nSTORY:\n" + story + "\n\nBLUEPRINT:\n" + json.dumps(bp.model_dump(), indent=2) +
            "\n\nWrite the spoken narration now.")
    return call_text(model_for("script_drafter"), sys, user, **SAMPLING["create"], max_tokens=1800)


def revise_script(draft, rf: Reconciled):
    if not rf.changes:
        return draft
    return call_text(model_for("script_reviser"), SCRIPT_REVISER_SYS,
                     SCRIPT_GOAL + "\n\nCURRENT NARRATION:\n" + draft + "\n\nRECONCILED FEEDBACK:\n" + _fmt_reconciled(rf),
                     **SAMPLING["revise"], max_tokens=1800)


def showrunner_script(best_draft, verdicts):
    notes = ("\n\nPANEL NOTES:\n" + json.dumps(
        {lane: {"score": c.score, "issues": [i.model_dump() for i in c.issues]} for lane, c in verdicts.items()}, indent=2)) if verdicts else ""
    return _ok_or(best_draft, call_text(model_for("script_showrunner"), SCRIPT_SHOWRUNNER_SYS,
                     SCRIPT_GOAL + "\n\nBEST DRAFT:\n" + best_draft + notes, **SAMPLING["finalize"], max_tokens=1800))


def enforce_script_length(text: str):
    lo, hi = CONFIG["script_len"]["low"], CONFIG["script_len"]["high"]
    wc = _wc(text)
    if lo <= wc <= hi:
        return text, wc, "in-band"
    action = f"Tighten it to {lo}–{hi} words." if wc > hi else f"Expand it slightly to {lo}–{hi} words."
    fixed = call_text(model_for("script_showrunner"), LENGTH_FIX_SYS,
                      f"CURRENT ({wc} words):\n{text}\n\n{action}", **SAMPLING["finalize"], max_tokens=1500)
    return fixed, _wc(fixed), f"adjusted from {wc}w"


def guard_script_names(text: str) -> str:
    return _ok_or(text, call_text(model_for("script_showrunner"), SCRIPT_NAME_GUARD_SYS,
                  SCRIPT_GOAL + "\n\nNARRATION:\n" + text, **SAMPLING["finalize"], max_tokens=1800))


def run_script(story: str) -> dict:
    set_stage("stage_b")
    _log_stage("STAGE B — SCRIPT")
    _log_step("brief (hook / keep-cut / rhythm)")
    bp = build_blueprint(story)
    _log_block("blueprint", json.dumps(bp.model_dump(), indent=2))
    _log_step("draft")
    script = draft_script(story, bp)
    _log_block(f"draft ({_wc(script)}w)", script)
    _log_step("critique ⇄ refine")
    cl = CONFIG["script_len"]
    gold = good_script_block()
    from .convergence import _length_note
    payload = lambda d: (
        SCRIPT_GOAL + "\n\n" + _length_note(d, cl["low"], cl["high"], cl["target"])
        + "\n\nSOURCE STORY (context — the script is a remix of this; use it only to check the core isn't contradicted):\n" + story
        + "\n\n" + gold
        + "\n\n---\nNARRATION (the piece you are judging):\n" + d + "\n---")
    res = converge(script, SCRIPT_CRITICS, payload, revise_script, max_loops=LOOPS["script_refine"], label="script")
    _log_step("showrunner + length guard")
    draft_wc = _wc(res.best)
    all_satisfied = bool(res.verdicts) and all(v.satisfied for v in res.verdicts.values())
    if all_satisfied and cl["low"] <= draft_wc <= cl["high"]:
        final, wc, how = res.best, draft_wc, "kept as-is"
        _log_kv("showrunner", "draft satisfied every lane and is in-band — kept verbatim (no rewrite)")
    else:
        final = showrunner_script(res.best, res.verdicts)
        final, wc, how = enforce_script_length(final)
    # ALWAYS run a first-time-listener NAME check.
    guarded = guard_script_names(final)
    if guarded.strip() != final.strip():
        final, wc, _ = enforce_script_length(guarded)
        how = how + " + name-intro fixed"
        _log_kv("name guard", "introduced a name a later edit had left unexplained")
    _log_kv("length", f"{wc} words ({how})")
    _log_block(f"FINAL SCRIPT ({wc}w)", final)
    return {"script": final, "blueprint": bp, "words": wc, "score": res.score}


# ============================================================================
# STAGE C — SCENES & PROMPTS
# ============================================================================

# ── Extractor ─────────────────────────────────────────────────────────────────

def extract_sheets(story, script) -> ExtractResult:
    user = "STORY:\n" + story + "\n\nNARRATION SCRIPT:\n" + script
    return call_struct(model_for("extractor"), EXTRACTOR_SYS, user, ExtractResult,
                       temperature=0.2, max_tokens=6000)


# ── Segmenter ─────────────────────────────────────────────────────────────────

_CONJ = {"and", "then", "but", "so", "which", "who", "because", "while", "when", "yet", "or", "nor", "as"}
_PUNCT = set(".,;:!?—–")


def _token_spans(text):
    spans, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace(): i += 1
        if i >= n: break
        j = i
        while j < n and not text[j].isspace(): j += 1
        spans.append((i, j)); i = j
    return spans


def _atomic_units(script, spans):
    """Deterministically split the script into small clause-level units at sentence/clause boundaries."""
    ends, n = [], len(spans)
    for i in range(n):
        tok = script[spans[i][0]:spans[i][1]]
        after_punct = tok.rstrip("'\"”’)").endswith(tuple(_PUNCT))
        nxt = script[spans[i + 1][0]:spans[i + 1][1]].strip(",.;:!?'\"").lower() if i + 1 < n else ""
        if after_punct or (nxt in _CONJ):
            c = i + 1
            if not ends or ends[-1] != c: ends.append(c)
    if not ends or ends[-1] != n: ends.append(n)
    texts, start = [], 0
    for e in ends:
        texts.append(script[spans[start][0]:spans[e - 1][1]].strip()); start = e
    return ends, texts


def _verify_lossless(script, segments):
    orig = " ".join(script.split())
    recon = " ".join(" ".join(s.narration.split()) for s in segments)
    return orig == recon, orig, recon


def _divergence_hint(orig, recon):
    o, r = orig.split(), recon.split()
    for i in range(min(len(o), len(r))):
        if o[i] != r[i]:
            a = " ".join(o[max(0, i - 3):i + 4])
            return 'near "' + a + '" the script says "' + o[i] + '" but you wrote "' + r[i] + '"'
    if len(o) > len(r): return 'you dropped text after "' + " ".join(r[-5:]) + '"'
    if len(r) > len(o): return 'you added text not in the script: "' + " ".join(r[len(o):len(o) + 6]) + '"'
    return "the beats do not reconstruct the script exactly"


_SENTINEL = re.compile(r"\|\s*\|\s*\|+")


def _segments_from_marked(raw: str) -> SegmentResult:
    txt = (raw or "").strip()
    if "```" in txt:
        txt = max(txt.split("```"), key=len).strip()
    pieces = [p.strip() for p in _SENTINEL.split(txt) if p.strip()]
    if not pieces:
        pieces = [txt] if txt else []
    return SegmentResult(segments=[Segment(id=i + 1, narration=p) for i, p in enumerate(pieces)])


def _cuts_from_segments(script, spans, segments):
    orig_toks = [script[a:b] for a, b in spans]
    seg_counts = [max(1, len(s.narration.split())) for s in segments] or [len(orig_toks)]
    model_toks = [t for s in segments for t in s.narration.split()]
    m2o = {}
    for ai, bi, size in difflib.SequenceMatcher(a=model_toks, b=orig_toks, autojunk=False).get_matching_blocks():
        for k in range(size): m2o[ai + k] = bi + k
    cuts, consumed, last = [], 0, 0
    for cnt in seg_counts:
        consumed += cnt; end = None
        for idx in range(consumed - 1, -1, -1):
            if idx in m2o: end = m2o[idx] + 1; break
        if end is None or end <= last: end = last + cnt
        end = max(last + 1, min(len(orig_toks), end))
        guard = 0
        while end < len(orig_toks) and guard < 12 and orig_toks[end - 1].rstrip("'\"”’)")[-1:] not in _PUNCT:
            end += 1; guard += 1
        cuts.append(end); last = end
    if cuts: cuts[-1] = len(orig_toks)
    return [c for c in cuts if c > 0]


def _segments_from_cuts(script, spans, cuts):
    out, start = [], 0
    for end in cuts:
        end = max(start + 1, min(len(spans), end))
        if start >= len(spans): break
        piece = script[spans[start][0]:spans[end - 1][1]].strip()
        if piece: out.append(piece)
        start = end
    return [Segment(id=i + 1, narration=t) for i, t in enumerate(out)]


def _best_split(script, spans, start, end):
    mid = (start + end) // 2
    best, bestd = None, 10 ** 9
    for k in range(start + 1, end):
        tok = script[spans[k - 1][0]:spans[k - 1][1]]
        if tok.rstrip("'\"”’)")[-1:] in _PUNCT:
            d = abs(k - mid)
            if d < bestd: best, bestd = k, d
    return best if best is not None else (mid if start < mid < end else start + 1)


def _count_fix(script, spans, cuts, lo, hi, target=None):
    ntok = len(spans)
    cuts = sorted({c for c in cuts if 1 <= c <= ntok})
    if not cuts or cuts[-1] != ntok: cuts.append(ntok); cuts = sorted(set(cuts))
    lo = min(lo, ntok)
    tgt = max(lo, min(hi, target if target else lo)); tgt = min(tgt, ntok)
    guard = 0
    while len(cuts) < tgt and guard < 2000:
        guard += 1
        bounds, p = [], 0
        for c in cuts: bounds.append((p, c)); p = c
        ci = max(range(len(bounds)), key=lambda i: bounds[i][1] - bounds[i][0])
        s, e = bounds[ci]
        if e - s < 2: break
        cuts = sorted(set(cuts + [_best_split(script, spans, s, e)]))
    guard = 0
    while len(cuts) > hi and len(cuts) > 1 and guard < 2000:
        guard += 1
        bounds, p = [], 0
        for c in cuts: bounds.append((p, c)); p = c
        bt, bc = None, 10 ** 9
        for t in range(len(cuts) - 1):
            comb = (bounds[t][1] - bounds[t][0]) + (bounds[t + 1][1] - bounds[t + 1][0])
            if comb < bc: bc, bt = comb, t
        if bt is None: break
        del cuts[bt]
    return cuts


def _limit_lengths(script, spans, segments, min_w=5, max_w=24):
    counts = [max(1, len(s.narration.split())) for s in segments]
    cuts, acc = [], 0
    for c in counts: acc += c; cuts.append(acc)
    ntok = len(spans)
    cuts = sorted({c for c in cuts if 0 < c <= ntok})
    if not cuts or cuts[-1] != ntok: cuts.append(ntok); cuts = sorted(set(cuts))
    guard, changed = 0, True
    while changed and guard < 2000:
        guard += 1; changed = False
        bounds, p = [], 0
        for c in cuts: bounds.append((p, c)); p = c
        for (s, e) in bounds:
            if (e - s) > max_w:
                k = _best_split(script, spans, s, e)
                if s < k < e and k not in cuts:
                    cuts = sorted(set(cuts + [k])); changed = True; break
    guard, changed = 0, True
    while changed and guard < 2000 and len(cuts) > 1:
        guard += 1; changed = False
        bounds, p = [], 0
        for c in cuts: bounds.append((p, c)); p = c
        for i, (s, e) in enumerate(bounds):
            if (e - s) < min_w:
                if i == 0:
                    del cuts[0]
                elif i == len(bounds) - 1:
                    del cuts[-2]
                else:
                    left = bounds[i-1][1]-bounds[i-1][0]; right = bounds[i+1][1]-bounds[i+1][0]
                    if left <= right: del cuts[i-1]
                    else: del cuts[i]
                changed = True; break
    cuts = sorted({c for c in cuts if 0 < c <= ntok})
    if not cuts or cuts[-1] != ntok: cuts.append(ntok); cuts = sorted(set(cuts))
    return _segments_from_cuts(script, spans, cuts)


def segment_dynamic(script, lo, hi, attempts, wps=8):
    spans = _token_spans(script)
    unit_ends, unit_texts = _atomic_units(script, spans)
    m, ntok = len(unit_ends), len(spans)
    target = max(lo, min(hi, round(len(script.split()) / max(1, wps))))

    def _assemble(chosen):
        valid = sorted({int(k) for k in (chosen or []) if isinstance(k, (int, float)) and 1 <= int(k) < m})
        cuts = sorted({unit_ends[k - 1] for k in valid} | {unit_ends[-1]})
        cuts = _count_fix(script, spans, cuts, lo, hi, target)
        segs = _limit_lengths(script, spans, _segments_from_cuts(script, spans, cuts), 5, 24)
        for i, s in enumerate(segs, 1): s.id = i
        return SegmentResult(segments=segs)

    menu = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(unit_texts))
    best = None
    for attempt in range(max(1, attempts)):
        try:
            cr = call_struct(model_for("segmenter"), SEGMENTER_SYS_TMPL.format(lo=lo, hi=hi, target=target),
                             "SCRIPT UNITS (in order):\n" + menu +
                             f"\n\nGroup these {m} units into about {target} beats (range {lo}-{hi}). "
                             "Return JSON with the unit numbers that END a beat.",
                             CutResult, temperature=0.0, max_tokens=1200)
            chosen = cr.cuts
        except Exception as e:
            _log_kv(f"segment attempt {attempt + 1}", f"model error ({type(e).__name__}) -> deterministic cuts")
            chosen = []
        segs = _assemble(chosen)
        n = len(segs.segments)
        ok = _verify_lossless(script, segs.segments)[0]
        _log_kv(f"segment attempt {attempt + 1}",
                f"{len(chosen)} cuts -> {n} beats · " + ("verbatim ✓" if ok else "MISMATCH") +
                (" · in band ✓" if lo <= n <= hi else f" · out of band ({lo}-{hi})"))
        if ok and lo <= n <= hi:
            return segs, ("indexed cuts" if attempt == 0 else f"indexed cuts (retry {attempt})")
        if best is None: best = segs
    segs = best if best is not None else _assemble([])
    ok2 = _verify_lossless(script, segs.segments)[0]
    return segs, (f"deterministic split -> {len(segs.segments)} beats" if ok2 else "MISMATCH")


# ── Style selector ────────────────────────────────────────────────────────────

def select_style(sheets: ExtractResult) -> StyleResult:
    rng = random.Random()
    pool = STYLE_POOL[:]; rng.shuffle(pool)
    offer = pool[:6]
    menu = "\n".join("- " + s["id"] + ": " + s["anchor"] for s in offer)
    user = ("STYLE MENU (choose exactly one style_id from THIS list):\n" + menu +
            "\n\nSTORY MOOD: " + sheets.overall_mood + "\nERA: " + sheets.era +
            "\n\nReminder: any of these can carry a dark story — do NOT default to black-and-white or woodcut; "
            "pick the freshest strong fit and favour colour/variety when it still suits the mood.")
    style = call_struct(model_for("style_selector"), STYLE_SYS, user, StyleResult, temperature=0.95, max_tokens=2500)
    offered = {s["id"]: s for s in offer}
    if style.style_id not in offered:
        base = rng.choice(offer)
        style.style_id, style.style_anchor = base["id"], base["anchor"]
        if not style.palette_hex: style.palette_hex = base["palette_hex"]
    elif not style.palette_hex:
        style.palette_hex = offered[style.style_id]["palette_hex"]
    return style


# ── Scene planner ─────────────────────────────────────────────────────────────

def _fallback_plan(segs: SegmentResult, sheets: ExtractResult) -> ScenePlan:
    setting = sheets.settings[0].id if sheets.settings else "set_main"
    names = [(c.name, c.id) for c in (sheets.characters or [])]
    frames = ["wide establishing shot", "medium shot", "close-up", "low angle medium shot", "insert detail close-up"]
    scenes = []
    for i, s in enumerate(segs.segments):
        present = [cid for nm, cid in names if nm and nm.lower() in s.narration.lower()]
        scenes.append(Scene(id=s.id, narration=s.narration,
                            visual_brief=("A single still image depicting: " + s.narration),
                            framing=frames[i % len(frames)],
                            characters_present=present, setting_id=setting,
                            continuity_note="auto: keep characters, palette and style consistent with neighbours"))
    return ScenePlan(scenes=scenes)


def plan_scenes(story, script, segs: SegmentResult, sheets: ExtractResult, style: StyleResult) -> ScenePlan:
    user = ("STORY:\n" + story + "\n\nFULL SCRIPT:\n" + script +
            "\n\nNARRATION BEATS (one scene each, keep id + narration):\n" + segs.model_dump_json(indent=2) +
            "\n\nCHARACTER & SETTING SHEETS:\n" + sheets.model_dump_json(indent=2) +
            "\n\nVISUAL STYLE:\n" + style.style_anchor)
    try:
        return call_struct(model_for("scene_planner"), PLANNER_SYS, user, ScenePlan,
                           temperature=0.8, min_p=0.05, max_tokens=14000)
    except Exception as e:
        _log_kv("plan fallback", f"planner could not return valid JSON ({type(e).__name__}) — "
               f"building a deterministic plan from the {len(segs.segments)} beats so the run continues")
        return _fallback_plan(segs, sheets)


# ── Plan judge / updater ──────────────────────────────────────────────────────

def judge_plan(story, script, plan: ScenePlan, prev_issues=None, style=None) -> PlanVerdict:
    style_ctx = ("\n\nVISUAL STYLE (2D, non-photorealistic — the plan should suit this hand-made look):\n" + style.style_anchor) if style else ""
    user = "STORY:\n" + story + "\n\nSCRIPT:\n" + script + style_ctx + "\n\nPLANNED SCENES:\n" + plan.model_dump_json(indent=2)
    if prev_issues:
        user += "\n\nISSUES LAST ROUND (don't re-raise resolved):\n" + json.dumps([i.model_dump() for i in prev_issues])
    return call_struct(model_for("plan_judge"), PLAN_JUDGE_SYS, user, PlanVerdict, **SAMPLING["judge"], max_tokens=5000)


def update_plan(story, script, plan: ScenePlan, sheets, style, issues) -> ScenePlan:
    user = ("STORY:\n" + story + "\n\nSCRIPT:\n" + script + "\n\nCURRENT SCENES:\n" + plan.model_dump_json(indent=2) +
            "\n\nSHEETS:\n" + sheets.model_dump_json(indent=2) + "\n\nSTYLE:\n" + style.style_anchor +
            "\n\nISSUES TO FIX (apply each suggestion):\n" + json.dumps([i.model_dump() for i in issues]))
    return call_struct(model_for("plan_updater"), PLAN_UPDATER_SYS, user, ScenePlan, temperature=0.4, max_tokens=14000)


# ── Prompt writer ─────────────────────────────────────────────────────────────

_BLUR_SUBS = [
    (re.compile(r"\bmotion[\s-]*blur(?:red)?\b", re.I), ""),
    (re.compile(r"\bsoft[\s-]*focus\b", re.I), ""),
    (re.compile(r"\b(?:shallow\s+)?depth\s+of\s+field\b", re.I), ""),
    (re.compile(r"\bbokeh\b", re.I), ""),
    (re.compile(r"\bout\s+of\s+focus\b", re.I), "indistinct"),
    (re.compile(r"\bdefocused\b", re.I), "indistinct"),
    (re.compile(r"\bblurred\b", re.I), "faded"),
    (re.compile(r"\bblurry\b", re.I), "faded"),
    (re.compile(r"\bblur\b", re.I), "haze"),
]
_WS = re.compile(r"\s{2,}")


def _descrub(text: str):
    if not text: return text, 0
    n = 0
    for rx, rep in _BLUR_SUBS:
        text, k = rx.subn(rep, text); n += k
    if n: text = _WS.sub(" ", text).replace(" ,", ",").replace(" .", ".").strip()
    return text, n


def descrub_blur(prompts: PromptBatch) -> int:
    total = 0
    for p in prompts.prompts:
        s = p.structured
        for f in ("subject", "action", "style", "setting", "lighting_mood", "palette", "composition"):
            v, k = _descrub(getattr(s, f)); setattr(s, f, v); total += k
        p.flat_prompt, k = _descrub(p.flat_prompt); total += k
    return total


def _compact_plan(plan: ScenePlan):
    return json.dumps([{"id": s.id, "brief": s.visual_brief, "chars": s.characters_present, "setting": s.setting_id} for s in plan.scenes])


def _present_people(scene: Scene, sheets: ExtractResult) -> str:
    by_id = {c.id: c for c in (sheets.characters or [])}
    lines = []
    for cid in scene.characters_present:
        c = by_id.get(cid)
        if c:
            line = "- " + c.visual_descriptor
            if c.signature_clothing:
                line += " | default wardrobe: " + c.signature_clothing + " (use unless this scene clearly calls for different clothing)"
            lines.append(line)
    if not lines:
        return "(no recurring named character here — render any figures generically, by appearance only)"
    return "\n".join(lines)


def _present_setting(scene: Scene, sheets: ExtractResult) -> str:
    by_id = {s.id: s for s in (sheets.settings or [])}
    st = by_id.get(scene.setting_id)
    if not st:
        return "(no fixed setting here — render the place generically, by description only)"
    return (st.description or "").strip()


def write_one_prompt(scene: Scene, plan: ScenePlan, sheets: ExtractResult, style: StyleResult,
                     story: str = "", script: str = "") -> PromptItem:
    bg = ""
    if story or script:
        bg = ("BACKGROUND — THE STORY (context only; depict THIS scene, not the whole story):\n" + story +
              "\n\nBACKGROUND — THE FULL NARRATION SCRIPT (context only):\n" + script + "\n\n")
    user = (bg + "FLUX.2 GUIDE:\n" + FLUX2_CARD + "\n\nVISUAL STYLE ANCHOR (2D, strictly NON-photorealistic — begin every prompt with this, keep it IDENTICAL in every scene, and never use photographic/camera words):\n" + style.style_anchor +
            "\nPALETTE (use only these hexes): " + ", ".join(style.palette_hex) +
            "\n\nCHARACTER & SETTING SHEETS (visual descriptions only, never the name field):\n" + sheets.model_dump_json(indent=2) +
            "\n\nPEOPLE IN THIS SCENE (describe EACH one ONLY by this appearance, NEVER by a name; if several appear, "
            "make each visually distinct and state where each is and who faces whom):\n" + _present_people(scene, sheets) +
            "\n\nTHE PLACE IN THIS SCENE (paint it with these exact words, never its name):\n" + _present_setting(scene, sheets) +
            "\n\nWHOLE PLAN (for continuity only):\n" + _compact_plan(plan) +
            "\n\n>>> WRITE THE PROMPT THAT DEPICTS THIS SCENE (use its visual_brief as the subject):\n" + scene.model_dump_json(indent=2))
    item = call_struct(model_for("prompt_writer"), PROMPT_WRITER_SYS, user, PromptItem, temperature=0.7, min_p=0.04, max_tokens=2200)
    item.id = scene.id
    return item


def write_prompts_parallel(plan: ScenePlan, sheets, style, workers=4, story="", script="") -> PromptBatch:
    scenes = plan.scenes; results = [None] * len(scenes)
    def job(ix): i, sc = ix; return i, write_one_prompt(sc, plan, sheets, style, story, script)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, item in ex.map(job, list(enumerate(scenes))): results[i] = item
    return PromptBatch(prompts=results)


# ── Prompt reviewer / rewriter ────────────────────────────────────────────────

def review_one_prompt(item, scene, briefs, sheets, style, prev_fixes_for=None, script=""):
    sc_ctx = ("\n\nFULL NARRATION SCRIPT (context for what the scenes, together, depict):\n" + script) if script else ""
    user = ("STYLE ANCHOR:\n" + style.style_anchor + "\n\nSHEETS:\n" + sheets.model_dump_json(indent=2) + sc_ctx +
            "\n\nALL SCENE BRIEFS (context only — you judge ONLY the one prompt below):\n" + briefs +
            "\n\nTHIS SCENE'S BRIEF (the prompt must depict this):\n" + scene.visual_brief +
            "\n\nTHE ONE PROMPT YOU ARE JUDGING (id " + str(item.id) + "):\n" + item.model_dump_json(indent=2))
    if prev_fixes_for:
        user += "\n\nTHIS PROMPT'S FIXES LAST ROUND (don't re-raise resolved):\n" + json.dumps([f.model_dump() for f in prev_fixes_for])
    try:
        v = call_struct(model_for("prompt_reviewer"), PROMPT_REVIEWER_SYS, user, PromptVerdict, **SAMPLING["judge"], max_tokens=1500)
    except Exception as e:
        _log_kv("review abstain", f"prompt {item.id} ({type(e).__name__}) — passing this prompt this round")
        return PromptVerdict(passed=True, fixes=[], overall_note="abstain")
    for f in v.fixes:
        f.prompt_id = item.id
    if (not v.passed) and (not v.fixes):
        return PromptVerdict(passed=True, fixes=[], overall_note="no concrete issue")
    return v


def review_prompts(prompts: PromptBatch, plan: ScenePlan, sheets, style, prev_fixes=None, script="", workers=4, only_ids=None) -> PromptVerdict:
    briefs = json.dumps([{"id": s.id, "brief": s.visual_brief} for s in plan.scenes])
    scene_by_id = {s.id: s for s in plan.scenes}
    prev_by = {}
    for f in (prev_fixes or []):
        prev_by.setdefault(f.prompt_id, []).append(f)
    items = prompts.prompts
    if only_ids is not None:
        idset = set(only_ids)
        items = [it for it in items if it.id in idset]
    def job(it):
        sc = scene_by_id.get(it.id)
        if sc is None:
            return PromptVerdict(passed=True, fixes=[], overall_note="no scene")
        return review_one_prompt(it, sc, briefs, sheets, style, prev_by.get(it.id), script)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        verdicts = list(ex.map(job, items))
    all_fixes = [f for v in verdicts for f in v.fixes]
    passed = not all_fixes
    note = "all prompts clean" if passed else (str(len(all_fixes)) + " fix(es) across " + str(len({f.prompt_id for f in all_fixes})) + " prompt(s)")
    return PromptVerdict(passed=passed, fixes=all_fixes, overall_note=note)


def _rewrite_one(pid, item, scene, fixes_for, sheets, style) -> PromptItem:
    probs = [{"problem": f.problem, "suggestion": f.suggestion} for f in fixes_for]
    user = ("FLUX.2 GUIDE:\n" + FLUX2_CARD + "\n\nSTYLE ANCHOR:\n" + style.style_anchor +
            "\nPALETTE (use only these hexes): " + ", ".join(style.palette_hex) +
            "\n\nSHEETS:\n" + sheets.model_dump_json(indent=2) +
            "\n\nSCENE BRIEF (the prompt must depict this):\n" + scene.visual_brief +
            "\n\nCURRENT PROMPT:\n" + item.flat_prompt +
            "\n\nFIX THESE (apply each suggestion):\n" + json.dumps(probs, indent=2))
    fixed = call_struct(model_for("prompt_rewriter"), PROMPT_REWRITER_SYS, user, PromptItem, temperature=0.5, max_tokens=2200)
    fixed.id = pid
    return fixed


def rewrite_prompts(prompts: PromptBatch, plan: ScenePlan, sheets, style, fixes, workers=4) -> PromptBatch:
    by_id = {p.id: p for p in prompts.prompts}; scene_by_id = {s.id: s for s in plan.scenes}
    fixes_by = {}
    for f in fixes:
        fixes_by.setdefault(f.prompt_id, []).append(f)
    flagged = [pid for pid in fixes_by if pid in by_id and pid in scene_by_id]
    if not flagged:
        return prompts
    def job(pid): return pid, _rewrite_one(pid, by_id[pid], scene_by_id[pid], fixes_by[pid], sheets, style)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, fixed in ex.map(job, flagged):
            by_id[pid] = fixed
    _log_block("rewritten prompts", "\n\n".join(f"[{pid}] {by_id[pid].flat_prompt}" for pid in sorted(flagged)))
    return PromptBatch(prompts=[by_id[i] for i in sorted(by_id)])


# ── Stage C orchestrator ─────────────────────────────────────────────────────

def run_scenes(story: str, script: str, *, workers=4) -> dict:
    lo, hi = CONFIG["scenes"]["low"], CONFIG["scenes"]["high"]
    set_stage("stage_c")
    _log_stage("STAGE C — SCENES & PROMPTS")

    _log_step("extract characters & settings")
    sheets = extract_sheets(story, script)
    _log_kv("extracted", f"{len(sheets.characters)} characters, {len(sheets.settings)} settings · mood: {sheets.overall_mood}")
    _log_block("sheets", sheets.model_dump_json(indent=2))

    _log_step(f"segment (dynamic, target {lo}-{hi} beats)")
    segs, seg_status = segment_dynamic(script, lo, hi, LOOPS["segment_attempts"], CONFIG["scenes"]["words_per_scene"])
    _log_kv("segmentation", f"{len(segs.segments)} beats · {seg_status}")
    _log_block("beats", "\n".join(f"[{s.id}] {s.narration}" for s in segs.segments))

    _log_step("select visual style")
    style = select_style(sheets)
    _log_kv("style", style.style_id)
    _log_block("style", style.model_dump_json(indent=2))

    _log_step("plan scenes")
    plan = plan_scenes(story, script, segs, sheets, style)
    _log_kv("planned", f"{len(plan.scenes)} scenes")
    _log_block("scene plan", plan.model_dump_json(indent=2))

    _log_step(f"plan judge ⇄ update (≤{LOOPS['plan_judge']})")
    plan = _keep_best_loop("plan", LOOPS["plan_judge"],
                           lambda p, prev: judge_plan(story, script, p, prev, style=style),
                           lambda p, issues: update_plan(story, script, p, sheets, style, issues),
                           lambda: plan)

    _log_step("write prompts (parallel)")
    prompts = write_prompts_parallel(plan, sheets, style, workers=workers, story=story, script=script)
    _blur = descrub_blur(prompts)
    _log_kv("prompts", f"{len(prompts.prompts)} written" + (f" · stripped {_blur} focus/blur term(s)" if _blur else ""))
    _log_block("prompts", "\n\n".join(f"[{p.id}] {p.flat_prompt}" for p in prompts.prompts))

    _log_step(f"prompt review ⇄ rewrite (≤{LOOPS['prompt_review']}; re-checks only what it rewrites)")
    to_check, prev_fixes = None, None
    for _it in range(LOOPS["prompt_review"]):
        verdict = review_prompts(prompts, plan, sheets, style, prev_fixes, script=script, workers=workers, only_ids=to_check)
        fixes = verdict.fixes
        if not fixes:
            _log_kv(f"prompt iter {_it + 1}", "✓ all clean"); break
        _log_kv(f"prompt iter {_it + 1}", f"{len(fixes)} issue(s) on {len({f.prompt_id for f in fixes})} prompt(s)")
        _log_issues(fixes, "prompt")
        prompts = rewrite_prompts(prompts, plan, sheets, style, fixes, workers=workers)
        to_check = sorted({f.prompt_id for f in fixes})
        prev_fixes = fixes
    else:
        _log_kv("prompt", "reached cap")
    descrub_blur(prompts)

    pmap = {p.id: p for p in prompts.prompts}
    seg_narr = {s.id: s.narration for s in segs.segments}
    scenes = []
    for i, sc in enumerate(plan.scenes):
        p = pmap.get(sc.id)
        scenes.append({
            "index": i + 1,
            "narration": seg_narr.get(sc.id, sc.narration),
            "prompt": (p.flat_prompt if p else ""),
            "shot": sc.framing,
            "visual_brief": sc.visual_brief,
            "characters_present": sc.characters_present,
            "setting_id": sc.setting_id,
            "structured": (p.structured.model_dump() if p else {}),
        })
    verbatim_ok = " ".join(script.split()) == " ".join(" ".join(s["narration"].split()) for s in scenes)
    _log_kv("final scenes", f"{len(scenes)} · narration reconstructs script: {verbatim_ok}")
    return {"scenes": scenes, "style": style.model_dump(), "sheets": sheets.model_dump(),
            "seg_status": seg_status, "verbatim_ok": verbatim_ok, "n_scenes": len(scenes)}


# ============================================================================
# FINAL ASSEMBLY
# ============================================================================

from .config import ROLE_MODELS

def build_final(story_res, script_res, scene_res, *, aspect):
    p = story_res["premise"]
    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "genre": CONFIG["genre"],
            "platform": CONFIG["platform"],
            "models": dict(ROLE_MODELS),
            "variety_axes": story_res["axes"],
            "creative_direction": story_res.get("direction"),
            "premise": {"logline": p.logline, "spine": p.spine},
            "scores": {"story": round(story_res["score"], 2), "script": round(script_res["score"], 2)},
            "counts": {"story_words": _wc(story_res["story"]),
                       "script_words": script_res["words"], "scenes": scene_res["n_scenes"]},
            "usage": METER.snapshot(),
            "style": scene_res["style"],
            "characters": scene_res["sheets"].get("characters", []),
            "settings": scene_res["sheets"].get("settings", []),
            "flux2": {"model": "black-forest-labs/FLUX.2-klein", "prompt_style": "plain prose, <512 tokens",
                      "negative_prompts": False, "text_encoder": "Qwen3", "aspect_ratio": aspect,
                      "resolution": DIMS.get(aspect, [752, 1328]),
                      "note": "render on your 5090 with the FLUX.2 generator notebook; choose image seeds there"},
            "narration_verbatim": scene_res["seg_status"],
            "narration_reconstructs_script": scene_res["verbatim_ok"],
        },
        "story": story_res["story"],
        "script": script_res["script"],
        "scenes": scene_res["scenes"],
    }


def run_all(*, aspect="9:16", workers=4) -> dict:
    """One independent, stateless run: a fresh story → spoken script → scene-by-scene prompts."""
    METER.reset()
    _log_stage(f"RUN · {CONFIG['genre'][:50]}")
    story_res = run_story()
    script_res = run_script(story_res["story"])
    scene_res = run_scenes(story_res["story"], script_res["script"], workers=workers)
    final = build_final(story_res, script_res, scene_res, aspect=aspect)
    _log_stage("DONE")
    c = final["meta"]["counts"]
    _log_kv("story words", c["story_words"])
    _log_kv("script words", c["script_words"])
    _log_kv("scenes", f"{c['scenes']} (band {CONFIG['scenes']['low']}-{CONFIG['scenes']['high']})")
    _log_kv("narration verbatim", final["meta"]["narration_reconstructs_script"])
    _log_kv("scores", final["meta"]["scores"])
    logger.info("Usage: %s", json.dumps(METER.snapshot(), indent=2))
    return final


# ── Filename helpers ──────────────────────────────────────────────────────────

def _slugify(text, maxlen=48):
    out = []
    for ch in (text or "").strip().lower():
        if ch.isalnum(): out.append(ch)
        elif (ch == " " or ch in "-_/\\\t.,:;") and out and out[-1] != "-": out.append("-")
    return "".join(out).strip("-")[:maxlen].strip("-") or "reel"


def unique_output_path(out_dir, final, prefix="ch_reel"):
    title = ((final.get("meta", {}).get("premise", {}) or {}).get("logline", "") or final.get("story", ""))
    base = os.path.join(out_dir, prefix + "__" + _slugify(" ".join(title.split()[:6])) + "__" + time.strftime("%Y%m%d-%H%M%S"))
    path, i = base + ".json", 2
    while os.path.exists(path):
        path, i = base + "-" + str(i) + ".json", i + 1
    return path

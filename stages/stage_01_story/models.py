"""
Content Factory — Pydantic schemas (all stages).

Every schema is extracted verbatim from the notebook.
Field names, types, defaults, and comments are preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, ConfigDict


# ── Base ──────────────────────────────────────────────────────────────────────

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Stage A — Story ───────────────────────────────────────────────────────────

class AxesUsed(Strict):
    era: str; region: str; domain: str; milieu: str
    motif: str; flavor: str; structure: str; telling_register: str


class Direction(Strict):
    title: str
    era: str                 # resolved, coherent
    place: str               # specific, fits the era
    premise_space: str       # 2-3 sentences: the world + the KIND of dark/plausible event (texture, not a plot)


class ConceptOut(Strict):
    axes: AxesUsed
    direction: Direction


class Premise(Strict):
    id: str
    logline: str            # one vivid sentence
    spine: str              # person · pressure · line crossed · turn · ending
    probability: float      # 0–1: how likely a typical writer is to pick this (VS)


class PremiseSet(Strict):
    premises: List[Premise]


class ScoredPremise(Strict):
    id: str
    engaging: int
    unique: int
    retellable: int
    on_genre: int
    visualizable: int
    overall: float


class PremiseRanking(Strict):
    ranked: List[ScoredPremise]
    pick_id: str
    why: str


class Questions(Strict):
    questions: List[str]


class AmateurTurn(Strict):
    thread_focus: str       # the part of the spine being nailed this round (round 1 = "building the story spine")
    questions: List[str]    # this round's questions about EVENTS (what happens & why); empty ONLY if done
    done: bool              # True as soon as there is ENOUGH rich material to write a really good story (leaving early is good)


class Issue(Strict):
    problem: str            # what's wrong, one specific sentence
    quote: str              # the EXACT phrase/sentence from the draft (verbatim); "" only if purely structural
    fix: str                # concrete BEFORE→AFTER illustration of the direction (never a line to paste)


class Critique(Strict):
    score: int              # 1–10 on THIS critic's lane (8–10 strong, 5–7 real issues, 1–4 broken)
    satisfied: bool         # the real gate: good enough on this lane (≈ score ≥ 8); don't withhold over a nitpick
    issues: List[Issue]     # ≤3; if satisfied, only tiny polish or EMPTY; a LOW score MUST name ≥1 concrete issue


class Reconciled(Strict):
    summary: str            # one line: how close, and the through-line of what's left
    changes: List[str]      # ≤5 prioritized, non-contradictory concrete changes (most important first)
    keep: List[str]         # ≤4 things working that MUST be preserved so the reviser doesn't break them


class Blueprint(Strict):
    hook: str               # the spoken first line (payoff-first, ~4–10 words)
    keep: str
    compress: str
    cut: str
    rhythm: str             # spoken cadence notes
    cta: str
    ending: str


# ── Stage C — Scenes & Prompts ────────────────────────────────────────────────

class Character(Strict):
    id: str; name: str
    visual_descriptor: str            # ONE concise, clothing-FREE identity (age, build, face, hair, marks) — constant every scene
    age_range: str
    signature_clothing: str           # DEFAULT wardrobe, kept SEPARATE so a scene can override it without touching identity
    palette_hex: List[str]


class Setting(Strict):
    id: str; name: str
    description: str                  # ONE concise place + mood line (atmosphere folded in)
    palette_hex: List[str]


class ExtractResult(Strict):
    characters: List[Character]; settings: List[Setting]; era: str; overall_mood: str


class Segment(Strict):
    id: int; narration: str


class SegmentResult(Strict):
    segments: List[Segment]


class StyleResult(Strict):
    style_id: str; style_anchor: str; palette_hex: List[str]; rationale: str


class Scene(Strict):
    id: int; narration: str; visual_brief: str; framing: str
    characters_present: List[str]; setting_id: str; continuity_note: str


class ScenePlan(Strict):
    scenes: List[Scene]


class PlanIssue(Strict):
    scene_id: int; problem: str; suggestion: str


class PlanVerdict(Strict):
    passed: bool; issues: List[PlanIssue]; overall_note: str


class StructuredPrompt(Strict):
    subject: str; action: str; style: str; setting: str
    lighting_mood: str; palette: str; composition: str


class PromptItem(Strict):
    id: int; structured: StructuredPrompt; flat_prompt: str


class PromptBatch(Strict):
    prompts: List[PromptItem]


class PromptFix(Strict):
    prompt_id: int; problem: str; suggestion: str


class PromptVerdict(Strict):
    passed: bool; fixes: List[PromptFix]; overall_note: str


class CutResult(Strict):
    cuts: List[int] = []          # unit numbers that END a beat; the model emits ONLY integers, so wording can't drift


# ── Convergence result ────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    best: object
    score: float            # headline = number of lanes satisfied for the chosen draft
    verdicts: object        # {lane: Critique} for the chosen draft (text + verdicts never diverge)

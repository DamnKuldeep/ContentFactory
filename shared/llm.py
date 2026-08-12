"""
Content Factory — OpenRouter LLM client with production-grade retries.

call_text / call_struct are the notebook's logic + tenacity retry wrapping.
Every prompt, temperature, max_tokens flow is preserved exactly.
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
import threading
import time
from typing import Optional, Type, TypeVar

import tenacity
from openai import OpenAI
from pydantic import BaseModel

from .config import PRICES

logger = logging.getLogger("contentfactory.llm")

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── Per-process state (each worker initialises its own) ───────────────────────

_client_lock = threading.Lock()
_client: Optional[OpenAI] = None
_cur_stage = {"name": "—"}


def set_stage(name: str):
    _cur_stage["name"] = name


def get_client() -> OpenAI:
    if _client is None:
        raise RuntimeError("OpenAI client not initialised — call init_client() first")
    return _client


def init_client(key: str = "") -> OpenAI:
    """Build the OpenRouter client from key or env var."""
    global _client
    k = (key or "").strip() or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "No OpenRouter API key — set OPENROUTER_API_KEY environment variable "
            "or pass it via --api-key."
        )
    with _client_lock:
        _client = OpenAI(base_url=OPENROUTER_BASE, api_key=k)
    return _client


# ── Token + cost meter (per-process, thread-safe) ─────────────────────────────

class Meter:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock if hasattr(self, '_lock') else threading.Lock():
            self.in_tok = self.out_tok = self.calls = 0
            self.cached_tok = 0          # measurement-only: cached input tokens (prompt caching)
            self.cost = 0.0
            self.by_stage = {}
            self.by_model = {}

    @staticmethod
    def _cached_tokens(usage):
        """Cached input tokens, if the provider reports them (OpenAI-style or Anthropic-style)."""
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None:
            c = getattr(ptd, "cached_tokens", None)
            if c is None and isinstance(ptd, dict):
                c = ptd.get("cached_tokens")
            if c:
                return int(c)
        me = getattr(usage, "model_extra", None) or {}
        return int(me.get("cache_read_input_tokens", 0) or 0)

    def _fallback_cost(self, model, i, o):
        pin, pout = PRICES.get(model, (None, None))
        return None if pin is None else (i / 1e6 * pin + o / 1e6 * pout)

    def add(self, model, usage, stage):
        i = getattr(usage, "prompt_tokens", 0) or 0
        o = getattr(usage, "completion_tokens", 0) or 0
        cached = self._cached_tokens(usage)                          # measurement-only
        cost = getattr(usage, "cost", None)                          # OpenRouter returns real cost when asked
        if cost is None:
            cost = (getattr(usage, "model_extra", None) or {}).get("cost")
        if cost is None:
            cost = self._fallback_cost(model, i, o) or 0.0           # price-table fallback
        with self._lock:
            self.in_tok += i
            self.out_tok += o
            self.cached_tok += cached
            self.calls += 1
            self.cost += cost
            for store, key in ((self.by_stage, stage), (self.by_model, model)):
                e = store.setdefault(key, {"in": 0, "out": 0, "calls": 0, "cost": 0.0})
                e["in"] += i
                e["out"] += o
                e["calls"] += 1
                e["cost"] += cost

    def snapshot(self):
        rnd = lambda d: {k: (round(v, 4) if k == "cost" else v) for k, v in d.items()}
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.in_tok,
                "cached_input_tokens": self.cached_tok,
                "output_tokens": self.out_tok,
                "cost_usd": round(self.cost, 4),
                "by_model": {m: rnd(d) for m, d in self.by_model.items()},
                "by_stage": {s: rnd(d) for s, d in self.by_stage.items()},
            }


METER = Meter()


# ── JSON repair (verbatim from notebook) ──────────────────────────────────────

def _repair_json(raw: str) -> str:
    """Mechanically make a reply parseable: drop any prose/fence before the object, remove trailing commas,
    and close brackets/strings left open by max_tokens truncation. A no-op on already-valid output — it only
    rescues the malformed/truncated cases (the common ways a weaker model breaks JSON)."""
    s = raw.strip()
    if "{" in s:
        s = s[s.find("{"):]                       # drop anything before the first object
    s = _re.sub(r",\s*([}\]])", r"\1", s)          # ,] or ,}  ->  ] or }
    stack, in_str, esc = [], False, False         # track string state so braces inside text don't fool us
    for ch in s:
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in "{[": stack.append(ch)
        elif ch in "}]":
            if stack: stack.pop()
    if in_str: s += '"'                            # close a value truncated mid-string
    s = _re.sub(r",\s*$", "", s.rstrip())          # a dangling comma at the very end
    for ch in reversed(stack):                     # close any objects/arrays left open
        s += "}" if ch == "{" else "]"
    return s


def _coerce(raw: str, model_cls: Type[T]) -> Optional[T]:
    """Best-effort parse: whole string, the outermost {...} slice, then a mechanically-repaired version
    (trailing commas removed, truncated brackets/strings closed)."""
    cands = [raw,
             raw[raw.find("{"):raw.rfind("}") + 1] if "{" in raw and "}" in raw else "",
             _repair_json(raw)]
    for cand in cands:
        cand = (cand or "").strip()
        if not cand:
            continue
        try:
            return model_cls.model_validate_json(cand)
        except Exception:
            try:
                return model_cls.model_validate(json.loads(cand))
            except Exception:
                continue
    return None


# ── Retry predicate for tenacity ──────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors we should retry."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, InternalServerError):
        return True
    # Catch generic connection / DNS / timeout errors
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    # Check for HTTP status codes in string representation
    s = str(exc)
    for code in ("429", "500", "502", "503", "504"):
        if code in s:
            return True
    return False


def _extract_retry_after(exc: BaseException) -> float:
    """Try to extract retry-after seconds from a 429 response."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", {})
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                return min(float(ra), 60.0)
            # OpenRouter x-ratelimit headers
            reset = headers.get("x-ratelimit-reset")
            if reset:
                wait = float(reset) - time.time()
                if wait > 0:
                    return min(wait, 60.0)
    except Exception:
        pass
    return 0.0


def _after_retry(retry_state: tenacity.RetryCallState):
    """Log retries."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number
    if exc:
        logger.warning(
            "LLM call retry #%d after %s: %s",
            attempt, type(exc).__name__, str(exc)[:200],
        )


# ── Production call_text ──────────────────────────────────────────────────────

@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=60) + tenacity.wait_random(0, 2),
    stop=tenacity.stop_after_attempt(8),
    after=_after_retry,
    reraise=True,
)
def _call_api(model, messages, temperature, max_tokens, extra_body):
    """Single API call with tenacity retry."""
    client = get_client()
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return r


def call_text(model, system, user, *, temperature=0.7, max_tokens=1600, min_p=None,
              presence_penalty=None, retries=2, cache=None) -> str:
    if cache is None:
        from shared.config import LLM_PROMPT_CACHE
        cache = LLM_PROMPT_CACHE
    extra = {"usage": {"include": True}}      # ask OpenRouter to return the real $ cost for this call
    if min_p is not None:
        extra["min_p"] = min_p
    if presence_penalty is not None:
        extra["presence_penalty"] = presence_penalty
    last = ""
    # cache=True marks the (verbatim) system prompt as a cache breakpoint — no content/model change.
    # OpenRouter forwards cache_control to Anthropic/Gemini; OpenAI/DeepSeek/Grok auto-cache.
    if cache:
        system_msg = {"role": "system",
                      "content": [{"type": "text", "text": system,
                                   "cache_control": {"type": "ephemeral"}}]}
    else:
        system_msg = {"role": "system", "content": system}
    messages = [system_msg, {"role": "user", "content": user}]
    for _ in range(retries + 1):
        try:
            r = _call_api(model, messages, temperature, max_tokens, extra)
            METER.add(model, r.usage, _cur_stage["name"])
            content = (r.choices[0].message.content or "").strip()
            if content:
                return content
            last = "empty completion"
            time.sleep(0.8)
            continue   # retry empties (refusals/transients)
        except Exception as e:
            last = str(e)
            time.sleep(1.2)
    if last == "empty completion":
        return ""        # soft-fail on persistent empties; callers fall back to a prior good draft
    raise RuntimeError(f"call_text failed: {last}")


def call_struct(model, system, user, schema: Type[T], *, temperature=0.5, max_tokens=1600,
                min_p=None, presence_penalty=None, redos=3, cache=None) -> T:
    """Get JSON matching `schema`. Strategy: run the task; if the reply is only MALFORMED JSON, reshape it
    WITHOUT losing content (a cheap repair pass); if that fails, let the model REDO the whole task. Up to
    `redos` attempts. If it still fails, STOP with an actionable error — the chosen model is weak at
    structured output for this role and should be switched."""
    sys_j = system + ("\n\nReturn ONLY a single JSON object that is a DATA INSTANCE conforming to this schema — "
                      "real values filled in, NOT the schema itself (do NOT output 'properties', '$defs', 'type', "
                      "or 'required'; output the actual data). No prose, no markdown, no fences. Schema:\n"
                      + json.dumps(schema.model_json_schema()))
    last = ""
    for attempt in range(max(1, redos)):
        raw = call_text(model, sys_j, user, temperature=temperature, max_tokens=max_tokens,
                        min_p=min_p, presence_penalty=presence_penalty, cache=cache)
        obj = _coerce(raw, schema)
        if obj is not None:
            return obj
        # content-preserving repair: reshape THIS reply into valid JSON (nothing lost, no task redo)
        fixed = call_text(model, "Reformat the text below into ONE valid JSON object matching the schema. "
                                 "Keep ALL of its content; change only the formatting. Output JSON only, no fences.",
                          "SCHEMA:\n" + json.dumps(schema.model_json_schema()) + "\n\nTEXT:\n" + raw,
                          temperature=0.0, max_tokens=max_tokens)
        obj = _coerce(fixed, schema)
        if obj is not None:
            return obj
        last = raw
        if attempt < redos - 1:
            logger.info("json retry: %s via %s — redoing the task (%d/%d)",
                        schema.__name__, model, attempt + 2, redos)
    raise RuntimeError(
        f"{model} could not return valid JSON for {schema.__name__} after {redos} attempts. This model is likely "
        f"weak at structured output for this role — switch it to another dense model in the Run panel. "
        f"Last reply began: {last[:160]!r}")

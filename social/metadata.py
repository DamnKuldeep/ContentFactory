"""
Generate engagement metadata for a finished video: a video DESCRIPTION + HASHTAGS (+ a short
YouTube title, required by the API). Reuses shared/llm.py (OpenRouter). Runs on the laptop inside
the uploader — NOT bundled into the public web app (keeps the OpenRouter key local).

Per-video source content (title, logline, narration script) is read from the stage-1 JSON in Drive
(the `videos` row only has title + link). Tone mirrors the pipeline's niche: dark "creepy-true-story"
micro-stories for Instagram Reels / YouTube Shorts, audience 16-35.

Assembly:
  Instagram caption = description + blank line + hashtags
  YouTube           = title + description + hashtags (in the description box)
"""

import json
import os
import sys
from typing import List

_CF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CF not in sys.path:
    sys.path.insert(0, _CF)

from pydantic import BaseModel, Field

CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "meta-llama/llama-3.3-70b-instruct")
NICHE = ("Dark, eerie 'creepy-true-story' / unsolved-mystery / dark-history micro-stories for "
         "Instagram Reels & YouTube Shorts (vertical), audience 16-35. Gripping, suggestive, never "
         "graphic; feels like it COULD be real.")

SYSTEM = (
    "You write high-engagement social copy for short vertical videos. Given a short dark micro-story, "
    "produce: (1) a YouTube title that stops the scroll (<= 70 chars, no clickbait lies, no emojis); "
    "(2) a description: 2-4 short sentences that hook curiosity and tease the story WITHOUT spoiling "
    "the ending, then a one-line call-to-engage (follow for more / what would you do?); "
    "(3) 12-20 hashtags that are SPECIFIC to this story's era/place/theme plus a few reach tags — "
    "lowercase, no '#', no spaces, no duplicates, avoid generic spam like 'fyp viral foryou'. "
    f"Niche/tone: {NICHE} Return ONLY the JSON object."
)


class SocialMeta(BaseModel):
    title: str = Field(..., description="YouTube title, <=70 chars")
    description: str = Field(..., description="2-4 hook sentences + a call to engage; no spoilers")
    hashtags: List[str] = Field(..., description="12-20 specific lowercase tags, no # prefix")


def generate(title: str, logline: str, script: str) -> SocialMeta:
    """One LLM call -> SocialMeta. Requires OPENROUTER_API_KEY (raise its spend limit first)."""
    from shared.llm import init_client, call_struct
    from shared.config import OPENROUTER_API_KEY
    try:
        init_client(OPENROUTER_API_KEY)
    except Exception:
        pass
    user = (f"TITLE: {title}\nLOGLINE: {logline}\n\nNARRATION SCRIPT:\n{script.strip()[:1500]}\n\n"
            "Write the JSON now.")
    meta = call_struct(CAPTION_MODEL, SYSTEM, user, SocialMeta, temperature=0.9, max_tokens=600)
    meta.title = meta.title[:70]
    # normalize hashtags
    seen, clean = set(), []
    for h in meta.hashtags:
        h = h.lstrip("#").strip().lower().replace(" ", "")
        if h and h not in seen:
            seen.add(h); clean.append(h)
    meta.hashtags = clean[:20]
    return meta


def ig_caption(meta: SocialMeta) -> str:
    return meta.description.strip() + "\n\n" + " ".join("#" + h for h in meta.hashtags)


def yt_description(meta: SocialMeta) -> str:
    return meta.description.strip() + "\n\n" + " ".join("#" + h for h in meta.hashtags)


def load_story_content(batch_id, story_num, manifest, uploader) -> dict:
    """Fetch the stage-1 JSON from Drive to get title/logline/script for the prompt."""
    job = manifest.get_completed_job(batch_id, "stage_01_story", story_num)
    if not job or not job.get("drive_file_id") or not uploader:
        return {"title": "", "logline": "", "script": ""}
    tmp = f"/tmp/s1_{batch_id}_{story_num}.json"
    if not uploader.download_file(job["drive_file_id"], tmp):
        return {"title": "", "logline": "", "script": ""}
    with open(tmp) as f:
        d = json.load(f)
    os.remove(tmp)
    m = d.get("meta", {})
    return {
        "title": (m.get("creative_direction") or {}).get("title", "") or "",
        "logline": (m.get("premise") or {}).get("logline", "") or "",
        "script": d.get("script", "") or "",
    }


def generate_for_video(batch_id, story_num, manifest, uploader) -> SocialMeta:
    c = load_story_content(batch_id, story_num, manifest, uploader)
    return generate(c["title"], c["logline"], c["script"])

"""
Gradio review app — reviewers log in, get a random PENDING video, Approve or Reject (+reason).
Writes decisions to the Google Sheet (single source of truth) and shows live queue counts.

Deploy free on Hugging Face Spaces: bundle this file + sheet.py + requirements.txt, and set Space
secrets: SHEET_ID, GOOGLE_TOKEN_JSON (contents of token.json with Drive+Sheets scope), REVIEW_USERS.
Run locally:  ../.venv_review/bin/python review_app/app.py

REVIEW_USERS format:  "alice:pw1,bob:pw2"  (each login name is recorded as the approver/rejecter).
"""

import os
import sys

# import sheet.py whether running from the repo (social/sheet.py) or bundled next to this file (Space)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from social import sheet as S
except Exception:
    import sheet as S

import gradio as gr


def _users():
    raw = os.environ.get("REVIEW_USERS", "admin:admin").strip()
    out = {}
    for pair in raw.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            out[u.strip()] = p.strip()
    return out


def auth_fn(username, password):
    return _users().get(username) == password


def _embed(video_url):
    if not video_url:
        return "<p>No video.</p>"
    return (f'<iframe src="{video_url}" width="360" height="640" '
            f'style="border:0;border-radius:12px" allow="autoplay; fullscreen"></iframe>')


def _metrics_md():
    c = S.queue_counts()
    return (f"### Queue\n"
            f"- **Pending review:** {c['pending']}\n"
            f"- **In queue (approved):** {c['in_queue']}  ·  IG waiting {c['ig_waiting']} · YT waiting {c['yt_waiting']}\n"
            f"- **Uploaded:** {c['uploaded']}  ·  **Rejected:** {c['rejected']}  ·  Total {c['total']}")


def _load_next():
    r = S.get_pending_random()
    if not r:
        return (None, _embed(""), "**No pending videos** 🎉", _metrics_md())
    state = {"batch_id": r["batch_id"], "story_num": r["story_num"]}
    title = f"### {r.get('title') or '(untitled)'}  \n`{r['batch_id']} / story {r['story_num']}`"
    return (state, _embed(r.get("video_url")), title, _metrics_md())


def _approve(state, request: gr.Request):
    if state:
        S.set_decision(state["batch_id"], state["story_num"], "in_queue", request.username)
    return _load_next()


def _reject(state, reason, request: gr.Request):
    if state:
        S.set_decision(state["batch_id"], state["story_num"], "rejected", request.username, reason or "")
    return _load_next() + ("",)   # also clear the reason box


def build():
    with gr.Blocks(title="Video Review Queue") as demo:
        gr.Markdown("# 🎬 Video Review Queue")
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.HTML()
            with gr.Column(scale=1):
                title = gr.Markdown()
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve", variant="primary")
                    skip_btn = gr.Button("⏭️ Skip")
                reason = gr.Textbox(label="Rejection reason (optional)", placeholder="why reject?")
                reject_btn = gr.Button("❌ Reject", variant="stop")
                metrics = gr.Markdown()

        demo.load(_load_next, outputs=[state, video, title, metrics])
        approve_btn.click(_approve, inputs=[state], outputs=[state, video, title, metrics])
        skip_btn.click(_load_next, outputs=[state, video, title, metrics])
        reject_btn.click(_reject, inputs=[state, reason],
                         outputs=[state, video, title, metrics, reason])
    return demo


if __name__ == "__main__":
    demo = build()
    demo.launch(auth=auth_fn, server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", "7860")))

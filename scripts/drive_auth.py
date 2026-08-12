#!/usr/bin/env python3
"""
drive_auth.py — ONE-TIME interactive Google Drive authorization.

Run this once on a machine WITH a browser. It opens the Google consent screen and writes
token.json next to credentials.json. Then copy BOTH credentials.json and token.json to every
other machine — the pipeline there will refresh silently and NEVER prompt for login again.

    python ContentFactory/scripts/drive_auth.py
    python ContentFactory/scripts/drive_auth.py --credentials /path/credentials.json --token /path/token.json
"""

import argparse
import os
import sys

_CF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ContentFactory/
if _CF_ROOT not in sys.path:
    sys.path.insert(0, _CF_ROOT)

from shared.drive import authorize_interactive, _resolve_client_secrets, _resolve_token_path


def main():
    ap = argparse.ArgumentParser(description="One-time Google Drive OAuth authorization.")
    ap.add_argument("--credentials", default=None, help="Path to OAuth client JSON (credentials.json).")
    ap.add_argument("--token", default=None, help="Where to write token.json.")
    args = ap.parse_args()

    cs = _resolve_client_secrets(args.credentials or "")
    if not cs:
        print("ERROR: credentials.json / secrets.json not found. Download the Desktop OAuth "
              "client JSON from Google Cloud Console and place it at the project root.",
              file=sys.stderr)
        sys.exit(1)

    tp = args.token or _resolve_token_path("", cs)
    print(f"Authorizing with client secrets: {cs}")
    print("A browser window will open for Google consent...")
    written = authorize_interactive(client_secrets=cs, token_path=tp)
    print(f"\n✅ Authorized. Token written to: {written}")
    print("Copy BOTH of these to every other machine (keep them out of git):")
    print(f"  - {cs}")
    print(f"  - {written}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post a message to the nexus comms bus.

Usage:
    python nexus/comms/post.py --to cami --topic baseline-results --from evie --body "..."
    python nexus/comms/post.py --to broadcast --topic hello --from evie --body "..."
"""
from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

HERE = Path(__file__).parent
AGENTS = ("evie", "cami", "irina", "chatgpt", "broadcast")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, choices=AGENTS)
    ap.add_argument("--from", dest="sender", required=True, choices=[a for a in AGENTS if a != "broadcast"])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--reply-to", default=None)
    args = ap.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}-{slug(args.topic)}.md"
    path = HERE / "inbox" / args.to / fname
    body = (
        "---\n"
        f"from: {args.sender}\n"
        f"to: {args.to}\n"
        f"date: {now.isoformat()}\n"
        f"topic: {args.topic}\n"
        f"reply_to: {args.reply_to}\n"
        "---\n\n"
        f"{args.body}\n"
    )
    path.write_text(body, encoding="utf-8")
    print(f"posted -> {path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()

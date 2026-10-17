# Nexus Comms

Async message bus for the four-agent HRMIT effort (Evie, Cami, Irina, ChatGPT).
We can't message each other directly, so the repo is the wire.

## Protocol

**Send:** write a markdown file to the recipient's inbox (or `broadcast/` for all):

```
nexus/comms/inbox/<agent>/YYYYMMDD-HHMMSS-<topic>.md
```

**Format:**

```markdown
---
from: evie
to: cami
date: 2026-10-17T13:10:00Z
topic: baseline-results
reply_to: null
---

Body here. Keep it short; link to repo paths, not pasted logs.
```

**Check:** `git pull`, then list `nexus/comms/inbox/<you>/` and `nexus/comms/inbox/broadcast/`
for files newer than your last check. Read them, act, optionally reply.

**Rules:**
- One topic per message. Filename topic slug matches the `topic:` field.
- `reply_to` names the file you're answering (or null).
- Never put secrets, PATs, or credentials in a message.
- Big artifacts (weights, corpora) go in their proper repo locations;
  messages link to them, never embed them.
- Zebulon relays anything urgent — this bus is async, not instant.

## Helper

```bash
python nexus/comms/post.py --to cami --topic baseline-results --from evie \
  --body "Baseline run finished: loss 2.31. Weights at results/weights/hrm_cogtext_evie_baseline.npz"
```

## Inboxes

- `inbox/evie/` — Evie
- `inbox/cami/` — Cami
- `inbox/irina/` — Irina
- `inbox/chatgpt/` — ChatGPT
- `inbox/broadcast/` — all agents

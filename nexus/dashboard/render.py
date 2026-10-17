#!/usr/bin/env python3
"""Regenerate the HRMIT Nexus dashboard from experiments/ and results/."""
import json
from pathlib import Path

HERE = Path(__file__).parent
NEXUS = HERE.parent

def load(p):
    return json.loads(p.read_text()) if p.exists() else None

experiments = sorted((NEXUS / "experiments").glob("*.json"))
results = {p.stem: load(p) for p in (NEXUS / "results").glob("*.json") if p.name != "template.json"}

rows = []
for exp_path in experiments:
    exp = load(exp_path)
    res = results.get(exp["id"])
    status = exp.get("status", "open")
    if res:
        status = "done"
    dot = {"open": "⚪", "claimed": "🟡", "done": "🟢"}.get(status, "⚪")
    loss = f"{res['final_loss']:.4f}" if res and res.get("final_loss") else "–"
    ppl = f"{res['test_perplexity']:.2f}" if res and res.get("test_perplexity") else "–"
    t = exp["training"]
    rows.append(
        f"<tr><td>{dot} <code>{exp['id']}</code></td><td>{exp['agent']}</td>"
        f"<td>cog_dim={exp['model']['cog_dim']}, lr={t['lr']}, ep={t['epochs']}, seq={t['seq_len']}</td>"
        f"<td>{status}</td><td>{loss}</td><td>{ppl}</td></tr>"
    )

# best run
best = None
for rid, res in results.items():
    if res and res.get("final_loss") is not None:
        if best is None or res["final_loss"] < best[1]:
            best = (rid, res["final_loss"])
best_html = f"<p>🏆 Best so far: <code>{best[0]}</code> — loss {best[1]:.4f}</p>" if best else "<p>No completed runs yet.</p>"

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>HRMIT Nexus Dashboard</title>
<style>
body {{ background:#05060e; color:#e8ebff; font-family:Inter,system-ui,sans-serif; max-width:960px; margin:2rem auto; padding:0 1rem; }}
h1 {{ color:#ffd479; }} table {{ width:100%; border-collapse:collapse; }}
th,td {{ border:1px solid rgba(120,140,255,.2); padding:.6rem; text-align:left; font-size:.9rem; }}
th {{ color:#7c8cff; }} code {{ color:#ffd479; }}
a {{ color:#7c8cff; }}
</style></head><body>
<h1>ζ HRMIT Nexus</h1>
<p>Four-agent training coordination — <a href="../NEXUS.md">protocol</a></p>
{best_html}
<table><tr><th>Experiment</th><th>Agent</th><th>Config</th><th>Status</th><th>Final loss</th><th>Test ppl</th></tr>
{''.join(rows)}
</table>
<p><small>Regenerate with <code>dashboard/render.py</code>. Weights live in <code>results/weights/</code>.</small></p>
</body></html>"""

(HERE / "index.html").write_text(html)
print(f"dashboard written: {len(rows)} experiments, {sum(1 for r in results.values() if r)} results")

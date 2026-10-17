# HRMIT Nexus

Central coordination hub for the four-agent HRMIT v1 training effort
(Evie, Cami, Irina, ChatGPT — relayed by Zebulon).

## How it works

1. **Claim** — pick an `open` experiment in `experiments/`, set its `status` to
   `claimed` with your name, commit.
2. **Run** — open `hrmit_nexus_colab.ipynb` in Colab (GPU runtime), set
   `CONFIG_ID` to your experiment id in the first code cell, run top to bottom.
3. **Report** — copy `results/template.json` to `results/<experiment-id>.json`,
   fill in your loss history, samples, and wall time. Append a line to
   `results/LEDGER.md`. Push your weights to `results/weights/`.
4. **Dashboard** — run `dashboard/render.py` to regenerate `dashboard/index.html`
   from the latest results.

## The parallel tracks (one intelligence, two learners, one wire)

**Evie's track (Colab):** trains the generator (`HRMCognitionText`) on
`nexus/corpus/wikitext-2.txt` **plus** every `nexus/corpus/interactive-*.txt`.
Checkpoints push back to `checkpoints/`.

**Cami & Irina's track (interactive):** teach through the Stage 6 loop using
`hrm/learning/teach.py`:

```python
from hrm.generation.hrm_text import HRMCognitionText
from hrm.learning.teach import TeachSession

gen = HRMCognitionText(cog_dim=256)   # load latest Colab checkpoint
gen.load("path/to/checkpoint")
session = TeachSession(gen)            # adapter critic wired in

cands = session.propose("The HRM system")      # adapter-ranked candidates
exp = session.teach("The HRM system", cands[0][0], score=0.9, teacher="cami")
candidate, report = session.adapt()            # train + gate + promote adapter
session.export_corpus("nexus/corpus/interactive-2026-10-17.txt")
```

**The wire** (`hrm/learning/bridge.py`): the promoted adapter becomes the
generator's critic (`GuidedGenerator` — generator proposes N, adapter picks the
best). High-reward teachings export as corpus, so promotion in the interactive
track literally becomes training data for the next Colab run, and each new
Colab checkpoint becomes the base the teachers teach against. That's the loop
that makes it one intelligence.

## Rules

- This repo is **private**. Never re-license, never publicize (Zebulon's directive).
- One agent per experiment. If a run diverges (loss spikes >2x within 50 epochs),
  halve `lr` and restart — note it in your results file.
- Fixed seeds everywhere. Report full loss curves, not just final numbers.
- Corpus lives at `nexus/corpus/wikitext-2.txt` (13.2MB, WikiText-2 train+valid+test,
  headers stripped). The notebook loads it from the local clone — no downloads needed.
- Weights are `.npz` (~2–8MB); commit inline, no LFS needed.

## Status board

| Experiment | Agent | Config | Status |
|---|---|---|---|
| `evie_baseline` | Evie | cog_dim=256, lr=0.01, 2000ep, seed=1 | open |
| `cami_lr_sweep` | Cami | cog_dim=256, lr=0.005, 2000ep, seed=2 | open |
| `irina_capacity` | Irina | cog_dim=384, lr=0.01, 1500ep, seed=3 | open |
| `chatgpt_context` | ChatGPT | cog_dim=256, lr=0.01, seq_len=256, 2000ep, seed=4 | open |

Winner is chosen by lowest byte-level perplexity on the held-out test slice,
then integrated into `hrm/generation/provider.py`.

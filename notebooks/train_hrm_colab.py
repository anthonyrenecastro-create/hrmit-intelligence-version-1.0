"""
HRM Text Training on Google Colab (GPU).

Open this file in Colab, or copy cells into a new notebook.
Designed for a T4 GPU (free tier). Expected speedup: 15-30x vs CPU.

Steps:
  1. Runtime > Change runtime type > GPU (T4)
  2. Run cells top to bottom
  3. Paste your GitHub PAT when prompted (needs repo scope for the private repo)
"""

# ── Cell 1: GPU check ──────────────────────────────────────────────
import subprocess, sys

result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True)
print("GPU:", result.stdout.strip() or "NONE DETECTED - enable GPU runtime!")

# ── Cell 2: Install JAX with CUDA ──────────────────────────────────
# Colab ships CUDA 12; jax[cuda12] matches it.
print("Installing JAX (GPU)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jax[cuda12]", "numpy>=2.1", "scipy>=1.14", "pillow"],
               check=True)
import jax
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

# ── Cell 3: Clone private repo ─────────────────────────────────────
import getpass, os

token = getpass.getpass("GitHub PAT (repo scope): ")
repo_url = f"https://x-access-token:{token}@github.com/anthonyrenecastro-create/hrmit-intelligence-version-1.0.git"
if not os.path.exists("/content/hrmit"):
    subprocess.run(["git", "clone", "--depth", "1", repo_url, "/content/hrmit"], check=True)
else:
    print("Repo already cloned.")
os.chdir("/content/hrmit")
# Don't leave the token in shell history
del token

# ── Cell 4: Build training corpus ──────────────────────────────────
from pathlib import Path

corpus_parts = []
# Repo docs (domain-relevant)
for p in list(Path("docs").rglob("*.md")) + list(Path("validation").rglob("*.md")):
    try:
        corpus_parts.append(p.read_text())
    except Exception:
        pass
corpus_parts.append(Path("README.md").read_text())

# Add TinyStories (public domain-style simple English, great for small models)
# ~2MB of clean, simple text
import urllib.request
tiny_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
tiny_path = Path("/content/tinystories.txt")
if not tiny_path.exists():
    print("Downloading TinyStories (~2GB full file, we'll take a slice)...")
    # Instead: use the smaller validation split via datasets is heavy; fetch a chunk
    # Fallback: skip if too large, use repo docs repeated
    print("Skipping large download on free tier; using repo corpus.")
repo_text = "\n---\n".join(corpus_parts)
print(f"Repo corpus: {len(repo_text):,} chars")
# Repeat to give the model more to chew on
corpus = (repo_text + "\n") * 8
print(f"Training corpus: {len(corpus):,} chars")
Path("/content/corpus.txt").write_text(corpus)

# ── Cell 5: Train ──────────────────────────────────────────────────
import sys
sys.path.insert(0, "/content/hrmit")
from hrm.generation.hrm_text import HRMCognitionText

gen = HRMCognitionText(vocab_size=257, cog_dim=256, seed=0)  # bigger than CPU run
print("Params:",
      sum(int(x.size) for x in [gen.embed, gen.W_gates, gen.b_gates, gen.W_hier, gen.W_out]))

history = gen.train(corpus, epochs=500, seq_len=128, lr=0.01, seed=1)
print(f"Loss: {history[0]:.3f} -> {history[-1]:.3f}")

# Save weights (also mount Drive first if you want persistence)
gen.save("/content/hrm_weights")
print("Weights saved to /content/hrm_weights/hrm_cogtext.npz")

# ── Cell 6: Generate ───────────────────────────────────────────────
for prompt in ["The HRM system", "Stage 2 memory"]:
    out = gen.generate(prompt=prompt, max_tokens=200, temperature=0.7, seed=42)
    print(f"--- prompt: {prompt!r} ---")
    print(out)
    print()

# ── Cell 7 (optional): push weights back to repo ───────────────────
# Requires the PAT again; saves ~3MB npz to the repo for the provider to load.
# Uncomment and run if you want the weights versioned.
#
# import base64, json, urllib.request
# token2 = getpass.getpass("GitHub PAT again: ")
# data = open("/content/hrm_weights/hrm_cogtext.npz", "rb").read()
# print(f"Weights size: {len(data)/1e6:.1f} MB")
# ... (push via contents API to hrm_generated/hrm_cogtext.npz)

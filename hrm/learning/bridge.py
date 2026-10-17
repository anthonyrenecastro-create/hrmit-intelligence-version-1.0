"""The missing wire: connects the Stage 6 interactive learning track to the
Colab generator training track, making them one intelligence learning both ways.

Direction 1 (teach -> train): high-reward text experiences become generator
training corpus. What Cami and Irina teach today, the Colab run trains on tomorrow.

Direction 2 (train -> teach): the promoted LinearAdapter becomes the generator's
critic. The generator proposes N candidates; the adapter scores each with the
feature extractor below and the best is served. Teaching feedback trains the
adapter's weights, so human preferences directly steer generation.
"""
from __future__ import annotations

import numpy as np

from .adapters import LinearAdapter
from .experience import ExperienceStore

# Interpretable v1 features over (prompt, candidate). Crude but honest —
# documented as heuristics; the adapter learns human preference weights over them.
FEATURE_NAMES = (
    "cand_len",       # candidate chars / 1000
    "prompt_len",     # prompt chars / 500
    "word_overlap",   # fraction of prompt words appearing in candidate
    "unique_ratio",   # unique words / total words in candidate
    "non_repeat",     # 1 - repeated-bigram ratio
    "terminal_punct", # ends with . ! ? : 0/1
    "mean_word_len",  # / 10
    "line_breaks",    # newline count / 10
)
FEATURE_DIM = len(FEATURE_NAMES)

TEXT_TASK_TYPE = "text_generate"


def text_features(prompt: str, candidate: str) -> np.ndarray:
    prompt = prompt or ""
    candidate = candidate or ""
    p_words = prompt.lower().split()
    c_words = candidate.lower().split()
    overlap = (sum(1 for w in p_words if w in set(c_words)) / max(1, len(p_words)))
    uniq = len(set(c_words)) / max(1, len(c_words))
    bigrams = list(zip(c_words, c_words[1:]))
    rep = (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams))
    feats = np.array([
        len(candidate) / 1000.0,
        len(prompt) / 500.0,
        overlap,
        uniq,
        1.0 - rep,
        1.0 if candidate.rstrip().endswith((".", "!", "?", ":")) else 0.0,
        (sum(len(w) for w in c_words) / max(1, len(c_words))) / 10.0,
        candidate.count("\n") / 10.0,
    ], dtype=np.float32)
    return feats


def experiences_to_corpus(store: ExperienceStore, min_reward: float = 0.7) -> str:
    """Export teach-track experiences as generator training text.

    Only text_generate tasks with reward >= min_reward and every feedback
    record marked suitable_for_training. Format: prompt, output, blank line.
    """
    chunks = []
    for exp in store.all():
        out = exp.task_outcome
        if out.task_type != TEXT_TASK_TYPE:
            continue
        if exp.reward < min_reward:
            continue
        if any(not f.suitable_for_training for f in exp.feedback):
            continue
        prompt = (out.inputs or {}).get("prompt", "")
        output = out.output or ""
        if not prompt or not output:
            continue
        chunks.append(f"{prompt}\n{output}\n\n")
    return "".join(chunks)


class GuidedGenerator:
    """Generator + adapter critic. The generator proposes, the adapter disposes."""

    def __init__(self, generator, adapter: LinearAdapter, n_candidates: int = 8, seed: int = 0):
        if adapter.feature_dim != FEATURE_DIM:
            raise ValueError(f"Adapter feature_dim={adapter.feature_dim}, need {FEATURE_DIM}")
        self.generator = generator
        self.adapter = adapter
        self.n_candidates = n_candidates
        self.seed = seed

    def propose(self, prompt: str, temperature: float = 0.8) -> list[tuple[str, float]]:
        scored = []
        for i in range(self.n_candidates):
            cand = self.generator.generate(prompt=prompt, max_tokens=200,
                                           temperature=temperature, seed=self.seed + i)
            feats = text_features(prompt, cand)
            score = float(self.adapter.probabilities(feats.reshape(1, -1))[0])
            scored.append((cand, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def generate(self, prompt: str, temperature: float = 0.8) -> str:
        return self.propose(prompt, temperature)[0][0]

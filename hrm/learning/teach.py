"""Interactive teaching session for the parallel-tracks loop.

Cami / Irina flow:
    session = TeachSession(generator, system, store)
    cands = session.propose("The HRM system")   # adapter-ranked candidates
    exp = session.teach("The HRM system", cands[0][0], score=0.9, teacher="cami")
    session.adapt()                             # train + gate + promote adapter
    session.export_corpus("nexus/corpus/interactive-2026-10-17.txt")

The exported corpus feeds the next Colab generator run; the Colab checkpoint
becomes the generator this session teaches against. One loop, two tracks.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from .adapters import LinearAdapter
from .bridge import FEATURE_DIM, TEXT_TASK_TYPE, experiences_to_corpus, text_features
from .experience import ExperienceStore
from .system import ControlledLearningSystem
from .types import FeedbackRecord, TaskOutcome


class TeachSession:
    def __init__(self, generator, system: ControlledLearningSystem | None = None,
                 store: ExperienceStore | None = None, n_candidates: int = 8):
        if system is None:
            system = ControlledLearningSystem(feature_dim=FEATURE_DIM)
        elif system.active.feature_dim != FEATURE_DIM:
            raise ValueError("System adapter feature_dim must match bridge FEATURE_DIM")
        self.generator = generator
        self.system = system
        self.store = store or ExperienceStore()
        self.guided = None
        self.n_candidates = n_candidates

    def _guided(self):
        from .bridge import GuidedGenerator
        if self.guided is None:
            self.guided = GuidedGenerator(self.generator, self.system.active,
                                          n_candidates=self.n_candidates)
        return self.guided

    def propose(self, prompt: str, temperature: float = 0.8) -> list[tuple[str, float]]:
        """Adapter-ranked candidate outputs for a prompt."""
        return self._guided().propose(prompt, temperature)

    def teach(self, prompt: str, output: str, score: float, teacher: str,
              notes: str = "") -> object:
        """Record one teaching interaction as an experience."""
        task_id = "teach-" + uuid.uuid4().hex[:12]
        outcome = TaskOutcome(
            task_id=task_id,
            task_type=TEXT_TASK_TYPE,
            inputs={"prompt": prompt},
            output=output,
            success=score >= 0.5,
            score=score,
            completion_time=0.0,
            module_ids=("guided_generator",),
            metadata={"teacher": teacher, "notes": notes},
        )
        feedback = (FeedbackRecord.create(
            task_id=task_id,
            source=teacher,
            feedback_type="numeric_score",
            value=score,
            confidence=0.9,
            scope="single",
            objective=False,
            suitable_for_training=True,
            metadata={"notes": notes},
        ),)
        exp = ControlledLearningSystem.capture(outcome, feedback)
        self.store.add(exp)
        # adapter weights may have been promoted since guided was built
        self.guided = None
        return exp

    def adapt(self, learning_rate: float = 0.1, epochs: int = 50,
              heldout_frac: float = 0.2):
        """Train an adapter candidate from stored experiences, gate, promote/rollback."""
        exps = self.store.all()
        if len(exps) < 4:
            raise ValueError(f"Need >= 4 experiences to adapt, have {len(exps)}")
        xs, ys = [], []
        for exp in exps:
            if exp.task_outcome.task_type != TEXT_TASK_TYPE:
                continue
            prompt = (exp.task_outcome.inputs or {}).get("prompt", "")
            output = exp.task_outcome.output or ""
            xs.append(text_features(prompt, output))
            ys.append(1.0 if exp.reward >= 0.5 else 0.0)
        if len(xs) < 4:
            raise ValueError("Need >= 4 text experiences to adapt")
        x = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, dtype=np.float32)
        n_held = max(1, int(len(x) * heldout_frac))
        # chronological split: oldest trains, newest held out
        train, heldout = (x[:-n_held], y[:-n_held]), (x[-n_held:], y[-n_held:])
        regression = (x[:max(1, len(x) // 4)], y[:max(1, len(y) // 4)])
        candidate, report = self.system.adapt(train, heldout, regression,
                                              learning_rate=learning_rate, epochs=epochs)
        self.guided = None  # rebuild critic with (possibly) new weights
        return candidate, report

    def export_corpus(self, path: str | Path, min_reward: float = 0.7) -> Path:
        """Write teach-track corpus for the next Colab generator run."""
        text = experiences_to_corpus(self.store, min_reward=min_reward)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

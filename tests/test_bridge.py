"""Tests for the teach<->train wire (hrm.learning.bridge + teach)."""
from __future__ import annotations

import numpy as np

from hrm.learning.adapters import LinearAdapter
from hrm.learning.bridge import (
    FEATURE_DIM,
    GuidedGenerator,
    experiences_to_corpus,
    text_features,
)
from hrm.learning.experience import ExperienceStore
from hrm.learning.system import ControlledLearningSystem
from hrm.learning.teach import TeachSession
from hrm.learning.types import FeedbackRecord, TaskOutcome


def _exp(prompt, output, reward, suitable=True, task_type="text_generate"):
    task_id = f"t-{abs(hash(prompt)) % 100000}"
    outcome = TaskOutcome(task_id=task_id, task_type=task_type,
                          inputs={"prompt": prompt}, output=output,
                          success=reward >= 0.5, score=reward)
    fb = (FeedbackRecord.create(task_id=task_id, source="test", feedback_type="numeric_score",
                                value=reward, confidence=0.9, scope="single",
                                objective=False, suitable_for_training=suitable),)
    return ControlledLearningSystem.capture(outcome, fb)


def test_text_features_dim_and_range():
    f = text_features("The HRM system", "The HRM system reasons in cycles.")
    assert f.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(f))
    assert 0.0 <= f[2] <= 1.0  # overlap bounded


def test_corpus_export_filters():
    store = ExperienceStore()
    store.add(_exp("p1", "good output here.", 0.9))                       # kept
    store.add(_exp("p2", "bad output.", 0.2))                            # low reward: dropped
    store.add(_exp("p3", "withheld output.", 0.95, suitable=False))       # not for training: dropped
    store.add(_exp("p4", "other task.", 0.9, task_type="vision"))          # wrong type: dropped
    corpus = experiences_to_corpus(store, min_reward=0.7)
    assert "good output here." in corpus
    assert "bad output." not in corpus
    assert "withheld" not in corpus
    assert "other task." not in corpus


class FakeGenerator:
    """Deterministic stub: returns seeded variants so the critic has something to rank."""

    def generate(self, prompt="", max_tokens=200, temperature=0.8, seed=0):
        if seed % 2 == 0:
            return prompt + " This is a complete, well-formed sentence."
        return "x"


def test_guided_generator_picks_best_scored():
    adapter = LinearAdapter(FEATURE_DIM)
    # hand-set weights: reward terminal punctuation + length, punish emptiness
    adapter.parameters[:] = 0.0
    adapter.parameters[5] = 5.0   # terminal_punct
    adapter.parameters[0] = 1.0   # cand_len
    guided = GuidedGenerator(FakeGenerator(), adapter, n_candidates=4, seed=0)
    best = guided.generate("The system")
    assert best.endswith(".")
    assert len(best) > 10


def test_teach_session_round_trip_and_adapt():
    system = ControlledLearningSystem(feature_dim=FEATURE_DIM)
    session = TeachSession(FakeGenerator(), system, n_candidates=2)
    for i, score in enumerate([0.9, 0.85, 0.2, 0.15, 0.8, 0.1]):
        cands = session.propose(f"prompt {i}")
        assert len(cands) == 2
        session.teach(f"prompt {i}", cands[0][0], score=score, teacher="test")
    assert len(session.store.all()) == 6
    candidate, report = session.adapt(epochs=10)
    assert candidate.candidate_id
    # corpus export contains only high-reward teachings
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = session.export_corpus(os.path.join(d, "interactive.txt"), min_reward=0.7)
        text = open(p).read()
        assert "prompt 0" in text
        assert "prompt 2" not in text  # score 0.2 -> reward 0.2 < 0.7

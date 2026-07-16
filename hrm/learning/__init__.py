"""Validation-gated adaptive learning for HRMIT Stage 6."""

from .system import ControlledLearningSystem
from .compatibility import LearningSystem, PreferenceModel
from .types import (AdaptationCandidate, EvaluationReport, ExperienceRecord,
                    FeedbackRecord, TaskOutcome)

__all__ = ["AdaptationCandidate", "ControlledLearningSystem", "EvaluationReport", "LearningSystem", "PreferenceModel",
           "ExperienceRecord", "FeedbackRecord", "TaskOutcome"]

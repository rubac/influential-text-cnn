"""
Discovering Influential Text Using Convolutional Neural Networks

A Python/PyTorch implementation of the method from:
Ayers, Sanford, Roberts, Yang (2024). "Discovering influential text using
convolutional neural networks." Findings of ACL 2024.
"""

from .model import InfluentialTextCNN, InfluentialTextLoss
from .pipeline import InfluentialTextPipeline
from .embedding import BERTEmbedder, PrecomputedEmbedder, EmbeddingResult
from .training import Trainer, TrainingHistory
from .interpretation import FilterInterpreter, InterpretationResult, FilterInfo
from .benchmarks import RegularizedLogisticRegression, BenchmarkResult
from .tuning import tune_hyperparameters, TuningReport

__version__ = "0.1.0"
__all__ = [
    "InfluentialTextCNN",
    "InfluentialTextLoss",
    "InfluentialTextPipeline",
    "BERTEmbedder",
    "PrecomputedEmbedder",
    "EmbeddingResult",
    "Trainer",
    "TrainingHistory",
    "FilterInterpreter",
    "InterpretationResult",
    "FilterInfo",
    "RegularizedLogisticRegression",
    "BenchmarkResult",
    "tune_hyperparameters",
    "TuningReport",
]

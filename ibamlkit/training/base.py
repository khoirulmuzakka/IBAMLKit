"""Training interfaces for IBAMLKit models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ibamlkit.models.forward import ForwardModelBase


@dataclass(frozen=True)
class TrainingBatch:
    """One supervised batch for model training or evaluation."""

    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    """Lightweight summary returned by a trainer."""

    epochs_completed: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelTrainer(ABC):
    """Abstract trainer interface.

    Trainers own optimization strategy, losses, batching, checkpoints,
    and evaluation loops. Models stay focused on inference semantics.
    """

    @abstractmethod
    def fit(self, model: ForwardModelBase, /, **kwargs: Any) -> TrainingResult:
        """Train a model and return a compact result summary."""

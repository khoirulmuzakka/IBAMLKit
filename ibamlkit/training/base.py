"""Training interfaces for IBAMLKit models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch


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


class TrainableTensorModel(Protocol):
    """Structural interface expected by the generic trainers."""

    def transform_inputs(self, inputs: Any) -> torch.Tensor:
        ...

    def validate_input_shape(self, inputs: torch.Tensor) -> None:
        ...

    def parameters(self):
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> Any:
        ...

    def to(self, device: torch.device | str) -> Any:
        ...

    def train(self, mode: bool = True) -> Any:
        ...

    def eval(self) -> Any:
        ...

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


class ModelTrainer(ABC):
    """Abstract trainer interface.

    Trainers own optimization strategy, losses, batching, checkpoints,
    and evaluation loops. Models stay focused on inference semantics.
    """

    @abstractmethod
    def fit(self, model: TrainableTensorModel, /, **kwargs: Any) -> TrainingResult:
        """Train a model and return a compact result summary."""

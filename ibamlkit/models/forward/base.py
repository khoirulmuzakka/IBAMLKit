"""Runtime base classes for IBAMLKit model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from ibamlkit.schema import ModelSchema


class ForwardModelBase(nn.Module, ABC):
    """Base class for surrogate forward models.

    The class owns the runtime contract for a model implementation:
    it knows its public ``ModelSchema`` and exposes predictable input
    transformation and prediction hooks.

    Training, packaging, and ONNX session handling are intentionally
    kept outside this class.
    """

    def __init__(self, schema: ModelSchema) -> None:
        super().__init__()
        self.schema = schema

    @property
    def input_dimension(self) -> int:
        return self.schema.inputs.dimension

    @property
    def output_feature_dimension(self) -> int:
        return self.schema.outputs.feature_dimension

    def transform_inputs(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert raw inputs to a float32 tensor expected by ``forward``."""
        if isinstance(inputs, torch.Tensor):
            return inputs if inputs.dtype == torch.float32 else inputs.to(dtype=torch.float32)
        return torch.as_tensor(inputs, dtype=torch.float32)

    def validate_input_shape(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 2:
            raise ValueError("Model inputs must be a 2D tensor of shape (batch, features).")
        if inputs.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected {self.input_dimension} input features, got {inputs.shape[1]}."
            )

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the model on a batch of transformed inputs."""

    def predict(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Inference helper using the model's ``forward`` implementation."""
        tensor = self.transform_inputs(inputs)
        self.validate_input_shape(tensor)
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        tensor = tensor.to(device=model_device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = self.forward(tensor)
        if was_training:
            self.train()
        return outputs

    def example_input(self, batch_size: int = 1) -> torch.Tensor:
        """Return a representative input tensor for tracing or export."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        return torch.zeros((batch_size, self.input_dimension), dtype=torch.float32)

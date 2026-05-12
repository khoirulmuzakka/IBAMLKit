"""Runtime base classes for IBAMLKit inverse-model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from ibamlkit.schema import InverseModelSchema


class InverseModelBase(nn.Module, ABC):
    """Base class for inverse models that map spectra to open parameters."""

    def __init__(self, schema: InverseModelSchema) -> None:
        super().__init__()
        self.schema = schema

    @property
    def input_dimension(self) -> int:
        return self.schema.inputs.dimension

    @property
    def output_dimension(self) -> int:
        return self.schema.outputs.feature_dimension

    def transform_inputs(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
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
        """Run the inverse model on transformed spectra."""

    def predict(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
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
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        return torch.zeros((batch_size, self.input_dimension), dtype=torch.float32)

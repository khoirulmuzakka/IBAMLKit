"""Pure MLP surrogate forward model for layered samples."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .base import ForwardModelBase
from .refiners import LocalSpectrumRefiner
from ibamlkit.schema import ForwardModelSchema


class MLPModel(ForwardModelBase):
    """Multi-layer Perceptron surrogate model.
    
    A simple feedforward neural network that flattens all input parameters
    (setup and layer features) into a single vector and processes them through
    fully connected layers.
    """

    def __init__(
        self,
        schema: ForwardModelSchema,
        *,
        hidden_sizes: Sequence[int] = (512, 512, 256),
        dropout_rate: float = 0.1,
        refiner_hidden_channels: int = 32,
        refiner_kernel_size: int = 17,
    ) -> None:
        super().__init__(schema)
        
        self.input_size = schema.inputs.dimension
        self.output_size = self._infer_output_size(schema)
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout_rate = float(dropout_rate)
        
        # Build the main MLP tower
        mlp_layers: list[nn.Module] = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0:
                mlp_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        mlp_layers.append(nn.Linear(prev_size, self.output_size))
        
        self.mlp_tower = nn.Sequential(*mlp_layers)
        
        # Spectrum refiner with FiLM conditioning on input features
        self.spectrum_refiner = LocalSpectrumRefiner(
            condition_dim=self.input_size,
            output_size=self.output_size,
            hidden_channels=refiner_hidden_channels,
            kernel_size=refiner_kernel_size,
        )
        
        # Output scaling parameter
        self.output_scale = nn.Parameter(torch.ones(self.output_size), requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP model.
        
        Args:
            inputs: Tensor of shape (batch_size, input_dimension)
            
        Returns:
            Tensor of shape (batch_size, output_size) with predicted spectrum
        """
        self.validate_input_shape(inputs)
        
        # Main MLP prediction
        logits = self.mlp_tower(inputs)
        coarse_spectrum = F.softplus(logits)
        
        # Refine spectrum using input-conditioned convolution
        refined_spectrum = F.softplus(self.spectrum_refiner(coarse_spectrum, inputs))
        
        # Apply learned output scaling
        return refined_spectrum * F.softplus(self.output_scale).to(
            device=logits.device,
            dtype=logits.dtype,
        )
    
    @staticmethod
    def _infer_output_size(schema: ForwardModelSchema) -> int:
        if schema.outputs.spectra_lengths:
            return sum(schema.outputs.spectra_lengths.values())
        if schema.outputs.feature_dimension > 0:
            return schema.outputs.feature_dimension
        raise ValueError("Unable to infer output size from schema.")



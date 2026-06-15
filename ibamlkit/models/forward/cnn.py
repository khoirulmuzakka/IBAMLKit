"""Pure CNN surrogate forward model for layered samples."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .base import ForwardModelBase
from .refiners import LocalSpectrumRefiner
from ibamlkit.schema import ForwardModelSchema


class CNNModel(ForwardModelBase):
    """Convolutional Neural Network surrogate model.
    
    Treats input parameters as a 1D sequence and processes them through
    1D convolutional layers followed by fully connected decoder layers
    to produce spectrum predictions.
    """

    def __init__(
        self,
        schema: ForwardModelSchema,
        *,
        initial_channels: int = 32,
        num_conv_layers: int = 3,
        kernel_size: int = 5,
        decoder_hidden_sizes: Sequence[int] = (512, 512, 256),
        dropout_rate: float = 0.1,
        refiner_hidden_channels: int = 32,
        refiner_kernel_size: int = 17,
    ) -> None:
        super().__init__(schema)
        
        self.input_size = schema.inputs.dimension
        self.output_size = self._infer_output_size(schema)
        self.initial_channels = int(initial_channels)
        self.num_conv_layers = int(num_conv_layers)
        self.kernel_size = int(kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.padding = self.kernel_size // 2
        self.dropout_rate = float(dropout_rate)
        
        # Build convolutional encoder
        conv_layers: list[nn.Module] = []
        
        # First conv layer: 1D -> initial_channels
        conv_layers.append(nn.Conv1d(1, self.initial_channels, kernel_size=self.kernel_size, padding=self.padding))
        conv_layers.append(nn.LeakyReLU())
        if self.dropout_rate > 0:
            conv_layers.append(nn.Dropout(self.dropout_rate))
        
        # Additional conv layers with channel growth
        prev_channels = self.initial_channels
        for i in range(1, self.num_conv_layers):
            next_channels = self.initial_channels * (2 ** i)
            conv_layers.append(
                nn.Conv1d(prev_channels, next_channels, kernel_size=self.kernel_size, padding=self.padding)
            )
            conv_layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0:
                conv_layers.append(nn.Dropout(self.dropout_rate))
            prev_channels = next_channels
        
        self.conv_encoder = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after convolutions
        # The sequence length is preserved by padding, so: input_size * prev_channels
        flattened_size = self.input_size * prev_channels
        
        # Build decoder: flattened conv output -> spectrum
        decoder_layers: list[nn.Module] = []
        prev_size = flattened_size
        
        for hidden_size in decoder_hidden_sizes:
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0:
                decoder_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_size, self.output_size))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Spectrum refiner with input-conditioned convolutions
        self.spectrum_refiner = LocalSpectrumRefiner(
            condition_dim=self.input_size,
            output_size=self.output_size,
            hidden_channels=refiner_hidden_channels,
            kernel_size=refiner_kernel_size,
        )
        
        # Output scaling parameter
        self.output_scale = nn.Parameter(torch.ones(self.output_size), requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN model.
        
        Args:
            inputs: Tensor of shape (batch_size, input_dimension)
            
        Returns:
            Tensor of shape (batch_size, output_size) with predicted spectrum
        """
        self.validate_input_shape(inputs)
        
        # Reshape to (batch_size, 1, input_size) for 1D convolutions
        x = inputs.unsqueeze(1)
        
        # Pass through convolutional encoder
        conv_output = self.conv_encoder(x)
        
        # Flatten: (batch_size, channels, length) -> (batch_size, channels * length)
        flattened = conv_output.reshape(conv_output.shape[0], -1)
        
        # Decode to spectrum
        logits = self.decoder(flattened)
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



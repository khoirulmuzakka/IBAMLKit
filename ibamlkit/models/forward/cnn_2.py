"""Alternative CNN surrogate model with explicit linear reshape and strided conv blocks."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .base import ForwardModelBase
from .refiners import LocalSpectrumRefiner
from ibamlkit.schema import ForwardModelSchema


class CNN2Model(ForwardModelBase):
    """CNN surrogate model with dense-to-reshape followed by strided conv blocks."""

    def __init__(
        self,
        schema: ForwardModelSchema,
        *,
        dense_size: int = 4096,
        channels_1: int = 256,
        channels_2: int = 512,
        bottleneck_length: int = 32,
        kernel_size: int = 3,
        decoder_hidden_sizes: Sequence[int] = (),
        dropout_rate: float = 0.0,
        refiner_hidden_channels: int = 32,
        refiner_kernel_size: int = 17,
    ) -> None:
        super().__init__(schema)

        self.input_size = schema.inputs.dimension
        self.output_size = self._infer_output_size(schema)
        self.dense_size = int(dense_size)
        self.channels_1 = int(channels_1)
        self.channels_2 = int(channels_2)
        self.bottleneck_length = int(bottleneck_length)
        self.kernel_size = int(kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.padding = self.kernel_size // 2
        self.dropout_rate = float(dropout_rate)

        # Dense projection from input to flattened conv tensor
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_size, self.dense_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
        )

        # Reshape to (batch, 128, 32) if dense_size allows
        self.reshape_channels = 128
        self.reshape_length = self.bottleneck_length
        if self.reshape_channels * self.reshape_length != self.dense_size:
            raise ValueError(
                f"dense_size must equal reshape_channels * reshape_length ({self.reshape_channels} * {self.reshape_length})"
            )

        self.conv_block = nn.Sequential(
            nn.Conv1d(self.reshape_channels, self.channels_1, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
            nn.Conv1d(self.channels_1, self.channels_2, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
        )

        # The final conv output has length 8 when stride=2 halves the length twice
        flattened_size = self.channels_2 * (self.reshape_length // 2 // 2)

        decoder_layers: list[nn.Module] = []
        prev_size = flattened_size
        for hidden_size in decoder_hidden_sizes:
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0.0:
                decoder_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        decoder_layers.append(nn.Linear(prev_size, self.output_size))
        self.decoder = nn.Sequential(*decoder_layers)

        self.spectrum_refiner = LocalSpectrumRefiner(
            condition_dim=self.input_size,
            output_size=self.output_size,
            hidden_channels=refiner_hidden_channels,
            kernel_size=refiner_kernel_size,
        )

        self.output_scale = nn.Parameter(torch.ones(self.output_size), requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.validate_input_shape(inputs)

        x = self.input_projection(inputs)
        x = x.view(x.shape[0], self.reshape_channels, self.reshape_length)
        x = self.conv_block(x)
        x = x.flatten(start_dim=1)
        logits = self.decoder(x)
        coarse_spectrum = F.softplus(logits)
        refined_spectrum = F.softplus(self.spectrum_refiner(coarse_spectrum, inputs))
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

"""Shared local spectrum refinement utilities for forward models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LocalSpectrumRefiner(nn.Module):
    """Inject local spectrum structure through a lightweight conv residual head."""

    def __init__(
        self,
        *,
        hidden_channels: int,
        kernel_size: int,
        condition_dim: int = 0,
        output_size: int | None = None,
        setup_dim: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1.")

        if setup_dim is not None:
            condition_dim = setup_dim

        self.output_size = int(output_size) if output_size is not None else None
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.padding = self.kernel_size // 2
        self._use_dummy_condition = condition_dim <= 0
        effective_condition_dim = 1 if self._use_dummy_condition else int(condition_dim)

        self.in_projection = nn.Conv1d(1, hidden_channels, kernel_size=self.kernel_size, padding=self.padding)
        self.residual_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.out_projection = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.condition_to_film = nn.Sequential(
            nn.Linear(effective_condition_dim, hidden_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )

    def forward(self, spectra: torch.Tensor, condition_inputs: torch.Tensor) -> torch.Tensor:
        if self._use_dummy_condition:
            condition_inputs = torch.ones(
                (spectra.shape[0], 1),
                device=spectra.device,
                dtype=spectra.dtype,
            )

        features = self.in_projection(spectra.unsqueeze(1))
        gamma, beta = torch.chunk(
            self.condition_to_film(condition_inputs.to(device=spectra.device, dtype=spectra.dtype)),
            2,
            dim=1,
        )
        gamma = torch.tanh(gamma).unsqueeze(-1) + 1.0
        beta = beta.unsqueeze(-1)
        features = F.leaky_relu(features * gamma + beta)
        residual = self.residual_conv(features)
        residual = F.leaky_relu(residual + features)
        residual = self.out_projection(residual).squeeze(1)
        return spectra + residual

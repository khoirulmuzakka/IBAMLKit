"""Deterministic inverse-model baselines."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from ibamlkit.schema import InverseModelSchema

from .base import InverseModelBase


class InverseMLPModel(InverseModelBase):
    """Simple dense baseline for spectra-to-parameter regression."""

    def __init__(
        self,
        schema: InverseModelSchema,
        *,
        hidden_sizes: Sequence[int] = (1024, 512, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(schema)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        layers: list[nn.Module] = []
        in_features = self.input_dimension
        for hidden_size in hidden_sizes:
            if hidden_size < 1:
                raise ValueError("hidden layer widths must be positive.")
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, self.output_dimension))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        self.validate_input_shape(inputs)
        return self.network(inputs)

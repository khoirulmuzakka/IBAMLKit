"""Transformer-based surrogate forward model for layered samples."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ibamlkit.models.base import ForwardModelBase
from ibamlkit.models.lrn import LRNModel
from ibamlkit.schema import ModelSchema


class _LocalSpectrumRefiner(nn.Module):
    """Lightweight setup-conditioned local residual correction."""

    def __init__(
        self,
        *,
        setup_dim: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1.")

        self.kernel_size = int(kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.padding = self.kernel_size // 2
        self._use_dummy_setup = setup_dim <= 0
        effective_setup_dim = 1 if self._use_dummy_setup else setup_dim

        self.in_projection = nn.Conv1d(1, hidden_channels, kernel_size=self.kernel_size, padding=self.padding)
        self.residual_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.out_projection = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.setup_to_film = nn.Sequential(
            nn.Linear(effective_setup_dim, hidden_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )

    def forward(self, spectra: torch.Tensor, setup_inputs: torch.Tensor) -> torch.Tensor:
        if self._use_dummy_setup:
            setup_inputs = torch.ones(
                (spectra.shape[0], 1),
                device=spectra.device,
                dtype=spectra.dtype,
            )

        features = self.in_projection(spectra.unsqueeze(1))
        gamma, beta = torch.chunk(
            self.setup_to_film(setup_inputs.to(device=spectra.device, dtype=spectra.dtype)),
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


class TransformerSurrogateModel(ForwardModelBase):
    """Encoder-only transformer surrogate for variable-layer samples."""

    def __init__(
        self,
        schema: ModelSchema,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        decoder_hidden_sizes: Sequence[int] = (1024, 1024),
        refiner_hidden_channels: int = 32,
        refiner_kernel_size: int = 17,
    ) -> None:
        super().__init__(schema)
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        layout = LRNModel._infer_input_layout(schema)
        self.input_layout = layout
        self.setup_param_size = layout.setup_param_size
        self.layer_param_size = layout.layer_param_size
        self.layer_count = layout.layer_count
        self.output_size = LRNModel._infer_output_size(schema)
        self.model_dim = int(model_dim)
        self.setup_feature_indices = torch.tensor(layout.setup_feature_indices, dtype=torch.int64)
        self.layer_feature_indices = tuple(
            torch.tensor(indices, dtype=torch.int64)
            for indices in layout.layer_feature_indices
        )
        self.concentration_indices = [list(offsets) for offsets in layout.concentration_feature_offsets]

        self.layer_projection = nn.Linear(max(self.layer_param_size, 1), model_dim)
        self.setup_projection = nn.Linear(max(self.setup_param_size, 1), model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.layer_count + 1, model_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(model_dim)

        decoder_layers: list[nn.Module] = []
        prev = model_dim * 3
        for width in decoder_hidden_sizes:
            decoder_layers.append(nn.Linear(prev, width))
            decoder_layers.append(nn.GELU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            prev = width
        decoder_layers.append(nn.Linear(prev, self.output_size))
        self.spectrum_decoder = nn.Sequential(*decoder_layers)
        self.spectrum_refiner = _LocalSpectrumRefiner(
            setup_dim=self.setup_param_size,
            hidden_channels=refiner_hidden_channels,
            kernel_size=refiner_kernel_size,
        )
        self.output_scale = nn.Parameter(torch.ones(self.output_size), requires_grad=True)

    def normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.concentration_indices or self.layer_param_size <= 0:
            return inputs

        normalized = inputs.clone()
        for layer_index, concentration_indices in enumerate(self.concentration_indices):
            if not concentration_indices:
                continue
            current = normalized.index_select(
                dim=1,
                index=self.layer_feature_indices[layer_index].to(device=inputs.device),
            )
            concentrations = current[:, concentration_indices].clamp_min(0.0)
            sums = concentrations.sum(dim=1, keepdim=True)
            concentrations = concentrations / torch.where(
                sums > 0.0,
                sums,
                torch.ones_like(sums),
            )
            current[:, concentration_indices] = concentrations
            normalized[:, self.layer_feature_indices[layer_index].to(device=inputs.device)] = current
        return normalized

    def split_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.setup_param_size > 0:
            setup = inputs.index_select(
                dim=1,
                index=self.setup_feature_indices.to(device=inputs.device),
            )
        else:
            setup = inputs.new_zeros((inputs.shape[0], 0))
        layers = [
            inputs.index_select(dim=1, index=indices.to(device=inputs.device))
            for indices in self.layer_feature_indices
        ]
        return setup, layers

    def _prepare_setup(self, setup_inputs: torch.Tensor) -> torch.Tensor:
        if self.setup_param_size <= 0:
            return torch.ones(
                (setup_inputs.shape[0], 1),
                device=setup_inputs.device,
                dtype=setup_inputs.dtype,
            )
        return setup_inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.validate_input_shape(inputs)
        inputs = self.normalize_inputs(inputs)
        setup_inputs, layer_inputs = self.split_inputs(inputs)
        batch_size = inputs.shape[0]

        setup_effective = self._prepare_setup(setup_inputs)
        setup_token = self.setup_projection(setup_effective).unsqueeze(1)

        if self.layer_count > 0:
            layer_tensor = torch.stack(layer_inputs, dim=1)
            layer_mask = (layer_tensor.abs().sum(dim=2) == 0.0)
            if self.layer_param_size <= 0:
                layer_tensor = torch.ones(
                    (batch_size, self.layer_count, 1),
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
            layer_tokens = self.layer_projection(layer_tensor)
        else:
            layer_mask = torch.zeros((batch_size, 0), device=inputs.device, dtype=torch.bool)
            layer_tokens = inputs.new_zeros((batch_size, 0, self.model_dim))

        cls_token = self.cls_token.to(device=inputs.device, dtype=inputs.dtype).expand(batch_size, -1, -1)
        cls_token = cls_token + setup_token
        tokens = torch.cat((cls_token, layer_tokens), dim=1)
        tokens = tokens + self.position_embeddings[:, : tokens.shape[1], :].to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        key_padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), device=inputs.device, dtype=torch.bool),
                layer_mask,
            ),
            dim=1,
        )

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.encoder_norm(encoded)
        cls_summary = encoded[:, 0, :]
        if layer_tokens.shape[1] > 0:
            valid_layers = (~layer_mask).unsqueeze(-1)
            layer_summary = (encoded[:, 1:, :] * valid_layers).sum(dim=1) / valid_layers.sum(dim=1).clamp_min(1)
            layer_max = encoded[:, 1:, :].masked_fill(layer_mask.unsqueeze(-1), float("-inf")).amax(dim=1)
            layer_max = torch.where(torch.isfinite(layer_max), layer_max, torch.zeros_like(layer_max))
        else:
            layer_summary = torch.zeros_like(cls_summary)
            layer_max = torch.zeros_like(cls_summary)

        decoder_input = torch.cat((cls_summary, layer_summary, layer_max), dim=1)
        logits = self.spectrum_decoder(decoder_input)
        coarse_spectrum = F.softplus(logits)
        refined_spectrum = F.softplus(self.spectrum_refiner(coarse_spectrum, setup_inputs))
        return refined_spectrum * F.softplus(self.output_scale).to(
            device=logits.device,
            dtype=logits.dtype,
        )

"""Transformer-based surrogate forward model for layered samples."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .base import ForwardModelBase
from .lrn import LRNModel
from .refiners import LocalSpectrumRefiner
from ibamlkit.schema import ForwardModelSchema


class LTNModel(ForwardModelBase):
    """Layerwise Transformer Network 
    Encoder-only transformer surrogate for variable-layer samples."""

    def __init__(
        self,
        schema: ForwardModelSchema,
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
        self.spectrum_refiner = LocalSpectrumRefiner(
            setup_dim=self.setup_param_size,
            hidden_channels=refiner_hidden_channels,
            kernel_size=refiner_kernel_size,
        )
        self.output_scale = nn.Parameter(torch.ones(self.output_size), requires_grad=True)

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

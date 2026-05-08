"""A second LRN variant with latent contribution accumulation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ibamlkit.models.base import ForwardModelBase
from ibamlkit.models.lrn import LRNModel
from ibamlkit.schema import ModelSchema


class _LayerwiseGRUBlock(nn.Module):
    """One recurrent layer block for context-aware per-layer processing."""

    def __init__(
        self,
        *,
        layer_param_size: int,
        setup_param_size: int,
        hidden_size: int,
        contribution_size: int,
        embedding_dim: int,
        block_hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self._use_dummy_layer = layer_param_size <= 0
        self._use_dummy_setup = setup_param_size <= 0
        layer_in_features = 1 if self._use_dummy_layer else layer_param_size
        setup_in_features = 1 if self._use_dummy_setup else setup_param_size

        self.layer_encoder = nn.Sequential(
            nn.Linear(layer_in_features, embedding_dim),
            nn.LeakyReLU(),
        )
        self.setup_encoder = nn.Sequential(
            nn.Linear(setup_in_features, embedding_dim),
            nn.LeakyReLU(),
        )
        self.setup_to_film = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 2 * embedding_dim),
        )
        self.hidden_projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LeakyReLU(),
        )

        fused_layers: list[nn.Module] = []
        prev = embedding_dim * 3
        for width in block_hidden_sizes:
            fused_layers.append(nn.Linear(prev, width))
            fused_layers.append(nn.LeakyReLU())
            prev = width
        self.fused_tower = nn.Sequential(*fused_layers) if fused_layers else nn.Identity()
        self.gru_cell = nn.GRUCell(prev, hidden_size)
        self.contribution_head = nn.Sequential(
            nn.Linear(prev + hidden_size, max(contribution_size, embedding_dim)),
            nn.LeakyReLU(),
            nn.Linear(max(contribution_size, embedding_dim), contribution_size),
        )

    def forward(
        self,
        layer_inputs: torch.Tensor,
        setup_inputs: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_dummy_layer:
            layer_inputs = torch.ones(
                (layer_inputs.shape[0], 1),
                device=layer_inputs.device,
                dtype=layer_inputs.dtype,
            )
        if self._use_dummy_setup:
            setup_inputs = torch.ones(
                (setup_inputs.shape[0], 1),
                device=setup_inputs.device,
                dtype=setup_inputs.dtype,
            )

        layer_feat = self.layer_encoder(layer_inputs)
        setup_feat = self.setup_encoder(setup_inputs)
        gamma, beta = torch.chunk(self.setup_to_film(setup_feat), 2, dim=1)
        gamma = torch.tanh(gamma) + 1.0
        conditioned_layer = layer_feat * gamma + beta
        hidden_feat = self.hidden_projection(hidden_state)
        fused = self.fused_tower(torch.cat((conditioned_layer, setup_feat, hidden_feat), dim=1))
        next_hidden = self.gru_cell(fused, hidden_state)
        contribution = self.contribution_head(torch.cat((fused, next_hidden), dim=1))
        return contribution, next_hidden


class LRNModelV2(ForwardModelBase):
    """Layerwise recurrent network with latent accumulation and shared decoder.

    Unlike :class:`LRNModel`, this variant does not force each layer step to
    predict a full spectrum. Each step emits a compact latent contribution,
    contributions are accumulated across the layer sequence, and a single shared
    decoder maps the accumulated representation to the final spectrum.
    """

    def __init__(
        self,
        schema: ModelSchema,
        *,
        hidden_size: int = 256,
        contribution_size: int = 256,
        setup_embedding_dim: int = 128,
        layer_embedding_dim: int = 256,
        block_hidden_sizes: Sequence[int] = (512, 512),
        decoder_hidden_sizes: Sequence[int] = (1024, 1024),
    ) -> None:
        super().__init__(schema)
        layout = LRNModel._infer_input_layout(schema)
        self.input_layout = layout
        self.setup_param_size = layout.setup_param_size
        self.layer_param_size = layout.layer_param_size
        self.layer_count = layout.layer_count
        self.output_size = LRNModel._infer_output_size(schema)
        self.hidden_size = hidden_size
        self.contribution_size = contribution_size
        self.setup_embedding_dim = setup_embedding_dim
        self.layer_embedding_dim = layer_embedding_dim
        self.setup_feature_indices = torch.tensor(layout.setup_feature_indices, dtype=torch.int64)
        self.layer_feature_indices = tuple(
            torch.tensor(indices, dtype=torch.int64)
            for indices in layout.layer_feature_indices
        )
        self.concentration_indices = [list(offsets) for offsets in layout.concentration_feature_offsets]
        self.layer_feature_names = layout.layer_feature_names

        self.setup_context_encoder = nn.Sequential(
            nn.Linear(max(self.setup_param_size, 1), setup_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(setup_embedding_dim, setup_embedding_dim),
            nn.LeakyReLU(),
        )
        self.layer_block = _LayerwiseGRUBlock(
            layer_param_size=self.layer_param_size,
            setup_param_size=self.setup_param_size,
            hidden_size=hidden_size,
            contribution_size=contribution_size,
            embedding_dim=layer_embedding_dim,
            block_hidden_sizes=block_hidden_sizes,
        )

        decoder_layers: list[nn.Module] = []
        prev = contribution_size + hidden_size + setup_embedding_dim
        for width in decoder_hidden_sizes:
            decoder_layers.append(nn.Linear(prev, width))
            decoder_layers.append(nn.LeakyReLU())
            prev = width
        decoder_layers.append(nn.Linear(prev, self.output_size))
        self.spectrum_decoder = nn.Sequential(*decoder_layers)
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

    def _encode_setup_context(self, setup_inputs: torch.Tensor) -> torch.Tensor:
        if self.setup_param_size <= 0:
            setup_inputs = torch.ones(
                (setup_inputs.shape[0], 1),
                device=setup_inputs.device,
                dtype=setup_inputs.dtype,
            )
        return self.setup_context_encoder(setup_inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.validate_input_shape(inputs)
        inputs = self.normalize_inputs(inputs)
        setup_inputs, layer_inputs = self.split_inputs(inputs)

        batch_size = inputs.shape[0]
        hidden_state = torch.zeros(
            (batch_size, self.hidden_size),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        contribution_total = torch.zeros(
            (batch_size, self.contribution_size),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        setup_context = self._encode_setup_context(setup_inputs)

        for current_layer in layer_inputs:
            mask = (~(current_layer == 0).all(dim=1)).float().unsqueeze(1)
            if mask.sum() == 0:
                break
            contribution, next_hidden = self.layer_block(current_layer, setup_inputs, hidden_state)
            contribution_total = contribution_total + contribution * mask
            hidden_state = hidden_state * (1.0 - mask) + next_hidden * mask

        decoder_input = torch.cat((contribution_total, hidden_state, setup_context), dim=1)
        logits = self.spectrum_decoder(decoder_input)
        return F.softplus(logits) * F.softplus(self.output_scale).to(
            device=logits.device,
            dtype=logits.dtype,
        )

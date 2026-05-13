"""An LRN variant with latent contribution accumulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


from .base import ForwardModelBase
from .base import ForwardModelBase
from ibamlkit.schema import (
    DatasetInputSpec,
    ForwardModelSchema,
    ModelInputSpec,
    ModelOutputSpec,
    ModelTaskSpec,
    TensorFeatureSpec,
)


@dataclass(frozen=True)
class _InputLayout:
    setup_feature_indices: tuple[int, ...]
    layer_feature_indices: tuple[tuple[int, ...], ...]
    concentration_feature_offsets: tuple[tuple[int, ...], ...]
    layer_feature_names: tuple[str, ...]

    @property
    def setup_param_size(self) -> int:
        return len(self.setup_feature_indices)

    @property
    def layer_param_size(self) -> int:
        if not self.layer_feature_indices:
            return 0
        return len(self.layer_feature_indices[0])

    @property
    def layer_count(self) -> int:
        return len(self.layer_feature_indices)


def _layer_slot_key(feature: TensorFeatureSpec) -> tuple[str, str, str]:
    slot = str(feature.metadata.get("layer_slot", "")).strip()
    if not slot:
        slot = feature.name
    element = str(feature.metadata.get("element", "")).strip()
    return (slot, feature.transform, element)


def build_lrn_model_schema(
    input_spec: DatasetInputSpec,
    *,
    model_name: str = "lrn",
    task_method_names: Sequence[str] | None = None,
    output_spectra_lengths: dict[str, int] | None = None,
) -> ForwardModelSchema:
    """Build a default LRN model schema from a dataset input spec.

    The resulting input spec preserves the dataset open-parameter order while
    annotating per-layer features with a canonical ``layer_slot`` so different
    layers can be recognized as instances of the same local feature layout.
    """

    task_method_names = tuple(task_method_names or [method.name for method in input_spec.methods])
    allowed_setup_methods = set(task_method_names)
    input_features: list[TensorFeatureSpec] = []
    for parameter in input_spec.open_parameters:
        if parameter.group == "setup":
            if parameter.method and parameter.method not in allowed_setup_methods:
                continue
            input_features.append(
                TensorFeatureSpec(
                    name=parameter.name,
                    source_parameter=parameter.name,
                    role="input",
                    group="setup",
                    unit=parameter.unit,
                    metadata={"method": parameter.method, "kind": parameter.kind},
                )
            )
            continue

        if parameter.group != "layer":
            continue

        kind = str(parameter.kind).strip().lower().replace("-", "_").replace(" ", "_")
        slot = kind
        transform = "identity"
        metadata = {
            "kind": parameter.kind,
            "element": parameter.element,
            "isotope": parameter.isotope,
        }
        if kind == "concentration":
            slot = f"concentration:{parameter.element}:{parameter.isotope}"
            transform = "normalize_concentration"
        elif parameter.element:
            slot = f"{kind}:{parameter.element}:{parameter.isotope}"

        metadata["layer_slot"] = slot
        input_features.append(
            TensorFeatureSpec(
                name=parameter.name,
                source_parameter=parameter.name,
                role="input",
                group="layer",
                layer_index=parameter.layer_index,
                transform=transform,
                unit=parameter.unit,
                metadata=metadata,
            )
        )

    if output_spectra_lengths is None:
        output_spectra_lengths = {method_name: 1 for method_name in task_method_names}

    return ForwardModelSchema(
        name=model_name,
        task=ModelTaskSpec(
            task_kind="surrogate",
            method_names=list(task_method_names),
        ),
        inputs=ModelInputSpec(features=input_features, layout="grouped_by_layer"),
        outputs=ModelOutputSpec(
            spectra_names=list(task_method_names),
            spectra_lengths=output_spectra_lengths,
        ),
        parameters=input_spec.parameters,
        metadata={"source": "DatasetInputSpec"},
    )


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


class _LocalSpectrumRefiner(nn.Module):
    """Inject local channel structure through a lightweight conv residual head."""

    def __init__(
        self,
        *,
        output_size: int,
        setup_dim: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1.")

        self.output_size = int(output_size)
        self.hidden_channels = int(hidden_channels)
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


class LRNModel(ForwardModelBase):
    """Layerwise recurrent network with latent accumulation and shared decoder.

    Unlike :class:`LRNModel`, this variant does not force each layer step to
    predict a full spectrum. Each step emits a compact latent contribution,
    contributions are accumulated across the layer sequence, and a single shared
    decoder maps the accumulated representation to the final spectrum.
    """

    def __init__(
        self,
        schema: ForwardModelSchema,
        *,
        hidden_size: int = 256,
        contribution_size: int = 256,
        setup_embedding_dim: int = 128,
        layer_embedding_dim: int = 256,
        block_hidden_sizes: Sequence[int] = (512, 512),
        decoder_hidden_sizes: Sequence[int] = (1024, 1024),
        refiner_hidden_channels: int = 32,
        refiner_kernel_size: int = 17,
    ) -> None:
        super().__init__(schema)
        layout = self._infer_input_layout(schema)
        self.input_layout = layout
        self.setup_param_size = layout.setup_param_size
        self.layer_param_size = layout.layer_param_size
        self.layer_count = layout.layer_count
        self.output_size = self._infer_output_size(schema)
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
        self.spectrum_refiner = _LocalSpectrumRefiner(
            output_size=self.output_size,
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
            layer_indices = self.layer_feature_indices[layer_index].to(device=inputs.device)
            concentration_index = torch.tensor(
                concentration_indices,
                device=inputs.device,
                dtype=torch.long,
            )
            current = normalized.index_select(
                dim=1,
                index=layer_indices,
            )
            concentrations = current.index_select(dim=1, index=concentration_index).clamp_min(0.0)
            sums = concentrations.sum(dim=1, keepdim=True)
            concentrations = concentrations / torch.where(
                sums > 0.0,
                sums,
                torch.ones_like(sums),
            )
            current = current.scatter(
                dim=1,
                index=concentration_index.unsqueeze(0).expand(current.shape[0], -1),
                src=concentrations,
            )
            normalized = normalized.scatter(
                dim=1,
                index=layer_indices.unsqueeze(0).expand(inputs.shape[0], -1),
                src=current,
            )
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
            contribution, next_hidden = self.layer_block(current_layer, setup_inputs, hidden_state)
            contribution_total = contribution_total + contribution * mask
            hidden_state = hidden_state * (1.0 - mask) + next_hidden * mask

        decoder_input = torch.cat((contribution_total, hidden_state, setup_context), dim=1)
        logits = self.spectrum_decoder(decoder_input)
        coarse_spectrum = F.softplus(logits)
        refined_spectrum = F.softplus(self.spectrum_refiner(coarse_spectrum, setup_inputs))
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

    @staticmethod
    def _infer_input_layout(schema: ForwardModelSchema) -> _InputLayout:
        setup_feature_indices: list[int] = []
        layer_groups: dict[int, list[tuple[int, TensorFeatureSpec]]] = {}

        for feature_index, feature in enumerate(schema.inputs.features):
            if feature.group == "setup":
                setup_feature_indices.append(feature_index)
                continue
            if feature.group == "layer":
                if feature.layer_index < 1:
                    raise ValueError(
                        "Layer input features must define a 1-based layer_index in the model schema."
                    )
                layer_groups.setdefault(feature.layer_index, []).append((feature_index, feature))

        if not layer_groups:
            return _InputLayout(
                setup_feature_indices=tuple(setup_feature_indices),
                layer_feature_indices=tuple(),
                concentration_feature_offsets=tuple(),
                layer_feature_names=tuple(),
            )

        ordered_layer_indices = sorted(layer_groups.keys())
        expected_layer_indices = list(range(1, ordered_layer_indices[-1] + 1))
        if ordered_layer_indices != expected_layer_indices:
            raise ValueError(
                "LRNModel requires contiguous layer indices starting at 1 in the model schema."
            )

        template_features = [feature for _, feature in layer_groups[ordered_layer_indices[0]]]
        template_signature = [_layer_slot_key(feature) for feature in template_features]

        layer_feature_indices: list[tuple[int, ...]] = []
        concentration_feature_offsets: list[tuple[int, ...]] = []
        for layer_index in ordered_layer_indices:
            current = layer_groups[layer_index]
            current_features = [feature for _, feature in current]
            current_signature = [_layer_slot_key(feature) for feature in current_features]
            if current_signature != template_signature:
                raise ValueError(
                    "LRNModel requires the same ordered feature schema in every layer."
                )
            layer_feature_indices.append(tuple(feature_index for feature_index, _ in current))
            concentration_feature_offsets.append(
                tuple(
                    offset
                    for offset, feature in enumerate(current_features)
                    if feature.transform in {
                        "normalize_group_sum",
                        "normalize_concentration",
                        "concentration_fraction",
                    }
                )
            )

        return _InputLayout(
            setup_feature_indices=tuple(setup_feature_indices),
            layer_feature_indices=tuple(layer_feature_indices),
            concentration_feature_offsets=tuple(concentration_feature_offsets),
            layer_feature_names=tuple(feature.name for feature in template_features),
        )

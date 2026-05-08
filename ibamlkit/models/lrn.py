"""LRN surrogate model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ibamlkit.models.base import ForwardModelBase
from ibamlkit.schema import (
    DatasetInputSpec,
    ModelInputSpec,
    ModelOutputSpec,
    ModelSchema,
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
) -> ModelSchema:
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

    return ModelSchema(
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


class _LRNLayerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_param_size: int,
        setup_param_size: int,
        output_size: int,
        hidden_nodes: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.max_exp = 9.0

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

        core_layers: list[nn.Module] = []
        prev = embedding_dim * 3
        for width in hidden_nodes:
            core_layers.append(nn.Linear(prev, width))
            core_layers.append(nn.LeakyReLU())
            prev = width
        self.core = nn.Sequential(*core_layers) if core_layers else nn.Identity()
        self.output_head = nn.Linear(prev, output_size)
        self.hidden_head = nn.Linear(prev, hidden_size)

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
        fused = self.core(torch.cat((conditioned_layer, setup_feat, hidden_feat), dim=1))
        logits = self.output_head(fused)
        next_hidden = hidden_state + self.hidden_head(fused)
        return torch.exp(torch.clamp(logits, max=self.max_exp)) - 1.0, next_hidden


class _LearnableKernelRefiner(nn.Module):
    def __init__(
        self,
        setup_dim: int,
        kernel_size: int = 65,
        kernel_hidden: int = 128,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.pad = self.kernel_size // 2
        self._use_dummy_setup = setup_dim <= 0
        effective_in = 1 if self._use_dummy_setup else setup_dim

        self.kernel_generator = nn.Sequential(
            nn.Linear(effective_in, kernel_hidden),
            nn.LeakyReLU(),
            nn.Linear(kernel_hidden, kernel_hidden),
            nn.LeakyReLU(),
            nn.Linear(kernel_hidden, self.kernel_size),
        )
        self.kernel_bias = nn.Parameter(torch.zeros(self.kernel_size))

        if padding_mode == "constant":
            self.pad_layer = nn.ConstantPad1d((self.pad, self.pad), 0.0)
        elif padding_mode == "replicate":
            self.pad_layer = nn.ReplicationPad1d(self.pad)
        else:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        extractor = torch.zeros(self.kernel_size, 1, self.kernel_size, dtype=torch.float32)
        for index in range(self.kernel_size):
            extractor[index, 0, index] = 1.0
        self.register_buffer("extractor", extractor)

    def _kernel(self, setup_inputs: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._use_dummy_setup:
            setup_inputs = torch.ones((setup_inputs.shape[0], 1), device=device, dtype=dtype)
        else:
            setup_inputs = setup_inputs.to(device=device, dtype=dtype)

        kernel = self.kernel_generator(setup_inputs) + self.kernel_bias.to(device=device, dtype=dtype)
        kernel = F.softplus(kernel)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-12)
        return kernel

    def forward(self, spectra: torch.Tensor, setup_inputs: torch.Tensor) -> torch.Tensor:
        original_is_2d = spectra.dim() == 2
        spectra = spectra.reshape(spectra.size(0), 1, -1)
        padded = self.pad_layer(spectra)
        windows = F.conv1d(padded, self.extractor, padding=0).permute(0, 2, 1)
        kernels = self._kernel(setup_inputs, spectra.device, spectra.dtype)
        refined = (windows * kernels.unsqueeze(1)).sum(dim=-1).unsqueeze(1)
        return refined.squeeze(1) if original_is_2d else refined


class LRNModel(ForwardModelBase):
    """Layerwise recurrent network surrogate model."""

    def __init__(
        self,
        schema: ModelSchema,
        *,
        hidden_size: int = 256,
        hidden_nodes: Sequence[int] = (512, 1024),
        layer_embedding_dim: int = 256,
        kernel_size: int = 65,
    ) -> None:
        super().__init__(schema)
        layout = self._infer_input_layout(schema)
        self.input_layout = layout
        self.setup_param_size = layout.setup_param_size
        self.layer_param_size = layout.layer_param_size
        self.layer_count = layout.layer_count
        self.output_size = self._infer_output_size(schema)
        self.hidden_size = hidden_size
        self.hidden_nodes = tuple(hidden_nodes)
        self.layer_embedding_dim = layer_embedding_dim
        self.setup_feature_indices = torch.tensor(
            layout.setup_feature_indices,
            dtype=torch.int64,
        )
        self.layer_feature_indices = tuple(
            torch.tensor(indices, dtype=torch.int64)
            for indices in layout.layer_feature_indices
        )
        self.concentration_indices = [list(offsets) for offsets in layout.concentration_feature_offsets]
        self.layer_feature_names = layout.layer_feature_names

        self.layer_block = _LRNLayerBlock(
            hidden_size=hidden_size,
            layer_param_size=self.layer_param_size,
            setup_param_size=self.setup_param_size,
            output_size=self.output_size,
            hidden_nodes=self.hidden_nodes,
            embedding_dim=layer_embedding_dim,
        )
        self.refiner = _LearnableKernelRefiner(
            setup_dim=self.setup_param_size,
            kernel_size=kernel_size,
            kernel_hidden=layer_embedding_dim,
        )
        self.norm_param = nn.Parameter(torch.ones(self.output_size), requires_grad=True)

    @staticmethod
    def _infer_output_size(schema: ModelSchema) -> int:
        if schema.outputs.spectra_lengths:
            return sum(schema.outputs.spectra_lengths.values())
        if schema.outputs.feature_dimension > 0:
            return schema.outputs.feature_dimension
        raise ValueError("Unable to infer output size from schema.")

    @staticmethod
    def _infer_input_layout(schema: ModelSchema) -> _InputLayout:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.validate_input_shape(inputs)
        inputs = self.normalize_inputs(inputs)
        setup_inputs, layer_inputs = self.split_inputs(inputs)

        batch_size = inputs.shape[0]
        hidden_state = torch.ones((batch_size, self.hidden_size), device=inputs.device, dtype=inputs.dtype)
        spectra = torch.zeros((batch_size, self.output_size), device=inputs.device, dtype=inputs.dtype)

        for current_layer in layer_inputs:
            mask = (~(current_layer == 0).all(dim=1)).float().unsqueeze(1)
            if mask.sum() == 0:
                break
            partial_spectrum, next_hidden = self.layer_block(current_layer, setup_inputs, hidden_state)
            spectra = spectra + partial_spectrum * mask
            hidden_state = hidden_state * (1.0 - mask) + next_hidden * mask

        spectra = self.refiner(spectra.unsqueeze(1), setup_inputs).squeeze(1)
        return spectra * F.softplus(self.norm_param).to(device=spectra.device, dtype=spectra.dtype)

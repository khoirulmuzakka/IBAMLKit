"""Dataset preparation helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ibamlkit.data import DatasetBatchReader
from ibamlkit.schema import IBADataset, ModelSchema, ParameterSpec


@dataclass(frozen=True)
class PreparedSurrogateDataset:
    """Canonical matrices prepared for surrogate-model training."""

    reference_dataset: IBADataset
    inputs_full: np.ndarray
    inputs_selected: np.ndarray
    targets: np.ndarray
    target_lengths: np.ndarray | None


def _open_parameter_index(parameters: list[ParameterSpec]) -> dict[str, int]:
    return {parameter.name: index for index, parameter in enumerate(parameters)}


def prepare_variable_layer_surrogate_dataset(
    datasets: list[IBADataset],
    *,
    schema: ModelSchema,
    method_name: str,
) -> PreparedSurrogateDataset:
    """Pad and merge variable-layer datasets into one max-layer training matrix.

    The function assumes each dataset shard was generated from the same setup
    schema family while varying only the number of physical layers. Inputs are
    projected into the open-parameter layout of the largest-layer dataset and
    then reduced to the ordered input features declared by ``schema``.
    """

    if not datasets:
        raise ValueError("At least one dataset is required.")

    reader = DatasetBatchReader()
    ordered = sorted(
        datasets,
        key=lambda dataset: int(dataset.input_spec.generation_info.get("n_layers", 0)),
    )
    reference_dataset = ordered[-1]
    reference_open_parameters = list(reference_dataset.input_spec.open_parameters)
    reference_setup_names = [
        parameter.name
        for parameter in reference_open_parameters
        if parameter.group == "setup"
    ]
    reference_index = _open_parameter_index(reference_open_parameters)

    full_input_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    length_blocks: list[np.ndarray] = []

    for dataset in ordered:
        open_parameters = list(dataset.input_spec.open_parameters)
        setup_names = [
            parameter.name
            for parameter in open_parameters
            if parameter.group == "setup"
        ]
        if setup_names != reference_setup_names:
            raise ValueError(
                "Setup open-parameter ordering differs across layer-count datasets."
            )
        if method_name not in dataset.spectra:
            raise KeyError(f"Method {method_name!r} not found in dataset spectra.")

        full_inputs = np.zeros(
            (dataset.sample_count, len(reference_open_parameters)),
            dtype=np.float32,
        )
        for local_index, parameter in enumerate(open_parameters):
            full_inputs[:, reference_index[parameter.name]] = dataset.open_parameter_values[
                :, local_index
            ]

        full_input_blocks.append(full_inputs)
        target_blocks.append(np.asarray(dataset.spectra[method_name], dtype=np.float32))
        if dataset.spectra_lengths is not None:
            length_blocks.append(
                np.asarray(dataset.spectra_lengths[method_name], dtype=np.int32)
            )

    inputs_full = np.concatenate(full_input_blocks, axis=0)
    targets = reader.concatenate_padded_spectra(target_blocks)
    target_lengths = np.concatenate(length_blocks, axis=0) if length_blocks else None

    selected_indices = []
    for feature in schema.inputs.features:
        try:
            selected_indices.append(reference_index[feature.source_parameter])
        except KeyError as exc:
            raise KeyError(
                f"Schema input feature {feature.name!r} references unknown parameter "
                f"{feature.source_parameter!r} for the reference dataset."
            ) from exc
    inputs_selected = np.asarray(inputs_full[:, selected_indices], dtype=np.float32)

    return PreparedSurrogateDataset(
        reference_dataset=reference_dataset,
        inputs_full=inputs_full,
        inputs_selected=inputs_selected,
        targets=targets,
        target_lengths=target_lengths,
    )

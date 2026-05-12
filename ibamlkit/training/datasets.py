"""Dataset preparation helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ibamlkit.data import DatasetBatchReader
from ibamlkit.schema import ForwardModelSchema, IBADataset, InverseModelSchema, ParameterSpec


@dataclass(frozen=True)
class PreparedSurrogateDataset:
    """Canonical matrices prepared for surrogate-model training."""

    reference_dataset: IBADataset
    inputs_full: np.ndarray
    inputs_selected: np.ndarray
    targets: np.ndarray
    target_lengths: np.ndarray | None


@dataclass(frozen=True)
class PreparedInverseDataset:
    """Canonical matrices prepared for inverse-model training."""

    reference_dataset: IBADataset
    inputs: np.ndarray
    targets_full: np.ndarray
    targets_selected: np.ndarray
    target_parameter_names: tuple[str, ...]
    input_lengths: np.ndarray | None


def _open_parameter_index(parameters: list[ParameterSpec]) -> dict[str, int]:
    return {parameter.name: index for index, parameter in enumerate(parameters)}


def prepare_variable_layer_surrogate_dataset(
    datasets: list[IBADataset],
    *,
    schema: ForwardModelSchema,
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


def prepare_inverse_dataset(
    dataset: IBADataset,
    *,
    schema: InverseModelSchema,
    method_name: str,
) -> PreparedInverseDataset:
    """Project one IBADataset into matrices for inverse-model training.

    Inputs come from one selected measured/simulated spectrum. Targets come from
    the dataset open-parameter matrix and are reduced to the parameter order
    declared in ``schema.outputs.features``.
    """

    if method_name not in dataset.spectra:
        raise KeyError(f"Method {method_name!r} not found in dataset spectra.")
    inputs = np.asarray(dataset.spectra[method_name], dtype=np.float32)
    targets_full = np.asarray(dataset.open_parameter_values, dtype=np.float32)
    open_parameters = list(dataset.input_spec.open_parameters)
    open_parameter_index = _open_parameter_index(open_parameters)

    selected_indices: list[int] = []
    target_parameter_names: list[str] = []
    for feature in schema.outputs.features:
        try:
            selected_indices.append(open_parameter_index[feature.source_parameter])
        except KeyError as exc:
            raise KeyError(
                f"Schema output feature {feature.name!r} references unknown open parameter "
                f"{feature.source_parameter!r}."
            ) from exc
        target_parameter_names.append(feature.source_parameter)

    targets_selected = np.asarray(targets_full[:, selected_indices], dtype=np.float32)
    input_lengths = None
    if dataset.spectra_lengths is not None:
        input_lengths = np.asarray(dataset.spectra_lengths[method_name], dtype=np.int32)

    return PreparedInverseDataset(
        reference_dataset=dataset,
        inputs=inputs,
        targets_full=targets_full,
        targets_selected=targets_selected,
        target_parameter_names=tuple(target_parameter_names),
        input_lengths=input_lengths,
    )

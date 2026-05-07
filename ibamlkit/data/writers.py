"""HDF5 writer for the canonical IBAMLKit dataset schema."""

from __future__ import annotations

import json

import numpy as np

from ..schema import IBADataset
from ..schema.constants import (
    DATASET_FORMAT_NAME,
    ROOT_ATTR_FORMAT,
    ROOT_ATTR_VERSION,
)


def _string_dtype():
    import h5py

    return h5py.string_dtype(encoding="utf-8")


def write_hdf5_dataset(path: str, dataset: IBADataset) -> None:
    """Serialize an :class:`IBADataset` to an HDF5 file."""

    import h5py

    with h5py.File(path, "w") as handle:
        handle.attrs[ROOT_ATTR_FORMAT] = DATASET_FORMAT_NAME
        handle.attrs[ROOT_ATTR_VERSION] = dataset.input_spec.format_version

        input_group = handle.create_group("input")
        methods_group = input_group.create_group("methods")
        methods_group.create_dataset(
            "order",
            data=np.asarray([method.name for method in dataset.input_spec.methods], dtype=_string_dtype()),
        )
        for method in dataset.input_spec.methods:
            method_group = methods_group.create_group(method.name)
            method_group.create_dataset("reference_file", data=method.reference_file, dtype=_string_dtype())
            method_group.create_dataset("file_type", data=method.file_type, dtype=_string_dtype())
            method_group.create_dataset(
                "metadata_json",
                data=json.dumps(dict(method.metadata)),
                dtype=_string_dtype(),
            )

        layers_group = input_group.create_group("layers")
        layer_dtype = np.dtype(
            [
                ("layer_index", np.int32),
                ("element", _string_dtype()),
                ("isotope", _string_dtype()),
            ]
        )
        layer_rows = np.empty(len(dataset.input_spec.layer_species), dtype=layer_dtype)
        for index, entry in enumerate(dataset.input_spec.layer_species):
            layer_rows[index] = (entry.layer_index, entry.element, entry.isotope)
        layers_group.create_dataset("species_table", data=layer_rows)
        layers_group.create_dataset(
            "num_layers",
            data=max((entry.layer_index for entry in dataset.input_spec.layer_species), default=0),
        )

        parameters_group = input_group.create_group("parameters")
        parameter_dtype = np.dtype(
            [
                ("name", _string_dtype()),
                ("group", _string_dtype()),
                ("kind", _string_dtype()),
                ("is_open", np.bool_),
                ("method", _string_dtype()),
                ("layer_index", np.int32),
                ("element", _string_dtype()),
                ("isotope", _string_dtype()),
                ("unit", _string_dtype()),
                ("lower_bound", np.float64),
                ("upper_bound", np.float64),
                ("fixed_value", np.float64),
            ]
        )
        parameter_rows = np.empty(len(dataset.input_spec.parameters), dtype=parameter_dtype)
        for index, entry in enumerate(dataset.input_spec.parameters):
            parameter_rows[index] = (
                entry.name,
                entry.group,
                entry.kind,
                entry.is_open,
                entry.method,
                entry.layer_index,
                entry.element,
                entry.isotope,
                entry.unit,
                np.nan if entry.lower_bound is None else float(entry.lower_bound),
                np.nan if entry.upper_bound is None else float(entry.upper_bound),
                np.nan if entry.fixed_value is None else float(entry.fixed_value),
            )
        parameters_group.create_dataset("table", data=parameter_rows)
        parameters_group.create_dataset(
            "open_names",
            data=np.asarray(
                [entry.name for entry in dataset.input_spec.open_parameters], dtype=_string_dtype()
            ),
        )
        parameters_group.create_dataset(
            "fixed_names",
            data=np.asarray(
                [entry.name for entry in dataset.input_spec.fixed_parameters], dtype=_string_dtype()
            ),
        )
        parameters_group.create_dataset(
            "fixed_values",
            data=np.asarray(
                [float(entry.fixed_value) for entry in dataset.input_spec.fixed_parameters],
                dtype=np.float32,
            ),
        )

        generation_info_group = input_group.create_group("generation_info")
        generation_info_group.create_dataset(
            "metadata_json",
            data=json.dumps(dict(dataset.input_spec.generation_info)),
            dtype=_string_dtype(),
        )

        samples_group = handle.create_group("samples")
        samples_group.create_dataset(
            "open_parameter_values",
            data=np.asarray(dataset.open_parameter_values, dtype=np.float32),
        )
        if dataset.sample_ids is not None:
            samples_group.create_dataset(
                "sample_ids",
                data=np.asarray(list(dataset.sample_ids), dtype=_string_dtype()),
            )

        spectra_group = samples_group.create_group("spectra")
        for method_name, spectrum_matrix in dataset.spectra.items():
            spectra_group.create_dataset(
                method_name,
                data=np.asarray(spectrum_matrix, dtype=np.float32),
            )
        if dataset.spectra_lengths is not None:
            lengths_group = samples_group.create_group("spectra_lengths")
            for method_name, lengths in dataset.spectra_lengths.items():
                lengths_group.create_dataset(
                    method_name,
                    data=np.asarray(lengths, dtype=np.int32),
                )

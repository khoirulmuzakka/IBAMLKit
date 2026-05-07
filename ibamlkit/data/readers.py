"""HDF5 reader for the canonical IBAMLKit dataset schema."""

from __future__ import annotations

import json

import numpy as np

from ..schema import (
    DatasetInputSpec,
    IBADataset,
    LayerSpeciesSpec,
    MethodSpec,
    ParameterSpec,
)


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_optional_float(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def read_hdf5_dataset(path: str) -> IBADataset:
    """Read an :class:`IBADataset` from an HDF5 file."""

    import h5py

    with h5py.File(path, "r") as handle:
        input_group = handle["input"]

        methods_order = [_decode_string(name) for name in input_group["methods"]["order"][()]]
        methods = []
        for name in methods_order:
            method_group = input_group["methods"][name]
            methods.append(
                MethodSpec(
                    name=name,
                    reference_file=_decode_string(method_group["reference_file"][()]),
                    file_type=_decode_string(method_group["file_type"][()]),
                    metadata=json.loads(_decode_string(method_group["metadata_json"][()])),
                )
            )

        layer_species = []
        for row in input_group["layers"]["species_table"][()]:
            layer_species.append(
                LayerSpeciesSpec(
                    layer_index=int(row["layer_index"]),
                    element=_decode_string(row["element"]),
                    isotope=_decode_string(row["isotope"]),
                )
            )

        parameters = []
        for row in input_group["parameters"]["table"][()]:
            parameters.append(
                ParameterSpec(
                    name=_decode_string(row["name"]),
                    group=_decode_string(row["group"]),
                    kind=_decode_string(row["kind"]),
                    is_open=bool(row["is_open"]),
                    method=_decode_string(row["method"]),
                    layer_index=int(row["layer_index"]),
                    element=_decode_string(row["element"]),
                    isotope=_decode_string(row["isotope"]),
                    unit=_decode_string(row["unit"]),
                    lower_bound=_decode_optional_float(float(row["lower_bound"])),
                    upper_bound=_decode_optional_float(float(row["upper_bound"])),
                    fixed_value=_decode_optional_float(float(row["fixed_value"])),
                )
            )

        input_spec = DatasetInputSpec(
            methods=methods,
            layer_species=layer_species,
            parameters=parameters,
            generation_info=json.loads(
                _decode_string(input_group["generation_info"]["metadata_json"][()])
            ),
        )

        samples_group = handle["samples"]
        sample_ids = None
        if "sample_ids" in samples_group:
            sample_ids = [_decode_string(value) for value in samples_group["sample_ids"][()]]

        spectra = {
            method_name: np.asarray(samples_group["spectra"][method_name][()], dtype=np.float32)
            for method_name in methods_order
        }
        spectra_lengths = None
        if "spectra_lengths" in samples_group:
            spectra_lengths = {
                method_name: np.asarray(
                    samples_group["spectra_lengths"][method_name][()],
                    dtype=np.int32,
                )
                for method_name in methods_order
            }

        return IBADataset(
            input_spec=input_spec,
            open_parameter_values=np.asarray(
                samples_group["open_parameter_values"][()], dtype=np.float32
            ),
            spectra=spectra,
            spectra_lengths=spectra_lengths,
            sample_ids=sample_ids,
        )

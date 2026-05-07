"""Canonical schema for rectangular IBA/ML datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import json

import numpy as np

from .layers import LayerSpeciesSpec
from .setup import MethodSpec
from .versions import DATASET_FORMAT_VERSION


@dataclass(frozen=True)
class ParameterSpec:
    """Semantic description of one parameter in the dataset."""

    name: str
    group: str
    kind: str
    is_open: bool
    method: str = ""
    layer_index: int = -1
    element: str = ""
    isotope: str = ""
    unit: str = ""
    lower_bound: float | None = None
    upper_bound: float | None = None
    fixed_value: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter name must not be empty.")
        if self.group not in {"setup", "layer"}:
            raise ValueError("Parameter group must be either 'setup' or 'layer'.")
        if self.layer_index == 0 or self.layer_index < -1:
            raise ValueError("layer_index must be -1 or a 1-based positive integer.")
        if self.is_open and self.fixed_value is not None:
            raise ValueError(f"Open parameter '{self.name}' must not define fixed_value.")
        if not self.is_open and self.fixed_value is None:
            raise ValueError(f"Fixed parameter '{self.name}' must define fixed_value.")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(f"Parameter '{self.name}' has lower_bound > upper_bound.")


@dataclass(frozen=True)
class DatasetInputSpec:
    """Dataset-level generation context shared by all samples."""

    methods: Sequence[MethodSpec]
    layer_species: Sequence[LayerSpeciesSpec]
    parameters: Sequence[ParameterSpec]
    generation_info: Mapping[str, Any] = field(default_factory=dict)
    format_version: str = DATASET_FORMAT_VERSION

    def __post_init__(self) -> None:
        method_names = [method.name for method in self.methods]
        if not method_names:
            raise ValueError("At least one method must be defined.")
        if len(method_names) != len(set(method_names)):
            raise ValueError("Method names must be unique.")

        parameter_names = [parameter.name for parameter in self.parameters]
        if not parameter_names:
            raise ValueError("At least one parameter must be defined.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Parameter names must be unique.")

        open_parameters = [parameter for parameter in self.parameters if parameter.is_open]
        if not open_parameters:
            raise ValueError("At least one open parameter must be defined.")

        fixed_parameters = [parameter for parameter in self.parameters if not parameter.is_open]
        if any(parameter.fixed_value is None for parameter in fixed_parameters):
            raise ValueError("Every fixed parameter must define a fixed_value.")

        method_name_set = set(method_names)
        for entry in self.parameters:
            if entry.method and entry.method not in method_name_set:
                raise ValueError(
                    f"Parameter '{entry.name}' references unknown method '{entry.method}'."
                )

        max_layer_index = max((item.layer_index for item in self.layer_species), default=0)
        for entry in self.parameters:
            if entry.layer_index > max_layer_index and max_layer_index > 0:
                raise ValueError(
                    f"Parameter '{entry.name}' references undefined layer {entry.layer_index}."
                )

        json.dumps(dict(self.generation_info))

    @property
    def open_parameters(self) -> list[ParameterSpec]:
        return [parameter for parameter in self.parameters if parameter.is_open]

    @property
    def fixed_parameters(self) -> list[ParameterSpec]:
        return [parameter for parameter in self.parameters if not parameter.is_open]


@dataclass(frozen=True)
class IBADataset:
    """Rectangular dataset for IBA machine-learning workflows."""

    input_spec: DatasetInputSpec
    open_parameter_values: np.ndarray
    spectra: Mapping[str, np.ndarray]
    spectra_lengths: Mapping[str, np.ndarray] | None = None
    sample_ids: Sequence[str] | None = None

    def __post_init__(self) -> None:
        open_values = np.asarray(self.open_parameter_values, dtype=np.float32)
        if open_values.ndim != 2:
            raise ValueError("open_parameter_values must be a 2D array.")

        expected_parameter_count = len(self.input_spec.open_parameters)
        if open_values.shape[1] != expected_parameter_count:
            raise ValueError(
                "open_parameter_values column count does not match the number of open parameters."
            )

        method_names = [method.name for method in self.input_spec.methods]
        if set(self.spectra.keys()) != set(method_names):
            raise ValueError("Spectra keys must exactly match the declared method names.")

        sample_count = open_values.shape[0]
        for method_name in method_names:
            spectrum_matrix = np.asarray(self.spectra[method_name], dtype=np.float32)
            if spectrum_matrix.ndim != 2:
                raise ValueError(f"Spectrum matrix for '{method_name}' must be 2D.")
            if spectrum_matrix.shape[0] != sample_count:
                raise ValueError(
                    f"Spectrum matrix for '{method_name}' has inconsistent sample count."
                )
            if self.spectra_lengths is not None:
                if method_name not in self.spectra_lengths:
                    raise ValueError(
                        f"Missing spectra_lengths entry for method '{method_name}'."
                    )
                lengths = np.asarray(self.spectra_lengths[method_name], dtype=np.int32)
                if lengths.ndim != 1:
                    raise ValueError(
                        f"spectra_lengths for '{method_name}' must be a 1D array."
                    )
                if lengths.shape[0] != sample_count:
                    raise ValueError(
                        f"spectra_lengths for '{method_name}' has inconsistent sample count."
                    )
                if np.any(lengths < 0):
                    raise ValueError(
                        f"spectra_lengths for '{method_name}' must be non-negative."
                    )
                if np.any(lengths > spectrum_matrix.shape[1]):
                    raise ValueError(
                        f"spectra_lengths for '{method_name}' exceeds padded spectrum width."
                    )

        if self.sample_ids is not None:
            if len(self.sample_ids) != sample_count:
                raise ValueError("sample_ids length must match the number of samples.")
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError("sample_ids must be unique.")

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.open_parameter_values).shape[0])

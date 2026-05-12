"""Backend-neutral validation interfaces and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from ibamlkit.schema import DatasetInputSpec, ParameterSpec


@dataclass(frozen=True)
class SimulationBatchResult:
    """Batch simulation output in canonical rectangular form."""

    spectra: Mapping[str, np.ndarray]
    spectra_lengths: Mapping[str, np.ndarray] | None = None

    @property
    def sample_count(self) -> int:
        if not self.spectra:
            return 0
        first = next(iter(self.spectra.values()))
        return int(np.asarray(first).shape[0])


class SpectrumSimulator(Protocol):
    """Protocol implemented by validation backends."""

    method_names: Sequence[str]

    def simulate_batch(
        self,
        open_parameter_values: np.ndarray,
        *,
        fixed_parameter_overrides: Mapping[str, float] | None = None,
        nthreads: int = 1,
    ) -> SimulationBatchResult:
        ...


def validate_open_parameter_matrix(
    input_spec: DatasetInputSpec,
    open_parameter_values: np.ndarray,
) -> np.ndarray:
    """Validate and normalize a batch of open-parameter vectors."""

    matrix = np.asarray(open_parameter_values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("open_parameter_values must be a 2D array.")
    expected_columns = len(input_spec.open_parameters)
    if matrix.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} open-parameter columns, got {matrix.shape[1]}."
        )
    return matrix


def resolve_fixed_parameter_values(
    input_spec: DatasetInputSpec,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve fixed parameters with optional shared overrides."""

    fixed_values = {
        parameter.name: float(parameter.fixed_value)
        for parameter in input_spec.fixed_parameters
    }
    if overrides is None:
        return fixed_values

    known_fixed = set(fixed_values.keys())
    unknown = sorted(set(overrides.keys()) - known_fixed)
    if unknown:
        raise KeyError(
            "fixed_parameter_overrides contains names that are not fixed parameters: "
            + ", ".join(unknown)
        )
    for name, value in overrides.items():
        fixed_values[name] = float(value)
    return fixed_values


def parameter_name_index(parameters: Sequence[ParameterSpec]) -> dict[str, int]:
    """Return the name-to-index mapping for one parameter sequence."""

    return {parameter.name: index for index, parameter in enumerate(parameters)}

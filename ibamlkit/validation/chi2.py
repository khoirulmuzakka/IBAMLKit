"""Chi-square style validation metrics for batch spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .base import SimulationBatchResult, SpectrumSimulator
from .simulate import simulate_batch


@dataclass(frozen=True)
class Chi2BatchResult:
    """Batch chi-square evaluation result."""

    total: np.ndarray
    per_method: Mapping[str, np.ndarray]
    simulated: SimulationBatchResult | None = None


def _resolve_channel_lengths(
    observed: np.ndarray,
    simulated: np.ndarray,
    observed_lengths: np.ndarray | None,
    simulated_lengths: np.ndarray | None,
) -> np.ndarray:
    sample_count = observed.shape[0]
    lengths = np.full((sample_count,), min(observed.shape[1], simulated.shape[1]), dtype=np.int32)
    if observed_lengths is not None:
        lengths = np.minimum(lengths, np.asarray(observed_lengths, dtype=np.int32))
    if simulated_lengths is not None:
        lengths = np.minimum(lengths, np.asarray(simulated_lengths, dtype=np.int32))
    return lengths


def _chi2_per_sample(
    observed: np.ndarray,
    simulated: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    if observed.shape[0] != simulated.shape[0]:
        raise ValueError("Observed and simulated spectra must share the same sample count.")

    values = np.zeros((observed.shape[0],), dtype=np.float32)
    for index in range(observed.shape[0]):
        width = int(lengths[index])
        if width <= 0:
            values[index] = np.inf
            continue
        obs = np.asarray(observed[index, :width], dtype=np.float32)
        sim = np.asarray(simulated[index, :width], dtype=np.float32)
        values[index] = float(np.mean((sim - obs) ** 2 / (obs + 1.0), dtype=np.float32))
    return values


def calculate_chi2_batch(
    simulator: SpectrumSimulator,
    open_parameter_values: np.ndarray,
    observed_spectra: Mapping[str, np.ndarray],
    *,
    observed_lengths: Mapping[str, np.ndarray] | None = None,
    fixed_parameter_overrides: Mapping[str, float] | None = None,
    nthreads: int = 1,
    return_simulated: bool = False,
) -> Chi2BatchResult:
    """Simulate spectra for one batch and compare them to observed spectra."""

    simulated = simulate_batch(
        simulator,
        open_parameter_values,
        fixed_parameter_overrides=fixed_parameter_overrides,
        nthreads=nthreads,
    )

    per_method: dict[str, np.ndarray] = {}
    total: np.ndarray | None = None
    for method_name in simulator.method_names:
        if method_name not in observed_spectra:
            raise KeyError(f"Observed spectra are missing method {method_name!r}.")
        observed = np.asarray(observed_spectra[method_name], dtype=np.float32)
        predicted = np.asarray(simulated.spectra[method_name], dtype=np.float32)
        lengths = _resolve_channel_lengths(
            observed,
            predicted,
            None if observed_lengths is None else observed_lengths.get(method_name),
            None if simulated.spectra_lengths is None else simulated.spectra_lengths.get(method_name),
        )
        method_chi2 = _chi2_per_sample(observed, predicted, lengths)
        per_method[method_name] = method_chi2
        total = method_chi2.copy() if total is None else total + method_chi2

    if total is None:
        total = np.zeros((0,), dtype=np.float32)

    return Chi2BatchResult(
        total=total,
        per_method=per_method,
        simulated=simulated if return_simulated else None,
    )

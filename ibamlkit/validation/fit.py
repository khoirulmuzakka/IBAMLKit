"""Parameter fitting by chi-square minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ibamlkit.schema import DatasetInputSpec

from .base import SpectrumSimulator, validate_open_parameter_matrix
from .chi2 import calculate_chi2_batch

try:
    import minionpy as mpy
except Exception:  # pragma: no cover - optional dependency guard
    mpy = None


@dataclass(frozen=True)
class FitSampleResult:
    """Optimization result for one observed spectrum sample."""

    best_open_parameter_values: np.ndarray
    best_chi2: float
    success: bool
    message: str
    nit: int
    sample_index: int


@dataclass(frozen=True)
class FitBatchResult:
    """Optimization results for one observed batch."""

    samples: Sequence[FitSampleResult]

    @property
    def best_open_parameter_values(self) -> np.ndarray:
        if not self.samples:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray([sample.best_open_parameter_values for sample in self.samples], dtype=np.float32)

    @property
    def best_chi2(self) -> np.ndarray:
        return np.asarray([sample.best_chi2 for sample in self.samples], dtype=np.float32)


def _default_initial_guess(input_spec: DatasetInputSpec) -> np.ndarray:
    values = []
    for parameter in input_spec.open_parameters:
        if parameter.lower_bound is not None and parameter.upper_bound is not None:
            values.append(0.5 * (parameter.lower_bound + parameter.upper_bound))
        elif parameter.fixed_value is not None:
            values.append(float(parameter.fixed_value))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _bounds_from_input_spec(input_spec: DatasetInputSpec) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for parameter in input_spec.open_parameters:
        bounds.append((parameter.lower_bound, parameter.upper_bound))
    return bounds


def _default_minion_options(algo: str) -> dict[str, Any]:
    normalized = str(algo).strip().upper()
    if normalized == "L_BFGS_B":
        return {
            "func_noise_ratio": 1e-8,
            "N_points_derivative": 3,
        }
    return {}


def fit_open_parameters(
    simulator: SpectrumSimulator,
    input_spec: DatasetInputSpec,
    observed_spectra: Mapping[str, np.ndarray],
    *,
    observed_lengths: Mapping[str, np.ndarray] | None = None,
    initial_open_parameter_values: np.ndarray | None = None,
    fixed_parameter_overrides: Mapping[str, float] | None = None,
    nthreads: int = 1,
    algo: str = "ARRDE",
    maxevals: int = 10000,
    rel_tol: float = 1e-4,
    seed: int | None = None,
    optimizer_options: Mapping[str, Any] | None = None,
    close_simulator: bool = True,
) -> FitBatchResult:
    """Fit one batch of open parameters by minimizing chi-square."""

    if mpy is None:
        raise RuntimeError("minionpy is required for fit_open_parameters but is not available.")

    sample_count = int(np.asarray(next(iter(observed_spectra.values())), dtype=np.float32).shape[0])
    user_provided_initial = initial_open_parameter_values is not None
    if initial_open_parameter_values is None:
        initial = np.repeat(_default_initial_guess(input_spec)[None, :], sample_count, axis=0)
    else:
        initial = validate_open_parameter_matrix(input_spec, initial_open_parameter_values)
        if initial.shape[0] != sample_count:
            raise ValueError("initial_open_parameter_values must match the observed sample count.")

    bounds = _bounds_from_input_spec(input_spec)
    results: list[FitSampleResult] = []

    try:
        for sample_index in range(sample_count):
            sample_observed_base = {
                method_name: np.asarray(values[sample_index : sample_index + 1], dtype=np.float32)
                for method_name, values in observed_spectra.items()
            }
            sample_lengths_base = None
            if observed_lengths is not None:
                sample_lengths_base = {
                    method_name: np.asarray(values[sample_index : sample_index + 1], dtype=np.int32)
                    for method_name, values in observed_lengths.items()
                }

            def objective(x: np.ndarray, data: object | None = None) -> list[float]:
                del data
                candidate_batch = np.asarray(x, dtype=np.float32)
                if candidate_batch.ndim == 1:
                    candidate_batch = candidate_batch.reshape(1, -1)
                repeated_observed = {
                    method_name: np.repeat(values, candidate_batch.shape[0], axis=0)
                    for method_name, values in sample_observed_base.items()
                }
                repeated_lengths = None
                if sample_lengths_base is not None:
                    repeated_lengths = {
                        method_name: np.repeat(values, candidate_batch.shape[0], axis=0)
                        for method_name, values in sample_lengths_base.items()
                    }
                chi2 = calculate_chi2_batch(
                    simulator,
                    candidate_batch,
                    repeated_observed,
                    observed_lengths=repeated_lengths,
                    fixed_parameter_overrides=fixed_parameter_overrides,
                    nthreads=nthreads,
                    return_simulated=False,
                )
                return np.asarray(chi2.total, dtype=np.float64).tolist()

            x0 = None
            if user_provided_initial:
                x0 = [np.asarray(initial[sample_index], dtype=np.float64).tolist()]
            normalized_algo = str(algo).strip().replace("-", "_")
            options = _default_minion_options(normalized_algo)
            if optimizer_options is not None:
                options.update(dict(optimizer_options))

            outcome = mpy.Minimizer(
                func=objective,
                bounds=bounds,
                x0=x0,
                algo=normalized_algo,
                relTol=float(rel_tol),
                maxevals=int(maxevals),
                seed=seed,
                options=options or None,
            ).optimize()
            success = bool(outcome.success)
            if not success:
                success = bool(
                    np.isfinite(float(outcome.fun))
                    and int(getattr(outcome, "nfev", maxevals)) < int(maxevals)
                )
            results.append(
                FitSampleResult(
                    best_open_parameter_values=np.asarray(outcome.x, dtype=np.float32),
                    best_chi2=float(outcome.fun),
                    success=success,
                    message=str(outcome.message),
                    nit=int(getattr(outcome, "nit", 0)),
                    sample_index=sample_index,
                )
            )
    finally:
        if close_simulator:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()

    return FitBatchResult(samples=results)

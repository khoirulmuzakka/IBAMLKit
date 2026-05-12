from __future__ import annotations

import os
import sys
from typing import Mapping, Sequence

import numpy as np

from ibamlkit.schema import DatasetInputSpec, ParameterSpec

current_file_directory = os.path.dirname(os.path.abspath(__file__))
custom_path = os.path.join(current_file_directory, "lib")
sys.path.append(custom_path)

# Configure OpenMP threads to use all available CPUs, unless user overrides.
if "OMP_NUM_THREADS" not in os.environ:
    try:
        import multiprocessing as _mp

        _n = _mp.cpu_count()
    except Exception:
        _n = os.cpu_count() or 1

    n_threads = min(_n, 32)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    del _n
    try:
        del _mp
    except Exception:
        pass

_PILEUPCPP_IMPORT_ERROR: ImportError | None = None
try:
    from pileupcpp import convert_to_channel_space_and_pileup_batch as convert_to_channel_space_and_pileup_batchpp
    from pileupcpp import fast_pileup_batch as fast_pileup_batchpp
    from pileupcpp import rebin_histogram as rebinpp
except ImportError as exc:
    _PILEUPCPP_IMPORT_ERROR = exc
    convert_to_channel_space_and_pileup_batchpp = None
    fast_pileup_batchpp = None
    rebinpp = None


def _require_pileupcpp() -> None:
    if _PILEUPCPP_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "The pileup C++ extension could not be imported. "
        "Python-side calibration helpers remain available, but numerical pileup "
        "operations require a working 'pileupcpp' build."
    ) from _PILEUPCPP_IMPORT_ERROR


def normalize_kind(kind: str) -> str:
    return str(kind).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_name(name: str) -> str:
    return normalize_kind(name)


def _name_matches_any(name: str, patterns: Sequence[str]) -> bool:
    normalized_name = _normalize_name(name)
    return any(pattern in normalized_name for pattern in patterns)


def _iter_matching_setup_parameters(
    input_spec: DatasetInputSpec,
    *,
    method_name: str,
    kind: str | None = None,
    name_patterns: Sequence[str] = (),
    is_open: bool | None = None,
) -> list[ParameterSpec]:
    normalized_kind = None if kind is None else normalize_kind(kind)
    matches: list[ParameterSpec] = []
    for parameter in input_spec.parameters:
        if parameter.group != "setup":
            continue
        if parameter.method and parameter.method != method_name:
            continue
        if is_open is not None and parameter.is_open != is_open:
            continue
        if normalized_kind is not None and normalize_kind(parameter.kind) == normalized_kind:
            matches.append(parameter)
            continue
        if name_patterns and _name_matches_any(parameter.name, name_patterns):
            matches.append(parameter)
    return matches


def get_setup_parameter_spec(
    input_spec: DatasetInputSpec,
    *,
    kind: str | None,
    method_name: str,
    name_patterns: Sequence[str] = (),
    is_open: bool | None = None,
) -> ParameterSpec | None:
    matches = _iter_matching_setup_parameters(
        input_spec,
        method_name=method_name,
        kind=kind,
        name_patterns=name_patterns,
        is_open=is_open,
    )
    if not matches:
        return None

    if is_open is None:
        open_matches = [parameter for parameter in matches if parameter.is_open]
        if open_matches:
            return open_matches[0]
        fixed_matches = [parameter for parameter in matches if not parameter.is_open]
        if fixed_matches:
            return fixed_matches[0]
    return matches[0]


def get_setup_parameter_vector(
    input_spec: DatasetInputSpec,
    open_parameter_names: Sequence[str],
    open_parameter_values: np.ndarray,
    *,
    kind: str | None,
    method_name: str,
    name_patterns: Sequence[str] = (),
    default: float | None = None,
) -> np.ndarray:
    parameter = get_setup_parameter_spec(
        input_spec,
        kind=kind,
        method_name=method_name,
        name_patterns=name_patterns,
    )
    sample_count = int(np.asarray(open_parameter_values).shape[0])
    if parameter is None:
        if default is None:
            raise KeyError(
                f"Could not resolve setup parameter for method={method_name!r}, "
                f"kind={kind!r}, patterns={tuple(name_patterns)!r}"
            )
        return np.full((sample_count,), float(default), dtype=np.float64)

    if parameter.is_open:
        name_to_index = {str(name): index for index, name in enumerate(open_parameter_names)}
        return np.asarray(open_parameter_values[:, name_to_index[parameter.name]], dtype=np.float64)
    return np.full((sample_count,), float(parameter.fixed_value), dtype=np.float64)


def resolve_channel_conversion_arrays(
    input_spec: DatasetInputSpec,
    open_parameter_names: Sequence[str],
    open_parameter_values: np.ndarray,
    *,
    method_name: str,
    energy_spectrum_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_parameter_values = np.asarray(open_parameter_values, dtype=np.float32)
    calibration_offset = get_setup_parameter_vector(
        input_spec,
        open_parameter_names,
        open_parameter_values,
        kind="calibration_offset",
        method_name=method_name,
        name_patterns=("calib_offset", "calibration_offset"),
        default=0.0,
    )
    calibration_linear = get_setup_parameter_vector(
        input_spec,
        open_parameter_names,
        open_parameter_values,
        kind="calibration_linear",
        method_name=method_name,
        name_patterns=("calib_linear", "calibration_linear"),
        default=1.0,
    )
    calibration_quadratic = get_setup_parameter_vector(
        input_spec,
        open_parameter_names,
        open_parameter_values,
        kind="calibration_quadratic",
        method_name=method_name,
        name_patterns=("calib_quadratic", "calibration_quadratic"),
        default=0.0,
    )

    particles_open = get_setup_parameter_spec(
        input_spec,
        kind="particles_sr",
        method_name=method_name,
        name_patterns=("particles_sr", "particlesr"),
        is_open=True,
    )
    if particles_open is not None:
        particles_open_values = get_setup_parameter_vector(
            input_spec,
            open_parameter_names,
            open_parameter_values,
            kind="particles_sr",
            method_name=method_name,
            name_patterns=("particles_sr", "particlesr"),
        )
        particles_fixed = get_setup_parameter_spec(
            input_spec,
            kind="particles_sr",
            method_name=method_name,
            name_patterns=("particles_sr", "particlesr"),
            is_open=False,
        )
        base_particles = (
            float(particles_fixed.fixed_value)
            if particles_fixed is not None and particles_fixed.fixed_value is not None
            else 0.0
        )
        if base_particles != 0.0:
            normalization_factors = particles_open_values / (float(energy_spectrum_scale) * base_particles)
        else:
            normalization_factors = np.ones((open_parameter_values.shape[0],), dtype=np.float64)
    else:
        normalization_factors = np.ones((open_parameter_values.shape[0],), dtype=np.float64)

    return (
        np.asarray(calibration_offset, dtype=np.float64),
        np.asarray(calibration_linear, dtype=np.float64),
        np.asarray(calibration_quadratic, dtype=np.float64),
        np.asarray(normalization_factors, dtype=np.float64),
    )


def rebin_histogram(bin_edges_old, bin_edges_new, spectrum):
    _require_pileupcpp()
    return rebinpp(bin_edges_old, bin_edges_new, spectrum)


def rebin_spectra_to_energy_space(
    channel_space_spectra: np.ndarray,
    channel_lengths: np.ndarray | None,
    *,
    calibration_offset,
    calibration_linear,
    calibration_quadratic,
    energy_bin_width: float = 1.0,
    target_width: int | None = None,
    energy_spectrum_scale: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_space_spectra = np.asarray(channel_space_spectra, dtype=np.float32)
    sample_count = int(channel_space_spectra.shape[0])
    if channel_lengths is None:
        channel_lengths = np.full((sample_count,), channel_space_spectra.shape[1], dtype=np.int32)
    else:
        channel_lengths = np.asarray(channel_lengths, dtype=np.int32)

    calibration_offset = np.asarray(calibration_offset, dtype=np.float64)
    calibration_linear = np.asarray(calibration_linear, dtype=np.float64)
    calibration_quadratic = np.asarray(calibration_quadratic, dtype=np.float64)

    max_energy = 0.0
    for index in range(sample_count):
        width = int(channel_lengths[index])
        channel_edges = np.arange(width + 1, dtype=np.float64)
        energy_edges_old = (
            calibration_offset[index]
            + calibration_linear[index] * channel_edges
            + calibration_quadratic[index] * (channel_edges**2)
        )
        max_energy = max(max_energy, float(np.max(energy_edges_old)))

    energy_edges = np.arange(
        0.0,
        np.ceil(max_energy) + float(energy_bin_width),
        float(energy_bin_width),
        dtype=np.float64,
    )
    rebinned: list[np.ndarray] = []
    rebinned_lengths = np.zeros((sample_count,), dtype=np.int32)
    output_width = None

    for index in range(sample_count):
        width = int(channel_lengths[index])
        channel_edges = np.arange(width + 1, dtype=np.float64)
        energy_edges_old = (
            calibration_offset[index]
            + calibration_linear[index] * channel_edges
            + calibration_quadratic[index] * (channel_edges**2)
        )
        rebinned_spectrum = np.asarray(
            rebin_histogram(energy_edges_old, energy_edges, channel_space_spectra[index, :width]),
            dtype=np.float32,
        ) * float(energy_spectrum_scale)
        if output_width is None:
            output_width = (
                rebinned_spectrum.shape[0]
                if target_width is None
                else min(int(target_width), rebinned_spectrum.shape[0])
            )
        current = rebinned_spectrum[:output_width]
        rebinned_lengths[index] = min(rebinned_spectrum.shape[0], output_width)
        if current.shape[0] < output_width:
            current = np.pad(current, (0, output_width - current.shape[0]), mode="constant")
        rebinned.append(np.asarray(current, dtype=np.float32))

    return np.stack(rebinned, axis=0), rebinned_lengths, energy_edges


def fast_pileup_batch(spectra_list, real_times, live_times, fudge_factors, clip_negative=True):
    _require_pileupcpp()
    return fast_pileup_batchpp(spectra_list, real_times, live_times, fudge_factors, clip_negative)


def convert_to_channel_space_and_pileup_batch(
    a,
    b,
    c,
    real_times,
    live_times,
    fudge_factors,
    r=1.0,
    E_space_spectra=None,
    clip_negative=True,
):
    """Convert E-space spectra to channel space using quadratic calibration and apply pileup."""
    _require_pileupcpp()
    if E_space_spectra is None:
        r_maybe_spec = np.asarray(r)
        if r_maybe_spec.ndim == 2:
            E_space_spectra = r_maybe_spec
            r = 1.0
        else:
            raise ValueError("E_space_spectra must be provided")

    E_space_spectra = np.asarray(E_space_spectra)
    if E_space_spectra.ndim != 2:
        raise ValueError("E_space_spectra must be 2D (n_spectra, K)")
    n_spec = E_space_spectra.shape[0]

    def _as_array(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return np.full((n_spec,), float(x), dtype=np.float64)
        if x.shape[0] != n_spec:
            raise ValueError("Length mismatch: inputs must be scalar or length n_spectra")
        return x.astype(np.float64, copy=False)

    a = _as_array(a)
    b = _as_array(b)
    c = _as_array(c)
    real_times = _as_array(real_times)
    live_times = _as_array(live_times)
    fudge_factors = _as_array(fudge_factors)
    r = _as_array(r)

    return convert_to_channel_space_and_pileup_batchpp(
        a,
        b,
        c,
        real_times,
        live_times,
        fudge_factors,
        r,
        E_space_spectra,
        clip_negative,
    )


def convert_energy_spectra_to_channel_space_and_pileup(
    energy_space_spectra: np.ndarray,
    *,
    calibration_offset,
    calibration_linear,
    calibration_quadratic,
    real_times,
    live_times,
    fudge_factors,
    normalization_factors=1.0,
    clip_negative: bool = True,
) -> np.ndarray:
    return np.asarray(
        convert_to_channel_space_and_pileup_batch(
            calibration_offset,
            calibration_linear,
            calibration_quadratic,
            real_times,
            live_times,
            fudge_factors,
            normalization_factors,
            energy_space_spectra,
            clip_negative,
        ),
        dtype=np.float32,
    )


def apply_channel_space_pileup(
    channel_space_spectra: np.ndarray,
    *,
    real_times,
    live_times,
    fudge_factors,
    clip_negative: bool = True,
) -> np.ndarray:
    return np.asarray(
        fast_pileup_batch(
            channel_space_spectra,
            real_times,
            live_times,
            fudge_factors,
            clip_negative,
        ),
        dtype=np.float32,
    )

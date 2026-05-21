"""SIMNRA-backed dataset generation for the canonical IBAMLKit schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback
from typing import Callable, Mapping, Sequence
import uuid

import numpy as np

from ...data import save_dataset
from ...schema import DatasetInputSpec, IBADataset, ParameterSpec
from ..base import SpectrumGenerator
from .SIMNRA import SIMNRA


def _normalize_kind(kind: str) -> str:
    return kind.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class _LayerBlueprint:
    layer_index: int
    elements: tuple[str, ...]


@dataclass(frozen=True)
class GenerationProgress:
    """Progress update emitted during generation."""

    phase: str
    completed: int
    total: int
    elapsed_seconds: float
    chunk_index: int | None = None
    chunk_total: int | None = None
    message: str = ""


@dataclass(frozen=True)
class GenerationFailure:
    """Structured error emitted for one failed sample."""

    sample_index: int
    open_parameter_values: np.ndarray
    error_type: str
    message: str
    traceback_text: str
    chunk_index: int | None = None


@dataclass(frozen=True)
class _ValidationFailure:
    method_name: str
    reason: str
    primary_length: int
    validation_length: int
    chi2: float | None = None


class _SIMNRAMethodSession:
    """One thread-local SIMNRA session for a single method."""

    def __init__(
        self,
        input_spec: DatasetInputSpec,
        method_name: str,
        log_concentration_corrections: bool = False,
        concentration_correction_threshold: float = 1e-3,
    ):
        self.input_spec = input_spec
        self.method_name = method_name
        self.log_concentration_corrections = bool(log_concentration_corrections)
        self.concentration_correction_threshold = max(0.0, float(concentration_correction_threshold))
        self.method_spec = next(method for method in input_spec.methods if method.name == method_name)
        self._temp_reference_path = self._copy_reference_file()
        self.simnra = SIMNRA()
        self._configure_mode()
        self._open_reference()
        self._layer_blueprints = self._build_layer_blueprints()
        self._concentration_params = self._build_concentration_index()
        self._roughness_layers = self._collect_layers_by_kind("roughness")
        self._porosity_layers = self._collect_layers_by_kind("porosity_fraction", "porosity_diameter")
        self._initialize_target_structure()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        simnra = getattr(self, "simnra", None)
        if simnra is not None:
            try:
                del self.simnra
            except Exception:
                pass
            self.simnra = None
        try:
            path = getattr(self, "_temp_reference_path", None)
            if path:
                Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        self._temp_reference_path = None

    def _copy_reference_file(self) -> str:
        reference_file = self.method_spec.reference_file
        if not reference_file:
            raise ValueError(f"Method '{self.method_name}' does not define a reference_file.")
        source = Path(reference_file)
        if not source.exists():
            raise FileNotFoundError(f"Reference file not found for method '{self.method_name}': {source}")

        temp_dir = Path(tempfile.gettempdir()) / "ibamlkit_simnra"
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix or ".xnra"
        temp_path = temp_dir / f"{self.method_name}_{uuid.uuid4().hex}{suffix}"
        shutil.copy2(source, temp_path)
        return str(temp_path)

    def _configure_mode(self) -> None:
        simulation_mode = str(self.method_spec.metadata.get("simulation_mode", "")).strip().lower()
        self.simnra.App.SimulationMode = 1 if simulation_mode == "pige" or "PIGE" in self.method_name.upper() else 0

    def _open_reference(self) -> None:
        if not self.simnra.App.Open(self._temp_reference_path, -1):
            message = self._last_message()
            raise RuntimeError(
                f"SIMNRA failed to open reference file for method '{self.method_name}': {message}"
            )

        self.simnra.Calc.ElementSpectra = True
        self.simnra.Fit.Chi2Evaluation = 0

    def _last_message(self) -> str:
        message = getattr(self.simnra.App, "LastMessage", "")
        if message is None:
            raise RuntimeError("SIMNRA did not provide an error message.")
        return str(message).strip()

    def _emit_warning(self, message: str) -> None:
        if not self.log_concentration_corrections:
            return
        print(f"[warning] {message}")

    def _build_layer_blueprints(self) -> list[_LayerBlueprint]:
        by_layer: dict[int, list[str]] = {}
        for entry in self.input_spec.layer_species:
            by_layer.setdefault(entry.layer_index, []).append(entry.element)
        return [
            _LayerBlueprint(layer_index=index, elements=tuple(elements))
            for index, elements in sorted(by_layer.items())
        ]

    def _build_concentration_index(self) -> dict[int, list[ParameterSpec]]:
        ret: dict[int, list[ParameterSpec]] = {}
        for parameter in self.input_spec.parameters:
            if parameter.group == "layer" and _normalize_kind(parameter.kind) == "concentration":
                ret.setdefault(parameter.layer_index, []).append(parameter)
        return ret

    def _collect_layers_by_kind(self, *kinds: str) -> set[int]:
        wanted = {_normalize_kind(kind) for kind in kinds}
        return {
            parameter.layer_index
            for parameter in self.input_spec.parameters
            if parameter.group == "layer" and _normalize_kind(parameter.kind) in wanted
        }

    def _parameter_value(self, parameter: ParameterSpec, full_parameter_values: Mapping[str, float]) -> float:
        return float(full_parameter_values[parameter.name])

    def _default_full_parameter_values(self) -> dict[str, float]:
        defaults: dict[str, float] = {}
        for parameter in self.input_spec.fixed_parameters:
            defaults[parameter.name] = float(parameter.fixed_value)
        for parameter in self.input_spec.open_parameters:
            if parameter.lower_bound is not None and parameter.upper_bound is not None:
                defaults[parameter.name] = 0.5 * (parameter.lower_bound + parameter.upper_bound)
            else:
                defaults[parameter.name] = 0.0
        return defaults

    def _initial_layer_concentration(self, layer_index: int, element: str) -> float:
        layer_parameters = self._concentration_params.get(layer_index, [])
        matches = [
            parameter
            for parameter in layer_parameters
            if parameter.element == element
        ]
        if not matches:
            return 0.0

        parameter = matches[0]
        if parameter.fixed_value is not None:
            return float(parameter.fixed_value)
        if parameter.lower_bound is not None and parameter.upper_bound is not None:
            return 0.5 * (parameter.lower_bound + parameter.upper_bound)
        return 0.0

    def _initial_layer_concentrations(self, layer_index: int, elements: Sequence[str]) -> list[float]:
        layer_parameters = self._concentration_params.get(layer_index, [])
        by_element = {
            parameter.element: parameter
            for parameter in layer_parameters
        }

        concentrations = [0.0] * len(elements)
        open_indexes: list[int] = []
        fixed_total = 0.0

        for index, element in enumerate(elements):
            parameter = by_element.get(element)
            if parameter is None:
                open_indexes.append(index)
                continue
            if parameter.fixed_value is not None:
                value = float(parameter.fixed_value)
                concentrations[index] = value
                fixed_total += value
            else:
                open_indexes.append(index)

        remaining = max(0.0, 1.0 - fixed_total)
        if open_indexes:
            fill_value = remaining / len(open_indexes)
            for index in open_indexes:
                concentrations[index] = fill_value

        return self._normalize_layer_concentrations(
            concentrations,
            layer_index,
            self.method_name,
        )

    def _initial_thickness(self, layer_index: int) -> float:
        matches = [
            parameter
            for parameter in self.input_spec.parameters
            if parameter.group == "layer"
            and parameter.layer_index == layer_index
            and _normalize_kind(parameter.kind) == "thickness"
        ]
        if not matches:
            return 1.0
        parameter = matches[0]
        if parameter.fixed_value is not None:
            return float(parameter.fixed_value)
        if parameter.lower_bound is not None and parameter.upper_bound is not None:
            return 0.5 * (parameter.lower_bound + parameter.upper_bound)
        return 1.0

    def _normalize_layer_concentrations(
        self,
        concentrations: Sequence[float],
        layer_index: int,
        method_name: str | None = None,
    ) -> list[float]:
        original = np.asarray(concentrations, dtype=np.float64)
        normalized = original.copy()
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError(f"Layer {layer_index} must define at least one concentration.")

        normalized = np.clip(normalized, 0.0, None)
        concentration_sum = float(normalized.sum(dtype=np.float64))
        if concentration_sum <= 0.0:
            normalized.fill(1.0 / normalized.size)
        else:
            normalized /= concentration_sum

        if normalized.size > 1:
            normalized[:-1] = np.clip(normalized[:-1], 0.0, 1.0)
            normalized[-1] = max(0.0, 1.0 - float(normalized[:-1].sum(dtype=np.float64)))
            tail_sum = float(normalized.sum(dtype=np.float64))
            if tail_sum <= 0.0:
                normalized.fill(1.0 / normalized.size)
            elif abs(tail_sum - 1.0) > 1e-12:
                normalized /= tail_sum
                normalized[-1] = max(0.0, 1.0 - float(normalized[:-1].sum(dtype=np.float64)))

        max_delta = float(np.max(np.abs(normalized - original)))
        sum_delta = abs(float(original.sum(dtype=np.float64)) - 1.0)
        if max(max_delta, sum_delta) >= self.concentration_correction_threshold:
            method_label = f" for method '{method_name}'" if method_name else ""
            self._emit_warning(
                "Normalized concentrations"
                f"{method_label} in layer {layer_index}: "
                f"sum {float(original.sum(dtype=np.float64)):.12g} -> 1.0, "
                f"max delta {max_delta:.6g}"
            )

        return normalized.tolist()

    def _initialize_target_structure(self) -> None:
        reference_layer_count = int(self.simnra.Target.NumberOfLayers)

        for blueprint in self._layer_blueprints:
            simnra_layer_index = reference_layer_count + blueprint.layer_index
            self.simnra.Target.AddLayer()
            self.simnra.Target.AddElements(simnra_layer_index, len(blueprint.elements))
            self.simnra.Target.SetLayerThickness(
                simnra_layer_index,
                self._initial_thickness(blueprint.layer_index),
            )

            for element_index, element in enumerate(blueprint.elements, start=1):
                self.simnra.Target.SetElementName(simnra_layer_index, element_index, element)
            concentrations = self._initial_layer_concentrations(
                blueprint.layer_index,
                blueprint.elements,
            )

            for element_index, concentration in enumerate(concentrations, start=1):
                self.simnra.Target.SetElementConcentration(
                    simnra_layer_index,
                    element_index,
                    concentration,
                )

        for _ in range(reference_layer_count):
            self.simnra.Target.DeleteLayer(1)

        for blueprint in self._layer_blueprints:
            if blueprint.layer_index in self._roughness_layers:
                self.simnra.Target.SetHasLayerRoughness(blueprint.layer_index, True)
            if blueprint.layer_index in self._porosity_layers:
                self.simnra.Target.SetHasLayerPorosity(blueprint.layer_index, True)

    def _apply_setup_parameters(self, full_parameter_values: Mapping[str, float]) -> None:
        for parameter in self.input_spec.parameters:
            if parameter.group != "setup":
                continue
            if parameter.method and parameter.method != self.method_name:
                continue

            value = self._parameter_value(parameter, full_parameter_values)
            kind = _normalize_kind(parameter.kind)
            if kind in {"particles_sr", "particle_sr"}:
                self.simnra.Setup.ParticlesSr = value
            elif kind in {"calibration_offset", "calib_offset"}:
                self.simnra.Setup.CalibrationOffset = value
            elif kind in {"calibration_linear", "calib_linear"}:
                self.simnra.Setup.CalibrationLinear = value
            elif kind in {"calibration_quadratic", "calib_quadratic"}:
                self.simnra.Setup.CalibrationQuadratic = value
            elif kind in {"fwhm", "detector_resolution"}:
                self.simnra.Setup.DetectorResolution = value
            elif kind == "beam_energy":
                self.simnra.Setup.Energy = value
            elif kind == "beam_spread":
                self.simnra.Setup.Beamspread = value

    def _apply_layer_parameters(self, full_parameter_values: Mapping[str, float]) -> None:
        layer_concentrations: dict[int, list[float]] = {}
        layer_blueprint_map = {blueprint.layer_index: blueprint for blueprint in self._layer_blueprints}
        species_index_map: dict[int, dict[tuple[str, str], int]] = {}
        for blueprint in self._layer_blueprints:
            species_index_map[blueprint.layer_index] = {
                element: index for index, element in enumerate(blueprint.elements)
            }
            layer_concentrations[blueprint.layer_index] = [0.0] * len(blueprint.elements)

        for parameter in self.input_spec.parameters:
            if parameter.group != "layer":
                continue
            if parameter.layer_index not in layer_blueprint_map:
                continue

            value = self._parameter_value(parameter, full_parameter_values)
            kind = _normalize_kind(parameter.kind)
            if kind == "thickness":
                self.simnra.Target.SetLayerThickness(parameter.layer_index, value)
            elif kind == "concentration":
                index = species_index_map[parameter.layer_index][parameter.element]
                layer_concentrations[parameter.layer_index][index] = value
            elif kind == "roughness":
                self.simnra.Target.SetLayerRoughness(parameter.layer_index, value)
            elif kind == "porosity_fraction":
                self.simnra.Target.SetPorosityFraction(parameter.layer_index, value)
            elif kind == "porosity_diameter":
                self.simnra.Target.SetPoreDiameter(parameter.layer_index, value)

        for layer_index, concentrations in layer_concentrations.items():
            concentrations = self._normalize_layer_concentrations(
                concentrations,
                layer_index,
                self.method_name,
            )
            for element_index, concentration in enumerate(concentrations, start=1):
                self.simnra.Target.SetElementConcentration(layer_index, element_index, concentration)

    def generate_spectrum(self, full_parameter_values: Mapping[str, float]) -> np.ndarray:
        self._apply_setup_parameters(full_parameter_values)
        self._apply_layer_parameters(full_parameter_values)
        result = self.simnra.App.CalculateSpectrum()
        if not result:
            message = self._last_message()
            if not message:
                message = "Unknown SIMNRA error."
            raise RuntimeError(
                f"SIMNRA CalculateSpectrum failed for method '{self.method_name}': {message}"
            )
        return np.asarray(self.simnra.Spectrum.GetDataArray(2), dtype=np.float32)


class _ThreadSessionPool:
    """Thread-local container of SIMNRA sessions, one per method."""

    def __init__(
        self,
        input_spec: DatasetInputSpec,
        log_concentration_corrections: bool = False,
        concentration_correction_threshold: float = 1e-3,
    ):
        self.input_spec = input_spec
        self.log_concentration_corrections = bool(log_concentration_corrections)
        self.concentration_correction_threshold = max(0.0, float(concentration_correction_threshold))
        self._local = threading.local()
        self._created_sessions: list[dict[str, _SIMNRAMethodSession]] = []
        self._lock = threading.Lock()

    def get_sessions(self) -> dict[str, _SIMNRAMethodSession]:
        sessions = getattr(self._local, "sessions", None)
        if sessions is None:
            sessions = self._create_sessions()
            self._local.sessions = sessions
        return sessions

    def _create_sessions(self) -> dict[str, _SIMNRAMethodSession]:
        sessions = {
            method.name: _SIMNRAMethodSession(
                self.input_spec,
                method.name,
                log_concentration_corrections=self.log_concentration_corrections,
                concentration_correction_threshold=self.concentration_correction_threshold,
            )
            for method in self.input_spec.methods
        }
        with self._lock:
            self._created_sessions.append(sessions)
        return sessions

    def reset_current_thread_sessions(self) -> dict[str, _SIMNRAMethodSession]:
        sessions = getattr(self._local, "sessions", None)
        if sessions is not None:
            for session in sessions.values():
                session.close()
        sessions = self._create_sessions()
        self._local.sessions = sessions
        return sessions

    def close_all(self) -> None:
        with self._lock:
            created_sessions = self._created_sessions
            self._created_sessions = []

        for session_map in created_sessions:
            for session in session_map.values():
                session.close()


class SIMNRASpectrumGenerator(SpectrumGenerator):
    """Generate rectangular IBA datasets using SIMNRA OLE in worker threads."""

    def __init__(
        self,
        input_spec: DatasetInputSpec,
        max_workers: int = 1,
        keep_alive: bool = False,
        progress_callback: Callable[[GenerationProgress], None] | None = None,
        error_callback: Callable[[GenerationFailure], None] | None = None,
        progress_every: int = 10,
        print_progress: bool = True,
        simnra_retry_limit: int = 2,
        allow_failed_samples: bool = True,
        validation_enabled: bool = True,
        validation_methods: Sequence[str] | None = None,
        validation_chi2_threshold: float = 10.0,
        validation_max_attempts: int = 3,
        validation_require_matching_length: bool = True,
        log_concentration_corrections: bool = False,
        concentration_correction_threshold: float = 1e-3,
    ):
        self.input_spec = input_spec
        self.max_workers = max(1, int(max_workers))
        self.keep_alive = bool(keep_alive)
        self._open_parameters = list(input_spec.open_parameters)
        self._fixed_parameter_values = {
            parameter.name: float(parameter.fixed_value)
            for parameter in input_spec.fixed_parameters
        }
        self.progress_callback = progress_callback
        self.error_callback = error_callback
        self.progress_every = max(1, int(progress_every))
        self.print_progress = print_progress
        self.simnra_retry_limit = max(0, int(simnra_retry_limit))
        self.allow_failed_samples = bool(allow_failed_samples)
        self.validation_enabled = bool(validation_enabled)
        available_method_names = {method.name for method in input_spec.methods}
        if validation_methods is None:
            self.validation_methods = tuple(method.name for method in input_spec.methods)
        else:
            resolved_validation_methods = tuple(str(name) for name in validation_methods)
            unknown_method_names = sorted(
                method_name
                for method_name in resolved_validation_methods
                if method_name not in available_method_names
            )
            if unknown_method_names:
                raise ValueError(
                    f"validation_methods contains unknown method names: {unknown_method_names}"
                )
            self.validation_methods = resolved_validation_methods
        self.validation_chi2_threshold = max(0.0, float(validation_chi2_threshold))
        self.validation_max_attempts = max(1, int(validation_max_attempts))
        self.validation_require_matching_length = bool(validation_require_matching_length)
        self.log_concentration_corrections = bool(log_concentration_corrections)
        self.concentration_correction_threshold = max(0.0, float(concentration_correction_threshold))
        self._warning_lock = threading.Lock()
        self._session_pool = _ThreadSessionPool(
            input_spec,
            log_concentration_corrections=self.log_concentration_corrections,
            concentration_correction_threshold=self.concentration_correction_threshold,
        )
        self._persistent_executor: ThreadPoolExecutor | None = None
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        executor = self._persistent_executor
        self._persistent_executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        self._session_pool.close_all()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _get_executor(self) -> tuple[ThreadPoolExecutor, bool]:
        if not self.keep_alive:
            return ThreadPoolExecutor(max_workers=self.max_workers), True
        if self._persistent_executor is None:
            self._persistent_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._persistent_executor, False

    def generate(
        self,
        open_parameter_values: np.ndarray,
        sample_ids: Sequence[str] | None = None,
    ) -> IBADataset:
        started_at = time.time()
        if self._closed:
            raise RuntimeError("SIMNRASpectrumGenerator has been closed.")
        executor, owns_executor = self._get_executor()
        try:
            open_values = self._validate_open_parameter_values(open_parameter_values)
            resolved_sample_ids = self._resolve_sample_ids(open_values.shape[0], sample_ids)
            self._emit_progress(
                phase="generate",
                completed=0,
                total=open_values.shape[0],
                started_at=started_at,
                message="Starting in-memory generation.",
            )
            accepted_open_values, accepted_sample_ids, spectra, spectra_lengths = self._generate_spectra_matrices(
                open_values,
                sample_ids=resolved_sample_ids,
                executor=executor,
                started_at=started_at,
            )
            return IBADataset(
                input_spec=self.input_spec,
                open_parameter_values=accepted_open_values,
                spectra=spectra,
                spectra_lengths=spectra_lengths,
                sample_ids=accepted_sample_ids,
            )
        finally:
            if owns_executor:
                executor.shutdown(wait=True)
            if not self.keep_alive:
                self._session_pool.close_all()

    def generate_to_files(
        self,
        open_parameter_values: np.ndarray,
        output_dir: str,
        base_name: str,
        chunk_size: int,
        sample_ids: Sequence[str] | None = None,
    ) -> list[str]:
        started_at = time.time()
        if self._closed:
            raise RuntimeError("SIMNRASpectrumGenerator has been closed.")
        executor, owns_executor = self._get_executor()
        try:
            open_values = self._validate_open_parameter_values(open_parameter_values)
            resolved_sample_ids = self._resolve_sample_ids(open_values.shape[0], sample_ids)
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer.")

            target_dir = Path(output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            file_paths: list[str] = []
            chunk_total = (open_values.shape[0] + chunk_size - 1) // chunk_size
            self._emit_progress(
                phase="generate_to_files",
                completed=0,
                total=open_values.shape[0],
                started_at=started_at,
                chunk_index=0,
                chunk_total=chunk_total,
                message="Starting chunked generation.",
            )

            for chunk_index, start in enumerate(range(0, open_values.shape[0], chunk_size), start=1):
                end = min(start + chunk_size, open_values.shape[0])
                chunk_open_values = open_values[start:end]
                chunk_sample_ids = resolved_sample_ids[start:end]
                self._emit_progress(
                    phase="chunk",
                    completed=start,
                    total=open_values.shape[0],
                    started_at=started_at,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    message=f"Generating chunk {chunk_index}/{chunk_total}.",
                )
                (
                    accepted_chunk_open_values,
                    accepted_chunk_sample_ids,
                    chunk_spectra,
                    chunk_spectra_lengths,
                ) = self._generate_spectra_matrices(
                    chunk_open_values,
                    sample_ids=chunk_sample_ids,
                    executor=executor,
                    started_at=started_at,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    global_offset=start,
                    global_total=open_values.shape[0],
                )
                if accepted_chunk_open_values.shape[0] == 0:
                    self._emit_progress(
                        phase="chunk_skipped",
                        completed=end,
                        total=open_values.shape[0],
                        started_at=started_at,
                        chunk_index=chunk_index,
                        chunk_total=chunk_total,
                        message=f"Skipped chunk {chunk_index}/{chunk_total} because all samples were dropped.",
                    )
                    continue
                dataset = IBADataset(
                    input_spec=self.input_spec,
                    open_parameter_values=accepted_chunk_open_values,
                    spectra=chunk_spectra,
                    spectra_lengths=chunk_spectra_lengths,
                    sample_ids=accepted_chunk_sample_ids,
                )
                file_path = target_dir / f"{base_name}_{chunk_index:06d}.h5"
                save_dataset(str(file_path), dataset)
                file_paths.append(str(file_path))
                self._emit_progress(
                    phase="chunk_written",
                    completed=end,
                    total=open_values.shape[0],
                    started_at=started_at,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    message=f"Wrote chunk {chunk_index}/{chunk_total} to {file_path}.",
                )

            return file_paths
        finally:
            if owns_executor:
                executor.shutdown(wait=True)
            if not self.keep_alive:
                self._session_pool.close_all()

    def _validate_open_parameter_values(self, open_parameter_values: np.ndarray) -> np.ndarray:
        open_values = np.asarray(open_parameter_values, dtype=np.float32)
        if open_values.ndim != 2:
            raise ValueError("open_parameter_values must be a 2D array.")
        if open_values.shape[1] != len(self._open_parameters):
            raise ValueError(
                "open_parameter_values column count must match the number of open parameters."
            )
        return open_values

    def _resolve_sample_ids(
        self,
        sample_count: int,
        sample_ids: Sequence[str] | None,
    ) -> list[str]:
        if sample_ids is None:
            return [f"sample-{index:08d}" for index in range(sample_count)]
        if len(sample_ids) != sample_count:
            raise ValueError("sample_ids length must match the number of samples.")
        return list(sample_ids)

    def _build_full_parameter_values(self, open_vector: np.ndarray) -> dict[str, float]:
        if len(open_vector) != len(self._open_parameters):
            raise ValueError("Open vector length does not match the number of open parameters.")

        values = dict(self._fixed_parameter_values)
        for parameter, value in zip(self._open_parameters, open_vector):
            values[parameter.name] = float(value)
        return values

    def _is_retryable_simnra_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        error_type = type(exc).__name__.lower()
        return any(
            token in message or token == error_type
            for token in (
                "com_error",
                "ole",
                "dispatch",
                "rpc",
                "marshalling",
                "marshal",
                "call was rejected",
                "server threw an exception",
                "server is unavailable",
                "rpc server is unavailable",
                "the object invoked has disconnected",
                "catastrophic failure",
            )
        )

    def _generate_method_spectrum_with_retries(
        self,
        method_name: str,
        full_parameter_values: Mapping[str, float],
    ) -> np.ndarray:
        attempts = self.simnra_retry_limit + 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            sessions = self._session_pool.get_sessions()
            try:
                return sessions[method_name].generate_spectrum(full_parameter_values)
            except Exception as exc:
                last_exc = exc
                should_retry = attempt + 1 < attempts and self._is_retryable_simnra_error(exc)
                if not should_retry:
                    raise
                self._emit_retry_message(
                    "[simnra] Recreating thread-local SIMNRA sessions after retryable failure "
                    f"for method '{method_name}' (attempt {attempt + 1}/{attempts}): "
                    f"{type(exc).__name__}: {exc}"
                )
                self._session_pool.reset_current_thread_sessions()
                time.sleep(min(0.05 * (attempt + 1), 0.2))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"SIMNRA generation failed for method '{method_name}'.")

    def _simulate_one_attempt(self, full_parameter_values: Mapping[str, float]) -> dict[str, np.ndarray]:
        return {
            method.name: self._generate_method_spectrum_with_retries(
                method.name,
                full_parameter_values,
            )
            for method in self.input_spec.methods
        }

    def _compute_validation_chi2(
        self,
        primary: np.ndarray,
        validation: np.ndarray,
    ) -> float:
        primary_array = np.asarray(primary, dtype=np.float64)
        validation_array = np.asarray(validation, dtype=np.float64)
        if primary_array.shape != validation_array.shape:
            raise ValueError("Validation chi2 requires equal-length spectra.")
        if primary_array.ndim != 1:
            raise ValueError("Validation chi2 requires 1D spectra.")
        if primary_array.size == 0:
            return 0.0
        chi2 = (primary_array - validation_array) ** 2 / (primary_array + 1.0)
        return float(np.mean(chi2, dtype=np.float64))

    def _validate_attempt_pair(
        self,
        primary_result: Mapping[str, np.ndarray],
        validation_result: Mapping[str, np.ndarray],
    ) -> _ValidationFailure | None:
        for method_name in self.validation_methods:
            primary = np.asarray(primary_result[method_name], dtype=np.float32)
            validation = np.asarray(validation_result[method_name], dtype=np.float32)
            primary_length = int(primary.shape[0])
            validation_length = int(validation.shape[0])

            if self.validation_require_matching_length and primary_length != validation_length:
                return _ValidationFailure(
                    method_name=method_name,
                    reason="length_mismatch",
                    primary_length=primary_length,
                    validation_length=validation_length,
                )

            if primary_length != validation_length:
                continue

            chi2_value = self._compute_validation_chi2(primary, validation)
            if chi2_value > self.validation_chi2_threshold:
                return _ValidationFailure(
                    method_name=method_name,
                    reason="chi2_exceeded",
                    primary_length=primary_length,
                    validation_length=validation_length,
                    chi2=chi2_value,
                )

        return None

    def _build_validation_failure_error(
        self,
        failure: _ValidationFailure,
        *,
        attempts: int,
    ) -> RuntimeError:
        if failure.reason == "length_mismatch":
            return RuntimeError(
                "SIMNRA validation failed after "
                f"{attempts} attempt(s) for method '{failure.method_name}': "
                f"spectrum length mismatch {failure.primary_length} != {failure.validation_length}."
            )
        chi2_value = "nan" if failure.chi2 is None else f"{failure.chi2:.6g}"
        return RuntimeError(
            "SIMNRA validation failed after "
            f"{attempts} attempt(s) for method '{failure.method_name}': "
            f"chi2={chi2_value} exceeds threshold {self.validation_chi2_threshold:.6g}."
        )

    def _emit_validation_message(
        self,
        *,
        attempt: int,
        attempts: int,
        failure: _ValidationFailure,
    ) -> None:
        details = (
            f"lengths {failure.primary_length} vs {failure.validation_length}"
            if failure.reason == "length_mismatch"
            else (
                f"chi2={failure.chi2:.6g}, threshold={self.validation_chi2_threshold:.6g}, "
                f"length={failure.primary_length}"
            )
        )
        self._emit_retry_message(
            "[simnra] Validation failed "
            f"(attempt {attempt}/{attempts}) for method '{failure.method_name}': {details}"
        )

    def _simulate_one(self, open_vector: np.ndarray) -> dict[str, np.ndarray]:
        full_parameter_values = self._build_full_parameter_values(open_vector)
        if not self.validation_enabled or not self.validation_methods:
            return self._simulate_one_attempt(full_parameter_values)

        last_failure: _ValidationFailure | None = None
        attempts = self.validation_max_attempts
        for attempt in range(1, attempts + 1):
            primary_result = self._simulate_one_attempt(full_parameter_values)
            validation_result = self._simulate_one_attempt(full_parameter_values)
            failure = self._validate_attempt_pair(primary_result, validation_result)
            if failure is None:
                return primary_result
            last_failure = failure
            if attempt < attempts:
                self._emit_validation_message(
                    attempt=attempt,
                    attempts=attempts,
                    failure=failure,
                )

        if last_failure is None:
            raise RuntimeError("SIMNRA validation failed without an explicit failure reason.")
        raise self._build_validation_failure_error(last_failure, attempts=attempts)

    def _emit_progress(
        self,
        phase: str,
        completed: int,
        total: int,
        started_at: float,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        message: str = "",
    ) -> None:
        progress = GenerationProgress(
            phase=phase,
            completed=completed,
            total=total,
            elapsed_seconds=time.time() - started_at,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            message=message,
        )
        if self.progress_callback is None:
            if self.print_progress:
                self._print_progress(progress)
            return
        self.progress_callback(progress)
        if self.print_progress:
            self._print_progress(progress)

    def _print_progress(self, progress: GenerationProgress) -> None:
        if progress.phase == "spectra":
            print(
                f"[{progress.elapsed_seconds:8.1f}s] "
                f"{progress.completed}/{progress.total} spectra computed"
                + (
                    f" (chunk {progress.chunk_index}/{progress.chunk_total})"
                    if progress.chunk_index is not None and progress.chunk_total is not None
                    else ""
                )
            )
            return

        if progress.message:
            print(f"[{progress.elapsed_seconds:8.1f}s] {progress.message}")

    def _emit_error(
        self,
        sample_index: int,
        open_parameter_values: np.ndarray,
        exc: BaseException,
        traceback_text: str,
        chunk_index: int | None = None,
    ) -> None:
        if self.error_callback is None:
            return
        self.error_callback(
            GenerationFailure(
                sample_index=sample_index,
                open_parameter_values=np.asarray(open_parameter_values, dtype=np.float32),
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback_text,
                chunk_index=chunk_index,
            )
        )

    def _emit_warning(self, message: str) -> None:
        if not self.log_concentration_corrections:
            return
        with self._warning_lock:
            print(f"[warning] {message}")

    def _emit_retry_message(self, message: str) -> None:
        with self._warning_lock:
            print(message)

    def _generate_spectra_matrices(
        self,
        open_parameter_values: np.ndarray,
        sample_ids: Sequence[str],
        executor: ThreadPoolExecutor,
        started_at: float,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        global_offset: int = 0,
        global_total: int | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        sample_count = open_parameter_values.shape[0]
        if sample_count == 0:
            empty_open_values = np.zeros((0, open_parameter_values.shape[1]), dtype=np.float32)
            empty_spectra = {
                method.name: np.zeros((0, 0), dtype=np.float32)
                for method in self.input_spec.methods
            }
            empty_lengths = {
                method.name: np.zeros((0,), dtype=np.int32)
                for method in self.input_spec.methods
            }
            return empty_open_values, [], empty_spectra, empty_lengths

        result_slots: list[dict[str, np.ndarray] | None] = [None] * sample_count
        failures: list[GenerationFailure] = []
        completed = 0
        progress_total = global_total if global_total is not None else sample_count
        future_to_index = {
            executor.submit(self._simulate_one, open_parameter_values[index]): index
            for index in range(sample_count)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_slots[index] = future.result()
            except Exception as exc:
                tb_text = traceback.format_exc()
                failure = GenerationFailure(
                    sample_index=global_offset + index,
                    open_parameter_values=np.asarray(
                        open_parameter_values[index],
                        dtype=np.float32,
                    ),
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback_text=tb_text,
                    chunk_index=chunk_index,
                )
                failures.append(failure)
                self._emit_error(
                    sample_index=global_offset + index,
                    open_parameter_values=open_parameter_values[index],
                    exc=exc,
                    traceback_text=tb_text,
                    chunk_index=chunk_index,
                )
            completed += 1
            if completed == sample_count or completed % self.progress_every == 0:
                self._emit_progress(
                    phase="spectra",
                    completed=global_offset + completed,
                    total=progress_total,
                    started_at=started_at,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    message=(
                        f"Computed {completed}/{sample_count} spectra"
                        if chunk_index is None
                        else f"Computed {completed}/{sample_count} spectra in chunk {chunk_index}/{chunk_total}"
                    ),
                )

        if failures and not self.allow_failed_samples:
            details = "; ".join(
                f"sample {failure.sample_index}: {failure.error_type}: {failure.message}"
                for failure in failures[:5]
            )
            raise RuntimeError(
                f"Generation failed for {len(failures)} sample(s). {details}"
            )

        kept_indexes = [
            index
            for index, result in enumerate(result_slots)
            if result is not None
        ]
        accepted_open_values = np.asarray(open_parameter_values[kept_indexes], dtype=np.float32)
        accepted_sample_ids = [str(sample_ids[index]) for index in kept_indexes]
        accepted_sample_count = len(kept_indexes)

        spectra: dict[str, np.ndarray] = {}
        spectra_lengths: dict[str, np.ndarray] = {}
        for method in self.input_spec.methods:
            method_rows = [
                result_slots[index][method.name]
                for index in kept_indexes
            ]
            lengths = np.asarray(
                [
                    row.shape[0]
                    for row in method_rows
                ],
                dtype=np.int32,
            )
            max_length = int(lengths.max()) if lengths.size else 0
            padded = np.zeros((accepted_sample_count, max_length), dtype=np.float32)
            for index, row in enumerate(method_rows):
                padded[index, : row.shape[0]] = row
            spectra[method.name] = padded
            spectra_lengths[method.name] = lengths
        return accepted_open_values, accepted_sample_ids, spectra, spectra_lengths

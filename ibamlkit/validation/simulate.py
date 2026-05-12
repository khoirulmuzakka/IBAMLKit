"""Batch simulation backends used by validation and fitting workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import threading

import numpy as np
import torch

from ibamlkit.models.forward import ForwardModelBase
from ibamlkit.schema import DatasetInputSpec, ForwardModelSchema, ParameterSpec
from ibamlkit.training.preprocessing import ArrayTransform

from .base import (
    SimulationBatchResult,
    SpectrumSimulator,
    parameter_name_index,
    resolve_fixed_parameter_values,
    validate_open_parameter_matrix,
)


def _clone_input_spec_with_fixed_overrides(
    input_spec: DatasetInputSpec,
    fixed_parameter_overrides: Mapping[str, float] | None,
) -> DatasetInputSpec:
    if not fixed_parameter_overrides:
        return input_spec

    fixed_values = resolve_fixed_parameter_values(input_spec, fixed_parameter_overrides)
    parameters: list[ParameterSpec] = []
    for parameter in input_spec.parameters:
        if parameter.name in fixed_values and not parameter.is_open:
            parameters.append(
                ParameterSpec(
                    name=parameter.name,
                    group=parameter.group,
                    kind=parameter.kind,
                    is_open=False,
                    method=parameter.method,
                    layer_index=parameter.layer_index,
                    element=parameter.element,
                    isotope=parameter.isotope,
                    unit=parameter.unit,
                    lower_bound=parameter.lower_bound,
                    upper_bound=parameter.upper_bound,
                    fixed_value=fixed_values[parameter.name],
                )
            )
        else:
            parameters.append(parameter)
    return DatasetInputSpec(
        methods=input_spec.methods,
        layer_species=input_spec.layer_species,
        parameters=parameters,
        generation_info=input_spec.generation_info,
        format_version=input_spec.format_version,
    )


def _split_spectra_by_schema(
    predictions: np.ndarray,
    schema: ForwardModelSchema,
) -> dict[str, np.ndarray]:
    if not schema.outputs.spectra_names:
        raise ValueError("Forward schema must declare output spectra_names for surrogate simulation.")
    widths = [int(schema.outputs.spectra_lengths[name]) for name in schema.outputs.spectra_names]
    if predictions.ndim != 2:
        raise ValueError("Surrogate predictions must be a 2D array.")
    if predictions.shape[1] != sum(widths):
        raise ValueError(
            f"Predicted feature width {predictions.shape[1]} does not match schema width {sum(widths)}."
        )

    ret: dict[str, np.ndarray] = {}
    offset = 0
    for name, width in zip(schema.outputs.spectra_names, widths):
        ret[str(name)] = np.asarray(predictions[:, offset : offset + width], dtype=np.float32)
        offset += width
    return ret


@dataclass
class SIMNRABatchSimulator(SpectrumSimulator):
    """Validation-time SIMNRA batch simulator."""

    input_spec: DatasetInputSpec

    def __post_init__(self) -> None:
        self._generator_cache: dict[tuple[int, tuple[tuple[str, float], ...]], object] = {}
        self._cache_lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def method_names(self) -> Sequence[str]:
        return [method.name for method in self.input_spec.methods]

    @staticmethod
    def _cache_key(
        nthreads: int,
        fixed_parameter_overrides: Mapping[str, float] | None,
    ) -> tuple[int, tuple[tuple[str, float], ...]]:
        override_items = tuple(
            sorted((name, float(value)) for name, value in (fixed_parameter_overrides or {}).items())
        )
        return (max(1, int(nthreads)), override_items)

    def _get_generator(
        self,
        *,
        fixed_parameter_overrides: Mapping[str, float] | None,
        nthreads: int,
    ):
        from ibamlkit.generation import SIMNRASpectrumGenerator

        key = self._cache_key(nthreads, fixed_parameter_overrides)
        with self._cache_lock:
            generator = self._generator_cache.get(key)
            if generator is not None:
                return generator

            effective_input_spec = _clone_input_spec_with_fixed_overrides(
                self.input_spec,
                fixed_parameter_overrides,
            )
            generator = SIMNRASpectrumGenerator(
                effective_input_spec,
                max_workers=max(1, int(nthreads)),
                keep_alive=True,
                print_progress=False,
            )
            self._generator_cache[key] = generator
            return generator

    def close(self) -> None:
        with self._cache_lock:
            generators = list(self._generator_cache.values())
            self._generator_cache.clear()
        for generator in generators:
            close = getattr(generator, "close", None)
            if callable(close):
                close()

    def simulate_batch(
        self,
        open_parameter_values: np.ndarray,
        *,
        fixed_parameter_overrides: Mapping[str, float] | None = None,
        nthreads: int = 1,
    ) -> SimulationBatchResult:
        open_matrix = validate_open_parameter_matrix(self.input_spec, open_parameter_values)
        generator = self._get_generator(
            fixed_parameter_overrides=fixed_parameter_overrides,
            nthreads=nthreads,
        )
        dataset = generator.generate(open_matrix)
        return SimulationBatchResult(
            spectra={name: np.asarray(values, dtype=np.float32) for name, values in dataset.spectra.items()},
            spectra_lengths=dataset.spectra_lengths,
        )


@dataclass
class SurrogateBatchSimulator(SpectrumSimulator):
    """Validation-time adapter around an in-memory forward surrogate model."""

    input_spec: DatasetInputSpec
    schema: ForwardModelSchema
    model: ForwardModelBase
    input_transform: ArrayTransform | None = None
    output_inverse_transform: ArrayTransform | None = None

    def __post_init__(self) -> None:
        self._open_parameters = list(self.input_spec.open_parameters)
        self._open_parameter_index = parameter_name_index(self._open_parameters)
        self._all_parameter_index = parameter_name_index(self.input_spec.parameters)

    @property
    def method_names(self) -> Sequence[str]:
        return list(self.schema.outputs.spectra_names)

    def _build_model_inputs(
        self,
        open_parameter_values: np.ndarray,
        fixed_parameter_overrides: Mapping[str, float] | None,
    ) -> np.ndarray:
        batch = validate_open_parameter_matrix(self.input_spec, open_parameter_values)
        fixed_values = resolve_fixed_parameter_values(self.input_spec, fixed_parameter_overrides)

        full_values = np.zeros((batch.shape[0], len(self.input_spec.parameters)), dtype=np.float32)
        for index, parameter in enumerate(self.input_spec.parameters):
            if parameter.is_open:
                full_values[:, index] = batch[:, self._open_parameter_index[parameter.name]]
            else:
                full_values[:, index] = float(fixed_values[parameter.name])

        selected_columns = []
        for feature in self.schema.inputs.features:
            try:
                selected_columns.append(self._all_parameter_index[feature.source_parameter])
            except KeyError as exc:
                raise KeyError(
                    f"Forward schema input feature {feature.name!r} references unknown parameter "
                    f"{feature.source_parameter!r}."
                ) from exc
        model_inputs = np.asarray(full_values[:, selected_columns], dtype=np.float32)
        if self.input_transform is not None:
            model_inputs = self.input_transform.transform(model_inputs)
        return np.asarray(model_inputs, dtype=np.float32)

    def simulate_batch(
        self,
        open_parameter_values: np.ndarray,
        *,
        fixed_parameter_overrides: Mapping[str, float] | None = None,
        nthreads: int = 1,
    ) -> SimulationBatchResult:
        del nthreads
        model_inputs = self._build_model_inputs(
            open_parameter_values,
            fixed_parameter_overrides,
        )
        predictions = self.model.predict(model_inputs)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        predictions = np.asarray(predictions, dtype=np.float32)
        if self.output_inverse_transform is not None:
            predictions = self.output_inverse_transform.inverse_transform(predictions)

        spectra = _split_spectra_by_schema(predictions, self.schema)
        spectra_lengths = {
            method_name: np.full((predictions.shape[0],), values.shape[1], dtype=np.int32)
            for method_name, values in spectra.items()
        }
        return SimulationBatchResult(
            spectra=spectra,
            spectra_lengths=spectra_lengths,
        )


def simulate_batch(
    simulator: SpectrumSimulator,
    open_parameter_values: np.ndarray,
    *,
    fixed_parameter_overrides: Mapping[str, float] | None = None,
    nthreads: int = 1,
) -> SimulationBatchResult:
    """Convenience wrapper for backend-neutral batch simulation."""

    return simulator.simulate_batch(
        open_parameter_values,
        fixed_parameter_overrides=fixed_parameter_overrides,
        nthreads=nthreads,
    )

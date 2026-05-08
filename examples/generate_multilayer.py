"""Example: generate 1-layer to N-layer training datasets.

This script generates one dataset per layer count:

- 1 layer
- 2 layers
- ...
- max_layers

Each case produces ``n_samples_per_case`` samples and writes a single HDF5 file
because the sample count is intentionally small.

Thickness sampling policy
-------------------------

Thickness parameter bounds are defined first:

- every layer thickness lower bound is ``0.0``
- thickness upper bounds increase strictly with depth
- the upper-bound profile uses exponential scaling
- the sum of the per-layer upper bounds grows from ``5e4`` for the 1-layer
  case to ``7e5`` for the ``max_layers`` case

Thickness values are then sampled in two stages:

1. Sample the total envelope thickness.
2. Build exponentially increasing per-layer envelope bounds whose sum equals
   that sampled total.
3. Sample each actual layer thickness independently inside its own envelope.

This gives:

- strictly increasing thickness envelopes with depth
- no rejection step
- thinner surface layers and broader deep-layer thickness ranges
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import sys 
sys.path.append("../")

from ibamlkit.generation import (
    LayerConcentrationSamplingConfig,
    SIMNRASpectrumGenerator,
    sample_open_parameter_matrix,
)
from ibamlkit.schema import DatasetInputSpec, LayerSpeciesSpec, MethodSpec, ParameterSpec


def build_methods() -> list[MethodSpec]:
    return [
        MethodSpec(
            name="NRA",
            reference_file="D:/IBAMLKit/xnra/Ref_all_nra_nopu.xnra",
            file_type="SIMNRA",
        ),
        MethodSpec(
            name="RBS",
            reference_file="D:/IBAMLKit/xnra/Ref_all_rbs_nopu.xnra",
            file_type="SIMNRA",
        ),
    ]


def build_setup_parameters() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="BeamEnergy_NRA",
            group="setup",
            kind="beam_energy",
            method="NRA",
            is_open=False,
            lower_bound=2655.0,
            upper_bound=3245.0,
            fixed_value=2974.0,
            unit="keV",
        ),
        ParameterSpec(
            name="BeamEnergy_RBS",
            group="setup",
            kind="beam_energy",
            method="RBS",
            is_open=False,
            lower_bound=2655.0,
            upper_bound=3245.0,
            fixed_value=2950.0,
            unit="keV",
        ),
        ParameterSpec(
            name="BeamSpread_NRA",
            group="setup",
            kind="beam_spread",
            method="NRA",
            is_open=False,
            lower_bound=0.0,
            upper_bound=0.0,
            fixed_value=0.0,
        ),
        ParameterSpec(
            name="BeamSpread_RBS",
            group="setup",
            kind="beam_spread",
            method="RBS",
            is_open=False,
            lower_bound=0.0,
            upper_bound=0.0,
            fixed_value=0.0,
        ),
        ParameterSpec(
            name="Calib_Linear_NRA",
            group="setup",
            kind="calibration_linear",
            method="NRA",
            is_open=False,
            lower_bound=7.1725,
            upper_bound=7.9275,
            fixed_value=7.55,
        ),
        ParameterSpec(
            name="Calib_Linear_RBS",
            group="setup",
            kind="calibration_linear",
            method="RBS",
            is_open=False,
            lower_bound=2.50528,
            upper_bound=2.769,
            fixed_value=2.637,
        ),
        ParameterSpec(
            name="Calib_Offset_NRA",
            group="setup",
            kind="calibration_offset",
            method="NRA",
            is_open=False,
            lower_bound=-20.0,
            upper_bound=20.0,
            fixed_value=0.0,
        ),
        ParameterSpec(
            name="Calib_Offset_RBS",
            group="setup",
            kind="calibration_offset",
            method="RBS",
            is_open=False,
            lower_bound=-20.0,
            upper_bound=20.0,
            fixed_value=0.0,
        ),
        ParameterSpec(
            name="Calib_Quadratic_NRA",
            group="setup",
            kind="calibration_quadratic",
            method="NRA",
            is_open=False,
            lower_bound=-0.01,
            upper_bound=0.01,
            fixed_value=0.0,
        ),
        ParameterSpec(
            name="Calib_Quadratic_RBS",
            group="setup",
            kind="calibration_quadratic",
            method="RBS",
            is_open=False,
            lower_bound=-0.01,
            upper_bound=0.01,
            fixed_value=-1.47615e-05,
        ),
        ParameterSpec(
            name="FWHM_NRA",
            group="setup",
            kind="fwhm",
            method="NRA",
            is_open=False,
            lower_bound=0.0,
            upper_bound=40.0,
            fixed_value=20.0,
        ),
        ParameterSpec(
            name="FWHM_RBS",
            group="setup",
            kind="fwhm",
            method="RBS",
            is_open=False,
            lower_bound=0.0,
            upper_bound=44.0,
            fixed_value=22.0,
        ),
        ParameterSpec(
            name="ParticlesSr_NRA",
            group="setup",
            kind="particles_sr",
            method="NRA",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1e+17,
            fixed_value=1e+15,
        ),
        ParameterSpec(
            name="ParticlesSr_RBS",
            group="setup",
            kind="particles_sr",
            method="RBS",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1e+17,
            fixed_value=1e+15,
        ),
    ]


def build_layer_elements() -> list[str]:
    return ["Li", "C", "O", "F", "Si", "P",  "Fe", "Al", "Mn", "Sr", "Ti", "K", "Ca" , "Cu"]


def make_layer_species(n_layers: int, elements: Sequence[str]) -> list[LayerSpeciesSpec]:
    return [
        LayerSpeciesSpec(layer_index=layer_index, element=element)
        for layer_index in range(1, n_layers + 1)
        for element in elements
    ]


def make_layer_parameters(
    n_layers: int,
    elements: Sequence[str],
    thickness_upper_bounds_per_layer: Sequence[float],
) -> list[ParameterSpec]:
    parameters: list[ParameterSpec] = []
    for layer_index in range(1, n_layers + 1):
        parameters.append(
            ParameterSpec(
                name=f"Thickness_{layer_index}",
                group="layer",
                kind="thickness",
                layer_index=layer_index,
                is_open=True,
                lower_bound=0.0,
                upper_bound=float(thickness_upper_bounds_per_layer[layer_index - 1]),
                unit="1e15 at/cm2",
            )
        )
        for element in elements:
            parameters.append(
                ParameterSpec(
                    name=f"Conc_{layer_index}_{element}",
                    group="layer",
                    kind="concentration",
                    layer_index=layer_index,
                    element=element,
                    is_open=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                )
            )
    return parameters


def build_input_spec(
    n_layers: int,
    methods: Sequence[MethodSpec],
    setup_parameters: Sequence[ParameterSpec],
    elements: Sequence[str],
    thickness_upper_bounds_per_layer: Sequence[float],
) -> DatasetInputSpec:
    return DatasetInputSpec(
        methods=list(methods),
        layer_species=make_layer_species(n_layers, elements),
        parameters=list(setup_parameters)
        + make_layer_parameters(
            n_layers,
            elements,
            thickness_upper_bounds_per_layer=thickness_upper_bounds_per_layer,
        ),
        generation_info={
            "example": "variable-layer generation",
            "n_layers": n_layers,
            "n_elements_per_layer": len(elements),
            "thickness_policy": (
                "lower bounds are 0.0; upper bounds increase exponentially with depth; "
                "sample total envelope thickness first, then sample each layer thickness "
                "independently within its exponentially scaled envelope"
            ),
        },
    )


def total_thickness_upper_sum(
    n_layers: int,
    max_layers: int,
    total_min_sum: float,
    total_max_sum: float,
) -> float:
    """Interpolate the total thickness envelope from 1-layer to max-layer cases."""

    if max_layers <= 1:
        return total_max_sum
    fraction = (n_layers - 1) / (max_layers - 1)
    return total_min_sum + fraction * (total_max_sum - total_min_sum)


def thickness_upper_bounds(
    n_layers: int,
    max_layers: int,
    total_min_sum: float,
    total_max_sum: float,
    growth_ratio: float,
) -> np.ndarray:
    """Build strictly increasing exponential upper bounds for layer thicknesses."""

    total_upper = total_thickness_upper_sum(
        n_layers=n_layers,
        max_layers=max_layers,
        total_min_sum=total_min_sum,
        total_max_sum=total_max_sum,
    )
    raw = np.asarray([growth_ratio ** layer_index for layer_index in range(n_layers)], dtype=np.float64)
    bounds = total_upper * raw / raw.sum()
    return bounds.astype(np.float32)


def sample_thickness_matrix(
    n_samples: int,
    n_layers: int,
    rng: np.random.Generator,
    total_min_sum: float,
    total_max_sum: float,
    growth_ratio: float,
) -> np.ndarray:
    """Sample thicknesses from per-sample exponentially scaled envelopes.

    For each sample:

    1. Sample a total envelope thickness ``T`` from
       ``[total_min_sum, total_max_sum]``.
    2. Split that envelope into strictly increasing exponential upper bounds
       ``t_1 < t_2 < ... < t_N`` such that ``sum_i t_i = T``.
    3. Sample each actual thickness independently from ``[0, t_i]``.
    """

    if n_layers < 1:
        raise ValueError("n_layers must be >= 1.")

    thicknesses = np.zeros((n_samples, n_layers), dtype=np.float32)
    raw = np.asarray([growth_ratio ** layer_index for layer_index in range(n_layers)], dtype=np.float64)
    raw = raw / raw.sum()

    for row_index in range(n_samples):
        total_envelope = float(rng.uniform(total_min_sum, total_max_sum))
        layer_envelopes = total_envelope * raw
        thicknesses[row_index, :] = rng.uniform(
            low=0.0,
            high=layer_envelopes,
            size=n_layers,
        ).astype(np.float32)

    return thicknesses


def apply_thickness_samples(
    sampled: np.ndarray,
    input_spec: DatasetInputSpec,
    thicknesses: np.ndarray,
) -> None:
    open_parameters = list(input_spec.open_parameters)
    thickness_columns = [
        index
        for index, parameter in enumerate(open_parameters)
        if parameter.group == "layer" and parameter.kind == "thickness"
    ]
    if len(thickness_columns) != thicknesses.shape[1]:
        raise ValueError("Thickness column count does not match the number of layers.")
    sampled[:, thickness_columns] = thicknesses


def build_layer_sampling_config(
    n_layers: int,
    default_config: LayerConcentrationSamplingConfig,
    single_layer_config: LayerConcentrationSamplingConfig,
) -> dict[int, LayerConcentrationSamplingConfig]:
    if n_layers == 1:
        return {1: single_layer_config}
    return {layer_index: default_config for layer_index in range(1, n_layers + 1)}


def main() -> None:
    max_layers = 10
    n_samples_per_case = [150000 for _ in range(max_layers)]
    n_threads = 64
    progress_every = 10000
    thickness_growth_ratio = 1.6
    total_thickness_min_sum = 5.0e4
    total_thickness_max_sum = 7.0e5
    output_root = Path("datasets/multilayer_14_elements")
    default_layer_sampling = LayerConcentrationSamplingConfig(
        anchor_fraction=0.05,
        pure_weight=0.15,
        single_dominant_weight=0.45,
        double_dominant_weight=0.25,
        triple_dominant_weight=0.05,
        sparse_tail_weight=0.1,
        balanced_weight=0.05,
        max_minor_active=1,
    )
    single_layer_sampling = LayerConcentrationSamplingConfig(
        anchor_fraction=0.05,
        pure_weight=0.1,
        single_dominant_weight=0.45,
        double_dominant_weight=0.25,
        triple_dominant_weight=0.05,
        sparse_tail_weight=0.1,
        balanced_weight=0.05,
        max_minor_active=1,
    )

    methods = build_methods()
    setup_parameters = build_setup_parameters()
    elements = build_layer_elements()

    if len(n_samples_per_case) != max_layers:
        raise ValueError("n_samples_per_case must define one entry for each layer-count case.")

    for n_layers in range(1, max_layers + 1):
        case_sample_count = int(n_samples_per_case[n_layers - 1])
        upper_bounds = thickness_upper_bounds(
            n_layers=n_layers,
            max_layers=max_layers,
            total_min_sum=total_thickness_min_sum,
            total_max_sum=total_thickness_max_sum,
            growth_ratio=thickness_growth_ratio,
        )
        input_spec = build_input_spec(
            n_layers=n_layers,
            methods=methods,
            setup_parameters=setup_parameters,
            elements=elements,
            thickness_upper_bounds_per_layer=upper_bounds,
        )

        rng = np.random.default_rng(n_layers)
        sampled = sample_open_parameter_matrix(
            input_spec,
            n_samples=case_sample_count,
            seed=n_layers,
            layer_sampling=build_layer_sampling_config(
                n_layers=n_layers,
                default_config=default_layer_sampling,
                single_layer_config=single_layer_sampling,
            ),
        )
        thicknesses = sample_thickness_matrix(
            n_samples=case_sample_count,
            n_layers=n_layers,
            rng=rng,
            total_min_sum=total_thickness_min_sum,
            total_max_sum=float(upper_bounds.sum()),
            growth_ratio=thickness_growth_ratio,
        )
        apply_thickness_samples(sampled, input_spec, thicknesses)

        generator = SIMNRASpectrumGenerator(
            input_spec=input_spec,
            max_workers=n_threads,
            progress_every=progress_every,
            print_progress=True,
            simnra_retry_limit=3,
            allow_failed_samples=True,
            log_concentration_corrections=True,
            concentration_correction_threshold=1e-4,
        )
        output_dir = output_root / f"layers_{n_layers}"
        written = generator.generate_to_files(
            open_parameter_values=sampled,
            output_dir=str(output_dir),
            base_name=f"multilayer_{n_layers}",
            chunk_size=case_sample_count,
            sample_ids=[f"layers-{n_layers:02d}-{index:06d}" for index in range(case_sample_count)],
        )
        print(f"{n_layers} layer(s): wrote {len(written)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()

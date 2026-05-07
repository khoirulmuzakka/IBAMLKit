"""Example: generate variable-layer datasets with GRR-style thickness bounds.

This script mirrors the core idea of the old ``DataGenerator_GRR``:

- loop from 1 layer up to a requested maximum layer count
- duplicate one base layer composition across all layers
- keep setup parameters unchanged
- expand layer parameters per layer index
- sample open parameters
- generate one dataset family per layer count

The thickness handling follows the old GRR logic:

- a target thickness list is computed from ``calc_thicknesses()``
- for every free thickness parameter in layer ``i``:
  - default value = ``0.5 * target_thickness[i]``
  - lower bound = ``50``
  - upper bound = ``target_thickness[i]``
- fixed thickness parameters stay fixed
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

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
            reference_file="../Ref_nra_nopu.xnra",
            file_type="SIMNRA",
        ),
        MethodSpec(
            name="RBS",
            reference_file="../Ref_rbs_nopu.xnra",
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
            fixed_value=2950.0,
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
            is_open=True,
            lower_bound=7.1725,
            upper_bound=7.9275,
        ),
        ParameterSpec(
            name="Calib_Linear_RBS",
            group="setup",
            kind="calibration_linear",
            method="RBS",
            is_open=True,
            lower_bound=2.50528,
            upper_bound=2.769,
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
            is_open=True,
            lower_bound=-20.0,
            upper_bound=20.0,
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
            is_open=True,
            lower_bound=0.0,
            upper_bound=908000000000.0,
        ),
        ParameterSpec(
            name="ParticlesSr_RBS",
            group="setup",
            kind="particles_sr",
            method="RBS",
            is_open=True,
            lower_bound=0.0,
            upper_bound=84049100000.0,
        ),
    ]


def build_base_layer_species() -> list[str]:
    return ["C", "O", "Y", "Zr", "Ba", "Ce", "D"]


def build_base_layer_parameters() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="Thickness",
            group="layer",
            kind="thickness",
            layer_index=1,
            is_open=True,
            lower_bound=10.0,
            upper_bound=2000.0,
            unit="1e15 at/cm2",
        ),
        ParameterSpec(
            name="Conc_C",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="C",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_O",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="O",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_Y",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="Y",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_Zr",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="Zr",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_Ba",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="Ba",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_Ce",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="Ce",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_D",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="D",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
    ]


def calc_thicknesses(
    n_layers: int,
    thickness_min: float = 50.0,
    thickness_max: float = 500000.0,
) -> list[float]:
    """Replicate the old GRR ``calcThickness`` ramp."""

    _ = thickness_min
    t_max = thickness_max + 0.25 * thickness_max
    return [2.0 * i * t_max / (n_layers * (n_layers + 1)) for i in range(1, n_layers + 1)]


def make_layer_species(n_layers: int, base_elements: Sequence[str]) -> list[LayerSpeciesSpec]:
    layer_species: list[LayerSpeciesSpec] = []
    for layer_index in range(1, n_layers + 1):
        for element in base_elements:
            layer_species.append(LayerSpeciesSpec(layer_index=layer_index, element=element))
    return layer_species


def make_layer_parameters(
    n_layers: int,
    base_layer_parameters: Sequence[ParameterSpec],
) -> list[ParameterSpec]:
    """Expand one base-layer parameter list across ``n_layers``.

    This follows the old GRR rule for free thickness parameters:

    - default = 0.5 * target thickness
    - lower bound = 50
    - upper bound = target thickness
    """

    thickness_targets = calc_thicknesses(n_layers)
    parameters: list[ParameterSpec] = []

    for layer_index in range(1, n_layers + 1):
        for template in base_layer_parameters:
            if template.kind == "thickness":
                target = thickness_targets[layer_index - 1]
                if template.is_open:
                    parameters.append(
                        ParameterSpec(
                            name=f"Thickness_{layer_index}",
                            group="layer",
                            kind="thickness",
                            layer_index=layer_index,
                            is_open=True,
                            lower_bound=50.0,
                            upper_bound=target,
                            unit=template.unit,
                        )
                    )
                else:
                    parameters.append(
                        ParameterSpec(
                            name=f"Thickness_{layer_index}",
                            group="layer",
                            kind="thickness",
                            layer_index=layer_index,
                            is_open=False,
                            lower_bound=template.lower_bound,
                            upper_bound=template.upper_bound,
                            fixed_value=template.fixed_value,
                            unit=template.unit,
                        )
                    )
                continue

            parameters.append(
                replace(
                    template,
                    name=f"{template.name}_{layer_index}" if not template.name.startswith("Conc_") else f"{template.name.split('_')[0]}_{layer_index}_{template.element}",
                    layer_index=layer_index,
                )
            )

    return parameters


def build_input_spec_for_layer_count(
    n_layers: int,
    methods: Sequence[MethodSpec],
    setup_parameters: Sequence[ParameterSpec],
    base_elements: Sequence[str],
    base_layer_parameters: Sequence[ParameterSpec],
) -> DatasetInputSpec:
    return DatasetInputSpec(
        methods=list(methods),
        layer_species=make_layer_species(n_layers, base_elements),
        parameters=list(setup_parameters) + make_layer_parameters(n_layers, base_layer_parameters),
        generation_info={
            "example": "multilayer GRR-style generation",
            "n_layers": n_layers,
            "thickness_rule": "default=0.5*target, lower=50, upper=target for free thickness parameters",
        },
    )


def main() -> None:
    max_layers = 5
    n_samples = 1000
    n_threads = 8
    chunk_size = 250
    output_root = Path("examples/datasets/multilayer")

    methods = build_methods()
    setup_parameters = build_setup_parameters()
    base_elements = build_base_layer_species()
    base_layer_parameters = build_base_layer_parameters()

    for n_layers in range(1, max_layers + 1):
        input_spec = build_input_spec_for_layer_count(
            n_layers=n_layers,
            methods=methods,
            setup_parameters=setup_parameters,
            base_elements=base_elements,
            base_layer_parameters=base_layer_parameters,
        )

        sampled = sample_open_parameter_matrix(
            input_spec,
            n_samples=(2 * n_samples if n_layers == 1 else n_samples),
            seed=n_layers,
            layer_sampling={
                layer_index: LayerConcentrationSamplingConfig(
                    anchor_fraction=0.2,
                    pure_weight=0.0,
                    single_dominant_weight=0.4,
                    double_dominant_weight=0.3,
                    triple_dominant_weight=0.1,
                    sparse_tail_weight=0.15,
                    balanced_weight=0.05,
                    max_minor_active=3,
                )
                for layer_index in range(1, n_layers + 1)
            },
        )

        generator = SIMNRASpectrumGenerator(input_spec=input_spec, max_workers=n_threads)
        output_dir = output_root / f"layers_{n_layers}"
        written = generator.generate_to_files(
            open_parameter_values=sampled,
            output_dir=str(output_dir),
            base_name=f"multilayer_{n_layers}",
            chunk_size=chunk_size,
            sample_ids=[f"layers-{n_layers:02d}-{index:06d}" for index in range(sampled.shape[0])],
        )

        print(f"{n_layers} layer(s): wrote {len(written)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()

"""Example: generate 1000 SIMNRA-backed dataset samples.

This example uses the provided BZCY setup and randomly samples the open
parameters from their declared bounds. 
"""

from __future__ import annotations

from pathlib import Path
import sys 

sys.path.append("../")


from ibamlkit.generation import (
    LayerConcentrationSamplingConfig,
    SIMNRASpectrumGenerator,
    sample_open_parameter_matrix,
)
from ibamlkit.schema import DatasetInputSpec, LayerSpeciesSpec, MethodSpec, ParameterSpec


def build_input_spec() -> DatasetInputSpec:
    methods = [
        MethodSpec(
            name="NRA",
            reference_file="D:/Developments/IBAMLKit/xnra/Ref_nra_nopu.xnra",
            file_type="SIMNRA",
        ),
        MethodSpec(
            name="RBS",
            reference_file="D:/Developments/IBAMLKit/xnra/Ref_rbs_nopu.xnra",
            file_type="SIMNRA",
        ),
    ]

    layer_species = [
        LayerSpeciesSpec(layer_index=1, element="C"),
        LayerSpeciesSpec(layer_index=1, element="O"),
        LayerSpeciesSpec(layer_index=2, element="C"),
        LayerSpeciesSpec(layer_index=2, element="O"),
        LayerSpeciesSpec(layer_index=2, element="Y"),
        LayerSpeciesSpec(layer_index=2, element="Zr"),
        LayerSpeciesSpec(layer_index=2, element="Ba"),
        LayerSpeciesSpec(layer_index=2, element="Ce"),
        LayerSpeciesSpec(layer_index=2, element="D"),
    ]

    parameters = [
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
        ParameterSpec(
            name="Thickness_1",
            group="layer",
            kind="thickness",
            layer_index=1,
            is_open=True,
            lower_bound=10.0,
            upper_bound=2000.0,
            unit="1e15 at/cm2",
        ),
        ParameterSpec(
            name="Thickness_2",
            group="layer",
            kind="thickness",
            layer_index=2,
            is_open=False,
            lower_bound=30000.0,
            upper_bound=600000.0,
            fixed_value=300000.0,
            unit="1e15 at/cm2",
        ),
        ParameterSpec(
            name="Conc_1_C",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="C",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_1_O",
            group="layer",
            kind="concentration",
            layer_index=1,
            element="O",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_value=0.5,
        ),
        # Resolved constant layer-2 composition from the provided Erf_*_A values.
        ParameterSpec(
            name="Conc_2_C",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="C",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_2_O",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="O",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        ParameterSpec(
            name="Conc_2_Y",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="Y",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_value=0.02,
        ),
        ParameterSpec(
            name="Conc_2_Zr",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="Zr",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_value=0.13,
        ),
        ParameterSpec(
            name="Conc_2_Ba",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="Ba",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_value=0.19,
        ),
        ParameterSpec(
            name="Conc_2_Ce",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="Ce",
            is_open=False,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_value=0.04,
        ),
        ParameterSpec(
            name="Conc_2_D",
            group="layer",
            kind="concentration",
            layer_index=2,
            element="D",
            is_open=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
    ]

    return DatasetInputSpec(
        methods=methods,
        layer_species=layer_species,
        parameters=parameters,
        generation_info={
            "example": "BZCY 2-layer constant-composition sampling",
            "settings_fit_nthreads": 8,
        },
    )

def main() -> None:
    n_samples = 100
    chunk_size = 100
    n_threads = 8
    output_dir = Path("examples/datasets/bzcy_1000")

    input_spec = build_input_spec()
    open_parameter_values = sample_open_parameter_matrix(
        input_spec,
        n_samples=n_samples,
        seed=1,
        layer_sampling={
            1: LayerConcentrationSamplingConfig(
                anchor_fraction=0.3,
                pure_weight=0.15,
                single_dominant_weight=0.45,
                double_dominant_weight=0.25,
                triple_dominant_weight=0.0,
                sparse_tail_weight=0.1,
                balanced_weight=0.05,
                max_minor_active=1,
            ),
            2: LayerConcentrationSamplingConfig(
                anchor_fraction=0.2,
                pure_weight=0.1,
                single_dominant_weight=0.35,
                double_dominant_weight=0.3,
                triple_dominant_weight=0.1,
                sparse_tail_weight=0.1,
                balanced_weight=0.05,
                max_minor_active=3,
            ),
        },
    )

    generator = SIMNRASpectrumGenerator(input_spec=input_spec, max_workers=n_threads)
    written_files = generator.generate_to_files(
        open_parameter_values=open_parameter_values,
        output_dir=str(output_dir),
        base_name="bzcy_dataset",
        chunk_size=chunk_size,
        sample_ids=[f"bzcy-{index:05d}" for index in range(n_samples)],
    )

    print(f"Generated {n_samples} samples into {len(written_files)} files:")
    for path in written_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()

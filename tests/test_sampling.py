import unittest

import numpy as np

from ibamlkit.generation.sampling import (
    LayerConcentrationSamplingConfig,
    sample_layer_concentrations,
    sample_open_parameter_matrix,
)
from ibamlkit.schema import DatasetInputSpec, LayerSpeciesSpec, MethodSpec, ParameterSpec


class SamplingTests(unittest.TestCase):
    def test_layer_concentration_sampler_preserves_total(self) -> None:
        rng = np.random.default_rng(1)
        config = LayerConcentrationSamplingConfig(anchor_fraction=0.5)
        samples = sample_layer_concentrations(
            n_elements=4,
            n_samples=50,
            total=0.7,
            rng=rng,
            config=config,
        )
        self.assertEqual(samples.shape, (50, 4))
        self.assertTrue(np.allclose(samples.sum(axis=1), 0.7, atol=1e-6))

    def test_open_parameter_sampler_respects_fixed_concentration_budget(self) -> None:
        input_spec = DatasetInputSpec(
            methods=[MethodSpec(name="RBS", reference_file="ref.xnra", file_type="SIMNRA")],
            layer_species=[
                LayerSpeciesSpec(layer_index=1, element="C"),
                LayerSpeciesSpec(layer_index=1, element="O"),
                LayerSpeciesSpec(layer_index=1, element="Li"),
            ],
            parameters=[
                ParameterSpec(
                    name="Thickness_1",
                    group="layer",
                    kind="thickness",
                    layer_index=1,
                    is_open=True,
                    lower_bound=10.0,
                    upper_bound=20.0,
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
                    is_open=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
                ParameterSpec(
                    name="Conc_1_Li",
                    group="layer",
                    kind="concentration",
                    layer_index=1,
                    element="Li",
                    is_open=False,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    fixed_value=0.2,
                ),
            ],
            generation_info={},
        )

        sampled = sample_open_parameter_matrix(input_spec, n_samples=64, seed=1)
        open_parameters = list(input_spec.open_parameters)
        concentration_columns = [
            index
            for index, parameter in enumerate(open_parameters)
            if parameter.kind == "concentration"
        ]
        self.assertTrue(
            np.allclose(sampled[:, concentration_columns].sum(axis=1), 0.8, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from ibamlkit.models.forward import build_lrn_model_schema
from ibamlkit.schema import DatasetInputSpec, LayerSpeciesSpec, MethodSpec, ParameterSpec
from ibamlkit.training import (
    ConstantFactorTransform,
    LayerwiseConcentrationNormalizer,
    ParameterBoundMinMaxScaler,
    SelectiveMinMaxScaler,
)


class PreprocessingTests(unittest.TestCase):
    def test_constant_factor_transform_supports_out_buffer(self) -> None:
        transform = ConstantFactorTransform(0.25)
        x = np.asarray([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32)
        transform.fit(x)

        out = np.empty_like(x)
        result = transform.inverse_transform(x, out=out)

        self.assertIs(result, out)
        self.assertTrue(np.allclose(out, [[16.0, 32.0], [48.0, 64.0]]))

    def test_selective_minmax_scaler_preserves_passthrough_columns(self) -> None:
        x = np.asarray(
            [
                [0.25, 10.0, 0.75, 100.0],
                [0.50, 20.0, 0.50, 200.0],
                [0.75, 30.0, 0.25, 400.0],
            ],
            dtype=np.float32,
        )
        transform = SelectiveMinMaxScaler(
            scale_columns=[False, True, False, True],
            low=0.0,
            high=1.0,
        ).fit(x)

        transformed = transform.transform(x)
        expected = np.asarray(
            [
                [0.25, 0.0, 0.75, 0.0],
                [0.50, 0.5, 0.50, 1.0 / 3.0],
                [0.75, 1.0, 0.25, 1.0],
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(transformed, expected, atol=1e-6))
        restored = transform.inverse_transform(transformed)
        self.assertTrue(np.allclose(restored, x, atol=1e-6))

    def test_selective_minmax_scaler_builds_passthrough_mask_from_concentration_parameters(self) -> None:
        parameters = [
            ParameterSpec(name="L1_Si", group="layer", kind="concentration", is_open=True, layer_index=1),
            ParameterSpec(name="L1_t", group="layer", kind="thickness", is_open=True, layer_index=1),
            ParameterSpec(name="L2_O", group="layer", kind="concentration", is_open=True, layer_index=2),
            ParameterSpec(name="beam", group="setup", kind="scalar", is_open=True),
        ]

        transform = SelectiveMinMaxScaler.from_passthrough_parameters(parameters)

        self.assertEqual(transform.scale_columns.tolist(), [False, True, False, True])

    def test_selective_minmax_scaler_supports_out_buffer(self) -> None:
        x = np.asarray([[0.1, 5.0], [0.9, 15.0]], dtype=np.float32)
        transform = SelectiveMinMaxScaler(scale_columns=[False, True]).fit(x)

        out = np.empty_like(x)
        result = transform.transform(x, out=out)

        self.assertIs(result, out)
        self.assertTrue(np.allclose(out, [[0.1, 0.0], [0.9, 1.0]], atol=1e-6))

    def test_layerwise_concentration_normalizer_normalizes_fixed_width_inputs(self) -> None:
        x = np.asarray(
            [
                [100.0, 45.0, 20.0, -1.0, 10.0, 30.0],
                [120.0, 50.0, 10.0, 2.0, 40.0, 0.0],
            ],
            dtype=np.float32,
        )
        transform = LayerwiseConcentrationNormalizer(
            setup_feature_count=2,
            layer_param_size=2,
            layer_feature_indices=((2, 3), (4, 5)),
            concentration_indices=((1,), (1,)),
            runtime_prefix_layout=True,
        ).fit(x)

        transformed = transform.transform(x)

        expected = np.asarray(
            [
                [100.0, 45.0, 20.0, 0.0, 10.0, 1.0],
                [120.0, 50.0, 10.0, 1.0, 40.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(transformed, expected, atol=1e-6))

    def test_layerwise_concentration_normalizer_supports_variable_width_inputs(self) -> None:
        parameters = [
            ParameterSpec(name="beam", group="setup", kind="scalar", is_open=True),
            ParameterSpec(name="angle", group="setup", kind="scalar", is_open=True),
            ParameterSpec(name="L1_t", group="layer", kind="thickness", is_open=True, layer_index=1),
            ParameterSpec(
                name="L1_Si",
                group="layer",
                kind="concentration",
                is_open=True,
                layer_index=1,
                element="Si",
                isotope="nat",
            ),
            ParameterSpec(name="L2_t", group="layer", kind="thickness", is_open=True, layer_index=2),
            ParameterSpec(
                name="L2_Si",
                group="layer",
                kind="concentration",
                is_open=True,
                layer_index=2,
                element="Si",
                isotope="nat",
            ),
        ]
        input_spec = DatasetInputSpec(
            methods=[MethodSpec(name="RBS")],
            layer_species=[
                LayerSpeciesSpec(layer_index=1, element="Si"),
                LayerSpeciesSpec(layer_index=2, element="Si"),
            ],
            parameters=parameters,
            generation_info={"n_layers": 2},
        )
        schema = build_lrn_model_schema(
            input_spec,
            task_method_names=["RBS"],
            output_spectra_lengths={"RBS": 4},
        )
        transform = LayerwiseConcentrationNormalizer.from_schema(schema).fit(
            np.zeros((1, schema.inputs.dimension), dtype=np.float32)
        )
        x = np.asarray([[100.0, 45.0, 20.0, 4.0]], dtype=np.float32)

        transformed = transform.transform(x)

        expected = np.asarray([[100.0, 45.0, 20.0, 1.0]], dtype=np.float32)
        self.assertTrue(np.allclose(transformed, expected, atol=1e-6))

    def test_parameter_bound_minmax_scaler_uses_fixed_thickness_bounds_and_passthrough_concentrations(self) -> None:
        parameters = [
            ParameterSpec(name="beam", group="setup", kind="beam_energy", is_open=True),
            ParameterSpec(name="L1_t", group="layer", kind="thickness", is_open=True, layer_index=1),
            ParameterSpec(name="L1_Si", group="layer", kind="concentration", is_open=True, layer_index=1),
        ]
        x = np.asarray(
            [
                [2600.0, 1000.0, 0.25],
                [3200.0, 50000.0, 0.75],
            ],
            dtype=np.float32,
        )
        transform = ParameterBoundMinMaxScaler.from_parameters(
            parameters,
            passthrough_kinds=("concentration",),
            fixed_bounds_by_kind={"thickness": (0.0, 100000.0)},
            low=0.0,
            high=1.0,
        ).fit(x)

        transformed = transform.transform(x)

        self.assertTrue(np.allclose(transformed[:, 0], [0.0, 1.0], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 1], [0.01, 0.5], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 2], x[:, 2], atol=1e-6))

    def test_parameter_bound_minmax_scaler_supports_out_buffer(self) -> None:
        parameters = [
            ParameterSpec(name="L1_t", group="layer", kind="thickness", is_open=True, layer_index=1),
            ParameterSpec(name="L1_Si", group="layer", kind="concentration", is_open=True, layer_index=1),
        ]
        x = np.asarray([[1000.0, 0.25], [50000.0, 0.75]], dtype=np.float32)
        transform = ParameterBoundMinMaxScaler.from_parameters(
            parameters,
            passthrough_kinds=("concentration",),
            fixed_bounds_by_kind={"thickness": (0.0, 100000.0)},
        ).fit(x)

        out = np.empty_like(x)
        result = transform.transform(x, out=out)

        self.assertIs(result, out)
        self.assertTrue(np.allclose(out[:, 0], [0.01, 0.5], atol=1e-6))
        self.assertTrue(np.allclose(out[:, 1], x[:, 1], atol=1e-6))

    def test_parameter_bound_minmax_scaler_supports_variable_width_inputs(self) -> None:
        parameters = [
            ParameterSpec(name="beam", group="setup", kind="beam_energy", is_open=True),
            ParameterSpec(name="angle", group="setup", kind="angle", is_open=True),
            ParameterSpec(name="L1_t", group="layer", kind="thickness", is_open=True, layer_index=1),
            ParameterSpec(name="L1_Si", group="layer", kind="concentration", is_open=True, layer_index=1),
            ParameterSpec(name="L2_t", group="layer", kind="thickness", is_open=True, layer_index=2),
            ParameterSpec(name="L2_Si", group="layer", kind="concentration", is_open=True, layer_index=2),
        ]
        fitted = np.asarray(
            [
                [2600.0, 30.0, 1000.0, 0.25, 5000.0, 0.75],
                [3200.0, 60.0, 50000.0, 0.75, 90000.0, 0.25],
            ],
            dtype=np.float32,
        )
        variable = np.asarray(
            [[2900.0, 45.0, 10000.0, 0.2, 20000.0, 0.3, 25000.0, 0.5]],
            dtype=np.float32,
        )
        transform = ParameterBoundMinMaxScaler.from_parameters(
            parameters,
            passthrough_kinds=("concentration",),
            fixed_bounds_by_kind={"thickness": (0.0, 100000.0)},
            low=0.0,
            high=1.0,
        ).fit(fitted)

        transformed = transform.transform(variable)

        self.assertTrue(np.allclose(transformed[:, 0], [0.5], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 1], [0.5], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 2], [0.1], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 3], [0.2], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 4], [0.2], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 5], [0.3], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 6], [0.25], atol=1e-6))
        self.assertTrue(np.allclose(transformed[:, 7], [0.5], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from ibamlkit.models.forward import ForwardModelBase
from ibamlkit.schema import (
    DatasetInputSpec,
    ForwardModelSchema,
    LayerSpeciesSpec,
    MethodSpec,
    ModelInputSpec,
    ModelOutputSpec,
    ModelTaskSpec,
    ParameterSpec,
    TensorFeatureSpec,
)
from ibamlkit.validation import (
    SurrogateBatchSimulator,
    calculate_chi2_batch,
    fit_open_parameters,
    simulate_batch,
)

try:
    import minionpy  # noqa: F401
    HAS_MINIONPY = True
except Exception:
    HAS_MINIONPY = False


class LinearForwardModel(ForwardModelBase):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x0 = inputs[:, 0]
        x1 = inputs[:, 1]
        y0 = x0 + 2.0 * x1
        y1 = 3.0 * x0 - x1
        return torch.stack((y0, y1), dim=1)


class ValidationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input_spec = DatasetInputSpec(
            methods=[MethodSpec(name="RBS", reference_file="ref.xnra", file_type="SIMNRA")],
            layer_species=[LayerSpeciesSpec(layer_index=1, element="Li")],
            parameters=[
                ParameterSpec(
                    name="SetupGain",
                    group="setup",
                    kind="gain",
                    is_open=False,
                    fixed_value=1.0,
                ),
                ParameterSpec(
                    name="Thickness_1",
                    group="layer",
                    kind="thickness",
                    is_open=True,
                    layer_index=1,
                    lower_bound=0.0,
                    upper_bound=2.0,
                ),
                ParameterSpec(
                    name="Conc_1_Li",
                    group="layer",
                    kind="concentration",
                    is_open=True,
                    layer_index=1,
                    element="Li",
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            ],
            generation_info={},
        )
        self.schema = ForwardModelSchema(
            name="linear_test_model",
            task=ModelTaskSpec(task_kind="surrogate", method_names=["RBS"]),
            inputs=ModelInputSpec(
                features=[
                    TensorFeatureSpec(
                        name="Thickness_1",
                        source_parameter="Thickness_1",
                        role="input",
                        group="layer",
                        layer_index=1,
                    ),
                    TensorFeatureSpec(
                        name="Conc_1_Li",
                        source_parameter="Conc_1_Li",
                        role="input",
                        group="layer",
                        layer_index=1,
                    ),
                ]
            ),
            outputs=ModelOutputSpec(
                spectra_names=["RBS"],
                spectra_lengths={"RBS": 2},
            ),
            parameters=self.input_spec.parameters,
        )
        self.model = LinearForwardModel(self.schema)
        self.simulator = SurrogateBatchSimulator(
            input_spec=self.input_spec,
            schema=self.schema,
            model=self.model,
        )

    def test_simulate_batch_returns_expected_spectrum(self) -> None:
        open_values = np.asarray([[1.0, 0.25], [0.5, 0.5]], dtype=np.float32)
        result = simulate_batch(self.simulator, open_values)
        expected = np.asarray([[1.5, 2.75], [1.5, 1.0]], dtype=np.float32)
        self.assertTrue(np.allclose(result.spectra["RBS"], expected))
        self.assertEqual(result.spectra_lengths["RBS"].tolist(), [2, 2])

    def test_calculate_chi2_batch_matches_zero_for_exact_data(self) -> None:
        open_values = np.asarray([[1.0, 0.25], [0.5, 0.5]], dtype=np.float32)
        observed = simulate_batch(self.simulator, open_values).spectra
        chi2 = calculate_chi2_batch(self.simulator, open_values, observed)
        self.assertTrue(np.allclose(chi2.total, 0.0))
        self.assertTrue(np.allclose(chi2.per_method["RBS"], 0.0))

    @unittest.skipUnless(HAS_MINIONPY, "minionpy is required for fitting test")
    def test_fit_open_parameters_recovers_known_solution(self) -> None:
        target_open_values = np.asarray([[1.0, 0.25]], dtype=np.float32)
        observed = simulate_batch(self.simulator, target_open_values).spectra
        initial = np.asarray([[0.1, 0.9]], dtype=np.float32)
        fit_result = fit_open_parameters(
            self.simulator,
            self.input_spec,
            observed,
            initial_open_parameter_values=initial,
            algo="L_BFGS_B",
            maxevals=200,
            rel_tol=0.0,
        )
        self.assertTrue(fit_result.samples[0].success)
        self.assertTrue(
            np.allclose(
                fit_result.best_open_parameter_values[0],
                target_open_values[0],
                atol=1e-4,
            )
        )
        self.assertLess(fit_result.best_chi2[0], 1e-8)


if __name__ == "__main__":
    unittest.main()

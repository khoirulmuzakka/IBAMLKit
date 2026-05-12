import unittest

import numpy as np

from ibamlkit.models.inverse import InverseMLPModel, build_inverse_model_schema
from ibamlkit.schema import DatasetInputSpec, IBADataset, LayerSpeciesSpec, MethodSpec, ParameterSpec
from ibamlkit.training import prepare_inverse_dataset


class InversePipelineTests(unittest.TestCase):
    def test_inverse_schema_and_dataset_prep_follow_open_parameter_order(self) -> None:
        dataset = IBADataset(
            input_spec=DatasetInputSpec(
                methods=[MethodSpec(name="RBS", reference_file="rbs.xnra", file_type="XNRA/IDF")],
                layer_species=[
                    LayerSpeciesSpec(layer_index=1, element="Li"),
                    LayerSpeciesSpec(layer_index=1, element="O"),
                ],
                parameters=[
                    ParameterSpec(
                        name="Thickness_1",
                        group="layer",
                        kind="thickness",
                        is_open=True,
                        layer_index=1,
                        lower_bound=100.0,
                        upper_bound=5000.0,
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
                    ParameterSpec(
                        name="Conc_1_O",
                        group="layer",
                        kind="concentration",
                        is_open=False,
                        layer_index=1,
                        element="O",
                        fixed_value=0.8,
                    ),
                ],
                generation_info={},
            ),
            open_parameter_values=np.asarray([[1000.0, 0.2], [1200.0, 0.25]], dtype=np.float32),
            spectra={"RBS": np.asarray([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=np.float32)},
            spectra_lengths={"RBS": np.asarray([3, 4], dtype=np.int32)},
            sample_ids=["sample-1", "sample-2"],
        )

        schema = build_inverse_model_schema(
            dataset,
            method_name="RBS",
            target_parameter_names=["Conc_1_Li", "Thickness_1"],
        )
        prepared = prepare_inverse_dataset(dataset, schema=schema, method_name="RBS")

        self.assertEqual(schema.task.task_kind, "inverse")
        self.assertEqual(schema.inputs.dimension, 4)
        self.assertEqual(
            [feature.source_parameter for feature in schema.outputs.features],
            ["Conc_1_Li", "Thickness_1"],
        )
        self.assertEqual(prepared.inputs.shape, (2, 4))
        self.assertEqual(prepared.targets_full.shape, (2, 2))
        self.assertEqual(prepared.targets_selected.shape, (2, 2))
        self.assertTrue(
            np.allclose(
                prepared.targets_selected,
                np.asarray([[0.2, 1000.0], [0.25, 1200.0]], dtype=np.float32),
            )
        )
        self.assertEqual(prepared.input_lengths.tolist(), [3, 4])

        model = InverseMLPModel(schema, hidden_sizes=(8, 4))
        predictions = model.predict(prepared.inputs)
        self.assertEqual(tuple(predictions.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from ibamlkit.models.forward import LRNModel, build_lrn_model_schema
from ibamlkit.schema import DatasetInputSpec, LayerSpeciesSpec, MethodSpec, ParameterSpec


def _make_input_spec(layer_count: int) -> DatasetInputSpec:
    parameters = [
        ParameterSpec(name="beam_energy", group="setup", kind="energy", is_open=True),
        ParameterSpec(name="detector_angle", group="setup", kind="angle", is_open=True),
    ]
    layer_species = []
    for layer_index in range(1, layer_count + 1):
        layer_species.append(LayerSpeciesSpec(layer_index=layer_index, element="Si"))
        parameters.extend(
            [
                ParameterSpec(
                    name=f"layer_{layer_index}_thickness",
                    group="layer",
                    kind="thickness",
                    layer_index=layer_index,
                    is_open=True,
                ),
                ParameterSpec(
                    name=f"layer_{layer_index}_si_concentration",
                    group="layer",
                    kind="concentration",
                    layer_index=layer_index,
                    element="Si",
                    isotope="nat",
                    is_open=True,
                ),
            ]
        )
    return DatasetInputSpec(
        methods=[MethodSpec(name="RBS")],
        layer_species=layer_species,
        parameters=parameters,
        generation_info={"n_layers": layer_count},
    )


def _make_model(layer_count: int) -> LRNModel:
    torch.manual_seed(7)
    schema = build_lrn_model_schema(
        _make_input_spec(layer_count),
        task_method_names=["RBS"],
        output_spectra_lengths={"RBS": 8},
    )
    return LRNModel(
        schema,
        hidden_size=16,
        contribution_size=12,
        setup_embedding_dim=8,
        layer_embedding_dim=10,
        block_hidden_sizes=(14,),
        decoder_hidden_sizes=(18,),
        refiner_hidden_channels=8,
        refiner_kernel_size=5,
    )


class LRNVariableInferenceTests(unittest.TestCase):
    def test_variable_length_matches_zero_padded_inference(self) -> None:
        model = _make_model(layer_count=2)
        short_input = torch.tensor([[1.2, 35.0, 120.0, 0.7]], dtype=torch.float32)
        padded_input = torch.tensor([[1.2, 35.0, 120.0, 0.7, 0.0, 0.0]], dtype=torch.float32)

        short_output = model.predict(short_input)
        padded_output = model.predict(padded_input)

        self.assertTrue(torch.allclose(short_output, padded_output, atol=1e-6, rtol=1e-6))

    def test_variable_length_accepts_more_layers_than_training_schema(self) -> None:
        model = _make_model(layer_count=1)
        inputs = torch.tensor([[1.2, 35.0, 120.0, 0.6, 80.0, 0.4]], dtype=torch.float32)

        outputs = model.predict(inputs)

        self.assertEqual(outputs.shape, (1, 8))

    def test_variable_length_rejects_partial_layer_blocks(self) -> None:
        model = _make_model(layer_count=2)
        inputs = torch.tensor([[1.2, 35.0, 120.0]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "setup \\+ N \\* layer_features"):
            model.predict(inputs)


if __name__ == "__main__":
    unittest.main()

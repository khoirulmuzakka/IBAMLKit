import tempfile
import unittest
from pathlib import Path

import numpy as np

from ibamlkit.data import load_dataset, save_dataset
from ibamlkit.schema import (
    DatasetInputSpec,
    IBADataset,
    LayerSpeciesSpec,
    MethodSpec,
    ParameterSpec,
)


class DatasetIORoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_core_shapes_and_metadata(self) -> None:
        dataset = IBADataset(
            input_spec=DatasetInputSpec(
                methods=[
                    MethodSpec(name="RBS", reference_file="rbs.xnra", file_type="XNRA/IDF"),
                    MethodSpec(name="NRA", reference_file="nra.xnra", file_type="XNRA/IDF"),
                ],
                layer_species=[
                    LayerSpeciesSpec(layer_index=1, element="Li"),
                    LayerSpeciesSpec(layer_index=1, element="O"),
                ],
                parameters=[
                    ParameterSpec(
                        name="BeamEnergy_RBS",
                        group="setup",
                        kind="beam_energy",
                        is_open=False,
                        method="RBS",
                        fixed_value=2000.0,
                        unit="keV",
                    ),
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
                generation_info={"source_application": "AutoNRA"},
            ),
            open_parameter_values=np.asarray([[1000.0, 0.2], [1200.0, 0.25]], dtype=np.float32),
            spectra={
                "RBS": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                "NRA": np.asarray([[7, 8, 0], [9, 10, 11]], dtype=np.float32),
            },
            spectra_lengths={
                "RBS": np.asarray([3, 3], dtype=np.int32),
                "NRA": np.asarray([2, 3], dtype=np.int32),
            },
            sample_ids=["sample-1", "sample-2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.h5"
            save_dataset(str(path), dataset)
            loaded = load_dataset(str(path))

        self.assertEqual(loaded.sample_count, 2)
        self.assertEqual([method.name for method in loaded.input_spec.methods], ["RBS", "NRA"])
        self.assertEqual(loaded.open_parameter_values.shape, (2, 2))
        self.assertEqual(loaded.spectra["RBS"].shape, (2, 3))
        self.assertEqual(loaded.spectra_lengths["NRA"].tolist(), [2, 3])
        self.assertEqual(
            [parameter.name for parameter in loaded.input_spec.fixed_parameters],
            ["BeamEnergy_RBS", "Conc_1_O"],
        )


if __name__ == "__main__":
    unittest.main()

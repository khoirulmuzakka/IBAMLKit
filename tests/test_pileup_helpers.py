import unittest

import numpy as np

from ibamlkit.pileup import resolve_channel_conversion_arrays
from ibamlkit.schema import DatasetInputSpec, LayerSpeciesSpec, MethodSpec, ParameterSpec


class PileupHelperTests(unittest.TestCase):
    def test_resolve_channel_conversion_arrays_prefers_kind_metadata(self) -> None:
        input_spec = DatasetInputSpec(
            methods=[MethodSpec(name="RBS", reference_file="ref.xnra", file_type="SIMNRA")],
            layer_species=[LayerSpeciesSpec(layer_index=1, element="Li")],
            parameters=[
                ParameterSpec(
                    name="Calib_Offset_RBS",
                    group="setup",
                    kind="calibration_offset",
                    is_open=False,
                    method="RBS",
                    fixed_value=10.0,
                ),
                ParameterSpec(
                    name="Calib_Linear_RBS",
                    group="setup",
                    kind="calibration_linear",
                    is_open=True,
                    method="RBS",
                    lower_bound=1.0,
                    upper_bound=2.0,
                ),
                ParameterSpec(
                    name="Calib_Quadratic_RBS",
                    group="setup",
                    kind="calibration_quadratic",
                    is_open=False,
                    method="RBS",
                    fixed_value=0.25,
                ),
                ParameterSpec(
                    name="ParticlesSr_RBS",
                    group="setup",
                    kind="particles_sr",
                    is_open=False,
                    method="RBS",
                    fixed_value=5000.0,
                ),
                ParameterSpec(
                    name="Thickness_1",
                    group="layer",
                    kind="thickness",
                    is_open=True,
                    layer_index=1,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            ],
            generation_info={},
        )
        open_parameter_names = ["Calib_Linear_RBS", "Thickness_1"]
        open_parameter_values = np.asarray([[1.5, 0.4], [1.75, 0.6]], dtype=np.float32)

        a, b, c, r = resolve_channel_conversion_arrays(
            input_spec,
            open_parameter_names,
            open_parameter_values,
            method_name="RBS",
            energy_spectrum_scale=1e-3,
        )

        self.assertTrue(np.allclose(a, [10.0, 10.0]))
        self.assertTrue(np.allclose(b, [1.5, 1.75]))
        self.assertTrue(np.allclose(c, [0.25, 0.25]))
        self.assertTrue(np.allclose(r, [1.0, 1.0]))

    def test_resolve_channel_conversion_arrays_supports_legacy_name_patterns(self) -> None:
        input_spec = DatasetInputSpec(
            methods=[MethodSpec(name="RBS", reference_file="ref.xnra", file_type="SIMNRA")],
            layer_species=[LayerSpeciesSpec(layer_index=1, element="Li")],
            parameters=[
                ParameterSpec(
                    name="Calib_Offset_RBS",
                    group="setup",
                    kind="legacy_offset",
                    is_open=False,
                    method="RBS",
                    fixed_value=5.0,
                ),
                ParameterSpec(
                    name="Calib_Linear_RBS",
                    group="setup",
                    kind="legacy_linear",
                    is_open=False,
                    method="RBS",
                    fixed_value=2.0,
                ),
                ParameterSpec(
                    name="Calib_Quadratic_RBS",
                    group="setup",
                    kind="legacy_quadratic",
                    is_open=False,
                    method="RBS",
                    fixed_value=0.125,
                ),
                ParameterSpec(
                    name="ParticlesSr_RBS",
                    group="setup",
                    kind="legacy_particles",
                    is_open=False,
                    method="RBS",
                    fixed_value=4000.0,
                ),
                ParameterSpec(
                    name="Thickness_1",
                    group="layer",
                    kind="thickness",
                    is_open=True,
                    layer_index=1,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
            ],
            generation_info={},
        )
        open_parameter_names = ["Thickness_1"]
        open_parameter_values = np.asarray([[0.4], [0.6]], dtype=np.float32)

        a, b, c, r = resolve_channel_conversion_arrays(
            input_spec,
            open_parameter_names,
            open_parameter_values,
            method_name="RBS",
            energy_spectrum_scale=1e-3,
        )

        self.assertTrue(np.allclose(a, [5.0, 5.0]))
        self.assertTrue(np.allclose(b, [2.0, 2.0]))
        self.assertTrue(np.allclose(c, [0.125, 0.125]))
        self.assertTrue(np.allclose(r, [1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()

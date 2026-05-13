import unittest
from unittest.mock import patch

import numpy as np

from ibamlkit.pileup import compute_rebin_energy_edges, resolve_channel_conversion_arrays
from ibamlkit.pileup import pileup as pileup_module
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

    def test_rebin_spectra_to_energy_space_writes_into_preallocated_output(self) -> None:
        channel_space_spectra = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        channel_lengths = np.asarray([3, 0], dtype=np.int32)
        calibration_offset = np.asarray([0.0, 0.0], dtype=np.float64)
        calibration_linear = np.asarray([1.0, 1.0], dtype=np.float64)
        calibration_quadratic = np.asarray([0.0, 0.0], dtype=np.float64)
        row_indices = np.asarray([1, 0], dtype=np.int64)
        energy_edges = compute_rebin_energy_edges(
            channel_space_spectra,
            channel_lengths,
            calibration_offset=calibration_offset,
            calibration_linear=calibration_linear,
            calibration_quadratic=calibration_quadratic,
            energy_bin_width=1.0,
            row_indices=row_indices,
            show_progress=False,
        )
        out = np.zeros((2, 3), dtype=np.float32)

        def fake_rebin_histogram(bin_edges_old, bin_edges_new, spectrum):
            del bin_edges_old, bin_edges_new, spectrum
            return np.asarray([10.0, 20.0], dtype=np.float64)

        with patch.object(pileup_module, "_require_pileupcpp", lambda: None), patch.object(
            pileup_module, "rebinpp", fake_rebin_histogram
        ):
            rebinned, rebinned_lengths, energy_edges = pileup_module.rebin_spectra_to_energy_space(
                channel_space_spectra,
                channel_lengths,
                calibration_offset=calibration_offset,
                calibration_linear=calibration_linear,
                calibration_quadratic=calibration_quadratic,
                energy_bin_width=1.0,
                target_width=None,
                energy_spectrum_scale=1.0,
                row_indices=row_indices,
                energy_edges=energy_edges,
                out=out,
                show_progress=False,
            )

        self.assertIs(rebinned, out)
        self.assertEqual(rebinned.shape, (2, 3))
        self.assertTrue(np.allclose(rebinned[0], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(rebinned[1], [10.0, 20.0, 0.0]))
        self.assertTrue(np.array_equal(rebinned_lengths, [0, 2]))
        self.assertTrue(np.allclose(energy_edges, [0.0, 1.0, 2.0, 3.0]))

    def test_convert_to_channel_space_and_pileup_batch_clamps_non_monotonic_calibration(self) -> None:
        captured = {}

        def fake_convert(a, b, c, real_times, live_times, fudge_factors, r, E_space_spectra, clip_negative):
            captured["a"] = np.asarray(a)
            captured["b"] = np.asarray(b)
            captured["c"] = np.asarray(c)
            captured["shape"] = np.asarray(E_space_spectra).shape
            return np.zeros_like(E_space_spectra, dtype=np.float32)

        with patch.object(pileup_module, "_require_pileupcpp", lambda: None), patch.object(
            pileup_module, "convert_to_channel_space_and_pileup_batchpp", fake_convert
        ):
            result = pileup_module.convert_to_channel_space_and_pileup_batch(
                np.asarray([0.0, 0.0], dtype=np.float64),
                np.asarray([-1.0, 2.0], dtype=np.float64),
                np.asarray([-0.1, -0.1], dtype=np.float64),
                real_times=np.asarray([1.0, 1.0], dtype=np.float64),
                live_times=np.asarray([1.0, 1.0], dtype=np.float64),
                fudge_factors=np.asarray([0.1, 0.1], dtype=np.float64),
                r=np.asarray([1.0, 1.0], dtype=np.float64),
                E_space_spectra=np.asarray([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                clip_negative=True,
            )

        self.assertEqual(captured["shape"], (2, 4))
        self.assertGreater(captured["b"][0], 0.0)
        self.assertGreaterEqual(captured["c"][0], 0.0)
        self.assertAlmostEqual(float(captured["b"][1]), 2.0)
        self.assertAlmostEqual(float(captured["c"][1]), -0.1)
        self.assertEqual(result.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()

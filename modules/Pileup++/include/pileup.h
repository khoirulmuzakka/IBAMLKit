#ifndef UTILITY_H
#define UTILITY_H

#include <windows.h>
#include <filesystem>
#include <string>
#include <locale>
#include <codecvt>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Enforce OpenMP at compile time for the whole project.
#ifndef _OPENMP
#error "OpenMP support is required. Please enable OpenMP in your compiler/toolchain (e.g., -fopenmp or /openmp)."
#endif

namespace py = pybind11;
// rebin: accepts numpy arrays (float64, C-contiguous). No implicit conversions => no elementwise copying.
py::array_t<double> rebin_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges_old,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges_new,
    py::array_t<double, py::array::c_style | py::array::forcecast> spectrum);


py::array_t<double> fast_pileup_batch_numpy(
    const py::array_t<double>& spectra,
    const py::array_t<double>& real_times,
    const py::array_t<double>& live_times,
    const py::array_t<double>& fudge_factors,
    bool clip_negative = true
);

// Convert E-space spectra to channel space (per-spectrum quadratic calibration) and apply pileup.
// a, b, c, real_times, live_times, fudge_factors: 1D arrays of length n_spectra
// E_space_spectra: 2D array (n_spectra, K)
py::array_t<double> convert_to_channel_space_and_pileup_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> c,
    py::array_t<double, py::array::c_style | py::array::forcecast> real_times,
    py::array_t<double, py::array::c_style | py::array::forcecast> live_times,
    py::array_t<double, py::array::c_style | py::array::forcecast> fudge_factors,
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> E_space_spectra,
    bool clip_negative = true);

#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector, std::wstring
#include "pileup.h"

namespace py = pybind11;

PYBIND11_MODULE(pileupcpp, m) {
    m.doc() = "Python bindings pileup";

     m.def("rebin_histogram", &rebin_numpy,
          py::arg("bin_edges_old"),
          py::arg("bin_edges_new"),
          py::arg("spectrum")
          //py::call_guard<py::gil_scoped_release>()
     );

     m.def("fast_pileup_batch", &fast_pileup_batch_numpy,
          py::arg("spectra"),
          py::arg("real_times"),
          py::arg("live_times"),
          py::arg("fudge_factors"),
          py::arg("clip_negative") = true
     );

     // Overload: scalar calibration/timing parameters for all spectra
     m.def("convert_to_channel_space_and_pileup_batch", &convert_to_channel_space_and_pileup_batch,
          py::arg("a"),
          py::arg("b"),
          py::arg("c"),
          py::arg("real_times"),
          py::arg("live_times"),
          py::arg("fudge_factors"),
          py::arg("r"),
          py::arg("E_space_spectra"),
          py::arg("clip_negative") = true
     );
}

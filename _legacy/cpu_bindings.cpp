#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include "pfaffian.h"

namespace py = pybind11;

double pfaffian_wrapper(py::array_t<double, py::array::c_style | py::array::forcecast> array, int panel_size) {
    py::buffer_info buf = array.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D NumPy array.");
    }

    if (buf.shape[0] != buf.shape[1]) {
        throw std::runtime_error("Matrix must be square.");
    }

    int n = static_cast<int>(buf.shape[0]);

    if (n % 2 != 0) {
        throw std::runtime_error("Matrix dimension must be even for Pfaffian.");
    }
    
    if (panel_size <= 0 || panel_size > n) {
        throw std::runtime_error("Panel size must be between 1 and n.");
    }

    py::array_t<double> copy = array.attr("copy")();
    py::buffer_info copy_buf = copy.request();

    double* data_ptr = static_cast<double*>(copy_buf.ptr);

    return pfaffian(data_ptr, n, panel_size);
}

PYBIND11_MODULE(pfaffian_module, m) {
    m.doc() = "Pfaffian computation using C++ implementation";

    // Added the argument name and a default value (optional but recommended)
    m.def("pfaffian", &pfaffian_wrapper,
          "Compute the Pfaffian of a skew-symmetric matrix",
          py::arg("array"), 
          py::arg("panel_size") = 128); // Defaulting to 128 or any sensible value
}
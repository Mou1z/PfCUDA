#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "pfaffian.h"

double pfaffian(const double* A, const long n);

namespace py = pybind11;

double pfaffian_numpy(py::array_t<double, py::array::f_style | py::array::forcecast> A) {
    if (A.ndim() != 2)
        throw std::runtime_error("Input must be a 2D matrix");

    if (A.shape(0) != A.shape(1))
        throw std::runtime_error("Matrix must be square");

    if (A.shape(0) % 2 != 0)
        throw std::runtime_error("Matrix dimension must be even");

    const long n = A.shape(0);
    const double* data = A.data();

    return pfaffian(data, n);
}

PYBIND11_MODULE(pfaffian_cuda, m) {
    m.doc() = "CUDA Pfaffian computation";
    m.def("pfaffian", &pfaffian_numpy, "Compute Pfaffian of skew-symmetric matrix");
}

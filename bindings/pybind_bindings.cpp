#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double pfaffian(double* A, const int n);

double pfaffian_wrapper(py::array_t<double, py::array::c_style | py::array::forcecast> array) {
    auto buf = array.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Input must be 2D");

    int n = buf.shape[0];

    if (n != buf.shape[1])
        throw std::runtime_error("Matrix must be square");

    if (n % 2 != 0)
        throw std::runtime_error("n must be even");

    double* ptr = static_cast<double*>(buf.ptr);

    return pfaffian(ptr, n);
}

PYBIND11_MODULE(cpupfaffian, m) {
    m.def("pfaffian", &pfaffian_wrapper,
          "Compute Pfaffian (modifies input matrix in-place)");
}
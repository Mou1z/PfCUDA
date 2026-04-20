#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

#include "pfaffian.cuh"
#include "slog_pfaffian.cuh"

namespace py = pybind11;

float pfaffian_float(py::array_t<float, py::array::f_style | py::array::forcecast> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Matrix must be square");

    return pfaffian<float>((float*)buf.ptr, buf.shape[0]);
}

double pfaffian_double(py::array_t<double, py::array::f_style | py::array::forcecast> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Matrix must be square");

    return pfaffian<double>((double*)buf.ptr, buf.shape[0]);
}

std::complex<float> pfaffian_cfloat(py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Matrix must be square");

    cuFloatComplex res = pfaffian<cuFloatComplex>(
        reinterpret_cast<cuFloatComplex*>(buf.ptr),
        buf.shape[0]
    );

    return std::complex<float>(res.x, res.y);
}

std::complex<double> pfaffian_cdouble(py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Matrix must be square");

    cuDoubleComplex res = pfaffian<cuDoubleComplex>(
        reinterpret_cast<cuDoubleComplex*>(buf.ptr),
        buf.shape[0]
    );

    return std::complex<double>(res.x, res.y);
}

template<typename T, typename LogT>
py::tuple slog_wrapper(py::array_t<T> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Matrix must be square");

    LogT log_abs;
    T phase;

    slog_pfaffian<T>(
        (T*)buf.ptr,
        buf.shape[0],
        &log_abs,
        &phase
    );

    return py::make_tuple(log_abs, phase);
}

py::tuple slog_pfaffian_float(py::array_t<float> arr) {
    return slog_wrapper<float, float>(arr);
}

py::tuple slog_pfaffian_double(py::array_t<double> arr) {
    return slog_wrapper<double, double>(arr);
}

py::tuple slog_pfaffian_cfloat(py::array_t<std::complex<float>> arr) {
    auto buf = arr.request();

    float log_abs;
    cuFloatComplex phase;

    slog_pfaffian<cuFloatComplex>(
        reinterpret_cast<cuFloatComplex*>(buf.ptr),
        buf.shape[0],
        &log_abs,
        &phase
    );

    return py::make_tuple(
        log_abs,
        reinterpret_cast<std::complex<float>&>(phase)
    );
}

py::tuple slog_pfaffian_cdouble(py::array_t<std::complex<double>> arr) {
    auto buf = arr.request();

    double log_abs;
    cuDoubleComplex phase;

    slog_pfaffian<cuDoubleComplex>(
        reinterpret_cast<cuDoubleComplex*>(buf.ptr),
        buf.shape[0],
        &log_abs,
        &phase
    );

    return py::make_tuple(
        log_abs,
        reinterpret_cast<std::complex<double>&>(phase)
    );
}

PYBIND11_MODULE(pfcuda, m) {
    m.def("pfaffian_f32", &pfaffian_float);
    m.def("pfaffian_f64", &pfaffian_double);
    m.def("pfaffian_c64", &pfaffian_cfloat);
    m.def("pfaffian_c128", &pfaffian_cdouble);

    m.def("slog_pfaffian_f32", &slog_pfaffian_float);
    m.def("slog_pfaffian_f64", &slog_pfaffian_double);
    m.def("slog_pfaffian_c64", &slog_pfaffian_cfloat);
    m.def("slog_pfaffian_c128", &slog_pfaffian_cdouble);
}
#pragma once
#include "pfaffian_utils.cuh"

inline void matrix_size_error(unsigned int n) {
    throw std::runtime_error(
        "Matrix size (" + std::to_string(n) + "x" + std::to_string(n) + ") is not supported by slog_pfaffian(). " +
        "Use pfaffian() for matrices smaller or equal to 32x32 in size. "
    );
}

// C++
template<typename T>
void slog_pfaffian(const T * h_A, const unsigned int n, typename ProjectionType<T>::type * h_log_abs, T * h_phase);

extern template void slog_pfaffian<float>(const float*, const unsigned int, float*, float*);
extern template void slog_pfaffian<double>(const double*, const unsigned int, double*, double*);
extern template void slog_pfaffian<cuFloatComplex>(const cuFloatComplex*, const unsigned int, float*, cuFloatComplex*);
extern template void slog_pfaffian<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*);

// Jax
template<typename T>
void slog_pfaffian(T * d_A, const unsigned int n, typename ProjectionType<T>::type * d_log_abs, T * d_phase, cudaStream_t stream);

extern template void slog_pfaffian<float>(float*, const unsigned int, float*, float*, cudaStream_t);
extern template void slog_pfaffian<double>(double*, const unsigned int, double*, double*, cudaStream_t);
extern template void slog_pfaffian<cuFloatComplex>(cuFloatComplex*, const unsigned int, float*, cuFloatComplex*, cudaStream_t);
extern template void slog_pfaffian<cuDoubleComplex>(cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*, cudaStream_t);

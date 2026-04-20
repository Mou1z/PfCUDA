#pragma once
#include "pfaffian_utils.cuh"

inline void matrix_size_error(unsigned int n) {
    throw std::runtime_error(
        "Matrix size (" + std::to_string(n) + "x" + std::to_string(n) + ") exceeds the 32x32 limit for pfaffian(). " +
        "Use slog_pfaffian() for larger matrices to avoid numerical overflow. "
    );
}

template<typename T>
void pfaffian_sm(const T * d_A, const unsigned int n, T * d_result, cudaStream_t stream = nullptr);

extern template void pfaffian_sm<float>(const float*, const unsigned int, float*, cudaStream_t);
extern template void pfaffian_sm<double>(const double*, const unsigned int, double*, cudaStream_t);
extern template void pfaffian_sm<cuFloatComplex>(const cuFloatComplex*, const unsigned int, cuFloatComplex*, cudaStream_t);
extern template void pfaffian_sm<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int, cuDoubleComplex*, cudaStream_t);
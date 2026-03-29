#pragma once
#include "pfaffian_utils.cuh"

// C++
template<typename T>
T pfaffian(const T * h_A, const unsigned int n);

extern template float pfaffian<float>(const float*, const unsigned int);
extern template double pfaffian<double>(const double*, const unsigned int);
extern template cuFloatComplex pfaffian<cuFloatComplex>(const cuFloatComplex* , const unsigned int);
extern template cuDoubleComplex pfaffian<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int);

// Jax
template<typename T>
void pfaffian(const T * d_A, const unsigned int n, T * d_result, cudaStream_t stream);

extern template void pfaffian<float>(const float*, const unsigned int, float*, cudaStream_t);
extern template void pfaffian<double>(const double*, const unsigned int, double*, cudaStream_t);
extern template void pfaffian<cuFloatComplex>(const cuFloatComplex* , const unsigned int, cuFloatComplex*, cudaStream_t);
extern template void pfaffian<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int, cuDoubleComplex*, cudaStream_t);

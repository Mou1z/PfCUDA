#pragma once
#include "pfaffian_utils.cuh"

template<typename T>
void slog_pfaffian_lg(T * d_A, const unsigned int n, typename ProjectionType<T>::type * h_log_abs, T * h_phase, cudaStream_t stream = nullptr);

extern template void slog_pfaffian_lg<float>(float*, const unsigned int, float*, float*, cudaStream_t);
extern template void slog_pfaffian_lg<double>(double*, const unsigned int, double*, double*, cudaStream_t);
extern template void slog_pfaffian_lg<cuFloatComplex>(cuFloatComplex*, const unsigned int, float*, cuFloatComplex*, cudaStream_t);
extern template void slog_pfaffian_lg<cuDoubleComplex>(cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*, cudaStream_t);

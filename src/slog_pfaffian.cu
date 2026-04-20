#include "slog_pfaffian.cuh"
#include "slog_pfaffian_lg.cuh"

template<typename T>
void slog_pfaffian(const T * h_A, const unsigned int n, typename ProjectionType<T>::type * h_log_abs, T * h_phase) {
    if(n <= 32) 
        matrix_size_error(n);

    if(n == 0 || n & 1) {
        *h_log_abs = -INFINITY;
        *h_phase = zero<T>();
        return;
    }
    
    T h_result = zero<T>();
    
    T * d_A;
    T * d_phase;
    typename ProjectionType<T>::type * d_log_abs;
    const int bytes =  n * n * sizeof(T);

    cudaMalloc(&d_log_abs, sizeof(typename ProjectionType<T>::type));
    cudaMalloc(&d_phase, sizeof(T));

    cudaMalloc(&d_A, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    slog_pfaffian_lg<T>(d_A, n, d_log_abs, d_phase);

    cudaMemcpy(h_log_abs, d_log_abs, sizeof(typename ProjectionType<T>::type), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phase, d_phase, sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void slog_pfaffian(T * d_A, const unsigned int n, typename ProjectionType<T>::type * d_log_abs, T * d_phase, cudaStream_t stream) {
    if(n <= 32) 
        matrix_size_error(n);

    slog_pfaffian_lg<T>(d_A, n, d_log_abs, d_phase, stream);
}

// C++
template void slog_pfaffian<float>(const float*, const unsigned int, float*, float*);
template void slog_pfaffian<double>(const double*, const unsigned int, double*, double*);
template void slog_pfaffian<cuFloatComplex>(const cuFloatComplex*, const unsigned int, float*, cuFloatComplex*);
template void slog_pfaffian<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*);

// Jax
template void slog_pfaffian<float>(float*, const unsigned int, float*, float*, cudaStream_t);
template void slog_pfaffian<double>(double*, const unsigned int, double*, double*, cudaStream_t);
template void slog_pfaffian<cuFloatComplex>(cuFloatComplex*, const unsigned int, float*, cuFloatComplex*, cudaStream_t);
template void slog_pfaffian<cuDoubleComplex>(cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*, cudaStream_t);

#include "pfaffian.cuh"
#include "pfaffian_sm.cuh"

template<typename T>
T pfaffian(const T * h_A, const unsigned int n) {
    if(n == 0 || n & 1)
        return zero<T>();

    if(n == 2)
        return CM(h_A, n, 0, 1);

    if(n == 4)
        return (
            CM(h_A, n, 0, 1) * CM(h_A, n, 2, 3) -
            CM(h_A, n, 0, 2) * CM(h_A, n, 1, 3) +
            CM(h_A, n, 0, 3) * CM(h_A, n, 1, 2)
        );

    T h_result = zero<T>();

    
    T * d_result;
    cudaMalloc(&d_result, sizeof(T));
    
    T * d_A;
    const int bytes = 
        n * n * sizeof(T);
    cudaMalloc(&d_A, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    if(n <= 32) 
        pfaffian_sm<T>(h_A, n, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    return h_result;
}

template<typename T>
void pfaffian(const T * d_A, const unsigned int n, T * d_result, cudaStream_t stream) {
    if(n <= 32) 
        pfaffian_sm<T>(d_A, n, d_result, stream);
}

// C++
template float pfaffian<float>(const float*, const unsigned int);
template double pfaffian<double>(const double*, const unsigned int);
template cuFloatComplex pfaffian<cuFloatComplex>(const cuFloatComplex* , const unsigned int);
template cuDoubleComplex pfaffian<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int);

// Jax
template void pfaffian<float>(const float*, const unsigned int, float*, cudaStream_t);
template void pfaffian<double>(const double*, const unsigned int, double*, cudaStream_t);
template void pfaffian<cuFloatComplex>(const cuFloatComplex* , const unsigned int, cuFloatComplex*, cudaStream_t);
template void pfaffian<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int, cuDoubleComplex*, cudaStream_t);
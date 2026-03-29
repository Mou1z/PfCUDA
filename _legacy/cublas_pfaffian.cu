#include "pfaffian.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

cublasHandle_t getCublasHandle() {
    static cublasHandle_t handle = nullptr;
    static bool initialized = false;

    if (!initialized) {
        cublasStatus_t status = cublasCreate(&handle);
        if(status != CUBLAS_STATUS_SUCCESS) 
            throw std::runtime_error("Failed to create the CUBLAS handle");
        initialized = true;
    }
    
    return handle;
}

__global__ void applyRowUpdate(
    double * __restrict__ A,
    int n, int k
) {
    int i = (k / 2);
    i *= 2;


    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int m = k + 1 + j;

    if(m >= n) return;
    if(k < 2) return;
    
    double sum = 0.0;
    for(int l = 0; l < i; l += 2) {
        sum +=
            (
                (
                    CM(A, n, l + 1, m) *
                    CM(A, n, l, k) 
                )
                -
                (
                    CM(A, n, l, m) *
                    CM(A, n, l + 1, k)
                )
            );
    }

    CM(A, n, k, m) += sum;
}

__global__ void applyUpdates(
    double * __restrict__ A,
    int n, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + 2 + i;
    if(j >= n) return;

    CM(A, n, k, j) /= CM(A, n, k, k + 1);
    CM(A, n, k + 1, j) *= -1;
}

template<typename T>
__device__ __host__ T gabs(T x) {
    return x < 0 ? -x : x;
}

template<typename T>
struct MaxPair {
    int index;
    T value;

    __host__ __device__ MaxPair(int i, T v) : index(i), value(v) {}
};

template<typename T>
__global__ void argmax(
    const T* __restrict__ A, 
    MaxPair<T>* __restrict__ blockResults, 
    const int n, int k, int elements
) {
    extern __shared__ MaxPair<T> section[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    MaxPair<double> current(-1, 0);

    if(i < (elements)) {
        current.index = k + 1 + i;
        current.value = CM(A, n, k, current.index);
    }

    section[tid] = current;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            if(gabs(section[tid].value) < gabs(section[tid + s].value)) {
                section[tid] = section[tid + s];
            }
        }
        __syncthreads();
    }

    if(tid == 0) {
        blockResults[blockIdx.x] = section[0];
    }
}

int getMinBlocks(int tpb, int elements) {
    return (elements + (tpb - 1)) / tpb;
}

// void debug(int k, int n, double * _A, const double * A) {
//     cudaMemcpy(_A, A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
//     std::cout << "k = " << k << std::endl;
//     for(int i = 0; i < n; i++) {
//         for(int j = 0; j < n; j++) {
//             std::cout << std::ceil(CM(_A, n, i, j) * 1000.0) / 1000.0 << "\t";
//         }
//         std::cout << "\n";
//     }
//     std::cout << "\n";
// }

double pfaffian(cublasHandle_t handle, const double * _A, const long n) {
    double* A;
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    
    int tpb = 256;

    MaxPair<double> pivot(-1, -1);
    MaxPair<double>* d_blockResults;
    int sharedBytes = tpb * sizeof(MaxPair<double>);
    cudaMalloc(&d_blockResults, getMinBlocks(tpb, n - 1) * sizeof(MaxPair<double>));
    
    double result = 1.0;

    int blocks;
    int currElements;

    for(int k = 0, i = 0; k < n - 1; k += 2, i++) {
        currElements = n - k - 1;

        blocks = getMinBlocks(tpb, currElements);

        applyRowUpdate<<<blocks, tpb>>>(A, n, k);
        
        argmax<<<blocks, tpb, sharedBytes>>>(A, d_blockResults, n, k, currElements);
        if(blocks > 1)
            argmax<<<1, tpb, sharedBytes>>>(A, d_blockResults, n, k, blocks);
        cudaMemcpy(&pivot, &d_blockResults[0], sizeof(MaxPair<double>), cudaMemcpyDeviceToHost);

        result *= pivot.value;
        if(pivot.index != k + 1) {
            cublasDswap(handle, n, A + (k + 1) * n, 1, A + pivot.index * n, 1);
            cublasDswap(handle, n, A + (k + 1), n, A + pivot.index, n);
            result *= -1;
        }

        blocks = getMinBlocks(tpb, currElements - 1);
        applyRowUpdate<<<blocks, tpb>>>(A, n, k + 1);
        applyUpdates<<<blocks, tpb>>>(A, n, k);
    }

    cudaDeviceSynchronize();
    cudaFree(d_blockResults);
    cudaFree(A);
    return result;
}

// double A[100] = {
//      0, -1, -2, -3, -4, -5, -6, -7, -8, -9,
//      1,  0, -11, -12, -13, -14, -15, -16, -17, -18,
//      2, 11,  0, -23, -24, -25, -26, -27, -28, -29,
//      3, 12, 23,  0, -34, -35, -36, -37, -38, -39,
//      4, 13, 24, 34,  0, -45, -46, -47, -48, -49,
//      5, 14, 25, 35, 45,  0, -56, -57, -58, -59,
//      6, 15, 26, 36, 46, 56,  0, -67, -68, -69,
//      7, 16, 27, 37, 47, 57, 67,  0, -78, -79,
//      8, 17, 28, 38, 48, 58, 68, 78,  0, -89,
//      9, 18, 29, 39, 49, 59, 69, 79, 89,  0
// };

// int main() {
//     cublasHandle_t handle = getCublasHandle();

//     double result = pfaffian(handle, A, 10);
//     std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
//           << ">> Result: " << result << std::endl;

//     return 0;
// }
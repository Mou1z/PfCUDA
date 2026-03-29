#include "pfaffian.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CM(A, P, ld, i, j) ((A)[(P[i]) + (P[j]) * (ld)])

template<typename T>
struct MaxPair {
    int index;
    T value;

    __host__ __device__ MaxPair(int i, T v) : index(i), value(v) {}
};

template<typename T>
__device__ __host__ T gabs(T x) {
    return x < 0 ? -x : x;
}

int getMinBlocks(int tpb, int elements) {
    return (elements + (tpb - 1)) / tpb;
}

__global__ void applyRowUpdate(
    double * __restrict__ A,
    int * P,
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
                    CM(A, P, n, l + 1, m) *
                    CM(A, P, n, l, k) 
                )
                -
                (
                    CM(A, P, n, l, m) *
                    CM(A, P, n, l + 1, k)
                )
            );
    }

    CM(A, P, n, k, m) += sum;
}

__global__ void applyUpdates(
    double * __restrict__ A,
    int * P,
    int n, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + 2 + i;
    if(j >= n) return;

    CM(A, P, n, k, j) /= CM(A, P, n, k, k + 1);
    CM(A, P, n, k + 1, j) *= -1;
}

template<typename T>
__global__ void pivot(
    const T* __restrict__ A, int * P, double * result,
    MaxPair<T>* __restrict__ blockResults, 
    const int n, int k, int elements
) {
    extern __shared__ MaxPair<T> section[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    MaxPair<double> current(-1, 0);

    if(i < (elements)) {
        current.index = k + 1 + i;
        current.value = gabs(CM(A, P, n, k, current.index));
    }

    section[tid] = current;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            if((section[tid].index > 0) && (section[tid].value < section[tid + s].value)) {
                section[tid] = section[tid + s];
            }
        }
        __syncthreads();
    }

    if(tid == 0) {
        blockResults[blockIdx.x] = section[0];

        if(gridDim.x == 1) {
            int kp = blockResults[0].index;

            int tmp = P[k + 1];
            P[k + 1] = P[kp];
            P[kp] = tmp;

            (*result) = (*result) * CM(A, P, n, k, k + 1) * -1.0;
        }
    }
}

// void debug(int k, int n, double * _A, const double * A, int * P) {
//     cudaMemcpy(_A, A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
//     std::cout << "k = " << k << std::endl;
//     for(int i = 0; i < n; i++) {
//         for(int j = 0; j < n; j++) {
//             std::cout << std::ceil(CM(_A, P, n, i, j) * 1000.0) / 1000.0 << "\t";
//         }
//         std::cout << "\n";
//     }
//     std::cout << "\n";
// }

__global__ void init_permutation_vector(int * P, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) P[i] = i;
}

double pfaffian(const double * _A, const long n) {
    double* A;
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    
    double result = 1.0;

    double * d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice);

    int tpb = 256;
    int blocks, currElements;

    MaxPair<double>* d_blockResults;
    int sharedBytes = tpb * sizeof(MaxPair<double>);
    cudaMalloc(&d_blockResults, getMinBlocks(tpb, n - 1) * sizeof(MaxPair<double>));
    
    int* d_P;
    blocks = getMinBlocks(tpb, n);
    cudaMalloc(&d_P, n * sizeof(int));
    init_permutation_vector<<<blocks, tpb>>>(d_P, n);

    cudaDeviceSynchronize();

    for(int k = 0, i = 0; k < n - 1; k += 2, i++) {
        currElements = n - k - 1;

        blocks = getMinBlocks(tpb, currElements);
        applyRowUpdate<<<blocks, tpb>>>(A, d_P, n, k);
        
        pivot<<<blocks, tpb, sharedBytes>>>(A, d_P, d_result, d_blockResults, n, k, currElements);
        if(blocks > 1)
            pivot<<<1, tpb, sharedBytes>>>(A, d_P, d_result, d_blockResults, n, k, blocks);

        blocks = getMinBlocks(tpb, currElements - 1);
        applyRowUpdate<<<blocks, tpb>>>(A, d_P, n, k + 1);
        applyUpdates<<<blocks, tpb>>>(A, d_P, n, k);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_blockResults);
    cudaFree(d_result);
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
//     double result = pfaffian(A, 10);
//     std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
//           << ">> Result: " << result << std::endl;
//     result = pfaffian(A, 10);
//     std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
//           << ">> Result: " << result << std::endl;
//     return 0;
// }
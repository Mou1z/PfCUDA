#include "pfaffian.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

int idx(int w, int x, int y) {
    if((x >> 2) > y) 
        return y * (2 * w - y - 1) + x - 1;
    return 0;
}

int getMinimumBlocks(int elements, int threadsPerBlock) {
    return (elements + threadsPerBlock - 1) / threadsPerBlock;
}

__global__ void updateV(
    const double* __restrict__ A,
    const double* __restrict__ R,
    double* V,
    int k,
    int i,
    int n
) {

    int rawIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int index = k + 2 + rawIndex;

    double k1 = CM(A, n, k, k + 1) + R[0];

    if (index < n) {
        CM(V, n, index, i) = (CM(A, n, k, index) + R[rawIndex + 1]) / k1;
    }
}

__global__ void updateW(
    const double* __restrict__ A,
    const double* __restrict__ R,
    double* W,
    int k,
    int i,
    int n
) {
    int rawIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int index = k + 2 + rawIndex;

    if(index < n) {
        CM(W, n, index, i) = - CM(A, n, k + 1, index) - R[rawIndex];
    }
}

double pfaffian(const double * _A, const long n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double* A;
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    
    long blockSize = (n - 2) / 2;
    
    double *V, *W, *R;
    cudaMalloc(&V, n * blockSize * sizeof(double));
    cudaMalloc(&W, n * blockSize * sizeof(double));
    cudaMalloc(&R, n * sizeof(double));

    double 
        a0 =  1.0, b0 = 0.0,
        a1 = -1.0, b1 = 1.0;

    int threadsPerBlock = 256, blocks, cols;

    for(int k = 0, i = 0; k < n - 2; k += 2, i++) {
        cols = n - k - 1;

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, cols, i, &a0, V + k, n, W + k + 1, n, &b0, R, 1);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, cols, i, &a1, W + k, n, V + k + 1, n, &b1, R, 1);

        blocks = getMinimumBlocks(cols - 1, threadsPerBlock);
        updateV<<<blocks, threadsPerBlock>>>(A, R, V, k, i, n);
        updateW<<<blocks, threadsPerBlock>>>(A, R, W, k, i, n);
    }

    double result = 0.0;

    return result;
}
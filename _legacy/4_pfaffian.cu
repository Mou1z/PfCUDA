#include "pfaffian.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

int getMinimumBlocks(int elements, int threadsPerBlock) {
    return (elements + threadsPerBlock - 1) / threadsPerBlock;
}

__global__ void reduceProduct(double* data, int n) {
    extern __shared__ double section[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    section[tid] = (i < n) ? data[i] : 1.0;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            section[tid] *= section[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0)
        data[blockIdx.x] = section[0];
}

__global__ void swap_rows(double * __restrict__ A, int n, int i, int j) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= n) return;

    double tmp;
    tmp = CM(A, n, i, k);
    CM(A, n, i, k) = CM(A, n, j, k);
    CM(A, n, j, k) = tmp;
}

__global__ void swap_cols(double * __restrict__ A, int n, int i, int j) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= n) return;

    double tmp;
    tmp = CM(A, n, k, i);
    CM(A, n, k, i) = CM(A, n, k, j);
    CM(A, n, k, j) = tmp;
}

void debug(int k, int n, double * _A, const double * A) {
    cudaMemcpy(_A, A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "k = " << k << std::endl;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << std::ceil(CM(_A, n, i, j) * 1000.0) / 1000.0 << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";
}

double pfaffian(double * _A, const long n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double * A;
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);

    double * R;
    double * hR;
    cudaMalloc(&R, n * sizeof(double));

    double a1 = 1.0, a2 = -1.0, b1 = 0.0, b2 = 1.0;

    int elements;

    double * V = A;
    double * W = A + n;
    int s = n * 2;

    for(int k = 0, i = 0; k < n - 1; k += 2, i++) {
        debug(k, n, _A, A);
        
        elements = n - k;

        if(k > 1) {
            // A + k*n + k + 1
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, elements, i, &a1, V + k, s, W + (k + 1), s, &b1, R, 1);
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, elements, i, &a2, W + k, s, V + (k + 1), s, &b2, R, 1);

            std::cout << "k = " << k << "\n";
            for(int j = 0; j < n; j++) {
                std::cout << hR[j] << " ";
            }
            std::cout << "\n\n";

            cudaMemcpy(hR, R, n, cudaMemcpyDeviceToHost);
        }

        cudaDeviceSynchronize(); 
    }

    return 0;
}

double A[100] = {
     0, -1, -2, -3, -4, -5, -6, -7, -8, -9,
     1,  0, -11, -12, -13, -14, -15, -16, -17, -18,
     2, 11,  0, -23, -24, -25, -26, -27, -28, -29,
     3, 12, 23,  0, -34, -35, -36, -37, -38, -39,
     4, 13, 24, 34,  0, -45, -46, -47, -48, -49,
     5, 14, 25, 35, 45,  0, -56, -57, -58, -59,
     6, 15, 26, 36, 46, 56,  0, -67, -68, -69,
     7, 16, 27, 37, 47, 57, 67,  0, -78, -79,
     8, 17, 28, 38, 48, 58, 68, 78,  0, -89,
     9, 18, 29, 39, 49, 59, 69, 79, 89,  0
};

int main() {
    double result = pfaffian(A, 10);
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
          << ">> Result: " << result << std::endl;
    return 0;
}
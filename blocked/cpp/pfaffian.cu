#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

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
    extern __shared__ double shared[];
    double& k1 = shared[0];

    int index = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

    if (index == k + 1) {
        k1 = CM(A, n, k, k + 1);
    }

    __syncthreads();

    if (index < n) {
        CM(V, n, index, i) = CM(A, n, k, index) / k1;
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
    int index = k + 2 + blockIdx * blockDim.x + threadIdx.x;

    if(index < n) {
        CM(W, n, index, i) = CM(A, n, index, k + 1)
    }
}

double pfaffian(const double * _A, const long n) {
    double* A;
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    long blockSize = (n - 2) / 2;

    double *V, *W, *R;

    cudaMalloc(&V, n * blockSize * sizeof(double));
    cudaMalloc(&W, n * blockSize * sizeof(double));
    cudaMalloc(&R, n * 2 * sizeof(double));

    int i;
    double alpha, beta;

    int threadsPerBlock = 256, blocks;

    for(int k = 0; k < n - 2; k += 2) {
        i = k / 2;

        blocks = getMinimumBlocks(n - k + 1, threadsPerBlock);
        updateV<<<blocks, threadsPerBlock, sizeof(double)>>>(A, R, V, k, i, n);

        blocks = getMinimumBlocks(n - k + 2, threadsPerBlock);
        updateW<<<blocks, threadsPerBlock>>>(A, R, W, k, i, n);
    }

    std::vector<double> V_host(n * blockSize);
    cudaMemcpy(V_host.data(), V, n * blockSize * sizeof(double),
            cudaMemcpyDeviceToHost);

    std::vector<double> W_host(n * blockSize);
    cudaMemcpy(W_host.data(), W, n * blockSize * sizeof(double),
            cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < blockSize; j++) {
            std::cout << CM(W_host, n, i, j) << "\t";
        }
        std::cout << "\n" << std::endl;
    }

    return 0;
}

double A[100] = {
     0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,
     1,   0, -11, -12, -13, -14, -15, -16, -17, -18,
     2,  11,   0, -23, -24, -25, -26, -27, -28, -29,
     3,  12,  23,   0, -34, -35, -36, -37, -38, -39,
     4,  13,  24,  34,   0, -45, -46, -47, -48, -49,
     5,  14,  25,  35,  45,   0, -56, -57, -58, -59,
     6,  15,  26,  36,  46,  56,   0, -67, -68, -69,
     7,  16,  27,  37,  47,  57,  67,   0, -78, -79,
     8,  17,  28,  38,  48,  58,  68,  78,   0, -89,
     9,  18,  29,  39,  49,  59,  69,  79,  89,   0
};

int main(void) {
    pfaffian(A, 10);
    return 0;
}
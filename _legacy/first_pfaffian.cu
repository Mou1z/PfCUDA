#include "pfaffian.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

constexpr double EPS = 1e-12;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

struct RowSlice {
    const double * data;
    int x;
    int y;
    int n;

    __host__ __device__
    double operator()(int i) const {
        return fabs(data[x + (y + i) * n]);
    }
};

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
    int n, int k, double pivot
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + 2 + i;
    if(j >= n) return;

    CM(A, n, k, k + 2 + i) /= pivot;
    CM(A, n, k + 1, k + 2 + i) *= -1;
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

double pfaffian(const double * _A, const long n) {
    double* A;
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    
    double result = 1.0;
    int blocks;
    
    int max_idx;
    double pivot;
    double curr_pivot;

    int* d_max_idx;
    double* d_max_val;

    cudaMalloc(&d_max_idx, sizeof(int));
    cudaMalloc(&d_max_val, sizeof(double));

    void * kp_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    RowSlice indexer { A, 0, 1, n };

    cub::DeviceReduce::ArgMax(
        kp_temp_storage,
        temp_storage_bytes,
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), indexer),
        d_max_val, d_max_idx, n - 1
    );
    cudaMalloc(&kp_temp_storage, temp_storage_bytes);

    for(int k = 0, i = 0; k < n - 1; k += 2, i++) {
        blocks = (n - k + 255) / 256;
        applyRowUpdate<<<blocks, 256>>>(A, n, k);

        indexer.x = k;
        indexer.y = k + 1;

        cub::DeviceReduce::ArgMax(
            kp_temp_storage,
            temp_storage_bytes,
            thrust::make_transform_iterator(thrust::make_counting_iterator(0), indexer),
            d_max_val, d_max_idx, n - (k + 1)
        );

        cudaMemcpy(&max_idx, d_max_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pivot, &CM(A, n, k, k + 1 + max_idx), sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&curr_pivot, &CM(A, n, k, k + 1), sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        bool first_is_max = fabs(curr_pivot - fabs(pivot)) <= EPS;
        if(max_idx != 0 && !first_is_max) {
            int x, y;
            x = k + 1;
            y = x + max_idx;

            blocks = (n + 255) / 256;

            swap_rows<<<blocks, 256>>>(A, n, x, y);
            swap_cols<<<blocks, 256>>>(A, n, x, y);

            result *= -1;
        }

        blocks = ((n - (k + 1)) + 255) / 256;
        applyRowUpdate<<<blocks, 256>>>(A, n, k + 1);

        blocks = ((n - k + 1) + 255) / 256;
        applyUpdates<<<blocks, 256>>>(A, n, k, pivot);

        result *= pivot;
    }

    CUDA_CHECK(cudaFree(A));
    return result;
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
          << "Result: " << result << std::endl;
    return 0;
}
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(err) if(err != cudaSuccess){ \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); }

#define CUBLAS_CHECK(err) if(err != CUBLAS_STATUS_SUCCESS){ \
    printf("CUBLAS error\n"); exit(1); }


// Column-major indexing
#define IDX(i,j,n) ((j)*(n) + (i))

// Kernel for fused skew-symmetric rank-2 update

// Fused skew-symmetric rank-2 update kernel
__global__ void skew_rank2_update(
    double* A22,      // submatrix A[k+2:, k+2:], size m x m
    const double* ak, // row vector ak = A[k, k+2:] / pivot, length m
    const double* Acol, // column vector A[k+2:, k+1], length m
    int m, int lda)   // lda = leading dimension of full matrix
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column

    if(i < m && j < m)
    {
        int idx = i + j * lda; // column-major index in full matrix
        A22[idx] += ak[j] * Acol[i] - Acol[j] * ak[i];
    }
}

double pfaffian(const double* h_A, const long n)
{
    if (n % 2 != 0) return 0.0;

    double result = 1.0;

    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n*n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, n*n*sizeof(double),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for(int k = 0; k < n-1; k += 2)
    {
        int col = k;
        int pivot_index;

        // ---- Pivot search ----
        CUBLAS_CHECK(
            cublasIdamax(handle,
                         n-k-1,
                         d_A + IDX(k+1,col,n),
                         1,
                         &pivot_index));

        pivot_index += k; // adjust 1-based index from cublasIdamax

        // ---- Row/Column swap if needed ----
        if(pivot_index != k+1)
        {
            CUBLAS_CHECK(
                cublasDswap(handle,
                            n,
                            d_A + IDX(k+1,0,n), n,
                            d_A + IDX(pivot_index,0,n), n));

            CUBLAS_CHECK(
                cublasDswap(handle,
                            n,
                            d_A + IDX(0,k+1,n), 1,
                            d_A + IDX(0,pivot_index,n), 1));

            result *= -1.0;
        }

        // ---- Get pivot value ----
        double pivot;
        CUDA_CHECK(cudaMemcpy(&pivot,
                              d_A + IDX(k,k+1,n),
                              sizeof(double),
                              cudaMemcpyDeviceToHost));

        if(pivot == 0.0)
        {
            result = 0.0;
            break;
        }

        result *= pivot;

        // ---- Skew-symmetric rank-2 update using fused kernel ----
        if(k+2 < n)
        {
            int m = n - (k+2);

            double alpha = 1.0 / pivot;

            // ak = A[k, k+2:] / pivot
            CUBLAS_CHECK(
                cublasDscal(handle,
                            m,
                            &alpha,
                            d_A + IDX(k, k+2, n),
                            n));

            double* ak_ptr = d_A + IDX(k, k+2, n);     // row vector
            double* Acol_ptr = d_A + IDX(k+2, k+1, n); // column vector
            double* A22_ptr = d_A + IDX(k+2, k+2, n);  // submatrix

            dim3 block(16,16);
            dim3 grid((m + block.x - 1)/block.x, (m + block.y - 1)/block.y);

            skew_rank2_update<<<grid, block>>>(A22_ptr, ak_ptr, Acol_ptr, m, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));

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
          << ">> Result: " << result << std::endl;
    return 0;
}
#include "pfaffian.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

template<typename T>
struct MaxPair {
    int index;
    T value;

    __host__ __device__ MaxPair() : index(0), value(T(0)) {}
    __host__ __device__ MaxPair(int i, T v) : index(i), value(v) {}
};

template<typename T>
__device__ __host__ T gabs(T x) {
    return x < 0 ? -x : x;
}

int getMinBlocks(int tpb, int elements) {
    return (elements + (tpb - 1)) / tpb;
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
    std::cout << "\n";
}

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

__global__ void rowUpdate(
    double * __restrict__ A,
    const int n, const int k,
    const int rows, const int cols
) {
    if(k < 2) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int c = k + 1 + (blockIdx.y * 32 + ty);
    const double * __restrict__ c_k = &CM(A, n, 0, k);
    const double * __restrict__ c_c = &CM(A, n, 0, c);

    double value = 0.0;
    for(int r = (tx * 2); r < rows - 1; r += 64) {
        const double ckr0 = c_k[r];
        const double ckr1 = c_k[r + 1];

        const double ccr0 = c_c[r];
        const double ccr1 = c_c[r + 1];

        const double a = ckr0 * ccr1;
        const double b = ckr1 * ccr0;
        value += (a - b);
    }

    for(int s = 16; s > 0; s >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, s);
    }

    if(tx == 0 && c < n) {
        CM(A, n, k, c) += value;
    }
}

template<typename T>
__global__ void pfaffian_sm_calc(const T * _A, const int n, T * d_result) {
    extern __shared__ T A[];

    __shared__ int kp;
    __shared__ int P[32];

    __shared__ T result;

    const int r = threadIdx.y;
    const int c = threadIdx.x;

    if(r < n && c < n) {
        CM(A, n, r, c) = CM(_A, n, r, c);

        if(r == 0) {
            P[c] = c;

            if(c == 0)
                result = T(1.0);
        }
    }
    __syncthreads();

    for(int k = 0; k < n - 1; k += 2) {
        if(r == 0) {
            int i = c;
            T v = 
                (k < c && c < n) ? 
                    gabs(CM(A, n, P[k], P[c])) :
                    0.0;
            
            unsigned mask = 0xffffffff;

            #pragma unroll
            for(int s = 16; s > 0; s >>= 1) {
                int _i = __shfl_down_sync(mask, i, s);
                T   _v = __shfl_down_sync(mask, v, s);

                if(_v > v) {
                    i = _i; 
                    v = _v;
                }
            }

            if(c == 0)
                kp = i;
        }
        __syncthreads();

        const T pivot = CM(A, n, P[k], P[kp]);
        const T scale = T(1.0) / pivot;

        if(r == 0 && c == 0) {
            if(kp != k + 1) {
                const int tmp = P[k + 1];
                P[k + 1] = P[kp];
                P[kp] = tmp;

                result *= T(-1.0);
            }

            result *= pivot;
        }
        __syncthreads();

        if(k + 2 < n) {
            T update_value = T(0.0);

            if(r > k + 1 && c > k + 1) {
                const T A_kc = CM(A, n, P[k], P[c]);
                const T A_rk1 = CM(A, n, P[r], P[k + 1]);

                const T A_kr = CM(A, n, P[k], P[r]);
                const T A_ck1 = CM(A, n, P[c], P[k + 1]);

                update_value += (A_kr * A_ck1 - A_kc * A_rk1) * scale;
            }
            __syncthreads();

            CM(A, n, P[r], P[c]) += update_value;
        }

        __syncthreads();
    }

    if(r == 0 && c == 0)
        *d_result = result;
}

template<typename T>
double pfaffian_sm(const T * h_A, const int n) {
    T * d_A;

    const int TOTAL_BYTES = n * n * sizeof(T);
    
    cudaMalloc(&d_A, TOTAL_BYTES);
    cudaMemcpy(d_A, h_A, TOTAL_BYTES, cudaMemcpyHostToDevice);

    T h_result;
    T * d_result;
    cudaMalloc(&d_result, sizeof(T));

    pfaffian_sm_calc<T><<<dim3(1, 1), dim3(32, 32), TOTAL_BYTES>>>(d_A, n, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    return h_result;
}

double pfaffian_md_to_lg(const double * _A, const long n) {
    double * A; 
    cudaMalloc(&A, n * n * sizeof(double));
    cudaMemcpy(A, _A, n * n * sizeof(double), cudaMemcpyHostToDevice);

    int tpb = 256;

    MaxPair<double> pivot(-1, -1);
    MaxPair<double>* d_blockResults;
    int sharedBytes = tpb * sizeof(MaxPair<double>);
    cudaMalloc(&d_blockResults, getMinBlocks(tpb, n - 1) * sizeof(MaxPair<double>));
    
    double result = 1.0;

    int blocks;
    dim3 block(32, 32);

    int rows, cols;

    for(int k = 0; k < n - 1; k += 2) {
        rows = k & ~1;
        cols = n - k - 1;

        blocks = getMinBlocks(32, cols);
        rowUpdate<<<dim3(1, blocks), block>>>(A, n, k, rows, cols);
        
        blocks = getMinBlocks(tpb, cols);
        argmax<<<blocks, tpb, sharedBytes>>>(A, d_blockResults, n, k, cols);
        if(blocks > 1)
            argmax<<<1, tpb, sharedBytes>>>(A, d_blockResults, n, k, blocks);
        cudaMemcpy(&pivot, &d_blockResults[0], sizeof(MaxPair<double>), cudaMemcpyDeviceToHost);

        result *= pivot.value;
        if(pivot.index != k + 1) {
            blocks = getMinBlocks(tpb, n);

            swap_rows<<<blocks, tpb>>>(A, n, k + 1, pivot.index);
            swap_cols<<<blocks, tpb>>>(A, n, k + 1, pivot.index);

            result *= -1;
        }

        blocks = getMinBlocks(32, cols - 1);
        rowUpdate<<<dim3(1, blocks), block>>>(A, n, k + 1, rows, cols - 1);

        blocks = getMinBlocks(tpb, cols - 1);
        applyUpdates<<<blocks, tpb>>>(A, n, k);
    }

    cudaDeviceSynchronize();
    cudaFree(d_blockResults);
    cudaFree(A);
    return result;
}

double pfaffian(const double * input_A, const long n) {
    if(n <= 32) {
        return pfaffian_sm<double>(input_A, n);
    } else {
        return pfaffian_md_to_lg(input_A, n);
    }
}

double gA[100] = {
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
    double result = pfaffian(gA, 10);
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
          << ">> Result: " << result << std::endl;
    return 0;
}
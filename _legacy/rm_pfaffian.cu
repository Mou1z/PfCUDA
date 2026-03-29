#include "pfaffian.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define RM(A, ld, i, j) ((A)[(i * ld) + j])

template<typename T>
struct MaxPair {
    int index;
    T value;

    __host__
    __device__
    MaxPair(int i, T v) : index(i), value(v) {}
};

template<typename T>
__host__
__device__
T gabs(T x) { return x < 0 ? -x : x; }

int min_blocks(int threads_per_block, int elements) {
    return (elements + (threads_per_block - 1)) / threads_per_block;
}

template<typename T>
__global__
void row_update(
    T * __restrict__ A,
    const int n, const int k,
    const int rows, const int cols
) {
    if(k < 2) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int c = k + 1 + (blockIdx.y * 32 + ty);

    T value(0);
    for(int r = (tx * 2); r < rows - 1; r += 64) {
        const T * __restrict__ r0 = &RM(A, n, r, 0);
        const T * __restrict__ r1 = &RM(A, n, r + 1, 0);

        const T a = r1[c] * r0[k];
        const T b = r0[c] * r1[k];

        value += (a - b);
    }

    for(int s = 16; s > 0; s >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, s);
    }

    if(tx == 0 && m < n) {
        RM(A, n, k, c) += value;
    }
}

template<typename T>
T pfaffian(const T * input_A, const int n) {
    T * A;
    
    const long total_bytes = n * n * sizeof(T);
    cudaMalloc(&A, total_bytes);
    cudaMemcpy(A, input_A, total_bytes, cudaMemcpyHostToDevice);

    MaxPair<T> pivot(0, 0);
    MaxPair<T> * d_max_pairs;
    
    const int THREADS_PER_BLOCK = 256;
    const int SHARED_BYTES = THREADS_PER_BLOCK * sizeof(MaxPair<T>);
    
    cudaMalloc(&d_max_pairs, min_blocks(THREADS_PER_BLOCK, n - 1) * sizeof(MaxPair<T>));

    int blocks, rows, cols;
    dim3 row_update_block(32, 32);

    double result = 1.0;

    for(int k = 0; k < n - 1; k += 2) {
        rows = k & ~1;
        cols = n - (k + 1);

        blocks = min_blocks(32, cols);
        update_row<<<dim3(1, blocks), row_update_block>>>(A, n, k, rows, cols);

        blocks = min_blocks(THREADS_PER_BLOCK, cols);
        argmax<<<blocks, THREADS_PER_BLOCK, SHARED_BYTES>>>(A, d_max_pairs, n, k, cols);

        if(blocks > 1)
            argmax<<<1, THREADS_PER_BLOCK, SHARED_BYTES>>>(A, d_max_pairs, n, k, blocks);

        cudaMemcpy(&pivot, &d_max_pairs[0], sizeof(MaxPair<double>), cudaMemcpyDeviceToHost);

        result *= pivot.value;
        if(pivot.index != k + 1) {
            blocks = min_blocks(THREADS_PER_BLOCK, n);

            swap_rows<<<blocks, THREADS_PER_BLOCK>>>(A, n, k + 1, pivot.index);
            swap_cols<<<blocks, THREADS_PER_BLOCK>>>(A, n, k + 1, pivot.index);

            result *= -1.0;
        }

        blocks = min_blocks(32, cols - 1);
        update_row<<<dim3(1, blocks), row_update_block>>>(A, n, k + 1, rows, cols - 1);

        blocks = getMinBlocks(THREADS_PER_BLOCK, cols - 1);
        transform_rows<<<block, THREADS_PER_BLOCK>>>(A, n, k);
    }

    cudaDeviceSynchronize();
    cudaFree(d_max_pairs);
    cudaFree(A);
    return result;
}
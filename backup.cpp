#include "slog_pfaffian_lg.cuh"

#define ARGMAX_BLOCK_SIZE 256

template<typename T>
struct MaxPair {
    int index;
    T value;

    __host__ __device__ MaxPair() : index(0), value(zero<T>()) {}
    __host__ __device__ MaxPair(int i, T v) : index(i), value(v) {}
};

inline int min_blocks(int tpb, int elements) {
    return (elements + (tpb - 1)) / tpb;
}

template<typename T>
__global__ void argmax(
    const T* __restrict__ A, 
    MaxPair<T>* __restrict__ block_results, 
    const unsigned int n, 
    const unsigned int k, 
    const unsigned int elements
) {
    extern __shared__ unsigned char shmem[];
    MaxPair<T> * section = reinterpret_cast<MaxPair<T>*>(shmem);

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    MaxPair<T> current(-1, zero<T>());

    if(i < elements) {
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
        block_results[blockIdx.x] = section[0];
    }
}

template<typename T>
__global__ void pivot(
    const T * __restrict__ A,
    const unsigned int n,
    const unsigned int k,
    const unsigned int elements
) {
    extern __shared__ unsigned char shmem[];
    MaxPair<T> * row = reinterpret_cast<MaxPair<T>*>(shmem);

    for(int i = threadIdx.x; i < elements; i += blockDim.x) {
        int index = k + 1 + i;
        row[threadIdx.x] = MaxPair<T>(index, CM(A, n, k, index));
    }

    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(threadIdx.x < s) {
            if(gabs(row[threadIdx.x].value) < gabs(row[threadIdx.x + s].value)) {
                row[threadIdx.x] = row[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        block_results[blockIdx.x] = row[0];
    }
}

template<typename T>
__device__ inline T __shfl_down_sync_ex(T value, int offset) {
    return __shfl_down_sync(0xffffffff, value, offset);
}

template<>
__device__ inline cuFloatComplex __shfl_down_sync_ex(cuFloatComplex value, int offset) {
    float real = __shfl_down_sync(0xffffffff, cuCrealf(value), offset);
    float imag = __shfl_down_sync(0xffffffff, cuCimagf(value), offset);
    return make_cuFloatComplex(real, imag);
}

template<>
__device__ inline cuDoubleComplex __shfl_down_sync_ex(cuDoubleComplex value, int offset) {
    double real = __shfl_down_sync(0xffffffff, cuCreal(value), offset);
    double imag = __shfl_down_sync(0xffffffff, cuCimag(value), offset);
    return make_cuDoubleComplex(real, imag);
}

template<typename T>
__global__ void row_update(
    T * __restrict__ A,
    const unsigned int n, const unsigned int k,
    const unsigned int rows, const unsigned int cols
) {
    if(k < 2) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int c = k + 1 + (blockIdx.y * 32 + ty);

    if(c >= n) return;

    const T * __restrict__ c_k = &CM(A, n, 0, k);
    const T * __restrict__ c_c = &CM(A, n, 0, c);

    T value = zero<T>();
    for(int r = (tx * 2); r < rows - 1; r += 64) {
        const T ckr0 = c_k[r];
        const T ckr1 = c_k[r + 1];

        const T ccr0 = c_c[r];
        const T ccr1 = c_c[r + 1];

        const T a = ckr0 * ccr1;
        const T b = ckr1 * ccr0;
        value += (a - b);
    }

    for(int s = 16; s > 0; s >>= 1) {
        value += __shfl_down_sync_ex<T>(value, s);
    }

    if(tx == 0)
        CM(A, n, k, c) += value;
}

template<typename T>
__global__ void swap_rows(T * __restrict__ A, int n, int i, int j) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= n) return;

    T tmp;
    tmp = CM(A, n, i, k);
    CM(A, n, i, k) = CM(A, n, j, k);
    CM(A, n, j, k) = tmp;
}

template<typename T>
__global__ void swap_cols(T * __restrict__ A, int n, int i, int j) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= n) return;

    T tmp;
    tmp = CM(A, n, k, i);
    CM(A, n, k, i) = CM(A, n, k, j);
    CM(A, n, k, j) = tmp;
}

template<typename T>
__global__ void apply_updates(
    T * __restrict__ A,
    const unsigned int n, 
    const unsigned int k,
    const T scale_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + 2 + i;
    if(j >= n) return;

    CM(A, n, k, j) *= scale_factor;
    CM(A, n, k + 1, j) *= minus_one<T>();
}

template<typename T>
void slog_pfaffian_lg(T * d_A, const unsigned int n, typename ProjectionType<T>::type * d_log_abs, T * d_phase, cudaStream_t stream) {
    const dim3 BLOCK(32, 32);

    typename ProjectionType<T>::type h_log_abs = 0.0;
    T h_phase = one<T>();

    MaxPair<T> h_pivot;
    MaxPair<T> * d_block_results;

    const unsigned int pair_size = sizeof(MaxPair<T>);
    const unsigned int shared_bytes = pair_size * ARGMAX_BLOCK_SIZE;
    cudaMallocAsync(&d_block_results, pair_size * min_blocks(ARGMAX_BLOCK_SIZE, n - 1), stream);

    int blocks, rows, cols;
    for(int k = 0; k < n - 1; k += 2) {
        rows = k & ~1;
        cols = n - k - 1;

        blocks = min_blocks(32, cols);
        row_update<T><<<dim3(1, blocks), BLOCK, 0, stream>>>(d_A, n, k, rows, cols);

        blocks = min_blocks(256, cols);
        argmax<T><<<blocks, 256, shared_bytes, stream>>>(d_A, d_block_results, n, k, cols);
        if(blocks > 1)
            argmax<T><<<1, 256, shared_bytes, stream>>>(d_A, d_block_results, n, k, blocks);
        
        cudaMemcpyAsync(&h_pivot, d_block_results, pair_size, cudaMemcpyDeviceToHost, stream);

        if(gabs(h_pivot.value) == zero<typename ProjectionType<T>::type>()) {
            h_log_abs = -INFINITY;
            h_phase = zero<T>();
            break;
        }

        h_log_abs += log(gabs(h_pivot.value));
        h_phase *= (h_pivot.value / gabs(h_pivot.value));
        if(h_pivot.index != k + 1) {
            blocks = min_blocks(256, n);

            swap_rows<<<blocks, 256, 0, stream>>>(d_A, n, k + 1, h_pivot.index);
            swap_cols<<<blocks, 256, 0, stream>>>(d_A, n, k + 1, h_pivot.index);

            h_phase *= minus_one<T>();
        }

        blocks = min_blocks(32, cols - 1);
        row_update<T><<<dim3(1, blocks), BLOCK, 0, stream>>>(d_A, n, k + 1, rows, cols - 1);

        blocks = min_blocks(256, cols - 1);
        const T scale_factor = one<T>() / h_pivot.value;
        apply_updates<<<blocks, 256, 0, stream>>>(d_A, n, k, scale_factor);
    }

    cudaMemcpyAsync(d_log_abs, &h_log_abs, sizeof(typename ProjectionType<T>::type), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_phase, &h_phase, sizeof(T), cudaMemcpyHostToDevice, stream);

    cudaFreeAsync(d_block_results, stream);
}

template void slog_pfaffian_lg<float>(float*, const unsigned int, float*, float*, cudaStream_t);
template void slog_pfaffian_lg<double>(double*, const unsigned int, double*, double*, cudaStream_t);
template void slog_pfaffian_lg<cuFloatComplex>(cuFloatComplex*, const unsigned int, float*, cuFloatComplex*, cudaStream_t);
template void slog_pfaffian_lg<cuDoubleComplex>(cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*, cudaStream_t);
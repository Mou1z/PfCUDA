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
__global__ void pivot(
    T * __restrict__ A,
    const unsigned int n,
    const unsigned int k,
    const unsigned int cols,
    T * d_phase,
    typename ProjectionType<T>::type * d_log_abs,
    T * d_pivot_value
) {
    extern __shared__ unsigned char shmem[];
    MaxPair<T> * section = reinterpret_cast<MaxPair<T>*>(shmem);
    
    __shared__ MaxPair<T> pivot;

    const unsigned int tid = threadIdx.x;
    
    MaxPair<T> thread_max(-1, minus_one<T>());
    for(unsigned int i = tid; i < cols; i += blockDim.x) {
        int col = k + 1 + i;
        if(col < n) {
            T value = CM(A, n, k, col);
            if(gabs(value) > gabs(thread_max.value)) {
                thread_max.index = col;
                thread_max.value = value;
            }
        }
    }

    section[tid] = thread_max;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            if(gabs(section[tid].value) < gabs(section[tid + s].value)) {
                section[tid] = section[tid + s];
            }
        }
        __syncthreads();
    }
    
    if(tid == 0)
        pivot = section[0];
    __syncthreads();

    if(gabs(pivot.value) == zero<typename ProjectionType<T>::type>()) {
        if(tid == 0) {
            *d_log_abs = -INFINITY;
            *d_phase = zero<T>();   
        }
        return;
    }

    int pivot_col = pivot.index;
    if(pivot_col != k + 1) {
        for(unsigned int i = tid; i < n; i += blockDim.x) {
            T tmp_row = CM(A, n, k, i);
            CM(A, n, k, i) = CM(A, n, pivot_col, i);
            CM(A, n, pivot_col, i) = tmp_row;

            T tmp_col = CM(A, n, i, k);
            CM(A, n, i, k) = CM(A, n, i, pivot_col);
            CM(A, n, i, pivot_col) = tmp_col;
        }

        if(tid == 0) {
            *d_phase *= minus_one<T>();
        }
    }

    if(tid == 0) {
        *d_log_abs += log(gabs(pivot.value));
        *d_phase *= pivot.value / gabs(pivot.value);
        *d_pivot_value = pivot.value;
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
    const T * __restrict__ d_pivot_value
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = k + 2 + i;
    if(j >= n) return;

    CM(A, n, k, j) *= (one<T>() / *d_pivot_value);
    CM(A, n, k + 1, j) *= minus_one<T>();
}

inline unsigned int min_threads(int elements) {
    int threads = 1;
    while(threads < elements)
        threads <<= 1;
    return min(threads, 1024);
}

template<typename T>
void slog_pfaffian_lg(T * d_A, const unsigned int n, typename ProjectionType<T>::type * d_log_abs, T * d_phase, cudaStream_t stream) {
    const dim3 BLOCK(32, 32);

    T h_one = one<T>();
    typename ProjectionType<T>::type h_log_zero = zero<typename ProjectionType<T>::type>();

    cudaMemcpy(d_phase, &h_one, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_abs, &h_log_zero, sizeof(typename ProjectionType<T>::type), cudaMemcpyHostToDevice);

    T * d_pivot_value;
    cudaMalloc(&d_pivot_value, sizeof(T));

    int blocks, rows, cols;
    for(int k = 0; k < n - 1; k += 2) {
        rows = k & ~1;
        cols = n - k - 1;

        blocks = min_blocks(32, cols);
        row_update<T><<<dim3(1, blocks), BLOCK, 0, stream>>>(d_A, n, k, rows, cols);

        unsigned int threads = min_threads(cols);
        size_t shared_bytes = threads * sizeof(MaxPair<T>);
        pivot<T><<<1, threads, shared_bytes, stream>>>(d_A, n, k, cols, d_phase, d_log_abs, d_pivot_value);

        if(cols == 1) break;

        blocks = min_blocks(32, cols - 1);
        row_update<T><<<dim3(1, blocks), BLOCK, 0, stream>>>(d_A, n, k + 1, rows, cols - 1);

        blocks = min_blocks(256, cols - 1);
        apply_updates<<<blocks, 256, 0, stream>>>(d_A, n, k, d_pivot_value);
    }
    
    cudaFree(d_pivot_value);
}

template void slog_pfaffian_lg<float>(float*, const unsigned int, float*, float*, cudaStream_t);
template void slog_pfaffian_lg<double>(double*, const unsigned int, double*, double*, cudaStream_t);
template void slog_pfaffian_lg<cuFloatComplex>(cuFloatComplex*, const unsigned int, float*, cuFloatComplex*, cudaStream_t);
template void slog_pfaffian_lg<cuDoubleComplex>(cuDoubleComplex*, const unsigned int, double*, cuDoubleComplex*, cudaStream_t);
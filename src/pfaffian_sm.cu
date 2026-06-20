#include "pfaffian_sm.cuh"

template<typename T>
__global__ void pf_sm_calc(const T * d_A, const unsigned int n, T * d_result) {
    __shared__ bool abort;

    __shared__ T A[32 * 33];
    
    __shared__ unsigned int P[32];
    __shared__ T scale;
    __shared__ T result;

    const int c = threadIdx.x;

    if(c < n) {
        for(int r = 0; r < n; r++)
            CM(A, n, r, c) = 
                CM(d_A, n, r, c);

        P[c] = c;

        if(c == 0) {
            abort = false;
            result = one<T>();
        }
    }
    __syncwarp();

    for(int k = 0; k < n - 1; k += 2) {
        double v = 
            (k < c && c < n) ? 
            gabs(CM(A, n, P[k], P[c])) : -1.0;
        
        unsigned int i = c;
        unsigned mask = 0xffffffff;

        #pragma unroll
        for(unsigned int s = 16; s > 0; s >>= 1) {
            const int    _i = __shfl_down_sync(mask, i, s);
            const double _v = __shfl_down_sync(mask, v, s);

            if(_v > v) {
                i = _i;
                v = _v;
            }
        }

        if(c == 0) {
            if(i != k + 1) {
                const int tmp = P[k + 1];
                P[k + 1] = P[i];
                P[i] = tmp;

                result *= minus_one<T>();
            }

            const T pivot = CM(A, n, P[k], P[k + 1]);

            if(is_zero<T>(pivot)) {
                abort = true;
            } else {
                scale = one<T>() / pivot;
                result *= pivot;
            }
        }
        __syncwarp();

        if(abort)
            break;

        if(k + 2 < n) {
            for(int r = k + 2; r < n; r++) {
                if(k + 1 < c && c < n) {
                    const T A_kc = CM(A, n, P[k], P[c]);
                    const T A_rk1 = CM(A, n, P[r], P[k + 1]);

                    const T A_kr = CM(A, n, P[k], P[r]);
                    const T A_ck1 = CM(A, n, P[c], P[k + 1]);

                    const T t1 = A_kr * A_ck1;
                    const T t2 = A_kc * A_rk1;

                    const T update = (t1 - t2) * scale;

                    CM(A, n, P[r], P[c]) += update;
                }
                __syncwarp();
            }
        }
        __syncwarp();
    }

    if(c == 0) {
        if(abort)
            *d_result = zero<T>();    
        else
            *d_result = result;
    }
}

template<typename T>
void pfaffian_sm(const T * d_A, const unsigned int n, T * d_result, cudaStream_t stream) {
    size_t shared_bytes = n * n * sizeof(T);
    pf_sm_calc<T><<<1, 32, shared_bytes, stream>>>(d_A, n, d_result);
}

template void pfaffian_sm<float>(const float*, const unsigned int, float*, cudaStream_t);
template void pfaffian_sm<double>(const double*, const unsigned int, double*, cudaStream_t);
template void pfaffian_sm<cuFloatComplex>(const cuFloatComplex*, const unsigned int, cuFloatComplex*, cudaStream_t);
template void pfaffian_sm<cuDoubleComplex>(const cuDoubleComplex*, const unsigned int, cuDoubleComplex*, cudaStream_t);
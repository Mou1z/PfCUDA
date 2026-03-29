__global__ void pfaffian_sm_calc(const double * _A, const int n, double * _result) {
    __shared__ double A[1024];
    
    __shared__ int kp;
    __shared__ int P[32];
    
    __shared__ double result;
    
    const int r = threadIdx.y;
    const int c = threadIdx.x;

    if(r == 0 && c == 0)
        result = 1.0;

    if(r == 0 && c < n)
        P[c] = c;

    CM(A, n, r, c) = (c < n && r < n) ? CM(_A, n, r, c) : 0.0;
    __syncthreads();

    for(int k = 0; k < n - 1; k += 2) {
        if(r == 0) { 
            double value = (k < c && c < n) ? gabs(CM(A, n, P[k], P[c])) : 0.0;
            int index = c;
            
            unsigned mask = 0xffffffff;
            
            #pragma unroll
            for(int s = 16; s > 0; s >>= 1) {
                int other_index = __shfl_down_sync(mask, index, s);
                double other_value = __shfl_down_sync(mask, value, s);

                if(other_value > value) {
                    value = other_value;
                    index = other_index;
                }
            }

            if(c == 0) {
                kp = index;
            }
        }
        __syncthreads();

        if(kp != k + 1) {
            if(r == 0 && c == 0) {
                int tmp = P[k + 1];
                P[k + 1] = P[kp];
                P[kp] = tmp;
            
                result *= -1.0;
            }
            __syncthreads();
        }

        const double pivot = CM(A, n, P[k], P[k + 1]);
        const double scale = 1.0 / pivot;

        if(r == 0 && c == 0) {
            result *= pivot;
        }

        __syncthreads();

        if(k + 2 < n) {
            double update_value = 0.0;
            if(r > k + 1 && c > k + 1) {
                const double A_kc = CM(A, n, P[k], P[c]);
                const double A_rk1 = CM(A, n, P[r], P[k + 1]);
                const double A_kr = CM(A, n, P[k], P[r]);
                const double A_ck1 = CM(A, n, P[c], P[k + 1]);
                update_value += (A_kr * A_ck1 - A_kc * A_rk1) * scale;
            }
            __syncthreads();

            CM(A, n, P[r], P[c]) += update_value;
        }

        __syncthreads();
    }

    if(r == 0 && c == 0)
        *_result = result;
}
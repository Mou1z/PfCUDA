#include <iostream>
#include <iomanip>
#include <limits>

#include "pfaffian_cpu.h"

double pfaffian(double * A, const int n) {
    if((n == 0) || (n & 1))
        return 0.0;

    if(n == 2)
        return RM(A, n, 0, 1);

    if(n == 4) {
        return (
            RM(A, n, 0, 1) * RM(A, n, 2, 3) -
            RM(A, n, 0, 2) * RM(A, n, 1, 3) +
            RM(A, n, 0, 3) * RM(A, n, 1, 2)
        );
    }

    double result = 1.0;

    for(int k = 0; k < n - 1; k += 2) {
        const int rows = (k & ~1);
        double * __restrict__ v_r = &RM(A, n, k, 0);
        double * __restrict__ w_r = &RM(A, n, k + 1, 0);
        
        int kp = k + 1;
        double kp_val = std::abs(v_r[kp]);

        for(int r = 0; r < rows; r += 2) {
            const double k0 = RM(A, n, r, k);
            const double k1 = -RM(A, n, r + 1, k);

            const double * __restrict__ curr_r0 = &RM(A, n, r, 0);
            const double * __restrict__ curr_r1 = &RM(A, n, r + 1, 0);

            for(int c = k + 1; c < n; c++) {
                const double c_r0 = curr_r0[c];
                const double c_r1 = -curr_r1[c];
                
                v_r[c] += (c_r1 * k0 - c_r0 * k1);
            }
        }

        for(int c = k + 2; c < n; c++) {
            const double c_val = std::abs(v_r[c]);
            if(c_val > kp_val) {
                kp = c; kp_val = c_val;
            }
        }

        if (kp != k + 1) {
            const int i = k + 1;
            const int j = kp;

            for (int r = 0; r < i; r++) {
                std::swap(RM(A, n, r, i), RM(A, n, r, j));
            }

            for (int c = j + 1; c < n; c++) {
                std::swap(RM(A, n, i, c), RM(A, n, j, c));
            }

            double tmp;
            for (int m = i + 1; m < j; m++) {
                tmp = RM(A, n, i, m);
                RM(A, n, i, m) = -RM(A, n, m, j);
                RM(A, n, m, j) = -tmp;
            }

            RM(A, n, i, j) = -RM(A, n, i, j);
            result *= -1.0;
        }
        
        const double pivot = v_r[k + 1];

        if(pivot == 0) 
            return 0.0;

        const double update_factor = 1.0 / pivot;
        result *= pivot;

        for(int r = 0; r < rows; r += 2) {
            const double k0 = RM(A, n, r, k + 1);
            const double k1 = -RM(A, n, r + 1, k + 1);

            const double * __restrict__ curr_r0 = &RM(A, n, r, 0);
            const double * __restrict__ curr_r1 = &RM(A, n, r + 1, 0);

            for(int c = k + 2; c < n; c++) {
                const double c_r0 = curr_r0[c];
                const double c_r1 = -curr_r1[c];
                
                w_r[c] += (c_r1 * k0 - c_r0 * k1);
            }
        }

        for(int c = k + 2; c < n; c++) {
            v_r[c] *= update_factor;
        }
    }

    return result;
}
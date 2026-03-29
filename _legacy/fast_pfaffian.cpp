#include "pfaffian.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstring>

#include "pfaffian.hpp"

#define RM(A, ld, i, j) ((A)[(i) * (ld) + (j)])

double pfaffian(double * A, const int n) {
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

double A[100] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    -1,  0, 11, 12, 13, 14, 15, 16, 17, 18,
    -2, -11, 0, 23, 24, 25, 26, 27, 28, 29,
    -3, -12, -23, 0, 34, 35, 36, 37, 38, 39,
    -4, -13, -24, -34, 0, 45, 46, 47, 48, 49,
    -5, -14, -25, -35, -45, 0, 56, 57, 58, 59,
    -6, -15, -26, -36, -46, -56, 0, 67, 68, 69,
    -7, -16, -27, -37, -47, -57, -67, 0, 78, 79,
    -8, -17, -28, -38, -48, -58, -68, -78, 0, 89,
    -9, -18, -29, -39, -49, -59, -69, -79, -89, 0
};

int main() {
    double result = pfaffian(A, 10);

    std::cout 
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "Result: " << result << std::endl;
}
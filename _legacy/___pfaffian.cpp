#include "pfaffian.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstring>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

// std::cout << "k = " << k << "\n";
// for(int x = 0; x < 10; x++) {
//     for(int y = 0; y < 10; y++) {
//         std::cout << std::ceil(CM(A, n, x, y) * 1000.0) / 1000.0 << "\t";
//     }
//     std::cout << "\n";
// }
// std::cout << "\n";

double pfaffian(double * A, const int n) {
    double result = 1.0;
    double * tmp_buffer = new double[n];

    double 
        * col_k,
        * col_c;

    double col_update;
    double pivot;

    for(int k = 0; k < n - 1; k += 2) {
        int rows = k & ~1;
        
        int kp = k + 1;
        col_k = &CM(A, n, 0, k);
        for(int c = k + 1; c < n; c++) {
            col_update = 0.0;
            col_c = &CM(A, n, 0, c);

            for(int r = 0; r + 1 < rows; r += 2) {
                col_update += (
                    col_c[r + 1] * col_k[r] -
                    col_c[r] * col_k[r + 1]
                );
            }

            col_c[k] += col_update;

            if(std::abs(CM(A, n, k, c)) > std::abs(CM(A, n, k, kp))) {
                kp = c;
            }
        }

        if(kp != k + 1) {
            double * col_k1 = &CM(A, n, 0, k + 1);
            double * col_kp = &CM(A, n, 0, kp);
            
            std::memcpy(tmp_buffer, col_k1, n * sizeof(double));
            std::memcpy(col_k1, col_kp, n * sizeof(double));
            std::memcpy(col_kp, tmp_buffer, n * sizeof(double));
            
            double tmp;
            for(int j = k + 1; j < n; j ++) {
                tmp = CM(A, n, k + 1, j);
                CM(A, n, k + 1, j) = CM(A, n, kp, j);
                CM(A, n, kp, j) = tmp;
            }
            
            result *= -1.0;
        }
        
        pivot = CM(A, n, k, k + 1);
        col_k = &CM(A, n, 0, k + 1);
        for(int c = k + 2; c < n; c++) {
            col_update = 0.0;
            col_c = &CM(A, n, 0, c);

            for(int r = 0; r + 1 < rows; r += 2) {
                col_update += (
                    col_c[r + 1] * col_k[r] -
                    col_c[r] * col_k[r + 1]
                );
            }

            col_c[k + 1] = -(col_c[k + 1] + col_update);
            col_c[k] /= pivot;
        }

        result *= pivot;
    }

    delete tmp_buffer;
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

    std::cout 
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "Result: " << result << std::endl;
}
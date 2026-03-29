#include "pfaffian.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstring>

#include "matrix.hpp"
#include "pfaffian.hpp"

#define RM(A, ld, i, j) ((A)[(i) * (ld) + (j)])

template <typename TScalar>
TScalar pfaffian_cpp(Matrix<TScalar> &matrix_in) {
    size_t n = matrix_in.cols;
    if(n == 0) return 1.0;
    if((n & 1) == 1) return 0;

    TScalar result = 1.0;

    size_t kp;
    for(size_t k = 0; k < (n - 1); k += 2) {
        kp = k + 1;

        for (size_t i = k + 2; i < n; i++) {
            size_t index_i = i * n + k;
            size_t index_kp = kp * n + k;
            
            if (std::abs(matrix_in[index_i]) > std::abs(matrix_in[index_kp])) {
                kp = i;
            }
        }

        if(kp != k + 1) {
            TScalar tmp;

            size_t k1_start = (k + 1) * n;
            size_t kp_start = kp * n;

            for(size_t i = 0; i < n; i++) 
            {
                size_t i_k1 = k1_start + i;
                size_t i_kp = kp_start + i;

                tmp = matrix_in[i_k1];
                matrix_in[i_k1] = matrix_in[i_kp];
                matrix_in[i_kp] = tmp;
            }
            
            for(size_t i = 0; i < n; i++) {
                size_t i_k1 = (i * n) + k + 1;
                size_t i_kp = (i * n) + kp;

                tmp = matrix_in[i_k1];
                matrix_in[i_k1] = matrix_in[i_kp];
                matrix_in[i_kp] = tmp;
            }

            result *= -1;
        }

        TScalar element = matrix_in[(k * n) + k + 1];

        if(element != 0) {
            result *= element;

            size_t tau_len = n - (k + 2);
            TScalar * tau = new TScalar[tau_len];

            for (size_t i = 0; i < tau_len; i++) {
                tau[i] = matrix_in[(k * n) + (k + 2 + i)] / element;
            }            
            
            if (k + 2 < n) {

                for(size_t i = k + 2; i < n; i++) {
                    for(size_t j = k + 2; j < n; j++) {
                        matrix_in[(i * n) + j] += 
                            (
                                (tau[i - (k + 2)] * matrix_in[(j * n) + k + 1]) - 
                                (tau[j - (k + 2)] * matrix_in[(i * n) + k + 1])
                            );
                    }
                }

            }
            
        } else {
            return 0.0;
        }

    }

    return result;
}

template float pfaffian_cpp<float>(Matrix<float> &matrix_in);
template double pfaffian_cpp<double>(Matrix<double> &matrix_in);

// double pfaffian(double * A, const int n) {
//     double result = 1.0;

//     for(int k = 0; k < n - 2; k += 2) {
//         int kp = k + 1;
//         double kp_val = std::abs(RM(A, n, k, kp));

//         for(int c = k + 2; c < n; c++) {
//             const double c_val = std::abs(RM(A, n, k, c));
//             if(c_val > kp_val) {
//                 kp = c; kp_val = c_val;
//             }
//         }

//         if (kp != k + 1) {
//             const int i = k + 1;
//             const int j = kp;

//             for (int r = 0; r < i; r++) {
//                 std::swap(RM(A, n, r, i), RM(A, n, r, j));
//             }

//             for (int c = j + 1; c < n; c++) {
//                 std::swap(RM(A, n, i, c), RM(A, n, j, c));
//             }

//             double tmp;
//             for (int m = i + 1; m < j; m++) {
//                 tmp = RM(A, n, i, m);
//                 RM(A, n, i, m) = -RM(A, n, m, j);
//                 RM(A, n, m, j) = -tmp;
//             }

//             RM(A, n, i, j) = -RM(A, n, i, j);
//             result *= -1.0;
//         }
        
//         const double pivot = RM(A, n, k, k + 1);
//         const double update_factor = 1.0 / pivot;
//         result *= pivot;

//         for(int c = k + 2; c < n; c++) { 
//             RM(A, n, k, c) *= update_factor;
//         }

//         const int rows = ((k + 1) & ~1) - 1;

//         const int k_tr0 = k + 1;
//         const int k_tr1 = k + 2;

//         double * tr0 = &RM(A, n, k + 1, 0);
//         double * tr1 = &RM(A, n, k + 2, 0);

//         for(int r = 0; r < rows; r += 2) {
//             const double tr0_k0 = RM(A, n, r, k_tr0);
//             const double tr0_k1 = -RM(A, n, r + 1, k_tr0);

//             const double tr1_k0 = RM(A, n, r, k_tr1);
//             const double tr1_k1 = -RM(A, n, r + 1, k_tr1);
            
//             const double * __restrict__ curr_r0 = &RM(A, n, r, 0);
//             const double * __restrict__ curr_r1 = &RM(A, n, r + 1, 0);

//             for(int c = k + 1; c < n; c++) {
//                 const double c_r0 = curr_r0[c];
//                 const double c_r1 = -curr_r1[c];

//                 tr0[c] += (c_r1 * tr0_k0 - c_r0 * tr0_k1);
//                 tr1[c] += (c_r1 * tr1_k0 - c_r0 * tr1_k1);
//             }
//         }
//     }

//     result *= RM(A, n, n - 2, n - 1);
//     return result;
// }

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
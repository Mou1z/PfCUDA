#include <iostream>
#include <iomanip>
#include <cblas.h>
#include <limits>
#include <cmath>

#define RM(A, ld, i, j) ((A)[(i) * (ld) + (j)])

double pfaffian(double * A, const int n) {
    double result = 1.0;

    int block_space = sizeof(double) * n * (n / 2);
    double * __restrict__ V = (double*) malloc(block_space);
    double * __restrict__ W = (double*) malloc(block_space);
    double * __restrict__ R = (double*) malloc(n * sizeof(double));

    const double 
        a0 = -1.0, a1 = 1.0,
        b0 = 0.0, b1 = 1.0;

    for(int i = 0, k = 0; k < n - 1; k += 2, i++) {
        int cols = n - k - 1;

        cblas_dgemv(CblasColMajor, CblasTrans, cols, i, a0, &RM(W, n, k + 1, 0), n, &RM(V, n, k, 0), n, b0, R, 1);
        cblas_dgemv(CblasColMajor, CblasTrans, cols, i, a1, &RM(V, n, k + 1, 0), n, &RM(W, n, k, 0), n, b1, R, 1);

        cblas_daxpy(cols, 1.0, &RM(A, n, k, k + 1), 1, R, 1);

        int km = cblas_idamax(cols, R, 1);
        int kp = k + 1 + km;

        double pivot = R[km];

        if(pivot == 0)
            return 0.0;

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
        
        result *= pivot;

        cblas_dscal(cols, 1.0 / pivot, R, 1);
        cblas_dcopy(cols - 1, R + 1, 1, &RM(V, n, i, k + 2), 1);

        cblas_dscal(cols, 0.0, R, 1);
        cblas_dgemv(CblasColMajor, CblasTrans, cols - 1, i, a1, &RM(V, n, k + 2, 0), n, &RM(W, n, k + 1, 0), n, b0, R, 1);
        cblas_dgemv(CblasColMajor, CblasTrans, cols - 1, i, a0, &RM(W, n, k + 2, 0), n, &RM(V, n, k + 1, 0), n, b1, R, 1);
        
        cblas_daxpy(cols - 1, 1.0, &RM(A, n, k + 1, k + 2), 1, R, 1);
        cblas_dcopy(cols - 1, R, 1, &RM(W, n, i, k + 2), 1);
    }

    free(V); free(W); free(R);
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
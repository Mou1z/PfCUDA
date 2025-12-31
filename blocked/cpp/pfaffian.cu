#include <iostream>

double pfaffian(double ** A, long n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    double result = 1.0;

    long blockSize = (n - 2) / 2;

    double * V = new double[n][blockSize];
    double * W = new double[n][blockSize];

    int i;
    double * updateVector = new double[n];
    for(int k = 0; k < n - 2; k += 2) {
        i = k / 2;

        

    }
}

int main(void) {
    return 0;
}
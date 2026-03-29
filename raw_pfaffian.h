#pragma once
#include <cublas_v2.h>

double pfaffian(cublasHandle_t handle, const double* A, const long n);
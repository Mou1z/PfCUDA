#pragma once

#include <string>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuComplex.h>

#define CM(A, ld, i, j) ((A)[(i) + (j) * (ld)])

template <typename T> struct ProjectionType { typedef T type; };
template <> struct ProjectionType<cuFloatComplex> { typedef float type; };
template <> struct ProjectionType<cuDoubleComplex> { typedef double type; };

__device__ __host__ inline float gabs(float x) { return x < 0 ? -x : x; }
__device__ __host__ inline double gabs(double x) { return x < 0 ? -x : x; }
__device__ __host__ inline float gabs(cuFloatComplex x) { return cuCabsf(x); }
__device__ __host__ inline double gabs(cuDoubleComplex x) { return cuCabs(x); }

template<typename T> __device__ __host__ inline T zero();
template<> __device__ __host__ inline float zero<float>() { return 0.0f; }
template<> __device__ __host__ inline double zero<double>() { return 0.0; }
template<> __device__ __host__ inline cuFloatComplex zero<cuFloatComplex>() { return make_cuFloatComplex(0.0f, 0.0f); }
template<> __device__ __host__ inline cuDoubleComplex zero<cuDoubleComplex>() { return make_cuDoubleComplex(0.0, 0.0); }

template<typename T> __device__ __host__ inline T one();
template<> __device__ __host__ inline float one<float>() { return 1.0f; }
template<> __device__ __host__ inline double one<double>() { return 1.0; }
template<> __device__ __host__ inline cuFloatComplex one<cuFloatComplex>() { return make_cuFloatComplex(1.0f, 0.0f); }
template<> __device__ __host__ inline cuDoubleComplex one<cuDoubleComplex>() { return make_cuDoubleComplex(1.0, 0.0); }

template<typename T> __device__ __host__ inline T minus_one();
template<> __device__ __host__ inline float minus_one<float>() { return -1.0f; }
template<> __device__ __host__ inline double minus_one<double>() { return -1.0; }
template<> __device__ __host__ inline cuFloatComplex minus_one<cuFloatComplex>() { return make_cuFloatComplex(-1.0f, 0.0f); }
template<> __device__ __host__ inline cuDoubleComplex minus_one<cuDoubleComplex>() { return make_cuDoubleComplex(-1.0, 0.0); }

__device__ __host__ inline cuFloatComplex operator * (cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
__device__ __host__ inline cuDoubleComplex operator * (cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
__device__ __host__ inline cuFloatComplex& operator *= (cuFloatComplex& a, cuFloatComplex b) { a = cuCmulf(a, b); return a; }
__device__ __host__ inline cuDoubleComplex& operator *= (cuDoubleComplex& a, cuDoubleComplex b) { a = cuCmul(a, b); return a; }

__device__ __host__ inline cuFloatComplex operator + (cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }
__device__ __host__ inline cuDoubleComplex operator + (cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ __host__ inline cuFloatComplex& operator += (cuFloatComplex& a, cuFloatComplex b) { a = cuCaddf(a, b); return a; }
__device__ __host__ inline cuDoubleComplex& operator += (cuDoubleComplex& a, cuDoubleComplex b) { a = cuCadd(a, b); return a; }

__device__ __host__ inline cuFloatComplex operator - (cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }
__device__ __host__ inline cuDoubleComplex operator - (cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }

__device__ __host__ inline cuFloatComplex operator / (cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a, b); }
__device__ __host__ inline cuDoubleComplex operator / (cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }

__device__ __host__ inline cuFloatComplex operator / (cuFloatComplex a, float b) { return make_cuFloatComplex(a.x / b, a.y / b); }
__device__ __host__ inline cuDoubleComplex operator / (cuDoubleComplex a, double b) { return make_cuDoubleComplex(a.x / b, a.y / b); }
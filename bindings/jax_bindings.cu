#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "pfaffian.cuh"
#include "slog_pfaffian.cuh"

namespace ffi = xla::ffi;

template <ffi::DataType> struct get_type;
template <> struct get_type<ffi::F32> { using Type = float; };
template <> struct get_type<ffi::F64> { using Type = double; };
template <> struct get_type<ffi::C64> { using Type = cuFloatComplex; };
template <> struct get_type<ffi::C128> { using Type = cuDoubleComplex; };

template <ffi::DataType T> struct get_result_type {  static constexpr ffi::DataType value = T; };
template <> struct get_result_type<ffi::C64> { static constexpr ffi::DataType value = ffi::F32; };
template <> struct get_result_type<ffi::C128> { static constexpr ffi::DataType value = ffi::F64; };

template <ffi::DataType T>
ffi::Error slog_pfaffian_mapper(
    cudaStream_t stream,
    ffi::Buffer<T> d_A,
    ffi::ResultBuffer<get_result_type<T>::value> d_log_pfaffian,
    ffi::ResultBuffer<T> d_sign
) {
    using cuda_type = typename get_type<T>::Type;
    using log_type = typename get_type<get_result_type<T>::value>::Type;

    auto dims = d_A.dimensions();
    if (dims.size() < 2) {
        return ffi::Error::InvalidArgument(
            "Pfaffian requires at least a 2D matrix (n, n). "
            "Received an array with rank " + std::to_string(dims.size())
        );
    }

    const unsigned int n = dims.back();
    const unsigned int step = n * n;
    const unsigned int total_elements = d_A.element_count();

    cuda_type* d_A_ptr = reinterpret_cast<cuda_type*>(d_A.typed_data());
    log_type* d_log_pfaffian_ptr = reinterpret_cast<log_type*>(d_log_pfaffian->typed_data());
    cuda_type* d_sign_ptr = reinterpret_cast<cuda_type*>(d_sign->typed_data());

    cuda_type* d_temp = nullptr;
    cudaError_t err = cudaMalloc(&d_temp, step * sizeof(cuda_type));
    if (err != cudaSuccess) {
        return ffi::Error::Internal("cudaMalloc failed");
    }

    for (unsigned int i = 0, batch_id = 0; i < total_elements; i += step, ++batch_id) {
        err = cudaMemcpy(
            d_temp,
            d_A_ptr + i,
            step * sizeof(cuda_type),
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) {
            cudaFree(d_temp);
            return ffi::Error::Internal("cudaMemcpy failed");
        }

        slog_pfaffian<cuda_type>(
            d_temp,
            n,
            d_log_pfaffian_ptr + batch_id,
            d_sign_ptr + batch_id,
            stream
        );
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(d_temp);
    if (err != cudaSuccess) {
        return ffi::Error::Internal("cudaStreamSynchronize failed");
    }

    return ffi::Error::Success();
}

template <ffi::DataType T>
ffi::Error pfaffian_mapper(
    cudaStream_t stream,
    ffi::Buffer<T> d_A,
    ffi::ResultBuffer<T> d_result
) {
    using cuda_type = typename get_type<T>::Type;

    auto dims = d_A.dimensions();

    if(dims.size() < 2) {
        return ffi::Error::InvalidArgument(
            "Pfaffian reqires at least a 2D matrix (n, n). "
            "Received an array with rank " + std::to_string(dims.size())
        );
    }

    const unsigned int n = dims.back();
    const unsigned int step = n * n;

    const unsigned int total_elements = d_A.element_count();

    cuda_type * d_A_ptr = reinterpret_cast<cuda_type*>(d_A.typed_data());
    cuda_type * d_result_ptr = reinterpret_cast<cuda_type*>(d_result->typed_data());

    for(unsigned int i = 0, batch_id = 0; i < total_elements; i += step, batch_id++) {
        pfaffian<cuda_type>(d_A_ptr + i, n, d_result_ptr + batch_id, stream);
    }

    return ffi::Error::Success();
}

// slog_pfaffian()
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    slog_pfaffian_f32, slog_pfaffian_mapper<ffi::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    slog_pfaffian_f64, slog_pfaffian_mapper<ffi::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    slog_pfaffian_c64, slog_pfaffian_mapper<ffi::C64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C64>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::C64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    slog_pfaffian_c128, slog_pfaffian_mapper<ffi::C128>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::C128>>()
);

// pfaffian()
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_f32, pfaffian_mapper<ffi::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_f64, pfaffian_mapper<ffi::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_c64, pfaffian_mapper<ffi::C64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C64>>()
        .Ret<ffi::Buffer<ffi::C64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_c128, pfaffian_mapper<ffi::C128>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>()
);

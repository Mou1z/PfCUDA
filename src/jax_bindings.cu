#include "pfaffian.cuh"

#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <ffi::DataType> struct get_type;
template <> struct get_type<ffi::F32> { using Type = float; };
template <> struct get_type<ffi::F64> { using Type = double; };
template <> struct get_type<ffi::C64> { using Type = cuFloatComplex; };
template <> struct get_type<ffi::C128> { using Type = cuDoubleComplex; };

template <ffi::DataType T>
ffi::Error pfaffian_impl(cudaStream_t stream, ffi::Buffer<T> d_A, ffi::ResultBuffer<T> d_result) {
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

    cuda_type* d_A_ptr = reinterpret_cast<cuda_type*>(d_A.typed_data());
    cuda_type* d_result_ptr = reinterpret_cast<cuda_type*>(d_result->typed_data());

    for(unsigned int i = 0; i < total_elements; i += step) {
        unsigned int batch_id = i / step;

        pfaffian<cuda_type>(d_A_ptr + i, n, d_result_ptr + batch_id, stream);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_f32, pfaffian_impl<ffi::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_f64, pfaffian_impl<ffi::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_c64, pfaffian_impl<ffi::C64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C64>>()
        .Ret<ffi::Buffer<ffi::C64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pfaffian_c128, pfaffian_impl<ffi::C128>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>()
);

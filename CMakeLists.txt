cmake_minimum_required(VERSION 3.18)
project(cu_pfaffian LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Build GPU implementation's object file
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND python3 -c "import jax; import os; print(os.path.dirname(jax.__file__))"
    OUTPUT_VARIABLE JAX_BASE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python3 -c "import jaxlib; import os; print(os.path.dirname(jaxlib.__file__))"
    OUTPUT_VARIABLE JAXLIB_BASE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(cu_pfaffian_lib SHARED
    src/pfaffian.cu
    src/pfaffian_sm.cu
    src/slog_pfaffian.cu
    src/slog_pfaffian_lg.cu
    bindings/jax_bindings.cu
)

target_include_directories(cu_pfaffian_lib PRIVATE 
    include
    ${JAX_BASE_DIR}
    ${JAXLIB_BASE_DIR}/include
)

set_target_properties(cu_pfaffian_lib PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "cupfaffian"
)

target_link_libraries(cu_pfaffian_lib PRIVATE
    CUDA::cudart
)

target_compile_options(cu_pfaffian_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)


# Build CPU implementation's object file
find_package(pybind11 REQUIRED)

pybind11_add_module(pfaffian_lib
    bindings/pybind_bindings.cpp
    src/pfaffian_cpu.cpp
)

target_include_directories(pfaffian_lib PRIVATE
    include
)

target_compile_options(pfaffian_lib PRIVATE -O3)

set_target_properties(pfaffian_lib PROPERTIES
    OUTPUT_NAME "cpupfaffian"
)

install(TARGETS cu_pfaffian_lib DESTINATION pfcuda)
install(TARGETS pfaffian_lib DESTINATION pfcuda)
cmake_minimum_required(VERSION 3.18)
project(pfcuda LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

file(GLOB CUDA_SOURCES src/*.cu)
list(REMOVE_ITEM CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/jax_bindings.cu")

add_library(pfcuda_core STATIC ${CUDA_SOURCES})

set_target_properties(pfcuda_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "pfcuda"
)

# Python module
pybind11_add_module(pfcuda bindings/pybind11_bindings.cpp)

target_link_libraries(pfcuda PRIVATE
    CUDA::cudart
    pfcuda_core
)

target_compile_options(pfcuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
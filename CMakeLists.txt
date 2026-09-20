cmake_minimum_required(VERSION 3.18)

# Where no GPU is visible at build time, -arch=native only warns and falls back
# to nvcc's default, yielding a binary that fails on the target card. A fixed
# architecture list is not portable either, since CUDA 13 dropped everything
# below sm_75, so query the toolkit instead. Read by project() below.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
        set(CMAKE_CUDA_ARCHITECTURES all-major)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    endif()
endif()

# pip install runs with the user's PATH, which commonly lacks nvcc. Search the
# usual install locations before project() fails on compiler detection.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(PFCUDA_NVCC nvcc
        HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT /usr/local/cuda /opt/cuda
        PATH_SUFFIXES bin)
    if(PFCUDA_NVCC)
        set(CMAKE_CUDA_COMPILER "${PFCUDA_NVCC}")
    else()
        message(FATAL_ERROR
            "Could not find nvcc, the CUDA compiler.\n"
            "PfCUDA compiles CUDA kernels from source, so the CUDA Toolkit must "
            "be installed: https://developer.nvidia.com/cuda-downloads\n"
            "If it is installed somewhere unusual, point CMake at it with "
            "CUDA_HOME=/path/to/cuda or CUDACXX=/path/to/nvcc.\n"
            "Searched: PATH, $CUDA_HOME/bin, $CUDA_PATH/bin, $CUDA_ROOT/bin, "
            "/usr/local/cuda/bin, /opt/cuda/bin.")
    endif()
endif()

project(cu_pfaffian LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optimisation comes from the build type, so a bare `cmake ..` must not be unoptimised.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Drops the built libraries into the source package so `import pfcuda` needs no
# install step. Off for wheel builds, which stage them through install() below.
option(PFCUDA_DEV_INPLACE "Write built libraries into the source pfcuda/ package" OFF)

find_package(CUDAToolkit REQUIRED)

# The interpreter that owns jax and pybind11: preset by scikit-build-core during
# pip install, passed explicitly by ./dev. A bare `python3` would resolve to the
# system interpreter, which has no jax.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jax; import os; print(os.path.dirname(jax.__file__))"
    OUTPUT_VARIABLE JAX_BASE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JAX_LOOKUP_RESULT
    ERROR_VARIABLE JAX_LOOKUP_ERROR
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jaxlib; import os; print(os.path.dirname(jaxlib.__file__))"
    OUTPUT_VARIABLE JAXLIB_BASE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JAXLIB_LOOKUP_RESULT
    ERROR_VARIABLE JAXLIB_LOOKUP_ERROR
)

if(NOT JAX_LOOKUP_RESULT EQUAL 0 OR NOT JAXLIB_LOOKUP_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not import jax/jaxlib with ${Python_EXECUTABLE}.\n"
        "The CUDA bindings need jaxlib's XLA FFI headers at compile time.\n"
        "Install them into that interpreter, or point CMake at the right one "
        "with -DPython_EXECUTABLE=/path/to/python.\n\n"
        "${JAX_LOOKUP_ERROR}${JAXLIB_LOOKUP_ERROR}")
endif()

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

# -O3 comes from the build type. Forcing it here, or fast-math, would leave a
# Debug build optimised and reassociated.
target_compile_options(cu_pfaffian_lib PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:--use_fast_math>
)


# Prefer the pybind11 shipped with Python_EXECUTABLE over any system-wide copy,
# so the extension module matches the interpreter that will import it.
if(NOT pybind11_DIR)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

# Use FindPython, and so Python_EXECUTABLE, rather than the deprecated
# PythonInterp path, unless the build backend already set this.
if(NOT DEFINED PYBIND11_FINDPYTHON)
    set(PYBIND11_FINDPYTHON ON)
endif()
find_package(pybind11 REQUIRED)

pybind11_add_module(pfaffian_lib
    bindings/pybind_bindings.cpp
    src/pfaffian_cpu.cpp
)

target_include_directories(pfaffian_lib PRIVATE
    include
)

set_target_properties(pfaffian_lib PROPERTIES
    OUTPUT_NAME "cpupfaffian"
)

if(PFCUDA_DEV_INPLACE)
    set_target_properties(cu_pfaffian_lib pfaffian_lib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pfcuda"
    )
endif()

install(TARGETS cu_pfaffian_lib DESTINATION pfcuda)
install(TARGETS pfaffian_lib DESTINATION pfcuda)

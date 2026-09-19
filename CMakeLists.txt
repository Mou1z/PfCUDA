cmake_minimum_required(VERSION 3.18)

# Build for every major architecture the installed toolkit supports. `native`
# would be smaller and faster to compile, but nvcc only warns when it cannot
# see a GPU ("Cannot find valid GPU for '-arch=native', default arch is used")
# and then targets its default, so builds on GPU-less machines -- containers,
# CI, HPC login nodes -- would silently produce a binary that fails on the
# card it eventually runs on. A fixed list is not portable either, since CUDA
# 13 dropped everything below sm_75; all-major asks nvcc what it supports.
# ./dev overrides this with `native` for fast local iteration.
# Must be decided before project(), which consumes it during compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
        set(CMAKE_CUDA_ARCHITECTURES all-major)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    endif()
endif()

# A plain `pip install .` runs with whatever PATH the user's shell has, and a
# CUDA Toolkit installed outside PATH is the common case. Look in the usual
# places before project() gives up with "CUDA compiler identification unknown".
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

# Drops the built libraries into the source package so `import pfcuda` needs no
# install step. Off for wheel builds, which stage them through install() below.
option(PFCUDA_DEV_INPLACE "Write built libraries into the source pfcuda/ package" OFF)

# Build GPU implementation's object file
find_package(CUDAToolkit REQUIRED)

# Resolve the interpreter that owns jax and pybind11. scikit-build-core presets
# Python_EXECUTABLE during `pip install`; ./dev passes it explicitly. Falling
# back to a bare `python3` would find the system interpreter, which has no jax.
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

target_compile_options(cu_pfaffian_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)


# Build CPU implementation's object file
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

# Let pybind11 use FindPython (and therefore Python_EXECUTABLE) rather than the
# deprecated PythonInterp path, unless a build backend already chose for us.
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

target_compile_options(pfaffian_lib PRIVATE -O3)

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

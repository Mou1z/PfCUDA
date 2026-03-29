cmake_minimum_required(VERSION 3.18)
project(pfaffian_cuda LANGUAGES CXX CUDA)

# CUDA
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Build shared library (IMPORTANT: SHARED, not MODULE)
add_library(pfaffian_cuda SHARED
    pfaffian.cu    # your CUDA kernel
)

# Include CUDA headers
target_include_directories(pfaffian_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    /mnt/c/Users/Mouiz/desktop/pfcuda/venv/lib/python3.12/site-packages/jaxlib/include
)

# Link CUDA libraries
target_link_libraries(pfaffian_cuda PRIVATE
    CUDA::cudart
)

# Required for Python/JAX dynamic loading
set_target_properties(pfaffian_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".so"
)

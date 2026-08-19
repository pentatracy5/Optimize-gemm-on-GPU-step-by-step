cmake_minimum_required(VERSION 3.24)

project(OptimizeGemm
    LANGUAGES CXX CUDA
    VERSION 1.0.0
)

# Default to building for the GPU present on this machine; override with
# -D CMAKE_CUDA_ARCHITECTURES=<arch> (e.g. 86) to target other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Global dependencies used by the library and/or the executable.

# CUDA Toolkit (matmul_lib uses cuda_runtime.h; matmul_ref uses cuBLAS).
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)

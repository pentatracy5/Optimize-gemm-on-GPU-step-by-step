# Core GEMM library: matmul implementations + helper utilities.
add_library(matmul_lib STATIC
    Timer.cpp
    matmul.cpp
    utils.cu
)

target_include_directories(matmul_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# cuBLAS is used by the reference implementation matmul_ref; cudart is part of
# matmul_lib's public interface (Timer.h exposes cudaStream_t/cudaEvent_t).
target_link_libraries(matmul_lib PUBLIC CUDA::cublas CUDA::cudart)

# Driver executable.
add_executable(matmul main.cpp)
target_link_libraries(matmul PRIVATE matmul_lib)

# Release-only optimization flags. Only apply to C++ sources; do not pass
# MSVC /O2 or /arch:AVX2 to the CUDA compiler (nvcc) in matmul_lib.
# Also enable -O3 for CUDA device code so utils.cu kernels are optimized.
if(MSVC)
    target_compile_options(matmul_lib PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /arch:AVX2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    )
    target_compile_options(matmul PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX2>)
else()
    target_compile_options(matmul_lib PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    )
    target_compile_options(matmul PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
endif()

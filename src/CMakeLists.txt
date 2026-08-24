# Core GEMM library: matmul implementations + helper utilities.
add_library(matmul_lib STATIC
    Timer.cpp
    matmul.cu
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
    # CUDA headers contain UTF-8 characters; without /utf-8 MSVC reads them in
    # the system code page (e.g. 936) and emits warning C4819. CUDA sources
    # need the flag forwarded to the host compiler via -Xcompiler.
    target_compile_options(matmul_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /arch:AVX2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    )
    target_compile_options(matmul PRIVATE
        /utf-8
        $<$<CONFIG:Release>:/O2 /arch:AVX2>
    )
else()
    target_compile_options(matmul_lib PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    )
    target_compile_options(matmul PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
endif()

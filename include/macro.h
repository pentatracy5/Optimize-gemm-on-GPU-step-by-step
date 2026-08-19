#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cstdlib>

#define N_BLOCKS_PER_DIM(total_threads, threads_per_block)  (((total_threads) + (threads_per_block) - 1) / (threads_per_block))
#define N_BLOCKS(total_threads, threads_per_block)          (dim3{N_BLOCKS_PER_DIM(total_threads.x, threads_per_block.x), N_BLOCKS_PER_DIM(total_threads.y, threads_per_block.y), N_BLOCKS_PER_DIM(total_threads.z, threads_per_block.z)})

#define CUDA_LAUNCH(kernel, total_threads, threads_per_block)										        kernel<<<N_BLOCKS(total_threads, threads_per_block), threads_per_block>>>
#define CUDA_LAUNCH_SHAREDMEM(kernel, total_threads, threads_per_block, shared_mem_bytes)				    kernel<<<N_BLOCKS(total_threads, threads_per_block), threads_per_block, shared_mem_bytes>>>
#define CUDA_LAUNCH_SHAREDMEM_STREAM(kernel, total_threads, threads_per_block, shared_mem_bytes, stream)	kernel<<<N_BLOCKS(total_threads, threads_per_block), threads_per_block, shared_mem_bytes, stream>>>

#define MALLOC_CHECK(ptr)                                                        \
    do {                                                                         \
        if ((ptr) == nullptr) {                                                  \
            std::cerr << "Host memory allocation failed"                         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_KERNEL_LAUNCH_CHECK()                                               \
    do {                                                                         \
        cudaError_t err = cudaGetLastError();                                    \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error: " << cublasGetStatusString(status)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CURAND_CHECK(call)                                                       \
    do {                                                                         \
        curandStatus_t status = call;                                            \
        if (status != CURAND_STATUS_SUCCESS) {                                   \
            std::cerr << "cuRAND error: "                                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#include <Timer.h>
#include <macro.h>

Timer::Timer()
{
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}

Timer::~Timer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void Timer::tic_gpu(cudaStream_t stream)
{
    cudaEventRecord(start, stream);
}

float Timer::toc_gpu(cudaStream_t stream)
{
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

void Timer::tic_cpu()
{
    begin = std::chrono::high_resolution_clock::now();
}

float Timer::toc_cpu()
{
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - begin;
    return elapsed.count() * 1000.0f;
}
#include <config.h>
#include <utils.h>
#include <macro.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <curand_kernel.h>

using std::cerr;
using std::cout;
using std::endl;

struct CudaFreeDeleter
{
	void operator()(void *p) const noexcept
	{
		if (p)
			cudaFree(p);
	}
};
static std::unique_ptr<void, CudaFreeDeleter> cacheFlushBuffer;
static size_t flushSize = 0;
void clear_cache()
{
	if (!cacheFlushBuffer)
	{
		int device = 0;
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
		flushSize = prop.l2CacheSize * 2;
		if (flushSize < 10 * 1024 * 1024)
			flushSize = 10 * 1024 * 1024;
		void *raw = nullptr;
		CUDA_CHECK(cudaMalloc(&raw, flushSize));
		cacheFlushBuffer.reset(raw);
	}
	CUDA_CHECK(cudaMemset(cacheFlushBuffer.get(), 0, flushSize));
}

void malloc_matrix(
	const int M,
	const int N,
	const int L,
	int &lda,
	int &ldb,
	int &ldc,
	float *&d_A,
	float *&d_B,
	float *&d_C,
	float *&d_REF)
{
	// padding for vectorize load and store
	lda = (L + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
	ldb = (N + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
	ldc = (N + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
	CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * lda));
	CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * L * ldb));
	CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * ldc));
	CUDA_CHECK(cudaMalloc(&d_REF, sizeof(float) * M * ldc));
}

void free_matrix(
	float *&d_A,
	float *&d_B,
	float *&d_C,
	float *&d_REF)
{
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	CUDA_CHECK(cudaFree(d_REF));
	d_A = nullptr;
	d_B = nullptr;
	d_C = nullptr;
	d_REF = nullptr;
}

__global__ void setup_states(
	curandStatePhilox4_32_10_t *states,
	const unsigned int size,
	const unsigned long long seed)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	curand_init(seed, idx, 0, states + idx);
}

__global__ void rand_init_kernel(
	curandStatePhilox4_32_10_t *states,
	float *output,
	const unsigned int rows,
	const unsigned int cols,
	const unsigned int ld)
{
	const unsigned int row_id = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (row_id >= rows || col_id >= ld)
		return;
	output[row_id * ld + col_id] = col_id >= cols ? 0.0f : curand_uniform(states + row_id * cols + col_id);
}

void init_ABCREF(
	const int M,
	const int N,
	const int L,
	const int lda,
	const int ldb,
	const int ldc,
	float *d_A,
	float *d_B,
	float *d_C,
	float *d_REF)
{
	// Generate the seed randomly once per program launch (hardware entropy
	// combined with the current time) so each run uses different matrices.
	auto device = static_cast<unsigned long long>(std::random_device{}()) << 32;
	auto time = static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	const unsigned long long SEED = device ^ time;

	curandStatePhilox4_32_10_t *d_states;
	const size_t state_size = std::max(static_cast<size_t>(M) * L, static_cast<size_t>(L) * N);
	CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandStatePhilox4_32_10_t) * state_size));

	// setup states
	dim3 total_threads_1d{static_cast<unsigned int>(state_size)};
	dim3 threads_per_block_1d{512};
	CUDA_LAUNCH(setup_states, total_threads_1d, threads_per_block_1d)(d_states,
																	  static_cast<unsigned int>(state_size),
																	  SEED);
	CUDA_KERNEL_LAUNCH_CHECK();

	// random init d_A
	dim3 total_threads_A{static_cast<unsigned int>(lda), static_cast<unsigned int>(M)};
	dim3 threads_per_block{16, 16};
	CUDA_LAUNCH(rand_init_kernel, total_threads_A, threads_per_block)(
		d_states,
		d_A,
		static_cast<unsigned int>(M),
		static_cast<unsigned int>(L),
		static_cast<unsigned int>(lda));
	CUDA_KERNEL_LAUNCH_CHECK();

	// setup states again, reseeded so that B is independent of A
	CUDA_LAUNCH(setup_states, total_threads_1d, threads_per_block_1d)(
		d_states,
		static_cast<unsigned int>(state_size),
		SEED + 1);
	CUDA_KERNEL_LAUNCH_CHECK();

	// random init d_B
	dim3 total_threads_B{static_cast<unsigned int>(ldb), static_cast<unsigned int>(L)};
	CUDA_LAUNCH(rand_init_kernel, total_threads_B, threads_per_block)(
		d_states,
		d_B,
		static_cast<unsigned int>(L),
		static_cast<unsigned int>(N),
		static_cast<unsigned int>(ldb));
	CUDA_KERNEL_LAUNCH_CHECK();

	// zero init d_C
	CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M * ldc));

	// zero init d_REF
	CUDA_CHECK(cudaMemset(d_REF, 0, sizeof(float) * M * ldc));

	CUDA_CHECK(cudaFree(d_states));
}

void print_ABC(
	const int M,
	const int N,
	const int L,
	const int lda,
	const int ldb,
	const int ldc,
	const float *d_A,
	const float *d_B,
	const float *d_C)
{
	float *h_A = (float *)malloc(sizeof(float) * M * lda);
	MALLOC_CHECK(h_A);
	CUDA_CHECK(cudaMemcpy(h_A, d_A, sizeof(float) * M * lda, cudaMemcpyDeviceToHost));
	float *h_B = (float *)malloc(sizeof(float) * L * ldb);
	MALLOC_CHECK(h_B);
	CUDA_CHECK(cudaMemcpy(h_B, d_B, sizeof(float) * L * ldb, cudaMemcpyDeviceToHost));
	float *h_C = (float *)malloc(sizeof(float) * M * ldc);
	MALLOC_CHECK(h_C);
	CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * M * ldc, cudaMemcpyDeviceToHost));

	cout << "Matrix A:" << endl;
	for (int i = 0; i < M; i++)
	{
		for (int k = 0; k < L; k++)
			cout << h_A[i * lda + k] << " ";
		cout << endl;
	}

	cout << "Matrix B:" << endl;
	for (int k = 0; k < L; k++)
	{
		for (int j = 0; j < N; j++)
			cout << h_B[k * ldb + j] << " ";
		cout << endl;
	}

	cout << "Matrix C:" << endl;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
			cout << h_C[i * ldc + j] << " ";
		cout << endl;
	}

	free(h_A);
	free(h_B);
	free(h_C);
}

bool check_result(
	const int M,
	const int N,
	const int ldc,
	const float *d_C,
	const float *d_REF,
	const float tolerance)
{
	float *h_C = (float *)malloc(sizeof(float) * M * ldc);
	MALLOC_CHECK(h_C);
	CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * M * ldc, cudaMemcpyDeviceToHost));
	float *h_REF = (float *)malloc(sizeof(float) * M * ldc);
	MALLOC_CHECK(h_REF);
	CUDA_CHECK(cudaMemcpy(h_REF, d_REF, sizeof(float) * M * ldc, cudaMemcpyDeviceToHost));

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		{
			const float expected = h_REF[static_cast<size_t>(i) * ldc + j];
			const float actual = h_C[static_cast<size_t>(i) * ldc + j];
			const float diff = std::fabs(actual - expected);
			// relative tolerance with an absolute floor for near-zero values
			if (diff > tolerance * std::max(1.0f, std::fabs(expected)))
			{
				cerr << "Error: C(" << i << ", " << j << ") = " << actual << ", but expected " << expected << endl;
				free(h_C);
				free(h_REF);
				return false;
			}
		}

	free(h_C);
	free(h_REF);
	return true;
}

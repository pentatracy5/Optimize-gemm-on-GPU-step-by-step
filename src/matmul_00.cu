#include <config.h>
#include <matmul.h>
#include <macro.h>

template <
    int BM,
    int BN,
    int BL,
    int RM,
    int RN>
__global__ void matmul_00_kernel(
    const int M,
    const int N,
    const int L,
    const int lda,
    const int ldb,
    const int ldc,
    const float alpha,
    const float beta,
    float *d_A,
    float *d_B,
    float *d_C)
{
    int block_row_id_start = blockIdx.y * BM;
    int block_col_id_start = blockIdx.x * BN;
    int thread_row_id_start = threadIdx.y * RM;
    int thread_col_id_start = threadIdx.x * RN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.y * blockDim.x;

    int current_block_A_M = min(BM, M - block_row_id_start);
    int current_block_A_L;
    int current_block_B_L;
    int current_block_B_N = min(BN, N - block_col_id_start);
    int current_thread_C_M = min(RM, current_block_A_M - thread_row_id_start);
    int current_thread_C_N = min(RN, current_block_B_N - thread_col_id_start);

    float *current_block_A_base;
    float *current_block_B_base;
    float *current_block_C_base = d_C + block_row_id_start * ldc + block_col_id_start;
    float *current_thread_A_base;
    float *current_thread_B_base;
    float *current_thread_C_base = current_block_C_base + thread_row_id_start * ldc + thread_col_id_start;

    __shared__ float A_panel[BL][BM];
    __shared__ float B_panel[BL][BN];
    float A_reg[RM]{0.0f};
    float B_reg[RN]{0.0f};
    float C_reg[RM][RN]{0.0f};

    int row_id;
    int col_id;
    for (int block_L_id_start = 0; block_L_id_start < L; block_L_id_start += BL)
    {
        // parallel load A from global memory to A panel
        current_block_A_base = d_A + block_L_id_start * lda + block_row_id_start;
        current_block_A_L = min(BL, L - block_L_id_start);
        for (int t = tid; t < BM * BL; t += threads_per_block)
        {
            col_id = t / BM;
            row_id = t - col_id * BM;
            A_panel[col_id][row_id] = (row_id < current_block_A_M && col_id < current_block_A_L)
                                          ? current_block_A_base[col_id * lda + row_id]
                                          : 0.0f;
        }
        // parallel load B from global memory to B panel
        current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
        current_block_B_L = current_block_A_L;
        for (int t = tid; t < BL * BN; t += threads_per_block)
        {
            row_id = t / BN;
            col_id = t - row_id * BN;
            B_panel[row_id][col_id] = (row_id < current_block_B_L && col_id < current_block_B_N)
                                          ? current_block_B_base[row_id * ldb + col_id]
                                          : 0.0f;
        }
        __syncthreads();
        // compute C
        for (int thread_L_id_start = 0; thread_L_id_start < BL; thread_L_id_start++)
        {
            // load A from A panel to A register
            current_thread_A_base = &A_panel[thread_L_id_start][thread_row_id_start];
            for (int i = 0; i < RM; i++)
            {
                A_reg[i] = current_thread_A_base[i];
            }
            // load B from B panel to B register
            current_thread_B_base = &B_panel[thread_L_id_start][thread_col_id_start];
            for (int j = 0; j < RN; j++)
            {
                B_reg[j] = current_thread_B_base[j];
            }
            // compute C
            for (int i = 0; i < RM; i++)
            {
                for (int j = 0; j < RN; j++)
                {
                    C_reg[i][j] += A_reg[i] * B_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // store C
    for (int i = 0; i < current_thread_C_M; i++)
    {
        for (int j = 0; j < current_thread_C_N; j++)
        {
            current_thread_C_base[i * ldc + j] = alpha * C_reg[i][j] + beta * current_thread_C_base[i * ldc + j];
        }
    }
}

void matmul_00(
    const int M,
    const int N,
    const int L,
    const int lda,
    const int ldb,
    const int ldc,
    float *d_A,
    float *d_B,
    float *d_C)
{
    auto kernel = matmul_00_kernel<BM, BN, BL, RM, RN>;
    const dim3 total_threads{static_cast<unsigned int>((N + RN - 1) / RN), static_cast<unsigned int>((M + RM - 1) / RM)};
    const dim3 threads_per_block{static_cast<unsigned int>(BN / RN), static_cast<unsigned int>(BM / RM)};
    CUDA_LAUNCH(kernel, total_threads, threads_per_block)(
        M,
        N,
        L,
        lda,
        ldb,
        ldc,
        ALPHA,
        BETA,
        d_A,
        d_B,
        d_C);
    CUDA_KERNEL_LAUNCH_CHECK();
}

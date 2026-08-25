#include <config.h>
#include <matmul.h>
#include <macro.h>
#include <types.h>

template <
    int BM,
    int BN,
    int BL,
    int RM,
    int RN,
    int VEC_SIZE,
    int THREADS_PER_BLOCK>
__global__ void matmul_02_kernel(
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
    using FVEC = VecN<float, VEC_SIZE>;
    constexpr FVEC ZERO_VEC = {};

    int block_row_id_start = blockIdx.y * BM;
    int block_col_id_start = blockIdx.x * BN;
    int thread_row_id_start = threadIdx.y * RM;
    int thread_col_id_start = threadIdx.x * RN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

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

    alignas(sizeof(FVEC)) __shared__ float A_panel[2][BL][BM];
    alignas(sizeof(FVEC)) __shared__ float B_panel[2][BL][BN];
    alignas(sizeof(FVEC)) float A_reg[RM]{0.0f};
    alignas(sizeof(FVEC)) float B_reg[RN]{0.0f};
    alignas(sizeof(FVEC)) float C_reg[RM][RN]{0.0f};
    constexpr int num_A_panel_reg = (BM * BL + THREADS_PER_BLOCK * VEC_SIZE - 1) / (THREADS_PER_BLOCK * VEC_SIZE);
    constexpr int num_B_panel_reg = (BL * BN + THREADS_PER_BLOCK * VEC_SIZE - 1) / (THREADS_PER_BLOCK * VEC_SIZE);
    FVEC A_panel_reg[num_A_panel_reg];
    FVEC B_panel_reg[num_B_panel_reg];

    int row_id;
    int col_id;
    int write_id = 0;
    int read_id = 1;
    for (int block_L_id_start = 0; block_L_id_start < L + BL; block_L_id_start += BL)
    {
        if (block_L_id_start < L)
        {
            // parallel load A from global memory to A panel reg
            current_block_A_base = d_A + block_row_id_start * lda + block_L_id_start;
            current_block_A_L = min(BL, L - block_L_id_start);
            for (int t = tid * VEC_SIZE, s = 0; t < BM * BL; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                row_id = t / BL;
                col_id = t - row_id * BL;
                A_panel_reg[s] = (row_id < current_block_A_M && col_id < current_block_A_L)
                                     ? *reinterpret_cast<FVEC *>(current_block_A_base + row_id * lda + col_id)
                                     : ZERO_VEC;
            }
            // parallel load B from global memory to B panel reg
            current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
            current_block_B_L = current_block_A_L;
            for (int t = tid * VEC_SIZE, s = 0; t < BL * BN; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                row_id = t / BN;
                col_id = t - row_id * BN;
                B_panel_reg[s] = (row_id < current_block_B_L && col_id < current_block_B_N)
                                     ? *reinterpret_cast<FVEC *>(current_block_B_base + row_id * ldb + col_id)
                                     : ZERO_VEC;
            }
        }

        if (block_L_id_start > 0)
        {
            // compute C
            for (int thread_L_id_start = 0; thread_L_id_start < BL; thread_L_id_start++)
            {
                // load A from A panel to A register
                current_thread_A_base = &A_panel[read_id][thread_L_id_start][thread_row_id_start];
#pragma unroll RM / VEC_SIZE
                for (int i = 0; i < RM; i += VEC_SIZE)
                {
                    *reinterpret_cast<FVEC *>(A_reg + i) = *reinterpret_cast<FVEC *>(current_thread_A_base + i);
                }
                // load B from B panel to B register
                current_thread_B_base = &B_panel[read_id][thread_L_id_start][thread_col_id_start];
#pragma unroll RN / VEC_SIZE
                for (int j = 0; j < RN; j += VEC_SIZE)
                {
                    *reinterpret_cast<FVEC *>(B_reg + j) = *reinterpret_cast<FVEC *>(current_thread_B_base + j);
                }
                // compute C
#pragma unroll RM
                for (int i = 0; i < RM; i++)
                {
#pragma unroll RN / VEC_SIZE
                    for (int j = 0; j < RN; j += VEC_SIZE)
                    {
                        *reinterpret_cast<FVEC *>(&C_reg[i][j]) += A_reg[i] * *reinterpret_cast<FVEC *>(B_reg + j);
                    }
                }
            }
        }

        if (block_L_id_start < L)
        {
            // parallel load A from reg to A panel
            current_block_A_base = d_A + block_row_id_start * lda + block_L_id_start;
            current_block_A_L = min(BL, L - block_L_id_start);
            for (int t = tid * VEC_SIZE, s = 0; t < BM * BL; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                row_id = t / BL;
                col_id = t - row_id * BL;
#pragma unroll VEC_SIZE
                for (int j = 0; j < VEC_SIZE; j++)
                {
                    A_panel[write_id][col_id + j][row_id] = A_panel_reg[s][j];
                }
            }
            // parallel load B from reg to B panel
            current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
            current_block_B_L = current_block_A_L;
            for (int t = tid * VEC_SIZE, s = 0; t < BL * BN; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                row_id = t / BN;
                col_id = t - row_id * BN;
                *reinterpret_cast<FVEC *>(&B_panel[write_id][row_id][col_id]) = B_panel_reg[s];
            }
        }

        write_id ^= 1;
        read_id ^= 1;
        __syncthreads();
    }

    // store C
    for (int i = 0; i < current_thread_C_M; i++)
    {
        for (int j = 0; j < current_thread_C_N; j += VEC_SIZE)
        {
            *reinterpret_cast<FVEC *>(&current_thread_C_base[i * ldc + j]) =
                alpha * *reinterpret_cast<FVEC *>(&C_reg[i][j]) +
                beta * *reinterpret_cast<FVEC *>(&current_thread_C_base[i * ldc + j]);
        }
    }
}

void matmul_02(
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
    const dim3 total_threads{static_cast<unsigned int>((N + RN - 1) / RN), static_cast<unsigned int>((M + RM - 1) / RM)};
    const dim3 threads_per_block{static_cast<unsigned int>(BN / RN), static_cast<unsigned int>(BM / RM)};
    constexpr int THREADS_PER_BLOCK = (BN * BM) / (RN * RM);
    auto kernel = matmul_02_kernel<BM, BN, BL, RM, RN, VEC_SIZE, THREADS_PER_BLOCK>;
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

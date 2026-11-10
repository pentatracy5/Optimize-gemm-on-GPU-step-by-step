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
    int VEC_SIZE>
__global__ void matmul_02_kernel(
    const int M,
    const int N,
    const int L,
    const int lda,
    const int ldb,
    const int ldc,
    const float alpha,
    const float beta,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    using FVEC = VecN<float, VEC_SIZE>;
    constexpr FVEC ZERO_VEC = {};

    constexpr int THREADS_PER_BLOCK_X = BN / RN;
    constexpr int THREADS_PER_BLOCK_Y = BM / RM;
    constexpr int THREADS_PER_BLOCK = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;
    constexpr int NUM_A_PANEL_REG = (BM * BL + THREADS_PER_BLOCK * VEC_SIZE - 1) / (THREADS_PER_BLOCK * VEC_SIZE);
    constexpr int NUM_B_PANEL_REG = (BL * BN + THREADS_PER_BLOCK * VEC_SIZE - 1) / (THREADS_PER_BLOCK * VEC_SIZE);

    const int block_row_id_start = blockIdx.y * BM;
    const int block_col_id_start = blockIdx.x * BN;

    const int current_block_M = min(BM, M - block_row_id_start);
    const int current_block_N = min(BN, N - block_col_id_start);
    int current_block_L;

    const int thread_row_id_start = threadIdx.y * RM;
    const int thread_col_id_start = threadIdx.x * RN;

    const int current_thread_M = min(RM, current_block_M - thread_row_id_start);
    const int current_thread_N = min(RN, current_block_N - thread_col_id_start);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float *current_block_A_base;
    float *current_block_B_base;
    float *current_block_C_base = d_C + block_row_id_start * ldc + block_col_id_start;
    float *current_thread_A_base;
    float *current_thread_B_base;
    float *current_thread_C_base = current_block_C_base + thread_row_id_start * ldc + thread_col_id_start;

    alignas(sizeof(FVEC)) __shared__ float A_panel[2][BL][BM];
    alignas(sizeof(FVEC)) __shared__ float B_panel[2][BL][BN];
    alignas(sizeof(FVEC)) float A_reg[2][RM]{0.0f};
    alignas(sizeof(FVEC)) float B_reg[2][RN]{0.0f};
    alignas(sizeof(FVEC)) float C_reg[RM][RN]{0.0f};
    FVEC A_panel_reg[NUM_A_PANEL_REG];
    FVEC B_panel_reg[NUM_B_PANEL_REG];

    int row_id;
    int col_id;
    int panel_write_id = 0;
    int panel_read_id = 1;
    int reg_write_id;
    int reg_read_id;
    for (int block_L_id_start = 0; block_L_id_start < L + BL; block_L_id_start += BL)
    {
        if (block_L_id_start < L)
        {
            current_block_L = min(BL, L - block_L_id_start);
            current_block_A_base = d_A + block_L_id_start * lda + block_row_id_start;
            for (int t = tid * VEC_SIZE, s = 0; t < BM * BL; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                col_id = t / BM;
                row_id = t - col_id * BM;
                A_panel_reg[s] =
                    (row_id < current_block_M && col_id < current_block_L)
                        ? *reinterpret_cast<FVEC *>(current_block_A_base + col_id * lda + row_id)
                        : ZERO_VEC;
            }
            current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
            for (int t = tid * VEC_SIZE, s = 0; t < BL * BN; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                row_id = t / BN;
                col_id = t - row_id * BN;
                B_panel_reg[s] =
                    (row_id < current_block_L && col_id < current_block_N)
                        ? *reinterpret_cast<FVEC *>(current_block_B_base + row_id * ldb + col_id)
                        : ZERO_VEC;
            }
        }

        if (block_L_id_start > 0)
        {
            reg_write_id = 0;
            reg_read_id = 1;
#pragma unroll
            for (int thread_L_id_start = 0; thread_L_id_start < BL + 1; thread_L_id_start++)
            {
                if (thread_L_id_start < BL)
                {
                    current_thread_A_base = &A_panel[panel_read_id][thread_L_id_start][thread_row_id_start];
#pragma unroll
                    for (int i = 0; i < RM; i += VEC_SIZE)
                    {
                        *reinterpret_cast<FVEC *>(A_reg[reg_write_id] + i) =
                            *reinterpret_cast<FVEC *>(current_thread_A_base + i);
                    }
                    current_thread_B_base = &B_panel[panel_read_id][thread_L_id_start][thread_col_id_start];
#pragma unroll
                    for (int j = 0; j < RN; j += VEC_SIZE)
                    {
                        *reinterpret_cast<FVEC *>(B_reg[reg_write_id] + j) =
                            *reinterpret_cast<FVEC *>(current_thread_B_base + j);
                    }
                }

                if (thread_L_id_start > 0)
                {
#pragma unroll
                    for (int i = 0; i < RM; i++)
                    {
#pragma unroll
                        for (int j = 0; j < RN; j += VEC_SIZE)
                        {
                            *reinterpret_cast<FVEC *>(&C_reg[i][j]) +=
                                A_reg[reg_read_id][i] *
                                *reinterpret_cast<FVEC *>(B_reg[reg_read_id] + j);
                        }
                    }
                }

                reg_write_id ^= 1;
                reg_read_id ^= 1;
            }
        }

        if (block_L_id_start < L)
        {
            current_block_L = min(BL, L - block_L_id_start);
            current_block_A_base = d_A + block_L_id_start * lda + block_row_id_start;
            for (int t = tid * VEC_SIZE, s = 0; t < BM * BL; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                col_id = t / BM;
                row_id = t - col_id * BM;
                *reinterpret_cast<FVEC *>(&A_panel[panel_write_id][col_id][row_id]) = A_panel_reg[s];
            }
            current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
            for (int t = tid * VEC_SIZE, s = 0; t < BL * BN; t += THREADS_PER_BLOCK * VEC_SIZE, s++)
            {
                row_id = t / BN;
                col_id = t - row_id * BN;
                *reinterpret_cast<FVEC *>(&B_panel[panel_write_id][row_id][col_id]) = B_panel_reg[s];
            }

            panel_write_id ^= 1;
            panel_read_id ^= 1;
            __syncthreads();
        }
    }

    for (int i = 0; i < current_thread_M; i++)
    {
        for (int j = 0; j < current_thread_N; j += VEC_SIZE)
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
    auto kernel = matmul_02_kernel<BM, BN, BL, RM, RN, VEC_SIZE>;
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

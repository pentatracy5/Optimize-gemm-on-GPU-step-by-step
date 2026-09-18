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
__global__ void matmul_07_kernel(
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
    constexpr int NUM_AREA_X = RN / VEC_SIZE;
    constexpr int NUM_AREA_Y = RM / VEC_SIZE;
    constexpr int AREA_X_SIZE = THREADS_PER_BLOCK_X * VEC_SIZE;
    constexpr int AREA_Y_SIZE = THREADS_PER_BLOCK_Y * VEC_SIZE;

    constexpr int LDGSTS_A_ROW_STRIDE = (BM / VEC_SIZE < THREADS_PER_BLOCK) ? BM / VEC_SIZE : THREADS_PER_BLOCK;
    constexpr int LDGSTS_A_COL_STRIDE = THREADS_PER_BLOCK / LDGSTS_A_ROW_STRIDE;
    constexpr int LDGSTS_B_COL_STRIDE = (BN / VEC_SIZE < THREADS_PER_BLOCK) ? BN / VEC_SIZE : THREADS_PER_BLOCK;
    constexpr int LDGSTS_B_ROW_STRIDE = THREADS_PER_BLOCK / LDGSTS_B_COL_STRIDE;
    constexpr int LDGSTS_A_ROW_TIMES = (BM / VEC_SIZE) / LDGSTS_A_ROW_STRIDE;
    constexpr int LDGSTS_A_COL_TIMES = (BL + LDGSTS_A_COL_STRIDE - 1) / LDGSTS_A_COL_STRIDE;
    constexpr int LDGSTS_B_COL_TIMES = (BN / VEC_SIZE) / LDGSTS_B_COL_STRIDE;
    constexpr int LDGSTS_B_ROW_TIMES = (BL + LDGSTS_B_ROW_STRIDE - 1) / LDGSTS_B_ROW_STRIDE;

    const int block_row_id_start = blockIdx.y * BM;
    const int block_col_id_start = blockIdx.x * BN;
    const int current_block_M = min(BM, M - block_row_id_start);
    const int current_block_N = min(BN, N - block_col_id_start);
    int current_block_L;

    int area_row_id_start[NUM_AREA_Y];
    int area_col_id_start[NUM_AREA_X];

    const int thread_row_id_start = threadIdx.y * VEC_SIZE;
    const int thread_col_id_start = threadIdx.x * VEC_SIZE;
    int thread_C_M[NUM_AREA_Y];
    int thread_C_N[NUM_AREA_X];

#pragma unroll NUM_AREA_Y
    for (int i = 0; i < NUM_AREA_Y; i++)
    {
        area_row_id_start[i] = AREA_Y_SIZE * i;
        int area_M = min(AREA_Y_SIZE, current_block_M - area_row_id_start[i]);
        thread_C_M[i] = min(VEC_SIZE, area_M - thread_row_id_start);
    }
#pragma unroll NUM_AREA_X
    for (int j = 0; j < NUM_AREA_X; j++)
    {
        area_col_id_start[j] = AREA_X_SIZE * j;
        int area_N = min(AREA_X_SIZE, current_block_N - area_col_id_start[j]);
        thread_C_N[j] = min(VEC_SIZE, area_N - thread_col_id_start);
    }

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ldgsts_A_col_id = tid / LDGSTS_A_ROW_STRIDE;
    const int ldgsts_A_row_id = tid - ldgsts_A_col_id * LDGSTS_A_ROW_STRIDE;
    const int ldgsts_B_row_id = tid / LDGSTS_B_COL_STRIDE;
    const int ldgsts_B_col_id = tid - ldgsts_B_row_id * LDGSTS_B_COL_STRIDE;

    int A_row_id[LDGSTS_A_ROW_TIMES];
    int A_col_id[LDGSTS_A_COL_TIMES];
    bool A_is_row_ok[LDGSTS_A_ROW_TIMES];
    int B_row_id[LDGSTS_B_ROW_TIMES];
    int B_col_id[LDGSTS_B_COL_TIMES];
    bool B_is_col_ok[LDGSTS_B_COL_TIMES];
#pragma unroll LDGSTS_A_COL_TIMES
    for (int j = 0; j < LDGSTS_A_COL_TIMES; j++)
    {
        A_col_id[j] = ldgsts_A_col_id + j * LDGSTS_A_COL_STRIDE;
    }
#pragma unroll LDGSTS_A_ROW_TIMES
    for (int i = 0; i < LDGSTS_A_ROW_TIMES; i++)
    {
        A_row_id[i] = (ldgsts_A_row_id + i * LDGSTS_A_ROW_STRIDE) * VEC_SIZE;
        A_is_row_ok[i] = A_row_id[i] < current_block_M;
    }
#pragma unroll LDGSTS_B_ROW_TIMES
    for (int i = 0; i < LDGSTS_B_ROW_TIMES; i++)
    {
        B_row_id[i] = ldgsts_B_row_id + i * LDGSTS_B_ROW_STRIDE;
    }
#pragma unroll LDGSTS_B_COL_TIMES
    for (int j = 0; j < LDGSTS_B_COL_TIMES; j++)
    {
        B_col_id[j] = (ldgsts_B_col_id + j * LDGSTS_B_COL_STRIDE) * VEC_SIZE;
        B_is_col_ok[j] = B_col_id[j] < current_block_N;
    }

    float *current_block_A_base;
    float *current_block_B_base;
    float *current_block_C_base = d_C + block_row_id_start * ldc + block_col_id_start;
    float *current_area_A_base;
    float *current_area_B_base;
    float *current_area_C_base;
    float *current_thread_A_base;
    float *current_thread_B_base;
    float *current_thread_C_base;

    alignas(sizeof(FVEC)) __shared__ float A_panel[2][BL][BM];
    alignas(sizeof(FVEC)) __shared__ float B_panel[2][BL][BN];
    alignas(sizeof(FVEC)) float A_reg[2][RM]{0.0f};
    alignas(sizeof(FVEC)) float B_reg[2][RN]{0.0f};
    alignas(sizeof(FVEC)) float C_reg[RM][RN]{0.0f};
    FVEC A_panel_reg[LDGSTS_A_COL_TIMES][LDGSTS_A_ROW_TIMES];
    FVEC B_panel_reg[LDGSTS_B_ROW_TIMES][LDGSTS_B_COL_TIMES];

    int panel_write_id = 0;
    int panel_read_id = 1;
    int reg_write_id;
    int reg_read_id;

    int block_L_id_start = 0;
    if constexpr (true)
    {
        current_block_L = min(BL, L - block_L_id_start);
        // parallel load A from global memory to A panel
        current_block_A_base = d_A + block_L_id_start * lda + block_row_id_start;
#pragma unroll LDGSTS_A_COL_TIMES
        for (int j = 0; j < LDGSTS_A_COL_TIMES; j++)
        {
            bool is_col_ok = A_col_id[j] < current_block_L;
#pragma unroll LDGSTS_A_ROW_TIMES
            for (int i = 0; i < LDGSTS_A_ROW_TIMES; i++)
            {
                A_panel_reg[j][i] =
                    (A_is_row_ok[i] && is_col_ok)
                        ? *reinterpret_cast<FVEC *>(current_block_A_base + A_col_id[j] * lda + A_row_id[i])
                        : ZERO_VEC;
            }
        }
        // parallel load B from global memory to B panel
        current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
#pragma unroll LDGSTS_B_ROW_TIMES
        for (int i = 0; i < LDGSTS_B_ROW_TIMES; i++)
        {
            bool is_row_ok = B_row_id[i] < current_block_L;
#pragma unroll LDGSTS_B_COL_TIMES
            for (int j = 0; j < LDGSTS_B_COL_TIMES; j++)
            {
                B_panel_reg[i][j] =
                    (is_row_ok && B_is_col_ok[j])
                        ? *reinterpret_cast<FVEC *>(current_block_B_base + B_row_id[i] * ldb + B_col_id[j])
                        : ZERO_VEC;
            }
        }

        // parallel load A from global memory to A panel
#pragma unroll LDGSTS_A_COL_TIMES
        for (int j = 0; j < LDGSTS_A_COL_TIMES; j++)
        {
            if (A_col_id[j] >= BL)
                break;
#pragma unroll LDGSTS_A_ROW_TIMES
            for (int i = 0; i < LDGSTS_A_ROW_TIMES; i++)
            {
                *reinterpret_cast<FVEC *>(&A_panel[panel_write_id][A_col_id[j]][A_row_id[i]]) = A_panel_reg[j][i];
            }
        }
        // parallel load B from global memory to B panel
#pragma unroll LDGSTS_B_ROW_TIMES
        for (int i = 0; i < LDGSTS_B_ROW_TIMES; i++)
        {
            if (B_row_id[i] >= BL)
                break;
#pragma unroll LDGSTS_B_COL_TIMES
            for (int j = 0; j < LDGSTS_B_COL_TIMES; j++)
            {
                *reinterpret_cast<FVEC *>(&B_panel[panel_write_id][B_row_id[i]][B_col_id[j]]) = B_panel_reg[i][j];
            }
        }

        panel_write_id ^= 1;
        panel_read_id ^= 1;
        __syncthreads();

#pragma unroll NUM_AREA_Y
        for (int i = 0; i < NUM_AREA_Y; i++)
        {
            current_area_A_base = &A_panel[panel_read_id][0][area_row_id_start[i]];
            current_thread_A_base = current_area_A_base + thread_row_id_start;
            *reinterpret_cast<FVEC *>(A_reg[0] + i * VEC_SIZE) =
                *reinterpret_cast<FVEC *>(current_thread_A_base);
        }
#pragma unroll NUM_AREA_X
        for (int j = 0; j < NUM_AREA_X; j++)
        {
            current_area_B_base = &B_panel[panel_read_id][0][area_col_id_start[j]];
            current_thread_B_base = current_area_B_base + thread_col_id_start;
            *reinterpret_cast<FVEC *>(B_reg[0] + j * VEC_SIZE) =
                *reinterpret_cast<FVEC *>(current_thread_B_base);
        }
    }

    for (block_L_id_start = BL; block_L_id_start < L; block_L_id_start += BL)
    {
        if constexpr (true)
        {
            current_block_L = min(BL, L - block_L_id_start);
            // parallel load A from global memory to A panel
            current_block_A_base = d_A + block_L_id_start * lda + block_row_id_start;
#pragma unroll LDGSTS_A_COL_TIMES
            for (int j = 0; j < LDGSTS_A_COL_TIMES; j++)
            {
                bool is_col_ok = A_col_id[j] < current_block_L;
#pragma unroll LDGSTS_A_ROW_TIMES
                for (int i = 0; i < LDGSTS_A_ROW_TIMES; i++)
                {
                    A_panel_reg[j][i] =
                        (A_is_row_ok[i] && is_col_ok)
                            ? *reinterpret_cast<FVEC *>(current_block_A_base + A_col_id[j] * lda + A_row_id[i])
                            : ZERO_VEC;
                }
            }
            // parallel load B from global memory to B panel
            current_block_B_base = d_B + block_L_id_start * ldb + block_col_id_start;
#pragma unroll LDGSTS_B_ROW_TIMES
            for (int i = 0; i < LDGSTS_B_ROW_TIMES; i++)
            {
                bool is_row_ok = B_row_id[i] < current_block_L;
#pragma unroll LDGSTS_B_COL_TIMES
                for (int j = 0; j < LDGSTS_B_COL_TIMES; j++)
                {
                    B_panel_reg[i][j] =
                        (is_row_ok && B_is_col_ok[j])
                            ? *reinterpret_cast<FVEC *>(current_block_B_base + B_row_id[i] * ldb + B_col_id[j])
                            : ZERO_VEC;
                }
            }
        }

        if constexpr (true)
        {
#pragma unroll BL - 1
            for (int thread_L_id_start = 1; thread_L_id_start < BL; thread_L_id_start++)
            {
                reg_write_id = thread_L_id_start % 2;
                reg_read_id = reg_write_id ^ 1;

#pragma unroll NUM_AREA_Y
                for (int i = 0; i < NUM_AREA_Y; i++)
                {
                    current_area_A_base = &A_panel[panel_read_id][thread_L_id_start][area_row_id_start[i]];
                    current_thread_A_base = current_area_A_base + thread_row_id_start;
                    *reinterpret_cast<FVEC *>(A_reg[reg_write_id] + i * VEC_SIZE) =
                        *reinterpret_cast<FVEC *>(current_thread_A_base);
                }
#pragma unroll NUM_AREA_X
                for (int j = 0; j < NUM_AREA_X; j++)
                {
                    current_area_B_base = &B_panel[panel_read_id][thread_L_id_start][area_col_id_start[j]];
                    current_thread_B_base = current_area_B_base + thread_col_id_start;
                    *reinterpret_cast<FVEC *>(B_reg[reg_write_id] + j * VEC_SIZE) =
                        *reinterpret_cast<FVEC *>(current_thread_B_base);
                }

#pragma unroll RM
                for (int i = 0; i < RM; i++)
                {
#pragma unroll RN / VEC_SIZE
                    for (int j = 0; j < RN; j += VEC_SIZE)
                    {
                        *reinterpret_cast<FVEC *>(&C_reg[i][j]) +=
                            A_reg[reg_read_id][i] *
                            *reinterpret_cast<FVEC *>(B_reg[reg_read_id] + j);
                    }
                }
            }
        }

        if constexpr (true)
        {
            // parallel load A from global memory to A panel
#pragma unroll LDGSTS_A_COL_TIMES
            for (int j = 0; j < LDGSTS_A_COL_TIMES; j++)
            {
                if (A_col_id[j] >= BL)
                    break;
#pragma unroll LDGSTS_A_ROW_TIMES
                for (int i = 0; i < LDGSTS_A_ROW_TIMES; i++)
                {
                    *reinterpret_cast<FVEC *>(&A_panel[panel_write_id][A_col_id[j]][A_row_id[i]]) = A_panel_reg[j][i];
                }
            }
            // parallel load B from global memory to B panel
#pragma unroll LDGSTS_B_ROW_TIMES
            for (int i = 0; i < LDGSTS_B_ROW_TIMES; i++)
            {
                if (B_row_id[i] >= BL)
                    break;
#pragma unroll LDGSTS_B_COL_TIMES
                for (int j = 0; j < LDGSTS_B_COL_TIMES; j++)
                {
                    *reinterpret_cast<FVEC *>(&B_panel[panel_write_id][B_row_id[i]][B_col_id[j]]) = B_panel_reg[i][j];
                }
            }

            panel_write_id ^= 1;
            panel_read_id ^= 1;
            __syncthreads();

#pragma unroll NUM_AREA_Y
            for (int i = 0; i < NUM_AREA_Y; i++)
            {
                current_area_A_base = &A_panel[panel_read_id][0][area_row_id_start[i]];
                current_thread_A_base = current_area_A_base + thread_row_id_start;
                *reinterpret_cast<FVEC *>(A_reg[0] + i * VEC_SIZE) =
                    *reinterpret_cast<FVEC *>(current_thread_A_base);
            }
#pragma unroll NUM_AREA_X
            for (int j = 0; j < NUM_AREA_X; j++)
            {
                current_area_B_base = &B_panel[panel_read_id][0][area_col_id_start[j]];
                current_thread_B_base = current_area_B_base + thread_col_id_start;
                *reinterpret_cast<FVEC *>(B_reg[0] + j * VEC_SIZE) =
                    *reinterpret_cast<FVEC *>(current_thread_B_base);
            }
        }

        if constexpr (true)
        {
#pragma unroll RM
            for (int i = 0; i < RM; i++)
            {
#pragma unroll RN / VEC_SIZE
                for (int j = 0; j < RN; j += VEC_SIZE)
                {
                    *reinterpret_cast<FVEC *>(&C_reg[i][j]) +=
                        A_reg[1][i] *
                        *reinterpret_cast<FVEC *>(B_reg[1] + j);
                }
            }
        }
    }

    if constexpr (true)
    {
#pragma unroll BL - 1
        for (int thread_L_id_start = 1; thread_L_id_start < BL; thread_L_id_start++)
        {
            reg_write_id = thread_L_id_start % 2;
            reg_read_id = reg_write_id ^ 1;

#pragma unroll NUM_AREA_Y
            for (int i = 0; i < NUM_AREA_Y; i++)
            {
                current_area_A_base = &A_panel[panel_read_id][thread_L_id_start][area_row_id_start[i]];
                current_thread_A_base = current_area_A_base + thread_row_id_start;
                *reinterpret_cast<FVEC *>(A_reg[reg_write_id] + i * VEC_SIZE) =
                    *reinterpret_cast<FVEC *>(current_thread_A_base);
            }
#pragma unroll NUM_AREA_X
            for (int j = 0; j < NUM_AREA_X; j++)
            {
                current_area_B_base = &B_panel[panel_read_id][thread_L_id_start][area_col_id_start[j]];
                current_thread_B_base = current_area_B_base + thread_col_id_start;
                *reinterpret_cast<FVEC *>(B_reg[reg_write_id] + j * VEC_SIZE) =
                    *reinterpret_cast<FVEC *>(current_thread_B_base);
            }

#pragma unroll RM
            for (int i = 0; i < RM; i++)
            {
#pragma unroll RN / VEC_SIZE
                for (int j = 0; j < RN; j += VEC_SIZE)
                {
                    *reinterpret_cast<FVEC *>(&C_reg[i][j]) +=
                        A_reg[reg_read_id][i] *
                        *reinterpret_cast<FVEC *>(B_reg[reg_read_id] + j);
                }
            }
        }
        
#pragma unroll RM
        for (int i = 0; i < RM; i++)
        {
#pragma unroll RN / VEC_SIZE
            for (int j = 0; j < RN; j += VEC_SIZE)
            {
                *reinterpret_cast<FVEC *>(&C_reg[i][j]) +=
                    A_reg[1][i] *
                    *reinterpret_cast<FVEC *>(B_reg[1] + j);
            }
        }
    }

    // store C
#pragma unroll NUM_AREA_Y
    for (int i = 0; i < NUM_AREA_Y; i++)
    {
#pragma unroll NUM_AREA_X
        for (int j = 0; j < NUM_AREA_X; j++)
        {
            current_area_C_base = current_block_C_base + area_row_id_start[i] * ldc + area_col_id_start[j];
            current_thread_C_base = current_area_C_base + thread_row_id_start * ldc + thread_col_id_start;
#pragma unroll VEC_SIZE
            for (int idx = 0; idx < VEC_SIZE; idx++)
            {
                if (idx >= thread_C_M[i] || 0 >= thread_C_N[j])
                    break;
                FVEC temp = *reinterpret_cast<FVEC *>(&current_thread_C_base[idx * ldc]);
                *reinterpret_cast<FVEC *>(&current_thread_C_base[idx * ldc]) =
                    alpha * *reinterpret_cast<FVEC *>(&C_reg[i * VEC_SIZE + idx][j * VEC_SIZE]) +
                    beta * temp;
            }
        }
    }
}

void matmul_07(
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
    auto kernel = matmul_07_kernel<BM, BN, BL, RM, RN, VEC_SIZE>;
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

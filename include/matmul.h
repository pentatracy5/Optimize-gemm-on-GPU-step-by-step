#pragma once

#define DECLARE_MATMUL(func_name)   \
void func_name(                     \
    const int M,                    \
    const int N,                    \
    const int L,                    \
    const int lda,                  \
    const int ldb,                  \
    const int ldc,                  \
    float* d_A,                     \
    float* d_B,                     \
    float* d_C)

DECLARE_MATMUL(matmul_ref);
DECLARE_MATMUL(matmul_00);
DECLARE_MATMUL(matmul_01);
DECLARE_MATMUL(matmul_02);
DECLARE_MATMUL(matmul_03);
DECLARE_MATMUL(matmul_04);
DECLARE_MATMUL(matmul_05);
DECLARE_MATMUL(matmul_06);

using MatMulFunc = void(*)(const int M, const int N, const int L, const int lda, const int ldb, const int ldc, float* d_A, float* d_B, float* d_C);

// Registry of all matmul versions, indexed by the `version` CLI argument.
extern const MatMulFunc matmul_funcs[];
extern const unsigned int matmul_funcs_count;
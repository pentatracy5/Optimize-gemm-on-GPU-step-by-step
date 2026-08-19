#pragma once

void matmul_ref(
    const int M, 
    const int N, 
    const int L, 
    const int lda, 
    const int ldb, 
    const int ldc, 
    float* d_A, 
    float* d_B, 
    float* d_C);

void matmul_00(
    const int M, 
    const int N, 
    const int L, 
    const int lda, 
    const int ldb, 
    const int ldc, 
    float* d_A, 
    float* d_B, 
    float* d_C);

using MatMulFunc = void(*)(const int M, const int N, const int L, const int lda, const int ldb, const int ldc, float* d_A, float* d_B, float* d_C);

// Registry of all matmul versions, indexed by the `version` CLI argument.
extern const MatMulFunc matmul_funcs[];
extern const unsigned int matmul_funcs_count;
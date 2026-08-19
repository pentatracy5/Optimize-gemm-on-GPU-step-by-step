#include <matmul.h>
#include <macro.h>
#include <cublas_v2.h>

void matmul_ref(
    const int M, 
    const int N, 
    const int L, 
    const int lda, 
    const int ldb, 
    const int ldc, 
    float* d_A, 
    float* d_B, 
    float* d_C)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    static cublasHandle_t handle = []()
    {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        L,
        &alpha,
        d_B,
        ldb,
        d_A,
        lda,
        &beta,
        d_C,
        ldc));
}

void matmul_00(
    const int M, 
    const int N, 
    const int L, 
    const int lda, 
    const int ldb, 
    const int ldc, 
    float* d_A, 
    float* d_B, 
    float* d_C)
{
    // implement later
}

const MatMulFunc matmul_funcs[] = { matmul_00 };
const unsigned int matmul_funcs_count = sizeof(matmul_funcs) / sizeof(matmul_funcs[0]);
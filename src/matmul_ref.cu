#include <config.h>
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
    float *d_A,
    float *d_B,
    float *d_C)
{
    static cublasHandle_t handle = []()
    {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        N,
        M,
        L,
        &ALPHA,
        d_B,
        ldb,
        d_A,
        lda,
        &BETA,
        d_C,
        ldc));
}

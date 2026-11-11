#include <matmul.h>

const MatMulFunc matmul_funcs[] = {
    matmul_00,
    matmul_01,
    matmul_02,
    matmul_03,
    matmul_04,
    matmul_05,
    matmul_06,
    matmul_07,
    matmul_08,
    matmul_09,
    matmul_10};
const unsigned int matmul_funcs_count = sizeof(matmul_funcs) / sizeof(matmul_funcs[0]);

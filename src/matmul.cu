#include <matmul.h>

const MatMulFunc matmul_funcs[] = {matmul_00, matmul_01};
const unsigned int matmul_funcs_count = sizeof(matmul_funcs) / sizeof(matmul_funcs[0]);

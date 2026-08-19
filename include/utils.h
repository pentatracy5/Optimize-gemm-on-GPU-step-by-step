#pragma once

void clear_cache();

void malloc_matrix(
	const int M,
	const int N,
	const int L,
	int &lda,
	int &ldb,
	int &ldc,
	float *&d_A,
	float *&d_B,
	float *&d_C,
	float *&d_REF);

void free_matrix(
	float *&d_A,
	float *&d_B,
	float *&d_C,
	float *&d_REF);

void init_ABCREF(
	const int M,
	const int N,
	const int L,
	const int lda,
	const int ldb,
	const int ldc,
	float *d_A,
	float *d_B,
	float *d_C,
	float *d_REF);

void print_ABC(
	const int M,
	const int N,
	const int L,
	const int lda,
	const int ldb,
	const int ldc,
	const float *d_A,
	const float *d_B,
	const float *d_C);

bool check_result(
	const int M,
	const int N,
	const int ldc,
	const float *d_C,
	const float *d_REF,
	const float tolerance);
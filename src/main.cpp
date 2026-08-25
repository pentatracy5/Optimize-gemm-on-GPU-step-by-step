#include <iostream>
#include <sstream>
#include <Timer.h>
#include <utils.h>
#include <matmul.h>
#include <config.h>

using std::cerr;
using std::cout;
using std::endl;
using std::istringstream;

bool test(int M, int N, int L, unsigned int version)
{
	Timer timer;

	MatMulFunc f{matmul_funcs[version]};
	MatMulFunc ref{matmul_ref};

	constexpr int nrepeats = NREPEATS;
	constexpr int warmup = WARMUP;
	const float tolerance = TOLERANCE;

	float *d_A, *d_B, *d_C, *d_REF;
	int lda, ldb, ldc;
	malloc_matrix(M, N, L, lda, ldb, ldc, d_A, d_B, d_C, d_REF);

	init_ABCREF(M, N, L, lda, ldb, ldc, d_A, d_B, d_C, d_REF);

	for (int i = 0; i < warmup; ++i)
	{
		ref(M, N, L, lda, ldb, ldc, d_A, d_B, d_REF);
		f(M, N, L, lda, ldb, ldc, d_A, d_B, d_C);
	}

	double elapsed = 0.0;
	for (int i = 0; i < nrepeats; i++)
	{
		clear_cache();
		timer.tic_gpu();
		ref(M, N, L, lda, ldb, ldc, d_A, d_B, d_REF);
		elapsed += timer.toc_gpu();
	}
	double time_ref = elapsed / nrepeats / 1000.0;

	elapsed = 0.0;
	for (int i = 0; i < nrepeats; i++)
	{
		clear_cache();
		timer.tic_gpu();
		f(M, N, L, lda, ldb, ldc, d_A, d_B, d_C);
		elapsed += timer.toc_gpu();
	}
	double time_f = elapsed / nrepeats / 1000.0;

	double flops = 2 * M / 1000.0 * N / 1000.0 * L / 1000.0;
	cout << "M\tN\tL\tref_GFLOPS\tf_GFLOPS" << endl;
	cout << M << '\t' << N << '\t' << L << '\t' << flops / time_ref << '\t' << '\t' << flops / time_f << endl;

	const bool pass = check_result(M, N, ldc, d_C, d_REF, tolerance);

	free_matrix(d_A, d_B, d_C, d_REF);
	return pass;
}

void run(int M, int N, int L, unsigned int version)
{
	MatMulFunc f;
	if constexpr (PROFILEREF)
		f = matmul_ref;
	else
		f = matmul_funcs[version];

	constexpr int nrepeats = NREPEATS;

	float *d_A, *d_B, *d_C, *d_REF;
	int lda, ldb, ldc;
	malloc_matrix(M, N, L, lda, ldb, ldc, d_A, d_B, d_C, d_REF);

	init_ABCREF(M, N, L, lda, ldb, ldc, d_A, d_B, d_C, d_REF);

	for (int i = 0; i < nrepeats; ++i)
	{
		f(M, N, L, lda, ldb, ldc, d_A, d_B, d_C);
	}

	free_matrix(d_A, d_B, d_C, d_REF);
}

int main(int argc, char *argv[])
{
	int M, N, L;
	unsigned int version;
	if (argc != 5)
	{
		cerr << "Usage: " << argv[0] << " M N L version" << endl;
		return 1;
	}

	istringstream iss1(argv[1]), iss2(argv[2]), iss3(argv[3]), iss4(argv[4]);
	if (!(iss1 >> M && iss1.eof()) || !(iss2 >> N && iss2.eof()) || !(iss3 >> L && iss3.eof()))
	{
		cerr << "Error: invalid integer arguments." << endl;
		return 1;
	}
	if (!(iss4 >> version && iss4.eof()))
	{
		cerr << "Error: invalid matmul version." << endl;
		return 1;
	}
	if (M <= 0 || N <= 0 || L <= 0)
	{
		cerr << "Error: invalid arguments integer value." << endl;
		return 1;
	}

	if (version >= matmul_funcs_count)
	{
		cerr << "Error: matmul version must be in [0, " << matmul_funcs_count - 1 << "], but got " << version << "." << endl;
		return 1;
	}

	if constexpr (PROFILE)
		run(M, N, L, version);
	else if (!test(M, N, L, version))
		return 1;

	return 0;
}

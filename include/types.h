#pragma once

template <typename T, unsigned int BYTE_SIZE>
struct alignas(BYTE_SIZE) Vec
{
    static_assert((BYTE_SIZE & (BYTE_SIZE - 1)) == 0, "BYTE_SIZE must be power of two");
    static_assert(BYTE_SIZE % sizeof(T) == 0, "Unsupported vectype");
    static constexpr unsigned int vec_len = BYTE_SIZE / sizeof(T);
    T x[vec_len];
    __host__ __device__ inline const T &operator[](unsigned int i) const { return x[i]; }
    __host__ __device__ inline T &operator[](unsigned int i) { return x[i]; }

    __host__ __device__ inline Vec operator+(const Vec &other) const
    {
        Vec out;
#pragma unroll
        for (unsigned int i = 0; i < vec_len; i++)
            out[i] = x[i] + other[i];
        return out;
    }

    __host__ __device__ inline Vec operator*(const Vec &other) const
    {
        Vec out;
#pragma unroll
        for (unsigned int i = 0; i < vec_len; i++)
            out[i] = x[i] * other[i];
        return out;
    }

    __host__ __device__ inline Vec &operator+=(const Vec &other)
    {
#pragma unroll
        for (unsigned int i = 0; i < vec_len; i++)
            x[i] += other[i];
        return *this;
    }

    __host__ __device__ inline Vec &operator*=(const Vec &other)
    {
#pragma unroll
        for (unsigned int i = 0; i < vec_len; i++)
            x[i] *= other[i];
        return *this;
    }
};

template <typename T, unsigned int N>
using VecN = Vec<T, N * sizeof(T)>;
